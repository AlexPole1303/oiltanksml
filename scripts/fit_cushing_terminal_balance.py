"""Infer hub inventory with separate tank, shared-scene, and internal-transfer errors."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC, learned_beta
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL, SCENE_SIGMA_M, parts, replay as shared_replay, scene_design

STEM = "cushing_terminal_balance"
MODELS = (PUBLIC,"carried_scene","terminal_fixed","terminal_learned","terminal_learned_direct")
DEFAULT = np.array([1.,SCENE_SIGMA_M**2,.6**2])
MIN_WEEKS = 32


def mapped_areas():
    result = defaultdict(dict)
    for phase,stem in (("open","sentinel2_space_time_coverage_terminal_weeks"),("holdout","holdout_optical_terminal_weeks")):
        for r in read(stem):
            if (r["study_phase"]=="holdout")!=(phase=="holdout"):
                continue
            site = r["terminal_id"]
            key = r["release_date"],r["week_ending"]
            if site in result[key]:
                raise ValueError(f"Duplicate terminal area on {key}/{site}")
            result[key][site] = float(r["dated_reference_area_m2"]),datetime.fromisoformat(r["cutoff_utc"])
    return result


def geometries(areas, state):
    observations = state["observed"]
    names = sorted(areas)
    if any(cutoff!=state["cutoff"] or area<0 for area,cutoff in areas.values()):
        raise ValueError(f"Terminal roster is inconsistent on {state['release_date']}")
    total = sum(area for area,_ in areas.values())
    if total<=0:
        raise ValueError(f"No dated terminal area on {state['release_date']}")
    shares = np.array([areas[name][0]/total for name in names])
    terminal = [names.index(r["terminal_id"]) for r in observations]
    if not np.isclose(shares.sum(),1.) or any(shares[i]<=0 for i in terminal):
        raise ValueError(f"Observed roof is absent from the dated roster on {state['release_date']}")
    # Conditional covariance of independent terminal deviations given their
    # zero sum over the complete mapped roster. Negative off-diagonal terms
    # represent a transfer between terminals that leaves the hub total unchanged.
    balanced = np.diag(shares)-np.outer(shares,shares)
    if not np.allclose(balanced.sum(axis=0),0,atol=1e-12):
        raise ValueError("Terminal deviations do not conserve the hub total")
    tank_share = np.array([r["area"]/total for r in observations])
    if np.any(tank_share<=0):
        raise ValueError(f"Invalid measured roof area on {state['release_date']}")
    normalized = tank_share/shares[terminal]
    current = np.array([r["sensitivity"] for r in observations])/tank_share
    previous = np.array([r["duration"] for r in observations])-current
    if not (np.all((0<current)&(current<=1+1e-9)) and np.all((-1e-9<=previous)&(previous<=1+1e-9))):
        raise ValueError(f"Price-independent observation fractions are invalid on {state['release_date']}")
    before = normalized*np.maximum(previous,0.)
    after = normalized*current
    transfer = balanced[np.ix_(terminal,terminal)]*(np.outer(before,before)+np.outer(after,after))
    return transfer


def source_rows():
    area = mapped_areas()
    for output,r in shared_replay():
        observations = r["observed"]
        if not observations:
            yield r | dict(public_ensemble=r["public"],transfers=None,precision_inputs=None),output
            continue
        if any(datetime.fromisoformat(q["available_utc"])>=r["cutoff"] for q in observations):
            raise ValueError(f"Satellite inputs arrived late on {r['release_date']}")
        transfers = geometries(area[r["release_date"],r["week_ending"]],r)
        _,residual,noise = parts(observations,r["actual"])
        names = sorted({q[k] for q in observations for k in ("scene_before","scene_after")})
        design = scene_design(observations,{n:i for i,n in enumerate(names)})
        components = np.stack((noise,design@design.T,transfers))
        # The fitting objective is a Gaussian approximation to signed, robustly
        # capped residuals. It treats release clusters as separate for calibration.
        cap = 4*np.sqrt(np.diag(np.einsum("i,ijk->jk",DEFAULT,components)))
        yield r | dict(public_ensemble=r["public"],transfers=transfers,
                       precision_inputs=(components,np.clip(residual,-cap,cap))),output


def variance_fit(past, now):
    if len(past)<MIN_WEEKS:
        return DEFAULT.copy()
    weight = np.array([2**(-(now-r["cutoff"]).days/(7*78.))/math.sqrt(len(r["observed"])) for r in past])
    weight /= sum(weight*np.array([len(r["observed"]) for r in past]))

    def objective(log_scale):
        variances = np.exp(log_scale)
        distance = log_scale-np.log(DEFAULT)
        result,derivative = .025*float(distance@distance),.05*distance
        for row,w in zip(past,weight,strict=True):
            m,y = row["precision_inputs"]
            cov = np.einsum("i,ijk->jk",variances,m)
            fact = cho_factor(cov)
            inverse = cho_solve(fact,np.eye(len(y)))
            alpha = inverse@y
            result += w*(2*np.log(np.diag(fact[0])).sum()+y@alpha)
            derivative += w*variances*np.einsum("ij,kij->k",inverse-np.outer(alpha,alpha),m)
        return result,derivative

    bounds = [(math.log(a),math.log(b)) for a,b in ((.25,9.),(.05,4.),(.025,9.))]
    optimum = minimize(objective,np.log(DEFAULT),jac=True,bounds=bounds,method="L-BFGS-B")
    if not optimum.success or not np.isfinite(optimum.x).all():
        raise ValueError(f"Variance fit failed at {now}: {optimum.message}")
    return np.exp(optimum.x)


def update(row, recent, var):
    scenes = {q[k] for q in row["observed"] for k in ("scene_before","scene_after")}
    connected = []
    for old in reversed(recent):
        used = {q[k] for q in old["observed"] for k in ("scene_before","scene_after")}
        if used & scenes:
            connected.append(old)
            scenes.update(used)
    names = {n:i for i,n in enumerate(sorted(scenes))}
    precision = np.eye(len(names))/var[1]
    score = np.zeros(len(names))
    for old in connected:
        m,y = old["precision_inputs"]
        cov = var[0]*m[0]+var[2]*m[2]
        design = scene_design(old["observed"],names)
        estimate = cho_solve(cho_factor(cov),np.column_stack((design,y)))
        precision += design.T@estimate[:,:-1]
        score += design.T@estimate[:,-1]
    design = scene_design(row["observed"],names)
    posterior = cho_solve(cho_factor(precision),np.column_stack((design.T,score)))
    sensitivity,residual,noise = parts(row["observed"],row[PUBLIC])
    cov = var[0]*noise+var[2]*row["transfers"]+design@posterior[:,:-1]
    deviation = residual-design@posterior[:,-1]
    clipped = np.clip(deviation,-3*np.sqrt(cov.diagonal()),3*np.sqrt(cov.diagonal()))
    projection = cho_solve(cho_factor(cov),sensitivity)
    information = float(sensitivity@projection)
    if information<0:
        raise ValueError(f"Invalid GLS information on {row['release_date']}")
    value = float(np.clip(projection@clipped/(1/PRIOR_INVENTORY_SIGMA_MBBL**2+information),-1.25,1.25))
    return value,information


def replay(rows):
    seen = []
    for r,output in rows:
        fixed = adjusted = info = 0.
        scales = DEFAULT.copy()
        past = [q for q in seen if q["known"]<r["cutoff"]][-104:]
        if r["observed"]:
            recent = [q for q in past if r["cutoff"]-q["cutoff"]<=MAX_SCENE_AGE]
            scales = variance_fit(past,r["cutoff"])
            fixed,_ = update(r,recent,DEFAULT)
            adjusted,info = update(r,recent,scales)
        beta_fixed,beta_learned = (learned_beta(past,k) for k in ("fixed_raw","adjusted_raw"))
        baseline = r[PUBLIC]
        result = r | dict(carried_scene=output["carried_scene"],terminal_fixed=baseline+beta_fixed*fixed,
            terminal_learned=baseline+beta_learned*adjusted,terminal_learned_direct=baseline+adjusted,
            fixed_raw=fixed,adjusted_raw=adjusted,information=info,noise_scale=float(math.sqrt(scales[0])),
            shared_scene_sigma_m=float(math.sqrt(scales[1])),terminal_transfer_sigma_mbbl=float(math.sqrt(scales[2])),
            training_weeks=len(past),latest_training_label_utc=past[-1]["known"].isoformat() if past else None)
        if r["observed"]:
            seen.append(result)
        yield result


def register(release):
    from register_cushing_prerelease_forecast import digest, freeze

    prefix = f"cushing_prerelease_forecast_{release}"
    path = DERIVED/f"{prefix}_terminal_balance.json"
    if path.exists():
        return json.loads(path.read_bytes())
    original_path = DERIVED/f"{prefix}.json"
    previous_path = DERIVED/f"{prefix}_shared_scene.json"
    previous = json.loads(previous_path.read_bytes())
    original = json.loads(original_path.read_bytes())
    cutoff,deadline = (datetime.fromisoformat(original[k]) for k in ("information_cutoff_utc","registration_deadline_utc"))
    if datetime.now(timezone.utc)>=deadline or previous["original_forecast_sha256"]!=digest(original_path):
        raise ValueError("Registration is late or its original forecast does not match")
    native_path = DERIVED/f"cushing_forward_native_optical_{release}_summary.json"
    native = json.loads(native_path.read_bytes())
    native_cutoff = datetime.fromisoformat(native["observation_cutoff"])
    if native["outcome_available"] or native_cutoff>cutoff or native["week_ending"]!=original["week_ending"] or (
        original["source_sha256"].get(native_path.name)!=digest(native_path)
    ):
        raise ValueError("The dated forward roof roster does not match the frozen satellite summary")
    reference = read("eia_wpsr_forward_releases")
    last = next(r for r in reference if r["release_date"]==original["prior_release"])
    observed = [dict(r,area=r["volume_per_m"]*BARREL_M3*1e6) for r in original["measured_pairs"]]
    if any(datetime.fromisoformat(r["source_available_utc"])>=cutoff for r in observed):
        raise ValueError("A frozen satellite input arrived after the cutoff")
    roster = [r for r in read("cushing_modern_lidar_optical_registry") if r["modern_input_status"]=="available" and
              r["latest_engineering_date"]<native_cutoff.date().isoformat() and r["first_seen"]<=last["latest_week_in_issue"]]
    if len(roster)!=native["dated_roofs"] or not math.isclose(math.fsum(float(r["circular_area_m2"]) for r in roster),
            native["dated_roof_area_m2"],abs_tol=1e-7):
        raise ValueError("The dated forward roof roster does not reproduce")
    by_terminal = defaultdict(float)
    for r in roster:
        by_terminal[r["terminal_id"]]+=float(r["circular_area_m2"])
    row = dict(release_date=release,cutoff=cutoff,observed=observed,public_ensemble=original["public_ensemble_kbbl"]/1000)
    row["transfers"] = geometries({site:(a,cutoff) for site,a in by_terminal.items()},row) if observed else None
    past = [r for r,_ in source_rows() if r["observed"] and r["known"]<cutoff][-104:]
    recent = [q for q in past if cutoff-q["cutoff"]<=MAX_SCENE_AGE]
    with threadpool_limits(limits=1):
        scales = variance_fit(past,cutoff)
        revision,information = update(row,recent,scales) if observed else (0.,0.)
    now = datetime.now(timezone.utc)
    if not cutoff<=now<deadline:
        raise ValueError("The terminal-balance forecast is past its registration deadline")
    record = dict(release_date=release,week_ending=original["week_ending"],recorded_at_utc=now.isoformat(),
        information_cutoff_utc=cutoff.isoformat(),registration_deadline_utc=deadline.isoformat(),
        public_kbbl=original["public_ensemble_kbbl"],shared_scene_kbbl=previous["shared_scene_kbbl"],
        terminal_model_kbbl=1000*(row[PUBLIC]+revision),model_update_kbbl=1000*revision,
        variance_parameters=scales.tolist(),information=information,observed_tanks=len(observed),
        training_releases=len(past),latest_training_label_utc=past[-1]["known"].isoformat(),
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in (original_path,previous_path,native_path,
            DERIVED/"cushing_modern_lidar_optical_registry.csv",DERIVED/f"{STEM}_forecasts.csv",ROOT/"scripts/fit_cushing_terminal_balance.py")})
    freeze(path,record)
    return record


def score_registered(release):
    from register_cushing_prerelease_forecast import digest, score

    original_path = DERIVED/f"cushing_prerelease_forecast_{release}.json"
    path = DERIVED/f"cushing_prerelease_forecast_{release}_terminal_balance.json"
    original,record = (json.loads(p.read_bytes()) for p in (original_path,path))
    training,cutoff,stamp,deadline = (datetime.fromisoformat(record[k]) for k in (
        "latest_training_label_utc","information_cutoff_utc","recorded_at_utc","registration_deadline_utc"))
    if not training<cutoff<=stamp<deadline or any(record[k]!=original[k] for k in ("release_date","week_ending",
        "information_cutoff_utc","registration_deadline_utc")) or any(
        digest(ROOT/p)!=h for p,h in record["source_sha256"].items() if p!=str((ROOT/"scripts/fit_cushing_terminal_balance.py").relative_to(ROOT))):
        raise ValueError("The terminal-balance receipt no longer matches the frozen inputs")
    outcome = score(original_path,original)
    result = dict(status=outcome["status"],release_date=release,receipt_sha256=digest(path))
    if outcome["status"]=="scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual,report_sha256=outcome["first_release_sha256"],
            absolute_error_kbbl={k:abs(actual-record[f"{k}_kbbl"]) for k in ("public","shared_scene","terminal_model")})
    write(path.with_name(f"{path.stem}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--register-release",type=date.fromisoformat)
    action.add_argument("--score-release",type=date.fromisoformat)
    args = parser.parse_args()
    release = args.register_release or args.score_release
    if release:
        result = register(release.isoformat()) if args.register_release else score_registered(release.isoformat())
        print(json.dumps(result,indent=2,allow_nan=False))
        return
    with threadpool_limits(limits=1):
        rows = list(replay(source_rows()))
        trades = list(trading(rows,MODELS))
    report = dict(status="historical_walk_forward_on_reviewed_period",forecasts={},trading={})
    for phase in PHASES:
        rows_in_phase = [r for r in rows if r["study_phase"]==phase]
        report["forecasts"][phase] = {scope:forecast_metrics(group,MODELS) for scope,group in (
            ("all",rows_in_phase),("measured",[r for r in rows_in_phase if r["observed"]]),
            ("multi_terminal",[r for r in rows_in_phase if len({q["terminal_id"] for q in r["observed"]})>=2]))}
        grouped = {m:[r for r in trades if r["study_phase"]==phase and r["method"]==m] for m in ("curve",*MODELS)}
        base = grouped[PUBLIC]
        result = {}
        for model,group in grouped.items():
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different market calendars in {phase}/{model}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(group,base,strict=True)]
            result[model] = robustness(group) | daily_risk(group) | dict(
                increment_usd=sum(delta),paired_delta_95_usd=block_bootstrap_mean(delta))
        report["trading"][phase] = result
        print(phase,{s:(d["weeks"],{m:round(v["mae_kbbl"],2) for m,v in d["models"].items()})
                     for s,d in report["forecasts"][phase].items()})
        print(phase,{m:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in result.items()})
    export(f"{STEM}_forecasts",[{k:r[k] for k in ("release_date","study_phase","actual","training_weeks","latest_training_label_utc",
        "noise_scale","shared_scene_sigma_m","terminal_transfer_sigma_mbbl","fixed_raw","adjusted_raw","information",*MODELS)} |
        dict(cutoff_utc=r["cutoff"].isoformat(),known_utc=r["known"].isoformat(),tanks=len(r["observed"]),
             terminals=len({q["terminal_id"] for q in r["observed"]})) for r in rows])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
