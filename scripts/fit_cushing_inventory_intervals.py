"""Calibrate inventory forecast intervals against errors known at each release cutoff."""

import argparse
import json
from datetime import datetime, timezone
from hashlib import file_digest

import numpy as np

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_robust_stack import indexed

STEM = "cushing_inventory_intervals"
EVALUATION = ("public_errors","hybrid_errors","hybrid_scaled_errors")
COVERAGE = .8


def source_rows():
    for issue,r in sorted(indexed("cushing_shared_scene_forecasts").items()):
        cutoff,known = (datetime.fromisoformat(r[k]) for k in ("forecast_cutoff_utc","target_known_by_utc"))
        if not cutoff<known or (r["latest_training_label_utc"] and datetime.fromisoformat(r["latest_training_label_utc"])>=cutoff):
            raise ValueError(f"Upstream forecasts or labels arrive out of order on {issue}")
        yield dict(release_date=issue,study_phase=r["study_phase"],cutoff=cutoff,known=known,actual=float(r["actual"]),
                   public=float(r["public_ensemble"]),hybrid=float(r["carried_scene"]),tanks=int(r["measured_roofs"]))


def width_scale(available):
    if len(available)<26:
        return 1.
    errors = [abs(q["actual"]-q["public"]) for q in available[-104:]]
    recent,pooled = [float(np.median(sample)) for sample in (errors[-13:],errors)]
    return float(np.clip((.5*recent+.5*pooled)/max(pooled,.1),.6,1.8))


def distribution(sorted_errors, estimate, scale):
    # One empirical forecast distribution gives a set of coherent quantiles.
    q10,q50,q90 = np.quantile(sorted_errors,(.1,.5,.9),method="linear")*scale+estimate
    support = sorted_errors*scale+estimate
    return (float(q10),float(q50),float(q90)),support


def replay(rows):
    seen = []
    for r in rows:
        history = [q for q in seen if q["known"]<r["cutoff"]][-104:]
        scale = width_scale(history)
        seen.append(r | dict(scale=scale))
        if len(history)<52:
            continue
        public_errors = np.array([q["actual"]-q["public"] for q in history])
        hybrid_errors = np.array([q["actual"]-q["hybrid"] for q in history])
        errors = (public_errors,hybrid_errors,hybrid_errors/np.array([q["scale"] for q in history]))
        predictions = {}
        for method,residuals,point,factor in zip(EVALUATION,errors,(r["public"],r["hybrid"],r["hybrid"]),(1.,1.,scale),strict=True):
            bounds,support = distribution(np.sort(residuals),point,factor)
            # Sorted samples let us compute the empirical CRPS in linear time.
            n = len(support)
            spread_penalty = (support@(2*np.arange(n)+1-n))/n**2
            predictions[method] = dict(quantiles=bounds,crps=float(np.mean(abs(support-r["actual"]))-spread_penalty))
        yield r | dict(scale=scale,estimates=predictions,training_releases=len(history),last_known=history[-1]["known"])


def scores(rows):
    targets = np.array([r["actual"] for r in rows])
    output = {}
    baseline = np.array([r["estimates"]["public_errors"]["crps"] for r in rows])
    for method in EVALUATION:
        low,med,high = np.array([r["estimates"][method]["quantiles"] for r in rows]).T
        crps = np.array([r["estimates"][method]["crps"] for r in rows])
        penalty = 2/(1-COVERAGE)*(np.maximum(low-targets,0.)+np.maximum(targets-high,0.))
        output[method] = dict(covered=int(np.count_nonzero((targets>=low)&(targets<=high))),
            below=int(np.count_nonzero(targets<low)),above=int(np.count_nonzero(targets>high)),
            width_kbbl=float(1000*(high-low).mean()),interval_score_kbbl=float(1000*(high-low+penalty).mean()),
            median_mae_kbbl=float(1000*abs(targets-med).mean()),crps_kbbl=float(1000*crps.mean()),
            paired_crps_delta_vs_public_95_kbbl=block_bootstrap_mean(1000*(crps-baseline)))
    return dict(weeks=len(rows),models=output)


def register(release):
    from register_cushing_prerelease_forecast import freeze

    original_path = DERIVED/f"cushing_prerelease_forecast_{release}.json"
    shared_path = DERIVED/f"cushing_prerelease_forecast_{release}_shared_scene.json"
    path = DERIVED/f"cushing_prerelease_forecast_{release}_intervals.json"
    if path.exists():
        return json.loads(path.read_bytes())
    original,shared = (json.loads(p.read_bytes()) for p in (original_path,shared_path))
    cutoff,deadline = (datetime.fromisoformat(original[k]) for k in ("information_cutoff_utc","registration_deadline_utc"))
    if datetime.now(timezone.utc)>=deadline:
        raise ValueError("This release is past the interval registration deadline")
    history = [r for r in source_rows() if r["known"]<cutoff][-104:]
    if len(history)<52 or datetime.fromisoformat(shared["information_cutoff_utc"])!=cutoff:
        raise ValueError("The frozen point forecast or its calibration history is inconsistent")
    estimates = {}
    for k,column,value in (("public_errors","public",original["public_ensemble_kbbl"]),
                           ("hybrid_errors","hybrid",shared["shared_scene_kbbl"])):
        residuals = np.sort([r["actual"]-r[column] for r in history])
        quantiles,_ = distribution(residuals,value/1000,1.)
        estimates[k] = dict(zip(("p10_kbbl","p50_kbbl","p90_kbbl"),(1000*q for q in quantiles),strict=True))
    now = datetime.now(timezone.utc)
    if not cutoff<=now<deadline:
        raise ValueError("The interval receipt time does not precede the release")
    sources = {}
    for p in (original_path,shared_path,DERIVED/"cushing_shared_scene_forecasts.csv"):
        with p.open("rb") as handle:
            sources[p.name] = file_digest(handle,"sha256").hexdigest()
    record = dict(release_date=release,week_ending=original["week_ending"],recorded_at_utc=now.isoformat(),
        information_cutoff_utc=cutoff.isoformat(),deadline_utc=deadline.isoformat(),calibration_releases=len(history),
        latest_calibration_release_utc=history[-1]["known"].isoformat(),nominal_coverage=COVERAGE,
        estimates=estimates,source_sha256=sources)
    freeze(path,record)
    return record


def score_registered(release):
    from register_cushing_prerelease_forecast import digest, score

    original_path = DERIVED/f"cushing_prerelease_forecast_{release}.json"
    path = DERIVED/f"cushing_prerelease_forecast_{release}_intervals.json"
    record = json.loads(path.read_bytes())
    original = json.loads(original_path.read_bytes())
    registered,cutoff,deadline,label = (datetime.fromisoformat(record[k]) for k in (
        "recorded_at_utc","information_cutoff_utc","deadline_utc","latest_calibration_release_utc"))
    if (not label<cutoff<=registered<deadline or record["nominal_coverage"]!=COVERAGE or
        record["release_date"]!=original["release_date"] or record["week_ending"]!=original["week_ending"] or
        original["registration_deadline_utc"]!=record["deadline_utc"] or
        any(digest(DERIVED/name)!=sha for name,sha in record["source_sha256"].items())):
        raise ValueError("The interval receipt or source checksums do not match")
    outcome = score(original_path,original)
    result = dict(release_date=release,status=outcome["status"],interval_forecast_sha256=digest(path))
    if outcome["status"]=="scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual,first_release_sha256=outcome["first_release_sha256"],models={})
        for method,estimates in record["estimates"].items():
            lo,med,hi = (estimates[f"p{n}_kbbl"] for n in (10,50,90))
            if not lo<=med<=hi:
                raise ValueError(f"Crossed registered intervals for {method}")
            result["models"][method] = dict(covered=lo<=actual<=hi,median_absolute_error_kbbl=abs(med-actual),
                interval_score_kbbl=hi-lo+2/(1-COVERAGE)*(max(lo-actual,0.)+max(actual-hi,0.)))
    write(path.with_name(f"{path.stem}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--register-release", metavar="YYYY-MM-DD")
    command.add_argument("--score-release", metavar="YYYY-MM-DD")
    args = parser.parse_args()
    release = args.register_release or args.score_release
    if release:
        datetime.strptime(release,"%Y-%m-%d")
        result = register(release) if args.register_release else score_registered(release)
        print(json.dumps(result,indent=2,allow_nan=False))
        return
    rows = list(replay(source_rows()))
    evaluation = {}
    for phase in PHASES:
        sample = [r for r in rows if r["study_phase"]==phase]
        evaluation[phase] = dict(all=scores(sample),measured=scores([r for r in sample if r["tanks"]]))
        print(phase,evaluation[phase]["all"])
    export(f"{STEM}_forecasts",[{k:r[k] for k in ("release_date","study_phase","actual","public","hybrid","tanks","scale","training_releases")} |
        {f"{k}_utc":r[k].isoformat() for k in ("cutoff","known","last_known")} | {
            f"{m}_{k}":float(v) for m,e in r["estimates"].items() for k,v in zip(("p10","p50","p90","crps"),(*e["quantiles"],e["crps"]),strict=True)}
        for r in rows])
    report = dict(status="historical_walk_forward_on_reviewed_period",nominal_coverage=COVERAGE,phases=evaluation)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
