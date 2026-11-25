"""Test probabilistic inventory forecasts and tail-aware CL signals on matched releases."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
from scipy.special import ndtr, ndtri
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import risk_prediction
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_late_pair_recovery import COMBINED, HYBRID, STEM as LATE
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from register_cushing_prerelease_forecast import digest, freeze, score as score_inventory

STEM = "cushing_tail_distribution"
LEVELS = (1.,1.5)
MODELS = {"public":(PUBLIC, False), "hybrid":(HYBRID, False),
          "combined":(COMBINED, False), "matched_combined":(COMBINED, True)}
POLICIES = ("public_conditional", "combined_conditional", "public_percentile", "combined_percentile",
            "matched_percentile", "combined_percentile_blend")
BANDWIDTH = .20


def distribution(train, row, point, local):
    residuals = np.array([q["actual"]-q[point] for q in train])
    ages = np.array([(row["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in train])
    weight = np.exp2(-ages/78)
    if local:
        delta = np.array([q["conditions"] for q in train])-row["conditions"]
        distance = np.square(delta).mean(axis=1)
        weight *= .35+.65*np.exp(-distance/2)
    weight /= weight.sum()
    support = row[point]+residuals
    probabilities = {f"{level:g}":(float(weight@ndtr((-level-support)/BANDWIDTH)),
                                     float(weight@ndtr((support-level)/BANDWIDTH))) for level in LEVELS}
    y = row.get("actual")
    # CDF evaluated at the subsequently reported value. This is used only
    # after the conservative EIA availability clock in the trading model.
    pit = float(weight@ndtr((y-support)/BANDWIDTH)) if y is not None else None
    return probabilities,pit,float(1/(weight@weight))


def source_rows():
    market = indexed("cushing_inventory_market_bridge_forecasts")
    for issue,r in sorted(indexed(f"{LATE}_forecasts").items()):
        cutoff,known = (datetime.fromisoformat(r[k]) for k in ("cutoff","known"))
        if not cutoff<known or any(datetime.fromisoformat(r[k])>=cutoff for k in (
                "latest_training_label_utc","latest_optical_at_utc") if r[k]):
            raise ValueError(f"Different inventory forecast clock on {issue}")
        q = market[issue]
        if datetime.fromisoformat(q["cutoff_utc"])!=cutoff or datetime.fromisoformat(q["known_utc"])!=known:
            raise ValueError(f"Different market feature clock on {issue}")
        optical = r["measured"]=="True" or int(r["late_roofs"])>0
        date = datetime.fromisoformat(r["week_ending"]).date()
        angle = 2*math.pi*date.timetuple().tm_yday/365.25
        change = float(q["survey_spread_move"]) if q["survey_spread_move"] else 0.
        features = np.array((2.*optical, math.sin(angle), math.cos(angle), min(abs(change),2.),
                             min(abs(float(r[PUBLIC])),2.)))
        yield {k:r[k] for k in ("release_date","week_ending","study_phase")} | dict(
            cutoff=cutoff,known=known,actual=float(r["actual"]),covered=optical,conditions=features,
            **{k:float(r[k]) for k in (PUBLIC,HYBRID,COMBINED)})


def forecasts():
    history = []
    for current in source_rows():
        train = [q for q in history if q["known"]<current["cutoff"]][-156:]
        if len(train)>=52:
            pred = {name:distribution(train,current,point,local) for name,(point,local) in MODELS.items()}
            yield current | dict(probabilities=pred,training_weeks=len(train),latest_label_utc=train[-1]["known"].isoformat())
        history.append(current)


def register_probabilities(issue):
    prefix = f"cushing_prerelease_forecast_{issue}"
    original_path = DERIVED/f"{prefix}.json"
    satellite_path = DERIVED/f"cushing_forward_cl_satellite_{issue}_inventory.json"
    path = DERIVED/f"{prefix}_tail_probabilities.json"
    original, satellite = (json.loads(p.read_bytes()) for p in (original_path,satellite_path))
    cutoff,deadline = (datetime.fromisoformat(original[k]) for k in ("information_cutoff_utc","registration_deadline_utc"))
    fields = ("release_date","week_ending","information_cutoff_utc","registration_deadline_utc")
    if any(original[k]!=satellite[k] for k in fields):
        raise ValueError("The point models use different releases or cutoffs")
    if path.exists():
        record = json.loads(path.read_bytes())
        if (any(digest(ROOT/p)!=checksum for p,checksum in record["source_sha256"].items())
                or not datetime.fromisoformat(record["latest_calibration_label_utc"])<cutoff
                or not cutoff<=datetime.fromisoformat(record["recorded_at_utc"])<deadline
                or any(record[k]!=original[k] for k in fields)):
            raise ValueError("The frozen probability forecast no longer matches its inputs")
        return record
    if issue in indexed("eia_wpsr_forward_releases") or not cutoff<=datetime.now(timezone.utc)<deadline:
        raise ValueError("A probability forecast requires an unpublished first report")
    train = [q for q in source_rows() if q["known"]<cutoff][-156:]
    if len(train)<52:
        raise ValueError("Insufficient published errors for probability estimation")
    public,sat = (satellite["estimates_kbbl"][key]/1000 for key in (PUBLIC,"outer_supplement"))
    current = {PUBLIC:public,"outer_supplement":sat,"cutoff":cutoff}
    archived = indexed("cushing_outer_capture_nowcast_forecasts")
    for r in train:
        previous = archived[r["release_date"]]
        if datetime.fromisoformat(previous["known_utc"])!=r["known"] or float(previous["actual"])!=r["actual"]:
            raise ValueError("The probability calibration vintages differ")
        r["outer_supplement"] = float(previous["outer_supplement"])
    forecasts = {}
    for key in (PUBLIC,"outer_supplement"):
        values,_,_ = distribution(train,current,key,False)
        forecasts[key] = {f"{level}_million_bbl":dict(draw=p[0],normal=1-sum(p),build=p[1]) for level,p in values.items()}
    now = datetime.now(timezone.utc)
    if now>=deadline:
        raise ValueError("Forecast finished outside the registration window")
    sources = (original_path,satellite_path,DERIVED/f"{LATE}_forecasts.csv",DERIVED/"cushing_outer_capture_nowcast_forecasts.csv",
               ROOT/"scripts"/f"fit_{STEM}.py")
    record = {k:original[k] for k in fields} | dict(recorded_at_utc=now.isoformat(),training_releases=len(train),
        latest_calibration_label_utc=train[-1]["known"].isoformat(),probabilities=forecasts,
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources})
    freeze(path,record)
    return record


def score_registered(issue):
    base = DERIVED/f"cushing_prerelease_forecast_{issue}"
    original_path = base.with_suffix(".json")
    record = register_probabilities(issue)
    outcome = score_inventory(original_path,json.loads(original_path.read_bytes()))
    result = dict(release_date=issue,status=outcome["status"])
    if outcome["status"]=="scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]/1000
        scores = {}
        for name,values in record["probabilities"].items():
            scores[name] = {}
            for label,probability in values.items():
                level = float(label.split("_")[0])
                event = "draw" if actual<=-level else "build" if actual>=level else "normal"
                scores[name][label] = dict(brier=sum((p-(key==event))**2 for key,p in probability.items()),outcome=event)
        result.update(actual_eia_change_kbbl=outcome["actual_eia_change_kbbl"],scores=scores,
                      first_release_sha256=outcome["first_release_sha256"])
    write(base.with_name(f"{base.name}_tail_probabilities_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


def tail_score(sample, boundary):
    actual = np.array([r["actual"] for r in sample])
    events = np.column_stack((actual<=-boundary,actual>=boundary))
    categories = np.column_stack((events[:,0],~events.any(axis=1),events[:,1]))
    base = None
    result = dict(weeks=len(sample),large_draws=int(events[:,0].sum()),large_builds=int(events[:,1].sum()),models={})
    for model in MODELS:
        extremes = np.array([r["probabilities"][model][0][f"{boundary:g}"] for r in sample])
        p = np.column_stack((extremes[:,0],1-extremes.sum(axis=1),extremes[:,1]))
        per_week = np.square(p-categories).sum(axis=1)
        if base is None:
            base = per_week
        event_p = extremes.sum(axis=1)
        y = events.any(axis=1)
        log = -np.log(np.clip(p[categories],1e-9,1.))
        result["models"][model] = dict(brier=float(per_week.mean()),log_loss=float(log.mean()),
            tail_brier=float(np.square(event_p-y).mean()),average_precision=float(average_precision_score(y,event_p)) if y.any() else None,
            draw_average_precision=float(average_precision_score(events[:,0],extremes[:,0])) if events[:,0].any() else None,
            build_average_precision=float(average_precision_score(events[:,1],extremes[:,1])) if events[:,1].any() else None,
            brier_delta_vs_public_95=block_bootstrap_mean(per_week-base))
    return result


def market(rows):
    forecasts = {r["release_date"]:r for r in rows}
    controls = {(r["release_date"],r["method"]):r for r in read(f"{LATE}_trades") if r["method"] in (PUBLIC,COMBINED)}
    history = []
    for row,_ in inputs():
        issue,decision = (row[k] for k in ("release_date","decision"))
        if issue not in forecasts or (issue,PUBLIC) not in controls:
            continue
        f = forecasts[issue]
        if not f["cutoff"]<f["known"]<=decision<row["entry"] or f["actual"]!=row["x"][0]:
            raise ValueError(f"Inconsistent market and target clocks on {issue}")
        refs = (controls[issue,PUBLIC],controls[issue,COMBINED])
        for ref in refs:
            if (any(ref[k]!=row[k] for k in ("near_month","far_month")) or any(
                datetime.fromisoformat(ref[k])!=row[field] for k,field in (
                    ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))):
                raise ValueError(f"Tail policy changed the CL contract month or date on {issue}")
        baselines = tuple(float(r["expected_long_move_usd"]) for r in refs)
        # Replace the linear surprise with an inventory-equivalent value
        # derived from the rank of the first release in its predicted distribution.
        distribution_names = ("public","combined","matched_combined")
        percentiles = tuple(f["probabilities"][m][1] for m in distribution_names)
        implied_surprises = tuple(.9*float(ndtri(np.clip(p,.01,.99))) for p in percentiles)
        vectors = tuple(replace_expectation(row,f["actual"]-s) for s in implied_surprises)
        current = row | dict(vectors=vectors)
        train = [q for q in history if q["exit_at"]<decision][-HISTORY:]
        history.append(current)
        fitted = (baselines[0],baselines[1],baselines[1])
        if len(train)>=MIN_WINDOWS:
            fitted = tuple(risk_prediction([q | dict(x=q["vectors"][i]) for q in train],current | dict(x=x),len(x))
                           for i,x in enumerate(vectors))
        values = (*baselines,*fitted,(fitted[1]+baselines[1])/2)
        for i,(m,estimate) in enumerate(zip(POLICIES,values,strict=True)):
            q = trade(row,m,estimate,3,training_at=train[-1]["exit_at"] if train else None)
            if i<2 and (q["near_side"]!=int(refs[i]["near_side"]) or q["net_170_usd"]!=int(refs[i]["net_170_usd"])):
                raise ValueError(f"Original risk payoff differs on {issue}/{m}")
            yield q | dict(public_percentile=percentiles[0],combined_percentile=percentiles[1],
                           matched_percentile=percentiles[2],probability_training_weeks=f["training_weeks"],
                           latest_probability_training_label_utc=f["latest_label_utc"])


def main():
    with threadpool_limits(limits=1):
        rows = list(forecasts())
        trades = list(market(rows))
    report = dict(status="exploratory_on_reviewed_history",thresholds_million_bbl=LEVELS,forecast={},market={})
    for phase in PHASES:
        subset = [r for r in rows if r["study_phase"]==phase]
        report["forecast"][phase] = {name:{f"{x:g}":tail_score(group,x) for x in LEVELS} for name,group in (
            ("all",subset),("measured",[r for r in subset if r["covered"]]),("unmeasured",[r for r in subset if not r["covered"]])) if group}
        block = report["market"][phase] = {}
        reference = [r for r in trades if r["study_phase"]==phase and r["method"]=="combined_conditional"]
        for method in POLICIES:
            sample = [r for r in trades if r["study_phase"]==phase and r["method"]==method]
            if [r["release_date"] for r in sample]!=[r["release_date"] for r in reference]:
                raise ValueError(f"Mismatched trade calendar for {phase}/{method}")
            block[method] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample),
                changes_from_combined=sum(r["near_side"]!=q["near_side"] for r,q in zip(sample,reference,strict=True)),
                paired_pnl_delta_95_usd=block_bootstrap_mean([r["net_170_usd"]-q["net_170_usd"]
                    for r,q in zip(sample,reference,strict=True)]))
        print(phase,{m:round(v["brier"],4) for m,v in report["forecast"][phase]["all"]["1"]["models"].items()},
              {m:(v["net_170_usd"],v["net_340_usd"]) for m,v in block.items()})
    export(f"{STEM}_forecasts",[{k:r[k] for k in ("release_date","week_ending","study_phase","cutoff","known","actual","covered",
        "training_weeks","latest_label_utc")}
        | {f"{m}_historical_effective_weeks":v[2] for m,v in r["probabilities"].items()}
        | {f"{m}_observed_percentile":v[1] for m,v in r["probabilities"].items()}
        | {f"{m}_{x}_{label}":value for m,v in r["probabilities"].items() for x,p in v[0].items()
           for label,value in zip(("draw","build"),p,strict=True)} for r in rows])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register-release",type=date.fromisoformat)
    mode.add_argument("--score-release",type=date.fromisoformat)
    options = parser.parse_args()
    if options.register_release:
        print(json.dumps(register_probabilities(options.register_release.isoformat()),indent=2,allow_nan=False))
    elif options.score_release:
        print(json.dumps(score_registered(options.score_release.isoformat()),indent=2,allow_nan=False))
    else:
        main()
