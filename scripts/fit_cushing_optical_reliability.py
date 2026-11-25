"""Calibrate inventory updates using the actual spatial and temporal optical coverage."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features

STEM = "cushing_optical_reliability"
MODELS = (PUBLIC,"carried_scene","global_weight","coverage_weight","coverage_verified_weight")
POLICIES = ("curve",*MODELS)
MIN_MEASURED = 32
COVERAGE_SCALE = np.array([.55,.35,.55])


def indexed_measurements():
    data = defaultdict(list)
    for r in read("cushing_observation_operator_measurements"):
        data[r["release_date"]].append(dict(r, sensitivity=float(r["sensitivity"]),area=float(r["area"])))
    return data


def coverage(measured, area_fraction):
    if not measured:
        return np.zeros(3)
    terminals = defaultdict(float)
    for r in measured:
        if r["sensitivity"]<=0 or r["area"]<=0 or not r["terminal_id"]:
            raise ValueError(f"Invalid measured area on {r['candidate_id']}")
        terminals[r["terminal_id"]] += r["sensitivity"]
    shares = np.array(list(terminals.values()))
    total = shares.sum()
    if not 0<total<=area_fraction+1e-10:
        raise ValueError("The observed area-time exposure exceeds the mapped area")
    return np.array([float(total),float(total/area_fraction),float(total**2/(shares@shares))])


def source_rows():
    pairs = indexed_measurements()
    for issue,r in sorted(indexed("cushing_shared_scene_forecasts").items()):
        cutoff,known = (datetime.fromisoformat(r[k]) for k in ("forecast_cutoff_utc","target_known_by_utc"))
        observed = pairs[issue]
        if not cutoff<known or len(observed)!=int(r["measured_roofs"]) or any(
            datetime.fromisoformat(q["available_utc"])>=cutoff for q in observed
        ):
            raise ValueError(f"Invalid input coverage or time for {issue}")
        for key in ("latest_training_label_utc","latest_available_scene_label_utc"):
            if r[key] and datetime.fromisoformat(r[key])>=cutoff:
                raise ValueError(f"Upstream satellite fit contains a late report on {issue}")
        measures = coverage(observed,float(r["area_fraction"]))
        # Compare how much of the hub was measured, how far into the week
        # the observed pairs reach, and how spread out the measurement is.
        coordinates = np.array([math.log1p(measures[0]/.04),measures[1],math.log1p(measures[2])])
        yield {k:r[k] for k in ("release_date","week_ending","study_phase")} | dict(
            actual=float(r["actual"]),cutoff=cutoff,known=known,observed=len(observed),
            public_ensemble=float(r[PUBLIC]),carried_scene=float(r["carried_scene"]),raw=float(r["carried_raw"]),
            spatial_fraction=float(r["area_fraction"]),quality=measures,coordinates=coordinates)


def betas(history, row):
    x = np.asarray([r["raw"] for r in history])
    y = np.asarray([r["actual"]-r[PUBLIC] for r in history])
    coords = np.asarray([r["coordinates"] for r in history])
    age = np.array([(row["cutoff"]-r["cutoff"]).days/7 for r in history])
    weight = np.exp2(-age/52.)
    local = np.exp(-.5*np.square((coords-row["coordinates"])/COVERAGE_SCALE).sum(axis=1))
    # Retain a pooled estimate when local history is sparse. A week with a
    # tiny, concentrated footprint cannot set its own unconstrained slope.
    weights = np.stack((weight,weight*(.35+.65*local)))
    w = weights.copy()
    for _ in range(6):
        b = np.clip((w@(x*y))/(w@(x*x)+.05*weights.sum(axis=1)),0.,1.5)
        w = weights/np.hypot(1.,(y-b[:,None]*x)/.6)
    effective_neighbors = float((weight@local)**2/max(np.sum(np.square(weight*local)),1e-12))
    return b,effective_neighbors


def replay(rows):
    seen = []
    for r in rows:
        cutoff = r["cutoff"]
        past = [q for q in seen if q["observed"] and q["known"]<cutoff][-104:]
        weights = np.zeros(2)
        support = 0.
        neighbor_count = 0.
        if len(past)>=MIN_MEASURED and r["observed"]:
            weights,neighbor_count = betas(past,r)
            nearest = np.stack([q["coordinates"] for q in past])-r["coordinates"]
            nearby = np.exp(-.5*np.square(nearest/COVERAGE_SCALE).sum(axis=1))
            # Use stored walk-forward predictions rather than fitted training residuals.
            valid = np.array([q["estimated"] for q in past]) & (nearby>=.25)
            actual,goal,base = (np.array([q[field] for q in past])[valid] for field in ("actual","coverage_weight",PUBLIC))
            if len(goal)>=16:
                improvement = np.abs(actual-base)-np.abs(actual-goal)
                se = max(float(np.std(improvement,ddof=1)/np.sqrt(len(improvement))),1e-9)
                support = float(np.clip(improvement.mean()/se,0.,1.))
        base = r[PUBLIC]
        fraction_threshold = float(np.median([q["quality"][0] for q in past])) if len(past)>=MIN_MEASURED else None
        above = bool(r["observed"] and fraction_threshold is not None and r["quality"][0]>=fraction_threshold)
        output = r | dict(global_weight=base+r["raw"]*weights[0],coverage_weight=base+r["raw"]*weights[1],
            coverage_verified_weight=base+r["raw"]*weights[1]*support,global_beta=float(weights[0]),local_beta=float(weights[1]),
            local_multiplier=support,effective_neighbors=neighbor_count,above_prior_median=above,
            prior_median_area_time=fraction_threshold,estimated=len(past)>=MIN_MEASURED and bool(r["observed"]),
            training_weeks=len(past),last_label_utc=past[-1]["known"].isoformat() if past else None)
        seen.append(output)
        yield output


def forecast_metrics(rows, models=MODELS):
    if not rows:
        return dict(weeks=0,models={})
    y = np.array([r["actual"] for r in rows])
    predictions = {m:np.array([r[m] for r in rows]) for m in models}
    errors = {m:1000*np.abs(y-p) for m,p in predictions.items()}
    return dict(weeks=len(rows),models={m:dict(mae_kbbl=float(err.mean()),direction=int(np.count_nonzero(predictions[m]*y>0)),
        delta_vs_public_95_kbbl=block_bootstrap_mean(err-errors[PUBLIC]),
        delta_vs_carried_95_kbbl=block_bootstrap_mean(err-errors["carried_scene"])) for m,err in errors.items()})


def trading(rows, models=MODELS):
    _,_,cases = market_calendar()
    lookup = {r["release_date"]:r for r in rows}
    incumbent = {r["release_date"]:r for r in read("cushing_regional_balance_trades") if r["method"]=="public_expectation"}
    seen = []
    for r in cases:
        issue = r["release_date"]
        f = lookup[issue]
        if not f["cutoff"]<f["known"]<=r["decision"]<r["entry"]<r["exit_at"] or f["actual"]!=r["x"][0]:
            raise ValueError(f"Invalid EIA or CL trading clocks on {issue}")
        past = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        design = (features(r)[5],*(replace_expectation(r,f[k]) for k in models))
        current = r | dict(design=design)
        seen.append(current)
        if len(past)<MIN_WINDOWS:
            continue
        for i,(method,x) in enumerate(zip(("curve",*models),design,strict=True)):
            value = risk_prediction([q | dict(x=q["design"][i]) for q in past],current | dict(x=x),len(x))
            if method==PUBLIC and not math.isclose(value,float(incumbent[issue]["expected_long_move_usd"]),abs_tol=1e-8):
                raise ValueError(f"Incumbent strategy changed on {issue}")
            side = int(value>HURDLE)-int(value<-HURDLE)
            gross = side*r["long_gross"]
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd","alternate_source")} | dict(
                method=method,decision_at_utc=r["decision"].isoformat(),entry_at_utc=r["entry"].isoformat(),
                exit_at_utc=r["exit_at"].isoformat(),settled_training_label_utc=past[-1]["exit_at"].isoformat(),
                expected_long_move_usd=value,near_side=side,gross_usd=gross,
                alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*m-(170 if n==5 else 85)*abs(side) for n,m in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    rows = list(replay(source_rows()))
    trades = list(trading(rows))
    report = dict(status="historical_walk_forward_on_reviewed_period",predictions={},trading={})
    for phase in PHASES:
        phase_rows = [r for r in rows if r["study_phase"]==phase]
        measured = [r for r in phase_rows if r["observed"]]
        report["predictions"][phase] = {scope:forecast_metrics(sample) for scope,sample in (
            ("all",phase_rows),("measured",measured),("above_prior_median",[r for r in measured if r["above_prior_median"]]),
            ("below_prior_median",[r for r in measured if r["estimated"] and not r["above_prior_median"]]))}
        group = {m:[r for r in trades if r["study_phase"]==phase and r["method"]==m] for m in POLICIES}
        base = group[PUBLIC]
        result = {}
        for method,sample in group.items():
            if [r["release_date"] for r in sample]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different calendars for {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(sample,base,strict=True)]
            result[method] = robustness(sample) | daily_risk(sample) | dict(increment_usd=sum(delta),
                paired_delta_95_usd=block_bootstrap_mean(delta),
                changed_directions=sum(r["near_side"]!=q["near_side"] for r,q in zip(sample,base,strict=True)))
        report["trading"][phase] = result
        print(phase,{s:(d["weeks"],{m:round(v["mae_kbbl"],2) for m,v in d["models"].items()}) for s,d in report["predictions"][phase].items()})
        print(phase,{m:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in result.items()})
    export(f"{STEM}_forecasts",[{k:r[k] for k in ("release_date","week_ending","study_phase","actual","observed","raw","spatial_fraction",
        "global_beta","local_beta","local_multiplier","effective_neighbors","above_prior_median","prior_median_area_time","estimated",
        "training_weeks","last_label_utc",*MODELS)} | dict(cutoff_utc=r["cutoff"].isoformat(),target_known_at_utc=r["known"].isoformat(),
        area_time=float(r["quality"][0]),average_fraction_of_week=float(r["quality"][1]),effective_terminals=float(r["quality"][2])) for r in rows])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
