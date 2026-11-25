"""Test whether a Friday forecast justifies continuing a position through the next report."""

import json
import math
from collections import defaultdict
from datetime import datetime, time

from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_following_report import prerelease
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import RISK_FLOOR_USD, risk_prediction
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_late_pair_recovery import COMBINED, STEM as LATE
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_robust_stack import indexed
from publication_clocks import EASTERN

STEM = "cushing_cl_paired_following_forecasts"
METHODS = ("hold", "exit_before", "risk_gate", "forecast_gate", "forecast_loss_guard", "optical_loss_guard")
STRATEGIES = {"combined":(LATE,COMBINED),"historical_leader":(OUTER,f"{SUPPLEMENT}_conditional")}


def current_strategies():
    return {name:{r["release_date"]:r for r in read(f"{source}_trades") if r["method"]==method}
            for name,(source,method) in STRATEGIES.items()}


def extension(row, next_row, early, current):
    index = row["last_before_next_release"]
    if index not in (1,2) or not row["stamps"][index]<row["next_report_clock"]<=row["exit_at"]:
        return None
    decision = datetime.combine(row["stamps"][index].date(),time.min,EASTERN)
    if not row["entry"]<decision<row["stamps"][index]:
        raise ValueError(f"Invalid exit timing on {row['release_date']}")
    if not early or early["cutoff"]>decision:
        return None
    if not early["cutoff"]<row["next_report_clock"]<early["known"]:
        raise ValueError(f"Premature next report target on {row['release_date']}")
    delayed = next_row is not None and datetime.fromisoformat(next_row["cutoff"])<=decision
    # Late pairs following the survey are usable for a Monday/Tuesday holding
    # decision only if their archived forecast was already fully available.
    # Otherwise use the earlier archived forecast, which has its own cutoff.
    public = float(next_row[PUBLIC]) if delayed else early["public"]
    optical = float(next_row[COMBINED]) if delayed else early["optical"]
    used_cutoff = datetime.fromisoformat(next_row["cutoff"]) if delayed else early["cutoff"]
    sigma = max(RISK_FLOOR_USD,row["volatility_usd"]*math.sqrt((3-index)/3))
    target,_,carry,change,*_ = row["x"]
    state = (carry,change,math.log(sigma/1000),target,row["spread_path_usd"][index-1]/sigma,
             float(current["expected_long_move_usd"])/sigma)
    prediction_set = (state,(*state,public,public*state[2]),(*state,public,public*state[2],optical-public))
    return row | dict(decision=decision,next_forecast_at=used_cutoff,early_optical_roofs=early["roofs"],forecast_uses_late_pairs=delayed,
        volatility_usd=sigma,long_gross=row["spread_path_usd"][-1]-row["spread_path_usd"][index],design=prediction_set)


def replay(windows, initial, late, early, strategy):
    history = []
    for row,_ in windows:
        issue = row["release_date"]
        if issue not in initial:
            continue
        ref = initial[issue]
        if (any(ref[k]!=row[k] for k in ("near_month","far_month")) or any(
                datetime.fromisoformat(ref[k])!=row[field] for k,field in (
                ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))):
            raise ValueError(f"Different CL positions on {issue}")
        value = float(ref["expected_long_move_usd"])
        side = int(ref["near_side"])
        before = row["last_before_next_release"]
        do_exit = before in (1,2) and row["stamps"][before]<row["next_report_clock"]<=row["exit_at"]
        context = extension(row,late.get(row["next_issue"]),early.get(row["next_issue"]),ref) if do_exit else None
        trained = []
        predictions = None
        if context is not None:
            trained = [q for q in history if q["exit_at"]<context["decision"]][-HISTORY:]
            history.append(context)
            if len(trained)>=MIN_WINDOWS:
                predictions = tuple(risk_prediction([q | dict(x=q["design"][i]) for q in trained],
                    context | dict(x=x),len(x)) for i,x in enumerate(context["design"]))
        if do_exit:
            indices = [3,before,3,3,3,3]
            if predictions is not None:
                a,b,c = (side*x for x in predictions)
                cap = .10*context["volatility_usd"]
                indices[2:] = [3 if a>0 else before, 3 if b>0 else before,
                               before if b< -cap else 3,before if c< -cap else 3]
        else:
            indices = [3]*len(METHODS)
        for method,index in zip(METHODS,indices,strict=True):
            result = trade(row,method,value,index)
            if result["near_side"]!=side or (method=="hold" and result["net_170_usd"]!=int(ref["net_170_usd"])):
                raise ValueError(f"Baseline entry or position changed on {issue}")
            yield result | dict(strategy=strategy, next_forecast_at_utc=context["next_forecast_at"].isoformat() if context else None,
                holding_decision_at_utc=context["decision"].isoformat() if context else None,
                latest_training_exit_for_holding_utc=trained[-1]["exit_at"].isoformat() if trained else None,
                forecast_uses_late_pairs=bool(context and context["forecast_uses_late_pairs"]),
                opted_out_before_following_release=bool(side and index!=3),
                horizon_curve_usd=predictions[0] if predictions is not None else None,
                horizon_public_usd=predictions[1] if predictions is not None else None,
                horizon_optical_usd=predictions[2] if predictions is not None else None)


def main():
    with threadpool_limits(limits=1):
        observed = list(inputs())
        late, early = indexed(f"{LATE}_forecasts"), prerelease()
        rows = [r for name, strategy in current_strategies().items() for r in replay(observed,strategy,late,early,name)]
    report = dict(status="exploratory_on_reviewed_history",phases={})
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["strategy"],r["method"]].append(r)
    for phase in PHASES:
        report["phases"][phase] = {}
        for strategy in STRATEGIES:
            reference = groups[phase,strategy,"hold"]
            block = report["phases"][phase][strategy] = {}
            for method in METHODS:
                sample = groups[phase,strategy,method]
                if [r["release_date"] for r in sample]!=[r["release_date"] for r in reference]:
                    raise ValueError(f"Different dates for {method}/{phase}/{strategy}")
                delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(sample,reference,strict=True)]
                block[method] = robustness(sample) | daily_risk(sample) | dict(
                    net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample),
                    changed_exits=sum(r["opted_out_before_following_release"] for r in sample),
                    usable_forecasts_among_positions=sum(bool(r["near_side"] and r["next_forecast_at_utc"]) for r in sample),
                    late_forecasts_among_positions=sum(bool(r["near_side"] and r["forecast_uses_late_pairs"]) for r in sample),
                    fitted_forecasts_among_positions=sum(bool(r["near_side"] and r["horizon_curve_usd"] is not None) for r in sample),
                    paired_delta_vs_hold_95_usd=block_bootstrap_mean(delta))
            print(phase,strategy,{m:(v["net_170_usd"],v["net_340_usd"],v["changed_exits"]) for m,v in block.items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
