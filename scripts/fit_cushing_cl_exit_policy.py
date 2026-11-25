"""Choose the exit session using only prices available before the fourth session."""

import json
import math
from collections import defaultdict
from datetime import datetime

from backtest_wti_cushing_spread import COSTS, NEW_YORK
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import PHASES, daily_risk, replay as delayed_calendar
from fit_cushing_cl_volatility import risk_prediction
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_volatility_interactions import features

STEM = "cushing_cl_exit_policy"
METHODS = ("session_five", "session_four", "forecast_last_session", "forecast_last_session_with_third_close",
           "forecast_last_session_with_third_close_and_inventory")


def replay(cases):
    anchors = {r["release_date"]:r for r in read("cushing_cl_volatility_interactions_trades") if r["method"]=="inventory_interaction"}
    times = {(r["release_date"],int(r["session_number"])):datetime.fromisoformat(r["settlement_reference_time"])
             for r in read("wti_cushing_event_bars") if r["leg"]=="near"}
    seen = []
    for r in cases:
        issue = r["release_date"]
        evaluated_from,close_four = (times[issue,n] for n in (3,4))
        decision = datetime.combine(close_four.date(),datetime.min.time(),NEW_YORK)
        if not r["entry"] < evaluated_from < decision < close_four < r["exit_at"]:
            raise ValueError(f"Invalid exit clocks on {issue}")
        move, four, five = r["spread_path_usd"][1:]
        sigma = r["volatility_usd"]
        _,_,_,inventory,_,curve = features(r)
        latest_move = move/sigma
        vectors = (curve,(*curve,latest_move),(*inventory,latest_move))
        current = dict(r, vectors=vectors, long_gross=five-four, volatility_usd=sigma/math.sqrt(3))
        past = [q for q in seen if q["exit_at"]<decision][-HISTORY:]
        seen.append(current)
        if issue not in anchors:
            continue
        if len(past)<MIN_WINDOWS:
            raise ValueError(f"Insufficient exit training history on {issue}")
        reference = anchors[issue]
        side = int(reference["near_side"])
        predictions = [risk_prediction([q | dict(x=q["vectors"][i]) for q in past],current | dict(x=x),len(x))
                       for i,x in enumerate(vectors)]
        for method,value in zip(METHODS,(None,None,*predictions),strict=True):
            go_early = method=="session_four" or (value is not None and side*value<0)
            # Both decisions incur the same closing cost. Buying or selling another
            # spread at the fourth session is outside this exit policy.
            gross = side*(four if go_early else five)
            path = [side*m-(170 if n==5 or (n==4 and go_early) else 85)*abs(side)
                    for n,m in enumerate(r["spread_path_usd"],2)]
            if go_early:
                path[-1] = path[-2]
            if path[-1]!=gross-170*abs(side) or (method=="session_five" and gross!=int(reference["gross_usd"])):
                raise ValueError(f"Exit accounting differs on {issue}/{method}")
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month")} | dict(
                method=method,near_side=side,volatility_usd=sigma,entry_at_utc=r["entry"].isoformat(),
                exit_information_mark_utc=evaluated_from.isoformat(),exit_decision_at_utc=decision.isoformat(),
                exit_at_utc=(close_four if go_early else r["exit_at"]).isoformat(),latest_exit_training_outcome_utc=past[-1]["exit_at"].isoformat(),
                expected_long_move_from_four_to_five_usd=value,exit_session_number=4 if go_early else 5,gross_usd=gross,
                alternate_source=r["alternate_source"] if not go_early else None,
                alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] and not go_early else None,
                **{f"session_{n}_net_170_usd":value for n,value in enumerate(path,2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    _,_,cases = delayed_calendar()
    rows = list(replay(cases))
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
    report = dict(status="historical_replay_on_reviewed_period",phases={})
    for p in PHASES:
        report["phases"][p] = {}
        base = groups[p,METHODS[0]]
        for method in METHODS:
            sample = groups[p,method]
            if [r["release_date"] for r in sample]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different exit-method calendars in {p}/{method}")
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(sample,base,strict=True)]
            report["phases"][p][method] = robustness(sample) | daily_risk(sample) | dict(
                early_exits=sum(r["exit_session_number"]==4 and r["near_side"]!=0 for r in sample),
                increment_usd=sum(delta),delta_mean_95_usd=block_bootstrap_mean(delta))
        print(p,{m:(v["trades"],v["early_exits"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in report["phases"][p].items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
