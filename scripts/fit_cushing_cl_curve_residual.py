"""Learn the incremental spread move beyond a prequential curve forecast."""

import json
import math
from collections import defaultdict
from datetime import datetime

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_curve_residual"
POLICIES = ("curve", "incumbent", "public_residual", "optical_residual", "shrunken_residual")


def replay(cases):
    # Index the curve and inventory predictions by model and date. Both
    # estimates were issued before the target move entered either training set.
    original = {(r["release_date"],r["method"]):r for r in read("cushing_optical_reliability_trades")
                if r["method"] in ("curve","public_ensemble")}
    photo = indexed("cushing_shared_scene_forecasts")
    seen = []
    for r in cases:
        issue = r["release_date"]
        if (issue,"curve") not in original:
            continue
        base,incumbent = (original[issue,k] for k in ("curve","public_ensemble"))
        if any(datetime.fromisoformat(q["settled_training_label_utc"])>=r["decision"] or
               datetime.fromisoformat(q["decision_at_utc"])!=r["decision"] for q in (base,incumbent)):
            raise ValueError(f"Curve estimate used a late outcome on {issue}")
        f = photo[issue]
        if not datetime.fromisoformat(f["forecast_cutoff_utc"])<datetime.fromisoformat(f["target_known_by_utc"])<=r["decision"]:
            raise ValueError(f"Inventory outcome not yet published on {issue}")
        curve,anchor = (float(q["expected_long_move_usd"]) for q in (base,incumbent))
        target,_,slope,*_ = r["x"]
        sigma = r["volatility_usd"]
        expectation = float(f["public_ensemble"])
        volume,hybrid = target-expectation,target-float(f["carried_scene"])
        vol = math.log(sigma/1000)
        designs = [(target,s,s*slope,target*vol,s*vol) for s in (volume,hybrid)]
        # Float residuals are dollars on the original units. Huber fitting
        # normalizes them by the risk estimate known at the event.
        current = r | dict(designs=designs,long_gross=r["long_gross"]-curve)
        past = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        seen.append(current)
        values = [curve,anchor]
        if len(past)>=MIN_WINDOWS:
            for i,x in enumerate(designs):
                residual = risk_prediction([q | dict(x=q["designs"][i]) for q in past],current | dict(x=x),len(x))
                values.append(curve+residual)
            # Equal-weight shrinkage controls estimation variance of the separate residual model.
            values.append((values[2]+anchor)/2)
        else:
            values.extend([anchor]*3)
        for method,value in zip(POLICIES,values,strict=True):
            side = int(value>HURDLE)-int(value<-HURDLE)
            gross = side*r["long_gross"]
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd","alternate_source")} | dict(
                method=method,decision_at_utc=r["decision"].isoformat(),entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),
                residual_training_windows=len(past),latest_residual_outcome_utc=past[-1]["exit_at"].isoformat() if past else None,
                forecast_long_move_usd=value,near_side=side,gross_usd=gross,
                alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*m-(170 if n==5 else 85)*abs(side) for n,m in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    _,_,cases = market_calendar()
    rows = list(replay(cases))
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
    report = dict(status="historical_walk_forward_on_reviewed_period",phases={})
    for phase in PHASES:
        base = groups[phase,"incumbent"]
        result = {}
        for method in POLICIES:
            group = groups[phase,method]
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Market calendars differ for {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(group,base,strict=True)]
            result[method] = robustness(group) | daily_risk(group) | dict(increment_usd=sum(delta),paired_delta_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group))
        report["phases"][phase] = result
        print(phase,{m:(v["windows"],v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in result.items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
