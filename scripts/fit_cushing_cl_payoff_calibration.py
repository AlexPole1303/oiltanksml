"""Calibrate the expected payoff of the published-inventory direction using closed windows."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.linalg import solve

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_volatility import HURDLE, PHASES
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_cl_payoff_calibration"
METHODS = ("inventory", "payoff_recalibration", "curve_agreement_payoff", "soft_curve_agreement")
MIN_SETTLED = 52
DOMAINS = ("cushing_regional_balance_trades", "cushing_optical_reliability_trades")


def calibration(history, current, extended):
    """Fit a shrunk correction to the model's normalized directional payoff."""
    fields = 4 if extended else 2
    x = np.array([r["vector"][:fields] for r in history])
    y = np.array([r["normalized_payoff"]-r["vector"][0] for r in history])
    age = np.array([(current["decision"]-r["decision"]).days/7 for r in history])
    time_weight = np.exp2(-age/78.)
    centre = np.average(x,axis=0,weights=time_weight)
    scale = np.maximum(np.sqrt(np.average(np.square(x-centre),axis=0,weights=time_weight)),.10)
    design = np.column_stack((np.ones(len(x)),np.clip((x-centre)/scale,-4.,4.)))
    weight = time_weight
    penalty = np.diag([.15]+[1.25]*fields)*sum(time_weight)
    for _ in range(5):
        beta = solve(design.T@(weight[:,None]*design)+penalty,design.T@(weight*y),assume_a="pos")
        weight = time_weight/np.hypot(1.,(y-design@beta)/.75)
    z = np.r_[1.,np.clip((np.asarray(current["vector"][:fields])-centre)/scale,-4.,4.)]
    return float(np.clip(z@beta,-.30,.30))*current["volatility_usd"]


def replay(cases):
    prior = {r["release_date"]:r for r in read(DOMAINS[0]) if r["method"]=="public_expectation"}
    curve = {r["release_date"]:r for r in read(DOMAINS[1]) if r["method"]=="curve"}
    seen = []
    for r in cases:
        issue = r["release_date"]
        if issue not in prior:
            continue
        a,b = prior[issue],curve[issue]
        if any(datetime.fromisoformat(q["decision_at_utc"])!=r["decision"] or datetime.fromisoformat(q[field])>=r["decision"]
               for q,field in ((a,"latest_training_exit"),(b,"settled_training_label_utc"))):
            raise ValueError(f"Input signal used an unsettled price on {issue}")
        inventory, carry = (float(q["expected_long_move_usd"]) for q in (a,b))
        side = 1 if inventory>=0 else -1
        sigma = r["volatility_usd"]
        vector = (abs(inventory)/sigma,math.log(sigma/1000),side*carry/sigma,abs(inventory-carry)/sigma)
        current = r | dict(vector=vector,normalized_payoff=side*r["long_gross"]/sigma)
        history = [q for q in seen if q["exit_at"]<r["decision"]][-156:]
        seen.append(current)
        adjustments = tuple(calibration(history,current,extra) if len(history)>=MIN_SETTLED else 0. for extra in (False,True))
        estimates = (abs(inventory),abs(inventory)+adjustments[0],abs(inventory)+adjustments[1],abs(inventory)+adjustments[1]/2)
        for method,value in zip(METHODS,estimates,strict=True):
            direction = side if value>HURDLE else 0
            gross = direction*r["long_gross"]
            if method=="inventory" and (direction!=int(a["near_side"]) or gross!=int(a["gross_usd"])):
                raise ValueError(f"Original trading decision changed on {issue}")
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd","alternate_source")} | dict(
                decision_at_utc=r["decision"].isoformat(),entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),
                latest_calibration_outcome_utc=history[-1]["exit_at"].isoformat() if history else None,
                calibration_windows=len(history),method=method,expected_gross_in_direction_usd=value,original_long_forecast_usd=inventory,
                curve_long_forecast_usd=carry,near_side=direction,gross_usd=gross,
                alternate_gross_usd=direction*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":direction*m-(170 if n==5 else 85)*abs(direction)
                   for n,m in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(direction) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(direction) for c in COSTS})


def main():
    _,_,cases = market_calendar()
    rows = list(replay(cases))
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
    report = dict(status="historical_walk_forward_on_reviewed_period",phases={})
    for phase in PHASES:
        base = groups[phase,"inventory"]
        result = {}
        for method in METHODS:
            group = groups[phase,method]
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different market calendars for {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(group,base,strict=True)]
            result[method] = robustness(group) | daily_risk(group) | dict(increment_usd=sum(delta),
                paired_delta_95_usd=block_bootstrap_mean(delta),changed_trades=sum(r["near_side"]!=q["near_side"]
                    for r,q in zip(group,base,strict=True)),net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group))
        report["phases"][phase] = result
        print(phase,{m:(q["trades"],q["net_170_usd"],q["daily_drawdown_usd"]) for m,q in result.items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
