"""Learn corrections to CL spread forecasts from previously settled similar reports."""

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
from fit_cushing_cl_volatility import HURDLE, PHASES
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PUBLIC

STEM = "cushing_cl_conditional_response"
METHODS = ("curve", "inventory", "optical", "curve_conditional", "inventory_conditional", "optical_conditional")
SOURCE_METHODS = ("curve", PUBLIC, "scene_direct")


def features(carry, volatility, surprise=None):
    market = (carry, math.log(volatility/1000))
    return market if surprise is None else (*market,surprise)


def conditional_residual(history, row):
    """Estimate a regularized correction to the scaled, earlier model predictions."""
    vectors = np.array([q["response_features"] for q in history])
    current = np.asarray(row["response_features"])
    # Columns stand for carry, log volatility, and optionally inventory surprise.
    distance_scale = np.maximum(vectors.std(axis=0), np.array((.20,.40,.40))[:len(current)])
    distance = np.square((vectors-current)/distance_scale).mean(axis=1)
    age = np.array([(row["decision"]-q["decision"]).days/7 for q in history])
    weights = np.exp2(-age/78)*np.exp(-.5*distance)
    residuals = np.array([(q["long_gross"]-q["base_estimate"])/q["volatility_usd"] for q in history])
    # Prior weight is the equivalent of 26 exact matches with zero correction.
    revision = float(weights@np.clip(residuals,-3.,3.)/(weights.sum()+26.))
    return row["volatility_usd"]*revision, float(weights.sum())


def base_inputs(cases):
    forecasts = {r["release_date"]:r for r in read("cushing_optical_stability_forecasts")}
    signals = {(r["release_date"],r["method"]):r for r in read("cushing_optical_stability_trades") if r["method"] in SOURCE_METHODS}
    for r in cases:
        issue = r["release_date"]
        if (issue, PUBLIC) not in signals:
            continue
        ref = [signals[issue,key] for key in SOURCE_METHODS]
        for q in ref:
            if (datetime.fromisoformat(q["settled_training_label_utc"]) >= r["decision"] or
                any(datetime.fromisoformat(q[k]) != r[field] for k,field in (
                    ("decision_at_utc","decision"), ("entry_at_utc","entry"), ("exit_at_utc","exit_at"))) or
                any(q[k] != r[k] for k in ("near_month","far_month"))):
                raise ValueError(f"Base forecast clock or delivery months differ on {issue}")
        f = forecasts[issue]
        cutoff, known = [datetime.fromisoformat(f[k]) for k in ("cutoff_utc","known_utc")]
        actual = float(f["actual"])
        if not (cutoff < known <= r["decision"] < r["entry"]) or actual != r["x"][0]:
            raise ValueError(f"Inventory release differs on {issue}")
        public_surprise, optical_surprise = (actual-float(f[k]) for k in (PUBLIC,"scene_direct"))
        columns = [features(r["x"][2],r["volatility_usd"],value) for value in (None,public_surprise,optical_surprise)]
        yield ref, [r | dict(base_estimate=float(q["expected_long_move_usd"]),response_features=x) for q,x in zip(ref,columns,strict=True)]


def replay(cases):
    seen = []
    for ref, models in base_inputs(cases):
        r = models[0]
        issue = r["release_date"]
        past = [q for q in seen if q[0]["exit_at"] < r["decision"]][-HISTORY:]
        seen.append(models)
        base_estimates = tuple(q["base_estimate"] for q in models)
        fitted = [conditional_residual([q[i] for q in past],current) if len(past) >= MIN_WINDOWS else (0.,0.)
                  for i,current in enumerate(models)]
        estimates = (*base_estimates, *(base+adj for base,(adj,_) in zip(base_estimates,fitted,strict=True)))
        for i,(method,estimate) in enumerate(zip(METHODS,estimates,strict=True)):
            side = int(estimate > HURDLE)-int(estimate < -HURDLE)
            gross = side*r["long_gross"]
            if i < 3 and (side != int(ref[i]["near_side"]) or gross != int(ref[i]["gross_usd"])):
                raise ValueError(f"Base trade differs on {issue}/{method}")
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd","alternate_source")} | dict(
                method=method, decision_at_utc=r["decision"].isoformat(), entry_at_utc=r["entry"].isoformat(), exit_at_utc=r["exit_at"].isoformat(),
                latest_calibration_exit_utc=past[-1][0]["exit_at"].isoformat() if past else None, calibration_samples=len(past),
                similar_report_weight=fitted[i-3][1] if i>=3 else None, expected_long_move_usd=estimate, near_side=side,
                gross_usd=gross, alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*v-(170 if n==5 else 85)*abs(side) for n,v in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    _,_,cases = market_calendar()
    trades = list(replay(cases))
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["method"]].append(r)
    report = dict(status="historical_walk_forward_on_reviewed_period", phases={})
    for phase in PHASES:
        incumbent = groups[phase,"inventory"]
        comparison = report["phases"][phase] = {}
        for i, method in enumerate(METHODS):
            rows, base = groups[phase,method], groups[phase,METHODS[i%3]]
            if [r["release_date"] for r in rows] != [r["release_date"] for r in incumbent]:
                raise ValueError(f"Trading windows differ for {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(rows,base,strict=True)]
            comparison[method] = robustness(rows) | daily_risk(rows) | dict(
                base=METHODS[i%3], incremental_net_usd=sum(delta), paired_increment_95_usd=block_bootstrap_mean(delta),
                changed_directions=sum(r["near_side"]!=q["near_side"] for r,q in zip(rows,base,strict=True)),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in rows))
        print(phase, {m:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in comparison.items()})
    export(f"{STEM}_trades", trades)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
