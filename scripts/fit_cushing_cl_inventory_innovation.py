"""Test curve-adjusted EIA inventory surprises on the dated CL calendar."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import PHASES
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PUBLIC
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features

STEM = "cushing_cl_inventory_innovation"
HYBRID = "basis_when_unmeasured"
MODELS = ("curve", "public_conditional", "reported_change", "public_surprise", "hybrid_surprise",
          "normalized_hybrid_surprise", "prior_report_control", "asymmetric_hybrid_surprise",
          "asymmetric_hybrid_blend")
ANCHOR = "public_conditional"


def forecasts():
    history = []
    for issue, q in sorted(indexed("cushing_public_state_forecasts").items()):
        cutoff, known = (datetime.fromisoformat(q[k]) for k in ("cutoff", "known"))
        ts = q["latest_training_label_utc"]
        if not cutoff < known or (ts and datetime.fromisoformat(ts) >= cutoff):
            raise ValueError(f"Bad release timeline on {issue}")
        actual, public, hybrid = (float(q[k]) for k in ("actual", PUBLIC, HYBRID))
        prior = [p for p in history if p["known"] < cutoff][-104:]
        # A robust scale keeps a large surprise from dominating the feature.
        errors = np.array([abs(p["hybrid_error"]) for p in prior])
        scale = max(.35, float(np.median(errors)) / .67449) if len(prior) >= 26 else 1.
        previous = prior[-1] if prior else dict(hybrid_error=0.)
        yield issue, dict(cutoff=cutoff, known=known, actual=actual, public=public, hybrid=hybrid,
                          scale=scale, prior_error=previous["hybrid_error"],
                          latest_scale_known_utc=prior[-1]["known"].isoformat() if prior else None)
        history.append(dict(known=known, hybrid_error=actual-hybrid))


def innovation(train, row):
    """Remove the curve's linear information from each inventory feature."""
    market, inventory, residual = (np.array([q[k] for q in train]) for k in ("market", "inventory", "residual"))
    elapsed = np.array([(row["decision"]-q["decision"]).total_seconds()/604800 for q in train])
    weights = np.exp2(-elapsed/78)
    weights /= weights.sum()
    center = weights @ market
    scale = np.maximum(np.sqrt(weights @ np.square(market-center)),.15)
    design = np.column_stack((np.ones(len(train)),np.clip((market-center)/scale,-4,4)))
    z = np.r_[1.,np.clip((row["market"]-center)/scale,-4,4)]
    # Leave the intercept unpenalized. Shrink the projected curve slopes.
    regularizer = np.diag([0., *([.20]*market.shape[1])])
    coefficients = solve(design.T@(weights[:,None]*design)+regularizer,
                         design.T@(weights[:,None]*inventory),assume_a="pos")
    centered = inventory-design@coefficients
    exogenous = row["inventory"]-z@coefficients
    amplitude = np.maximum(np.sqrt(weights@np.square(centered)),.35)
    x = np.clip(centered/amplitude,-4,4)
    z = np.clip(exogenous/amplitude,-4,4)
    y = np.clip(residual,-3.,3.)
    weight = weights[:,None]*np.ones(x.shape[1])
    coefficients = np.zeros(x.shape[1])
    # Fit each surprise independently, without an inventory-free intercept;
    # this isolates its contribution over the archived curve prediction.
    for _ in range(6):
        coefficients = (np.sum(weight*x*y[:,None],axis=0)
                        / (np.sum(weight*np.square(x),axis=0)+.65))
        weight = weights[:,None]/np.hypot(1.,(y[:,None]-x*coefficients)/.7)
    scaled = z*coefficients
    # Asymmetric effects can differ for an unexpected build and draw.
    # Split the curve-adjusted normalized surprise around zero, retain the
    # same regularization on both slopes, and allow no free return drift.
    tail = x[:,3]
    asymmetric = np.column_stack((np.minimum(tail,0.),np.maximum(tail,0.)))
    weight = weights.copy()
    penalty = .65*np.eye(2)
    for _ in range(6):
        beta = solve(asymmetric.T@(weight[:,None]*asymmetric)+penalty,asymmetric.T@(weight*y),assume_a="pos")
        weight = weights/np.hypot(1.,(y-asymmetric@beta)/.7)
    split = np.array((min(z[3],0.),max(z[3],0.)))
    return row["curve"] + row["volatility_usd"]*np.clip(np.r_[scaled,split@beta],-.75,.75), exogenous


def replay():
    book = dict(forecasts())
    market_book = {q["release_date"]:q for q in read("cushing_cl_volatility_interactions_trades")
                   if q["method"]=="carry_volatility_context"}
    history = []
    for r,anchors in inputs():
        issue = r["release_date"]
        if "inventory_conditional" not in anchors:
            continue
        anchor, curve = anchors["inventory_conditional"],market_book[issue]
        f = book[issue]
        for ref in (anchor,curve):
            if any(ref[k]!=r[k] for k in ("near_month","far_month")) or any(
                datetime.fromisoformat(ref[k])!=r[field] for k,field in (
                    ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")
                )):
                raise ValueError(f"Different dated trading row on {issue}")
        if (not f["cutoff"] < f["known"] <= r["decision"] or f["actual"] != r["x"][0]
                or datetime.fromisoformat(curve["latest_training_exit_utc"]) >= r["decision"]
                or anchor["latest_calibration_exit_utc"]
                and datetime.fromisoformat(anchor["latest_calibration_exit_utc"]) >= r["decision"]):
            raise ValueError(f"Out-of-order release labels on {issue}")
        x = float(curve["expected_long_gross_usd"])
        surprise = f["actual"]-f["hybrid"]
        now = r | dict(curve=x, residual=(r["long_gross"]-x)/r["volatility_usd"],
            market=np.array(features(r)[5]), inventory=np.array((f["actual"], f["actual"]-f["public"],
                surprise,surprise/f["scale"],f["prior_error"])))
        train = [q for q in history if q["exit_at"] < r["decision"]][-HISTORY:]
        history.append(now)
        prediction, adjusted = innovation(train,now) if len(train) >= MIN_WINDOWS else (np.full(6,x),np.zeros(5))
        baseline = float(anchor["expected_long_move_usd"])
        fitted = (x,baseline,*prediction,(prediction[-1]+baseline)/2)
        for i,(name,move) in enumerate(zip(MODELS,fitted,strict=True)):
            q = trade(r,name,float(move),3,training_at=train[-1]["exit_at"] if train and i>=2 else None)
            if i < 2:
                ref = (curve,anchor)[i]
                if q["near_side"]!=int(ref["near_side"]) or q["gross_usd"]!=int(ref["gross_usd"]):
                    raise ValueError(f"Control did not reproduce on {issue}/{name}")
            yield q | dict(base_curve_expected_usd=x,held_spread_move_usd=r["long_gross"],
                raw_inventory_feature=float(now["inventory"][i-2]) if 2<=i<7 else None,
                curve_adjusted_inventory=float(adjusted[i-2]) if len(train)>=MIN_WINDOWS and 2<=i<7 else None,
                weekly_hybrid_scale_million_bbl=f["scale"],latest_scale_known_utc=f["latest_scale_known_utc"],
                residual_training_windows=len(train))


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
    groups = defaultdict(list)
    for row in rows:
        groups[row["study_phase"], row["method"]].append(row)
    report = dict(status="exploratory_on_reviewed_history",phases={})
    for phase in PHASES:
        base,curve = (groups[phase,k] for k in (ANCHOR,"curve"))
        result = report["phases"][phase] = {}
        for model in MODELS:
            sample = groups[phase,model]
            if [r["release_date"] for r in sample] != [r["release_date"] for r in base]:
                raise ValueError(f"Different calendars for {phase}/{model}")
            confidence = [r["net_170_usd"]-c["net_170_usd"] for r,c in zip(sample,curve,strict=True)]
            errors = [abs(r["expected_long_move_usd"]-r["held_spread_move_usd"]) for r in sample]
            result[model] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample),
                spread_mae_usd=float(np.mean(errors)),
                increment_vs_curve_usd=sum(confidence),
                paired_delta_vs_curve_95_usd=block_bootstrap_mean(confidence),
                paired_delta_vs_public_conditional_95_usd=block_bootstrap_mean(
                    [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(sample,base,strict=True)]))
        print(phase,{m:(r["net_170_usd"], r["net_340_usd"], round(r["spread_mae_usd"],2)) for m,r in result.items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
