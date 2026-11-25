"""Test whether satellite information improves forecasts of losses on fixed CL signals."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_volatility import HURDLE, PHASES
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_late_pair_recovery import COMBINED, STEM as LATE
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_observation_operator import PUBLIC
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_conditional_loss"
ANCHORS = {"public": "public_ensemble_conditional", "historical_satellite": f"{SUPPLEMENT}_conditional"}
RISK_MODELS = ("pooled", "curve", "public", "combined")
GATES = ("pooled_tail_gate", "curve_tail_gate", "public_tail_gate", "satellite_tail_gate")
RISK_PENALTIES = ("curve_tail_penalty", "public_tail_penalty", "satellite_tail_penalty")
POLICIES = ("original", *GATES, *RISK_PENALTIES)
QUANTILE = .80
LOSS_TOLERANCE = .75  # Severe realized loss in units of pre-entry volatility.
GATE_PROBABILITY = .30
HISTORY = 156


def market_inputs():
    forecasts = indexed(f"{LATE}_forecasts")
    anchors = {(r["release_date"],r["method"]):r for r in read(f"{OUTER}_trades")
               if r["method"] in ANCHORS.values()}
    for r,_ in inputs():
        issue = r["release_date"]
        f = forecasts[issue]
        cutoff,known = (datetime.fromisoformat(f[k]) for k in ("cutoff","known"))
        if not cutoff<known<=r["decision"]<r["entry"] or float(f["actual"])!=r["x"][0]:
            raise ValueError(f"Inventory and market clocks do not match on {issue}")
        if any(f[k] and datetime.fromisoformat(f[k])>=cutoff for k in ("latest_training_label_utc","latest_optical_at_utc")):
            raise ValueError(f"Late inventory input on {issue}")
        matches = {name:anchors[issue,key] for name,key in ANCHORS.items() if (issue,key) in anchors}
        for a in matches.values():
            if (any(a[k]!=r[k] for k in ("near_month","far_month")) or any(
                    datetime.fromisoformat(a[k])!=r[field] for k,field in (
                        ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))):
                raise ValueError(f"Control execution window changed on {issue}")
        actual,_,carry,carry_change,*_ = r["x"]
        public = float(f[PUBLIC])
        combined = float(f[COMBINED])
        curve = (carry,carry_change,math.log(r["volatility_usd"]/1000.))
        fundamental = (*curve,actual,actual-public)
        full = (*fundamental,combined-public,float(f["measured"]=="True" or int(f["late_roofs"])>0))
        yield r | dict(design=((),curve,fundamental,full),anchor_rows=matches)


def calibrate(history, row, side, cost):
    sigma = row["volatility_usd"]
    date = row["decision"]
    age = np.array([(date-r["decision"]).total_seconds()/604800 for r in history])
    decay = np.exp2(-age/78.)
    # Fit the same hypothetical side on every settled window. Scale each
    # historical payoff by its own ex ante volatility, then apply today's
    # volatility and identical costs to every probability model.
    x = np.array([side*r["long_gross"]/r["volatility_usd"] for r in history])
    loss = cost/sigma-x
    adverse = loss>LOSS_TOLERANCE
    realized = cost/sigma-side*row["long_gross"]/sigma
    pooled = float((decay@adverse + 2.)/(decay.sum()+8.))
    result = {}
    for name,a,z in zip(RISK_MODELS,(None, *(np.array([q["design"][i] for q in history]) for i in range(1,4))),
                        row["design"],strict=True):
        w = decay
        if a is not None:
            scales = np.array((.5,.25,.5,.75,.75,.30,1.))[:len(z)]
            distances = np.square(np.clip((a-z)/scales,-5.,5.)).mean(axis=1)
            # Partial pooling limits overconfidence on weeks with uncommon conditions.
            w = decay*(.35+.65*np.exp(-distances/2))
        probability = float((w@adverse+8.*pooled)/(w.sum()+8.))
        order = np.argsort(loss,kind="stable")
        values,weights = loss[order],w[order]/w.sum()
        running = np.cumsum(weights)
        tail = np.maximum(running-QUANTILE,0.)-np.maximum(running-weights-QUANTILE,0.)
        quantile = float(values[min(int(np.searchsorted(running,QUANTILE)),len(values)-1)])
        es = float(tail@values/(1-QUANTILE))
        error = realized-quantile
        result[name] = dict(loss_probability=probability,tail_loss_usd=sigma*es,loss_quantile_usd=sigma*quantile,
                            quantile_loss_usd=sigma*max(QUANTILE*error,(QUANTILE-1)*error))
    return result,realized>LOSS_TOLERANCE


def replay():
    history = []
    for r in market_inputs():
        prior = [q for q in history if q["exit_at"]<r["decision"]][-HISTORY:]
        history.append(r)
        for strategy,anchor in r["anchor_rows"].items():
            expectation, side = float(anchor["expected_long_move_usd"]), int(anchor["near_side"])
            if len(prior)<52:
                raise ValueError(f"Loss calibration lacks settled windows on {r['release_date']}")
            risks, event = calibrate(prior,r,side,HURDLE) if side else ({},None)
            hard_gate = [expectation if not side or risks[m]["loss_probability"]<=GATE_PROBABILITY else 0. for m in RISK_MODELS]
            # Penalize excess expected loss relative to the model's pooled
            # forecast for the same side, rather than the absolute volatility.
            penalty = [side*max(0.,abs(expectation)-.5*max(0.,risks[m]["tail_loss_usd"]-risks["pooled"]["tail_loss_usd"]))
                       if side else expectation for m in RISK_MODELS[1:]]
            expectation_by_policy = (expectation,*hard_gate,*penalty)
            for policy,estimate in zip(POLICIES,expectation_by_policy,strict=True):
                value = trade(r,policy,estimate,3,training_at=prior[-1]["exit_at"])
                if policy=="original" and (value["net_170_usd"]!=int(anchor["net_170_usd"]) or value["near_side"]!=side):
                    raise ValueError(f"Control P&L changed on {r['release_date']}/{strategy}")
                yield value | dict(strategy=strategy,original_side=side,original_severe_loss=event,
                    training_windows=len(prior), **{f"{m}_{k}":risks[m][k] if side else None
                        for m in RISK_MODELS for k in ("loss_probability","tail_loss_usd","loss_quantile_usd","quantile_loss_usd")})


def risk_metrics(reference):
    positions = [r for r in reference if r["original_side"]]
    event = np.asarray([r["original_severe_loss"] for r in positions])
    baseline = np.asarray([r["curve_loss_probability"] for r in positions])
    public = np.asarray([r["public_loss_probability"] for r in positions])
    result = dict(trades=len(positions),severe_losses=int(event.sum()),models={})
    for m in RISK_MODELS:
        p = np.asarray([r[f"{m}_loss_probability"] for r in positions])
        brier = np.square(p-event)
        quantile = [r[f"{m}_quantile_loss_usd"] for r in positions]
        result["models"][m] = dict(brier=float(brier.mean()),mean_quantile_loss_usd=float(np.mean(quantile)),
              mean_probability=float(p.mean()),delta_vs_curve_95=block_bootstrap_mean(brier-np.square(baseline-event)),
              delta_vs_public_95=block_bootstrap_mean(brier-np.square(public-event)))
    return result


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["strategy"],r["method"]].append(r)
    report = dict(status="exploratory_on_reviewed_history",gate_probability=GATE_PROBABILITY,
                  severe_loss_preentry_volatility=LOSS_TOLERANCE,phases={})
    for phase in PHASES:
        report["phases"][phase] = {}
        for strategy in ANCHORS:
            reference = groups[phase,strategy,"original"]
            risk = risk_metrics(reference)
            stats = {}
            for m in POLICIES:
                group = groups[phase,strategy,m]
                if [r["release_date"] for r in group]!=[r["release_date"] for r in reference]:
                    raise ValueError(f"Mismatched execution cohort on {phase}/{strategy}/{m}")
                stats[m] = robustness(group) | daily_risk(group) | dict(
                    net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group),
                    severe_losses=sum(bool(r["near_side"] and r["original_severe_loss"]) for r in group),
                    skipped_trades=sum(bool(r["original_side"] and not r["near_side"]) for r in group),
                    paired_delta_vs_original_95_usd=block_bootstrap_mean([r["net_170_usd"]-q["net_170_usd"]
                        for r,q in zip(group,reference,strict=True)]))
            report["phases"][phase][strategy] = dict(risk_fit=risk,policies=stats)
            print(phase,strategy,{m:(v["net_170_usd"],v["net_340_usd"],v["severe_losses"]) for m,v in stats.items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
