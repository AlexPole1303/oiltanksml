"""Model the probability and size of CL spread gains on settled historical windows."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_payoff_distribution"
BASE = "inventory_conditional"
METHODS = ("curve", BASE, "mixture", "mixture_optical", "mixture_blended", "mixture_optical_blended")


def distribution(sample, row, dimension):
    weights = np.exp2(-np.array([(row["decision"]-q["decision"]).total_seconds()/604800 for q in sample])/78)
    raw = np.array([q["design"][:dimension] for q in sample])
    center = np.average(raw,axis=0,weights=weights)
    scale = np.maximum(np.sqrt(np.average((raw-center)**2,axis=0,weights=weights)),.15)
    x = np.clip((raw-center)/scale,-4,4)
    z = np.clip((np.asarray(row["design"][:dimension])-center)/scale,-4,4)
    normalized = np.asarray([q["long_gross"]/q["volatility_usd"] for q in sample])
    active = normalized!=0
    positive = normalized>0
    if np.count_nonzero(positive)<10 or np.count_nonzero(normalized<0)<10:
        return row["anchor"], None, None
    model = LogisticRegression(C=.10,max_iter=300)
    model.fit(x[active],positive[active],sample_weight=weights[active])
    p = float(model.predict_proba(z[None,:])[0,1])
    local = weights*np.exp(-.5*np.mean((x-z)**2,axis=1))
    magnitude = np.minimum(abs(normalized),4.)
    values = []
    for mask in (positive,normalized<0):
        pooled = float(np.average(magnitude[mask],weights=weights[mask]))
        w = local[mask]
        values.append(float((w@magnitude[mask]+24*pooled)/(w.sum()+24)))
    probability_nonzero = weights[active].sum()/weights.sum()
    expected = row["volatility_usd"]*probability_nonzero*(p*values[0]-(1-p)*values[1])
    unconditional_reference = float(weights[positive].sum()/weights[active].sum())
    return float(expected), p, unconditional_reference


def replay():
    sources = indexed("cushing_inventory_market_bridge_forecasts")
    controls = {(q["release_date"],q["method"]):q for q in read("cushing_cl_conditional_response_trades")
                if q["method"] in METHODS[:2]}
    _,_,cases = market_calendar()
    seen = []
    for r in cases:
        issue, decision = r["release_date"], r["decision"]
        if (issue,BASE) not in controls:
            continue
        f = sources[issue]
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc","known_utc"))
        actual = float(f["actual"])
        if not cutoff<known<=decision or actual!=r["x"][0] or any(
            f[k]!=r[k] for k in ("near_month","far_month")
        ):
            raise ValueError(f"Different EIA or contract metadata on {issue}")
        ref = [controls[issue,m] for m in METHODS[:2]]
        for q in ref:
            if (any(datetime.fromisoformat(q[key])!=r[field] for key,field in (
                    ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at"))) or
                any(q[k]!=r[k] for k in ("near_month","far_month")) or
                q["latest_calibration_exit_utc"] and datetime.fromisoformat(q["latest_calibration_exit_utc"])>=decision):
                raise ValueError(f"Unmatched spread control on {issue}")
        curve, anchor = (float(q["expected_long_move_usd"]) for q in ref)
        sigma = r["volatility_usd"]
        _,_,carry,carry_change,*_ = r["x"]
        surprise = actual-float(f[PUBLIC])
        survey = float(f["survey_spread_move"]) if f["survey_spread_move"] else 0.
        late = float(f["post_survey_spread_move"]) if f["post_survey_spread_move"] else 0.
        optical = float(f["all_recovered"])-float(f[PUBLIC])
        row = r | dict(anchor=anchor, design=(anchor/sigma,curve/sigma,carry,carry_change,
                                             math.log(sigma/1000),surprise,survey,late,optical,optical*math.log(sigma/1000)))
        past = [q for q in seen if q["exit_at"]<decision][-HISTORY:]
        seen.append(row)
        estimates = [(anchor,None,None)]*2
        if len(past)>=MIN_WINDOWS:
            estimates = [distribution(past,row,n) for n in (8,10)]
        blended = [((value+anchor)/2,p,baseline) for value,p,baseline in estimates]
        for method,(value,probability,baseline) in zip(METHODS,((curve,None,None),(anchor,None,None),*estimates,*blended),strict=True):
            side = int(value>HURDLE)-int(value<-HURDLE)
            gross = side*r["long_gross"]
            if method in METHODS[:2]:
                q = controls[issue,method]
                if side!=int(q["near_side"]) or gross!=int(q["gross_usd"]):
                    raise ValueError(f"Original position differs on {issue}/{method}")
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd","alternate_source")} | dict(
                method=method,decision_at_utc=decision.isoformat(),entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),
                latest_training_exit_utc=past[-1]["exit_at"].isoformat() if past else None,training_windows=len(past),
                estimated_probability_long_move_positive_given_nonzero=probability,expected_long_move_usd=value,
                historic_positive_rate_given_nonzero=baseline,long_move_usd=r["long_gross"],
                near_side=side,gross_usd=gross,alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*m-(170 if n==5 else 85)*abs(side) for n,m in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
    groups = defaultdict(list)
    for q in rows:
        groups[q["study_phase"],q["method"]].append(q)
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        base = groups[phase,BASE]
        result = {}
        for m in METHODS:
            arm = groups[phase,m]
            if [q["release_date"] for q in arm]!=[q["release_date"] for q in base]:
                raise ValueError(f"Different payoff-model calendar on {phase}/{m}")
            changes = [q["net_170_usd"]-b["net_170_usd"] for q,b in zip(arm,base,strict=True)]
            nonzero = [q for q in arm if q["long_move_usd"] and q["estimated_probability_long_move_positive_given_nonzero"] is not None]
            brier = {}
            if nonzero:
                actual = np.array([q["long_move_usd"]>0 for q in nonzero])
                scores = {key:(np.array([q[field] for q in nonzero])-actual)**2 for key,field in (
                    ("probability_brier","estimated_probability_long_move_positive_given_nonzero"),
                    ("reference_brier","historic_positive_rate_given_nonzero"))}
                brier = {k:float(v.mean()) for k,v in scores.items()}
                brier["paired_brier_difference_95"] = block_bootstrap_mean(scores["probability_brier"]-scores["reference_brier"])
            result[m] = robustness(arm) | daily_risk(arm) | brier | dict(
                nonzero_probability_windows=len(nonzero),
                net_340_usd=sum(q["gross_usd"]-340*abs(q["near_side"]) for q in arm),
                pnl_delta_vs_candidate_usd=sum(changes), paired_delta_vs_candidate_95_usd=block_bootstrap_mean(changes))
        report["phases"][phase] = result
        print(phase,{m:(q["net_170_usd"],q["net_340_usd"],q["daily_drawdown_usd"]) for m,q in result.items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
