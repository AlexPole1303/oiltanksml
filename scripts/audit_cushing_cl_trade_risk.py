"""Compare one-spread risk rules on the dated public and satellite CL signals."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from audit_cushing_cl_selection import BASE, CONTROL, OPTICAL
from backtest_wti_cushing_spread import NEW_YORK
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_volatility import HURDLE, PHASES
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_cl_trade_risk"
SOURCES = {
    "public": ("cushing_cl_conditional_response_trades",BASE.split(":")[1],"expected_long_move_usd"),
    "optical": ("cushing_outer_capture_nowcast_trades",OPTICAL.split(":")[1],"expected_long_move_usd"),
    "curve": ("cushing_cl_volatility_interactions_trades",CONTROL.split(":")[1],"expected_long_gross_usd"),
}
RISK_CAPS = (2000,3000,5000)
ADVERSE_THRESHOLDS = (1.,1.5)
RULES = ("original", *(f"sigma_cap_{v}" for v in RISK_CAPS), "risk_adjusted_entry",
         *(f"adverse_close_{v:g}_sigma" for v in ADVERSE_THRESHOLDS), "observed_risk_cap")
MIN_RISK_HISTORY = 52


def loaded():
    sources = {}
    for method,(stem,key,value) in SOURCES.items():
        records = [r for r in read(stem) if r["method"]==key]
        table = {r["release_date"]:(r,float(r[value])) for r in records}
        if len(table)!=len(records) or not table:
            raise ValueError(f"Incomplete source decisions in {stem}")
        sources[method] = table
    if any(s.keys()!=sources["public"].keys() for s in sources.values()):
        raise ValueError("The source policies disagree on eligible trade dates")
    return sources


def risk_estimate(history, row, side):
    train = [q for q in history if q["exit_at"]<row["decision"]][-156:]
    if len(train)<MIN_RISK_HISTORY or not side:
        return None,train
    # Estimate a high quantile of adverse daily marks under the specified side.
    # The denominator is each previous trade window's own pre-release risk.
    excursions = [max(0., -min(side*move for move in q["spread_path_usd"]))/q["volatility_usd"] for q in train]
    return HURDLE+row["volatility_usd"]*float(np.quantile(excursions,.90)),train


def exit_on_adverse_mark(row, side, threshold):
    # The second mark is the only adverse observation that can lead to
    # an earlier close when there are four marks and a one-session lag.
    observed, exit_at = row["stamps"][1:3]
    decision = datetime.combine(exit_at.astimezone(NEW_YORK).date(),datetime.min.time(),NEW_YORK)
    if not observed < decision < exit_at:
        raise ValueError(f"Invalid stop timing on {row['release_date']}")
    if side and -side*row["spread_path_usd"][1] >= threshold:
        return 2,observed,decision
    return 3,None,None


def replay():
    source = loaded()
    history = []
    for r,_ in inputs():
        issue = r["release_date"]
        if issue not in source["public"]:
            history.append(r)
            continue
        for model in SOURCES:
            anchor,expectation = source[model][issue]
            side = int(anchor["near_side"])
            sigma = r["volatility_usd"]
            if (not math.isclose(float(anchor["volatility_usd"]),sigma,abs_tol=1e-8)
                    or any(anchor[k]!=r[k] for k in ("near_month","far_month"))
                    or any(datetime.fromisoformat(anchor[k])!=r[field] for k,field in (
                         ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))):
                raise ValueError(f"Risk scenarios do not share the source calendar on {issue}/{model}")
            risk,history_used = risk_estimate(history,r,side)
            entry_tests = (True,*(sigma<=cap for cap in RISK_CAPS),
                            abs(expectation)>HURDLE+.10*sigma)
            exit_tests = [exit_on_adverse_mark(r,side,v*sigma) for v in ADVERSE_THRESHOLDS]
            alternatives = ([(expectation if allowed else 0.,3,None,None) for allowed in entry_tests]
                            + [(expectation,*close) for close in exit_tests]
                            + [(expectation if risk is not None and risk<=5000 else 0.,3,None,None)])
            for rule,(value,end,observed,decided) in zip(RULES,alternatives,strict=True):
                q = trade(r,rule,value,end)
                if rule=="original" and (q["near_side"]!=side or q["gross_usd"]!=int(anchor["gross_usd"])):
                    raise ValueError(f"The original absolute P&L changed on {issue}/{model}")
                worst = min(q["near_side"]*move for move in r["spread_path_usd"][:end+1])
                yield q | dict(model=model, original_near_side=side,
                    estimated_adverse_risk_usd=risk,
                    hypothetical_original_adverse_usd=max(0.,-min(side*move for move in r["spread_path_usd"]))+HURDLE*abs(side),
                    learned_risk_training_weeks=len(history_used),
                    latest_risk_label_utc=history_used[-1]["exit_at"].isoformat() if history_used else None,
                    exit_signal_input_utc=observed.isoformat() if observed else None,
                    exit_signal_clock_utc=decided.isoformat() if decided else None,
                    adverse_daily_mark_usd=worst,
                    net_340_usd=q["gross_usd"]-340*abs(q["near_side"]))
        history.append(r)


def main():
    rows = list(replay())
    report = dict(status="exploratory_on_reviewed_history",phases={})
    for phase in PHASES:
        group = report["phases"][phase] = {}
        for model in SOURCES:
            by_rule = defaultdict(list)
            for r in rows:
                if r["model"]==model and r["study_phase"]==phase:
                    by_rule[r["method"]].append(r)
            reference = by_rule["original"]
            for rule, sample in by_rule.items():
                if [q["release_date"] for q in sample]!=[q["release_date"] for q in reference]:
                    raise ValueError(f"Different release calendar for {phase}/{model}/{rule}")
                change = [q["net_170_usd"]-r["net_170_usd"] for q,r in zip(sample,reference,strict=True)]
                admitted = [q for q in sample if q["near_side"] and q["estimated_adverse_risk_usd"] is not None]
                group[f"{model}:{rule}"] = robustness(sample) | daily_risk(sample) | dict(
                    net_340_usd=sum(q["net_340_usd"] for q in sample),
                    worst_daily_before_cost_usd=min(q["adverse_daily_mark_usd"] for q in sample),
                    blocked_trades=sum(q["original_near_side"]!=0 and q["near_side"]==0 for q in sample),
                    exited_early=sum(q["near_side"]!=0 and q["exit_session_number"]<5 for q in sample),
                    original_path_risk_estimates=len(admitted),
                    original_path_risk_exceedances=sum(q["hypothetical_original_adverse_usd"]>q["estimated_adverse_risk_usd"] for q in admitted),
                    paired_delta_vs_original_95_usd=block_bootstrap_mean(change))
        print(phase,{m:(r["trades"],r["net_170_usd"],r["net_340_usd"],r["daily_drawdown_usd"]) for m,r in group.items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
