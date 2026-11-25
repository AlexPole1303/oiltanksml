"""Check whether inventory-model selection transfers across calendar years."""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_cl_transfer"
CONTROLS = ("curve","inventory","inventory_conditional","optical","optical_conditional")
METHODS = ("yearly_net_selection","yearly_stable_selection")


def matched_rows():
    groups = defaultdict(dict)
    for r in read("cushing_cl_conditional_response_trades"):
        if r["method"] not in CONTROLS:
            continue
        issue = r["release_date"]
        if r["method"] in groups[issue]:
            raise ValueError(f"Repeated candidate on {issue}")
        integers = ("near_side","gross_usd","net_50_usd","net_90_usd","net_170_usd",
                    "cost_50_usd","cost_90_usd","cost_170_usd",*(f"session_{n}_net_170_usd" for n in range(2,6)))
        groups[issue][r["method"]] = r | {k:int(r[k]) for k in integers} | dict(
            alternate_gross_usd=int(r["alternate_gross_usd"]) if r["alternate_source"] else None,
            volatility_usd=float(r["volatility_usd"]))
    for issue,choices in sorted(groups.items()):
        if set(choices)!=set(CONTROLS):
            raise ValueError(f"Unmatched candidates on {issue}")
        if any(len({r[k] for r in choices.values()})!=1 for k in (
            "study_phase","decision_at_utc","entry_at_utc","exit_at_utc","near_month","far_month","volatility_usd"
        )):
            raise ValueError(f"Candidates use differing dates or contracts on {issue}")
        clocks = [datetime.fromisoformat(choices["curve"][k]) for k in ("decision_at_utc","entry_at_utc","exit_at_utc")]
        if clocks!=sorted(set(clocks)):
            raise ValueError(f"Invalid trading dates on {issue}")
        yield choices


def select(history, cutoff):
    eligible = [g for g in history if datetime.fromisoformat(g["curve"]["exit_at_utc"])<cutoff]
    by_year = defaultdict(list)
    for g in eligible:
        by_year[g["curve"]["release_date"][:4]].append(g)
    # Require at least forty release windows in each comparison year.
    complete = [g for g in by_year.values() if len(g)>=40]
    if not complete:
        return ("curve",)*len(METHODS),{},len(eligible)
    performance = {m:[sum(r[m]["net_170_usd"]-r["curve"]["net_170_usd"] for r in year)
                      for year in complete] for m in CONTROLS}
    most = max(CONTROLS,key=lambda m:sum(performance[m]))
    stable_eligible = [m for m in CONTROLS[1:] if len(complete)>=2 and min(performance[m])>0]
    stable = max(stable_eligible,key=lambda m:float(np.median(performance[m])),default="curve")
    return (most,stable),performance,len(eligible)


def main():
    matched = list(matched_rows())
    history, selected, decisions = [], [], []
    year = None
    for choices in matched:
        r = choices["curve"]
        issue = r["release_date"]
        if issue[:4]!=year:
            year = issue[:4]
            cutoff = datetime.fromisoformat(r["decision_at_utc"])
            names,performance,n = select(history,cutoff)
            decisions.append(dict(year=year,selected_on=issue,settled_previous_windows=n,yearly_train_deltas_vs_curve=performance,
                                  selected=dict(zip(METHODS,names,strict=True))))
        for method,name in zip(METHODS,names,strict=True):
            selected.append(choices[name] | dict(method=method,selected_model=name))
        history.append(choices)
    records = [*((r | dict(selected_model=r["method"])) for g in matched for r in g.values()),*selected]
    results = {}
    for m in (*CONTROLS,*METHODS):
        candidates = [r for r in records if r["method"]==m]
        base = [g["curve"] for g in matched]
        if [r["release_date"] for r in candidates]!=[r["release_date"] for r in base]:
            raise ValueError(f"Yearly selection changed the calendar for {m}")
        holdout = [(r,q) for r,q in zip(candidates,base,strict=True) if r["study_phase"]=="holdout"]
        h = [r for r,_ in holdout]
        differences = [r["net_170_usd"]-q["net_170_usd"] for r,q in holdout]
        quiet = [(r,q) for r,q in holdout if r["volatility_usd"]<=3000]
        before_2026 = [(r,q) for r,q in holdout if r["release_date"]<"2026-01-01"]
        yearly = defaultdict(list)
        for r,q in zip(candidates,base,strict=True):
            yearly[r["release_date"][:4]].append((r,q))
        results[m] = dict(holdout=robustness(h) | daily_risk(h) | dict(delta_vs_curve_usd=sum(differences),
                          paired_delta_95_usd=block_bootstrap_mean(differences),
                          net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in h),
                          net_through_2025_usd=sum(r["net_170_usd"] for r,_ in before_2026),
                          delta_vs_curve_through_2025_usd=sum(r["net_170_usd"]-q["net_170_usd"] for r,q in before_2026),
                          windows_excluding_high_volatility=len(quiet),
                          delta_vs_curve_excluding_high_volatility_usd=sum(r["net_170_usd"]-q["net_170_usd"] for r,q in quiet)),
            yearly={year:dict(windows=len(group), trades=sum(r["near_side"]!=0 for r,_ in group),
                net_170_usd=sum(r["net_170_usd"] for r,_ in group),delta_vs_curve_usd=sum(r["net_170_usd"]-q["net_170_usd"] for r,q in group))
                for year,group in yearly.items()})
    report = dict(status="yearly_reselection_on_reviewed_history",yearly_selection=decisions,results=results)
    export(f"{STEM}_trades",selected)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    print({m:(v["holdout"]["net_170_usd"],v["holdout"]["delta_vs_curve_usd"]) for m,v in results.items()})


if __name__=="__main__":
    main()
