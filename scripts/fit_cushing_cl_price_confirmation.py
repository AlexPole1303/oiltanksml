"""Reprice the inventory signal with observable WTI marks on the delayed entry calendar."""

import json
import math
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta

from backtest_wti_cushing_spread import COSTS, settlement, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as delayed_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features

STEM = "cushing_cl_price_confirmation"
METHODS = ("inventory", "curve_report_mark", "inventory_report_mark", "inventory_report_first_mark", "curve_report_first_mark")


def marked(cases):
    clocks = indexed("eia_release_clock")
    first = {r["release_date"]:r["session_date"] for r in read("wti_cushing_event_bars")
             if r["session_number"]=="1" and r["leg"]=="near"}
    end = max(first[r["release_date"]] for r in cases)
    with closing(read_only(SOURCE)) as db:
        sources = {source:{(day,month):tick_price(price) for day,month,_,price in price_rows(db,source,numeric_through=end)}
                   for source in ("archive","ibkr")}
        prior = {r["release_date"]:db.execute("SELECT max(session_date) FROM wti_session_basis WHERE session_date<? "
                    "AND full_cmes_session=1 AND cme_no_settlement_notice_url IS NULL",(r["release_date"],)).fetchone()[0] for r in cases}
    for r in cases:
        issue = r["release_date"]
        stamp = clocks[issue]["scheduled_table_release_at_utc"]
        prior_mark,report_mark, first_mark = map(settlement,(prior[issue],issue,first[issue]))
        eligible = bool(stamp and prior_mark<datetime.fromisoformat(stamp)<report_mark<r["decision"]<first_mark<r["entry"])
        months = r["near_month"],r["far_month"]
        keys = [(day,month) for day in (prior[issue],issue,first[issue]) for month in months]
        source = next((name for name,book in sources.items() if eligible and all(k in book for k in keys)),None)
        if source:
            before_near,before_far,report_near,report_far,first_near,first_far = [sources[source][k] for k in keys]
            before, report, next_day = before_near-before_far,report_near-report_far,first_near-first_far
            sigma = r["volatility_usd"]
            reaction = (10*(report-before)/sigma,(report_near-before_near)/max(abs(before_near),1000))
            later = (10*(next_day-report)/sigma,(first_near-report_near)/max(abs(report_near),1000))
        else:
            reaction,later = (math.nan,math.nan),(math.nan,math.nan)
        yield r | dict(clock_eligible=eligible,mark_source=source,report_mark=report_mark,first_mark=first_mark,
                       reaction=reaction,later=later,earliest_mark=prior_mark)


def replay(cases):
    seen = []
    reference = {r["release_date"]:r for r in read("cushing_cl_volatility_interactions_trades") if r["method"]=="inventory_interaction"}
    for r in marked(cases):
        issue = r["release_date"]
        _,_,_,inventory,_,curve = features(r)
        surprise = r["x"][1]
        spread,flat = r["reaction"]
        response = (spread,flat,spread*surprise)
        designs = (inventory,(*curve,spread,flat),(*inventory,*response),(*inventory,*response,*r["later"]),
                   (*curve,spread,flat,*r["later"]))
        history = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        current = r | dict(designs=designs)
        seen.append(current)
        if len(history)<MIN_WINDOWS:
            continue
        data = dict(release_date=issue,study_phase=r["study_phase"],near_month=r["near_month"],far_month=r["far_month"],
            report_mark=r["report_mark"].isoformat(),first_mark=r["first_mark"].isoformat(),earliest_mark=r["earliest_mark"].isoformat(),
            reaction_source=r["mark_source"],reaction_over_sigma=spread if r["mark_source"] else None,
            following_session_over_sigma=r["later"][0] if r["mark_source"] else None,
            entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),latest_training_exit=history[-1]["exit_at"].isoformat(),
            volatility_usd=r["volatility_usd"],alternate_source=r["alternate_source"])
        for i,(method,x) in enumerate(zip(METHODS,designs,strict=True)):
            # The next calendar day admits the closed daily bar, allowing time
            # for publication rather than using its value at the closing mark.
            decision = (r["first_mark"]+timedelta(days=1)).replace(hour=0,minute=0) if i>=3 and r["mark_source"] else r["decision"]
            if not decision<r["entry"]<r["exit_at"] or (r["mark_source"] and r["report_mark"]>=decision):
                raise ValueError(f"Price feature timed after decision on {issue}/{method}")
            value = risk_prediction([q | dict(x=q["designs"][i]) for q in history],current | dict(x=x),len(x))
            side = int(value>HURDLE)-int(value<-HURDLE)
            gross = side*r["long_gross"]
            if i==0 and not math.isclose(value,float(reference[issue]["expected_long_gross_usd"]),abs_tol=1e-8):
                raise ValueError(f"Inventory replay differs on {issue}")
            yield data | dict(method=method,decision_at_utc=decision.isoformat(),
                latest_feature_mark=(r["first_mark"].isoformat() if i>=3 else r["report_mark"].isoformat()) if r["mark_source"] and i else None,
                expected_long_gross_usd=value,
                near_side=side,gross_usd=gross,alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*v-(170 if n==5 else 85)*abs(side) for n,v in enumerate(r["spread_path_usd"],2)},
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
                raise ValueError(f"Different calendar in {p}/{method}")
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(sample,base,strict=True)]
            report["phases"][p][method] = robustness(sample) | daily_risk(sample) | dict(
                reaction_windows=sum(bool(r["reaction_source"]) for r in sample),increment_usd=sum(delta),
                delta_mean_95_usd=block_bootstrap_mean(delta))
        print(p,{m:(q["trades"],q["reaction_windows"],q["net_170_usd"],q["daily_drawdown_usd"]) for m,q in report["phases"][p].items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
