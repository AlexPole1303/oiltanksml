"""Reconcile the evaluated holdout forecasts and trades to saved source data."""

import json
import math
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from fractions import Fraction
from hashlib import file_digest
from statistics import mean

from audit_cushing_holdout_eia_vintages import stocks
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import ROOT, write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean


def require(value, reason):
    if not value:
        raise ValueError(reason)


def index(rows, *fields):
    out = {tuple(r[k] for k in fields):r for r in rows}
    require(len(out)==len(rows),f"Duplicate ledger row for {fields}")
    return out


def close(actual, expected):
    return math.isclose(float(actual),float(expected),abs_tol=1e-7,rel_tol=1e-11)


def load(name):
    return json.loads((DERIVED/f"{name}.json").read_bytes())


def share_factors():
    splits = defaultdict(list)
    for row in read_csv("study_equity_action_events"):
        if row["event_type"] == "share_split":
            terms = json.loads(row["issuer_terms_json"])
            factor = Fraction(terms["old_shares"], terms["new_shares"])
            splits[row["instrument"]].append((row["first_relevant_date"], factor))
    return splits


def main():
    release = index(read_csv("cushing_holdout_release_index"),"release_date")
    forecasts = index(read_csv("cushing_holdout_replay_forecasts"),"release_date")
    require(release.keys()==forecasts.keys(),"Forecast release IDs differ from the fixed calendar")
    for key,r in forecasts.items():
        ref = release[key]
        week,_,_,_,change,_,_,_,_ = stocks(key[0])
        require(r["week_ending"]==week==ref["week_ending"] and close(float(change)/1000,r["target"]),f"EIA source differs on {key}")
        require(r["forecast_cutoff_utc"]==ref["forecast_cutoff_utc"] and r["target_known_by_utc"]==ref["target_available_at_utc"]
            and r["latest_training_label_utc"]<r["forecast_cutoff_utc"]<r["target_known_by_utc"],f"Forecast clock differs on {key}")
    wti_report = load("cushing_holdout_replay_summary")
    equity_report = load("cushing_holdout_operator_replay_summary")
    require(wti_report["evaluation"] and equity_report["evaluated"],"One holdout is a synthetic run")
    with closing(read_only(SOURCE)) as db:
        futures = {"archive":{(d,m):p for d,m,p in db.execute("SELECT session_date,delivery_month,reported_close FROM wti_tradingcharts_bar WHERE volume>0 AND session_date<capture_date_chicago AND session_date>='2024-01-01'")},
            "ibkr":{(d,m):p for d,m,p in db.execute("SELECT session_date,contract_month,close FROM tws_futures_closed_sessions WHERE instrument='CL' AND volume>0 AND session_date>='2024-01-01'")}}
        shares = {(s,d):(a,b) for s,d,a,b in db.execute("SELECT instrument,session_date,open,close FROM market_study_quote WHERE currency='USD' AND venue='XNYS' AND session_date>='2024-01-01'")}
    wti = read_csv("cushing_holdout_replay_trades")
    wti_by_method = defaultdict(list)
    wti_bars = defaultdict(list)
    for b in read_csv("wti_cushing_event_bars"):
        if b["study_phase"]=="holdout":
            wti_bars[b["release_date"]].append(b)
    for r in wti:
        day = r["release_date"]
        ref, f = release[day,],forecasts[day,]
        side = int(r["near_side"])
        target_time, decision, entry = map(datetime.fromisoformat, (
            f["target_known_by_utc"], ref["decision_at_utc"], ref["entry_mark"]))
        require(side==-int(r["far_side"]) and r["source"]==ref["wti_replay_source"]
                and target_time<=decision<entry,f"Invalid CL timing or legs {day}")
        require(all(r[k]==ref[k] for k in ("near_month","far_month","entry_session","exit_session")),f"Invalid CL contract or calendar {day}")
        require(close(r["published_million_bbl"],f["target"]),f"CL target differs on {day}")
        expectation = 0.0 if r["method"]=="published_change" else None if r["method"]=="always_long_spread" else float(f[r["method"]])
        difference = float(f["target"])-expectation if expectation is not None else None
        expected_side = 1 if difference is None else int(difference<0)-int(difference>0)
        require(side==expected_side and (difference is None or close(r["surprise_million_bbl"],difference)),f"CL rule differs on {day}")
        book = futures[r["source"]]
        prices = {(int(b["session_number"]),b["leg"]):round(100*book[b["session_date"],b["delivery_month"]]) for b in wti_bars[day]}
        require(len(prices)==10,f"CL five-session coverage differs on {day}")
        spread = [10*(prices[i,"near"]-prices[i,"far"]) for i in range(1,6)]
        gross = side*(spread[-1]-spread[0])
        require(close(gross,r["gross_usd"]) and all(close(gross-abs(side)*c,r[f"net_{c}_usd"]) and
                    close(abs(side)*c,r[f"cost_{c}_usd"]) for c in (50,90,170)),f"CL P&L differs on {day}")
        wti_by_method[r["scope"],r["method"]].append(r)
    for (scope,method),group in wti_by_method.items():
        col = "nonoverlapping_window_all" if scope=="all" else "nonoverlapping_window_b04"
        expected = [k[0] for k,r in release.items() if r[col]=="True"]
        require([r["release_date"] for r in group]==expected and all(a["exit_session"]<b["entry_session"]
                for a,b in zip(group,group[1:])),f"CL calendar differs for {scope}/{method}")
        stats = wti_report["trading"][scope]["metrics"][method]
        require(all(close(sum(float(r[k]) for r in group),stats[k]) for k in ("gross_usd","net_50_usd","net_90_usd","net_170_usd")),
                f"CL aggregate differs for {scope}/{method}")
    equity = read_csv("cushing_holdout_operator_replay_trades")
    equity_manifest = index(read_csv("cushing_holdout_operator_replay_manifest"),"release_date","instrument")
    common = {r["release_date"] for r in equity_manifest.values() if r["common_nonoverlapping_issue"]=="True"}
    require(len(common)==equity_report["paired_nonoverlapping_release_dates"],"Wrong equity common cohort")
    splits = share_factors()
    by_strategy = defaultdict(list)
    for r in equity:
        day,symbol = r["release_date"],r["instrument"]
        m = equity_manifest[day,symbol]
        require(day in common and r["latest_training_close_utc"]<r["decision_at_utc"]<r["entry_open_utc"]<r["fifth_close_utc"]
                and m["eia_target_available_at_utc"]<=r["decision_at_utc"] and all(r[k]==m[k] for k in
                    ("entry_session","entry_open_utc","fifth_session","fifth_close_utc")),f"Wrong equity chronology {day}/{symbol}")
        require(close(r["eia_change_kbbl"],1000*float(forecasts[day,]["target"])),f"Wrong equity target {day}/{symbol}")
        side = int(r["instrument_side"])
        price_pnl = 0.0
        for name,ticker,leg_side in (("instrument",symbol,side),("amlp","AMLP",-side)):
            opening, closing_price = shares[ticker,r["entry_session"]][0], shares[ticker,r["fifth_session"]][1]
            require(close(r[f"{name}_entry_quote"],opening) and close(r[f"{name}_exit_quote"],closing_price),
                    f"Wrong equity quotes {day}/{symbol}")
            share_factor = math.prod(factor for effective, factor in splits[ticker] if r["entry_session"] < effective)
            quote_quantity = int(r[f"{name}_entry_shares"])/share_factor
            price_pnl += leg_side*quote_quantity*(closing_price-opening)
        require(close(r["gross_price_pnl_usd"],price_pnl),f"Wrong equity price P&L {day}/{symbol}")
        gross = price_pnl+float(r["cash_distributions_pnl_usd"])
        variable = float(r["spread_cost_usd"])+float(r["borrow_usd"])
        fixed = float(r["commission_usd"])+float(r["regulatory_usd"])
        require(close(r["gross_pnl_usd"],gross) and close(r["net_pnl_usd"],gross-variable-fixed) and
                close(r["stressed_net_pnl_usd"],gross-2*variable-fixed),f"Wrong equity accounting {day}/{symbol}")
        by_strategy[r["strategy"]].append(r)
    for name,group in by_strategy.items():
        stats = equity_report["methods"][name]
        require(len(group)==stats["trades"] and all(a["fifth_close_utc"]<b["entry_open_utc"] for a,b in zip(group,group[1:])),
                f"Equity windows overlap for {name}")
        require(close(sum(float(r["net_pnl_usd"]) for r in group)/1000,stats["net_simple_return_pct"]),f"Equity aggregate differs for {name}")
    optical_increment = {}
    for scope,value in wti_report["forecast"].items():
        cohort = [r for key,r in forecasts.items() if scope=="all" or (scope=="with_b04" and r["measured"]=="True") or
                  (scope=="with_b04_and_all_three_views" and release[key]["optical_views_available"]=="3")]
        cohort.sort(key=lambda r: r["forecast_cutoff_utc"])
        require(len(cohort)==value["releases"],f"Forecast cohort differs for {scope}")
        for model,metric in value["metrics"].items():
            mae = sum(abs(float(r[model])-float(r["target"])) for r in cohort)/len(cohort)
            require(close(mae,metric["mae_million_bbl"]),f"Forecast MAE differs for {scope}/{model}")
        if scope in ("all", "with_b04"):
            paired = [
                abs(float(r["optical_ensemble"]) - float(r["target"]))
                - abs(float(r["public_ensemble"]) - float(r["target"]))
                for r in cohort
            ]
            optical_increment[scope] = dict(releases=len(cohort), mae_delta_million_bbl=mean(paired),
                block_bootstrap_95pct_interval_million_bbl=block_bootstrap_mean(paired))
    artifacts = [ROOT/"data/reference"/f"cushing_holdout{k}_protocol.json" for k in ("","_operator")]
    artifacts.extend(DERIVED/f"{name}.csv" for name in ("cushing_holdout_replay_forecasts","cushing_holdout_replay_trades",
                                                       "cushing_holdout_operator_replay_trades",
                                                       "cushing_holdout_operator_replay_residual_forecasts"))
    artifacts.extend(DERIVED/f"cushing_holdout{k}_replay_summary.json" for k in ("","_operator"))
    hashes = {}
    for path in artifacts:
        with path.open("rb") as stream:
            hashes[str(path.relative_to(ROOT))] = file_digest(stream,"sha256").hexdigest()
    result = dict(actual_first_release_targets=len(forecasts),verified_wti_rows=len(wti),verified_equity_rows=len(equity),
        common_equity_windows=len(common),verified_forecast_cohorts=len(wti_report["forecast"]),
        secondary_paired_optical_ensemble_minus_public=optical_increment,sha256=hashes)
    payload = json.dumps(result,indent=2)+"\n"
    write(DERIVED/"cushing_final_replay_audit.json",payload.encode())
    print(payload,end="")


if __name__=="__main__":
    main()
