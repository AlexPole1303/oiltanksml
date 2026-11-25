"""Evaluate inventory-surprise rules on fixed, dated NYMEX calendar spreads."""
import argparse
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import exchange_calendars as ec
import numpy as np

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from build_wti_curve import MONTH_CODES
from collect_public import write
from fit_cushing_nowcast_ensemble import MAX_RELEASE, STEM as ML_STEM, block_bootstrap_mean

STEM = "wti_cushing_event"
NEW_YORK = ZoneInfo("America/New_York")
SOURCES = ("official", "archive", "ibkr")
COSTS = (50, 90, 170)
BASE = ("always_long_spread", "published_change", "prior_surprise")
ML = ("public_ensemble_surprise", "optical_ensemble_surprise")
ADAPTIVE = {"adaptive_public_surprise":"adaptive_public", "adaptive_optical_surprise":"adaptive_optical",
            "raw_stack_surprise":"stack_raw", "multiview_stack_surprise":"stack_multiview"}
SENSOR = {"sensor_only_surprise":"sensor_only", "wti_boost_surprise":"wti_boost", "wti_sensor_surprise":"wti_sensor",
          "terminal_weighted_surprise":"terminal_weighted", "development_selected_surprise":"development_selected"}


def settlement(day):
    return datetime.combine(date.fromisoformat(day),time(14,30),NEW_YORK)


def tick_price(value):
    if value is None:
        return None
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite WTI close: {value}")
    ticks = round(value*100)
    if not math.isclose(value*100,ticks,abs_tol=1e-6):
        raise ValueError(f"Close is off the WTI tick grid: {value}")
    return ticks


def price_books(db):
    # SQL exports presence only beyond the market holdout boundary.
    query = {
        "official": "SELECT session_date,delivery_month,CASE WHEN session_date<=? THEN official_close END,source_url "
                    "FROM wti_eia_dated_close WHERE official_close IS NOT NULL",
        "archive": "SELECT session_date,delivery_month,CASE WHEN session_date<=? THEN reported_close END,source_file "
                   "FROM wti_tradingcharts_bar WHERE reported_close IS NOT NULL AND volume>0 AND session_date<capture_date_chicago",
        "ibkr": "SELECT session_date,contract_month,CASE WHEN session_date<=? THEN close END,source_file "
                "FROM tws_futures_closed_sessions WHERE instrument='CL' AND close IS NOT NULL AND volume>0",
    }
    books = {}
    for source, sql in query.items():
        records = {}
        for day, month, close, reference in db.execute(sql,(MAX_RELEASE,)):
            if (day,month) in records:
                raise ValueError(f"Duplicate {source} close for {day}/{month}")
            records[day,month] = tick_price(close),reference
        books[source] = records
    return books


def full_sessions(db, latest_issue):
    full = [r[0] for r in db.execute("""SELECT session_date FROM wti_session_basis
        WHERE full_cmes_session=1 AND cme_no_settlement_notice_url IS NULL ORDER BY session_date""")]
    latest_basis, = db.execute("SELECT max(session_date) FROM wti_session_basis").fetchone()
    end = date.fromisoformat(latest_issue)+timedelta(days=22)
    future = ec.get_calendar("CMES",start=latest_basis,end=end)
    early = set(future.early_closes)
    return full+[d.date().isoformat() for d in future.sessions if d.date().isoformat()>latest_basis and d not in early]


def build_windows():
    events = sorted((r for r in read_csv("eia_phase_market_calendar") if r["calendar_venue"]=="XNYS"),
                    key=lambda r:r["release_date"])
    if len({r["release_date"] for r in events}) != len(events):
        raise ValueError("Duplicate WTI event dates")
    with closing(read_only(SOURCE)) as db:
        books = price_books(db)
        sessions = full_sessions(db,events[-1]["release_date"])
        contracts = list(db.execute("""SELECT delivery_month,last_trading_date,expiry_basis
            FROM wti_delivery_calendar ORDER BY delivery_month"""))
    expiries = [r[1] for r in contracts]
    windows, bars = [], []
    moves = defaultdict(dict)
    for r in events:
        issue = r["release_date"]
        decision = datetime.fromisoformat(r["decision_at_utc"])
        idx = bisect_right(sessions,issue)
        while settlement(sessions[idx]) <= decision:
            idx += 1
        days = sessions[idx:idx+5]
        guard = sessions[idx+6]
        near_index = bisect_right(expiries,guard)
        pair = (contracts[near_index],contracts[near_index+3])
        near, far = (contract[0] for contract in pair)
        start, stop = map(settlement,(days[0],days[-1]))
        boundary = r["next_phase_first_decision_at_utc"]
        phase_ok = not boundary or stop < datetime.fromisoformat(boundary)
        open_numeric = r["study_phase"] in ("development","validation") and days[-1]<=MAX_RELEASE and phase_ok
        covered = {source:sum((d,m) in book for d in days for m in (near,far)) for source,book in books.items()}
        windows.append(dict(**{k:r[k] for k in ("study_phase","week_ending","release_date","decision_at_utc",
                            "next_phase_first_decision_at_utc","eia_release_clock_source_url")},
            entry_session=days[0],exit_session=days[-1],entry_mark=start.isoformat(),exit_mark=stop.isoformat(),
            expiry_cushion_through=guard,near_month=near,far_month=far,near_expiry=pair[0][1],far_expiry=pair[1][1],
            near_expiry_basis=pair[0][2],far_expiry_basis=pair[1][2],phase_ok=phase_ok,open_numeric=open_numeric,
            **{f"{s}_available_leg_sessions":n for s,n in covered.items()}))
        for number, day in enumerate(days,1):
            for role,(month,expiry,_) in zip(("near","far"),pair):
                row = dict(release_date=issue,study_phase=r["study_phase"],session_number=number,session_date=day,
                    settlement_reference_time=settlement(day).isoformat(),leg=role,delivery_month=month,
                    symbol=f"CL{MONTH_CODES[int(month[4:])-1]}{month[2:4]}",expiry=expiry)
                for source,book in books.items():
                    observed = book.get((day,month))
                    row[f"{source}_present"] = observed is not None
                    row[f"{source}_close_ticks"] = observed[0] if observed and open_numeric else None
                    row[f"{source}_reference"] = observed[1] if observed else None
                bars.append(row)
        if open_numeric:
            for source in ("official","archive"):
                if covered[source]==10:
                    book = books[source]
                    marks = [(book[day,near][0]-book[day,far][0])*10 for day in days]
                    moves[issue][source] = [m-marks[0] for m in marks]
    return windows, bars, moves


def available_signals(windows, compare_forecasts=False, compare_sensor=False):
    open_issues = {r["release_date"]:r for r in windows if r["open_numeric"]}
    inventory = {(r["release_date"],r["week_ending"]):r for r in read_csv("sentinel2_space_time_coverage_eia_comparison")}
    forecasts = {r["release_date"]:r for r in read_csv(f"{ML_STEM}_forecasts")}
    legacy = {(r["release_date"],r["week_ending"]):r for r in read_csv("cushing_eia_walk_forward_forecasts")}
    adaptive = {r["release_date"]:r for r in read_csv("cushing_multiview_nowcast_forecasts")} if compare_forecasts or compare_sensor else {}
    sensor = {r["release_date"]:r for r in read_csv("cushing_sensor_market_nowcast_forecasts")} if compare_sensor else {}
    result = {}
    for issue,w in open_issues.items():
        week = w["week_ending"]
        row = inventory.get((issue,week))
        if not row or not row["eia_first_release_change_kbbl"] or not row["prior_kbbl"]:
            continue
        actual, prior = (float(row[k])/1000 for k in ("eia_first_release_change_kbbl","prior_kbbl"))
        expectations = {"published_change":0.,"prior_surprise":prior}
        optical = bool(row["raw_kbbl"])
        if optical:
            expectations["prior_plus_optical_surprise"] = prior+float(row["raw_kbbl"])/1000
        f, old = forecasts.get(issue), legacy.get((issue,week))
        for model in (f,old):
            if model and (model["week_ending"]!=week or float(model["target_million_bbl"])!=actual or
                datetime.fromisoformat(model["latest_training_release_available_utc"])>=datetime.fromisoformat(model["forecast_cutoff_utc"]) or
                datetime.fromisoformat(model["forecast_cutoff_utc"])!=datetime.fromisoformat(row["cutoff_utc"]) or
                datetime.fromisoformat(model["target_known_by_utc"])>datetime.fromisoformat(w["decision_at_utc"])):
                raise ValueError(f"Forecast clock, target or period mismatch on {issue}")
        if f:
            expectations.update({k:float(f[k.removesuffix("_surprise")]) for k in ML})
        if old and optical:
            expectations.update({f"{k}_surprise":float(old[k]) for k in ("ridge_public","ridge_public_optical")})
        newer = adaptive.get(issue)
        if newer:
            fit_before = newer["latest_stack_training_release_available_utc"]
            cutoff = datetime.fromisoformat(newer["forecast_cutoff_utc"])
            if (not f or newer["week_ending"]!=week or float(newer["target"])!=actual
                or optical!=(newer["measured"]=="True") or cutoff!=datetime.fromisoformat(row["cutoff_utc"])
                or (fit_before and datetime.fromisoformat(fit_before)>=cutoff)
                or datetime.fromisoformat(newer["target_known_by_utc"])>datetime.fromisoformat(w["decision_at_utc"])):
                raise ValueError(f"Adaptive forecast clock, target or period mismatch on {issue}")
            expectations.update({k:float(newer[v]) for k,v in ADAPTIVE.items()})
        calibrated = sensor.get(issue)
        if calibrated:
            keys = ("week_ending", "target", "measured", "forecast_cutoff_utc", "target_known_by_utc")
            fit_before = calibrated["latest_training_label_utc"]
            if (not newer or any(newer[k]!=calibrated[k] for k in keys)
                or float(calibrated["incumbent"])!=float(newer["stack_multiview"])
                or (fit_before and datetime.fromisoformat(fit_before)>=cutoff)):
                raise ValueError(f"Sensor forecast clock, target or baseline mismatch on {issue}")
            expectations.update({k:float(calibrated[v]) for k,v in SENSOR.items()})
        result[issue] = actual,expectations
    return result


def replay(windows,moves,signals):
    last_exit = {}
    for w in windows:
        issue, phase = w["release_date"],w["study_phase"]
        if issue not in signals:
            continue
        actual, expected = signals[issue]
        optical = "prior_plus_optical_surprise" in expected
        ml = all(m in expected for m in ML)
        controls = (*BASE,"prior_plus_optical_surprise") if optical else BASE
        ridge = tuple(k for k in ("ridge_public_surprise","ridge_public_optical_surprise") if k in expected)
        scopes = [("all","official",BASE), ("all","archive",BASE)]
        if optical:
            scopes += [("optical",source,controls) for source in ("official","archive")]
        if ridge:
            scopes += [("optical_legacy_ridge",source,(*controls,*ridge)) for source in ("official","archive")]
        if ml:
            scopes += [("ml_all","official",(*BASE,*ML)),("ml_all","archive",(*BASE,*ML))]
        if ml and optical and {"official","archive"}<=moves.get(issue,{}).keys():
            scopes += [("ml_optical_common",source,(*controls,*ML)) for source in ("official","archive")]
        if ml and optical:
            scopes += [("ml_optical",source,(*controls,*ML)) for source in ("official","archive")]
        if ADAPTIVE.keys()<=expected.keys():
            scopes += [("adaptive_all",source,(*BASE,*ML,*ADAPTIVE)) for source in ("official","archive")]
            if optical:
                scopes += [("adaptive_optical",source,(*controls,*ML,*ADAPTIVE)) for source in ("official","archive")]
        if SENSOR.keys()<=expected.keys():
            newest = ("multiview_stack_surprise",*SENSOR)
            scopes += [("sensor_all",source,(*BASE,*newest)) for source in ("official","archive")]
            if optical:
                scopes += [("sensor_optical",source,(*controls,*newest)) for source in ("official","archive")]
        for scope,source,methods in scopes:
            key = phase,scope,source
            if source not in moves.get(issue,{}) or w["entry_session"]<=last_exit.get(key,""):
                continue
            last_exit[key] = w["exit_session"]
            spread_move = moves[issue][source]
            for method in methods:
                yield settlement_trade(w,scope,source,method,actual,expected.get(method),spread_move)


def settlement_trade(w,scope,source,method,actual,expected,spread_move):
    surprise = actual-expected if expected is not None else None
    side = 1 if surprise is None else int(surprise<0)-int(surprise>0)
    path = [side*mark for mark in spread_move]
    gross = path[-1]
    return {k:w[k] for k in ("study_phase","week_ending","release_date","entry_session","exit_session","near_month","far_month")} | dict(
        scope=scope,source=source,method=method,near_side=side,far_side=-side,published_million_bbl=actual,
        expected_million_bbl=expected,surprise_million_bbl=surprise,
        gross_usd=gross,minimum_settlement_gross_usd=min(path),maximum_settlement_gross_usd=max(path),
        **{f"cost_{c}_usd":c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def statistics(trades):
    gross = sum(r["gross_usd"] for r in trades)
    base_net = np.array([r["net_50_usd"] for r in trades])
    equity = np.r_[0,np.cumsum(base_net)]
    return dict(windows=len(trades),trades=sum(r["near_side"]!=0 for r in trades),gross_usd=gross,
        **{f"cost_{c}_usd":sum(r[f"cost_{c}_usd"] for r in trades) for c in COSTS},
        **{f"net_{c}_usd":sum(r[f"net_{c}_usd"] for r in trades) for c in COSTS},
        profitable_base_trades=int(np.count_nonzero(base_net>0)),
        closed_trade_max_drawdown_usd=int(np.max(np.maximum.accumulate(equity)-equity)),
        mean_net_50_per_window_95pct_block_bootstrap_usd=block_bootstrap_mean(base_net))


def summarize(windows,trades):
    phases = {}
    for phase in dict.fromkeys(r["study_phase"] for r in windows):
        subset = [r for r in windows if r["study_phase"]==phase]
        coverage = {s:sum(r[f"{s}_available_leg_sessions"]==10 for r in subset) for s in SOURCES}
        overlap = Counter("+".join(s for s in SOURCES if r[f"{s}_available_leg_sessions"]==10) or "none" for r in subset)
        grouped = defaultdict(list)
        for t in trades:
            if t["study_phase"]==phase:
                grouped[t["scope"],t["source"],t["method"]].append(t)
        results = defaultdict(dict)
        for (scope,source,method),rows in grouped.items():
            results[f"{scope}:{source}"][method] = statistics(rows)
        effects = {}
        for scope,source in {(a,b) for a,b,_ in grouped}:
            public,optical = (grouped.get((scope,source,k),[]) for k in ML)
            if public and [r["release_date"] for r in public]==[r["release_date"] for r in optical]:
                changes = np.array([o["net_50_usd"]-p["net_50_usd"] for p,o in zip(public,optical)])
                changed = [{"release_date":p["release_date"], "public_near_side":p["near_side"],
                            "optical_near_side":o["near_side"], "delta_net_usd":int(delta)}
                           for p,o,delta in zip(public,optical,changes) if p["near_side"]!=o["near_side"]]
                effects[f"{scope}:{source}"] = dict(windows=len(changes),changed_directions=len(changed),
                    total_delta_net_usd=int(changes.sum()),changed_events=changed,
                    mean_delta_per_window_95pct_block_bootstrap_usd=block_bootstrap_mean(changes))
        phases[phase] = dict(calendar_issues=len(subset),complete_source_windows=coverage,
                             complete_window_source_overlap=dict(overlap),covered_by_any_source=len(subset)-overlap["none"],
                             open_numeric_windows=sum(r["open_numeric"] for r in subset),results=results,paired_ml_effect=effects)
    return dict(phases=phases,cost_scenarios_usd=COSTS,tick_value_usd=10,
        interpretation="One dated CL spread per eligible release. Scheduled settlement references and scenario costs. "
        "Numerical market labels and trades end in 2023. Source-specific eligibility conditions the historical sample.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-forecasts",action="store_true",help="Add adaptive and multiview forecasts to matched historical windows")
    parser.add_argument("--compare-sensor",action="store_true",help="Also test the WTI and optical reliability forecast variants")
    args = parser.parse_args()
    windows,bars,moves = build_windows()
    trades = list(replay(windows,moves,available_signals(windows,args.compare_forecasts,args.compare_sensor)))
    export(f"{STEM}_windows",windows)
    export(f"{STEM}_bars",bars)
    export(f"{STEM}_trades",trades)
    report = summarize(windows,trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    for phase,values in report["phases"].items():
        print(phase,values["calendar_issues"],values["complete_source_windows"],
            {scope:result["total_delta_net_usd"] for scope,result in values["paired_ml_effect"].items()})


if __name__ == "__main__":
    main()
