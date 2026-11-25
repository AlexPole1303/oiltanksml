"""Reproduce the public model and score forward CL releases with archived inputs."""

import argparse
import hashlib
import json
import math
import re
from bisect import bisect_left, bisect_right
from contextlib import closing
from datetime import date, datetime, timedelta, timezone

import numpy as np
from threadpoolctl import threadpool_limits

import fit_cushing_adaptive_nowcast as adaptive
import fit_cushing_nowcast_ensemble as ensemble
import fit_cushing_robust_stack as stack
from audit_cushing_cl_horizons import paths
from audit_cushing_forward_wti_coverage import archives
from audit_cushing_holdout_cftc_reports import original, positions
from backtest_wti_cushing_spread import COSTS, full_sessions, settlement, tick_price
from build_cushing_holdout_market_features import make_plan, read_protocol
from build_cushing_holdout_release_index import price_rows
from build_eia_market_inputs import METRICS, standardized
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import RAW, ROOT, write
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, inputs, predictions
from fit_cushing_cl_volatility import HURDLE, RISK_FLOOR_USD, RISK_LOOKBACK, risk_inputs, risk_prediction, spread_volatility
from fit_cushing_delayed_calendar import replay as delayed_calendar_replay
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_sensor_market_nowcast import EASTERN
from fit_cushing_volatility_interactions import features as interaction_features
from publication_clocks import utc
from replay_cushing_holdout import closed_inputs, indexed as holdout_index
from audit_cushing_holdout_eia_vintages import archive, report_date

STEM = "cushing_forward_public_cl"
REGISTERED = DERIVED/f"{STEM}_registered"
EXPANDED_METHOD = "public_volatility_second_session_expanded_calendar"
INTERACTION_METHOD = "inventory_interaction_second_session"
CARRY_INTERACTION_METHOD = "carry_volatility_context_second_session"
ENSEMBLE_INTERACTION_METHOD = "inventory_public_expectation_second_session"
EXPANDED_MODELS = (EXPANDED_METHOD, INTERACTION_METHOD, CARRY_INTERACTION_METHOD, ENSEMBLE_INTERACTION_METHOD)
MODEL_WINDOWS = {"public": 1, "public_volatility": 1, "carry_only_volatility": 1,
                 "public_volatility_second_session": 2} | dict.fromkeys(EXPANDED_MODELS,2)
DECISION_FIELDS = ("release_date","method","week_ending","forecast_cutoff_utc","forecast_million_bbl","actual_million_bbl",
                   "decision_at_utc","registration_deadline_utc","entry_at","exit_at","near_month","far_month","estimated_long_move_usd","near_side",
                   "volatility_usd","volatility_source","eligible_nonoverlap","training_windows","latest_training_exit_utc",
                   "curve_day","curve_source","carry","carry_change","cftc_file","cftc_sha256")


def forward_report_vintage(issue, week, prior_week, vintage):
    directory = RAW/"eia/wpsr_forward"
    for table in sorted({m.table for m in METRICS}):
        rows, stubs, proof = archive(issue,table,directory=directory)
        dates = list(map(report_date,rows[0][stubs:stubs+2]))
        if dates!=[week,prior_week]:
            raise ValueError(f"Unexpected report weeks in {issue} Table {table}")
        for metric in METRICS:
            if metric.table!=table:
                continue
            matches = [(number,row) for number,row in enumerate(rows[1:],2)
                       if len(row)>=stubs+2 and (metric.row is None or metric.row==number)
                       and row[stubs-1].strip()==metric.label and (not metric.group or row[0].strip()==metric.group)]
            if len(matches)!=1:
                raise ValueError(f"Missing unique {metric.key} in {issue} Table {table}")
            number, row = matches[0]
            for i, observed in enumerate(dates):
                value = standardized(metric,issue,observed,row[stubs+i].strip(),table,"archived_csv",
                                     number,stubs+i+1,metric.label,proof["source_url"])[4]
                vintage[issue,observed][metric.key] = ensemble.number(value)


def cftc(cutoff):
    day = cutoff.astimezone(EASTERN).date()
    last_tuesday = day-timedelta(days=(day.weekday()-1)%7 or 7)
    for lag in range(1,4):
        report = (last_tuesday-timedelta(weeks=lag-1)).isoformat()
        try:
            html,path,meta = original(report)
        except OSError:
            continue
        (interest,long,short,*_), _ = positions(html,report)
        footer, = re.findall(r">\s*Updated ([A-Za-z]+ \d{1,2}, \d{4})\s*</p>",html)
        reported = datetime.strptime(footer,"%B %d, %Y").date()
        clock = datetime.fromisoformat(utc(reported+timedelta(days=1)))
        if date.fromisoformat(report)<=reported and clock<=cutoff:
            return clock.isoformat(),interest,long-short,path,meta["sha256"]
    raise ValueError(f"No verified CFTC vintage before {cutoff.isoformat()}")


def curve_inputs(issue, sessions, contracts, books):
    stop = bisect_left(sessions,issue)-1
    day = sessions[stop]
    active = [month for month,expiry in contracts if expiry>=day][:4]
    if len(active)!=4:
        raise ValueError(f"Missing active delivery months on {day}")
    source = next((name for name, book in books.items() if all((day,m) in book for m in active)),None)
    if not source:
        raise ValueError(f"No complete previous-session WTI curve for {issue}")
    book = books[source]
    front,second,_,far = (book[day,m] for m in active)
    lag = sessions[stop-5]
    lag_eligible = {month for month,expiry in contracts if expiry>=lag}
    lag_ranks = [m for m,_ in contracts if m in lag_eligible][:4]
    older = [book.get((lag,m)) if m in lag_ranks else None for m in active[:2]]
    slope = round(front-second,4)
    change = slope-(older[0]-older[1]) if all(v is not None for v in older) else np.nan
    return day,source,slope,round(front-far,4),change


def new_features(releases, *, pending_as_of=None):
    clocks = stack.indexed("eia_release_clock")
    _, archive_marks = archives(as_of=pending_as_of)
    with closing(read_only(SOURCE)) as db:
        vintage = ensemble.public_vintages(db,max(clocks))
        sessions = full_sessions(db,releases[-1]["release_date"])
        contracts = list(db.execute("SELECT delivery_month,last_trading_date FROM wti_delivery_calendar ORDER BY delivery_month"))
        books = {s:{(d,m):float(v) for d,m,_,v in price_rows(db,s,numeric_through=releases[-1]["release_date"])} for s in ("archive","ibkr")}
    books["archive"].update({k:v[0]/100 for k,v in archive_marks.items()})
    geometry, context, metadata, rows = {}, {}, {}, []
    for r in releases:
        issue,week,prior_week = (r[k] for k in ("release_date","latest_week_in_issue","previous_week_ending"))
        cutoff,known = (r[k] for k in ("conservative_prerelease_before_utc","conservative_postrelease_from_utc"))
        available = [k for k,v in clocks.items() if k<issue and datetime.fromisoformat(v["conservative_postrelease_from_utc"])<=datetime.fromisoformat(cutoff)]
        previous_issue = max(available)
        previous_week = clocks[previous_issue]["latest_week_in_issue"]
        previous_base = (date.fromisoformat(previous_week)-timedelta(days=7)).isoformat()
        prior = (vintage[previous_issue,previous_week]["cushing_crude"]-vintage[previous_issue,previous_base]["cushing_crude"])/1000
        curve_asof = issue if r["cushing_change_kbbl"] is not None else min(issue,(pending_as_of.astimezone(EASTERN).date()+timedelta(days=1)).isoformat())
        day,source,carry,far,delta = curve_inputs(curve_asof,sessions,contracts,books)
        cot_time, interest, net, cot_path, cot_hash = cftc(datetime.fromisoformat(cutoff))
        clocks[issue] = dict(r,oldest_week_in_issue=prior_week)
        geometry[week,issue] = dict(raw_kbbl=None,overlap_kbbl=None,baseline_b04_space_time_fraction=None)
        context[issue] = (cutoff,previous_issue,previous_week,day,carry,far,cot_time,interest,net)
        metadata[issue] = dict(curve_day=day,curve_source=source,carry=carry,carry_change=delta,cftc_file=cot_path,cftc_sha256=cot_hash)
        angle = 2*math.pi*date.fromisoformat(week).timetuple().tm_yday/365.25
        label = float(r["cushing_change_kbbl"])/1000 if r["cushing_change_kbbl"] is not None else float("nan")
        rows.append(dict(study_phase="forward",release_date=issue,week_ending=week,forecast_cutoff_utc=cutoff,target_known_by_utc=known,
                         target_million_bbl=label,label=label,prior_million_bbl=prior,raw_optical_million_bbl=None,optical_available=False,
                         eia_features=(prior,math.sin(angle),math.cos(angle),carry)))
        if r["cushing_change_kbbl"] is not None:
            forward_report_vintage(issue,week,prior_week,vintage)
    enriched = list(ensemble.enrich_feature_rows(rows,geometry,clocks,vintage,context))
    return enriched, metadata, sessions, contracts, books, archive_marks


def public_forecasts(releases, *, pending_as_of=None):
    plan = list(make_plan(read_protocol()))
    closed,_ = closed_inputs(plan,holdout_index("holdout_optical_expanded_view_features"),evaluate=True)
    added, metadata, sessions, contracts, books, archive = new_features(releases,pending_as_of=pending_as_of)
    study = list(ensemble.feature_rows())+closed+added
    published = list(adaptive.archived_training_rows())
    history = [dict(r,label=r["target_million_bbl"],cutoff=datetime.fromisoformat(r["forecast_cutoff_utc"]),
                   known_at=datetime.fromisoformat(r["target_known_by_utc"])) for r in published]
    with threadpool_limits(limits=1):
        first = {r["release_date"]:r for r in ensemble.replay([dict(q) for q in study])}
        second = {r["release_date"]:r for r in adaptive.replay(history,[dict(q,previous_level_million_bbl=ensemble.number(q["core"][4])/1000)
                                                                       for q in study],first)}
    historic = {r["release_date"]:r for r in stack.source_rows()}
    original = stack.indexed("cushing_robust_stack_forecasts")
    for issue,ref in original.items():
        if (not math.isclose(first[issue]["public_ensemble"],float(ref["public_ensemble"]),abs_tol=1e-10)
                or not math.isclose(second[issue]["adaptive_public"],float(ref["adaptive_public"]),abs_tol=1e-10)):
            raise ValueError(f"Historical public forecasts changed on {issue}")
    for r in added:
        issue = r["release_date"]
        public = dict(public_ensemble=first[issue]["public_ensemble"],adaptive_public=second[issue]["adaptive_public"],
                      prior=r["prior_million_bbl"],zero=0.)
        historic[issue] = dict(r,target=r["label"],known=r["known_at"],forecasts=public,
                              public=np.array([public[k] for k in stack.PUBLIC]),optical=None)
    predictions_by_date = {}
    for r in stack.replay(historic[k] for k in sorted(historic)):
        issue = r["release_date"]
        if issue in metadata:
            predictions_by_date[issue] = r
        elif not math.isclose(r["robust_public"],float(original[issue]["robust_public"]),abs_tol=1e-10):
            raise ValueError(f"Historical robust forecast changed on {issue}")
    return predictions_by_date, metadata, sessions, contracts, books, archive


def prospective(releases):
    new, metadata, sessions, contracts, books, archives_by_leg = public_forecasts(releases)
    risk_book = list(risk_inputs(list(inputs())))
    delayed_book = [r | dict(long_gross=moves[-1]-moves[1],
                             volatility_usd=max(RISK_FLOOR_USD, r["volatility_usd"]*math.sqrt(3/4)))
                    for r, _, _, moves in paths(risk_book)]
    _, _, expanded_book = delayed_calendar_replay()
    original_forecasts = stack.indexed("cushing_robust_stack_forecasts")
    public_expectations = {issue:float(r["public_ensemble"]) for issue,r in (original_forecasts | new).items()}
    expiries = [expiry for _,expiry in contracts]
    last_exit = risk_book[-1]["exit_at"]
    expanded_last_exit = expanded_book[-1]["exit_at"]
    for r in releases:
        issue = r["release_date"]
        decision = datetime.fromisoformat(r["conservative_postrelease_from_utc"])
        start = bisect_right(sessions,issue)
        while settlement(sessions[start])<=decision:
            start += 1
        days = sessions[start:start+5]
        front = bisect_right(expiries,sessions[start+6])
        near,far = contracts[front][0],contracts[front+3][0]
        entry,exit_at = map(settlement,(days[0],days[-1]))
        dates = sessions[bisect_left(sessions,issue)-RISK_LOOKBACK-1:bisect_left(sessions,issue)]
        risk_source = next((s for s,book in books.items() if all((d,m) in book for d in dates for m in (near,far))),None)
        if risk_source is None:
            raise ValueError(f"No complete prior fixed-contract risk series on {issue}")
        risk_prices = books[risk_source]
        spread = np.array([10*(tick_price(risk_prices[d,near])-tick_price(risk_prices[d,far])) for d in dates])
        sigma = spread_volatility(spread)
        forecast = new[issue]["robust_public"]
        target = new[issue]["target"]
        eligible = entry>last_exit
        expanded_eligible = settlement(days[1])>expanded_last_exit
        history = [row for row in risk_book if row["exit_at"]<decision][-HISTORY:]
        expanded_history = [row for row in expanded_book if row["exit_at"]<decision][-HISTORY:]
        estimates = dict.fromkeys(MODEL_WINDOWS)
        delayed_history = [row for row in delayed_book if row["exit_at"]<decision][-HISTORY:]
        delayed_sigma = max(RISK_FLOOR_USD,sigma*math.sqrt(3/4))
        if target is not None:
            carry,change = (metadata[issue][k] for k in ("carry","carry_change"))
            surprise = target-forecast
            current = dict(x=(target,surprise,carry,change,surprise*carry),volatility_usd=sigma,exit_at=exit_at)
            expanded_current = current | dict(x=(*current["x"],0.,0.),volatility_usd=delayed_sigma)
        if eligible and target is not None and len(history)>=MIN_WINDOWS:
            estimates["public"] = predictions(history,current,5)
            estimates["public_volatility"] = risk_prediction(history,current,5)
            carry_history = [q | {"x":q["x"][2:4]} for q in history]
            estimates["carry_only_volatility"] = risk_prediction(carry_history,dict(x=(carry,change),volatility_usd=sigma),2)
            estimates["public_volatility_second_session"] = risk_prediction(delayed_history,current | {"volatility_usd":delayed_sigma},5)
        if expanded_eligible and target is not None and len(expanded_history)>=MIN_WINDOWS:
            estimates[EXPANDED_METHOD] = risk_prediction(expanded_history,expanded_current,5)
            for method,column in ((INTERACTION_METHOD,3),(CARRY_INTERACTION_METHOD,5)):
                design = interaction_features(expanded_current)[column]
                train = [row | dict(x=interaction_features(row)[column]) for row in expanded_history]
                estimates[method] = risk_prediction(train,expanded_current | dict(x=design),len(design))
            train = [row | dict(x=replace_expectation(row,public_expectations[row["release_date"]])) for row in expanded_history]
            design = replace_expectation(expanded_current,public_expectations[issue])
            estimates[ENSEMBLE_INTERACTION_METHOD] = risk_prediction(train,expanded_current | dict(x=design),len(design))
        exit_marks = [archives_by_leg.get((d,m)) for d in days for m in (near,far)]
        marks_by_start = {first:exit_marks[2*(first-1):] for first in set(MODEL_WINDOWS.values())}
        moves = {first:10*(marks[-2][0]-marks[-1][0]-marks[0][0]+marks[1][0]) if all(marks) else None
                 for first,marks in marks_by_start.items()}
        if eligible:
            last_exit = exit_at
            if estimates["public"] is not None:
                if moves[1] is not None:
                    risk_book.append(current | dict(long_gross=moves[1]))
                if moves[2] is not None:
                    delayed_book.append(current | dict(long_gross=moves[2],volatility_usd=delayed_sigma))
        if expanded_eligible:
            expanded_last_exit = exit_at
            if moves[2] is not None and target is not None:
                expanded_book.append(expanded_current | dict(long_gross=moves[2],release_date=issue))
        for name,value in estimates.items():
            first = MODEL_WINDOWS[name]
            move = moves[first]
            model_history = expanded_history if name in EXPANDED_MODELS else history if first == 1 else delayed_history
            model_marks = marks_by_start[first]
            side = None if value is None else int(value>HURDLE)-int(value<-HURDLE)
            gross = 0 if side==0 else None if side is None or move is None else side*move
            yield dict(release_date=issue,week_ending=r["latest_week_in_issue"],forecast_cutoff_utc=r["conservative_prerelease_before_utc"],
                       forecast_million_bbl=public_expectations[issue] if name==ENSEMBLE_INTERACTION_METHOD else forecast,
                       actual_million_bbl=target,decision_at_utc=decision.isoformat(),
                       registration_deadline_utc=settlement(days[first-1]).astimezone(timezone.utc).isoformat(),
                       entry_at=settlement(days[first-1]).isoformat(),
                       exit_at=exit_at.isoformat(),near_month=near,far_month=far,method=name,estimated_long_move_usd=value,
                       near_side=side,volatility_usd=sigma if first == 1 else delayed_sigma,volatility_source=risk_source,
                       eligible_nonoverlap=expanded_eligible if name in EXPANDED_MODELS else eligible,
                       observed_leg_sessions=sum(v is not None for v in model_marks), expected_leg_sessions=len(model_marks),
                       market_window_complete=move is not None,training_windows=len(model_history),
                       latest_training_exit_utc=model_history[-1]["exit_at"].isoformat() if model_history else None,
                       historical_reconstruction=True,gross_usd=gross,
                       **{f"net_{c}_usd":gross-c*abs(side) if gross is not None else None for c in COSTS},
                       **{k:None if isinstance(v,float) and not math.isfinite(v) else v for k,v in metadata[issue].items()})


def registered_decision(row, *, register, now):
    path = REGISTERED/f"{row['release_date']}_{row['method']}.json"
    fields = {k:row[k] for k in DECISION_FIELDS}
    start, deadline = (datetime.fromisoformat(row[k]) for k in ("decision_at_utc","registration_deadline_utc"))
    if register and row["near_side"] is not None and not path.exists():
        if start<=now<deadline:
            receipt = dict(recorded_at_utc=now.isoformat(),decision=fields,model_source_sha256={
                name:hashlib.sha256((ROOT/"scripts"/name).read_bytes()).hexdigest() for name in
                ("evaluate_cushing_forward_public_cl.py","fit_cushing_cl_volatility.py","fit_cushing_regional_balance.py")})
            path.parent.mkdir(parents=True,exist_ok=True)
            try:
                with path.open("x",encoding="utf-8") as file:
                    json.dump(receipt,file,indent=2,allow_nan=False)
                    file.write("\n")
            except FileExistsError:
                pass
    if not path.exists():
        return row | dict(recorded_at_utc=None)
    receipt = json.loads(path.read_bytes())
    stamp = datetime.fromisoformat(receipt["recorded_at_utc"])
    if not start<=stamp<deadline:
        raise ValueError(f"Registration outside its decision window for {path.name}")
    old = receipt["decision"]
    if old.keys()!=fields.keys() or any(
        not math.isclose(old[k],v,abs_tol=1e-9,rel_tol=1e-10) if isinstance(v,float) and isinstance(old[k],float)
        else old[k]!=v for k,v in fields.items()
    ):
        raise ValueError(f"Registered decision does not reproduce for {path.name}")
    return row | dict(recorded_at_utc=stamp.isoformat(),historical_reconstruction=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true",help="Save decisions before the first eligible session mark")
    args = parser.parse_args()
    releases = read("eia_wpsr_forward_releases")
    rows = list(prospective(releases))
    now = datetime.now(timezone.utc)
    rows = [registered_decision(r,register=args.register,now=now) for r in rows]
    export(f"{STEM}_decisions",rows)
    report = dict(status="forward_decision_and_reconstruction_ledger", historical_public_forecasts_reproduced=len(stack.indexed("cushing_robust_stack_forecasts")),
                  candidate_source_sha256={name:hashlib.sha256((DERIVED/f"{name}.json").read_bytes()).hexdigest() for name in
                                           ("cushing_cl_volatility_summary","cushing_coverage_nowcast_summary","cushing_cl_execution_windows_summary",
                                            "cushing_cl_delayed_calendar_summary","cushing_cl_volatility_interactions_summary",
                                            "cushing_regional_balance_summary")},
                  release_rows=rows, completed_windows=len({r["release_date"] for r in rows if r["market_window_complete"]}),
                  registered_windows=len({r["release_date"] for r in rows if r["recorded_at_utc"]}),
                  registered_completed_windows=len({r["release_date"] for r in rows if r["recorded_at_utc"] and r["market_window_complete"]}))
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    for r in rows:
        print(r["release_date"],r["method"],r["forecast_million_bbl"],r["near_side"],r["observed_leg_sessions"],r["net_170_usd"])


if __name__=="__main__":
    main()
