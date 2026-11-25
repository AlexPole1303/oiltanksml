"""Nowcast weekly inventory using dated WTI moves around the survey boundary."""

import json
import math
from bisect import bisect_left
from contextlib import closing
from datetime import datetime, timedelta

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import settlement, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean, transformed
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_robust_stack import indexed
from publication_clocks import EASTERN

STEM = "cushing_inventory_market_bridge"
OPTICAL = "all_recovered"
MODELS = (PUBLIC, "carried_scene", OPTICAL, "public_curve", "optical_curve", "public_early_curve", "optical_early_curve")
DESIGNS = {"public_curve": (PUBLIC, "curve"), "optical_curve": (OPTICAL, "curve"),
           "public_early_curve": (PUBLIC, "early_curve"), "optical_early_curve": (OPTICAL, "early_curve")}


def source_rows():
    forecasts = indexed("cushing_recovered_optical_forecasts")
    contracts = indexed("wti_cushing_event_windows")
    clocks = indexed("eia_release_clock")
    early = {q["issue"]: q for q in read("cushing_prerelease_sensor_forecasts")}
    geometry = {(q["release_date"],q["week_ending"]): q
                for name in ("sentinel2_space_time_coverage_weeks","holdout_optical_weeks") for q in read(name)}
    with closing(read_only(SOURCE)) as db:
        sessions = [day for day, in db.execute("SELECT session_date FROM wti_session_basis WHERE full_cmes_session=1 "
                                               "AND cme_no_settlement_notice_url IS NULL ORDER BY session_date")]
        books = {source:{(d,m):tick_price(p) for d,m,_,p in price_rows(db,source,numeric_through=max(forecasts))}
                 for source in ("archive","ibkr")}
    for issue, f in sorted(forecasts.items()):
        week = f["week_ending"]
        g = geometry[issue,week]
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc","known_utc"))
        start, end = (datetime.fromisoformat(g[f"{which}_measurement_at_utc"]) for which in ("previous","current"))
        release_known = datetime.fromisoformat(clocks[issue]["conservative_postrelease_from_utc"])
        if not start < end < cutoff < known or known != release_known or cutoff != datetime.fromisoformat(g["cutoff_utc"]):
            raise ValueError(f"Different inventory clocks on {issue}")
        if f["last_training_label_utc"] and datetime.fromisoformat(f["last_training_label_utc"]) >= cutoff:
            raise ValueError(f"The optical forecast used a late label on {issue}")
        marks = []
        for when in (start,end,cutoff):
            day = when.astimezone(EASTERN).date().isoformat()
            pos = bisect_left(sessions,day)-1
            if pos < 0 or settlement(sessions[pos]) >= when:
                raise ValueError(f"Unavailable prior close on {issue}")
            marks.append(sessions[pos])
        if not marks[0] < marks[1] <= marks[2]:
            raise ValueError(f"Reversed weekly close dates on {issue}")
        w = contracts[issue]
        legs = (w["near_month"],w["far_month"])
        keys = [(day,m) for day in marks for m in legs]
        source = next((s for s,b in books.items() if all(k in b for k in keys)),None)
        vector = (math.nan,)*4
        if source:
            prices = np.asarray([books[source][k] for k in keys],dtype=float).reshape(3,2)/100
            flat = np.diff(prices[:,0])/max(abs(prices[0,0]),15)
            spread = np.diff(prices[:,0]-prices[:,1])
            vector = (float(flat[0]),float(spread[0]),float(flat[1]),float(spread[1]))
        raw = {k:float(f[k]) for k in ("actual",PUBLIC,"carried_scene",OPTICAL)}
        early_public, early_optical = math.nan, math.nan
        e = early.get(issue)
        if e is not None:
            early_cutoff = datetime.fromisoformat(e["cutoff_utc"])
            if (not start < early_cutoff < cutoff or e["week_ending"]!=week or
                    float(e["inventory"])!=raw["actual"] or datetime.fromisoformat(e["known_utc"])!=known or
                    any(e[k] and datetime.fromisoformat(e[k]) >= early_cutoff
                        for k in ("latest_inventory_training_utc","latest_optical_input_utc"))):
                raise ValueError(f"Invalid earlier forecast on {issue}")
            early_public = float(e["early_public_ml"])-raw[PUBLIC]
            early_optical = float(e["early_recovered_scene"])-raw[OPTICAL]
        yield raw | dict(release_date=issue,week_ending=week,study_phase=f["study_phase"],cutoff=cutoff,known=known,
                         source=source,mark_sessions=marks,near_month=legs[0],far_month=legs[1],
                         measured=int(f["total_recovered_tanks"])>0,curve=vector,
                         early_curve=(*vector, early_public, early_optical))


def model(sample, current, target, design):
    x, z = transformed(sample,current,design)
    y = np.asarray([q["actual"]-q[target] for q in sample])
    lag = np.asarray([(current["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in sample])
    timeliness = np.exp2(-lag/78)
    penalty = np.eye(x.shape[1])*.8*timeliness.sum()
    beta = np.zeros(x.shape[1])
    # Penalized median regression, centered covariates and no free intercept.
    for _ in range(6):
        weights = timeliness/np.maximum(abs(y-x@beta),.20)
        beta = solve(x.T@(weights[:,None]*x)+penalty,x.T@(weights*y),assume_a="pos")
    return float(np.clip(z@beta,-.6,.6))


def replay():
    history = []
    for q in source_rows():
        train = [r for r in history if r["known"] < q["cutoff"] and r["source"]][-HISTORY:]
        forecast = {m:q[target]+model(train,q,target,x) if len(train)>=MIN_WINDOWS and q["source"] else q[target]
                    for m,(target,x) in DESIGNS.items()}
        history.append(q)
        yield q | forecast | dict(latest_training_label_utc=train[-1]["known"].isoformat() if train else None,
                                  training_releases=len(train))


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        trades = list(trading(rows,MODELS))
    report = dict(status="exploratory_on_reviewed_history", forecasts={}, trading={})
    for phase in PHASES:
        sample = [q for q in rows if q["study_phase"]==phase]
        report["forecasts"][phase] = {}
        for scope,group in (("all",sample),("dated_market",[q for q in sample if q["source"]]),
                            ("measured",[q for q in sample if q["measured"]])):
            stats = forecast_metrics(group,MODELS)
            for key,(target,_) in DESIGNS.items():
                delta = [1000*(abs(q["actual"]-q[key])-abs(q["actual"]-q[target])) for q in group]
                if group:
                    stats["models"][key]["delta_vs_matched_baseline_95_kbbl"] = block_bootstrap_mean(delta)
            report["forecasts"][phase][scope] = stats
        report["trading"][phase] = {}
        base = [q for q in trades if q["study_phase"]==phase and q["method"]==OPTICAL]
        for key in ("curve",*MODELS):
            group = [q for q in trades if q["study_phase"]==phase and q["method"]==key]
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different CL dates on {phase}/{key}")
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(group,base,strict=True)]
            report["trading"][phase][key] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(q["gross_usd"]-340*abs(q["near_side"]) for q in group),
                delta_vs_optical_95_usd=block_bootstrap_mean(delta))
        print(phase,{m:round(z["mae_kbbl"],2) for m,z in report["forecasts"][phase]["all"]["models"].items()})
        print(phase,{m:z["net_170_usd"] for m,z in report["trading"][phase].items()})
    export(f"{STEM}_forecasts",[{k:q[k] for k in ("release_date","week_ending","study_phase","actual","source","near_month","far_month",
            "measured","latest_training_label_utc","training_releases",*MODELS)}
        | dict(cutoff_utc=q["cutoff"].isoformat(),known_utc=q["known"].isoformat(),
               survey_start_previous_close=q["mark_sessions"][0],survey_end_previous_close=q["mark_sessions"][1],
               forecast_previous_close=q["mark_sessions"][2])
        | {k:v if math.isfinite(v) else None for k,v in zip(
            ("survey_flat_return","survey_spread_move","post_survey_flat_return","post_survey_spread_move",
             "early_public_minus_public","early_optical_minus_optical"),q["early_curve"],strict=True)} for q in rows])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
