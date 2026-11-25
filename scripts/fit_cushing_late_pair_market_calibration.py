"""Estimate post-survey inventory drift separately from the week being forecast."""

import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import closing
from datetime import datetime, time, timedelta

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from backtest_wti_cushing_spread import settlement, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import estimate, prior_scale
from fit_cushing_late_pair_recovery import HYBRID, COMBINED, accepted_pairs, geometry
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL
from publication_clocks import EASTERN

STEM = "cushing_late_pair_market_calibration"
METHODS = ("calendar_drift", "price_drift", "price_drift_and_variance")
MODELS = (PUBLIC,"carried_scene",HYBRID,COMBINED,*METHODS)
ERROR_WEIGHT = np.array((.25,.35,.30,.10))


def pricing(through):
    with closing(read_only(SOURCE)) as db:
        sessions = [r[0] for r in db.execute("SELECT session_date FROM wti_session_basis WHERE full_cmes_session=1 "
                                              "AND cme_no_settlement_notice_url IS NULL ORDER BY session_date")]
        books = {source:{(d,m):tick_price(p) for d,m,_,p in price_rows(db,source,numeric_through=through)}
                 for source in ("official","archive","ibkr")}
    # A mark at settlement on the endpoint day is after the standard survey clock.
    return sessions,books


def market_state(sessions, books, boundary, cutoff, near, far):
    day = boundary.astimezone(EASTERN).date().isoformat()
    first = bisect_right(sessions,day)-1
    if first<0 or settlement(sessions[first])<boundary:
        first += 1
    end = bisect_left(sessions,cutoff.astimezone(EASTERN).date().isoformat())-1
    if end<first or not boundary<=settlement(sessions[first])<=settlement(sessions[end])<cutoff:
        return None,None,None
    previous = bisect_left(sessions,day)-1
    if previous<0 or settlement(sessions[previous])>=boundary:
        return None,None,None
    dates = (sessions[previous],sessions[end])
    look = [(d,m) for d in dates for m in (near,far)]
    reference = next((source for source,prices in books.items() if all(k in prices for k in look)),None)
    if reference is None:
        return None,None,None
    n0,f0,n1,f1 = [books[reference][key]/100 for key in look]
    return np.array(((n1-n0)/max(abs(n0),15.),(n1-f1)-(n0-f0),n0-f0)),dates[-1],reference


def state_rows():
    raw = indexed("cushing_late_pair_recovery_forecasts")
    windows = indexed("wti_cushing_event_windows")
    sessions,books = pricing(max(raw))
    chronological = sorted(raw.values(),key=lambda r:r["week_ending"])
    if len({r["week_ending"] for r in chronological})!=len(chronological):
        raise ValueError("Weekly labels are not unique")
    labels = {r["week_ending"]:r for r in chronological}
    for r in chronological:
        issue = r["release_date"]
        week = datetime.fromisoformat(r["week_ending"]).date()
        boundary = datetime.combine(week,time(7),EASTERN)
        cutoff,known = (datetime.fromisoformat(r[k]) for k in ("cutoff","known"))
        if not boundary<cutoff<known:
            raise ValueError(f"Wrong storage survey clock on {issue}")
        for key in ("latest_training_label_utc","latest_optical_at_utc"):
            if r[key] and datetime.fromisoformat(r[key])>=cutoff:
                raise ValueError(f"Forecast input arrives after the cutoff on {issue}")
        target = labels.get((week+timedelta(days=7)).isoformat())
        if target and not cutoff<datetime.fromisoformat(target["known"]):
            raise ValueError(f"The following week's EIA label is already known on {issue}")
        market,mark,provider = market_state(sessions,books,boundary,cutoff,windows[issue]["near_month"],windows[issue]["far_month"])
        angle = 2*math.pi*week.timetuple().tm_yday/365.25
        available = [q for q in chronological if q["week_ending"]<r["week_ending"] and datetime.fromisoformat(q["known"])<cutoff]
        prior = float(available[-1]["actual"]) if available else 0.
        design = np.array((math.sin(angle),math.cos(angle),prior,float(r[PUBLIC])))
        yield r | dict(week=week,cutoff=cutoff,known=known,actual=float(r["actual"]),
            after_survey_mark=mark,market_provider=provider,calendar_design=design,
            market_design=np.r_[design,market] if market is not None else None,
            following_value=float(target["actual"]) if target else None,
            following_known=datetime.fromisoformat(target["known"]) if target else None)


def fit_drift(rows, current, key):
    train = [r for r in rows if r["following_known"] and r["following_known"]<current["cutoff"] and r[key] is not None][-156:]
    if len(train)<52 or current[key] is None:
        return 0.,PRIOR_INVENTORY_SIGMA_MBBL,None,len(train)
    a = np.stack([r[key] for r in train]); y = np.array([r["following_value"] for r in train])
    age = np.array([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in train])
    w = np.exp2(-age/78.)
    center = np.average(a,axis=0,weights=w)
    scale = np.maximum(np.sqrt(np.average((a-center)**2,axis=0,weights=w)),1e-4)
    x,z = np.clip((a-center)/scale,-4,4),np.clip((current[key]-center)/scale,-4,4)
    x = np.column_stack((np.ones(len(x)),x)); z = np.r_[1.,z]
    penalty = np.diag([.7,*([1.]*(x.shape[1]-1))])*w.sum()
    beta = np.zeros(x.shape[1])
    for _ in range(6):
        weights = w/np.hypot(1.,(y-x@beta)/.75)
        beta = solve(x.T@(weights[:,None]*x)+penalty,x.T@(weights*y),assume_a="pos")
    magnitude = np.minimum(abs(y-x@beta),2.5)
    residual_sigma = np.sqrt(np.average(magnitude**2,weights=w))
    condition = float(np.clip(.5*residual_sigma+.5*PRIOR_INVENTORY_SIGMA_MBBL,.6,1.25))
    return float(np.clip(z@beta,-.65,.65)),condition,train[-1]["following_known"],len(train)


def replay():
    registered = accepted_pairs()
    history,errors = [],[]
    calibrated = defaultdict(list)
    for r in state_rows():
        issue,week,cutoff,known,actual = (r[k] for k in ("release_date","week","cutoff","known","actual"))
        base = float(r[HYBRID]); public = float(r[PUBLIC]); combined = float(r[COMBINED])
        photos = registered[issue]
        can_use = photos and r["measured"]=="False"
        raw = dict.fromkeys(METHODS,combined)
        market_estimate = fit_drift(history,r,"market_design")
        calendar_estimate = fit_drift(history,r,"calendar_design")
        if photos:
            public_errors = [q for q in errors if q["known"]<cutoff]
            scale = prior_scale(public_errors)
            specs = ((0.,PRIOR_INVENTORY_SIGMA_MBBL),
                     (calendar_estimate[0],PRIOR_INVENTORY_SIGMA_MBBL),
                     (market_estimate[0],PRIOR_INVENTORY_SIGMA_MBBL),market_estimate[:2])
            for key,(mean,sigma) in zip(("reference",*METHODS),specs,strict=True):
                previous = [p for p in calibrated[key] if p["known"]<cutoff][-78:]
                recent = [p["prepared"] for p in previous if cutoff-p["cutoff"]<=MAX_SCENE_AGE]
                prediction = geometry(photos,base,week,following_mean=mean,following_sigma=sigma)
                options = estimate(prediction,recent)
                change = float(ERROR_WEIGHT@np.clip([score/(1/scale**2+info) for _,info,score in options],-.6,.6))
                if key=="reference":
                    if not math.isclose(change,float(r["late_pair_delta"]),abs_tol=1e-10):
                        raise ValueError(f"Original late-pair forecast does not reproduce on {issue}")
                elif can_use:
                    raw[key] = base+change
                fitted = geometry(photos,actual,week,following_mean=mean,following_sigma=sigma)
                calibrated[key].append(dict(cutoff=cutoff,known=known,prepared=fitted))
        errors.append(dict(known=known,error=actual-public))
        history.append(r)
        yield {k:r[k] for k in ("release_date","week_ending","study_phase","actual","cutoff","known","late_roofs")} | dict(
               **{PUBLIC:public,HYBRID:base,COMBINED:combined,"carried_scene":float(r["carried_scene"])},
               **raw,applied=bool(can_use),after_survey_mark=r["after_survey_mark"],market_provider=r["market_provider"],
               estimated_calendar_drift=calendar_estimate[0],estimated_market_drift=market_estimate[0],
               estimated_drift_sigma=market_estimate[1],latest_calendar_training_utc=calendar_estimate[2],
               latest_market_training_utc=market_estimate[2],market_training_weeks=market_estimate[3])


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        base = list(trading(rows,MODELS))
        trades = list(replay_conditional_market(rows,base,(PUBLIC,HYBRID,COMBINED,*METHODS),references={PUBLIC:f"{PUBLIC}_conditional"}))
    report = dict(status="exploratory_on_reviewed_history",forecasts={},market={})
    for phase in PHASES:
        study = [r for r in rows if r["study_phase"]==phase]
        report["forecasts"][phase] = {}
        for k,sample in (("all",study),("late_pair_used",[r for r in study if r["applied"]])):
            block = forecast_metrics(sample,MODELS)
            for name,v in block["models"].items():
                d = [1000*(abs(r["actual"]-r[name])-abs(r["actual"]-r[COMBINED])) for r in sample]
                v["delta_vs_original_95_kbbl"] = block_bootstrap_mean(d)
            report["forecasts"][phase][k] = block
        market = report["market"][phase] = {}
        ref = [r for r in trades if r["study_phase"]==phase and r["method"]==COMBINED]
        for m in (PUBLIC,HYBRID,COMBINED,*METHODS):
            group = [r for r in trades if r["study_phase"]==phase and r["method"]==m]
            if [r["release_date"] for r in group]!=[r["release_date"] for r in ref]:
                raise ValueError(f"Different CL calendar for {phase}/{m}")
            market[m] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group),
                decisions_different=sum(r["near_side"]!=q["near_side"] for r,q in zip(group,ref,strict=True)))
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in report["forecasts"][phase]["all"]["models"].items()},
              {m:(v["net_170_usd"],v["net_340_usd"]) for m,v in market.items()})
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
