"""Evaluate regional stock and refinery signals against frozen Cushing forecasts."""

import calendar
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean, ridge_fit, transformed
from fit_cushing_observation_operator import PHASES
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features

STEM = "cushing_regional_balance"
WEEKLY = ("cushing_crude","padd2_commercial_crude","padd3_commercial_crude","padd2_crude_refiner_inputs","padd3_crude_refiner_inputs")
MONTHLY = ("padd3_to_padd2_pipeline_kbbl","padd2_to_padd3_pipeline_kbbl","padd4_to_padd2_pipeline_kbbl","padd2_to_padd4_pipeline_kbbl")
FORECASTS = ("public", "seasonal", "weekly", "weekly_monthly", "weekly_monthly_optical", "prior_public_plus_monthly")
POLICIES = ("original", "public_expectation", "public_expectation_optical", "regional_expectation", "regional_expectation_optical",
            "regional_release_only", "regional_release_and_cushing", "regional_release_and_cushing_optical")


def vintage_rows():
    clocks = indexed("eia_release_clock")
    source = defaultdict(dict)
    for r in read("eia_control_vintage"):
        metric = r["metric"]
        if metric not in WEEKLY or r["release_date"] not in clocks:
            continue
        key = r["release_date"],r["report_date"]
        is_stock = WEEKLY.index(metric)<3
        allowed_tables = {"4"} if is_stock else {"2"}
        if (metric in source[key] or r["unit"]!=("thousand_bbl" if is_stock else "thousand_bbl_per_day")
                or r["table_number"] not in allowed_tables):
            raise ValueError(f"Unexpected original WPSR value for {key}/{metric}")
        source[key][metric] = float(r["value"])/1000
    for issue, clock in sorted(clocks.items()):
        week = date.fromisoformat(clock["latest_week_in_issue"])
        current = source[issue,week.isoformat()]
        previous = source[issue,(week-timedelta(days=7)).isoformat()]
        if len(current)<3 or len(previous)<3:
            raise ValueError(f"Missing key inventory vintages on {issue}")
        values = np.array([[d.get(k,np.nan) for k in WEEKLY] for d in (current,previous)])
        values[:,1] -= values[:,0]
        yield dict(issue=issue,week=week,known=datetime.fromisoformat(clock["conservative_postrelease_from_utc"]),
                   values=values[0],change=values[0]-values[1])


def monthly_rates(row, cutoff):
    if not row or row["psm_used_for_features"]!="True":
        return np.full(2,np.nan),None
    stamp = datetime.fromisoformat(row["psm_available_at_utc"])
    if stamp>cutoff:
        raise ValueError(f"Monthly report timestamp exceeds the cutoff {cutoff}")
    month = date.fromisoformat(row["psm_data_month"])
    values = np.array([float(row[k]) if row[k] else np.nan for k in MONTHLY])
    per_day = values/(1000*calendar.monthrange(month.year,month.month)[1])
    return per_day[::2]-per_day[1::2],stamp


def weekly_features(known, week):
    eligible = [q for q in known if q["week"]<week]
    if not eligible:
        raise ValueError(f"Regional history is unavailable for {week}")
    # Use the original vintage of each report. Estimate seasonal refinery run
    # rates from a +/- four-week window in earlier years, then compare to today.
    latest = eligible[-1]
    seasonal = [q["values"][3:] for q in eligible if 45<(week-q["week"]).days and
                abs(((week-q["week"]).days+182.625)%365.25-182.625)<=28]
    normals = np.array([np.nan,np.nan])
    if seasonal:
        for i, column in enumerate(np.array(seasonal).T):
            available = column[np.isfinite(column)]
            if len(available)>=4:
                normals[i] = np.median(available)
    trend = np.array([q["change"][1:3] for q in eligible[-4:]])
    return (*trend.mean(axis=0),*(7*(latest["values"][3:]-normals)),*(7*latest["change"][3:]))


def source_rows():
    book = list(vintage_rows())
    measured = indexed("cushing_shared_scene_forecasts")
    prior_monthly = indexed("cushing_flow_extensions_forecasts")
    monthly = indexed("eia_monthly_context_weekly_features")
    by_issue = {q["issue"]:q for q in book}
    for issue,r in sorted(measured.items()):
        cutoff,target_known = (datetime.fromisoformat(r[k]) for k in ("forecast_cutoff_utc","target_known_by_utc"))
        week = date.fromisoformat(r["week_ending"])
        if not cutoff<target_known or (issue in by_issue and by_issue[issue]["known"]!=target_known):
            raise ValueError(f"Different outcome cutoffs on {issue}")
        history = [q for q in book if q["known"]<=cutoff and q["week"]<week]
        last = history[-1]
        if issue in by_issue and not math.isclose(by_issue[issue]["change"][0],float(r["actual"]),abs_tol=1e-10):
            raise ValueError(f"Different reported inventory on {issue}")
        recent = [q["change"][0] for q in history[-4:]]
        angle = 2*math.pi*week.timetuple().tm_yday/365.25
        psm,stamp = monthly_rates(monthly.get(issue),cutoff)
        public,photo = [float(r[k]) for k in ("public_ensemble","carried_scene")]
        daily = weekly_features(history,week)
        basic = (public,last["values"][0],np.mean(recent),math.sin(angle),math.cos(angle))
        f = prior_monthly[issue]
        if f["forecast_cutoff_utc"]!=r["forecast_cutoff_utc"] or float(f["target"])!=float(r["actual"]):
            raise ValueError(f"Previous monthly forecast does not match on {issue}")
        yield dict(release_date=issue,week_ending=r["week_ending"],study_phase=r["study_phase"],cutoff=cutoff,known=target_known,
                   actual=float(r["actual"]),last_wpsr_issue=last["issue"],last_wpsr_known_at=last["known"],monthly_known_at=stamp,
                   current_report=by_issue.get(issue),prior_stock_changes=tuple(q["change"] for q in history[-4:]),
                   public=public,optical=photo-public,measured_roofs=int(r["measured_roofs"]),
                   basic=basic,weekly_features=(*basic,*daily),monthly_features=(*basic,*daily,*psm),
                   prior_public_plus_monthly=float(f["public_plus_monthly"]))


def replay(rows):
    seen = []
    for r in rows:
        history = [q for q in seen if q["known"]<r["cutoff"]][-156:]
        seen.append(r)
        predictions = []
        for key in ("basic","weekly_features","monthly_features"):
            increment = 0.
            if len(history)>=52:
                x,z = transformed(history,r,key)
                y = np.asarray([q["actual"]-q["public"] for q in history])
                increment = .5*float(np.clip(ridge_fit(x,y,z,1.,robust=True),-1.,1.))
            predictions.append(r["public"]+increment)
        seasonal,weekly,monthly = predictions
        yield r | dict(seasonal=seasonal,weekly=weekly,weekly_monthly=monthly,weekly_monthly_optical=monthly+r["optical"],
                       training_releases=len(history),latest_training_label=max((q["known"] for q in history),default=None))


def evaluate_forecasts(rows):
    y = np.array([r["actual"] for r in rows])
    vectors = {k:np.array([r[k] for r in rows]) for k in FORECASTS}
    error = {k:1000*abs(v-y) for k,v in vectors.items()}
    return dict(weeks=len(rows),monthly_available=sum(r["monthly_known_at"] is not None for r in rows),methods={k:dict(
        mae_kbbl=float(e.mean()),direction=int(np.count_nonzero(vectors[k]*y>0)),delta_vs_public_95_kbbl=block_bootstrap_mean(e-error["public"]))
        for k,e in error.items()},monthly_vs_weekly_95_kbbl=block_bootstrap_mean(error["weekly_monthly"]-error["weekly"]),
        optical_vs_monthly_95_kbbl=block_bootstrap_mean(error["weekly_monthly_optical"]-error["weekly_monthly"]))


def replace_expectation(r, expected):
    target,_,carry,change,_,*extra = r["x"]
    surprise = target-expected
    return features(r | dict(x=(target,surprise,carry,change,surprise*carry,*extra)))[3]


def market(rows):
    _,_,cases = market_calendar()
    forecasts = {r["release_date"]:r for r in rows}
    incumbent = {r["release_date"]:r for r in read("cushing_cl_volatility_interactions_trades") if r["method"]=="inventory_interaction"}
    previous = []
    for r in cases:
        issue = r["release_date"]
        f = forecasts[issue]
        report = f["current_report"]
        if not (report and f["cutoff"]<report["known"]<=r["decision"]<r["entry"]<r["exit_at"] and f["actual"]==r["x"][0]):
            raise ValueError(f"Bad clock or label on {issue}")
        _,_,_,original,_,curve = features(r)
        known = np.asarray(f["prior_stock_changes"])
        expected = np.asarray([np.mean(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in known.T])
        latest = report["change"]-expected
        regional = (*latest[1:3],*np.multiply(latest[3:],7))
        public = replace_expectation(r,f["public"])
        public_optical = replace_expectation(r,f["public"]+f["optical"])
        balance = replace_expectation(r,f["weekly_monthly"])
        balance_optical = replace_expectation(r,f["weekly_monthly_optical"])
        designs = (original,public,public_optical,balance,balance_optical,(*curve,*regional),(*balance,*regional),(*balance_optical,*regional))
        history = [q for q in previous if q["exit_at"]<r["decision"]][-HISTORY:]
        current = r | dict(designs=designs)
        previous.append(current)
        if len(history)<MIN_WINDOWS:
            continue
        for idx,(method,x) in enumerate(zip(POLICIES,designs,strict=True)):
            value = risk_prediction([q | dict(x=q["designs"][idx]) for q in history],current | dict(x=x),len(x))
            side = int(value>HURDLE)-int(value<-HURDLE)
            gross = side*r["long_gross"]
            if idx==0 and not math.isclose(value,float(incumbent[issue]["expected_long_gross_usd"]),abs_tol=1e-8):
                raise ValueError(f"Original strategy changed on {issue}")
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd","alternate_source")} | dict(
                method=method,report_known_at=report["known"].isoformat(),decision_at_utc=r["decision"].isoformat(),
                entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),latest_training_exit=history[-1]["exit_at"].isoformat(),
                expected_long_move_usd=value,near_side=side,gross_usd=gross,
                alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*m-(170 if n==5 else 85)*abs(side) for n,m in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    rows = list(replay(source_rows()))
    trades = list(market(rows))
    report = dict(status="historical_replay_on_reviewed_period",forecast={},trading={})
    for p in PHASES:
        group = [r for r in rows if r["study_phase"]==p]
        report["forecast"][p] = {scope:evaluate_forecasts(subset) for scope,subset in (
            ("all",group),("optical",[r for r in group if r["measured_roofs"]]),("monthly",[r for r in group if r["monthly_known_at"]]))}
        data = report["trading"][p] = {}
        base = [r for r in trades if r["study_phase"]==p and r["method"]=="original"]
        for method in POLICIES:
            matched = [r for r in trades if r["study_phase"]==p and r["method"]==method]
            if [r["release_date"] for r in matched]!=[r["release_date"] for r in base]:
                raise ValueError(f"Trade calendar differs in {p}/{method}")
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(matched,base,strict=True)]
            data[method] = robustness(matched) | daily_risk(matched) | dict(increment_usd=sum(delta),delta_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in matched))
        print(p,{k:round(v["mae_kbbl"],2) for k,v in report["forecast"][p]["all"]["methods"].items()})
        print(p,{k:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for k,v in data.items()})
    export(f"{STEM}_forecasts",[{k:r[k] for k in ("release_date","week_ending","study_phase","actual","measured_roofs","last_wpsr_issue",
        "training_releases",*FORECASTS)} | {f"{k}_utc":r[k].isoformat() if r[k] else None for k in
            ("cutoff","known","last_wpsr_known_at","monthly_known_at","latest_training_label")} for r in rows])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
