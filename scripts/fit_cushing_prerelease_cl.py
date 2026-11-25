"""Test satellite information before publication using exact dated CL contracts."""

import json
import math
from bisect import bisect_left
from collections import defaultdict
from contextlib import closing
from datetime import datetime

import numpy as np

from backtest_wti_cushing_spread import COSTS, settlement, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, RISK_FLOOR_USD, risk_inputs, risk_prediction
from fit_cushing_nowcast_ensemble import block_bootstrap_mean, ridge_fit, transformed
from fit_cushing_observation_operator import physical_update, source_rows
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_prerelease"
MODELS = ("curve_only", "public_nowcast", "public_and_optical", "public_and_lagged_optical", "early_public_ml",
          "early_public_ml_and_optical", "optical_direction", "always_long")


def inputs():
    baseline = indexed("cushing_robust_stack_forecasts")
    observation = indexed("cushing_observation_operator_forecasts")
    clocks = indexed("eia_release_clock")
    windows = indexed("wti_cushing_event_windows")
    released = sorted((datetime.fromisoformat(clocks[r["release_date"]]["conservative_postrelease_from_utc"]),
                       r["week_ending"],float(r["change_kbbl_as_known_at_release"])/1000,float(r["cushing_kbbl"])/1000)
                      for r in read("eia_inventory_release_panel") if r["release_date"] in clocks and r["change_kbbl_as_known_at_release"])
    with closing(read_only(SOURCE)) as db:
        sessions = [r[0] for r in db.execute("SELECT session_date FROM wti_session_basis WHERE full_cmes_session=1 "
                                           "AND cme_no_settlement_notice_url IS NULL ORDER BY session_date")]
        books = {s:{(day,month):(tick_price(value),ref) for day,month,ref,value in price_rows(db,s,numeric_through=sessions[-1])}
                 for s in ("archive","ibkr")}
    coverage = defaultdict(lambda:defaultdict(int))
    rows = []
    for issue,f in sorted(baseline.items()):
        phase = f["study_phase"]
        clock = clocks[issue]
        start = bisect_left(sessions,issue)
        if start<2 or start==len(sessions) or sessions[start]!=issue:
            coverage[phase]["no_regular_release_session"]+=1
            continue
        if not clock["scheduled_table_release_at_utc"] or not clock["timing_basis"].startswith(("standard_schedule","dated_holiday_schedule")):
            coverage[phase]["release_clock_unavailable"]+=1
            continue
        day = sessions[start-1]
        entry = settlement(day)
        reported = datetime.fromisoformat(clock["scheduled_table_release_at_utc"])
        exit_at = settlement(issue)
        cutoff = datetime.fromisoformat(f["forecast_cutoff_utc"])
        if not entry < cutoff <= reported < exit_at:
            coverage[phase]["release_outside_mark_window"]+=1
            continue
        # Use the same months as the audited post-release strategy. Use market
        # prices through the penultimate session and allow daily archive delivery
        # through the beginning of the entry session's local calendar day.
        information_day = sessions[start-2]
        decision = datetime.fromisoformat(day).replace(tzinfo=entry.tzinfo)
        w = windows[issue]
        near,far = w["near_month"],w["far_month"]
        legs = [(d,m) for d in (information_day,day,issue) for m in (near,far)]
        source = next((s for s in books if all(k in books[s] for k in legs)),None)
        if not source:
            coverage[phase]["missing_leg_marks"]+=1
            continue
        old_near,old_far,near_before,far_before,near_after,far_after = [books[source][k][0] for k in legs]
        prior_public = {week:(value,level) for stamp,week,value,level in released if stamp<=decision}
        # The stored public ensemble may use the entry-day close. Rebuild a leaner
        # expectation from earlier releases and the penultimate exact-contract mark.
        weeks = sorted(prior_public)[-4:]
        recent = [prior_public[week][0] for week in weeks]
        if len(recent)!=4:
            raise ValueError(f"No complete past inventory series on {issue}")
        lag1 = recent[-1]
        lag4 = sum(recent)/len(recent)
        angle = 2*math.pi*datetime.fromisoformat(f["week_ending"]).timetuple().tm_yday/365.25
        o = observation[issue]
        # Rebuild the optical regression using only pairs archived before the close
        # of the previous session. The optical feature is attached in replay().
        if float(o["actual"])!=float(f["target"]):
            raise ValueError(f"Inconsistent EIA inventory on {issue}")
        rows.append(dict(release_date=issue,study_phase=phase,near_month=near,far_month=far,source=source,
            decision=decision,entry=entry,exit_at=exit_at,reported_at=reported,
            known=datetime.fromisoformat(f["target_known_by_utc"]),week_ending=f["week_ending"],inventory=float(f["target"]),
            x=(10*(old_near-old_far),old_near/100),preliminary_inventory=.5*lag1+.5*lag4,
            inventory_scale=sum(map(abs,recent))/len(recent),
            short_nowcast_features=(lag1,lag4,prior_public[weeks[-1]][1],math.sin(angle),math.cos(angle),
                                     (old_near-old_far)/100,old_near/100),
            long_gross=10*(near_after-far_after-near_before+far_before),
            clock_basis=clock["timing_basis"]))
    return rows,coverage


def cases():
    optical = {r["release_date"]:r for r in source_rows()}
    windows,coverage = inputs()
    # Compute risk at the information cutoff. Risk_inputs ends at the session
    # strictly before its date argument, so pass the entry session date.
    marked = list(risk_inputs([r | dict(release_date=r["entry"].date().isoformat(),issue=r["release_date"]) for r in windows],allow_missing=True))
    output = []
    nowcast_history = []
    for r in marked:
        issue = r["issue"]
        if not r["decision"] < r["entry"] < r["reported_at"] < r["exit_at"]:
            raise ValueError(f"Invalid forecast and settlement clocks on {issue}")
        if r["volatility_usd"] is None:
            coverage[r["study_phase"]]["missing_risk_marks"]+=1
            continue
        if output and output[-1]["exit_at"]>=r["entry"]:
            coverage[r["study_phase"]]["overlap"]+=1
            continue
        observed = [q for q in optical[issue]["observed"] if datetime.fromisoformat(q["available_utc"])<r["decision"] and
                    q["prior_issue"] and datetime.fromisoformat(q["prior_available_utc"])<=r["decision"]]
        public = r["preliminary_inventory"]
        signal,_ = physical_update(observed,public,correlated=True)
        available_labels = [q for q in nowcast_history if q["known"]<r["decision"]][-156:]
        if len(available_labels)>=52:
            predictors,design = transformed(available_labels,r,"short_nowcast_features")
            y = np.asarray([q["inventory"] for q in available_labels])
            public_ml = ridge_fit(predictors,y,design,.45,robust=True)
        else:
            public_ml = public
        nowcast_history.append(r)
        optical_ml,information = physical_update(observed,public_ml,correlated=True)
        lag = output[-1]["signal"] if output and output[-1]["known"]<=r["decision"] and observed else 0.
        sigma = max(RISK_FLOOR_USD,r["volatility_usd"]/2)
        curve = (*r["x"],math.log(sigma/1000))
        base = (*curve,public,public*curve[-1])
        ml = (*curve,public_ml,public_ml*curve[-1])
        design = (curve,base,(*base,signal,signal*curve[-1]),(*base,lag,lag*curve[-1]),ml,(*ml,optical_ml,optical_ml*curve[-1]))
        output.append(r | dict(volatility_usd=sigma,design=design,signal=signal,early_public_ml=public_ml,
            early_optical_ml=optical_ml,early_optical_information=information,measured_roofs=len(observed),
            latest_nowcast_training_label_utc=available_labels[-1]["known"] if available_labels else None))
    return output,coverage


def replay():
    marked,coverage = cases()
    history = []
    output = []
    for r in marked:
        issue,design,signal = r["issue"],r["design"],r["signal"]
        past = [q for q in history if q["exit_at"]<r["decision"]][-HISTORY:]
        history.append(r)
        if len(past)<MIN_WINDOWS:
            continue
        values = [risk_prediction([q | dict(x=q["design"][i]) for q in past],r | dict(x=x),len(x)) for i,x in enumerate(design)]
        sides = [int(v>HURDLE)-int(v<-HURDLE) for v in values] + [-int(signal>.02)+int(signal<-.02),1]
        for method,expected,side in zip(MODELS,(*values,None,None),sides,strict=True):
            gross = side*r["long_gross"]
            output.append({k:r[k] for k in ("study_phase","near_month","far_month","source","clock_basis")} | dict(
                release_date=issue,method=method,model_information_at_utc=r["decision"].isoformat(),
                entry_at_utc=r["entry"].isoformat(),scheduled_report_at_utc=r["reported_at"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),
                latest_training_exit_utc=past[-1]["exit_at"].isoformat(),measured_roofs=r["measured_roofs"],
                early_public_forecast_million_bbl=r["early_public_ml"],actual_eia_change_million_bbl=r["inventory"],
                latest_nowcast_training_label_utc=r["latest_nowcast_training_label_utc"].isoformat() if r["latest_nowcast_training_label_utc"] else None,
                optical_update_million_bbl=signal,expected_long_move_usd=expected,near_side=side,gross_usd=gross,
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS}))
    return output,coverage


def main():
    rows,coverage = replay()
    phases = {}
    for p in PHASES:
        phases[p] = {}
        control = [r for r in rows if r["study_phase"]==p and r["method"]=="public_nowcast"]
        for method in MODELS:
            group = [r for r in rows if r["study_phase"]==p and r["method"]==method]
            if [r["release_date"] for r in group] != [r["release_date"] for r in control]:
                raise ValueError(f"Unequal windows for {p}/{method}")
            delta = [r["net_170_usd"]-c["net_170_usd"] for r,c in zip(group,control,strict=True)]
            phases[p][method] = robustness(group) | dict(increment_usd=sum(delta),delta_mean_95_usd=block_bootstrap_mean(delta),
                measured_windows=sum(r["measured_roofs"]>0 for r in group))
        print(p,{m:(q["windows"],q["trades"],q["net_170_usd"],q["measured_windows"]) for m,q in phases[p].items()})
    report = dict(status="exploratory_prepublication_event_replay",phase_exclusions=coverage,phases=phases)
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
