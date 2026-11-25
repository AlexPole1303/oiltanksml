"""Estimate weekly Cushing stock changes from dated, partial tank observations."""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.linalg import solve

from backtest_wti_cushing_spread import COSTS, NEW_YORK
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as delayed_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features
from fit_cushing_weekly_change_ml import VOLUMES
from publication_clocks import utc

STEM = "cushing_observation_operator"
PUBLIC = "public_ensemble"
METHODS = (PUBLIC, "independent_observations", "correlated_observations", "joint_temporal_model", "adaptive_tank_model",
           "lagged_optical_control")
PHASES = ("development", "validation", "holdout")
NOISE_FLOOR_M = .9
LOCAL_ACTIVITY_M = 1.45
COMMON_ACTIVITY_M = .45
TRADE_METHODS = ("observed_inventory", "optical_as_feature", "optical_in_surprise", "lagged_optical_as_feature")


def source_rows():
    labels = indexed("cushing_robust_stack_forecasts")
    archived = defaultdict(list)
    for r in read("eia_inventory_release_panel"):
        if r["change_kbbl_as_known_at_release"]:
            archived[r["week_ending"]].append(r)
    clocks = indexed("eia_release_clock")
    geometry = indexed("cushing_coverage_nowcast_forecasts")
    parts = defaultdict(list)
    for stem in VOLUMES:
        for r in read(stem):
            parts[r["release_date"]].append(r)
    for issue,f in sorted(labels.items()):
        cutoff,known = (datetime.fromisoformat(f[k]) for k in ("forecast_cutoff_utc","target_known_by_utc"))
        if not cutoff < known:
            raise ValueError(f"Invalid public chronology on {issue}")
        measured = parts[issue]
        if bool(measured) != (f["optical_available"]=="True") or len({r["candidate_id"] for r in measured}) != len(measured):
            raise ValueError(f"Unexpected measured roster on {issue}")
        old = geometry[issue]
        area_fraction = float(old["area_fraction"])
        total_area = math.fsum(float(r["circular_area_m2"]) for r in measured)/area_fraction if measured else None
        baselines = {}
        rows = []
        for r in measured:
            first,start,last,end,available = [datetime.fromisoformat(r[k]) for k in (
                "earlier_capture_at_utc","previous_measurement_at_utc","later_capture_at_utc","current_measurement_at_utc","archive_available_at_utc")]
            if not first <= start < last <= end or not last <= available < cutoff or datetime.fromisoformat(r["cutoff_utc"])!=cutoff:
                raise ValueError(f"Invalid optical chronology on {issue}/{r['candidate_id']}")
            if start not in baselines:
                candidates = [q for q in archived[start.date().isoformat()] if q["release_date"] in clocks and
                              datetime.fromisoformat(clocks[q["release_date"]]["conservative_postrelease_from_utc"])<cutoff]
                if candidates:
                    prior = min(candidates,key=lambda q:q["release_date"])
                    issue_date = prior["release_date"]
                    baselines[start] = (float(prior["change_kbbl_as_known_at_release"])/1000,issue_date,
                                        clocks[issue_date]["conservative_postrelease_from_utc"])
                else:
                    baselines[start] = float(f["prior"]),None,None
            prior,prior_issue,prior_available = baselines[start]
            prior_friday = start.date()-timedelta(days=7)
            previous = datetime.fromisoformat(utc(prior_friday,start.astimezone(NEW_YORK).time()))
            if first<previous:
                raise ValueError(f"Prior fraction spans more than a week on {issue}")
            previous_fraction = (start-first)/(start-previous)
            current_fraction = (last-start)/(end-start)
            area = float(r["circular_area_m2"])
            k = float(r["circular_kbbl_per_m"])/1000
            low,high = (float(r[name])/1000 for name in ("joint_loss_min_change_kbbl","joint_loss_max_change_kbbl"))
            value = float(r["capture_change_kbbl"])/1000
            if k<=0 or not low-1e-10<=value<=high+1e-10:
                raise ValueError(f"Invalid pair estimate on {issue}/{r['candidate_id']}")
            # Conditional noise allowance combines native-pixel ambiguity and local tank
            # throughput variability. Its units are million barrels for this exact tank.
            duration = previous_fraction+current_fraction
            variance = ((NOISE_FLOOR_M*k)**2+((high-low)/2)**2+(LOCAL_ACTIVITY_M*k)**2*duration)
            rows.append(dict(candidate_id=r["candidate_id"],terminal_id=r["terminal_id"],prior_issue=prior_issue,
                available_utc=available.isoformat(),prior_available_utc=prior_available,
                pair_signal=value, prior_component=area/total_area*previous_fraction*prior,
                sensitivity=area/total_area*current_fraction, volume_per_m=k, variance=variance, duration=duration,
                area=area, scene_before=r["earlier_sentinel2_version"],scene_after=r["later_sentinel2_version"]))
        yield dict(release_date=issue,week_ending=f["week_ending"],study_phase=f["study_phase"],cutoff=cutoff,known=known,
                   observed=rows,actual=float(f["target"]),public=float(f[PUBLIC]),area_fraction=area_fraction)


def physical_update(observations, baseline, *, correlated, activity=None):
    if not observations:
        return 0.,0.
    s = np.array([r["sensitivity"] for r in observations])
    e = np.array([r["pair_signal"]-r["prior_component"]-r["sensitivity"]*baseline for r in observations])
    variance = np.array([r["variance"] for r in observations])
    if activity is not None:
        variance += np.array([(activity[r["candidate_id"]]-LOCAL_ACTIVITY_M**2)*r["volume_per_m"]**2*r["duration"] for r in observations])
    # A soft cap prevents a single anomalous estimate from representing the entire hub.
    e = np.clip(e,-3*np.sqrt(variance),3*np.sqrt(variance))
    covariance = np.diag(variance)
    if correlated:
        u = np.asarray([COMMON_ACTIVITY_M*r["volume_per_m"]*math.sqrt(r["duration"]) for r in observations])
        for key,weight in (("terminal_id",1.),("scene_before",.5),("scene_after",.5)):
            values = np.asarray([r[key] for r in observations])
            covariance += weight*np.outer(u,u)*(values[:,None]==values[None,:])
    joint = solve(covariance,np.column_stack((s,e)),assume_a="pos")
    information = s@joint[:,0]
    score = s@joint[:,1]
    return float(np.clip(score/(1/.9**2+information),-1.25,1.25)),float(information)


def learned_beta(rows,name):
    if len(rows)<32:
        return 0.
    # Use one slope, past releases only. The penalty favours the public forecast on sparse data.
    x = np.asarray([r[name] for r in rows]); y = np.array([r["actual"]-r["public"] for r in rows])
    weights = np.exp2(-np.arange(len(rows)-1,-1,-1)/52)
    beta = 0.
    for _ in range(6):
        w = weights/np.hypot(1.,(y-x*beta)/.6)
        beta = float(np.clip((w*x)@y/(w@(x*x)+.05*weights.sum()),0.,1.5))
    return beta


def reported_activity(history, observations):
    """Shrink past tank-level deviations toward the pooled rate before using them as weights."""
    counts = defaultdict(list)
    for q in history[-104:]:
        for o in q["observed"]:
            mismatch = ((o["pair_signal"]-o["prior_component"]-o["sensitivity"]*q["actual"])/o["volume_per_m"])
            counts[o["candidate_id"]].append(min(mismatch*mismatch/max(o["duration"],.25),16.))
    return {r["candidate_id"]:float(np.clip((12*LOCAL_ACTIVITY_M**2+math.fsum(counts[r["candidate_id"]]))/
            (12+len(counts[r["candidate_id"]])),.5,8.)) for r in observations}


def replay():
    seen = []
    for r in source_rows():
        obs = r["observed"]
        history = [q for q in seen if q["known"]<r["cutoff"]]
        measured = [q for q in history if q["observed"]][-104:]
        independent, info_ind = physical_update(obs,r["public"],correlated=False)
        correlated, info_corr = physical_update(obs,r["public"],correlated=True)
        beta = learned_beta(measured,"correlated_raw")
        adaptive, info_adapt = physical_update(obs,r["public"],correlated=True,activity=reported_activity(history,obs))
        adaptive_beta = learned_beta(measured,"adaptive_raw")
        last = measured[-1] if measured else None
        lag = last["correlated_raw"] if last and obs and r["cutoff"]-last["cutoff"]<=timedelta(days=14) else 0.
        lag_beta = learned_beta(measured,"lag_raw")
        result = {k:r[k] for k in ("release_date","week_ending","study_phase","actual","area_fraction")} | dict(
            forecast_cutoff_utc=r["cutoff"].isoformat(),target_known_by_utc=r["known"].isoformat(),
            last_training_release_utc=history[-1]["known"].isoformat() if history else None,
            measured_roofs=len(obs),terminal_count=len({q["terminal_id"] for q in obs}),
            independent_information=info_ind,correlated_information=info_corr,adaptive_information=info_adapt,
            prior_measured_releases=len(measured),
            public_ensemble=r["public"],independent_observations=r["public"]+independent,
            correlated_observations=r["public"]+correlated,learned_weight=beta,
            joint_temporal_model=r["public"]+beta*correlated,adaptive_tank_model=r["public"]+adaptive_beta*adaptive,
            adaptive_weight=adaptive_beta,lagged_optical_control=r["public"]+lag_beta*lag)
        seen.append(r | dict(correlated_raw=correlated,adaptive_raw=adaptive,lag_raw=lag))
        yield result, [dict(release_date=r["release_date"],**q) for q in obs], seen[-1]


def summary(group):
    actual = np.asarray([r["actual"] for r in group]); base = np.abs(actual-[r[PUBLIC] for r in group])
    methods = {}
    for method in METHODS:
        prediction = np.asarray([r[method] for r in group]); error = np.abs(actual-prediction)
        methods[method] = dict(mae_kbbl=float(1000*error.mean()),direction=int(np.count_nonzero(actual*prediction>0)),
                               paired_95_kbbl=block_bootstrap_mean(1000*(error-base)))
    return dict(releases=len(group),methods=methods)


def market_replay(forecasts):
    by_issue = {r["release_date"]:r for r in forecasts}
    control = {r["release_date"]:r for r in read("cushing_cl_volatility_interactions_trades")
               if r["method"]=="inventory_interaction"}
    _,_,windows = delayed_calendar()
    seen = []
    for r in windows:
        issue = r["release_date"]
        f = by_issue[issue]
        if not datetime.fromisoformat(f["forecast_cutoff_utc"])<datetime.fromisoformat(f["target_known_by_utc"])<=r["decision"]:
            raise ValueError(f"Inventory unavailable on {issue}")
        x = r["x"]
        increment = f["joint_temporal_model"]-f[PUBLIC]
        lag = f["lagged_optical_control"]-f[PUBLIC]
        surprise = x[1]-increment
        replacement = (x[0],surprise,x[2],x[3],surprise*x[2],0.,0.)
        vector = (features(r)[3],features(r | dict(x=(*x[:5],increment,increment*x[2])))[4],
                  features(r | dict(x=replacement))[3],features(r | dict(x=(*x[:5],lag,lag*x[2])))[4])
        train = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        seen.append(r | dict(vector=vector))
        if len(train)<MIN_WINDOWS:
            continue
        for i,method in enumerate(TRADE_METHODS):
            design = vector[i]
            value = risk_prediction([q | dict(x=q["vector"][i]) for q in train],r | dict(x=design),len(design))
            side = int(value>HURDLE)-int(value<-HURDLE)
            gross = side*r["long_gross"]
            if method==TRADE_METHODS[0] and (not math.isclose(value,float(control[issue]["expected_long_gross_usd"]),abs_tol=1e-9)
                                           or gross-HURDLE*abs(side)!=int(control[issue]["net_170_usd"])):
                raise ValueError(f"Trading controls do not reproduce on {issue}")
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd")} | dict(
                method=method,entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),
                decision_at_utc=r["decision"].isoformat(),latest_training_exit_utc=train[-1]["exit_at"].isoformat(),
                measured_roofs=f["measured_roofs"],optical_increment_million_bbl=increment,
                expected_long_gross_usd=value,near_side=side,gross_usd=gross,alternate_source=r["alternate_source"],
                alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*v-(170 if n==5 else 85)*abs(side) for n,v in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def market_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
    for phase in PHASES:
        control = groups[phase,TRADE_METHODS[0]]
        methods = {}
        for method in TRADE_METHODS:
            sample = groups[phase,method]
            if [r["release_date"] for r in sample] != [r["release_date"] for r in control]:
                raise ValueError(f"Trading calendars differ in {phase}/{method}")
            delta = [a["net_170_usd"]-b["net_170_usd"] for a,b in zip(sample,control,strict=True)]
            methods[method] = robustness(sample) | daily_risk(sample) | dict(
                increment_usd=sum(delta),delta_mean_95_usd=block_bootstrap_mean(delta),
                changed_directions=sum(a["near_side"]!=b["near_side"] for a,b in zip(sample,control,strict=True)))
        yield phase,methods


def main():
    rows,pairs = [],[]
    for r,obs,_ in replay():
        rows.append(r)
        pairs.extend(obs)
    phases = {p:{scope:summary([r for r in rows if r["study_phase"]==p and (scope=="all" or r["measured_roofs"])])
                 for scope in ("all","measured")} for p in PHASES}
    trades = list(market_replay(rows))
    trading = dict(market_summary(trades))
    report = dict(status="historical_point_in_time_model_comparison",measurements=len(pairs),phases=phases,trading=trading)
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_measurements",pairs)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    for p,g in phases.items():
        print(p,{s:{m:round(v["mae_kbbl"],1) for m,v in q["methods"].items()} for s,q in g.items()})
        print(p,{m:(v["trades"],v["net_170_usd"],v["increment_usd"]) for m,v in trading[p].items()})


if __name__=="__main__":
    main()
