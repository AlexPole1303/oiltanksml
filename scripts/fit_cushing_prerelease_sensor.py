"""Rebuild optical inventory estimates at the earlier, tradable pre-EIA cutoff."""

import json
import math
from datetime import date, datetime, timedelta

import numpy as np
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, risk_prediction
from fit_cushing_intraweek import FIXED_MODELS, predict, preparation
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_prerelease_cl import cases
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed

STEM = "cushing_prerelease_sensor"
FORECASTS = ("early_public_ml","early_public_scene","early_adaptive_scene","early_recovered_scene")
METHODS = ("curve","simple","simple_optical","public_ml","original_optical","direct_scene","adaptive_scene","recovered_scene",
           "direct_scene_measured_only","adaptive_scene_measured_only","recovered_scene_measured_only",
           "symmetric_release_response","one_sided_release_response",
           "public_direction","adaptive_direction","recovered_direction")


def source_book():
    area = {r["release_date"]:float(r["dated_reference_area_m2"]) for name in (
        "sentinel2_space_time_coverage_weeks","holdout_optical_weeks") for r in read(name)}
    archives = indexed("eia_release_clock")
    priors = {}
    for r in read("eia_inventory_release_panel"):
        if r["reported_as"]!="current_week" or r["release_date"] not in archives or not r["change_kbbl_as_known_at_release"]:
            continue
        stamp = datetime.fromisoformat(archives[r["release_date"]]["conservative_postrelease_from_utc"])
        week = r["week_ending"]
        if week not in priors or stamp<priors[week][0]:
            priors[week] = stamp,float(r["change_kbbl_as_known_at_release"])/1000
    baseline = indexed("cushing_robust_stack_forecasts")
    book = {}
    for state,arms,_ in combined_sources():
        issue = state["release_date"]
        week = date.fromisoformat(state["week_ending"])
        clock, confirmed = priors.get((week-timedelta(days=7)).isoformat(),(None,None))
        expected = float(baseline[issue]["prior"])
        # Expanded measurements carry the previous report's published change
        # in their prior component. Admit them only when its vintage is exact.
        original = state["observed"]
        occupied = {r["candidate_id"] for r in original}
        expanded = [r for r in arms[2] if r["candidate_id"] not in occupied]
        if clock is None or not math.isclose(expected,confirmed,abs_tol=1e-12):
            expanded = []
        book[issue] = state["known"],area[issue],original,expanded,clock
    return book


def visible(r, record):
    known,area,stored,added,prior_at = record
    if known!=r["known"] or not r["decision"]<r["entry"]<r["reported_at"]<r["exit_at"]:
        raise ValueError(f"The optical and market cutoffs differ on {r['issue']}")
    cutoff = r["decision"]
    original = [q for q in stored if datetime.fromisoformat(q["available_utc"])<cutoff and q["prior_issue"] and
                datetime.fromisoformat(q["prior_available_utc"])<=cutoff]
    if len(original)!=r["measured_roofs"]:
        raise ValueError(f"The initial pre-release optical control differs on {r['issue']}")
    recovered = [q for q in added if prior_at<=cutoff and datetime.fromisoformat(q["available_utc"])<cutoff]
    return area,original,recovered


def event_response(training, row, curve_estimate, innovation, *, symmetric):
    past = [q for q in training if q["curve_estimate"] is not None and q["known"]<row["decision"]][-104:]
    if len(past)<52 or not innovation:
        return curve_estimate
    x = np.array([q["inventory"]-q["early_public_ml"] for q in past])
    y = np.array([(q["long_gross"]-q["curve_estimate"])/q["volatility_usd"] for q in past])
    vol = np.array([q["design"][0][-1] for q in past])
    age = np.array([(row["decision"]-q["decision"]).days/7 for q in past])
    time_weight = np.exp2(-age/52)*np.exp(-.5*np.square((vol-row["design"][0][-1])/.8))

    def slope(mask):
        w = time_weight*mask
        beta = 0.
        for _ in range(5):
            robust = w/np.hypot(1., (y-x*beta)/1.5)
            beta = float(np.clip((robust*x)@y/(robust@(x*x)+.15*time_weight.sum()),-1.,0.))
        return beta

    beta = slope(np.ones(len(past)))
    if not symmetric:
        beta = (beta+slope((x*innovation)>0))/2
    return curve_estimate+row["volatility_usd"]*beta*innovation


def reaction_metrics(rows):
    # Outcomes are used only here, after all decisions. The sign of the actual
    # surprise is unavailable at entry and is never included in strategy ranking.
    rows = [r for r in rows if r["curve_estimate"] is not None]
    curve = np.array([r["curve_estimate"] for r in rows])
    marks = np.array([r["long_gross"] for r in rows])
    actual = np.array([r["inventory"]-r["early_public_ml"] for r in rows])
    revision = np.array([r["early_adaptive_scene"]-r["early_public_ml"] for r in rows])
    measured = np.array([bool(r["measured_roofs"]) for r in rows])
    controls = {}
    for name, signal in (("realized_inventory_surprise",actual),("satellite_revision",revision)):
        valid = measured & (signal!=0)
        value = signal[valid]
        residual = marks[valid]-curve[valid]
        sides = -np.sign(value)
        directional = marks[valid]*sides
        controls[name] = dict(windows=len(value),same_sign_as_realized_surprise=int(np.count_nonzero(value*actual[valid]>0)),
                              spread_sign_correct=int(np.count_nonzero(directional>0)),
                              correlation_with_curve_residual=float(np.corrcoef(value,residual)[0,1]) if len(value)>1 else None,
                              mean_signed_gross_usd=float(directional.mean()) if len(value) else None)
    return controls


def replay(marked):
    measurements = source_book()
    seen, errors, optical_history = [], [], {name:[] for name in ("base","expanded")}
    for r in marked:
        issue,cutoff,known = (r[k] for k in ("issue","decision","known"))
        base = r["early_public_ml"]
        public_past = [q for q in errors if q["known"]<cutoff]
        total,observed,added = visible(r,measurements[issue])
        levels = []
        for key,roofs in (("base",observed),("expanded",observed+added)):
            history = [q for q in optical_history[key] if q["known"]<cutoff]
            current = dict(cutoff=cutoff,known=known,week_ending=r["week_ending"], public=base,observed=roofs,
                           area_fraction=math.fsum(q["area"] for q in roofs)/total)
            forecast = predict(current,history,public_past)
            levels.append(forecast)
            if roofs:
                errors_now = np.abs([forecast[m]-r["inventory"] for m in FIXED_MODELS])
                optical_history[key].append(current | dict(prepared=preparation(current,r["inventory"]),absolute_errors=errors_now))
        direct,adaptive = (levels[0][k] for k in ("scene_direct","adaptive_prior_and_timing"))
        expanded = levels[1]["adaptive_prior_and_timing"]
        errors.append(dict(known=known,error=r["inventory"]-base))
        curve, simple, simple_optical, _, public_ml, original_optical = r["design"]
        vol = curve[-1]
        designs = (curve,simple,simple_optical,public_ml,original_optical,
                   *((*public_ml,value-base,(value-base)*vol) for value in (direct,adaptive,expanded)))
        training = [q for q in seen if q["exit_at"]<cutoff][-HISTORY:]
        current = r | dict(designs=designs, early_public_scene=direct, early_adaptive_scene=adaptive,
                           early_recovered_scene=expanded,recovered_roofs=len(added),
                           latest_optical_input_utc=max((datetime.fromisoformat(q["available_utc"]) for q in observed+added),
                                                        default=None),
                           latest_inventory_training_utc=public_past[-1]["known"].isoformat() if public_past else None,
                           curve_estimate=None)
        seen.append(current)
        if len(training)<MIN_WINDOWS:
            yield current,[]
            continue
        estimates = [risk_prediction([q | dict(x=q["designs"][i]) for q in training],current | dict(x=x),len(x))
                     for i,x in enumerate(designs)]
        # A model trained with optical features can change its curve coefficients
        # even when the current image is missing. Keep the public-only execution
        # exactly on those dates in the explicitly gated versions.
        estimates.extend((estimates[5] if observed else estimates[3],
                          estimates[6] if observed else estimates[3],
                          estimates[7] if observed or added else estimates[3]))
        current["curve_estimate"] = estimates[0]
        estimates.extend(event_response(training,current,estimates[0],adaptive-base,symmetric=flag) for flag in (True,False))
        sides = [int(v>HURDLE)-int(v<-HURDLE) for v in estimates]
        # A common inventory threshold is expressed in million barrels.
        # A build points toward short nearby CL relative to the deferred leg.
        threshold = .50
        sides.extend(int(v < -threshold)-int(v > threshold) for v in (base,adaptive,expanded))
        trades = []
        for method,value,side in zip(METHODS,(*estimates,None,None,None),sides,strict=True):
            gross = side*r["long_gross"]
            trades.append({k:r[k] for k in ("study_phase","near_month","far_month","source","measured_roofs")} | dict(
                method=method, release_date=issue, cutoff_utc=cutoff.isoformat(), entry_at_utc=r["entry"].isoformat(),
                scheduled_report_at_utc=r["reported_at"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),
                latest_training_exit_utc=training[-1]["exit_at"].isoformat(),recovered_roofs=len(added),
                expected_long_move_usd=value,near_side=side,gross_usd=gross,
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS}))
        yield current,trades


def inventory_metrics(rows):
    actual = np.array([r["inventory"] for r in rows])
    error = {m:1000*np.abs(np.array([r[m] for r in rows])-actual) for m in FORECASTS}
    return dict(releases=len(rows), methods={m:dict(mae_kbbl=float(e.mean()),
        paired_vs_early_public_95_kbbl=block_bootstrap_mean(e-error[FORECASTS[0]])) for m,e in error.items()})


def main():
    with threadpool_limits(limits=1):
        marked,coverage = cases()
        evaluated = list(replay(marked))
    forecasts = [f for f,_ in evaluated]
    trades = [r for _,sample in evaluated for r in sample]
    anchors = {(r["release_date"],r["method"]):r for r in read("cushing_cl_prerelease_trades")}
    for r in trades:
        legacy = {"curve":"curve_only","simple":"public_nowcast","simple_optical":"public_and_optical",
                  "public_ml":"early_public_ml","original_optical":"early_public_ml_and_optical"}.get(r["method"])
        if legacy and any(str(r[k])!=anchors[r["release_date"],legacy][k] for k in ("near_side","gross_usd","net_170_usd")):
            raise ValueError(f"The earlier baseline trade differs on {r['release_date']}/{legacy}")
    report = dict(status="historical_prepublication_replay_on_reviewed_period", exclusions=coverage, forecasts={},trading={},
                  realized_reaction_diagnostic={})
    changes = []
    for phase in PHASES:
        sample = [r for r in forecasts if r["study_phase"]==phase]
        measured = [r for r in sample if r["measured_roofs"] or r["recovered_roofs"]]
        report["forecasts"][phase] = dict(all=inventory_metrics(sample),measured=inventory_metrics(measured),
            added_roofs=sum(r["recovered_roofs"] for r in sample), measured_releases=len(measured))
        report["realized_reaction_diagnostic"][phase] = reaction_metrics(sample)
        group = {m:[r for r in trades if r["study_phase"]==phase and r["method"]==m] for m in METHODS}
        metrics = report["trading"][phase] = {}
        for method,rows in group.items():
            control_method = "public_direction" if method.endswith("direction") else "curve" if method.endswith("response") else "public_ml"
            control = group[control_method]
            if [r["release_date"] for r in rows]!=[r["release_date"] for r in control]:
                raise ValueError(f"Different early trading dates in {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(rows,control,strict=True)]
            versus_curve = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(rows,group["curve"],strict=True)]
            changed = [(r,q,e) for r,q,e in zip(rows,control,delta,strict=True) if r["near_side"]!=q["near_side"]]
            changes.extend(dict(release_date=r["release_date"],study_phase=phase,method=method,control=q["method"],
                measured_roofs=r["measured_roofs"],recovered_roofs=r["recovered_roofs"],near_side=r["near_side"],
                control_side=q["near_side"],net_170_usd=r["net_170_usd"],control_net_170_usd=q["net_170_usd"],increment_usd=e)
                for r,q,e in changed)
            metrics[method] = robustness(rows) | dict(increment_usd=sum(delta), paired_increment_95_usd=block_bootstrap_mean(delta),
                incremental_vs_curve_usd=sum(versus_curve),paired_vs_curve_95_usd=block_bootstrap_mean(versus_curve),
                changed_decisions=len(changed), changed_without_current_optics=sum(not (r["measured_roofs"] or r["recovered_roofs"])
                                                                                  for r,_,_ in changed),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in rows))
        print(phase, {m:round(v["mae_kbbl"],2) for m,v in report["forecasts"][phase]["measured"]["methods"].items()})
        print(phase, {m:(v["trades"],v["net_170_usd"]) for m,v in metrics.items()})
    fields = ("issue","week_ending","study_phase","inventory","measured_roofs","recovered_roofs","latest_inventory_training_utc",*FORECASTS)
    export(f"{STEM}_forecasts",[{k:r[k] for k in fields} | dict(cutoff_utc=r["decision"].isoformat(),known_utc=r["known"].isoformat(),
             latest_optical_input_utc=r["latest_optical_input_utc"].isoformat() if r["latest_optical_input_utc"] else None)
             for r in forecasts])
    export(f"{STEM}_trades",trades)
    if changes:
        export(f"{STEM}_changed_decisions",changes)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
