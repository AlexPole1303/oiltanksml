"""Nowcast a survey week from captures that span it and finish before publication."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_intraweek import estimate, learned_weights, preparation, prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, NOISE_FLOOR_M, PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL, parts
from publication_clocks import EASTERN, utc

STEM = "cushing_outer_capture_nowcast"
BASE = "all_recovered"
ARMS = ("outer_zero_drift","outer_persistence")
FOLLOWUP = "outer_shifted_followup"
SUPPLEMENT = "outer_supplement"
SHRUNK = "outer_supplement_shrunk"
ROBUST = "outer_robust_roofs"
SCALED = "outer_learned_gain"
SCALED_LOCAL = "outer_learned_gain_by_coverage"
INTERNAL = "outer_and_internal"
INTERNAL_CALIBRATION = "outer_with_internal_calibration"
ADDITIONS = (*ARMS, FOLLOWUP, SUPPLEMENT, SHRUNK, ROBUST, SCALED, SCALED_LOCAL, INTERNAL, INTERNAL_CALIBRATION)
MODELS = (PUBLIC,"carried_scene",BASE,*ADDITIONS)
CONDITIONAL_INPUTS = (PUBLIC,BASE,*ADDITIONS)
CONDITIONAL_BASE = f"{PUBLIC}_conditional"


def outer_pairs(stem):
    collected = defaultdict(list)
    for q in read(stem):
        if q["registered_both_shadow_diagnostics_pass"]=="True":
            if q["registration_status"]!="checked_offset":
                raise ValueError(f"Missing independent registration on {q['release_date']}")
            collected[q["release_date"]].append(q)
    return collected


def observation(q, r, hub_area, prior, *, internal=False):
    first,start,end,last,ready = (datetime.fromisoformat(q[k]) for k in (
        "earlier_capture_at_utc","previous_measurement_at_utc","current_measurement_at_utc",
        "later_capture_at_utc","latest_geometry_available_utc"))
    cutoff = r["cutoff"]
    start_local, end_local = (x.astimezone(EASTERN) for x in (start,end))
    previous = datetime.fromisoformat(utc(start_local.date()-timedelta(days=7),start_local.time()))
    following = datetime.fromisoformat(utc(end_local.date()+timedelta(days=7),end_local.time()))
    valid = (start<first<last<=end and last-first>=timedelta(hours=48)) if internal else (
        previous<=first<=start<end<last and last-end<=timedelta(hours=72))
    if (not valid or not last<=ready<cutoff or cutoff!=datetime.fromisoformat(q["inputs_available_before_utc"])
            or not q["terminal_id"]):
        raise ValueError(f"Invalid outer capture clock on {r['release_date']}/{q['candidate_id']}")
    area = float(q["circular_area_m2"])
    if not 0 < area <= hub_area or q["earlier_version"] == q["later_version"]:
        raise ValueError(f"Invalid outer capture geometry on {r['release_date']}/{q['candidate_id']}")
    per_m = area/BARREL_M3/1e6
    volume = float(q["registered_change_m"])*per_m
    early_gap, late_gap = (float(q[f"{s}_registered_gap_m"]) for s in ("earlier","later"))
    if not math.isclose(volume,(early_gap-late_gap)*per_m,abs_tol=1e-10):
        raise ValueError(f"Height and volume differ on {r['release_date']}")
    pre, post = (start-first)/(start-previous), (last-end)/(following-end)
    fraction = area/hub_area
    width = float(q["registered_loss_range_max_m"])-float(q["registered_loss_range_min_m"])
    interval = (last-first)/(end-start) if internal else 1.
    duration = interval if internal else 1+pre+post
    row = dict(candidate_id=q["candidate_id"],terminal_id=q["terminal_id"],pair_signal=volume,
        prior_component=0. if internal else fraction*pre*prior,sensitivity=fraction*interval,
        volume_per_m=per_m,area=area,variance=per_m**2*(NOISE_FLOOR_M**2+(width/2)**2+LOCAL_ACTIVITY_M**2*duration),
        duration=duration,scene_before=q["earlier_version"],scene_after=q["later_version"],
        available_utc=ready.isoformat(),pre_fraction=max(pre,0.),post_fraction=max(post,0.),
        after_survey_hours=max(0.,(last-end).total_seconds()/3600),
        mapped_hub_fraction=fraction,survey_end_utc=end.isoformat(),late_capture_utc=last.isoformat())
    if internal:
        row.update(internal_first=(first-start)/(end-start), internal_last=(last-start)/(end-start),
                   early_capture_utc=first.isoformat(), survey_start_utc=start.isoformat())
    return row


def prepared(row, target, outer, followup=(), *, next_target_known=False):
    if row["observed"] and "internal_first" in row["observed"][0]:
        sample = row["observed"]
        mapped = np.array([q["mapped_hub_fraction"] for q in sample])
        endpoints = tuple((np.array([q[f"internal_{edge}"] for q in sample]), sign*mapped)
                          for edge,sign in (("first",-1),("last",1)))
        sensitivity,residual,local = parts(sample,target)
        scenes = {q[k] for q in sample for k in ("scene_before","scene_after")}
        return dict(observations=sample,sensitivity=sensitivity,residual=residual,local=local,
                    clocks={date.fromisoformat(row["week_ending"]):endpoints},scenes=scenes)
    if not followup and not outer:
        return preparation(row, target)
    observations = row["observed"]
    normal = [q for q in observations if "post_fraction" not in q]
    spanning = [q for q in observations if "post_fraction" in q]
    sample = normal + spanning + list(followup)
    week = date.fromisoformat(row["week_ending"])
    before, after = week-timedelta(days=7), week+timedelta(days=7)
    groups = []
    if normal:
        fraction = math.fsum(q["area"] for q in normal) / row["hub_area"]
        groups.append((len(normal), preparation(row | dict(observed=normal, area_fraction=fraction),target)["clocks"]))
    for group, start, finish in ((spanning, before, after), (followup, before-timedelta(days=7), week)):
        if group:
            mapped = np.array([q["mapped_hub_fraction"] for q in group])
            pre,post = (np.array([q[f"{s}_fraction"] for q in group]) for s in ("pre","post"))
            groups.append((len(group), {start:(1-pre,-mapped), finish:(post,mapped)}))
    clocks = {}
    for key in {key for _, group in groups for key in group}:
        pieces = [group.get(key, (np.zeros(count),) * 2) for count, group in groups]
        clocks[key] = tuple(np.concatenate(column) for column in zip(*pieces, strict=True))
    fraction,residual,local = parts(sample,target)
    if outer and not next_target_known:
        times, area = clocks[after]
        next_exposure = area*times
        local += PRIOR_INVENTORY_SIGMA_MBBL**2*np.outer(next_exposure,next_exposure)
    scenes = {q[k] for q in sample for k in ("scene_before","scene_after")}
    return dict(observations=sample,sensitivity=fraction,residual=residual,local=local,clocks=clocks,scenes=scenes)


def run_forecast(history, errors, row, outer, *, robust=False, update=True):
    cutoff, known, base = (row[k] for k in ("cutoff","known","public"))
    previous = [q for q in history if q["known"]<cutoff]
    if not row["observed"]:
        return base, previous[-1]["known"].isoformat() if previous else None
    recent = [q["prepared"] for q in previous if cutoff-q["cutoff"]<=MAX_SCENE_AGE]
    model = prepared(row,base,outer)
    fits = estimate(model,recent,robust=robust)
    scale = prior_scale(errors)
    increments = np.clip([score/(1/scale**2+info) for _,info,score in fits],-1.25,1.25)
    prediction = base+float(learned_weights(previous)@increments)
    if update:
        history.append(dict(cutoff=cutoff,known=known,prepared=prepared(row,row["actual"],outer),
            absolute_errors=np.abs([base+revision-row["actual"] for revision,_,_ in fits])))
    return prediction, previous[-1]["known"].isoformat() if previous else None


def calibrate_gain(history, current):
    """Fit two regularized slopes toward 1 from already scored optical forecasts."""
    if not current["measured"]:
        return np.ones(2), 0, None
    eligible = [r for r in history if r["known"] < current["cutoff"]][-104:]
    latest = eligible[-1]["known"].isoformat() if eligible else None
    if len(eligible) < 32:
        return np.ones(2), len(eligible), latest
    x, y = (np.array([r[k] for r in eligible]) for k in ("optical_update", "residual"))
    age = np.array([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in eligible])
    weight = np.exp2(-age/78.)
    area = np.array([r["area_time"] for r in eligible])
    local = np.exp(-.5*np.square(np.log1p(area/.04)-math.log1p(current["area_time"]/.04))/.55**2)
    similar = np.array([r["spanning"] == current["spanning"] for r in eligible])
    weights = np.stack((weight, weight*(.35+.65*local*similar)))
    # Prior precision has the weight of an update of 0.25 million barrels
    # on each of eight previous weeks. Zero optical update yields zero change.
    prior = 8*.25**2
    slopes = np.ones(2)
    for _ in range(6):
        scale = np.hypot(1., (y-slopes[:,None]*x)/.6)
        w = weights/scale
        slopes = np.clip(((w@(x*y))+prior)/((w@np.square(x))+prior),0.,2.)
    return slopes, len(eligible), latest


def forecast_with_followup(history, errors, row, late, releases):
    cutoff, known, base = (row[k] for k in ("cutoff", "known", "public"))
    week = date.fromisoformat(row["week_ending"])
    before = week-timedelta(days=7)
    occupied = {q["candidate_id"] for q in row["observed"]}
    followup = []
    if before in late:
        source, previous = late[before]
        if not source["known"] < cutoff or source["week_ending"] != before.isoformat():
            raise ValueError(f"The earlier full-week target is unavailable on {row['release_date']}")
        for q in previous:
            start, end = row["survey_interval"]
            inherited_end, captured, available = (datetime.fromisoformat(q[k])
                for k in ("survey_end_utc", "late_capture_utc", "available_utc"))
            if not (inherited_end == start < captured <= end and captured <= available < cutoff):
                raise ValueError(f"The next-week observation clock is invalid on {row['release_date']}/{q['candidate_id']}")
            if q["candidate_id"] not in occupied:
                followup.append(q | dict(
                    prior_component=q["prior_component"]+q["mapped_hub_fraction"]*source["actual"],
                    sensitivity=q["mapped_hub_fraction"]*q["post_fraction"], source_release_date=source["release_date"],
                    source_report_known_utc=source["known"].isoformat()))
    outer = week in late
    training = [q for q in history if q["known"] < cutoff]
    lineage = [training[-1]["known"]] if training else []
    prediction = base
    if row["observed"] or followup:
        recent = []
        for old in training:
            previous_week = date.fromisoformat(old["row"]["week_ending"])
            if not old["row"]["observed"] or cutoff-old["row"]["cutoff"] > MAX_SCENE_AGE:
                continue
            if old["outer"]:
                successor = releases.get(previous_week+timedelta(days=7))
                # The late image contributes to the target this week. Conditioning
                # on another copy of it would count its signal twice.
                if successor is None or successor["known"] >= cutoff:
                    continue
                adjusted = [q | dict(prior_component=q["prior_component"]
                    + q["mapped_hub_fraction"]*q["post_fraction"]*successor["actual"])
                    for q in old["row"]["observed"]]
                past = old["row"] | dict(observed=adjusted)
                lineage.append(successor["known"])
                recent.append(prepared(past,past["actual"],True,next_target_known=True))
            else:
                recent.append(prepared(old["row"],old["row"]["actual"],False))
        if followup:
            lineage.append(late[before][0]["known"])
        fits = estimate(prepared(row,base,outer,followup),recent)
        scale = prior_scale(errors)
        increments = np.clip([score/(1/scale**2+info) for _,info,score in fits],-1.25,1.25)
        prediction = base+float(learned_weights(training)@increments)
        history.append(dict(known=known,row=row,outer=outer,
            absolute_errors=np.abs([base+update-row["actual"] for update,_,_ in fits])))
    details = [dict(release_date=row["release_date"],week_ending=row["week_ending"],study_phase=row["study_phase"],
                    cutoff_utc=cutoff.isoformat(),**q) for q in followup]
    return prediction, max(lineage).isoformat() if lineage else None, details


def replay(*, state=None, before=None):
    geometry = {(q["release_date"],q["week_ending"]):q for stem in ("sentinel2_space_time_coverage_weeks","holdout_optical_weeks")
                for q in read(stem)}
    baseline = indexed("cushing_recovered_optical_forecasts")
    public = indexed("cushing_robust_stack_forecasts")
    pairs = outer_pairs("cushing_outer_capture_pairs_fits")
    supplements = outer_pairs("cushing_outer_capture_pairs_supplement_fits")
    internal = outer_pairs("cushing_outer_capture_pairs_internal_fits")
    history = {m:[] for m in (BASE,*ARMS,SUPPLEMENT,ROBUST,INTERNAL)}
    followup_history, late, releases = [], {}, {}
    measured_history = []
    errors = []
    for r,arms,_ in combined_sources():
        issue, week, cutoff, known = (r[k] for k in ("release_date","week_ending","cutoff","known"))
        if before is not None and known >= before:
            break
        ref = baseline[issue]
        reference = geometry[issue,week]
        area = float(reference["dated_reference_area_m2"])
        if not cutoff<known or cutoff!=datetime.fromisoformat(ref["cutoff_utc"]):
            raise ValueError(f"Conflicting inventory clocks on {issue}")
        past = [p for p in errors if p["known"]<cutoff]
        existing = arms[-1]
        outer = pairs[issue]
        if outer and existing:
            raise ValueError(f"An outer pair entered a nonempty release on {issue}")
        previous = float(public[issue]["prior"])
        base = float(ref[BASE])
        values, labels = {}, {}
        details = []
        outer_observations = [observation(q,r,area,previous) for q in outer]
        extra = [observation(q,r,area,previous) for q in supplements[issue]]
        inside = [observation(q,r,area,previous,internal=True) for q in internal[issue]]
        if inside and (existing or outer):
            raise ValueError(f"Internal pairs may fill only an otherwise missing week on {issue}")
        for m in (BASE,*ARMS,SUPPLEMENT,ROBUST,INTERNAL):
            # A fixed, conservative persistence sensitivity uses one quarter
            # of the already available forecast for the current survey week.
            observations = outer_observations if outer and m!=BASE else existing
            if outer and m in (ARMS[1],ROBUST,INTERNAL):
                drift = .25*r["public"]
                observations = [q | dict(prior_component=q["prior_component"]
                    + q["mapped_hub_fraction"]*q["post_fraction"]*drift) for q in observations]
            if m == SUPPLEMENT:
                observations = observations + extra
            if m == INTERNAL and inside:
                observations = inside
            if len({q["candidate_id"] for q in observations})!=len(observations) or any(
                datetime.fromisoformat(q["available_utc"])>=cutoff for q in observations
            ):
                raise ValueError(f"Duplicate or late optical observation on {issue}")
            row = r | dict(observed=observations,hub_area=area,area_fraction=math.fsum(q["area"] for q in observations)/area)
            values[m],labels[m] = run_forecast(history[m],past,row,bool((outer and m!=BASE) or (extra and m==SUPPLEMENT)),
                                                 robust=m==ROBUST)
            if m==ARMS[0] and outer:
                details = [dict(release_date=issue,week_ending=week,study_phase=r["study_phase"],cutoff_utc=cutoff.isoformat(),**q)
                           for q in observations]
        if not math.isclose(values[BASE],base,abs_tol=1e-9):
            raise ValueError(f"Original forecast changed on {issue}")
        # Evaluate internal pairs as training data for scene offsets and
        # timing weights. Keep the live forecast unchanged on weeks where
        # the only optical input covers part of the reporting interval.
        values[INTERNAL_CALIBRATION] = values[ARMS[1]] if inside else values[INTERNAL]
        labels[INTERNAL_CALIBRATION] = labels[ARMS[1]] if inside else labels[INTERNAL]
        as_week = date.fromisoformat(week)
        releases[as_week] = r
        observations_for_followup = outer_observations or existing
        if outer:
            late[as_week] = (r, observations_for_followup)
        followup_row = r | dict(observed=observations_for_followup,
            hub_area=area,
            area_fraction=math.fsum(q["area"] for q in observations_for_followup)/area,
            survey_interval=tuple(datetime.fromisoformat(reference[k]) for k in
                ("previous_measurement_at_utc", "current_measurement_at_utc")))
        value, label, shifted = forecast_with_followup(followup_history,past,followup_row,late,releases)
        values[FOLLOWUP], labels[FOLLOWUP] = value, label
        # Shrink the marginal effect of unlike survey intervals using a fixed
        # weight, including their effect on future calibration.
        values[SHRUNK] = .75*values[ARMS[0]] + .25*values[SUPPLEMENT]
        labels[SHRUNK] = max((labels[m] for m in (ARMS[0],SUPPLEMENT) if labels[m]), default=None)
        public_expected = r["public"]
        update = values[ARMS[1]] - public_expected
        current = dict(cutoff=cutoff,known=known,measured=bool(existing or outer),spanning=bool(outer),
            optical_update=update,residual=r["actual"]-public_expected,
            area_time=math.fsum(q["sensitivity"] for q in (outer_observations or existing)))
        gains, training_weeks, training_label = calibrate_gain(measured_history,current)
        if current["measured"]:
            measured_history.append(current)
        for name, gain in zip((SCALED,SCALED_LOCAL), gains, strict=True):
            values[name] = public_expected + float(gain)*update
            labels[name] = max((stamp for stamp in (training_label,labels[ARMS[1]]) if stamp), default=None)
        errors.append(dict(known=known,error=r["actual"]-r["public"]))
        yield r | values | dict(public_ensemble=r["public"],carried_scene=float(ref["carried_scene"]),
            baseline_roofs=len(existing),outer_roofs=len(outer),followup_roofs=len(shifted),supplement_roofs=len(extra),
            internal_roofs=len(inside),
            gain_training_weeks=training_weeks,optical_gain=float(gains[0]),optical_gain_by_coverage=float(gains[1]),
            **{f"{m}_latest_label_utc":stamp for m,stamp in labels.items()}),details,shifted,inside
    if state is not None:
        state.update(history=history,errors=errors)


def conditional_trading(rows, base_trades):
    forecasts = {r["release_date"]: r for r in rows}
    base = {(r["release_date"], r["method"]): r for r in base_trades}
    public_anchor = {r["release_date"]: r for r in read("cushing_cl_conditional_response_trades")
                     if r["method"] == "inventory_conditional"}
    histories = defaultdict(list)
    _, _, cases = market_calendar()
    for r in cases:
        issue = r["release_date"]
        if issue not in public_anchor:
            continue
        forecast = forecasts[issue]
        actual, _, carry, *_ = r["x"]
        for method in CONDITIONAL_INPUTS:
            original = base[issue, method]
            if (any(datetime.fromisoformat(original[stamp]) != r[field] for stamp, field in (
                    ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))
                    or any(original[month] != r[month] for month in ("near_month", "far_month"))
                    or datetime.fromisoformat(original["settled_training_label_utc"]) >= r["decision"]
                    or not forecast["cutoff"] < forecast["known"] <= r["decision"] or forecast["actual"] != actual):
                raise ValueError(f"Different market or inventory lineage on {issue}/{method}")
            point = float(original["expected_long_move_usd"])
            vector = features(carry, r["volatility_usd"], actual - forecast[method])
            current = r | dict(base_estimate=point, response_features=vector)
            past = [q for q in histories[method] if q["exit_at"] < r["decision"]][-HISTORY:]
            correction, support = conditional_residual(past, current) if len(past) >= MIN_WINDOWS else (0., 0.)
            histories[method].append(current)
            expected = point + correction
            side = int(expected > HURDLE) - int(expected < -HURDLE)
            gross = side * r["long_gross"]
            if method == PUBLIC:
                ref = public_anchor[issue]
                if (not math.isclose(expected, float(ref["expected_long_move_usd"]), abs_tol=1e-8)
                        or side != int(ref["near_side"]) or gross != int(ref["gross_usd"])):
                    raise ValueError(f"Public conditional strategy changed on {issue}")
            yield original | dict(method=f"{method}_conditional", expected_long_move_usd=expected, near_side=side,
                gross_usd=gross, alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                latest_calibration_exit_utc=past[-1]["exit_at"].isoformat() if past else None,
                calibration_samples=len(past), similar_report_weight=support,
                **{f"session_{n}_net_170_usd":side*v-(170 if n==5 else 85)*abs(side)
                   for n,v in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},
                **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def forward(issue, *, scoring):
    from register_cushing_prerelease_forecast import digest, freeze, score

    original_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    recovered_path = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
    path = original_path.with_name(f"{original_path.stem}_internal_calibration.json")
    original = json.loads(original_path.read_bytes())
    recovered = json.loads(recovered_path.read_bytes())
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in ("information_cutoff_utc", "registration_deadline_utc"))
    if not path.exists() and not scoring:
        now = datetime.now(timezone.utc)
        if not cutoff <= now < deadline:
            raise ValueError("The satellite forecast missed the registration deadline")
        if any(original[k] != recovered[k] for k in (
            "release_date", "week_ending", "information_cutoff_utc", "registration_deadline_utc"
        )) or any(digest(DERIVED/name) != checksum for name,checksum in recovered["source_sha256"].items()):
            raise ValueError("The previously registered satellite panel does not match its source data")
        sources = [original_path, recovered_path, *(DERIVED/f"{stem}.csv" for stem in (
            "cushing_outer_capture_pairs_fits", "cushing_outer_capture_pairs_internal_fits",
            "cushing_recovered_optical_forecasts", "holdout_optical_weeks",
        )), ROOT/"scripts/fit_cushing_outer_capture_nowcast.py", ROOT/"scripts/fit_cushing_intraweek.py"]
        observed = [q | dict(area=q["volume_per_m"]*BARREL_M3*1e6,
            available_utc=q["source_available_utc"]) for q in original["measured_pairs"]]
        observed.extend(recovered["recovered_roofs"])
        native = json.loads((DERIVED/f"cushing_forward_native_optical_{issue}_summary.json").read_bytes())
        area = float(native["dated_roof_area_m2"])
        if (not observed or len(observed) != recovered["expanded_roofs"]
                or len({q["candidate_id"] for q in observed}) != len(observed)
                or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observed)):
            raise ValueError("The new forecast must reuse the frozen, timely measurements")
        state = {}
        with threadpool_limits(limits=1):
            list(replay(state=state,before=cutoff))
            previous = indexed("eia_wpsr_forward_releases")[original["prior_release"]]
            previous_known = datetime.fromisoformat(previous["conservative_postrelease_from_utc"])
            if previous_known >= cutoff or previous["source_sha256"] != original["prior_release_sha256"]:
                raise ValueError("The previous EIA outcome is unavailable at the forecast cutoff")
            ledger = json.loads((DERIVED/"cushing_forward_public_cl_summary.json").read_bytes())["release_rows"]
            expectation, = (r for r in ledger if r["release_date"]==previous["release_date"]
                             and r["method"]=="inventory_public_expectation_second_session")
            if datetime.fromisoformat(expectation["forecast_cutoff_utc"]) >= previous_known:
                raise ValueError("The previous-release public expectation has an inconsistent cutoff")
            errors = state["errors"] + [dict(known=previous_known,
                error=float(previous["cushing_change_kbbl"])/1000-expectation["forecast_million_bbl"])]
            row = dict(cutoff=cutoff, known=deadline, public=original["public_ensemble_kbbl"]/1000,
                week_ending=original["week_ending"], observed=observed,hub_area=area,
                area_fraction=math.fsum(q["area"] for q in observed)/area)
            estimates = {PUBLIC:original["public_ensemble_kbbl"]}
            for name, method in ((ARMS[1],ARMS[1]),(INTERNAL_CALIBRATION,INTERNAL),(ROBUST,ROBUST)):
                value,_ = run_forecast(state["history"][method],errors,row,False,robust=method==ROBUST,update=False)
                estimates[name] = value*1000
        stamp = datetime.now(timezone.utc)
        if not cutoff <= stamp < deadline:
            raise ValueError("The satellite forecast finished after deadline")
        record = dict(release_date=issue,week_ending=original["week_ending"],recorded_at_utc=stamp.isoformat(),
            information_cutoff_utc=cutoff.isoformat(),registration_deadline_utc=deadline.isoformat(),
            latest_training_label_utc=previous_known.isoformat(),measured_roofs=len(observed),estimates_kbbl=estimates,
            source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
            preceding_public_forecast_million_bbl=expectation["forecast_million_bbl"],
            preceding_eia_source_sha256=previous["source_sha256"])
        freeze(path,record)
        return record
    record = json.loads(path.read_bytes())
    stamp = datetime.fromisoformat(record["recorded_at_utc"])
    if (not datetime.fromisoformat(record["latest_training_label_utc"])<cutoff<=stamp<deadline or any(
            record[k] != original[k] for k in ("week_ending","release_date","information_cutoff_utc","registration_deadline_utc"))
            or any(digest(p) != record["source_sha256"][str(p.relative_to(ROOT))] for p in (original_path,recovered_path))):
        raise ValueError("The frozen forecast does not match its source data")
    if not scoring:
        return record
    result = score(original_path,original)
    comparison = dict(status=result["status"],release_date=issue,forecast_sha256=digest(path))
    if result["status"] == "scored_first_release":
        value = result["actual_eia_change_kbbl"]
        comparison.update(actual_eia_change_kbbl=value,first_release_sha256=result["first_release_sha256"],
            absolute_error_kbbl={m:abs(value-estimate) for m,estimate in record["estimates_kbbl"].items()})
    write(path.with_name(f"{path.stem}_score.json"),(json.dumps(comparison,indent=2,allow_nan=False)+"\n").encode())
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register-release",type=date.fromisoformat)
    mode.add_argument("--score-release",type=date.fromisoformat)
    args = parser.parse_args()
    issue = args.register_release or args.score_release
    if issue:
        print(json.dumps(forward(issue.isoformat(),scoring=args.score_release is not None),indent=2,allow_nan=False))
        return
    with threadpool_limits(limits=1):
        series = list(replay())
        rows = [r for r,*_ in series]
        trades = list(trading(rows,MODELS))
        trades.extend(conditional_trading(rows,trades))
    report = dict(status="exploratory_on_reviewed_history", forecasts={}, market={}, market_conditional={})
    for phase in PHASES:
        group = [r for r in rows if r["study_phase"]==phase]
        largest = sorted(group,key=lambda r:abs(r[PUBLIC]-r["actual"]),reverse=True)[:round(len(group)*.10)]
        public_absolute_error = math.fsum(abs(r[PUBLIC]-r["actual"]) for r in group)
        report["forecasts"][phase] = {}
        for name,sample in (("all",group),("largest_public_errors",largest),
                            ("no_measurement",[r for r in group if r["baseline_roofs"]+r["outer_roofs"]==0]),
                            ("new_internal_weeks",[r for r in group if r["internal_roofs"]]),
                            ("newly_measured",[r for r in group if r["outer_roofs"]]),
                            ("following_week_measured",[r for r in group if r["followup_roofs"]]),
                            ("supplemented_weeks",[r for r in group if r["supplement_roofs"]])):
            result = forecast_metrics(sample,MODELS)
            result["new_roof_week_pairs"] = sum(r["outer_roofs"] for r in sample)
            result["baseline_measured_weeks"] = sum(r["baseline_roofs"]>0 for r in sample)
            result["expanded_measured_weeks"] = sum(r["baseline_roofs"]+r["outer_roofs"]>0 for r in sample)
            result["followup_pairs"] = sum(r["followup_roofs"] for r in sample)
            result["followup_measured_weeks"] = sum(r["baseline_roofs"]+r["outer_roofs"]+r["followup_roofs"]>0 for r in sample)
            result["supplement_pairs"] = sum(r["supplement_roofs"] for r in sample)
            result["internal_pairs"] = sum(r["internal_roofs"] for r in sample)
            result["internal_expanded_weeks"] = sum(r["baseline_roofs"]+r["outer_roofs"]+r["internal_roofs"]>0 for r in sample)
            result["share_of_public_absolute_error"] = math.fsum(abs(r[PUBLIC]-r["actual"]) for r in sample)/public_absolute_error
            for method in ADDITIONS:
                delta = [1000*(abs(q[method]-q["actual"])-abs(q[BASE]-q["actual"])) for q in sample]
                if delta:
                    result["models"][method]["paired_delta_vs_base_95_kbbl"] = block_bootstrap_mean(delta)
                    difference = [1000*(abs(q[method]-q["actual"])-abs(q[ARMS[1]]-q["actual"])) for q in sample]
                    result["models"][method]["paired_delta_vs_outer_persistence_95_kbbl"] = block_bootstrap_mean(difference)
            report["forecasts"][phase][name] = result
        arms = defaultdict(list)
        for r in trades:
            if r["study_phase"] == phase:
                arms[r["method"]].append(r)
        for section, baseline, methods in (
            ("market", BASE, ("curve", *MODELS)),
            ("market_conditional", CONDITIONAL_BASE, tuple(f"{m}_conditional" for m in CONDITIONAL_INPUTS)),
        ):
            origin = arms[baseline]
            result = report[section][phase] = {}
            for m in methods:
                sample = arms[m]
                if [q["release_date"] for q in sample] != [q["release_date"] for q in origin]:
                    raise ValueError(f"Unmatched market calendar on {phase}/{m}")
                pairs = list(zip(sample,origin,strict=True))
                diff = [r["net_170_usd"]-q["net_170_usd"] for r,q in pairs]
                result[m] = robustness(sample) | daily_risk(sample) | dict(
                    net_340_usd=sum(q["gross_usd"]-340*abs(q["near_side"]) for q in sample),
                    changed_decisions=sum(r["near_side"]!=q["near_side"] for r,q in pairs),
                    paired_delta_vs_base_95_usd=block_bootstrap_mean(diff))
            print(phase,section,{m:v["net_170_usd"] for m,v in result.items()})
    export(f"{STEM}_forecasts",[{k:r[k] for k in ("release_date","week_ending","study_phase","actual","baseline_roofs","outer_roofs",
        "followup_roofs","supplement_roofs","internal_roofs","gain_training_weeks","optical_gain","optical_gain_by_coverage",
        *MODELS,*(f"{m}_latest_label_utc" for m in (BASE,*ADDITIONS)))} |
        dict(cutoff_utc=r["cutoff"].isoformat(),known_utc=r["known"].isoformat()) for r in rows])
    export(f"{STEM}_measurements",[q for _,group,_,_ in series for q in group])
    export(f"{STEM}_followup_measurements",[q for _,_,group,_ in series for q in group])
    partial = [dict(release_date=r["release_date"],week_ending=r["week_ending"],cutoff_utc=r["cutoff"].isoformat(),**q)
               for r,_,_,group in series for q in group]
    if partial:
        export(f"{STEM}_internal_measurements",partial)
    calibration = ("latest_calibration_exit_utc", "calibration_samples", "similar_report_weight")
    export(f"{STEM}_trades",[{k:r.get(k) for k in (*list(trades[0]), *calibration)} for r in trades])
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
