"""Calibrate the B04 pair error model from independently dated tandem controls."""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from audit_cushing_weekly_change_controls import aerial
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_windows, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, NOISE_FLOOR_M, PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT, observation, outer_pairs, run_forecast
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from replay_cushing_dated_height_calibration import indexed_by_roof, pair_inputs

STEM = "cushing_optical_noise_floor"
ADJUSTED = "tandem_calibrated_b04_noise"
BASE = np.array((NOISE_FLOOR_M**2, 1.))
MIN_CONTROLS = 40
MIN_DAYS = 2


def tandem_controls():
    frames = {r["observation_id"]: r for r in read("shadow_change_control_estimates")}
    terminals = {r["candidate_id"]: r["terminal_id"] for r in read("sentinel2_space_time_coverage_roof_registry")}
    pairs = defaultdict(list)
    for r in read("shadow_tandem_pass_pairs"):
        tank = r["candidate_id"]
        if (r["selected_for_summary"] != "True" or r["both_resolved"] != "True" or r["roof_type"] != "EFR"
                or abs(float(r["other_minus_a_seconds"])) > 60 or tank not in terminals):
            continue
        gaps, spans, available = [], [], []
        versions = (r["a_version"], r["other_version"])
        for name, version in zip(("a", "other"), versions, strict=True):
            q = frames[r[f"{name}_observation_id"]]
            gap, lo, hi = (float(q[k]) for k in ("estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m"))
            observed = datetime.fromisoformat(r[f"{name}_capture_at_utc"])
            ready = max(datetime.fromisoformat(r[f"{name}_{k}_available_at_utc"]) for k in ("archive", "geometry"))
            if (q["version_id"] != version or q["candidate_id"] != tank or not 0 <= lo <= gap <= hi or ready < observed):
                raise ValueError(f"Invalid tandem observation for {tank}/{version}")
            gaps.append(gap)
            spans.append(hi-lo)
            available.append(ready)
        gap_delta = gaps[1]-gaps[0]
        if not math.isclose(gap_delta, float(r["other_minus_a_gap_m"])) or versions[0] == versions[1]:
            raise ValueError(f"Incorrect paired roof difference for {tank}")
        pairs[versions].append(dict(candidate_id=tank, terminal_id=terminals[tank], capture_day=r["capture_day"],
            earlier_version=versions[0], later_version=versions[1], difference_m=gap_delta,
            width_term=((spans[0]+spans[1])/2)**2, ready=max(available)))
    for rows in pairs.values():
        for q in rows:
            donors = [r["difference_m"] for r in rows if r["terminal_id"] != q["terminal_id"] and r["ready"] <= q["ready"]]
            if len(donors) < 8:
                continue
            offset = float(np.median(donors))
            yield q | dict(scene_offset_m=offset, residual_m=q["difference_m"]-offset, independent_donor_count=len(donors))


def learn(rows):
    counts = Counter(r["capture_day"] for r in rows)
    if len(rows) < MIN_CONTROLS or len(counts) < MIN_DAYS:
        return BASE
    width = np.array([r["width_term"] for r in rows])
    error = np.array([r["residual_m"] for r in rows])
    weight = np.array([1 / counts[r["capture_day"]] for r in rows])
    weight /= weight.sum()

    def objective(log_scale):
        floor_sq, slope = BASE * np.exp(log_scale)
        variance = floor_sq + slope * width
        # Student-t reference with four degrees of freedom limits the effect
        # of an isolated shadow failure on every future pair's weight.
        loss = .5 * np.log(variance) + 2.5 * np.log1p(error**2 / (2 * variance))
        return float(weight @ loss + .20 * (log_scale @ log_scale))

    fit = minimize(objective, np.zeros(2), method="L-BFGS-B", bounds=[(-1., 1.), (-1., 1.)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"Pair-noise calibration failed: {fit.message}")
    return BASE*np.exp(fit.x)


def validation(controls):
    fits = {}
    for q in controls:
        key = (q["capture_day"], q["terminal_id"])
        if key not in fits:
            day, site = key
            fits[key] = learn([r for r in controls if r["capture_day"] != day and r["terminal_id"] != site])
        yield q | validation_row(q, fits[key])


def validation_row(q, params):
    reference, calibrated = (float(floor + scale * q["width_term"]) for floor, scale in (BASE, params))
    return dict(reference_sd_m=math.sqrt(reference), calibrated_sd_m=math.sqrt(calibrated),
                reference_score=.5*(math.log(reference)+q["residual_m"]**2/reference),
                calibrated_score=.5*(math.log(calibrated)+q["residual_m"]**2/calibrated))


def adjust_observations(row, observations, red, pairs, calibration):
    models = {site: learn([r for r in calibration if r["terminal_id"] != site])
              for site in {q["terminal_id"] for q in observations}}
    revised = []
    changed = 0
    for q in observations:
        key = (row["release_date"], q["candidate_id"], q["scene_before"], q["scene_after"])
        per_m = q["volume_per_m"]
        matched = [x for x in pairs.get(key, ()) if math.isclose((x[0]-x[1])*per_m, q["pair_signal"], abs_tol=1e-10)]
        if q["candidate_id"] not in red or len(matched) != 1:
            revised.append(q)
            continue
        floor, slope = models[q["terminal_id"]]
        activity = LOCAL_ACTIVITY_M**2*q["duration"]
        apparent_width = q["variance"]/per_m**2 - BASE[0] - activity
        if apparent_width < -1e-10:
            raise ValueError(f"The original B04 variance does not reproduce for {key}")
        variance = per_m**2*(floor + slope*max(apparent_width, 0.) + activity)
        changed += not math.isclose(variance, q["variance"], rel_tol=1e-10)
        revised.append(q | dict(variance=variance))
    return revised, changed


def forecasts(controls, *, before=None, state=None):
    geometry = {(q["release_date"], q["week_ending"]): q for stem in (
        "sentinel2_space_time_coverage_weeks", "holdout_optical_weeks") for q in read(stem)}
    archive = indexed(f"{OUTER}_forecasts")
    prior_reports = indexed("cushing_robust_stack_forecasts")
    pairs = pair_inputs()
    first = outer_pairs("cushing_outer_capture_pairs_fits")
    supplement = outer_pairs("cushing_outer_capture_pairs_supplement_fits")
    published, histories = [], defaultdict(list)
    for row, arms, _ in combined_sources():
        issue, cutoff, known = (row[k] for k in ("release_date", "cutoff", "known"))
        if before is not None and known >= before:
            break
        ref = archive[issue]
        if (cutoff != datetime.fromisoformat(ref["cutoff_utc"]) or known != datetime.fromisoformat(ref["known_utc"])
                or row["actual"] != float(ref["actual"])):
            raise ValueError(f"Unmatched EIA time or value on {issue}")
        area = float(geometry[issue, row["week_ending"]]["dated_reference_area_m2"])
        outside = first[issue]+supplement[issue]
        observations = arms[-1] + [observation(q, row, area, float(prior_reports[issue]["prior"])) for q in outside]
        red = {q["candidate_id"] for q in arms[1]+outside}
        if (len({q["candidate_id"] for q in observations}) != len(observations)
                or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observations)):
            raise ValueError(f"Duplicated or unavailable optical input on {issue}")
        eligible = [q for q in controls if q["ready"] < cutoff]
        new, count = adjust_observations(row, observations, red, pairs, eligible)
        past = [q for q in published if q["known"] < cutoff]
        estimates = {}
        for m, sample in ((SUPPLEMENT, observations), (ADJUSTED, new)):
            q = row | dict(observed=sample, hub_area=area, area_fraction=math.fsum(r["area"] for r in sample)/area)
            estimates[m], _ = run_forecast(histories[m], past, q, bool(outside))
        if not math.isclose(estimates[SUPPLEMENT], float(ref[SUPPLEMENT]), abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"The source-dated inventory model differs on {issue}")
        published.append(dict(known=known, error=row["actual"]-row["public"]))
        yield {k: row[k] for k in ("release_date", "study_phase", "week_ending", "cutoff", "known", "actual")} | estimates | {
            PUBLIC: row["public"]} | dict(available_tandem_controls=len(eligible),
                available_tandem_days=len({q["capture_day"] for q in eligible}), revised_roofs=count,
                latest_control_at=max((q["ready"] for q in eligible), default=None),
                previous_inventory_training_label=past[-1]["known"] if past else None)
    if state is not None:
        state.update(histories=histories, published=published)


def physical_checks(controls):
    tests = list(validation(controls))
    by_day = defaultdict(list)
    for r in tests:
        by_day[r["capture_day"]].append(r)
    def summary(rows):
        return dict(roofs=len(rows), mean_absolute_residual_m=float(np.mean([abs(r["residual_m"]) for r in rows])),
                    reference_score=float(np.mean([r["reference_score"] for r in rows])),
                    calibrated_score=float(np.mean([r["calibrated_score"] for r in rows])))
    # The independent aerial interval also includes synchronization error and
    # any real movement between the aerial and satellite endpoint times.
    a = [r | validation_row(dict(width_term=((r["high"]-r["low"])/2)**2,
                                residual_m=r["physical"]-r["target"]),
             learn([q for q in controls if q["terminal_id"] != r["terminal_id"]])) for r in aerial()]
    aerial_summary = dict(roofs=len(a), pooled_reference_score=float(np.mean([r["reference_score"] for r in a])),
                         pooled_calibrated_score=float(np.mean([r["calibrated_score"] for r in a])))
    return tests, dict(pooled_tandem=summary(tests), by_tandem_date={day:summary(rows) for day,rows in sorted(by_day.items())},
                       aerial_interval=aerial_summary, pooled_historical_parameters=learn(controls).tolist())


def market(projected):
    book = {r["release_date"]: r for r in projected}
    histories = defaultdict(list)
    for r, _ in market_windows():
        f = book[r["release_date"]]
        decision = r["decision"]
        if not f["cutoff"] < f["known"] <= decision < r["entry"] < r["exit_at"] or f["actual"] != r["x"][0]:
            raise ValueError(f"Invalid inventory vintage on {r['release_date']}")
        for m in (SUPPLEMENT, ADJUSTED):
            history = histories[m]
            current = r | dict(x=replace_expectation(r, f[m]), response_features=features(
                r["x"][2], r["volatility_usd"], f["actual"] - f[m]))
            settled = [q for q in history if q["exit_at"] < decision][-HISTORY:]
            history.append(current)
            if len(settled) < MIN_WINDOWS:
                continue
            point = volatility_forecast(settled, current)
            calibration = [q for q in settled if "base_estimate" in q]
            residual, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            current["base_estimate"] = point
            t = trade(r, m, point + residual, 3, training_at=settled[-1]["exit_at"],
                      calibration_at=calibration[-1]["exit_at"] if calibration else None)
            yield t | dict(realized_long_move_usd=r["long_gross"], revised_roofs=f["revised_roofs"],
                           net_340_usd=t["gross_usd"] - 340*abs(t["near_side"]))


def register(issue):
    prefix = DERIVED/f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    recovered_path = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
    leader_path = DERIVED/f"cushing_forward_cl_satellite_{issue}_inventory.json"
    original, recovered, leader = (json.loads(p.read_bytes()) for p in (original_path, recovered_path, leader_path))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(r[k] != original[k] for r in (recovered, leader) for k in LINEAGE):
        raise ValueError("The registered input dates do not match")
    checked_hashes(leader["sources_sha256"])
    if not path.exists():
        if issue in indexed("eia_wpsr_forward_releases") or not cutoff <= datetime.now(timezone.utc) < deadline:
            raise ValueError("The optical error model must be registered before the EIA report")
        stem = f"cushing_forward_native_optical_{issue}"
        native = DERIVED/f"{stem}_summary.json"
        native_pairs = DERIVED/f"{stem}_pairs.csv"
        hub_area = float(json.loads(native.read_bytes())["dated_roof_area_m2"])
        observations = [r | dict(area=r["volume_per_m"]*BARREL_M3*1e6, available_utc=r["source_available_utc"])
                        for r in original["measured_pairs"]] + recovered["recovered_roofs"]
        red = {r["candidate_id"] for r in original["measured_pairs"]}
        geometry = indexed_by_roof("cushing_modern_lidar_optical_registry")
        if (len(observations) != leader["measured_roofs"] or len({r["candidate_id"] for r in observations}) != len(observations)
                or any(datetime.fromisoformat(r["available_utc"]) >= cutoff for r in observations)):
            raise ValueError("The frozen observation panel is incomplete or late")
        pairs = defaultdict(set)
        for r in read(f"{stem}_pairs"):
            if (r["candidate_id"] not in red or r["registered_both_shadow_diagnostics_pass"] != "True"
                    or r["registration_status"] != "checked_offset"):
                continue
            g = geometry[r["candidate_id"]]
            if (g["latest_engineering_date"] >= cutoff.date().isoformat()
                    or not math.isclose(float(g["circular_area_m2"]), float(r["area_m2"]), abs_tol=1e-8)):
                raise ValueError(f"Mismatched frozen geometry for {r['candidate_id']}")
            key = (issue, r["candidate_id"], r["earlier_version"], r["later_version"])
            pairs[key].add(tuple(float(r[f"{s}_registered_gap_m"]) for s in ("earlier", "later")))
        row = dict(release_date=issue, week_ending=original["week_ending"], cutoff=cutoff, known=deadline,
                   public=original["public_ensemble_kbbl"]/1000, hub_area=hub_area,
                   area_fraction=math.fsum(r["area"] for r in observations)/hub_area)
        qualified = [r for r in tandem_controls() if r["ready"] < cutoff]
        state = {}
        with threadpool_limits(limits=1):
            list(forecasts(qualified, before=cutoff, state=state))
            adjusted, revised = adjust_observations(row, observations, red, pairs, qualified)
            estimates = {m:1000*run_forecast(state["histories"][m], state["published"], row | dict(observed=sample),
                                             False, update=False)[0] for m,sample in ((SUPPLEMENT, observations), (ADJUSTED, adjusted))}
        stamp = datetime.now(timezone.utc)
        if (not math.isclose(estimates[SUPPLEMENT], leader["estimates_kbbl"][SUPPLEMENT], abs_tol=1e-8)
                or not cutoff <= stamp < deadline or revised == 0):
            raise ValueError("The online replay did not reproduce a timely adjusted comparison")
        sources = (original_path, recovered_path, leader_path, native, native_pairs, Path(__file__).resolve(),
                   ROOT/"scripts/replay_cushing_dated_height_calibration.py",
                   *(DERIVED/f"{name}.csv" for name in ("shadow_tandem_pass_pairs", "shadow_change_control_estimates",
                       "cushing_modern_lidar_optical_registry", "sentinel2_space_time_coverage_roof_registry", f"{STEM}_forecasts")))
        freeze(path, {k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), measured_roofs=len(observations),
            revised_roofs=revised, available_tandem_controls=len(qualified), latest_control_at=max(r["ready"] for r in qualified).isoformat(),
            estimates_kbbl=estimates, sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources}))
    frozen = json.loads(path.read_bytes())
    checked_hashes(frozen["sources_sha256"])
    if (any(frozen[k] != original[k] for k in LINEAGE) or not datetime.fromisoformat(frozen["latest_control_at"])
            < cutoff <= datetime.fromisoformat(frozen["recorded_at_utc"]) < deadline):
        raise ValueError("The optical noise registration failed its date checks")
    outcome = score_release(original_path, original)
    result = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path), estimates_kbbl=frozen["estimates_kbbl"])
    if result["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_errors_kbbl={m:abs(v-actual) for m,v in frozen["estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(result, indent=2))


def main():
    controls = list(tandem_controls())
    checked, physical = physical_checks(controls)
    with threadpool_limits(limits=1):
        projected = list(forecasts(controls))
        trades = list(market(projected))
    anchor = {r["release_date"]: r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    for r in trades:
        if r["method"] == SUPPLEMENT:
            a = anchor[r["release_date"]]
            if (any(int(a[k]) != r[k] for k in ("near_side", "net_170_usd", "net_340_usd"))
                    or not math.isclose(float(a["expected_long_move_usd"]), r["expected_long_move_usd"], abs_tol=1e-8)):
                raise ValueError(f"The trading baseline differs on {r['release_date']}")
    report = dict(status="exploratory_on_reviewed_history", physical_reference=physical, phases={})
    for phase in PHASES:
        sample = [r for r in projected if r["study_phase"] == phase]
        inventory = {}
        for scope, chosen in (("all",sample), ("revised",[r for r in sample if r["revised_roofs"]])):
            inventory[scope] = dict(releases=len(chosen), models={})
            for method in (PUBLIC,SUPPLEMENT,ADJUSTED):
                e = np.array([abs(r[method]-r["actual"])*1000 for r in chosen])
                paired = np.array([abs(r[SUPPLEMENT]-r["actual"])*1000 for r in chosen])
                if len(e):
                    inventory[scope]["models"][method] = dict(mae_kbbl=float(e.mean()), change_95_kbbl=block_bootstrap_mean(e-paired))
        metrics = {}
        incumbent = [r for r in trades if r["study_phase"] == phase and r["method"] == SUPPLEMENT]
        for m in (SUPPLEMENT,ADJUSTED):
            arm = [r for r in trades if r["study_phase"] == phase and r["method"] == m]
            if [r["release_date"] for r in arm] != [r["release_date"] for r in incumbent]:
                raise ValueError(f"Trading calendars differ in {phase}")
            metrics[m] = robustness(arm) | daily_risk(arm) | dict(net_340_usd=sum(r["net_340_usd"] for r in arm),
                changed_sides=sum(r["near_side"] != a["near_side"] for r,a in zip(arm,incumbent,strict=True)))
        report["phases"][phase] = dict(inventory=inventory, trading=metrics, revised_roofs=sum(r["revised_roofs"] for r in sample))
        print(phase, inventory["revised"], {m:(v["trades"],v["net_170_usd"],v["net_340_usd"]) for m,v in metrics.items()})
    export(f"{STEM}_controls", checked)
    export(f"{STEM}_forecasts", projected)
    export(f"{STEM}_trades", trades)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    if args.release:
        register(args.release.isoformat())
    else:
        main()
