"""Propagate a source-dated roof-height calibration into inventory and CL replay."""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_weekly_change_controls import aerial
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from calibrate_cushing_height_multisource import STEM as HEIGHT, learn, predict
from collect_public import ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_windows, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC, VOLUMES
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT, observation, outer_pairs, run_forecast
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_dated_height_calibration"
MIDPOINT = "dated_midpoint"
MULTIVARIATE = "dated_multivariate"
MODELS = (SUPPLEMENT, MIDPOINT, MULTIVARIATE)
HEIGHT_METHODS = {MIDPOINT: "midpoint_calibration", MULTIVARIATE: "multisource_ml"}


def training_rows():
    geometry = indexed_by_roof("cushing_modern_lidar_optical_registry")
    archive = {r["l1c_product_name"]: r["with_red_detector_mask_objects_created_by_utc"]
               for r in read("sentinel2_l1c_archive_availability") if r["with_red_detector_mask_objects_created_by_utc"]}
    heights = read("shadow_geometry_estimates")
    for q in read(f"{HEIGHT}_controls"):
        # This replay requires verified original label publication. The aerial
        # and orbital controls still enter the separate engineering assessment.
        if q["source"] != "airborne_lidar":
            continue
        row = heights[int(q["source_record"].rsplit(":", 1)[1]) - 2]
        g = geometry[q["candidate_id"]]
        if (row["candidate_id"] != q["candidate_id"] or row["optical_capture_at_utc"] != q["optical_captured_at_utc"]
                or float(row["lidar_gap_m"]) != float(q["reference_gap_m"])):
            raise ValueError("The dated height source does not match its label")
        # USGS provides a date for availability, so allow the label from
        # the next UTC day. Use the latest timestamp among its dependencies.
        limits = [datetime.fromisoformat(g[k]).replace(tzinfo=timezone.utc) + timedelta(days=1)
                  for k in ("lidar_source_available_date", "roof_evidence_date")]
        limits.append(datetime.fromisoformat(archive[row["version_id"]]))
        yield q | {k: float(q[k]) for k in ("raw_gap_m", "reference_gap_m", "diameter_m", "shell_m", "relative_gap", "profile_span_m")} | dict(
            optical_version=row["version_id"], lidar_source_available_date=g["lidar_source_available_date"], ready=max(limits))


def indexed_by_roof(stem):
    rows = read(stem)
    out = {r["candidate_id"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"Repeated roof in {stem}")
    return out


def pair_inputs():
    pairs = defaultdict(set)
    for stem in VOLUMES:
        for q in read(stem):
            key = (q["release_date"], q["candidate_id"], q["earlier_sentinel2_version"], q["later_sentinel2_version"])
            pairs[key].add((float(q["earlier_gap_m"]), float(q["later_gap_m"]), float(q["shell_above_ground_m"]),
                            float(q["diameter_m"])))
    roof = indexed_by_roof("sentinel2_space_time_coverage_roof_registry")
    for stem in ("cushing_outer_capture_pairs_fits", "cushing_outer_capture_pairs_supplement_fits", "sentinel2_wider_registration_gap_pairs"):
        status = "wider_registration_status" if stem.startswith("sentinel2") else "registration_status"
        for q in read(stem):
            if q[status] != "checked_offset" or q["registered_both_shadow_diagnostics_pass"] != "True":
                continue
            key = (q["release_date"], q["candidate_id"], q["earlier_version"], q["later_version"])
            g = roof[q["candidate_id"]]
            gaps = (float(q["earlier_registered_gap_m"]), float(q["later_registered_gap_m"]))
            pairs[key].add((*gaps, float(g["shell_above_ground_m"]), float(g["diameter_m"])))
    return pairs


def calibrated_change(first, last, shell, diameter, model, method):
    field = HEIGHT_METHODS[method]
    # A range for the change is not a range for either endpoint. For the
    # multivariate fit, use the training mean to neutralize this missing feature.
    typical_width = float(model[0][-1])
    a, b = (predict(dict(raw_gap_m=gap, shell_m=shell, relative_gap=gap / shell, diameter_m=diameter,
                         profile_span_m=typical_width), model)[field] for gap in (first, last))
    return a - b


def corrected_observations(row, observations, red, fits, eligible, audit):
    issue, cutoff = row["release_date"], row["cutoff"]
    models = {}
    counts = {}
    for terminal in {q["terminal_id"] for q in observations}:
        train = [q for q in eligible if q["terminal_id"] != terminal]
        counts[terminal] = len(train)
        if len(train) >= 20:
            models[terminal] = learn(train)
    by_arm = {m: [] for m in MODELS}
    applied = 0
    for q in observations:
        terminal = q["terminal_id"]
        key = (issue, q["candidate_id"], q["scene_before"], q["scene_after"])
        model = models.get(terminal)
        value = q["pair_signal"]
        matching = [e for e in fits.get(key, ()) if q["candidate_id"] in red
                    and math.isclose((e[0] - e[1]) * q["volume_per_m"], value, abs_tol=1e-10)
                    and min(e[0], e[1]) >= 0 and max(e[0], e[1]) <= e[2] and e[3] > 0]
        endpoints = matching[0] if len(matching) == 1 else None
        used = endpoints is not None and model is not None
        volumes = {}
        for method in MODELS:
            volume = value
            if method != SUPPLEMENT and used:
                volume = calibrated_change(*endpoints, model, method) * q["volume_per_m"]
            by_arm[method].append(q | dict(pair_signal=volume))
            volumes[method] = 1000 * volume
        applied += int(used)
        if audit is not None:
            audit.append(dict(release_date=issue, cutoff=cutoff, candidate_id=q["candidate_id"], terminal_id=terminal,
                observed_before_version=key[2], observed_after_version=key[3], unique_b04_pair=endpoints is not None, applied=used,
                available_training_labels_excluding_terminal=counts[terminal], **{f"{m}_kbbl": v for m, v in volumes.items()}))
    return by_arm, applied


def replay(controls, measurement_audit, *, state=None, before=None):
    geometry = {(q["release_date"], q["week_ending"]): q for s in ("sentinel2_space_time_coverage_weeks", "holdout_optical_weeks")
                for q in read(s)}
    base = indexed(f"{OUTER}_forecasts")
    past_reports = indexed("cushing_robust_stack_forecasts")
    fits = pair_inputs()
    regular = outer_pairs("cushing_outer_capture_pairs_fits")
    supplement = outer_pairs("cushing_outer_capture_pairs_supplement_fits")
    err_history, histories = [], {m: [] for m in MODELS}
    for row, arms, _ in combined_sources():
        issue, cutoff, known = (row[k] for k in ("release_date", "cutoff", "known"))
        if before is not None and known >= before:
            break
        ref = base[issue]
        if (cutoff != datetime.fromisoformat(ref["cutoff_utc"]) or known != datetime.fromisoformat(ref["known_utc"])
                or row["actual"] != float(ref["actual"])):
            raise ValueError(f"Invalid optical cohort on {issue}")
        area = float(geometry[issue, row["week_ending"]]["dated_reference_area_m2"])
        prior = float(past_reports[issue]["prior"])
        outer = regular[issue] + supplement[issue]
        if regular[issue] and arms[-1]:
            raise ValueError(f"Outer pairs overlap the primary cohort on {issue}")
        observations = arms[-1] + [observation(q, row, area, prior) for q in outer]
        red = {q["candidate_id"] for q in arms[1]} | {q["candidate_id"] for q in outer}
        if (len({q["candidate_id"] for q in observations}) != len(observations)
                or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observations)):
            raise ValueError(f"Duplicated or unavailable observation on {issue}")
        eligible = [q for q in controls if q["ready"] < cutoff]
        by_arm, applied = corrected_observations(row, observations, red, fits, eligible, measurement_audit)
        errors = [q for q in err_history if q["known"] < cutoff]
        values = {}
        for method, sample in by_arm.items():
            model_row = row | dict(observed=sample, hub_area=area, area_fraction=math.fsum(q["area"] for q in sample) / area)
            values[method], _ = run_forecast(histories[method], errors, model_row, bool(outer))
        if not math.isclose(values[SUPPLEMENT], float(ref[SUPPLEMENT]), abs_tol=1e-9):
            raise ValueError(f"The existing satellite forecast differs on {issue}")
        err_history.append(dict(known=known, error=row["actual"] - row["public"]))
        yield {k: row[k] for k in ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual")} | values | dict(
            public_ensemble=row["public"], observed_roofs=len(observations), corrected_roofs=applied,
            available_height_labels=len(eligible), latest_height_label_utc=max((q["ready"] for q in eligible), default=None))
    if state is not None:
        state.update(histories=histories, errors=err_history)


def held_out_aerial(controls):
    fields = ("candidate_id", "flight_date", "satellite_version", "satellite_captured_at_utc", "satellite_sensor")
    refs = {tuple(r[k] for k in fields): r for r in read("aerialok_native_optical_pairs")}
    geometry = indexed_by_roof("cushing_modern_lidar_optical_registry")
    rows = aerial()
    by_terminal = {t: learn([r for r in controls if r["terminal_id"] != t]) for t in {r["terminal_id"] for r in rows}}
    for r in rows:
        tank = r["candidate_id"]
        first, last = (refs[tank, r[f"{m}_flight"], r[f"{m}_satellite_version"], r[f"{m}_satellite_capture_utc"],
                            r[f"{m}_satellite_sensor"]] for m in ("first", "last"))
        g = geometry[tank]
        shell = float(g["prior_rim_z_p95_native"]) - float(g["prior_ground_z_p50_native"])
        endpoints = (float(first["satellite_gap_m"]), float(last["satellite_gap_m"]), shell, float(g["diameter_m"]))
        a = {m: calibrated_change(*endpoints, by_terminal[r["terminal_id"]], m) for m in MODELS[1:]}
        yield {k: r[k] for k in ("candidate_id", "terminal_id", "first_flight", "last_flight", "max_gap_hours",
                                 "roof_area_m2", "target", "physical")} | a


def score(rows):
    if not rows:
        return dict(releases=0, models={})
    reference = np.array([1000 * abs(r[SUPPLEMENT] - r["actual"]) for r in rows])
    result = {}
    for m in (PUBLIC, *MODELS):
        loss = np.array([1000 * abs(r[m] - r["actual"]) for r in rows])
        result[m] = dict(mae_kbbl=float(loss.mean()), delta_vs_reference_95_kbbl=block_bootstrap_mean(loss - reference))
    return dict(releases=len(rows), models=result)


def market(forecasts):
    book = {r["release_date"]: r for r in forecasts}
    history = {m: [] for m in MODELS}
    for r, _ in market_windows():
        f = book[r["release_date"]]
        decision = r["decision"]
        if not f["cutoff"] < f["known"] <= decision < r["entry"] < r["exit_at"] or f["actual"] != r["x"][0]:
            raise ValueError(f"Different inventory vintage on {r['release_date']}")
        for m, rows in history.items():
            case = r | dict(x=replace_expectation(r, f[m]),
                            response_features=features(r["x"][2], r["volatility_usd"], f["actual"] - f[m]))
            train = [p for p in rows if p["exit_at"] < decision][-HISTORY:]
            rows.append(case)
            if len(train) < MIN_WINDOWS:
                continue
            point = volatility_forecast(train, case)
            calibration = [p for p in train if "base_estimate" in p]
            delta, _ = conditional_residual(calibration, case) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            case["base_estimate"] = point
            result = trade(r, f"{m}:volatility", point + delta, 3, training_at=train[-1]["exit_at"],
                           calibration_at=calibration[-1]["exit_at"] if calibration else None)
            yield result | dict(realized_long_move_usd=r["long_gross"], base_estimate_usd=point,
                                training_windows=len(train),
                                net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def register_forward(issue, *, scoring):
    path = DERIVED / f"{STEM}_{issue}_registration.json"
    original = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    recovered = original.with_name(f"{original.stem}_recovered_optical.json")
    leader = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    initial, expanded, leader_record = (json.loads(p.read_bytes()) for p in (original, recovered, leader))
    lineage = ("release_date", "week_ending", "information_cutoff_utc", "registration_deadline_utc")
    cutoff, deadline = (datetime.fromisoformat(initial[k]) for k in lineage[-2:])
    if not path.exists():
        if scoring or not cutoff <= datetime.now(timezone.utc) < deadline:
            raise ValueError("The calibrated forecast requires registration before release")
        if (any(q[k] != initial[k] for q in (expanded, leader_record) for k in lineage)
                or any(digest(ROOT / k) != v for k, v in leader_record["sources_sha256"].items())):
            raise ValueError("The previously registered forecast sources differ")
        if not all(cutoff <= datetime.fromisoformat(q["recorded_at_utc"]) < deadline
                   for q in (initial, expanded, leader_record)):
            raise ValueError("The original inputs missed their forecast deadline")
        if issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The target release is already in the first-release archive")
        stem = f"cushing_forward_native_optical_{issue}"
        native = DERIVED / f"{stem}_summary.json"
        native_pairs = DERIVED / f"{stem}_pairs.csv"
        area = float(json.loads(native.read_bytes())["dated_roof_area_m2"])
        roof = indexed_by_roof("cushing_modern_lidar_optical_registry")
        red = {q["candidate_id"] for q in initial["measured_pairs"]}
        fits = defaultdict(set)
        for q in read(f"{stem}_pairs"):
            if (q["candidate_id"] not in red or q["registered_both_shadow_diagnostics_pass"] != "True"
                    or q["registration_status"] != "checked_offset"):
                continue
            key = (issue, q["candidate_id"], q["earlier_version"], q["later_version"])
            g = roof[q["candidate_id"]]
            if (g["modern_input_status"] != "available" or g["latest_engineering_date"] >= cutoff.date().isoformat()
                    or not math.isclose(float(g["circular_area_m2"]), float(q["area_m2"]), abs_tol=1e-8)):
                raise ValueError(f"Unmatched forward roof geometry on {q['candidate_id']}")
            shell = float(g["prior_rim_z_p95_native"]) - float(g["prior_ground_z_p50_native"])
            fits[key].add((float(q["earlier_registered_gap_m"]), float(q["later_registered_gap_m"]),
                           shell, float(g["diameter_m"])))
        observed = [q | dict(area=q["volume_per_m"] * BARREL_M3 * 1e6, available_utc=q["source_available_utc"])
                    for q in initial["measured_pairs"]] + expanded["recovered_roofs"]
        if (len(observed) != int(leader_record["measured_roofs"]) or len({q["candidate_id"] for q in observed}) != len(observed)
                or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observed)):
            raise ValueError("The forward observation panel is incomplete or arrived after cutoff")
        row = dict(release_date=issue, cutoff=cutoff, known=deadline, public=initial["public_ensemble_kbbl"] / 1000,
                   week_ending=initial["week_ending"], hub_area=area,
                   area_fraction=math.fsum(q["area"] for q in observed) / area)
        controls = [q for q in training_rows() if q["ready"] < cutoff]
        state = {}
        with threadpool_limits(limits=1):
            for _ in replay(controls, None, state=state, before=cutoff):
                pass
            arms, applied = corrected_observations(row, observed, red, fits, controls, None)
            estimates = {m: 1000 * run_forecast(state["histories"][m], state["errors"], row | dict(observed=sample),
                                                 False, update=False)[0] for m, sample in arms.items()}
        if not math.isclose(estimates[SUPPLEMENT], leader_record["estimates_kbbl"][SUPPLEMENT], abs_tol=1e-8):
            raise ValueError("The original satellite point forecast differs")
        stamp = datetime.now(timezone.utc)
        if not cutoff <= stamp < deadline or applied == 0:
            raise ValueError("No timely calibrated satellite observation for registration")
        input_stems = (f"{HEIGHT}_controls", "shadow_geometry_estimates", "cushing_modern_lidar_optical_registry",
                       "sentinel2_l1c_archive_availability", "sentinel2_space_time_coverage_roof_registry", f"{STEM}_forecasts",
                       f"{STEM}_measurements", f"{STEM}_training")
        sources = (original, recovered, leader, native, native_pairs, Path(__file__).resolve(),
                   ROOT / "scripts/calibrate_cushing_height_multisource.py", *(DERIVED / f"{k}.csv" for k in input_stems))
        evidence = leader_record["sources_sha256"] | {str(p.relative_to(ROOT)): digest(p) for p in sources}
        freeze(path, {k: initial[k] for k in lineage} | dict(recorded_at_utc=stamp.isoformat(),
            calibration_labels=len(controls), latest_height_label_utc=max(q["ready"] for q in controls).isoformat(),
            latest_eia_training_label_utc=max(q["known"] for q in state["errors"]).isoformat(),
            measured_roofs=len(observed), corrected_b04_roofs=applied, estimates_kbbl=estimates, sources_sha256=evidence))
    record = json.loads(path.read_bytes())
    if (any(record[k] != initial[k] for k in lineage)
            or not all(datetime.fromisoformat(record[k]) < cutoff for k in ("latest_height_label_utc", "latest_eia_training_label_utc"))
            or not cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline
            or any(digest(ROOT / k) != v for k, v in record["sources_sha256"].items())):
        raise ValueError("The calibrated registration does not verify")
    if not scoring:
        return record
    outcome = score_release(original, initial)
    scorecard = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path),
                     estimates_kbbl=record["estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        scorecard.update(actual_eia_change_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                         absolute_errors_kbbl={m: abs(v - actual) for m, v in record["estimates_kbbl"].items()})
    write(path.with_name(f"{STEM}_{issue}_score.json"), (json.dumps(scorecard, indent=2, allow_nan=False) + "\n").encode())
    return scorecard


def main():
    controls = list(training_rows())
    measurement_audit = []
    with threadpool_limits(limits=1):
        aerial_rows = list(held_out_aerial(controls))
        rows = list(replay(controls, measurement_audit))
        trades = list(market(rows))
    anchors = {r["release_date"]: r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"], r["method"]].append(r)
        if r["method"] == f"{SUPPLEMENT}:volatility":
            a = anchors[r["release_date"]]
            if (any(r[k] != int(a[k]) for k in ("near_side", "gross_usd", "net_170_usd", "net_340_usd"))
                    or not math.isclose(r["expected_long_move_usd"], float(a["expected_long_move_usd"]), abs_tol=1e-8)):
                raise ValueError(f"The historical trading leader differs on {r['release_date']}")
    report = dict(status="exploratory_source_dated_calibration", height_training_labels=len(controls),
        held_out_aerial_change=dict(roofs=len(aerial_rows), pairs=len({(r["first_flight"], r["last_flight"]) for r in aerial_rows}),
            methods={m: dict(mae_m=float(np.mean([abs(r[m] - r["target"]) for r in aerial_rows])),
                same_sign=sum(r[m] * r["target"] > 0 for r in aerial_rows),
                opposite_sign=sum(r[m] * r["target"] < 0 for r in aerial_rows),
                mean_abs_signal_m=float(np.mean([abs(r[m]) for r in aerial_rows])))
                for m in ("physical", *MODELS[1:])}), phases={})
    for phase in PHASES:
        series = [r for r in rows if r["study_phase"] == phase]
        reference = groups[phase, f"{SUPPLEMENT}:volatility"]
        risk = {}
        for m in MODELS:
            arm = groups[phase, f"{m}:volatility"]
            if [r["release_date"] for r in arm] != [r["release_date"] for r in reference]:
                raise ValueError(f"The comparison changed the trading cohort on {phase}/{m}")
            matched = list(zip(arm, reference, strict=True))
            delta = [r["net_170_usd"] - a["net_170_usd"] for r, a in matched]
            risk[m] = robustness(arm) | daily_risk(arm) | dict(net_340_usd=sum(r["net_340_usd"] for r in arm),
                changed_directions=sum(r["near_side"] != a["near_side"] for r, a in matched),
                increment_vs_reference_usd=sum(delta), increment_per_window_95_usd=block_bootstrap_mean(delta))
        report["phases"][phase] = dict(all=score(series), direct=score([r for r in series if r["corrected_roofs"]]),
            corrected_roofs=sum(r["corrected_roofs"] for r in series), trades=risk)
        print(phase, report["phases"][phase]["direct"],
              {m: (q["trades"], q["net_170_usd"], q["net_340_usd"]) for m, q in risk.items()}, flush=True)
    export(f"{STEM}_training", controls)
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_trades", trades)
    export(f"{STEM}_measurements", measurement_audit)
    export(f"{STEM}_aerial", aerial_rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.score and not args.release:
        parser.error("--score requires --release")
    if args.release:
        result = register_forward(args.release, scoring=args.score)
        print(json.dumps({k: v for k, v in result.items() if k != "sources_sha256"}, indent=2))
    else:
        main()
