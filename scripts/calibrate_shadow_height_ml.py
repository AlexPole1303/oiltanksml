"""Learn and verify a bounded correction to the optical floating-roof gap."""
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from statistics import median

import numpy as np
from scipy.optimize import least_squares

from build_ok_aerial_sentinel2_controls import DB as AERIAL_DB
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write
from fit_shadow_geometry import estimate, inputs
from train_shadow_quality import RESOLVED

STEM = "shadow_height_ml_calibration"
TRAIN_DAY = "2022-12-30"
TEST_DAY = "2023-01-04"
FEATURES = ("raw_gap_m", "diameter_m", "profile_span_m", "gain_over_flat", "contrast_to_residual", "box_gap_minus_raw_gap_m")
METHODS = ("raw", "offset", "ml")
L2 = 0.35
ROBUST_SCALE_M = 1.5
MAX_CORRECTION_M = 5


def field_vector(row):
    gap = float(row["estimated_gap_m"])
    rmse, flat = (float(row[k]) for k in ("pixel_rmse", "flat_rmse"))
    low, high = (float(row[f"near_optimal_gap_{edge}_m"]) for edge in ("min", "max"))
    return np.asarray([gap, float(row["diameter_m"]), high-low, 1-rmse/max(flat, 1e-9),
                       float(row["rim_shadow_reflectance_difference"])/max(rmse, 1e-9),
                       float(row["box_response_gap_m"])-gap])


def clip_gap(value, shell):
    return min(max(float(value), 0.), float(shell))


def learn(rows):
    matrix = np.array([field_vector(r) for r in rows])
    y = np.array([r["reference_gap_m"]-float(r["estimated_gap_m"]) for r in rows])
    site_counts = Counter(r["terminal_id"] for r in rows)
    weights = np.asarray([1/site_counts[r["terminal_id"]] for r in rows])
    weights /= weights.mean()
    centre = np.average(matrix, axis=0, weights=weights)
    scale = np.sqrt(np.average((matrix-centre)**2, axis=0, weights=weights))
    scale[scale == 0] = 1.
    x = (matrix-centre)/scale
    design = np.column_stack((np.ones(len(rows)), np.clip(x, -3, 3)))
    root = np.sqrt(weights)

    def residual(coef):
        return np.r_[root*(design @ coef-y), np.sqrt(L2*len(rows))*coef[1:]]

    bound = np.full(design.shape[1], np.inf)
    bound[1] = 0.25*scale[0]
    result = least_squares(residual, np.zeros(design.shape[1]), bounds=(-bound, bound),
                           loss="soft_l1", f_scale=ROBUST_SCALE_M)
    if not result.success:
        raise RuntimeError(result.message)
    return dict(feature_names=list(FEATURES), centre=centre.tolist(), scale=scale.tolist(),
        intercept=float(result.x[0]), weights=result.x[1:].tolist(), offset_m=float(median(y)),
        training_observations=len(rows), training_terminals=sorted(site_counts))


def predict(row, model):
    raw, shell = float(row["estimated_gap_m"]), float(row["shell_above_ground_m"])
    vector = np.clip((field_vector(row)-model["centre"])/model["scale"], -3, 3)
    residual = model["intercept"]+vector @ np.asarray(model["weights"])
    return dict(raw=raw, offset=clip_gap(raw+model["offset_m"], shell),
                ml=clip_gap(raw+np.clip(residual, -MAX_CORRECTION_M, MAX_CORRECTION_M), shell))


def reference_rows(terminal_for):
    for r in read_csv("shadow_geometry_estimates"):
        day = r["optical_capture_at_utc"][:10]
        terminal = terminal_for.get(r["candidate_id"])
        if (day in (TRAIN_DAY, TEST_DAY) and r["roof_type"] == "EFR" and r["selected_for_pass_metrics"] == "True"
                and r["fit_status"] == RESOLVED and r["lidar_gap_m"] and terminal):
            yield dict(r, terminal_id=terminal, reference_gap_m=float(r["lidar_gap_m"]),
                       time_separation_hours=float(r["maximum_sensor_gap_hours"]))


def aerial_rows():
    selected = [r for r in read_csv("ok_aerial_2025_sentinel2_roof_comparisons") if r["roof_type"] == "EFR"
                and r["selected_for_comparison"] == r["both_diagnostics_pass"] == "True"
                and r["exterior_clock_status"] == "exterior_matches_tile_time"]
    with closing(read_only(AERIAL_DB)) as db:
        db.row_factory = sqlite3.Row
        stored, kernels = inputs(db)
    observations = {str(r["observation_id"]):r for r in stored}

    def refit(row):
        observation = observations[row["optical_observation_id"]]
        if (observation["candidate_id"], observation["version_id"]) != (row["candidate_id"], row["optical_version"]):
            raise ValueError("Incorrect aerial comparison observation")
        result = estimate(observation, float(row["shared_lidar_rim_to_ground_m"]), kernels[observation["platform"]],
                          calibration_db=AERIAL_DB)
        if float(row["optical_gap_m"]) != result["estimated_gap_m"] or result["fit_status"] != RESOLVED:
            raise ValueError("Aerial baseline fit did not reproduce")
        return dict(result, terminal_id=row["terminal_id"], reference_gap_m=float(row["aerial_gap_m"]),
                    time_separation_hours=float(row["maximum_nominal_separation_hours"]))

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(refit, selected))


def scored(rows, models, test):
    for row in rows:
        model = models[row["terminal_id"]]
        estimate = predict(row, model)
        yield dict(test=test, candidate_id=row["candidate_id"], terminal_id=row["terminal_id"], version_id=row["version_id"],
                   optical_day=row["optical_capture_at_utc"][:10], reference_gap_m=row["reference_gap_m"],
                   time_separation_hours=row["time_separation_hours"], calibration_samples=model["training_observations"],
                   candidate_terminal_in_training=row["terminal_id"] in model["training_terminals"],
                   **{f"{k}_gap_m": v for k, v in estimate.items()})


def metrics(rows):
    result = {}
    for method in METHODS:
        errors = np.array([r[f"{method}_gap_m"]-r["reference_gap_m"] for r in rows])
        result[method] = dict(n=len(rows), mean_absolute_m=float(np.mean(abs(errors))),
            median_absolute_m=float(np.median(abs(errors))), p90_absolute_m=float(np.quantile(abs(errors), 0.9)),
            mean_signed_m=float(np.mean(errors)), within_one_m=int(np.sum(abs(errors)<=1)))
    return result


def tandem_volume_checks(model):
    fits = {r["observation_id"]:r for r in read_csv("shadow_change_control_estimates")}
    roof = {r["candidate_id"]:r for r in read_csv("sentinel2_space_time_coverage_roof_registry")}
    days = {}
    for r in read_csv("shadow_tandem_pass_pairs"):
        if r["selected_for_summary"] == r["both_resolved"] == "True" and r["candidate_id"] in roof:
            key = r["capture_day"], r["candidate_id"], r["other_sensor"]
            days[key] = [predict(fits[r[f"{side}_observation_id"]], model) for side in ("a", "other")]
    groups = defaultdict(list)
    for r in read_csv("shadow_tandem_change_pairs"):
        tank = r["candidate_id"]
        if r["elapsed_days"] != "10" or r["all_four_resolved"] != "True" or tank not in roof:
            continue
        before, after = (days[r[k], tank, r["other_sensor"]] for k in ("earlier_day", "later_day"))
        if (before[0]["raw"]-after[0]["raw"], before[1]["raw"]-after[1]["raw"]) != (float(r["a_change_m"]), float(r["other_change_m"])):
            raise ValueError("Tandem baseline changed while replaying calibration")
        factor = float(roof[tank]["circular_area_m2"])/BARREL_M3/1000
        estimates = {m: [(b[m]-a[m])*factor for b,a in zip(before,after,strict=True)] for m in METHODS}
        groups[r["earlier_day"],r["later_day"],roof[tank]["terminal_id"]].append(estimates)
    for (start, end, terminal), values in sorted(groups.items()):
        row = dict(earlier_day=start, later_day=end, terminal_id=terminal, tanks=len(values))
        for method in METHODS:
            first, second = (math.fsum(v[method][i] for v in values) for i in (0,1))
            row.update({f"{method}_first_change_kbbl":first, f"{method}_second_change_kbbl":second,
                        f"{method}_absolute_disagreement_kbbl":abs(first-second), f"{method}_opposite_nonzero_sign":first*second<0})
        yield row


def main():
    with closing(read_only(SOURCE)) as db:
        terminal_for = dict(db.execute("SELECT candidate_id,documented_terminal_id FROM tank_engineering_registry"))
    known = list(reference_rows(terminal_for))
    train = [r for r in known if r["optical_capture_at_utc"][:10] == TRAIN_DAY]
    later = [r for r in known if r["optical_capture_at_utc"][:10] == TEST_DAY]
    other = aerial_rows()
    full = learn(train)
    terminals = sorted({r["terminal_id"] for r in train+later+other})
    withheld = {site:learn([r for r in train if r["terminal_id"] != site]) for site in terminals}
    checks = {"training_day_withheld_terminal": list(scored(train, withheld, "training_day_withheld_terminal")),
              "later_optical_day_withheld_terminal": list(scored(later, withheld, "later_optical_day_withheld_terminal")),
              "aerial_2025_withheld_terminal": list(scored(other, withheld, "aerial_2025_withheld_terminal"))}
    volume = list(tandem_volume_checks(full))
    result = {key:metrics(rows) for key, rows in checks.items()}
    preferred = min(METHODS, key=lambda k:result["training_day_withheld_terminal"][k]["mean_absolute_m"])
    volume_metrics = {m: dict(terminal_intervals=len(volume), median_absolute_disagreement_kbbl=median(
        r[f"{m}_absolute_disagreement_kbbl"] for r in volume), opposite_nonzero_sign=sum(r[f"{m}_opposite_nonzero_sign"] for r in volume))
        for m in METHODS}
    report = dict(training_optical_date=TRAIN_DAY, held_out_optical_date=TEST_DAY, preferred_from_training_day=preferred,
        reference_results=result, tandem_volume_results=volume_metrics,
        interpretation="Calibration uses only Dec 30 optical inputs and a single Dec 31 LiDAR campaign. All direct reference "
        "checks withhold the evaluated terminal from the model. Jan 4 uses the same LiDAR campaign at a longer delay. "
        "The independent 2025 aerial reference has its own shadow measurement and is not simultaneous. The tandem check "
        "uses a frozen full calibration and compares two ten-day estimates with almost simultaneous endpoints. No market or EIA outcomes are used.")
    full.update(training_optical_date=TRAIN_DAY, l2=L2, robust_scale_m=ROBUST_SCALE_M, max_correction_m=MAX_CORRECTION_M)
    export(f"{STEM}_reference_checks", [row for values in checks.values() for row in values])
    export(f"{STEM}_tandem_volumes", volume)
    for suffix, values in (("summary", report), ("model", full)):
        write(DERIVED/f"{STEM}_{suffix}.json", (json.dumps(values, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
