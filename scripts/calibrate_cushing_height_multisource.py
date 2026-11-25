"""Learn bounded Sentinel-2 height corrections from dated laser and aerial campaigns."""

import json
import math
from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import least_squares

from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_public import write
from fit_tandem_shadow_residual import measured_pairs

STEM = "cushing_height_multisource_ml"
MAX_LASER_GAP_HOURS = 24.
MAX_AERIAL_GAP_HOURS = 24.
MAX_CORRECTION_M = 2.
PENALTY = 0.2
FEATURES = ("relative_gap", "diameter_m", "profile_span_m")
METHODS = ("physical", "constant_offset", "multisource_ml", "midpoint_calibration")


def sample(row, source, day, optical_stamp, reference_hours, gap_fields, registry, source_record):
    candidate = row["candidate_id"]
    geometry = registry[candidate]
    raw, lower, upper, reference = (float(row[k]) for k in gap_fields)
    diameter, shell = geometry
    if (not all(map(math.isfinite, (raw, lower, upper, reference, diameter, shell)))
            or not 0 <= lower <= raw <= upper or diameter <= 0 or shell <= 0):
        raise ValueError(f"Invalid gap range for {source}/{day}/{candidate}")
    return dict(candidate_id=candidate, terminal_id=row["terminal_id"], source=source, reference_date=day,
                optical_captured_at_utc=optical_stamp, maximum_time_separation_hours=reference_hours,
                raw_gap_m=raw, reference_gap_m=reference, diameter_m=diameter, shell_m=shell,
                relative_gap=raw/shell, profile_span_m=upper-lower, source_record=source_record)


def controls():
    roster = {r["candidate_id"]:r for r in read("tank_engineering_registry") if r["documented_roof_type"] == "EFR"}
    registry = {r["candidate_id"]:(float(r["diameter_m"]), float(r["prior_rim_z_p95_native"])-float(r["prior_ground_z_p50_native"]))
                for r in read("cushing_modern_lidar_optical_registry") if r["modern_input_status"] == "available"
                and r["candidate_id"] in roster}
    lidar_fields = ("estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m", "lidar_gap_m")
    for line, row in enumerate(read("shadow_geometry_estimates"), 2):
        key = row["candidate_id"]
        if (key in registry and row["selected_for_pass_metrics"] == "True" and row["fit_status"] == RESOLVED
                and row["lidar_gap_m"] and float(row["maximum_sensor_gap_hours"]) <= MAX_LASER_GAP_HOURS):
            yield sample(dict(row, terminal_id=roster[key]["documented_terminal_id"]), "airborne_lidar", "2022-12-31",
                         row["optical_capture_at_utc"], float(row["maximum_sensor_gap_hours"]), lidar_fields, registry,
                         f"shadow_geometry_estimates.csv:{line}")

    for line, row in enumerate(read("aerialok_native_optical_pairs"), 2):
        if (row["candidate_id"] in registry and row["satellite_sensor"].startswith("Sentinel-2 B04")
                and row["both_pass_at_inferred_best"] == row["all_near_clock_fits_visible"] == "True"
                and float(row["maximum_inferred_separation_hours"]) <= MAX_AERIAL_GAP_HOURS):
            yield sample(row, "oklahoma_aerial", row["flight_date"], row["satellite_captured_at_utc"],
                         float(row["maximum_inferred_separation_hours"]),
                         ("satellite_gap_m", "satellite_gap_min_m", "satellite_gap_max_m", "aerial_gap_m"), registry,
                         f"aerialok_native_optical_pairs.csv:{line}")

    for line, row in enumerate(read("cushing_gedi_native_optical_fits"), 2):
        if (row["candidate_id"] in registry and row["fit_status"] == RESOLVED
                and abs(float(row["sentinel2_minus_gedi_time_hours"])) <= MAX_LASER_GAP_HOURS):
            yield sample(row, "gedi", row["gedi_capture_utc"][:10], row["sentinel2_capture_utc"],
                         abs(float(row["sentinel2_minus_gedi_time_hours"])),
                         ("b04_wall_to_roof_m", "b04_profile_min_m", "b04_profile_max_m", "gedi_corrected_wall_to_roof_m"),
                         registry, f"cushing_gedi_native_optical_fits.csv:{line}")

    for line, row in enumerate(read("cushing_icesat2_native_optical_controls"), 2):
        if row["candidate_id"] in registry and float(row["max_time_separation_hours"]) <= MAX_LASER_GAP_HOURS:
            yield sample(row, "icesat2", row["icesat_granule_start_utc"][:10], row["sentinel2_captured_at_utc"],
                         float(row["max_time_separation_hours"]), ("sentinel2_estimated_gap_m", "sentinel2_near_optimal_gap_min_m",
                         "sentinel2_near_optimal_gap_max_m", "icesat_wall_to_roof_m"), registry,
                         f"cushing_icesat2_native_optical_controls.csv:{line}")


def independent_rows(rows):
    selected = {}
    def rank(row):
        return row["maximum_time_separation_hours"], row["optical_captured_at_utc"], row["profile_span_m"]

    for r in rows:
        if not r["terminal_id"]:
            continue
        key = r["candidate_id"], r["source"], r["reference_date"]
        if key not in selected or rank(r) < rank(selected[key]):
            selected[key] = r
    return list(selected.values())


def learn(rows):
    x = np.array([[r[k] for k in FEATURES] for r in rows])
    target = np.array([r["reference_gap_m"]-r["raw_gap_m"] for r in rows])
    by_campaign = Counter((r["source"], r["reference_date"]) for r in rows)
    weight = np.array([1 / by_campaign[r["source"], r["reference_date"]] for r in rows])
    weight /= weight.sum()
    centre = np.average(x, axis=0, weights=weight)
    scale = np.sqrt(np.average((x-centre)**2, axis=0, weights=weight))
    scale[scale == 0] = 1.
    x = np.column_stack((np.ones(len(x)), np.clip((x-centre)/scale, -3., 3.)))
    root = np.sqrt(weight)
    result = least_squares(lambda beta: np.r_[root * (x @ beta - target), np.sqrt(PENALTY)*beta],
                           np.zeros(x.shape[1]), loss="soft_l1", f_scale=1.5)
    if not result.success:
        raise RuntimeError(result.message)
    order = np.argsort(target)
    median = target[order[np.searchsorted(np.cumsum(weight[order]), 0.5)]]
    # The correction has no intercept and preserves the wall midpoint.  A fitted
    # slope changes amplitude without learning a universal shift between sources.
    gap = np.array([r["relative_gap"]-.5 for r in rows])
    midpoint = least_squares(lambda beta:np.r_[root*(gap*beta[0]-target),np.sqrt(.07)*beta],
                             np.zeros(1),bounds=(-4.,4.),loss="soft_l1",f_scale=1.)
    if not midpoint.success:
        raise RuntimeError(midpoint.message)
    return centre, scale, result.x, float(median), float(midpoint.x[0])


def predict(row, model):
    centre, scale, beta, offset, midpoint = model
    raw, shell = row["raw_gap_m"], row["shell_m"]
    features = np.array([row[k] for k in FEATURES])
    correction = beta[0] + np.clip((features-centre)/scale, -3., 3.) @ beta[1:]
    estimates = {method:float(np.clip(raw + np.clip(value, -MAX_CORRECTION_M, MAX_CORRECTION_M), 0., shell))
                 for method, value in (("constant_offset", offset), ("multisource_ml", correction),
                                       ("midpoint_calibration", midpoint*(row["relative_gap"]-.5)))}
    return {"physical": raw} | estimates


def compare(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source"], row["reference_date"], row["terminal_id"]].append(row)
    for (_, day, terminal), test in sorted(groups.items()):
        # Withhold the entire reference campaign, including measurements by other
        # terminals, and withhold this terminal across all training campaigns.
        optical_days = {r["optical_captured_at_utc"][:10] for r in test}
        train = [r for r in rows if r["terminal_id"] != terminal and r["reference_date"] != day
                 and r["optical_captured_at_utc"][:10] not in optical_days]
        campaigns = {(r["source"], r["reference_date"]) for r in train}
        if len(train) < 20 or len(campaigns) < 3:
            continue
        model = learn(train)
        for r in test:
            yield dict(r, training_rows=len(train), training_campaigns=len(campaigns),
                       training_reference_dates="|".join(sorted({q["reference_date"] for q in train})),
                       training_terminals="|".join(sorted({q["terminal_id"] for q in train})),
                       **{f"{method}_gap_m": value for method, value in predict(r, model).items()})


def tandem_changes(controls):
    fits = {r["observation_id"]: r for r in read("shadow_change_control_estimates")}
    pairs = {(r["capture_day"], r["candidate_id"], r["other_sensor"]): r for r in measured_pairs()}
    endpoints = {}
    for r in read("shadow_tandem_pass_pairs"):
        key = r["capture_day"], r["candidate_id"], r["other_sensor"]
        if key not in pairs or r["selected_for_summary"] != "True" or r["both_resolved"] != "True" or abs(float(r["other_minus_a_seconds"])) > 60:
            continue
        if key in endpoints:
            raise ValueError(f"Duplicate tandem endpoint for {key}")
        endpoints[key] = [fits[r[f"{side}_observation_id"]] for side in ("a", "other")]
    intervals = defaultdict(list)
    for row in read("terminal_change_repeatability_tank_controls"):
        if row["elapsed_days"] == "10" and row["all_four_resolved"] == "True":
            intervals[row["earlier_day"], row["later_day"], row["terminal_id"]].append(row)
    for (start, end, terminal), rows in sorted(intervals.items()):
        train = [r for r in controls if r["terminal_id"] != terminal and r["reference_date"] < start
                 and r["optical_captured_at_utc"][:10] < start]
        model = learn(train)
        changes = defaultdict(list)
        for row in rows:
            tank, sensor = row["candidate_id"], row["other_sensor"]
            predictions = []
            for day in (start, end):
                current = []
                for fit in endpoints[day, tank, sensor]:
                    raw, shell, diameter, lower, upper = (float(fit[k]) for k in (
                        "estimated_gap_m", "shell_above_ground_m", "diameter_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m"))
                    current.append(predict(dict(raw_gap_m=raw, shell_m=shell, relative_gap=raw/shell,
                                                diameter_m=diameter, profile_span_m=upper-lower), model))
                predictions.append(current)
            scale = float(row["kbbl_per_m"])
            for method in METHODS:
                changes[method].append([(before[method]-after[method])*scale for before, after in zip(*predictions, strict=True)])
            if any(not math.isclose(a, float(row[f"{label}_platform_change_kbbl"]), abs_tol=1e-9)
                   for label, a in zip(("first", "second"), changes["physical"][-1], strict=True)):
                raise ValueError(f"Raw tandem change differs for {tank}/{start}/{end}")
        result = dict(earlier_day=start, later_day=end, terminal_id=terminal, tanks=len(rows), training_rows=len(train))
        for method, values in changes.items():
            first, second = map(math.fsum, zip(*values, strict=True))
            result.update({f"{method}_first_change_kbbl":first, f"{method}_second_change_kbbl":second,
                           f"{method}_absolute_disagreement_kbbl":abs(first-second),
                           f"{method}_opposite_sign":first*second < 0})
        yield result


def metrics(rows):
    campaign_sizes = Counter((r["source"], r["reference_date"]) for r in rows)
    weights = [1 / campaign_sizes[r["source"], r["reference_date"]] for r in rows]
    methods = {}
    for method in METHODS:
        errors = [abs(r[f"{method}_gap_m"] - r["reference_gap_m"]) for r in rows]
        methods[method] = dict(mae_m=float(np.mean(errors)), p90_m=float(np.quantile(errors, .9)),
                               campaign_weighted_mae_m=float(np.average(errors, weights=weights)))
    return dict(observations=len(rows), terminals=len({r["terminal_id"] for r in rows}),
                campaigns=len(campaign_sizes), methods=methods)


def campaign_differences(rows):
    by_campaign = defaultdict(list)
    for r in rows:
        actual = r["reference_gap_m"]
        candidate = abs(r["midpoint_calibration_gap_m"]-actual)
        by_campaign[r["source"],r["reference_date"]].append(
            [candidate-abs(r[f"{control}_gap_m"]-actual) for control in METHODS[:-1]])
    delta = np.array([np.mean(group,axis=0) for group in by_campaign.values()])
    rng = np.random.default_rng(532)
    draws = delta[rng.integers(len(delta),size=(10000,len(delta)))].mean(axis=1)
    return {m:dict(mean_delta_m=float(value),bootstrap_interval_95_m=interval.tolist())
            for m,value,interval in zip(METHODS[:-1],delta.mean(axis=0),np.quantile(draws,[.025,.975],axis=0).T,strict=True)}


def main():
    rows = independent_rows(controls())
    output = list(compare(rows))
    tandem = list(tandem_changes(rows))
    result = dict(status="retrospective_campaign_and_terminal_holdout", admitted_controls=len(rows),
                  evaluated=metrics(output), by_source={source: metrics([r for r in output if r["source"] == source])
                                                       for source in sorted({r["source"] for r in output})},
                  equal_campaign_midpoint_differences=campaign_differences(output),
                  ten_day_tandem_changes={method:dict(intervals=len(tandem), median_disagreement_kbbl=float(np.median(
                      [r[f"{method}_absolute_disagreement_kbbl"] for r in tandem])),
                      opposite_sign=sum(r[f"{method}_opposite_sign"] for r in tandem)) for method in METHODS})
    export(f"{STEM}_controls", rows)
    export(f"{STEM}_evaluated", output)
    export(f"{STEM}_tandem_changes", tandem)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
