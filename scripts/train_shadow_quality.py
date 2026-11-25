"""Train and test an optical shadow-quality classifier on real engineering controls."""
import json
import math
from collections import defaultdict
from statistics import median

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write
from fit_shadow_geometry import MAX_NEAR_OPTIMAL_SPAN_M, SHADOW_RMSE_GAIN_FRACTION

STEM = "shadow_ml_repeatability"
RESOLVED = "resolved_in_geometry_diagnostic"
GAP_TOLERANCE_M = 2.0
L2 = 0.05
FEATURES = ("fit_gain_over_flat", "fit_gain_over_reversed_sun", "profile_span_m", "contrast_to_fit_residual",
    "relative_distance_to_gap_bound", "fit_residual_reflectance", "fitted_pixels", "diameter_m", "sun_elevation_deg",
    "view_zenith_deg", "registration_displacement_m")


def features(row):
    error, flat, reversed_sun = (float(row[k]) for k in ("pixel_rmse", "flat_rmse", "reversed_sun_rmse"))
    lo, hi = (float(row[k]) for k in ("near_optimal_gap_min_m", "near_optimal_gap_max_m"))
    gap, bound = (float(row[k]) for k in ("estimated_gap_m", "model_maximum_gap_m"))
    east, north = (float(row[k]) for k in ("shift_east_m", "shift_north_m"))
    result = (1-error/max(flat, 1e-9), 1-error/max(reversed_sun, 1e-9), hi-lo,
        -float(row["rim_shadow_reflectance_difference"])/max(error, 1e-9), min(gap, bound-gap)/bound, error,
        float(row["fitted_pixels"]), float(row["diameter_m"]), float(row["sun_elevation_deg"]),
        float(row["view_zenith_deg"]), math.hypot(east, north))
    if not all(math.isfinite(v) for v in result):
        raise ValueError(f"Nonfinite optical diagnostic for {row['candidate_id']}/{row['version_id']}")
    return result


def physics_rank(vector):
    flat, reversed_sun, span, _, gap_bound, *_ = vector
    return min(flat/SHADOW_RMSE_GAIN_FRACTION-1, reversed_sun/SHADOW_RMSE_GAIN_FRACTION,
               (MAX_NEAR_OPTIMAL_SPAN_M-span)/MAX_NEAR_OPTIMAL_SPAN_M, 10*gap_bound)


def fit_logistic(x, y, feature_names, l2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    centre = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(len(y)), (x-centre)/scale))

    def loss(weights):
        logit = design @ weights
        reg = np.r_[0, weights[1:]]
        value = np.mean(np.logaddexp(0, logit)-y*logit) + l2*(reg @ reg)/2
        gradient = (design.T @ (expit(logit)-y))/len(y) + l2*reg
        return value, gradient

    result = minimize(loss, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B", options={"gtol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Optical score training failed: {result.message}")
    return dict(feature_names=list(feature_names), feature_centre=centre.tolist(), feature_scale=scale.tolist(),
                intercept=float(result.x[0]), weights=result.x[1:].tolist())


def fit(records):
    features = [v for r in records for v in r["endpoints"]]
    labels = [r["agree"] for r in records for _ in r["endpoints"]]
    return dict(fit_logistic(features, labels, FEATURES, L2), training_pairs=len(records))


def score(model, vectors):
    scaled = (np.asarray(vectors, dtype=float)-model["feature_centre"])/model["feature_scale"]
    return expit(model["intercept"]+scaled @ np.asarray(model["weights"]))


def pairs():
    fitted = {r["observation_id"]: r for r in read_csv("shadow_change_control_estimates")}
    selected = [r for r in read_csv("shadow_tandem_pass_pairs")
                if r["selected_for_summary"] == "True" and r["roof_type"] == "EFR"]
    identity = set()
    for pair in selected:
        day, candidate = pair["capture_day"], pair["candidate_id"]
        if (day, candidate) in identity or abs(float(pair["other_minus_a_seconds"])) > 60:
            raise ValueError(f"Duplicate or asynchronous calibration pair: {day}/{candidate}")
        identity.add((day, candidate))
        endpoints = [fitted[pair[f"{side}_observation_id"]] for side in ("a", "other")]
        if any((r["candidate_id"], r["version_id"]) != (candidate, pair[f"{side}_version"])
               for r, side in zip(endpoints, ("a", "other"), strict=True)):
            raise ValueError(f"Mismatched native control scene: {day}/{candidate}")
        difference = float(endpoints[1]["estimated_gap_m"])-float(endpoints[0]["estimated_gap_m"])
        if difference != float(pair["other_minus_a_gap_m"]):
            raise ValueError(f"Recorded and recomputed differences disagree: {day}/{candidate}")
        vectors = [features(row) for row in endpoints]
        yield dict(capture_day=day, candidate_id=candidate, cohort_id=endpoints[0]["cohort_id"],
            a_version=pair["a_version"], other_version=pair["other_version"], elapsed_seconds=float(pair["other_minus_a_seconds"]),
            original_pass=pair["both_resolved"] == "True", gap_difference_m=difference,
            absolute_difference_m=abs(difference), agree=abs(difference) <= GAP_TOLERANCE_M,
            physics_rank=min(map(physics_rank, vectors)), endpoints=vectors)


def metrics(rows, difference="absolute_difference_m"):
    errors = [r[difference] for r in rows]
    return dict(n=len(rows), within_2m=sum(e <= GAP_TOLERANCE_M for e in errors),
        within_2m_fraction=sum(e <= GAP_TOLERANCE_M for e in errors)/len(errors) if errors else None,
        median_absolute_m=median(errors) if errors else None,
        p90_absolute_m=float(np.percentile(errors, 90)) if errors else None)


def best_at_count(rows, field, count):
    return sorted(rows, key=lambda r: (-r[field], r["candidate_id"], r.get("version_id", "")))[:count]


def assess(rows, threshold):
    original = [r for r in rows if r["original_pass"]]
    return dict(all_numeric_pairs=metrics(rows), original_rules=metrics(original),
        physics_rank_same_count=metrics(best_at_count(rows, "physics_rank", len(original))),
        ml_rank_same_count=metrics(best_at_count(rows, "ml_score", len(original))),
        ml_threshold=metrics([r for r in rows if threshold is not None and r["ml_score"] >= threshold]))


def choose_threshold(rows):
    original = metrics([r for r in rows if r["original_pass"]])
    ordered = best_at_count(rows, "ml_score", len(rows))
    chosen = None
    successes = 0
    for index, row in enumerate(ordered, 1):
        successes += row["agree"]
        last_at_score = index == len(ordered) or ordered[index]["ml_score"] < row["ml_score"]
        if last_at_score and index >= original["n"] and successes/index >= original["within_2m_fraction"]:
            chosen = row["ml_score"]
    return chosen


def predict_pairs(rows, model):
    for row in rows:
        first, second = score(model, row["endpoints"])
        row.update(a_ml_score=float(first), other_ml_score=float(second), ml_score=float(min(first, second)))


def lidar_check(model, threshold):
    rows = []
    for row in read_csv("shadow_geometry_estimates"):
        if (row["selected_for_pass_metrics"] != "True" or row["roof_type"] != "EFR" or not row["lidar_gap_m"]
                or not row["estimated_gap_m"]):
            continue
        vector = features(row)
        rows.append(dict(candidate_id=row["candidate_id"], version_id=row["version_id"],
            optical_day=row["optical_capture_at_utc"][:10], sensor_gap_hours=float(row["minimum_sensor_gap_hours"]),
            absolute_difference_m=abs(float(row["estimated_gap_m"])-float(row["lidar_gap_m"])),
            original_pass=row["fit_status"] == RESOLVED, physics_rank=physics_rank(vector),
            ml_score=float(score(model, vector))))
    dates = sorted({r["optical_day"] for r in rows})
    return rows, {day: assess([r for r in rows if r["optical_day"] == day], threshold) for day in dates}


def main():
    rows = list(pairs())
    by_day = defaultdict(list)
    for row in rows:
        by_day[row["capture_day"]].append(row)
    *training_days, test_day = sorted(by_day)
    if len(training_days) < 3:
        raise ValueError("At least three training dates and a separate later test date are required")
    train = [r for day in training_days for r in by_day[day]]
    for day in training_days:
        model = fit([r for r in train if r["capture_day"] != day])
        predict_pairs(by_day[day], model)
    threshold = choose_threshold(train)
    model = fit(train)
    test = by_day[test_day]
    predict_pairs(test, model)
    test_metrics = assess(test, threshold)
    spatial = []
    for cohort in sorted({r["cohort_id"] for r in test}):
        spatial_model = fit([r for r in train if r["cohort_id"] != cohort])
        subset = [dict(r) for r in test if r["cohort_id"] == cohort]
        predict_pairs(subset, spatial_model)
        spatial.extend(subset)
    lidar, lidar_metrics = lidar_check(model, threshold)
    for row, withheld in zip(sorted(test, key=lambda r: (r["cohort_id"], r["candidate_id"])),
                             sorted(spatial, key=lambda r: (r["cohort_id"], r["candidate_id"])), strict=True):
        row["ml_score_withheld_cohort"] = withheld["ml_score"]
    model.update(training_dates=training_days, test_date=test_day, agreement_tolerance_m=GAP_TOLERANCE_M,
                 l2=L2, threshold_from_out_of_date_predictions=threshold, use="engineering_experiment")
    summary = dict(training_dates=training_days, later_test_date=test_day, tested_pairs=len(rows),
        selected_training_threshold=threshold, training_predictions_by_left_out_day=assess(train, threshold),
        later_test=test_metrics, later_test_with_each_cohort_withheld_from_training=assess(spatial, threshold),
        lidar_separate_pass_diagnostic=lidar_metrics,
        interpretation="Targets are agreement within two metres on paired optical passes up to 60 seconds apart. "
        "Both endpoint labels use the pair agreement, but predictors use only that endpoint's own optical fit. "
        "Candidates already require an externally documented floating roof and a numeric physical fit. No learned gap, "
        "tank detection or volume is produced. LiDAR is evaluated separately and its acquisitions can differ in time. "
        "The engineering model uses 2024 training scenes and a 2026 test scene. It must not be applied to a historical "
        "investment test with a decision before training labels were available. EIA and market data are absent.")
    pair_rows = [{k: v for k, v in r.items() if k != "endpoints"} | dict(split="later_test" if r["capture_day"] == test_day
        else "training_held_out_date", ml_score_withheld_cohort=r.get("ml_score_withheld_cohort")) for r in rows]
    export(f"{STEM}_pairs", pair_rows)
    export(f"{STEM}_lidar", lidar)
    for label, result in (("model", model), ("summary", summary)):
        write(DERIVED / f"{STEM}_{label}.json", (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
