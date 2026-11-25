"""Compare saved optical fits on LiDAR, fixed-roof and tandem controls."""
import json
from collections import defaultdict
from statistics import mean, median

import numpy as np

from build_operator_development_inputs import one_per_key
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, RESOLVED
from collect_public import write


def describe(values):
    if not values:
        return dict(observations=0)
    absolute = [abs(value) for value in values]
    return dict(observations=len(values), median_absolute_m=median(absolute),
                p90_absolute_m=float(np.percentile(absolute, 90)), mean_signed_m=mean(values))


def difference(earlier, later):
    lower = float(earlier["near_optimal_gap_min_m"]) - float(later["near_optimal_gap_max_m"])
    upper = float(earlier["near_optimal_gap_max_m"]) - float(later["near_optimal_gap_min_m"])
    return lower, upper


def sign_summary(bounds):
    return dict(pairs=len(bounds), zero_outside_near_optimal_range=sum(a > 0 or b < 0 for a, b in bounds),
                median_range_width_m=median(b-a for a, b in bounds) if bounds else None)


def grouped_errors(groups):
    return {group: dict(esa_response=describe([a for a, _ in values]), box_response=describe([b for _, b in values]))
            for group, values in sorted(groups.items())}


def lidar_control():
    groups = defaultdict(list)
    for row in read_csv("shadow_geometry_estimates"):
        if row["roof_type"] != "EFR" or row["selected_for_pass_metrics"] != "True" or not row["lidar_gap_m"]:
            continue
        if not row["box_response_gap_m"] or not row["estimated_gap_m"]:
            continue
        day, reference = row["optical_capture_at_utc"][:10], float(row["lidar_gap_m"])
        errors = float(row["estimated_gap_m"])-reference, float(row["box_response_gap_m"])-reference
        groups[day + "_all"].append(errors)
        if row["fit_status"] == RESOLVED:
            groups[day + "_resolved"].append(errors)
    return grouped_errors(groups)


def fixed_roof_control(observations):
    groups = defaultdict(list)
    bounds = {phase: [] for phase in ("all", "development", "validation", "holdout")}
    for row in read_csv("shadow_fixed_weekly_pairs"):
        before, after = (observations[row["candidate_id"], row[f"{side}_version"]] for side in ("earlier", "later"))
        if (before["estimated_gap_m"], after["estimated_gap_m"]) != (row["earlier_gap_m"], row["later_gap_m"]):
            raise ValueError(f"Fixed roof pair does not match its saved fits for {row['candidate_id']}")
        roof = row["roof_type"]
        errors = (float(row["apparent_level_change_m"]),
                  float(before["box_response_gap_m"])-float(after["box_response_gap_m"]))
        groups[f"{roof}_all"].append(errors)
        if row["both_resolved"] == "True":
            groups[f"{roof}_resolved"].append(errors)
            interval = difference(before, after)
            bounds["all"].append(interval)
            bounds[row["study_phase"]].append(interval)
        if roof == "IFR":
            changed = (before["shift_east_m"], before["shift_north_m"]) != (after["shift_east_m"], after["shift_north_m"])
            groups["IFR_different_fitted_offsets" if changed else "IFR_same_fitted_offsets"].append(errors)
            high_sun = max(float(r["sun_elevation_deg"]) for r in (before, after)) >= 50
            groups["IFR_max_sun_at_least_50" if high_sun else "IFR_max_sun_below_50"].append(errors)
    return grouped_errors(groups), {phase: sign_summary(values) for phase, values in bounds.items()}


def tandem_control(observations):
    groups = defaultdict(list)
    for row in read_csv("shadow_tandem_pass_pairs"):
        if (row["roof_type"], row["selected_for_summary"], row["both_resolved"]) != ("EFR", "True", "True"):
            continue
        before, after = (observations[row[field],] for field in ("a_observation_id", "other_observation_id"))
        if (before["candidate_id"] != row["candidate_id"] or after["candidate_id"] != row["candidate_id"]
                or before["version_id"] != row["a_version"] or after["version_id"] != row["other_version"]):
            raise ValueError(f"Tandem control uses another observation for {row['candidate_id']}")
        errors = (float(row["other_minus_a_gap_m"]),
                  float(after["box_response_gap_m"])-float(before["box_response_gap_m"]))
        groups["all"].append(errors)
        groups[row["other_sensor"]].append(errors)
    return grouped_errors(groups)


def validation_sampling():
    observations = one_per_key(read_csv("weekly_shadow_validation_observations"), "geometry_mode", "observation_id")
    by_mode = defaultdict(list)
    for row in read_csv("weekly_shadow_validation_pairs"):
        if row["roof_type"] != "EFR" or row["both_shadow_diagnostics_pass"] != "True":
            continue
        mode = row["geometry_mode"]
        before, after = (observations[mode, row[f"{side}_observation_id"]] for side in ("earlier", "later"))
        if before["candidate_id"] != row["candidate_id"] or after["candidate_id"] != row["candidate_id"]:
            raise ValueError(f"Validation pair uses another observation for {row['candidate_id']}")
        by_mode[mode].append((difference(before, after), float(row["fraction_of_eia_interval_observed"])))
    return {mode: dict(**sign_summary([b for b, _ in rows]),
                       median_fraction_of_eia_interval=median(f for _, f in rows),
                       pairs_observing_less_than_half_the_eia_interval=sum(f < .5 for _, f in rows))
            for mode, rows in sorted(by_mode.items())}


def main():
    rows = read_csv("shadow_change_control_estimates")
    fixed, bounds = fixed_roof_control(one_per_key(rows, "candidate_id", "version_id"))
    report = dict(
        lidar_controls=lidar_control(), fixed_roof_controls=fixed, fixed_roof_change_ranges=bounds,
        tandem_controls=tandem_control(one_per_key(rows, "observation_id")),
        validation_physical_sampling=validation_sampling(),
        interpretation="Box and ESA response estimates are compared on identical saved observations. "
                       "The resolved groups are selected by the existing ESA diagnostics. "
                       "Offset and solar groups are descriptive. "
                       "Near-optimal ranges describe model loss. Their coverage is checked against the fixed-roof controls. "
                       "Tandem observations are engineering controls. No EIA or market outcomes are used.",
    )
    write(DERIVED / "shadow_precision_audit.json", json.dumps(report, indent=2).encode() + b"\n")
    print(json.dumps(report["fixed_roof_change_ranges"], indent=2), flush=True)


if __name__ == "__main__":
    main()
