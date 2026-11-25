"""Compare the same optical control pairs with separate and registered shared centres."""
import json
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import lru_cache

import numpy as np

from audit_shadow_precision import describe
from audit_shadow_registration_pixels import index
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, RESOLVED, export
from collect_public import write
from fit_shadow_geometry import (bearing_vector, gap_trials, inputs, pixel_grid, read_only,
                                 response_profiles, shared_centre_minimum, shifted_grid)

CONTROL_DB = DERIVED / "shadow_change_controls.sqlite"


def source_pairs():
    sources = index("sources", "source_id")
    pairs = index("scene_pairs", "scene_pair_id")
    for row in read_csv("shadow_registration_control_offsets"):
        if row["registration_status"] != "checked_offset":
            continue
        pair = pairs[row["scene_pair_id"]]
        versions = tuple(sources[pair[side + "_source_id"]]["version_id"] for side in ("earlier", "later"))
        yield row, versions


def common_centre(first, last):
    shifts, earlier_gaps, earlier_loss = first
    later_shifts, later_gaps, later_loss = last
    if shifts != later_shifts:
        raise ValueError("Centre trials must match between dates")
    optimum, before, after, score = shared_centre_minimum(earlier_loss, later_loss)
    return float(earlier_gaps[before]), float(later_gaps[after]), *shifts[optimum], score


def refit(pairs):
    with closing(read_only(CONTROL_DB)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
    observed = {(r["candidate_id"], r["version_id"]): r for r in observations}
    saved = {(r["candidate_id"], r["version_id"]): r for r in read_csv("shadow_change_control_estimates")}

    @lru_cache(maxsize=2000)
    def profile(key, east, north):
        observation, old = observed[key], saved[key]
        if int(old["observation_id"]) != observation["observation_id"]:
            raise ValueError(f"The saved baseline uses different pixels: {key}")
        with closing(read_only(CONTROL_DB)) as db:
            grid = shifted_grid(pixel_grid(db, observation), east, north)
        shell = float(old["shell_above_ground_m"]) if old["shell_above_ground_m"] else None
        sun = bearing_vector(observation["sun_zenith_deg"], observation["sun_azimuth_grid_deg"])
        view = bearing_vector(observation["red_view_zenith_deg"], observation["red_view_azimuth_grid_deg"])
        gaps = gap_trials(shell, observation["roof_type"])
        shifts, loss, _, _ = response_profiles(grid, observation["radius_m"], shell if shell and shell >= 5 else 0,
                                               gaps, sun, view, kernels[observation["platform"]], reversed_sun=False)
        if east == north == 0:
            best = int(np.unravel_index(np.argmin(loss), loss.shape)[1])
            if float(gaps[best]) != float(old["estimated_gap_m"]):
                raise ValueError(f"The unshifted profile differs from the saved baseline: {key}")
        return shifts, gaps, loss/grid[-1].sum()

    def solve(item):
        row, versions = item
        keys = tuple((row["candidate_id"], v) for v in versions)
        first, last = (saved[key] for key in keys)
        prior_first, prior_last = (observed[key] for key in keys)
        geometry = ("center_east_m", "center_north_m", "radius_m", "geometry_basis", "roof_type")
        if any(prior_first[k] != prior_last[k] for k in geometry):
            raise ValueError(f"The two observations use different tank geometry: {keys}")
        zero = common_centre(*(profile(key, 0, 0) for key in keys))
        east, north = (float(row[f"later_minus_earlier_{axis}_m"]) for axis in ("east", "north"))
        registered = common_centre(profile(keys[0], -east/2, -north/2), profile(keys[1], east/2, north/2))
        return dict(row, roof_type=first["roof_type"], earlier_version=versions[0], later_version=versions[1],
            baseline_earlier_fit_status=first["fit_status"], baseline_later_fit_status=last["fit_status"],
            baseline_both_resolved=first["fit_status"] == last["fit_status"] == RESOLVED,
            separate_change_m=float(first["estimated_gap_m"])-float(last["estimated_gap_m"]),
            shared_unregistered_change_m=zero[0]-zero[1], shared_registered_change_m=registered[0]-registered[1],
            unregistered_centre_east_m=zero[2], unregistered_centre_north_m=zero[3],
            registered_centre_east_m=registered[2], registered_centre_north_m=registered[3],
            unregistered_sum_weighted_residual_variance=zero[4], registered_sum_weighted_residual_variance=registered[4])

    with ThreadPoolExecutor(max_workers=3) as pool:
        for number, row in enumerate(pool.map(solve, pairs), 1):
            if number % 100 == 0 or number == len(pairs):
                print(f"Refit {number}/{len(pairs)} matched optical controls", flush=True)
            yield row


def main():
    rows = list(refit(list(source_pairs())))
    groups = defaultdict(list)
    for row in rows:
        label = row["comparison"] + ("_EFR" if row["comparison"] == "tandem" and row["roof_type"] == "EFR" else "")
        groups[label].append(row)
        if row["baseline_both_resolved"]:
            groups[label + "_previously_resolved"].append(row)
        if float(row["later_minus_earlier_east_m"]) or float(row["later_minus_earlier_north_m"]):
            groups[label + "_nonzero_offset"].append(row)
    fields = ("separate_change_m", "shared_unregistered_change_m", "shared_registered_change_m")
    comparison = {name: {k: describe([r[k] for r in sample]) for k in fields} for name, sample in sorted(groups.items())}
    coverage = {name: dict(tanks=len({r["candidate_id"] for r in sample}),
                           scene_pairs=len({r["scene_pair_id"] for r in sample}),
                           dates=len({r["week_or_control_day"] for r in sample}))
                for name, sample in sorted(groups.items())}
    export("shadow_registered_control_fit", rows)
    report = dict(compared_pairs=len(rows), coverage=coverage, methods=comparison,
        interpretation="All three methods use the same pixels, angles, tank geometry and ESA B04 response. "
        "Sample inclusion is based on exterior registration availability. Fixed roofs are a zero-change control. "
        "Tandem EFR estimates diagnose repeatability between closely timed captures. No EIA or price outcomes are used.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / "shadow_registered_control_fit_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
