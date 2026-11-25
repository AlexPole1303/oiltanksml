"""Refit mixed sensor pairs at a shared centre using checked exterior offsets."""

import json
import math
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime
from functools import lru_cache

import numpy as np
from pyproj import Transformer

from build_cushing_landsat7_mixed_inputs import MIXED
from build_registered_shadow_development import registered_pair
from build_sentinel2_angles import ground_coordinates
from build_terminal_development_measurements import BARREL_M3
from collect_cushing_landsat7 import DERIVED, export, legacy_geometry, rows, save_json
from fit_shadow_geometry import bearing_vector, fit_response, gap_trials, inputs, pixel_grid, read_only, response_profiles
from measure_cushing_landsat7 import CENTRAL_FWHM, FWHM_M, PIXEL_M, STEP_M, native_chip, native_raster, response_kernel

OFFSETS = (-6, -3, 0, 3, 6)
CENTRES = ((0, 0), *((x, y) for x in OFFSETS for y in OFFSETS if x or y))
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
CRS = "EPSG:32614"
TO_GEOGRAPHIC = Transformer.from_crs(CRS, 4326, always_xy=True)


def pair_key(r):
    return r["week_ending"], r["candidate_id"], r["landsat_product_id"], r["sentinel_version_id"]


def offset_key(r):
    return r["week_or_control_day"], r["candidate_id"], r["earlier_version_id"], r["later_version_id"]


@lru_cache(maxsize=4)
def sentinel_packet(source):
    name = "weekly_shadow_development.sqlite" if source == "core_terminal_legacy_observations.csv" else source.replace("_observations.csv", ".sqlite")
    path = DERIVED / name
    with closing(read_only(path)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
        scenes = {r["scene_id"]: (r["scene_item_id"], r["crs"]) for r in db.execute("SELECT scene_id, scene_item_id, crs FROM scene")}
    return path, {r["observation_id"]: r for r in observations}, kernels, scenes


def sentinel_chip(pair, geometry):
    path, observations, kernels, scenes = sentinel_packet(pair["sentinel_observation_source"])
    row = observations[int(pair["sentinel_observation_id"])]
    canonical, crs = scenes[row["scene_id"]]
    if (row["candidate_id"], row["version_id"], canonical, crs, datetime.fromisoformat(row["captured_at_utc"])) != (
            pair["candidate_id"], pair["sentinel_version_id"], pair["sentinel_canonical_id"], CRS,
            datetime.fromisoformat(pair["sentinel_captured_at_utc"])):
        raise ValueError(f"Saved B04 observation identity differs for {pair_key(pair)}")
    east, north, radius, shell = geometry
    row = dict(row, center_east_m=east, center_north_m=north, radius_m=radius)
    with closing(read_only(path)) as db:
        grid = pixel_grid(db, row)
    sun = bearing_vector(row["sun_zenith_deg"], row["sun_azimuth_grid_deg"])
    view = bearing_vector(row["red_view_zenith_deg"], row["red_view_azimuth_grid_deg"])
    kernel = kernels[row["platform"]]
    fit = fit_response(grid, radius, shell, gap_trials(shell, "EFR"), sun, view, kernel, reversed_sun=True)
    if len(grid[-2]) != int(pair["sentinel_fitted_pixels"]) or fit[0] != float(pair["sentinel_gap_m"]):
        raise ValueError(f"Saved B04 pixels or height do not reproduce for {pair_key(pair)}")
    return (grid, radius, shell, sun, view), kernel


@lru_cache(maxsize=6)
def pan_packet(product):
    item, data = native_raster(product)
    return item, data, {r["candidate_id"]: r for r in item["roofs"]}


def landsat_chip(pair, geometry):
    item, data, roofs = pan_packet(pair["landsat_product_id"])
    roof = roofs[pair["candidate_id"]]
    east, north, radius, shell = geometry
    if (item["native_packet"] != pair["landsat_packet"] or not roof["angle_valid"]
        or not np.allclose((east, north, radius), [roof[k] for k in ("center_east_m", "center_north_m", "radius_m")], atol=1e-8, rtol=0)):
        raise ValueError(f"Saved B8 observation identity differs for {pair_key(pair)}")
    bearing = ground_coordinates(CRS, (TO_GEOGRAPHIC.transform(east, north),))[2][0]
    sun = bearing_vector(roof["sun_zenith_deg"], roof["sun_azimuth_deg"]+bearing)
    view = bearing_vector(roof["view_zenith_deg"], roof["view_azimuth_deg"]+bearing)
    raw, _, _ = native_chip(data, roof, (east, north))
    signal = (raw[-2]*item["b8_reflectance_mult"] + item["b8_reflectance_add"]) / math.cos(math.radians(roof["sun_zenith_deg"]))
    if len(signal) != int(pair["landsat_fitted_pixels"]):
        raise ValueError(f"Saved B8 pixels do not reproduce for {pair_key(pair)}")
    return (*raw[:-2], signal, raw[-1]), radius, shell, sun, view


def profiles(chip, kernel, shifts, pixel_m, step_m, subcells):
    grid, radius, shell, sun, view = chip
    gaps = gap_trials(shell, "EFR")
    points = tuple(dict.fromkeys((x+e, y+n) for e,n in shifts for x,y in CENTRES))
    order, loss, _, placebo = response_profiles(grid, radius, shell, gaps, sun, view, kernel,
        reversed_sun=True, pixel_m=pixel_m, step_m=step_m, subcell_samples=subcells, centre_offsets=points)
    lookup = {point: i for i, point in enumerate(order)}
    for e, n in shifts:
        ids = [lookup[x+e, y+n] for x,y in CENTRES]
        yield CENTRES, gaps, loss[ids], placebo[ids], len(grid[-1]), float(grid[-1].sum())


def solve(pair, offset, g, kernels):
    e, n = (float(offset[f"later_minus_earlier_{k}_m"]) for k in ("east", "north"))
    residual = max(2., float(offset["check_median_residual_m"]), float(offset["fit_median_residual_m"]))
    shifts = [(e, n), *((e+residual*x, n+residual*y) for x,y in DIRECTIONS)]
    geometry = (float(g["center_east_m"]), float(g["center_north_m"]), float(g["diameter_m"])/2,
                float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"]))
    pan, (sentinel, kernel) = landsat_chip(pair, geometry), sentinel_chip(pair, geometry)
    b04 = list(profiles(sentinel, kernel, [(x/2, y/2) for x,y in shifts], 10, 2, 1))
    first_pan = pair["route"] == "B8_RT_to_B04"
    fits = []
    for width, response in kernels.items():
        b8 = profiles(pan, response, [(-x/2, -y/2) for x,y in shifts], PIXEL_M, STEP_M, 3)
        for i, (a, b) in enumerate(zip(b8, b04, strict=True)):
            r = registered_pair(a, b) if first_pan else registered_pair(b, a)
            interior = max(abs(r[f"registered_centre_{axis}_m"]) for axis in ("east", "north")) < max(OFFSETS)
            fits.append(dict(r, effective_fwhm_m=width, offset_scenario=i, fit_pass=interior and r["registered_both_shadow_diagnostics_pass"]))
    return residual, fits


def main():
    pairs = [r for r in rows(DERIVED / f"{MIXED}_candidate_pairs.csv") if r["landsat_native_height_diagnostic_pass"] == "True"]
    offsets = {offset_key(r): r for r in rows(DERIVED / f"{MIXED}_registration_offsets.csv")}
    if len(offsets) != len(pairs) or offsets.keys() != {pair_key(r) for r in pairs}:
        raise ValueError("Mixed pairs differ from the exterior registration ledger")
    geometry = legacy_geometry()
    kernels = {width: response_kernel(width) for width in FWHM_M}
    results, cases = [], []
    for p in pairs:
        offset = offsets[pair_key(p)]
        checked = offset["registration_status"] == "checked_offset" and offset["red_context_before_cutoff"] == "True"
        r = dict(p, registration_status=offset["registration_status"], registration_before_cutoff=offset["red_context_before_cutoff"] == "True",
            checked_exterior_offset=checked, offset_east_m=offset["later_minus_earlier_east_m"], offset_north_m=offset["later_minus_earlier_north_m"],
            stress_radius_m=None, joint_central_fit_pass=False, all_scenarios_pass=False, central_capture_change_bbl=None,
            stress_capture_change_min_bbl=None, stress_capture_change_max_bbl=None, robust_nonzero_change=False)
        if checked:
            residual, fits = solve(p, offset, geometry[p["candidate_id"]], kernels)
            central = next(v for v in fits if v["effective_fwhm_m"] == CENTRAL_FWHM and v["offset_scenario"] == 0)
            scale = float(p["area_m2"])/BARREL_M3
            stable = all(v["fit_pass"] for v in fits)
            low = min(v["registered_loss_range_min_m"] for v in fits)*scale if stable else None
            high = max(v["registered_loss_range_max_m"] for v in fits)*scale if stable else None
            r.update(stress_radius_m=residual, joint_central_fit_pass=central["fit_pass"], all_scenarios_pass=stable,
                     central_capture_change_bbl=central["registered_change_m"]*scale if central["fit_pass"] else None,
                     stress_capture_change_min_bbl=low, stress_capture_change_max_bbl=high,
                     robust_nonzero_change=stable and (low>0 or high<0))
            fields = ("week_ending", "candidate_id", "landsat_product_id", "sentinel_version_id", "route")
            cases.extend({k:p[k] for k in fields} | fit for fit in fits)
        results.append(r)
    export(f"{MIXED}_pair_fits", results)
    if cases:
        export(f"{MIXED}_sensitivity_cases", cases)
    robust = [r for r in results if r["robust_nonzero_change"]]
    report = dict(independent_height_pairs=len(pairs), registration_status=dict(Counter(r["registration_status"] for r in results)),
        checked_exterior_offsets_before_cutoff=sum(r["checked_exterior_offset"] for r in results),
        joint_central_height_pairs=sum(r["joint_central_fit_pass"] for r in results), sensitivity_cases=len(cases),
        pairs_passing_all_offsets_and_response_widths=sum(r["all_scenarios_pass"] for r in results),
        robust_nonzero_capture_change_pairs=len(robust), unique_robust_roof_weeks=len({(r["week_ending"],r["candidate_id"]) for r in robust}),
        weeks_with_robust_pairs=len({r["week_ending"] for r in robust}),
        weeks_with_robust_pairs_without_any_b04_measurement=len({r["week_ending"] for r in robust if r["b04_baseline_week_tanks"]=="0"}),
        note="Exploratory inter-capture geometric changes. Cross-sensor level bias and reporting-clock alignment still apply.")
    save_json(DERIVED / f"{MIXED}_fit_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
