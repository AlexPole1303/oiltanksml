"""Trace the 25C-6 shadow fit to native pixels and model assumptions."""

import hashlib
import json
import math
import sqlite3
from contextlib import closing

import numpy as np
import rasterio
from rasterio.transform import Affine

from audit_plains_25c6_detector_angles import PLAINS, STEM as ANGLES
from audit_sentinel2_alternate_registration import packet_inputs
from build_cross_sensor_shadow import PAN_PIXEL, PAN_SHIFTS, PAN_STEP, pan_grid, pan_psf, published_lsf
from build_landsat_c2_b8_measurements import native_pan_samples
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED, read_only
from collect_public import ROOT, write
from collect_landsat import catalog
from fit_shadow_geometry import (PIXEL_M, SHIFTS_M, STEP_M, bearing_vector, gap_fit_status, gap_trials, pixel_grid, response_profiles)

STEM = f"{PLAINS}_shadow_support"


def sentinel_inputs(packet=f"{PLAINS}_sentinel_native.sqlite", stem=ANGLES, fit_suffix="single_scene_fits", *, subset=False):
    path, observations, _, kernels = packet_inputs(packet)
    geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry")}
    angle_suffix = "sentinel_angles" if subset else "angle_nodes"
    angles = {(r["candidate_id"], r["version_id"]): r for r in read(f"{stem}_{angle_suffix}") if r["band"] == "B04"}
    baselines = {(r["candidate_id"], r["version_id"]): r for r in read(f"{stem}_{fit_suffix}") if r["method"] == "red"}
    if angles.keys() != baselines.keys():
        raise ValueError(f"Angle and fit identities differ for {stem}")
    with closing(read_only(path)) as db:
        db.row_factory = sqlite3.Row
        for o in observations.values():
            key = o["candidate_id"], o["version_id"]
            if subset and key not in baselines:
                continue
            a, g = angles[key], geometry[key[0]]
            row = dict(o, center_east_m=float(g["center_east_m"]), center_north_m=float(g["center_north_m"]),
                       radius_m=float(g["diameter_m"])/2, red_detector_id=int(a["detector_id"]))
            shell = float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"])
            sun = bearing_vector(o["sun_zenith_deg"], o["sun_azimuth_grid_deg"])
            view = bearing_vector(float(a["zenith_deg"]), float(a["azimuth_true_deg"])+o["true_north_grid_bearing_deg"])
            yield dict(key=key, band="B04", row=row, capture=o["captured_at_utc"], shell=shell, grid=pixel_grid(db, row), sun=sun,
                       view=view, kernel=kernels[o["platform"]], pixel=PIXEL_M, step=STEP_M, shifts=SHIFTS_M, baseline=baselines[key],
                       packet=path.name, source_type="sentinel2_level1c")


def landsat_inputs(stem=PLAINS, year=2026, *, baselines=None):
    path = DERIVED/f"{stem}_landsat_native.sqlite"
    sources = {r["product_id"]: r for r in read(f"{stem}_landsat_sources")}
    if baselines is None:
        baselines = {(r["candidate_id"], r["product_id"]): r for r in read(f"{stem}_landsat_fits")}
    items = {r["id"]: r for r in catalog(year, "usgs") if r["id"] in sources}
    lsf = published_lsf()
    with closing(read_only(path)) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute("SELECT * FROM roof_window JOIN scene USING (product_id) ORDER BY pan_observation_id"):
            source = sources[row["product_id"]]
            folder = (ROOT / source["reflectance_path"]).parent
            metadata = json.loads((folder / "MTL.json").read_bytes())["LANDSAT_METADATA_FILE"]
            # Read the exact native window origin from the matched LandsatLook asset grid.
            item = items[row["product_id"]]
            official = Affine(*item["assets"]["pan"]["proj:transform"][:6])
            source_path = ROOT/source["reflectance_path"]
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != source["reflectance_sha256"]:
                raise ValueError("Stored B8 reflectance has changed")
            with rasterio.open(source_path) as raster:
                x, y = (~official)*(raster.transform.c, raster.transform.f)
                if max(abs(x-round(x)), abs(y-round(y))) > 1e-8 or raster.crs.to_string() != row["crs"]:
                    raise ValueError("Landsat reflectance does not align with the official B8 grid")
                raw, problem = native_pan_samples(raster, db, row, (round(y), round(x)),
                                                  metadata["IMAGE_ATTRIBUTES"]["SATURATION_BAND_8"], apparent_reflectance_scale=.0001)
            if problem:
                raise ValueError(f"Missing native B8 evidence for {row['candidate_id']}: {problem}")
            grid, _ = pan_grid(row, raw)
            north = row["true_north_grid_bearing_deg"]
            key = row["candidate_id"], row["product_id"]
            yield dict(key=key, band="B8", row=row, capture=row["captured_at_utc"], shell=row["shell_above_ground_m"], grid=grid,
                       sun=bearing_vector(row["solar_zenith_deg"], row["solar_azimuth_true_deg"]+north),
                       view=bearing_vector(row["view_zenith_deg"], row["view_azimuth_true_deg"]+north),
                       kernel=pan_psf(lsf, row["track_bearing_deg"]), pixel=PAN_PIXEL, step=PAN_STEP, shifts=PAN_SHIFTS,
                       baseline=baselines[key], packet=path.name, source_type=source["status"]+"_esri_reflectance")


def restrict(grid, keep):
    return (*grid[:3], *(a[keep] for a in grid[3:]))


def coordinates(source):
    _, _, origin, row, col, _, _ = source["grid"]
    x, y = origin[0] + source["pixel"]*col, origin[1] - source["pixel"]*row
    return x, y, np.hypot(x, y)


def scores(source, grid, shell, radius, *, fine=False, subdivisions=1, exterior=True, pixel_losses=False, gap_limit=None):
    gaps = gap_trials(shell if gap_limit is None else min(shell, gap_limit), source["row"]["roof_type"])
    offsets = [(float(x), float(y)) for x in range(-6, 7, 2) for y in range(-6, 7, 2)] if fine else None
    shifts, loss, _, reversed_loss = response_profiles(grid, radius, shell, gaps, source["sun"], source["view"], source["kernel"],
        reversed_sun=True, pixel_m=source["pixel"], step_m=source["step"], shifts=source["shifts"], centre_offsets=offsets,
        subcell_samples=subdivisions, include_exterior_shadow=exterior, pixel_losses=pixel_losses)
    return gaps, shifts, loss, reversed_loss


def summarize(gaps, shifts, loss, reverse, grid, nominal_gap):
    i, j = np.unravel_index(loss.argmin(), loss.shape)
    best = float(loss[i, j])
    weights = float(grid[-1].sum())
    count = len(grid[-1])
    near = gaps[loss.min(axis=0) <= best*(1+1/max(1, count-6)) + 1e-10]
    status = gap_fit_status(gaps, gaps[j], near.min(), near.max(), math.sqrt(best/weights),
                            math.sqrt(loss[:, 0].min()/weights), math.sqrt(reverse.min()/weights))
    nominal_j = int(abs(gaps-nominal_gap).argmin())
    nominal_i = int(loss[:, nominal_j].argmin())
    ratio = float(loss[nominal_i, nominal_j]/best) if best else None
    return dict(fit_status=status, gap_m=float(gaps[j]), near_min_m=float(near.min()), near_max_m=float(near.max()),
                shift_east_m=shifts[i][0], shift_north_m=shifts[i][1], pixels=count, min_loss=best, rmse=math.sqrt(best/weights),
                model_maximum_gap_m=float(gaps[-1]), nearest_nominal_gap_m=float(gaps[nominal_j]), nominal_loss_to_best_ratio=ratio), (i,j,nominal_i,nominal_j)


def compare(source, nominal_gap, nominal_shell, nominal_radius):
    radius, shell, grid = source["row"]["radius_m"], source["shell"], source["grid"]
    x, y, radial = coordinates(source)
    trials = (
        ("baseline", grid, shell, radius, False, 1, True),
        ("subcell4", grid, shell, radius, False, 4, True),
        ("fine_centre", grid, shell, radius, True, 1, True),
        ("fine_centre_subcell4", grid, shell, radius, True, 4, True),
        ("operator_shell_only", grid, nominal_shell, radius, True, 4, True),
        ("operator_diameter_only", grid, shell, nominal_radius, True, 4, True),
        ("operator_shell_and_diameter", grid, nominal_shell, nominal_radius, True, 4, True),
        ("omit_exterior_shadow_term", grid, shell, radius, False, 1, False),
        ("inside_rim_pixel_centres", restrict(grid, radial <= radius), shell, radius, False, 1, False),
    )
    identity = dict(candidate_id=source["key"][0], version_id=source["key"][1], band=source["band"],
                    capture_at_utc=source["capture"], source_type=source["source_type"], native_packet=source["packet"])
    fits, pixels = [], []
    for name, subset, h, r, fine, n, exterior in trials:
        if len(subset[-1]) < 12:
            raise ValueError(f"Insufficient pixels for {source['key']}/{name}")
        save_pixels = name == "baseline"
        gaps, shifts, loss, reverse = scores(source, subset, h, r, fine=fine, subdivisions=n, exterior=exterior, pixel_losses=save_pixels)
        aggregate, control = (loss.sum(axis=-1), reverse.sum(axis=-1)) if save_pixels else (loss, reverse)
        fit, (i, j, k, l) = summarize(gaps, shifts, aggregate, control, subset, nominal_gap)
        if save_pixels:
            old = source["baseline"]
            if (fit["gap_m"], fit["fit_status"]) != (float(old.get("gap_m") or old["estimated_gap_m"]), old["fit_status"]):
                raise ValueError(f"Baseline replay changed for {source['key']}")
            dx, dy = shifts[i]
            inside = np.hypot(x-dx, y-dy) <= r
            delta = loss[k,l] - loss[i,j]
            if not math.isclose(float(delta.sum()), aggregate[k,l]-aggregate[i,j], abs_tol=1e-12):
                raise ValueError("Pixel evidence does not sum to the reported loss difference")
            for q, (py, px) in enumerate(zip(grid[3], grid[4], strict=True)):
                pixels.append(dict(**identity, window_row=int(py), window_column=int(px), east_of_lidar_centre_m=float(x[q]),
                                   north_of_lidar_centre_m=float(y[q]), inside_fitted_rim=bool(inside[q]),
                                   source_reflectance=float(grid[-2][q]), pixel_weight=float(grid[-1][q]),
                                   weighted_best_squared_residual=float(loss[i,j,q]), weighted_nominal_squared_residual=float(loss[k,l,q]),
                                   nominal_minus_best_loss=float(delta[q])))
        fits.append(dict(**identity, test=name, shell_height_m=h, radius_m=r, **fit))
    # Refit after removing each native pixel in turn.
    loo = []
    for omitted in range(len(grid[-1])):
        keep = np.arange(len(grid[-1])) != omitted
        gaps, shifts, omitted_loss, _ = scores(source, restrict(grid, keep), shell, radius)
        i, j = np.unravel_index(omitted_loss.argmin(), omitted_loss.shape)
        loo.append(dict(**identity, removed_window_row=int(grid[3][omitted]), removed_window_column=int(grid[4][omitted]),
                        refit_gap_m=float(gaps[j]), refit_shift_east_m=shifts[i][0], refit_shift_north_m=shifts[i][1]))
    return fits, pixels, loo


def main():
    ground, = read(f"{PLAINS}_ground_record")
    shell = float(ground["operator_shell_height_ft"])*.3048
    nominal_gap = shell-float(ground["liquid_level_m"])
    radius = float(ground["operator_diameter_ft"])*.3048/2
    rows, pixels, loo = [], [], []
    for source in (*sentinel_inputs(), *landsat_inputs()):
        fit, values, omitted = compare(source, nominal_gap, shell, radius)
        rows.extend(fit)
        pixels.extend(values)
        loo.extend(omitted)
        print(source["key"], fit[0]["gap_m"], flush=True)
    grouped = {}
    for source in [r for r in rows if r["test"] == "baseline"]:
        key = source["candidate_id"], source["version_id"]
        p = [r for r in pixels if (r["candidate_id"], r["version_id"]) == key]
        l = [r["refit_gap_m"] for r in loo if (r["candidate_id"], r["version_id"]) == key]
        inner = sum(r["nominal_minus_best_loss"] for r in p if r["inside_fitted_rim"])
        total = sum(r["nominal_minus_best_loss"] for r in p)
        alternatives = [r for r in rows if (r["candidate_id"], r["version_id"]) == key and r["test"] != "baseline"]
        grouped[key] = {k: source[k] for k in ("candidate_id", "version_id", "band", "capture_at_utc", "fit_status", "gap_m", "pixels")} | dict(
            nearest_nominal_gap_m=source["nearest_nominal_gap_m"],
            interior_loss_difference=inner, exterior_loss_difference=total-inner, total_loss_difference=total,
            interior_fraction_of_loss_difference=inner/total if total else None,
            leave_one_pixel_gap_min_m=min(l), leave_one_pixel_gap_max_m=max(l), alternative_gap_min_m=min(r["gap_m"] for r in alternatives),
            alternative_gap_max_m=max(r["gap_m"] for r in alternatives), alternatives_passing=sum(r["fit_status"] == RESOLVED for r in alternatives))
    for name, records in (("tests", rows), ("pixels", pixels), ("leave_one_pixel", loo), ("observations", list(grouped.values()))):
        export(f"{STEM}_{name}", records)
    report = dict(observations=len(grouped), sensitivity_fits=len(rows), native_pixel_rows=len(pixels), leave_one_pixel_refits=len(loo),
                  conditional_nominal_gap_m=nominal_gap,
                  passed_by_test={test: sum(r["test"] == test and r["fit_status"] == RESOLVED for r in rows)
                                  for test in dict.fromkeys(r["test"] for r in rows)})
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
