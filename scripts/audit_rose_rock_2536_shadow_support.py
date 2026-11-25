"""Reprofile the PHMSA-linked Rose Rock tank on the original Sentinel-2 pixels."""

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from itertools import product
from zoneinfo import ZoneInfo

import numpy as np

from audit_sentinel2_alternate_registration import packet_inputs
from build_cushing_nrc_rose_rock_landsat import STEM as LANDSAT
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED, read_only
from check_weekly_multiband_shadow import METHODS, detector_geometry, matched_signals, spectral_kernels
from collect_public import ROOT, write
from collect_s2_spectral_detector_masks import BANDS
from fit_shadow_geometry import bearing_vector, gap_fit_status, gap_trials, pixel_grid, response_profiles, shifted_grid

STEM = "cushing_rose_rock_2536_shadow_support"
INPUT = f"{LANDSAT}_lidar_2022"
FINE_OFFSETS = list(product(range(-6, 7, 2), repeat=2))
METHOD_BAND = {"red": "B04", "nir": "B08"}
LOCAL = ZoneInfo("America/Chicago")


def source_inputs(candidate):
    packet, observations, scenes, _ = packet_inputs(f"{INPUT}_sentinel2_native.sqlite")
    archived = {r["version_id"]: r for r in read(f"{INPUT}_spectral_fits") if r["candidate_id"] == candidate}
    geometry, = [r for r in read("cushing_modern_lidar_optical_registry") if r["candidate_id"] == candidate]
    ground = float(geometry["prior_ground_z_p50_native"])
    masks = {}
    for r in read(f"{INPUT}_spectral_masks"):
        path = ROOT/r["local_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["source_sha256"]:
            raise ValueError(f"Original detector footprint differs from its archive checksum: {path}")
        masks[r["version_id"], r["band"]] = r["local_file"]
    centre = tuple(float(geometry[f"center_{axis}_m"]) for axis in ("east", "north"))
    radius = float(geometry["diameter_m"])/2
    shell = float(geometry["prior_rim_z_p95_native"])-ground
    with closing(read_only(packet)) as db:
        db.row_factory = sqlite3.Row
        kernels = spectral_kernels(db)
        for original in observations.values():
            if original["candidate_id"] != candidate:
                continue
            version = original["version_id"]
            row = dict(original, center_east_m=centre[0], center_north_m=centre[1], radius_m=radius)
            native = pixel_grid(db, row)
            signals, keep = matched_signals(db, row, native)
            status, views, detectors = detector_geometry(row, native, keep, scenes[row["scene_id"]],
                                                           tuple(masks[version, band] for band in BANDS))
            old = archived[version]
            if views is None or int(keep.sum()) != int(old["common_pixels"]):
                raise ValueError(f"Detector or common-pixel checks changed for {version}: {status}")
            height, width, origin, y, x, _, weight = native
            yield dict(row=row, base=(height, width, origin, y[keep], x[keep]), weights=weight[keep], signals=signals,
                       kernels={b: kernels[row["platform"], b] for b in signals}, views=views, detectors=detectors,
                       sun=bearing_vector(row["sun_zenith_deg"], row["sun_azimuth_grid_deg"]),
                       centre=centre, shell=shell, ground=ground, radius=radius, old=old, packet=packet.name)


def scenarios(source, rings):
    origin = (*source["centre"], source["radius"], source["shell"])
    yield "published_geometry", origin, 1, None, True
    yield "subcell_original_geometry", origin, 4, None, True
    yield "fine_alignment_original_geometry", origin, 4, FINE_OFFSETS, True
    yield "no_exterior_shadow_original_geometry", origin, 4, FINE_OFFSETS, False
    for r in rings:
        shape = (float(r["ring_east_m"]), float(r["ring_north_m"]), float(r["ring_diameter_m"])/2,
                 float(r["ring_z_m"])-source["ground"])
        yield f"lidar_ring_{r['ring_rank']}", shape, 4, FINE_OFFSETS, True


def profiles(source, shape, subdivisions, offsets, exterior, *, keep=None, pixel_losses=False):
    east, north, radius, shell = shape
    dx, dy = east-source["centre"][0], north-source["centre"][1]
    h, w, origin, y, x = source["base"]
    weight = source["weights"]
    if keep is None:
        keep = np.ones(len(weight), dtype=bool)
    weights = weight[keep]
    gaps = gap_trials(shell, "EFR")
    by_band = {}
    for band, values in source["signals"].items():
        values = values[keep]
        variance = float(np.dot(weights, (values-np.average(values, weights=weights))**2))
        if variance <= 0:
            raise ValueError(f"No spectral variance on {source['row']['version_id']}/{band}")
        grid = shifted_grid((h, w, origin, y[keep], x[keep], values, weights), dx, dy)
        shifts, loss, _, reverse = response_profiles(grid, radius, shell, gaps, source["sun"], source["views"][band],
            source["kernels"][band], reversed_sun=True, centre_offsets=offsets, subcell_samples=subdivisions,
            include_exterior_shadow=exterior, pixel_losses=pixel_losses)
        by_band[band] = loss, reverse, variance
    result = {method: (by_band[band][0]/weights.sum(), by_band[band][1]/weights.sum()) for method, band in METHOD_BAND.items()}
    result["joint"] = tuple(np.mean([row[i]/row[2] for row in by_band.values()], axis=0) for i in range(2))
    return gaps, shifts, result, int(keep.sum())


def evaluate(gaps, shifts, loss, reverse, pixels):
    i, j = np.unravel_index(loss.argmin(), loss.shape)
    error, flat, placebo = math.sqrt(loss[i, j]), math.sqrt(loss[:, 0].min()), math.sqrt(reverse.min())
    near = gaps[loss.min(axis=0) <= loss[i,j]*(1+1/max(1, pixels-6))+1e-10]
    lo, hi = float(near.min()), float(near.max())
    fit = dict(estimated_gap_m=float(gaps[j]), near_min_m=lo, near_max_m=hi, best_shift_east_m=shifts[i][0],
               best_shift_north_m=shifts[i][1], fit_status=gap_fit_status(gaps, gaps[j], lo, hi, error, flat, placebo),
               rms_fit=error, rms_flat=flat, rms_reverse_sun=placebo, fractional_rms_improvement_over_flat=1-error/flat if flat else None)
    return fit, (i, j)


def pixel_support(source, shape, identity):
    gaps, shifts, result, count = profiles(source, shape, 1, None, True, pixel_losses=True)
    _, _, origin, y, x = source["base"]
    east = origin[0]+10*x
    north = origin[1]-10*y
    for method, (loss, reverse) in result.items():
        aggregate = loss.sum(axis=2)
        _, (i, j) = evaluate(gaps, shifts, aggregate, reverse.sum(axis=2), count)
        flat = int(np.argmin(aggregate[:, 0]))
        dx, dy = shifts[i]
        gain = loss[flat, 0]-loss[i, j]
        if not math.isclose(float(gain.sum()), float(aggregate[flat, 0]-aggregate[i,j]), abs_tol=1e-12):
            raise ValueError("Pixel attribution failed to reproduce the aggregate loss difference")
        for q in range(count):
            yield dict(**identity, method=method, native_chip_row=int(y[q]), native_chip_column=int(x[q]),
                       east_of_geometry_centre_m=float(east[q]), north_of_geometry_centre_m=float(north[q]),
                       inside_fitted_circle=bool(math.hypot(east[q]-dx, north[q]-dy) <= source["radius"]),
                       b03_reflectance=float(source["signals"]["B03"][q]), b04_reflectance=float(source["signals"]["B04"][q]),
                       b08_reflectance=float(source["signals"]["B08"][q]), weight=float(source["weights"][q]),
                       weighted_flat_loss=float(loss[flat, 0, q]), weighted_best_loss=float(loss[i,j,q]),
                       flat_minus_best_loss=float(gain[q]))


def sector_checks(source, shape, identity):
    _, _, origin, y, x = source["base"]
    east, north = origin[0]+10*x, origin[1]-10*y
    azimuth = np.mod(np.degrees(np.arctan2(north, east)), 360)
    for sector, lower in zip(("NE", "NW", "SW", "SE"), range(0, 360, 90), strict=True):
        keep = (azimuth < lower) | (azimuth >= lower+90)
        gaps, shifts, methods, count = profiles(source, shape, 4, FINE_OFFSETS, True, keep=keep)
        for method, (loss, reverse) in methods.items():
            fit, _ = evaluate(gaps, shifts, loss, reverse, count)
            yield dict(**identity, excluded_sector=sector, method=method, retained_pixels=count, **fit)


def main():
    link = json.loads((DERIVED/"cushing_phmsa_controls_summary.json").read_text())["rose_rock_inventory_case"]
    candidate = link["preferred_candidate_id"]
    event = datetime.fromisoformat(link["incident_confirmation_utc"])
    rings = [r for r in read("cushing_lidar_shell_rings_flight_lines") if r["candidate_id"] == candidate and r["status"] == "coherent_circle"]
    if len(rings) != 3 or {r["ring_rank"] for r in rings} != {"1", "2", "3"}:
        raise ValueError("The three source LiDAR rings for the linked contour are required")
    rows, curves, pixels, sectors = [], [], [], []
    for s in source_inputs(candidate):
        o, old = s["row"], s["old"]
        stamp = datetime.fromisoformat(o["captured_at_utc"])
        identity = dict(candidate_id=candidate, equipment_id=link["equipment_id"], version_id=o["version_id"],
                        capture_utc=o["captured_at_utc"], native_packet=s["packet"], observation_id=o["observation_id"],
                        signed_hours_from_incident=(stamp-event).total_seconds()/3600,
                        same_local_day=stamp.astimezone(LOCAL).date() == event.astimezone(LOCAL).date())
        incident_day = identity["same_local_day"]
        for name, shape, subdivision, offsets, exterior in scenarios(s, rings):
            gaps, shifts, methods, count = profiles(s, shape, subdivision, offsets, exterior)
            for method, (loss, reverse) in methods.items():
                fit, _ = evaluate(gaps, shifts, loss, reverse, count)
                if name == "published_geometry" and (fit["estimated_gap_m"], fit["fit_status"]) != (
                    float(old[f"{method}_gap_m"]), old[f"{method}_fit_status"]
                ):
                    raise ValueError(f"Original gap or fit status changed for {o['version_id']}/{method}")
                key = dict(**identity, scenario=name, method=method)
                rows.append(dict(**key, native_pixels=count, shell_above_ground_m=shape[3], diameter_m=2*shape[2],
                                 sun_elevation_deg=90-o["sun_zenith_deg"],
                                 projected_direct_shadow_length_m=fit["estimated_gap_m"]*float(np.linalg.norm(s["sun"])),
                                 admitted_absolute_inventory_label=False, **fit))
                for gap, a, b in zip(gaps, loss.min(axis=0), reverse.min(axis=0), strict=True):
                    curves.append(dict(**key, gap_m=float(gap), minimum_normalized_loss=float(a), reverse_sun_minimum_loss=float(b)))
            if incident_day and name == "published_geometry":
                pixels.extend(pixel_support(s, shape, identity))
            if incident_day and name == "fine_alignment_original_geometry":
                sectors.extend(sector_checks(s, shape, identity))
    incident = [r for r in rows if r["same_local_day"]]
    by_scenario = defaultdict(list)
    for row in incident:
        by_scenario[row["scenario"]].append(row)
    result = dict(candidate_id=candidate, equipment_id=link["equipment_id"], source_versions=len({r["version_id"] for r in rows}),
                  fits=len(rows), incident_day_fits=len(incident), incident_day_fit_status=Counter(r["fit_status"] for r in incident),
                  incident_day_both_tiles_passing_scenario_methods=[f"{key}/{method}" for key, group in by_scenario.items() for method in METHODS
                      if sum(r["method"] == method and r["fit_status"] == RESOLVED for r in group) == 2],
                  incident_day_max_fractional_rms_improvement=max(r["fractional_rms_improvement_over_flat"] for r in incident),
                  incident_day_baseline_rms_gain_range=[fn(r["fractional_rms_improvement_over_flat"] for r in incident
                                                          if r["scenario"] == "published_geometry") for fn in (min,max)],
                  incident_day_sector_refits=len(sectors), incident_day_sector_passing_fits=sum(r["fit_status"] == RESOLVED for r in sectors),
                  pixel_support_rows=len(pixels), stock_measurement_time_known=False, admitted_absolute_inventory_label=False)
    common_pixels = defaultdict(dict)
    contributions = defaultdict(list)
    for row in pixels:
        contributions[row["version_id"], row["method"]].append(row)
        if row["method"] == "red":
            cell = row["native_chip_row"], row["native_chip_column"]
            common_pixels[row["version_id"]][cell] = row
    first, second = common_pixels.values()
    if first.keys() != second.keys() or any(any(first[k][axis] != second[k][axis] for axis in (
        "east_of_geometry_centre_m", "north_of_geometry_centre_m")) for k in first):
        raise ValueError("The incident-day tiles do not use the same physical pixel grid")
    result["incident_day_shared_grid_cells"] = len(first)
    result["incident_day_tile_max_absolute_reflectance_difference"] = {band: max(abs(first[k][band]-second[k][band]) for k in first)
        for band in ("b03_reflectance", "b04_reflectance", "b08_reflectance")}
    regional_support = []
    for (version, method), samples in contributions.items():
        total = math.fsum(r["flat_minus_best_loss"] for r in samples)
        interior = math.fsum(r["flat_minus_best_loss"] for r in samples if r["inside_fitted_circle"])
        regional_support.append(dict(version_id=version, method=method, interior_cells=sum(r["inside_fitted_circle"] for r in samples),
                                     all_cells=len(samples), interior_loss_gain=interior, exterior_loss_gain=total-interior,
                                     total_loss_gain=total, interior_fraction_of_total_gain=interior/total if total else None))
    for suffix, table in (("fits", rows), ("loss_curves", curves), ("pixel_support", pixels), ("sector_refits", sectors),
                          ("regional_support", regional_support)):
        export(f"{STEM}_{suffix}", table)
    payload = json.dumps(result, indent=2, allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
