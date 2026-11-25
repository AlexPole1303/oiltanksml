"""Compare B04 and a joint B03/B04/B08 physical fit against dated LiDAR controls."""
import json
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
from statistics import fmean

import numpy as np

from audit_shadow_precision import describe
from build_sentinel2_angles import interpolate, stencil
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, RESOLVED, export
from collect_public import ROOT, write
from collect_s2_l1c import source_metadata_bytes
from fit_shadow_geometry import (CALIBRATION, bearing_vector, gap_fit_status, gap_trials,
                                 pixel_grid, read_only, response_profiles)

BANDS = {"B03": (2, "dn_green", 2), "B04": (3, "dn_red", 4), "B08": (7, "dn_nir", 8)}
METHODS = ("red", "joint", "red_fine_centre")


@lru_cache(maxsize=24)
def tile_metadata(path):
    return ET.fromstring(source_metadata_bytes(ROOT / path))


def viewing_vectors(o, shape):
    root = tile_metadata(o["tile_metadata_path"])
    tile = root.find(".//Tile_Geocoding/Geoposition[@resolution='10']")
    left, top = float(tile.findtext("ULX")), float(tile.findtext("ULY"))
    rows, cols, origin = shape
    west, north = o["center_east_m"] + origin[0], o["center_north_m"] + origin[1]
    east, south = west + 10 * (cols - 1), north - 10 * (rows - 1)
    x = np.array([o["center_east_m"], west, west, east, east])
    y = np.array([o["center_north_m"], north, south, north, south])
    weights = stencil((x - left) / 5000, (top - y) / 5000)
    views = {"B04": bearing_vector(o["red_view_zenith_deg"], o["red_view_azimuth_grid_deg"])}
    for band, (band_id, _, _) in BANDS.items():
        if band == "B04":
            continue
        valid = []
        for grid in root.findall(f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}']"):
            zenith = interpolate(grid, "Zenith", weights)
            azimuth = interpolate(grid, "Azimuth", weights, True)
            defined = np.isfinite(zenith) & np.isfinite(azimuth)
            if defined.any():
                valid.append((defined.all(), zenith[0], azimuth[0]))
        if len(valid) == 1 and valid[0][0]:
            _, zenith, azimuth = valid[0]
            views[band] = bearing_vector(zenith, azimuth + o["true_north_grid_bearing_deg"])
    return views


def matched_signals(db, o, red_grid):
    rows, cols, _, row, col, reflectance, _ = red_grid
    bands = {r["band"]: r for r in db.execute("SELECT * FROM band WHERE scene_id=?", (o["scene_id"],))}
    pixels = db.execute("""SELECT dn_green,dn_red,dn_nir,raster_valid_band_bits FROM pixel
        WHERE observation_id=? ORDER BY tile_row,tile_col""", (o["observation_id"],)).fetchall()
    if len(pixels) != rows * cols:
        raise ValueError("The spectral samples do not align with the B04 chip")
    index = row * cols + col
    signals = {"B04": reflectance}
    valid = np.ones(len(index), dtype=bool)
    for band, (_, channel, bit) in BANDS.items():
        if band == "B04":
            continue
        meta = bands[band]
        dn = np.asarray([pixels[i][channel] for i in index], dtype=float)
        raster_valid = np.array([bool(pixels[i]["raster_valid_band_bits"] & bit) for i in index])
        valid &= raster_valid & np.isfinite(dn) & (dn != meta["nodata_dn"]) & (dn != meta["saturated_dn"])
        signals[band] = dn * meta["reflectance_scale"] + meta["reflectance_offset"]
    return {band: values[valid] for band, values in signals.items()}, valid


def best(gaps, main, reversed_sun, sample_size):
    _, i = np.unravel_index(np.argmin(main), main.shape)
    error = main.min()
    near = gaps[main.min(axis=0) <= error * (1 + 1 / max(1, sample_size - 6)) + 1e-10]
    rmse, flat, placebo = np.sqrt([error, main[:, 0].min(), reversed_sun.min()])
    return float(gaps[i]), gap_fit_status(gaps, gaps[i], near.min(), near.max(), rmse, flat, placebo)


def compare(db, o, reference, kernels):
    grid = pixel_grid(db, o)
    views = viewing_vectors(o, grid[:3])
    if views.keys() != BANDS.keys():
        return "ambiguous_nonred_detector", None
    signals, keep = matched_signals(db, o, grid)
    if keep.sum() < 12:
        return "insufficient_common_pixels", None
    sun = bearing_vector(o["sun_zenith_deg"], o["sun_azimuth_grid_deg"])
    radius, shell = o["radius_m"], float(reference["shell_above_ground_m"] or 0)
    gaps = gap_trials(shell, o["roof_type"])
    rows, cols, origin, row, col, _, weight = grid
    weights = weight[keep]
    profiles = {}
    for band, signal in signals.items():
        variation = float(np.dot(weights, (signal - np.average(signal, weights=weights)) ** 2))
        if variation <= 0 or not np.isfinite(variation):
            return "no_band_variation", None
        _, main, _, reversed_sun = response_profiles((rows, cols, origin, row[keep], col[keep], signal, weights),
            radius, shell if shell >= 5 else 0, gaps, sun, views[band], kernels[o["platform"], band],
            reversed_sun=True)
        profiles[band] = main / variation, reversed_sun / variation
    red, red_status = best(gaps, *profiles["B04"], int(keep.sum()))
    if keep.all() and red != float(reference["estimated_gap_m"]):
        raise ValueError(f"B04 baseline differs from the saved fit for {o['observation_id']}")
    joint_profiles = np.mean(list(profiles.values()), axis=0)
    joint, joint_status = best(gaps, *joint_profiles, int(keep.sum()))
    _, fine_loss, _, fine_placebo = response_profiles(
        (rows, cols, origin, row[keep], col[keep], signals["B04"], weights), radius, shell if shell >= 5 else 0,
        gaps, sun, views["B04"], kernels[o["platform"], "B04"], reversed_sun=True, shifts=(-4, -2, 0, 2, 4))
    fine, fine_status = best(gaps, fine_loss / weights.sum(), fine_placebo / weights.sum(), int(keep.sum()))
    return "compared", dict(observation_id=o["observation_id"], candidate_id=o["candidate_id"],
        version_id=o["version_id"], capture_date=o["captured_at_utc"][:10], pixels=int(keep.sum()),
        red_gap_m=red, joint_gap_m=joint, red_status=red_status, joint_status=joint_status,
        red_fine_centre_gap_m=fine, red_fine_centre_status=fine_status,
        lidar_gap_m=float(reference["lidar_gap_m"]), minimum_lidar_separation_hours=float(reference["minimum_sensor_gap_hours"]))


def metrics(rows):
    result = {name: describe([r[f"{name}_gap_m"] - r["lidar_gap_m"] for r in rows]) for name in METHODS}
    for name in result:
        errors = [abs(r[f"{name}_gap_m"] - r["lidar_gap_m"]) for r in rows]
        result[name].update(mean_absolute_m=fmean(errors) if errors else None,
                            diagnostic_passes=sum(r[f"{name}_status"] == RESOLVED for r in rows))
    return result


def spectral_kernels(db):
    kernels = {}
    for platform, band in db.execute("SELECT platform,band FROM instrument_psf WHERE band IN ('B03','B04','B08')"):
        values = [r[0] for r in db.execute("""SELECT coefficient FROM psf_coefficient WHERE platform=? AND band=?
            ORDER BY psf_row,psf_col""", (platform, band))]
        kernel = np.asarray(values, dtype="float32").reshape(33, 33)
        kernels[platform, band] = (kernel + kernel.T) / (2 * kernel.sum())
    return kernels


def main():
    references = {int(r["observation_id"]): r for r in read_csv("shadow_geometry_estimates")
                  if r["roof_type"] == "EFR" and r["selected_for_pass_metrics"] == "True"
                  and r["lidar_gap_m"] and r["estimated_gap_m"]}
    statuses = Counter()
    rows = []
    with closing(read_only(CALIBRATION)) as db:
        db.row_factory = sqlite3.Row
        kernels = spectral_kernels(db)
        for identity, reference in references.items():
            o = db.execute("""SELECT o.*,s.version_id,s.platform,s.captured_at_utc,s.tile_metadata_path FROM observation o
                JOIN scene s USING(scene_id) WHERE observation_id=? AND s.level='L1C'""", (identity,)).fetchone()
            if o is None or o["version_id"] != reference["version_id"]:
                raise ValueError(f"Different source observation for LiDAR comparison {identity}")
            status, result = compare(db, o, reference, kernels)
            statuses[status] += 1
            if result:
                rows.append(result)
    by_day = defaultdict(list)
    for row in rows:
        by_day[row["capture_date"]].append(row)
    comparisons = {day: dict(all_matched=metrics(sample), all_methods_resolved=metrics(
        [r for r in sample if all(r[f"{name}_status"] == RESOLVED for name in METHODS)]))
        for day, sample in sorted(by_day.items())}
    report = dict(statuses=dict(statuses), bands=list(BANDS), comparisons=comparisons,
        interpretation="Both fits use identical native pixels. Band-specific viewing angles, point spread responses "
        "and radiometry feed a common gap and centre, with individual background and contrast fits. Equal weights are "
        "applied to residual profiles normalized by within-chip weighted reflectance variance. The fine-centre B04 "
        "fit keeps the same allowed displacement and uses 2 m centre trials rather than 4 m trials. Detector uniqueness "
        "outside B04 is screened using the angle grid at chip corners and centre. LiDAR separations are nonzero. "
        "Results describe a two-date optical diagnostic. No inventory or market outcomes are used.")
    export("multiband_shadow_lidar_comparison", rows)
    write(DERIVED / "multiband_shadow_lidar_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
