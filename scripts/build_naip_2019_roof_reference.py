"""Estimate NAIP shadow lengths for the dated 2019 radar comparison set."""
import argparse
import hashlib
import importlib.util
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import median

import numpy as np
import rasterio
from pyproj import Geod, Transformer
from rasterio.transform import Affine
from rasterio.windows import Window

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import ROOT, write

DAY = "2019-08-12"
TIME_SAMPLES = 9
SUN_GEOD = Geod(ellps="WGS84")
ROOF_SAMPLE_GAP = 0.25
MIN_GAIN = 0.25
MIN_CONTRAST = 15
MIN_DIRECTION_GAIN = 0.05
MAX_NEAR_SPAN = 3.0
MAX_CENTER_SHIFT = 8.0

SOURCE_SQL = """
WITH matches AS (
    SELECT *,row_number() OVER (PARTITION BY candidate_id ORDER BY contour_fraction DESC,source_polygon_id) AS pos
    FROM naip_tank_source_clock WHERE source_acquisition_date=? AND contour_fraction>=0.99
        AND json_array_length(dated_asset_paths_json)>0
), lidar AS (
    SELECT *,row_number() OVER (PARTITION BY candidate_id ORDER BY first_upper_return_at_utc DESC) AS pos
    FROM tank_lidar_outline WHERE fit_status='well_supported_circular_fit'
)
SELECT t.candidate_id,t.documented_terminal_id,t.documented_roof_type,t.longitude,t.latitude,t.first_seen,
    coalesce(l.center_east_m,t.footprint_east_m) AS east,coalesce(l.center_north_m,t.footprint_north_m) AS north,
    coalesce(l.diameter_m,t.footprint_diameter_m)/2 AS radius,t.lidar_rim_to_ground_native_z AS shell_m,
    CASE WHEN l.center_east_m IS NOT NULL THEN l.dataset ELSE 'mapped_footprint' END AS geometry_source,
    m.source_polygon_id,m.start_utc,m.end_exclusive_utc,m.dated_asset_paths_json,m.source_file
FROM matches m JOIN tank_engineering_registry t USING(candidate_id) LEFT JOIN lidar l ON l.candidate_id=t.candidate_id AND l.pos=1
WHERE m.pos=1 AND t.first_seen<=? AND t.documented_roof_type IN ('EFR','IFR','Domed EFR')
ORDER BY t.candidate_id
"""


@lru_cache(maxsize=8)
def project(origin, target):
    return Transformer.from_crs(origin, target, always_xy=True).transform


def native_crop(dataset, east, north, radius):
    row, col = dataset.index(east, north)
    if dataset.crs.is_geographic:
        if dataset.crs.to_epsg() != 4326:
            raise ValueError("Geographic aerial crop must use the WGS84 ellipsoid")
        dx, dy = dataset.res
        x_m = SUN_GEOD.inv(east, north, east + dx, north)[2]
        y_m = SUN_GEOD.inv(east, north, east, north + dy)[2]
        # Change the coordinate units locally and retain every original source sample.
        metric = Affine.scale(x_m/dx, y_m/dy) * Affine.translation(-east, -north)
        east, north = 0., 0.
        spacing = min(x_m, y_m)
    else:
        metric = Affine.identity()
        spacing = min(dataset.res)
    pixels = int(np.ceil((radius + MAX_CENTER_SHIFT + 6) / spacing))
    r0, c0 = max(0, row - pixels), max(0, col - pixels)
    r1, c1 = min(dataset.height, row + pixels + 1), min(dataset.width, col + pixels + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    window = Window(c0, r0, c1 - c0, r1 - r0)
    rgb = dataset.read((1, 2, 3), window=window).astype(float)
    gray = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    valid = np.all(dataset.read_masks((1, 2, 3), window=window) > 0, axis=0)
    affine = metric * dataset.window_transform(window)
    if affine.b or affine.d or affine.a <= 0 or affine.e >= 0:
        raise ValueError("NAIP audit expects an unrotated source raster grid")
    i, j = np.indices(gray.shape)
    x, y = affine * (j + 0.5, i + 0.5)
    distance = np.hypot(x - east, y - north)
    footprint = (distance <= radius + MAX_CENTER_SHIFT + 3)
    area = np.pi * (radius + MAX_CENTER_SHIFT + 3) ** 2 / abs(affine.a * affine.e)
    coverage = min(1.0, float(np.count_nonzero(footprint & valid) / area))
    gray[~valid] = np.nan
    return gray, valid, x - east, y - north, affine, coverage


def find_asset(tank, sources):
    options = []
    for path in json.loads(tank["dated_asset_paths_json"]):
        dataset = sources[path]
        east, north = project("EPSG:32614", dataset.crs)(tank["east"], tank["north"])
        crop = native_crop(dataset, east, north, tank["radius"])
        if crop:
            if dataset.crs.is_geographic:
                east, north = 0., 0.
            options.append((crop[-1], path, east, north, crop))
    return max(options, key=lambda row: (row[0], row[1])) if options else None


def ring_samples(gray, affine, east, north, dx, dy, radius):
    bearings = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    radials = np.array([radius - 1.25, radius + 1.25])
    x = east + dx[:, None, None] + np.sin(bearings)[None, None, :] * radials[None, :, None]
    y = north + dy[:, None, None] + np.cos(bearings)[None, None, :] * radials[None, :, None]
    col, row = (~affine) * (x, y)
    i, j = np.floor(row).astype(int), np.floor(col).astype(int)
    inside = (i >= 0) & (i < gray.shape[0]) & (j >= 0) & (j < gray.shape[1])
    values = gray[np.clip(i, 0, gray.shape[0] - 1), np.clip(j, 0, gray.shape[1] - 1)]
    values[~inside] = np.nan
    difference = np.abs(values[:, 1] - values[:, 0])
    valid = np.isfinite(difference).sum(axis=1) >= 88
    sector = np.nanmedian(np.where(valid[:, None], difference, 0).reshape(-1, 8, 12), axis=2)
    support = np.nanpercentile(sector, 25, axis=1)
    return np.where(valid, support, -np.inf)


def independent_centre(gray, affine, east, north, radius):
    offsets = np.arange(-MAX_CENTER_SHIFT, MAX_CENTER_SHIFT + 0.01, 1)
    dx, dy = (a.ravel() for a in np.meshgrid(offsets, offsets))
    allowed = np.hypot(dx, dy) <= MAX_CENTER_SHIFT
    dx, dy = dx[allowed], dy[allowed]
    scores = ring_samples(gray, affine, east, north, dx, dy, radius)
    if not np.isfinite(scores).any():
        return None
    order = np.lexsort((np.hypot(dx, dy), -scores))
    best = order[0]
    return float(dx[best]), float(dy[best]), float(scores[best])


def sky(spa, tank, crs):
    begin, finish = (datetime.fromisoformat(tank[k]) for k in ("start_utc", "end_exclusive_utc"))
    times = np.linspace(begin.timestamp(), finish.timestamp(), TIME_SAMPLES)
    lon, lat = tank["longitude"], tank["latitude"]
    forward = project("EPSG:4326", crs)
    origin = forward(lon, lat)
    n_lon, n_lat, _ = SUN_GEOD.fwd(lon, lat, 0, 100)
    north = forward(n_lon, n_lat)
    north_bearing = np.degrees(np.arctan2(north[0] - origin[0], north[1] - origin[1]))
    delta_t = spa.calculate_deltat(begin.year, begin.month)
    _, z, _, e, azimuth, _ = spa.solar_position(times, lat, lon, 300, 0, 15, delta_t, 0.5667)
    if np.min(e) <= 0:
        raise ValueError("NAIP daytime shadow experiment must have positive solar elevation")
    return times, e, np.radians(azimuth + north_bearing), np.tan(np.radians(z))


def signal_setup(x, y, gray, radius):
    design = np.column_stack((np.ones(len(x)), x / radius, y / radius))
    gram_inv = np.linalg.inv(design.T @ design)
    residual = gray - design @ (gram_inv @ (design.T @ gray))
    return design, gram_inv, residual, float(residual @ residual)


def profile(x, y, radius, angles, ratios, gaps, setup):
    design, gram_inv, residual, sse = setup
    gains, contrasts = [], []
    rows = np.column_stack((design, residual))
    threshold_support = max(16, int(np.ceil(0.02 * len(x))))
    for angle, ratio in zip(angles, ratios, strict=True):
        projected = x * np.sin(angle) + y * np.cos(angle)
        reach = np.sqrt(np.maximum(0, radius**2 - x**2 - y**2 + projected**2)) - projected
        order = np.argsort(reach)
        limit = np.searchsorted(reach[order], gaps * ratio, side="left")
        cumulative = np.vstack((np.zeros(4), np.cumsum(rows[order], axis=0)))[limit]
        background, covariance = cumulative[:, :3], cumulative[:, 3]
        info = limit - np.einsum("gi,ij,gj->g", background, gram_inv, background)
        usable = (limit >= threshold_support) & (len(x) - limit >= threshold_support) & (info > 1) & (covariance < 0)
        contrast = np.divide(-covariance, info, out=np.zeros(len(gaps)), where=usable)
        gain = -covariance * contrast / max(sse, 1e-9)
        gains.append(gain)
        contrasts.append(contrast)
    return np.asarray(gains), np.asarray(contrasts)


def fit(tank, selected, spa, sources, *, model_crs=None):
    coverage, path, east, north, crop = selected
    gray, valid, x, y, affine, _ = crop
    radius = tank["radius"]
    centre = independent_centre(gray, affine, east, north, radius)
    result = dict(source_raster=path, source_crs=str(sources[path].crs), native_pixel_m=math.hypot(affine.a, affine.d),
                  native_clip_valid_fraction=coverage, source_fit_east_m=None, source_fit_north_m=None,
                  east_shift_m=None, north_shift_m=None, rim_edge_lower_quartile_dn=None,
                  fitted_pixels=None, elevation_low_deg=None, elevation_high_deg=None, best_time_utc=None,
                  best_gap_m=None, fitted_horizontal_shadow_m=None, near_gap_min_m=None, near_gap_max_m=None,
                  midpoint_time_gap_m=None, time_only_gap_min_m=None, time_only_gap_max_m=None,
                  best_gain=None, reversed_direction_gain=None, fitted_shadow_contrast_dn=None,
                  shadow_reference_status="insufficient_aerial_coverage")
    if centre is None or coverage < 0.985:
        return result
    dx, dy, rim_score = centre
    shifted_x, shifted_y = x - dx, y - dy
    inside = np.hypot(shifted_x, shifted_y) <= radius - 1.5
    keep = inside & valid
    result.update(source_fit_east_m=east + dx, source_fit_north_m=north + dy, east_shift_m=dx, north_shift_m=dy,
                  rim_edge_lower_quartile_dn=rim_score, fitted_pixels=int(keep.sum()))
    if keep.sum() < 200 or np.count_nonzero(inside & ~valid) > 0.01 * inside.sum():
        return result
    sun_times, elevations, bearings, ratios = sky(spa, tank, model_crs or sources[path].crs)
    x_fit, y_fit, signal = shifted_x[keep], shifted_y[keep], gray[keep]
    setup = signal_setup(x_fit, y_fit, signal, radius)
    shell = min(24.0, float(tank["shell_m"]))
    gaps = np.arange(0, shell + 1e-8, ROOF_SAMPLE_GAP)
    positive, contrast = profile(x_fit, y_fit, radius, bearings, ratios, gaps, setup)
    reversed_sun, _ = profile(x_fit, y_fit, radius, bearings + np.pi, ratios, gaps, setup)
    time_index, gap_index = np.unravel_index(np.argmax(positive), positive.shape)
    gain, placebo = float(positive[time_index, gap_index]), float(reversed_sun.max())
    near = gaps[positive.max(axis=0) >= gain - 0.025]
    fitted_contrast = float(contrast[time_index, gap_index])
    time_fit = gaps[np.argmax(positive, axis=1)]
    result.update(elevation_low_deg=float(elevations.min()), elevation_high_deg=float(elevations.max()),
        best_time_utc=datetime.fromtimestamp(sun_times[time_index], tz=datetime.fromisoformat(tank["start_utc"]).tzinfo).isoformat(),
        best_gap_m=float(gaps[gap_index]), fitted_horizontal_shadow_m=float(gaps[gap_index] * ratios[time_index]),
        near_gap_min_m=float(near.min()), near_gap_max_m=float(near.max()), midpoint_time_gap_m=float(time_fit[TIME_SAMPLES // 2]),
        time_only_gap_min_m=float(time_fit.min()), time_only_gap_max_m=float(time_fit.max()), best_gain=gain,
        reversed_direction_gain=placebo, fitted_shadow_contrast_dn=fitted_contrast)
    if rim_score < 3 or np.hypot(dx, dy) >= MAX_CENTER_SHIFT - 0.25:
        status = "weak_or_search_boundary_rim"
    elif gap_index in (0, len(gaps) - 1) or gain < MIN_GAIN or fitted_contrast < MIN_CONTRAST:
        status = "weak_or_boundary_shadow"
    elif gain - placebo < MIN_DIRECTION_GAIN:
        status = "reversed_direction_not_rejected"
    elif near.max() - near.min() > MAX_NEAR_SPAN:
        status = "broad_profile"
    else:
        status = "resolved_aerial_shadow"
    return dict(result, shadow_reference_status=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spa-module", type=Path, required=True, help="spa.py from pvlib 0.13.1")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("naip_roof_spa", args.spa_module)
    spa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spa)
    selected = {r["candidate_id"]: r for r in read("s1_reference_nearest_products")
                if r["reference_day"] == DAY and r["product_type"] == "IW_GRDH_1S"
                and float(r["maximum_separation_hours"]) <= 24}
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        tanks = [dict(row) for row in db.execute(SOURCE_SQL, (DAY, DAY))]
    if {t["candidate_id"] for t in tanks if t["documented_roof_type"] == "EFR"} != selected.keys():
        raise ValueError("The open-roof reference population differs from the dated radar comparison")
    rows = []
    with ExitStack() as stack:
        paths = {path for tank in tanks for path in json.loads(tank["dated_asset_paths_json"])}
        sources = {p: stack.enter_context(rasterio.open(ROOT / p)) for p in sorted(paths)}
        for tank in tanks:
            reference = selected.get(tank["candidate_id"], {})
            if reference and (tank["source_polygon_id"], tank["start_utc"], tank["end_exclusive_utc"]) != (
                    reference["reference_id"], reference["reference_start_utc"], reference["reference_end_utc"]):
                raise ValueError(f"Aerial flight clock differs for {tank['candidate_id']}")
            asset = find_asset(tank, sources)
            if asset is None or not tank["shell_m"]:
                raise ValueError(f"Missing aerial crop or shell geometry for {tank['candidate_id']}")
            rows.append(dict(tank, radar_product_id=reference.get("product_id"),
                radar_minimum_separation_hours=reference.get("minimum_separation_hours"),
                radar_maximum_separation_hours=reference.get("maximum_separation_hours"),
                **fit(tank, asset, spa, sources)))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["documented_roof_type"]].append(row)
    resolved = [row for row in grouped["EFR"] if row["shadow_reference_status"] == "resolved_aerial_shadow"]
    span = [r["near_gap_max_m"] - r["near_gap_min_m"] for r in resolved]
    times = [r["time_only_gap_max_m"] - r["time_only_gap_min_m"] for r in resolved]
    report = dict(naip_source_date=DAY, matched_efr=len(grouped["EFR"]), resolved_efr=len(resolved),
        status_counts_by_roof_type={kind: Counter(r["shadow_reference_status"] for r in group) for kind, group in grouped.items()},
        median_near_gap_span_m=median(span) if span else None,
        median_gap_span_from_time_trials_m=median(times) if times else None,
        median_gap_m=median(r["best_gap_m"] for r in resolved) if resolved else None,
        distinct_source_rasters=len({r["source_raster"] for r in rows}),
        spa_module_sha256=hashlib.sha256(args.spa_module.read_bytes()).hexdigest(), time_trials=TIME_SAMPLES,
        interpretation="The physical circle and contrast fits use native NAIP pixel samples. The centre uses rim edges, "
        "and the roof fit uses the interior and the known flight time interval. The estimated level is a provisional "
        "aerial shadow reference. The profile spans are sensitivity ranges without a probabilistic meaning; the unknown "
        "camera look direction and orthorectification effects can shift the apparent lip and shadow. IFR and domed roofs "
        "are optical negative controls. Radar levels and inventory outcomes are excluded.")
    export("naip_2019_roof_shadow_reference", rows)
    write(DERIVED / "naip_2019_roof_shadow_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
