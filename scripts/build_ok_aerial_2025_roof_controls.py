"""Measure roofs at 15 cm and check the tile clock against exterior shadows."""
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from datetime import datetime, timedelta

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.merge import merge
from rasterio.transform import Affine

from audit_shadow_precision import describe
from build_aerialok_october_controls import scores
from build_naip_2019_roof_reference import MAX_CENTER_SHIFT, fit, project
from build_naip_2025_roof_reference import solar_module
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_ok_aerial import TARGET
from collect_public import ROOT, write

UTM = "EPSG:32614"
STATE = "EPSG:6350"
TO_STATE = Transformer.from_crs(UTM, STATE, always_xy=True)
TO_UTM = Transformer.from_crs(STATE, UTM, always_xy=True)
GRID_M = 0.15
EXTERIOR_REACH_M = 65
RESOLVED = "resolved_aerial_shadow"


def input_geometry():
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        geometry = {r["candidate_id"]: dict(r) for r in source.execute("""
            SELECT t.candidate_id,t.documented_roof_type,t.documented_terminal_id,t.longitude,t.latitude,t.first_seen,
                coalesce(l.center_east_m,t.footprint_east_m) AS east,coalesce(l.center_north_m,t.footprint_north_m) AS north,
                coalesce(l.diameter_m,t.footprint_diameter_m)/2 AS radius,t.lidar_rim_to_ground_native_z AS shell_m,
                CASE WHEN l.center_east_m IS NOT NULL THEN l.dataset ELSE 'mapped_footprint' END AS geometry_source
            FROM tank_engineering_registry t LEFT JOIN tank_lidar_outline l
                ON l.candidate_id=t.candidate_id AND l.fit_status='well_supported_circular_fit'
        """)}
    grouped = defaultdict(list)
    for r in read("ok_aerial_tank_image_coverage"):
        t = geometry[r["candidate_id"]]
        if (t["documented_roof_type"] and t["first_seen"] and t["first_seen"] <= r["tile_majority_date"]
                and r["tile_time_raw"]):
            grouped[t["candidate_id"], r["tile_majority_date"], r["tile_time_raw"]].append(r)
    return geometry, grouped


def affine_state_to_utm(x, y):
    e, n = TO_UTM.transform(x, y)
    ex, nx = TO_UTM.transform(x+1, y)
    ey, ny = TO_UTM.transform(x, y+1)
    return Affine(ex-e, ey-e, e, nx-n, ny-n, n) * Affine.translation(-x, -y)


def rgb_tiles():
    paths = {p.parent.name: p for p in sorted((TARGET / "tank_clips").glob("*/item.json"))}
    paths.update({p.parent.name: p for p in sorted((TARGET / "tiles").glob("*/item.json"))})
    by_clock = defaultdict(list)
    for identity, path in sorted(paths.items()):
        props = json.loads(path.read_bytes())["properties"]
        if (stamp := props["time"]):
            by_clock[props["majorityda"][:10], stamp].append((identity, path.parent / "image.tif"))
    return by_clock


def read_crop(stack, options, sources, tank, reach):
    crop = read_native_bands(stack, options, sources, tank, reach, (1, 2, 3))
    if crop is None:
        return None
    rgb, valid, east, north, affine, cover = crop
    gray = np.einsum("ijk,i->jk", rgb, np.array([0.2126, 0.7152, 0.0722]))
    gray[~valid] = np.nan
    return gray, valid, east, north, affine, cover


def read_native_bands(stack, options, sources, tank, reach, bands):
    x, y = TO_STATE.transform(tank["east"], tank["north"])
    extent = tank["radius"] + reach
    cols = [(tile, path, sources.get(path)) for tile, path in options]
    datasets = []
    for tile, path, ds in cols:
        if ds is None:
            ds = sources[path] = stack.enter_context(rasterio.open(path))
        if ds.crs.to_epsg() != 6350 or not all(math.isclose(r, GRID_M, abs_tol=1e-6) for r in ds.res):
            raise ValueError(f"Unexpected state orthophoto grid for {tile}")
        l, b, r, t = ds.bounds
        if l < x+extent and r > x-extent and b < y+extent and t > y-extent:
            datasets.append(ds)
    if not datasets:
        return None
    source = datasets[0]
    col0, row0 = (~source.transform) * (x-extent, y+extent)
    col1, row1 = (~source.transform) * (x+extent, y-extent)
    west, top = source.transform * (math.floor(col0), math.floor(row0))
    east, bottom = source.transform * (math.ceil(col1), math.ceil(row1))
    for other in datasets[1:]:
        offset = (~source.transform) * (other.transform.c, other.transform.f)
        if any(abs(v-round(v)) > 1e-5 for v in offset):
            raise ValueError("Source tiles are not on the same 15 cm grid")
    samples, affine = merge(datasets, bounds=(west, bottom, east, top), indexes=bands, masked=True)
    valid = ~np.ma.getmaskarray(samples).any(axis=0)
    affine = affine_state_to_utm(x, y) * affine
    rows, cols = np.indices(valid.shape, sparse=True)
    east, north = affine * (cols+0.5, rows+0.5)
    east, north = east-tank["east"], north-tank["north"]
    radius = tank["radius"] + MAX_CENTER_SHIFT + 3
    area = np.pi*radius**2/abs(affine.determinant)
    cover = min(1., np.count_nonzero((east**2+north**2 <= radius**2) & valid)/area)
    return samples.data, valid, east, north, affine, cover


def exterior_clock_profile(crop, tank, geometry, spa, stamps):
    gray, _, _, _, affine, _ = crop
    radius, shell = tank["radius"], tank["shell_m"]
    if not shell or not 5 <= shell <= 30:
        return None
    axis = np.arange(-radius-EXTERIOR_REACH_M, radius+EXTERIOR_REACH_M+1, 2.)
    xx, yy = (v.ravel() for v in np.meshgrid(axis, axis))
    distance = np.hypot(xx, yy)
    valid = (distance > radius+3) & (distance < radius+EXTERIOR_REACH_M)
    for other in geometry.values():
        if other is tank:
            continue
        dx, dy = other["east"]-tank["east"], other["north"]-tank["north"]
        limit = radius+EXTERIOR_REACH_M+other["radius"]+3
        if abs(dx) < limit and abs(dy) < limit:
            valid &= np.hypot(xx-dx, yy-dy) > other["radius"]+3
    col, row = (~affine) * (xx+tank["east"], yy+tank["north"])
    i, j = np.floor(row).astype(int), np.floor(col).astype(int)
    valid &= (i >= 0) & (j >= 0) & (i < gray.shape[0]) & (j < gray.shape[1])
    values = gray[i.clip(0, gray.shape[0]-1), j.clip(0, gray.shape[1]-1)]
    valid &= np.isfinite(values)
    if valid.sum() < 160:
        return None
    times = np.array([t.timestamp() for t in stamps])
    longitude, latitude = tank["longitude"], tank["latitude"]
    delta_t = spa.calculate_deltat(stamps[0].year, stamps[0].month)
    _, zenith, _, elevation, azimuth, _ = spa.solar_position(times, latitude, longitude, 300, 0, 15, delta_t, 0.5667)
    x, y = project("EPSG:4326", UTM)(longitude, latitude+0.001)
    dx, dy = project("EPSG:4326", UTM)(longitude, latitude)
    direction = np.deg2rad(azimuth) + math.atan2(x-dx, y-dy)
    ex, ey = -np.sin(direction)[:, None], -np.cos(direction)[:, None]
    xx, yy = xx[valid], yy[valid]
    reach = (shell*np.tan(np.deg2rad(zenith)))[:, None]
    path = np.clip(ex*xx+ey*yy, 0, reach)
    shadow = (xx-ex*path)**2 + (yy-ey*path)**2 <= radius**2
    gain, _ = scores(values[valid], shadow)
    gain[elevation < 12] = 0
    return gain


def clock_comparisons(rows, profiles, trial_times):
    by_tile = defaultdict(list)
    for row in rows:
        key = row["candidate_id"], row["tile_group"]
        if key in profiles:
            by_tile[row["tile_group"]].append((row["candidate_id"], profiles[key]))
    for row in rows:
        data = [g for tank, g in by_tile[row["tile_group"]] if tank != row["candidate_id"]]
        row.update(exterior_donor_tanks=len(data), exterior_best_utc=None, exterior_best_gain=None,
            exterior_nominal_gain=None, exterior_time_difference_minutes=None, exterior_clock_status="insufficient_other_tanks")
        if len(data) < 3:
            continue
        times = trial_times[row["tile_group"]]
        gains = np.median(data, axis=0)
        best = int(np.argmax(gains))
        nominal = datetime.fromisoformat(row["start_utc"])
        differences = np.array([(t-nominal).total_seconds()/60 for t in times])
        distant = np.abs(differences-differences[best]) >= 30
        margin = gains[best] - gains[distant].max()
        if gains[best] < .08 or margin < .025:
            status = "exterior_clock_ambiguous"
        else:
            status = "exterior_matches_tile_time" if abs(differences[best]) <= 15 else "exterior_differs_from_tile_time"
        row.update(exterior_best_utc=times[best].isoformat(), exterior_best_gain=float(gains[best]),
            exterior_nominal_gain=float(gains[differences == 0][0]), exterior_time_difference_minutes=float(differences[best]),
            exterior_clock_status=status)


def build_observations():
    geometry, groups = input_geometry()
    tiles = rgb_tiles()
    spa = solar_module()
    rows, exterior, trial_times = [], {}, {}
    with ExitStack() as stack:
        sources = {}
        for n, ((tank, day, time), coverage) in enumerate(sorted(groups.items()), 1):
            t = geometry[tank]
            nominal = datetime.fromisoformat(f"{day}T{time}:00+00:00")
            primary = max(coverage, key=lambda r: (float(r["valid_fraction"]), r["tile_id"]))
            group = primary["tile_id"]
            path = primary["asset_path"]
            # A group can include neighboring tiles only when their catalog clock labels agree.
            options = tiles[day, time]
            crop = read_crop(stack, options, sources, t, MAX_CENTER_SHIFT+7)
            if crop is None or not t["shell_m"]:
                raise ValueError(f"State aerial source or height geometry missing for {tank}")
            info = dict(t, reference_day=day, tile_group=group, tile_nominal_time_utc=time,
                nominal_tile_id=group, nominal_tile_catalog_file=str((ROOT / path).parent.joinpath("item.json").relative_to(ROOT)),
                start_utc=nominal.isoformat(), end_exclusive_utc=(nominal+timedelta(minutes=1)).isoformat(),
                timing_basis="tile_majority_nominal_minute_source_frame_unverified")
            image = sources[ROOT / path] if ROOT / path in sources else next(ds for p, ds in sources.items() if p.parent.name == group)
            result = fit(info, (crop[-1], path, t["east"], t["north"], crop), spa, {path: image}, model_crs=UTM)
            rows.append(dict(info, **result))
            midnight = nominal.replace(hour=0, minute=0)
            if group not in trial_times:
                times = [midnight+timedelta(minutes=m) for m in range(14*60, 23*60+31, 10)]
                trial_times[group] = sorted(set((*times, nominal)))
            if crop[-1] >= .985:
                outer = read_crop(stack, options, sources, t, EXTERIOR_REACH_M+3)
                values = exterior_clock_profile(outer, t, geometry, spa, trial_times[group])
                if values is not None:
                    exterior[tank, group] = values
            if n % 50 == 0 or n == len(groups):
                print(f"Measured {n}/{len(groups)} dated tank tile groups", flush=True)
    clock_comparisons(rows, exterior, trial_times)
    return rows


def compare_umbra(rows):
    photos = {(r["candidate_id"], r["nominal_tile_id"], r["reference_day"]): r for r in rows}
    radar = {(r["candidate_id"], r["product_id"]): r for r in read("umbra_sicd_all_roof_arc_fits")}
    for m in read("umbra_cushing_reference_matches"):
        if m["reference_kind"] != "ok_aerial_tile_majority":
            continue
        photo = photos.get((m["candidate_id"], m["reference_id"], m["reference_day"]))
        radar_fit = radar.get((m["candidate_id"], m["product_id"]))
        if photo is None or radar_fit is None:
            continue
        if photo["documented_roof_type"] != "EFR" or radar_fit["roof_type"] != "EFR" or not math.isclose(
                photo["shell_m"], float(radar_fit["lidar_rim_minus_ground_m"]), rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"Different geometry in state aerial and radar for {m['candidate_id']}")
        if datetime.fromisoformat(photo["start_utc"]) != datetime.fromisoformat(m["reference_start_utc"]):
            raise ValueError(f"Tile clocks disagree for {m['candidate_id']}")
        gap = photo["best_gap_m"]
        aerial = photo["shell_m"] - gap if gap is not None else None
        radar_height = float(radar_fit["scene_registered_height_m"]) if radar_fit["scene_registered_height_m"] else None
        radar_pass = radar_fit["scene_registered_use_status"] == "resolved_arc_diagnostic"
        yield dict(m, aerial_fit_status=photo["shadow_reference_status"], exterior_clock_status=photo["exterior_clock_status"],
                   aerial_roof_height_m=aerial, radar_scene_fit_status=radar_fit["scene_registered_use_status"],
                   radar_roof_height_m=radar_height, both_diagnostics_pass=radar_pass and photo["shadow_reference_status"] == RESOLVED,
                   radar_minus_aerial_height_m=radar_height-aerial if None not in (radar_height, aerial) else None)


def main():
    rows = build_observations()
    pairs = list(compare_umbra(rows))
    export("ok_aerial_2025_roof_shadow_reference", rows)
    export("ok_aerial_2025_umbra_roof_comparisons", pairs)
    roofs = [r for r in rows if r["documented_roof_type"] == "EFR"]
    resolved = [r for r in roofs if r["shadow_reference_status"] == RESOLVED]
    accepted = [r for r in pairs if r["both_diagnostics_pass"]]
    clock_pass = [r for r in accepted if r["exterior_clock_status"] == "exterior_matches_tile_time"]
    report = dict(aerial_observations=len(rows), distinct_tanks=len({r["candidate_id"] for r in rows}),
        distinct_efr_tanks=len({r["candidate_id"] for r in roofs}), efr_observations=len(roofs), resolved_efr=len(resolved),
        covered_roof_observations=len(rows)-len(roofs), covered_roof_false_passes=sum(
            r["documented_roof_type"] != "EFR" and r["shadow_reference_status"] == RESOLVED for r in rows),
        clock_status=dict(Counter(r["exterior_clock_status"] for r in rows)), radar_matches=len(pairs),
        all_radar_disagreement=describe([r["radar_minus_aerial_height_m"] for r in accepted]),
        clock_corroborated_radar_disagreement=describe([r["radar_minus_aerial_height_m"] for r in clock_pass]),
        interpretation="State WFS records a nominal tile majority clock. Other tanks' exterior shadows check consistency independently of roof pixels. "
        "The tests do not identify the exact exposure of an individual roof or eliminate real inventory movement between captures. "
        "Original RGB values are retained with local conversion of the projected geometry to UTM. No EIA outcome or price is used.")
    write(DERIVED / "ok_aerial_2025_roof_shadow_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
