"""Index shared native-grid image windows for tank-independent registration."""
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from math import floor, hypot

import numpy as np
import rasterio

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, write
from collect_s2_l1c import parse_product, source_metadata_path

PIXEL_M = 10
PATCH_M = 160
PADDING_M = 20
PATCH_STRIDE_M = PATCH_M + 2 * PADDING_M
SEARCH_RADIUS_M = 800
TANK_EXCLUSION_M = 120
CRS = "EPSG:32614"
WEEKLY_PACKETS = (
    "weekly_shadow_development", "weekly_shadow_terminal_development",
    "weekly_shadow_validation", "weekly_shadow_holdout",
)
CONTROL_PACKET = "shadow_change_controls"


def checked_path(value):
    path = source_metadata_path(ROOT / value)
    if not path.is_file():
        raise ValueError(f"Registration source is missing: {value}")
    return str(path.relative_to(ROOT))


def scene_sources():
    scenes, controls = {}, {}
    for packet in (*WEEKLY_PACKETS, CONTROL_PACKET):
        with closing(read_only(DERIVED / f"{packet}.sqlite")) as db:
            db.row_factory = sqlite3.Row
            for raw in registration_sources(db):
                row = dict(raw)
                key = row["version_id"], row["source_directory"]
                old = scenes.setdefault(key, row)
                if any(row[field] != old[field] for field in row if field != "scene_id"):
                    raise ValueError(f"Source metadata differs between physical packets: {key}")
            if packet == CONTROL_PACKET:
                for tank, version, folder in db.execute("""SELECT o.candidate_id,s.version_id,s.source_directory
                        FROM observation o JOIN scene s USING(scene_id) WHERE s.level='L1C'"""):
                    key = tank, version
                    if key in controls:
                        raise ValueError(f"Ambiguous control source: {key}")
                    controls[key] = version, folder
    result = {}
    for number, (key, row) in enumerate(sorted(scenes.items()), 1):
        result[key] = describe_source(number, row)
    return result, controls


def registration_sources(db):
    return db.execute("""
        SELECT s.*,r.source_raster AS red_raster,r.reflectance_scale,r.reflectance_offset,
            r.nodata_dn,r.saturated_dn,g.source_raster AS green_raster,n.source_raster AS nir_raster
        FROM scene s JOIN band r ON r.scene_id=s.scene_id AND r.band='B04'
        JOIN band g ON g.scene_id=s.scene_id AND g.band='B03'
        JOIN band n ON n.scene_id=s.scene_id AND n.band='B08' WHERE s.level='L1C'
    """)


def describe_source(number, row):
    version, folder = row["version_id"], row["source_directory"]
    (platform, _, orbit, tile), baseline, _ = parse_product(version)
    basis = row["cloud_mask_basis"]
    mask_name = {"gml_opaque_cirrus": "MSK_CLOUDS_B00.gml",
                 "jp2_opaque_cirrus_snow_ice": "MSK_CLASSI_B00.jp2"}.get(basis)
    if mask_name is None:
        raise ValueError(f"No usable original cloud mask for {version}: {basis}")
    red, green, nir = (checked_path(row[f"{band}_raster"]) for band in ("red", "green", "nir"))
    with rasterio.open(ROOT / red) as raster:
        affine = tuple(row[f"affine_{element}"] for element in "abcdef")
        if (raster.crs.to_string(), tuple(raster.transform)[:6], raster.width, raster.height) != (
            row["crs"], affine, row["width"], row["height"]
        ) or row["crs"] != CRS or (affine[0], affine[1], affine[3], affine[4]) != (PIXEL_M, 0, 0, -PIXEL_M):
            raise ValueError(f"Source raster differs from the native geometry in {version}")
        west, south, east, north = raster.bounds
    return dict(source_id=number, version_id=version, canonical_l2a_id=row["scene_item_id"], platform=platform,
                relative_orbit=orbit, mgrs_tile=tile, processing_baseline=baseline, captured_at_utc=row["captured_at_utc"],
                crs=CRS, pixel_m=PIXEL_M, west=west, south=south, east=east, north=north,
                width=row["width"], height=row["height"], original_tile_row_offset=row["tile_row_offset"],
                original_tile_col_offset=row["tile_col_offset"], red_raster=red, green_raster=green, nir_raster=nir,
                red_reflectance_scale=row["reflectance_scale"], red_reflectance_offset=row["reflectance_offset"],
                red_nodata_dn=row["nodata_dn"], red_saturated_dn=row["saturated_dn"], cloud_mask_basis=basis,
                cloud_mask=checked_path(f"{folder}/{mask_name}"), tile_metadata=checked_path(row["tile_metadata_path"]),
                product_manifest=checked_path(row["manifest_path"]))


def pair_inputs(sources, controls):
    for stem in ("weekly_shadow_endpoint_manifest", "weekly_shadow_terminal_endpoint_manifest"):
        pairs = defaultdict(dict)
        for row in read_csv(stem):
            key = row["study_phase"], row["week_ending"], row["candidate_id"]
            if row["endpoint"] in pairs[key]:
                raise ValueError(f"Duplicate weekly endpoint in {stem}: {key}")
            pairs[key][row["endpoint"]] = row
        for (phase, week, tank), pair in sorted(pairs.items()):
            first, last = pair["previous"], pair["current"]
            cutoff = first["inputs_available_before_utc"]
            if cutoff != last["inputs_available_before_utc"] or first["terminal_id"] != last["terminal_id"]:
                raise ValueError(f"Inconsistent weekly pair: {week}/{tank}")
            earlier, later = (sources[r["version_id"], r["source_directory"]] for r in (first, last))
            for endpoint, scene in ((first, earlier), (last, later)):
                if scene["canonical_l2a_id"] != endpoint["canonical_l2a_id"] or scene["captured_at_utc"] != endpoint["captured_at_utc"]:
                    raise ValueError(f"Weekly pair points to a different native scene: {week}/{tank}")
                if max(map(datetime.fromisoformat, (endpoint["archive_available_at_utc"],
                       endpoint["geometry_available_at_utc"], endpoint["captured_at_utc"]))) >= datetime.fromisoformat(cutoff):
                    raise ValueError(f"Weekly registration input arrives at or after cutoff: {week}/{tank}")
            yield ("weekly", phase, week, tank, last["roof_type"], last["terminal_id"], cutoff,
                   first["archive_available_at_utc"], last["archive_available_at_utc"], earlier, later)
    for row in read_csv("shadow_fixed_weekly_pairs"):
        tank = row["candidate_id"]
        earlier, later = (sources[controls[tank, row[f"{side}_version"]]] for side in ("earlier", "later"))
        yield ("fixed_roof", row["study_phase"], row["week_ending"], tank, row["roof_type"], "",
               row["inputs_available_before_utc"], row["earlier_archive_available_at_utc"],
               row["later_archive_available_at_utc"], earlier, later)
    for row in read_csv("shadow_tandem_pass_pairs"):
        if row["selected_for_summary"] != "True":
            continue
        tank = row["candidate_id"]
        entries = sorted(((sources[controls[tank, row[f"{side}_version"]]], row[f"{side}_archive_available_at_utc"])
                         for side in ("a", "other")), key=lambda entry: datetime.fromisoformat(entry[0]["captured_at_utc"]))
        (earlier, first_available), (later, last_available) = entries
        yield ("tandem", "engineering", row["capture_day"], tank, row["roof_type"], "", "",
               first_available, last_available, earlier, later)


def tank_points():
    with closing(read_only(SOURCE)) as db:
        rows = db.execute("""SELECT candidate_id,footprint_east_m,footprint_north_m,footprint_diameter_m/2
            FROM tank_engineering_registry ORDER BY candidate_id""").fetchall()
    return {tank: (east, north) for tank, east, north, _ in rows}, np.asarray([r[1:] for r in rows], dtype=float)


def anchor_grid(tanks, exclusion, *, search_radius_m=SEARCH_RADIUS_M, patch_m=PATCH_M, padding_m=PADDING_M):
    stride = patch_m + 2 * padding_m

    @lru_cache(maxsize=None)
    def outside_tank_zones(west, south):
        east, north = west + patch_m, south + patch_m
        dx = np.maximum(0, np.maximum(west - padding_m - exclusion[:, 0], exclusion[:, 0] - east - padding_m))
        dy = np.maximum(0, np.maximum(south - padding_m - exclusion[:, 1], exclusion[:, 1] - north - padding_m))
        return bool((np.hypot(dx, dy) >= exclusion[:, 2] + TANK_EXCLUSION_M).all())

    candidates = {}
    for tank, (east, north) in tanks.items():
        e0, n0 = floor(east / stride), floor(north / stride)
        tiles = []
        steps = search_radius_m // stride + 1
        for i in range(e0 - steps, e0 + steps + 1):
            for j in range(n0 - steps, n0 + steps + 1):
                west, south = i * stride + padding_m, j * stride + padding_m
                if (hypot(west + patch_m / 2 - east, south + patch_m / 2 - north) <= search_radius_m
                        and outside_tank_zones(west, south)):
                    tiles.append((west, south, (i + j) % 2))
        candidates[tank] = tiles
    return candidates


def shared_bounds(earlier, later):
    if any((earlier[k] - later[k]) % PIXEL_M for k in ("west", "north")):
        raise ValueError("The two image sources do not share a native lattice")
    return (max(earlier["west"], later["west"]), max(earlier["south"], later["south"]),
            min(earlier["east"], later["east"]), min(earlier["north"], later["north"]))


def build_windows(inputs, anchors):
    comparisons, windows, used_anchors = {}, [], {}
    counters = defaultdict(Counter)
    keys = set()
    for comparison, phase, day, tank, roof, terminal, cutoff, first_available, last_available, earlier, later in inputs:
        identity = earlier["source_id"], later["source_id"]
        if datetime.fromisoformat(earlier["captured_at_utc"]) >= datetime.fromisoformat(later["captured_at_utc"]):
            raise ValueError(f"Source chronology is inconsistent for {comparison}/{day}/{tank}")
        if identity not in comparisons:
            west, south, east, north = shared_bounds(earlier, later)
            comparisons[identity] = dict(scene_pair_id=len(comparisons)+1, earlier_source_id=identity[0],
                later_source_id=identity[1], same_relative_orbit=earlier["relative_orbit"] == later["relative_orbit"],
                same_mgrs_tile=earlier["mgrs_tile"] == later["mgrs_tile"], shared_west=west, shared_south=south,
                shared_east=east, shared_north=north)
        pair = comparisons[identity]
        key = comparison, phase, day, tank, pair["scene_pair_id"]
        if key in keys:
            raise ValueError(f"Duplicate registration comparison: {key}")
        keys.add(key)
        inside = [[], []]
        for west, south, split in anchors[tank]:
            if (pair["shared_west"] <= west-PADDING_M and pair["shared_east"] >= west+PATCH_M+PADDING_M
                    and pair["shared_south"] <= south-PADDING_M and pair["shared_north"] >= south+PATCH_M+PADDING_M):
                anchor_id = f"{west}_{south}"
                role = ("fit", "spatial_check")[split]
                inside[split].append(anchor_id)
                used_anchors.setdefault(anchor_id, dict(anchor_id=anchor_id, west=west, south=south,
                    east=west+PATCH_M, north=south+PATCH_M, padding_m=PADDING_M, role=role, crs=CRS))
        status = "both_patch_groups_in_bounds" if all(inside) else "missing_one_patch_group" if any(inside) else "no_patch_in_bounds"
        counters[f"{comparison}/{phase}"][status] += 1
        windows.append(dict(comparison=comparison, study_phase=phase, week_or_control_day=day,
            candidate_id=tank, roof_type=roof, terminal_id=terminal, scene_pair_id=pair["scene_pair_id"],
            inputs_available_before_utc=cutoff, earlier_archive_available_at_utc=first_available,
            later_archive_available_at_utc=last_available, fit_patch_count=len(inside[0]),
            spatial_check_patch_count=len(inside[1]), patch_geometry_status=status,
            fit_patch_ids="|".join(inside[0]), spatial_check_patch_ids="|".join(inside[1])))
    return list(comparisons.values()), windows, list(used_anchors.values()), dict(counters)


def main():
    sources, controls = scene_sources()
    pairs, windows, anchors, coverage = build_windows(pair_inputs(sources, controls), anchor_grid(*tank_points()))
    used_ids = {r[f"{side}_source_id"] for r in pairs for side in ("earlier", "later")}
    source_rows = [r for r in sources.values() if r["source_id"] in used_ids]
    for name, rows in (("sources", source_rows), ("scene_pairs", pairs), ("tank_windows", windows), ("patches", anchors)):
        export(f"shadow_registration_{name}", rows)
    summary = dict(sources=len(source_rows), scene_pairs=len(pairs), tank_windows=len(windows), candidate_patches=len(anchors),
        patch_m=PATCH_M, padding_m=PADDING_M, patch_stride_m=PATCH_STRIDE_M,
        search_radius_m=SEARCH_RADIUS_M, tank_exclusion_m=TANK_EXCLUSION_M,
        coverage=coverage, interpretation="Patch counts establish geometry and local source availability. Stable landmarks, "
        "cloud-free pixels and registration quality require pixel-level checks. No EIA or market outcomes are read.")
    report = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / "shadow_registration_input_summary.json", report.encode())
    print(report, end="", flush=True)


if __name__ == "__main__":
    main()
