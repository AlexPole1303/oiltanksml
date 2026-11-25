"""Compare archived Sentinel-2 angle grids and sensing clocks against independent SPA."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from statistics import median

import numpy as np
from pyproj import Transformer

from build_weekly_shadow_panel import DERIVED, export, read_only
from collect_public import ROOT, write
from collect_s2_l1c import source_metadata_bytes

CONTROLS = DERIVED / "shadow_change_controls.sqlite"
SPA_URL = "https://docs.nrel.gov/docs/fy08osti/34302.pdf"
PVLIB_URL = "https://github.com/pvlib/pvlib-python/blob/v0.13.1/pvlib/spa.py"


@lru_cache(maxsize=8)
def geographic_transform(crs):
    return Transformer.from_crs(crs, 4326, always_xy=True).transform


@lru_cache(maxsize=1500)
def xml_time(path):
    root = ET.fromstring(source_metadata_bytes(ROOT / path))
    text, = (node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "SENSING_TIME")
    return datetime.fromisoformat(text)


def solar(spa, stamp, lon, lat, xml_zenith, xml_azimuth):
    delta_t = spa.calculate_deltat(stamp.year, stamp.month)
    _, z, _, _, a, _ = spa.solar_position(np.array([stamp.timestamp()]), lat, lon, 300, 0, 15, delta_t, 0.5667)[:, 0]
    da = float((a - xml_azimuth + 180) % 360 - 180)
    dz = float(z - xml_zenith)
    ref_z, model_z = map(math.radians, (xml_zenith, z))
    angle = math.acos(max(-1.0, min(1.0, math.cos(ref_z) * math.cos(model_z)
        + math.sin(ref_z) * math.sin(model_z) * math.cos(math.radians(da)))))
    return dz, da, math.degrees(angle), abs(3 * math.tan(ref_z) / math.tan(model_z) - 3)


def diagnose(db, spa):
    scenes = db.execute("""SELECT crs,center_east_m,center_north_m,captured_at_utc,sun_zenith_deg,
        sun_azimuth_grid_deg-true_north_grid_bearing_deg,version_id,tile_metadata_path,scene_item_id,candidate_id
        FROM (SELECT s.*,o.candidate_id,o.center_east_m,o.center_north_m,o.sun_zenith_deg,o.sun_azimuth_grid_deg,
              o.true_north_grid_bearing_deg,row_number() OVER (PARTITION BY s.scene_id ORDER BY o.observation_id) AS rn
              FROM observation o JOIN scene s USING(scene_id)
              WHERE o.sun_zenith_deg IS NOT NULL AND o.sun_azimuth_grid_deg IS NOT NULL) WHERE rn=1
        ORDER BY captured_at_utc,version_id,tile_metadata_path""")
    for crs, x, y, captured, zenith, azimuth, version, path, scene, candidate in scenes:
        lon, lat = geographic_transform(crs)(x, y)
        table, xml = datetime.fromisoformat(captured), xml_time(path)
        cat, source = (solar(spa, stamp, lon, lat, zenith, azimuth) for stamp in (table, xml))
        gap = (table - xml).total_seconds()
        if abs(gap) <= 1:
            flag = "clock_agreement" if cat[2] <= 0.1 else "angle_grid_conflict_with_both_clocks"
        elif source[2] <= 0.05 and cat[2] - source[2] >= 0.07:
            flag = "angle_grid_supports_native_xml_clock"
        elif cat[2] <= 0.05 and source[2] - cat[2] >= 0.07:
            flag = "angle_grid_supports_inherited_catalog_clock"
        else:
            flag = "clock_conflict_unresolved"
        yield dict(version_id=version, scene_item_id=scene, candidate_id=candidate, source_xml=path,
            inherited_captured_at_utc=captured, native_xml_sensing_time_utc=xml.isoformat(), catalog_minus_xml_seconds=gap,
            native_xml_sun_zenith_deg=zenith, native_xml_sun_azimuth_true_deg=azimuth,
            inherited_spa_minus_xml_zenith_deg=cat[0], inherited_spa_minus_xml_azimuth_deg=cat[1],
            inherited_spa_solar_direction_difference_deg=cat[2], inherited_spa_gap_effect_at_3m=cat[3],
            native_time_spa_minus_xml_zenith_deg=source[0], native_time_spa_minus_xml_azimuth_deg=source[1],
            native_time_spa_solar_direction_difference_deg=source[2], native_time_spa_gap_effect_at_3m=source[3],
            diagnostic=flag)


def supported_replacements(rows):
    passes = defaultdict(list)
    for row in rows:
        if row["diagnostic"] == "angle_grid_supports_native_xml_clock" and abs(row["catalog_minus_xml_seconds"]) >= 30:
            name = row["version_id"].split("_")
            passes[(*name[:5], name[-1])].append(row)
    replacements = {}
    for sample in passes.values():
        native = [datetime.fromisoformat(r["native_xml_sensing_time_utc"]) for r in sample]
        distinct_tiles = {r["version_id"].split("_")[5] for r in sample}
        if len(distinct_tiles) < 2 or (max(native) - min(native)).total_seconds() > 6:
            continue
        for row in sample:
            key = (row["version_id"], row["scene_item_id"])
            if key in replacements and row["native_xml_sensing_time_utc"] != replacements[key]["native_xml_sensing_time_utc"]:
                raise ValueError(f"Multiple source sensing times for {key}")
            replacements[key] = row
    return replacements


def corrected_interpolation(rows, replacements):
    def gap(row, label):
        stamps, heights = [], []
        for side in ("before", "after"):
            prefix = f"{label}_{side}_"
            override = replacements.get((row[prefix + "version_id"], row[prefix + "canonical_l2a_id"]))
            stamp = datetime.fromisoformat(override["native_xml_sensing_time_utc"] if override else row[prefix + "captured_at_utc"])
            stamps.append(stamp)
            heights.append(float(row[prefix + "gap_m"]))
        at = datetime.fromisoformat(row[f"{label}_measurement_at_utc"])
        if not stamps[0] <= at <= stamps[1] or stamps[0] == stamps[1]:
            raise ValueError(f"Clock replacement changes the bracket for {row['week_ending']}")
        weight = (at - stamps[0]).total_seconds() / (stamps[1] - stamps[0]).total_seconds()
        return heights[0] + weight * (heights[1] - heights[0])

    for row in rows:
        if row["timing_status"] != "both_boundaries_interpolated":
            continue
        uses_replacement = any((row[f"{boundary}_{side}_version_id"], row[f"{boundary}_{side}_canonical_l2a_id"]) in replacements
            for boundary in ("previous", "current") for side in ("before", "after"))
        if not uses_replacement:
            continue
        corrected = (gap(row, "previous") - gap(row, "current")) * float(row["circular_area_m2"]) / 0.158987294928
        old = 1000 * float(row["interpolated_visible_kbbl"])
        yield dict(week_ending=row["week_ending"], candidate_id=row["candidate_id"], geometry_mode=row["geometry_mode"],
            inherited_change_bbl=old, xml_time_change_bbl=corrected, difference_bbl=corrected - old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spa-module", type=Path, required=True, help="spa.py from pvlib 0.13.1")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("solar_position_audit_spa", args.spa_module)
    spa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spa)
    with closing(read_only(CONTROLS)) as db:
        rows = list(diagnose(db, spa))
    replacements = supported_replacements(rows)
    with (DERIVED / "shadow_timing_development_pairs.csv").open(newline="", encoding="utf-8") as source:
        interpolated = list(corrected_interpolation(csv.DictReader(source), replacements))
    by_week = defaultdict(float)
    for row in interpolated:
        by_week[row["week_ending"], row["geometry_mode"]] += row["difference_bbl"]
    disagreements = [row for row in rows if abs(row["catalog_minus_xml_seconds"]) > 1]
    report = dict(scene_records=len(rows), distinct_scene_versions=len({r["version_id"] for r in rows}),
        time_conflicts_over_one_second=len(disagreements),
        maximum_absolute_time_conflict_seconds=max(abs(r["catalog_minus_xml_seconds"]) for r in rows),
        diagnostic_counts={name: sum(r["diagnostic"] == name for r in rows) for name in sorted({r["diagnostic"] for r in rows})},
        inherited_median_absolute_zenith_disagreement_deg=median(abs(r["inherited_spa_minus_xml_zenith_deg"]) for r in rows),
        inherited_median_absolute_azimuth_disagreement_deg=median(abs(r["inherited_spa_minus_xml_azimuth_deg"]) for r in rows),
        native_clock_replacements_supported_by_two_tiles=len(replacements), development_interpolated_tank_weeks_changed=len(interpolated),
        maximum_absolute_interpolated_weekly_difference_bbl=max(map(abs, by_week.values()), default=0.0),
        spa_reference=SPA_URL, spa_implementation=PVLIB_URL, spa_module_sha256=hashlib.sha256(args.spa_module.read_bytes()).hexdigest(),
        interpretation="A representative contour is tested per archived scene record. Geometric solar position uses "
            "native projection coordinates and each timestamp independently. Thresholds mark engineering conflicts. "
            "A native XML clock is carried into a separate calculation when two tiles of a pass support the native time "
            "with at most six seconds separation and both inherited offsets are at least 30 seconds. "
            "The interpolation comparison keeps the existing observation pairs and fitted roof levels.")
    export("s2_sun_clock_audit", rows)
    if replacements:
        export("s2_sensing_clock_supported_replacements", list(replacements.values()))
    if interpolated:
        export("s2_sensing_clock_interpolation_recheck", interpolated)
    write(DERIVED / "s2_sun_clock_audit_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
