"""Measure dated 2025 NAIP roof shadows with the unchanged 2019 physical fit."""
import hashlib
import importlib.util
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import ExitStack, closing
from datetime import datetime
from statistics import median

import rasterio

from audit_shadow_precision import describe
from build_naip_2019_roof_reference import SOURCE_SQL, find_asset, fit
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import RAW, ROOT, get, write

SPA = RAW / "solar_position" / "pvlib_0_13_1_spa.py"
SPA_URL = "https://raw.githubusercontent.com/pvlib/pvlib-python/v0.13.1/pvlib/spa.py"
SPA_SHA256 = "ad7762ccca5fd9611ef9e2a9b1e37f95fe4c5b31628616b2ad3e84ea1e5f9202"
RESOLVED = "resolved_aerial_shadow"


def solar_module():
    if hashlib.sha256(get(SPA_URL, SPA)).hexdigest() != SPA_SHA256:
        raise ValueError("Solar implementation differs from the checked 2019 version")
    spec = importlib.util.spec_from_file_location("naip_2025_roof_spa", SPA)
    spa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(spa)
    return spa


def aerial_observations():
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        dates = [r[0] for r in db.execute("""SELECT DISTINCT source_acquisition_date FROM naip_tank_source_clock
            WHERE acquisition_year=2025 AND contour_fraction>=0.99 AND json_array_length(dated_asset_paths_json)>0
            ORDER BY source_acquisition_date""")]
        tanks = [dict(row, reference_date=day) for day in dates for row in db.execute(SOURCE_SQL, (day, day))]
    spa = solar_module()
    rows = []
    with ExitStack() as stack:
        paths = {p for row in tanks for p in json.loads(row["dated_asset_paths_json"])}
        sources = {p: stack.enter_context(rasterio.open(ROOT / p)) for p in sorted(paths)}
        for tank in tanks:
            asset = find_asset(tank, sources)
            if asset is None or not tank["shell_m"]:
                raise ValueError(f"Missing dated aerial crop or geometry for {tank['candidate_id']}")
            crop = asset[-1]
            rows.append(dict(tank, measurement_coordinate_basis="local_tangent_metres_from_lidar_centre"
                if sources[asset[1]].crs.is_geographic else "projected_source_metres",
                native_pixel_north_south_m=abs(crop[4].e), **fit(tank, asset, spa, sources)))
    return rows


def compare_radar(observations):
    aerial = {(r["candidate_id"], r["reference_date"], str(r["source_polygon_id"])): r for r in observations}
    radar = {(r["candidate_id"], r["product_id"]): r for r in read("umbra_sicd_all_roof_arc_fits")}
    for match in read("umbra_cushing_reference_matches"):
        if match["reference_kind"] != "naip_source_polygon":
            continue
        a = aerial.get(tuple(match[k] for k in ("candidate_id", "reference_day", "reference_id")))
        r = radar.get((match["candidate_id"], match["product_id"]))
        if not a or not r:
            continue
        if (a["documented_roof_type"] != r["roof_type"] or r["roof_type"] != "EFR"
                or not math.isclose(a["shell_m"], float(r["lidar_rim_minus_ground_m"]), rel_tol=0, abs_tol=1e-6)
                or any(datetime.fromisoformat(a[left]) != datetime.fromisoformat(match[right])
                       for left, right in (("start_utc", "reference_start_utc"), ("end_exclusive_utc", "reference_end_utc")))):
            raise ValueError(f"Aerial and radar identities differ for {match['candidate_id']}")
        gap = a["best_gap_m"]
        height = a["shell_m"]-gap if gap is not None else None
        result = dict(match, aerial_shadow_status=a["shadow_reference_status"],
                      aerial_shadow_gap_m=gap, aerial_roof_height_m=height, aerial_shadow_source=a["source_raster"])
        for method, height_key, status_key in (("local", "aligned_height_above_ground_m", "roof_use_status"),
                                               ("scene", "scene_registered_height_m", "scene_registered_use_status")):
            radar_height = float(r[height_key]) if r[height_key] else None
            result.update({f"radar_{method}_status": r[status_key], f"radar_{method}_roof_height_m": radar_height,
                f"both_aerial_radar_{method}_resolved": a["shadow_reference_status"] == RESOLVED and r[status_key] == "resolved_arc_diagnostic",
                f"radar_{method}_minus_aerial_height_m": radar_height-height if None not in (radar_height, height) else None})
        yield result


def radar_comparison_summary(rows, method):
    accepted = [r for r in rows if r[f"both_aerial_radar_{method}_resolved"]]
    field = f"radar_{method}_minus_aerial_height_m"
    before_after = defaultdict(dict)
    for row in accepted:
        key = row["candidate_id"], row["reference_day"], row["reference_id"]
        side = row["radar_side"]
        if side in before_after[key]:
            raise ValueError(f"More than one selected radar capture on the same side of {key}")
        before_after[key][side] = row[field]
    bracketed = [r for r in before_after.values() if r.keys() == {"before", "after"}]
    close = [r for r in accepted if float(r["maximum_separation_hours"]) <= 72]
    return dict(disagreement=describe([r[field] for r in accepted]),
        within_72_hours=describe([r[field] for r in close]),
        radar_higher_within_72_hours=sum(r[field] > 0 for r in close),
        distinct_capture_pairs=len({(r["product_id"], r["reference_day"]) for r in accepted}),
        aerial_with_resolved_radar_before_and_after=len(bracketed),
        radar_higher_on_both_sides=sum(min(r.values()) > 0 for r in bracketed))


def main():
    rows = aerial_observations()
    pairs = list(compare_radar(rows))
    export("naip_2025_roof_shadow_reference", rows)
    export("naip_2025_umbra_roof_comparisons", pairs)
    resolved = [r for r in rows if r["documented_roof_type"] == "EFR" and r["shadow_reference_status"] == RESOLVED]
    stats = {method: radar_comparison_summary(pairs, method) for method in ("local", "scene")}
    report = dict(aerial_records=len(rows), distinct_tanks=len({r["candidate_id"] for r in rows}),
        source_dates=sorted({r["reference_date"] for r in rows}),
        roof_types=dict(Counter(r["documented_roof_type"] for r in rows)), resolved_efr=len(resolved),
        fixed_roof_false_passes=sum(r["shadow_reference_status"] == RESOLVED and r["documented_roof_type"] != "EFR" for r in rows),
        median_gap_m=median(r["best_gap_m"] for r in resolved) if resolved else None,
        median_gap_span_from_aerial_time_trials_m=median(r["time_only_gap_max_m"]-r["time_only_gap_min_m"] for r in resolved)
        if resolved else None,
        radar_pairs=len(pairs), radar_matches_with_maximum_gap_at_most_72_hours=sum(float(r["maximum_separation_hours"]) <= 72 for r in pairs),
        minimum_radar_gap_hours=min(float(r["minimum_separation_hours"]) for r in pairs), joint_diagnostics=stats,
        spa_sha256=SPA_SHA256,
        interpretation="The USDA flight interval is linked to the state 2025 NAIP orthophoto crop and fixed LiDAR geometry. "
        "WCS geographic sample coordinates are converted to local tangent metres without resampling the pixel values. "
        "The unchanged 2019 optical model measures the shadow of a floating roof. Orthorectification and camera look remain unmeasured. "
        "Radar comparisons across days include possible changes in real inventory. No EIA outcomes or prices are used.")
    write(DERIVED / "naip_2025_roof_shadow_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
