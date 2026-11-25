"""Match Sentinel-1 product clocks and footprints to independent Cushing reference acquisitions."""
import csv
import json
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime

from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import write, write_csv
from collect_s1_roof_catalog import GRD, SLC


REFERENCE_SQL = """
WITH refs AS (
    SELECT candidate_id,'naip_source_polygon' AS reference_kind,source_polygon_id AS reference_id,
        source_acquisition_date AS day,start_utc AS begin,end_exclusive_utc AS finish,
        contour_fraction AS fraction,minimum_nominal_gsd_m AS gsd,dated_asset_paths_json AS assets,
        NULL AS measured_gap_m,source_file AS clock_file
    FROM naip_tank_source_clock WHERE json_array_length(dated_asset_paths_json)>0
    UNION ALL
    SELECT candidate_id,'ok_aerial_tile_majority',tile_id,tile_majority_date,
        tile_majority_date||'T'||tile_time_raw||':00Z',
        strftime('%Y-%m-%dT%H:%M:%SZ',tile_majority_date||'T'||tile_time_raw||':00Z','+1 minute'),
        valid_fraction,0.15,json_array(asset_path),NULL,replace(asset_path,'/image.tif','/item.json')
    FROM ok_aerial_tank_image_coverage WHERE length(tile_time_raw)=5
    UNION ALL
    SELECT candidate_id,'lidar_return_interval',dataset||':'||point_source_id,acquisition_date_utc,
        first_return_at_utc,last_return_at_utc,1.0,NULL,json_array(source_directory),
        rim_minus_center_plane_native_z,'data/derived/lidar_roof_elevation_reference.csv'
    FROM lidar_roof_elevation_reference WHERE dataset='OK_Statewide_2_D22'
        AND rim_minus_center_plane_native_z IS NOT NULL AND first_return_at_utc IS NOT NULL
        AND last_return_at_utc IS NOT NULL
)
SELECT r.* FROM refs r JOIN tank_engineering_registry t USING(candidate_id)
WHERE t.documented_roof_type='EFR' AND t.first_seen<=r.day AND r.fraction>=0.99
ORDER BY r.day,r.candidate_id,r.reference_kind,r.reference_id
"""


def read(stem):
    with (DERIVED / f"{stem}.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def export(stem, rows):
    if not rows:
        raise ValueError(f"No rows for {stem}")
    fields = tuple(rows[0])
    write_csv(DERIVED / f"{stem}.csv", fields, ([row[field] for field in fields] for row in rows))


def nearest(rows, products):
    by_tank = defaultdict(list)
    for radar in products:
        start, stop = (datetime.fromisoformat(radar[field]) for field in ("capture_start_utc", "capture_end_utc"))
        for tank in json.loads(radar["covered_candidate_ids_json"]):
            by_tank[tank].append((radar, start, stop))
    for tank, kind, reference, day, first, last, fraction, gsd, assets, gap, clock_file in rows:
        begin, finish = map(datetime.fromisoformat, (first, last))
        if finish < begin:
            raise ValueError(f"Reference interval is reversed for {tank}/{day}")
        closest = {}
        for radar, start, stop in by_tank[tank]:
            maximum = max(abs((start - finish).total_seconds()), abs((stop - begin).total_seconds())) / 3600
            if maximum > 14 * 24:
                continue
            minimum = max(0, (start - finish).total_seconds(), (begin - stop).total_seconds()) / 3600
            side = "before" if stop < begin else "after" if start > finish else "overlap"
            key = (radar["product_type"], side)
            rank = (maximum, radar["product_id"])
            if key not in closest or rank < closest[key][0]:
                closest[key] = (rank, dict(candidate_id=tank, reference_kind=kind, reference_id=reference,
                    reference_day=day, reference_start_utc=first, reference_end_utc=last,
                    reference_contour_fraction=fraction, reference_gsd_m=gsd, reference_gap_m=gap,
                    reference_assets_json=assets, reference_clock_file=clock_file, radar_side=side,
                    product_type=radar["product_type"], product_id=radar["product_id"], product_name=radar["product_name"],
                    radar_start_utc=radar["capture_start_utc"], radar_end_utc=radar["capture_end_utc"],
                    direction=radar["direction"], relative_orbit=radar["relative_orbit"],
                    minimum_separation_hours=round(minimum, 6), maximum_separation_hours=round(maximum, 6)))
        yield from (data for _, data in closest.values())


def pilot_list(rows, products):
    reasons = defaultdict(set)
    optical = {r["week_ending"] for r in read("weekly_shadow_validation_weeks")
               if r["geometry_mode"] == "dated_2011_lidar" and int(r["resolved_efr_pairs"]) > 0}
    for row in read("s1_cushing_weekly_catalog_coverage"):
        if row["study_phase"] != "validation" or row["week_ending"] in optical or row["grd_common_relative_track"] != "True":
            continue
        for side in ("previous", "current"):
            for identity in json.loads(row[f"grd_{side}_product_ids_json"]):
                reasons[identity].add(f"optical_gap_{row['week_ending']}_{side}")

    by_scene = defaultdict(set)
    for row in rows:
        if row["product_type"] != GRD:
            continue
        if row["reference_kind"] == "lidar_return_interval":
            if float(row["maximum_separation_hours"]) <= 7 * 24:
                reasons[row["product_id"]].add(f"lidar_{row['reference_day']}_{row['radar_side']}")
        elif float(row["maximum_separation_hours"]) <= 24:
            by_scene[row["product_id"], row["reference_day"]].add(row["candidate_id"])
    for (identity, day), tanks in by_scene.items():
        reasons[identity].add(f"aerial_{day}_{len(tanks)}_contours_within_24h")
    return [dict(p, selection_reasons="|".join(sorted(reasons[p["product_id"]])))
            for p in products if p["product_id"] in reasons]


def main():
    with closing(read_only(SOURCE)) as db:
        reference = db.execute(REFERENCE_SQL).fetchall()
    products = [r for r in read("s1_cushing_catalog") if r["product_type"] in {SLC, GRD}]
    matches = list(nearest(reference, products))
    selected = pilot_list(matches, products)
    export("s1_reference_nearest_products", matches)
    export("s1_grd_native_pilot_manifest", selected)
    best_aerial = defaultdict(set)
    for row in matches:
        if row["reference_kind"] != "lidar_return_interval" and float(row["maximum_separation_hours"]) <= 24:
            best_aerial[row["reference_day"], row["product_type"], row["product_id"]].add(row["candidate_id"])
    lidar_min = {}
    for kind in (SLC, GRD):
        lidar_min[kind] = min(r["minimum_separation_hours"] for r in matches
                              if r["reference_kind"] == "lidar_return_interval" and r["product_type"] == kind)
    summary = dict(reference_records=len(reference), match_rows=len(matches),
        references_by_kind=dict(Counter(row[1] for row in reference)), pilot_grd_products=len(selected),
        aerial_products_within_24h=[dict(reference_day=day, product_type=kind, product_id=identity, tanks=len(tanks))
                                  for (day, kind, identity), tanks in sorted(best_aerial.items())],
        nearest_lidar_minimum_separation_hours_by_type=lidar_min,
        interpretation="Footprints cover contour centres and reference rasters cover at least 99 percent of the contour. "
        "Matches preserve the whole reference acquisition interval and nearest before/after products within 14 days. "
        "The 2019 aerial reference has image clocks and coverage, while a roof-gap value requires separate measurement. "
        "The engineering comparison uses no EIA values or equity outcomes.")
    write(DERIVED / "s1_reference_matches_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
