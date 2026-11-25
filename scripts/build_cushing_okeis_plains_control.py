"""Replay the Plains operator's 25C-6 liquid-level observation on the two possible roofs."""

import hashlib
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone

from shapely.geometry import Point

from build_cushing_inspection_b8 import recover
from build_cushing_inspection_controls import LOCAL_TIME, source_image_candidates
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_timing_measurements import fit_additional
from build_weekly_shadow_panel import RESOLVED, SOURCE, read_only
from build_weekly_shadow_validation import DATED
from collect_landsat import LANDSAT, catalog
from collect_public import ROOT, write

STEM = "cushing_okeis_plains_25c6"
EQUIPMENT = "25C6"
DAY = date(2026, 3, 28)
REPORT = "5018336167410780747"
REPORT_HASH = "8f50fa5001d11fa49f467d379e7534621607f140ffca6003628640a903762ff9"
LEVEL_M = (32 * 12 + 5) * .0254
DESIGN_SHELL_M = 48 * .3048


def operator_record():
    source, = [r for r in read("cushing_okeis_documents") if r["document_id"] == REPORT]
    if hashlib.sha256((ROOT / source["local_pdf"]).read_bytes()).hexdigest() != REPORT_HASH:
        raise ValueError("The visually verified Plains inspection report has changed")
    permit, = [r for r in read("permit_tanks") if r["document_id"] == "8795" and r["tank_id"] == EQUIPMENT]
    if (permit["facility_id"], permit["roof_type"], float(permit["capacity_bbl"])) != ("7955", "EFR", 250_000):
        raise ValueError("The operator and regulator records refer to different equipment")
    candidates = {r["candidate_id"]: Point(float(r["longitude"]), float(r["latitude"])) for r in read("tank_engineering_registry")
                  if r["documented_terminal_id"] == "cushing_connect" and EQUIPMENT in json.loads(r["possible_equipment_ids_json"] or "[]")}
    if len(candidates) != 2:
        raise ValueError("The engineering registry no longer has the two independently mapped 25C-6 candidates")
    row = dict(equipment_id=EQUIPMENT, operator_equipment_id="25C-6", terminal_id="cushing_connect",
               state_facility_id=permit["facility_id"], reporting_permit="2020-0308-TVR2",
               source_document_id=REPORT, source_pdf=source["local_pdf"], source_url=source["document_url"], source_pdf_sha256=REPORT_HASH,
               source_form_pdf_page=22, source_field_notes_pdf_page=23, source_listing_timestamp=source["created_or_received_at"],
               source_local_date=DAY.isoformat(), source_local_time=None, level_raw="32 ft 5 in", liquid_level_m=LEVEL_M,
               level_datum_reported=False, roof_type="EFR_pontoon", observed_floating_state_basis="completed_secondary_seal_measurement",
               operator_shell_height_ft=48, operator_diameter_ft=194, nominal_capacity_bbl=250_000,
               permit_reference_url=permit["source_url"], permit_reference_page=permit["pdf_page"],
               possible_candidate_ids_json=json.dumps(sorted(candidates)), individual_contour_verified=False,
               observed_volume_bbl=None, admitted_synchronized_roof_label=False)
    return row, candidates


def scenes(points, day=DAY):
    rows = []
    for ident, (local_day, stamp, sensor, cloud, path, footprint) in source_image_candidates().items():
        offset = (local_day - day).days
        covered = sorted(k for k, p in points.items() if footprint.covers(p))
        if abs(offset) <= 2 and covered:
            rows.append(dict(catalog_id=ident, sensor=sensor, local_date=local_day.isoformat(), scene_time_utc=stamp,
                             scene_minus_inspection_days=offset, catalog_cloud_percent=cloud, source_catalog=str(path.relative_to(ROOT)),
                             covered_candidate_ids_json=json.dumps(covered)))
    return sorted(rows, key=lambda r: (r["scene_time_utc"], r["catalog_id"]))


def sentinel(rows, geometry, *, stem=STEM):
    masks, wanted = {}, {}
    catalogs = {p: {i["id"]: i for i in json.loads((ROOT / p).read_bytes())["features"]}
                for p in {r["source_catalog"] for r in rows if r["sensor"] == "Sentinel-2 L1C"}}
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        for r in rows:
            if r["sensor"] != "Sentinel-2 L1C":
                continue
            item = catalogs[r["source_catalog"]][r["catalog_id"]]
            version = item["properties"]["s2:product_uri"]
            for tank in json.loads(r["covered_candidate_ids_json"]):
                if tank not in geometry:
                    continue
                mask = db.execute("""SELECT *, scene_item_id AS canonical_l2a_id, l1c_product_name AS version_id
                                    FROM sentinel2_l1c_tank_quality WHERE candidate_id=? AND l1c_product_name=?""",
                                  (tank, version)).fetchone()
                if mask is None:
                    raise ValueError(f"No exact source mask for {tank}/{version}")
                row = dict(mask, study_phase="holdout")
                key = tank, version, row["canonical_l2a_id"]
                wanted[key] = row
                masks[key] = {k: row[k] for k in ("opaque_roof_cells", "cirrus_roof_cells", "snow_ice_roof_cells",
                                                "optical_cells_valid_in_all_bands", "optical_touched_cells_expected",
                                                "red_detector_covered_roof_cells", "red_detector_ambiguous_roof_cells",
                                                "red_detector_center_ids_json", "red_detector_with_view_options_json")}
    fits, _ = fit_additional("holdout", f"{stem}_sentinel_native", {k: {DATED} for k in wanted}, wanted, geometry, reuse_pixels=False)
    return [dict(sensor="Sentinel-2 L1C", product_id=version, candidate_id=tank, captured_at_utc=wanted[tank, version, item]["captured_at_utc"],
                 **masks[tank, version, item]) | fit for (_, tank, version, item), fit in fits.items()]


def landsat(rows, geometry):
    covered = {r["catalog_id"]: set(json.loads(r["covered_candidate_ids_json"])) for r in rows if r["sensor"] == "Landsat 8/9 OLI"}
    exact = {r["id"]: r for r in catalog(DAY.year, "usgs") if r["id"] in covered}
    sources, fits, _ = recover(covered, exact, store=LANDSAT / STEM, native=DERIVED / f"{STEM}_landsat_native.sqlite",
                              purpose="retrospective_plains_liquid_level", dated_geometry=geometry, geometry_basis="dated_2022_lidar")
    export(f"{STEM}_landsat_sources", sources)
    captured = {r["product_id"]: r["captured_at_utc"] for r in sources}
    return [dict(r, sensor="Landsat 8/9 OLI", captured_at_utc=captured[r["product_id"]]) for r in fits]


def main():
    record, points = operator_record()
    geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry")
                if r["candidate_id"] in points and r["modern_input_status"] == "available" and r["latest_engineering_date"] < DAY.isoformat()}
    if geometry.keys() != points.keys():
        raise ValueError("The inspection candidates lack usable dated LiDAR geometry")
    rows = scenes(points)
    export(f"{STEM}_ground_record", [record])
    export(f"{STEM}_scene_catalog", rows)
    with ThreadPoolExecutor(max_workers=2) as pool:
        s2 = pool.submit(sentinel, rows, geometry)
        b8 = pool.submit(landsat, rows, geometry)
        fits = s2.result() + b8.result()
    earliest = datetime.combine(DAY, time(), LOCAL_TIME).astimezone(timezone.utc)
    latest = datetime.combine(DAY + timedelta(days=1), time(), LOCAL_TIME).astimezone(timezone.utc)
    checks = []
    for r in fits:
        t = datetime.fromisoformat(r["captured_at_utc"])
        gap = r["estimated_gap_m"] if r["fit_status"] == RESOLVED else None
        g = geometry[r["candidate_id"]]
        shell = float(g["prior_rim_z_p95_native"]) - float(g["prior_ground_z_p50_native"])
        checks.append(dict(equipment_id=EQUIPMENT, candidate_id=r["candidate_id"], sensor=r["sensor"], product_id=r["product_id"],
                           capture_at_utc=t.isoformat(), same_local_day=t.astimezone(LOCAL_TIME).date() == DAY,
                           minimum_possible_hours_from_inspection=max(0., (earliest-t).total_seconds(), (t-latest).total_seconds())/3600,
                           maximum_possible_hours_from_inspection=max(abs((t-earliest).total_seconds()), abs((t-latest).total_seconds()))/3600,
                           fit_status=r["fit_status"], fitted_gap_m=gap,
                           fit_near_min_m=r.get("near_optimal_gap_min_m") if gap is not None else None,
                           fit_near_max_m=r.get("near_optimal_gap_max_m") if gap is not None else None,
                           operator_liquid_level_m=LEVEL_M, conditional_shell_minus_level_design_m=DESIGN_SHELL_M-LEVEL_M,
                           conditional_shell_minus_level_lidar_m=shell-LEVEL_M,
                           individual_contour_verified=False, simultaneous_measurement_verified=False,
                           admitted_roof_gap_error=False))
    for name, part in (("sentinel_fits", [r for r in fits if r["sensor"] == "Sentinel-2 L1C"]),
                       ("landsat_fits", [r for r in fits if r["sensor"] == "Landsat 8/9 OLI"]), ("comparisons", checks)):
        if part:
            keys = dict.fromkeys(k for r in part for k in r)
            export(f"{STEM}_{name}", [{k: r.get(k) for k in keys} for r in part])
    summary = dict(equipment_id=EQUIPMENT, inspection_date=DAY.isoformat(), observed_liquid_level_m=LEVEL_M,
                   identity_candidates=len(points), scene_catalog_products=len(rows), fits=len(checks),
                   fit_status=Counter(r["fit_status"] for r in checks), same_day_fits=sum(r["same_local_day"] for r in checks),
                   same_day_resolved_fits=sum(r["same_local_day"] and r["fitted_gap_m"] is not None for r in checks),
                   admitted_individual_roof_height_errors=0)
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
