"""Archive Landsat windows around the 2024 Rose Rock NRC inventory notification."""

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from statistics import median
from urllib.parse import urlsplit

import numpy as np
import pdfplumber
from shapely.geometry import Point, shape

from audit_cushing_b8_archive import scene_file
from audit_landsat_c2_alternative_access import copernicus, probe
from audit_sentinel2_alternate_registration import packet_inputs
from build_cushing_inspection_b8 import recover
from build_cushing_icesat2_controls import CMR, COLLECTION, query
from build_cushing_nrc_incidents import STEM as NRC
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_timing_measurements import fit_additional
from build_weekly_shadow_panel import RESOLVED, SOURCE, read_only
from build_weekly_shadow_validation import DATED, legacy_geometry
from collect_landsat import LANDSAT, catalog
from collect_public import RAW, ROOT, get, write
from collect_s2_spectral_detector_masks import BANDS, collect, jobs
from check_weekly_multiband_shadow import METHODS, spectral_kernels, spectral_profiles
from fit_shadow_geometry import gap_fit_status
from stage_landsat_c2_b8_inputs import expected_sha512

STEM = "cushing_nrc_rose_rock_landsat"
STORE = LANDSAT / STEM
RENEWAL_URL = "https://applications.deq.ok.gov/permitspublic/storedpermits/8063.pdf"


def inputs(event):
    stamp = datetime.fromisoformat(event["incident_local_time"])
    candidates = {r["product_id"]: r for r in read(f"{NRC}_rose_rock_scenes") if r["sensor"] == "Landsat 8/9 OLI"}
    scenes = {r["id"]: r for r in catalog(stamp.year, "usgs") if r["id"] in candidates}
    if scenes.keys() != candidates.keys():
        raise ValueError("Some NRC Landsat candidates do not have exact USGS product records")
    roofs = {r["candidate_id"]: Point(float(r["longitude"]), float(r["latitude"])) for r in read("tank_engineering_registry")
             if r["cohort_id"].startswith("rose_rock_north_") and r["documented_roof_type"] == "EFR"
             and r["first_seen"] and r["first_seen"] <= stamp.date().isoformat()}
    if len(roofs) != event["roof_candidates"] or any(len(roofs) != int(r["roof_candidates_covered"]) for r in candidates.values()):
        raise ValueError("The dated Rose Rock geometry differs from the NRC scene inventory")
    coverage, regional_start = {}, {}
    for product, item in scenes.items():
        footprint = shape(item["geometry"])
        coverage[product] = sum(footprint.covers(p) for p in roofs.values())
        acquisition = datetime.fromisoformat(item["properties"]["datetime"])
        start = datetime.fromisoformat(candidates[product]["scene_capture_utc"])
        if not timedelta(0) <= acquisition - start < timedelta(minutes=1):
            raise ValueError(f"The USGS centre time is inconsistent with the regional start time for {product}")
        regional_start[product] = start.isoformat()
    # The footprint can follow the edge of valid image data instead of the B8 raster rectangle.
    # Fetch every catalog-corner candidate, then measure actual pixel and QA coverage.
    return stamp, scenes, {product: set(roofs) for product in scenes}, coverage, regional_start


def optical_controls(scene_rows, fits, optical=None):
    if optical is None:
        geometry = legacy_geometry()
        optical = [dict(r, diameter_m=geometry[r["candidate_id"]]["diameter_m"])
                   for r in read(f"{NRC}_rose_rock_sentinel2_fits") if r["fit_status"] == RESOLVED]
    by_roof = defaultdict(list)
    for row in optical:
        if row["fit_status"] == RESOLVED:
            by_roof[row["candidate_id"]].append(row)
    times = {r["product_id"]: datetime.fromisoformat(r["captured_at_utc"]) for r in scene_rows}
    rows = []
    for fit in fits:
        if fit["fit_status"] != RESOLVED:
            continue
        t = times[fit["product_id"]]
        for ref in by_roof[fit["candidate_id"]]:
            hours = (t - datetime.fromisoformat(ref["captured_at_utc"])).total_seconds() / 3600
            if abs(hours) <= 48 and abs(fit["diameter_m"] - float(ref["diameter_m"])) < 1e-6:
                b04, b8 = float(ref["estimated_gap_m"]), fit["estimated_gap_m"]
                rows.append(dict(candidate_id=fit["candidate_id"], landsat_product=fit["product_id"],
                    landsat_capture_utc=t.isoformat(), sentinel_version=ref["version_id"], sentinel_capture_utc=ref["captured_at_utc"],
                    landsat_minus_sentinel_hours=hours, landsat_gap_m=b8, sentinel_gap_m=b04, landsat_minus_sentinel_gap_m=b8-b04,
                    admitted_nrc_inventory_label=False))
    return rows


def matching_sentinel_geometry(geometry):
    versions = [r["product_id"] for r in read(f"{NRC}_rose_rock_scenes") if r["sensor"] == "Sentinel-2 L1C"]
    sources = {}
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        for roof in sorted(geometry):
            for version in versions:
                source = db.execute("""SELECT candidate_id, scene_item_id AS canonical_l2a_id, l1c_product_name AS version_id,
                       captured_at_utc, source_directory, tile_metadata_path
                       FROM sentinel2_l1c_tank_quality WHERE candidate_id=? AND l1c_product_name=?""", (roof, version)).fetchone()
                if source:
                    sources[roof, version, source["canonical_l2a_id"]] = dict(source, study_phase="holdout")
    if len(sources) != len(geometry) * len(versions):
        raise ValueError("Missing exact Sentinel-2 roof-window inputs for the common LiDAR geometry")
    measured, _ = fit_additional("holdout", f"{STEM}_lidar_2022_sentinel2_native",
                                 {k: {DATED} for k in sources}, sources, geometry, reuse_pixels=False)
    rows = []
    for (_, *key), fit in measured.items():
        source = sources[tuple(key)]
        rows.append(source | {field: fit.get(field) for field in ("fit_status", "diameter_m", "estimated_gap_m",
                    "near_optimal_gap_min_m", "near_optimal_gap_max_m", "measurement_source", "observation_id")}
                    | dict(geometry_basis="retrospective_2022_lidar", admitted_nrc_inventory_label=False))
    return rows


def permit_renewal():
    path = RAW / "documents" / "source_pdfs" / "8063.pdf"
    raw = get(RENEWAL_URL, path)
    rows = {}
    with pdfplumber.open(path) as doc:
        if "2020-0213-TVR2" not in (doc.pages[0].extract_text() or ""):
            raise ValueError("Unexpected Rose Rock permit renewal")
        for page in (doc.pages[2], doc.pages[3]):
            for line in page.extract_text().splitlines():
                match = re.fullmatch(r"T-(\d+) (\d+) EFR ([\d,]+) [\d,]+ (\d{4})", line)
                if match:
                    equipment, point, capacity, year = match.groups()
                    if equipment != point or equipment in rows:
                        raise ValueError(f"Ambiguous Rose Rock equipment {equipment}")
                    rows[equipment] = dict(equipment_id=equipment, reported_capacity_bbl=int(capacity.replace(",", "")),
                        construction_year=int(year), source_pdf_page=page.page_number, source_url=RENEWAL_URL,
                        source_file=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(raw).hexdigest())
    return rows


def candidate_roofs(event, roofs, optical, landsat, renewal):
    """Preserve all contours and screen only by the documented capacity class."""
    registry = {r["candidate_id"]: r for r in read("tank_engineering_registry") if r["candidate_id"] in roofs}
    provenance = {r["candidate_id"]: r for r in read("north_farm_contour_reconciliation") if r["candidate_id"] in roofs}
    if registry.keys() != roofs or provenance.keys() != roofs:
        raise ValueError("Missing engineering evidence for a Rose Rock contour")
    result = []
    for roof in sorted(roofs):
        r, p = registry[roof], provenance[roof]
        capacity = float(r["reported_capacity_max_bbl"])
        equipment = json.loads(p["possible_equipment_ids_json"])
        if (p["facility_id"] != "7238" or float(p["reported_capacity_min_bbl"]) != capacity
                or r["documented_roof_type"] != "EFR" or r["cohort_id"] != p["cohort_id"]
                or not equipment or any(renewal.get(e, {}).get("reported_capacity_bbl") != capacity for e in equipment)):
            raise ValueError(f"Inconsistent DEQ cohort for {roof}")
        valid_optical = [x for x in optical if x["candidate_id"] == roof and x["fit_status"] == RESOLVED]
        valid_landsat = [x for x in landsat if x["candidate_id"] == roof and x["fit_status"] == RESOLVED]
        result.append(dict(candidate_id=roof, cohort_id=r["cohort_id"], facility_id=p["facility_id"],
            possible_equipment_ids_json=p["possible_equipment_ids_json"], reported_capacity_bbl=capacity,
            reported_capacity_basis=r["reported_capacity_basis"], permit_document_id=p["document_id"],
            permit_equipment_pages=p["equipment_pdf_pages"], permit_source_url=p["engineering_source_url"],
            renewal_source_pages_json=json.dumps(sorted({renewal[e]["source_pdf_page"] for e in equipment})),
            renewal_source_url=RENEWAL_URL,
            reported_inventory_bbl=event["reported_amount_in_tank_bbl"],
            capacity_at_least_reported_inventory=capacity >= event["reported_amount_in_tank_bbl"],
            sentinel2_passing_versions_json=json.dumps([x["version_id"] for x in valid_optical]),
            landsat_passing_versions_json=json.dumps([x["product_id"] for x in valid_landsat]),
            has_2022_lidar_optical_attempt=any(x["candidate_id"] == roof for x in optical),
            individual_identity_verified=False, admitted_nrc_inventory_label=False))
    return result


def matched_optical_dates(optical, stamp):
    groups = defaultdict(list)
    for row in optical:
        day = datetime.fromisoformat(row["captured_at_utc"]).astimezone(stamp.tzinfo).date().isoformat()
        groups[day, row["candidate_id"]].append(row)
    rows = []
    for (day, roof), versions in sorted(groups.items()):
        pair = {row["version_id"].split("_")[5]: row for row in versions}
        if len(versions) != 2 or pair.keys() != {"T14SPE", "T14SQE"}:
            raise ValueError(f"Expected two tiles of one Sentinel-2 acquisition for {day}/{roof}")
        a, b = (pair[k] for k in ("T14SPE", "T14SQE"))
        if abs(datetime.fromisoformat(a["captured_at_utc"]) - datetime.fromisoformat(b["captured_at_utc"])) > timedelta(seconds=10):
            raise ValueError(f"The Sentinel-2 tile times differ for {day}/{roof}")
        passed = a["fit_status"] == b["fit_status"] == RESOLVED
        rows.append(dict(candidate_id=roof, local_date=day, both_tiles_passed=passed,
            t14spe_version=a["version_id"], t14sqe_version=b["version_id"], t14spe_status=a["fit_status"],
            t14sqe_status=b["fit_status"], t14spe_gap_m=a["estimated_gap_m"], t14sqe_gap_m=b["estimated_gap_m"],
            t14sqe_minus_t14spe_gap_m=b["estimated_gap_m"]-a["estimated_gap_m"] if passed else None))
    return rows


def spectral_checks(optical, geometry, stem):
    packet, observations, scenes, _ = packet_inputs(f"{stem}_sentinel2_native.sqlite")
    with closing(read_only(packet)) as db:
        kernels = spectral_kernels(db)
    masks = [collect(job) for job in jobs({r["version_id"]: r["manifest_path"] for r in scenes.values()})]
    files = {(r["version_id"], r["band"]): r["local_file"] for r in masks}

    def check(row):
        obs = observations[row["observation_id"]]
        roof, version = row["candidate_id"], row["version_id"]
        if (obs["candidate_id"], obs["version_id"]) != (roof, version):
            raise ValueError(f"Spectral observation differs from saved B04 input {roof}/{version}")
        geom = geometry[roof]
        shell = float(geom["prior_rim_z_p95_native"]) - float(geom["prior_ground_z_p50_native"])
        obs = dict(obs, center_east_m=float(geom["center_east_m"]), center_north_m=float(geom["center_north_m"]),
                   radius_m=float(geom["diameter_m"])/2)
        if row["fit_status"] in ("incomplete_optical_geometry_or_quality", "dated_geometry_outside_chip"):
            status, pixels, profiles = row["fit_status"], None, None
        else:
            status, pixels, profiles, _ = spectral_profiles(packet, obs, scenes[obs["scene_id"]], shell, kernels,
                tuple(files[version, band] for band in BANDS), 0, 0)
        result = dict(candidate_id=roof, version_id=version, captured_at_utc=row["captured_at_utc"],
                      pixel_status=status, common_pixels=pixels)
        for method in METHODS:
            gap = lower = upper = fit_status = None
            if profiles:
                _, gaps, loss, reverse, count, weight = profiles[method]
                score, placebo = loss/weight, reverse/weight
                _, i = np.unravel_index(np.argmin(score), score.shape)
                minimum = score.min()
                near = gaps[score.min(axis=0) <= minimum*(1 + 1/max(1, count-6)) + 1e-10/weight]
                gap, lower, upper = map(float, (gaps[i], near.min(), near.max()))
                fit_status = gap_fit_status(gaps, gap, lower, upper, np.sqrt(minimum),
                                            np.sqrt(score[:, 0].min()), np.sqrt(placebo.min()))
            result.update({f"{method}_{k}": v for k, v in dict(gap_m=gap, near_gap_min_m=lower,
                          near_gap_max_m=upper, fit_status=fit_status).items()})
        if status == "compared_full_pixels" and (
            result["red_gap_m"] != row["estimated_gap_m"] or result["red_fit_status"] != row["fit_status"]
        ):
            raise ValueError(f"Full-pixel B04 replay differs for {roof}/{version}")
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(check, optical)), masks


def spectral_dates(spectral, dates, compatible):
    by_version = {(r["candidate_id"], r["version_id"]): r for r in spectral}
    rows = []
    for day in dates:
        roof = day["candidate_id"]
        left, right = (by_version[roof, day[f"{tile}_version"]] for tile in ("t14spe", "t14sqe"))
        row = dict(candidate_id=roof, local_date=day["local_date"],
                   capacity_at_least_reported_inventory=roof in compatible,
                   t14spe_pixel_status=left["pixel_status"], t14sqe_pixel_status=right["pixel_status"])
        for method in METHODS:
            status = f"{method}_fit_status"
            passed = left[status] == right[status] == RESOLVED
            row[f"{method}_both_tiles_passed"] = passed
            row[f"{method}_t14spe_gap_m"] = left[f"{method}_gap_m"]
            row[f"{method}_t14sqe_gap_m"] = right[f"{method}_gap_m"]
        row["all_three_methods_both_tiles_passed"] = all(row[f"{m}_both_tiles_passed"] for m in METHODS)
        rows.append(row)
    return rows


def original_access(scenes):
    queue = [dict(product_id=product, platform=item["properties"]["platform"], purpose="retrospective_nrc_rose_rock",
                  captured_at_utc=item["properties"]["datetime"], original_b8_filename=f"{product}_B8.TIF",
                  b8_status="awaiting_original_b8", expected_original_b8_sha512=expected_sha512(item["assets"]["pan"]),
                  usgs_url=item["assets"]["pan"]["href"]) for product, item in sorted(scenes.items())]
    archive = STORE / "cdse"
    found = copernicus(queue, store=archive)

    def check(row):
        product = row["product_id"]
        original = scene_file(row, found[product], store=archive) if product in found else {}
        cdse = probe(original["cdse_b8_download_url"]) if original else {}
        usgs = probe(row["usgs_url"])
        relative = urlsplit(row["usgs_url"]).path.removeprefix("/data/")
        aws_url = f"https://usgs-landsat.s3.us-west-2.amazonaws.com/{relative}"
        gcs_url = f"https://storage.googleapis.com/gee-public-data-landsat/{relative}"
        aws, gcs = probe(aws_url), probe(gcs_url)
        return row | original | dict(copernicus_http_status=cdse.get("http_status"),
            copernicus_tiff_signature=cdse.get("tiff_signature"), usgs_http_status=usgs["http_status"],
            usgs_response_host=usgs["response_host"], usgs_tiff_signature=usgs["tiff_signature"],
            aws_url=aws_url, aws_http_status=aws["http_status"], aws_tiff_signature=aws["tiff_signature"],
            gcs_url=gcs_url, gcs_http_status=gcs["http_status"], gcs_tiff_signature=gcs["tiff_signature"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, queue))
    fields = tuple(dict.fromkeys(k for row in rows for k in row))
    export(f"{STEM}_original_access", [{k: row.get(k) for k in fields} for row in rows])


def lidar_catalog_checks():
    sites = (
        ("rose_rock_north", "-96.762,36.014,-96.751,36.023", "2024-06-17T00:00:00Z,2024-06-28T23:59:59Z", 1402673),
        ("grandstaff_north", "-96.760,36.012,-96.752,36.024", "2025-11-17T00:00:00Z,2025-11-22T23:59:59Z", 1381197),
    )
    rows = []
    for site, bounds, period, report in sites:
        url = query(CMR, dict(collection_concept_id=COLLECTION, bounding_box=bounds, temporal=period, page_size=100))
        path = STORE / f"atl03_{site}.json"
        payload = json.loads(get(url, path))
        if payload["hits"] != len(payload["items"]):
            raise ValueError(f"Incomplete ATL03 catalog response for {site}")
        rows.append(dict(site=site, nrc_report_id=report, bounding_box=bounds, search_interval_utc=period,
                         atl03_v7_matching_granules=payload["hits"], query_url=url, source_file=str(path.relative_to(ROOT))))
    export(f"{STEM}_atl03_catalog_checks", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidar-2022", action="store_true", help="Use 2022 roof geometry in the retrospective analysis")
    parser.add_argument("--probe-originals", action="store_true", help="Verify exact original B8 file listings and access")
    parser.add_argument("--spectral", action="store_true", help="Compare B04, B08 and a three-band fit on common pixels")
    args = parser.parse_args()
    if args.spectral and not args.lidar_2022:
        parser.error("--spectral requires --lidar-2022")
    event = json.loads((DERIVED / f"{NRC}_summary.json").read_bytes())["rose_rock_inventory_event"]
    stamp, sources, coverage, footprint_count, regional_start = inputs(event)
    if args.probe_originals:
        original_access(sources)
        lidar_catalog_checks()
    geometry, basis = None, "dated_2011_lidar"
    stem, store = STEM, STORE
    if args.lidar_2022:
        basis = "retrospective_2022_lidar"
        stem = f"{STEM}_lidar_2022"
        store = LANDSAT / stem
        roofs = set().union(*coverage.values())
        geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry")
                    if r["candidate_id"] in roofs and r["modern_input_status"] == "available"
                    and datetime.fromisoformat(r["last_upper_return_at_utc"]) < stamp}
    scene_rows, fits, stats = recover(coverage, sources, store=store, native=DERIVED / f"{stem}_native.sqlite",
                                     purpose="retrospective_nrc_rose_rock", dated_geometry=geometry, geometry_basis=basis)
    for scene in scene_rows:
        scene["stac_footprint_roofs"] = footprint_count[scene["product_id"]]
        scene["regional_catalog_start_utc"] = regional_start[scene["product_id"]]
        scene["hours_from_reported_incident"] = (datetime.fromisoformat(scene["captured_at_utc"]) - stamp).total_seconds()/3600
    before, after = set(), set()
    for scene in scene_rows:
        passed = {r["candidate_id"] for r in fits if r["product_id"] == scene["product_id"] and r["fit_status"] == RESOLVED}
        scene["resolved_roof_heights"] = len(passed)
        (before if scene["hours_from_reported_incident"] < 0 else after).update(passed)
    matched_geometry = matching_sentinel_geometry(geometry) if args.lidar_2022 else []
    controls = optical_controls(scene_rows, fits, matched_geometry if args.lidar_2022 else None)
    export(f"{stem}_scenes", scene_rows)
    if fits:
        export(f"{stem}_fits", fits)
    if controls:
        export(f"{stem}_optical_controls", controls)
    if matched_geometry:
        export(f"{stem}_sentinel2_fits", matched_geometry)
        renewal = permit_renewal()
        candidates = candidate_roofs(event, roofs, matched_geometry, fits, renewal)
        if {e for r in candidates for e in json.loads(r["possible_equipment_ids_json"])} != renewal.keys():
            raise ValueError("The two Rose Rock permits cover different EFR equipment")
        dates = matched_optical_dates(matched_geometry, stamp)
        export(f"{stem}_inventory_candidates", candidates)
        export(f"{stem}_permit_renewal_equipment", list(renewal.values()))
        export(f"{stem}_sentinel2_tiles", dates)
    else:
        candidates, dates = [], []
    summary = dict(stats, nrc_report_id=event["report_id"], usgs_products=len(sources), requested_roofs=len(next(iter(coverage.values()))),
        geometry_basis=basis, geometry_roofs=len(geometry) if geometry is not None else len({r["candidate_id"] for r in fits}),
        lidar_2022_latest_public_availability_date=max(r["latest_engineering_date"] for r in geometry.values()) if geometry else None,
        fit_status=Counter(r["fit_status"] for r in fits), roofs_resolved_before_event=len(before),
        roofs_resolved_after_event=len(after), roofs_resolved_on_both_sides=len(before & after),
        same_roof_b8_b04_controls_within_48h=len(controls),
        same_geometry_sentinel2_fits=len(matched_geometry),
        median_absolute_b8_b04_difference_m=median(abs(r["landsat_minus_sentinel_gap_m"]) for r in controls) if controls else None,
        original_b8_verified=0, admitted_nrc_inventory_labels=0)
    if candidates:
        compatible = {r["candidate_id"] for r in candidates if r["capacity_at_least_reported_inventory"]}
        grouped = defaultdict(list)
        for row in dates:
            grouped[row["local_date"]].append(row)
        summary.update(deq_capacity_compatible_roofs=len(compatible), capacity_compatible_roofs_with_lidar_2022=len(compatible & geometry.keys()),
            deq_renewal_equipment_crosschecked=len(renewal),
            landsat_resolved_roofs_capacity_compatible=len((before | after) & compatible),
            sentinel2_dates=[dict(local_date=day, roofs=len(rows), both_tiles_passed=sum(r["both_tiles_passed"] for r in rows),
                capacity_compatible_roofs_both_tiles_passed=sum(r["both_tiles_passed"] and r["candidate_id"] in compatible for r in rows))
                for day, rows in grouped.items()])
    if args.spectral:
        spectral, masks = spectral_checks(matched_geometry, geometry, stem)
        comparisons = spectral_dates(spectral, dates, compatible)
        export(f"{stem}_spectral_fits", spectral)
        export(f"{stem}_spectral_masks", masks)
        export(f"{stem}_spectral_tiles", comparisons)
        summary.update(spectral_detector_masks=len(masks), spectral_fit_rows=len(spectral),
            spectral_all_three_passed=sum(all(r[f"{m}_fit_status"] == RESOLVED for m in METHODS) for r in spectral))
        for report in summary["sentinel2_dates"]:
            matching = [r for r in comparisons if r["local_date"] == report["local_date"]
                        and r["capacity_at_least_reported_inventory"]]
            report["capacity_compatible_both_tiles_by_spectral_method"] = {
                m: sum(r[f"{m}_both_tiles_passed"] for r in matching) for m in METHODS}
            report["capacity_compatible_both_tiles_all_three_methods"] = sum(
                r["all_three_methods_both_tiles_passed"] for r in matching)
    result = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{stem}_summary.json", result.encode())
    print(result, end="")


if __name__ == "__main__":
    main()
