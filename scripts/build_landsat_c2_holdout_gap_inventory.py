"""Find exact Landsat 8/9 B8 versions for gaps in the closed Sentinel-2 series."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone
from functools import cache
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from shapely.geometry import Point, shape

from build_holdout_optical_inputs import indexed
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_landsat import LANDSAT
from collect_public import ROOT, TLS, write

STEM = "landsat_c2_holdout_gap"
ASSETS = ("pan", "qa_pixel", "qa_radsat", "MTL.json", "ANG.txt")
SIDES = ("previous", "current")
DOWNLOAD_ASSETS = ("pan_url", "qa_pixel_url", "qa_radsat_url", "mtl_json_url", "ang_txt_url")


def parse(value):
    return datetime.fromisoformat(value)


@cache
def level1_generation(file):
    path = ROOT / file
    metadata = json.loads(path.read_bytes())["LANDSAT_METADATA_FILE"]
    record = metadata["LEVEL1_PROCESSING_RECORD"]
    return record["LANDSAT_PRODUCT_ID"], parse(record["DATE_PRODUCT_GENERATED"])


def landsat_catalog():
    for year in range(2023, 2027):
        file = LANDSAT / "usgs_catalog" / f"{year}.geojson"
        for item in json.loads(file.read_bytes())["features"]:
            props = item["properties"]
            if props["platform"] in ("LANDSAT_8", "LANDSAT_9") and props["view:sun_elevation"] > 0:
                yield item, str(file.relative_to(ROOT))


def sentinel_observations():
    with (DERIVED / "shadow_timing_holdout_observations.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["geometry_mode"] != "dated_2011_lidar" or row["fit_status"] != "resolved_in_geometry_diagnostic":
                continue
            capture, available = (parse(row[key]) for key in ("captured_at_utc", "geometry_available_at_utc"))
            if capture > available:
                raise ValueError(f"Sentinel-2 archive clock precedes capture for {row['version_id']}")
            yield capture, available, row


def sentinel_candidates(windows, tanks):
    by_week, by_capture_day = defaultdict(list), defaultdict(list)
    targets = defaultdict(list)
    for week, sides in windows.items():
        for side, start, end, cutoff in sides:
            day = start.date()
            while day <= end.date():
                targets[day].append((week, side, start, end, cutoff))
                day += timedelta(days=1)
    for capture, available, r in sentinel_observations():
        tank = r["candidate_id"]
        item = capture, available, r
        by_capture_day[tank,capture.date()].append(item)
        for week, side, start, end, cutoff in targets[capture.date()]:
            if start < capture <= end and available < cutoff and tank in tanks[week]:
                if r["terminal_id"] != tanks[week][tank]["terminal_id"]:
                    raise ValueError(f"Different dated tank identity for {week}/{tank}")
                by_week[week,side,tank].append(item)
    return by_week, by_capture_day


def build(db):
    weeks = {key: r for key,r in indexed("holdout_optical_weeks").items() if r["b04_status"] != "measured"}
    windows = defaultdict(list)
    for w, side, issue, start, end, cutoff in db.execute("""SELECT week_ending,endpoint,release_date,
        window_starts_after_utc,window_ends_by_utc,inputs_available_before_utc
        FROM eia_satellite_window WHERE study_phase='holdout' ORDER BY week_ending,endpoint"""):
        if w in weeks:
            if issue != weeks[w]["release_date"] or parse(cutoff) != parse(weeks[w]["cutoff_utc"]):
                raise ValueError(f"Inconsistent release clock for {w}")
            windows[w].append((side, parse(start), parse(end), parse(cutoff)))
    with (DERIVED / "holdout_optical_tank_weeks.csv").open(newline="") as handle:
        tanks = defaultdict(dict)
        for r in csv.DictReader(handle):
            if r["week_ending"] in weeks:
                tanks[r["week_ending"]][r["candidate_id"]] = r
    centers = {tank: Point(lon,lat) for tank,lon,lat in db.execute(
        "SELECT candidate_id,longitude,latitude FROM tank_engineering_registry")}
    products = {}
    for local, path, available in db.execute("""SELECT l.product_id,l.metadata_path,a.latest_archive_object_time_utc
        FROM landsat_local_product l LEFT JOIN landsat_product_archive_clock a USING(product_id)
        WHERE l.source_kind='pc_collection2_level2_context' AND substr(l.captured_at_utc,1,10)>='2023-12-22'"""):
        product, generated = level1_generation(path)
        if product in products:
            raise ValueError(f"Multiple local Level-2 files for exact Landsat product {product}")
        products[product] = (local, path, generated, parse(available) if available else None)
    clear = defaultdict(set)
    for product, tank in db.execute("""SELECT q.product_id,q.candidate_id FROM landsat_tank_scene_quality q
        JOIN landsat_local_product p USING(product_id)
        WHERE p.source_kind='pc_collection2_level2_context' AND substr(p.captured_at_utc,1,10)>='2023-12-22'
          AND q.qa_roof_cells>0 AND q.qa_roof_cells=q.valid_qa_roof_cells
          AND q.qa_roof_cells=q.clear_qa_roof_cells AND q.signal_cells=q.valid_signal_cells"""):
        clear[product].add(tank)
    rows, coverage = [], defaultdict(set)
    for item, catalog_file in landsat_catalog():
        product, props = item["id"], item["properties"]
        captured = parse(props["datetime"])
        slots = [(week,side,upper,cutoff) for week,sides in windows.items() for side,lower,upper,cutoff in sides
                 if lower < captured <= upper]
        if not slots:
            continue
        polygon = shape(item["geometry"])
        local, path, generated, available = products.get(product, (None,None,None,None))
        day_processed = datetime.strptime(product.split("_")[4], "%Y%m%d").date()
        if generated and (generated < captured or generated.date() != day_processed):
            raise ValueError(f"Level-1 clock conflicts with catalog for {product}")
        assets = {key: item["assets"][key]["href"] for key in ASSETS}
        for week, side, upper, cutoff in slots:
            matched = {tank for tank in tanks[week] if polygon.covers(centers[tank])}
            if not matched:
                continue
            screened = matched & clear[local]
            could_precede = generated < cutoff if generated else day_processed <= cutoff.date()
            qa_precedes = available is not None and available < cutoff
            if could_precede and qa_precedes:
                coverage[week,side].update(screened)
            rows.append(dict(study_phase="holdout", week_ending=week, release_date=weeks[week]["release_date"],
                endpoint=side, cutoff_utc=cutoff.isoformat(), platform=props["platform"], product_id=product,
                tier=product.split("_")[-1], captured_at_utc=props["datetime"],
                hours_before_boundary=(upper-captured).total_seconds()/3600,
                processed_date_from_id=day_processed.isoformat(), level1_generated_at_utc=generated.isoformat() if generated else None,
                level1_could_be_generated_by_cutoff=could_precede, level1_stac_created_at_utc=props.get("created"),
                local_exact_level2_product=local, local_level2_metadata_path=path,
                local_level2_assets_at_utc=available.isoformat() if available else None, local_level2_before_cutoff=qa_precedes,
                dated_efr_centers=len(matched), clear_efr_centers_via_exact_level2_qa=len(screened),
                clear_efr_candidate_ids_json=json.dumps(sorted(screened)), catalog_file=catalog_file,
                **{f"{key.lower().replace('.','_')}_url":url for key,url in assets.items()}))
    sentinel_observations_by_slot, by_capture_day = sentinel_candidates(windows, tanks)
    sentinel = defaultdict(set)
    for week, side, tank in sentinel_observations_by_slot:
        sentinel[week,side].add(tank)
    weeks_out = []
    for week, r in sorted(weeks.items()):
        sides = [coverage[week, side] for side in SIDES]
        both = sides[0] & sides[1]
        mixed_previous = sides[0] & sentinel[week,"current"]
        mixed_current = sentinel[week,"previous"] & sides[1]
        mixed = mixed_previous | mixed_current
        any_b8 = both | mixed
        weeks_out.append(dict(study_phase="holdout", week_ending=week, release_date=r["release_date"],
            forecast_cutoff_utc=r["cutoff_utc"], b04_status=r["b04_status"],
            dated_efr=len(tanks[week]), previous_clear_possible_b8=len(sides[0]), current_clear_possible_b8=len(sides[1]),
            paired_clear_possible_b8=len(both), paired_area_m2=sum(float(tanks[week][tank]["circular_area_m2"]) for tank in sorted(both)),
            paired_candidate_ids_json=json.dumps(sorted(both)), mixed_b8_and_sentinel2_candidate_tanks=len(mixed),
            mixed_b8_and_sentinel2_candidate_ids_json=json.dumps(sorted(mixed)), any_b8_pair_candidate_tanks=len(any_b8),
            previous_b8_pair_candidate_ids_json=json.dumps(sorted(both | mixed_previous)),
            current_b8_pair_candidate_ids_json=json.dumps(sorted(both | mixed_current)),
            previous_b8_mixed_candidate_ids_json=json.dumps(sorted(mixed_previous)),
            current_b8_mixed_candidate_ids_json=json.dumps(sorted(mixed_current))))
    return sorted(rows,key=lambda r:(r["week_ending"],r["endpoint"],r["product_id"])), weeks_out, by_capture_day


def calibration_controls(rows, by_capture_day):
    best = {}
    for r in rows:
        if not r["level1_could_be_generated_by_cutoff"] or not r["local_level2_before_cutoff"]:
            continue
        capture, cutoff = parse(r["captured_at_utc"]), parse(r["cutoff_utc"])
        for tank in json.loads(r["clear_efr_candidate_ids_json"]):
            key = r["product_id"], tank
            for observed, available, s in by_capture_day[tank,capture.date()]:
                minutes = (observed-capture).total_seconds()/60
                order = abs(minutes), cutoff, s["version_id"], s["canonical_l2a_id"]
                if available >= cutoff or abs(minutes)>45 or (key in best and best[key][0]<=order):
                    continue
                best[key] = order, dict(candidate_id=tank, terminal_id=s["terminal_id"],
                    landsat_platform=r["platform"], landsat_product_id=r["product_id"],
                    landsat_capture_at_utc=r["captured_at_utc"], landsat_exact_level2_product=r["local_exact_level2_product"],
                    sentinel2_capture_at_utc=s["captured_at_utc"], sentinel2_version_id=s["version_id"],
                    sentinel2_canonical_id=s["canonical_l2a_id"], sentinel2_geometry_available_at_utc=s["geometry_available_at_utc"],
                    sentinel2_metadata_path=s["tile_metadata_path"], sentinel2_minus_landsat_minutes=minutes,
                    first_matching_release_date=r["release_date"], forecast_cutoff_utc=r["cutoff_utc"])
    return [best[key][1] for key in sorted(best)]


def download_manifest(rows, weeks, controls):
    candidates = {(r["week_ending"],side): set(json.loads(r[f"{side}_b8_pair_candidate_ids_json"]))
                  for r in weeks for side in SIDES}
    control_counts = Counter(r["landsat_product_id"] for r in controls)
    catalog = {}
    qualified = defaultdict(list)
    for r in rows:
        catalog[r["product_id"]] = r
        if r["level1_could_be_generated_by_cutoff"] and r["local_level2_before_cutoff"]:
            linked = candidates[r["week_ending"],r["endpoint"]] & set(json.loads(r["clear_efr_candidate_ids_json"]))
            if linked:
                qualified[r["product_id"]].append((r, linked))
    manifest = []
    for product in qualified.keys() | control_counts.keys():
        entries = qualified[product]
        r = catalog[product]
        manifest.append(dict(product_id=product, platform=r["platform"], tier=r["tier"],
            captured_at_utc=r["captured_at_utc"], level1_generated_at_utc=r["level1_generated_at_utc"],
            source_catalog_file=r["catalog_file"], contributing_gap_weeks=len({a["week_ending"] for a,_ in entries}),
            contributing_gap_tank_weeks=len({(a["week_ending"],tank) for a,ids in entries for tank in ids}),
            sentinel2_calibration_candidates=control_counts[product],
            week_endpoints_json=json.dumps([dict(week_ending=a["week_ending"], endpoint=a["endpoint"],
                                                forecast_cutoff_utc=a["cutoff_utc"],
                                                paired_candidate_tanks=len(ids)) for a,ids in entries]),
            **{key:r[key] for key in DOWNLOAD_ASSETS}))
    return sorted(manifest,key=lambda r:(-r["contributing_gap_tank_weeks"],-r["sentinel2_calibration_candidates"],r["product_id"]))


def probe_source(job):
    row, asset = job
    reference = row[asset]
    parsed = urlsplit(reference)
    if parsed.scheme != "https" or parsed.netloc != "landsatlook.usgs.gov" or not parsed.path.startswith("/data/"):
        raise ValueError(f"Unexpected original Landsat asset URL: {reference}")
    routes = {
        "landsatlook": reference,
        "unsigned_usgs_s3": "https://usgs-landsat.s3.us-west-2.amazonaws.com/" + parsed.path.removeprefix("/data/"),
    }
    for route, url in routes.items():
        request = Request(url, headers={"Range": "bytes=0-31", "User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=20, context=TLS) as response:
                status, host = response.status, urlsplit(response.url).hostname
                content_type, sample = response.headers.get_content_type(), response.read(32)
        except HTTPError as error:
            status, host = error.code, urlsplit(error.url).hostname
            content_type, sample = error.headers.get_content_type(), error.read(32)
        if host == "ers.cr.usgs.gov":
            evidence = "usgs_login_page"
        elif status in (401,403):
            evidence = "authorization_required"
        elif asset in DOWNLOAD_ASSETS[:3] and sample[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
            evidence = "tiff_signature"
        elif status in (200,206) and asset=="mtl_json_url" and sample.lstrip().startswith(b"{"):
            evidence = "json_signature"
        elif status in (200,206) and asset=="ang_txt_url" and b"GROUP" in sample:
            evidence = "angle_text_signature"
        else:
            evidence = "unverified_response"
        yield dict(product_id=row["product_id"], platform=row["platform"], asset=asset, route=route,
                   url=url, checked_at_utc=datetime.now(timezone.utc).isoformat(), status=status,
                   response_host=host, content_type=content_type, evidence=evidence)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-access", action="store_true", help="Probe five source assets on each of Landsat 8 and 9.")
    args = parser.parse_args()
    with closing(read_only(SOURCE)) as db:
        rows, weeks, sentinel = build(db)
    controls = calibration_controls(rows, sentinel)
    manifest = download_manifest(rows, weeks, controls)
    summary = dict(b04_gap_weeks=len(weeks), gap_weeks_with_catalog_candidates=len({r["week_ending"] for r in rows}),
                   distinct_products=len({r["product_id"] for r in rows}), product_endpoints=len(rows),
                   product_endpoints_by_platform=dict(Counter(r["platform"] for r in rows)),
                   product_endpoints_with_exact_level2_qa=sum(r["local_exact_level2_product"] is not None for r in rows),
                   product_endpoints_with_clear_qa_and_time=sum(r["level1_could_be_generated_by_cutoff"]
                       and r["local_level2_before_cutoff"] and r["clear_efr_centers_via_exact_level2_qa"]>0 for r in rows),
                   gap_weeks_with_previous_qualified_endpoint=sum(r["previous_clear_possible_b8"]>0 for r in weeks),
                   gap_weeks_with_current_qualified_endpoint=sum(r["current_clear_possible_b8"]>0 for r in weeks),
                   gap_weeks_with_two_possible_b8_endpoints=sum(r["paired_clear_possible_b8"]>0 for r in weeks),
                   possible_b8_tank_week_pairs=sum(r["paired_clear_possible_b8"] for r in weeks),
                   gap_weeks_with_mixed_sentinel2_and_b8_candidates=sum(r["mixed_b8_and_sentinel2_candidate_tanks"]>0 for r in weeks),
                   mixed_sentinel2_b8_tank_week_pairs=sum(r["mixed_b8_and_sentinel2_candidate_tanks"] for r in weeks),
                   gap_weeks_with_any_candidate_b8_pair=sum(r["any_b8_pair_candidate_tanks"]>0 for r in weeks),
                   unique_tank_week_pairs_with_any_candidate_b8=sum(r["any_b8_pair_candidate_tanks"] for r in weeks),
                   exact_level1_products_in_download_manifest=len(manifest),
                   exact_level1_products_contributing_gap_pairs=sum(r["contributing_gap_weeks"]>0 for r in manifest),
                   near_coincident_sentinel2_calibration_candidates=len(controls),
                   calibration_landsat_products=len({r["landsat_product_id"] for r in controls}),
                   calibration_candidates_by_platform=dict(Counter(r["landsat_platform"] for r in controls)),
                   interpretation="Local Level-2 QA is linked to the exact underlying Level-1 product ID. "
                                  "Level-1 generation is a lower bound on availability. "
                                  "The remote B8, QA and angle versions and B8 roof height remain to be verified. "
                                  "Sentinel-2 mixed candidates use passing single-view fits. Pairwise cross-sensor validation is pending.")
    export(f"{STEM}_products", rows)
    export(f"{STEM}_weeks", weeks)
    if not manifest:
        raise ValueError("No downloadable B8 versions for any candidate pair")
    export(f"{STEM}_download_manifest", manifest)
    export(f"{STEM}_calibration_controls", controls)
    if args.probe_access:
        examples = [next(r for r in manifest if r["platform"] == platform) for platform in ("LANDSAT_8","LANDSAT_9")]
        with ThreadPoolExecutor(max_workers=6) as pool:
            probes = [r for result in pool.map(lambda j: list(probe_source(j)),
                      ((r,key) for r in examples for key in DOWNLOAD_ASSETS)) for r in result]
        export(f"{STEM}_access_probe", probes)
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
