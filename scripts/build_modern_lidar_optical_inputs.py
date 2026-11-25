"""Audit source-dated 2022 LiDAR geometry and replay archived Sentinel-2 physical inputs."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from email.utils import parsedate_to_datetime
from statistics import median
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from pyproj import Transformer
from shapely import transform, union_all
from shapely.geometry import box, shape

from audit_lidar_tanks import DATASET
from audit_sentinel2_weekly_volume import observations
from build_registered_shadow_development import (RETROSPECTIVE, observation_sources, prepare_jobs,
                                                 registered_pair, registered_profile, study_weeks)
from build_shadow_timing_measurements import date_evidence, engineering_dates
from build_tank_lidar_outlines import OUTLINE_FIELDS, elevation_priors, footprint_data, outline_rows
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_public import RAW, ROOT, TLS, write

MODE = "dated_2022_lidar"
STEM = "cushing_modern_lidar_optical"
GOOD_FIT = "well_supported_circular_fit"
SOURCE_DIR = RAW / "usgs" / "modern_source_metadata"
BUCKET = ("https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/LPC/Projects/"
          "OK_Statewide_D22/OK_Statewide_2_D22/")
DOWNLOAD_LIST = BUCKET + "0_file_download_links.txt"
PSD_REGISTER = "https://oklahoma.gov/content/dam/ok/en/deq/documents/air-quality/Permits_issued_PSD_permits.csv"
WILDHORSE_PERMIT = "https://applications.deq.ok.gov/permitspublic/storedpermits/8910.pdf"


def archive(url, name):
    path = SOURCE_DIR / name
    receipt = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not receipt.is_file():
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=35, context=TLS) as response:
            data = response.read()
            modified = response.headers.get("Last-Modified")
        meta = dict(url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    last_modified_utc=parsedate_to_datetime(modified).isoformat() if modified else None)
        write(path, data)
        write(receipt, json.dumps(meta, indent=2).encode() + b"\n")
    data = path.read_bytes()
    meta = json.loads(receipt.read_bytes())
    if (meta["url"], meta["bytes"], meta["sha256"]) != (url, len(data), hashlib.sha256(data).hexdigest()):
        raise ValueError(f"Archived source changed in {path.name}")
    return data, meta


def registry_and_tiles():
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        roofs = [dict(r) for r in db.execute("""SELECT t.*,documented_terminal_id AS terminal_id
            FROM tank_engineering_registry t WHERE documented_roof_type='EFR' ORDER BY candidate_id""")]
        tiles = [dict(r) for r in db.execute("""SELECT * FROM lidar_product
            WHERE url LIKE '%/OK_Statewide_2_D22/%' ORDER BY id""")]
    for tile in tiles:
        bounds = json.loads(tile["bounding_box_json"])
        tile["bounds"] = box(*(bounds[k] for k in ("minX", "minY", "maxX", "maxY")))
    return roofs, tiles


def check_tile(tile, links, list_date):
    basename = tile["url"].rsplit("/", 1)[-1].removesuffix(".laz")
    url = BUCKET + f"metadata/{basename}.xml"
    payload, receipt = archive(url, f"{basename}.xml")
    metadata = ElementTree.fromstring(payload)
    citation = metadata.find("./idinfo/citation/citeinfo")
    bounds = metadata.find("./idinfo/spdom/bounding")
    pub_date = dt.datetime.strptime(citation.findtext("pubdate"), "%Y%m%d").date().isoformat()
    extent = tuple(float(bounds.findtext(k)) for k in ("westbc", "southbc", "eastbc", "northbc"))
    if (citation.findtext("title") != tile["title"] or pub_date != tile["publication_date"]
            or extent != tile["bounds"].bounds or tile["url"] not in links):
        raise ValueError(f"USGS tile metadata disagrees with catalog for {tile['id']}")
    if not receipt["last_modified_utc"]:
        raise ValueError(f"Missing USGS metadata upload date for {tile['id']}")
    catalog_date = dt.datetime.fromisoformat(tile["date_created"]).astimezone(dt.timezone.utc).date().isoformat()
    return dict(usgs_source_id=tile["id"], title=tile["title"], tnm_publication_date=pub_date,
                tnm_catalog_created_at=tile["date_created"], fgdc_last_modified_at_utc=receipt["last_modified_utc"],
                usgs_file_list_last_modified_at_utc=list_date,
                source_available_date=max(pub_date, catalog_date, receipt["last_modified_utc"][:10],
                                          list_date[:10]),
                bounds_wgs84_json=json.dumps(extent), source_laz_expected_bytes=tile["size_bytes"],
                source_laz_url=tile["url"], fgdc_url=url, fgdc_sha256=receipt["sha256"])


def wildhorse_issued_date():
    payload, _ = archive(PSD_REGISTER, "okdeq_final_psd.csv")
    rows = [r for r in csv.DictReader(payload.decode("utf-8-sig").splitlines())
            if r["Permit #"] == "2016-1066-C M-2 PSD" and r["County"] == "Lincoln"]
    if len(rows) != 1 or rows[0]["Facility Name"] not in ("Wildhorse Terminal", "Cushing West Terminal"):
        raise ValueError("Cannot identify Wildhorse M-2 in the Oklahoma DEQ issuance register")
    return dt.datetime.strptime(rows[0]["Issue Date"], "%m/%d/%Y").date().isoformat()


def source_footprints(roofs, outlines, tiles):
    footprint = {r[0]: r for scope in ("osm", "supplemental") for r in footprint_data(scope)}
    to_utm = Transformer.from_crs(4326, 32614, always_xy=True)
    to_lonlat = Transformer.from_crs(32614, 4326, always_xy=True)
    ground_limits = {}
    for source_file in {r[-1] for r in footprint.values()}:
        for feature in json.loads((ROOT / source_file).read_bytes())["features"]:
            polygon = transform(shape(feature["geometry"]), to_utm.transform, interleaved=False)
            west, south, east, north = polygon.bounds
            ground_limits[feature["properties"]["candidate_id"]] = (west - 35, south - 35, east + 35, north + 35)
    roof_tiles = {}
    for roof in roofs:
        tank = roof["candidate_id"]
        if not outlines[tank]["diameter_m"]:
            continue
        _, x, y, _, _, diameter, *_ = footprint[tank]
        radius = diameter / 2 + 12
        west, south, east, north = ground_limits[tank]
        limits = box(*to_lonlat.transform_bounds(
            min(x-radius, west), min(y-radius, south), max(x+radius, east), max(y+radius, north)))
        hits = [t for t in tiles if t["bounds"].intersects(limits)]
        if not hits or not union_all([t["bounds"] for t in hits]).covers(limits):
            raise ValueError(f"USGS catalog footprints do not cover the LiDAR search area for {tank}")
        roof_tiles[tank] = hits
    return footprint, roof_tiles


def replay_outlines(registry, footprint, outlines):
    for scope, name in (("osm", "lidar_tank_observations"), ("supplemental", "lidar_supplemental_tank_observations")):
        targets = [footprint[r["candidate_id"]] for r in registry if (
            r["outline_source"] == "OSM") == (scope == "osm") and r["modern_input_status"] == "available"]
        if not targets:
            continue
        priors = elevation_priors(DERIVED / f"{name}.csv")
        directory = ROOT / outlines[targets[0][0]]["source_directory"]
        for values in outline_rows(targets, priors, directory):
            actual = dict(zip(OUTLINE_FIELDS, values, strict=True))
            tank = actual["candidate_id"]
            saved = outlines[tank]
            for key, value in actual.items():
                if isinstance(value, (int, float)):
                    equal = math.isclose(value, float(saved[key]), rel_tol=0, abs_tol=1e-8)
                else:
                    equal = value == saved[key]
                if not equal:
                    raise ValueError(f"Raw EPT replay disagrees on {tank}/{key}")
            yield tank


def geometry_registry(roofs, outlines, roof_tiles, metadata, legacy):
    evidence = date_evidence()
    _, permits, aerial, _, _ = evidence
    wildhorse = wildhorse_issued_date()
    result = []
    for roof in roofs:
        tank = roof["candidate_id"]
        outline = outlines[tank]
        sources = [metadata[tile["id"]] for tile in roof_tiles.get(tank, ())]
        source_date = max((r["source_available_date"] for r in sources), default=None)
        _, roof_date = engineering_dates(roof, evidence)
        doc = roof["engineering_source_url"].rsplit("/", 1)[-1].removesuffix(".pdf")
        _, roof_clock_basis, roof_clock_url = aerial.get(tank, permits.get(doc, (None, None, None)))
        if roof_date and not roof_clock_basis:
            roof_clock_basis = "existing_dated_roof_evidence"
        if roof["terminal_id"] == "wildhorse":
            if roof["engineering_source_url"] != WILDHORSE_PERMIT or roof["source_document_date"] >= wildhorse:
                raise ValueError(f"Wildhorse as-built roster no longer matches the issued permit for {tank}")
            roof_date, roof_clock_basis = wildhorse, "okdeq_matching_psd_permit_issuance"
            roof_clock_url = PSD_REGISTER
        status = "available"
        if outline["fit_status"] != GOOD_FIT:
            status = "insufficient_lidar_outline_quality"
        elif not source_date or not roof_date or not roof["first_seen"]:
            status = "missing_dated_evidence"
        area = math.pi * float(outline["diameter_m"])**2 / 4 if outline["diameter_m"] else None
        if status == "available" and not (
                float(outline["prior_rim_z_p95_native"]) > float(outline["prior_ground_z_p50_native"])
                and dt.datetime.fromisoformat(outline["last_upper_return_at_utc"]).date().isoformat() < source_date):
            raise ValueError(f"Invalid geometry or source clocks for {tank}")
        result.append(dict(outline, terminal_id=roof["terminal_id"], cohort_id=roof["cohort_id"],
            outline_source=roof["outline_source"], first_seen=roof["first_seen"], has_legacy_geometry=tank in legacy,
            engineering_source_url=roof["engineering_source_url"], roof_evidence_date=roof_date,
            roof_evidence_basis=roof_clock_basis, roof_clock_source_url=roof_clock_url,
            lidar_source_available_date=source_date,
            latest_engineering_date=max(source_date, roof_date) if source_date and roof_date else None,
            source_usgs_ids_json=json.dumps([r["usgs_source_id"] for r in sources]),
            circular_area_m2=area, modern_input_status=status))
    return result


def optical_replay(registry, weeks):
    geometry = {r["candidate_id"]:r for r in registry if r["modern_input_status"] == "available"}
    manifest = {(r["week_ending"],r["candidate_id"]):r for r in read_csv("weekly_shadow_holdout_input_manifest")}
    ledger = {}
    for row in read_csv("weekly_shadow_registered_holdout_pairs"):
        tank, week = row["candidate_id"], row["week_ending"]
        g = geometry.get(tank)
        if row["geometry_mode"] != RETROSPECTIVE or g is None:
            continue
        start, _, release, cutoff = weeks[week]
        if g["latest_engineering_date"] >= cutoff[:10] or g["first_seen"] > start.date().isoformat():
            continue
        source = manifest[week, tank]
        if (week, tank) in ledger or (release, cutoff, g["terminal_id"], g["cohort_id"]) != (
                row["release_date"], row["inputs_available_before_utc"], row["terminal_id"], source["cohort_id"]):
            raise ValueError(f"Optical window or tank identity differs for {week}/{tank}")
        for side, label in (("previous", "earlier"), ("current", "later")):
            captured = dt.datetime.fromisoformat(source[f"{side}_capture_at_utc"])
            archived = dt.datetime.fromisoformat(source[f"{side}_archive_available_at_utc"])
            ready = dt.datetime.fromisoformat(source[f"{side}_geometry_available_at_utc"])
            if (source[f"{side}_version"] != row[f"{label}_version"]
                    or captured != dt.datetime.fromisoformat(row[f"{label}_capture_at_utc"])
                    or archived != dt.datetime.fromisoformat(row[f"{label}_archive_available_at_utc"])
                    or not captured <= archived <= ready < dt.datetime.fromisoformat(cutoff)):
                raise ValueError(f"Wrong scene product or availability for {week}/{tank}/{side}")
        if not math.isclose(float(row["circular_area_m2"]), g["circular_area_m2"], rel_tol=1e-9):
            raise ValueError(f"Saved shadow fit uses a different LiDAR geometry for {week}/{tank}")
        details = dict(lidar_source_available_date=g["lidar_source_available_date"], roof_evidence_date=g["roof_evidence_date"],
                       latest_engineering_date=g["latest_engineering_date"], has_legacy_geometry=g["has_legacy_geometry"],
                       previous_detector_geometry_available_at_utc=source["previous_geometry_available_at_utc"],
                       current_detector_geometry_available_at_utc=source["current_geometry_available_at_utc"])
        ledger[week, tank] = dict(row, geometry_mode=MODE, **details)
    volumes = [dict(r, geometry_mode=MODE,
                    latest_engineering_date=ledger[r["week_ending"],r["candidate_id"]]["latest_engineering_date"],
                    has_legacy_geometry=geometry[r["candidate_id"]]["has_legacy_geometry"])
               for r in observations("holdout", weeks, geometry, ledger.values(), geometry_mode=MODE)]
    return ledger, volumes


def verify_native_fits(registry, weeks, ledger, gap_weeks):
    native, saved, _ = observation_sources("holdout")
    geometry = {r["candidate_id"]:r for r in registry}
    verified = set()
    replayed = set()
    for mode, base, _, offset, keys, *_ in prepare_jobs(weeks, native, saved, "holdout"):
        key = base["week_ending"], base["candidate_id"]
        if mode != RETROSPECTIVE or key not in ledger:
            continue
        g = geometry[key[1]]
        shell = float(g["prior_rim_z_p95_native"]) - float(g["prior_ground_z_p50_native"])
        fits = []
        reference = ledger[key]
        for side, (packet, stem, saved_mode, number) in enumerate(keys):
            path, rows, _, kernels = native[packet]
            row, original = rows[number], saved[stem, saved_mode, number]
            if (not all(math.isclose(float(original[field]), float(g[field]), rel_tol=0, abs_tol=1e-8)
                        for field in ("center_east_m", "center_north_m", "diameter_m"))
                    or not math.isclose(float(original["shell_above_ground_m"]), shell, rel_tol=0, abs_tol=1e-8)):
                raise ValueError(f"The original native chip uses different geometry for {key}")
            if key[0] in gap_weeks and reference["registered_both_shadow_diagnostics_pass"] == "True":
                if offset["registration_status"] != "checked_offset":
                    raise ValueError(f"Missing independent alignment for {key}")
                sign = -0.5 if side == 0 else 0.5
                east, north = (sign * float(offset[f"later_minus_earlier_{axis}_m"]) for axis in ("east", "north"))
                fits.append(registered_profile(path, row, original, shell, kernels[row["platform"]], east, north))
        verified.add(key)
        if fits:
            for field, value in registered_pair(*fits).items():
                expected = reference[field]
                if isinstance(value, float):
                    matches = math.isclose(value, float(expected), rel_tol=0, abs_tol=1e-8)
                else:
                    matches = str(value) == expected
                if not matches:
                    raise ValueError(f"Native B04 refit differs for {key}/{field}")
            replayed.add(key)
    expected_replays = {key for key, row in ledger.items() if key[0] in gap_weeks
                        and row["registered_both_shadow_diagnostics_pass"] == "True"}
    if verified != ledger.keys() or replayed != expected_replays:
        raise ValueError("Native geometry verification omitted optical inputs")
    return verified, replayed


def weekly_summary(weeks, registry, ledger, volumes):
    baseline = {r["week_ending"]:r for r in read_csv("holdout_optical_weeks")}
    baseline_tanks = {(r["week_ending"],r["candidate_id"]) for r in read_csv("holdout_optical_tank_weeks")
                      if r["baseline_b04_resolved"] == "True"}
    verified, checked, resolved = (defaultdict(list) for _ in range(3))
    for row in ledger.values():
        verified[row["week_ending"]].append(row)
        if row["offset_status"] == "checked_offset":
            checked[row["week_ending"]].append(row)
    for row in volumes:
        resolved[row["week_ending"]].append(row)
    for week, (start, _, release, cutoff) in weeks.items():
        available = [r for r in registry if r["modern_input_status"] == "available"
                     and r["latest_engineering_date"] < cutoff[:10] and r["first_seen"] <= start.date().isoformat()]
        measured = resolved[week]
        additional = [r for r in measured if (week,r["candidate_id"]) not in baseline_tanks]
        available_area = sum(r["circular_area_m2"] for r in available)
        added_area = sum(r["circular_area_m2"] for r in additional)
        yield dict(week_ending=week, release_date=release, cutoff_utc=cutoff,
                   baseline_b04_status=baseline[week]["b04_status"],
                   baseline_b04_tanks=int(baseline[week]["baseline_b04_tanks"]),
                   modern_available_tanks=len(available), modern_available_area_m2=available_area,
                   modern_input_pairs=len(verified[week]),
                   modern_checked_pairs=len(checked[week]), modern_resolved_pairs=len(measured),
                   modern_resolved_without_legacy_geometry=sum(not r["has_legacy_geometry"] for r in measured),
                   modern_additional_resolved_tanks=len(additional),
                   modern_additional_area_m2=added_area,
                   modern_additional_area_fraction=added_area / available_area if available_area else None,
                   modern_additional_with_loss_interval_excluding_zero=sum(
                       r["minimum_magnitude_change_kbbl"] != 0 for r in additional))


def compare_geometry(weeks, modern, legacy):
    old = {(r["week_ending"],r["candidate_id"]):r for r in observations("holdout", weeks, legacy)}
    for row in modern:
        week, tank = row["week_ending"], row["candidate_id"]
        previous = old.get((week,tank))
        if previous is None:
            continue
        if any(row[f"{side}_{field}"] != previous[f"{side}_{field}"] for side in ("earlier", "later")
               for field in ("sentinel2_version", "capture_at_utc")):
            raise ValueError(f"Comparison uses different satellite captures for {week}/{tank}")
        before, after = previous["capture_change_kbbl"], row["capture_change_kbbl"]
        old_change_m = previous["earlier_gap_m"] - previous["later_gap_m"]
        new_change_m = row["earlier_gap_m"] - row["later_gap_m"]
        yield dict(week_ending=week, release_date=row["release_date"], candidate_id=tank, terminal_id=row["terminal_id"],
                   earlier_sentinel2_version=row["earlier_sentinel2_version"], later_sentinel2_version=row["later_sentinel2_version"],
                   legacy_diameter_m=previous["diameter_m"], modern_diameter_m=row["diameter_m"],
                   legacy_change_m=old_change_m, modern_change_m=new_change_m, modern_minus_legacy_change_m=new_change_m-old_change_m,
                   legacy_change_kbbl=before, modern_change_kbbl=after, modern_minus_legacy_change_kbbl=after-before,
                   legacy_loss_range_excludes_zero=previous["minimum_magnitude_change_kbbl"] != 0,
                   modern_loss_range_excludes_zero=row["minimum_magnitude_change_kbbl"] != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    roofs, tiles = registry_and_tiles()
    outlines = {r["candidate_id"]:r for r in read_csv("tank_lidar_outlines")}
    footprint, roof_tiles = source_footprints(roofs, outlines, tiles)
    used = {t["id"]:t for group in roof_tiles.values() for t in group}
    link_file, receipt = archive(DOWNLOAD_LIST, "download_links.txt")
    links = set(link_file.decode().splitlines())
    list_date = receipt["last_modified_utc"]
    if not list_date:
        raise ValueError("Missing USGS publication manifest upload date")
    with ThreadPoolExecutor(max_workers=4) as pool:
        metadata = {r["usgs_source_id"]:r for r in pool.map(
            lambda tile: check_tile(tile, links, list_date), used.values())}
    legacy = legacy_geometry()
    registry = geometry_registry(roofs, outlines, roof_tiles, metadata, legacy)
    raw_replayed = set(replay_outlines(registry, footprint, outlines))
    if raw_replayed != {r["candidate_id"] for r in registry if r["modern_input_status"] == "available"}:
        raise ValueError("EPT replay did not cover every eligible roof")
    for row in registry:
        row["raw_ept_outline_replayed"] = row["candidate_id"] in raw_replayed
    weeks = study_weeks("holdout")
    ledger, volumes = optical_replay(registry, weeks)
    weekly = list(weekly_summary(weeks, registry, ledger, volumes))
    overlap = list(compare_geometry(weeks, volumes, legacy))
    added_weeks = [r["week_ending"] for r in weekly if
                   r["baseline_b04_status"] != "measured" and r["modern_additional_resolved_tanks"]]
    verified, replayed = verify_native_fits(registry, weeks, ledger, set(added_weeks))
    for key, row in ledger.items():
        row["raw_b04_refit_verified"] = key in replayed
    for row in volumes:
        row["raw_b04_refit_verified"] = (row["week_ending"], row["candidate_id"]) in replayed
    additional = [r for r in registry if not r["has_legacy_geometry"] and r["modern_input_status"] == "available"]
    comparison = [r for r in registry if r["has_legacy_geometry"] and r["modern_input_status"] == "available"]
    for row in registry:
        old = legacy.get(row["candidate_id"])
        row["legacy_diameter_m"] = float(old["diameter_m"]) if old else None
        row["modern_minus_legacy_area_pct"] = (
            100*((float(row["diameter_m"])/row["legacy_diameter_m"])**2-1)
            if old and row["diameter_m"] else None)
    summary = dict(source_tiles=len(metadata), documented_efr_tanks=len(registry),
                   modern_status_counts=dict(Counter(r["modern_input_status"] for r in registry)),
                   additional_tanks_without_legacy_geometry=len(additional),
                   additional_tanks_by_terminal=dict(Counter(r["terminal_id"] for r in additional)),
                   modern_dated_tanks_with_legacy_geometry=len(comparison),
                   median_absolute_lidar_diameter_difference_m=median(
                       abs(float(r["diameter_m"])-r["legacy_diameter_m"]) for r in comparison),
                   median_absolute_lidar_area_difference_pct=median(
                       abs(r["modern_minus_legacy_area_pct"]) for r in comparison),
                   same_capture_pairs_with_both_geometries=len(overlap),
                   median_absolute_same_capture_change_difference_m=median(
                       abs(r["modern_minus_legacy_change_m"]) for r in overlap),
                   median_absolute_same_capture_change_difference_kbbl=median(
                       abs(r["modern_minus_legacy_change_kbbl"]) for r in overlap),
                   raw_ept_outlines_replayed=len(raw_replayed),
                   holdout_input_pairs=len(ledger), holdout_resolved_pairs=len(volumes),
                   holdout_pairs_with_native_geometry_verified=len(verified),
                   raw_b04_refits_on_baseline_gap_weeks=len(replayed),
                   holdout_resolved_pairs_without_legacy_geometry=sum(not r["has_legacy_geometry"] for r in volumes),
                   holdout_additional_resolved_pairs=sum(r["modern_additional_resolved_tanks"] for r in weekly),
                   holdout_weeks_with_additional_measurements=sum(r["modern_additional_resolved_tanks"]>0 for r in weekly),
                   weeks_with_measurements_where_baseline_missing=added_weeks,
                   scope="Source-dated modern geometry on a retrospective contour registry. Coverage checks only. "
                         "Individual captured tank changes are not weekly total Cushing inventory estimates. "
                         "USGS metadata proves catalog footprint coverage, not LAS point origin IDs.")
    for suffix, rows in (("source_tiles", list(metadata.values())), ("registry", registry),
                         ("pairs", list(ledger.values())), ("volumes", volumes), ("weeks", weekly),
                         ("geometry_comparison", overlap)):
        export(f"{STEM}_{suffix}", rows)
    write(DERIVED / f"{STEM}_summary.json", json.dumps(summary, indent=2).encode() + b"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
