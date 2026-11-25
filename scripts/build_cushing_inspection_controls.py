"""Recover dated operator gauges and match them to possible, independently mapped roofs."""

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import pdfplumber
from pypdf import PdfReader
from shapely.geometry import Point, Polygon, shape

from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import RAW, ROOT, write

STEM = "cushing_operator_inspection"
FACILITIES = {
    "110007388674": ("enbridge_blueknight", "1105"),
    "110007161588": ("enbridge_cushing", "1514"),
}
TERMINALS = {terminal for terminal, _ in FACILITIES.values()}
LOCAL_TIME = ZoneInfo("America/Chicago")
DOCS = RAW / "documents"
INSPECTION_PAGES = {
    "86232292": (2, 5),
    "86334379": (2,),
    "86467455": (2, 5),
    "86829489": tuple(range(2, 93, 3)),
    "86831645": (2,),
    "86943748": (2, 5, 8, 11),
    "87047598": (2, 5),
    "86943744": (2,),
    "87047602": (2, 5),
    "87047606": (2, 5),
    "87047610": (2, 5),
    "87047614": (2, 5),
    "87094907": (2, 5, 8, 11),
    "87176028": (2,),
}
RESOLVED = "resolved_in_geometry_diagnostic"
CORNERS = ("Upper Left", "Upper Right", "Lower Right", "Lower Left")


def source_inventory():
    rows = []
    for source in sorted(DOCS.glob("epa_echo_dfr_*.json")):
        report = json.loads(source.read_text())["Results"]
        if report["Message"] != "Success" or source.stem.rsplit("_", 1)[1] != report["RegistryID"]:
            raise ValueError(f"Unexpected EPA ECHO response in {source.name}")
        for entry in report["WebFireDocuments"]:
            ident, = parse_qs(urlparse(entry["DocumentLink"]).query)["ID"]
            path, = (DOCS / "source_pdfs").glob(f"epa_webfire_{ident}.*")
            rows.append(dict(
                document_id=ident, frs_id=report["RegistryID"],
                listed_submission_date=entry["ReportSubmissionDate"],
                title=entry["DocumentName"], source_url=entry["DocumentLink"],
                source_file=str(path.relative_to(ROOT)),
                source_index_file=str(source.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ))
    return rows


def inspection_value(raw):
    if raw in {"Empty", "OOS"}:
        return raw, None
    match = re.fullmatch(r"(\d+)'(?:\s*(\d+)\x22)?", raw)
    if not match:
        raise ValueError(f"Unexpected gauge entry {raw}")
    feet, inches = int(match[1]), int(match[2] or 0)
    if inches >= 12:
        raise ValueError(f"Invalid gauge entry {raw}")
    return "measured_level", feet + inches / 12


def date_value(raw):
    raw = raw.replace(".", "/")
    pattern = "%m/%d/%Y" if len(raw.split("/")[-1]) == 4 else "%m/%d/%y"
    return datetime.strptime(raw, pattern).date().isoformat()


def inspection_fields(text, page):
    values = {}
    for reference in page.get("/Annots", []):
        widget = reference.get_object()
        if widget.get("/Subtype") == "/Widget":
            parent_ref = widget.get("/Parent")
            parent = parent_ref.get_object() if parent_ref else {}
            values[widget.get("/T", parent.get("/T"))] = widget.get("/V", parent.get("/V"))
    if values.get("Liquid level during inspection"):
        return (values["Cushing EFRT Seal Gap Inspection Record"], values["Tank number"],
                values["Liquid level during inspection"], "pdf_form_values")
    text = text.replace("(cid:3)", " ")
    match = re.search(
        r"Date: (\d+/\d+/\d+) Age of seal.*?Tank number: (\d+) .*?"
        r"Liquid level during inspection: (\d+'\s*\d+\")", text, re.S,
    )
    if match:
        return (*match.groups(), "pdf_inline_text")
    lines = text.splitlines()
    if "Liquidlevelduringinspection:" not in lines[7].replace("::", ":"):
        raise ValueError("Unable to locate liquid level field on the flattened inspection form")
    return lines[2].split()[0], lines[4], lines[6].split(maxsplit=1)[1], "pdf_flattened_text"


def gauges(sources, by_equipment):
    found = {}
    for ident, pages in INSPECTION_PAGES.items():
        source = sources[ident]
        terminal, facility = FACILITIES[source["frs_id"]]
        path = ROOT / source["source_file"]
        reader = PdfReader(path)
        with pdfplumber.open(path) as document:
            for number in pages:
                raw_date, equipment, value, extraction = inspection_fields(
                    document.pages[number - 1].extract_text() or "", reader.pages[number - 1],
                )
                day = date_value(raw_date)
                state, feet = inspection_value(value)
                roofs = by_equipment[terminal, equipment]
                cohort, = {r["cohort_id"] for r in roofs}
                key = (terminal, equipment, day)
                if key in found:
                    previous = found[key]
                    if previous["source_document_id"] != ident or previous["reported_liquid_level"] != value:
                        raise ValueError(f"Conflicting reports for {key}")
                    previous["source_pages"] += f"|{number}"
                    continue
                found[key] = dict(
                    terminal_id=terminal, facility_id=facility, equipment_id=equipment, cohort_id=cohort,
                    inspection_local_date=day, inspection_timezone=str(LOCAL_TIME), inspection_time_local=None,
                    reported_liquid_level=value, reported_state=state, reported_liquid_level_ft=feet,
                    reported_liquid_level_m=feet * .3048 if feet is not None else None,
                    gauge_datum_verified=False, absolute_volume_verified=False,
                    possible_candidate_count=len(roofs),
                    possible_candidate_ids_json=json.dumps(sorted(r["candidate_id"] for r in roofs)),
                    individual_contour_verified=len(roofs) == 1,
                    source_document_id=ident, source_pages=str(number), source_value_extraction=extraction,
                    listed_submission_date=source["listed_submission_date"], source_url=source["source_url"],
                    source_file=source["source_file"], source_sha256=source["sha256"],
                )
    return sorted(found.values(), key=lambda r: (r["inspection_local_date"], r["equipment_id"]))


def refill_notices(sources, by_equipment):
    rows = []
    for source in sources.values():
        if source["frs_id"] not in FACILITIES or "refill" not in source["title"].lower():
            continue
        terminal, facility = FACILITIES[source["frs_id"]]
        with pdfplumber.open(ROOT / source["source_file"]) as document:
            body = " ".join(document.pages[0].extract_text().split())
        match = re.search(
            r"projected date of refill for Tanks? (.*?) (?:is|will be|would be) on or after ([A-Za-z]+ \d+, \d{4})",
            body,
        )
        if not match:
            raise ValueError(f"Could not read expected refill notice {source['document_id']}")
        revision = re.search(r"new refill date was moved to ([A-Za-z]+ \d+, \d{4})", body)
        for equipment in re.findall(r"\b\d{4}\b", match[1]):
            rows.append(dict(
                terminal_id=terminal, facility_id=facility, equipment_id=equipment,
                original_date_cited_in_revision=datetime.strptime(match[2], "%B %d, %Y").date().isoformat() if revision else None,
                projected_refill_local_date=datetime.strptime(
                    revision[1] if revision else match[2], "%B %d, %Y",
                ).date().isoformat(),
                projected_refill_date_basis="revised_target_date" if revision else "on_or_after",
                possible_candidate_count=len(by_equipment.get((terminal, equipment), ())),
                actual_refill_local_date=None, actual_landing_local_date=None, source_pdf_page=1, **source,
            ))
    return sorted(rows, key=lambda r: (r["projected_refill_local_date"], r["equipment_id"]))


def source_image_candidates():
    scenes = {}
    for year in (2025, 2026):
        path = RAW / "stac_supplemental" / "sentinel-2-l1c" / f"{year}.geojson"
        for item in json.loads(path.read_text())["features"]:
            properties = item["properties"]
            stamp = datetime.fromisoformat(properties["datetime"].replace("Z", "+00:00"))
            scenes[item["id"]] = (stamp.astimezone(LOCAL_TIME).date(), properties["datetime"],
                                  "Sentinel-2 L1C", properties.get("eo:cloud_cover"), path, shape(item["geometry"]))
    path = DERIVED / "cushing_usgs_b8_metadata_regional_catalog.csv"
    for item in read(path.stem):
        ident = item["Landsat Product Identifier L1"]
        if not ident.startswith(("LC08_", "LC09_")) or item["Day/Night Indicator"] != "DAY":
            continue
        stamp = datetime.fromisoformat(item["Start Time"]).replace(tzinfo=timezone.utc)
        if stamp.date().isoformat() != item["Date Acquired"].replace("/", "-"):
            raise ValueError(f"Landsat sensing day differs from catalog acquisition day for {ident}")
        footprint = Polygon([(float(item[f"Corner {c} Longitude"]), float(item[f"Corner {c} Latitude"])) for c in CORNERS])
        if not footprint.is_valid:
            raise ValueError(f"Invalid Landsat metadata footprint for {ident}")
        scenes[ident] = (stamp.astimezone(LOCAL_TIME).date(), stamp.isoformat(),
                         "Landsat 8/9 OLI", item["Scene Cloud Cover L1"], path, footprint)
    return scenes


def optical_observations():
    # Timing observations repeat some weekly fits. Keep one row per geometry, roof and exact product.
    fits = {}
    for source, clock in (("weekly_shadow_holdout_observations", "optical_capture_at_utc"),
                          ("shadow_timing_holdout_observations", "captured_at_utc")):
        for row in read(source):
            if row["terminal_id"] not in TERMINALS:
                continue
            key = tuple(row[k] for k in ("candidate_id", "geometry_mode", "version_id"))
            if key not in fits:
                fits[key] = dict(**row, capture_utc=row[clock], source_fit_file=f"data/derived/{source}.csv")
    return fits.values()


def match_scenes(ground, points):
    optical = defaultdict(list)
    for row in optical_observations():
        stamp = datetime.fromisoformat(row["capture_utc"].replace("Z", "+00:00"))
        optical[(row["candidate_id"], stamp.astimezone(LOCAL_TIME).date())].append(row)
    scenes = source_image_candidates()
    catalog, fits = [], []
    for row in ground:
        day = date.fromisoformat(row["inspection_local_date"])
        key = {k: row[k] for k in ("terminal_id", "equipment_id", "cohort_id", "inspection_local_date", "reported_state")}
        candidates = json.loads(row["possible_candidate_ids_json"])
        for ident, (capture_day, timestamp, sensor, cloud, path, footprint) in sorted(scenes.items()):
            offset = (capture_day - day).days
            if abs(offset) > 1:
                continue
            covered = [candidate for candidate in candidates if footprint.covers(points[candidate])]
            if covered:
                catalog.append(dict(
                    **key, source_catalog_id=ident, sensor=sensor, scene_capture_utc=timestamp,
                    scene_minus_inspection_local_days=offset, scene_cloud_cover_percent=cloud,
                    covered_possible_candidate_ids_json=json.dumps(covered),
                    source_catalog_file=str(path.relative_to(ROOT)), verified_roof_measurement=False,
                ))
        for candidate in candidates:
            for offset in (-1, 0, 1):
                capture_day = day + timedelta(days=offset)
                for fit in optical[candidate, capture_day]:
                    fields = {k: fit[k] for k in ("candidate_id", "geometry_mode", "observation_id", "version_id",
                                                 "capture_utc", "fit_status", "estimated_gap_m", "near_optimal_gap_min_m",
                                                 "near_optimal_gap_max_m", "source_fit_file")}
                    fits.append(dict(**key, **fields, scene_minus_inspection_local_days=offset,
                                     admissible_as_paired_gauge=False))
    return catalog, fits


def sentinel_quality(catalog, fits):
    """Explain which exact L1C chips can support a same-day inspection check."""
    candidates = [r for r in catalog if r["sensor"] == "Sentinel-2 L1C"]
    wanted = {r["source_catalog_id"] for r in candidates}
    products = {}
    for path in {ROOT / r["source_catalog_file"] for r in candidates}:
        for item in json.loads(path.read_bytes())["features"]:
            if item["id"] in wanted:
                products[item["id"]] = item["properties"]["s2:product_uri"]
    if products.keys() != wanted:
        raise ValueError("An inspection scene has no exact Sentinel-2 version")
    masks = {}
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        versions = sorted(set(products.values()))
        placeholders = ",".join("?" for _ in versions)
        for row in db.execute(f"""SELECT * FROM sentinel2_l1c_tank_quality
                               WHERE l1c_product_name IN ({placeholders})""", versions):
            key = row["l1c_product_name"], row["candidate_id"]
            if key in masks:
                raise ValueError(f"Multiple mask rows for one Sentinel-2 roof and version {key}")
            masks[key] = dict(row)
    existing = defaultdict(set)
    for row in fits:
        existing[row["version_id"], row["candidate_id"]].add(row["fit_status"])
    fields = ("optical_touched_cells_expected", "optical_cells_valid_in_all_bands", "opaque_roof_cells",
              "cirrus_roof_cells", "snow_ice_roof_cells", "red_detector_covered_roof_cells",
              "red_detector_ambiguous_roof_cells", "cloud_mask_basis", "source_directory", "tile_metadata_path")
    for row in candidates:
        version = products[row["source_catalog_id"]]
        for tank in json.loads(row["covered_possible_candidate_ids_json"]):
            mask = masks.get((version, tank), {})
            statuses = existing[version, tank]
            if mask and abs((datetime.fromisoformat(mask["captured_at_utc"])
                             - datetime.fromisoformat(row["scene_capture_utc"])).total_seconds()) >= 1:
                raise ValueError(f"Sentinel-2 mask time differs from exact catalog item {version}")
            if not mask:
                status = "exact_version_roof_mask_missing"
            elif any(mask[k] for k in ("opaque_roof_cells", "cirrus_roof_cells", "snow_ice_roof_cells")):
                status = "cloud_or_snow_flagged_roof"
            elif mask["optical_cells_valid_in_all_bands"] != mask["optical_touched_cells_expected"]:
                status = "incomplete_optical_coverage"
            elif (mask["red_detector_covered_roof_cells"] != mask["optical_touched_cells_expected"]
                  or mask["red_detector_ambiguous_roof_cells"]
                  or len(json.loads(mask["red_detector_with_view_options_json"])) != 1):
                status = "unresolved_b04_detector"
            else:
                status = "clear_existing_fit" if statuses else "clear_needs_fit"
            yield {k: row[k] for k in ("terminal_id", "equipment_id", "cohort_id", "inspection_local_date",
                                      "source_catalog_id", "scene_capture_utc", "scene_minus_inspection_local_days")} | dict(
                candidate_id=tank, version_id=version, quality_status=status,
                existing_fit_statuses_json=json.dumps(sorted(statuses)), **{k: mask.get(k) for k in fields},
            )


def main():
    by_equipment = defaultdict(list)
    for row in read("cushing_tank_volume_geometry_registry"):
        if row["terminal_id"] in TERMINALS:
            for equipment in json.loads(row["possible_equipment_ids_json"]):
                by_equipment[row["terminal_id"], equipment.removeprefix("CG-")].append(row)
    inventory = source_inventory()
    sources = {r["document_id"]: r for r in inventory}
    ground = gauges(sources, by_equipment)
    refill = refill_notices(sources, by_equipment)
    points = {r["candidate_id"]: Point(float(r["longitude"]), float(r["latitude"]))
              for r in read("tank_engineering_registry") if r["documented_terminal_id"] in TERMINALS}
    catalog, fits = match_scenes(ground, points)
    quality = list(sentinel_quality(catalog, fits))
    for name, rows in (("sources", inventory), ("levels", ground), ("refill_notices", refill),
                       ("catalog_candidates", catalog), ("optical_fit_candidates", fits), ("sentinel_quality", quality)):
        export(f"{STEM}_{name}", rows)
    states = Counter(row["reported_state"] for row in ground)
    summary = dict(
        archived_echo_detailed_facility_reports=len(list(DOCS.glob("epa_echo_dfr_*.json"))),
        archived_webfire_documents=len(inventory), webfire_documents_by_frs=dict(Counter(r["frs_id"] for r in inventory)),
        inspection_reports=len(INSPECTION_PAGES), tank_date_observations=len(ground),
        unique_equipment_ids=len({(r["terminal_id"], r["equipment_id"]) for r in ground}), state_counts=dict(states),
        inspection_rows_by_terminal=dict(Counter(r["terminal_id"] for r in ground)),
        numeric_height_ft_range=[min(r["reported_liquid_level_ft"] for r in ground if r["reported_state"] == "measured_level"),
                                 max(r["reported_liquid_level_ft"] for r in ground if r["reported_state"] == "measured_level")],
        projected_refill_tank_dates=len(refill),
        distinct_local_inspection_dates=len({r["inspection_local_date"] for r in ground}),
        same_day_catalog_tank_dates=len({(r["terminal_id"], r["equipment_id"], r["inspection_local_date"]) for r in catalog
                                         if r["scene_minus_inspection_local_days"] == 0}),
        same_day_resolved_fit_tank_dates=len({(r["terminal_id"], r["equipment_id"], r["inspection_local_date"]) for r in fits
                                              if r["scene_minus_inspection_local_days"] == 0 and r["fit_status"] == RESOLVED}),
        adjacent_day_or_same_day_resolved_fit_tank_dates=len({(r["terminal_id"], r["equipment_id"], r["inspection_local_date"]) for r in fits
                                                             if r["fit_status"] == RESOLVED}),
        individually_mapped_gauges=sum(r["individual_contour_verified"] for r in ground),
        verified_satellite_gauge_pairs=sum(r["admissible_as_paired_gauge"] for r in fits),
        sentinel_exact_products_screened=len({r["version_id"] for r in quality}),
        sentinel_inspection_candidate_status=dict(Counter(r["quality_status"] for r in quality)),
        satellite_catalog_scope="Local Sentinel-2 L1C STAC and USGS regional Landsat metadata snapshots. "
                               "Catalog matches are not confirmed roof observations or a complete pass search.",
    )
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
