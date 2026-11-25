"""Index the NRC's original notifications and locate potential tank controls."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta, timezone

from openpyxl import load_workbook
from pyproj import Geod
from shapely.geometry import Point, Polygon, box, shape

from build_cushing_inspection_controls import CORNERS, LOCAL_TIME, RESOLVED
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from build_weekly_shadow_validation import DATED
from collect_public import BBOX, RAW, ROOT, write

STEM = "cushing_nrc_incidents"
STORE = RAW / "documents" / "uscg_nrc"
URL = "https://nrc.uscg.mil/FOIAFiles/"
YEARS = range(2017, 2027)
WEST, SOUTH, EAST, NORTH = map(float, BBOX.split(","))
UPDATE = re.compile(r"\b(?:NRC\s+)?REPORT\s*(?:NUMBER|#)?\s*(\d{6,8})\b", re.I)
GEOD = Geod(ellps="WGS84")
ROSE_ROCK_ADDRESS_SOURCE = "https://www.sec.gov/Archives/edgar/data/1489136/000119312516727586/d438323dex101.htm"
ROSE_ROCK_CURRENT_NAME_SOURCE = "https://mcgpamidstream.org/event/2025-crude-terminal-tour/"


def workbook_rows(workbook, name, fields):
    rows = iter(workbook[name].values)
    header = next(rows)
    indexes = [header.index(field) for field in ("SEQNOS", *fields)]
    if len(set(indexes)) != len(indexes):
        raise ValueError(f"Duplicate fields in {name}")
    for number, row in enumerate(rows, 2):
        if row[indexes[0]] not in (None, ""):
            yield number, int(row[indexes[0]]), {field: row[i] for field, i in zip(fields, indexes[1:], strict=True)}


def coordinates(row):
    result = []
    for prefix, hemispheres, limit in (("LONG", "EW", 180), ("LAT", "NS", 90)):
        values = [row[f"{prefix}_{part}"] for part in ("DEG", "MIN", "SEC", "QUAD")]
        if any(value in ("", None) for value in values) or values[-1] not in hemispheres:
            return None, None
        degrees, minutes, seconds = map(float, values[:3])
        if not (0 <= degrees <= limit and 0 <= minutes < 60 and 0 <= seconds < 60):
            return None, None
        value = degrees + minutes / 60 + seconds / 3600
        if value > limit:
            return None, None
        result.append(-value if values[-1] in ("S", "W") else value)
    return tuple(result)


def file_record(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return dict(source_file=str(path.relative_to(ROOT)), source_url=URL + path.name, source_bytes=path.stat().st_size,
                source_sha256=digest)


def annual(year):
    path = STORE / f"CY{year % 100:02d}.xlsx"
    commons = ("INCIDENT_DATE_TIME", "INCIDENT_DTG", "TYPE_OF_INCIDENT", "INCIDENT_CAUSE", "LOCATION_NEAREST_CITY",
               "LOCATION_STATE", "LOCATION_COUNTY", "INCIDENT_LOCATION", "LOCATION_ADDRESS", "DESCRIPTION_OF_INCIDENT", *(
                   f"{axis}_{part}" for axis in ("LAT", "LONG") for part in ("DEG", "MIN", "SEC", "QUAD")))
    reports = {}
    materials = []
    with closing(load_workbook(path, read_only=True, data_only=True)) as workbook:
        count = 0
        for number, seq, row in workbook_rows(workbook, "INCIDENT_COMMONS", commons):
            count += 1
            lon, lat = coordinates(row)
            city, state = (str(row[f"LOCATION_{f}"] or "").strip().upper() for f in ("NEAREST_CITY", "STATE"))
            city_match = city == "CUSHING" and state == "OK"
            bbox_match = lon is not None and WEST <= lon <= EAST and SOUTH <= lat <= NORTH
            if not (city_match or bbox_match):
                continue
            if seq in reports:
                raise ValueError(f"Duplicate NRC report {seq}")
            narrative = row["DESCRIPTION_OF_INCIDENT"] or ""
            update = UPDATE.search(narrative[:350]) if "UPDATE" in narrative[:160].upper() else None
            reports[seq] = dict(report_id=seq, workbook_year=year, commons_excel_row=number,
                in_research_bbox=bbox_match, city_is_cushing_ok=city_match, incident_local_time_as_reported=row["INCIDENT_DATE_TIME"],
                time_basis_as_reported=row["INCIDENT_DTG"], incident_type=row["TYPE_OF_INCIDENT"], cause=row["INCIDENT_CAUSE"],
                nearest_city=city, state=state, county=row["LOCATION_COUNTY"],
                incident_location_as_reported=row["INCIDENT_LOCATION"], incident_address_as_reported=row["LOCATION_ADDRESS"],
                longitude=lon, latitude=lat,
                follow_up_to_report_id=int(update[1]) if update and int(update[1]) != seq else None)

        joins = (
            ("CALLS", dict(responsible_company="RESPONSIBLE_COMPANY", nrc_received_time_as_reported="DATE_TIME_RECEIVED")),
            ("INCIDENTS", dict(tank_id="TANK_ID", tank_description="DESCRIPTION_OF_TANK",
                reported_tank_capacity="CAPACITY_OF_TANK", capacity_unit="CAPACITY_OF_TANK_UNITS",
                reported_amount_in_tank="ACTUAL_AMOUNT", amount_in_tank_unit="ACTUAL_AMOUNT_UNITS")),
        )
        for name, fields in joins:
            found = set()
            for number, seq, row in workbook_rows(workbook, name, fields.values()):
                if seq in reports:
                    if seq in found:
                        raise ValueError(f"Duplicate {name} row for NRC report {seq}")
                    found.add(seq)
                    reports[seq].update({f"{name.lower()}_excel_row": number, **{
                        key: row[value] for key, value in fields.items()}})
            for seq in reports.keys() - found:
                reports[seq].update(dict.fromkeys((f"{name.lower()}_excel_row", *fields)))

        fields = ("NAME_OF_MATERIAL", "AMOUNT_OF_MATERIAL", "UNIT_OF_MEASURE")
        for number, seq, row in workbook_rows(workbook, "MATERIAL_INVOLVED", fields):
            if seq in reports:
                materials.append(dict(report_id=seq, workbook_year=year, material_excel_row=number,
                    material=row[fields[0]], reported_released_amount=row[fields[1]], released_amount_unit=row[fields[2]]))

    return dict(workbook_year=year, commons_records_scanned=count, local_records=len(reports), **file_record(path)), list(reports.values()), materials


def possible_contours(reports):
    with closing(read_only(SOURCE)) as db:
        records = db.execute("""SELECT t.candidate_id,t.geometry_geojson,g.longitude,g.latitude,g.documented_terminal_id,
                                      g.documented_roof_type,g.cohort_id,g.possible_equipment_ids_json,g.first_seen
                               FROM tank_candidate t JOIN tank_engineering_registry g USING(candidate_id)""").fetchall()
    geometries = [(shape(json.loads(geo)), roof, lon, lat, terminal, roof_type, cohort, json.loads(ids or "[]"), first_seen)
                  for roof, geo, lon, lat, terminal, roof_type, cohort, ids, first_seen in records]
    output = []
    half_second = 0.5 / 3600
    for row in reports:
        lon, lat = row["longitude"], row["latitude"]
        if not row["in_research_bbox"]:
            continue
        point = Point(lon, lat)
        envelope = box(lon - half_second, lat - half_second, lon + half_second, lat + half_second)
        for geo, roof, x, y, terminal, roof_type, cohort, numbers, first_seen in geometries:
            if not geo.intersects(envelope):
                continue
            equipment = str(row["tank_id"] or "").upper().removeprefix("CG-").removeprefix("T-")
            normalized = {str(n).upper().removeprefix("CG-").removeprefix("T-") for n in numbers}
            output.append(dict(report_id=row["report_id"], candidate_id=roof, terminal_id=terminal,
                roof_type=roof_type, cohort_id=cohort, equipment_id_in_contour_cohort=bool(equipment and equipment in normalized),
                rounded_point_inside_contour=geo.covers(point), half_arcsecond_box_inside_contour=geo.covers(envelope),
                centre_distance_m=GEOD.inv(lon, lat, x, y)[2], first_seen_aerial_date=first_seen,
                individual_contour_verified=False))
    return sorted(output, key=lambda r: (r["report_id"], r["centre_distance_m"], r["candidate_id"]))


def rose_rock_inventory_coverage(by_id):
    event, update = (by_id[ident] for ident in (1402673, 1402790))
    if any(r["incident_address_as_reported"] != "3710 N LITTLE AVENUE" or r["tank_id"]
           or r["reported_amount_in_tank"] != 175000 or r["amount_in_tank_unit"] != "BARREL(S)" for r in (event, update)):
        raise ValueError("The retained inventory notifications disagree with the site evidence")
    if update["follow_up_to_report_id"] != event["report_id"]:
        raise ValueError("The inventory update no longer links to its original notification")
    incident = datetime.strptime(event["incident_local_time_as_reported"], "%m/%d/%Y %H:%M").replace(tzinfo=LOCAL_TIME)
    source = RAW / "stac" / "sentinel-2-l1c" / f"{incident.year}.geojson"
    features = json.loads(source.read_bytes())["features"]
    output = []
    measurement_keys = set()
    with closing(read_only(SOURCE)) as db:
        roofs = {tank: Point(lon, lat) for tank, lon, lat in db.execute("""SELECT candidate_id, longitude, latitude
            FROM tank_engineering_registry WHERE cohort_id LIKE 'rose_rock_north_%'
                AND documented_roof_type = 'EFR' AND first_seen <= ?""", (incident.date().isoformat(),))}
        for item in features:
            props = item["properties"]
            stamp = datetime.fromisoformat(props["datetime"])
            if abs(stamp.astimezone(LOCAL_TIME).date() - incident.date()) > timedelta(days=5):
                continue
            footprint = shape(item["geometry"])
            covered = {tank for tank, point in roofs.items() if footprint.covers(point)}
            if not covered:
                continue
            version = props["s2:product_uri"]
            quality = {r[0]: r[1:] for r in db.execute("""SELECT candidate_id, opaque_roof_cells, cirrus_roof_cells,
                        snow_ice_roof_cells, optical_touched_cells_expected, optical_cells_valid_in_all_bands
                        FROM sentinel2_l1c_tank_quality WHERE l1c_product_name=?""", (version,)) if r[0] in covered}
            output.append(dict(report_id=event["report_id"], sensor="Sentinel-2 L1C", product_id=version,
                scene_capture_utc=stamp.isoformat(), hours_from_reported_incident=(stamp - incident).total_seconds()/3600,
                roof_candidates_covered=len(covered), roof_quality_rows=len(quality),
                no_cloud_snow_and_all_bands_valid_roofs=sum(not any(q[:3]) and q[3] == q[4] for q in quality.values()),
                scene_cloud_percent=props["eo:cloud_cover"], catalog_source_file=str(source.relative_to(ROOT))))
            measurement_keys.update((version, roof) for roof in covered)
    source = DERIVED / "cushing_usgs_b8_metadata_regional_catalog.csv"
    for item in read(source.stem):
        ident = item["Landsat Product Identifier L1"]
        if not ident.startswith(("LC08_", "LC09_")) or item["Day/Night Indicator"] != "DAY":
            continue
        stamp = datetime.fromisoformat(item["Start Time"]).replace(tzinfo=timezone.utc)
        if abs(stamp.astimezone(LOCAL_TIME).date() - incident.date()) > timedelta(days=5):
            continue
        footprint = Polygon([(float(item[f"Corner {c} Longitude"]), float(item[f"Corner {c} Latitude"])) for c in CORNERS])
        covered = {tank for tank, point in roofs.items() if footprint.covers(point)}
        if covered:
            output.append(dict(report_id=event["report_id"], sensor="Landsat 8/9 OLI", product_id=ident,
                scene_capture_utc=stamp.isoformat(), hours_from_reported_incident=(stamp - incident).total_seconds()/3600,
                roof_candidates_covered=len(covered), roof_quality_rows=None, no_cloud_snow_and_all_bands_valid_roofs=None,
                scene_cloud_percent=item["Scene Cloud Cover L1"], catalog_source_file=str(source.relative_to(ROOT))))
    fits = []
    for row in read("shadow_timing_holdout_observations"):
        if (row["version_id"], row["candidate_id"]) in measurement_keys and row["geometry_mode"] == DATED:
            fields = ("version_id", "candidate_id", "captured_at_utc", "geometry_mode", "measurement_source",
                      "observation_id", "fit_status", "estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m")
            fits.append(dict(report_id=event["report_id"], **{f: row[f] for f in fields}, admissible_as_paired_gauge=False))
    sentinel = [r for r in output if r["sensor"] == "Sentinel-2 L1C"]
    nearest = min(sentinel, key=lambda r: abs(r["hours_from_reported_incident"]))
    same_day_passes = {r["candidate_id"] for r in fits
        if r["fit_status"] == RESOLVED
        and datetime.fromisoformat(r["captured_at_utc"]).astimezone(LOCAL_TIME).date() == incident.date()}
    return sorted(output, key=lambda r: r["scene_capture_utc"]), fits, dict(report_id=event["report_id"],
        follow_up_report_id=update["report_id"], incident_local_time=incident.isoformat(), reported_amount_in_tank_bbl=175000,
        operator_asset="Energy Transfer Cushing North / Rose Rock", address_disclosure_url=ROSE_ROCK_ADDRESS_SOURCE,
        current_asset_name_source_url=ROSE_ROCK_CURRENT_NAME_SOURCE, roof_candidates=len(roofs),
        catalog_sentinel2_products=len(sentinel), catalog_landsat_products=len(output)-len(sentinel),
        nearest_sentinel2_capture_utc=nearest["scene_capture_utc"],
        nearest_sentinel2_hours_from_reported_incident=nearest["hours_from_reported_incident"],
        same_day_roofs_with_dated_passing_optical_fit=len(same_day_passes), exact_roof_id_known=False)


def main():
    with ProcessPoolExecutor(max_workers=3) as pool:
        years = list(pool.map(annual, YEARS))
    sources, reports, materials = [], [], []
    for source, annual_reports, annual_materials in years:
        sources.append(source)
        reports.extend(annual_reports)
        materials.extend(annual_materials)
    dictionary = file_record(STORE / "DataDictionary.xlsx")
    sources.append(dict(workbook_year=None, commons_records_scanned=None, local_records=None, **dictionary))
    by_id = {r["report_id"]: r for r in reports}
    if len(by_id) != len(reports):
        raise ValueError("Duplicate local report across NRC years")
    contours = possible_contours(reports)
    candidates = defaultdict(list)
    for row in contours:
        candidates[row["report_id"]].append(row)
    inventories = [r for r in reports if r["reported_amount_in_tank"] not in (None, "")
                   and r["amount_in_tank_unit"] not in (None, "", "UNKNOWN AMOUNT")]
    inspections = []
    for row in read("cushing_operator_inspection_levels"):
        for candidate in contours:
            nrc = by_id[candidate["report_id"]]
            if (str(nrc["tank_id"]) == row["equipment_id"] and candidate["terminal_id"] == row["terminal_id"]
                    and candidate["equipment_id_in_contour_cohort"] and candidate["rounded_point_inside_contour"]
                    and candidate["candidate_id"] in json.loads(row["possible_candidate_ids_json"])):
                inspections.append(dict(report_id=nrc["report_id"], candidate_id=candidate["candidate_id"],
                    terminal_id=row["terminal_id"], equipment_id=row["equipment_id"],
                    inspection_local_date=row["inspection_local_date"], reported_level_m=row["reported_liquid_level_m"],
                    reported_state=row["reported_state"], inspection_document_id=row["source_document_id"],
                    inspection_source_pages=row["source_pages"], individual_contour_verified=False,
                    admissible_as_paired_gauge=False))
    for row in reports:
        row["spatial_candidate_count"] = len(candidates[row["report_id"]])
    coverage, fits, inventory_event = rose_rock_inventory_coverage(by_id)
    for suffix, rows in (("sources", sources), ("reports", reports), ("materials", materials),
                         ("contours", contours), ("inspection_links", inspections),
                         ("rose_rock_scenes", coverage), ("rose_rock_sentinel2_fits", fits)):
        export(f"{STEM}_{suffix}", rows)
    summary = dict(source_index="https://nrc.uscg.mil/Default.aspx", archive_years=list(YEARS),
        commons_records_scanned=sum(source["commons_records_scanned"] for source in sources[:-1]),
        local_reports=len(reports), material_rows=len(materials),
        records_by_type=Counter(r["incident_type"] for r in reports),
        linked_follow_up_reports=sum(r["follow_up_to_report_id"] in by_id for r in reports),
        coordinate_reports_in_bbox=sum(r["in_research_bbox"] for r in reports), contour_candidates=len(contours),
        inspection_links=len(inspections), reported_inventory_record_ids=[r["report_id"] for r in inventories],
        rose_rock_inventory_event=inventory_event,
        first_public_availability_verified=False, admitted_inventory_measurements=0)
    payload = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
