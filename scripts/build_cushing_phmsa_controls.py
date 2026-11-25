"""Index PHMSA reports and cross-check named tank incidents against the optical registry."""

import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from zipfile import ZipFile
from zoneinfo import ZoneInfo

from pypdf import PdfReader
from pyproj import Geod
from shapely.geometry import Point

from build_l1c_quality import candidate_layer
from build_cushing_height_validation_groups import optical_pass
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_public import BBOX, RAW, ROOT, write

STEM = "cushing_phmsa_controls"
STORE = RAW/"documents"/"phmsa_hazardous_liquid_incidents"
ARCHIVE = "accident_hazardous_liquid_jan2010_present"
ARCHIVE_URL = f"https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/data_statistics/pipeline/{ARCHIVE}.zip"
FORM = "https://portal.phmsa.dot.gov/PDFGenerator/getPublicReport/ODES_HazLiquid.pdf?rptid="
OCC = RAW/"documents"/"occ_cushing"/"ogcd-incidents.csv.gz"
OCC_URL = "https://oklahoma.gov/content/dam/ok/en/occ/documents/og/ogdatafiles/ogcd-incidents.csv"
WEST, SOUTH, EAST, NORTH = map(float, BBOX.split(","))
CHICAGO = ZoneInfo("America/Chicago")
WGS84 = Geod(ellps="WGS84")
FIELD_MAP = dict(report_id="REPORT_NUMBER", revision_id="SUPPLEMENTAL_NUMBER", report_kind="REPORT_TYPE",
    initial_report_received="REPORT_RECEIVED_DATE", operator_id="OPERATOR_ID", operator="NAME", facility="PIPE_FAC_NAME",
    segment="SEGMENT_NAME", city="ONSHORE_CITY_NAME", county="ONSHORE_COUNTY_NAME", state="ONSHORE_STATE_ABBREVIATION",
    longitude="LOCATION_LONGITUDE", latitude="LOCATION_LATITUDE", reported_event_local="LOCAL_DATETIME",
    reported_identified_local="INCIDENT_IDENTIFIED_DATETIME", reported_confirmed_local="CONFIRMED_DISCOVERY_DATETIME",
    reported_zone="TIME_ZONE", reported_daylight_saving="DAYLIGHT_SAVINGS_IND", commodity="COMMODITY_RELEASED_TYPE",
    system_part="SYSTEM_PART_INVOLVED", item="ITEM_INVOLVED", tank_failure_path="TANK_VESSEL_SUBTYPE",
    installation_year="INSTALLATION_YEAR", safe_fill_limit_ft="SAFE_FILL_LEVEL",
    safe_fill_limit_exceeded_as_reported="SAFE_FILL_LEVEL_EXCEEDED_IND", unintentional_release_bbl="UNINTENTIONAL_RELEASE_BBLS",
    intentional_release_bbl="INTENTIONAL_RELEASE_BBLS", recovered_bbl="RECOVERED_BBLS", nrc="NRC_RPT_NUM",
    additional_nrc="ADDITIONAL_NRC_REPORT_NUMBERS", cause="CAUSE", narrative="NARRATIVE")

# The clock belongs to the stated event. Later actions in the same report may have no timestamp.
CASES = {
    "20260087": ("enbridge_cushing", "3360", "tank_disconnected_out_of_service_at_line_incident", "TANK LINE 3360"),
    "20250064": ("enbridge_blueknight", "2525", "release_onto_roof_then_out_of_service", "TMG TANK 2525"),
    "20240170": ("semgroup_cushing", "2536", "mixer_incident_then_pump_down", "TANK 2536 MIXER"),
    "20230259": ("enbridge_blueknight", "2514", "release_onto_roof_and_cleanup", "BURPED THE SEAL ON TANK 2514"),
    "20230127": ("plains_cushing", "3600", "earliest_possible_release_then_out_of_service", "EARLIEST POSSIBLE TIME"),
    "20230011": ("wildhorse", "215", "mixer_leak_during_operation", "BREAKOUT TANK 215"),
    "20220074": ("enbridge_blueknight", "2522", "roof_release_then_drain_for_inspection", "THE TANK WAS DRAINED AND DISCONNECTED"),
    "20210034": ("amid_cushing", "41154", "prior_inadvertent_roof_landing", "INADVERTENTLY LANDING ON THIS SINGLE LEG"),
    "20200339": ("enbridge_cushing", "3012", "prior_internal_roof_landings", "SET DOWN ON ITS LEGS MULTIPLE TIMES"),
    "20190234": ("enterprise_cushing", "41123", "tank_drained_then_returned_to_service", "TANK #41123"),
    "20190011": ("magellan_shinn_pence", "2533", "initial_fill_then_emptied_and_cleaned", "IN THE PROCESS OF BEING FILLED"),
    "20160078": ("enbridge_cushing", "2229", "prior_roof_landing_and_later_product_transfer", "LANDING THE ROOF"),
    "20150464": ("enterprise_cushing", "41123", "fill_line_release", "TANK 23 FLUSH LINE"),
    "20150096": ("enbridge_cushing", "3010", "roof_floated_before_mixer_discovery", "AFTER FLOATING THE ROOF ON TANK 3010"),
    "20130208": ("enbridge_cushing", "3013", "fill_line_release_then_tank_deinventory", "TANK 3013"),
    "20130042": ("magellan_cushing", "2512", "near_empty_and_invalid_radar_reading", "TANK 2512 WAS NEAR EMPTY"),
}


def source_rows(path):
    result = []
    with ZipFile(path) as archive:
        with archive.open(ARCHIVE+".txt") as stream:
            rows = csv.DictReader(io.TextIOWrapper(stream, encoding="cp1252", newline=""), delimiter="\t", quoting=csv.QUOTE_NONE)
            if len(set(rows.fieldnames)) != len(rows.fieldnames):
                raise ValueError("Duplicate PHMSA fields")
            for record, row in enumerate(rows, 2):
                if None in row or row["NARRATIVE"] is None:
                    raise ValueError(f"Malformed PHMSA record {record}")
                lon, lat = (float(row[k]) if row[k] else None for k in ("LOCATION_LONGITUDE", "LOCATION_LATITUDE"))
                inside = lon is not None and lat is not None and WEST <= lon <= EAST and SOUTH <= lat <= NORTH
                city = row["ONSHORE_CITY_NAME"].strip().upper() == "CUSHING" and row["ONSHORE_STATE_ABBREVIATION"] == "OK"
                if inside or city:
                    result.append(dict(source_record_number=record, in_research_bbox=inside, city_is_cushing_ok=city,
                                       **{k: row[v] for k, v in FIELD_MAP.items()}))
    if len({row["report_id"] for row in result}) != len(result):
        raise ValueError("More than one archive row per local PHMSA report")
    return record-1, result


def norm(number):
    return str(number).upper().removeprefix("CG-").removeprefix("T-")


def candidates():
    roofs = {row["candidate_id"]: row for row in read("tank_engineering_registry")}
    polygons = dict((*candidate_layer("osm"), *candidate_layer("supplemental")))
    shinn = {row["candidate_id"]: row for row in read("shinn_pence_contour_reconciliation")}
    for key, roof in roofs.items():
        if key in shinn:
            s = shinn[key]
            yield roof, polygons[key], s["site_id"], s["documented_group_roof_type"], json.loads(s["possible_tank_numbers_json"] or "[]")
        else:
            yield (roof, polygons[key], roof["documented_terminal_id"], roof["documented_roof_type"],
                   json.loads(roof["possible_equipment_ids_json"] or "[]"))


def selected_cases(reports):
    polygons = list(candidates())
    entries, spatial = [], []
    for ident, (terminal, number, kind, anchor) in CASES.items():
        row = reports[ident]
        if anchor not in row["narrative"].upper():
            raise ValueError(f"Reviewed narrative changed for {ident}")
        stamp = datetime.strptime(row["reported_event_local"], "%m/%d/%Y %H:%M")
        zone = row["reported_zone"]
        utc = None
        if zone == "CENTRAL" and row["reported_daylight_saving"] in {"YES", "NO"}:
            local = stamp.replace(tzinfo=CHICAGO)
            if bool(local.dst()) == (row["reported_daylight_saving"] == "YES"):
                utc = local.astimezone(ZoneInfo("UTC")).isoformat()
        point = Point(float(row["longitude"]), float(row["latitude"]))
        matches, assigned = [], []
        for roof, outline, site, roof_type, numbers in polygons:
            if site != terminal:
                continue
            listed = number in {norm(n) for n in numbers}
            inside = outline.covers(point)
            if not (listed or inside):
                continue
            key = roof["candidate_id"]
            if listed:
                matches.append(key)
            if inside:
                assigned.append(key)
            distance = WGS84.inv(point.x, point.y, float(roof["longitude"]), float(roof["latitude"]))[2]
            spatial.append(dict(report_id=ident, terminal_id=terminal, equipment_id=number, candidate_id=key, roof_type=roof_type,
                                equipment_in_documented_group=listed, incident_point_inside_outline=inside, centre_distance_m=distance,
                                first_aerial_detection=roof["first_seen"],
                                first_aerial_detection_by_incident=bool(roof["first_seen"] and roof["first_seen"] <= stamp.date().isoformat()),
                                individual_contour_verified=False))
        entries.append(dict(report_id=ident, terminal_id=terminal, equipment_id=number, observation_kind=kind,
                            event_local_date=stamp.date().isoformat(), reported_event_local=row["reported_event_local"],
                            reported_event_utc=utc, event_clock_basis="reported_zone_and_dst" if utc else "timezone_unverified_in_form",
                            safe_fill_limit_ft=row["safe_fill_limit_ft"], safe_fill_limit_exceeded_as_reported=row["safe_fill_limit_exceeded_as_reported"],
                            unintentional_release_bbl=row["unintentional_release_bbl"],
                            group_candidate_ids_json=json.dumps(sorted(matches)), incident_point_inside_candidate_ids_json=json.dumps(sorted(assigned)),
                            individual_contour_verified=False, admitted_absolute_inventory_label=False))
    return entries, spatial


def ground_links(cases, spatial):
    by_tank = defaultdict(list)
    for entry in cases:
        by_tank[entry["terminal_id"], entry["equipment_id"]].append(entry)
    hits = defaultdict(set)
    for row in spatial:
        if row["incident_point_inside_outline"] and row["equipment_in_documented_group"] and row["first_aerial_detection_by_incident"]:
            hits[row["report_id"]].add(row["candidate_id"])
    optical = defaultdict(list)
    for row in read("cushing_validation_registry_candidate_optical_links"):
        optical[row["event_id"], row["candidate_id"]].append(row)
    for row in read("cushing_validation_registry_operator_events"):
        for case in by_tank[row["terminal_id"], norm(row["equipment_id"])]:
            possible = set(json.loads(row["possible_candidate_ids_json"]))
            overlaps = hits[case["report_id"]] & possible
            preferred = next(iter(overlaps)) if len(overlaps) == 1 and len(hits[case["report_id"]]) == 1 else None
            fits = optical[row["event_id"], preferred]
            same_day = [r for r in fits if r["local_date_offset"] == "0"]
            yield dict(event_id=row["event_id"], report_id=case["report_id"], terminal_id=row["terminal_id"],
                       equipment_id=row["equipment_id"], inspection_local_date=row["local_date"], incident_local_date=case["event_local_date"],
                       inspection_state=row["state"], inspection_liquid_level_m=row["reported_liquid_level_m"],
                       original_candidate_count=len(possible), preferred_candidate_from_incident_point=preferred,
                       different_dates=(date.fromisoformat(row["local_date"]) != date.fromisoformat(case["event_local_date"])),
                       existing_optical_fit_rows=len(fits), same_day_fit_statuses_json=json.dumps(sorted({r["fit_status"] for r in same_day})),
                       same_day_passing_fits=sum(r["fit_status"] == RESOLVED for r in same_day),
                       individual_contour_verified=False, admitted_absolute_inventory_label=False)


def pdf_sources(reports):
    phmsa = "https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/docs/"
    magellan = "https://primis.phmsa.dot.gov/enforcement-documents/420205009/420205009_"
    files = {
        "20130208_investigation": (STORE/"enbridge_2013_3013.pdf", phmsa+"safety-reports/17981/enbridge2013-05-17-final-internet.pdf"),
        "20150464_investigation": (STORE.parent/"phmsa_enterprise_cushing_west_2015.pdf",
            phmsa+"inspections-and-investigations/17861/enterprisecrudepipelinellccushingwestterminal20151201redacted.pdf"),
        "20110206_investigation": (STORE.parent/"phmsa_enterprise_cushing_east_2011.pdf", phmsa+"Enterprise_Crude_Pipeline_022111_Cushing_Terminal_OK.pdf"),
        "20190011_notice": (STORE/"magellan_2018_2533_notice.pdf", magellan+"%20NOPV%20PCP_05122020.pdf"),
        "20190011_final_order": (STORE/"magellan_2018_2533_final_order.pdf", magellan+"Final%20Order_09112020_text.pdf"),
        "20190011_operator_response": (STORE/"magellan_2018_2533_response.pdf", magellan+
            "Operator%20Response%20to%20Notice%20and%20Request%20for%20a%20Revised%20Case%20File%20and%20a%20Revised%20Proposed%20Civil%20Penalty%20Worksheet_06152020.pdf"),
        "20190011_form": (STORE/"magellan_2018_2533_7000_1.pdf", FORM+"ctzvcfvv61tbbH8RwUReXA%3D%3D"),
    }
    for ident in CASES:
        revision = reports[ident]["revision_id"]
        path = STORE/f"form_{ident}_{revision}.pdf"
        if path.is_file():
            files[f"{ident}_current_form"] = path, FORM+revision
    for key, (path, url) in files.items():
        blob = path.read_bytes()
        pdf = PdfReader(io.BytesIO(blob))
        ident = key.split("_", 1)[0]
        if key.endswith("_form") and not re.search(fr"\b{ident}\s*-\s*{reports[ident]['revision_id']}\b", pdf.pages[0].extract_text()):
            raise ValueError(f"Unexpected PHMSA form number or revision for {path}")
        yield dict(source_id=key, report_id=ident, local_pdf=str(path.relative_to(ROOT)), source_url=url,
                   pages=len(pdf.pages), bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())


def reported_quantities(documents):
    sources = {row["source_id"]: row for row in documents}
    source = sources["20130208_investigation"]
    pages = PdfReader(ROOT/Path(source["local_pdf"])).pages
    for day, page, number, kind, anchor in (
        ("2013-05-08", 2, 50000, "injected_to_float_internal_roof", "injection of 50,000 barrels"),
        ("2013-05-10", 2, 1879, "gauge_recalibration_without_oil_added", "1,879 barrels although no additional oil"),
        ("2013-05-18", 7, -1573, "24_hour_accounted_inventory_change", "1,573-barrel-loss in tank volume"),
        ("2013-05-18", 2, 2246, "released_from_fill_line_during_incident", "2,246 barrels of crude oil were released"),
    ):
        if anchor not in " ".join(pages[page-1].extract_text().split()):
            raise ValueError(f"PHMSA investigation excerpt differs on PDF page {page}")
        yield dict(report_id="20130208", terminal_id="enbridge_cushing", equipment_id="3013", roof_type="IFR",
                   reported_local_date=day, quantity_kind=kind, quantity_bbl=number, source_id=source["source_id"],
                   source_pdf_page=page, source_sha256=source["sha256"], admitted_absolute_inventory_label=False)


def rose_rock_inventory(report, case, spatial):
    ids = set(json.loads(report["local_nrc_report_ids_json"]))
    records = [r for r in read("cushing_nrc_incidents_reports") if r["report_id"] in ids]
    links = [r for r in spatial if r["report_id"] == report["report_id"] and r["incident_point_inside_outline"]
             and r["equipment_in_documented_group"] and r["first_aerial_detection_by_incident"]]
    match, = links
    if ids != {"1402673", "1402790"} or len(records) != 2 or case["equipment_id"] != "2536":
        raise ValueError("The Rose Rock incident and NRC notifications do not reconcile")
    confirmed_local = datetime.strptime(report["reported_confirmed_local"], "%m/%d/%Y %H:%M").replace(tzinfo=CHICAGO)
    if any(r["incident_local_time_as_reported"] != report["reported_confirmed_local"] or
           (r["reported_amount_in_tank"], r["amount_in_tank_unit"]) != ("175000", "BARREL(S)") for r in records):
        raise ValueError("Rose Rock incident clock or reported stock differs across NRC reports")
    earlier = [r for r in read("cushing_nrc_rose_rock_landsat_lidar_2022_inventory_candidates")
               if r["capacity_at_least_reported_inventory"] == "True"]
    roof = match["candidate_id"]
    permit, = [r for r in earlier if r["candidate_id"] == roof]
    rows = []
    for sensor, suffix, clock, method_fields in (
        ("Sentinel-2", "spectral_fits", "captured_at_utc", [(m, f"{m}_fit_status", f"{m}_gap_m") for m in ("red", "nir", "joint")]),
        ("Landsat", "fits", "optical_capture_at_utc", [("b8", "fit_status", "estimated_gap_m")]),
    ):
        stem = f"cushing_nrc_rose_rock_landsat_lidar_2022_{suffix}"
        for i, row in enumerate(read(stem), 2):
            if row["candidate_id"] != roof:
                continue
            capture = datetime.fromisoformat(row[clock])
            for method, fit, gap in method_fields:
                rows.append(dict(report_id=report["report_id"], candidate_id=roof, sensor=sensor, method=method,
                                 product_id=row["version_id"], optical_acquisition_id=optical_pass(row["version_id"]),
                                 optical_capture_utc=capture.isoformat(), hours_from_incident_confirmation=(capture-confirmed_local).total_seconds()/3600,
                                 same_local_day=capture.astimezone(CHICAGO).date() == confirmed_local.date(),
                                 fit_status=row[fit], fitted_roof_gap_m=row[gap], source_table=f"data/derived/{stem}.csv", source_record_number=i,
                                 stock_measurement_time_known=False, admitted_absolute_inventory_label=False))
    qualified = [r for r in rows if r["fit_status"] == RESOLVED]
    nearest = min(qualified, key=lambda r: abs(r["hours_from_incident_confirmation"]))
    sources = {r["workbook_year"]: r for r in read("cushing_nrc_incidents_sources")}
    source = sources[records[0]["workbook_year"]]
    with (ROOT/source["source_file"]).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source["source_sha256"]:
            raise ValueError("The original NRC workbook differs from its recorded checksum")
    summary = dict(phmsa_report_id=report["report_id"], phmsa_archive_record_number=report["source_record_number"],
                   nrc_report_ids=sorted(ids), nrc_source_path=source["source_file"], nrc_source_sha256=source["source_sha256"],
                   nrc_stock_record_rows={r["report_id"]: int(r["incidents_excel_row"]) for r in records},
                   equipment_id=case["equipment_id"], preferred_candidate_id=roof, phmsa_point_to_candidate_centre_m=match["centre_distance_m"],
                   earlier_capacity_eligible_contours=len(earlier), permit_capacity_bbl=float(permit["reported_capacity_bbl"]),
                   reported_stock_bbl=175000, reported_release_bbl=float(report["unintentional_release_bbl"]),
                   incident_confirmation_utc=confirmed_local.astimezone(ZoneInfo("UTC")).isoformat(), stock_measurement_time_known=False,
                   optical_fit_rows=len(rows), optical_acquisitions=len({r["optical_acquisition_id"] for r in rows}),
                   same_local_day_fit_rows=sum(r["same_local_day"] for r in rows),
                   same_local_day_passing_fit_rows=sum(r["same_local_day"] for r in qualified),
                   nearest_passing_fit_hours_from_incident_confirmation=nearest["hours_from_incident_confirmation"],
                   admitted_absolute_inventory_label=False)
    return summary, rows


def occ_controls(reports):
    records = []
    matches = {"18524OGDO10913": ("20240170", 2, "MIXER CAUGHT FIRE", "ENERGY TRANSFER"),
               "18515OGDO13137": ("20150464", 1, "SPILL AT TANK FARM", "1000 BBLS OIL")}
    fields = ("incident_no", "incident_type", "county", "latitude", "longitude", "initial_date", "investigation_date",
              "resolved_date", "oil_released", "oil_recovered", "incident_confirmation_status")
    with gzip.open(OCC, "rt", encoding="utf-8-sig", newline="") as stream:
        for i, row in enumerate(csv.DictReader(stream), 2):
            try:
                lon, lat = float(row["longitude"]), float(row["latitude"])
            except (TypeError, ValueError):
                continue
            if not (WEST <= lon <= EAST and SOUTH <= lat <= NORTH):
                continue
            ident = row["incident_no"]
            related, separation = None, None
            if ident in matches:
                related, expected_day_lag, complaint, findings = matches[ident]
                if complaint not in row["incident_complaint"].upper() or findings not in row["findings"].upper():
                    raise ValueError(f"OCC report no longer matches the reviewed case {ident}")
                r = reports[related]
                state_date = datetime.strptime(row["initial_date"], "%m/%d/%Y %H:%M:%S").date()
                federal_date = datetime.strptime(r["reported_event_local"], "%m/%d/%Y %H:%M").date()
                if (state_date-federal_date).days != expected_day_lag:
                    raise ValueError(f"OCC and PHMSA dates no longer reconcile for {ident}")
                separation = WGS84.inv(lon, lat, float(r["longitude"]), float(r["latitude"]))[2]
            records.append(dict(source_record_number=i, **{k: row[k] for k in fields},
                                possible_phmsa_report_id=related, phmsa_to_occ_report_coordinate_separation_m=separation,
                                source_url=OCC_URL, admitted_absolute_inventory_label=False))
    if {r["incident_no"] for r in records if r["possible_phmsa_report_id"]} != matches.keys():
        raise ValueError("Reviewed state incident absent from the regional archive")
    with gzip.open(OCC, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = dict(archive_url=OCC_URL, archive_path=str(OCC.relative_to(ROOT)), original_csv_sha256=digest,
                   statewide_records_scanned=i-1, regional_records=len(records),
                   regional_records_by_type=Counter(r["incident_type"] for r in records),
                   possible_phmsa_links=sum(bool(r["possible_phmsa_report_id"]) for r in records))
    return summary, records


def main():
    archive = STORE/f"{ARCHIVE}.zip"
    scanned, rows = source_rows(archive)
    reports = {row["report_id"]: row for row in rows}
    nrc = {row["report_id"] for row in read("cushing_nrc_incidents_reports")}
    for row in rows:
        linked = sorted(set(re.findall(r"\b\d{6,8}\b", row["nrc"]+" "+row["additional_nrc"])) & nrc)
        row["local_nrc_report_ids_json"] = json.dumps(linked)
    cases, spatial = selected_cases(reports)
    links = list(ground_links(cases, spatial))
    documents = list(pdf_sources(reports))
    quantities = list(reported_quantities(documents))
    rose_case, = [r for r in cases if r["report_id"] == "20240170"]
    rose, optical = rose_rock_inventory(reports[rose_case["report_id"]], rose_case, spatial)
    occ, regional = occ_controls(reports)
    for stem, data in (("reports", rows), ("cases", cases), ("contours", spatial), ("ground_links", links),
                       ("documents", documents), ("reported_quantities", quantities), ("rose_rock_optical", optical),
                       ("occ_regional", regional)):
        export(f"{STEM}_{stem}", data)
    counts = Counter(r["report_id"] for r in spatial if r["incident_point_inside_outline"] and r["equipment_in_documented_group"])
    result = dict(archive_url=ARCHIVE_URL, archive_path=str(archive.relative_to(ROOT)),
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), national_records_scanned=scanned,
                  local_records=len(rows), by_commodity=Counter(row["commodity"].strip('"') for row in rows),
                  local_reports_with_cross_linked_nrc=sum(row["local_nrc_report_ids_json"] != "[]" for row in rows),
                  reported_safe_fill_limits=sum(bool(row["safe_fill_limit_ft"]) for row in rows), reviewed_tank_cases=len(cases),
                  cases_with_one_incident_point_inside_a_number_group_contour=sum(n == 1 for n in counts.values()),
                  matched_ground_records=len(links), ground_records_with_one_preferred_contour=sum(bool(r["preferred_candidate_from_incident_point"]) for r in links),
                  original_documents=len(documents), rose_rock_inventory_case=rose, occ_archive=occ, admitted_absolute_inventory_labels=0)
    payload = json.dumps(result, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
