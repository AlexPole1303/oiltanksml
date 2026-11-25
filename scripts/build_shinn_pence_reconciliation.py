"""Reconcile Shinn Pence equipment, county parcels and dated physical contours."""
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone

import pdfplumber
from pyproj import Transformer
from shapely import make_valid, transform
from shapely.geometry import Point, shape

from collect_public import RAW, ROOT, write, write_csv

DERIVED = ROOT / "data/derived"
DOCUMENTS = RAW / "documents"
PARCEL = "17N05E-12-2-00000-000-0251"
LANDOWNER = "Magellan Pipeline Company, LP"
FRS_ID = "110007733943"
PERMIT = DOCUMENTS / "source_pdfs/8447.pdf"
PHMSA = DOCUMENTS / "source_pdfs/phmsa_magellan_2016_0072_analysis_findings.pdf"
SPECIAL_PERMIT = DOCUMENTS / "phmsa_magellan_2016_0072.pdf"
PHMSA_ROWS = ROOT / "data/reference/shinn_pence_phmsa_capacity.csv"
DEQ_URL = "https://applications.deq.ok.gov/permitspublic/storedpermits/8447.pdf"
EPA_TIMELINE_URL = "https://www.epa.gov/caa-permitting/operating-permit-timeline-oklahoma"
PHMSA_URL = ("https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/2020-08/"
             "2016-0072-Magellan-Midstream-Partner-Special-Permit-Analysis-and-Findings.pdf")
SPECIAL_PERMIT_URL = ("https://www.phmsa.dot.gov/sites/phmsa.dot.gov/files/docs/subdoc/3951/"
                      "special-permit-phmsa-2016-0072-magellan.pdf")
ACQUISITION_URL = "https://www.sec.gov/Archives/edgar/data/1126975/000119312510175294/dex101.htm"
ACQUISITION_CLOSE_URL = "https://www.sec.gov/Archives/edgar/data/1126975/000112697511000032/mmp12311010k.htm"
MERGER_URL = "https://www.sec.gov/Archives/edgar/data/1126975/000121390023079220/ea185726-8k_magellan.htm"
OLDER = "historical_117_ft_group"
NEWER = "2019_group"
MEMO_ROW = re.compile(r"\s*(\d+)\s+(\d+)\s+(IFR)\s+([\d,]+)\s+[\d,]+\s+(\d{4})\s*")
CONDITIONS_ROW = re.compile(r"\s*(\d+)\s+(\d+)\s+(IFR)\s+([\d,]+)\s+\S+\s*")
DECOMMISSIONED = re.compile(r"\s*(\d+)\s+Decommissioned\s*", re.IGNORECASE)


def roster(pdf, pages, pattern):
    units = {}
    for page in pages:
        for line in pdf.pages[page - 1].extract_text(layout=True).splitlines():
            active = pattern.fullmatch(line)
            decommissioned = DECOMMISSIONED.fullmatch(line)
            if active:
                unit, number, roof, gallons, *year = active.groups()
                data = (int(number), roof, int(gallons.replace(",", "")), int(year[0]) if year else None, page)
            elif decommissioned:
                unit = decommissioned[1]
                data = (None, None, None, None, page)
            else:
                continue
            if int(unit) in units:
                raise ValueError(f"Repeated emission unit {unit} on page {page}")
            units[int(unit)] = data
    return units


def equipment():
    with pdfplumber.open(PERMIT) as pdf:
        cover = pdf.pages[0].extract_text()
        for expected in ("DRAFT/PROPOSED", "April 26, 2023", "2020-0212-TVR", "Shinn Pence Tank Farm", "Facility ID: 29"):
            if expected not in cover:
                raise ValueError(f"Missing Shinn Pence permit identification {expected}")
        dimensions = pdf.pages[5].extract_text(layout=True)
        if not re.search(r"Diameter\s+117 ft", dimensions) or not re.search(r"Height\s+42 ft", dimensions):
            raise ValueError("Missing reported dimensions of the older group")
        memo = roster(pdf, (3, 4), MEMO_ROW)
        conditions = roster(pdf, (18, 19, 20), CONDITIONS_ROW)
    if set(memo) != set(range(1, 44)) | {45, 46, 47, 48}:
        raise ValueError("Unexpected draft permit equipment positions")
    if {unit: row[:3] for unit, row in memo.items()} != {unit: row[:3] for unit, row in conditions.items()}:
        raise ValueError("DEQ memorandum and draft specific conditions differ")
    if {unit for unit, row in memo.items() if row[0] is None} != {5, 11, 31, 38}:
        raise ValueError("Unexpected set of decommissioned positions")
    with PHMSA_ROWS.open(newline="", encoding="utf-8") as source:
        references = list(csv.DictReader(source))
    phmsa = {int(row["tank_number"]): row for row in references}
    older_ids = {data[0] for unit, data in memo.items() if unit < 44 and data[0] is not None}
    if len(references) != 21 or len(phmsa) != 21 or not phmsa.keys() <= older_ids:
        raise ValueError("PHMSA equipment identities do not reconcile with the older DEQ group")

    rows = []
    for unit, (number, roof, gallons, year, page) in sorted(memo.items()):
        old = unit < 44
        if number is not None and (gallons, year) != ((3360000, 1923) if old else (10500000, 2019)):
            raise ValueError(f"Unexpected capacity or construction year for emission unit {unit}")
        reference = phmsa.get(number, {})
        rows.append(dict(emissions_unit_id=unit, tank_number=number, equipment_group=OLDER if old else NEWER,
            status_in_draft="listed_active" if number is not None else "decommissioned", roof_type=roof,
            deq_stated_capacity_us_gal=gallons, deq_stated_capacity_bbl=gallons // 42 if gallons else None,
            deq_reported_construction_year=year, deq_group_diameter_ft=117 if old and number else None,
            deq_group_height_ft=42 if old and number else None, deq_group_dimension_pdf_page=6 if old and number else None,
            deq_memo_pdf_page=page, deq_draft_conditions_pdf_page=conditions[unit][-1],
            phmsa_reported_tank_capacity_bbl=int(reference["phmsa_reported_tank_capacity_bbl"]) if reference else None,
            phmsa_analysis_pdf_page=int(reference["analysis_pdf_page"]) if reference else None,
            phmsa_special_permit_pdf_page=int(reference["special_permit_pdf_page"]) if reference else None,
            deq_source_url=DEQ_URL, phmsa_analysis_source_url=PHMSA_URL if reference else None,
            phmsa_special_permit_source_url=SPECIAL_PERMIT_URL if reference else None))
    return rows


def features(path):
    return json.loads(path.read_bytes())["features"]


def parcels():
    project = Transformer.from_crs(4326, 32614, always_xy=True).transform

    def projected(geometry):
        return transform(make_valid(shape(geometry)), project, interleaved=False)

    boundary_inputs = [DERIVED / f"{name}_tank_candidates.geojson" for name in ("osm", "supplemental")]
    outlines = {f["properties"]["candidate_id"]: projected(f["geometry"])
                for path in boundary_inputs for f in features(path)}
    echo_path = DOCUMENTS / "epa_echo_cushing.geojson"
    echo, = (f for f in features(echo_path) if f["properties"]["REGISTRY_ID"] == FRS_ID)
    epa_point = projected(echo["geometry"])
    deq_point = Point(*project(-96.73094, 35.95694))
    overlap = defaultdict(dict)
    snapshots = []
    for year in range(2021, 2026):
        path = DOCUMENTS / f"payne_parcels/{year}.geojson"
        snapshot = json.loads(path.read_bytes())
        parcel, = (f for f in snapshot["features"] if f["properties"]["ParcelID"] == PARCEL)
        props, area = parcel["properties"], projected(parcel["geometry"])
        if props["OwnerNameFormatted"] != LANDOWNER or not area.covers(deq_point):
            raise ValueError(f"Parcel identity disagrees with the DEQ location in {year}")
        for candidate, polygon in outlines.items():
            if polygon.intersects(area):
                fraction = polygon.intersection(area).area / polygon.area
                if fraction > 0.0001:
                    overlap[candidate][year] = fraction
        snapshots.append(dict(snapshot_year=year, assessor_object_id=props["OBJECTID"],
            recorded_owner=props["OwnerNameFormatted"], deed_reference=props["LegalReference"],
            recorded_sale_date=datetime.fromtimestamp(props["SaleDate"] / 1000, timezone.utc).date().isoformat(),
            epa_reference_point_distance_to_parcel_m=round(area.distance(epa_point), 3),
            source_file=str(path.relative_to(ROOT)), source_url=snapshot["source_layer"]))
    years = {s["snapshot_year"] for s in snapshots}
    if len(overlap) != 49 or any(f.keys() != years or min(f.values()) < 0.99 for f in overlap.values()):
        raise ValueError("Shinn Pence parcel matches disagree across annual cadastral snapshots")
    return overlap, snapshots, echo["properties"]


def contours(db, overlaps, units):
    candidates = sorted(overlaps)
    markers = ",".join("?" for _ in candidates)
    cursor = db.cursor()
    cursor.row_factory = sqlite3.Row
    records = cursor.execute(f"""SELECT candidate_id,first_seen,last_absence,latest_aerial_date,
        latest_aerial_top_observation,footprint_diameter_m,lidar_upper_outline_diameter_m,lidar_fit_status,
        first_upper_return_at_utc FROM tank_engineering_registry
        WHERE candidate_id IN ({markers}) ORDER BY candidate_id""", candidates).fetchall()
    if len(records) != len(candidates):
        raise ValueError("Missing engineering data for a Shinn Pence contour")
    counts = Counter()
    output = []
    group_numbers = [r["tank_number"] for r in units if r["equipment_group"] == NEWER]
    for row in records:
        candidate = row["candidate_id"]
        year_seen = row["first_seen"][:4]
        lidar = row["lidar_upper_outline_diameter_m"]
        if row["lidar_fit_status"] != "well_supported_circular_fit" or lidar is None:
            raise ValueError(f"Unverified LiDAR outline for {candidate}")
        if row["latest_aerial_top_observation"] not in {"top_without_distinct_inset_shadow", "no_resolved_inset_shadow"}:
            raise ValueError(f"Unexpected visual roof evidence for {candidate}")
        if 50 < lidar < 65 and year_seen == "2019" and row["last_absence"].startswith("2017-"):
            group = NEWER
        elif abs(lidar - 117 * 0.3048) < 1.5 and year_seen == "2010":
            group = OLDER
        else:
            raise ValueError(f"Contour does not match either size and construction cohort: {candidate}")
        counts[group] += 1
        fractions = {f"parcel_fraction_{year}": round(fraction, 6) for year, fraction in overlaps[candidate].items()}
        is_new = group == NEWER
        output.append(dict(row, site_id="magellan_shinn_pence", assessor_parcel_id=PARCEL, **fractions,
            equipment_group=group, match_status="documented_equipment_group" if is_new else "active_unit_unresolved",
            documented_group_roof_type="IFR" if is_new else None,
            group_capacity_per_tank_bbl=250000 if is_new else None,
            possible_tank_numbers_json=json.dumps(group_numbers) if is_new else None))
    if counts != {OLDER: 45, NEWER: 4}:
        raise ValueError(f"Unexpected Shinn Pence physical groups: {counts}")
    return output


def group_summary(group, units, outlines):
    equipment_rows = [r for r in units if r["equipment_group"] == group]
    active = [r for r in equipment_rows if r["status_in_draft"] == "listed_active"]
    controls = [r for r in outlines if r["equipment_group"] == group]
    diameters = [r["lidar_upper_outline_diameter_m"] for r in controls]
    return dict(contours=len(controls), listed_active=len(active), decommissioned=len(equipment_rows) - len(active),
        contours_minus_listed_active=len(controls) - len(active), lidar_diameter_range_m=[min(diameters), max(diameters)],
        listed_capacity_bbl=sum(r["deq_stated_capacity_bbl"] for r in active),
        group_matches=sum(r["match_status"] == "documented_equipment_group" for r in controls))


def main():
    units = equipment()
    overlap, snapshots, epa = parcels()
    database = DERIVED / "cushing.sqlite"
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as db:
        outlines = contours(db, overlap, units)
    phmsa_capacity = [r["phmsa_reported_tank_capacity_bbl"] for r in units if r["phmsa_reported_tank_capacity_bbl"]]
    summary = dict(site_id="magellan_shinn_pence", deq_facility_id=29, deq_permit="2020-0212-TVR",
        permit_document_status="draft_proposed", permit_memo_date="2023-04-26", first_public_availability=None,
        permit_applicant="Magellan Terminals Holdings, L.P.",
        epa_review_timeline=[dict(listed_review_end="2023-07-02", status="review_temporarily_suspended"),
                             dict(listed_review_end="2023-10-23", petition_deadline="2023-12-22")],
        epa_timeline_source_url=EPA_TIMELINE_URL,
        parcel_id=PARCEL, deq_reference_point_inside_all_parcels=True, parcel_snapshots=snapshots,
        epa_frs_id=FRS_ID, epa_location_accuracy_m=epa["FAC_ACCURACY_METERS"], epa_source_url=epa["DFR_URL"],
        epa_reference_point=epa["FAC_REFERENCE_POINT"],
        epa_location_source_file="data/raw/documents/epa_echo_cushing.geojson",
        groups={group: group_summary(group, units, outlines) for group in (OLDER, NEWER)},
        phmsa_overlap_count=len(phmsa_capacity), phmsa_reported_capacity_range_bbl=[min(phmsa_capacity), max(phmsa_capacity)],
        phmsa_total_reported_capacity_bbl=sum(phmsa_capacity),
        ownership_evidence=[dict(event="asset_purchase_agreement", date="2010-07-12", seller="BP Pipelines (North America) Inc.",
            buyer="Magellan Pipeline Company, L.P.", source_url=ACQUISITION_URL),
            dict(event="completed_cushing_asset_purchase", month="2010-09", parent="Magellan Midstream Partners, L.P.",
                 ticker="MMP", source_url=ACQUISITION_CLOSE_URL),
            dict(event="magellan_parent_merger_closed", date="2023-09-25", parent="ONEOK, Inc.", ticker="OKE",
                 source_url=MERGER_URL)],
        source_pdfs={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in (PERMIT, PHMSA, SPECIAL_PERMIT)},
        interpretation="The four 2019 contours are matched to a group. Individual tank numbers remain unassigned. "
            "The 45 older contours cannot be reconciled individually to 39 active and four decommissioned DEQ positions. "
            "DEQ capacities and PHMSA reported tank capacities retain separate definitions. This is a retrospective "
            "engineering and attribution reconciliation outside the existing optical and market samples.")
    for stem, rows in (("permit_equipment", units), ("contour_reconciliation", outlines)):
        fields = tuple(rows[0])
        write_csv(DERIVED / f"shinn_pence_{stem}.csv", fields, ([r[f] for f in fields] for r in rows))
    write(DERIVED / "shinn_pence_reconciliation_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary["groups"], indent=2))


if __name__ == "__main__":
    main()
