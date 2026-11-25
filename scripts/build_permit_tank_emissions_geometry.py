"""Extract geometry used in a dated permit application's emissions calculations."""
import re

import pdfplumber

from build_tank_geometry_registry import FEET_TO_METRES, MAGELLAN_SOUTH_COHORT
from collect_public import RAW, ROOT

SOURCE = RAW / "documents" / "source_pdfs" / "deq_magellan_south_6418026.pdf"
SCAN = RAW / "documents" / "source_pdfs" / "deq_magellan_south_6423620.pdf"
SOURCE_URL = "https://applications.deq.ok.gov/PermitsPublicReview/permitinformation.aspx?id=1223161"
SCHEMA = f"""
CREATE TABLE permit_tank_emissions_geometry (
    cohort_id TEXT,tank_id TEXT,application_received_date TEXT,calculation_kind TEXT,modeled_period TEXT,
    diameter_ft REAL,source_pdf_page INTEGER,source_file TEXT,scanned_copy_pdf_page INTEGER,scanned_copy_file TEXT,
    source_url TEXT,PRIMARY KEY (cohort_id,tank_id,application_received_date,calculation_kind));
CREATE VIEW tank_emissions_geometry_comparison AS
    WITH dimensions AS (
        SELECT e.candidate_id,e.cohort_id,d.application_received_date,
            json_array_length(e.possible_equipment_ids_json) AS possible_equipment_ids,
            count(DISTINCT d.tank_id) AS matched_equipment_ids,
            count(*) AS source_rows,count(DISTINCT d.diameter_ft) AS distinct_diameters,
            min(d.diameter_ft) AS diameter_ft_min,max(d.diameter_ft) AS diameter_ft_max,d.source_url,d.source_file
        FROM tank_engineering_cohort e JOIN json_each(e.possible_equipment_ids_json) p
        JOIN permit_tank_emissions_geometry d ON d.cohort_id=e.cohort_id AND d.tank_id=p.value
        GROUP BY e.candidate_id,e.cohort_id,d.application_received_date,d.source_url,d.source_file
    ), geometry AS (
        SELECT d.*,t.lidar_upper_outline_diameter_m,t.lidar_fit_status,t.footprint_diameter_m,
            CASE WHEN d.distinct_diameters=1 AND d.matched_equipment_ids=d.possible_equipment_ids
                THEN d.diameter_ft_min*{FEET_TO_METRES} END AS emissions_diameter_m
        FROM dimensions d JOIN tank_engineering_registry t USING (candidate_id,cohort_id)
    )
    SELECT *,
        round(lidar_upper_outline_diameter_m-emissions_diameter_m,6) AS lidar_minus_emissions_diameter_m,
        round(100*((lidar_upper_outline_diameter_m*lidar_upper_outline_diameter_m)
            /(emissions_diameter_m*emissions_diameter_m)-1),6) AS lidar_circle_area_above_emissions_circle_pct
    FROM geometry;
"""


def decode_cid(match, page):
    value = int(match[1])
    if page == 30 and 3 <= value <= 97:
        return chr(value + 29)
    if page == 32 and 1004 <= value <= 1013:
        return str(value - 1004)
    if page == 32:
        return {3: " ", 856: "."}.get(value, match[0])
    return match[0]


def rows():
    expected = {str(number) for number in range(2500, 2517)}
    with pdfplumber.open(SOURCE) as pdf:
        for page, scan_page, kind, period in ((30, 29, "normal_operation", "2020"),
                                             (32, 31, "roof_landing", "2020-05")):
            text = pdf.pages[page - 1].extract_text() or ""
            decoded = re.sub(r"\(cid:(\d+)\)", lambda match: decode_cid(match, page), text)
            values = re.findall(r"^(25\d\d)\s+(\d+(?:\.\d+)?)\s+", decoded, re.MULTILINE)
            if len(values) != len(expected) or {tank for tank, _ in values} != expected:
                raise ValueError(f"Unable to reconcile Magellan Cushing South equipment IDs on PDF page {page}")
            for tank, diameter in values:
                yield (MAGELLAN_SOUTH_COHORT, tank, "2026-09-22", kind, period, float(diameter), page,
                       str(SOURCE.relative_to(ROOT)), scan_page if SCAN.is_file() else None,
                       str(SCAN.relative_to(ROOT)) if SCAN.is_file() else None, SOURCE_URL)


def add_permit_tank_emissions_geometry(db):
    if SOURCE.is_file():
        db.executemany("INSERT INTO permit_tank_emissions_geometry VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows())
