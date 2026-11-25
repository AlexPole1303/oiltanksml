"""Reconcile terminal scope, dated imagery, independent geometry, and capacity evidence."""
import csv
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing

import pdfplumber
from pyproj import Geod, Transformer
from shapely import transform
from shapely.geometry import Point, shape

from build_lidar_roof_reference import WEEKLY_ROOF_COVERAGE_VIEW
from build_nei_roof_evidence import install_nei_roof_evidence
from build_tank_geometry_registry import (TERMINAL_DATA_COVERAGE_VIEW, cushing_connect_rows,
                                          enterprise_and_sunoco_rows, phillips_nei_rows)
from collect_public import RAW, ROOT, write, write_csv
from terminal_aerial_roof_evidence import TABLE as EARLY_ROOF_TABLE, install_early_roof_review

DERIVED = ROOT/"data/derived"
REFERENCE = ROOT/"data/reference"
DB = DERIVED/"cushing.sqlite"
PLAN = RAW/"documents/ok_deq_tallgrass_cushing_south_2017_proposed.pdf"
PLAN_URL = "https://applications.deq.ok.gov/permitspublic/storedpermits/4290.pdf"
EAST_SOURCE_POINTS = ((-96.748064,35.93964),(-96.749159,35.9401912))
TALLGRASS_DRAFT_POINT = (-96.75811,35.92031)
LITTLE_FARM_PARCEL = "18N05E-35-3-CZ999-000-0517"

SCOPE_TABLE = """
CREATE TABLE IF NOT EXISTS terminal_parcel_scope (
    scope_id TEXT,terminal_id TEXT,county TEXT,snapshot_kind TEXT,snapshot_year INTEGER,parcel_id TEXT,
    recorded_owner TEXT,evidence_kind TEXT,source_url TEXT,source_file TEXT,source_page INTEGER,notes TEXT,
    PRIMARY KEY(scope_id,county,snapshot_year,parcel_id));
"""
SCOPE_VIEW = """
CREATE VIEW tank_terminal_scope_parcel_evidence AS
SELECT p.candidate_id,s.scope_id,s.terminal_id,p.county,p.snapshot_kind,p.snapshot_year,
    group_concat(p.parcel_id,'|') AS parcel_ids,p.owner_name_recorded,sum(p.tank_area_fraction) AS combined_area_fraction,
    min(p.source_url) AS parcel_source_url,s.evidence_kind,s.source_url AS scope_source_url,s.source_file AS scope_source_file,
    s.source_page AS scope_source_page,s.notes
FROM terminal_parcel_scope s JOIN tank_parcel_intersection p
 ON p.parcel_id=s.parcel_id AND p.county=s.county AND p.snapshot_kind=s.snapshot_kind AND p.snapshot_year=s.snapshot_year
 AND upper(trim(p.owner_name_recorded))=upper(trim(s.recorded_owner))
WHERE (s.scope_id<>'hcs_ground_lease' OR EXISTS (SELECT 1 FROM tank_asset_cohort c
    WHERE c.candidate_id=p.candidate_id AND c.terminal_id=s.terminal_id AND c.cohort_id='amid_five_2009'))
  AND (s.scope_id<>'transmontaigne_ground_lease' OR EXISTS (SELECT 1 FROM tank_engineering_cohort c
    WHERE c.candidate_id=p.candidate_id AND c.terminal_id='enbridge_blueknight' AND c.cohort_id='grandstaff_central_2012'))
GROUP BY p.candidate_id,s.scope_id,s.terminal_id,p.county,p.snapshot_kind,p.snapshot_year,p.owner_name_recorded,
    s.evidence_kind,s.source_url,s.source_file,s.source_page,s.notes
HAVING sum(p.tank_area_fraction)>=0.99;
"""
ATTRIBUTION = """
CREATE VIEW tank_terminal_research_attribution AS
    SELECT candidate_id,terminal_id,'documented_asset_cohort' AS basis FROM tank_asset_cohort
    UNION
    SELECT candidate_id,terminal_id,'documented_parcel_scope' AS basis FROM tank_terminal_scope_parcel_evidence p
    WHERE NOT EXISTS (SELECT 1 FROM tank_asset_cohort c WHERE c.candidate_id=p.candidate_id)
    UNION
    SELECT candidate_id,terminal_id,'recorded_parcel_association' AS basis FROM tank_terminal_parcel_evidence p
    WHERE NOT EXISTS (SELECT 1 FROM tank_asset_cohort c WHERE c.candidate_id=p.candidate_id)
      AND NOT EXISTS (SELECT 1 FROM tank_terminal_scope_parcel_evidence s
          WHERE s.candidate_id=p.candidate_id AND (s.terminal_id=p.terminal_id OR
              (s.scope_id='transmontaigne_ground_lease' AND p.terminal_id='enbridge_blueknight')));
"""


def csv_rows(path):
    with path.open(newline="",encoding="utf-8") as source:
        rows = csv.DictReader(source)
        return [tuple(v or None for v in row.values()) for row in rows]


def supplemental_shapes():
    items = json.loads((DERIVED / "supplemental_tank_candidates.geojson").read_bytes())["features"]
    return {row["properties"]["candidate_id"]: row["geometry"] for row in items}


def install_little_farm_parcels(db):
    sites = json.loads((RAW / "documents/epa_echo_cushing.geojson").read_bytes())["features"]
    site, = (row for row in sites if row["properties"]["REGISTRY_ID"] == "110022404423")
    point = shape(site["geometry"])
    project = Transformer.from_crs(4326, 32614, always_xy=True).transform
    tanks = {candidate: transform(shape(geometry), project, interleaved=False)
             for candidate, geometry in supplemental_shapes().items() if candidate.startswith("ortho-little-farm-")}
    if len(tanks) != 5:
        raise ValueError("The Little Farm aerial group must contain five contours")
    rows = []
    for year in range(2021, 2026):
        path = RAW / f"documents/payne_parcels_little_farm/{year}.geojson"
        snapshot = json.loads(path.read_bytes())
        parcel, = (r for r in snapshot["features"] if r["properties"]["ParcelID"] == LITTLE_FARM_PARCEL)
        props, geometry = parcel["properties"], shape(parcel["geometry"])
        if not geometry.covers(point) or props["OwnerNameFormatted"] != "Sun Pipe Line Co":
            raise ValueError(f"EPA location or recorded owner changed for Little Farm, {year}")
        projected = transform(geometry, project, interleaved=False)
        for candidate, tank in sorted(tanks.items()):
            fraction = projected.intersection(tank).area / tank.area
            if fraction < 0.99:
                raise ValueError(f"Little Farm contour outside the parcel in {year}: {candidate}")
            rows.append((candidate, year, props["OBJECTID"], LITTLE_FARM_PARCEL, props["PropertyID"], props["YearID"],
                         props["OwnerNameFormatted"], props["BusinessName"], props["LandUseCode"], props["SaleDate"],
                         props["LegalReference"], round(fraction, 6), snapshot["source_layer"], "payne", "annual_assessment"))
    db.execute("DELETE FROM tank_parcel_intersection WHERE candidate_id LIKE 'ortho-little-farm-%' AND county='payne'")
    db.executemany("INSERT INTO tank_parcel_intersection VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def install_terminal_scopes(db):
    install_nei_roof_evidence(db)
    install_little_farm_parcels(db)
    db.execute(SCOPE_TABLE)
    db.execute("DELETE FROM terminal_parcel_scope")
    db.executemany("INSERT INTO terminal_parcel_scope VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   csv_rows(REFERENCE/"terminal_parcel_scopes.csv"))
    db.execute("DROP VIEW IF EXISTS tank_terminal_research_attribution")
    db.execute("DROP VIEW IF EXISTS tank_terminal_scope_parcel_evidence")
    db.execute(SCOPE_VIEW)
    db.execute(ATTRIBUTION)


def verify_east_parcel(db):
    county = RAW/"documents/lincoln_parcels/2026.geojson"
    prefix = "Lincoln_Oklahoma.DBO."
    with county.open() as source:
        parcels = json.load(source)["features"]
    parcel, = (p for p in parcels if p["properties"][prefix+"Parcels.Parcel_ID"]=="0000-23-17N-05E-2-001-00")
    legal = parcel["properties"][prefix+"GIS_Out.PARCELLGL"]
    geometry = shape(parcel["geometry"])
    if "TEPPCO" not in legal or not all(geometry.covers(Point(*xy)) for xy in EAST_SOURCE_POINTS):
        raise ValueError("The source locations and Lincoln legal description do not confirm the East parcel")
    return dict(checked_east_source_points=len(EAST_SOURCE_POINTS),parcel_legal_description=legal,
                parcel_source_file=str(county.relative_to(ROOT)),
                incident_position_sources=[db.execute("SELECT source_url FROM terminal_parcel_scope WHERE scope_id='enterprise_east'").fetchone()[0],
                                            "https://response.epa.gov/site/polrep_printer.aspx?counter=27892&format=pdf"])


def verify_cvr_and_phillips_parcels(db):
    documents = RAW/"documents"
    cvr_source = documents/"epa_dfr_110043685615.json"
    incidents_source = documents/"phmsa_cushing_phillips_2010_2014_incidents.geojson"
    cvr = json.loads(cvr_source.read_text())["Results"]["Permits"]
    facility, = (r for r in cvr if r["SourceID"]=="OK0000004008100059")
    incidents = {str(r["properties"]["Rpt_Num"]): r["properties"]
                 for r in json.loads(incidents_source.read_text())["features"]}
    scopes = (
        ("cushing_connect_terminal", (facility,), cvr_source, 6),
        ("phillips_buxton", (incidents["20140125"],), incidents_source, 2),
        ("phillips_named_cushing_terminal", (incidents["20140325"], incidents["20100026"]), incidents_source, 2),
    )
    audit = {}
    for scope, records, source, expected_count in scopes:
        county, year, parcel_id = db.execute(
            "SELECT county,snapshot_year,parcel_id FROM terminal_parcel_scope WHERE scope_id=?", (scope,)).fetchone()
        parcel_source = documents/f"{county}_parcels/{year}.geojson"
        key = "ParcelID" if county=="payne" else "Lincoln_Oklahoma.DBO.Parcels.Parcel_ID"
        parcel, = (r for r in json.loads(parcel_source.read_text())["features"] if r["properties"][key]==parcel_id)
        points = [Point(float(record["Longitude"]), float(record["Latitude"])) for record in records]
        if not all(shape(parcel["geometry"]).covers(point) for point in points):
            raise ValueError(f"Regulatory locations do not intersect the {scope} parcel")
        contours = list(db.execute("""SELECT s.candidate_id,t.geometry_geojson
            FROM tank_terminal_scope_parcel_evidence s JOIN tank_candidate t USING(candidate_id)
            WHERE s.scope_id=? ORDER BY s.candidate_id""", (scope,)))
        if len(contours)!=expected_count:
            raise ValueError(f"The {scope} parcel does not contain the expected number of large contours")
        audit[scope] = dict(parcel_id=parcel_id, parcel_source_file=str(parcel_source.relative_to(ROOT)),
                            point_source_file=str(source.relative_to(ROOT)),
                            source_ids=[str(record.get("Rpt_Num", record.get("SourceID"))) for record in records],
                            matched_contours=[candidate for candidate, _ in contours])
        if scope=="phillips_named_cushing_terminal":
            tank, = (candidate for candidate, geo in contours if shape(json.loads(geo)).covers(points[-1]))
            audit[scope]["equipment_location_match"] = dict(equipment_id="824", candidate_id=tank,
                incident_report="20100026", source_pdf_page=7, identification_basis="incident_point_inside_single_contour")
    return audit


def planned_equipment():
    with pdfplumber.open(PLAN) as document:
        intro,page = (document.pages[i].extract_text() for i in (0,1))
    pattern = r"^(T-\d+) Crude Oil (EFR|IFR) Liquid Heel ([\d,]+) ([\d.]+) ([\d.]+) (TBD)$"
    rows = [dict(facility_scope_id="tallgrass_cushing_south_draft",facility_id=17009,equipment_id=tank,roof_type=roof,
        draft_capacity_bbl=int(capacity.replace(",","")),draft_height_ft=float(height),draft_diameter_ft=float(diameter),
        construction_date_as_printed=status,document_date="2017-09-20",introduction_application_number="2017-0121-C (PSD)",
        roster_page_header_number="2017-0212-C (PSD)",source_pdf_page=2,source_url=PLAN_URL,
        source_file=str(PLAN.relative_to(ROOT))) for tank,roof,capacity,height,diameter,status in re.findall(pattern,page,re.M)]
    counts = Counter((r["roof_type"],r["draft_capacity_bbl"]) for r in rows)
    if counts!={('EFR',500000):6,('EFR',250000):10,('IFR',1000):4} or len({r["equipment_id"] for r in rows})!=20:
        raise ValueError("The proposed Tallgrass equipment roster changed")
    if "2017-0121-C (PSD)" not in intro or "2017-0212-C (PSD)" not in page or "DRAFT/PROPOSED" not in page:
        raise ValueError("The proposed Tallgrass document identity changed")
    return rows


def scope_contours(db):
    sql = """
    WITH visual AS (
      SELECT candidate_id,count(DISTINCT CASE WHEN structure_observation='aboveground_tank_seen'
                AND top_observation IN ('inset_wall_shadow_visible','clear_inset_wall_shadow') THEN observed_date END) AS clear_inset_shadow_dates,
          min(CASE WHEN structure_observation='aboveground_tank_seen' AND top_observation IN ('inset_wall_shadow_visible','clear_inset_wall_shadow')
                THEN observed_date END) AS first_clear_inset_shadow_date,
          max(CASE WHEN structure_observation='aboveground_tank_seen' AND top_observation IN ('inset_wall_shadow_visible','clear_inset_wall_shadow')
                THEN observed_date END) AS last_clear_inset_shadow_date
      FROM tank_aerial_visual_observation GROUP BY candidate_id
    ), legacy AS (
      SELECT candidate_id,max(rim_z_p95-ground_z_p50) AS max_2011_rim_to_ground_m,
          sum(interior_points) AS interior_2011_returns
      FROM lidar_tank_observation WHERE dataset LIKE '%2011%' GROUP BY candidate_id
    )
    SELECT s.*,t.longitude,t.latitude,t.first_seen AS first_visible_aerial_date,t.last_absence AS latest_dated_absence,
      t.latest_aerial_date,t.latest_aerial_top_observation,v.clear_inset_shadow_dates,v.first_clear_inset_shadow_date,
      v.last_clear_inset_shadow_date,t.footprint_diameter_m,t.lidar_upper_outline_diameter_m,t.lidar_fit_status,
      t.lidar_rim_to_ground_native_z AS lidar_rim_above_ground_m,t.lidar_circular_area_bbl_per_m,
      l.max_2011_rim_to_ground_m,l.interior_2011_returns,t.documented_terminal_id AS engineering_registry_terminal_id,
      t.cohort_id AS engineering_cohort_id,t.possible_equipment_ids_json,t.documented_roof_type,t.reported_capacity_min_bbl,t.reported_capacity_max_bbl,
      t.reported_capacity_basis,t.reported_height_basis,t.reported_height_min_m,t.reported_height_max_m,t.permit_diameter_m,
      t.engineering_source_url,t.engineering_source_file,
      CASE WHEN s.scope_id='hcs_ground_lease' THEN 'five_larger_efr_same_operated_site_different_lessee_label'
           WHEN v.clear_inset_shadow_dates>0 THEN 'externally_visible_inset_shadow'
           ELSE 'roof_geometry_unresolved' END AS roof_observation_basis
    FROM tank_terminal_scope_parcel_evidence s JOIN tank_engineering_registry t USING(candidate_id)
    LEFT JOIN visual v USING(candidate_id) LEFT JOIN legacy l USING(candidate_id)
    ORDER BY s.scope_id,s.candidate_id
    """
    return [dict(row) for row in db.execute(sql)]


def rollup(contours):
    groups = defaultdict(list)
    for row in contours:
        groups[row["scope_id"]].append(row)
    for scope_id,group in sorted(groups.items()):
        good = [r for r in group if r["lidar_fit_status"]=="well_supported_circular_fit"]
        yield dict(scope_id=scope_id,terminal_id=group[0]["terminal_id"],mapped_contours=len(group),
            first_seen_aerial_min=min(r["first_visible_aerial_date"] for r in group if r["first_visible_aerial_date"]),
            first_seen_aerial_max=max(r["first_visible_aerial_date"] for r in group if r["first_visible_aerial_date"]),
            distinct_dated_inset_shadow_contours=sum((r["clear_inset_shadow_dates"] or 0)>0 for r in group),
            well_supported_lidar_diameters=len(good),min_supported_diameter_m=min(r["lidar_upper_outline_diameter_m"] for r in good),
            max_supported_diameter_m=max(r["lidar_upper_outline_diameter_m"] for r in good),
            documented_roof_contours=sum(r["documented_roof_type"] is not None for r in group),
            contours_with_reported_capacity=sum(r["reported_capacity_min_bbl"] is not None for r in group))


def export(stem,rows):
    rows = list(rows)
    columns = tuple(rows[0])
    write_csv(DERIVED/f"{stem}.csv",columns,([r[k] for k in columns] for r in rows))


def all_view_rows(db,name,order):
    return [dict(row) for row in db.execute(f"SELECT * FROM {name} ORDER BY {order}")]


def development_economic_coverage(contours):
    economic = {r["candidate_id"]:r["terminal_id"] for r in contours if r["scope_id"]=="transmontaigne_ground_lease"}
    rows = defaultdict(list)
    path = DERIVED/"weekly_shadow_development_pairs.csv"
    with path.open(newline="",encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["roof_type"]=="EFR":
                terminal = economic.get(row["candidate_id"],row["terminal_id"])
                rows[row["week_ending"],terminal].append(row)
    for (week,terminal),group in sorted(rows.items()):
        fitted = [r for r in group if r["optical_change_equivalent_bbl"]!=""]
        resolved = [r for r in fitted if r["both_shadow_diagnostics_pass"]=="True"]
        yield dict(week_ending=week,release_date=group[0]["release_date"],economic_terminal_id=terminal,
            input_efr_pairs=len(group),fitted_efr_pairs=len(fitted),resolved_efr_pairs=len(resolved),
            input_circle_area_m2=sum(float(r["circular_area_m2"]) for r in group),
            resolved_circle_area_m2=sum(float(r["circular_area_m2"]) for r in resolved),
            fitted_visible_change_equivalent_bbl=sum(float(r["optical_change_equivalent_bbl"]) for r in fitted) if fitted else None,
            resolved_visible_change_equivalent_bbl=sum(float(r["optical_change_equivalent_bbl"]) for r in resolved) if resolved else None,
            reassigned_from_permit_scope_tank_pairs=sum(r["terminal_id"]!=terminal for r in group))


def write_scope_outputs(db):
    db.row_factory = sqlite3.Row
    east = verify_east_parcel(db)
    cvr_and_phillips = verify_cvr_and_phillips_parcels(db)
    contours = scope_contours(db)
    summary = list(rollup(contours))
    plan = planned_equipment()
    if len({r["candidate_id"] for r in contours})!=len(contours):
        raise ValueError("A tank was assigned to more than one named physical scope")
    export("terminal_scope_contours",contours)
    export("terminal_scope_rollup",summary)
    export("tallgrass_cushing_south_planned_equipment",plan)
    economic = list(development_economic_coverage(contours))
    export("weekly_shadow_development_economic_terminals",economic)
    export("terminal_scope_aerial_history",[dict(r) for r in db.execute("""SELECT s.scope_id,s.terminal_id,a.*
        FROM tank_terminal_scope_parcel_evidence s JOIN tank_aerial_visual_observation a USING(candidate_id)
        ORDER BY s.scope_id,a.candidate_id,a.observed_date,a.image_source""")])
    shapes = {candidate: json.loads(geometry) for candidate, geometry in db.execute(
        "SELECT candidate_id,geometry_geojson FROM tank_candidate")}
    shapes.update(supplemental_shapes())
    features = [dict(type="Feature",geometry=shapes[r["candidate_id"]],properties=r) for r in contours]
    write(DERIVED/"terminal_scope_contours.geojson",json.dumps(dict(type="FeatureCollection",features=features),ensure_ascii=False).encode())
    geodesic = Geod(ellps="WGS84")
    near = min((geodesic.inv(*TALLGRASS_DRAFT_POINT,lon,lat)[-1],candidate) for candidate,lon,lat in db.execute(
        "SELECT candidate_id,longitude,latitude FROM tank_engineering_registry"))
    report = dict(contours=len(contours),scope_counts={r["scope_id"]:r["mapped_contours"] for r in summary},
        enterprise_parcel_audit=east,cvr_and_phillips_parcel_audit=cvr_and_phillips,
        planned_tallgrass_roster_units=len(plan),planned_tallgrass_roster_capacity_bbl=sum(r["draft_capacity_bbl"] for r in plan),
        nearest_existing_registry_contour_to_tallgrass_draft_point=dict(distance_m=near[0],candidate_id=near[1]),
        development_efr_pairs_reassigned_from_permit_scope=sum(r["reassigned_from_permit_scope_tank_pairs"] for r in economic),
        development_economic_terminal_weeks=len(economic),
        scope="Parcel and independent visual reconciliation. The Tallgrass draft has no assigned observed contours. Group capacities and cadastral title are carried with their distinct source definitions.")
    write(DERIVED/"terminal_scope_reconciliation_summary.json",json.dumps(report,indent=2,ensure_ascii=False).encode())
    return report


def main():
    with closing(sqlite3.connect(DB)) as db:
        db.execute("BEGIN IMMEDIATE")
        with db:
            install_terminal_scopes(db)
            install_early_roof_review(db)
            db.execute("DROP VIEW IF EXISTS terminal_data_coverage")
            db.execute(TERMINAL_DATA_COVERAGE_VIEW)
            db.execute("DROP VIEW IF EXISTS eia_roof_cohort_weekly_coverage")
            db.execute(WEEKLY_ROOF_COVERAGE_VIEW)
            verify_east_parcel(db)
            verify_cvr_and_phillips_parcels(db)
            db.execute("""DELETE FROM tank_engineering_cohort
                WHERE (terminal_id='cushing_connect' AND cohort_id LIKE 'former_cvr_efr_%')
                   OR cohort_id IN ('phillips_buxton_efr','phillips_cushing_efr','enterprise_east_efr',
                       'enterprise_west_owned_efr','enterprise_west_owned_ifr','amid_five_2009','hcs_five_large_efr',
                       'sunoco_little_farm_efr')""")
            db.executemany("INSERT INTO tank_engineering_cohort VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (*cushing_connect_rows(db), *phillips_nei_rows(db), *enterprise_and_sunoco_rows(db)))
            db.executemany("INSERT OR REPLACE INTO terminal_disclosed_capacity VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           csv_rows(REFERENCE/"cushing_terminal_disclosed_capacity.csv"))
            terminal_ids = ("cushing_connect", "former_cvr_cushing", "phillips66_cushing")
            db.executemany("INSERT OR REPLACE INTO terminal VALUES (?,?,?,?,?,?,?,?)",
                           (row for row in csv_rows(REFERENCE/"terminals.csv") if row[0] in terminal_ids))
            db.execute("DELETE FROM terminal_company_history WHERE terminal_id IN ('cushing_connect','former_cvr_cushing')")
            db.executemany("INSERT INTO terminal_company_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (row for row in csv_rows(REFERENCE/"terminal_company_history.csv") if row[0]=="cushing_connect"))
        report = write_scope_outputs(db)
        export("sunoco_little_farm_parcel_history", [dict(row) for row in db.execute("""SELECT * FROM tank_parcel_intersection
            WHERE candidate_id LIKE 'ortho-little-farm-%' ORDER BY snapshot_year,candidate_id""")])
        for name,order in (("terminal_disclosed_capacity","terminal_id,scope_key,reference_date"),
                           (EARLY_ROOF_TABLE,"scope_id,candidate_id,observed_date"),
                           ("terminal_data_coverage","terminal_id"),("tank_engineering_registry","candidate_id"),
                           ("tank_engineering_cohort","candidate_id,cohort_id"),("roof_physics_input_coverage","candidate_id"),
                           ("nei_roof_equipment_evidence","inventory_year,terminal_id,cohort_id,equipment_id"),
                           ("eia_roof_cohort_weekly_coverage","week_ending,cohort_id,ticker")):
            export(name,all_view_rows(db,name,order))
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
