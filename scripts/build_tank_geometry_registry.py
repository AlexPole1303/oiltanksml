"""Join dated permit evidence, independent geometry, and aerial observations."""
import datetime as dt
import json
import math

from build_tank_lidar_outlines import DATASET as CURRENT_LIDAR_DATASET, outline_rows
from collect_public import RAW, ROOT
from terminal_aerial_roof_evidence import TABLE as EARLY_ROOF_TABLE

CP_SOURCE = "https://www.cpenergy.com/facilities/tanks-terminals"
CP_LOCAL = RAW / "documents" / "cp_energy_tanks.html"
MAGELLAN_SOUTH_COHORT = "magellan_cushing_south_2011"
MAGELLAN_CENTRAL_PREFIX = "magellan_cushing_central_"
STROUD_COHORT = "plains_cushing_south_stroud_ifr_2009"
PLAINS_ORIGINAL_PREFIX = "plains_cushing_original_"
PLAINS_ORIGINAL_PARCEL = "0000-23-17N-05E-2-002-00"
PLAINS_DESIGN_SOURCE = "https://www.ogj.com/home/article/17211456/cushing-terminal-hits-full-operation"
PLAINS_DESIGN_FILE = RAW / "documents" / "plains_cushing_1994_ogj.html"
PLAINS_ORIGINAL_DESIGNS = ((100_000, 14, 120, 48), (150_000, 4, 150, 48))
PLAINS_EXPANSION_PREFIX = "plains_cushing_phase_"
PLAINS_EXPANSIONS = (
    ("i_v", 20, 250_000, 270_000, "1997", "2005", 63, 65, "issuer_existing_tank_capacity"),
    ("vi", 6, 570_000, 568_000, "2006", "2006", 85, 86, "issuer_planned_tank_capacity"),
)
PLAINS_EXPANSION_SOURCE = "https://www.sec.gov/Archives/edgar/data/1070423/000110465906061871/a06-19926_1425.htm"
PLAINS_EXPANSION_FILE = RAW / "documents" / "paa_cushing_2006_phase_vi_prnewswire.html"
PLAINS_AS_BUILT_SOURCE = "https://www.sec.gov/Archives/edgar/data/1070423/000095012908001329/h53460e10vk.htm"
PLAINS_AS_BUILT_FILE = RAW / "documents" / "paa_fy2007_10k.html"
GRANDSTAFF_PREFIX = "grandstaff_"
ROSE_ROCK_PREFIX = "rose_rock_north_"
ENBRIDGE_MAIN_PREFIX = "enbridge_main_"
NGL_LEGACY_PREFIX = "ngl_cushing_2011_"
NGL_LEGACY_DATE = "2011-12-08"
TERMINAL_GEOMETRY_STUDY = ("cushing_connect", "phillips66_cushing", "enterprise_cushing", "energy_transfer_cushing", "amid_cushing")
BBL_PER_CUBIC_METRE = 1 / 0.158987294928
FEET_TO_METRES = 0.3048

SCHEMA = """
CREATE TABLE permit_planned_tank_cohort (
    facility_key TEXT, document_id TEXT, unit_group TEXT, proposed_tanks INTEGER, stored_product TEXT,
    roof_type TEXT, capacity_bbl_each REAL, diameter_ft_in_emissions_calculation REAL, roster_pdf_page INTEGER,
    dimension_pdf_page INTEGER, source_url TEXT, PRIMARY KEY (document_id, unit_group));
CREATE TABLE tank_footprint_geometry (
    candidate_id TEXT PRIMARY KEY, center_east_m REAL, center_north_m REAL, area_m2 REAL, perimeter_m REAL,
    equivalent_diameter_m REAL, minimum_rectangle_short_m REAL, minimum_rectangle_long_m REAL,
    circularity REAL, volume_gradient_bbl_per_m REAL, source_file TEXT);
CREATE TABLE tank_lidar_outline (
    candidate_id TEXT PRIMARY KEY, dataset TEXT, first_upper_return_at_utc TEXT, last_upper_return_at_utc TEXT,
    prior_rim_z_p95_native REAL, prior_ground_z_p50_native REAL, upper_points INTEGER, sectors_present INTEGER,
    center_east_m REAL, center_north_m REAL, diameter_m REAL, center_shift_m REAL, sectors_within_055m INTEGER,
    radial_residual_median_m REAL, radial_residual_p90_m REAL, fit_status TEXT, source_directory TEXT);
CREATE TABLE tank_engineering_cohort (
    candidate_id TEXT, cohort_id TEXT, terminal_id TEXT, evidence_basis TEXT,
    possible_equipment_ids_json TEXT, documented_roof_type TEXT, permit_diameter_m REAL,
    reported_height_min_m REAL, reported_height_max_m REAL, reported_height_basis TEXT,
    reported_capacity_min_bbl REAL, reported_capacity_max_bbl REAL, reported_capacity_basis TEXT,
    volume_gradient_permit_diameter_bbl_per_m REAL, source_document_date TEXT,
    earliest_group_construction TEXT, latest_group_construction TEXT, construction_date_precision TEXT,
    source_pdf_page INTEGER, source_url TEXT, source_file TEXT, PRIMARY KEY (candidate_id, cohort_id));
CREATE TABLE permit_tank_capacity_crosscheck (
    facility_key TEXT,tank_id TEXT,document_id TEXT,memo_date TEXT,pdf_page INTEGER,document_section TEXT,
    capacity_gal REAL,capacity_bbl REAL,source_url TEXT,source_file TEXT,
    PRIMARY KEY (facility_key,tank_id,document_id,pdf_page));
CREATE TABLE plains_original_design_reference (
    cohort_id TEXT PRIMARY KEY, nominal_capacity_bbl INTEGER, reported_tank_count INTEGER,
    design_diameter_ft REAL, design_height_ft REAL, publication_date TEXT, source_author TEXT,
    source_organization TEXT, source_url TEXT, source_file TEXT);
CREATE TABLE plains_expansion_capacity_reference (
    cohort_id TEXT, reported_tank_count INTEGER, corporate_capacity_bbl_each INTEGER,
    corporate_capacity_basis TEXT, statement_date TEXT, source_url TEXT, source_file TEXT,
    PRIMARY KEY (cohort_id,statement_date));
CREATE TABLE plains_cushing_2008_aerial_observation (
    candidate_id TEXT PRIMARY KEY, review_number INTEGER, parcel_ids TEXT, lidar_upper_outline_diameter_m REAL,
    original_cohort_id TEXT, source_coverage_year INTEGER, capture_date_status TEXT, structure_observation TEXT,
    detail TEXT, previous_first_dated_aerial_seen TEXT, source_file TEXT, review_atlas TEXT, source_url TEXT);
CREATE TABLE enbridge_main_contour_group (
    candidate_id TEXT PRIMARY KEY, cohort_id TEXT NOT NULL, parcel_id TEXT NOT NULL);
CREATE TABLE enbridge_main_group_source (
    cohort_id TEXT, source_group_key TEXT, source_kind TEXT, statement_date TEXT, pdf_page INTEGER,
    reported_tank_count INTEGER, reported_group_capacity_bbl INTEGER, capacity_basis TEXT,
    source_url TEXT, source_file TEXT, evidence_note TEXT, PRIMARY KEY (cohort_id,source_group_key));
CREATE TABLE terminal_disclosed_capacity (
    disclosure_id TEXT PRIMARY KEY, terminal_id TEXT NOT NULL, scope_key TEXT NOT NULL,
    reference_date TEXT, reference_date_precision TEXT, publication_date TEXT,
    reported_tank_count INTEGER, capacity_bbl INTEGER NOT NULL, capacity_basis TEXT NOT NULL,
    source_url TEXT NOT NULL, source_file TEXT, evidence_note TEXT NOT NULL);
CREATE TABLE enbridge_main_covered_roof_outline (
    candidate_id TEXT PRIMARY KEY, dataset TEXT, first_upper_return_at_utc TEXT, last_upper_return_at_utc TEXT,
    prior_interior_p50_z_m REAL, prior_ground_p50_z_m REAL, upper_points INTEGER, sectors_present INTEGER,
    center_east_m REAL, center_north_m REAL, diameter_m REAL, center_shift_m REAL, sectors_within_055m INTEGER,
    radial_residual_median_m REAL, radial_residual_p90_m REAL, fit_status TEXT, source_directory TEXT,
    prior_interior_p10_z_m REAL, diameter_with_p10_prior_m REAL);
CREATE VIEW plains_permit_equipment_history AS
    WITH old AS (SELECT * FROM permit_tank WHERE facility_key='plains_cushing' AND document_id='5618'),
         recent AS (SELECT * FROM permit_tank WHERE facility_key='plains_cushing' AND document_id='8795'),
         ids AS (SELECT tank_id FROM old UNION SELECT tank_id FROM recent)
    SELECT i.tank_id, a.unit_group AS group_2019, b.unit_group AS group_2024,
        a.roof_type AS roof_2019, b.roof_type AS roof_2024, a.capacity_bbl AS capacity_bbl_2019,
        b.capacity_bbl AS capacity_bbl_2024, a.construction_or_modification_raw AS source_year_2019,
        b.construction_or_modification_raw AS source_year_2024, a.source_url AS source_url_2019,
        b.source_url AS source_url_2024, a.pdf_page AS pdf_page_2019, b.pdf_page AS pdf_page_2024,
        CASE WHEN a.tank_id IS NULL THEN 'only_in_2024_roster'
             WHEN b.tank_id IS NULL THEN 'only_in_2019_roster'
             WHEN a.roof_type IS NOT b.roof_type OR a.capacity_bbl IS NOT b.capacity_bbl THEN 'roof_or_capacity_differs'
             ELSE 'roof_and_capacity_agree' END AS comparison
    FROM ids i LEFT JOIN old a USING (tank_id) LEFT JOIN recent b USING (tank_id);
CREATE VIEW enbridge_main_permit_equipment_history AS
    WITH old AS (SELECT * FROM permit_tank WHERE facility_key='enbridge_cushing' AND document_id='3329'),
         recent AS (SELECT * FROM permit_tank WHERE facility_key='enbridge_cushing' AND document_id='8499'),
         ids AS (SELECT tank_id FROM old UNION SELECT tank_id FROM recent)
    SELECT i.tank_id,a.unit_group AS group_2016,b.unit_group AS group_2023,
        a.roof_type AS roof_2016,b.roof_type AS roof_2023,a.capacity_bbl AS capacity_bbl_2016,
        b.capacity_bbl AS capacity_bbl_2023,a.status_in_draft AS status_2016,b.status_in_draft AS status_2023,
        a.construction_or_modification_raw AS construction_2016,b.construction_or_modification_raw AS construction_2023,
        a.pdf_page AS pdf_page_2016,b.pdf_page AS pdf_page_2023,a.source_url AS source_url_2016,b.source_url AS source_url_2023,
        CASE WHEN b.emissions_unit_id='6487' THEN 'removed_and_demolished' END AS narrative_status_2023,
        CASE WHEN b.emissions_unit_id='6487' THEN 1 END AS narrative_pdf_page_2023,
        CASE WHEN a.tank_id IS NULL THEN 'only_in_2023_roster'
             WHEN b.tank_id IS NULL THEN 'only_in_2016_roster'
             WHEN b.emissions_unit_id='6487' AND b.status_in_draft<>'removed' THEN 'narrative_roster_status_conflict'
             WHEN a.status_in_draft='uncertain_merged_cell' OR b.status_in_draft='uncertain_merged_cell' THEN 'ambiguous_document_cell'
             WHEN a.status_in_draft<>b.status_in_draft THEN 'documented_status_changed'
             WHEN a.status_in_draft='removed' AND b.capacity_bbl IS NULL AND a.capacity_bbl IS NOT NULL
                 THEN 'earlier_capacity_not_repeated_after_removal'
             WHEN a.roof_type IS NOT b.roof_type OR a.capacity_bbl IS NOT b.capacity_bbl THEN 'roof_or_capacity_differs'
             ELSE 'status_roof_and_capacity_agree' END AS comparison
    FROM ids i LEFT JOIN old a USING (tank_id) LEFT JOIN recent b USING (tank_id);
CREATE VIEW north_farm_parcel_evidence AS
    WITH parcels AS (
        SELECT candidate_id,parcel_id,owner_name_recorded,snapshot_year,tank_area_fraction,source_url,
            CASE WHEN owner_name_recorded='BKEP Crude Storage, LLC' THEN 'enbridge_blueknight'
                 ELSE 'semgroup_cushing' END AS terminal_id,
            CASE WHEN parcel_id LIKE '18N05E-27-%' THEN 'central' ELSE 'north' END AS farm_block
        FROM tank_parcel_intersection WHERE county='payne' AND snapshot_kind='annual_assessment' AND snapshot_year=2025
          AND ((owner_name_recorded='BKEP Crude Storage, LLC' AND (parcel_id LIKE '18N05E-22-%' OR parcel_id LIKE '18N05E-27-%'))
             OR (owner_name_recorded='Semcrude, LP' AND parcel_id LIKE '18N05E-22-%'))
    )
    SELECT candidate_id,terminal_id,farm_block,snapshot_year,owner_name_recorded,
        group_concat(DISTINCT parcel_id) AS parcel_ids,sum(tank_area_fraction) AS same_owner_area_fraction,
        min(source_url) AS parcel_source_url
    FROM parcels GROUP BY candidate_id,terminal_id,farm_block,snapshot_year,owner_name_recorded
    HAVING sum(tank_area_fraction) >= 0.99;
CREATE VIEW tank_engineering_registry AS
    WITH aerial_dates AS (
        SELECT candidate_id, min(CASE WHEN structure_observation='aboveground_tank_seen' THEN observed_date END) AS first_seen,
            max(CASE WHEN structure_observation='no_aboveground_tank_seen' THEN observed_date END) AS last_absence,
            max(observed_date) AS latest_aerial_date
        FROM tank_aerial_visual_observation GROUP BY candidate_id
    ), terminals AS (
        SELECT candidate_id, group_concat(DISTINCT terminal_id) AS associated_terminal_ids
        FROM tank_terminal_research_attribution GROUP BY candidate_id
    ), candidates AS (
        SELECT candidate_id,longitude,latitude,'OSM' AS outline_source FROM tank_candidate
        UNION ALL
        SELECT candidate_id,longitude,latitude,'state_orthophoto' FROM supplemental_tank_candidate
    )
    SELECT c.*, g.center_east_m AS footprint_east_m,g.center_north_m AS footprint_north_m,
        g.area_m2 AS footprint_area_m2,g.equivalent_diameter_m AS footprint_diameter_m,g.circularity,
        g.volume_gradient_bbl_per_m AS footprint_bbl_per_m,g.source_file AS footprint_source_file,
        l.diameter_m AS lidar_upper_outline_diameter_m,l.center_shift_m AS lidar_center_shift_m,
        l.radial_residual_median_m AS lidar_median_radial_residual_m,l.sectors_within_055m AS lidar_supported_sectors,
        l.fit_status AS lidar_fit_status,l.first_upper_return_at_utc,l.last_upper_return_at_utc,
        l.prior_rim_z_p95_native-l.prior_ground_z_p50_native AS lidar_rim_to_ground_native_z,
        (3.141592653589793 / (4 * 0.158987294928))*l.diameter_m*l.diameter_m AS lidar_circular_area_bbl_per_m,
        a.first_seen,a.last_absence,a.latest_aerial_date,
        (SELECT group_concat(DISTINCT v.top_observation) FROM tank_aerial_visual_observation v
            WHERE v.candidate_id=c.candidate_id AND v.observed_date=a.latest_aerial_date) AS latest_aerial_top_observation,
        t.associated_terminal_ids,e.terminal_id AS documented_terminal_id,e.cohort_id,e.evidence_basis,
        e.possible_equipment_ids_json,e.documented_roof_type,e.permit_diameter_m,
        e.reported_height_min_m,e.reported_height_max_m,e.reported_height_basis,
        e.reported_capacity_min_bbl,e.reported_capacity_max_bbl,e.reported_capacity_basis,
        e.volume_gradient_permit_diameter_bbl_per_m,e.source_document_date,
        e.earliest_group_construction,e.latest_group_construction,e.construction_date_precision,
        e.source_pdf_page,e.source_url AS engineering_source_url,e.source_file AS engineering_source_file
    FROM candidates c
    LEFT JOIN tank_footprint_geometry g USING (candidate_id)
    LEFT JOIN tank_lidar_outline l USING (candidate_id)
    LEFT JOIN aerial_dates a USING (candidate_id)
    LEFT JOIN terminals t USING (candidate_id)
    LEFT JOIN tank_engineering_cohort e USING (candidate_id);
CREATE VIEW magellan_central_fixed_roof_check AS
    WITH unassigned AS (
        SELECT DISTINCT t.candidate_id,t.lidar_upper_outline_diameter_m,t.lidar_rim_to_ground_native_z,
            t.lidar_circular_area_bbl_per_m,t.latest_aerial_top_observation
        FROM tank_facility_evidence p JOIN tank_engineering_registry t USING(candidate_id)
        WHERE p.facility_key='magellan_cushing_central' AND t.cohort_id IS NULL
    )
    SELECT p.*,u.candidate_id AS unassigned_parcel_contour_id,u.lidar_upper_outline_diameter_m,
        u.lidar_rim_to_ground_native_z,u.latest_aerial_top_observation,
        round(p.capacity_bbl/u.lidar_circular_area_bbl_per_m,6) AS equivalent_fill_depth_at_contour_area_m,
        'equipment_to_contour_unresolved' AS match_status
    FROM permit_tank_capacity_crosscheck p LEFT JOIN unassigned u ON 1
    WHERE p.facility_key='magellan_cushing_central' AND p.tank_id='0011';
CREATE VIEW stroud_tank_specification_check AS
    SELECT t.candidate_id,t.documented_terminal_id,t.documented_roof_type,t.possible_equipment_ids_json,
        t.earliest_group_construction,t.construction_date_precision,t.first_seen AS first_seen_aerial_date,
        t.reported_capacity_min_bbl AS nominal_capacity_bbl,t.reported_height_min_m AS application_shell_height_m,
        t.reported_height_basis,t.permit_diameter_m AS application_diameter_m,t.footprint_diameter_m,
        t.lidar_upper_outline_diameter_m,t.lidar_fit_status,
        round(t.lidar_upper_outline_diameter_m-t.permit_diameter_m,6) AS lidar_minus_application_diameter_m,
        t.lidar_rim_to_ground_native_z,t.source_document_date AS application_receipt_date,
        t.source_pdf_page,t.engineering_source_url,t.engineering_source_file,
        a.roof_capacity_memo_date,a.roof_capacity_memo_url,
        s.fgdc_flight_start AS legacy_flight_window_start,s.fgdc_flight_end AS legacy_flight_window_end,
        s.date_status AS legacy_date_status,s.interior_first_returns AS legacy_interior_first_returns
    FROM tank_engineering_registry t
    LEFT JOIN lidar_stroud_legacy_observation s USING(candidate_id)
    CROSS JOIN (SELECT min(memo_date) AS roof_capacity_memo_date,min(source_url) AS roof_capacity_memo_url
                FROM permit_tank WHERE document_id='9030') a
    WHERE t.cohort_id='plains_cushing_south_stroud_ifr_2009';
CREATE VIEW plains_cushing_contour_reconciliation AS
    WITH parcels AS (
        SELECT candidate_id,group_concat(DISTINCT parcel_id) AS parcel_ids
        FROM tank_facility_evidence WHERE facility_key='plains_cushing' GROUP BY candidate_id
    ), contours AS (
        SELECT candidate_id FROM parcels
        UNION SELECT candidate_id FROM tank_terminal_research_attribution WHERE terminal_id='plains_cushing'
        UNION SELECT candidate_id FROM tank_engineering_cohort WHERE terminal_id='plains_cushing'
    )
    SELECT c.candidate_id,p.parcel_ids,t.longitude,t.latitude,t.first_seen AS first_seen_aerial_date,
        t.cohort_id,t.possible_equipment_ids_json,t.documented_roof_type,t.reported_capacity_min_bbl AS capacity_min_bbl,
        t.reported_capacity_max_bbl AS capacity_max_bbl,t.reported_capacity_basis,t.source_document_date,
        t.earliest_group_construction AS roster_construction_year_min,t.latest_group_construction AS roster_construction_year_max,
        t.engineering_source_url,t.source_pdf_page,
        t.evidence_basis,t.footprint_diameter_m,t.lidar_upper_outline_diameter_m,t.lidar_rim_to_ground_native_z,
        t.lidar_fit_status,d.design_diameter_ft,d.design_height_ft,
        round(d.design_diameter_ft*0.3048,6) AS design_diameter_m,
        round(d.design_height_ft*0.3048,6) AS design_height_m,
        round(t.lidar_upper_outline_diameter_m-d.design_diameter_ft*0.3048,6) AS lidar_minus_design_diameter_m,
        round(t.lidar_rim_to_ground_native_z-d.design_height_ft*0.3048,6) AS lidar_minus_design_height_m,
        round(3.141592653589793/0.158987294928*(d.design_diameter_ft*0.3048/2)*(d.design_diameter_ft*0.3048/2),6)
            AS design_circular_area_bbl_per_m,
        t.lidar_circular_area_bbl_per_m,d.publication_date AS design_publication_date,
        d.source_url AS design_source_url,d.source_file AS design_source_file,
        e.corporate_capacity_bbl_each,e.corporate_capacity_basis,e.statement_date AS corporate_statement_date,
        e.source_url AS corporate_source_url,e.source_file AS corporate_source_file,
        q.corporate_capacity_bbl_each AS earlier_corporate_capacity_bbl_each,
        q.corporate_capacity_basis AS earlier_corporate_capacity_basis,q.statement_date AS earlier_corporate_statement_date,
        q.source_url AS earlier_corporate_source_url,q.source_file AS earlier_corporate_source_file,
        a.source_coverage_year AS aerial_coverage_year,a.capture_date_status AS aerial_capture_date_status,
        a.structure_observation AS aerial_structure_observation,a.source_url AS aerial_source_url,
        CASE WHEN t.documented_terminal_id='plains_cushing' THEN 'equipment_group_matched'
             ELSE 'equipment_to_contour_unresolved' END AS match_status
    FROM contours c JOIN tank_engineering_registry t USING(candidate_id)
    LEFT JOIN parcels p USING(candidate_id)
    LEFT JOIN plains_original_design_reference d USING(cohort_id)
    LEFT JOIN plains_expansion_capacity_reference e ON e.cohort_id=t.cohort_id
        AND e.corporate_capacity_basis='issuer_as_built_tank_capacity'
    LEFT JOIN plains_expansion_capacity_reference q ON q.cohort_id=t.cohort_id
        AND q.corporate_capacity_basis IN ('issuer_existing_tank_capacity','issuer_planned_tank_capacity')
    LEFT JOIN plains_cushing_2008_aerial_observation a USING(candidate_id);
CREATE VIEW north_farm_contour_reconciliation AS
    WITH contours AS (
        SELECT candidate_id,terminal_id FROM north_farm_parcel_evidence
        UNION SELECT candidate_id,terminal_id FROM tank_terminal_research_attribution
        WHERE terminal_id IN ('enbridge_blueknight','semgroup_cushing')
    )
    SELECT c.candidate_id,c.terminal_id,p.farm_block,p.parcel_ids,p.same_owner_area_fraction,p.snapshot_year,
        p.owner_name_recorded,p.parcel_source_url,t.longitude,t.latitude,t.first_seen AS first_seen_aerial_date,
        t.latest_aerial_date,t.latest_aerial_top_observation,t.last_absence AS most_recent_aerial_absence,
        t.footprint_diameter_m,t.lidar_upper_outline_diameter_m,t.lidar_rim_to_ground_native_z,t.lidar_fit_status,
        t.cohort_id,t.possible_equipment_ids_json,t.documented_roof_type,t.reported_capacity_min_bbl,
        t.earliest_group_construction,t.latest_group_construction,t.source_document_date,t.source_pdf_page,
        d.facility_id,d.document_id,t.engineering_source_url,t.engineering_source_file,t.evidence_basis,
        (SELECT group_concat(DISTINCT r.pdf_page) FROM json_each(t.possible_equipment_ids_json) i
         JOIN permit_tank r ON r.document_id=d.document_id AND r.facility_key=d.facility_key AND r.tank_id=i.value)
            AS equipment_pdf_pages,
        CASE WHEN p.candidate_id IS NULL THEN 'outside_matched_north_central_parcels'
             WHEN t.documented_terminal_id=c.terminal_id THEN 'equipment_group_matched'
             ELSE 'equipment_to_contour_unresolved' END AS match_status
    FROM contours c JOIN tank_engineering_registry t USING(candidate_id)
    LEFT JOIN north_farm_parcel_evidence p USING(candidate_id,terminal_id)
    LEFT JOIN permit_document d ON d.source_url=t.engineering_source_url;
CREATE VIEW enbridge_main_contour_reconciliation AS
    WITH parcels AS (
        SELECT candidate_id,group_concat(DISTINCT parcel_id) AS parcel_ids
        FROM tank_facility_evidence WHERE facility_key='enbridge_cushing' GROUP BY candidate_id
    ), contours AS (
        SELECT candidate_id FROM parcels
        UNION SELECT candidate_id FROM enbridge_main_contour_group
    ), legacy AS (
        SELECT candidate_id,max(rim_z_p95-ground_z_p50) AS rim_above_ground_m,
            max(interior_z_p90-ground_z_p50) AS interior_p90_above_ground_m,
            sum(interior_points) AS interior_points
        FROM lidar_tank_observation WHERE acquisition_date_utc='2011-12-08' GROUP BY candidate_id
    )
    SELECT c.candidate_id,coalesce(m.parcel_id,p.parcel_ids) AS supporting_parcel_ids,
        (SELECT max(p2.tank_area_fraction) FROM tank_parcel_intersection p2 WHERE p2.candidate_id=c.candidate_id
            AND p2.parcel_id=m.parcel_id AND lower(p2.owner_name_recorded) IN ('enbridge storage (cushing), llc','enbridge storage (cushing) llc'))
            AS matched_parcel_fraction,
        t.longitude,t.latitude,t.first_seen AS first_seen_aerial_date,t.last_absence AS latest_aerial_absence,
        t.latest_aerial_top_observation,t.footprint_diameter_m,t.lidar_upper_outline_diameter_m,t.lidar_fit_status,
        o.diameter_m AS lidar_covered_surface_outline_diameter_m,o.fit_status AS lidar_covered_surface_fit_status,
        t.lidar_rim_to_ground_native_z,l.rim_above_ground_m AS lidar_2011_rim_above_ground_m,
        l.interior_p90_above_ground_m AS lidar_2011_interior_p90_above_ground_m,l.interior_points AS lidar_2011_interior_points,
        t.cohort_id,t.documented_roof_type,t.possible_equipment_ids_json,t.reported_capacity_min_bbl,t.reported_capacity_max_bbl,
        t.earliest_group_construction,t.latest_group_construction,t.construction_date_precision,t.source_document_date,
        t.engineering_source_url,t.engineering_source_file,t.evidence_basis,
        (SELECT group_concat(DISTINCT r.pdf_page) FROM json_each(t.possible_equipment_ids_json) i
         JOIN permit_tank r ON r.document_id='3329' AND r.facility_key='enbridge_cushing' AND r.tank_id=i.value) AS permit_pdf_pages,
        CASE WHEN t.documented_terminal_id='enbridge_cushing' THEN 'equipment_group_matched'
             ELSE 'equipment_to_contour_unresolved' END AS match_status
    FROM contours c JOIN tank_engineering_registry t USING(candidate_id)
    LEFT JOIN parcels p USING(candidate_id) LEFT JOIN enbridge_main_contour_group m USING(candidate_id)
    LEFT JOIN legacy l USING(candidate_id) LEFT JOIN enbridge_main_covered_roof_outline o USING(candidate_id);
CREATE VIEW enbridge_main_cohort_capacity AS
    WITH groups AS (
        SELECT cohort_id,count(*) AS contours,max(documented_roof_type) AS roof_type,
            max(possible_equipment_ids_json) AS equipment_ids_json,
            min(coalesce(lidar_upper_outline_diameter_m,lidar_covered_surface_outline_diameter_m)) AS lidar_diameter_min_m,
            max(coalesce(lidar_upper_outline_diameter_m,lidar_covered_surface_outline_diameter_m)) AS lidar_diameter_max_m
        FROM enbridge_main_contour_reconciliation WHERE match_status='equipment_group_matched' GROUP BY cohort_id
    )
    SELECT g.*,count(r.tank_id) AS permit_equipment_count,
        round(sum(CASE WHEN r.capacity_unit='million_gal' THEN r.capacity_value*1000000.0/42 ELSE r.capacity_bbl END),3)
            AS permit_capacity_total_bbl
    FROM groups g JOIN json_each(g.equipment_ids_json) i
    JOIN permit_tank r ON r.document_id='3329' AND r.facility_key='enbridge_cushing' AND r.tank_id=i.value
    GROUP BY g.cohort_id;
CREATE VIEW ngl_cushing_contour_reconciliation AS
    WITH legacy AS (
        SELECT candidate_id,max(rim_z_p95-ground_z_p50) AS rim_above_ground_m
        FROM lidar_tank_observation WHERE acquisition_date_utc='2011-12-08' GROUP BY candidate_id
    )
    SELECT p.candidate_id,p.parcel_id,p.owner_name_recorded,p.snapshot_year,p.tank_area_fraction,
        p.source_url AS parcel_source_url,
        CASE p.parcel_id WHEN '0000-24-17N-05E-1-003-00' THEN 'ngl_older_parcel'
                         WHEN '0000-24-17N-05E-1-001-00' THEN 'ngl_later_parcel'
                         WHEN '0000-24-17N-05E-1-002-00' THEN 'cushing_connect_parcel'
                         WHEN '0000-24-17N-05E-1-005-00' THEN 'canyon_crossing_parcel' END AS parcel_group,
        t.first_seen AS first_seen_aerial_date,t.last_absence AS latest_aerial_absence,
        t.lidar_upper_outline_diameter_m,t.lidar_fit_status,l.rim_above_ground_m AS lidar_2011_rim_above_ground_m,
        t.lidar_rim_to_ground_native_z,t.lidar_circular_area_bbl_per_m,t.latest_aerial_top_observation,
        a.top_observation AS aerial_top_2021,b.top_observation AS aerial_top_2023,
        t.cohort_id,t.documented_roof_type,t.reported_capacity_min_bbl AS nominal_capacity_bbl,
        t.reported_capacity_basis,t.source_document_date,t.engineering_source_url,t.evidence_basis,
        CASE WHEN t.documented_terminal_id='ngl_cushing' THEN 'contractor_group_partial_match'
             WHEN p.parcel_id='0000-24-17N-05E-1-001-00' THEN 'later_contour_equipment_unresolved'
             WHEN l.rim_above_ground_m>=13 THEN 'adjacent_2011_contour_contract_scope_unresolved'
             ELSE 'adjacent_later_contour' END AS match_status
    FROM tank_parcel_intersection p JOIN tank_engineering_registry t USING(candidate_id)
    LEFT JOIN legacy l USING(candidate_id)
    LEFT JOIN tank_aerial_visual_observation a ON a.candidate_id=p.candidate_id AND a.image_source='USDA NAIP'
        AND a.observed_date='2021-05-06'
    LEFT JOIN tank_aerial_visual_observation b ON b.candidate_id=p.candidate_id AND b.image_source='USDA NAIP'
        AND b.observed_date='2023-06-15'
    WHERE p.county='lincoln' AND p.snapshot_kind='retrieved_current' AND p.tank_area_fraction>=0.99
        AND p.parcel_id IN ('0000-24-17N-05E-1-001-00','0000-24-17N-05E-1-002-00',
                            '0000-24-17N-05E-1-003-00','0000-24-17N-05E-1-005-00');
CREATE VIEW enterprise_cushing_contour_reconciliation AS
    SELECT c.candidate_id,c.terminal_id AS asset_scope_id,p.parcel_id,p.owner_name_recorded AS land_owner,
        p.snapshot_year,p.tank_area_fraction,p.source_url AS parcel_source_url,
        t.first_seen AS first_seen_aerial_date,t.lidar_upper_outline_diameter_m,t.lidar_fit_status,
        t.lidar_rim_to_ground_native_z,t.lidar_circular_area_bbl_per_m,t.latest_aerial_top_observation,
        a.top_observation AS aerial_top_2021,b.top_observation AS aerial_top_2023,t.cohort_id,
        t.documented_roof_type,t.reported_capacity_min_bbl AS nominal_capacity_bbl,t.reported_capacity_basis,
        t.engineering_source_url,
        CASE WHEN c.terminal_id='amid_cushing' THEN 'five_tank_ground_lease_asset'
             ELSE 'enterprise_west_parcel_contour_equipment_unresolved' END AS match_status
    FROM tank_terminal_research_attribution c JOIN tank_engineering_registry t USING(candidate_id)
    JOIN tank_parcel_intersection p ON p.candidate_id=c.candidate_id AND p.county='payne'
        AND p.snapshot_kind='annual_assessment' AND p.snapshot_year=2025 AND p.tank_area_fraction>=0.99
    LEFT JOIN tank_aerial_visual_observation a ON a.candidate_id=c.candidate_id AND a.image_source='USDA NAIP'
        AND a.observed_date='2021-05-06'
    LEFT JOIN tank_aerial_visual_observation b ON b.candidate_id=c.candidate_id AND b.image_source='USDA NAIP'
        AND b.observed_date='2023-06-15'
    WHERE c.terminal_id IN ('enterprise_cushing','amid_cushing');
"""

TERMINAL_DATA_COVERAGE_VIEW = """
CREATE VIEW terminal_data_coverage AS
    WITH economic_scope AS (
        SELECT candidate_id,terminal_id FROM tank_terminal_scope_parcel_evidence
        WHERE scope_id='transmontaigne_ground_lease'
    ), permit_parcels AS (
        SELECT DISTINCT e.candidate_id,coalesce(s.terminal_id,p.terminal_id) AS terminal_id
        FROM tank_facility_evidence e JOIN permit_document p USING(facility_key)
        LEFT JOIN economic_scope s ON s.candidate_id=e.candidate_id AND p.terminal_id='enbridge_blueknight'
        WHERE p.terminal_id IS NOT NULL
    ), engineering AS (
        SELECT e.candidate_id,coalesce(s.terminal_id,e.terminal_id) AS terminal_id,e.cohort_id,
            e.documented_roof_type,e.reported_capacity_min_bbl,e.reported_height_min_m
        FROM tank_engineering_cohort e
        LEFT JOIN economic_scope s ON s.candidate_id=e.candidate_id AND e.terminal_id='enbridge_blueknight'
    ), candidates AS (
        SELECT candidate_id,terminal_id FROM tank_terminal_research_attribution
        UNION SELECT candidate_id,terminal_id FROM engineering
        UNION SELECT candidate_id,terminal_id FROM permit_parcels
    ), candidate_terminals AS (
        SELECT candidate_id,count(*) AS terminal_candidates FROM candidates GROUP BY candidate_id
    ), coverage AS (
        SELECT c.terminal_id,count(*) AS associated_contours,
            count(CASE WHEN t.terminal_candidates>1 THEN 1 END) AS contours_with_multiple_terminal_associations,
            count(CASE WHEN p.candidate_id IS NOT NULL THEN 1 END) AS parcel_and_permit_contours,
            count(CASE WHEN e.documented_roof_type IS NOT NULL THEN 1 END) AS contours_with_documented_roof_type,
            count(CASE WHEN e.documented_roof_type='EFR' THEN 1 END) AS documented_open_roof_contours,
            count(CASE WHEN e.documented_roof_type IN ('IFR','Domed EFR','FR') THEN 1 END) AS documented_covered_roof_contours,
            count(CASE WHEN e.reported_capacity_min_bbl IS NOT NULL THEN 1 END) AS contours_with_reported_capacity,
            count(CASE WHEN e.reported_height_min_m IS NOT NULL THEN 1 END) AS contours_with_reported_height,
            count(CASE WHEN e.has_independent_design_dimensions THEN 1 END) AS contours_with_independent_design_dimensions
        FROM candidates c JOIN candidate_terminals t USING(candidate_id)
        LEFT JOIN permit_parcels p USING(candidate_id,terminal_id)
        LEFT JOIN (
            SELECT candidate_id,terminal_id,max(documented_roof_type) AS documented_roof_type,
                max(reported_capacity_min_bbl) AS reported_capacity_min_bbl,max(reported_height_min_m) AS reported_height_min_m,
                max(d.cohort_id IS NOT NULL) AS has_independent_design_dimensions
            FROM engineering LEFT JOIN plains_original_design_reference d USING(cohort_id)
            GROUP BY candidate_id,terminal_id
        ) e USING(candidate_id,terminal_id)
        GROUP BY c.terminal_id
    ), documents AS (
        SELECT terminal_id,count(*) AS permit_documents FROM permit_document WHERE terminal_id IS NOT NULL GROUP BY terminal_id
    ), company AS (
        SELECT terminal_id,group_concat(DISTINCT ticker) AS historical_tickers,count(*) AS company_history_rows,
            min(CASE WHEN date_precision='day' THEN valid_from END) AS first_day_precision_start,
            min(CASE WHEN date_precision='documented_from' THEN valid_from END) AS first_documented_exposure_bound
        FROM terminal_company_history GROUP BY terminal_id
    )
    SELECT t.*,c.historical_tickers,coalesce(c.company_history_rows,0) AS company_history_rows,
        c.first_day_precision_start,c.first_documented_exposure_bound,coalesce(d.permit_documents,0) AS permit_documents,
        coalesce(v.associated_contours,0) AS associated_contours,
        coalesce(v.contours_with_multiple_terminal_associations,0) AS contours_with_multiple_terminal_associations,
        coalesce(v.parcel_and_permit_contours,0) AS parcel_and_permit_contours,
        coalesce(v.contours_with_documented_roof_type,0) AS contours_with_documented_roof_type,
        coalesce(v.documented_open_roof_contours,0) AS documented_open_roof_contours,
        coalesce(v.documented_covered_roof_contours,0) AS documented_covered_roof_contours,
        coalesce(v.contours_with_reported_capacity,0) AS contours_with_reported_capacity,
        coalesce(v.contours_with_reported_height,0) AS contours_with_reported_height,
        coalesce(v.contours_with_independent_design_dimensions,0) AS contours_with_independent_design_dimensions
    FROM terminal t LEFT JOIN coverage v USING(terminal_id) LEFT JOIN documents d USING(terminal_id)
    LEFT JOIN company c USING(terminal_id);
"""

SCHEMA += TERMINAL_DATA_COVERAGE_VIEW


def raw_tank_specs(db, document):
    return {row[0]: row[1:] for row in db.execute("""SELECT tank_id,roof_type,diameter_ft,height_ft,height_basis,
        capacity_bbl,capacity_basis,construction_or_modification_raw,memo_date,pdf_page,source_url
        FROM permit_tank WHERE document_id=? AND status_in_draft IN ('listed','as_built_reported')""", (document,))}


def grouped_specs(identity, terminal, cohort, match, ids, equipment, source_file, *, start, end, precision):
    rows = [equipment[tank] for tank in ids]
    roof, diameter_ft, _, height_basis, _, capacity_basis, _, document_date, _, url = rows[0]
    if any((row[0], row[1], row[3], row[5], row[7], row[9]) != (roof, diameter_ft, height_basis, capacity_basis,
                                                              document_date, url) for row in rows):
        raise ValueError(f"Inconsistent engineering definitions within {cohort}")
    pages = {row[8] for row in rows}
    page = next(iter(pages)) if len(pages) == 1 else None
    heights = [row[2] * FEET_TO_METRES for row in rows if row[2] is not None]
    capacities = [row[4] for row in rows]
    diameter = diameter_ft * FEET_TO_METRES if diameter_ft is not None else None
    gradient = math.pi * (diameter / 2)**2 * BBL_PER_CUBIC_METRE if diameter is not None else None
    return (identity, cohort, terminal, match, json.dumps(ids), roof, diameter, min(heights, default=None),
            max(heights, default=None), height_basis,
            min(capacities), max(capacities), capacity_basis, gradient, document_date, start, end, precision,
            page, url, source_file)


def cushing_connect_rows(db):
    equipment = raw_tank_specs(db, "8795")
    contours = list(db.execute("""SELECT s.candidate_id,r.first_seen,a.structure_observation
        FROM tank_terminal_scope_parcel_evidence s JOIN tank_engineering_registry r USING(candidate_id)
        JOIN tank_aerial_visual_observation a ON a.candidate_id=s.candidate_id AND a.observed_date='2015-06-23'
        WHERE s.scope_id='cushing_connect_terminal' ORDER BY s.candidate_id"""))
    if not contours:
        return
    groups = (
        ("2011", ("25C1", "25C2", "25C3", "25C4"), "2013-06-11", "aboveground_tank_seen"),
        ("2015", ("25C5", "25C6"), "2017-05-07", "shell_or_construction_seen"),
    )
    if len(contours) != sum(len(ids) for _, ids, _, _ in groups):
        raise ValueError("Cushing Connect parcel and former CVR permit tank counts disagree")
    for year, ids, first_seen, structure in groups:
        group = [candidate for candidate, first, observed in contours if (first, observed) == (first_seen, structure)]
        if len(group) != len(ids) or any(
                equipment[tank][0] != "EFR" or equipment[tank][4] != 250_000 or equipment[tank][6] != year for tank in ids):
            raise ValueError(f"Cushing Connect construction cohort {year} needs review")
        for candidate in group:
            yield grouped_specs(candidate, "cushing_connect", f"former_cvr_efr_{year}",
                                "former_cvr_permit_roster_current_parcel_and_aerial_construction_cohort",
                                ids, equipment, "data/raw/documents/source_pdfs/8795.pdf",
                                start=year, end=year, precision="year")


def phillips_nei_rows(db):
    scopes = (("phillips_buxton", "phillips_buxton_efr"),
              ("phillips_named_cushing_terminal", "phillips_cushing_efr"))
    for scope, cohort in scopes:
        mapped = list(db.execute("""SELECT candidate_id FROM tank_terminal_scope_parcel_evidence
            WHERE scope_id=? ORDER BY candidate_id""", (scope,)))
        if not mapped:
            continue
        equipment = list(db.execute("""SELECT equipment_id,documented_roof_type,archive_version_date,source_url,source_file
            FROM nei_roof_equipment_evidence WHERE cohort_id=? AND inventory_year=2020 ORDER BY equipment_id""", (cohort,)))
        if len(mapped) != 2 or len(equipment) != len(mapped) or len({row[1:] for row in equipment}) != 1:
            raise ValueError(f"Phillips 66 NEI and parcel inventories disagree for {scope}")
        ids = [row[0] for row in equipment]
        _, roof, archive_date, url, file = equipment[0]
        if scope == "phillips_named_cushing_terminal" and (
                "T824" not in ids or ("osm-way-460925471",) not in mapped):
            raise ValueError("The documented Phillips 66 Tank 824 association needs review")
        for candidate, in mapped:
            possible_ids = ids
            basis = "nei_roof_equipment_and_named_site_parcel_group"
            if scope == "phillips_named_cushing_terminal":
                exact = candidate == "osm-way-460925471"
                possible_ids = ["T824"] if exact else [tank for tank in ids if tank != "T824"]
                basis = ("nei_roof_equipment_and_phmsa_point_inside_contour" if exact
                         else "nei_roof_equipment_and_remaining_contour_in_two_tank_group")
            yield (candidate, cohort, "phillips66_cushing", basis, json.dumps(possible_ids),
                   roof, None, None, None, None, None, None, None, None, archive_date, None, None, None, None, url, file)


def enterprise_and_sunoco_rows(db):
    def contours(scope):
        return list(db.execute("""SELECT s.candidate_id,r.first_seen,r.latest_aerial_top_observation,
            r.lidar_upper_outline_diameter_m FROM tank_terminal_scope_parcel_evidence s
            JOIN tank_engineering_registry r USING(candidate_id) WHERE s.scope_id=? ORDER BY s.candidate_id""", (scope,)))

    east = contours("enterprise_east")
    west = contours("enterprise_west_owned")
    hcs = contours("hcs_ground_lease")
    little_farm = contours("sunoco_little_farm")
    if not (east or west or hcs or little_farm):
        return
    if tuple(map(len, (east, west, hcs, little_farm))) != (10, 10, 5, 5):
        raise ValueError("Enterprise and Sunoco parcel counts disagree with the reviewed groups")
    if any(row[2] != "inset_wall_shadow_visible" for row in (*east, *hcs)):
        raise ValueError("An Enterprise East or HCS contour lacks the reviewed exterior shadow")
    if min(row[3] for row in hcs) <= max(row[3] for row in west):
        raise ValueError("The five large HCS contours are no longer distinct from Enterprise West")
    west_open = [row for row in west if row[2] == "inset_wall_shadow_visible"]
    west_covered = [row for row in west if row[2] == "top_without_distinct_inset_shadow"]
    sunoco_open = [row for row in little_farm if row[2] == "clear_inset_wall_shadow"]
    if (len(west_open), len(west_covered), len(sunoco_open)) != (3, 7, 1):
        raise ValueError("Enterprise West and Sunoco visual roof groups need review")
    early_dates = {}
    for scope, mapped in (("enterprise_west_owned", west), ("sunoco_little_farm", little_farm)):
        reviews = list(db.execute(f"""SELECT candidate_id,top_observation,observed_date,publication_date
            FROM {EARLY_ROOF_TABLE} WHERE scope_id=? ORDER BY candidate_id""", (scope,)))
        if ([row[:2] for row in reviews] != [(row[0], row[2]) for row in mapped]
                or len({row[2:] for row in reviews}) != 1):
            raise ValueError(f"The earlier dated aerial roof review disagrees with the confirmed group for {scope}")
        early_dates[scope] = reviews[0][3]

    groups = (
        ("enterprise_east_efr", "enterprise_cushing", east, None,
         "nei_ten_efr_units_and_ten_historical_parcel_contours"),
        ("enterprise_west_owned_efr", "enterprise_cushing", west_open, early_dates["enterprise_west_owned"],
         "nei_west_roster_phmsa_leased_tank_count_and_aerial_inset_shadows"),
        ("enterprise_west_owned_ifr", "enterprise_cushing", west_covered, early_dates["enterprise_west_owned"],
         "nei_west_roster_remaining_covered_contours_after_open_group_reconciliation"),
        ("sunoco_little_farm_efr", "energy_transfer_cushing", sunoco_open, early_dates["sunoco_little_farm"],
         "nei_single_external_roof_same_parcel_and_one_clear_aerial_inset_shadow"),
    )
    for cohort, terminal, mapped, visual_date, basis in groups:
        equipment = list(db.execute("""SELECT equipment_id,documented_roof_type,archive_version_date,source_url,source_file
            FROM nei_roof_equipment_evidence WHERE cohort_id=? AND inventory_year=2017 ORDER BY equipment_id""", (cohort,)))
        if len(mapped) != len(equipment) or len({row[1:] for row in equipment}) != 1:
            raise ValueError(f"The published NEI roster disagrees with the reconciled contours for {cohort}")
        _, roof, archive_date, url, file = equipment[0]
        ids = json.dumps([row[0] for row in equipment])
        for candidate, _, _, _ in mapped:
            yield (candidate, cohort, terminal, basis, ids, roof, None, None, None, None, None, None, None, None,
                   max(archive_date, visual_date or archive_date), None, None, None, None, url, file)

    for candidate, _, _, _ in hcs:
        capacity, documented, year, page, url, file = db.execute("""SELECT capacity_bbl,documented_by,
            documented_build_year,map_pdf_page,source_url,local_pdf FROM tank_asset_cohort
            WHERE candidate_id=? AND cohort_id='amid_five_2009'""", (candidate,)).fetchone()
        yield (candidate, "amid_five_2009", "amid_cushing", "phmsa_five_large_efr_and_separate_issuer_ground_lease",
               None, "EFR", None, None, None, None, capacity, capacity, "issuer_shell_capacity", None,
               documented, year, year, "year", page, url, file)


def southbow_rows(db):
    specs = raw_tank_specs(db, "6816")
    for identity, cohort, possible, evidence in db.execute("""SELECT candidate_id,cohort_id,possible_permit_tank_ids,
        diameter_evidence FROM tank_permit_cohort_evidence WHERE facility_key='southbow_cushing' ORDER BY candidate_id"""):
        ids = possible.split("|")
        dates = sorted(specs[tank][6] for tank in ids)
        yield grouped_specs(identity, "southbow_cushing", cohort, f"permit_cohort_{evidence}", ids, specs,
                            "data/raw/documents/source_pdfs/6816.pdf", start=dates[0], end=dates[-1], precision="source_text")


def wildhorse_rows(db):
    equipment = {tank: spec for tank, spec in raw_tank_specs(db, "8910").items() if spec[0] == "EFR"}
    if not equipment:
        return
    mapped = list(db.execute("""SELECT DISTINCT r.candidate_id,l.diameter_m,l.fit_status
        FROM tank_terminal_research_attribution r JOIN tank_lidar_outline l USING (candidate_id)
        WHERE r.terminal_id='wildhorse' ORDER BY l.diameter_m"""))
    if not mapped:
        return
    groups = {diameter: [tank for tank, spec in equipment.items() if spec[1] == diameter]
              for diameter in sorted({spec[1] for spec in equipment.values()})}
    if len(mapped) != len(equipment) or any(status != "well_supported_circular_fit" for _, _, status in mapped):
        raise ValueError("Wildhorse contours and documented EFR groups require reconciliation")
    offset = 0
    for diameter, ids in groups.items():
        group = mapped[offset:offset + len(ids)]
        first, last = group[0][1], group[-1][1]
        if last - first > 1 or (offset and first - mapped[offset-1][1] < 5):
            raise ValueError("Wildhorse LiDAR diameter groups overlap")
        dates = sorted(dt.datetime.strptime(equipment[tank][6], "%m/%d/%Y").date().isoformat() for tank in ids)
        for identity, _, _ in group:
            yield grouped_specs(identity, "wildhorse", f"wildhorse_efr_{int(diameter)}ft",
                                "as_built_permit_group_matched_by_site_and_lidar_diameter_rank", ids, equipment,
                                "data/raw/documents/source_pdfs/8910.pdf", start=dates[0], end=dates[-1], precision="day")
        offset += len(ids)


def magellan_south_rows(db):
    equipment = raw_tank_specs(db, "8118")
    if not equipment:
        return
    contours = list(db.execute("""SELECT DISTINCT e.candidate_id,t.first_seen
        FROM tank_facility_evidence e JOIN tank_engineering_registry t USING(candidate_id)
        WHERE e.facility_key='magellan_cushing_south'
        ORDER BY e.candidate_id"""))
    if not contours:
        return
    if len(equipment) != 17 or len(contours) != len(equipment):
        raise ValueError("Magellan Cushing South parcel and 2021 equipment inventory require reconciliation")
    if any(spec[0] != "EFR" or spec[4] != 250_000 or spec[6] != "2011" for spec in equipment.values()):
        raise ValueError("Magellan Cushing South equipment inventory no longer describes one homogeneous group")
    document_date = next(iter(equipment.values()))[7]
    if any(not first_seen or first_seen > document_date for _, first_seen in contours):
        raise ValueError("Magellan Cushing South lacks dated aerial evidence for one or more contours")
    for identity, _ in contours:
        yield grouped_specs(identity, "magellan_cushing", MAGELLAN_SOUTH_COHORT,
                            "draft_homogeneous_permit_roster_and_parcel_match", sorted(equipment), equipment,
                            "data/raw/documents/source_pdfs/8118.pdf", start="2011", end="2011", precision="year")


def magellan_central_rows(db):
    primary = raw_tank_specs(db, "8699")
    corroboration = raw_tank_specs(db, "8766")
    if not primary and not corroboration:
        return
    if set(primary) != set(corroboration) or any(
        (spec[0], spec[4], spec[6]) != (corroboration[tank][0], corroboration[tank][4], corroboration[tank][6])
        for tank, spec in primary.items()
    ):
        raise ValueError("Cushing Central memoranda do not agree on the equipment rosters")
    if len(primary) != 13 or primary.get("0011", (None,))[0] != "FR":
        raise ValueError("Cushing Central requires twelve floating roofs and one fixed roof in the roster")

    contours = list(db.execute("""SELECT t.candidate_id,t.lidar_upper_outline_diameter_m,
            count(a.observed_date) AS visual_dates,
            sum(a.top_observation IN ('inset_wall_shadow_visible','thin_or_uncertain_inset_shadow')) AS open_roof_observations,
            sum(a.top_observation='top_without_distinct_inset_shadow') AS covered_roof_observations
        FROM tank_engineering_registry t
        JOIN (SELECT DISTINCT candidate_id FROM tank_facility_evidence WHERE facility_key='magellan_cushing_central') p
            USING(candidate_id)
        LEFT JOIN tank_aerial_visual_observation a ON a.candidate_id=t.candidate_id AND a.image_source='USDA NAIP'
            AND a.observed_date IN ('2021-05-06','2023-06-15')
        GROUP BY t.candidate_id,t.lidar_upper_outline_diameter_m
        ORDER BY t.lidar_upper_outline_diameter_m,t.candidate_id"""))
    if len(contours) != 13 or any(not diameter or dates != 2 for _, diameter, dates, _, _ in contours):
        raise ValueError("Cushing Central requires two dated aerial reviews and a LiDAR outline for all thirteen contours")
    exposed = [(identity, diameter) for identity, diameter, _, open_seen, _ in contours if open_seen == 2]
    covered = [(identity, diameter) for identity, diameter, _, _, covered_seen in contours if covered_seen == 2]
    if len(exposed) != 9 or len(covered) != 4 or covered[0][1] >= min(diameter for _, diameter in exposed):
        raise ValueError("Cushing Central roof observations require review before assigning equipment groups")

    for roof, candidates in (("EFR", exposed), ("IFR", covered[1:])):
        capacities = sorted({spec[4] for spec in primary.values() if spec[0] == roof})
        offset = 0
        for capacity in capacities:
            equipment = sorted(tank for tank, spec in primary.items() if spec[0] == roof and spec[4] == capacity)
            group = candidates[offset:offset + len(equipment)]
            if len(group) != len(equipment) or group[-1][1] - group[0][1] > 1:
                raise ValueError("Cushing Central measured diameter groups do not match the permit capacity groups")
            if offset and group[0][1] - candidates[offset - 1][1] < 5:
                raise ValueError("Cushing Central measured diameter groups are not distinct")
            dates = sorted(primary[tank][6] for tank in equipment)
            for identity, _ in group:
                yield grouped_specs(identity, "magellan_cushing", f"{MAGELLAN_CENTRAL_PREFIX}{roof.lower()}_{int(capacity)}bbl",
                                    "corroborated_permit_group_parcel_aerial_roof_and_lidar_size", equipment, primary,
                                    "data/raw/documents/source_pdfs/8699.pdf", start=dates[0], end=dates[-1], precision="year")
            offset += len(equipment)
        if offset != len(candidates):
            raise ValueError("Cushing Central permit groups do not cover the reviewed floating-roof contours")


def stroud_rows(db):
    application = raw_tank_specs(db, "6184830")
    if not application:
        return
    earlier = raw_tank_specs(db, "9030")
    if set(earlier) != {"TANK1", "TANK2"} or set(application) != set(earlier):
        raise ValueError("Stroud permit and application equipment IDs do not agree")
    if any((spec[0], spec[4], spec[6]) != (earlier[tank][0], earlier[tank][4], earlier[tank][6])
           for tank, spec in application.items()):
        raise ValueError("Stroud roof types, capacities or installation dates do not agree")
    contours = list(db.execute("""SELECT candidate_id,lidar_upper_outline_diameter_m,first_seen
        FROM tank_engineering_registry WHERE candidate_id LIKE 'ortho-stroud-%' ORDER BY candidate_id"""))
    if len(contours) != len(application) or any(not diameter or not first_seen for _, diameter, first_seen in contours):
        raise ValueError("Stroud requires two observed contours with LiDAR outlines")
    months = sorted(dt.datetime.strptime(spec[6], "%m/%Y").strftime("%Y-%m") for spec in application.values())
    for identity, _, _ in contours:
        yield grouped_specs(identity, "plains_cushing_south_stroud", STROUD_COHORT,
                            "corroborated_permit_group_aerial_and_lidar_dimensions", sorted(application), application,
                            "data/raw/documents/source_pdfs/deq_plains_south_6184830.pdf",
                            start=months[0], end=months[-1], precision="month")


def plains_original_rows(db):
    primary = raw_tank_specs(db, "5618")
    corroboration = raw_tank_specs(db, "8795")
    if not primary and not corroboration:
        return
    contours = list(db.execute("""SELECT t.candidate_id,t.lidar_upper_outline_diameter_m,t.longitude,t.latitude,
            t.first_seen,t.lidar_fit_status
        FROM tank_engineering_registry t JOIN (
            SELECT DISTINCT candidate_id FROM tank_facility_evidence WHERE facility_key='plains_cushing' AND parcel_id=?
        ) p USING(candidate_id) ORDER BY t.lidar_upper_outline_diameter_m,t.candidate_id""", (PLAINS_ORIGINAL_PARCEL,)))
    original_count = sum(count for _, count, _, _ in PLAINS_ORIGINAL_DESIGNS)
    if len(contours) <= original_count or any(
        diameter is None or status != "well_supported_circular_fit"
        for _, diameter, _, _, _, status in contours[:original_count + 1]
    ):
        raise ValueError("Plains Cushing original parcel requires independently measured circular outlines")
    north = sorted(contours[:14], key=lambda row: row[2])
    south = contours[14:original_count]
    if (north[7][2] - north[6][2] < 0.001 or max(row[3] for row in south) >= min(row[3] for row in north)
        or any(first_seen != "2010-05-04" for _, _, _, _, first_seen, _ in north + south)):
        raise ValueError("Plains Cushing original parcel does not reproduce the documented three-cluster layout")

    offset = 0
    for capacity, count, diameter_ft, _ in PLAINS_ORIGINAL_DESIGNS:
        equipment, corroborated = (
            sorted(tank for tank, spec in specs.items() if (spec[0], spec[4], spec[6]) == ("EFR", capacity, "1993"))
            for specs in (primary, corroboration)
        )
        if len(equipment) != count or equipment != corroborated:
            raise ValueError(f"Plains Cushing original equipment rosters disagree for {capacity} bbl")
        end = offset + count
        group = contours[offset:end]
        first, last = group[0][1], group[-1][1]
        if last - first > 1 or contours[end][1] - last < 5 or abs((first + last) / 2 - diameter_ft * FEET_TO_METRES) > 2:
            raise ValueError(f"Plains Cushing measured diameter group does not match the {capacity} bbl design")
        for identity, _, _, _, _, _ in group:
            yield grouped_specs(identity, "plains_cushing", f"{PLAINS_ORIGINAL_PREFIX}{capacity}bbl",
                                "two_permit_rosters_matched_by_original_parcel_layout_and_lidar_size",
                                equipment, primary, "data/raw/documents/source_pdfs/5618.pdf",
                                start="1993", end="1993", precision="year")
        offset = end


def plains_expansion_rows(db):
    observations = list(db.execute("""SELECT a.candidate_id,t.diameter_m,t.fit_status
        FROM plains_cushing_2008_aerial_observation a JOIN tank_lidar_outline t USING(candidate_id)
        WHERE a.source_coverage_year=2008 AND a.structure_observation='aboveground_circular_shell_seen'
            AND a.original_cohort_id IS NULL ORDER BY t.diameter_m,a.candidate_id"""))
    if not observations:
        return
    primary = raw_tank_specs(db, "5618")
    corroboration = raw_tank_specs(db, "8795")
    if len(observations) != sum(count for _, count, *_ in PLAINS_EXPANSIONS):
        raise ValueError("Plains 2008 aerial coverage requires reconciliation before matching expansion groups")
    for phase, count, capacity, _, first_year, last_year, diameter_min, diameter_max, _ in PLAINS_EXPANSIONS:
        ids, corroborated = (
            sorted(tank for tank, spec in specs.items()
                   if spec[0] == "EFR" and spec[4] == capacity and first_year <= (spec[6] or "") <= last_year)
            for specs in (primary, corroboration)
        )
        if len(ids) != count or ids != corroborated or any(primary[tank][6] != corroboration[tank][6] for tank in ids):
            raise ValueError(f"Plains Cushing expansion equipment rosters disagree for Phase {phase}")
        contours = [identity for identity, diameter, status in observations
                    if status == "well_supported_circular_fit" and diameter_min <= diameter <= diameter_max]
        if len(contours) != count:
            raise ValueError(f"Plains Cushing 2008 outlines do not reproduce Phase {phase} equipment count")
        for identity in contours:
            yield grouped_specs(identity, "plains_cushing", f"{PLAINS_EXPANSION_PREFIX}{phase}",
                                "two_permit_rosters_and_issuer_count_matched_by_2008_shell_and_lidar_size",
                                ids, primary, "data/raw/documents/source_pdfs/5618.pdf",
                                start=first_year, end=last_year, precision="year")


def plains_expansion_reference_rows():
    earlier_file = str(PLAINS_EXPANSION_FILE.relative_to(ROOT)) if PLAINS_EXPANSION_FILE.is_file() else None
    report_file = str(PLAINS_AS_BUILT_FILE.relative_to(ROOT)) if PLAINS_AS_BUILT_FILE.is_file() else None
    for phase, count, _, earlier_capacity, _, _, _, _, basis in PLAINS_EXPANSIONS:
        yield (f"{PLAINS_EXPANSION_PREFIX}{phase}", count, earlier_capacity, basis, "2006-09-19",
               PLAINS_EXPANSION_SOURCE, earlier_file)
        reported_capacity = {"i_v": 270_000, "vi": 570_000}[phase]
        yield (f"{PLAINS_EXPANSION_PREFIX}{phase}", count, reported_capacity, "issuer_as_built_tank_capacity",
               "2008-02-29", PLAINS_AS_BUILT_SOURCE, report_file)


def north_farm_rows(db):
    grandstaff = raw_tank_specs(db, "8122")
    grandstaff_crosscheck = raw_tank_specs(db, "8303")
    rose_rock = raw_tank_specs(db, "6844")
    if not (grandstaff or grandstaff_crosscheck or rose_rock):
        return
    if len(grandstaff) != 41 or grandstaff.keys() != grandstaff_crosscheck.keys() or any(
        (row[0], row[4], row[6]) != (grandstaff_crosscheck[tank][0], grandstaff_crosscheck[tank][4], grandstaff_crosscheck[tank][6])
        for tank, row in grandstaff.items()
    ):
        raise ValueError("Grandstaff permit rosters disagree on equipment, roof type, capacity or installation year")
    if len(rose_rock) != 35:
        raise ValueError("Rose Rock permit requires 31 crude storage tanks and four ancillary tanks")

    contours = list(db.execute("""SELECT t.candidate_id,p.terminal_id,p.farm_block,t.lidar_upper_outline_diameter_m,
            t.footprint_diameter_m,t.first_seen
        FROM tank_engineering_registry t JOIN north_farm_parcel_evidence p USING(candidate_id) ORDER BY t.candidate_id"""))
    for terminal, block, prefix, document, specs, groups in (
        ("enbridge_blueknight", "central", GRANDSTAFF_PREFIX, "8303", grandstaff_crosscheck, (
            ("central_2004_2006", 13, 250_000, 60, 63, "2004", "2006", "2010-05-04", "2010-05-04"),
            ("central_2012", 4, 250_000, 57, 58, "2012", "2012", "2013-06-11", "2013-06-11"),
        )),
        ("enbridge_blueknight", "north", GRANDSTAFF_PREFIX, "8303", grandstaff_crosscheck, (
            ("north_100000bbl", 3, 100_000, 36, 41, "2002", "2006", "2010-05-04", "2010-05-04"),
            ("north_200000bbl", 3, 200_000, 54, 56, "2002", "2002", "2010-05-04", "2010-05-04"),
            ("north_250000bbl", 8, 250_000, 60, 63, "2007", "2007", "2010-05-04", "2010-05-04"),
        )),
        ("semgroup_cushing", "north", ROSE_ROCK_PREFIX, "6844", rose_rock, (
            ("100000bbl_2009", 1, 100_000, 37, 41, "2009", "2009", "2010-05-04", "2010-05-04"),
            ("100000bbl_2011_2012", 4, 100_000, 37, 41, "2011", "2012", "2013-06-11", "2013-06-11"),
            ("250000bbl", 20, 250_000, 60, 63, "2009", "2012", "2010-05-04", "2013-06-11"),
            ("350000bbl", 6, 350_000, 72, 74, "2009", "2009", "2010-05-04", "2010-05-04"),
        )),
    ):
        for name, count, capacity, diameter_min, diameter_max, year_min, year_max, aerial_min, aerial_max in groups:
            ids = sorted(tank for tank, row in specs.items()
                         if row[0] == "EFR" and row[4] == capacity and year_min <= (row[6] or "") <= year_max)
            matched = [(identity, diameter) for identity, owner, farm, diameter, footprint, first_seen in contours
                       if owner == terminal and farm == block and aerial_min <= (first_seen or "") <= aerial_max
                       and diameter_min <= (diameter or footprint or 0) <= diameter_max]
            if len(ids) != count or len(matched) != count:
                raise ValueError(f"North farm group {name} has {len(ids)} documented IDs and {len(matched)} contours, expected {count}")
            for identity, diameter in matched:
                basis = "permit_group_parcel_aerial_chronology_and_lidar_outline" if diameter else "permit_group_parcel_aerial_chronology_and_footprint"
                yield grouped_specs(identity, terminal, prefix + name, basis, ids, specs,
                                    f"data/raw/documents/source_pdfs/{document}.pdf", start=year_min, end=year_max, precision="year")


def add_enbridge_main_covered_roof_outline(db):
    footprints = list(db.execute("""SELECT g.* FROM tank_footprint_geometry g
        JOIN enbridge_main_contour_group m USING(candidate_id) JOIN tank_lidar_outline l USING(candidate_id)
        WHERE m.cohort_id LIKE 'enbridge_main_south_ifr_%' AND l.diameter_m IS NULL"""))
    if not footprints:
        return
    directory = RAW / "usgs" / "ept" / CURRENT_LIDAR_DATASET / "nodes"
    for footprint in footprints:
        candidate = footprint[0]
        p10, p50, ground = db.execute("""SELECT interior_z_p10,interior_z_p50,ground_z_p50
            FROM lidar_tank_observation WHERE candidate_id=? AND dataset=? ORDER BY interior_points DESC LIMIT 1""",
            (candidate, CURRENT_LIDAR_DATASET)).fetchone()
        if p10 - ground < 10:
            raise ValueError(f"Elevated returns do not support a covered roof outline for {candidate}")
        median_fit, lower_fit = (next(outline_rows([footprint], {candidate: (height, ground)}, directory))
                                 for height in (p50, p10))
        if (median_fit[15] != "well_supported_circular_fit" or lower_fit[15] != "well_supported_circular_fit"
            or abs(median_fit[10] - lower_fit[10]) > 0.25):
            raise ValueError(f"Covered roof outline is sensitive to the elevation prior for {candidate}")
        db.execute("INSERT INTO enbridge_main_covered_roof_outline VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (*median_fit, p10, lower_fit[10]))


def enbridge_main_rows(db):
    contours = list(db.execute("""SELECT m.candidate_id,m.cohort_id,m.parcel_id,t.first_seen,
        t.lidar_upper_outline_diameter_m,l.lidar_covered_surface_outline_diameter_m,l.lidar_2011_rim_above_ground_m,
        l.matched_parcel_fraction,
        (SELECT count(DISTINCT observed_date) FROM tank_aerial_visual_observation a WHERE a.candidate_id=m.candidate_id
         AND observed_date IN ('2021-05-06','2023-06-15') AND top_observation='top_without_distinct_inset_shadow')
        FROM enbridge_main_contour_group m JOIN tank_engineering_registry t USING(candidate_id)
        JOIN enbridge_main_contour_reconciliation l USING(candidate_id)"""))
    if not contours:
        return
    primary, recent = (raw_tank_specs(db, doc) for doc in ("3329", "8499"))
    old_aerial, recent_aerial = "2010-05-04", "2013-06-11"
    # Group names retain the public project or enclosure number where the source gives one.
    groups = (
        ("west_355000_area_2", range(2238, 2243), (67, 68), ("2007", "2008"), old_aerial, None, "EFR"),
        ("west_355000_area_3", range(2235, 2238), (67, 68), ("2006", "2006"), old_aerial, None, "EFR"),
        ("west_355000_area_4", range(2229, 2235), (67, 68), ("2007", "2007"), old_aerial, None, "EFR"),
        ("west_365000_area_3", range(2243, 2245), (68, 69), ("2011", "2011"), recent_aerial, (15, 20), "EFR"),
        ("south_575000_area_5", range(3361, 3365), (84, 86), (None, None), old_aerial, (15, 20), "EFR"),
        ("south_400000_m7", range(3014, 3017), (71, 72), ("2012", "2012"), recent_aerial, (15, 20), "EFR"),
        ("south_316905_m10", range(3017, 3021), (63, 65), ("2012", "2012"), recent_aerial, (-2, 5), "EFR"),
        ("north_310000_m8", (1040, 1041, 1045, 1046), (63, 64), ("2012", "2012"), recent_aerial, (15, 20), "EFR"),
        ("north_288167_m11", range(1048, 1051), (63, 64), ("2013", "2013"), recent_aerial, (-2, 12), "EFR"),
        ("south_ifr_190000_m12", (3009, 3011), (49, 50), ("1975", "1975"), old_aerial, None, "IFR"),
        ("south_ifr_250000_m12", (3010, 3012, 3013), (55, 57), ("1975", "1975"), old_aerial, None, "IFR"),
    )
    if len(contours) != sum(len(ids) for _, ids, *_ in groups):
        raise ValueError("Enbridge Cushing match inputs do not cover the expected equipment groups")
    for name, numbers, diameter_range, years, first_date, legacy_rim_range, roof in groups:
        cohort = ENBRIDGE_MAIN_PREFIX + name
        ids = [f"CG-{number}" for number in numbers]
        if any(tank not in primary or tank not in recent or primary[tank][0] != roof
               or (primary[tank][0], primary[tank][4]) != (recent[tank][0], recent[tank][4]) for tank in ids):
            raise ValueError(f"Enbridge Cushing permit rosters disagree for {cohort}")
        matching = [row for row in contours if row[1] == cohort]
        if len(matching) != len(ids):
            raise ValueError(f"Enbridge Cushing contour and equipment counts differ for {cohort}")
        for identity, _, _, first_seen, measured, covered_outline, legacy_rim, parcel_fraction, covered_dates in matching:
            diameter = measured if measured is not None else covered_outline
            minimum_fraction = 0.95 if identity == "osm-way-460925444" else 0.99
            if (first_seen != first_date or parcel_fraction is None or parcel_fraction < minimum_fraction
                or diameter is None or not diameter_range[0] <= diameter <= diameter_range[1]
                or (legacy_rim_range and (legacy_rim is None or not legacy_rim_range[0] <= legacy_rim <= legacy_rim_range[1]))
                or (roof == "IFR" and covered_dates != 2)):
                raise ValueError(f"Enbridge Cushing geometry or chronology does not corroborate {identity} in {cohort}")
            basis = "two_permit_rosters_parcel_group_aerial_and_lidar"
            if minimum_fraction < 0.99:
                basis = "two_permit_rosters_enclosure_and_majority_parcel_overlap"
            elif measured is None:
                basis = "two_permit_rosters_parcel_group_and_independent_covered_surface_outline"
            yield grouped_specs(identity, "enbridge_cushing", cohort, basis, ids, primary,
                                "data/raw/documents/source_pdfs/3329.pdf", start=years[0], end=years[1],
                                precision="year" if years[0] else "unresolved")


def ngl_legacy_rows(db):
    spec = db.execute("""SELECT reported_tank_count,capacity_bbl,source_url,source_file
        FROM terminal_disclosed_capacity WHERE disclosure_id='matrix_gavilon_as_built'""").fetchone()
    if spec is None:
        return
    project_count, project_capacity, source, path = spec
    small_count, small_capacity = 2, 50_000
    large_count = project_count - small_count
    large_capacity, remainder = divmod(project_capacity - small_count * small_capacity, large_count)
    if (project_count, large_capacity, remainder) != (18, 250_000, 0):
        raise ValueError("Matrix Gavilon as-built total does not reconcile with the two stated size classes")
    contours = list(db.execute("""SELECT candidate_id,first_seen_aerial_date,lidar_2011_rim_above_ground_m,
        lidar_upper_outline_diameter_m,aerial_top_2021,aerial_top_2023
        FROM ngl_cushing_contour_reconciliation WHERE parcel_group='ngl_older_parcel'"""))
    remaining = {small_capacity: small_count, large_capacity: 12}
    inset_evidence = {"inset_wall_shadow_visible", "thin_or_uncertain_inset_shadow"}
    for candidate, first_seen, rim, diameter, top_2021, top_2023 in contours:
        if (first_seen != "2013-06-11" or rim is None or not 13 <= rim <= 16
            or diameter is None or not {top_2021, top_2023} <= inset_evidence):
            raise ValueError(f"NGL legacy parcel, chronology or roof evidence is incomplete for {candidate}")
        if 30 <= diameter <= 33:
            capacity, basis = small_capacity, "contractor_nominal_tank_capacity"
        elif 59 <= diameter <= 63:
            capacity, basis = large_capacity, "contractor_nominal_tank_capacity_reconciled_from_total"
        else:
            raise ValueError(f"NGL legacy contour {candidate} is outside the corroborated diameter classes")
        remaining[capacity] -= 1
        yield (candidate, f"{NGL_LEGACY_PREFIX}{capacity}bbl", "ngl_cushing",
               "contractor_open_top_group_current_parcel_dated_aerial_and_lidar", None, "EFR",
               None, None, None, None, capacity, capacity, basis, None, None,
               None, NGL_LEGACY_DATE, "documented_by_lidar", None, source, path)
    if any(remaining.values()):
        raise ValueError(f"NGL legacy contour counts changed: {remaining}")


def corporate_rows(db):
    for identity, terminal, cohort, capacity, basis, date, build_year, page, source, path in db.execute("""
        SELECT candidate_id,terminal_id,cohort_id,capacity_bbl,capacity_basis,documented_by,
            documented_build_year,map_pdf_page,source_url,local_pdf FROM tank_asset_cohort
        WHERE capacity_bbl IS NOT NULL AND cohort_id<>'amid_five_2009'"""):
        year = str(build_year) if build_year is not None else None
        yield (identity, cohort, terminal, "documented_asset_cohort", None, None, None, None, None, None,
               capacity, capacity, basis, None, date, year, year, "year" if year else None, page, source, path)
    if not CP_LOCAL.is_file():
        return
    content = CP_LOCAL.read_text()
    if "175,000-barrel" not in content or "700,000-barrel" not in content:
        raise ValueError("CP Energy capacity evidence changed")
    contours = list(db.execute("""SELECT DISTINCT candidate_id FROM tank_terminal_research_attribution
        WHERE terminal_id='cp_bpv_cushing' ORDER BY candidate_id"""))
    if len(contours) != 4:
        raise ValueError("CP Energy corporate tank count disagrees with the dated parcel association")
    for identity, in contours:
        yield (identity, "cp_bpv_four_2015", "cp_bpv_cushing", "issuer_four_tank_capacity_and_parcel_group",
               None, None, None, None, None, None, 175000, 175000, "issuer_tank_capacity_unspecified", None,
               None, "2015", "2015", "year", None, CP_SOURCE, str(CP_LOCAL.relative_to(ROOT)))


def add_tank_geometry_registry(db):
    add_enbridge_main_covered_roof_outline(db)
    plains_original = list(plains_original_rows(db))
    plains_expansions = list(plains_expansion_rows(db))
    if plains_original:
        db.executemany("INSERT INTO plains_original_design_reference VALUES (?,?,?,?,?,?,?,?,?,?)", (
            (f"{PLAINS_ORIGINAL_PREFIX}{capacity}bbl", capacity, count, diameter, height, "1994-08-15", "Allen H. Smith",
             "Willbros Butler Engineers Inc.", PLAINS_DESIGN_SOURCE, str(PLAINS_DESIGN_FILE.relative_to(ROOT)))
            for capacity, count, diameter, height in PLAINS_ORIGINAL_DESIGNS
        ))
    if plains_expansions:
        db.executemany("INSERT INTO plains_expansion_capacity_reference VALUES (?,?,?,?,?,?,?)", plains_expansion_reference_rows())
    db.executemany("INSERT INTO tank_engineering_cohort VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (*southbow_rows(db), *wildhorse_rows(db), *magellan_south_rows(db), *magellan_central_rows(db),
                    *stroud_rows(db), *plains_original, *plains_expansions, *north_farm_rows(db), *enbridge_main_rows(db),
                    *ngl_legacy_rows(db), *corporate_rows(db), *cushing_connect_rows(db), *phillips_nei_rows(db),
                    *enterprise_and_sunoco_rows(db)))
