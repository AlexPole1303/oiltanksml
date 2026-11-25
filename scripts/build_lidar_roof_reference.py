"""Extract centre and rim elevation references for tanks with documented roof types."""
import argparse
import datetime as dt
import json
import sqlite3
from contextlib import closing

import numpy as np

from audit_lidar_tanks import DATASET, GPS_EPOCH, LEGACY_DATASET, SURVEYS, as_utc
from build_lidar_radial_profiles import RADII, ZONES, Circle, reference_circles, select_returns
from build_nei_roof_evidence import NEI_ROOF_SCHEMA
from build_tank_geometry_registry import (
    ENBRIDGE_MAIN_PREFIX, GRANDSTAFF_PREFIX, MAGELLAN_CENTRAL_PREFIX, MAGELLAN_SOUTH_COHORT, NGL_LEGACY_PREFIX,
    PLAINS_EXPANSION_PREFIX, PLAINS_ORIGINAL_PREFIX, ROSE_ROCK_PREFIX,
)
from collect_public import RAW, ROOT, write_csv
from publication_clocks import export

DERIVED = ROOT / "data" / "derived"
REFERENCE_CSV = DERIVED / "lidar_roof_elevation_reference.csv"
SCHEMA = """
CREATE TABLE lidar_roof_elevation_reference (
    candidate_id TEXT,dataset TEXT,acquisition_date_utc TEXT,point_source_id INTEGER,geometry_basis TEXT,outline_fit_status TEXT,
    center_east_m REAL,center_north_m REAL,radius_m REAL,center_radius_fraction_upper REAL,
    rim_radius_fraction_lower REAL,rim_radius_fraction_upper REAL,first_return_at_utc TEXT,last_return_at_utc TEXT,
    center_nonground_returns INTEGER,center_sectors INTEGER,rim_nonground_returns INTEGER,rim_sectors INTEGER,
    center_z_p50_native REAL,center_z_p90_minus_p10_native REAL,center_plane_z_native REAL,
    center_plane_slope_east_native_per_m REAL,center_plane_slope_north_native_per_m REAL,
    center_plane_residual_p90_minus_p10_native REAL,rim_pooled_z_p95_native REAL,rim_sector_p95_median_native REAL,
    rim_sector_p95_range_native REAL,rim_minus_center_plane_native_z REAL,source_directory TEXT,
    source_node_ids_json TEXT,PRIMARY KEY (candidate_id,dataset,acquisition_date_utc,point_source_id));
CREATE VIEW lidar_roof_optical_reference AS
    SELECT r.candidate_id,r.dataset,r.acquisition_date_utc,r.point_source_id,r.geometry_basis,r.center_plane_z_native,
        r.rim_sector_p95_median_native,r.rim_minus_center_plane_native_z,r.center_sectors,r.rim_sectors,
        r.first_return_at_utc,r.last_return_at_utc,t.documented_roof_type,t.cohort_id,t.source_document_date,
        o.level,o.scene_item_id,o.version_id,o.optical_captured_at_utc,
        CASE WHEN julianday(o.optical_captured_at_utc)<julianday(r.first_return_at_utc) THEN 'before_lidar'
            WHEN julianday(o.optical_captured_at_utc)>julianday(r.last_return_at_utc) THEN 'after_lidar'
            ELSE 'within_lidar_return_interval' END AS optical_time_relation,
        round(24*(julianday(o.optical_captured_at_utc)-julianday(r.last_return_at_utc)),3) AS optical_minus_lidar_end_hours,
        round(24*(julianday(o.optical_captured_at_utc)-julianday(r.first_return_at_utc)),3) AS optical_minus_lidar_start_hours,
        o.capture_period,o.optical_cells,o.valid_cells,o.roof_clear_by_source_mask,o.resolved_b04_detector,
        o.sun_zenith_deg,o.sun_azimuth_grid_deg,o.red_view_options_json,o.red_radiometry_status,
        o.source_directory AS optical_source_directory,o.tile_metadata_path,r.source_node_ids_json
    FROM lidar_roof_elevation_reference r JOIN lidar_optical_calibration_pair o
        ON o.candidate_id=r.candidate_id AND o.lidar_dataset=r.dataset
        AND o.lidar_acquisition_date_utc=r.acquisition_date_utc AND o.lidar_point_source_id=r.point_source_id
    JOIN tank_engineering_registry t ON t.candidate_id=r.candidate_id AND t.documented_roof_type IS NOT NULL;
"""
COVERAGE_VIEW = """
CREATE VIEW roof_physics_input_coverage AS
    WITH aerial AS (
        SELECT t.candidate_id,
            count(DISTINCT a.acquisition_year) AS dated_naip_years_through_2022,
            max(a.source_acquisition_date) AS latest_dated_naip_through_2022
        FROM tank_engineering_registry t JOIN naip_tank_source_clock a
            ON a.candidate_id=t.candidate_id AND a.source_acquisition_date>=t.first_seen
        WHERE a.source_acquisition_date<'2023-01-01' AND a.contour_fraction>=0.999999
            AND json_array_length(a.dated_asset_paths_json)>0
        GROUP BY t.candidate_id
    ), visual AS (
        SELECT candidate_id,
            max(CASE WHEN observed_date<'2022-01-01' THEN top_observation END) AS aerial_top_2021,
            max(CASE WHEN observed_date>='2023-01-01' THEN top_observation END) AS aerial_top_2023
        FROM tank_aerial_visual_observation WHERE image_source='USDA NAIP'
            AND (observed_date LIKE '2021-%' OR observed_date LIKE '2023-%')
            AND top_observation<>'not_assessed'
        GROUP BY candidate_id
    ), lidar AS (
        SELECT candidate_id,count(*) AS lidar_flight_lines
        FROM lidar_tank_observation WHERE dataset='OK_Statewide_2_D22' GROUP BY candidate_id
    ), reference AS (
        SELECT candidate_id,
            count(CASE WHEN dataset='OK_Statewide_2_D22' THEN 1 END) AS reference_flight_lines,
            count(CASE WHEN dataset='OK_Statewide_2_D22' THEN rim_minus_center_plane_native_z END) AS numeric_references,
            count(CASE WHEN dataset='USGS_LPC_OK_Area1_1A_2011_LAS_2016' THEN 1 END) AS legacy_reference_flight_lines,
            count(CASE WHEN dataset='USGS_LPC_OK_Area1_1A_2011_LAS_2016' THEN rim_minus_center_plane_native_z END)
                AS legacy_numeric_references
        FROM lidar_roof_elevation_reference GROUP BY candidate_id
    ), optical AS (
        SELECT candidate_id,
            count(DISTINCT CASE WHEN optical_captured_at_utc<'2023-01-01' THEN version_id END)
                AS clear_l1c_versions_2022,
            count(DISTINCT CASE WHEN optical_captured_at_utc>='2023-01-01' THEN version_id END)
                AS clear_l1c_versions_2023
        FROM lidar_optical_calibration_pair WHERE level='L1C' AND optical_cells>0 AND optical_cells=valid_cells
            AND roof_clear_by_source_mask AND resolved_b04_detector GROUP BY candidate_id
    )
    SELECT t.candidate_id,t.outline_source,t.first_seen,t.cohort_id,t.documented_roof_type,t.source_document_date,
        t.engineering_source_url,t.latest_aerial_date,t.latest_aerial_top_observation,t.footprint_diameter_m,
        t.lidar_upper_outline_diameter_m,t.lidar_fit_status,t.permit_diameter_m,t.reported_height_min_m,t.reported_height_max_m,
        coalesce(a.dated_naip_years_through_2022,0) AS dated_naip_years_through_2022,a.latest_dated_naip_through_2022,
        v.aerial_top_2021,v.aerial_top_2023,
        coalesce(l.lidar_flight_lines,0) AS lidar_flight_lines,coalesce(r.reference_flight_lines,0) AS reference_flight_lines,
        coalesce(r.numeric_references,0) AS numeric_references,
        coalesce(r.legacy_reference_flight_lines,0) AS legacy_reference_flight_lines,
        coalesce(r.legacy_numeric_references,0) AS legacy_numeric_references,
        coalesce(o.clear_l1c_versions_2022,0) AS clear_l1c_versions_2022,
        coalesce(o.clear_l1c_versions_2023,0) AS clear_l1c_versions_2023
    FROM tank_engineering_registry t LEFT JOIN aerial a USING(candidate_id) LEFT JOIN visual v USING(candidate_id)
    LEFT JOIN lidar l USING(candidate_id)
    LEFT JOIN reference r USING(candidate_id) LEFT JOIN optical o USING(candidate_id);
"""

NEI_ROOF_DATES = """
        SELECT terminal_id,cohort_id,documented_roof_type,min(archive_version_date) AS earliest_nei_roof_archive_date
        FROM nei_roof_equipment_evidence
        WHERE group_mapping_basis IN ('equipment_label_and_roof_codes_in_archive','four_pre_expansion_efr_units_on_single_cvr_site',
                                      'ten_efr_units_on_enterprise_east_site')
        GROUP BY terminal_id,cohort_id,documented_roof_type
"""
WEEKLY_ROOF_COVERAGE_VIEW = f"""
CREATE VIEW eia_roof_cohort_weekly_coverage AS
    WITH nei_roof_dates AS ({NEI_ROOF_DATES}), cohorts AS (
        SELECT w.week_ending,w.release_date,w.study_phase,w.inputs_available_before_utc,t.cohort_id,t.documented_terminal_id,
            t.documented_roof_type,t.source_document_date,count(*) AS documented_contours,
            count(CASE WHEN t.first_seen<=w.week_ending THEN 1 END) AS contours_seen_by_week_end,
            count(CASE WHEN c.l1c_resolved_red_view AND p.l1c_resolved_red_view
                AND c.latest_prior_aerial_structure='aboveground_tank_seen'
                AND p.latest_prior_aerial_structure='aboveground_tank_seen' THEN 1 END) AS paired_l1c_prior_visible_contours,
            count(CASE WHEN c.l2a_scl_roof_clear AND p.l2a_scl_roof_clear
                AND c.latest_prior_aerial_structure='aboveground_tank_seen'
                AND p.latest_prior_aerial_structure='aboveground_tank_seen' THEN 1 END) AS paired_l2a_prior_visible_contours
        FROM eia_satellite_window w CROSS JOIN tank_engineering_registry t
        LEFT JOIN eia_satellite_tank_screen c ON c.week_ending=w.week_ending
            AND c.endpoint='current' AND c.candidate_id=t.candidate_id
        LEFT JOIN eia_satellite_tank_screen p ON p.week_ending=w.week_ending
            AND p.endpoint='previous' AND p.candidate_id=t.candidate_id
        WHERE w.endpoint='current' AND t.documented_roof_type IS NOT NULL
        GROUP BY w.week_ending,t.cohort_id,t.documented_terminal_id,t.documented_roof_type,t.source_document_date
    )
    SELECT c.*,c.source_document_date<substr(c.inputs_available_before_utc,1,10) AS source_document_dated_before_cutoff,
        n.earliest_nei_roof_archive_date,
        n.earliest_nei_roof_archive_date<substr(c.inputs_available_before_utc,1,10) AS nei_roof_archive_dated_before_cutoff,
        x.ticker,x.role AS terminal_relation_role,x.valid_from,x.valid_to_exclusive,x.date_precision,x.exposure_date_status,
        x.source_url AS exposure_source_url,
        CASE WHEN c.documented_roof_type='EFR' THEN 'external_roof_candidate'
             ELSE 'covered_surface_control' END AS roof_measurement_role
    FROM cohorts c LEFT JOIN nei_roof_dates n ON n.terminal_id=c.documented_terminal_id
        AND n.cohort_id=c.cohort_id AND n.documented_roof_type=c.documented_roof_type
    LEFT JOIN eia_event_operator_exposure x
        ON x.release_date=c.release_date AND x.terminal_id=c.documented_terminal_id;
"""
SCHEMA += NEI_ROOF_SCHEMA + COVERAGE_VIEW + WEEKLY_ROOF_COVERAGE_VIEW


def centre_summary(points):
    if not len(points):
        return (None,) * 6
    z = points[:, 5]
    p10, median, p90 = np.percentile(z, (10, 50, 90))
    design = np.column_stack((np.ones(len(points)), points[:, 8:10]))
    coefficients, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank < 3:
        return (median, p90 - p10, None, None, None, None)
    residual_p10, residual_p90 = np.percentile(z - design @ coefficients, (10, 90))
    return (median, p90 - p10, *coefficients, residual_p90 - residual_p10)


def rim_summary(points):
    if not len(points):
        return (None, None, None)
    z = points[:, 5]
    sectors = points[:, 3]
    sector_p95 = [np.percentile(z[sectors == sector], 95) for sector in np.unique(sectors)]
    return np.percentile(z, 95), np.median(sector_p95), max(sector_p95) - min(sector_p95)


def reference_rows(circles, survey, directory):
    node_paths = sorted(directory.glob("*.laz"))
    if not node_paths:
        raise ValueError(f"No local point files in {directory}")
    selected = select_returns(circles, survey, node_paths, include_xy=True)
    center_band, rim_band = ZONES.index("center"), ZONES.index("boundary")
    for index, groups in sorted(selected.items()):
        circle = circles[index]
        points = np.concatenate(groups)
        points = points[(points[:, 6] == 0) & np.isin(points[:, 2], (center_band, rim_band))]
        for day, line in np.unique(points[:, :2], axis=0):
            flight = points[(points[:, 0] == day) & (points[:, 1] == line)]
            center, rim = (flight[flight[:, 2] == band] for band in (center_band, rim_band))
            center_stats, rim_stats = centre_summary(center), rim_summary(rim)
            center_plane, rim_median = center_stats[2], rim_stats[1]
            depth = rim_median - center_plane if rim_median is not None and center_plane is not None else None
            stats = (*center_stats, *rim_stats, depth)
            start, end = (as_utc(value, survey.gps_to_utc_seconds).isoformat()
                          for value in (flight[:, 4].min(), flight[:, 4].max()))
            date = (GPS_EPOCH.date() + dt.timedelta(days=int(day))).isoformat()
            nodes = [node_paths[int(node)].stem for node in np.unique(flight[:, 7])]
            yield (circle.candidate, survey.dataset, date, int(line), circle.basis, circle.fit_status,
                   circle.east, circle.north, circle.radius,
                   RADII[center_band + 1], RADII[rim_band], RADII[rim_band + 1], start, end,
                   len(center), len(np.unique(center[:, 3])), len(rim), len(np.unique(rim[:, 3])),
                   *(round(value, 6) if value is not None else None for value in stats),
                   str(directory.relative_to(ROOT)), json.dumps(nodes))


def covered_surface_reference_rows(db):
    for candidate, status, east, north, diameter, directory in db.execute("""SELECT
        o.candidate_id,o.fit_status,o.center_east_m,o.center_north_m,o.diameter_m,o.source_directory
        FROM enbridge_main_covered_roof_outline o JOIN tank_engineering_cohort t USING(candidate_id)
        WHERE t.documented_roof_type='IFR'"""):
        circle = Circle(candidate, "lidar_covered_surface_circle", status, east, north, diameter / 2)
        yield from reference_rows([circle], SURVEYS[DATASET], ROOT / directory)


def local_reference_rows(db):
    groups = list(db.execute("""SELECT candidate_id,cohort_id,first_seen FROM tank_engineering_registry
        WHERE documented_roof_type IN ('EFR','Domed EFR','IFR')"""))
    documented = {candidate for candidate, _, _ in groups}
    legacy = {candidate for candidate, cohort, first_seen in groups
              if cohort in (MAGELLAN_SOUTH_COHORT, "former_cvr_efr_2011")
              or cohort.startswith((MAGELLAN_CENTRAL_PREFIX, PLAINS_ORIGINAL_PREFIX, PLAINS_EXPANSION_PREFIX,
                                    NGL_LEGACY_PREFIX))
              or (cohort.startswith((GRANDSTAFF_PREFIX, ROSE_ROCK_PREFIX, ENBRIDGE_MAIN_PREFIX, "phillips_", "enterprise_",
                                     "amid_five_", "sunoco_little_farm_"))
                  and first_seen and first_seen <= '2011-12-08')}
    rows = []
    for scope, circles in reference_circles().items():
        for dataset, candidates, basis in ((DATASET, documented, "lidar_elevated_circle"),
                                           (LEGACY_DATASET, legacy, "mapped_footprint_circle")):
            eligible = [circle for circle in circles if circle.basis == basis and circle.candidate in candidates]
            if eligible:
                rows.extend(reference_rows(eligible, SURVEYS[dataset], RAW / "usgs" / scope / dataset / "nodes"))
    if not rows:
        raise ValueError("No documented roof references from local LiDAR")
    rows.extend(covered_surface_reference_rows(db))
    return sorted(rows)


def add_lidar_roof_reference(db):
    placeholders = ",".join("?" for _ in db.execute("PRAGMA table_info(lidar_roof_elevation_reference)"))
    db.executemany(f"INSERT INTO lidar_roof_elevation_reference VALUES ({placeholders})", local_reference_rows(db))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh the SQLite references and their dependent CSVs.")
    args = parser.parse_args()
    database = DERIVED / "cushing.sqlite"
    with closing(sqlite3.connect(database.as_uri() + ("?mode=rw" if args.refresh else "?mode=ro"), uri=True)) as db:
        if args.refresh:
            db.execute("BEGIN IMMEDIATE")
        rows = local_reference_rows(db)
        fields = [row[1] for row in db.execute("PRAGMA table_info(lidar_roof_elevation_reference)")]
        if args.refresh:
            with db:
                db.execute("DELETE FROM lidar_roof_elevation_reference")
                placeholders = ",".join("?" for _ in fields)
                db.executemany(f"INSERT INTO lidar_roof_elevation_reference VALUES ({placeholders})", rows)
            export(db, "lidar_roof_optical_reference",
                   order_by="candidate_id,acquisition_date_utc,point_source_id,optical_captured_at_utc,level,version_id")
            export(db, "roof_physics_input_coverage", order_by="candidate_id")
        write_csv(REFERENCE_CSV, fields, rows)
        print(f"{len(rows)} flight references across {len({row[0] for row in rows})} documented roofs")


if __name__ == "__main__":
    main()
