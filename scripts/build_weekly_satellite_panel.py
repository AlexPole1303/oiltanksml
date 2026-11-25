"""Index exact satellite inputs and tank-level screening for each first EIA vintage."""
import datetime as dt

from publication_clocks import utc

L1C_INVENTORY_VIEW = """
CREATE VIEW sentinel2_l1c_input_inventory AS
    WITH catalog AS (
        SELECT g.canonical_l2a_id,g.captured_at_utc,g.status AS gcs_status,a.status AS alternate_status,
            CASE WHEN g.all_required_objects_created_by_utc IS NOT NULL THEN 'gcs'
                 WHEN a.all_inputs_available_at_utc IS NOT NULL THEN 'aws' END AS archive,
            CASE WHEN g.all_required_objects_created_by_utc IS NOT NULL THEN g.l1c_product_name
                 ELSE a.l1c_product_name END AS version_id,
            coalesce(g.all_required_objects_created_by_utc,a.all_inputs_available_at_utc) AS archive_available_at_utc,
            CASE WHEN g.all_required_objects_created_by_utc IS NOT NULL THEN g.with_red_detector_mask_objects_created_by_utc
                 ELSE a.all_inputs_available_at_utc END AS geometry_available_at_utc,
            CASE WHEN g.all_required_objects_created_by_utc IS NOT NULL THEN g.gcs_product_url
                 ELSE a.aws_tile_info_url END AS archive_source_url
        FROM sentinel2_l1c_archive_availability g LEFT JOIN sentinel2_l1c_alternate_evidence a USING (canonical_l2a_id)
    ), folders AS (
        SELECT scene_item_id,count(*) AS local_folders_checked,
            count(CASE WHEN status='screened' THEN 1 END) AS local_folders_screened,
            count(CASE WHEN status='failed_local_read_or_validation' THEN 1 END) AS local_folders_failed
        FROM sentinel2_l1c_local_file_check GROUP BY scene_item_id
    ), flags AS (
        SELECT scene_item_id,l1c_product_name,source_directory,
            optical_touched_cells_expected>0 AND optical_cells_valid_in_all_bands=optical_touched_cells_expected
                AS complete_optical,
            opaque_roof_cells=0 AND cirrus_roof_cells=0 AND coalesce(snow_ice_roof_cells,0)=0 AS clear,
            red_detector_covered_roof_cells=optical_touched_cells_expected AND red_detector_ambiguous_roof_cells=0
                AND json_array_length(red_detector_with_view_options_json)=1 AND sun_zenith_deg IS NOT NULL
                AND sun_azimuth_grid_deg IS NOT NULL AS geometry
        FROM sentinel2_l1c_tank_quality
    ), quality AS (
        SELECT scene_item_id,l1c_product_name,min(source_directory) AS example_local_directory,
            count(*) AS screened_contours,count(CASE WHEN complete_optical THEN 1 END) AS complete_optical_contours,
            count(CASE WHEN complete_optical AND clear THEN 1 END) AS clear_optical_contours,
            count(CASE WHEN complete_optical AND clear AND geometry THEN 1 END) AS clear_geometry_contours
        FROM flags GROUP BY scene_item_id,l1c_product_name
    ), radiometry AS (
        SELECT version_id,count(CASE WHEN reflectance_scale IS NOT NULL AND reflectance_offset IS NOT NULL
            AND product_xml IS NOT NULL AND band IN ('B02','B03','B04','B08') THEN 1 END) AS calibrated_bands
        FROM sentinel2_band_radiometry WHERE level='L1C' GROUP BY version_id
    ), study AS (
        SELECT canonical_l2a_id,version_id,count(*) AS eligible_week_endpoints
        FROM eia_satellite_window_input WHERE level='L1C' GROUP BY canonical_l2a_id,version_id
    )
    SELECT c.*,substr(c.captured_at_utc,1,4) AS capture_year,q.example_local_directory,
        coalesce(f.local_folders_checked,0) AS local_folders_checked,
        coalesce(f.local_folders_screened,0) AS local_folders_screened,
        coalesce(f.local_folders_failed,0) AS local_folders_failed,
        coalesce(q.screened_contours,0) AS exact_version_screened_contours,
        coalesce(q.complete_optical_contours,0) AS complete_optical_contours,
        coalesce(q.clear_optical_contours,0) AS clear_optical_contours,
        coalesce(q.clear_geometry_contours,0) AS clear_geometry_contours,
        coalesce(r.calibrated_bands,0) AS calibrated_bands,
        coalesce(s.eligible_week_endpoints,0) AS eligible_week_endpoints,
        CASE WHEN c.archive IS NULL THEN 'unresolved_archive_version'
             WHEN q.screened_contours IS NULL THEN 'exact_version_without_local_screen'
             WHEN coalesce(r.calibrated_bands,0)<4 THEN 'incomplete_local_radiometry'
             ELSE 'local_exact_version_with_radiometry' END AS input_status
    FROM catalog c LEFT JOIN folders f ON f.scene_item_id=c.canonical_l2a_id
    LEFT JOIN quality q ON q.scene_item_id=c.canonical_l2a_id AND q.l1c_product_name=c.version_id
    LEFT JOIN radiometry r ON r.version_id=c.version_id
    LEFT JOIN study s ON s.canonical_l2a_id=c.canonical_l2a_id AND s.version_id=c.version_id;
"""

SCHEMA = """
CREATE TABLE sentinel2_l1c_tank_quality (
    candidate_id TEXT, scene_item_id TEXT, captured_at_utc TEXT, l1c_product_name TEXT,
    optical_touched_cells_expected INTEGER, optical_cells_valid_in_all_bands INTEGER,
    cloud_mask_basis TEXT, opaque_roof_cells INTEGER, cirrus_roof_cells INTEGER, snow_ice_roof_cells INTEGER,
    context_valid_cells INTEGER, context_opaque_cells INTEGER, context_cirrus_cells INTEGER,
    context_snow_ice_cells INTEGER, red_detector_mask_basis TEXT, red_detector_covered_roof_cells INTEGER,
    red_detector_ambiguous_roof_cells INTEGER, red_detector_center_ids_json TEXT,
    red_detector_with_view_options_json TEXT, sun_zenith_deg REAL, sun_azimuth_grid_deg REAL,
    source_directory TEXT, tile_metadata_path TEXT, PRIMARY KEY (scene_item_id, candidate_id));
CREATE INDEX sentinel2_quality_scene_idx ON sentinel2_tank_scene_quality (scene_item_id);
CREATE TABLE sentinel2_l1c_local_file_check (
    scene_item_id TEXT, capture_year TEXT, source_directory TEXT PRIMARY KEY,
    status TEXT, tank_rows INTEGER, detail TEXT);
CREATE TABLE sentinel2_l1c_alternate_evidence (
    canonical_l2a_id TEXT PRIMARY KEY, captured_at_utc TEXT, earth_search_l1c_id TEXT, l1c_product_name TEXT,
    earth_search_created_at_utc TEXT, cdse_published_at_utc TEXT, status TEXT, all_inputs_available_at_utc TEXT,
    local_directory TEXT, aws_tile_info_url TEXT, cdse_product_url TEXT, source_file TEXT);
CREATE VIEW sentinel2_l1c_tandem_pair AS
    WITH archives AS (
        SELECT canonical_l2a_id,l1c_product_name AS product,captured_at_utc,
            all_required_objects_created_by_utc AS available_at_utc,'gcs' AS archive
        FROM sentinel2_l1c_archive_availability WHERE all_required_objects_created_by_utc IS NOT NULL
        UNION ALL
        SELECT canonical_l2a_id,l1c_product_name,captured_at_utc,all_inputs_available_at_utc,'aws'
        FROM sentinel2_l1c_alternate_evidence WHERE all_inputs_available_at_utc IS NOT NULL
    ), tandem_period AS (
        SELECT *,substr(product,1,3) AS satellite,substr(product,34,4) AS orbit,substr(product,40,5) AS tile
        FROM archives WHERE substr(captured_at_utc,1,10) BETWEEN '2024-10-29' AND '2024-12-19'
    ), screened AS (
        SELECT scene_item_id,l1c_product_name,min(source_directory) AS source_directory
        FROM sentinel2_l1c_tank_quality WHERE substr(captured_at_utc,1,10) BETWEEN '2024-10-29' AND '2024-12-19'
        GROUP BY scene_item_id,l1c_product_name
    )
    SELECT substr(c.captured_at_utc,1,10) AS capture_day,c.orbit,c.tile,
        a.canonical_l2a_id AS s2a_scene_item_id,c.canonical_l2a_id AS s2c_scene_item_id,
        a.product AS s2a_product,c.product AS s2c_product,
        a.captured_at_utc AS s2a_captured_at_utc,c.captured_at_utc AS s2c_captured_at_utc,
        round((julianday(c.captured_at_utc)-julianday(a.captured_at_utc))*86400,3) AS s2c_minus_s2a_seconds,
        a.archive AS s2a_archive,c.archive AS s2c_archive,
        a.available_at_utc AS s2a_available_at_utc,c.available_at_utc AS s2c_available_at_utc,
        qa.source_directory AS s2a_source_directory,qc.source_directory AS s2c_source_directory
    FROM tandem_period a JOIN tandem_period c ON a.tile=c.tile AND a.orbit=c.orbit
        AND abs(julianday(c.captured_at_utc)-julianday(a.captured_at_utc))*86400<=30
    LEFT JOIN screened qa ON qa.scene_item_id=a.canonical_l2a_id AND qa.l1c_product_name=a.product
    LEFT JOIN screened qc ON qc.scene_item_id=c.canonical_l2a_id AND qc.l1c_product_name=c.product
    WHERE a.satellite='S2A' AND c.satellite='S2C'
    ORDER BY capture_day,c.tile;
CREATE TABLE eia_satellite_window (
    week_ending TEXT, endpoint TEXT, endpoint_week_ending TEXT, release_date TEXT, study_phase TEXT,
    target_reported_as TEXT, target_precision_kbbl INTEGER, publication_timing_basis TEXT,
    window_starts_after_utc TEXT, window_ends_by_utc TEXT, inputs_available_before_utc TEXT,
    PRIMARY KEY (week_ending, endpoint));
CREATE TABLE eia_satellite_window_input (
    week_ending TEXT, endpoint TEXT, level TEXT, version_id TEXT, canonical_l2a_id TEXT,
    captured_at_utc TEXT, archive_available_at_utc TEXT, geometry_available_at_utc TEXT, availability_basis TEXT,
    core_catalog INTEGER, local_exact_version_screened INTEGER,
    PRIMARY KEY (week_ending, endpoint, level, version_id));
CREATE TABLE eia_satellite_tank_screen (
    week_ending TEXT, endpoint TEXT, candidate_id TEXT,
    l1c_local_versions INTEGER, l2a_local_versions INTEGER,
    l1c_complete_optical INTEGER, l2a_complete_optical INTEGER,
    l1c_cloud_mask_clear INTEGER, l2a_scl_roof_clear INTEGER, l1c_resolved_red_view INTEGER,
    first_local_capture_at_utc TEXT, latest_prior_aerial_date TEXT, latest_prior_aerial_structure TEXT,
    PRIMARY KEY (week_ending, endpoint, candidate_id));
CREATE VIEW eia_weekly_satellite_readiness AS
    WITH inputs AS (
        SELECT week_ending,
            sum(endpoint='current' AND level='L1C') AS current_l1c_archive_products,
            sum(endpoint='previous' AND level='L1C') AS previous_l1c_archive_products,
            sum(endpoint='current' AND level='L1C' AND local_exact_version_screened) AS current_l1c_screened_products,
            sum(endpoint='previous' AND level='L1C' AND local_exact_version_screened) AS previous_l1c_screened_products,
            sum(endpoint='current' AND level='L2A') AS current_l2a_archive_versions,
            sum(endpoint='previous' AND level='L2A') AS previous_l2a_archive_versions,
            sum(endpoint='current' AND level='L2A' AND local_exact_version_screened) AS current_l2a_screened_versions,
            sum(endpoint='previous' AND level='L2A' AND local_exact_version_screened) AS previous_l2a_screened_versions
        FROM eia_satellite_window_input GROUP BY week_ending
    ), tanks AS (
        SELECT c.week_ending,
            sum(c.l1c_resolved_red_view AND p.l1c_resolved_red_view) AS paired_l1c_candidates,
            sum(c.l2a_scl_roof_clear AND p.l2a_scl_roof_clear) AS paired_l2a_candidates,
            sum((c.l1c_resolved_red_view AND p.l1c_resolved_red_view)
                 OR (c.l2a_scl_roof_clear AND p.l2a_scl_roof_clear)) AS paired_candidates_same_level,
            sum(((c.l1c_resolved_red_view AND p.l1c_resolved_red_view)
                  OR (c.l2a_scl_roof_clear AND p.l2a_scl_roof_clear))
                 AND c.latest_prior_aerial_structure='aboveground_tank_seen'
                 AND p.latest_prior_aerial_structure='aboveground_tank_seen') AS paired_with_prior_aerial_tank_seen
        FROM eia_satellite_tank_screen c JOIN eia_satellite_tank_screen p USING (week_ending, candidate_id)
        WHERE c.endpoint='current' AND p.endpoint='previous' GROUP BY c.week_ending
    )
    SELECT w.week_ending, w.release_date, w.study_phase, w.target_reported_as, w.target_precision_kbbl,
        w.publication_timing_basis, w.inputs_available_before_utc,
        coalesce(i.current_l1c_archive_products,0) AS current_l1c_archive_products,
        coalesce(i.previous_l1c_archive_products,0) AS previous_l1c_archive_products,
        coalesce(i.current_l1c_screened_products,0) AS current_l1c_screened_products,
        coalesce(i.previous_l1c_screened_products,0) AS previous_l1c_screened_products,
        coalesce(i.current_l2a_archive_versions,0) AS current_l2a_archive_versions,
        coalesce(i.previous_l2a_archive_versions,0) AS previous_l2a_archive_versions,
        coalesce(i.current_l2a_screened_versions,0) AS current_l2a_screened_versions,
        coalesce(i.previous_l2a_screened_versions,0) AS previous_l2a_screened_versions,
        coalesce(t.paired_l1c_candidates,0) AS paired_l1c_candidates,
        coalesce(t.paired_l2a_candidates,0) AS paired_l2a_candidates,
        coalesce(t.paired_candidates_same_level,0) AS paired_candidates_same_level,
        coalesce(t.paired_with_prior_aerial_tank_seen,0) AS paired_with_prior_aerial_tank_seen
    FROM eia_satellite_window w LEFT JOIN inputs i USING (week_ending) LEFT JOIN tanks t USING (week_ending)
    WHERE w.endpoint='current';
""" + L1C_INVENTORY_VIEW


def windows(db):
    for week, release, role, precision, timing, cutoff in db.execute("""
        WITH first_vintage AS (
            SELECT *, row_number() OVER (PARTITION BY week_ending ORDER BY release_date, reported_as) AS n
            FROM eia_cushing_as_published WHERE week_ending >= '2017-01-01'
        )
        SELECT v.week_ending,v.release_date,v.reported_as,v.precision_kbbl,
            e.timing_basis,e.conservative_prerelease_before_utc
        FROM first_vintage v JOIN eia_release_clock e USING (release_date) WHERE v.n=1
        ORDER BY v.week_ending"""):
        if week <= "2022-12-31":
            phase = "development"
        elif week <= "2023-12-31":
            phase = "validation"
        elif week <= "2026-10-31":
            phase = "holdout"
        else:
            phase = "after_holdout"
        date = dt.date.fromisoformat(week)
        for endpoint, days_before in (("current", 0), ("previous", 7)):
            end = date - dt.timedelta(days=days_before)
            start = end - dt.timedelta(days=7)
            yield (week, endpoint, end.isoformat(), release, phase, role, precision, timing,
                   utc(start, dt.time(7)), utc(end, dt.time(7)), cutoff)


def build_weekly_satellite_panel(db):
    db.executemany("INSERT INTO eia_satellite_window VALUES (?,?,?,?,?,?,?,?,?,?,?)", windows(db))
    db.execute("""INSERT INTO eia_satellite_window_input
        WITH versions AS (
            SELECT 'L1C' AS level, a.l1c_product_name AS version_id, a.canonical_l2a_id,
                a.captured_at_utc, a.all_required_objects_created_by_utc AS available,
                a.with_red_detector_mask_objects_created_by_utc AS geometry_available,
                'gcs_exact_object_creation_times' AS basis,
                EXISTS (SELECT 1 FROM sentinel2_l1c_tank_quality q
                    WHERE q.scene_item_id=a.canonical_l2a_id AND q.l1c_product_name=a.l1c_product_name) AS local
            FROM sentinel2_l1c_archive_availability a WHERE a.all_required_objects_created_by_utc IS NOT NULL
            UNION ALL
            SELECT 'L1C', a.l1c_product_name, a.canonical_l2a_id, a.captured_at_utc, a.all_inputs_available_at_utc,
                a.all_inputs_available_at_utc, 'aws_exact_objects_and_earth_search_catalog',
                EXISTS (SELECT 1 FROM sentinel2_l1c_tank_quality q
                    WHERE q.scene_item_id=a.canonical_l2a_id AND q.l1c_product_name=a.l1c_product_name)
            FROM sentinel2_l1c_alternate_evidence a WHERE a.all_inputs_available_at_utc IS NOT NULL
            UNION ALL
            SELECT 'L2A', a.item_id, a.canonical_item_id, a.captured_at_utc, a.catalog_route_conservative_from_utc,
                a.catalog_route_conservative_from_utc, 'aws_exact_objects_and_earth_search_catalog',
                EXISTS (SELECT 1 FROM sentinel2_tank_scene_quality q WHERE q.scene_item_id=a.item_id)
            FROM sentinel2_archive_availability a WHERE a.catalog_route_conservative_from_utc IS NOT NULL
        ), core AS (SELECT DISTINCT item_id FROM satellite_scene WHERE canonical_sentinel2=1)
        SELECT w.week_ending,w.endpoint,v.level,v.version_id,v.canonical_l2a_id,v.captured_at_utc,v.available,
            v.geometry_available,v.basis,s.item_id IS NOT NULL,v.local
        FROM eia_satellite_window w JOIN versions v
            ON julianday(v.captured_at_utc)>julianday(w.window_starts_after_utc)
            AND julianday(v.captured_at_utc)<=julianday(w.window_ends_by_utc)
            AND julianday(v.available)<julianday(w.inputs_available_before_utc)
        LEFT JOIN core s ON s.item_id=v.canonical_l2a_id""")
    db.execute("""INSERT INTO eia_satellite_tank_screen
        WITH l1c AS (
            SELECT i.week_ending,i.endpoint,q.candidate_id,i.captured_at_utc,
                q.optical_cells_valid_in_all_bands=q.optical_touched_cells_expected AS optical,
                q.opaque_roof_cells=0 AND q.cirrus_roof_cells=0 AND coalesce(q.snow_ice_roof_cells,0)=0 AS clear,
                q.red_detector_covered_roof_cells=q.optical_touched_cells_expected
                    AND q.red_detector_ambiguous_roof_cells=0
                    AND json_array_length(q.red_detector_with_view_options_json)=1
                    AND julianday(i.geometry_available_at_utc)<julianday(w.inputs_available_before_utc) AS view
            FROM eia_satellite_window_input i JOIN eia_satellite_window w USING (week_ending,endpoint)
            JOIN sentinel2_l1c_tank_quality q ON q.scene_item_id=i.canonical_l2a_id AND q.l1c_product_name=i.version_id
            WHERE i.level='L1C' AND i.local_exact_version_screened
        ), l2a AS (
            SELECT i.week_ending,i.endpoint,q.candidate_id,i.captured_at_utc,
                q.optical_cells_valid_in_all_bands=q.optical_touched_cells_expected AS optical,
                q.scl_touched_cells_expected>0
                    AND q.scl_touched_cells_expected=q.scl_class_2+q.scl_class_4+q.scl_class_5+q.scl_class_6+q.scl_class_7 AS clear
            FROM eia_satellite_window_input i JOIN sentinel2_tank_scene_quality q ON q.scene_item_id=i.version_id
            WHERE i.level='L2A' AND i.local_exact_version_screened
        ), flags AS (
            SELECT week_ending,endpoint,candidate_id,1 AS l1c,0 AS l2a,optical AS l1c_optical,0 AS l2a_optical,
                optical AND coalesce(clear,0) AS l1c_clear,0 AS l2a_clear,
                optical AND coalesce(clear,0) AND coalesce(view,0) AS l1c_view,captured_at_utc FROM l1c
            UNION ALL
            SELECT week_ending,endpoint,candidate_id,0,1,0,optical,0,optical AND clear,0,captured_at_utc FROM l2a
        ), tank_windows AS (
            SELECT week_ending,endpoint,candidate_id,sum(l1c) AS l1c_versions,sum(l2a) AS l2a_versions,
                max(l1c_optical) AS l1c_optical,max(l2a_optical) AS l2a_optical,max(l1c_clear) AS l1c_clear,
                max(l2a_clear) AS l2a_clear,max(l1c_view) AS l1c_view,min(captured_at_utc) AS first_capture
            FROM flags GROUP BY week_ending,endpoint,candidate_id
        )
        SELECT t.*, (
            SELECT max(observed_date) FROM tank_aerial_visual_observation a
            WHERE a.candidate_id=t.candidate_id AND a.observed_date<substr(t.first_capture,1,10)
        ), (
            SELECT CASE WHEN count(DISTINCT structure_observation)>1 THEN 'conflicting_observations'
                        ELSE min(structure_observation) END
            FROM tank_aerial_visual_observation a WHERE a.candidate_id=t.candidate_id
              AND a.observed_date=(SELECT max(observed_date) FROM tank_aerial_visual_observation b
                   WHERE b.candidate_id=t.candidate_id AND b.observed_date<substr(t.first_capture,1,10))
        )
        FROM tank_windows t""")
