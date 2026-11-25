"""Index the two optical passes bracketing the dated 2022 Cushing LiDAR survey."""
import json
import xml.etree.ElementTree as ET

import numpy as np

from build_sentinel2_angles import array, ground_coordinates, observation_rows, stencil
from collect_public import RAW, ROOT
from collect_s2_l1c import CHANNELS, source_metadata_bytes

CAPTURE_DAYS = ("2022-12-30", "2023-01-04")
SCHEMA = """
CREATE TABLE sentinel2_band_radiometry (
    level TEXT, version_id TEXT, product_name TEXT, captured_at_utc TEXT, processing_baseline TEXT,
    band TEXT, channel TEXT, quantification_value REAL, add_offset_dn REAL, reflectance_scale REAL,
    reflectance_offset REAL, nodata_dn INTEGER, saturated_dn INTEGER, stac_scale REAL, stac_offset REAL,
    metadata_status TEXT, product_xml TEXT, source_manifest TEXT, PRIMARY KEY (level,version_id,band));
CREATE TABLE lidar_calibration_capture_day (capture_day TEXT PRIMARY KEY);
CREATE TABLE lidar_calibration_l2a_geometry (
    scene_item_id TEXT, candidate_id TEXT, source_directory TEXT, tile_metadata_path TEXT,
    sun_zenith_deg REAL, sun_azimuth_grid_deg REAL, red_view_options_json TEXT,
    PRIMARY KEY (scene_item_id,candidate_id));
CREATE TABLE lidar_calibration_b04_angle_node (
    scene_item_id TEXT, version_id TEXT, candidate_id TEXT, detector_id INTEGER, grid_row INTEGER, grid_col INTEGER,
    node_role TEXT, stencil_weight REAL, source_zenith_deg REAL, source_azimuth_deg REAL, distance_to_center_m REAL,
    node_east_m REAL,node_north_m REAL,center_east_m REAL,center_north_m REAL,crs TEXT,tile_metadata_path TEXT,
    PRIMARY KEY (scene_item_id,version_id,candidate_id,detector_id,grid_row,grid_col));
CREATE VIEW lidar_optical_calibration_pair AS
    WITH optical AS (
        SELECT q.candidate_id,'L1C' AS level,q.scene_item_id,q.l1c_product_name AS version_id,
            q.captured_at_utc,q.source_directory,q.tile_metadata_path,q.optical_touched_cells_expected AS optical_cells,
            q.optical_cells_valid_in_all_bands AS valid_cells,q.cloud_mask_basis,
            q.opaque_roof_cells=0 AND q.cirrus_roof_cells=0 AND coalesce(q.snow_ice_roof_cells,0)=0 AS roof_clear,
            q.context_valid_cells AS context_cells,q.context_opaque_cells,q.context_cirrus_cells,q.context_snow_ice_cells,
            NULL AS context_scl_excluded_cells,q.red_detector_mask_basis,q.red_detector_covered_roof_cells,
            q.red_detector_ambiguous_roof_cells,q.sun_zenith_deg,q.sun_azimuth_grid_deg,
            q.red_detector_with_view_options_json AS red_view_options_json,
            q.red_detector_covered_roof_cells=q.optical_touched_cells_expected AND q.red_detector_ambiguous_roof_cells=0
                AND json_array_length(q.red_detector_with_view_options_json)=1 AS resolved_b04_detector
        FROM sentinel2_l1c_tank_quality q JOIN lidar_calibration_capture_day d ON substr(q.captured_at_utc,1,10)=d.capture_day
        UNION ALL
        SELECT q.candidate_id,'L2A',q.scene_item_id,q.scene_item_id,q.captured_at_utc,
            g.source_directory,g.tile_metadata_path,q.optical_touched_cells_expected,q.optical_cells_valid_in_all_bands,
            'l2a_scl',q.scl_touched_cells_expected>0
                AND q.scl_touched_cells_expected=q.scl_class_2+q.scl_class_4+q.scl_class_5+q.scl_class_6+q.scl_class_7,
            q.context_scl_cells,NULL,NULL,NULL,q.context_nodata_defect_cloud_shadow_snow_cells,
            NULL,NULL,NULL,g.sun_zenith_deg,g.sun_azimuth_grid_deg,g.red_view_options_json,NULL
        FROM sentinel2_tank_scene_quality q JOIN lidar_calibration_l2a_geometry g USING (scene_item_id,candidate_id)
    ), aerial AS (
        SELECT candidate_id,max(CASE WHEN observed_date<'2022-12-31' THEN observed_date END) AS prior_date,
            min(CASE WHEN observed_date>'2023-01-01' THEN observed_date END) AS later_date
        FROM tank_aerial_visual_observation GROUP BY candidate_id
    ), selected_aerial AS (
        SELECT a.candidate_id,a.prior_date,a.later_date,
            (SELECT CASE WHEN count(DISTINCT v.structure_observation)>1 THEN 'conflicting_observations'
                         ELSE min(v.structure_observation) END FROM tank_aerial_visual_observation v
             WHERE v.candidate_id=a.candidate_id AND v.observed_date=a.prior_date) AS prior_structure,
            (SELECT CASE WHEN count(DISTINCT v.structure_observation)>1 THEN 'conflicting_observations'
                         ELSE min(v.structure_observation) END FROM tank_aerial_visual_observation v
             WHERE v.candidate_id=a.candidate_id AND v.observed_date=a.later_date) AS later_structure
        FROM aerial a
    )
    SELECT l.candidate_id,l.dataset AS lidar_dataset,l.acquisition_date_utc AS lidar_acquisition_date_utc,
        l.point_source_id AS lidar_point_source_id,l.first_interior_return_at_utc,l.last_interior_return_at_utc,
        l.interior_points,l.rim_points,l.ground_points,l.interior_z_p10,l.interior_z_p50,l.interior_z_p90,
        l.rim_z_p95,l.ground_z_p50,round(l.rim_z_p95-l.interior_z_p50,3) AS rim_p95_minus_interior_p50_native_z,
        coalesce(l.rim_z_p95<l.interior_z_p50,0) AS rim_quantile_below_interior_median,
        o.level,o.scene_item_id,o.version_id,o.captured_at_utc AS optical_captured_at_utc,
        CASE WHEN julianday(o.captured_at_utc)<julianday(l.first_interior_return_at_utc) THEN 'before_lidar'
             WHEN julianday(o.captured_at_utc)>julianday(l.last_interior_return_at_utc) THEN 'after_lidar'
             ELSE 'within_lidar_return_interval' END AS optical_time_relation,
        round(24*(julianday(o.captured_at_utc)-julianday(l.last_interior_return_at_utc)),3) AS optical_minus_lidar_end_hours,
        round(24*(julianday(o.captured_at_utc)-julianday(l.first_interior_return_at_utc)),3) AS optical_minus_lidar_start_hours,
        CASE WHEN l.acquisition_date_utc<'2023-01-01' AND substr(o.captured_at_utc,1,10)<'2023-01-01'
             THEN 'development_capture_dates' ELSE 'includes_2023_capture' END AS capture_period,
        o.optical_cells,o.valid_cells,o.cloud_mask_basis,coalesce(o.roof_clear,0) AS roof_clear_by_source_mask,
        o.context_cells,o.context_opaque_cells,o.context_cirrus_cells,o.context_snow_ice_cells,o.context_scl_excluded_cells,
        o.red_detector_mask_basis,o.red_detector_covered_roof_cells,o.red_detector_ambiguous_roof_cells,
        o.sun_zenith_deg,o.sun_azimuth_grid_deg,o.red_view_options_json,o.resolved_b04_detector,
        o.source_directory,o.tile_metadata_path,r.processing_baseline,r.reflectance_scale AS red_scale,
        r.reflectance_offset AS red_offset,r.nodata_dn AS red_nodata_dn,r.saturated_dn AS red_saturated_dn,
        r.metadata_status AS red_radiometry_status,r.product_xml,
        a.prior_date AS prior_aerial_date,a.prior_structure AS prior_aerial_structure,
        a.later_date AS later_aerial_date,a.later_structure AS later_aerial_structure,
        t.footprint_diameter_m,t.lidar_upper_outline_diameter_m,t.lidar_fit_status,
        t.documented_roof_type,t.source_document_date AS roof_type_document_date
    FROM lidar_tank_observation l JOIN optical o USING (candidate_id)
    LEFT JOIN sentinel2_band_radiometry r ON r.level=o.level AND r.version_id=o.version_id AND r.band='B04'
    LEFT JOIN selected_aerial a USING (candidate_id) LEFT JOIN tank_engineering_registry t USING (candidate_id)
    WHERE l.dataset='OK_Statewide_2_D22';
"""


def add_lidar_optical_calibration(db):
    db.executemany("INSERT INTO lidar_calibration_capture_day VALUES (?)", ((day,) for day in CAPTURE_DAYS))
    rows = db.execute("""SELECT q.scene_item_id,q.candidate_id,q.captured_at_utc,t.longitude,t.latitude,
        EXISTS (SELECT 1 FROM supplemental_tank_candidate s WHERE s.candidate_id=q.candidate_id) AS supplemental
        FROM sentinel2_tank_scene_quality q JOIN lidar_calibration_capture_day d ON substr(q.captured_at_utc,1,10)=d.capture_day
        JOIN tank_engineering_registry t USING (candidate_id) ORDER BY q.scene_item_id,q.candidate_id""")
    coordinates, groups = {}, {}
    for scene, identity, captured, longitude, latitude, supplemental in rows:
        coordinates[identity] = longitude, latitude
        store = "sentinel2_supplemental" if supplemental else "imagery_core"
        directory = RAW / store / "sentinel2" / captured[:4] / scene
        groups.setdefault((scene, directory), []).append(identity)
    for (scene, directory), identities in groups.items():
        xml = RAW / "sentinel2_metadata" / scene / "granule_metadata.xml"
        files = [directory / f"{band}.tif" for band in (*CHANNELS, "scl")]
        if not all(path.is_file() for path in (*files, directory / "item.json", xml)):
            raise ValueError(f"Incomplete local calibration inputs for {scene} in {directory}")
        source = str(directory.relative_to(ROOT))
        for angle in observation_rows(scene, tuple(identities), coordinates, xml):
            db.execute("INSERT INTO lidar_calibration_l2a_geometry VALUES (?,?,?,?,?,?,?)",
                       (scene, angle[1], source, angle[14], angle[5], angle[7], angle[11]))
    add_l1c_angle_nodes(db)


def add_l1c_angle_nodes(db):
    rows = db.execute("""SELECT q.scene_item_id,q.l1c_product_name,q.tile_metadata_path,q.candidate_id,
        t.longitude,t.latitude,q.red_detector_center_ids_json
        FROM sentinel2_l1c_tank_quality q JOIN lidar_calibration_capture_day d ON substr(q.captured_at_utc,1,10)=d.capture_day
        JOIN tank_engineering_registry t USING (candidate_id) ORDER BY q.scene_item_id,q.candidate_id""")
    groups = {}
    for scene, version, path, candidate, longitude, latitude, detectors in rows:
        groups.setdefault((scene, version, path), []).append((candidate, (longitude, latitude), json.loads(detectors)))
    for (scene, version, path), contours in groups.items():
        xml = ET.fromstring(source_metadata_bytes(ROOT / path))
        position = xml.find(".//Tile_Geocoding/Geoposition[@resolution='10']")
        crs = xml.findtext(".//Tile_Geocoding/HORIZONTAL_CS_CODE")
        left, top = float(position.findtext("ULX")), float(position.findtext("ULY"))
        grids = {int(node.get("detectorId")): (array(node, "Zenith"), array(node, "Azimuth"))
                 for node in xml.findall(".//Viewing_Incidence_Angles_Grids[@bandId='3']")}
        east, north, _ = ground_coordinates(crs, tuple(row[1] for row in contours))
        row_nodes, col_nodes, weights = stencil((east - left) / 5000, (top - north) / 5000)
        for i, (candidate, _, detectors) in enumerate(contours):
            for detector in detectors:
                if detector not in grids:
                    continue
                zenith, azimuth = grids[detector]
                nodes = {(int(row), int(col)): ("bilinear_stencil", float(weight))
                         for row, col, weight in zip(row_nodes[:, i], col_nodes[:, i], weights[:, i], strict=True)}
                finite = np.isfinite(zenith) & np.isfinite(azimuth)
                if not all(finite[row, col] for row, col in nodes):
                    valid = np.argwhere(finite)
                    distance = np.hypot(left + valid[:, 1] * 5000 - east[i], top - valid[:, 0] * 5000 - north[i])
                    for index in np.argsort(distance, kind="stable")[:4]:
                        row, col = map(int, valid[index])
                        nodes.setdefault((row, col), ("nearby_finite_node", None))
                for (row, col), (role, weight) in nodes.items():
                    node_east, node_north = left + col * 5000, top - row * 5000
                    distance = np.hypot(node_east - east[i], node_north - north[i])
                    db.execute("INSERT INTO lidar_calibration_b04_angle_node VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                               (scene, version, candidate, detector, row, col, role, weight,
                                float(zenith[row, col]) if np.isfinite(zenith[row, col]) else None,
                                float(azimuth[row, col]) if np.isfinite(azimuth[row, col]) else None,
                                round(float(distance), 3), node_east, node_north, float(east[i]), float(north[i]), crs, path))
