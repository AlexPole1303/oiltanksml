"""Package native optical pixels, independent LiDAR returns and instrument references."""
import csv
import hashlib
import io
import json
import math
import sqlite3
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import ExitStack, closing
from zipfile import ZipFile

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from shapely import area, box, intersection, transform
from shapely.geometry import Point

from build_l1c_quality import candidate_layer, provider_masks
from build_scene_quality import grid, grid_candidates
from build_sentinel2_angles import ground_coordinates
from collect_public import RAW, ROOT, get, write
from collect_s2_l1c import CHANNELS, source_metadata_bytes, source_metadata_path

DERIVED = ROOT / "data" / "derived"
OUTPUT = DERIVED / "shadow_calibration.sqlite"
PSF_SOURCE = "https://sentiwiki.copernicus.eu/web/s2-mission"
PSF_ARCHIVES = {
    "S2A": "a_5d1d206d9a40f649af3d757911e281c32fe4dfbb3cc324dac4090283efd7bd98/S2A_PSF.zip?cb=26847f9cf35711611888f4b47f9c62b1",
    "S2B": "a_daff9ee363bc255ee7ea2c0c199d5d2c9e2167ed96d6c598e805072735b9366b/S2B_PSF.zip?cb=0ad156d7a3c7e1883d003ed24c2cf52a",
    "S2C": "a_4b3ec8cd2059c485f640d94e0e35569a786aacb7d0ab86f25b06ccad0c0aefda/S2C_PSF.zip?cb=6e58e5c013ab8527dab541fbdadd8a7b",
}
CONTEXT_M = 30
SCHEMA = """
CREATE TABLE scene (
    scene_id INTEGER PRIMARY KEY,level TEXT,scene_item_id TEXT,version_id TEXT,captured_at_utc TEXT,
    platform TEXT,source_directory TEXT,tile_metadata_path TEXT,manifest_path TEXT,crs TEXT,
    affine_a REAL,affine_b REAL,affine_c REAL,affine_d REAL,affine_e REAL,affine_f REAL,
    width INTEGER,height INTEGER,tile_col_offset INTEGER,tile_row_offset INTEGER,
    cloud_mask_basis TEXT,red_detector_mask_basis TEXT,
    UNIQUE(level,version_id,source_directory));
CREATE TABLE band (
    scene_id INTEGER REFERENCES scene,band TEXT,channel TEXT,quantification_value REAL,add_offset_dn REAL,
    reflectance_scale REAL,reflectance_offset REAL,nodata_dn INTEGER,saturated_dn INTEGER,
    metadata_status TEXT,product_xml TEXT,source_raster TEXT,PRIMARY KEY(scene_id,band));
CREATE TABLE observation (
    observation_id INTEGER PRIMARY KEY,scene_id INTEGER REFERENCES scene,candidate_id TEXT,
    roof_type TEXT,cohort_id TEXT,engineering_source_date TEXT,first_seen_aerial TEXT,
    geometry_basis TEXT,outline_fit_status TEXT,center_east_m REAL,center_north_m REAL,radius_m REAL,
    sun_zenith_deg REAL,sun_azimuth_grid_deg REAL,true_north_grid_bearing_deg REAL,
    red_view_options_true_json TEXT,red_detector_id INTEGER,red_view_zenith_deg REAL,red_view_azimuth_grid_deg REAL,
    roof_cells_expected INTEGER,roof_cells_valid_all_bands INTEGER,roof_clear_by_source_mask INTEGER,
    resolved_b04_detector INTEGER,source_mask_status TEXT,stored_pixel_count INTEGER,
    retained_footprint_cells INTEGER,circle_area_bbl_per_m REAL,UNIQUE(scene_id,candidate_id));
CREATE TABLE pixel (
    observation_id INTEGER REFERENCES observation,tile_row INTEGER,tile_col INTEGER,
    mapped_footprint_cell INTEGER,mapped_context_cell INTEGER,lidar_circle_fraction REAL,
    dn_blue INTEGER,dn_green INTEGER,dn_red INTEGER,dn_nir INTEGER,raster_valid_band_bits INTEGER,
    opaque INTEGER,cirrus INTEGER,snow_ice INTEGER,red_detector_bits INTEGER,scl_at_pixel_center INTEGER,
    PRIMARY KEY(observation_id,tile_row,tile_col)) WITHOUT ROWID;
CREATE TABLE lidar_reference (
    candidate_id TEXT,dataset TEXT,acquisition_date_utc TEXT,point_source_id INTEGER,geometry_basis TEXT,
    outline_fit_status TEXT,center_east_m REAL,center_north_m REAL,radius_m REAL,first_return_at_utc TEXT,
    last_return_at_utc TEXT,center_nonground_returns INTEGER,center_sectors INTEGER,rim_nonground_returns INTEGER,
    rim_sectors INTEGER,center_z_p50_native REAL,center_plane_z_native REAL,center_plane_slope_east_native_per_m REAL,
    center_plane_slope_north_native_per_m REAL,center_plane_residual_p90_minus_p10_native REAL,
    rim_pooled_z_p95_native REAL,rim_sector_p95_median_native REAL,rim_sector_p95_range_native REAL,
    rim_minus_center_plane_native_z REAL,source_directory TEXT,source_node_ids_json TEXT,
    PRIMARY KEY(candidate_id,dataset,acquisition_date_utc,point_source_id));
CREATE TABLE lidar_pair (
    observation_id INTEGER REFERENCES observation,candidate_id TEXT,dataset TEXT,acquisition_date_utc TEXT,
    point_source_id INTEGER,optical_time_relation TEXT,first_return_at_utc TEXT,last_return_at_utc TEXT,
    optical_minus_lidar_start_hours REAL,optical_minus_lidar_end_hours REAL,
    PRIMARY KEY(observation_id,dataset,acquisition_date_utc,point_source_id),
    FOREIGN KEY(candidate_id,dataset,acquisition_date_utc,point_source_id) REFERENCES lidar_reference);
CREATE TABLE instrument_psf (
    platform TEXT,band TEXT,estimated_from_acquisition_year INTEGER,processing_level TEXT,
    oversample_factor INTEGER,side_length INTEGER,center_row_zero_based INTEGER,center_col_zero_based INTEGER,
    coefficient_sum REAL,minimum_coefficient REAL,negative_coefficients INTEGER,archive_path TEXT,member_path TEXT,
    archive_sha256 TEXT,member_sha256 TEXT,source_url TEXT,mission_reference_url TEXT,PRIMARY KEY(platform,band));
CREATE TABLE psf_coefficient (
    platform TEXT,band TEXT,psf_row INTEGER,psf_col INTEGER,coefficient REAL,
    PRIMARY KEY(platform,band,psf_row,psf_col),FOREIGN KEY(platform,band) REFERENCES instrument_psf) WITHOUT ROWID;
CREATE TABLE same_day_aerial_pair (
    candidate_id TEXT,aerial_id TEXT,aerial_date TEXT,aerial_start_utc TEXT,aerial_end_exclusive_utc TEXT,
    aerial_assets_json TEXT,aerial_nominal_gsd_m REAL,aerial_clock_file TEXT,level TEXT,scene_item_id TEXT,version_id TEXT,
    satellite_captured_at_utc TEXT,satellite_source_directory TEXT,minimum_separation_seconds REAL,
    maximum_separation_seconds REAL,optical_cells INTEGER,resolved_b04_detector INTEGER,documented_roof_type TEXT,
    aerial_top_observation TEXT,aerial_review_note TEXT,aerial_review_source_file TEXT,
    PRIMARY KEY(candidate_id,aerial_date,aerial_id,level,version_id));
CREATE VIEW pixel_red_reflectance AS
    SELECT p.*,s.crs,s.affine_c+(p.tile_col-s.tile_col_offset+0.5)*s.affine_a AS center_east_m,
        s.affine_f+(p.tile_row-s.tile_row_offset+0.5)*s.affine_e AS center_north_m,
        CASE WHEN p.raster_valid_band_bits&4 AND p.dn_red NOT IN (b.nodata_dn,b.saturated_dn)
             THEN p.dn_red*b.reflectance_scale+b.reflectance_offset END AS red_reflectance
    FROM pixel p JOIN observation o USING(observation_id) JOIN scene s USING(scene_id)
    JOIN band b ON b.scene_id=s.scene_id AND b.band='B04';
CREATE VIEW observation_lidar AS
    SELECT o.*,s.level,s.version_id,s.platform,s.captured_at_utc,l.dataset,l.acquisition_date_utc,l.point_source_id,
        l.first_return_at_utc,l.last_return_at_utc,l.optical_time_relation,
        l.optical_minus_lidar_start_hours,l.optical_minus_lidar_end_hours,
        r.rim_minus_center_plane_native_z,r.center_plane_residual_p90_minus_p10_native,r.rim_sector_p95_range_native
    FROM observation o JOIN scene s USING(scene_id) JOIN lidar_pair l USING(observation_id)
    JOIN lidar_reference r USING(candidate_id,dataset,acquisition_date_utc,point_source_id);
CREATE VIEW lidar_flight_line_spread AS
    SELECT candidate_id,dataset,acquisition_date_utc,count(*) AS flight_lines,
        count(rim_minus_center_plane_native_z) AS numeric_references,
        min(rim_minus_center_plane_native_z) AS minimum_gap_native_z,max(rim_minus_center_plane_native_z) AS maximum_gap_native_z,
        CASE WHEN count(rim_minus_center_plane_native_z)>1
             THEN max(rim_minus_center_plane_native_z)-min(rim_minus_center_plane_native_z) END AS gap_spread_native_z,
        (julianday(max(last_return_at_utc))-julianday(min(first_return_at_utc)))*86400 AS total_return_span_seconds
    FROM lidar_reference GROUP BY candidate_id,dataset,acquisition_date_utc;
"""


def insert(db, table, rows):
    columns = ",".join("?" for _ in db.execute(f"PRAGMA table_info({table})"))
    db.executemany(f"INSERT INTO {table} VALUES ({columns})", rows)


def add_psf(db):
    for platform, suffix in PSF_ARCHIVES.items():
        path = RAW / "sentinel2_instrument" / f"{platform}_PSF.zip"
        url = f"https://sentiwiki.copernicus.eu/__attachments/{suffix}"
        content = path.read_bytes() if path.is_file() else get(url)
        digest = hashlib.sha256(content).hexdigest()
        with ZipFile(io.BytesIO(content)) as archive:
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt PSF archive {path}")
            members = sorted(name for name in archive.namelist() if name.endswith(".csv"))
            if {name[-7:-4] for name in members} != {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                                                       "B8A", "B09", "B11", "B12"}:
                raise ValueError(f"Unexpected PSF band inventory in {path}")
            for name in members:
                band = name[-7:-4]
                member = archive.read(name)
                coefficients = np.loadtxt(io.BytesIO(member), delimiter=",")
                total = float(coefficients.sum())
                if coefficients.shape != (33, 33) or not np.isfinite(coefficients).all() or not math.isclose(total, 1, abs_tol=1e-4):
                    raise ValueError(f"Invalid PSF matrix in {path}: {name} has sum {total}")
                insert(db, "instrument_psf", [(platform, band, 2023 if platform == "S2B" else 2024, "L1B", 5, 33, 16, 16,
                       total, float(coefficients.min()), int((coefficients < 0).sum()), str(path.relative_to(ROOT)), name,
                       digest, hashlib.sha256(member).hexdigest(), url, PSF_SOURCE)])
                insert(db, "psf_coefficient", ((platform, band, row, col, float(value))
                       for (row, col), value in np.ndenumerate(coefficients)))
        if not path.is_file():
            write(path,content)


def geospatial_inputs(source):
    candidates = dict((*candidate_layer("osm"), *candidate_layer("supplemental")))
    rows = source.execute("""SELECT DISTINCT o.candidate_id,o.level,o.scene_item_id,o.version_id,o.optical_source_directory,
        o.optical_captured_at_utc,o.tile_metadata_path,o.documented_roof_type,t.cohort_id,t.source_document_date,t.first_seen,
        t.longitude,t.latitude,r.geometry_basis,r.outline_fit_status,r.center_east_m,r.center_north_m,r.radius_m,
        o.sun_zenith_deg,o.sun_azimuth_grid_deg,o.red_view_options_json,o.optical_cells,o.valid_cells,
        o.roof_clear_by_source_mask,o.resolved_b04_detector,q.cloud_mask_basis
        FROM lidar_roof_optical_reference o JOIN tank_engineering_registry t USING(candidate_id)
        JOIN lidar_roof_elevation_reference r USING(candidate_id,dataset,acquisition_date_utc,point_source_id)
        JOIN lidar_optical_calibration_pair q ON q.candidate_id=o.candidate_id AND q.level=o.level
            AND q.scene_item_id=o.scene_item_id AND q.version_id=o.version_id AND q.lidar_point_source_id=o.point_source_id
            AND q.lidar_dataset=o.dataset AND q.lidar_acquisition_date_utc=o.acquisition_date_utc
        ORDER BY o.level,o.version_id,o.optical_source_directory,o.candidate_id""")
    groups = defaultdict(list)
    for row in rows:
        groups[row["level"],row["version_id"],row["optical_source_directory"]].append(row)
    return candidates, groups


def mask_arrays(directory, level, reference, metadata, external_detector=None):
    if level == "L1C":
        clouds, cloud_basis, detector, detector_basis = provider_masks(directory, external_detector, reference, metadata)
        return clouds, cloud_basis, detector, detector_basis, None
    scl = np.zeros(reference.shape, dtype="uint8")
    with rasterio.open(directory / "scl.tif") as source:
        reproject(rasterio.band(source, 1), scl, dst_transform=reference.transform, dst_crs=reference.crs,
                  dst_nodata=0, resampling=Resampling.nearest)
    return (None, None, None), "l2a_scl", None, None, scl.ravel()


def pixel_rows(identity, reference, circle, mapped_polygon, footprint, context, offsets, dn, valid, clouds, detector, scl):
    context_bounds = box(*mapped_polygon.bounds).union(box(*circle.bounds)).buffer(CONTEXT_M).bounds
    window = from_bounds(*context_bounds, reference.transform)
    row, col = np.mgrid[max(0, math.floor(window.row_off)):min(reference.height, math.ceil(window.row_off+window.height)),
                        max(0, math.floor(window.col_off)):min(reference.width, math.ceil(window.col_off+window.width))]
    indices = row.ravel()*reference.width+col.ravel()
    e = reference.transform.c + col.ravel()*reference.transform.a
    n = reference.transform.f + row.ravel()*reference.transform.e
    cells = box(e, n+reference.transform.e, e+reference.transform.a, n)
    fractions = area(intersection(circle, cells)) / (-reference.transform.e*reference.transform.a)
    footprint_flags, context_flags = np.isin(indices, footprint), np.isin(indices, context)
    row_offset, col_offset = offsets
    return [(identity, int(r+row_offset), int(c+col_offset), int(f), int(x), round(float(fraction), 9),
             *(int(channel[i]) for channel in dn), int(valid[i]),
             *(int(mask[i]) if mask is not None else None for mask in clouds),
             int(detector[i]) if detector is not None else None, int(scl[i]) if scl is not None else None)
            for r, c, i, f, x, fraction in zip(row.ravel(), col.ravel(), indices, footprint_flags,
                                                context_flags, fractions, strict=True)]


def add_scene(db, source, scene_id, key, observations, candidates, identities):
    level, version, path = key
    directory = ROOT / path
    example = observations[0]
    tile_path = source_metadata_path(ROOT / example["tile_metadata_path"])
    metadata = ET.fromstring(source_metadata_bytes(tile_path))
    metadata_position = metadata.find(".//Tile_Geocoding/Geoposition[@resolution='10']")
    tile_transform = Affine(float(metadata_position.findtext("XDIM")), 0, float(metadata_position.findtext("ULX")),
                            0, float(metadata_position.findtext("YDIM")), float(metadata_position.findtext("ULY")))
    with ExitStack() as stack:
        band_sources = [stack.enter_context(rasterio.open(directory/f"{channel}.tif")) for channel in CHANNELS]
        reference = band_sources[0]
        if any(grid(dataset) != grid(reference) for dataset in band_sources):
            raise ValueError(f"Calibration band grids differ in {directory}")
        raster_transform = reference.transform
        col_offset, row_offset = ~tile_transform * (raster_transform.c, raster_transform.f)
        if (reference.crs.to_string() != metadata.findtext(".//Tile_Geocoding/HORIZONTAL_CS_CODE")
            or abs(col_offset-round(col_offset))>1e-7 or abs(row_offset-round(row_offset))>1e-7
            or (raster_transform.a,raster_transform.b,raster_transform.d,raster_transform.e)!=(10,0,0,-10)):
            raise ValueError(f"Local raster is not on its original 10 m tile grid in {directory}")
        row_offset,col_offset = round(row_offset),round(col_offset)
        dn = [dataset.read(1).ravel() for dataset in band_sources]
        valid = np.zeros(reference.width*reference.height, dtype="uint8")
        for bit, dataset in enumerate(band_sources):
            valid |= (dataset.read_masks(1).ravel()>0).astype("uint8") << bit
        external_detector = None
        if level == "L1C" and not source_metadata_path(directory/"MSK_DETFOO_B04.gml").is_file():
            external = source.execute("""SELECT local_file FROM sentinel2_l1c_early_detector_footprint
                WHERE scene_item_id=? AND l1c_product_name=?""", (example["scene_item_id"],version)).fetchone()
            if external and external[0]:
                external_detector = source_metadata_path(ROOT/external[0])
        clouds, cloud_basis, detector, detector_basis, scl = mask_arrays(directory, level, reference, metadata, external_detector)
        insert(db,"scene",[(scene_id,level,example["scene_item_id"],version,example["optical_captured_at_utc"],version[:3],path,
               str(tile_path.relative_to(ROOT)),str((directory/"item.json").relative_to(ROOT)),reference.crs.to_string(),
               *tuple(raster_transform)[:6],reference.width,reference.height,col_offset,row_offset,cloud_basis,detector_basis)])
        radiometry = list(source.execute("""SELECT band,channel,quantification_value,add_offset_dn,reflectance_scale,
            reflectance_offset,nodata_dn,saturated_dn,metadata_status,product_xml FROM sentinel2_band_radiometry
            WHERE level=? AND version_id=? ORDER BY band""", (level,version)))
        if {row[0] for row in radiometry} != set(CHANNELS.values()) or any(None in tuple(r)[2:8] for r in radiometry):
            raise ValueError(f"Missing four-band radiometry in {directory}")
        insert(db,"band",((scene_id,*r,str((directory/f"{r['channel']}.tif").relative_to(ROOT))) for r in radiometry))
        mapped = [(r["candidate_id"],candidates[r["candidate_id"]]) for r in observations]
        covers = grid_candidates(reference,mapped)
        project = Transformer.from_crs(4326,reference.crs,always_xy=True).transform
        lidar_transform = Transformer.from_crs(32614,reference.crs,always_xy=True).transform
        bearings = ground_coordinates(reference.crs.to_string(),tuple((r["longitude"],r["latitude"]) for r in observations))[2]
        for row,bearing in zip(observations,bearings,strict=True):
            candidate = row["candidate_id"]
            if (key,candidate) in identities:
                raise ValueError(f"More than one geometry for {candidate} in {version}")
            sample_id = len(identities)+1
            identities[key,candidate] = sample_id
            expected, footprint, context = covers[candidate]
            four_band_valid = valid[footprint]==15
            if level=="L1C":
                four_band_valid &= np.logical_and.reduce([band[footprint]!=0 for band in dn])
            if (expected,int(four_band_valid.sum()),cloud_basis)!=(row["optical_cells"],row["valid_cells"],row["cloud_mask_basis"]):
                raise ValueError(f"Native pixels do not reproduce the stored QA counts for {candidate} in {version}")
            center = lidar_transform(row["center_east_m"],row["center_north_m"])
            radius = row["radius_m"]
            circle = Point(*center).buffer(radius,quad_segs=128)
            views = json.loads(row["red_view_options_json"])
            detector_id, view_zenith, view_azimuth = views[0] if row["resolved_b04_detector"] else (None,None,None)
            pixels = pixel_rows(sample_id,reference,circle,transform(candidates[candidate],project,interleaved=False),
                                footprint,context,(row_offset,col_offset),dn,valid,clouds,detector,scl)
            coverage_status = "archive_roof_clear" if row["roof_clear_by_source_mask"] else "archive_roof_mask_flagged"
            insert(db,"observation",[(sample_id,scene_id,candidate,row["documented_roof_type"],row["cohort_id"],row["source_document_date"],
                   row["first_seen"],row["geometry_basis"],row["outline_fit_status"],*center,radius,row["sun_zenith_deg"],
                   row["sun_azimuth_grid_deg"],float(bearing),row["red_view_options_json"],detector_id,view_zenith,
                   (view_azimuth+float(bearing))%360 if view_azimuth is not None else None,expected,int(four_band_valid.sum()),
                   row["roof_clear_by_source_mask"],row["resolved_b04_detector"],coverage_status,len(pixels),len(footprint),
                   math.pi*radius*radius/0.158987294928)])
            insert(db,"pixel",pixels)


def load_lidar(db, source, identities):
    fields = ",".join(row[1] for row in db.execute("PRAGMA table_info(lidar_reference)"))
    insert(db,"lidar_reference",source.execute(f"SELECT {fields} FROM lidar_roof_elevation_reference"))
    rows = source.execute("""SELECT level,version_id,optical_source_directory,candidate_id,dataset,acquisition_date_utc,
        point_source_id,optical_time_relation,first_return_at_utc,last_return_at_utc,optical_minus_lidar_start_hours,
        optical_minus_lidar_end_hours FROM lidar_roof_optical_reference""")
    insert(db,"lidar_pair",((identities[(level,version,directory),candidate],candidate,*values)
           for level,version,directory,candidate,*values in rows))


def add_same_day_aerial(db, source):
    review_path = ROOT / "data" / "reference" / "shadow_same_day_aerial_review.csv"
    with review_path.open(newline="") as stream:
        reviews = {(r["candidate_id"],r["aerial_date"]): (r["top_observation"],r["detail"],str(review_path.relative_to(ROOT)))
                   for r in csv.DictReader(stream)}
    rows = source.execute("""SELECT a.candidate_id,a.aerial_id,a.aerial_date,a.aerial_start_utc,a.aerial_end_exclusive_utc,
        a.aerial_assets_json,a.aerial_nominal_gsd_m,a.aerial_clock_file,a.level,a.scene_item_id,a.version_id,
        a.satellite_captured_at_utc,a.satellite_source_directory,a.minimum_separation_seconds,a.maximum_separation_seconds,
        a.optical_cells,q.red_detector_covered_roof_cells=a.optical_cells AND q.red_detector_ambiguous_roof_cells=0
            AND json_array_length(q.red_detector_with_view_options_json)=1,t.documented_roof_type
        FROM aerial_optical_pair a JOIN tank_engineering_registry t USING(candidate_id)
        LEFT JOIN sentinel2_l1c_tank_quality q ON a.candidate_id=q.candidate_id AND a.version_id=q.l1c_product_name
            AND a.level='L1C'
        WHERE a.aerial_source='usda_naip_source_polygon' AND a.aerial_contour_fraction>=0.999999
            AND a.optical_cells=a.valid_cells AND a.roof_clear_by_source_mask=1
            AND substr(a.satellite_captured_at_utc,1,10)=a.aerial_date
        ORDER BY a.aerial_date,a.candidate_id,a.aerial_id,a.level,a.version_id""").fetchall()
    if reviews.keys()-{(r[0],r[2]) for r in rows}:
        raise ValueError("An audited same-day aerial pair is missing from the source database")
    for row in rows:
        if not all((ROOT / path).is_file() for path in json.loads(row[5])):
            raise ValueError(f"Missing same-day NAIP assets for {row[0]} on {row[2]}")
    insert(db,"same_day_aerial_pair",((*row,*reviews.get((row[0],row[2]),(None,None,None))) for row in rows))


def same_day_l1c_inputs(db, source):
    groups = defaultdict(list)
    pairs = db.execute("""SELECT DISTINCT candidate_id,version_id FROM same_day_aerial_pair
        WHERE level='L1C' AND resolved_b04_detector ORDER BY version_id,candidate_id""")
    for candidate,version in pairs:
        row = source.execute("""SELECT q.candidate_id,'L1C' AS level,q.scene_item_id,q.l1c_product_name AS version_id,
            q.source_directory AS optical_source_directory,q.captured_at_utc AS optical_captured_at_utc,q.tile_metadata_path,
            t.documented_roof_type,t.cohort_id,t.source_document_date,t.first_seen,t.longitude,t.latitude,
            'lidar_elevated_circle' AS geometry_basis,l.fit_status AS outline_fit_status,l.center_east_m,l.center_north_m,
            l.diameter_m/2 AS radius_m,q.sun_zenith_deg,q.sun_azimuth_grid_deg,
            q.red_detector_with_view_options_json AS red_view_options_json,q.optical_touched_cells_expected AS optical_cells,
            q.optical_cells_valid_in_all_bands AS valid_cells,q.opaque_roof_cells=0 AND q.cirrus_roof_cells=0
            AND coalesce(q.snow_ice_roof_cells,0)=0 AS roof_clear_by_source_mask,1 AS resolved_b04_detector,q.cloud_mask_basis
            FROM sentinel2_l1c_tank_quality q JOIN tank_lidar_outline l USING(candidate_id)
            JOIN tank_engineering_registry t USING(candidate_id) WHERE q.candidate_id=? AND q.l1c_product_name=?""",
            (candidate,version)).fetchone()
        if row is None:
            raise ValueError(f"Missing native L1C inputs or circle for {candidate} in {version}")
        groups[row["level"],row["version_id"],row["optical_source_directory"]].append(row)
    return groups


def output_report(db):
    summary = {table: db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in
               ("scene","observation","pixel","lidar_reference","lidar_pair","instrument_psf","psf_coefficient",
                "same_day_aerial_pair")}
    summary["coverage"] = [dict(row) for row in db.execute("""SELECT level,substr(captured_at_utc,1,10) AS capture_day,roof_type,
        count(DISTINCT observation_id) AS candidate_products,count(DISTINCT candidate_id) AS candidates,count(*) AS lidar_pairs,
        count(DISTINCT CASE WHEN roof_clear_by_source_mask AND roof_cells_expected=roof_cells_valid_all_bands
                             AND resolved_b04_detector THEN candidate_id END) AS clear_with_resolved_detector_candidates,
        min(abs(optical_minus_lidar_end_hours)) AS closest_lidar_end_hours,max(abs(optical_minus_lidar_end_hours)) AS farthest_lidar_end_hours
        FROM observation_lidar GROUP BY 1,2,3 ORDER BY 1,2,3""")]
    summary["lidar_flight_line_repeatability"] = [dict(row) for row in db.execute("""SELECT dataset,
        count(*) AS candidate_dates,count(gap_spread_native_z) AS repeated_candidate_dates,
        min(gap_spread_native_z) AS min_spread_native_z,max(gap_spread_native_z) AS max_spread_native_z
        FROM lidar_flight_line_spread GROUP BY dataset ORDER BY dataset""")]
    summary["same_day_aerial_coverage"] = [dict(row) for row in db.execute("""SELECT aerial_date,aerial_top_observation,
        count(DISTINCT candidate_id) AS candidates,min(minimum_separation_seconds) AS minimum_separation_seconds,
        max(maximum_separation_seconds) AS maximum_separation_seconds
        FROM same_day_aerial_pair WHERE level='L1C' GROUP BY aerial_date,aerial_top_observation
        ORDER BY aerial_date,aerial_top_observation""")]
    return summary


def export_observations(db):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    rows = db.execute("""SELECT observation_id,candidate_id,roof_type,cohort_id,level,version_id,platform,captured_at_utc,
        dataset,acquisition_date_utc,point_source_id,optical_time_relation,optical_minus_lidar_start_hours,
        optical_minus_lidar_end_hours,rim_minus_center_plane_native_z,center_plane_residual_p90_minus_p10_native,
        rim_sector_p95_range_native,roof_cells_expected,roof_cells_valid_all_bands,roof_clear_by_source_mask,
        resolved_b04_detector,stored_pixel_count FROM observation_lidar ORDER BY observation_id,point_source_id""")
    writer.writerow(column[0] for column in rows.description)
    writer.writerows(rows)
    write(DERIVED/"shadow_calibration_observations.csv",buffer.getvalue().encode())


def main():
    temporary = OUTPUT.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    source_uri = (DERIVED/"cushing.sqlite").as_uri()+"?mode=ro"
    with closing(sqlite3.connect(source_uri,uri=True)) as source, closing(sqlite3.connect(temporary)) as db, db:
        source.row_factory = db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        candidates, groups = geospatial_inputs(source)
        if not groups:
            raise ValueError("No LiDAR and Sentinel-2 calibration pairs in the source database")
        add_same_day_aerial(db,source)
        for key,observations in same_day_l1c_inputs(db,source).items():
            groups[key].extend(observations)
        identities = {}
        add_psf(db)
        for number,(key, observations) in enumerate(groups.items(),1):
            add_scene(db,source,number,key,observations,candidates,identities)
            print(number,key[0],key[1],len(observations),flush=True)
        load_lidar(db,source,identities)
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Calibration database integrity check failed")
        summary = output_report(db)
        export_observations(db)
    temporary.replace(OUTPUT)
    write(DERIVED/"shadow_calibration_summary.json",json.dumps(summary,indent=2).encode())
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
