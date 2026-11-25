"""Match aerial source-image acquisition intervals to local Sentinel-2 pixels."""
import datetime as dt
import json
from collections import defaultdict
from zoneinfo import ZoneInfo

from pyproj import Transformer
from shapely import make_valid, transform
from shapely.geometry import shape
from shapely.strtree import STRtree

from collect_naip_seamlines import SEAMLINES, SERVICES, candidates
from collect_public import ROOT

SCHEMA = """
CREATE TABLE naip_2025_wcs_coverage (candidate_id TEXT PRIMARY KEY,clip_id TEXT,asset_path TEXT,
    candidate_pixels INTEGER,valid_pixels INTEGER);
CREATE TABLE naip_tank_source_clock (
    candidate_id TEXT,acquisition_year INTEGER,source_polygon_id TEXT,source_acquisition_date TEXT,
    source_start_local TEXT,source_end_local TEXT,start_utc TEXT,end_exclusive_utc TEXT,
    contour_fraction REAL,intersecting_source_polygons INTEGER,nominal_tile_dates_json TEXT,
    nominal_date_matches INTEGER,asset_paths_json TEXT,dated_asset_paths_json TEXT,asset_time_basis TEXT,
    minimum_nominal_gsd_m REAL,source_file TEXT,
    PRIMARY KEY(candidate_id,acquisition_year,source_polygon_id));
CREATE INDEX aerial_l1c_candidate_capture ON sentinel2_l1c_tank_quality(candidate_id,captured_at_utc);
CREATE INDEX aerial_l2a_candidate_capture ON sentinel2_tank_scene_quality(candidate_id,captured_at_utc);
CREATE VIEW aerial_optical_pair AS
    WITH aerial AS (
        SELECT candidate_id,'usda_naip_source_polygon' AS aerial_source,source_polygon_id AS aerial_id,
            source_acquisition_date AS aerial_date,start_utc AS aerial_start_utc,end_exclusive_utc AS aerial_end_exclusive_utc,
            contour_fraction AS aerial_contour_fraction,intersecting_source_polygons,dated_asset_paths_json AS aerial_assets_json,
            minimum_nominal_gsd_m AS aerial_nominal_gsd_m,source_file AS aerial_clock_file,asset_time_basis
        FROM naip_tank_source_clock WHERE json_array_length(dated_asset_paths_json)>0
        UNION ALL
        SELECT candidate_id,'ok_aerial_2025_tile_majority',tile_id,tile_majority_date,
            tile_majority_date||'T'||tile_time_raw||':00Z',
            strftime('%Y-%m-%dT%H:%M:%SZ',tile_majority_date||'T'||tile_time_raw||':00Z','+1 minute'),
            valid_fraction,NULL,json_array(asset_path),0.15,replace(asset_path,'/image.tif','/item.json'),'tile_majority_clock'
        FROM ok_aerial_tank_image_coverage WHERE length(tile_time_raw)=5
    ), pairs AS (
        SELECT a.*,'L1C' AS level,q.scene_item_id,q.l1c_product_name AS version_id,q.captured_at_utc AS satellite_captured_at_utc,
            q.source_directory AS satellite_source_directory,q.optical_touched_cells_expected AS optical_cells,
            q.optical_cells_valid_in_all_bands AS valid_cells,q.cloud_mask_basis,
            q.opaque_roof_cells=0 AND q.cirrus_roof_cells=0 AND coalesce(q.snow_ice_roof_cells,0)=0 AS roof_clear_by_source_mask,
            q.sun_zenith_deg,q.sun_azimuth_grid_deg
        FROM aerial a JOIN sentinel2_l1c_tank_quality q ON q.candidate_id=a.candidate_id
            AND q.captured_at_utc>=date(a.aerial_start_utc,'-7 days')
            AND q.captured_at_utc<date(a.aerial_end_exclusive_utc,'+8 days')
        WHERE q.optical_cells_valid_in_all_bands>0
        UNION ALL
        SELECT a.*,'L2A',q.scene_item_id,q.scene_item_id,q.captured_at_utc,
            'data/raw/'||CASE WHEN s.candidate_id IS NULL THEN 'imagery_core' ELSE 'sentinel2_supplemental' END
                ||'/sentinel2/'||substr(q.captured_at_utc,1,4)||'/'||q.scene_item_id,
            q.optical_touched_cells_expected,q.optical_cells_valid_in_all_bands,'l2a_scl',
            q.scl_touched_cells_expected>0
                AND q.scl_touched_cells_expected=q.scl_class_2+q.scl_class_4+q.scl_class_5+q.scl_class_6+q.scl_class_7,
            NULL,NULL
        FROM aerial a JOIN sentinel2_tank_scene_quality q ON q.candidate_id=a.candidate_id
            AND q.captured_at_utc>=date(a.aerial_start_utc,'-7 days')
            AND q.captured_at_utc<date(a.aerial_end_exclusive_utc,'+8 days')
        LEFT JOIN supplemental_tank_candidate s ON s.candidate_id=a.candidate_id
        WHERE q.optical_cells_valid_in_all_bands>0
    ), separation AS (
        SELECT p.*,
            round((julianday(satellite_captured_at_utc)-julianday(aerial_end_exclusive_utc))*86400,3)
                AS satellite_minus_aerial_end_seconds,
            round((julianday(satellite_captured_at_utc)-julianday(aerial_start_utc))*86400,3)
                AS satellite_minus_aerial_start_seconds
        FROM pairs p
    )
    SELECT *,max(satellite_minus_aerial_end_seconds,-satellite_minus_aerial_start_seconds,0) AS minimum_separation_seconds,
        max(abs(satellite_minus_aerial_end_seconds),abs(satellite_minus_aerial_start_seconds)) AS maximum_separation_seconds
    FROM separation WHERE max(satellite_minus_aerial_end_seconds,-satellite_minus_aerial_start_seconds,0)<=7*86400;
"""

LOCAL_ZONE = ZoneInfo("America/Chicago")


def acquisition_minute(raw):
    return dt.datetime.strptime(raw, "%m/%d/%Y %H:%M").replace(tzinfo=LOCAL_ZONE)


def asset_evidence(db):
    assets = defaultdict(list)
    for identity, date, path, gsd in db.execute("SELECT candidate_id,acquisition_date,asset_path,pixel_width_m FROM naip_tank_image_coverage"):
        assets[identity, int(date[:4])].append((date, path, gsd))
    for identity, path in db.execute("SELECT candidate_id,asset_path FROM naip_2025_wcs_coverage"):
        assets[identity, 2025].append((None, path, 0.6))
    return assets


def source_clock_rows(db):
    project = Transformer.from_crs(4326, 32614, always_xy=True).transform
    tanks = [(identity, transform(tank, project, interleaved=False)) for identity, tank in candidates()]
    assets = asset_evidence(db)
    for year in SERVICES:
        path = SEAMLINES / str(year) / "features.geojson"
        if not path.is_file():
            continue
        features = json.loads(path.read_bytes())["features"]
        shapes = [transform(make_valid(shape(f["geometry"])), project, interleaved=False) for f in features]
        index = STRtree(shapes)
        for identity, tank in tanks:
            candidates_in_year = assets[identity, year]
            dates = sorted({d for d, _, _ in candidates_in_year if d})
            paths = sorted({p for _, p, _ in candidates_in_year})
            if any(not (ROOT / file).is_file() for file in paths):
                raise ValueError(f"Missing local NAIP image for {identity} in {year}")
            matches = [(int(i), tank.intersection(shapes[i]).area / tank.area) for i in index.query(tank, predicate="intersects")]
            matches = [(i, fraction) for i, fraction in matches if fraction > 0]
            if not matches:
                raise ValueError(f"No USDA source polygon for {identity} in {year}")
            for i, fraction in sorted(matches):
                f = features[i]
                attrs = f["properties"]
                date = attrs["IDATE"]
                compatible = [(p, m) for d, p, m in candidates_in_year if d==date or (d is None and year==2025)]
                dated_paths = sorted({p for p, _ in compatible})
                gsd = min((m for _, m in compatible), default=None)
                basis = ("state_naip_2025_mosaic_and_usda_source_polygon" if year == 2025 else
                         "stac_tile_date_agrees_with_usda_polygon") if dated_paths else "source_date_unconfirmed_in_local_rasters"
                start, end = map(acquisition_minute, (attrs["SDATE"], attrs["EDATE"]))
                if start.date().isoformat() != date or end < start:
                    raise ValueError(f"Conflicting source times for NAIP {year} polygon {f['id']}")
                upper = end.astimezone(dt.timezone.utc) + dt.timedelta(minutes=1)
                yield (identity,year,str(f["id"]),date,attrs["SDATE"],attrs["EDATE"],
                       start.astimezone(dt.timezone.utc).isoformat(),upper.isoformat(),round(fraction,8),len(matches),
                       json.dumps(dates),int(date in dates) if dates else None,json.dumps(paths),json.dumps(dated_paths),
                       basis,gsd,str(path.relative_to(ROOT)))


def add_aerial_optical_pairs(db):
    db.executemany("INSERT INTO naip_tank_source_clock VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", source_clock_rows(db))
