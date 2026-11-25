"""Index Landsat acquisition clocks, local bands, and native-grid roof QA."""
import csv
import datetime as dt
import json
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.features import geometry_mask, geometry_window
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from collect_hls_landsat_gaps import HLS, HLS_ASSETS, HLS_SPECTRAL
from collect_landsat import CONTEXT_ASSETS, LANDSAT, daylight, identity, scene_key, source_coverage
from collect_naip_seamlines import candidates
from collect_public import ROOT

SCHEMA = """
CREATE TABLE landsat_scene_catalog (provider TEXT,item_id TEXT,acquisition_key TEXT,platform TEXT,captured_at_utc TEXT,
    wrs_path_row TEXT,collection_number TEXT,processing_category TEXT,sun_azimuth_deg REAL,sun_elevation_deg REAL,
    scene_cloud_pct REAL,catalog_created_utc TEXT,pan_source_url TEXT,qa_source_url TEXT,source_file TEXT,
    PRIMARY KEY(provider,item_id));
CREATE TABLE landsat_local_product (source_kind TEXT,product_id TEXT PRIMARY KEY,acquisition_key TEXT,captured_at_utc TEXT,
    native_signal_gsd_m INTEGER,native_qa_gsd_m INTEGER,source_created_utc TEXT,clock_basis TEXT,
    sun_azimuth_deg REAL,sun_elevation_deg REAL,source_directory TEXT,metadata_path TEXT);
CREATE TABLE landsat_reflectance_coefficients (product_id TEXT,band TEXT,multiplier REAL,additive REAL,
    reflectance_basis TEXT,PRIMARY KEY(product_id,band));
CREATE TABLE landsat_tank_scene_quality (product_id TEXT,candidate_id TEXT,signal_cells INTEGER,valid_signal_cells INTEGER,
    qa_roof_cells INTEGER,valid_qa_roof_cells INTEGER,cloud_roof_cells INTEGER,cirrus_roof_cells INTEGER,
    cloud_shadow_roof_cells INTEGER,snow_roof_cells INTEGER,saturated_roof_cells INTEGER,terrain_occluded_roof_cells INTEGER,
    clear_qa_roof_cells INTEGER,context_qa_cells INTEGER,valid_context_qa_cells INTEGER,clear_context_qa_cells INTEGER,
    PRIMARY KEY(product_id,candidate_id));
CREATE TABLE landsat_archive_asset_clock (provider TEXT,product_id TEXT,asset_label TEXT,archive_url TEXT,
    object_created_at_utc TEXT,object_last_modified_at_utc TEXT,source_bytes INTEGER,object_version TEXT,
    version_basis TEXT,source_file TEXT,PRIMARY KEY(provider,product_id,asset_label));
CREATE VIEW landsat_product_archive_clock AS
    SELECT provider,product_id,count(*) AS assets,max(object_created_at_utc) AS all_assets_created_by_utc,
        max(coalesce(object_last_modified_at_utc,object_created_at_utc)) AS latest_archive_object_time_utc
    FROM landsat_archive_asset_clock GROUP BY provider,product_id;
CREATE TABLE landsat_hls_gap_evidence (acquisition_key TEXT PRIMARY KEY,gap_reason TEXT,usgs_product_ids_json TEXT,usgs_scene_ids_json TEXT,
    pc_hls_day_items INTEGER,cmr_hls_day_items INTEGER,exact_hls_matches_json TEXT);
CREATE TABLE landsat_pc_asset_exception (product_id TEXT,asset_label TEXT,acquisition_key TEXT,source_url TEXT,
    http_status INTEGER,content_type TEXT,content_range TEXT,etag TEXT,first_bytes_hex TEXT,validation_status TEXT,
    source_file TEXT,PRIMARY KEY(product_id,asset_label));
CREATE TABLE landsat_hls_source_scene_match (acquisition_key TEXT,hls_id TEXT,contributing_scene_ids_json TEXT,
    metadata_path TEXT,PRIMARY KEY(acquisition_key,hls_id));
CREATE VIEW landsat_hls_gap_coverage AS
    SELECT g.acquisition_key,g.gap_reason,g.usgs_scene_ids_json,g.pc_hls_day_items,g.cmr_hls_day_items,
        json_array_length(g.exact_hls_matches_json) AS exact_hls_catalog_matches,
        count(DISTINCT h.hls_id) AS local_hls_granules,
        count(DISTINCT CASE WHEN q.valid_signal_cells>0 THEN q.candidate_id END) AS contours_with_local_pixels,
        count(DISTINCT CASE WHEN q.signal_cells=q.valid_signal_cells AND q.qa_roof_cells=q.clear_qa_roof_cells
                            THEN q.candidate_id END) AS contours_fully_sampled_with_clear_qa,
        max(l.source_created_utc) AS latest_local_granule_archive_time_utc
    FROM landsat_hls_gap_evidence g LEFT JOIN landsat_hls_source_scene_match h USING (acquisition_key)
    LEFT JOIN landsat_local_product l ON l.product_id=h.hls_id
    LEFT JOIN landsat_tank_scene_quality q ON q.product_id=h.hls_id GROUP BY g.acquisition_key;
CREATE VIEW landsat_route_coverage AS
    WITH keys AS (SELECT DISTINCT acquisition_key,platform,substr(captured_at_utc,1,10) AS acquired_date,wrs_path_row
                  FROM landsat_scene_catalog), catalog AS (
        SELECT acquisition_key,max(provider='usgs') AS usgs_level1_catalog,max(provider='pc') AS pc_level2_catalog
        FROM landsat_scene_catalog GROUP BY acquisition_key
    ), staged AS (
        SELECT acquisition_key,max(source_kind='gcs_collection1_level1_pan') AS gcs_pan_local,
            max(source_kind='pc_collection2_level2_context') AS pc_level2_local
        FROM landsat_local_product GROUP BY acquisition_key
    )
    SELECT k.*,c.usgs_level1_catalog,c.pc_level2_catalog,coalesce(s.gcs_pan_local,0) AS gcs_pan_local,
        coalesce(s.pc_level2_local,0) AS pc_level2_local,
        EXISTS(SELECT 1 FROM landsat_hls_source_scene_match h WHERE h.acquisition_key=k.acquisition_key) AS hls_l30_local
    FROM keys k JOIN catalog c USING (acquisition_key) LEFT JOIN staged s USING (acquisition_key);
"""


def add_catalog(db, area):
    for provider in ("usgs", "pc"):
        for path in sorted((LANDSAT / f"{provider}_catalog").glob("*.geojson")):
            source = str(path.relative_to(ROOT))
            for item in json.loads(path.read_bytes())["features"]:
                if not daylight(item) or not shape(item["geometry"]).intersects(area):
                    continue
                p, assets = item["properties"], item["assets"]
                captured, platform = p["datetime"], p["platform"].lower().replace("_", "-")
                grid, _ = scene_key(item)
                db.execute("INSERT INTO landsat_scene_catalog VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (provider,item["id"],identity(platform,grid,captured),platform,captured,grid,
                            p.get("landsat:collection_number"),p.get("landsat:collection_category"),p["view:sun_azimuth"],
                            p["view:sun_elevation"],p.get("eo:cloud_cover"),p.get("created"),assets.get("pan",{}).get("href"),
                            assets.get("qa_pixel",{}).get("href"),source))


def window_values(raster, geometry, values):
    window = geometry_window(raster, [geometry], boundless=True)
    row, col, height, width = map(int, (window.row_off,window.col_off,window.height,window.width))
    region = geometry_mask([geometry], out_shape=(height,width), transform=raster.window_transform(window),
                           invert=True, all_touched=True)
    r0,r1 = max(0,row),min(raster.height,row+height)
    c0,c1 = max(0,col),min(raster.width,col+width)
    expected = int(np.count_nonzero(region))
    if r0 >= r1 or c0 >= c1:
        return expected, values.reshape((len(values), -1))[:, :0]
    inside = region[r0-row:r1-row,c0-col:c1-col]
    return expected, values[:,r0:r1,c0:c1][:,inside]


def bit(values, index):
    return (values & (1 << index)) != 0


def native_masks(qa, saturation, kind):
    if kind=="pan":
        high = lambda index: ((qa >> index) & 3) == 3
        return (bit(qa,4) | high(5), high(11), high(7), high(9), (qa & 12)!=0, bit(qa,1))
    if kind=="hls":
        return (bit(qa,1) | bit(qa,2),None,bit(qa,3),bit(qa,4),None,None)
    return (bit(qa,1) | bit(qa,3),bit(qa,2),bit(qa,4),bit(qa,5),(saturation & 126)!=0,bit(saturation,11))


def quality_rows(directory, product, kind, geometries):
    spectral, qa_label = {"pan":(("pan",),"bqa"),"context":(CONTEXT_ASSETS[:6],"qa_pixel"),
                          "hls":(HLS_SPECTRAL,"Fmask")}[kind]
    with ExitStack() as stack:
        qa = stack.enter_context(rasterio.open(directory / f"{qa_label}.tif"))
        signals = [stack.enter_context(rasterio.open(directory / f"{label}.tif")) for label in spectral]
        signal = signals[0]
        signature = lambda raster: (raster.crs,raster.transform,raster.shape)
        if any(signature(raster) != signature(signal) for raster in signals):
            raise ValueError(f"Landsat spectral grids disagree for {product}")
        if signal.crs != qa.crs or (kind!="pan" and signature(signal) != signature(qa)):
            raise ValueError(f"Landsat signal and QA grids disagree for {product}")
        values = qa.read(1)
        fill = values==255 if kind=="hls" else bit(values,0)
        valid = (qa.read_masks(1)>0) & ~fill
        saturation = None
        if kind=="context":
            radsat = stack.enter_context(rasterio.open(directory / "qa_radsat.tif"))
            if signature(radsat) != signature(qa):
                raise ValueError(f"Landsat saturation and cloud grids disagree for {product}")
            saturation = radsat.read(1)
        flags = native_masks(values,saturation,kind)
        mask_values = [valid & f if f is not None else np.zeros_like(valid) for f in flags]
        qa_stack = np.stack((valid,*mask_values,valid & ~np.logical_or.reduce(mask_values)))
        context_stack = qa_stack[[0,-1]]
        spectral_valid = [raster.read_masks(1)>0 for raster in signals]
        if kind!="hls":
            spectral_valid = [valid & (raster.read(1)>0) for valid,raster in zip(spectral_valid,signals,strict=True)]
        sig_stack = np.array([np.logical_and.reduce(spectral_valid)])
        for candidate_id, roof, context in geometries(qa.crs):
            signal_count, signal_pixels = window_values(signal,roof,sig_stack)
            valid_signal = int(np.count_nonzero(signal_pixels))
            roof_count, roof_pixels = window_values(qa,roof,qa_stack)
            roof_values = [int(x) for x in np.count_nonzero(roof_pixels,axis=1)]
            for index,flag in enumerate(flags,1):
                if flag is None:
                    roof_values[index] = None
            if not valid_signal and not roof_values[0]:
                continue
            context_count, context_pixels = window_values(qa,context,context_stack)
            context_values = [int(x) for x in np.count_nonzero(context_pixels,axis=1)]
            yield (product,candidate_id,signal_count,valid_signal,roof_count,*roof_values,context_count,*context_values)


def add_azure_clocks(db,path,provider):
    value = json.loads(path.read_bytes())
    source = str(path.relative_to(ROOT))
    db.executemany("INSERT INTO landsat_archive_asset_clock VALUES (?,?,?,?,?,?,?,?,?,?)",
                   ((provider,value["product_id"],label,entry["url"],entry["created_at_utc"],
                     entry["last_modified_at_utc"],entry["bytes"],entry["etag"],"azure_etag",source)
                    for label,entry in value["assets"].items()))
    return value


def local_clock(value,files):
    if set(value["assets"])!=set(files):
        return None
    for label,path in files.items():
        clock = dt.datetime.fromisoformat(value["assets"][label]["last_modified_at_utc"])
        if clock.timestamp()>path.stat().st_mtime:
            return None
    return value["latest_asset_modified_at_utc"]


def add_gcs_clocks(db,item,path):
    index = path.parent / "objects.json"
    objects = json.loads(index.read_bytes())["records"]
    product = item["product_id"]
    metadata, = (x for x in objects if x["name"].endswith(f"/{product}_MTL.txt"))
    assets = {**item["assets"],"mtl.txt":metadata}
    db.executemany("INSERT INTO landsat_archive_asset_clock VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (("gcs_landsat_collection1",product,label,"https://storage.googleapis.com/gcp-public-data-landsat/"+entry["name"],
                     entry["timeCreated"],None,int(entry["size"]),entry["crc32c"],"gcs_crc32c",str(index.relative_to(ROOT)))
                    for label,entry in assets.items()))


def add_gap_evidence(db):
    path = ROOT / "data" / "derived" / "landsat_hls_gap_evidence.csv"
    if path.is_file():
        with path.open() as file:
            db.executemany("INSERT INTO landsat_hls_gap_evidence VALUES (?,?,?,?,?,?,?)",
                           ((row["acquisition_key"],row.get("gap_reason","missing_from_pc_catalog"),row["usgs_product_ids_json"],
                             row["usgs_scene_ids_json"],row["pc_hls_day_items"],row["cmr_hls_day_items"],row["exact_hls_matches_json"])
                            for row in csv.DictReader(file)))
    fields = ("product_id","asset_label","acquisition_key","source_url","http_status","content_type","content_range",
              "etag","first_bytes_hex","validation_status")
    for path in sorted((LANDSAT/"pc_asset_exceptions").glob("*.json")):
        value = json.loads(path.read_bytes())
        db.execute("INSERT INTO landsat_pc_asset_exception VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   (*[value[field] for field in fields],str(path.relative_to(ROOT))))


def add_landsat(db):
    _, area = source_coverage()
    add_catalog(db, area)
    add_gap_evidence(db)
    pc_clocks = {p.stem:add_azure_clocks(db,p,"azure_landsat_collection2")
                 for p in sorted((LANDSAT/"pc_asset_times").glob("*/*.json"))}
    shapes = [(candidate_id,shape(transform_geom(4326,32614,mapping(roof)))) for candidate_id,roof in candidates()]
    cache = {}

    def geometries(crs):
        if crs not in cache:
            cache[crs] = [(identity,transform_geom(32614,crs,mapping(roof)),
                           transform_geom(32614,crs,mapping(roof.buffer(60)))) for identity,roof in shapes]
        return cache[crs]

    for store,kind in ((LANDSAT/"legacy_pan","pan"),(LANDSAT/"collection2_context","context"),(HLS/"granules","hls")):
        for path in sorted(store.glob("*/*/item.json")):
            item = json.loads(path.read_bytes())
            directory = path.parent
            if kind=="pan":
                product, captured = item["product_id"],item["captured_at_utc"]
                key = identity("landsat-8",product.split("_")[2],captured)
                source, clock = "gcs_collection1_level1_pan","latest_gcs_object_creation"
                created = item["archive_assets_created_by_utc"]
                azimuth,elevation = item["sun_azimuth_deg"],item["sun_elevation_deg"]
                metadata = directory / "MTL.txt"
                radiometry = [("B8",item["reflectance_mult_b8"],item["reflectance_add_b8"],"toa_before_solar_elevation_correction")]
                add_gcs_clocks(db,item,path)
            elif kind=="context":
                p = item["properties"]
                product,captured = item["id"],p["datetime"]
                key = identity(p["platform"],scene_key(item)[0],captured)
                source,clock = "pc_collection2_level2_context","provider_catalog_creation"
                created,azimuth,elevation = p.get("created"),p["view:sun_azimuth"],p["view:sun_elevation"]
                label = item.get("local_metadata_asset","mtl.json")
                metadata = directory / ("MTL.xml" if label=="mtl.xml" else "MTL.json")
                coefficients = json.loads((directory/"MTL.json").read_bytes())["LANDSAT_METADATA_FILE"]["LEVEL2_SURFACE_REFLECTANCE_PARAMETERS"]
                radiometry = [(f"B{n}",float(coefficients[f"REFLECTANCE_MULT_BAND_{n}"]),
                               float(coefficients[f"REFLECTANCE_ADD_BAND_{n}"]),"surface_reflectance") for n in range(2,8)]
                if product in pc_clocks:
                    archived = local_clock(pc_clocks[product],{label:metadata,**{band:directory/f"{band}.tif" for band in CONTEXT_ASSETS}})
                    if archived:
                        created,clock = archived,"latest_azure_asset_modified"
            else:
                p = item["item"]["properties"]
                product,captured = item["item"]["id"],p["datetime"]
                key = None
                source = "azure_hls_l30_gap_context"
                metadata = ROOT / item["source_tags_file"]
                azimuth,elevation = p.get("view:sun_azimuth"),p.get("view:sun_elevation")
                coefficients = item["reflectance_coefficients"]
                radiometry = [(label,*coefficients[label],"hls_surface_reflectance") for label in HLS_SPECTRAL]
                archive = add_azure_clocks(db,directory/"asset_times.json","azure_hls_l30")
                created = local_clock(archive,{band:directory/f"{band}.tif" for band in HLS_ASSETS})
                clock = "latest_azure_asset_modified" if created else "archive_version_not_verified_for_local_files"
                db.executemany("INSERT INTO landsat_hls_source_scene_match VALUES (?,?,?,?)",
                               ((matched,product,json.dumps(item["source_scene_ids"]),item["source_tags_file"])
                                for matched in item["matched_acquisition_keys"]))
            db.execute("INSERT INTO landsat_local_product VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (source,product,key,captured,15 if kind=="pan" else 30,30,created,clock,azimuth,elevation,
                        str(directory.relative_to(ROOT)),str(metadata.relative_to(ROOT))))
            db.executemany("INSERT INTO landsat_reflectance_coefficients VALUES (?,?,?,?,?)",
                           ((product,*row) for row in radiometry))
            db.executemany("INSERT INTO landsat_tank_scene_quality VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           quality_rows(directory,product,kind,geometries))
