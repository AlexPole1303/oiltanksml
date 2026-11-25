"""Recover a Landsat 9 B8 roof window near the second March SLC endpoint."""

import hashlib
import json
import math
from contextlib import ExitStack, closing
from datetime import datetime, timezone

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile

from audit_landsat_esri_b8_registration import context_crop, padded_chip_bounds
from build_cross_sensor_shadow import published_lsf
from build_landsat_c2_b8_measurements import (
    QA_LABELS, engineering_inputs, prepare_native, prepare_tool, roof_touched, source_tables, verified_file,
)
from build_landsat_esri_b8_controls import fit_tile
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, read_only
from collect_landsat_c2_gap_geometry import native_b8_grid, roof_angles
from collect_landsat_esri_b8 import query, raster_request
from collect_public import RAW, ROOT, get, write
from stage_landsat_c2_b8_inputs import roof_windows, scene_qa

STEM = "cushing_slc_march_b8_control"
PRODUCT = "LC09_L1TP_027035_20250317_20250317_02_T1"


def selected_roof():
    row, = [r for r in read("ceda_s1_slc_roof_profiles_weekly")
            if r["week_ending"] == "2025-03-21" and r["both_pass"] == "True"]
    return row["candidate_id"]


def native_geometry(tank, product, stage, database):
    _, legacy = engineering_inputs()
    roof = legacy[tank]
    (angle,) = roof_angles(prepare_tool(), product, [tank], legacy, "dated_2011_lidar")
    if angle["angle_status"] != "resolved":
        raise ValueError("Landsat B8 angles unresolved for the selected roof")
    to_native = Transformer.from_crs(32614, product["crs"], always_xy=True)
    x, y = to_native.transform(float(roof["center_east_m"]), float(roof["center_north_m"]))
    col, line = (~native_b8_grid(product)) * (x, y)
    center = (math.floor(line), math.floor(col))
    position = (tank, x, y, float(roof["diameter_m"]) / 2, *center)
    with ExitStack() as stack:
        rasters = [stack.enter_context(rasterio.open(verified_file(ROOT / stage[f"{key}_path"], stage[f"{key}_sha512"])))
                   for key in QA_LABELS]
        windows = list(roof_windows(product, [position], scene_qa(product, rasters, [position])))
    expected = (*center, *(float(angle[key]) for key in ("solar_azimuth_deg", "solar_zenith_deg", "view_azimuth_deg", "view_zenith_deg")))
    prepare_native({PRODUCT: stage}, {PRODUCT: product}, {PRODUCT: windows}, {(PRODUCT, tank): expected}, database=database)
    return windows[0]


def esri_window(product, roof):
    url, payload, (item,) = query(f"Name = '{PRODUCT}'")
    when = datetime.fromtimestamp(item["AcquisitionDate"] / 1000, timezone.utc)
    if (item["Name"] != PRODUCT or item["LANDSAT_PRODUCT_ID"] != PRODUCT
            or item["LANDSAT_SCENE_ID"] != product["level1_scene_id"] or item["SensorName"] != "Landsat 9"
            or abs((when - datetime.fromisoformat(product["captured_at_utc"])).total_seconds()) >= 1):
        raise ValueError("Esri B8 query did not resolve to the exact USGS version")
    store = RAW / "landsat" / STEM
    listing = store / "esri_catalog.json"
    write(listing, payload)
    r0, r1, c0, c1 = [int(roof[field]) for field in ("b8_first_line", "b8_last_line", "b8_first_sample", "b8_last_sample")]
    bounds = (r0, r1 + 1, c0, c1 + 1)
    identity = item["OBJECTID"]
    source, expected = raster_request(product, identity, bounds)
    content = get(source)
    with MemoryFile(content) as mem, mem.open() as raster:
        if (raster.transform != expected or raster.crs.to_string() != product["crs"]
                or raster.shape != (r1 - r0 + 1, c1 - c0 + 1) or raster.dtypes != ("int16",)):
            raise ValueError("Esri crop is not on the requested USGS grid")
    path = store / "roof_reflectance.tif"
    write(path, content)
    tile = dict(product_id=PRODUCT, object_id=identity, first_line=r0, end_line_exclusive=r1 + 1,
                first_sample=c0, end_sample_exclusive=c1 + 1, reflectance_scale=.0001,
                local_path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(content).hexdigest(),
                source_url=source, source_catalog_url=url, source_catalog_file=str(listing.relative_to(ROOT)),
                source_catalog_sha256=hashlib.sha256(payload).hexdigest())
    overlap = context_crop(product, identity, padded_chip_bounds(product, [tile]), [tile])
    return tile, overlap


def main():
    tank = selected_roof()
    stages, products, _, _ = source_tables()
    product, stage = products[PRODUCT], stages[PRODUCT]
    database = DERIVED / f"{STEM}_angles.sqlite"
    roof = native_geometry(tank, product, stage, database)
    tile, overlap = esri_window(product, roof)
    fit, = fit_tile(tile, [{"candidate_id": tank}], product, stage, published_lsf(), database)
    fit["esri_exterior_overlap_status"] = overlap["overlap_status"]
    with closing(read_only(database)) as db:
        pixel_count, = db.execute("SELECT count(*) FROM native_pixel").fetchone()
        native = np.asarray(db.execute(
            "SELECT center_east_m,center_north_m,touching_qa_valid,touching_qa_flag_bits,angle_status "
            "FROM native_pixel ORDER BY native_line,native_sample").fetchall(), dtype=object)
    expected = (tile["end_line_exclusive"] - tile["first_line"]) * (tile["end_sample_exclusive"] - tile["first_sample"])
    if len(native) != pixel_count or pixel_count != expected:
        raise ValueError("Landsat reflectance and QA windows have different pixel counts")
    inside = roof_touched(native[:, 0].astype(float), native[:, 1].astype(float),
                          *(roof[key] for key in ("center_east_m", "center_north_m", "radius_m")))
    flags = native[inside, 3].astype(int)
    with rasterio.open(ROOT / tile["local_path"]) as raster:
        dn, valid = raster.read(1).ravel(), raster.read_masks(1).ravel() > 0
    fit.update(roof_touching_pixels=int(inside.sum()),
               roof_touching_pixels_clear_with_angles=int(sum(
                   bool(valid_qa) and not flag and status == "resolved" for valid_qa, flag, status in native[inside, 2:])),
               roof_touching_pixels_with_valid_esri_reflectance=int(np.count_nonzero(inside & valid & (dn > 0) & (dn < 10000))),
               roof_touching_pixels_cloud_flags_json=json.dumps({key: int(np.count_nonzero(flags & (1 << bit)))
                   for key, bit in (("dilated_cloud", 1), ("cirrus", 2), ("cloud", 3), ("cloud_shadow", 4), ("snow", 5),
                                    ("terrain_occlusion", 11))}, sort_keys=True))
    export(f"{STEM}_fits", [fit])
    export(f"{STEM}_sources", [tile | dict(
        angle_file=stage["local_angle_file"], angle_sha512=stage["source_angle_sha512"],
        qa_pixel_file=stage["qa_pixel_path"], qa_pixel_sha512=stage["qa_pixel_sha512"],
        qa_radsat_file=stage["qa_radsat_path"], qa_radsat_sha512=stage["qa_radsat_sha512"],
        local_geometry_database=str(database.relative_to(ROOT)), native_angle_qa_pixels=pixel_count,
        second_crop_file=overlap["raster_path"], second_crop_sha256=overlap["raster_sha256"],
        shared_valid_pixel_comparisons=overlap["shared_valid_pixel_comparisons"],
        shared_value_disagreements=overlap["shared_value_disagreements"],
        shared_mask_disagreements=overlap["shared_mask_disagreements"], overlap_status=overlap["overlap_status"])])
    print(json.dumps({key: fit.get(key) for key in ("candidate_id", "optical_capture_at_utc", "fit_status",
          "estimated_gap_m", "esri_exterior_overlap_status", "roof_touching_pixels", "roof_touching_pixels_clear_with_angles",
          "roof_touching_pixels_with_valid_esri_reflectance", "roof_touching_pixels_cloud_flags_json")}, indent=2))


if __name__ == "__main__":
    main()
