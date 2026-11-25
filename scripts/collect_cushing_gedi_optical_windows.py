"""Preserve native Sentinel-2 pixels and masks around the public GEDI roof candidates."""
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from statistics import median

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely import transform
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from audit_s2_sun_clock import xml_time
from build_cushing_icesat2_controls import TO_GEO, identities
from build_l1c_quality import QUALITY_FIELDS, provider_masks, scene_rows
from build_s1_reference_matches import read
from build_sentinel2_angles import ground_coordinates
from build_weekly_shadow_panel import DERIVED
from collect_cushing_gedi_olm import STORE, export
from collect_public import ROOT, write
from collect_s2_l1c import download, earliest_products, parse_product, source_metadata_bytes, source_metadata_path
from extract_s2_radiometry import BAND_ID, read_xml
from fit_shadow_geometry import CALIBRATION, bearing_vector, geometry_fit, read_only


STEM = "cushing_gedi_optical"
DEST = STORE / "sentinel2_pixels"
CONTEXT_M = 120


def inventory():
    by_granule = defaultdict(list)
    for r in read("cushing_gedi_olm_v003_original_candidates"):
        by_granule[r["gedi_acquisition_key"]].append(r)
    features = {}
    shots = defaultdict(list)
    for r in read("cushing_gedi_olm_sentinel2_catalog"):
        payload = (ROOT / r["earth_search_snapshot"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != r["earth_search_sha256"]:
            raise ValueError("Sentinel-2 catalog snapshot differs from archived checksum")
        for feature in json.loads(payload)["features"]:
            key = parse_product(feature["properties"]["s2:product_uri"])[0]
            coverage = shape(feature["geometry"])
            for shot in by_granule[r["gedi_acquisition_key"]]:
                if not coverage.covers(Point(float(shot["longitude"]), float(shot["latitude"]))):
                    continue
                previous = features.get(key)
                if previous is None or feature["id"] < previous["id"]:
                    features[key] = feature
                if shot not in shots[key]:
                    shots[key].append(shot)
    return features, shots


def windows(key, feature, shots, tanks, versions):
    circles = {r["candidate_id"]: Point(tanks[r["candidate_id"]]["east_m"], tanks[r["candidate_id"]]["north_m"]).buffer(
        tanks[r["candidate_id"]]["radius_m"]) for r in shots}
    footprint = unary_union([poly.buffer(CONTEXT_M) for poly in circles.values()])
    bounds = transform(footprint, TO_GEO.transform, interleaved=False).bounds
    acquisition = feature["properties"]["datetime"]
    directory = DEST / "granules" / acquisition[:4] / feature["id"]
    status = download(feature, versions, DEST, bounds, None, verify=True, pack_text=True)
    if status["status"] != "ready":
        return dict(feature_id=feature["id"], product_key="_".join(key), **status), []
    manifest = directory / "item.json"
    geometry = [(k, transform(poly, TO_GEO.transform, interleaved=False)) for k, poly in circles.items()]
    points = {k: (tanks[k]["longitude"], tanks[k]["latitude"]) for k in circles}
    qa = {q[0]: dict(zip(QUALITY_FIELDS, q, strict=True)) for q in scene_rows(manifest, geometry, points, {}, None)}
    results = []
    for shot in shots:
        identity = shot["candidate_id"]
        row = qa.get(identity)
        if not row:
            continue
        observed = xml_time(row["tile_metadata_path"])
        hours = (abs((observed - datetime.fromisoformat(shot["gedi_approximate_capture_utc"])).total_seconds()) + 1) / 3600
        mask_fields = ("opaque_roof_cells", "cirrus_roof_cells", "snow_ice_roof_cells")
        full = row["optical_touched_cells_expected"] == row["optical_cells_valid_in_all_bands"]
        clear = row["cloud_mask_basis"] != "missing" and all(not row[k] for k in mask_fields)
        resolved = (row["red_detector_covered_roof_cells"] == row["optical_touched_cells_expected"]
                    and not row["red_detector_ambiguous_roof_cells"] and len(json.loads(row["red_detector_with_view_options_json"])) == 1)
        results.append(dict(row, source_tile_id=shot["source_tile_id"], gedi_extract_row=shot["extract_row"],
            gedi_approximate_capture_utc=shot["gedi_approximate_capture_utc"], sentinel2_xml_capture_utc=observed.isoformat(),
            max_separation_hours=hours, native_optical_valid=full, provider_cloud_mask_clear=clear,
            b04_detector_and_view_resolved=resolved, inputs_pass=full and clear and resolved and hours <= 48))
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
    return dict(feature_id=feature["id"], product_key="_".join(key), local_manifest=str(manifest.relative_to(ROOT)),
        version=json.loads(manifest.read_bytes())["product_name"], file_sha256_json=json.dumps(files, sort_keys=True), **status), results


def kernels(platforms):
    with closing(read_only(CALIBRATION)) as db:
        result = {}
        for platform in platforms:
            values = [v for v, in db.execute("""SELECT coefficient FROM psf_coefficient WHERE platform=? AND band='B04'
                                                ORDER BY psf_row,psf_col""", (platform,))]
            kernel = np.asarray(values, dtype="float32").reshape(33, 33)
            result[platform] = (kernel + kernel.T) / (2 * kernel.sum())
        return result


@lru_cache(maxsize=4)
def native_b04_inputs(source_directory, tile_metadata_path, product):
    directory = ROOT / source_directory
    quant, offsets, nodata, saturated, _ = read_xml(source_metadata_path(directory / "MTD_MSIL1C.xml"), "L1C", product)
    if quant is None or BAND_ID["B04"] not in offsets:
        return None
    metadata = ET.fromstring(source_metadata_bytes(ROOT / tile_metadata_path))
    with rasterio.open(directory / "red.tif") as source:
        clouds, _, bits, _ = provider_masks(directory, None, source, metadata)
        return (quant, offsets[BAND_ID["B04"]], nodata, saturated, source.transform, source.crs.to_string(),
                source.read(1), source.read_masks(1), clouds, bits)


def shadow_geometry_inputs(row, tank):
    product = row["l1c_product_name"]
    native = native_b04_inputs(row["source_directory"], row["tile_metadata_path"], product)
    if native is None:
        return dict(fit_status="missing_product_radiometry")
    quant, offset, nodata, saturated, affine, crs, data, mask, clouds, bits = native
    (detector, view_zenith, view_azimuth), = json.loads(row["red_detector_with_view_options_json"])
    radius = tank["radius_m"]
    shell = tank["lidar_top_navd88_m"] - tank["lidar_ground_navd88_m"]
    (east,), (north,), (bearing,) = ground_coordinates(crs, ((tank["longitude"], tank["latitude"]),))
    extent = radius + 25
    window = from_bounds(east - extent, north - extent, east + extent, north + extent, affine)
    height, width = data.shape
    rows, columns = np.mgrid[max(0, int(np.floor(window.row_off))):min(height, int(np.ceil(window.row_off + window.height))),
                             max(0, int(np.floor(window.col_off))):min(width, int(np.ceil(window.col_off + window.width)))]
    x, y = affine * (columns + 0.5, rows + 0.5)
    distance = np.hypot(x - east, y - north)
    index = rows * width + columns
    dn = data[rows, columns]
    valid = (mask[rows, columns] > 0) & (dn != nodata) & (dn != saturated)
    for cloud in clouds:
        if cloud is not None:
            valid &= cloud[index] == 0
    valid &= (bits[index] == 1 << (detector - 1)) & (distance <= radius + 16)
    count = int(valid.sum())
    if count < 12:
        return dict(fitted_pixels=count, fit_status="insufficient_valid_pixels")
    reflectance = (dn[valid].astype(float) + offset) / quant
    background = np.exp(-0.5 * (np.maximum(distance[valid] - radius, 0) / 7) ** 2)
    grid = (*rows.shape, (x[0, 0] - east, y[0, 0] - north), rows[valid] - rows.min(),
            columns[valid] - columns.min(), reflectance, background)
    sun = bearing_vector(row["sun_zenith_deg"], row["sun_azimuth_grid_deg"])
    view = bearing_vector(view_zenith, (view_azimuth + bearing) % 360)
    return grid, radius, shell, sun, view


def fit_geometry(row, tank, kernel):
    geometry = shadow_geometry_inputs(row, tank)
    if isinstance(geometry, dict):
        return geometry
    grid, radius, shell, sun, view = geometry
    return dict(fitted_pixels=len(grid[-2]), **geometry_fit(grid, radius, shell, "EFR", sun, view, kernel))


def comparisons(quality, tanks):
    guarded = {(r["source_tile_id"], r["extract_row"]): r for r in read("cushing_gedi_olm_roof_candidates")
               if r["footprint_inside_with_10m_margin"] == "True"}
    selected = [r for r in quality if r["inputs_pass"]]
    psf = kernels({r["l1c_product_name"][:3] for r in selected})
    for row in selected:
        shot = guarded[row["source_tile_id"], row["gedi_extract_row"]]
        result = fit_geometry(row, tanks[row["candidate_id"]], psf[row["l1c_product_name"][:3]])
        optical = result.get("estimated_gap_m")
        gedi = float(shot["wall_to_lowest_mode_approx_m"]) if shot["wall_to_lowest_mode_approx_m"] else None
        yield dict(row, **result, gedi_wall_to_lowest_mode_approx_m=gedi,
                   sentinel2_minus_gedi_m=optical - gedi if optical is not None and gedi is not None else None)


def main():
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    os.environ.setdefault("OPJ_NUM_THREADS", "1")
    tanks, _, _ = identities()
    features, shots = inventory()
    versions = earliest_products(features.values(), 3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pairs = list(pool.map(lambda key: windows(key, features[key], shots[key], tanks, versions), sorted(features)))
    sources = [s for s, _ in pairs]
    qa = [r for _, subset in pairs for r in subset]
    fits = list(comparisons(qa, tanks))
    resolved = [r for r in fits if r["fit_status"] == "resolved_in_geometry_diagnostic"]
    by_shot = defaultdict(list)
    for row in resolved:
        by_shot[row["source_tile_id"], row["gedi_extract_row"]].append(row)
    selected = []
    for (tile, index), group in sorted(by_shot.items()):
        values = [r["sentinel2_minus_gedi_m"] for r in group if r["sentinel2_minus_gedi_m"] is not None]
        selected.append(dict(candidate_id=group[0]["candidate_id"], source_tile_id=tile, gedi_extract_row=index,
            gedi_approximate_capture_utc=group[0]["gedi_approximate_capture_utc"],
            max_separation_hours=max(r["max_separation_hours"] for r in group), tiles=len(group),
            optical_versions_json=json.dumps(sorted(r["l1c_product_name"] for r in group)),
            median_sentinel2_minus_gedi_m=median(values) if values else None))
    absolute = [abs(r["median_sentinel2_minus_gedi_m"]) for r in selected if r["median_sentinel2_minus_gedi_m"] is not None]
    export(DERIVED / f"{STEM}_sources.csv", sources, ("feature_id", "status"))
    export(DERIVED / f"{STEM}_roof_quality.csv", qa, QUALITY_FIELDS)
    export(DERIVED / f"{STEM}_fits.csv", fits, ("candidate_id", "scene_item_id", "fit_status"))
    export(DERIVED / f"{STEM}_comparisons.csv", selected, ("candidate_id", "source_tile_id", "gedi_extract_row"))
    result = dict(granules_requested=len(features), source_status=Counter(r["status"] for r in sources),
        roof_tile_shot_rows=len(qa), within_48h=sum(r["max_separation_hours"] <= 48 for r in qa),
        source_mask_clear=sum(r["provider_cloud_mask_clear"] for r in qa),
        passes_optical_inputs=sum(r["inputs_pass"] for r in qa),
        distinct_gedi_shots_with_passing_inputs=len({(r["source_tile_id"], r["gedi_extract_row"]) for r in qa if r["inputs_pass"]}),
        fit_status=Counter(r["fit_status"] for r in fits),
        distinct_gedi_shots_with_resolved_optical_fit=len(selected), resolved_optical_fit_tile_rows=len(resolved),
        resolved_shots_within_24h=sum(r["max_separation_hours"] <= 24 for r in selected),
        median_absolute_sensor_disagreement_m=median(absolute) if absolute else None,
        interpretation="Optical fits use source B04 and LiDAR rim geometry. GEDI waveform confirmation is still required.")
    payload = json.dumps(result, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
