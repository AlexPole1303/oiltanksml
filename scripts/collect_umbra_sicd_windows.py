"""Read native complex Umbra pixels around LiDAR-surveyed tanks."""
import argparse
import hashlib
import io
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from urllib.parse import urlencode

import numpy as np
import rasterio
from lxml import etree
from pyproj import Transformer
from rasterio.windows import Window
from sarpy.io.complex.sicd_elements.SICD import SICDType

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import ROOT, get, write
from collect_umbra_cushing import STORE

TO_GEO = Transformer.from_crs(32614, 4326, always_xy=True)
GRID_STEP_DEG = 0.05
GEOID_URL = "https://geodesy.noaa.gov/api/geoid/ght?"
WINDOW_GROUP_PIXELS = 512
MARGIN_M = 35


def tank_inputs():
    with closing(read_only(SOURCE)) as db:
        registry = {tank: (roof, terminal, first) for tank, roof, terminal, first in db.execute("""
            SELECT candidate_id,documented_roof_type,documented_terminal_id,first_seen
            FROM tank_engineering_registry WHERE documented_roof_type IS NOT NULL
        """)}
    records = {}
    for row in read("tank_lidar_outlines"):
        tank = row["candidate_id"]
        if (tank not in registry or row["dataset"] != "OK_Statewide_2_D22"
                or row["fit_status"] != "well_supported_circular_fit"):
            continue
        roof, terminal, first = registry[tank]
        if not first or first > row["first_upper_return_at_utc"][:10]:
            continue
        east, north, diameter, ground, top = (float(row[k]) for k in (
            "center_east_m", "center_north_m", "diameter_m", "prior_ground_z_p50_native", "prior_rim_z_p95_native"))
        if not all(map(math.isfinite, (east, north, diameter, ground, top))) or not 5 <= top - ground <= 30:
            continue
        lon, lat = TO_GEO.transform(east, north)
        records[tank] = dict(candidate_id=tank, roof_type=roof, terminal_id=terminal, longitude=lon, latitude=lat,
            east_m=east, north_m=north, radius_m=diameter/2, lidar_ground_navd88_m=ground, lidar_top_navd88_m=top,
            lidar_first_upper_return_utc=row["first_upper_return_at_utc"])
    return records


def geoid_node(lat_index, lon_index):
    lat, lon = (round(i * GRID_STEP_DEG, 6) for i in (lat_index, lon_index))
    url = GEOID_URL + urlencode(dict(lat=lat, lon=lon, model=14))
    path = STORE / "geoid18" / f"{lat}_{lon}.json"
    record = json.loads(get(url, path))
    if (record["geoidModel"], record["units"], record["lat"], record["lon"]) != ("GEOID18", "m", lat, lon):
        raise ValueError(f"NOAA GEOID18 response mismatch for {url}")
    return (lat_index, lon_index), float(record["geoidHeight"])


def with_ellipsoid_heights(tanks):
    positions = {k: (t["latitude"]/GRID_STEP_DEG, t["longitude"]/GRID_STEP_DEG) for k, t in tanks.items()}
    nodes = {(i + dy, j + dx) for lat, lon in positions.values() for i, j in [(math.floor(lat), math.floor(lon))]
             for dy in (0, 1) for dx in (0, 1)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        grid = dict(pool.map(lambda indices: geoid_node(*indices), sorted(nodes)))
    for tank, (lat, lon) in positions.items():
        i, j = math.floor(lat), math.floor(lon)
        fy, fx = lat - i, lon - j
        low = grid[i, j] * (1-fx) + grid[i, j+1] * fx
        high = grid[i+1, j] * (1-fx) + grid[i+1, j+1] * fx
        offset = low * (1-fy) + high * fy
        row = tanks[tank]
        row.update(geoid18_interpolated_m=offset, ground_hae_approx_m=row["lidar_ground_navd88_m"] + offset,
                   top_hae_approx_m=row["lidar_top_navd88_m"] + offset)
    return tanks


def sicd_metadata(dataset, path):
    des = dataset.tags(ns="xml:DES")["xml:DES"]
    root = etree.fromstring(des.encode())
    node, = root.xpath(".//*[local-name()='SICD']")
    payload = etree.tostring(node, encoding="UTF-8")
    sicd = SICDType.from_xml_string(payload)
    write(path, payload)
    return sicd


def project(sicd, east, north, height):
    lon, lat = TO_GEO.transform(east, north)
    row_col, error, _ = sicd.project_ground_to_image_geo(np.stack((lat, lon, height), axis=-1),
                                                       tolerance=0.002, max_iterations=20)
    if not np.isfinite(row_col).all() or np.max(error) > 0.02:
        raise ValueError(f"SICD ground projection did not converge: {np.max(error)} m")
    return row_col


def grouped_windows(sicd, dataset, tanks):
    groups = defaultdict(list)
    for t in tanks:
        east, north, pad = t["east_m"], t["north_m"], t["radius_m"] + MARGIN_M
        xyz = np.array([(east + de, north + dn, h) for h in (t["ground_hae_approx_m"], t["top_hae_approx_m"])
                        for de, dn in ((0, 0), (pad, pad), (pad, -pad), (-pad, pad), (-pad, -pad))])
        points = project(sicd, *xyz.T)
        r0, c0 = np.floor(points.min(axis=0) - 2).astype(int)
        r1, c1 = np.ceil(points.max(axis=0) + 3).astype(int)
        if r0 < 0 or c0 < 0 or r1 > dataset.height or c1 > dataset.width:
            continue
        row, col = points[0]
        key = (int(row)//WINDOW_GROUP_PIXELS, int(col)//WINDOW_GROUP_PIXELS)
        groups[key].append((t, (r0, c0, r1, c1), points[0], points[5]))
    for key, sample in sorted(groups.items()):
        bounds = np.asarray([b for _, b, _, _ in sample])
        r0, c0 = bounds[:, :2].min(axis=0)
        r1, c1 = bounds[:, 2:].max(axis=0)
        yield key, Window(int(c0), int(r0), int(c1-c0), int(r1-r0)), sample


def scene_windows(scene, tanks, *, store=STORE):
    directory = store / scene["product_id"]
    xml_path = directory / "sicd.xml"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="35", GDAL_HTTP_MAX_RETRY="2"), \
            rasterio.open(scene["sicd_url"]) as dataset:
        sicd = sicd_metadata(dataset, xml_path)
        if (dataset.dtypes not in (("complex64",), ("complex_int16",))
                or dataset.shape != (sicd.ImageData.NumRows, sicd.ImageData.NumCols)
                or sicd.CollectionInfo.CollectorName.casefold() != scene["platform"].casefold()):
            raise ValueError(f"Unexpected source SICD grid or platform {scene['product_id']}")
        license_source = scene.get("licence_source", "SICD NITF TEXT header")
        if license_source == "SICD NITF TEXT header":
            license_header = dataset.tags(ns="TEXT").get("HEADER_0", "")
            if "Creative Common By 4.0" not in license_header:
                raise ValueError(f"Unexpected source SICD licence {scene['product_id']}")
        products = []
        for key, window, sample in grouped_windows(sicd, dataset, tanks):
            path = directory / f"complex_{key[0]:03d}_{key[1]:03d}.npz"
            values = dataset.read(1, window=window)
            if values.shape != (int(window.height), int(window.width)) or not np.isfinite(values).all():
                raise ValueError(f"Incomplete SICD range read {path}")
            packed = io.BytesIO()
            np.savez_compressed(packed, complex_amplitude=values)
            payload = packed.getvalue()
            write(path, payload)
            digest = hashlib.sha256(payload).hexdigest()
            for tank, _, centre, top in sample:
                r, c = (int(round(v)) for v in centre - np.array([window.row_off, window.col_off]))
                products.append(dict(tank, product_id=scene["product_id"], capture_start_utc=scene["capture_start_utc"],
                    source_url=scene["sicd_url"], source_etag=scene["sicd_etag"], licence="CC BY 4.0", licence_source=license_source,
                    row_spacing_m=sicd.Grid.Row.SS, column_spacing_m=sicd.Grid.Col.SS,
                    row_impulse_response_m=sicd.Grid.Row.ImpRespWid, column_impulse_response_m=sicd.Grid.Col.ImpRespWid,
                    native_centre_row=centre[0], native_centre_column=centre[1],
                    native_top_centre_row=top[0], native_top_centre_column=top[1], native_ground_centre_nonzero=bool(values[r, c]),
                    window_row_offset=window.row_off, window_column_offset=window.col_off,
                    window_rows=window.height, window_columns=window.width,
                    window_file=str(path.relative_to(ROOT)), window_sha256=digest,
                    sicd_metadata_file=str(xml_path.relative_to(ROOT))))
    print(f"{scene['product_name']}: {len(products)} native tank windows indexed", flush=True)
    return products


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--all-scenes", action="store_true")
    args = parser.parse_args()
    pairs = read("umbra_cushing_close_capture_pairs")
    identities = {r[key] for r in pairs for key in ("first_product_id", "last_product_id")}
    scenes = [r for r in read("umbra_cushing_catalog") if args.all_scenes or r["product_id"] in identities]
    coverage = {r["product_id"]: set(json.loads(r["covered_candidate_ids_json"])) for r in scenes}
    tanks = with_ellipsoid_heights({k: t for k, t in tank_inputs().items() if any(k in ids for ids in coverage.values())})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda scene: scene_windows(scene, [t for k, t in tanks.items()
                                  if k in coverage[scene["product_id"]]]), scenes))
    rows = [r for batch in results for r in batch]
    stem = "umbra_sicd_all" if args.all_scenes else "umbra_sicd_close_pass"
    export(stem + "_window_index", rows)
    report = dict(scenes_requested=len(scenes), scenes_with_pixels=len({r["product_id"] for r in rows}),
        tank_scene_rows=len(rows), efr_tank_scene_rows=sum(r["roof_type"] == "EFR" for r in rows),
        fixed_roof_tank_scene_rows=sum(r["roof_type"] != "EFR" for r in rows),
        unique_tanks=len({r["candidate_id"] for r in rows}), native_windows=len({r["window_file"] for r in rows}),
        max_pixels_per_window=max(r["window_rows"] * r["window_columns"] for r in rows),
        interpretation="Native complex values from public byte-range reads. The SICD XML and window offsets preserve sensor geometry. "
        "GEOID18 is interpolated on a local grid. Treating NAD83 ellipsoid heights as an approximation to WGS84 is used for search windows. "
        "Subsequent height estimates require common horizontal alignment and independent accuracy checks. Source imagery © Umbra Lab, CC BY 4.0.")
    write(DERIVED / (stem + "_summary.json"), (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
