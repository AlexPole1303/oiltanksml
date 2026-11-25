"""Stage public imagery for the research AOI at the source pixel grids."""
import argparse
import concurrent.futures
import datetime as dt
import json
import math
import os
import re
import shutil
import threading
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds, transform_geom
from rasterio.windows import Window, bounds as window_bounds, from_bounds
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from shapely.prepared import prep

from collect_public import BBOX, RAW, ROOT, get, write

BOUNDS = tuple(map(float, BBOX.split(",")))
AREA = box(*BOUNDS)
IMAGERY = Path(os.environ.get("OILTANKS_RASTER_STORE", ROOT / "data" / "rasters"))
PC = "https://planetarycomputer.microsoft.com"
BANDS = ("blue", "green", "red", "nir", "scl")
CONFIG = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.jp2",
    "GDAL_HTTP_CONNECTTIMEOUT": "15",
    "GDAL_HTTP_TIMEOUT": "45",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_CACHEMAX": 100_000_000,
}
if Path("/etc/ssl/cert.pem").exists():
    CONFIG["GDAL_CURL_CA_BUNDLE"] = "/etc/ssl/cert.pem"
SAS = {}
SAS_LOCK = threading.Lock()


def candidate_footprint(path, buffer_m):
    features = json.loads(path.read_bytes())["features"]
    geometries = [shape(transform_geom("EPSG:4326", "EPSG:32614", item["geometry"])) for item in features]
    metric = unary_union(geometries).buffer(buffer_m)
    geographic = shape(transform_geom("EPSG:32614", "EPSG:4326", mapping(metric)))
    return metric, geographic


def acquisition_key(item):
    properties = item["properties"]
    return (properties.get("platform"), properties.get("grid:code"), properties["datetime"][:10],
            properties.get("sat:relative_orbit"))


def level2a_items(catalog_root=RAW / "stac", area=AREA):
    for collection in ("sentinel-2-l2a", "sentinel-2-c1-l2a"):
        for path in sorted((catalog_root / collection).glob("*.geojson")):
            for item in json.loads(path.read_text())["features"]:
                if not all(item["assets"].get(band, {}).get("href", "").startswith("https://")
                           for band in BANDS):
                    continue
                if shape(item["geometry"]).intersects(area):
                    yield item


def canonical_items(catalog_root=RAW / "stac", area=AREA):
    items = {acquisition_key(item): item for item in level2a_items(catalog_root, area)}
    return sorted(items.values(), key=lambda item: item["properties"]["datetime"])


def earliest_alternate_items(catalog_root=RAW / "stac", area=AREA):
    ordered = sorted(level2a_items(catalog_root, area), key=lambda item: item["properties"]["s2:generation_time"], reverse=True)
    earliest = {acquisition_key(item): item for item in ordered}
    alternatives = [earliest[acquisition_key(item)] for item in canonical_items(catalog_root, area)
                    if earliest[acquisition_key(item)]["id"] != item["id"]]
    return sorted(alternatives, key=lambda item: item["properties"]["datetime"])


def clip(url, destination, bounds, footprint=None, *, replace=False, mask_to_footprint=False):
    if destination.is_file() and not replace:
        with rasterio.open(destination) as existing:
            return existing.width, existing.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{os.getpid()}.part.tif")
    with rasterio.Env(**CONFIG), rasterio.open(url) as source:
        bounds = transform_bounds("EPSG:4326", source.crs, *bounds)
        left, bottom, right, top = bounds
        left, bottom = max(left, source.bounds.left), max(bottom, source.bounds.bottom)
        right, top = min(right, source.bounds.right), min(top, source.bounds.top)
        if left >= right or bottom >= top:
            return 0, 0
        fractional = from_bounds(left, bottom, right, top, source.transform)
        col, row = max(0, math.floor(fractional.col_off)), max(0, math.floor(fractional.row_off))
        right = min(source.width, math.ceil(fractional.col_off + fractional.width))
        bottom = min(source.height, math.ceil(fractional.row_off + fractional.height))
        window = Window(col, row, right - col, bottom - row)
        profile = source.profile.copy()
        profile.update(driver="GTiff", width=int(window.width), height=int(window.height),
                       transform=source.window_transform(window), tiled=True, compress="deflate",
                       blockxsize=256, blockysize=256, predictor=2)
        if str(profile.get("photometric", "")).lower() == "ycbcr":
            profile["photometric"] = "RGB"
        if footprint is not None:
            profile.update(sparse_ok=True)
            footprint_geometry = transform_geom("EPSG:32614", source.crs, mapping(footprint))
            projected = prep(shape(footprint_geometry))
        with rasterio.open(temporary, "w", **profile) as output:
            if footprint is not None:
                for _, block in output.block_windows(1):
                    read_window = Window(col + block.col_off, row + block.row_off, block.width, block.height)
                    if projected.intersects(box(*window_bounds(read_window, source.transform))):
                        pixels = source.read(window=read_window)
                        valid = source.dataset_mask(window=read_window)
                        if mask_to_footprint:
                            keep = geometry_mask([footprint_geometry], out_shape=valid.shape,
                                                 transform=source.window_transform(read_window), invert=True, all_touched=True)
                            pixels[:, ~keep] = source.nodata or 0
                            valid[~keep] = 0
                        output.write(pixels, window=block)
                        output.write_mask(valid, window=block)
            elif source.count * window.width * window.height < 8_000_000:
                output.write(source.read(window=window))
            else:
                for offset in range(0, int(window.height), 512):
                    height = min(512, int(window.height) - offset)
                    read_window = Window(col, row + offset, window.width, height)
                    write_window = Window(0, offset, window.width, height)
                    output.write(source.read(window=read_window), window=write_window)
            output.scales = source.scales
            output.offsets = source.offsets
    temporary.replace(destination)
    return int(window.width), int(window.height)


def sentinel(item, bounds, footprint=None):
    identity = item["id"]
    captured = item["properties"]["datetime"]
    directory = IMAGERY / "sentinel2" / captured[:4] / identity
    dimensions = {band: clip(item["assets"][band]["href"], directory / f"{band}.tif", bounds, footprint) for band in BANDS}
    if not all(width and height for width, height in dimensions.values()):
        raise ValueError("At least one band has no overlap with the research AOI")
    write(directory / "item.json", json.dumps(item).encode())
    return dimensions


def naip_items(bbox=BBOX):
    cache_name = "planetary_computer.geojson" if bbox == BBOX else f"planetary_computer_{bbox}.geojson"
    cached = RAW / "naip" / cache_name
    if cached.is_file():
        return json.loads(cached.read_bytes())["features"]
    url = PC + "/api/stac/v1/search?" + urlencode({"collections": "naip", "bbox": bbox, "limit": 100})
    features = []
    while url:
        page = json.loads(get(url))
        features.extend(page["features"])
        url = next((link["href"] for link in page["links"] if link["rel"] == "next"), None)
    write(cached, json.dumps({"type": "FeatureCollection", "features": features}).encode())
    return features


def sign(url):
    location = urlsplit(url)
    account = location.hostname.split(".")[0]
    container = location.path.split("/")[1]
    key = (account, container)
    with SAS_LOCK:
        token = SAS.get(key)
        now = dt.datetime.now(dt.timezone.utc)
        if not token or dt.datetime.fromisoformat(token["msft:expiry"].replace("Z", "+00:00")) < now + dt.timedelta(minutes=5):
            token = json.loads(get(PC + f"/api/sas/v1/token/{account}/{container}"))
            SAS[key] = token
    return url + "?" + token["token"]


def naip(item, bounds, footprint=None):
    identity = item["id"]
    captured = item["properties"]["datetime"]
    directory = IMAGERY / "naip" / captured[:4] / identity
    url = item["assets"]["image"]["href"]
    dimensions = clip(sign(url), directory / "image.tif", bounds, footprint)
    if all(dimensions):
        write(directory / "item.json", json.dumps(item).encode())
    return dimensions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["sentinel2", "sentinel2-original", "naip"])
    parser.add_argument("--start", default="2017-01-01")
    parser.add_argument("--end", default="2026-11-11")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bbox", default=BBOX, help="west,south,east,north in EPSG:4326")
    parser.add_argument("--min-free-gib", type=float, default=3)
    parser.add_argument("--tank-buffer-m", type=float)
    parser.add_argument("--candidate-layer", type=Path, help="Alternative candidate GeoJSON for a tank-buffer crop")
    parser.add_argument("--extended-catalog", action="store_true", help="Select Sentinel-2 from the supplemental search")
    parser.add_argument("--exclude-core-catalog", action="store_true", help="Collect only acquisitions absent from the core search")
    args = parser.parse_args()
    if not IMAGERY.is_dir():
        raise RuntimeError("Public raster storage is not mounted")
    bounds = tuple(map(float, args.bbox.split(",")))
    if len(bounds) != 4 or not (-180 <= bounds[0] < bounds[2] <= 180 and -90 <= bounds[1] < bounds[3] <= 90):
        parser.error("--bbox must contain west,south,east,north in increasing order")
    if args.candidate_layer and args.tank_buffer_m is None:
        parser.error("--candidate-layer requires --tank-buffer-m")
    if args.extended_catalog and (not args.candidate_layer or args.source == "naip"):
        parser.error("--extended-catalog requires a Sentinel-2 source and --candidate-layer")
    if args.exclude_core_catalog and not args.extended_catalog:
        parser.error("--exclude-core-catalog requires --extended-catalog")
    requested_area = box(*bounds)
    footprint = None
    if args.tank_buffer_m is not None:
        candidates = args.candidate_layer or ROOT / "data" / "derived" / "osm_tank_candidates.geojson"
        footprint, requested_area = candidate_footprint(candidates, args.tank_buffer_m)
    if args.source == "naip":
        if args.candidate_layer:
            bounds = requested_area.bounds
        catalog_bbox = ",".join(f"{value:.6f}" for value in bounds) if args.candidate_layer else BBOX
        items, function = naip_items(catalog_bbox), naip
    else:
        select = earliest_alternate_items if args.source == "sentinel2-original" else canonical_items
        items = select(RAW / "stac_supplemental", requested_area) if args.extended_catalog else select()
        if args.exclude_core_catalog:
            core_keys = {acquisition_key(item) for item in select()}
            items = [item for item in items if acquisition_key(item) not in core_keys]
        function = sentinel
    selected = [item for item in items
                if args.start <= item["properties"]["datetime"][:10] <= args.end
                and shape(item["geometry"]).intersects(requested_area)]
    if args.limit is not None:
        selected = selected[:args.limit]
    if not selected:
        raise RuntimeError("No items in the requested time range")
    suffix = "_extended" if args.extended_catalog else ""
    log = ROOT / "data" / "logs" / f"{args.source}{suffix}_{args.start}_{args.end}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"{args.source}: {len(selected)} items", flush=True)
    free_floor = args.min_free_gib * 1024 ** 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor, log.open("a") as stream:
        for i in range(0, len(selected), args.workers):
            if shutil.disk_usage(IMAGERY).free < free_floor:
                print("Stopped at the disk-space floor. Completed files can be reused.", flush=True)
                break
            pending = {executor.submit(function, item,
                                       requested_area.intersection(shape(item["geometry"])).bounds if args.extended_catalog else bounds,
                                       footprint): item["id"]
                       for item in selected[i:i + args.workers]}
            for future in concurrent.futures.as_completed(pending):
                row = {"id": pending[future], "bbox": bounds, "storage": str(IMAGERY), "tank_buffer_m": args.tank_buffer_m}
                try:
                    width_data = future.result()
                    if isinstance(width_data, tuple) and not all(width_data):
                        row["error"] = "Raster bounds do not overlap the requested area"
                    else:
                        row["dimensions"] = width_data
                except Exception as error:
                    row["error"] = re.sub(r"https?://[^\s]+", "[remote URL]", str(error))
                print(json.dumps(row), file=stream, flush=True)
                if len(selected) <= 20 or "error" in row:
                    print(json.dumps(row), flush=True)



if __name__ == "__main__":
    main()
