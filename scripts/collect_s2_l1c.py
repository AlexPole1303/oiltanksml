"""Retrieve the earliest matching L1C products in the public GCS archive."""
import argparse
import base64
import concurrent.futures
import datetime as dt
import gzip
import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote, urlencode

import rasterio
from shapely.geometry import shape

from collect_imagery import candidate_footprint, canonical_items, clip
from collect_public import RAW, ROOT, get, write

BUCKET = "gcp-public-data-sentinel-2"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
OBJECTS = f"https://storage.googleapis.com/{BUCKET}/"
OUT = RAW / "sentinel2_l1c_original"
URI = re.compile(r"(S2[A-Z])_MSIL[12][CA]_(\d{8}T\d{6})_N(\d{4})_R(\d{3})_T(\d{2}[A-Z]{3})_(\d{8}T\d{6})\.SAFE")
CHANNELS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08"}
BOUNDS = (-96.78, 35.91, -96.70, 36.03)


def parse_product(name):
    sensor, sensing, baseline, orbit, tile, generated = URI.fullmatch(name).groups()
    return (sensor, sensing, orbit, tile), baseline, generated


def query_archive(prefix, *, directories=False, cache=None):
    if cache is not None and cache.is_file():
        return json.loads(cache.read_bytes())
    options = {"prefix": prefix, "maxResults": 1000,
               "fields": "prefixes,nextPageToken" if directories else "items(name,size,timeCreated,updated,crc32c),nextPageToken"}
    if directories:
        options["delimiter"] = "/"
    entries = []
    while True:
        data = json.loads(get(API + "?" + urlencode(options)))
        entries.extend(data.get("prefixes" if directories else "items", []))
        if not data.get("nextPageToken"):
            if cache is not None:
                write(cache, json.dumps(entries).encode())
            return entries
        options["pageToken"] = data["nextPageToken"]


def indexed_products(tile, sensor, year):
    directory = f"tiles/{tile[:2]}/{tile[2]}/{tile[3:]}"
    prefix = f"{directory}/{sensor}_MSIL1C_{year}"
    cache = OUT / "indexes" / f"{tile}_{sensor}_{year}.json"
    return query_archive(prefix, directories=True, cache=cache)


def earliest_products(items, workers):
    jobs = sorted({(key[3], key[0], key[1][:4]) for item in items
                   for key, _, _ in [parse_product(item["properties"]["s2:product_uri"])]})
    versions = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for rows in executor.map(lambda job: indexed_products(*job), jobs):
            for prefix in rows:
                product = prefix.rstrip("/").rsplit("/", 1)[-1]
                if not URI.fullmatch(product):
                    continue
                key, _, generated = parse_product(product)
                entry = generated, product, prefix
                if key not in versions or entry < versions[key]:
                    versions[key] = entry
    return versions


@lru_cache(maxsize=1)
def crc32c_table():
    table = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value)
    return table


def crc32c_base64(data):
    table = crc32c_table()
    value = 0xFFFFFFFF
    for byte in data:
        value = table[(value ^ byte) & 0xFF] ^ (value >> 8)
    return base64.b64encode((value ^ 0xFFFFFFFF).to_bytes(4, "big")).decode()


def source_metadata_path(path):
    packed = path.with_suffix(path.suffix + ".gz")
    return packed if not path.is_file() and packed.is_file() else path


def source_metadata_bytes(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == ".gz" else data


def download_source_metadata(url, path, pack_text):
    path = source_metadata_path(path)
    if path.is_file():
        return source_metadata_bytes(path)
    if not pack_text or path.suffix not in {".xml", ".gml"}:
        return get(url, path)
    data = get(url)
    packed = path.with_suffix(path.suffix + ".gz")
    write(packed, gzip.compress(data, compresslevel=6, mtime=0))
    if source_metadata_bytes(packed) != data:
        raise ValueError(f"Packed source metadata readback differs in {packed}")
    return data


def local_granule_files(directory, canonical_id, product, record=None):
    if record is None:
        record = json.loads((directory / "item.json").read_bytes())
    if record["canonical_l2a_id"] != canonical_id or record["product_name"] != product:
        raise ValueError(f"Mismatched L1C product in {directory}")
    first_grid = None
    for band in CHANNELS:
        path = directory / f"{band}.tif"
        with rasterio.open(path) as src:
            dimensions = (src.width, src.height)
            grid = (src.transform, src.crs)
            if dimensions != tuple(record["assets"][band]["clip_dimensions"]) or src.count != 1:
                raise ValueError(f"Unexpected dimensions for {path}")
            if first_grid is not None and grid != first_grid:
                raise ValueError(f"Unexpected band grid in {path}")
            first_grid = grid
            src.read(1)
            src.read_masks(1)
    for item in record["granule_metadata"]:
        filename = item["local_name"] if "local_name" in item else Path(item["name"]).name
        path = source_metadata_path(directory / filename)
        data = source_metadata_bytes(path)
        if "crc32c" in item:
            matched = crc32c_base64(data) == item["crc32c"]
        else:
            matched = hashlib.md5(data, usedforsecurity=False).hexdigest() == item["etag"]
        if len(data) != int(item["size"]) or not matched:
            raise ValueError(f"Source metadata checksum mismatch for {path}")
    product_path = source_metadata_path(directory / "MTD_MSIL1C.xml")
    ET.fromstring(source_metadata_bytes(product_path))
    return record


def download(canonical, versions, out, bounds, footprint, *, verify=False, min_free_bytes=0, pack_text=False,
             object_index_store=OUT/"object_indexes"):
    properties = canonical["properties"]
    key, _, _ = parse_product(properties["s2:product_uri"])
    dated = properties["datetime"]
    earliest = versions.get(key)
    if earliest is None:
        return {"id": canonical["id"], "status": "no_matching_public_l1c_product"}
    _, product, prefix = earliest
    captured = dt.datetime.fromisoformat(dated)
    destination = out / "granules" / dated[:4] / canonical["id"]
    manifest = destination / "item.json"
    if manifest.is_file() and all((destination / (band + ".tif")).is_file() for band in CHANNELS):
        record = (local_granule_files(destination, canonical["id"], product) if verify
                  else json.loads(manifest.read_bytes()))
        if record["product_name"] == product:
            return {"id": canonical["id"], "status": "ready", "readback_verified": verify,
                    "generation_lag_hours": record["generation_lag_hours"]}
    if min_free_bytes and shutil.disk_usage(out).free < min_free_bytes:
        return {"id": canonical["id"], "status": "waiting_for_storage"}
    objects = query_archive(prefix, cache=object_index_store/f"{product}.json")
    members = [asset for asset in objects if "/GRANULE/" in asset["name"]]
    assets = {}
    for channel, band in CHANNELS.items():
        options = [item for item in members if item["name"].endswith(f"_{band}.jp2") and "/IMG_DATA/" in item["name"]]
        if len(options) != 1:
            raise ValueError(f"Expected one {band} file in {product}, found {len(options)}")
        asset = options[0]
        url = OBJECTS + quote(asset["name"], safe="/")
        dimensions = clip(url, destination / f"{channel}.tif", bounds, footprint)
        if not all(dimensions):
            raise ValueError(f"No AOI pixels for {band} in {product}")
        assets[channel] = {**asset, "url": url, "clip_dimensions": dimensions}
    metadata = [item for item in members if item["name"].endswith(("/MTD_TL.xml", "/MSK_CLOUDS_B00.gml",
                                                                   "/MSK_CLASSI_B00.jp2", "/MSK_DETFOO_B04.gml",
                                                                   "/MSK_DETFOO_B04.jp2"))]
    for item in metadata:
        basename = item["name"].rsplit("/", 1)[-1]
        download_source_metadata(OBJECTS + quote(item["name"], safe="/"), destination / basename, pack_text)
    product_metadata = OBJECTS + quote(prefix + "MTD_MSIL1C.xml", safe="/")
    xml = ET.fromstring(download_source_metadata(product_metadata, destination / "MTD_MSIL1C.xml", pack_text))
    reported = next(node.text for node in xml.iter() if node.tag.rsplit("}", 1)[-1] == "GENERATION_TIME")
    generation = dt.datetime.fromisoformat(reported)
    archive_files = list(assets.values()) + metadata
    archive_ready = max(dt.datetime.fromisoformat(item["timeCreated"]) for item in archive_files)
    if archive_ready < captured:
        raise ValueError("Public archive object dates predate acquisition")
    lag = round((generation - captured).total_seconds() / 3600, 6)
    record = {"canonical_l2a_id": canonical["id"], "product_name": product,
          "captured_at_utc": dated, "generated_at_utc": generation.isoformat(), "generation_lag_hours": lag,
          "reported_generation_before_tile_sensing": generation < captured,
          "band_and_granule_metadata_objects_created_by_utc": archive_ready.isoformat(),
          "processing_level": "L1C", "assets": assets, "granule_metadata": metadata,
          "product_metadata_url": product_metadata, "archive": OBJECTS}
    if verify:
        local_granule_files(destination, canonical["id"], product, record)
    payload = json.dumps(record, indent=2).encode()
    write(manifest, payload)
    if verify and manifest.read_bytes() != payload:
        raise ValueError(f"L1C manifest readback differs in {destination}")
    return {"id": canonical["id"], "status": "ready", "readback_verified": verify,
            "source": "public_archive_download", "generation_lag_hours": lag}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--retry-recorded-errors", action="store_true")
    area_options = parser.add_mutually_exclusive_group()
    area_options.add_argument("--supplemental", action="store_true")
    area_options.add_argument("--combined", action="store_true")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--capture-day", action="append", type=dt.date.fromisoformat,
                        help="Restrict the year selection to the specified UTC acquisition day; repeat for more days")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-local", action="store_true")
    parser.add_argument("--pack-source-text", action="store_true",
                        help="Store newly downloaded XML and GML in lossless gzip files")
    parser.add_argument("--minimum-free-gib", type=float, default=0)
    parser.add_argument("--log-name")
    args = parser.parse_args()
    # Limit nested JPEG2000 decoding while processing multiple products concurrently.
    os.environ.setdefault("GDAL_NUM_THREADS", "1")
    os.environ.setdefault("OPJ_NUM_THREADS", "1")
    items, footprint, area = canonical_items(), None, None
    derived = ROOT / "data" / "derived"
    out, area_label = OUT, "original"
    if args.supplemental or args.combined:
        layer = derived / "supplemental_tank_candidates.geojson"
        footprint, area = candidate_footprint(layer, 120)
        if args.combined:
            core_footprint, core_area = candidate_footprint(derived / "osm_tank_candidates.geojson", 120)
            footprint, area = footprint.union(core_footprint), area.union(core_area)
        items = canonical_items(RAW / "stac_supplemental", area)
        area_label = "combined" if args.combined else "supplemental"
        out = OUT / area_label
    if args.output is not None:
        out = args.output
    if args.start_year > args.end_year:
        parser.error("--start-year must be at or before --end-year")
    requested_days = {day.isoformat() for day in args.capture_day or []}
    candidates = [item for item in items
                  if args.start_year <= int(item["properties"]["datetime"][:4]) <= args.end_year
                  and (not requested_days or item["properties"]["datetime"][:10] in requested_days)]
    if requested_days - {item["properties"]["datetime"][:10] for item in candidates}:
        parser.error("At least one requested capture day has no item in the selected archive and year range")
    period = "" if (args.start_year, args.end_year) == (2017, 2018) else f"_{args.start_year}_{args.end_year}"
    log_name = args.log_name or f"sentinel2_l1c_{area_label}{period}.jsonl"
    log = ROOT / "data" / "logs" / log_name
    if args.retry_recorded_errors:
        latest = {row["id"]: row for row in map(json.loads, log.read_text().splitlines())}
        candidates = [item for item in candidates if latest.get(item["id"], {}).get("status") == "error"]
    versions = earliest_products(candidates, args.workers)
    print(f"{len(candidates)} L2A acquisitions, {sum(parse_product(item['properties']['s2:product_uri'])[0] in versions for item in candidates)} matching L1C products", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, item, versions, out,
                               area.intersection(shape(item["geometry"])).bounds if area is not None else BOUNDS,
                               footprint, verify=args.verify_local,
                               min_free_bytes=int(args.minimum_free_gib * 1024 ** 3),
                               pack_text=args.pack_source_text): item["id"] for item in candidates}
        for future in concurrent.futures.as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                row = {"id": futures[future], "status": "error", "error": str(error)}
                print(json.dumps(row), flush=True)
            print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    main()
