"""Stage Landsat B8 and independent multispectral context over the tank footprints."""
import argparse
import datetime as dt
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import rasterio
from shapely.geometry import shape

from collect_imagery import PC, candidate_footprint, clip, sign
from collect_public import RAW, ROOT, get, write
from collect_s2_l1c import crc32c_base64

LANDSAT = RAW / "landsat"
BBOX = "-96.90,35.68,-96.52,36.12"
USGS = "https://landsatlook.usgs.gov/stac-server/search?"
BUCKET = "gcp-public-data-landsat"
GCS_INDEX = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?"
GCS_OBJECT = f"https://storage.googleapis.com/{BUCKET}/"
PAN_PRODUCT = re.compile(r"LC08_L1(?:TP|GT|GS)_(\d{6})_(\d{8})_(\d{8})_01_(T1|T2|RT)")
PAN_ASSETS = {"pan": ("B8.TIF", 15), "bqa": ("BQA.TIF", 30)}
CONTEXT_ASSETS = ("blue", "green", "red", "nir08", "swir16", "swir22", "qa_pixel", "qa_radsat")


def catalog(year, route):
    endpoint, collection = ((USGS, "landsat-c2l1") if route == "usgs" else
                            (PC + "/api/stac/v1/search?", "landsat-c2-l2"))
    path = LANDSAT / f"{route}_catalog" / f"{year}.geojson"
    if path.is_file():
        return json.loads(path.read_bytes())["features"]
    end = "2026-11-11" if year == 2026 else f"{year}-12-31"
    url = endpoint + urlencode({"collections": collection, "bbox": BBOX, "limit": 100,
        "datetime": f"{year}-01-01T00:00:00Z/{end}T23:59:59.999999Z"})
    pages, features, matched = [], {}, None
    while url:
        if url in pages:
            raise ValueError(f"Repeated Landsat catalog page for {route} {year}")
        page = json.loads(get(url))
        if not pages:
            matched = page.get("numberMatched", page.get("context", {}).get("matched"))
        pages.append(url)
        features.update((item["id"], item) for item in page["features"])
        url = next((link["href"] for link in page.get("links", []) if link["rel"] == "next"), None)
    if matched is not None and matched != len(features):
        raise ValueError(f"Incomplete Landsat catalog for {route} {year}: {len(features)} of {matched}")
    write(path, json.dumps({"type": "FeatureCollection", "features": list(features.values()),
          "page_urls": pages, "number_matched": matched}, separators=(",", ":")).encode())
    print(route, year, len(features), "catalog records", flush=True)
    return list(features.values())


def gcs_index(prefix, path, directories=False):
    if path.is_file():
        return json.loads(path.read_bytes())["records"]
    params = {"prefix": prefix, "maxResults": 1000,
              "fields": "prefixes,nextPageToken" if directories else "items(name,size,timeCreated,crc32c),nextPageToken"}
    if directories:
        params["delimiter"] = "/"
    result, pages = [], []
    while True:
        url = GCS_INDEX + urlencode(params)
        data = json.loads(get(url))
        pages.append(url)
        result.extend(data.get("prefixes" if directories else "items", []))
        if not data.get("nextPageToken"):
            break
        params["pageToken"] = data["nextPageToken"]
    write(path, json.dumps({"page_urls": pages, "records": result}, separators=(",", ":")).encode())
    return result


def daylight(item):
    p = item["properties"]
    return p["platform"].lower().replace("_", "-") in {"landsat-8", "landsat-9"} and p["view:sun_elevation"] > 0


def source_coverage():
    derived = ROOT / "data" / "derived"
    core, core_area = candidate_footprint(derived / "osm_tank_candidates.geojson", 120)
    supplemental, supplemental_area = candidate_footprint(derived / "supplemental_tank_candidates.geojson", 120)
    return core.union(supplemental), core_area.union(supplemental_area)


def scene_key(item):
    p = item["properties"]
    return p["landsat:wrs_path"] + p["landsat:wrs_row"], p["datetime"][:10].replace("-", "")


def identity(platform, grid, acquired):
    return f"{platform.lower().replace('_', '-')}:{grid}:{acquired[:10]}"


def pan_jobs(items, area, years, pool):
    references = {scene_key(item): item for item in items if item["properties"]["platform"] == "LANDSAT_8"
                  and daylight(item) and shape(item["geometry"]).intersects(area)}
    grids = sorted({grid for grid, _ in references})

    def scan(job):
        grid, year = job
        base = f"LC08/01/{grid[:3]}/{grid[3:]}/"
        path = LANDSAT / "legacy_gcs_index" / f"{grid}_{year}.json"
        return gcs_index_year(base, grid, year, path)

    variants = {}
    order = {"T1": 0, "T2": 1, "RT": 2}
    for rows in pool.map(scan, ((grid, year) for grid in grids for year in years)):
        for prefix in rows:
            product = prefix.rstrip("/").rsplit("/", 1)[-1]
            match = PAN_PRODUCT.fullmatch(product)
            if not match:
                continue
            grid, acquired, processed, tier = match.groups()
            key = grid, acquired
            if key in references:
                variants.setdefault(key, []).append((order[tier], processed, product, prefix))
    return [(min(variants[key])[-1], references[key]) for key in sorted(variants, key=lambda key: (key[1], key[0]))]


def gcs_index_year(base, grid, year, path):
    rows = []
    for correction in ("TP", "GT", "GS"):
        typed_path = path.with_name(path.stem + f"_{correction}.json")
        rows.extend(gcs_index(base + f"LC08_L1{correction}_{grid}_{year}", typed_path, True))
    return rows


def check_clip(path, resolution, dimensions):
    with rasterio.open(path) as raster:
        if raster.count != 1 or raster.dtypes != ("uint16",) or raster.res != (resolution, resolution):
            raise ValueError(f"Unexpected Landsat pixel type or grid: {path}")
        if (raster.width, raster.height) != tuple(dimensions) or not all(dimensions):
            raise ValueError(f"Missing Landsat raster pixels: {path}")
        raster.read(1)
        raster.read_masks(1)


def pan_record(job, area, footprint):
    prefix, reference = job
    product = prefix.rstrip("/").rsplit("/", 1)[-1]
    year = product.split("_")[3][:4]
    directory = LANDSAT / "legacy_pan" / year / product
    records = {obj["name"].rsplit("/", 1)[-1]: obj for obj in gcs_index(prefix, directory / "objects.json")}
    source = directory / "MTL.txt"
    meta = records[f"{product}_MTL.txt"]
    payload = get(GCS_OBJECT + meta["name"], source)
    if crc32c_base64(payload) != meta["crc32c"]:
        raise ValueError(f"Wrong source checksum for {product}")
    mtl = dict(re.findall(r"^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$", payload.decode(), re.MULTILINE))
    if mtl["LANDSAT_PRODUCT_ID"].strip('"') != product:
        raise ValueError(f"Wrong Landsat product metadata for {product}")
    captured = mtl["DATE_ACQUIRED"] + "T" + mtl["SCENE_CENTER_TIME"].strip('"')
    if abs((dt.datetime.fromisoformat(captured) - dt.datetime.fromisoformat(reference["properties"]["datetime"])).total_seconds()) > 1:
        raise ValueError(f"Collection 1 and Collection 2 acquisition clocks disagree for {product}")
    assets = {}
    for label, (suffix, resolution) in PAN_ASSETS.items():
        obj = records[f"{product}_{suffix}"]
        path = directory / f"{label}.tif"
        dimensions = clip(GCS_OBJECT + obj["name"], path, area.bounds, footprint, mask_to_footprint=True)
        check_clip(path, resolution, dimensions)
        assets[label] = {**obj, "local_file": path.name, "clip_dimensions": dimensions}
    clocks = [obj["timeCreated"] for obj in (*assets.values(), meta)]
    record = {"product_id": product, "captured_at_utc": captured, "collection2_reference_id": reference["id"],
              "archive_assets_created_by_utc": max(clocks, key=dt.datetime.fromisoformat),
              "sun_azimuth_deg": float(mtl["SUN_AZIMUTH"]), "sun_elevation_deg": float(mtl["SUN_ELEVATION"]),
              "reflectance_mult_b8": float(mtl["REFLECTANCE_MULT_BAND_8"]),
              "reflectance_add_b8": float(mtl["REFLECTANCE_ADD_BAND_8"]), "assets": assets}
    write(directory / "item.json", json.dumps(record, indent=2).encode())
    return product


def context_record(item, area, footprint):
    props = item["properties"]
    directory = LANDSAT / "collection2_context" / props["datetime"][:4] / item["id"]
    meta_path = directory / "MTL.json"
    xml_path = directory / "MTL.xml"
    source_label = "mtl.xml" if xml_path.is_file() else "mtl.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_bytes())
    else:
        try:
            meta = json.loads(get(sign(item["assets"]["mtl.json"]["href"])))
        except HTTPError as error:
            if error.code != 404:
                raise
            source_label = "mtl.xml"
            data = get(sign(item["assets"][source_label]["href"]),xml_path)
            root = ET.fromstring(data)
            meta = {root.tag:{group.tag:{field.tag:field.text for field in group} for group in root}}
    source = item["assets"][source_label]["href"]
    contents = meta["LANDSAT_METADATA_FILE"]["PRODUCT_CONTENTS"]
    if contents["LANDSAT_PRODUCT_ID"] not in source or contents["LANDSAT_PRODUCT_ID"].split("_")[3] != props["datetime"][:10].replace("-", ""):
        raise ValueError(f"Wrong Landsat Collection 2 metadata for {item['id']}")
    if not meta_path.is_file():
        write(meta_path, json.dumps(meta, indent=2).encode())
    for label in CONTEXT_ASSETS:
        path = directory / f"{label}.tif"
        href = item["assets"][label]["href"]
        dimensions = clip(sign(href) if not path.is_file() else href, path, area.bounds, footprint, mask_to_footprint=True)
        check_clip(path, 30, dimensions)
    write(directory / "item.json", json.dumps({**item,"local_metadata_asset":source_label}, separators=(",", ":")).encode())
    return item["id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=("catalog", "pan", "context"))
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retry-errors-from",type=Path)
    args = parser.parse_args()
    years = tuple(range(args.start_year, args.end_year+1))
    if not years or args.workers < 1:
        parser.error("Use an increasing year interval and a positive worker count")
    if args.retry_errors_from and args.route!="context":
        parser.error("--retry-errors-from applies to Collection 2 context")
    footprint, area = source_coverage()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        route = "usgs" if args.route == "pan" else "pc"
        rows = [item for items in pool.map(lambda year: catalog(year, route), years) for item in items]
        if args.route == "catalog":
            list(pool.map(lambda year: catalog(year, "usgs"), years))
            return
        if args.route == "pan":
            jobs = pan_jobs(rows, area, years, pool)
            collect = pan_record
        else:
            jobs = sorted((item for item in rows if daylight(item) and shape(item["geometry"]).intersects(area)),
                          key=lambda item: item["properties"]["datetime"])
            if args.retry_errors_from:
                last = {row["product_id"]:row["status"] for row in map(json.loads,args.retry_errors_from.read_text().splitlines())}
                jobs = [item for item in jobs if last.get(item["id"])=="error"]
            collect = context_record
        print(args.route, len(jobs), "eligible scenes", flush=True)
        failures = 0
        suffix = "_retry" if args.retry_errors_from else ""
        log = ROOT / "data" / "logs" / f"landsat_{args.route}_{years[0]}_{years[-1]}{suffix}.jsonl"
        with log.open("a") as stream:
            for offset in range(0, len(jobs), args.workers):
                if shutil.disk_usage(LANDSAT).free < 2.5 * 1024**3:
                    raise RuntimeError("Landsat download paused at the working disk reserve")
                pending = {pool.submit(collect, job, area, footprint): job for job in jobs[offset:offset+args.workers]}
                for future in as_completed(pending):
                    job = pending[future]
                    identity = job[0].rstrip("/").rsplit("/", 1)[-1] if args.route == "pan" else job["id"]
                    try:
                        future.result()
                    except Exception as error:
                        failures += 1
                        detail = re.sub(r"https?://\S+", "[archive URL]", str(error))
                        status = {"product_id": identity, "status": "error", "detail": f"{type(error).__name__}: {detail}"}
                    else:
                        status = {"product_id": identity, "status": "ready"}
                    print(json.dumps(status), file=stream, flush=True)
                    print(json.dumps(status), flush=True)
        if failures:
            raise RuntimeError(f"{failures} Landsat scenes could not be staged. See {log.name}")


if __name__ == "__main__":
    main()
