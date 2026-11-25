"""Cache Oklahoma's 2025 orthophotos that intersect the project area."""
import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlencode, urlsplit

import rasterio
from pyproj import Transformer
from shapely import transform, union_all
from shapely.geometry import shape

from collect_imagery import clip
from collect_public import RAW, ROOT, get, write

TYPE_NAME = "ogi:aerial2025_tile_index_rgb75_4326"
BBOX = "-96.79,35.91,-96.69,36.03"
TARGET = RAW / "ok_aerial_2025"


def catalog(bbox=BBOX):
    url = "https://okmaps.org/geoserver/wfs?" + urlencode({
        "service": "WFS", "version": "1.0.0", "request": "GetFeature", "typeName": TYPE_NAME,
        "outputFormat": "application/json", "bbox": bbox, "srsName": "EPSG:4326", "maxFeatures": 1000,
    })
    index = (TARGET / "tile_index.geojson" if bbox == BBOX else
             TARGET / "additional_indexes" / f"{bbox}.geojson")
    source = json.loads(get(url, index))
    if not source.get("features") or len(source["features"]) != source.get("totalFeatures"):
        raise ValueError("Empty or incomplete state imagery tile index")
    write(index.with_suffix(".source"), (url + "\n").encode())
    return source["features"]


def download(item):
    data = item["properties"]
    identity = data["strtile"]
    directory = TARGET / "tiles" / identity
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "image.tif"
    world = directory / "image.tfw"
    for key in ("ftp_tif", "ftp_tfw"):
        if urlsplit(data[key]).hostname != "okmaps.org":
            raise ValueError("Tile index points outside the state imagery host")
    if not image.is_file():
        temp = directory / "image.part"
        proc = subprocess.run(["/usr/bin/curl", "-fLsS", "--max-time", "180", "-C", "-",
                               "-o", str(temp), data["ftp_tif"]], capture_output=True, text=True)
        if proc.returncode:
            raise OSError(proc.stderr.strip())
        temp.replace(image)
    get(data["ftp_tfw"], world)
    with rasterio.open(image) as dataset:
        dx, dy = dataset.res
        if abs(dx - 0.15) > 0.001 or abs(dy - 0.15) > 0.001 or dataset.count != 3:
            raise ValueError(f"Unexpected raster layout: {dataset.res}, {dataset.count} bands")
        crs = dataset.crs.to_string() if dataset.crs else "EPSG:6350 (state metadata)"
        size = [dataset.width, dataset.height]
        dataset.read(1, window=((0, 1), (0, 1)))
    write(directory / "item.json", json.dumps(item, ensure_ascii=False).encode())
    return {"id": identity, "dimensions": size, "crs": crs, "bytes": image.stat().st_size}


def tank_tiles(items):
    path = ROOT / "data" / "derived" / "osm_tank_candidates.geojson"
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:32614", "EPSG:4326", always_xy=True)
    tanks = [shape(item["geometry"]) for item in json.loads(path.read_bytes())["features"]]
    footprint = union_all([transform(tank, to_utm.transform, interleaved=False) for tank in tanks]).buffer(120)
    area = transform(footprint, to_geo.transform, interleaved=False)
    selected = []
    for item in items:
        tile = shape(item["geometry"])
        overlap = tile.intersection(area)
        if not overlap.is_empty:
            count = sum(tile.covers(tank.centroid) for tank in tanks)
            selected.append((count, item, overlap.bounds, footprint))
    return sorted(selected, key=lambda row: (-row[0], row[1]["properties"]["strtile"]))


def clip_tile(item, bounds, footprint):
    properties = item["properties"]
    identity = properties["strtile"]
    if urlsplit(properties["ftp_tif"]).hostname != "okmaps.org":
        raise ValueError("Tile index points outside the state imagery host")
    directory = TARGET / "tank_clips" / identity
    image = directory / "image.tif"
    source_dir = TARGET / "tiles" / identity
    source = str(source_dir / "image.tif") if (source_dir / "item.json").is_file() else properties["ftp_tif"]
    replace = False
    if image.is_file():
        left, bottom = float(properties["x_lwrlft"]), float(properties["y_lwrlft"])
        width_m = 8_192 * float(properties["dblgsd"])
        with rasterio.open(image) as cached:
            edges = cached.bounds
            tolerance = 0.01
            replace = (edges.left < left - tolerance or edges.right > left + width_m + tolerance or
                       edges.bottom < bottom - tolerance or edges.top > bottom + width_m + tolerance)
    width, height = clip(source, image, bounds, footprint, replace=replace)
    if not width or not height:
        raise ValueError("Tile raster has no overlap with the mapped tank footprint")
    with rasterio.open(image) as dataset:
        if dataset.crs.to_epsg() != 6350 or dataset.count != 3 or any(abs(value - 0.15) > 0.001 for value in dataset.res):
            raise ValueError(f"Unexpected state imagery CRS, bands or resolution: {image}")
    write(directory / "item.json", json.dumps(item, ensure_ascii=False).encode())
    return {"id": identity, "dimensions": [width, height], "crs": "EPSG:6350",
            "bytes": image.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tank-clips", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--bbox", default=BBOX, help="west,south,east,north in EPSG:4326")
    args = parser.parse_args()
    try:
        west, south, east, north = map(float, args.bbox.split(","))
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise ValueError
    except ValueError:
        parser.error("--bbox must contain west,south,east,north in increasing order")
    bbox = ",".join(format(value, "g") for value in (west, south, east, north))
    if args.tank_clips and bbox != BBOX:
        parser.error("--tank-clips uses the existing core tank inventory and its default bbox")
    items = catalog(bbox)
    if args.tank_clips:
        jobs = [(item["properties"]["strtile"], clip_tile, (item, bounds, footprint))
                for _, item, bounds, footprint in tank_tiles(items)]
    else:
        jobs = [(item["properties"]["strtile"], download, (item,)) for item in items]
    log = ROOT / "data" / "logs" / ("ok_aerial_2025_tank_clips.jsonl" if args.tank_clips else "ok_aerial_2025.jsonl")
    print(f"{len(jobs)} state aerial tiles", flush=True)
    with log.open("a") as stream, ThreadPoolExecutor(max_workers=args.workers) as executor:
        for offset in range(0, len(jobs), args.workers):
            if shutil.disk_usage(TARGET).free < 3 * 1024 ** 3:
                break
            pending = {executor.submit(collect, *params): identity
                       for identity, collect, params in jobs[offset:offset + args.workers]}
            for future in as_completed(pending):
                try:
                    row = future.result()
                except Exception as error:
                    row = {"id": pending[future], "error": str(error)}
                    print(json.dumps(row), flush=True)
                print(json.dumps(row), file=stream, flush=True)


if __name__ == "__main__":
    main()
