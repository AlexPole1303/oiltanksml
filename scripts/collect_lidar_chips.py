"""Preserve the original LiDAR points near mapped tanks without storing entire source tiles."""
import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import laspy
from pyproj import Transformer
from shapely import intersects_xy
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union

from collect_public import RAW, ROOT, write

TARGET = RAW / "lidar_tanks"
CANDIDATES = ROOT / "data" / "derived" / "osm_tank_candidates.geojson"


def footprint(buffer, candidates=CANDIDATES):
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    from_utm = Transformer.from_crs("EPSG:32614", "EPSG:4326", always_xy=True)
    features = json.loads(candidates.read_bytes())["features"]
    metric = unary_union([transform(to_utm.transform, shape(item["geometry"])) for item in features])
    return transform(from_utm.transform, metric.buffer(buffer))


def overlaps(item, geometry):
    bounds = item.get("boundingBox", {})
    if any(key not in bounds for key in ("minX", "minY", "maxX", "maxY")):
        return False
    return geometry.intersects(box(bounds["minX"], bounds["minY"], bounds["maxX"], bounds["maxY"]))


def fetch_and_clip(item, geometry, buffer):
    identity = item["sourceId"]
    directory = TARGET / identity
    metadata = directory / "item.json"
    destination = directory / "points.laz"
    if metadata.is_file() and destination.is_file():
        return json.loads(metadata.read_bytes())["selected_points"]
    url = item.get("downloadLazURL") or item.get("downloadURL")
    if not url or not url.lower().endswith((".laz", ".las")):
        raise ValueError("No source LAS or LAZ URL")
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "source.part"
    if not original.exists() or original.stat().st_size != item.get("sizeInBytes"):
        process = subprocess.run(
            ["/usr/bin/curl", "-fLsS", "--max-time", "300", "-C", "-", "-o", str(original), url],
            capture_output=True, text=True,
        )
        if process.returncode:
            raise OSError(process.stderr.strip())
    if item.get("sizeInBytes") and original.stat().st_size != item["sizeInBytes"]:
        raise ValueError("Source file size disagrees with the USGS catalog")
    temporary = directory / "points.part.laz"
    kept = 0
    with laspy.open(original) as source:
        crs = source.header.parse_crs()
        if crs is None:
            raise ValueError("Source LAS header has no parseable CRS")
        converter = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        projected = transform(converter.transform, geometry)
        with laspy.open(temporary, mode="w", header=source.header, do_compress=True) as result:
            for points in source.chunk_iterator(500_000):
                inside = intersects_xy(projected, points.x, points.y)
                if inside.any():
                    result.write_points(points[inside])
                    kept += int(inside.sum())
        total = source.header.point_count
        dimensions = list(source.header.point_format.dimension_names)
    with laspy.open(temporary) as check:
        if check.header.point_count != kept:
            raise ValueError("Clipped point count does not match the written LAS header")
    with original.open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    temporary.replace(destination)
    write(metadata, json.dumps({
        "usgs_product": item,
        "source_sha256": sha256,
        "source_points": total,
        "selected_points": kept,
        "point_dimensions": dimensions,
        "crs_wkt": crs.to_wkt(),
        "buffer_m": buffer,
        "footprint_source": str(CANDIDATES.relative_to(ROOT)),
    }, ensure_ascii=False).encode())
    original.unlink()
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--buffer", type=float, default=120)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", default="OK_Statewide_D22")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    geometry = footprint(args.buffer)
    catalog = json.loads((RAW / "usgs" / "lidar_products.json").read_bytes())["items"]
    selected = [item for item in catalog
                if args.project in item.get("title", "") and overlaps(item, geometry)]
    if args.limit is not None:
        selected = selected[:args.limit]
    print(f"{len(selected)} selected source tiles ({sum(i.get('sizeInBytes', 0) for i in selected):,} bytes before clipping)", flush=True)
    if args.inspect:
        return
    TARGET.mkdir(parents=True, exist_ok=True)
    log = ROOT / "data" / "logs" / "lidar_chips.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index in range(0, len(selected), args.workers):
            if shutil.disk_usage(TARGET).free < 3 * 1024 ** 3:
                print("Stopped at the disk-space floor", flush=True)
                break
            pending = {executor.submit(fetch_and_clip, item, geometry, args.buffer): item["sourceId"]
                       for item in selected[index:index + args.workers]}
            rate_limited = False
            for future in concurrent.futures.as_completed(pending):
                row = {"id": pending[future]}
                try:
                    row["points"] = future.result()
                except Exception as error:
                    row["error"] = str(error)
                    rate_limited |= "error: 429" in row["error"]
                print(json.dumps(row), file=stream, flush=True)
                print(json.dumps(row), flush=True)
            if rate_limited:
                print("HTTP 429 from RockyWeb. Stopping this download run.", flush=True)
                break


if __name__ == "__main__":
    main()
