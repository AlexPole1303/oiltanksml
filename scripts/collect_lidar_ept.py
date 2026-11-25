"""Fetch the public USGS Entwine point-cloud nodes intersecting all mapped tanks."""
import argparse
import concurrent.futures
import hashlib
import io
import json
import shutil

import laspy
from pyproj import CRS, Transformer
from shapely import intersects_xy, transform
from shapely.geometry import box
from shapely.prepared import prep

from collect_lidar_chips import CANDIDATES, footprint
from collect_public import RAW, ROOT, get, write

BASE = "https://usgs-lidar-public.s3.us-west-2.amazonaws.com/"


def node_bounds(key, bounds):
    depth, ix, iy, _ = map(int, key.split("-"))
    scale = 2 ** depth
    dx = (bounds[3] - bounds[0]) / scale
    dy = (bounds[4] - bounds[1]) / scale
    xmin = bounds[0] + ix * dx
    ymin = bounds[1] + iy * dy
    return box(xmin, ymin, xmin + dx, ymin + dy)


def hierarchy(url, directory, metadata, area):
    pending = ["0-0-0-0"]
    visited, points = set(), {}
    spatial = prep(area)
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        source = get(f"{url}/ept-hierarchy/{current}.json", directory / "hierarchy" / f"{current}.json")
        for key, count in json.loads(source).items():
            if not spatial.intersects(node_bounds(key, metadata["bounds"])):
                continue
            if count < 0:
                if key == current:
                    raise ValueError(f"Recursive unresolved EPT hierarchy node {key}")
                pending.append(key)
            elif count > 0:
                points[key] = count
    return points


def save_node(key, count, dataset, url, directory, area, crs, buffer):
    metadata_file = directory / "nodes" / (key + ".json")
    if metadata_file.is_file():
        row = json.loads(metadata_file.read_bytes())
        if row["selected_points"] == 0 or (metadata_file.parent / (key + ".laz")).is_file():
            return row
    source_url = f"{url}/ept-data/{key}.laz"
    raw = get(source_url)
    data = laspy.read(io.BytesIO(raw))
    if len(data) != count:
        raise ValueError(f"EPT hierarchy declared {count} points, read {len(data)}")
    inside = intersects_xy(area, data.x, data.y)
    kept = int(inside.sum())
    row = {"id": key, "dataset": dataset, "source_url": source_url, "source_points": count,
           "selected_points": kept, "source_sha256": hashlib.sha256(raw).hexdigest(),
           "horizontal_crs": crs.to_string(), "buffer_m": buffer}
    if kept:
        result = laspy.LasData(data.header)
        result.points = data.points[inside]
        if result.header.parse_crs() is None:
            result.header.add_crs(crs)
        output = metadata_file.parent / (key + ".laz")
        output.parent.mkdir(parents=True, exist_ok=True)
        part = output.with_suffix(".part.laz")
        result.write(part)
        with laspy.open(part) as check:
            if check.header.point_count != kept:
                raise ValueError("Clipped point count did not match the LAZ header")
        part.replace(output)
    write(metadata_file, json.dumps(row, separators=(",", ":")).encode())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="OK_Statewide_2_D22")
    parser.add_argument("--buffer", type=float, default=120)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--supplemental", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset
    url = BASE + dataset
    scope = "ept_supplemental" if args.supplemental else "ept"
    candidates = CANDIDATES.with_name("supplemental_tank_candidates.geojson") if args.supplemental else CANDIDATES
    directory = RAW / "usgs" / scope / dataset
    data = json.loads(get(url + "/ept.json", directory / "ept.json"))
    if data["dataType"] != "laszip" or data["hierarchyType"] != "json":
        raise ValueError("Expected LASzip data with a JSON hierarchy")
    srs = data["srs"]
    crs = CRS.from_user_input(srs.get("wkt") or f"EPSG:{srs['horizontal']}")
    converter = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    area = transform(footprint(args.buffer, candidates), converter.transform, interleaved=False)
    nodes = hierarchy(url, directory, data, area)
    write(directory / "selection.json", json.dumps({"dataset": dataset, "buffer_m": args.buffer,
          "selected_nodes": nodes, "horizontal_crs": crs.to_string()}, indent=2).encode())
    print(f"{dataset}: {len(nodes)} candidate EPT nodes with {sum(nodes.values()):,} source points", flush=True)
    if args.inspect:
        return
    log = ROOT / "data" / "logs" / f"lidar_{scope}_{dataset}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(nodes.items(), key=lambda pair: tuple(map(int, pair[0].split("-"))))
    with log.open("a") as stream, concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index in range(0, len(ordered), args.workers):
            if shutil.disk_usage(directory).free < 3 * 1024 ** 3:
                print("Stopped at the disk-space floor", flush=True)
                break
            batch = {executor.submit(save_node, key, count, dataset, url, directory, area, crs, args.buffer): key
                     for key, count in ordered[index:index + args.workers]}
            for future in concurrent.futures.as_completed(batch):
                try:
                    result = future.result()
                except Exception as error:
                    result = {"id": batch[future], "error": str(error)}
                    print(json.dumps(result), flush=True)
                print(json.dumps(result), file=stream, flush=True)


if __name__ == "__main__":
    main()
