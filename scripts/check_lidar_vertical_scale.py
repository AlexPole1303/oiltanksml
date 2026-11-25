"""Compare native ground-return Z with USGS elevation responses in metres."""
import argparse
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import laspy
import numpy as np
from pyproj import Transformer

from audit_lidar_tanks import DATASET, LEGACY_DATASET, SURVEYS
from collect_public import RAW, ROOT, get, write

EPQS = "https://epqs.nationalmap.gov/v1/json"


def check_point(sample, directory):
    node, longitude, latitude, value = sample
    query = urlencode({"x": longitude, "y": latitude, "units": "Meters", "wkid": 4326, "includeDate": "true"})
    url = EPQS + "?" + query
    target = directory / (node + ".json")
    response = json.loads(get(url, target))
    location = response.get("location", {})
    if abs(location.get("x", 0) - longitude) > 1e-8 or abs(location.get("y", 0) - latitude) > 1e-8:
        response = json.loads(get(url, target, refresh=True))
    elevation = float(response["value"])
    if not np.isfinite(elevation) or not (150 < elevation < 500):
        raise ValueError(f"Unexpected elevation from EPQS for {node}")
    return (node, longitude, latitude, value, elevation, round(value - elevation, 3),
            response["resolution"], response.get("attributes", {}).get("AcquisitionDate"), url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SURVEYS, default=DATASET)
    args = parser.parse_args()
    root = RAW / "usgs"
    legacy = args.dataset == LEGACY_DATASET
    directory = root / ("epqs_ground_checks_2011" if legacy else "epqs_ground_checks")
    folder = root / "ept" / args.dataset / "nodes"
    nodes = sorted(folder.glob("*.laz" if legacy else "11-*.laz"))
    if legacy:
        depth = max(int(node.name.split("-", 1)[0]) for node in nodes)
        nodes = [node for node in nodes if int(node.name.split("-", 1)[0]) == depth]
    to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    sampled = []
    for position in np.linspace(0, len(nodes) - 1, num=min(16, len(nodes)), dtype=int):
        node = nodes[position]
        data = laspy.read(node)
        eligible = ~np.asarray(data.withheld, dtype=bool) & ~np.asarray(data.synthetic, dtype=bool)
        selection = np.flatnonzero((np.asarray(data.classification) == 2) & eligible)
        if not len(selection):
            continue
        index = int(selection[len(selection) // 2])
        longitude, latitude = to_geo.transform(float(data.x[index]), float(data.y[index]))
        sampled.append((node.stem, round(longitude, 9), round(latitude, 9), round(float(data.z[index]), 3)))
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lambda sample: check_point(sample, directory), sampled))
    if not rows:
        raise ValueError(f"No eligible ground returns for {args.dataset}")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("ept_node_id", "longitude", "latitude", "ept_ground_z", "epqs_elevation_m",
                     "z_minus_epqs_m", "epqs_resolution_degrees", "epqs_acquisition_date_raw", "source_url"))
    writer.writerows(rows)
    filename = "lidar_2011_vertical_scale_checks.csv" if legacy else "lidar_vertical_scale_checks.csv"
    write(ROOT / "data" / "derived" / filename, stream.getvalue().encode())
    print(f"{len(rows)} USGS elevation comparisons. Median absolute difference: {np.median([abs(row[5]) for row in rows]):.3f} m")


if __name__ == "__main__":
    main()
