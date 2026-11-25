"""Preserve USDA source-image timing and the Oklahoma NAIP 2025 RGB coverage."""
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask, geometry_window
from shapely import transform, union_all
from shapely.geometry import box, mapping, shape

from collect_public import RAW, ROOT, get, write, write_csv

DERIVED = ROOT / "data" / "derived"
SEAMLINES = RAW / "naip" / "seamlines"
RGB = RAW / "naip" / "ok_2025_rgb"
ARC = "https://services.arcgis.com/LLVEmB8Lsae3Um4s/arcgis/rest/services/"
ITEM = "https://www.arcgis.com/sharing/rest/content/items/"
SERVICES = {
    2017: ("OK2017_Seamlines", "df23fcda629647a883f04dc16135c29e"),
    2019: ("OK_Seamlines2019", "2d6e5955904e4f8a875a3b1929ab0576"),
    2021: ("Seamlines_OK40_2021", "1429a85754d7426994b561897e8d880e"),
    2023: ("2023_Oklahoma_Image_Dates", "f30a1bfeba0544cdbef7bc533e845f40"),
    2025: ("2025_Oklahoma_Image_Dates", "ed3243311669427aa91fa56b0b71522c"),
}
WCS = "https://okmaps.org/geoserver/ogi/wcs?"
COVERAGE = "ogi__NAIP2025"


def candidates():
    for layer in ("osm", "supplemental"):
        path = DERIVED / f"{layer}_tank_candidates.geojson"
        for item in json.loads(path.read_bytes())["features"]:
            yield item["properties"]["candidate_id"], shape(item["geometry"])


def collect_seamlines(year):
    name, item_id = SERVICES[year]
    directory = SEAMLINES / str(year)
    layer = ARC + name + "/FeatureServer/0"
    metadata_url = ITEM + item_id + "/info/metadata/metadata.xml"
    metadata = get(metadata_url, directory / "metadata.xml")
    if b"local 24 hour clock" not in metadata.lower():
        raise ValueError(f"USDA clock description missing for {year}")
    get(layer + "?f=pjson", directory / "layer.json")
    search = {"where": "1=1", "geometry": "-96.90,35.68,-96.52,36.12", "geometryType": "esriGeometryEnvelope",
              "inSR": 4326, "outSR": 4326, "outFields": "*", "returnGeometry": "true", "f": "geojson"}
    query = layer + "/query?"
    source = query + urlencode(search)
    data = json.loads(get(source, directory / "features.geojson"))
    check = query + urlencode({**search, "returnCountOnly": "true", "f": "json"})
    count = json.loads(get(check, directory / "feature_count.json"))["count"]
    if data.get("exceededTransferLimit") or not count or len(data["features"]) != count:
        raise ValueError(f"Incomplete USDA seamline coverage for {year}")
    write(directory / "features.source", (source + "\n").encode())
    write(directory / "metadata.source", (metadata_url + "\n").encode())
    print(year, count, "USDA source-image polygons", flush=True)


def coverage_description():
    source = WCS + urlencode({"service": "WCS", "version": "2.0.1", "request": "DescribeCoverage",
                               "coverageId": COVERAGE})
    data = get(source, RGB / "coverage.xml")
    document = ET.fromstring(data)
    if document.findtext(".//{*}CoverageId") != COVERAGE:
        raise ValueError("Unexpected Oklahoma WCS coverage")
    (zero_y, dx), (dy, zero_x) = [tuple(map(float, row.text.split()))
                                  for row in document.findall(".//{*}offsetVector")]
    center_y, center_x = map(float, document.findtext(".//{*}origin/{*}Point/{*}pos").split())
    if zero_x or zero_y or dx <= 0 or dy >= 0:
        raise ValueError("Unexpected Oklahoma WCS grid orientation")
    write(RGB / "coverage.source", (source + "\n").encode())
    return dx, dy, center_x - dx / 2, center_y - dy / 2


def tile_jobs():
    groups = defaultdict(list)
    for identity, geometry in candidates():
        point = geometry.centroid
        key = math.floor((point.x + 97) * 100), math.floor((point.y - 35) * 100)
        groups[key].append((identity, geometry))
    forward = Transformer.from_crs(4326, 32614, always_xy=True).transform
    backward = Transformer.from_crs(32614, 4326, always_xy=True).transform
    for key, group in sorted(groups.items()):
        buffered = union_all([transform(g, forward, interleaved=False).buffer(125) for _, g in group])
        bounds = transform(buffered, backward, interleaved=False).bounds
        yield f"{key[0]:03d}_{key[1]:03d}", bounds, group


def collect_tile(job, grid):
    key, bounds, group = job
    west, south, east, north = bounds
    query = [("service", "WCS"), ("version", "2.0.1"), ("request", "GetCoverage"), ("coverageId", COVERAGE),
             ("format", "image/tiff"), ("subset", f"Lat({south:.9f},{north:.9f})"),
             ("subset", f"Long({west:.9f},{east:.9f})")]
    url = WCS + urlencode(query)
    directory = RGB / key
    path = directory / "image.tif"
    manifest = directory / "source.json"
    cached = path.is_file() and manifest.is_file()
    payload = path.read_bytes() if cached else get(url)
    checksum = hashlib.sha256(payload).hexdigest()
    if cached:
        previous = json.loads(manifest.read_bytes())
        if previous["request_url"] != url or previous["sha256"] != checksum:
            raise ValueError(f"Cached Oklahoma WCS response differs for {key}")
    dx, dy, origin_x, origin_y = grid
    with rasterio.MemoryFile(payload) as memory, memory.open() as dataset:
        af = dataset.transform
        offsets = (af.c - origin_x) / dx, (af.f - origin_y) / dy
        if dataset.crs.to_epsg() != 4326 or dataset.count != 3 or dataset.dtypes != ("uint8",) * 3:
            raise ValueError(f"Unexpected Oklahoma WCS raster format for {key}")
        if abs(af.a - dx) > 1e-13 or abs(af.e - dy) > 1e-13 or any(abs(o-round(o)) > 1e-5 for o in offsets):
            raise ValueError(f"Oklahoma WCS changed the coverage grid for {key}")
        rows = []
        for identity, tank in group:
            if not box(*dataset.bounds).covers(tank):
                raise ValueError(f"Oklahoma WCS clip misses part of {identity}")
            geom = mapping(tank)
            window = geometry_window(dataset, [geom])
            region = geometry_mask([geom], out_shape=(int(window.height), int(window.width)),
                                   transform=dataset.window_transform(window), invert=True)
            valid = np.all(dataset.read_masks(window=window) > 0, axis=0)
            expected, pixels = int(region.sum()), int((region & valid).sum())
            if not expected or expected != pixels:
                raise ValueError(f"Oklahoma WCS raster has missing roof pixels for {identity}")
            rows.append((identity, key, str(path.relative_to(ROOT)), expected, pixels))
        dimensions = [dataset.width, dataset.height]
    if not cached:
        write(path, payload)
        write(manifest, json.dumps({"request_url": url, "sha256": checksum, "dimensions": dimensions,
              "bounds_requested": bounds, "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=2).encode())
    print(key, len(rows), "NAIP 2025 roof footprints verified", flush=True)
    return rows


def main():
    grid = coverage_description()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(collect_seamlines, SERVICES))
        rows = [row for result in pool.map(lambda job: collect_tile(job, grid), tile_jobs()) for row in result]
    write_csv(DERIVED / "naip_2025_wcs_coverage.csv", ("candidate_id", "clip_id", "asset_path",
              "candidate_pixels", "valid_pixels"), sorted(rows))
    print(len(rows), "verified NAIP 2025 candidates")


if __name__ == "__main__":
    main()
