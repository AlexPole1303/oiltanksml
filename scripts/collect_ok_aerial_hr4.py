"""Archive original four-band Oklahoma orthophoto samples around every tank."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window
from shapely import STRtree, transform, union_all
from shapely.geometry import shape

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_imagery import CONFIG, clip
from collect_naip_seamlines import candidates as mapped_tanks
from collect_ok_aerial import BBOX, TARGET
from collect_public import ROOT, TLS, get, write

TYPE = "aerial2025_tile_index_hr4_4326"
STORE = TARGET / "hr4_tank_clips"
META = TARGET / "source_metadata"
TO_UTM = Transformer.from_crs(4326, 32614, always_xy=True)
TO_GEO = Transformer.from_crs(32614, 4326, always_xy=True)
CONTEXT = 70


def index(bbox):
    params = dict(service="WFS", version="1.0.0", request="GetFeature", typeName=f"ogi:{TYPE}",
                  outputFormat="application/json", bbox=bbox, srsName="EPSG:4326", maxFeatures=1000)
    url = "https://okmaps.org/geoserver/wfs?" + urlencode(params)
    name = TYPE if bbox == BBOX else f"{TYPE}_{bbox}"
    payload = json.loads(get(url, META / f"{name}.geojson"))
    if len(payload["features"]) != payload["totalFeatures"]:
        raise ValueError(f"Truncated Oklahoma four-band index for {bbox}")
    write(META / f"{name}.source", (url + "\n").encode())
    return payload["features"]


def planned_tiles():
    paths = sorted((TARGET / "additional_indexes").glob("*.geojson"))
    items = {}
    for bbox in (BBOX, *(p.stem for p in paths)):
        for item in index(bbox):
            identity = item["properties"]["strtile"]
            if items.setdefault(identity, item) != item:
                raise ValueError(f"Conflicting four-band catalog records for {identity}")
    original = {(r["tile_id"], r["tile_majority_date"], r["tile_time_raw"]) for r in read("ok_aerial_tank_image_coverage")}
    for tile, day, stamp in original:
        props = items[tile]["properties"]
        if (props["majorityda"][:10], props["time"]) != (day, stamp):
            raise ValueError(f"RGB and four-band tile time labels differ for {tile}")
    tanks = [(tank, transform(polygon, TO_UTM.transform, interleaved=False).buffer(CONTEXT)) for tank, polygon in mapped_tanks()]
    spatial = STRtree([region for _, region in tanks])
    for identity, item in sorted(items.items()):
        native = transform(shape(item["geometry"]), TO_UTM.transform, interleaved=False)
        selected = spatial.query(native, predicate="intersects")
        if len(selected) == 0:
            continue
        footprint = union_all([tanks[i][1] for i in selected])
        bounds = transform(footprint.intersection(native), TO_GEO.transform, interleaved=False).bounds
        yield identity, item, bounds, footprint, sorted(tanks[i][0] for i in selected)


def collect(item):
    tile, feature, bounds, footprint, candidates = item
    props = feature["properties"]
    url, world_url = props["ftp_tif"], props["ftp_tfw"]
    if any(urlsplit(u).scheme != "https" or urlsplit(u).netloc != "okmaps.org" for u in (url, world_url)):
        raise ValueError(f"Unexpected four-band source host for {tile}")
    directory = STORE / tile
    image = directory / "image.tif"
    manifest = directory / "source.json"
    if manifest.is_file() and image.is_file():
        existing = json.loads(manifest.read_bytes())
        with image.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if (existing["candidate_ids_json"] == json.dumps(candidates, separators=(",", ":"))
                and existing["source_sha256_cropped_tiff"] == digest):
            return existing
    get(world_url, directory / "source.tfw")
    with urlopen(Request(url, method="HEAD"), timeout=25, context=TLS) as response:
        headers = response.headers
        if response.status != 200 or headers.get("Accept-Ranges") != "bytes":
            raise ValueError(f"Byte-range reads unavailable for {tile}")
    dims = clip(url, image, bounds, footprint, replace=image.is_file())
    if not all(dims):
        raise ValueError(f"Empty four-band crop for {tile}")
    with rasterio.Env(**CONFIG), rasterio.open(url) as source, rasterio.open(image, "r+") as local:
        if (source.crs.to_epsg() != 6350 or source.count != 4 or source.res != (0.15, 0.15)
                or source.dtypes != ("uint8",)*4 or local.crs != source.crs or local.count != source.count):
            raise ValueError(f"Unexpected original grid or bands for {tile}")
        source_offset = (~source.transform) * (local.transform.c, local.transform.f)
        left, top = (round(x) for x in source_offset)
        if any(abs(a-b) > 1e-6 for a, b in zip(source_offset, (left, top), strict=True)):
            raise ValueError(f"The crop is off the original 15 cm grid for {tile}")
        local.colorinterp = source.colorinterp
        local.set_band_description(4, "Near infrared")
        for _, w in local.block_windows(1):
            valid = local.dataset_mask(window=w) != 0
            if not valid.any():
                continue
            original = source.read(window=Window(left+w.col_off, top+w.row_off, w.width, w.height))
            if not np.array_equal(local.read(window=w)[:, valid], original[:, valid]):
                raise ValueError(f"Lossless source-pixel comparison failed for {tile}")
            break
        else:
            raise ValueError(f"No valid sampled pixels to cross-check for {tile}")
    with image.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    record = dict(tile_id=tile, tile_majority_date=props["majorityda"][:10], tile_nominal_time_utc=props["time"],
        source_url=url, source_world_file_url=world_url, source_object_bytes=int(headers["Content-Length"]),
        source_last_modified=headers["Last-Modified"], source_etag=headers["ETag"], crop_origin_source_column=left,
        crop_origin_source_row=top, crop_width=dims[0], crop_height=dims[1], crop_path=str(image.relative_to(ROOT)),
        source_sha256_cropped_tiff=digest, candidate_ids_json=json.dumps(candidates, separators=(",", ":")))
    write(directory / "item.json", json.dumps(feature, indent=2).encode())
    write(manifest, json.dumps(record, indent=2).encode())
    print(f"{tile}: {len(candidates)} contour contexts", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    jobs = list(planned_tiles())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(collect, jobs))
    export("ok_aerial_2025_hr4_window_index", rows)
    report = dict(tiles=len(rows), candidate_contexts=len({tank for r in rows for tank in json.loads(r["candidate_ids_json"])}),
        archived_crop_bytes=sum((ROOT / r["crop_path"]).stat().st_size for r in rows), context_buffer_m=CONTEXT,
        interpretation="Four bands retain original sample values and grid, verified against the remote source. "
        "The catalog clock describes the tile majority. Object times describe the downloadable copies.")
    write(DERIVED / "ok_aerial_2025_hr4_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
