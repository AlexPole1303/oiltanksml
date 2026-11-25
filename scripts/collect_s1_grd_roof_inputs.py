"""Read Sentinel-1 GRD native windows without resampling and retain the source annotation."""
import argparse
import hashlib
import io
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from urllib.parse import quote, urlencode

import numpy as np
import rasterio
from rasterio.transform import GCPTransformer
from rasterio.windows import Window

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import RAW, ROOT, get, write

BUCKET = "https://sentinel-s1-l1c.s3.eu-central-1.amazonaws.com/"
STORE = RAW / "sentinel1_grd_native"
GROUP_PIXELS = 512
CONTEXT_M = 120
POLARIZATIONS = {"DV": ("VV", "VH"), "SV": ("VV",), "DH": ("HH", "HV"), "SH": ("HH",)}


def prefix(product):
    name = product["product_name"].removesuffix(".SAFE")
    mission, mode, kind, pol, stamp, *_ = name.split("_")
    if not mission.startswith("S1") or mode != "IW" or kind != "GRDH" or pol[2:] not in POLARIZATIONS:
        raise ValueError(f"Unexpected original Sentinel-1 GRD product {name}")
    date = datetime.strptime(stamp, "%Y%m%dT%H%M%S")
    return f"GRD/{date.year}/{date.month}/{date.day}/{mode}/{pol[2:]}/{name}/", POLARIZATIONS[pol[2:]]


def listing(path, directory):
    payload = get(BUCKET + "?" + urlencode({"list-type": 2, "prefix": path}), directory / "aws_listing.xml")
    root = ET.fromstring(payload)
    if root.findtext("{*}IsTruncated") != "false":
        raise ValueError(f"Incomplete AWS listing for {path}")
    objects = {}
    for entry in root.findall("{*}Contents"):
        key = entry.findtext("{*}Key")
        if not key.startswith(path):
            raise ValueError(f"AWS object outside {path}")
        objects[key.removeprefix(path)] = dict(source_url=BUCKET + quote(key, safe="/"),
            last_modified_utc=entry.findtext("{*}LastModified"), etag=entry.findtext("{*}ETag").strip('"'),
            bytes=int(entry.findtext("{*}Size")))
    return objects


def save_metadata(objects, name, directory):
    asset = objects[name]
    path = directory / name
    payload = get(asset["source_url"], path)
    if len(payload) != asset["bytes"]:
        raise ValueError(f"Source object size differs for {name}")
    if name.endswith((".xml", ".safe")):
        ET.fromstring(payload)
    elif name.endswith(".json"):
        json.loads(payload)
    asset.update(local_file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(payload).hexdigest())
    return payload


def annotation(payload, product, pol):
    root = ET.fromstring(payload)
    header = root.find("adsHeader")
    first = datetime.fromisoformat(header.findtext("startTime")).replace(tzinfo=timezone.utc)
    last = datetime.fromisoformat(header.findtext("stopTime")).replace(tzinfo=timezone.utc)
    if (header.findtext("polarisation") != pol or header.findtext("missionId") != product["satellite"]
            or int(header.findtext("absoluteOrbitNumber")) != int(product["absolute_orbit"])
            or abs((first - datetime.fromisoformat(product["capture_start_utc"])).total_seconds()) > 1
            or abs((last - datetime.fromisoformat(product["capture_end_utc"])).total_seconds()) > 1):
        raise ValueError(f"Annotation identity differs for {product['product_name']} {pol}")
    details = root.find("imageAnnotation/imageInformation")
    return dict(annotation_start_utc=first.isoformat(), annotation_end_utc=last.isoformat(),
        source_shape=[int(details.findtext(k)) for k in ("numberOfLines", "numberOfSamples")],
        pixel_spacing_m=[float(details.findtext(k)) for k in ("azimuthPixelSpacing", "rangePixelSpacing")])


def windows(dataset, tank_ids, tanks, spacing):
    gcps, crs = dataset.gcps
    if len(gcps) < 20 or crs.to_epsg() != 4326:
        raise ValueError("Missing native WGS84 geolocation grid")
    longitude, latitude = ([tanks[tank][index] for tank in tank_ids] for index in (0, 1))
    with GCPTransformer(gcps, tps=True) as transformer:
        line, pixel = transformer.rowcol(longitude, latitude, op=float)
    groups = defaultdict(list)
    for tank, row, col in zip(tank_ids, line, pixel, strict=True):
        if not (0 <= row < dataset.height and 0 <= col < dataset.width):
            continue
        pad = math.ceil((CONTEXT_M + tanks[tank][2] / 2) / min(spacing))
        groups[int(row) // GROUP_PIXELS, int(col) // GROUP_PIXELS].append((tank, row, col, pad))
    for (tile_row, tile_col), positions in sorted(groups.items()):
        r0 = max(0, math.floor(min(row - pad for _, row, _, pad in positions)))
        c0 = max(0, math.floor(min(col - pad for _, _, col, pad in positions)))
        r1 = min(dataset.height, math.ceil(max(row + pad + 1 for _, row, _, pad in positions)))
        c1 = min(dataset.width, math.ceil(max(col + pad + 1 for _, _, col, pad in positions)))
        yield (tile_row, tile_col), Window(c0, r0, c1 - c0, r1 - r0), positions


def read_band(product, directory, objects, pol, tank_ids, tanks):
    pol_lower = pol.lower()
    label = f"iw-{pol_lower}"
    xml_name = f"annotation/{label}.xml"
    raw = save_metadata(objects, xml_name, directory)
    details = annotation(raw, product, pol)
    for name in (f"annotation/calibration/calibration-{label}.xml", f"annotation/calibration/noise-{label}.xml"):
        save_metadata(objects, name, directory)
    measurement = objects[f"measurement/{label}.tiff"]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="25",
                      GDAL_HTTP_MAX_RETRY="2", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff"):
        with rasterio.open(measurement["source_url"]) as dataset:
            if list(dataset.shape) != details["source_shape"] or dataset.dtypes != ("uint16",):
                raise ValueError(f"Unexpected native radar raster grid for {product['product_name']} {pol}")
            gcps, crs = dataset.gcps
            details.update(source_dtype=dataset.dtypes[0], source_nodata=dataset.nodata,
                geolocation_crs=str(crs), native_gcps=[[g.row, g.col, g.x, g.y, g.z] for g in gcps])
            rows = []
            for (tile_row, tile_col), window, positions in windows(dataset, tank_ids, tanks, details["pixel_spacing_m"]):
                path = directory / pol / f"native_{tile_row:03d}_{tile_col:03d}.npz"
                values = dataset.read(1, window=window)
                valid = dataset.read_masks(1, window=window) > 0
                if values.shape != (int(window.height), int(window.width)) or valid.shape != values.shape:
                    raise ValueError(f"Window read differs for {path.name}")
                output = io.BytesIO()
                np.savez_compressed(output, dn=values, valid=valid)
                payload = output.getvalue()
                write(path, payload)
                digest = hashlib.sha256(payload).hexdigest()
                for tank, row, col, _ in positions:
                    i, j = math.floor(row) - int(window.row_off), math.floor(col) - int(window.col_off)
                    rows.append(dict(product_id=product["product_id"], product_name=product["product_name"],
                        polarization=pol, candidate_id=tank, native_line=round(float(row), 6), native_sample=round(float(col), 6),
                        centre_pixel_valid=bool(valid[i, j]), window_row_offset=int(window.row_off),
                        window_col_offset=int(window.col_off), window_rows=int(window.height), window_cols=int(window.width),
                        window_file=str(path.relative_to(ROOT)), window_sha256=digest, source_url=measurement["source_url"],
                        source_last_modified_utc=measurement["last_modified_utc"], selection_reasons=product["selection_reasons"]))
            return details, rows


def collect(product, tanks, *, store=STORE):
    name = product["product_name"].removesuffix(".SAFE")
    directory = store / name
    source_prefix, polarizations = prefix(product)
    objects = listing(source_prefix, directory)
    for name in ("manifest.safe", "productInfo.json"):
        save_metadata(objects, name, directory)
    ids = [tank for tank in json.loads(product["covered_candidate_ids_json"]) if tank in tanks]
    bands, all_rows = {}, []
    for pol in polarizations:
        details, rows = read_band(product, directory, objects, pol, ids, tanks)
        if not rows:
            raise ValueError(f"No native windows for {product['product_name']} {pol}")
        bands[pol] = dict(details, native_windows=len({r["window_file"] for r in rows}),
                          located_tanks=len(rows), valid_centre_pixels=sum(r["centre_pixel_valid"] for r in rows))
        all_rows.extend(rows)
    manifest = dict(product=product, bands=bands, objects=objects, catalogue_contours=len(ids),
                    retrieved_utc=datetime.now(timezone.utc).isoformat())
    write(directory / "native_manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    print(f"{product['product_name']}: {len(all_rows)} contour-polarization rows", flush=True)
    return manifest, all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--holdout-gaps", action="store_true")
    args = parser.parse_args()
    products = read("s1_grd_holdout_gap_manifest" if args.holdout_gaps else "s1_grd_native_pilot_manifest")
    stem = "s1_grd_holdout_gap" if args.holdout_gaps else "s1_grd"
    with closing(read_only(SOURCE)) as db:
        tanks = {tank: (lon, lat, diameter) for tank, lon, lat, diameter in db.execute(
            "SELECT candidate_id,longitude,latitude,footprint_diameter_m FROM tank_engineering_registry")}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda item: collect(item, tanks), products))
    rows = [row for _, group in results for row in group]
    export(f"{stem}_native_window_index", rows)
    report = dict(product_count=len(results), contour_polarization_rows=len(rows),
        native_windows=len({r["window_file"] for r in rows}), valid_centre_pixels=sum(r["centre_pixel_valid"] for r in rows),
        candidate_contours=len({r["candidate_id"] for r in rows}),
        interpretation="The files retain unsigned native GRD digital numbers and source masks from byte-range reads. "
        "Coordinates were located approximately using the native WGS84 geolocation grid. No map reprojection, "
        "calibration to backscatter, roof height estimation, SLC phase or market outcomes are included.")
    summary = f"{stem}_native_summary" if args.holdout_gaps else "s1_grd_native_pilot_summary"
    write(DERIVED / f"{summary}.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
