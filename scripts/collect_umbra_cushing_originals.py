"""Materialize complete public Umbra GEC and SICD objects on the raster drive."""
import argparse
import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.windows import Window

from build_s1_reference_matches import export, read
from collect_public import ROOT, TLS, write
from collect_umbra_cushing import BUCKET


CHUNK_BYTES = 4 * 1024 * 1024
RASTERS = ROOT / "data" / "rasters"
CONTROL_ORIGINALS = ROOT / "data" / "raw" / "umbra_original_controls"
RECEIPTS = ROOT / "data" / "raw" / "umbra_cushing" / "original_raster_receipts"


def task(row, kind, destination):
    url = row[f"{kind}_url"]
    if not url.startswith(BUCKET):
        raise ValueError(f"Not an Umbra public object: {url}")
    filename = unquote(urlsplit(url).path.rsplit("/", 1)[-1])
    if filename != row["product_name"] + {"gec": "_GEC.tif", "sicd": "_SICD.nitf"}[kind]:
        raise ValueError(f"Unexpected source filename: {filename}")
    directory = destination / row["product_id"]
    path = directory / filename
    result = dict(product_id=row["product_id"], collect_id=row["collect_id"],
                  capture_start_utc=row["capture_start_utc"], asset_type=kind,
                  platform=row["platform"], source_url=url, object_bytes=int(row[f"{kind}_bytes"]),
                  s3_etag=row[f"{kind}_etag"], s3_last_modified_utc=row[f"{kind}_s3_last_modified_utc"],
                  local_file=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
    return path, result


def transfer(path, asset):
    etag = asset["s3_etag"]
    stamp = datetime.fromisoformat(asset["s3_last_modified_utc"])
    request = Request(asset["source_url"], headers={"If-Match": f'"{etag}"', "Accept-Encoding": "identity"})
    digest = hashlib.sha256()
    written = 0
    with urlopen(request, context=TLS, timeout=60) as source:
        headers = source.headers
        if (source.status != 200 or headers["ETag"].strip('"') != etag
                or int(headers["Content-Length"]) != asset["object_bytes"]
                or parsedate_to_datetime(headers["Last-Modified"]) != stamp):
            raise ValueError(f"Source object changed: {asset['source_url']}")
        with path.open("wb") as handle:
            while chunk := source.read(CHUNK_BYTES):
                written += len(chunk)
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    if written != asset["object_bytes"]:
        raise OSError(f"Incomplete transfer for {path.name}: {written} bytes")
    return digest.hexdigest()


def raster_details(path, asset):
    with rasterio.open(path) as dataset:
        if asset["asset_type"] == "sicd":
            xml = dataset.tags(ns="xml:DES").get("xml:DES", "")
            if (dataset.driver != "NITF" or dataset.dtypes not in (("complex64",), ("complex_int16",))
                    or asset["platform"].casefold() not in xml.casefold() or "SICD" not in xml):
                raise ValueError(f"Unexpected complex radar data in {path.name}")
        elif dataset.driver != "GTiff" or dataset.count != 1 or dataset.crs is None:
            raise ValueError(f"Unexpected geocoded radar data in {path.name}")
        if min(dataset.shape) <= 0:
            raise ValueError(f"Empty raster in {path.name}")
        return dict(rows=dataset.height, columns=dataset.width, dtype=dataset.dtypes[0],
                    crs=str(dataset.crs) if dataset.crs else None)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def copy_and_sync(source, destination):
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
        outgoing.flush()
        os.fsync(outgoing.fileno())


def receipt_path(path):
    folder = hashlib.sha256(str(path.parent.parent).encode()).hexdigest()[:24]
    return RECEIPTS / folder / path.parent.name / (path.name + ".json")


def existing_record(path, asset):
    if not path.is_file() or path.stat().st_size != asset["object_bytes"]:
        return None
    for marker in (receipt_path(path), path.with_suffix(path.suffix + ".verified.json")):
        if not marker.is_file():
            continue
        try:
            old = json.loads(marker.read_bytes())
        except (UnicodeError, json.JSONDecodeError):
            continue
        if (isinstance(old, dict) and all(old.get(k) == v for k, v in asset.items() if k != "local_file")
                and old.get("sha256") and old["sha256"] == file_hash(path)):
            return dict(old, local_file=asset["local_file"])
    return None


def download(path, asset):
    for attempt in range(3):
        try:
            digest = transfer(path, asset)
            return dict(asset, sha256=digest, **raster_details(path, asset))
        except OSError as error:
            if attempt == 2 or isinstance(error, HTTPError) and error.code in {400, 401, 403, 404, 412}:
                raise
            time.sleep(2 ** attempt)


def collect(item, stage=None):
    path, asset = item
    marker = receipt_path(path)
    if old := existing_record(path, asset):
        return old
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name("partial-" + path.name)
    seed = CONTROL_ORIGINALS / asset["product_id"] / path.name
    try:
        result = existing_record(seed, asset) if seed != path else None
        if result:
            if not partial.is_file() or file_hash(partial) != result["sha256"]:
                copy_and_sync(seed, partial)
        elif stage is not None:
            with TemporaryDirectory(dir=stage) as temporary:
                staged = Path(temporary) / path.name
                result = download(staged, asset)
                copy_and_sync(staged, partial)
        else:
            result = download(partial, asset)
        if partial.stat().st_size != asset["object_bytes"] or file_hash(partial) != result["sha256"]:
            raise OSError(f"Destination file differs from verified source for {path.name}")
        os.replace(partial, path)
        if file_hash(path) != result["sha256"]:
            raise OSError(f"Published destination file differs from verified source for {path.name}")
        write(marker, (json.dumps(result, indent=2) + "\n").encode())
        print(f"Verified {path.name}, {asset['object_bytes']:,} bytes.", flush=True)
        return result
    finally:
        partial.unlink(missing_ok=True)


def verify_native_windows(results):
    originals = {r["product_id"]: r for r in results if r["asset_type"] == "sicd"}
    windows = defaultdict(dict)
    for row in read("umbra_sicd_all_window_index"):
        product = row["product_id"]
        if product in originals:
            windows[product][row["window_file"]] = row
    output = []
    for product, source in originals.items():
        if not windows[product]:
            raise ValueError(f"No archived complex windows for {product}")
        with rasterio.open(ROOT / source["local_file"]) as dataset:
            for file, row in windows[product].items():
                if row["source_url"] != source["source_url"] or row["source_etag"] != source["s3_etag"]:
                    raise ValueError(f"Range-read source version differs for {file}")
                x, y, w, h = (int(float(row[k])) for k in
                              ("window_column_offset", "window_row_offset", "window_columns", "window_rows"))
                with np.load(ROOT / file, allow_pickle=False) as archive:
                    saved = archive["complex_amplitude"]
                pixels = dataset.read(1, window=Window(x, y, w, h))
                if pixels.dtype != saved.dtype or not np.array_equal(pixels, saved):
                    raise ValueError(f"Range-read samples differ from the full SICD object for {file}")
                output.append(dict(product_id=product, full_sicd_file=source["local_file"],
                                   full_sicd_sha256=source["sha256"], native_window_file=file,
                                   native_window_sha256=row["window_sha256"], pixels=w * h))
    return output


def export_progress(jobs, stem):
    rows = []
    for path, asset in jobs:
        saved = existing_record(path, asset)
        present = path.is_file()
        rows.append(dict(asset, destination_size_bytes=path.stat().st_size if path.is_file() else None,
                         receipt_status="verified_sha256" if saved else "present_unverified" if present else "pending",
                         sha256=saved["sha256"] if saved else None))
    export(stem + "_progress", rows)
    for kind in ("gec", "sicd"):
        group = [r for r in rows if r["asset_type"] == kind]
        count = sum(r["receipt_status"] == "verified_sha256" for r in group)
        print(f"{kind.upper()} SHA-256 verified: {count}/{len(group)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--product-id", action="append", help="Restrict to specified catalog acquisitions")
    parser.add_argument("--output", type=Path, default=RASTERS / "umbra" / "originals")
    parser.add_argument("--stage", type=Path, help="Verify incoming files locally before copying to the destination")
    parser.add_argument("--inventory-only", action="store_true", help="Export progress from existing receipts")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    rows = read("umbra_cushing_catalog")
    if args.product_id:
        rows = [row for row in rows if row["product_id"] in args.product_id]
        if {r["product_id"] for r in rows} != set(args.product_id):
            parser.error("Unknown Umbra product identifier")
    output = args.output.absolute()
    if output.is_relative_to(RASTERS) and not RASTERS.is_dir():
        raise ValueError("Mount the existing external raster drive first")
    jobs = [task(row, kind, output) for kind in ("gec", "sicd") for row in rows]
    stem = "umbra_cushing_original_rasters"
    stem += "_selected" if args.product_id else ""
    if args.inventory_only:
        export_progress(jobs, stem)
        return
    output.mkdir(parents=True, exist_ok=True)
    remaining_bytes = sum(asset["object_bytes"] for path, asset in jobs
                          if not path.is_file() or path.stat().st_size != asset["object_bytes"])
    if shutil.disk_usage(output).free < remaining_bytes:
        raise ValueError("Insufficient space for complete radar objects")
    if args.stage is not None:
        args.stage.mkdir(parents=True, exist_ok=True)
        largest = sorted((asset["object_bytes"] for _, asset in jobs), reverse=True)[:args.workers]
        if shutil.disk_usage(args.stage).free < sum(largest):
            raise ValueError("Insufficient space for staging the largest concurrent transfers")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda item: collect(item, args.stage), jobs))
    except OSError:
        export_progress(jobs, stem)
        raise
    checks = verify_native_windows(results)
    export(stem, results)
    export(stem + "_window_check", checks)
    export_progress(jobs, stem)
    print(f"Verified {len(results)} complete objects, {sum(r['object_bytes'] for r in results):,} bytes.", flush=True)
    print(f"Matched {len(checks)} original native windows with {sum(r['pixels'] for r in checks):,} complex pixels.",
          flush=True)


if __name__ == "__main__":
    main()
