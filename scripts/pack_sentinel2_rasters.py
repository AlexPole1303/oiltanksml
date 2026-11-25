"""Losslessly compress completed local Sentinel-2 L2A GeoTIFF clips."""
import argparse
import hashlib
import itertools
import json
import os
import stat
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import ColorInterp, Compression
from rasterio.shutil import copy as copy_raster

from collect_public import RAW, ROOT
from pack_local_satellite_metadata import creation_api, creation_time, extended_attributes, restore_times, sha256

STORES = ("imagery_core", "imagery_historical", "sentinel2_supplemental", "sentinel2_supplemental_original")
BANDS = {"blue.tif", "green.tif", "red.tif", "nir.tif", "scl.tif"}
FILE_FIELDS = ("st_mtime_ns", "st_mode", "st_uid", "st_gid", "st_flags")
RASTER_FIELDS = ("width", "height", "count", "crs", "transform", "dtypes", "nodatavals", "scales", "offsets", "units",
                 "descriptions", "colorinterp", "mask_flag_enums")


def tags(dataset, index=0):
    excluded = {"IMAGE_STRUCTURE", "DERIVED_SUBDATASETS"}
    return {name: dataset.tags(index, ns=name) for name in (None, *dataset.tag_namespaces(index)) if name not in excluded}


def verify_copy(source, packed):
    if any(getattr(source, field) != getattr(packed, field) for field in RASTER_FIELDS):
        raise ValueError("Raster geometry or band metadata changed")
    if tags(source) != tags(packed) or tags(source, 1) != tags(packed, 1):
        raise ValueError("Raster metadata tags changed")
    if source.colorinterp == (ColorInterp.palette,) and source.colormap(1) != packed.colormap(1):
        raise ValueError("Raster palette changed")
    if packed.compression != Compression.lerc_zstd or len(packed.files) != 1:
        raise ValueError("Unexpected output compression or an external output sidecar")
    pixels, mask = source.read(1), source.read_masks(1)
    if not np.array_equal(pixels, packed.read(1)) or not np.array_equal(mask, packed.read_masks(1)):
        raise ValueError("Pixel values or masks changed")
    return hashlib.sha256(pixels.tobytes()).hexdigest(), hashlib.sha256(mask.tobytes()).hexdigest()


def pack(libc, path):
    record = {"path": str(path.relative_to(ROOT))}
    original = path.lstat()
    if not stat.S_ISREG(original.st_mode) or original.st_nlink != 1:
        return {**record, "status": "unsupported_file"}
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(path) as source:
        if source.compression != Compression.deflate:
            return {**record, "status": "already_packed_or_other_codec"}
        if (len(source.files) != 1 or source.count != 1 or source.dtypes[0] not in {"uint8", "uint16"}
                or source.overviews(1) or source.gcps[0] or source.rpcs is not None or not source.is_tiled):
            return {**record, "status": "unsupported_raster_layout"}
        created = creation_time(libc, path)
        with tempfile.TemporaryDirectory(prefix=".raster-pack-", dir=path.parent) as work:
            output = Path(work) / path.name
            copy_raster(path, output, driver="GTiff", tiled=True, blockxsize=256, blockysize=256,
                        sparse_ok=True, compress="LERC_ZSTD", zstd_level=15, max_z_error=0)
            if output.stat().st_blocks >= original.st_blocks:
                return {**record, "status": "no_allocation_saving"}
            with rasterio.open(output) as packed:
                pixel_sha, mask_sha = verify_copy(source, packed)
            before_sha, after_sha = sha256(path), sha256(output)
            if extended_attributes(path) != extended_attributes(output):
                raise ValueError("Extended file attributes changed")
            os.chmod(output, stat.S_IMODE(original.st_mode))
            restore_times(libc, output, original, created)
            packed_stat = output.stat()
            if (any(getattr(original, field) != getattr(packed_stat, field) for field in FILE_FIELDS)
                    or creation_time(libc, output) != created):
                raise ValueError("File permissions or timestamps changed")
            current = path.lstat()
            if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
                    original.st_ino, original.st_size, original.st_mtime_ns, original.st_ctime_ns):
                raise ValueError("Source changed during verification")
            output.replace(path)
    return {**record, "status": "packed", "source_sha256": before_sha, "packed_sha256": after_sha,
            "pixel_sha256": pixel_sha, "mask_sha256": mask_sha,
            "before_allocated_bytes": original.st_blocks * 512, "after_allocated_bytes": packed_stat.st_blocks * 512}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=STORES, action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if sys.platform != "darwin" or args.workers < 1:
        parser.error("Requires macOS and a positive worker count")
    libc = creation_api()
    paths = (path for scope in args.scope or STORES
             for path in sorted((RAW / scope / "sentinel2").glob("*/*/*.tif"))
             if path.name in BANDS and path.stat().st_size >= 131072 and (path.parent / "item.json").is_file())
    selected = itertools.islice(paths, args.limit)
    counts, saved = Counter(), 0
    log = ROOT / "data" / "logs" / "sentinel2_raster_packing.jsonl"
    with log.open("a", buffering=1) as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for batch in itertools.batched(selected, args.workers * 8):
            pending = [(path, pool.submit(pack, libc, path)) for path in batch]
            for path, future in pending:
                try:
                    result = future.result()
                except Exception as error:
                    result = {"path": str(path.relative_to(ROOT)), "status": "original_preserved", "detail": str(error)}
                print(json.dumps(result), file=stream)
                counts[result["status"]] += 1
                if result["status"] == "packed":
                    saved += result["before_allocated_bytes"] - result["after_allocated_bytes"]
            if sum(counts.values()) % (args.workers * 128) == 0:
                print(f"{sum(counts.values()):,} clips checked, {saved / 1024 ** 3:.3f} GiB saved", flush=True)
    print(f"{dict(counts)}, {saved / 1024 ** 3:.3f} GiB saved", flush=True)


if __name__ == "__main__":
    main()
