"""Verify all exact B8 originals in a directory before staging pixel crops."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import rasterio
from affine import Affine

from audit_cushing_usgs_b8_metadata import stac_records
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write
from stage_landsat_c2_b8_inputs import expected_sha512, raster_grid

STEM = "cushing_b8_original_verification"


def check(originals, row, items):
    identity = row["product_id"]
    item,_ = items[identity]
    pan = item["assets"]["pan"]
    filename = row["b8_filename"]
    expected = row["b8_expected_sha512"]
    if filename!=f"{identity}_B8.TIF" or expected!=expected_sha512(pan):
        raise ValueError(f"Verification manifest differs from USGS for {identity}")
    path = originals/filename
    result = dict(product_id=identity,download_priority=int(row["download_priority"]),purpose=row["purpose"],
        filename=filename,expected_sha512=expected,local_path=str(path),local_bytes=None,local_sha512=None,
        status="awaiting_original_b8",archive_first_availability_verified=False)
    if not path.is_file():
        return result
    with path.open("rb") as handle:
        actual = hashlib.file_digest(handle,"sha512").hexdigest()
    result.update(local_bytes=path.stat().st_size,local_sha512=actual)
    if actual!=expected:
        result["status"] = "checksum_mismatch"
        return result
    crs = f"EPSG:{item['properties']['proj:epsg']}"
    transform = Affine(*pan["proj:transform"][:6])
    with rasterio.open(path) as raster:
        if raster_grid(raster,{"crs":crs},15,transform)!=(0,0) or raster.shape!=tuple(pan["proj:shape"]):
            raise ValueError(f"Verified B8 bytes have an unexpected grid for {identity}")
    result["status"] = "verified_original_b8"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("originals",type=Path)
    parser.add_argument("--inspection", action="store_true", help="Verify the nine B8 scenes near operator inspections")
    args = parser.parse_args()
    directory = args.originals.resolve(strict=True)
    if not directory.is_dir():
        parser.error("originals must be a directory")
    items = stac_records()
    if args.inspection:
        requests = [dict(product_id=r["product_id"], download_priority=i, purpose=r["purpose"],
                         b8_filename=r["original_b8_filename"], b8_expected_sha512=r["expected_original_b8_sha512"])
                    for i, r in enumerate(read_csv("cushing_operator_inspection_b8_download_files"), 1)]
        stem = "cushing_operator_inspection_b8_original_verification"
    else:
        requests = read_csv("cushing_usgs_b8_metadata_scenes")
        stem = STEM
    rows = [check(directory, r, items) for r in requests]
    if len(rows)!=len({r["product_id"] for r in rows}):
        raise ValueError("Duplicate products in the B8 verification manifest")
    export(stem, rows)
    report = dict(directory=str(directory),requested_originals=len(rows),status=Counter(r["status"] for r in rows),
        scope="Whole-file SHA-512 and original 15 m TIFF grid. Verification of bytes held now supplies no historical availability timestamp.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload,end="",flush=True)
    if any(r["status"]=="checksum_mismatch" for r in rows):
        raise SystemExit(1)


if __name__=="__main__":
    main()
