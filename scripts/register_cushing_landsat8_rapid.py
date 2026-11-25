"""Collect and register exterior context for the dated Landsat 8 RT pair inventory."""

import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.io import MemoryFile

from audit_cushing_landsat8_rapid_archive import DERIVED, ROOT, STEM, STORE, check_grid, export, pan_pixels, parse_mtl, rows, save_json, source_bytes
from audit_landsat_c2_context_registration import estimate, sentinel_source
from build_shadow_registration_inputs import CRS
from collect_landsat_c2_registration_context import collect_sentinel, gcs_version, source_bounds
from collect_public import write
from register_cushing_landsat7 import analysis_raster, masked_scores

CONTEXT = STORE / "registration"
SIDES = ("previous", "current")


def exterior_b8(scene, bounds):
    product = scene["product_id"]
    folder = CONTEXT / product
    local, receipt = folder / "pan.npz", folder / "item.json"
    record = dict(product_id=product, band="Landsat8_B8", bounds_wgs84=list(bounds),
        archive_url=scene["b8_url"], archive_generation=scene["b8_archive_generation"],
        archive_bytes=int(scene["b8_archive_bytes"]), archive_crc32c=scene["b8_archive_crc32c"],
        archive_modified_at_utc=scene["archive_bundle_available_utc"], raster_path=str(local.relative_to(ROOT)))
    if local.is_file() and receipt.is_file():
        prior = json.loads(receipt.read_bytes())
        if all(prior.get(k)==v for k,v in record.items()) and prior["raster_sha256"]==hashlib.sha256(local.read_bytes()).hexdigest():
            return prior
    manifest = {Path(a["name"]).name:a for a in json.loads((ROOT/scene["source_index"]).read_bytes())["records"]}
    pan_asset, qa_asset, mtl_asset = (manifest[f"{product}_{suffix}"] for suffix in ("B8.TIF", "BQA.TIF", "MTL.txt"))
    mtl = parse_mtl(source_bytes(mtl_asset, STORE/product/"MTL.txt"))
    url = record["archive_url"]
    if mtl["LANDSAT_PRODUCT_ID"] != product or gcs_version(url,pan_asset) != record["archive_generation"]:
        raise ValueError(f"B8 context source mismatch for {product}")
    with (rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="2"),
          rasterio.open(url) as pan, MemoryFile(source_bytes(qa_asset)) as qa_file, qa_file.open() as qa):
        check_grid(pan,mtl,15,"uint16","PANCHROMATIC")
        check_grid(qa,mtl,30,"uint16","REFLECTIVE")
        if pan.crs != CRS:
            raise ValueError(f"B8 context CRS mismatch for {product}")
        metric = Transformer.from_crs(4326, pan.crs, always_xy=True).transform_bounds(*bounds)
        dn, grid, window, q, qa_grid, _, _, usable = pan_pixels(pan,qa,metric)
        with io.BytesIO() as stream:
            np.savez_compressed(stream, pan_dn=dn, pan_usable=usable, bqa=q, pan_transform=tuple(grid)[:6],
                                qa_transform=tuple(qa_grid)[:6], crs=pan.crs.to_string())
            payload = stream.getvalue()
    if gcs_version(url,pan_asset) != record["archive_generation"]:
        raise ValueError(f"B8 context generation changed for {product}")
    packet = ROOT/scene["native_packet"]
    if hashlib.sha256(packet.read_bytes()).hexdigest() != scene["native_sha256"]:
        raise ValueError(f"Corrupt roof packet for {product}")
    with np.load(packet) as roof:
        t = Affine(*roof["pan_transform"])
        col, row = (~grid)*(t.c,t.f)
        if max(abs(col-round(col)),abs(row-round(row))) > 1e-8:
            raise ValueError(f"B8 context and roof pixels have different grids for {product}")
        c, r = round(col), round(row)
        old = roof["pan_dn"]
        r0,r1 = max(0,r),min(dn.shape[0],r+old.shape[0])
        c0,c1 = max(0,c),min(dn.shape[1],c+old.shape[1])
        if r0>=r1 or c0>=c1 or not np.array_equal(dn[r0:r1,c0:c1],old[r0-r:r1-r,c0-c:c1-c]):
            raise ValueError(f"B8 context disagrees with the stored roof pixels for {product}")
    write(local,payload)
    record.update(raster_sha256=hashlib.sha256(payload).hexdigest(), native_first_line=int(window.row_off),
                  native_first_sample=int(window.col_off), checked_overlapping_pixels=(r1-r0)*(c1-c0))
    save_json(receipt,record)
    return record


def main():
    pairs = rows(DERIVED / f"{STEM}_height_pair_candidates.csv")
    sources = {r["product_id"]:r for r in rows(DERIVED / f"{STEM}_native_scenes.csv")}
    wanted_pan, wanted_s2 = set(), set()
    for r in pairs:
        for side in SIDES:
            if r[f"{side}_sensor"]=="B8":
                wanted_pan.add(r[f"{side}_version_id"])
            else:
                wanted_s2.add(tuple(r[f"{side}_{field}"] for field in ("version_id","source","canonical_id","captured_at_utc")))
    bounds = source_bounds(pairs)
    with ThreadPoolExecutor(max_workers=8) as pool:
        pan_jobs = [pool.submit(exterior_b8,sources[p],bounds) for p in sorted(wanted_pan)]
        s2_jobs = [pool.submit(collect_sentinel,*job,bounds,store=CONTEXT) for job in sorted(wanted_s2)]
        pans = [job.result() for job in pan_jobs]
        sentinels = [sentinel_source(job.result()) for job in s2_jobs]
    indexed = {(sensor,r[field]):(sensor,r[field],r["raster_path"]) for sensor,field,items in
               (("B8","product_id",pans),("B04","version_id",sentinels)) for r in items}
    lookup = {indexed[sensor,r[field]]:r for sensor,field,items in
              (("B8","product_id",pans),("B04","version_id",sentinels)) for r in items}
    paired = defaultdict(list)
    for r in pairs:
        keys = tuple(indexed[r[f"{side}_sensor"],r[f"{side}_version_id"]] for side in SIDES)
        paired[keys].append(dict(r, forecast_cutoff_utc=r["cutoff_utc"]))
    _, groups, offsets, patches = estimate(pairs,sources_and_pairs=(lookup,paired),load_pixels=analysis_raster,
        context_label="Landsat 8 RT and Sentinel-2",study_phase="engineering",partial_patch_scores=masked_scores)
    for suffix, records in (("pan_sources",pans),("sentinel_sources",sentinels),("scene_pairs",groups),("offsets",offsets),("patches",patches)):
        if records:
            export(f"{STEM}_registration_{suffix}",records)
    checked = [r for r in offsets if r["registration_status"]=="checked_offset" and r["red_context_before_cutoff"]]
    report = dict(roof_pairs=len(pairs), b8_sources=len(pans), b04_sources=len(sentinels), scene_pairs=len(groups),
        exterior_patch_profiles=len(patches), checked_exterior_offsets_before_cutoff=len(checked),
        checked_by_route=dict(Counter(r["comparison"] for r in checked)),
        status_counts=dict(Counter(r["registration_status"] for r in offsets)))
    save_json(DERIVED / f"{STEM}_registration_summary.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
