"""Collect scene-locked panchromatic reflectance on the exact USGS B8 output grid."""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
from urllib.parse import urlencode

import numpy as np
from affine import Affine
from rasterio.io import MemoryFile

from build_cushing_icesat2_b8_inputs import NATIVE as LASER_NATIVE, source_tables as laser_sources
from build_landsat_c2_b8_measurements import NATIVE, source_tables
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export, read_only
from collect_landsat_c2_gap_geometry import native_b8_grid
from collect_public import RAW, ROOT, get, write

STEM = "landsat_esri_b8"
STORE = RAW / "landsat" / STEM
SERVER = "https://landsat.arcgis.com/arcgis/rest/services/Landsat/Pan/ImageServer"
FIELD = ("OBJECTID", "Name", "LANDSAT_PRODUCT_ID", "LANDSAT_SCENE_ID", "AcquisitionDate", "DateUpdated", "SensorName", "Category")
PRODUCT = re.compile(r"LC0[89]_L1TP_\d{6}_\d{8}_\d{8}_02_T1")
BLOCK = 256


def query(where):
    url = f"{SERVER}/query?" + urlencode(dict(f="json", where=where, returnGeometry="false", outFields=",".join(FIELD)))
    raw = get(url)
    data = json.loads(raw)
    if "error" in data or data.get("exceededTransferLimit"):
        raise ValueError(f"Incomplete Esri catalogue reply: {data.get('error')}")
    return url,raw,[r["attributes"] for r in data["features"]]


def catalog(products):
    raw = get(f"{SERVER}?f=json", STORE / "service.json", refresh=True)
    schema = json.loads(raw)
    if schema["pixelType"] != "S16" or schema["bandCount"] != 1 or "10000" not in schema["serviceDescription"]:
        raise ValueError("Esri PAN service description or value encoding changed")
    names = sorted(products)
    if any(not PRODUCT.fullmatch(n) for n in names):
        raise ValueError("Invalid Landsat product identifier in the acquisition queue")
    exact, alternatives = {}, defaultdict(list)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(url,raw,items) in enumerate(pool.map(query, ("Name IN (" + ",".join(f"'{n}'" for n in names[i:i+10]) + ")"
                    for i in range(0,len(names),10)))):
            write(STORE / f"exact_{i:02}.json", raw)
            for r in items:
                n = r["Name"]
                if n not in products or r["LANDSAT_PRODUCT_ID"] != n or r["Category"] != 1 or n in exact:
                    raise ValueError(f"Esri result differs from the requested exact item: {r}")
                exact[n] = r
        missing = [n for n in names if n not in exact]
        jobs = ["Name LIKE '" + "_".join(n.split("_")[:4]) + "_%'" for n in missing]
        for n,(url,raw,items) in zip(missing,pool.map(query,jobs),strict=True):
            write(STORE / "alternatives" / f"{n}.json", raw)
            alternatives[n] = sorted(r["Name"] for r in items if r["Name"].startswith("_".join(n.split("_")[:4])+"_"))
    return exact,alternatives


def roof_blocks(db, product):
    blocks = defaultdict(list)
    for name,r0,r1,c0,c1 in db.execute("""SELECT candidate_id,first_native_line,last_native_line,
            first_native_sample,last_native_sample FROM roof_window WHERE product_id=?""",(product,)):
        blocks[(r0+r1)//(2*BLOCK),(c0+c1)//(2*BLOCK)].append((name,r0,r1,c0,c1))
    for (block_r,block_c),members in sorted(blocks.items()):
        yield block_r,block_c,(min(x[1] for x in members),max(x[2] for x in members)+1,
                              min(x[3] for x in members),max(x[4] for x in members)+1),members


def raster_request(product, source, window, route=SERVER):
    r0,r1,c0,c1 = window
    grid = native_b8_grid(product)
    left,top = grid*(c0,r0)
    right,bottom = grid*(c1,r1)
    wkid = int(product["crs"].split(":")[1])
    mosaic = dict(mosaicMethod="esriMosaicLockRaster",lockRasterIds=[source],mosaicOperation="MT_FIRST")
    q = dict(f="image",bbox=f"{left},{bottom},{right},{top}",bboxSR=wkid,imageSR=wkid,size=f"{c1-c0},{r1-r0}",
             format="tiff",interpolation="RSP_NearestNeighbor",mosaicRule=json.dumps(mosaic),
             renderingRule=json.dumps(dict(rasterFunction="None")))
    return route+"/exportImage?"+urlencode(q),grid * Affine.translation(c0,r0)


def collect(product, source, block, receipts):
    br,bc,window,members = block
    name = product["product_id"]
    r0,r1,c0,c1 = window
    url,expected = raster_request(product,source,window)
    path = STORE / "reflectance" / name / f"{br}_{bc}.tif"
    receipt = receipts.get(str(path.relative_to(ROOT)))
    raw = None
    if path.is_file() and receipt and receipt["source_url"] == url:
        saved = path.read_bytes()
        if hashlib.sha256(saved).hexdigest() == receipt["sha256"]:
            raw = saved
    cached = raw is not None
    if raw is None:
        raw = get(url)
    if not raw.startswith((b"II*\0",b"MM\0*")):
        raise ValueError(f"Expected TIFF for {name}/{br}/{bc}: {raw[:160]!r}")
    with MemoryFile(raw) as mem,mem.open() as raster:
        if (raster.crs.to_string()!=product["crs"] or not raster.transform.almost_equals(expected,precision=1e-7)
                or raster.count!=1 or raster.dtypes!=("int16",) or raster.shape!=(r1-r0,c1-c0)):
            raise ValueError(f"Esri output does not match the requested Landsat grid for {name}/{br}/{bc}")
        data, mask = raster.read(1), raster.read_masks(1)>0
    useful = mask & (data>0) & (data<10000)
    if not cached:
        write(path,raw)
    checksum = hashlib.sha256(raw).hexdigest()
    tiles = dict(product_id=name, object_id=source, block_row=br, block_col=bc, first_line=r0, end_line_exclusive=r1,
                 first_sample=c0, end_sample_exclusive=c1, reflectance_scale=.0001, useful_pixels=int(useful.sum()),
                 source_url=url, local_path=str(path.relative_to(ROOT)), sha256=checksum, size_bytes=len(raw),
                 local_file_written_utc=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat())
    result = []
    for tank,a,b,c,d in members:
        tile = useful[a-r0:b-r0+1,c-c0:d-c0+1]
        result.append(dict(product_id=name,candidate_id=tank,block_row=br,block_col=bc,chip_pixels=int(tile.size),
                           chip_pixels_valid=int(tile.sum()),local_path=tiles["local_path"],sha256=checksum))
    return tiles,result


def inputs():
    stages,products,_,_ = source_tables()
    laser,targets,_ = laser_sources()
    laser_stages = {r["product_id"]:r for r in read_csv("cushing_icesat2_b8_products")}
    queued = {r["product_id"]:r for r in read_csv("cushing_b8_download_manifest")}
    if (laser.keys()!=laser_stages.keys() or products.keys() & laser.keys()
        or queued.keys()!=products.keys() | laser.keys()):
        raise ValueError("Prepared B8 scenes differ from the original-file download queue")
    if any(p["captured_at_utc"]!=laser_stages[name]["captured_at_utc"] for name,p in laser.items()):
        raise ValueError("ICESat-2 control scene times differ from their geometry")
    return stages | laser_stages,products | laser,{p:LASER_NATIVE if p in laser else NATIVE for p in queued},targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-product",action="append",default=[],metavar="PRODUCT_ID",
                        help="Download the named exact products again instead of reusing stored TIFFs")
    args = parser.parse_args()
    stages,products,databases,_ = inputs()
    if set(args.refresh_product)-products.keys():
        parser.error("Refresh product must be an exact ID from the B8 queue")
    exact, alternatives = catalog(products)
    scenes = []
    for n,product in sorted(products.items()):
        r = exact.get(n)
        expected = datetime.fromisoformat(product["captured_at_utc"])
        stamp = datetime.fromtimestamp(r["AcquisitionDate"]/1000,timezone.utc) if r else None
        if r and (r["LANDSAT_SCENE_ID"]!=product["level1_scene_id"] or abs((stamp-expected).total_seconds())>=1):
            raise ValueError(f"Esri acquisition identity differs for {n}: {r}")
        scenes.append(dict(product_id=n,platform=product["platform"],purpose=stages[n]["purpose"],
            exact_product_in_esri=r is not None,object_id=r["OBJECTID"] if r else None,source_scene_id=r["LANDSAT_SCENE_ID"] if r else None,
            acquisition_utc=expected.isoformat(),esri_acquisition_utc=stamp.isoformat() if stamp else None,
            source_date_updated=r["DateUpdated"] if r else None, esri_sensor_label=r["SensorName"] if r else None,
            esri_sensor_label_matches_product=r["SensorName"]==f"Landsat {n[3]}" if r else None,
            alternate_names_json=json.dumps(alternatives[n]),
            source_type="esri_apparent_reflectance",usgs_original_verified=False,archive_available_before_eia_verified=False))
    export(f"{STEM}_scenes",scenes)
    tiles, windows, failed = [],[],[]
    jobs = []
    for database in sorted(set(databases.values())):
        with closing(read_only(database)) as db:
            jobs.extend((products[n],r["OBJECTID"],b) for n,r in exact.items() if databases[n]==database
                        for b in roof_blocks(db,n))
    receipts = {}
    path = DERIVED / f"{STEM}_tiles.csv"
    if path.is_file():
        with path.open(newline="") as stream:
            receipts = {r["local_path"]:r for r in csv.DictReader(stream) if r["product_id"] not in args.refresh_product}
    print(f"Esri matched {len(exact)}/{len(products)} exact Landsat products, requesting {len(jobs)} small grids",flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        pending = {pool.submit(collect,*job,receipts): (job[0]["product_id"],job[2][:2]) for job in jobs}
        for i,future in enumerate(as_completed(pending),1):
            name, (br,bc) = pending[future]
            try:
                tile,roof = future.result()
            except (ValueError,OSError) as error:
                failed.append(dict(product_id=name,block_row=br,block_col=bc,issue=str(error)))
            else:
                tiles.append(tile)
                windows.extend(roof)
            if i%30==0:
                print(f"Esri B8 {i}/{len(jobs)}, {len(failed)} failures",flush=True)
    export(f"{STEM}_tiles",sorted(tiles,key=lambda r:(r["product_id"],r["block_row"],r["block_col"])))
    export(f"{STEM}_windows",sorted(windows,key=lambda r:(r["product_id"],r["candidate_id"])))
    summary = dict(requested_products=len(products), exact_products_in_service=len(exact), exact_by_purpose=Counter(stages[n]["purpose"] for n in exact),
        esri_sensor_label_disagreements=sum(r["esri_sensor_label_matches_product"] is False for r in scenes),
        products_with_alternate_version=sum(bool(v) for v in alternatives.values()), staged_tiles=len(tiles), staged_roof_windows=len(windows),
        products_with_tiles=len({r["product_id"] for r in tiles}), stored_bytes=sum(r["size_bytes"] for r in tiles), failed_tiles=failed,
        scope="ArcGIS Landsat/Pan scene-locked 16-bit apparent reflectance scaled by 0.0001, exported on the USGS output grid. "
        "The service performs on-the-fly processing. No original digital numbers, USGS checksum match, or historical file clock are inferred.")
    raw = json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",raw.encode())
    print(raw,end="",flush=True)


if __name__ == "__main__":
    main()
