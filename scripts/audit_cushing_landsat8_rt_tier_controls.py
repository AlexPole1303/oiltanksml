"""Match Landsat 8 RT and Tier 1 products of the same acquisition on the same roof."""

import hashlib
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from audit_shadow_precision import describe
from audit_cushing_landsat8_rapid_archive import STORE, pan_pixels
from build_cross_sensor_shadow import along_track_bearing, pan_psf, published_lsf
from build_landsat_pan_roof_angles import angle_values, prepare_tool
from collect_cushing_landsat7 import LANDSAT, check_grid, parse_mtl, source_bytes
from collect_landsat import GCS_OBJECT
from collect_landsat_c2_registration_context import gcs_version
from collect_public import write
from measure_cushing_landsat8_rapid import CRS, DERIVED, ROOT, STEM, RESOLVED, export, fit_roof, legacy_geometry, roof_inputs, rows, save_json


def final_scenes():
    index = {}
    for path in sorted((LANDSAT/"legacy_pan").glob("*/*_T1/item.json")):
        item = json.loads(path.read_bytes())
        product = item["product_id"]
        key = tuple(product.split("_")[2:4])
        if key in index:
            raise ValueError(f"Duplicate final version for {key}")
        index[key] = item,path.parent
    return index


def closest_controls():
    # Use the same roofs as the published near-coincident Sentinel-2 diagnostic.
    grouped = defaultdict(list)
    for r in rows(DERIVED/f"{STEM}_height_sentinel_controls.csv"):
        if float(r["capture_separation_hours"])<=3 and r["signed_gap_difference_m"]:
            grouped[r["landsat_product_id"]].append(r)
    return grouped


def collect(job, tool, geometry, lsf):
    rt, controls, (final,folder) = job
    product = final["product_id"]
    before = datetime.fromisoformat(rt["captured_at_utc"])
    after = datetime.fromisoformat(final["captured_at_utc"])
    if abs((before-after).total_seconds())>1 or final["collection2_reference_id"]!=rt["collection2_reference_id"]:
        raise ValueError(f"RT and T1 do not identify the same Landsat 8 capture: {product}")
    objects = {Path(x["name"]).name:x for x in json.loads((folder/"objects.json").read_bytes())["records"]}
    qa, pan, mtl, ang = (objects[f"{product}_{label}"] for label in ("BQA.TIF","B8.TIF","MTL.txt","ANG.txt"))
    metadata = parse_mtl(source_bytes(mtl,folder/"MTL.txt"))
    angles = folder/"ANG.txt"
    source_bytes(ang,angles)
    if metadata["LANDSAT_PRODUCT_ID"]!=product:
        raise ValueError(f"T1 metadata names a different Landsat 8 source: {product}")
    url = GCS_OBJECT+pan["name"]
    generation = gcs_version(url,pan)
    kernel = pan_psf(lsf,along_track_bearing(angles,CRS))
    gain,bias = (float(metadata[f"REFLECTANCE_{k}_BAND_8"]) for k in ("MULT","ADD"))
    output, arrays = [], {}
    with (rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",GDAL_HTTP_TIMEOUT="30",GDAL_HTTP_MAX_RETRY="2"),
          rasterio.open(url) as image, MemoryFile(source_bytes(qa)) as qa_file, qa_file.open() as quality):
        check_grid(image,metadata,15,"uint16","PANCHROMATIC")
        check_grid(quality,metadata,30,"uint16","REFLECTIVE")
        roof_rows = []
        for r in controls:
            g = geometry[r["candidate_id"]]
            x,y = (float(g[f"center_{axis}_m"]) for axis in ("east","north"))
            roof_rows.append((r,g,x,y,float(g["diameter_m"])/2,image.index(x,y)))
        resolved = angle_values(tool,angles,[row[-1] for row in roof_rows])
        for r,g,x,y,radius,point in roof_rows:
            fit = dict(fitted_pixels=0,fit_status="unresolved_t1_angles",gap_m=None,center_at_boundary=None)
            angle = resolved[point]
            if angle["angle_status"]=="resolved":
                size = radius+35
                dn, grid, _, quality_values, qa_grid, _, _, usable = pan_pixels(image,quality,(x-size,y-size,x+size,y+size))
                tank = r["candidate_id"]
                arrays.update({f"{tank}_{name}":v for name,v in (("dn",dn),("usable",usable),("grid",tuple(grid)[:6]),
                                                                 ("bqa",quality_values),("bqa_grid",tuple(qa_grid)[:6]))})
                roof = dict(candidate_id=r["candidate_id"],center_east_m=x,center_north_m=y,radius_m=radius)|angle
                inputs, valid = roof_inputs((dn,usable,grid),roof,g,gain,bias)
                fit = fit_roof(inputs,kernel) if len(inputs[0][-1])>=12 and valid>=4 else dict(
                    fit, fitted_pixels=len(inputs[0][-1]),fit_status="insufficient_t1_pixels")
            passes = fit["fit_status"]==RESOLVED and not fit["center_at_boundary"]
            rt_gap = float(r["landsat_gap_m"])
            output.append(dict(candidate_id=r["candidate_id"],rt_product_id=rt["product_id"],t1_product_id=product,
                acquisition_utc=rt["captured_at_utc"], t1_archive_bundle_at_utc=max(
                    (x["timeCreated"] for x in (qa,pan,mtl,ang)),key=datetime.fromisoformat), t1_b8_url=url,
                t1_b8_generation=generation,t1_b8_crc32c=pan["crc32c"],rt_gap_m=rt_gap,t1_gap_m=fit["gap_m"],
                t1_fitted_pixels=fit["fitted_pixels"],t1_fit_status=fit["fit_status"],t1_center_at_boundary=fit["center_at_boundary"],
                same_acquisition_rt_minus_t1_m=rt_gap-fit["gap_m"] if passes else None,
                sentinel_version_id=r["sentinel_version_id"],sentinel_captured_at_utc=r["sentinel_captured_at_utc"],
                rt_minus_sentinel_m=float(r["signed_gap_difference_m"])))
    if generation!=gcs_version(url,pan):
        raise ValueError(f"T1 archive generation changed: {product}")
    with io.BytesIO() as stream:
        np.savez_compressed(stream,**arrays)
        payload = stream.getvalue()
    packet = STORE/"rt_t1_controls"/f"{product}.npz"
    write(packet,payload)
    return [dict(r,t1_native_packet=str(packet.relative_to(ROOT)),t1_packet_sha256=hashlib.sha256(payload).hexdigest(),
                 t1_angles=str(angles.relative_to(ROOT)),t1_angle_sha256=hashlib.sha256(angles.read_bytes()).hexdigest()) for r in output]


def main():
    versions, controls = final_scenes(),closest_controls()
    rapid = {r["product_id"]:r for r in rows(DERIVED/f"{STEM}_native_scenes.csv")}
    jobs = [(rapid[p],g,versions[tuple(p.split("_")[2:4])]) for p,g in sorted(controls.items())
            if tuple(p.split("_")[2:4]) in versions]
    tool,geometry,lsf = prepare_tool(),legacy_geometry(),published_lsf()
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = [r for g in pool.map(lambda j:collect(j,tool,geometry,lsf),jobs) for r in g]
    if records:
        export(f"{STEM}_rt_t1_controls",records)
    passed = [r for r in records if r["same_acquisition_rt_minus_t1_m"] is not None]
    report = dict(reference_rt_roof_heights=sum(map(len,controls.values())),matched_t1_scenes=len(jobs),tested_roofs=len(records),
        passed_same_acquisition_controls=describe([r["same_acquisition_rt_minus_t1_m"] for r in passed]),
        passing_t1_scenes=len({r["t1_product_id"] for r in passed}),
        scope="The exact same Landsat 8 acquisition in RT and T1 processing. The cohort starts from accepted near-coincident Sentinel-2 controls. T1 serves as a retrospective processing reference.")
    save_json(DERIVED/f"{STEM}_rt_t1_summary.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
