"""Measure recovered 2023 Landsat reflectance with checked Level-2 angles and masks."""

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from contextlib import closing

import rasterio

from audit_cushing_landsat_catalog_recovery import STEM as RECOVERY
from audit_cushing_usgs_b8_metadata import STEM as BULK
from audit_shadow_precision import describe
from build_cross_sensor_shadow import published_lsf
from build_cushing_icesat2_b8_inputs import b8_window
from build_landsat_c2_b8_measurements import ANGLES, RESOLVED, prepare_native
from build_landsat_c2_holdout_gap_inventory import parse
from build_landsat_esri_b8_controls import fit_tile
from build_landsat_pan_roof_angles import prepare_tool
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, read_only
from build_weekly_shadow_validation import DATED, legacy_geometry
from collect_landsat_asset_times import asset_snapshot
from collect_landsat_c2_gap_geometry import packet, read_azure_asset, roof_angles
from collect_landsat_esri_b8 import raster_request
from collect_public import RAW, ROOT, write
from stage_landsat_c2_b8_inputs import QA_LABELS

STEM = "cushing_recovered_landsat_b8"
STORE = RAW/"landsat"/STEM
NATIVE = DERIVED/f"{STEM}_native.sqlite"
CONTROL_TOLERANCE = timedelta(hours=48)


def scene(recovered, bulk, geometry, tool, *, store=STORE, purpose="catalog_recovery_validation"):
    name = recovered["product_id"]
    previous_mtl = ROOT/recovered["level2_metadata_path"]
    item = json.loads(previous_mtl.with_name("item.json").read_bytes())
    folder = store/name
    evidence = folder/"level2_objects.json"
    archive = asset_snapshot(item, QA_LABELS, evidence, ("mtl.json", "mtl.xml"))["assets"]
    mtl = folder/"MTL.json"
    old_metadata = json.loads(previous_mtl.read_bytes())["LANDSAT_METADATA_FILE"]
    required = ("PRODUCT_CONTENTS", "IMAGE_ATTRIBUTES", "LEVEL1_PROCESSING_RECORD", "PROJECTION_ATTRIBUTES",
                "LEVEL1_RADIOMETRIC_RESCALING", "LEVEL1_PROJECTION_PARAMETERS")
    if "mtl.json" in archive:
        metadata = json.loads(read_azure_asset(archive["mtl.json"], mtl))["LANDSAT_METADATA_FILE"]
        agrees = all(metadata[k] == old_metadata[k] for k in required)
    else:
        original = ET.fromstring(read_azure_asset(archive["mtl.xml"], folder / "MTL.xml"))
        agrees = original.tag == "LANDSAT_METADATA_FILE" and all(
            {field.tag: field.text for field in original.findall(f"{k}/*")} == old_metadata[k] for k in required)
        metadata = old_metadata
        if agrees:
            write(mtl, previous_mtl.read_bytes())
    if not agrees:
        raise ValueError(f"Original Level-2 MTL differs from the geometry used for Esri export for {name}")
    rec = metadata["LEVEL1_PROCESSING_RECORD"]
    stamp = parse(metadata["IMAGE_ATTRIBUTES"]["DATE_ACQUIRED"]+"T"+metadata["IMAGE_ATTRIBUTES"]["SCENE_CENTER_TIME"])
    if (name!=rec["LANDSAT_PRODUCT_ID"] or rec["LANDSAT_SCENE_ID"]!=recovered["usgs_scene_id"]
        or stamp!=parse(item["properties"]["datetime"]) or stamp.date().isoformat()!=recovered["acquisition_day"]):
        raise ValueError(f"Level-2 archive metadata identifies a different Level-1 source for {name}")
    inputs = dict(product_id=name,platform=f"LANDSAT_{bulk['Satellite']}",captured_at_utc=stamp.isoformat(),
                  level1_generated_at_utc=rec["DATE_PRODUCT_GENERATED"])
    # The exact scene identifier here comes from the USGS bulk export, with the time and grid from Level-2 MTL.
    product = packet(inputs,{"properties":{"landsat:scene_id":bulk["Landsat Scene Identifier"]}},item,mtl,store=store)
    if (product["native_lines"],product["native_samples"])!=tuple(int(bulk[k]) for k in ("Panchromatic Lines","Panchromatic Samples")):
        raise ValueError(f"USGS bulk B8 dimensions disagree with ANG for {name}")
    stage = dict(purpose=purpose,local_angle_file=product["angle_local_file"],
        source_angle_sha512=hashlib.sha512((ROOT/product["angle_local_file"]).read_bytes()).hexdigest(),
        b8_saturation_flag_in_scene_mtl=metadata["IMAGE_ATTRIBUTES"]["SATURATION_BAND_8"],
        expected_original_b8_sha512=recovered["b8_expected_usgs_sha512"] or None)
    if stage["b8_saturation_flag_in_scene_mtl"] not in ("Y","N"):
        raise ValueError(f"Missing B8 saturation flag for {name}")
    for label in QA_LABELS:
        path = folder/f"{label}.tif"
        data = read_azure_asset(archive[label],path)
        stage.update({f"{label}_path":str(path.relative_to(ROOT)),f"{label}_sha512":hashlib.sha512(data).hexdigest()})
    support = max(parse(product["angle_last_modified_at_utc"]),*(parse(r["last_modified_at_utc"]) for r in archive.values()))
    stage["level2_support_available_at_utc"] = support.isoformat()
    stage["level2_assets_snapshot"] = str(evidence.relative_to(ROOT))
    candidates = set(json.loads(recovered["covered_registry_ids_json"]))&geometry.keys()
    windows,angles,unresolved = [],{},[]
    for row in roof_angles(tool,product,candidates,geometry,DATED):
        tank = row["candidate_id"]
        g = geometry[tank]
        if row["angle_status"]!="resolved":
            unresolved.append(tank)
            continue
        location = {k:g[k] for k in ("center_east_m","center_north_m")} | dict(product_id=name,candidate_id=tank,
            radius_m=float(g["diameter_m"])/2)
        window = b8_window(location,product,row,g)
        windows.append(window)
        angles[name,tank] = (window["center_native_line"],window["center_native_sample"],*(row[k] for k in ANGLES))
    r0,r1,c0,c1 = json.loads(recovered["esri_reflectance_grid_window_json"])
    url,transform = raster_request(product,int(recovered["esri_exact_object_id"]),(r0,r1,c0,c1))
    if url!=recovered["esri_reflectance_url"]:
        raise ValueError(f"Esri request differs from the independently recovered native grid for {name}")
    with rasterio.open(ROOT/recovered["esri_reflectance_path"]) as image:
        if image.transform!=transform or image.shape!=(r1-r0,c1-c0) or image.crs.to_string()!=product["crs"]:
            raise ValueError(f"Stored reflectance has changed grid for {name}")
    tile = dict(local_path=recovered["esri_reflectance_path"],sha256=recovered["esri_reflectance_sha256"],
        object_id=recovered["esri_exact_object_id"],first_line=r0,end_line_exclusive=r1,first_sample=c0,end_sample_exclusive=c1,
        reflectance_scale=0.0001)
    return product,stage,windows,angles,tile,unresolved


def optical_controls(fits):
    dates = [parse(r["optical_capture_at_utc"]) for r in fits]
    start,stop = min(dates)-CONTROL_TOLERANCE,max(dates)+CONTROL_TOLERANCE
    registry = {r["candidate_id"] for r in fits}
    indexed = defaultdict(list)
    with (DERIVED/"shadow_timing_validation_observations.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["candidate_id"] not in registry or row["geometry_mode"]!=DATED or row["fit_status"]!=RESOLVED:
                continue
            stamp = parse(row["captured_at_utc"])
            if start<=stamp<=stop:
                indexed[row["candidate_id"]].append((stamp,row))
    dates_by_tank = {}
    for tank,items in indexed.items():
        items.sort(key=lambda x:(x[0],x[1]["version_id"],x[1]["canonical_l2a_id"]))
        dates_by_tank[tank] = [x[0] for x in items]
    for b8 in fits:
        if b8["fit_status"]!=RESOLVED:
            continue
        tank = b8["candidate_id"]
        stamp = parse(b8["optical_capture_at_utc"])
        times = dates_by_tank.get(tank,[])
        left,right = bisect_left(times,stamp-CONTROL_TOLERANCE),bisect_right(times,stamp+CONTROL_TOLERANCE)
        choices = [(abs(t-stamp),s["version_id"],s["canonical_l2a_id"],s) for t,s in indexed[tank][left:right]]
        if not choices:
            continue
        span,_,_,s = min(choices,key=lambda x:x[:3])
        if b8["terminal_id"]!=s["terminal_id"] or b8["diameter_m"]!=float(s["diameter_m"]):
            raise ValueError(f"Sentinel-2 uses different dated geometry for {tank}")
        error = b8["estimated_gap_m"]-float(s["estimated_gap_m"])
        yield dict(product_id=b8["product_id"],candidate_id=tank,terminal_id=b8["terminal_id"],
            b8_capture_utc=b8["optical_capture_at_utc"],b8_gap_m=b8["estimated_gap_m"],b8_tiff=b8["local_path"],
            b8_tiff_sha256=b8["raster_sha256"],b04_version_id=s["version_id"],b04_canonical_id=s["canonical_l2a_id"],
            b04_capture_utc=s["captured_at_utc"],b04_gap_m=float(s["estimated_gap_m"]),
            b04_native_observation_id=int(s["observation_id"]),b04_observation_source=s["measurement_source"],
            b04_tile_metadata=s["tile_metadata_path"],absolute_capture_separation_hours=span.total_seconds()/3600,
            b8_minus_b04_gap_m=error)


def main():
    recovered = [r for r in read(RECOVERY) if r["esri_reflectance_path"] and r["acquisition_day"][:4]=="2023"]
    names = {r["product_id"] for r in recovered}
    bulk = {r["Landsat Product Identifier L1"]:r for r in read(f"{BULK}_regional_catalog") if r["Landsat Product Identifier L1"] in names}
    if bulk.keys()!=names or len(recovered)!=len(names):
        raise ValueError("Missing exact USGS bulk scene identities")
    geometry,tool = legacy_geometry(),prepare_tool()
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared = list(pool.map(lambda r:scene(r,bulk[r["product_id"]],geometry,tool),recovered))
    staged = {p["product_id"]:s for p,s,*_ in prepared}
    products = {p["product_id"]:p for p,*_ in prepared}
    windows = {p["product_id"]:w for p,_,w,*_ in prepared}
    angles = {k:v for _,_,_,a,*_ in prepared for k,v in a.items()}
    stats = prepare_native(staged,products,windows,angles,database=NATIVE)
    lsf = published_lsf()
    with ThreadPoolExecutor(max_workers=4) as pool:
        fits = [r for result in pool.map(lambda item:fit_tile(item[4],item[2],item[0],item[1],lsf,NATIVE),prepared) for r in result]
    if len(fits)!=len(angles) or len({(r["product_id"],r["candidate_id"]) for r in fits})!=len(angles):
        raise ValueError("Recovered scene fits do not match the native angle ledger")
    controls = list(optical_controls(fits))
    with closing(read_only(NATIVE)) as db:
        full = dict(db.execute("""SELECT product_id,sum(roof_touching_b8_cells>0 AND
                    roof_touching_b8_cells=roof_touching_b8_cells_passing_qa) FROM roof_window GROUP BY product_id"""))
    scenes = [p | s | dict(roofs_with_angles=len(w),roofs_without_angles=len(missing),
        roofs_with_full_footprint_qa=full[p["product_id"]],
        roof_fits_resolved=sum(r["product_id"]==p["product_id"] and r["fit_status"]==RESOLVED for r in fits),
        esri_tiff=t["local_path"],esri_tiff_sha256=t["sha256"],qa_basis="azure_exact_level2_object_etag_size_md5",
        original_b8_verified=False,historical_b8_available_before_eia_verified=False) for p,s,w,_,t,missing in prepared]
    fields = tuple(dict.fromkeys(k for r in fits for k in r))
    export(f"{STEM}_fits",[{k:r.get(k) for k in fields} for r in fits])
    export(f"{STEM}_scenes",scenes)
    if controls:
        export(f"{STEM}_sentinel2_controls",controls)
    report = dict(stats,fit_status=Counter(r["fit_status"] for r in fits),
        fits_resolved_by_tier=Counter(r["product_id"].split("_")[-1] for r in fits if r["fit_status"]==RESOLVED),
        roofs_without_angles=sum(r["roofs_without_angles"] for r in scenes),
        scenes_with_resolved_fits=sum(r["roof_fits_resolved"]>0 for r in scenes),
        distinct_roofs_with_resolved_fits=len({r["candidate_id"] for r in fits if r["fit_status"]==RESOLVED}),
        sentinel2_nearest_within_48h=describe([r["b8_minus_b04_gap_m"] for r in controls]),
        sentinel2_nearest_within_24h=describe([r["b8_minus_b04_gap_m"] for r in controls if r["absolute_capture_separation_hours"]<=24]),
        sentinel2_nearest_within_45m=describe([r["b8_minus_b04_gap_m"] for r in controls if r["absolute_capture_separation_hours"]<=.75]),
        historical_pairs_admitted=0,scope="Retrospective Esri B8 measurements on the native grid. QA and ANG come from the exact "
        "underlying Level-1 identity in Level-2 MTL and are verified against Azure object metadata. Cross-sensor comparisons use the "
        "nearest resolved Sentinel-2 observation per Landsat roof, with possible actual roof movement. No EIA or price values are used.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
