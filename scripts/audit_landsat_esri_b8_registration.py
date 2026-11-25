"""Check exterior registration on Esri panchromatic pixels and exact Sentinel-2 versions."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from statistics import median
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.windows import Window

from audit_landsat_c2_context_registration import (PADDING_M, PATCH_M, SEARCH_RADIUS_M, analysis_grid, analysis_pixels,
    estimate, resample_pixels, source_inputs)
from build_cross_sensor_shadow import pan_qa
from build_landsat_c2_b8_measurements import verified_file
from build_s1_reference_matches import export, read
from build_shadow_registration_inputs import anchor_grid, tank_points
from build_weekly_shadow_panel import DERIVED
from collect_landsat_c2_gap_geometry import native_b8_grid
from collect_landsat_esri_b8 import STORE, inputs, raster_request
from collect_public import ROOT, get, write
from stage_landsat_c2_b8_inputs import QA_LABELS, scene_qa

STEM = "landsat_esri_b8_registration"
PAIR_ID = ("week_ending","candidate_id","previous_version_id","current_version_id","previous_source_file","current_source_file")


def context_crop(product, object_id, bounds, roof_chips):
    grid = native_b8_grid(product)
    west,south,east,north = bounds
    left,top = (~grid)*(west,north)
    right,bottom = (~grid)*(east,south)
    win = math.floor(top),math.ceil(bottom),math.floor(left),math.ceil(right)
    r0,r1,c0,c1 = win
    if not (0<=r0<r1<=product["native_lines"] and 0<=c0<c1<=product["native_samples"]):
        raise ValueError(f"Esri exterior window exceeds the source scene {product['product_id']}")
    url,transform = raster_request(product,object_id,win)
    path = STORE / "exterior" / product["product_id"] / f"{object_id}_{r0}_{r1}_{c0}_{c1}.tif"
    receipt = path.with_suffix(".json")
    raw = path.read_bytes() if path.is_file() and receipt.is_file() else None
    if raw is not None and json.loads(receipt.read_bytes()) != dict(url=url,sha256=hashlib.sha256(raw).hexdigest()):
        raw = None
    cached = raw is not None
    if not cached:
        raw = get(url)
    with MemoryFile(raw) as memory,memory.open() as raster:
        if (raster.crs.to_string()!=product["crs"] or raster.transform!=transform or raster.shape!=(r1-r0,c1-c0)
                or raster.dtypes!=("int16",)):
            raise ValueError(f"Esri exterior raster differs from the requested native output grid: {path.name}")
        shared_tiles,shared_valid,disagreements,mask_differences = 0,0,0,0
        for chip in roof_chips:
            a,b,c,d = [int(chip[k]) for k in ("first_line","end_line_exclusive","first_sample","end_sample_exclusive")]
            lo,hi,lft,rgt = max(a,r0),min(b,r1),max(c,c0),min(d,c1)
            if lo>=hi or lft>=rgt:
                continue
            if int(chip["object_id"])!=object_id:
                raise ValueError(f"Esri roof chip and exterior use different raster IDs: {path.name}")
            piece = Window(lft-c0,lo-r0,rgt-lft,hi-lo)
            earlier_piece = Window(lft-c,lo-a,rgt-lft,hi-lo)
            raw_chip = (ROOT/chip["local_path"]).read_bytes()
            if hashlib.sha256(raw_chip).hexdigest()!=chip["sha256"]:
                raise ValueError(f"Esri roof chip digest changed: {chip['local_path']}")
            with MemoryFile(raw_chip) as previous,previous.open() as small:
                y,x = raster.read(1,window=piece),small.read(1,window=earlier_piece)
                y_mask,x_mask = raster.read_masks(1,window=piece)>0,small.read_masks(1,window=earlier_piece)>0
            shared = x_mask & y_mask
            shared_tiles += 1
            shared_valid += int(shared.sum())
            disagreements += int((shared & (x!=y)).sum())
            mask_differences += int((x_mask!=y_mask).sum())
    digest = hashlib.sha256(raw).hexdigest()
    if not cached:
        write(path,raw)
        write(receipt,json.dumps(dict(url=url,sha256=digest)).encode())
    status = ("no_shared_valid_pixels" if not shared_valid else "exports_disagree" if disagreements or mask_differences
              else "shared_pixels_match")
    return dict(product_id=product["product_id"],object_id=object_id,raster_path=str(path.relative_to(ROOT)),
        raster_sha256=digest,source_url=url,raster_size_bytes=len(raw),roof_tiles_with_shared_pixels=shared_tiles,
        shared_valid_pixel_comparisons=shared_valid,shared_value_disagreements=disagreements,
        shared_mask_disagreements=mask_differences,overlap_status=status)


def pan_pixels(source, grid):
    path = ROOT/source["raster_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest()!=source["raster_sha256"]:
        raise ValueError(f"Esri exterior TIFF digest changed: {path.name}")
    with ExitStack() as stack:
        raster = stack.enter_context(rasterio.open(path))
        qa = [stack.enter_context(rasterio.open(verified_file(ROOT/source[f"{k}_path"],source[f"{k}_sha512"])))
              for k in QA_LABELS]
        west,south,east,north = raster.bounds
        corners = [("",x,y,0,0,0) for x,y in ((west,north),(east,south))]
        q,valid,flags,transform = scene_qa(source,qa,corners)
        view = SimpleNamespace(transform=transform,height=q.shape[0],width=q.shape[1])
        rows,cols = np.indices(raster.shape)
        x,y = raster.transform*(cols.ravel()+.5,rows.ravel()+.5)
        _,valid_qa,flags_qa = pan_qa(view,x,y,q,valid,flags)
        raw = raster.read(1)
        keep = (raster.read_masks(1)>0)&(raw>0)&(raw<10000)&(valid_qa&(flags_qa==0)).reshape(raw.shape)
        return resample_pixels((raw,keep,raster.transform),grid)


def pixels(source, grid):
    return pan_pixels(source,grid) if source["band"]=="Esri_B8" else analysis_pixels(source,grid)


def padded_chip_bounds(product, chips):
    grid = native_b8_grid(product)
    r0,c0 = (max(0,min(int(r[key]) for r in chips)-1) for key in ("first_line","first_sample"))
    r1,c1 = (min(product[limit],max(int(r[key]) for r in chips)+1) for limit,key in (
        ("native_lines","end_line_exclusive"),("native_samples","end_sample_exclusive")))
    west,north = grid*(c0,r0)
    east,south = grid*(c1,r1)
    return west,south,east,north


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources-only",action="store_true",help="Check larger TIFFs against roof chips before fitting shadows")
    args = parser.parse_args()
    products = {r["product_id"]:int(r["object_id"]) for r in read("landsat_esri_b8_scenes") if r["object_id"]}
    rows = [r for r in read("landsat_c2_b8_weekly_pairs")
            if all(r[f"{s}_sensor"]!="B8" or r[f"{s}_version_id"] in products for s in ("previous","current"))]
    stages, geometry, _, _ = inputs()
    tanks, exclusion = tank_points()
    anchors = anchor_grid({r["candidate_id"]:tanks[r["candidate_id"]] for r in rows},exclusion,
        search_radius_m=SEARCH_RADIUS_M,patch_m=PATCH_M,padding_m=PADDING_M)
    _,_,bounds = analysis_grid({p for group in anchors.values() for p in group})
    chips = defaultdict(list)
    for r in read("landsat_esri_b8_tiles"):
        chips[r["product_id"]].append(r)
    needed = {r[f"{s}_version_id"] for r in rows for s in ("previous","current") if r[f"{s}_sensor"]=="B8"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        crops = list(pool.map(lambda p:dict(context_crop(geometry[p],products[p],
            bounds if p in needed else padded_chip_bounds(geometry[p],chips[p]),chips[p]),
            used_in_weekly_exterior_search=p in needed),sorted(products)))
    by_product = {r["product_id"]:r for r in crops}
    export(f"{STEM}_sources",crops)
    rejected = sorted(p for p,r in by_product.items() if r["overlap_status"]!="shared_pixels_match")
    print(f"Checked {len(crops)} Esri exterior exports against the roof tiles, flagged {len(rejected)} scenes",flush=True)
    if args.sources_only:
        return
    rows = [r for r in rows if all(r[f"{s}_version_id"] not in rejected for s in ("previous","current"))]
    sources,pairs = source_inputs(rows)
    for key in sources:
        if key[0]=="B8":
            name = key[1]
            sources[key].update(geometry[name] | stages[name] | by_product[name],band="Esri_B8")
    _,scene_pairs,results,_ = estimate(rows,sources_and_pairs=(sources,pairs),load_pixels=pixels,context_label="B8/B04")
    if len(results)!=len(rows):
        raise ValueError("Esri exterior audit lost one or more exact candidate pairs")
    roof_fits = {tuple(r[k] for k in PAIR_ID):r for r in read("landsat_esri_b8_controls_candidate_pairs")}
    clock = {tuple(r[k] for k in PAIR_ID):r for r in read("cushing_b8_hls_lineage_pairs")}
    if roof_fits.keys()!=clock.keys():
        raise ValueError("Esri roof and USGS lineage ledgers disagree on the exact weekly candidate pairs")
    registered_by_week = defaultdict(list)
    for r in results:
        key = tuple(r[k] for k in ("week_or_control_day","candidate_id","earlier_version_id","later_version_id","earlier_source","later_source"))
        roof = roof_fits[key]
        evidence = clock[key]
        if (roof["esri_inputs_usable_for_retrospective_fit"]!="True" or r["release_date"]!=roof["release_date"]
            or (evidence["release_date"],evidence["forecast_cutoff_utc"])!=(roof["release_date"],roof["forecast_cutoff_utc"])):
            raise ValueError(f"Esri exterior and roof source lists differ for {key}")
        r.update(forecast_cutoff_utc=roof["forecast_cutoff_utc"],both_roof_fits_resolved=roof["both_fits_resolved"]=="True",
                 prior_red_registration_status=roof["context_registration_status"],
                 prior_red_context_before_cutoff=r.pop("red_context_before_cutoff"),
                 prior_red_context_available_at_utc=r.pop("red_context_available_at_utc"),
                 earlier_exterior_raster=r.pop("earlier_red_raster"),later_exterior_raster=r.pop("later_red_raster"),
                 cdse_source_bundle_clocks_before_cutoff=evidence["cdse_catalog_all_clocks_before_cutoff"]=="True",
                 hls_exact_scene_lineage_before_cutoff=evidence["all_hls_lineage_before_cutoff"]=="True",
                 usgs_original_b8_files_before_cutoff_confirmed=evidence["historical_b8_file_availability_confirmed"]=="True",
                 esri_pixels_before_cutoff_verified=False)
        registered_by_week[r["week_or_control_day"]].append(r)
    export(f"{STEM}_pairs",results)
    calendar = []
    for week in read("landsat_esri_b8_controls_weeks"):
        group = registered_by_week.pop(week["week_ending"],[])
        checked = [r for r in group if r["registration_status"]=="checked_offset"]
        resolved = [r for r in checked if r["both_roof_fits_resolved"]]
        if (len(group)!=int(week["pairs_with_usable_esri_inputs"]) or any(
            (r["release_date"],r["forecast_cutoff_utc"])!=(week["release_date"],week["forecast_cutoff_utc"]) for r in group)):
            raise ValueError(f"Esri exterior registration differs from the weekly roof-fit ledger on {week['week_ending']}")
        calendar.append({k:week[k] for k in ("week_ending","release_date","forecast_cutoff_utc","b04_status")} |
            dict(candidate_pairs=len(group),pairs_with_checked_exterior_offsets=len(checked),
                 checked_pairs_with_both_roof_fits=len(resolved),
                 checked_tanks_with_both_roof_fits=len({r["candidate_id"] for r in resolved}),
                 checked_tanks_with_both_roof_fits_and_source_clock=len({r["candidate_id"] for r in resolved if
                     r["cdse_source_bundle_clocks_before_cutoff"] or r["hls_exact_scene_lineage_before_cutoff"]})))
    if registered_by_week:
        raise ValueError("Unexpected Esri exterior registration weeks")
    export(f"{STEM}_weeks",calendar)
    checked = [r for r in results if r["registration_status"]=="checked_offset"]
    resolved = [r for r in checked if r["both_roof_fits_resolved"]]
    summary = dict(esri_exterior_rasters=len(crops),esri_exterior_raster_bytes=sum(r["raster_size_bytes"] for r in crops),
        rasters_for_weekly_exterior_search=len(needed),control_rasters_checked_for_overlap=len(crops)-len(needed),
        roof_tiles_compared=sum(r["roof_tiles_with_shared_pixels"] for r in crops),
        shared_valid_pixel_comparisons=sum(r["shared_valid_pixel_comparisons"] for r in crops),
        sources_with_consistent_overlap=len(crops)-len(rejected),rejected_products=rejected,
        shared_value_disagreements=sum(r["shared_value_disagreements"] for r in crops),
        shared_mask_disagreements=sum(r["shared_mask_disagreements"] for r in crops),
        scene_pairs=len(scene_pairs),candidate_pairs=len(results),registration_status=Counter(r["registration_status"] for r in results),
        checked_pairs=len(checked),checked_pairs_with_both_roof_fits=len(resolved),
        checked_pairs_with_both_roof_fits_and_prior_checked_red_context=sum(
            r["prior_red_registration_status"]=="checked_offset" and r["prior_red_context_before_cutoff"] for r in resolved),
        checked_roof_pairs_median_offset_m=median(math.hypot(r["later_minus_earlier_east_m"],r["later_minus_earlier_north_m"])
            for r in resolved) if resolved else None,
        checked_roof_pairs_median_spatial_check_residual_m=median(r["check_median_residual_m"] for r in resolved) if resolved else None,
        checked_tank_weeks_with_both_roof_fits=len({(r["week_or_control_day"],r["candidate_id"]) for r in resolved}),
        checked_weeks_with_both_roof_fits=len({r["week_or_control_day"] for r in resolved}),
        checked_tank_weeks_with_both_roof_fits_and_source_clock=sum(r["checked_tanks_with_both_roof_fits_and_source_clock"] for r in calendar),
        checked_weeks_with_both_roof_fits_and_source_clock=sum(r["checked_tanks_with_both_roof_fits_and_source_clock"]>0 for r in calendar),
        scope="Retrospective B8 exterior registration on a common 30 m analysis grid, with separate patches for fitting and checking. "
              "The offsets have not been applied to the roof fits. The Esri files have no established pre-release archive timestamps.")
    payload = json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
