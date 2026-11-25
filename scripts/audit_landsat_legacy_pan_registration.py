"""Compare B4 and B8 offsets against the same near-coincident Sentinel-2 B04."""
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from audit_landsat_c2_context_registration import (
    MAX_DISAGREEMENT_M, PADDING_M, PATCH_M, PATCH_PIXELS, PEAK_SEPARATION_M, SAMPLING,
    SEARCH_RADIUS_M, SHIFTS, ZERO, analysis_grid, analysis_pixels, resample_pixels, sentinel_source,
)
from build_cross_sensor_shadow import pan_qa
from build_landsat_c2_holdout_gap_inventory import parse
from build_landsat_coverage import bit, native_masks
from build_shadow_registration_inputs import CRS, anchor_grid, tank_points
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_imagery import clip
from collect_landsat import GCS_OBJECT, LANDSAT, catalog
from collect_landsat_c2_gap_geometry import read_azure_asset
from collect_landsat_c2_registration_context import collect_scene, collect_sentinel, gcs_version, source_bounds
from collect_public import ROOT, write
from fit_shadow_registration import peak, shifted_gradients, tank_offset
from stage_landsat_c2_b8_inputs import QA_LABELS, expected_sha512, sha256_file

STEM = "landsat_legacy_pan_registration"
STORE = LANDSAT / STEM


def legacy_crop(product, directory, label, resolution, bounds, objects):
    filename = f"{product}_{label}.TIF"
    asset = objects[filename]
    url = GCS_OBJECT + asset["name"]
    destination = STORE / product / filename
    sidecar = destination.with_suffix(".json")
    identity = dict(product_id=product,band=label,source_url=url,crc32c=asset["crc32c"],
        source_bytes=int(asset["size"]),source_created_at_utc=asset["timeCreated"], bounds_wgs84=list(bounds),
        raster_path=str(destination.relative_to(ROOT)))
    if destination.is_file() and sidecar.is_file():
        saved = json.loads(sidecar.read_bytes())
        if all(saved.get(k)==v for k,v in identity.items()) and saved["sha256"] == sha256_file(destination):
            return saved
    generation = gcs_version(url,asset)
    width,height = clip(url,destination,bounds,replace=True)
    if not width or not height or gcs_version(url,asset) != generation:
        raise ValueError(f"Missing context coverage or changing source for {filename}")
    with rasterio.open(destination) as raster:
        if raster.crs != CRS or raster.res != (resolution,resolution) or raster.dtypes != ("uint16",):
            raise ValueError(f"Incorrect native context grid for {filename}")
        if not (raster.read(1)>0).any():
            raise ValueError(f"Empty source crop for {filename}")
    identity.update(archive_generation=generation,sha256=sha256_file(destination),
                    source_directory=str(directory.relative_to(ROOT)))
    write(sidecar,json.dumps(identity,indent=2).encode())
    return identity


def collection2_context(source,mtl,bounds):
    identity = source["id"]
    item = json.loads(mtl.with_name("item.json").read_bytes())
    captured = source["properties"]["datetime"]
    clock = LANDSAT / "pc_asset_times" / captured[:4] / f"{item['id']}.json"
    archive = json.loads(clock.read_bytes())["assets"]
    pan = source["assets"]["pan"]
    a,b,x,d,e,y = pan["proj:transform"][:6]
    if (a,b,d,e) != (15,0,0,-15):
        raise ValueError(f"Unexpected Collection 2 source grid for {identity}")
    lines,samples = pan["proj:shape"]
    product = dict(product_id=identity,level2_metadata_path=str(mtl.relative_to(ROOT)),
        level2_item_id=item["id"],captured_at_utc=captured,crs=CRS,ul_b8_center_x_m=x+7.5,ul_b8_center_y_m=y-7.5,
        native_lines=lines,native_samples=samples)
    stage = dict(qa_clock_source=str(clock.relative_to(ROOT)))
    for label in QA_LABELS:
        path = STORE / identity / f"{label}.tif"
        if archive[label]["url"] != item["assets"][label]["href"]:
            raise ValueError(f"Collection 2 QA object differs for {identity}/{label}")
        read_azure_asset(archive[label],path)
        stage[f"{label}_path"] = str(path.relative_to(ROOT))
        stage[f"{label}_sha512"] = expected_sha512(source["assets"][label])
    return collect_scene(product,stage,bounds)


def context(group,bounds,reference,c2_products):
    row = group[0]
    product = row["landsat_product_id"]
    directory = ROOT / row["landsat_source_directory"]
    manifest = json.loads((directory/"item.json").read_bytes())
    if manifest["product_id"]!=product or manifest["captured_at_utc"]!=row["landsat_capture_at_utc"]:
        raise ValueError(f"Legacy metadata differs for {product}")
    c2_id = manifest["collection2_reference_id"]
    c2 = reference[c2_id]
    if (product.split("_")[2:4] != c2_id.split("_")[2:4]
        or abs((parse(c2["properties"]["datetime"])-parse(manifest["captured_at_utc"])).total_seconds()) > 1):
        raise ValueError(f"Collection 1 and 2 do not identify the same acquisition for {product}")
    objects = {a["name"].rsplit("/",1)[-1]:a for a in json.loads((directory/"objects.json").read_bytes())["records"]}
    assets = {name:legacy_crop(product,directory,name,size,bounds,objects)
              for name,size in (("B4",30),("B8",15),("BQA",30))}
    tile = ROOT / row["sentinel2_tile_metadata_path"]
    canonical = json.loads(tile.with_name("item.json").read_bytes())["canonical_l2a_id"]
    s2 = collect_sentinel(row["sentinel2_product_id"],row["sentinel2_tile_metadata_path"],canonical,
                          row["sentinel2_capture_at_utc"],bounds)
    c2 = collection2_context(c2,c2_products[c2_id],bounds)
    return assets,s2,c2


def landsat_arrays(assets,grid):
    with rasterio.open(ROOT/assets["BQA"]["raster_path"]) as qa:
        q = qa.read(1)
        good = (qa.read_masks(1)>0)&~bit(q,0)
        flags = np.logical_or.reduce(native_masks(q,None,"pan")).astype("uint8")
        for name in ("B4","B8"):
            with rasterio.open(ROOT/assets[name]["raster_path"]) as image:
                values = image.read(1)
                valid = (image.read_masks(1)>0)&(values>0)&(values<65535)
                if name == "B4":
                    if image.transform!=qa.transform or image.shape!=qa.shape:
                        raise ValueError(f"Collection 1 B4 and BQA grids differ for {assets[name]['product_id']}")
                    valid &= good&(flags==0)
                else:
                    y,x = np.indices(image.shape)
                    east,north = image.transform*(x.ravel()+.5,y.ravel()+.5)
                    _,covered,flag = pan_qa(qa,east,north,q,good,flags)
                    valid &= (covered&(flag==0)).reshape(image.shape)
                yield resample_pixels((values,valid,image.transform),grid)


def control_estimates(number,group,assets,sentinel,c2,anchors,grid):
    images = [*landsat_arrays(assets,grid),analysis_pixels(dict(c2,band="Landsat_SR_B4"),grid),
              analysis_pixels(sentinel_source(sentinel),grid)]
    inverse = ~grid[0]
    profiles = {"B4":{},"B8":{},"C2_B4":{}}
    patches = []
    for west,south,role in sorted({anchor for r in group for anchor in anchors[r["candidate_id"]]}):
        col,line = map(round,inverse*(west-PADDING_M,south+PATCH_M+PADDING_M))
        cells = np.s_[line:line+PATCH_PIXELS,col:col+PATCH_PIXELS]
        if not all(mask[cells].all() for _,mask in images):
            continue
        identifier = f"{west}_{south}"
        vectors = [shifted_gradients(signal[cells],SAMPLING) for signal,_ in images]
        for band, landsat in zip(profiles,vectors[:-1],strict=True):
            scores = np.clip(np.einsum("ij,ij->i",landsat[::-1],vectors[-1]),-1,1)
            best,separation,status = peak(scores,shifts=SHIFTS,separation_distance_m=PEAK_SEPARATION_M)
            profiles[band][number,identifier] = scores,best,status
            patches.append(dict(scene_pair_id=number,anchor_id=identifier,role=("fit","spatial_check")[role],
                band=band,peak_status=status,east_m=int(SHIFTS[best,0]),north_m=int(SHIFTS[best,1]),
                peak_correlation=float(scores[best]),peak_separation=separation))
    results = []
    for r in group:
        ids = [[],[]]
        for west,south,role in anchors[r["candidate_id"]]:
            name = f"{west}_{south}"
            if (number,name) in profiles["B4"]:
                ids[role].append(name)
        job = dict(comparison="band_transfer",study_phase="engineering",week_or_control_day=r["landsat_capture_at_utc"][:10],
            candidate_id=r["candidate_id"],scene_pair_id=number,fit_screened_patch_ids="|".join(ids[0]),
            spatial_check_screened_patch_ids="|".join(ids[1]))
        outcomes = {band:tank_offset(job,profile,shifts=SHIFTS,zero=ZERO,disagreement_m=MAX_DISAGREEMENT_M)
                    for band,profile in profiles.items()}
        passed = {band for band,p in outcomes.items() if p["registration_status"]=="checked_offset"}
        columns = ("registration_status","later_minus_earlier_east_m","later_minus_earlier_north_m",
                   "check_median_residual_m","fit_inliers","spatial_check_inliers")
        comparisons = {}
        for band in ("B4","C2_B4"):
            a,b = ({"B4":"B8","C2_B4":"B4"}[band],band)
            check = {a,b} <= passed
            delta = [outcomes[a][f"later_minus_earlier_{axis}_m"]-outcomes[b][f"later_minus_earlier_{axis}_m"]
                     for axis in ("east","north")] if check else (None,None)
            prefix = "b8_minus_b4" if band=="B4" else "c1_b4_minus_c2_b4"
            comparisons.update({f"{prefix}_east_m":delta[0],f"{prefix}_north_m":delta[1],
                                f"{prefix}_difference_m":math.hypot(*delta) if check else None})
        results.append(dict(scene_pair_id=number,candidate_id=r["candidate_id"],roof_type=r["roof_type"],
            landsat_product_id=r["landsat_product_id"],landsat_c2_product_id=c2["product_id"],
            sentinel2_version_id=r["sentinel2_product_id"],
            landsat_capture_at_utc=r["landsat_capture_at_utc"],sentinel2_capture_at_utc=r["sentinel2_capture_at_utc"],
            shared_fit_patches=len(ids[0]),shared_check_patches=len(ids[1]),**comparisons,
            **{f"{band.lower()}_{k}":v[k] for band,v in outcomes.items() for k in columns}))
    return results,patches


def main():
    matches = read_csv("landsat_sentinel2_pan_calibration_matches")
    groups = defaultdict(list)
    for r in matches:
        groups[r["landsat_product_id"],r["sentinel2_product_id"]].append(r)
    tanks,exclusion = tank_points()
    tanks = {r["candidate_id"]:tanks[r["candidate_id"]] for r in matches}
    anchors = anchor_grid(tanks,exclusion,search_radius_m=SEARCH_RADIUS_M,patch_m=PATCH_M,padding_m=PADDING_M)
    grid = analysis_grid({p for rows in anchors.values() for p in rows})
    bounds = source_bounds(matches)
    references = {json.loads((ROOT/r["landsat_source_directory"]/"item.json").read_bytes())["collection2_reference_id"]
                  for r in (group[0] for group in groups.values())}
    products = {f["id"]:f for year in sorted({int(p.split("_")[3][:4]) for p in references})
                for f in catalog(year,"usgs") if f["id"] in references}
    contexts = {}
    for mtl in sorted((LANDSAT/"collection2_context").glob("*/*/MTL.json")):
        identity = json.loads(mtl.read_bytes())["LANDSAT_METADATA_FILE"]["LEVEL1_PROCESSING_RECORD"]["LANDSAT_PRODUCT_ID"]
        if identity in references:
            if identity in contexts:
                raise ValueError(f"Repeated Collection 2 context for {identity}")
            contexts[identity] = mtl
    if references != products.keys() or references != contexts.keys():
        raise ValueError("Collection 1 controls lack an exact paired Collection 2 red source")
    results,patches,scenes,source_rows,c2_sources = [],[],[],[],[]
    ordered = [groups[key] for key in sorted(groups)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        collected = pool.map(lambda g:context(g,bounds,products,contexts),ordered)
        for number, (group,(assets,sentinel,c2)) in enumerate(zip(ordered,collected,strict=True),1):
            values,checks = control_estimates(number,group,assets,sentinel,c2,anchors,grid)
            results.extend(values)
            patches.extend(checks)
            source_rows.extend(assets.values())
            c2_sources.append(c2)
            scenes.append(dict(scene_pair_id=number,landsat_product_id=group[0]["landsat_product_id"],
                landsat_c2_product_id=c2["product_id"],
                sentinel2_version_id=sentinel["version_id"],sentinel2_path=sentinel["raster_path"],
                sentinel2_sha256=sentinel["raster_sha256"],sentinel2_generation=sentinel["archive_generation"],
                tank_comparisons=len(values),both_bands_checked=sum(r["b8_minus_b4_difference_m"] is not None for r in values),
                both_red_versions_checked=sum(r["c1_b4_minus_c2_b4_difference_m"] is not None for r in values)))
            print(f"Checked original B4 and B8 on {number}/{len(groups)} Landsat scene pairs",flush=True)
    if len(results)!=len(matches) or len({(r["scene_pair_id"],r["candidate_id"]) for r in results})!=len(results):
        raise ValueError("Legacy band audit does not reconcile with the near-coincident source roster")
    for name, rows in (("scenes",scenes),("sources",source_rows),("c2_sources",c2_sources),("tanks",results),("patches",patches)):
        export(f"{STEM}_{name}",rows)
    comparisons = {}
    for label in ("b8_minus_b4","c1_b4_minus_c2_b4"):
        field = f"{label}_difference_m"
        differences = [r[field] for r in results if r[field] is not None]
        per_scene = defaultdict(list)
        for r in results:
            if r[field] is not None:
                per_scene[r["scene_pair_id"]].append(r[field])
        comparisons[label] = dict(tank_comparisons=len(differences),scene_pairs=len(per_scene),
            median_difference_m=float(np.median(differences)) if differences else None,
            p90_difference_m=float(np.quantile(differences,.9)) if differences else None,
            median_of_scene_medians_m=float(np.median([np.median(v) for v in per_scene.values()])) if per_scene else None)
    report = dict(scene_pairs=len(scenes),tank_comparisons=len(results),
        band_status={band:dict(Counter(r[f"{band}_registration_status"] for r in results)) for band in ("b4","b8","c2_b4")},
        c1_pan_checked_c1_red_unchecked=sum(r["b8_registration_status"]=="checked_offset"
            and r["b4_registration_status"]!="checked_offset" for r in results),
        **comparisons,scope="Collection 1 Landsat 8 B4 and B8 and matched Collection 2 SR B4 compared to the same "
        "Sentinel-2 B04 on identical exterior patches. Offsets are Sentinel-2 minus Landsat. "
        "The Collection 2 comparison combines collection and radiometry changes. No Collection 2 B8 was available.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
