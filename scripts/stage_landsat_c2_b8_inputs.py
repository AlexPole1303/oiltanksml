"""Verify original Landsat B8 bytes and stage native-grid QA and roof windows."""
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_bounds

from build_cross_sensor_shadow import pan_qa
from build_landsat_c2_holdout_gap_inventory import SIDES, STEM as GAP, parse
from build_landsat_c2_external_controls import STEM as CONTROLS
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from build_weekly_shadow_validation import legacy_geometry
from collect_landsat import LANDSAT, catalog, source_coverage
from collect_landsat_c2_gap_geometry import STEM as GEOMETRY, STORE, native_b8_grid, read_azure_asset
from collect_public import ROOT, write

STEM = "landsat_c2_b8_stage"
BUFFER_METRES = 30
QA_LABELS = ("qa_pixel", "qa_radsat")


def expected_sha512(asset):
    checksum = asset["file:checksum"]
    # STAC uses a hexadecimal multihash. 0x13 is SHA-512 and 0x40 is the byte count.
    if not re.fullmatch(r"1340[0-9a-f]{128}",checksum):
        raise ValueError(f"Unsupported Landsat checksum for {asset['href']}")
    return checksum[4:]


def normalized(row):
    return row | {k:float(row[k]) for k in ("ul_b8_center_x_m","ul_b8_center_y_m")} | {
        k:int(row[k]) for k in ("native_lines","native_samples")}


def footprints():
    dated = legacy_geometry()
    for r in read_csv(f"{GEOMETRY}_roof_angles"):
        tank = dated[r["candidate_id"]]
        yield (r["product_id"],r["candidate_id"],float(tank["center_east_m"]),float(tank["center_north_m"]),
               float(tank["diameter_m"])/2,int(r["b8_native_line"]),int(r["b8_native_sample"]))
    seen = {}
    for r in read_csv(f"{CONTROLS}_roof_pairs"):
        key = r["product_id"],r["candidate_id"]
        location = (float(r["center_east_m"]),float(r["center_north_m"]),float(r["radius_m"]),
                    int(r["b8_native_line"]),int(r["b8_native_sample"]))
        if key in seen:
            if seen[key]!=location:
                raise ValueError(f"Independent control geometry differs within {key}")
            continue
        seen[key] = location
        yield (*key,*location)


def raster_grid(raster, product, resolution, grid):
    if (raster.count!=1 or raster.dtypes!=("uint16",) or not raster.crs or raster.crs.to_string()!=product["crs"]
        or (raster.transform.a,raster.transform.b,raster.transform.d,raster.transform.e)!=(resolution,0,0,-resolution)):
        raise ValueError(f"Wrong Landsat grid or pixel format in {Path(raster.name).name}")
    col,row = (~grid)*(raster.transform.c,raster.transform.f)
    if max(abs(col-round(col)),abs(row-round(row)))>1e-8:
        raise ValueError(f"Off-grid Landsat origin in {Path(raster.name).name}")
    return round(row),round(col)


def stage_pan(product, original, output, asset, extent):
    expected = expected_sha512(asset)
    with original.open("rb") as handle:
        actual = hashlib.file_digest(handle,"sha512").hexdigest()
    if actual!=expected:
        raise ValueError(f"B8 SHA-512 differs from USGS for {product['product_id']}")
    grid = native_b8_grid(product)
    with rasterio.open(original) as raster:
        if (raster_grid(raster,product,15,grid)!=(0,0) or
            raster.shape!=tuple(asset["proj:shape"]) or raster.shape!=(product["native_lines"],product["native_samples"])):
            raise ValueError(f"B8 source dimensions differ from exact USGS product {product['product_id']}")
        bounds = transform_bounds("EPSG:4326",raster.crs,*extent)
        area = from_bounds(*bounds,transform=raster.transform)
        first_line,first_sample = max(0,math.floor(area.row_off)),max(0,math.floor(area.col_off))
        stop_line = min(raster.height,math.ceil(area.row_off+area.height))
        stop_sample = min(raster.width,math.ceil(area.col_off+area.width))
        if first_line>=stop_line or first_sample>=stop_sample:
            raise ValueError(f"No B8 coverage of the project area in {original}")
        window = Window(first_sample,first_line,stop_sample-first_sample,stop_line-first_line)
        dn,mask = raster.read(1,window=window),raster.read_masks(1,window=window)
        if not ((dn>0)&(mask>0)).any():
            raise ValueError(f"No usable native B8 pixels in {output}")
        profile = raster.profile | dict(driver="GTiff",width=int(window.width),height=int(window.height),
            transform=raster.window_transform(window),tiled=True,blockxsize=256,blockysize=256,compress="deflate")
        output.parent.mkdir(parents=True,exist_ok=True)
        with TemporaryDirectory(dir=output.parent) as directory:
            temporary = Path(directory)/output.name
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(temporary,"w",**profile) as cropped:
                cropped.write(dn,1)
                cropped.write_mask(mask)
            with rasterio.open(temporary) as cropped:
                if (raster_grid(cropped,product,15,grid)!=(first_line,first_sample)
                    or not np.array_equal(cropped.read(1),dn) or not np.array_equal(cropped.read_masks(1),mask)):
                    raise ValueError(f"B8 crop altered native samples or masks for {product['product_id']}")
            temporary.replace(output)
    return actual,first_line,first_sample


def sha256_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scene_qa(product, rasters, roofs):
    qa,saturation = rasters
    qa_origin = Affine(30,0,product["ul_b8_center_x_m"]-15,0,-30,product["ul_b8_center_y_m"]+15)
    for raster in rasters:
        if raster_grid(raster,product,30,qa_origin)!=(0,0):
            raise ValueError(f"Wrong QA origin in {Path(raster.name).name}")
    shape = tuple((product[k]+1)//2 for k in ("native_lines","native_samples"))
    if any(raster.shape!=shape for raster in rasters):
        raise ValueError(f"Wrong QA dimensions for {product['product_id']}")
    margin = BUFFER_METRES + 30
    first_row,first_col = qa.index(min(e-r-margin for _,e,n,r,_,_ in roofs),
                                   max(n+r+margin for _,e,n,r,_,_ in roofs))
    last_row,last_col = qa.index(max(e+r+margin for _,e,n,r,_,_ in roofs),
                                 min(n-r-margin for _,e,n,r,_,_ in roofs))
    if not (0<=first_row<=last_row<qa.height and 0<=first_col<=last_col<qa.width):
        raise ValueError(f"QA windows extend beyond the original grid for {product['product_id']}")
    window = Window(first_col,first_row,last_col-first_col+1,last_row-first_row+1)
    q = qa.read(1,window=window)
    s = saturation.read(1,window=window)
    valid = (qa.read_masks(1,window=window)>0)&(saturation.read_masks(1,window=window)>0)&((q&1)==0)
    # Pixel QA bits 1–5 represent dilated cloud, cirrus, cloud, cloud shadow and snow.
    # QA_RADSAT bit 11 represents terrain occlusion. It has no B8 saturation bit.
    flags = (q&62) | (s&(1<<11))
    return q,valid,flags,qa.window_transform(window)


def roof_windows(product, roofs, qa):
    grid = native_b8_grid(product)
    inverse = ~grid
    q,valid,flags,transform = qa
    view = SimpleNamespace(transform=transform,height=q.shape[0],width=q.shape[1])
    for tank,e,n,r,line,sample in roofs:
        col,row = inverse*(e,n)
        if (math.floor(row),math.floor(col))!=(line,sample):
            raise ValueError(f"Roof center differs from prepared B8 angles for {product['product_id']}/{tank}")
        lo_c,lo_r = inverse*(e-r-BUFFER_METRES,n+r+BUFFER_METRES)
        hi_c,hi_r = inverse*(e+r+BUFFER_METRES,n-r-BUFFER_METRES)
        start_c,start_r = math.floor(lo_c),math.floor(lo_r)
        end_c,end_r = math.ceil(hi_c),math.ceil(hi_r)
        if not (0<=start_c<end_c<=product["native_samples"] and 0<=start_r<end_r<=product["native_lines"]):
            raise ValueError(f"Roof window exceeds the original Landsat grid for {product['product_id']}/{tank}")
        rr,cc = np.indices((end_r-start_r,end_c-start_c))
        x,y = grid*(cc+start_c+0.5,rr+start_r+0.5)
        _,ok,bits = pan_qa(view,x,y,q,valid,flags)
        usable = ok&(bits==0)
        distance = np.hypot(x-e,y-n)
        roof = distance<=r
        surrounding = (distance>r)&(distance<=r+BUFFER_METRES)
        yield dict(product_id=product["product_id"],candidate_id=tank,center_east_m=e,center_north_m=n,radius_m=r,
            center_native_line=line,center_native_sample=sample,b8_first_line=start_r,b8_last_line=end_r-1,
            b8_first_sample=start_c,b8_last_sample=end_c-1,b8_window_cells=int(rr.size),
            b8_roof_center_cells=int(roof.sum()),b8_roof_center_cells_pass_qa=int((roof&usable).sum()),
            b8_surrounding_center_cells=int(surrounding.sum()),
            b8_surrounding_center_cells_pass_qa=int((surrounding&usable).sum()))


def stage(job, input_dir, output_dir, bounds):
    request,product,item,roofs = job
    name = product["product_id"]
    context_dir = ROOT / product["level2_metadata_path"]
    context = json.loads((context_dir.parent/"item.json").read_bytes())
    metadata = json.loads(context_dir.read_bytes())["LANDSAT_METADATA_FILE"]
    if metadata["LEVEL1_PROCESSING_RECORD"]["LANDSAT_PRODUCT_ID"]!=name:
        raise ValueError(f"Level-2 metadata has a different Level-1 product for {name}")
    saturation = metadata["IMAGE_ATTRIBUTES"]["SATURATION_BAND_8"]
    if saturation not in ("N","Y"):
        raise ValueError(f"Unrecognized scene-level B8 saturation flag for {name}")
    clock_file = LANDSAT/"pc_asset_times"/product["captured_at_utc"][:4]/f"{product['level2_item_id']}.json"
    archive = json.loads(clock_file.read_bytes())["assets"]
    directory = STORE/name
    angles = ROOT / product["angle_local_file"]
    exact = expected_sha512(item["assets"]["ANG.txt"])
    if hashlib.sha512(angles.read_bytes()).hexdigest()!=exact:
        raise ValueError(f"Level-2 ANG differs from the original Level-1 ANG for {name}")
    records, paths = {}, []
    for label in QA_LABELS:
        path = directory/f"{label.upper()}.TIF"
        source = archive[label]
        if source["url"]!=context["assets"][label]["href"]:
            raise ValueError(f"Level-2 QA archive URL differs from the exact item for {name}")
        payload = read_azure_asset(source,path)
        checksum = expected_sha512(item["assets"][label])
        if hashlib.sha512(payload).hexdigest()!=checksum:
            raise ValueError(f"Level-2 {label} differs from original Level-1 QA for {name}")
        paths.append(path)
        records[label] = dict(path=str(path.relative_to(ROOT)),sha512=checksum,source=source["url"],
                              bytes=len(payload),archive_at_utc=source["last_modified_at_utc"])
    with ExitStack() as stack:
        rasters = [stack.enter_context(rasterio.open(path)) for path in paths]
        windows = list(roof_windows(product,roofs,scene_qa(product,rasters,roofs)))
    asset = item["assets"]["pan"]
    if (asset["href"]!=request["pan_url"] or tuple(asset["proj:shape"])!=(product["native_lines"],product["native_samples"])
        or Affine(*asset["proj:transform"][:6])!=native_b8_grid(product)):
        raise ValueError(f"Level-1 STAC disagrees with the B8 geometry for {name}")
    original = input_dir / f"{name}_B8.TIF"
    cropped = output_dir / name / "pan.tif"
    digest,first_line,first_sample = (stage_pan(product,original,cropped,asset,bounds)
                                      if original.is_file() else (None,None,None))
    evidence = dict(product_id=name,priority=int(request["priority"]),platform=product["platform"],
                    purpose=request["purpose"],original_b8_filename=original.name,
                    b8_download_url=asset["href"],expected_original_b8_sha512=expected_sha512(asset),
                    original_b8_sha512=digest,b8_status="verified_native_crop" if digest else "awaiting_original_b8",
                    b8_crop_path=str(cropped) if digest else None,b8_crop_sha256=sha256_file(cropped) if digest else None,
                    b8_crop_first_line=first_line,b8_crop_first_sample=first_sample,
                    source_angle_sha512=exact,local_angle_file=product["angle_local_file"],
                    angle_archive_at_utc=product["angle_last_modified_at_utc"],level2_metadata_path=product["level2_metadata_path"],
                    b8_saturation_flag_in_scene_mtl=saturation,
                    qa_clock_source=str(clock_file.relative_to(ROOT)),roof_windows=len(windows),
                    qa_and_angles_archive_at_utc=max(product["angle_last_modified_at_utc"],
                                                     *(r["archive_at_utc"] for r in records.values()),key=parse))
    for label,info in records.items():
        evidence.update({f"{label}_{k}":v for k,v in info.items()})
    return evidence,windows


def clear_roof(r):
    return r["b8_roof_center_cells"]>0 and r["b8_roof_center_cells"]==r["b8_roof_center_cells_pass_qa"]


def planned_b8_endpoints(evidence, windows, plan):
    by_product = {r["product_id"]:r for r in evidence}
    clear = {(r["product_id"],r["candidate_id"]) for r in windows if clear_roof(r)}
    for r in read_csv(f"{GAP}_products"):
        key = r["week_ending"],r["endpoint"]
        item = by_product.get(r["product_id"])
        if (item is None or r["level1_could_be_generated_by_cutoff"]!="True" or r["local_level2_before_cutoff"]!="True"
            or parse(item["qa_and_angles_archive_at_utc"])>=parse(r["cutoff_utc"])):
            continue
        wanted = set(json.loads(plan[key[0]][f"{key[1]}_b8_pair_candidate_ids_json"]))
        for tank in sorted(wanted.intersection(json.loads(r["clear_efr_candidate_ids_json"]))):
            if (r["product_id"],tank) in clear:
                yield r, tank


def weekly_qa(evidence, windows):
    plan = {r["week_ending"]:r for r in read_csv(f"{GAP}_weeks")}
    cover = defaultdict(set)
    for r,tank in planned_b8_endpoints(evidence, windows, plan):
        cover[r["week_ending"],r["endpoint"]].add(tank)
    rows = []
    for week,r in sorted(plan.items()):
        previous,current = (cover[week,side] for side in SIDES)
        both = previous & current
        mixed = (previous & set(json.loads(r["previous_b8_mixed_candidate_ids_json"]))) | (
                 current & set(json.loads(r["current_b8_mixed_candidate_ids_json"])))
        union = both | mixed
        if (not both<=set(json.loads(r["paired_candidate_ids_json"]))
            or not mixed<=set(json.loads(r["mixed_b8_and_sentinel2_candidate_ids_json"]))):
            raise ValueError(f"Native QA unexpectedly introduced new candidate pairs for {week}")
        rows.append(dict(study_phase=r["study_phase"],week_ending=week,release_date=r["release_date"],
                         forecast_cutoff_utc=r["forecast_cutoff_utc"],b8_previous_clear_roofs=len(previous),
                         b8_current_clear_roofs=len(current),clear_two_b8_pair_candidates=len(both),
                         clear_b8_sentinel2_pair_candidates=len(mixed),clear_any_b8_pair_candidates=len(union),
                         clear_candidate_ids_json=json.dumps(sorted(union))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals",type=Path,default=LANDSAT/"collection2_b8_original")
    parser.add_argument("--output",type=Path,default=LANDSAT/"collection2_b8_native")
    args = parser.parse_args()
    requests = read_csv("landsat_c2_b8_acquisition_queue")
    products = {r["product_id"]:normalized(r) for stem in (GEOMETRY,CONTROLS) for r in read_csv(stem+"_products")}
    years = sorted({int(r["captured_at_utc"][:4]) for r in requests})
    ids = {r["product_id"] for r in requests}
    scenes = {r["id"]:r for year in years for r in catalog(year,"usgs") if r["id"] in ids}
    roofs = defaultdict(list)
    for product,*roof in footprints():
        roofs[product].append(roof)
    if ids!=products.keys() or ids!=scenes.keys() or ids!=roofs.keys() or len(ids)!=len(requests):
        raise ValueError("B8 queue, angle products, exact STAC products and roof windows differ")
    _,area = source_coverage()
    bounds = area.bounds
    jobs = [(r,products[r["product_id"]],scenes[r["product_id"]],roofs[r["product_id"]]) for r in requests]
    with ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(lambda job:stage(job,args.originals,args.output,bounds),jobs))
    evidence = [e for e,_ in items]
    windows = [r for _,group in items for r in group]
    weeks = weekly_qa(evidence,windows)
    export(f"{STEM}_products",evidence)
    export(f"{STEM}_roof_windows",windows)
    export(f"{STEM}_weeks",weeks)
    report = dict(scenes=len(evidence), status=dict(Counter(e["b8_status"] for e in evidence)),
                  exact_original_angle_files=len(evidence),exact_original_qa_files=2*len(evidence),
                  downloaded_qa_bytes=sum(e[f"{k}_bytes"] for e in evidence for k in QA_LABELS),
                  planned_native_roof_windows=len(windows),
                  roof_windows_with_all_roof_center_cells_clear=sum(map(clear_roof,windows)),
                  gap_weeks_with_native_qa_pair_candidates=sum(r["clear_any_b8_pair_candidates"]>0 for r in weeks),
                  tank_week_pair_candidates_with_native_qa=sum(r["clear_any_b8_pair_candidates"] for r in weeks),
                  scenes_with_b8_saturation_flag=sum(e["b8_saturation_flag_in_scene_mtl"]=="Y" for e in evidence),
                  interpretation="Original SHA-512 matches use the USGS Level-1 STAC. Identical QA and angle bytes "
                                 "are sourced from the corresponding Level-2 archive. These dates describe that archive. "
                                 "A clear QA window is not evidence of B8 pixel access or a resolved roof measurement.")
    output=json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",output.encode())
    print(output,end="",flush=True)


if __name__=="__main__":
    main()
