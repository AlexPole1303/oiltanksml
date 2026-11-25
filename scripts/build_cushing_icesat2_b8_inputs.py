"""Prepare native Landsat B8 windows and compare verified original pixels with ICESat-2."""
import argparse
import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from affine import Affine
from pyproj import Transformer

from audit_shadow_precision import describe
from build_cushing_icesat2_optical_controls import profiles, separation
from build_cross_sensor_shadow import published_lsf
from build_landsat_c2_holdout_gap_inventory import parse
from build_landsat_c2_b8_measurements import ANGLES, fit_scene, prepare_native, verified_file
from build_shadow_change_controls import RESOLVED, scale
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_landsat import LANDSAT
from collect_landsat_c2_gap_geometry import native_b8_grid
from collect_public import ROOT, write, write_csv
from stage_landsat_c2_b8_inputs import expected_sha512, normalized, sha256_file, stage_pan

STEM = "cushing_icesat2_b8"
OPTICAL = "cushing_icesat2_optical_landsat_c2"
NATIVE = DERIVED/f"{STEM}_native.sqlite"
MARGIN_M = 30
FIT_FIELDS = ("product_id","candidate_id","terminal_id","landsat_capture_at_utc","icesat_granule","icesat_cmr_id",
    "icesat_granule_start_utc","icesat_granule_end_utc","maximum_capture_separation_hours","icesat_photon_csv",
    "diameter_m","liquid_volume_equivalent_bbl_per_m","b8_input_status","b8_fit_status","b8_gap_m",
    "b8_gap_min_m","b8_gap_max_m","icesat_gap_m","b8_minus_icesat_m","b8_minus_icesat_liquid_volume_equivalent_bbl",
    "original_b8_sha512","b8_crop_sha256","psf_basis")


def source_tables():
    geometry = {r["product_id"]:normalized(r) for r in read_csv(f"{OPTICAL}_geometry")}
    angles = {}
    for r in read_csv(f"{OPTICAL}_angles"):
        key = r["product_id"],r["candidate_id"]
        if key in angles or r["angle_status"]!="resolved" or r["product_id"] not in geometry:
            raise ValueError(f"Unexpected B8 center angles for {key}")
        angles[key] = r
    targets = {}
    for r in read_csv(f"{OPTICAL}_targets"):
        key = r["product_id"],r["candidate_id"]
        if key not in angles:
            continue
        if (key in targets or r["icesat_granule"]!=angles[key]["icesat_granule"]
            or r["native_level2_roof_qa_pass"]!="True"):
            raise ValueError(f"Unexpected B8 target for {key}")
        targets[key] = r
    if not geometry or targets.keys()!=angles.keys() or set(geometry)!={p for p,_ in targets}:
        raise ValueError("B8 scene, target and angle tables disagree")
    return geometry,targets,angles


def b8_window(row,product,angle,circle):
    xyz = tuple(float(row[k]) for k in ("center_east_m","center_north_m","radius_m"))
    expected = (*[float(circle[k]) for k in ("center_east_m","center_north_m")],float(circle["diameter_m"])/2)
    if xyz!=expected:
        raise ValueError(f"Circle differs from the dated LiDAR for {row['candidate_id']}")
    east,north = Transformer.from_crs(32614,product["crs"],always_xy=True).transform(*xyz[:2])
    radius = xyz[2]
    inverse = ~native_b8_grid(product)
    col,line = inverse*(east,north)
    center = int(angle["b8_native_line"]),int(angle["b8_native_sample"])
    if (math.floor(line),math.floor(col))!=center:
        raise ValueError(f"Native center differs for {row['product_id']}/{row['candidate_id']}")
    extent = radius+MARGIN_M
    left,top = inverse*(east-extent,north+extent)
    right,bottom = inverse*(east+extent,north-extent)
    return dict(candidate_id=row["candidate_id"],center_east_m=east,center_north_m=north,radius_m=radius,
        center_native_line=center[0],center_native_sample=center[1],
        b8_first_line=math.floor(top),b8_last_line=math.ceil(bottom)-1,
        b8_first_sample=math.floor(left),b8_last_sample=math.ceil(right)-1)


def scene_stage(product,targets,windows,originals,output):
    sample = targets[0]
    identity = product["product_id"]
    catalog = ROOT/sample["landsatlook_catalog"]
    if sha256_file(catalog)!=sample["landsatlook_catalog_sha256"]:
        raise ValueError(f"LandsatLook snapshot digest differs for {identity}")
    data = json.loads(catalog.read_bytes())
    item, = [r for r in data["features"] if r["id"]==identity]
    assets = item["assets"]
    asset = assets["pan"]
    expected = expected_sha512(asset)
    if (tuple(asset["proj:shape"])!=(product["native_lines"],product["native_samples"])
        or Affine(*asset["proj:transform"][:6])!=native_b8_grid(product)
        or any((r["landsatlook_catalog"],r["b8_asset"],r["b8_sha512"])!=(sample["landsatlook_catalog"],asset["href"],expected)
               for r in targets)):
        raise ValueError(f"B8 catalog grid or source differs for {identity}")
    angle_path = ROOT/product["angle_local_file"]
    angle_digest = expected_sha512(assets["ANG.txt"])
    verified_file(angle_path,angle_digest)
    row = dict(product_id=identity,platform=product["platform"],captured_at_utc=product["captured_at_utc"],
        purpose="icesat2_laser_reference",source_angle_sha512=angle_digest,local_angle_file=product["angle_local_file"],
        expected_original_b8_sha512=expected,original_b8_filename=asset["href"].rsplit("/",1)[-1],
        b8_download_url=asset["href"],landsatlook_catalog=sample["landsatlook_catalog"],
        level2_metadata_path=product["level2_metadata_path"])
    for label in ("qa_pixel","qa_radsat"):
        digest = expected_sha512(assets[label])
        if product[f"{label}_sha512"]!=digest:
            raise ValueError(f"Source {label} digest differs for {identity}")
        row.update({f"{label}_sha512":digest,f"{label}_path":product[f"{label}_file"]})
    metadata = json.loads((ROOT/product["level2_metadata_path"]).read_bytes())["LANDSAT_METADATA_FILE"]
    saturation = metadata["IMAGE_ATTRIBUTES"]["SATURATION_BAND_8"]
    if metadata["LEVEL1_PROCESSING_RECORD"]["LANDSAT_PRODUCT_ID"]!=identity or saturation not in ("Y","N"):
        raise ValueError(f"Invalid radiometric metadata for {identity}")
    row["b8_saturation_flag_in_scene_mtl"] = saturation
    original = originals/row["original_b8_filename"]
    if original.is_file():
        grid = native_b8_grid(product)
        projection = Transformer.from_crs(product["crs"],4326,always_xy=True)
        left = min(r["b8_first_sample"] for r in windows)
        top = min(r["b8_first_line"] for r in windows)
        right = max(r["b8_last_sample"] for r in windows)+1
        bottom = max(r["b8_last_line"] for r in windows)+1
        west,north = grid*(left,top)
        east,south = grid*(right,bottom)
        bounds = projection.transform_bounds(west,south,east,north,densify_pts=21)
        crop = output/identity/"pan.tif"
        digest,line,sample = stage_pan(product,original,crop,asset,bounds)
        row.update(b8_status="verified_native_crop",original_b8_sha512=digest,b8_crop_path=str(crop.resolve()),
            b8_crop_sha256=sha256_file(crop),b8_crop_first_line=line,b8_crop_first_sample=sample)
    else:
        row.update(b8_status="awaiting_original_b8",original_b8_sha512=None,b8_crop_path=None,
                   b8_crop_sha256=None,b8_crop_first_line=None,b8_crop_first_sample=None)
    return row


def observations(targets,staged,geometry):
    laser = profiles()
    fits = {}
    products = {p for p,r in staged.items() if r["b8_status"]=="verified_native_crop"}
    if products:
        lsf = published_lsf()
        for p in sorted(products):
            for fit in fit_scene(staged[p],geometry[p],lsf,database=NATIVE):
                key = p,fit["candidate_id"]
                if key in fits:
                    raise ValueError(f"Repeated B8 fit for {key}")
                fits[key] = fit
    if fits.keys()!={key for key in targets if key[0] in products}:
        raise ValueError("Verified B8 fits differ from the prepared laser comparison targets")
    for (product,tank),target in targets.items():
        stage = staged[product]
        source = laser[target["icesat_granule"]][tank]
        capture = parse(stage["captured_at_utc"])
        hours = separation(source,capture)
        if (source["cmr_id"]!=target["icesat_cmr_id"] or source["terminal_id"]!=target["terminal_id"]
            or capture!=parse(target["landsat_capture_at_utc"])
            or not math.isclose(hours,float(target["max_separation_hours"]),rel_tol=0,abs_tol=1e-9)
            or any(float(source[f"geometry_{k}"])!=float(target[k])
                   for k in ("center_east_m","center_north_m","radius_m"))):
            raise ValueError(f"Laser source identity differs for {product}/{tank}")
        ref = float(source["estimated_wall_to_roof_m"])
        diameter = 2*float(target["radius_m"])
        bbl_per_m = scale({"diameter_m":diameter})
        fit = fits.get((product,tank),{})
        if fit and (fit["terminal_id"]!=source["terminal_id"] or fit["roof_type"]!="EFR"
                    or fit["diameter_m"]!=diameter):
            raise ValueError(f"Optical fit uses different roof geometry for {product}/{tank}")
        status = fit.get("fit_status",stage["b8_status"])
        gap = fit.get("estimated_gap_m")
        difference = gap-ref if status==RESOLVED else None
        yield dict(product_id=product,candidate_id=tank,terminal_id=source["terminal_id"],
            landsat_capture_at_utc=capture.isoformat(),icesat_granule=source["filename"],icesat_cmr_id=source["cmr_id"],
            icesat_granule_start_utc=source["granule_start_utc"],icesat_granule_end_utc=source["granule_end_utc"],
            maximum_capture_separation_hours=hours,icesat_photon_csv=source["photon_csv"],diameter_m=diameter,
            liquid_volume_equivalent_bbl_per_m=bbl_per_m,
            b8_input_status=stage["b8_status"],b8_fit_status=status,b8_gap_m=gap,
            b8_gap_min_m=fit.get("near_optimal_gap_min_m"),b8_gap_max_m=fit.get("near_optimal_gap_max_m"),
            icesat_gap_m=ref,b8_minus_icesat_m=difference,
            b8_minus_icesat_liquid_volume_equivalent_bbl=-difference*bbl_per_m if difference is not None else None,
            original_b8_sha512=stage["original_b8_sha512"],b8_crop_sha256=stage["b8_crop_sha256"],
            psf_basis=fit.get("psf_basis"))


def export(kind,rows,fields=None):
    columns = fields or tuple(rows[0])
    write_csv(DERIVED/f"{STEM}_{kind}.csv",columns,([r[k] for k in columns] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals",type=Path,default=LANDSAT/"collection2_b8_original")
    parser.add_argument("--output",type=Path,default=LANDSAT/"collection2_b8_icesat2_native")
    args = parser.parse_args()
    geometry,targets,angles = source_tables()
    circles = legacy_geometry()
    windows,matched = defaultdict(list),defaultdict(list)
    center_angles = {}
    for key,row in targets.items():
        product,tank = key
        angle = angles[key]
        window = b8_window(row,geometry[product],angle,circles[tank])
        windows[product].append(window)
        matched[product].append(row)
        center_angles[key] = (window["center_native_line"],window["center_native_sample"],*(float(angle[k]) for k in ANGLES))
    staged = {p:scene_stage(r,matched[p],windows[p],args.originals,args.output) for p,r in geometry.items()}
    stats = prepare_native(staged,geometry,windows,center_angles,database=NATIVE)
    with closing(read_only(NATIVE)) as db:
        quality = [dict(product_id=p,candidate_id=t,touched_cells=n,cells_passing_qa_and_angle=ok,
                        full_roof_qa_pass=0<n==ok) for p,t,n,ok in db.execute(
            "SELECT product_id,candidate_id,roof_touching_b8_cells,roof_touching_b8_cells_passing_qa FROM roof_window")]
    results = list(observations(targets,staged,geometry))
    export("products",list(staged.values()))
    export("quality",quality)
    export("laser_comparisons",results,FIT_FIELDS)
    report = dict(stats,original_b8_status=dict(Counter(r["b8_status"] for r in staged.values())),
        b8_fit_status=dict(Counter(r["b8_fit_status"] for r in results)),
        full_roof_qa_count_by_scene={p:sum(r["full_roof_qa_pass"] and r["product_id"]==p for r in quality) for p in staged},
        laser_comparison_disagreement=describe([r["b8_minus_icesat_m"] for r in results if r["b8_minus_icesat_m"] is not None]),
        native_database=str(NATIVE.relative_to(ROOT)),native_database_sha256=sha256_file(NATIVE),
        scope="The native database contains quality and angles but no B8 signal. Measurements require an exact source B8 checksum. "
              "Comparisons are retrospective, with a possible actual roof movement between sensors.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
