"""Prepare native Landsat 8/9 B8 windows and measure roofs from verified source pixels."""
import argparse
import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import Window

from build_cross_sensor_shadow import PAN_PIXEL, PSF_SHA256, fit_pan_samples, pan_psf, pan_qa, published_lsf
from build_landsat_c2_external_controls import STEM as CONTROLS
from build_landsat_c2_holdout_gap_inventory import STEM as GAP, parse
from build_landsat_pan_roof_angles import angle_values, prepare_tool
from build_sentinel2_angles import ground_coordinates
from build_shadow_change_controls import RESOLVED, scale
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_landsat_c2_gap_geometry import STEM as GEOMETRY, native_b8_grid
from collect_public import ROOT, write, write_csv
from fit_shadow_geometry import FIT_FIELDS
from stage_landsat_c2_b8_inputs import QA_LABELS, normalized, raster_grid, scene_qa, sha256_file, weekly_qa

STEM = "landsat_c2_b8_measurements"
NATIVE = DERIVED / "landsat_c2_b8_native_geometry.sqlite"
ANGLES = ("solar_azimuth_deg", "solar_zenith_deg", "view_azimuth_deg", "view_zenith_deg")
FIT_OUTPUT = ("product_id", "platform", "terminal_id", "geometry_basis", "psf_basis", "b8_crop_path",
              "b8_crop_sha256", "historical_b8_archive_availability_verified", *FIT_FIELDS,
              "max_roof_view_vector_delta", "max_roof_sun_vector_delta")
SCHEMA = """
CREATE TABLE scene (
    product_id TEXT PRIMARY KEY, platform TEXT NOT NULL, captured_at_utc TEXT NOT NULL,
    crs TEXT NOT NULL, original_b8_sha512 TEXT, angle_sha512 TEXT NOT NULL,
    qa_pixel_sha512 TEXT NOT NULL, qa_radsat_sha512 TEXT NOT NULL, sun_view_path TEXT NOT NULL,
    reflectance_mult REAL NOT NULL, reflectance_add REAL NOT NULL, track_bearing_deg REAL NOT NULL,
    psf_basis TEXT NOT NULL, psf_source_sha256 TEXT NOT NULL);
CREATE TABLE roof_window (
    pan_observation_id INTEGER PRIMARY KEY, product_id TEXT NOT NULL REFERENCES scene,
    candidate_id TEXT NOT NULL, terminal_id TEXT, roof_type TEXT NOT NULL, geometry_basis TEXT NOT NULL,
    center_east_m REAL NOT NULL, center_north_m REAL NOT NULL, radius_m REAL NOT NULL,
    shell_above_ground_m REAL NOT NULL, center_native_line INTEGER NOT NULL, center_native_sample INTEGER NOT NULL,
    true_north_grid_bearing_deg REAL NOT NULL, solar_azimuth_true_deg REAL NOT NULL, solar_zenith_deg REAL NOT NULL,
    view_azimuth_true_deg REAL NOT NULL, view_zenith_deg REAL NOT NULL, first_native_line INTEGER NOT NULL,
    last_native_line INTEGER NOT NULL, first_native_sample INTEGER NOT NULL, last_native_sample INTEGER NOT NULL,
    roof_touching_b8_cells INTEGER NOT NULL, roof_touching_b8_cells_passing_qa INTEGER NOT NULL,
    UNIQUE(product_id,candidate_id));
CREATE TABLE native_pixel (
    product_id TEXT NOT NULL REFERENCES scene, native_line INTEGER NOT NULL, native_sample INTEGER NOT NULL,
    center_east_m REAL NOT NULL, center_north_m REAL NOT NULL, touching_qa_valid INTEGER NOT NULL,
    touching_qa_flag_bits INTEGER NOT NULL, angle_status TEXT NOT NULL, solar_azimuth_deg REAL, solar_zenith_deg REAL,
    view_azimuth_deg REAL, view_zenith_deg REAL, PRIMARY KEY(product_id,native_line,native_sample)) WITHOUT ROWID;
"""


def verified_file(path, digest):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha512").hexdigest() != digest:
            raise ValueError(f"Original Level-1 digest mismatch for {path.name}")
    return path


def source_tables():
    staged = {r["product_id"]:r for r in read_csv("landsat_c2_b8_stage_products")}
    products = {r["product_id"]:normalized(r) for stem in (GEOMETRY, CONTROLS) for r in read_csv(stem+"_products")}
    windows = defaultdict(list)
    for r in read_csv("landsat_c2_b8_stage_roof_windows"):
        windows[r["product_id"]].append(r)
    angles = {}
    for stem in (GEOMETRY+"_roof_angles", CONTROLS+"_roof_pairs"):
        for r in read_csv(stem):
            key = r["product_id"],r["candidate_id"]
            a = (int(r["b8_native_line"]),int(r["b8_native_sample"]),*(float(r[k]) for k in ANGLES))
            if angles.setdefault(key,a) != a:
                raise ValueError(f"Different center angles for {key}")
    if staged.keys() != products.keys() or staged.keys() != windows.keys():
        raise ValueError("B8 scene, QA, and native window products differ")
    if sum(map(len, windows.values())) != len(angles):
        raise ValueError("The staged roof windows differ from the saved angle centers")
    return staged, products, windows, angles


def engineering_inputs():
    with closing(read_only(SOURCE)) as db:
        registry = {tank:(terminal,roof,shell) for tank,terminal,roof,shell in db.execute("""
            SELECT candidate_id,documented_terminal_id,documented_roof_type,lidar_rim_to_ground_native_z
            FROM tank_engineering_registry WHERE documented_roof_type IS NOT NULL""")}
    return registry, legacy_geometry()


def roof_touched(x,y,east,north,radius):
    dx = np.maximum(np.abs(x-east)-PAN_PIXEL/2,0)
    dy = np.maximum(np.abs(y-north)-PAN_PIXEL/2,0)
    return dx**2+dy**2 < radius**2


def prepare_scene(job, tool, registry, dated_geometry, angles, geometry_basis):
    stage, product, windows = job
    identity = product["product_id"]
    angle_path = verified_file(ROOT/stage["local_angle_file"], stage["source_angle_sha512"])
    qa_paths = [verified_file(ROOT/stage[f"{label}_path"], stage[f"{label}_sha512"]) for label in QA_LABELS]
    grid = native_b8_grid(product)
    to_geographic = Transformer.from_crs(product["crs"],4326,always_xy=True)
    to_native = Transformer.from_crs(32614,product["crs"],always_xy=True)
    chips, locations, points, known = [], [], set(), []
    historical = stage["purpose"] not in ("lidar_2022","state_aerial_2025")
    for r in windows:
        tank = r["candidate_id"]
        terminal, roof, shell = registry[tank]
        if historical:
            g = dated_geometry[tank]
            shell = float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"])
        x,y,radius = (float(r[k]) for k in ("center_east_m","center_north_m","radius_m"))
        center = (int(r["center_native_line"]),int(r["center_native_sample"]))
        line0,line1,col0,col1 = (int(r[k]) for k in ("b8_first_line","b8_last_line","b8_first_sample","b8_last_sample"))
        col,line = (~grid)*(x,y)
        expected = angles[identity,tank]
        if ((math.floor(line),math.floor(col)) != center or center != expected[:2]
                or not line0<=center[0]<=line1 or not col0<=center[1]<=col1 or not shell or shell<=0):
            raise ValueError(f"Invalid B8 roof geometry for {identity}/{tank}")
        if historical and (roof != "EFR" or not np.allclose(
                (x,y,radius), (*to_native.transform(float(g["center_east_m"]),float(g["center_north_m"])),
                                float(g["diameter_m"])/2), atol=1e-8, rtol=0)):
            raise ValueError(f"Native B8 geometry differs from the dated outline for {identity}/{tank}")
        points.update((line,col) for line in range(line0,line1+1) for col in range(col0,col1+1))
        locations.append(to_geographic.transform(x,y))
        known.append((tank,x,y,radius,*center))
        chips.append((identity,tank,terminal,roof,geometry_basis if historical else "independent_control_geometry",
                      x,y,radius,shell,*center,line0,line1,col0,col1,expected[2:]))
    solved = angle_values(tool,angle_path,points)
    north = ground_coordinates(product["crs"],tuple(locations))[2]
    with ExitStack() as stack:
        qa = [stack.enter_context(rasterio.open(path)) for path in qa_paths]
        values, valid, flags, transform = scene_qa(product,qa,known)
    ordered = np.asarray(sorted(points))
    x,y = grid*(ordered[:,1]+.5,ordered[:,0]+.5)
    view = SimpleNamespace(transform=transform,height=values.shape[0],width=values.shape[1])
    _,ok,bits = pan_qa(view,x,y,values,valid,flags)
    qa_by_pixel = {tuple(p):bool(usable) for p,usable in zip(ordered,ok&(bits==0),strict=True)}
    roof_rows = []
    for chip, correction in zip(chips,north,strict=True):
        current = solved[chip[9],chip[10]]
        vals = tuple(current[k] for k in ANGLES)
        if current["angle_status"] != "resolved" or vals != chip[-1]:
            raise ValueError(f"Native RPC replay differs from the saved roof-center angles for {chip[:2]}")
        line0,line1,col0,col1 = chip[11:-1]
        lines,cols = np.indices((line1-line0+1,col1-col0+1)).reshape(2,-1)
        lines,cols = lines+line0,cols+col0
        e,n = grid*(cols+.5,lines+.5)
        touched = roof_touched(e,n,*chip[5:8])
        passed = sum(qa_by_pixel[a] and solved[a]["angle_status"]=="resolved" for a in
                     zip(lines[touched],cols[touched],strict=True))
        roof_rows.append((*chip[:11],float(correction),*vals,*chip[11:-1],int(touched.sum()),passed))
    pixel_rows = [(identity,int(line),int(col),float(px),float(py),int(use),int(flag),a["angle_status"],
                   *(a[k] for k in ANGLES))
                  for (line,col),px,py,use,flag in zip(ordered,x,y,ok,bits,strict=True)
                  for a in (solved[line,col],)]
    psf = {"LANDSAT_8":"usgs_landsat8_prelaunch_lsf",
           "LANDSAT_9":"landsat8_lsf_proxy_for_oli2"}[product["platform"]]
    scene_row = (identity,product["platform"],product["captured_at_utc"],product["crs"],stage["expected_original_b8_sha512"],
                 stage["source_angle_sha512"],stage["qa_pixel_sha512"],stage["qa_radsat_sha512"],stage["local_angle_file"],
                 float(product["reflectance_mult_b8"]),float(product["reflectance_add_b8"]),
                 float(product["along_track_grid_bearing_deg"]),psf,PSF_SHA256)
    return scene_row,roof_rows,pixel_rows


def prepare_native(staged,products,windows,angles,*,database=NATIVE, dated_geometry=None, geometry_basis="dated_2011_lidar"):
    tool = prepare_tool()
    registry,legacy = engineering_inputs()
    geometry = legacy if dated_geometry is None else dated_geometry
    temporary = database.with_suffix(".part.sqlite")
    temporary.unlink(missing_ok=True)
    jobs = [(row,products[p],windows[p]) for p,row in staged.items()]
    stats = Counter()
    with closing(sqlite3.connect(temporary)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = pool.map(lambda job:prepare_scene(job,tool,registry,geometry,angles,geometry_basis),jobs)
            for i,(scene,roofs,pixels) in enumerate(results,1):
                db.execute("INSERT INTO scene VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",scene)
                db.executemany("INSERT INTO roof_window VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",roofs)
                db.executemany("INSERT INTO native_pixel VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",pixels)
                stats.update(dict(scenes=1,roof_windows=len(roofs),unique_native_pixels=len(pixels),
                    native_pixels_with_angles=sum(r[7]=="resolved" for r in pixels),
                    native_pixels_passing_qa=sum(r[5] and not r[6] for r in pixels),
                    roof_windows_with_full_footprint_qa=sum(0<r[-2]==r[-1] for r in roofs)))
                if i%12==0 or i==len(jobs):
                    print(f"Prepared exact QA and angle pixels for {i}/{len(jobs)} Landsat scenes",flush=True)
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Native B8 inputs failed integrity check")
    temporary.replace(database)
    return dict(stats)


def crop_window(raster, o, origin):
    first_line,first_col = o["first_native_line"]-origin[0],o["first_native_sample"]-origin[1]
    height = o["last_native_line"]-o["first_native_line"]+1
    width = o["last_native_sample"]-o["first_native_sample"]+1
    if not (0<=first_line<=raster.height-height and 0<=first_col<=raster.width-width):
        raise ValueError(f"Roof window exceeds verified B8 crop for {o['product_id']}/{o['candidate_id']}")
    return Window(first_col,first_line,width,height)


def native_pan_samples(raster, db, o, origin, saturation, *, apparent_reflectance_scale=None):
    win = crop_window(raster,o,origin)
    values = raster.read(1,window=win).ravel()
    valid = raster.read_masks(1,window=win).ravel()>0
    pixels = np.array(db.execute("""SELECT native_line,native_sample,center_east_m,center_north_m,
        solar_zenith_deg,touching_qa_valid,touching_qa_flag_bits,view_azimuth_deg,view_zenith_deg,
        solar_azimuth_deg,solar_zenith_deg FROM native_pixel WHERE product_id=?
        AND native_line BETWEEN ? AND ? AND native_sample BETWEEN ? AND ?
        ORDER BY native_line,native_sample""",(o["product_id"],o["first_native_line"],o["last_native_line"],
                                              o["first_native_sample"],o["last_native_sample"])).fetchall(),dtype=float)
    if len(values) != len(pixels):
        raise ValueError(f"Missing native angle or QA pixels for {o['product_id']}/{o['candidate_id']}")
    if apparent_reflectance_scale is None:
        usable = valid & (values!=0) & (values!=65535) & np.isfinite(pixels[:,4])
        reflectance = (values*o["reflectance_mult"]+o["reflectance_add"])/np.cos(np.deg2rad(pixels[:,4]))
    else:
        reflectance = values*apparent_reflectance_scale
        usable = valid & (reflectance>0) & (reflectance<1) & np.isfinite(pixels[:,4])
    roof = roof_touched(pixels[:,2],pixels[:,3],o["center_east_m"],o["center_north_m"],o["radius_m"])
    pixels[:,4] = np.where(usable,reflectance,np.nan)
    # QA_RADSAT has no B8 flag per pixel. A scene flag leaves local saturation unresolved.
    problem = ("b8_scene_saturation_flag" if saturation=="Y" else
               "b8_roof_native_pixels_or_qa_incomplete" if not (
                    0<o["roof_touching_b8_cells"]==o["roof_touching_b8_cells_passing_qa"] and usable[roof].all()) else None)
    if problem:
        pixels[:,4] = np.nan
    return pixels,problem


def fit_native_crop(raster, db, o, origin, kernel, saturation, *, apparent_reflectance_scale=None):
    pixels,problem = native_pan_samples(raster,db,o,origin,saturation,apparent_reflectance_scale=apparent_reflectance_scale)
    result = fit_pan_samples(o,pixels,kernel)
    if problem:
        result["fit_status"] = problem
    return result


def fit_scene(stage,product,lsf,*,database=NATIVE):
    path = ROOT / stage["b8_crop_path"]
    if (stage["original_b8_sha512"] != stage["expected_original_b8_sha512"]
            or sha256_file(path) != stage["b8_crop_sha256"]):
        raise ValueError(f"B8 crop disagrees with its verified staging record for {product['product_id']}")
    with closing(read_only(database)) as db,rasterio.open(path) as raster:
        db.row_factory = sqlite3.Row
        origin = raster_grid(raster,product,15,native_b8_grid(product))
        if origin != (int(stage["b8_crop_first_line"]),int(stage["b8_crop_first_sample"])):
            raise ValueError(f"B8 crop origin differs from its native stage for {product['product_id']}")
        rows = db.execute("SELECT * FROM roof_window JOIN scene USING(product_id) WHERE product_id=? ORDER BY pan_observation_id",
                          (product["product_id"],))
        kernel = pan_psf(lsf,float(product["along_track_grid_bearing_deg"]))
        for o in rows:
            fit = fit_native_crop(raster,db,o,origin,kernel,stage["b8_saturation_flag_in_scene_mtl"])
            yield dict(fit,product_id=o["product_id"],platform=o["platform"],terminal_id=o["terminal_id"],
                       geometry_basis=o["geometry_basis"],psf_basis=o["psf_basis"],b8_crop_path=stage["b8_crop_path"],
                       b8_crop_sha256=stage["b8_crop_sha256"],historical_b8_archive_availability_verified=False)


def control_disagreements(fits):
    for r in read_csv(CONTROLS+"_roof_pairs"):
        fit = fits.get((r["product_id"],r["candidate_id"]))
        if fit is None:
            continue
        if r["roof_type"] != fit["roof_type"] or not math.isclose(
                2*float(r["radius_m"]),fit["diameter_m"],abs_tol=1e-8):
            raise ValueError("B8 control comparison uses different roof geometry")
        resolved = fit["fit_status"] == RESOLVED and fit["roof_type"] == "EFR"
        delta = fit["estimated_gap_m"]-float(r["reference_gap_m"]) if resolved else None
        yield dict(r,b8_fit_status=fit["fit_status"],b8_gap_m=fit.get("estimated_gap_m"),
                   b8_minus_reference_gap_m=delta,b8_minus_reference_equivalent_bbl=scale(fit)*delta if resolved else None)


def sentinel_controls(staged,fits):
    candidates = read_csv(GAP+"_calibration_controls")
    key_fields = ("candidate_id","sentinel2_version_id","sentinel2_canonical_id")
    keys = {tuple(r[k] for k in key_fields) for r in candidates}
    reference = {}
    with (DERIVED/"shadow_timing_holdout_observations.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["candidate_id"],row["version_id"],row["canonical_l2a_id"]
            if key in keys and row["geometry_mode"]=="dated_2011_lidar":
                if key in reference:
                    raise ValueError(f"Duplicate Sentinel-2 control for {key}")
                reference[key] = row
    for r in candidates:
        key = tuple(r[k] for k in key_fields)
        s = reference[key]
        p = staged[r["landsat_product_id"]]
        if (s["fit_status"] != RESOLVED or s["terminal_id"] != r["terminal_id"]
                or parse(s["captured_at_utc"]) != parse(r["sentinel2_capture_at_utc"])
                or parse(s["geometry_available_at_utc"]) != parse(r["sentinel2_geometry_available_at_utc"])
                or max(parse(s["geometry_available_at_utc"]),parse(p["qa_and_angles_archive_at_utc"])) >= parse(r["forecast_cutoff_utc"])):
            raise ValueError(f"Sentinel-2 control differs from the expected version or clock for {key}")
        a = fits.get((r["landsat_product_id"],r["candidate_id"]))
        if a and (a["terminal_id"] != r["terminal_id"] or a["fit_status"] == RESOLVED and
                  not math.isclose(a["diameter_m"],float(s["diameter_m"]),abs_tol=1e-8)):
            raise ValueError(f"Different B8 and B04 tank geometry for {key}")
        gap = float(s["estimated_gap_m"])
        delta = a["estimated_gap_m"]-gap if a and a["fit_status"]==RESOLVED else None
        yield dict(r,sentinel2_fit_status=s["fit_status"],sentinel2_gap_m=gap,
                   sentinel2_near_gap_min_m=float(s["near_optimal_gap_min_m"]),
                   sentinel2_near_gap_max_m=float(s["near_optimal_gap_max_m"]),
                   b8_status=a["fit_status"] if a else p["b8_status"],b8_gap_m=a.get("estimated_gap_m") if a else None,
                   b8_minus_b04_gap_m=delta,b8_minus_b04_equivalent_bbl=scale(a)*delta if delta is not None else None)


def save_csv(suffix,fields,rows):
    write_csv(DERIVED/f"{STEM}_{suffix}.csv",fields,([r.get(k) for k in fields] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    staged,products,windows,angles = source_tables()
    stats = prepare_native(staged,products,windows,angles)
    with closing(read_only(NATIVE)) as db:
        footprint_qa = [dict(product_id=p,candidate_id=t,b8_roof_center_cells=total,b8_roof_center_cells_pass_qa=passed)
                        for p,t,total,passed in db.execute("""SELECT product_id,candidate_id,roof_touching_b8_cells,
                            roof_touching_b8_cells_passing_qa FROM roof_window""")]
    weeks = weekly_qa(staged.values(),footprint_qa)
    eligible = [p for p,r in staged.items() if r["b8_status"]=="verified_native_crop"]
    fits = {}
    if eligible and not args.prepare_only:
        lsf = published_lsf()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = pool.map(lambda p:list(fit_scene(staged[p],products[p],lsf)),eligible)
            for i,rows in enumerate(results,1):
                fits.update({(r["product_id"],r["candidate_id"]):r for r in rows})
                print(f"Fit B8 roofs for {i}/{len(eligible)} native products",flush=True)
    controls = list(control_disagreements(fits))
    sentinel = list(sentinel_controls(staged,fits))
    control_fields = (*read_csv(CONTROLS+"_roof_pairs")[0],"b8_fit_status","b8_gap_m",
                      "b8_minus_reference_gap_m","b8_minus_reference_equivalent_bbl")
    save_csv("roofs", FIT_OUTPUT, fits.values())
    save_csv("external_controls", control_fields, controls)
    save_csv("sentinel2_controls", tuple(sentinel[0]), sentinel)
    save_csv("footprint_qa_weeks", tuple(weeks[0]), weeks)
    summary = dict(stats,native_database=str(NATIVE.relative_to(ROOT)),native_database_sha256=sha256_file(NATIVE),
                   scenes_with_verified_b8=len(eligible),awaiting_original_b8=len(staged)-len(eligible),
                   gap_weeks_with_full_footprint_qa_candidates=sum(r["clear_any_b8_pair_candidates"]>0 for r in weeks),
                   gap_tank_weeks_with_full_footprint_qa_candidates=sum(r["clear_any_b8_pair_candidates"] for r in weeks),
                   roof_fits=len(fits),roof_fit_status=dict(Counter(r["fit_status"] for r in fits.values())),
                   external_comparisons=len(controls),external_efr_comparisons_with_resolved_b8=sum(
                       r["b8_minus_reference_gap_m"] is not None for r in controls),
                   sentinel2_resolved_reference_observations=len(sentinel),
                   sentinel2_comparisons_with_resolved_b8=sum(r["b8_minus_b04_gap_m"] is not None for r in sentinel),
                   scope="Native angle and QA records include no B8 values until a verified original is staged. "
                         "Single-scene estimates do not establish historical B8 delivery or independently registered weekly changes.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
