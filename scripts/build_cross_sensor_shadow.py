"""Build native Landsat B8 and Sentinel-2 B04 comparisons without EIA or ML."""
import argparse
import hashlib
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds

from build_l1c_quality import candidate_layer
from build_landsat_coverage import bit, native_masks
from build_landsat_pan_roof_angles import angle_values, native_origin, prepare_tool
from build_sentinel2_angles import ground_coordinates
from build_shadow_calibration import SCHEMA, add_psf, add_scene, insert
from build_shadow_change_controls import RESOLVED, SELECT_QUALITY
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, get, write
from fit_shadow_geometry import bearing_vector, estimate, geometry_fit, inputs

DB = DERIVED/"landsat_sentinel2_shadow.sqlite"
PSF_NAME = "OLI_TIRS_Prelaunch_PointSpreadFunctions-web.xlsx"
PSF_URL = f"https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/{PSF_NAME}"
PSF_SHA256 = "bdafde376e3a5b8557daf664e08cd1e664c256b3a7f6ad2a114093e824f0bc33"
PSF_DOC = "https://www.usgs.gov/landsat-missions/spatial-performance-landsat-8-instruments"
PAN_STEP = 3
PAN_PIXEL = 15
PAN_SHIFTS = (-6,0,6)
PAN_SCHEMA = """
CREATE TABLE pan_lsf(distance_m REAL PRIMARY KEY, across_track REAL NOT NULL, along_track REAL NOT NULL);
CREATE TABLE pan_product (
    product_id TEXT PRIMARY KEY,captured_at_utc TEXT,inputs_available_at_utc TEXT,source_directory TEXT,
    mtl_file TEXT,angle_file TEXT,crs TEXT,first_native_line INTEGER,first_native_sample INTEGER,
    ul_pixel_east_m REAL,ul_pixel_north_m REAL,reflectance_mult REAL,reflectance_add REAL,
    along_track_grid_bearing_deg REAL,psf_sha256 TEXT,psf_source_url TEXT);
CREATE TABLE pan_observation (
    pan_observation_id INTEGER PRIMARY KEY,product_id TEXT REFERENCES pan_product,candidate_id TEXT,roof_type TEXT,
    terminal_id TEXT,geometry_basis TEXT,center_east_m REAL,center_north_m REAL,radius_m REAL,shell_above_ground_m REAL,
    center_native_line INTEGER,center_native_sample INTEGER,true_north_grid_bearing_deg REAL,
    solar_azimuth_true_deg REAL,solar_zenith_deg REAL,view_azimuth_true_deg REAL,view_zenith_deg REAL,
    first_native_line INTEGER,last_native_line INTEGER,first_native_sample INTEGER,last_native_sample INTEGER,
    UNIQUE(product_id,candidate_id));
CREATE TABLE pan_pixel (
    pan_observation_id INTEGER REFERENCES pan_observation,native_line INTEGER,native_sample INTEGER,
    center_east_m REAL,center_north_m REAL,b8_dn INTEGER,b8_raster_valid INTEGER,
    bqa_upper_left INTEGER,bqa_upper_right INTEGER,bqa_lower_left INTEGER,bqa_lower_right INTEGER,
    all_touching_bqa_valid INTEGER,touching_bqa_flag_bits INTEGER,angle_status TEXT,
    solar_azimuth_true_deg REAL,solar_zenith_deg REAL,view_azimuth_true_deg REAL,view_zenith_deg REAL,
    PRIMARY KEY(pan_observation_id,native_line,native_sample)) WITHOUT ROWID;
CREATE VIEW pan_pixel_reflectance AS
    SELECT p.*,CASE WHEN p.b8_raster_valid AND p.b8_dn NOT IN (0,65535) AND p.angle_status='resolved'
        THEN (p.b8_dn*s.reflectance_mult+s.reflectance_add)/cos(radians(p.solar_zenith_deg)) END AS b8_toa_reflectance
    FROM pan_pixel p JOIN pan_observation o USING(pan_observation_id) JOIN pan_product s USING(product_id);
CREATE TABLE sensor_match (
    pan_observation_id INTEGER REFERENCES pan_observation,sentinel_observation_id INTEGER REFERENCES observation,
    candidate_id TEXT,roof_type TEXT,terminal_id TEXT,landsat_minus_sentinel2_minutes REAL,
    both_inputs_before_matching_cutoff INTEGER,matching_release_cutoff_at_utc TEXT,
    sentinel2_archive_available_at_utc TEXT,sentinel2_geometry_available_at_utc TEXT,
    PRIMARY KEY(pan_observation_id,sentinel_observation_id));
"""


def published_lsf():
    path = ROOT/"data/raw/landsat/psf"/PSF_NAME
    payload = path.read_bytes() if path.exists() else get(PSF_URL)
    if hashlib.sha256(payload).hexdigest()!=PSF_SHA256:
        raise ValueError("Landsat point spread workbook differs from the verified USGS version")
    if not path.exists():
        write(path,payload)
    namespace = {"x":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as book:
        if book.testzip() is not None:
            raise ValueError("Corrupt Landsat spatial-response workbook")
        strings = ["".join(r.itertext()) for r in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("x:si",namespace)]
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.findall(".//x:row",namespace):
        values = {c.get("r").rstrip("0123456789"):v if c.get("t")!="s" else strings[int(v)] for c in row
                  if (v:=c.findtext("x:v",None,namespace)) is not None}
        rows.append(values)
    if tuple(rows[1][k] for k in ("L","M","Z","AA"))!=("meters","B8","meters","B8"):
        raise ValueError("The USGS workbook column layout changed")
    values = np.array([[float(r[k]) for k in ("L","M","Z","AA")] for r in rows[2:] if "M" in r and "AA" in r])
    if (values.shape!=(81,4) or not np.array_equal(values[:,0],values[:,2]) or not np.allclose(np.diff(values[:,0]),1.41)
            or not np.allclose(values[:,[1,3]].sum(axis=0),1) or (values[:,[1,3]]<0).any()):
        raise ValueError("Unexpected sampling or coefficient values in the USGS B8 profiles")
    return values[:,[0,1,3]]


def along_track_bearing(angle_file, crs):
    source = angle_file.read_text(encoding="ascii")
    samples = [np.fromstring(re.search(rf"EPHEMERIS_ECEF_{axis}\s*=\s*\(([^)]+)\)",source)[1],sep=",") for axis in "XYZ"]
    n = len(samples[0])//2
    project = Transformer.from_crs(4978,crs,always_xy=True)
    east,north,_ = project.transform(*(p[n-1:n+2:2] for p in samples))
    bearing = math.degrees(math.atan2(east[1]-east[0],north[1]-north[0]))%360
    if not 170<bearing<215:
        raise ValueError(f"Unexpected Landsat descending track bearing for {angle_file}: {bearing}")
    return bearing


def pan_psf(lsf,bearing):
    distance,across,along = lsf.T
    size = math.ceil(math.hypot(distance[-1],distance[-1])/PAN_STEP)
    grid = np.arange(-size,size+1)*PAN_STEP
    east,north = np.meshgrid(grid,-grid)
    angle = math.radians(bearing)
    x = east*math.cos(angle)-north*math.sin(angle)
    y = east*math.sin(angle)+north*math.cos(angle)
    # The workbook gives the axes. Symmetrize each profile before mapping it to the UTM grid.
    across = (across+across[::-1])/2
    along = (along+along[::-1])/2
    kernel = np.interp(x,distance,across,left=0,right=0)*np.interp(y,distance,along,left=0,right=0)
    return (kernel/kernel.sum()).astype("float32")


def sentinel_source(source,matches):
    wanted = defaultdict(set)
    for match in matches:
        wanted[match["sentinel2_product_id"],match["sentinel2_source_directory"]].add(match["candidate_id"])
    groups = {}
    for (version,directory),candidates in sorted(wanted.items()):
        rows = [r for r in source.execute(SELECT_QUALITY+"AND q.l1c_product_name=? AND q.source_directory=? ORDER BY q.candidate_id",
                                         (version,directory)) if r["candidate_id"] in candidates]
        if len(rows)!=len(candidates) or {r["candidate_id"] for r in rows}!=candidates:
            raise ValueError(f"Sentinel-2 QA selection disagrees with the cross-sensor manifest for {version}")
        groups["L1C",version,directory] = rows
    return groups


def pan_qa(qa, x, y, dn, valid, bad_flags):
    values,checks,flags = [],[],[]
    for dy,dx in ((1,-1),(1,1),(-1,-1),(-1,1)):
        u,v = (~qa.transform)*(x+dx*7.499999,y+dy*7.499999)
        r,c = np.floor(v).astype(int),np.floor(u).astype(int)
        inside = (0<=r)&(r<qa.height)&(0<=c)&(c<qa.width)
        index = np.clip(r,0,qa.height-1),np.clip(c,0,qa.width-1)
        values.append(np.where(inside,dn[index].astype("int32"),-1))
        checks.append(inside&valid[index])
        flags.append(np.where(inside,bad_flags[index],0))
    return values,np.logical_and.reduce(checks),np.bitwise_or.reduce(flags)


def chip_indices(pan,x,y,radius):
    extent = radius+30
    win = from_bounds(x-extent,y-extent,x+extent,y+extent,pan.transform)
    r0,c0 = max(0,math.floor(win.row_off)),max(0,math.floor(win.col_off))
    r1,c1 = min(pan.height,math.ceil(win.row_off+win.height)),min(pan.width,math.ceil(win.col_off+win.width))
    rows,cols = np.indices((r1-r0,c1-c0))
    return rows.ravel()+r0,cols.ravel()+c0


def add_pan_product(db,source,tool,product,matches,geometry,shells,identities):
    example = matches[0]
    directory,angle_file = ROOT/example["landsat_source_directory"],ROOT/example["landsat_angle_file"]
    multiplier,additive = source.execute("SELECT multiplier,additive FROM landsat_reflectance_coefficients WHERE product_id=? AND band='B8'",
                                         (product,)).fetchone()
    x0,y0 = native_origin(directory,angle_file)
    with rasterio.open(directory/"pan.tif") as pan,rasterio.open(directory/"bqa.tif") as qa:
        if pan.crs!=qa.crs or pan.res!=(15,15) or qa.res!=(30,30) or pan.transform.b or pan.transform.d or qa.transform.b or qa.transform.d:
            raise ValueError(f"Unexpected native B8 or BQA grid for {product}")
        x,y = pan.xy(0,0)
        row0,col0 = round((y0-y)/15),round((x-x0)/15)
        if not (math.isclose(y0-row0*15,y,abs_tol=1e-6,rel_tol=0) and math.isclose(x0+col0*15,x,abs_tol=1e-6,rel_tol=0)):
            raise ValueError(f"B8 local crop is off its original grid for {product}")
        bearing = along_track_bearing(angle_file,pan.crs.to_string())
        insert(db,"pan_product",[(product,example["landsat_capture_at_utc"],example["landsat_inputs_available_at_utc"],
              example["landsat_source_directory"],str((directory/"MTL.txt").relative_to(ROOT)),example["landsat_angle_file"],
              pan.crs.to_string(),row0,col0,x0,y0,multiplier,additive,bearing,PSF_SHA256,PSF_URL)])
        pan_dn,pan_mask = pan.read(1),pan.read_masks(1)>0
        qa_dn = qa.read(1)
        qa_valid = (qa.read_masks(1)>0)&~bit(qa_dn,0)
        qa_flags = np.zeros(qa_dn.shape,dtype="uint8")
        for i,flag in enumerate(native_masks(qa_dn,None,"pan")):
            qa_flags |= (qa_valid&flag).astype("uint8")<<i
        project = Transformer.from_crs(32614,pan.crs,always_xy=True).transform
        coordinates = tuple((geometry[r["candidate_id"]]["longitude"],geometry[r["candidate_id"]]["latitude"]) for r in matches)
        north_bearings = ground_coordinates(pan.crs.to_string(),coordinates)[2]
        chips = []
        points = set()
        for match,north in zip(matches,north_bearings,strict=True):
            candidate = match["candidate_id"]
            row = geometry[candidate]
            center = project(row["center_east_m"],row["center_north_m"])
            radius = row["radius_m"]
            r,c = chip_indices(pan,*center,radius)
            line,col = pan.index(*center)
            cp = row0+line,col0+col
            if cp!=(int(match["landsat_b8_native_line"]),int(match["landsat_b8_native_sample"])):
                raise ValueError(f"Landsat roof-center pixel disagrees with the matched manifest for {product} {candidate}")
            points.update(zip(r+row0,c+col0,strict=True))
            points.add(cp)
            chips.append((match,center,radius,float(north),cp,r,c))
        angles = angle_values(tool,angle_file,points)
        for match,center,radius,north,cp,r,c in chips:
            candidate = match["candidate_id"]
            if (product,candidate) in identities:
                raise ValueError(f"Duplicate B8 match for {product} {candidate}")
            identity = len(identities)+1
            identities[product,candidate] = identity
            center_angle = angles[cp]
            if center_angle["angle_status"]!="resolved" or any(abs(center_angle[field]-float(match[f"landsat_b8_{field}"]))>1e-9
                    for field in ("solar_azimuth_deg","solar_zenith_deg","view_azimuth_deg","view_zenith_deg")):
                raise ValueError(f"Landsat native angles differ from the matched manifest for {product} {candidate}")
            insert(db,"pan_observation",[(identity,product,candidate,match["roof_type"],match["terminal_id"],geometry[candidate]["geometry_basis"],
                   *center,radius,shells[candidate],*cp,north,*angle_tuple(center_angle),
                   row0+int(r.min()),row0+int(r.max()),col0+int(c.min()),col0+int(c.max()))])
            east,northing = pan.transform*(c+0.5,r+0.5)
            qa_values,clear_flags,flags = pan_qa(qa,east,northing,qa_dn,qa_valid,qa_flags)
            insert(db,"pan_pixel",((identity,int(line),int(sample),float(px),float(py),int(b8),int(valid),
                    *(int(v) if v>=0 else None for v in q),int(usable),int(flag),a["angle_status"],*angle_tuple(a))
                    for line,sample,px,py,b8,valid,q,usable,flag in zip(r+row0,c+col0,east,northing,pan_dn[r,c],pan_mask[r,c],
                    zip(*qa_values,strict=True),clear_flags,flags,strict=True) for a in (angles[line,sample],)))


def angle_tuple(row):
    return tuple(row[name] for name in ("solar_azimuth_deg","solar_zenith_deg","view_azimuth_deg","view_zenith_deg"))


def build_pixels(source,matches,shells):
    s2_groups = sentinel_source(source,matches)
    p_groups = defaultdict(list)
    for row in matches:
        p_groups[row["landsat_product_id"]].append(row)
    geometry = {r["candidate_id"]:r for rows in s2_groups.values() for r in rows}
    shapes = dict((*candidate_layer("osm"),*candidate_layer("supplemental")))
    tool = prepare_tool()
    temporary = DB.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    with closing(sqlite3.connect(temporary)) as db,db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA+PAN_SCHEMA)
        add_psf(db)
        insert(db,"pan_lsf",(tuple(map(float,r)) for r in published_lsf()))
        sentinel_ids,pan_ids = {},{}
        for i,(key,rows) in enumerate(s2_groups.items(),1):
            add_scene(db,source,i,key,rows,shapes,sentinel_ids)
        print(f"Stored native Sentinel-2 pixels from {len(s2_groups)} exact products",flush=True)
        for i,(product,rows) in enumerate(sorted(p_groups.items()),1):
            add_pan_product(db,source,tool,product,rows,geometry,shells,pan_ids)
            if i%4==0 or i==len(p_groups):
                print(f"Stored native Landsat pixels from {i}/{len(p_groups)} products",flush=True)
        insert(db,"sensor_match",((pan_ids[r["landsat_product_id"],r["candidate_id"]],
               sentinel_ids[("L1C",r["sentinel2_product_id"],r["sentinel2_source_directory"]),r["candidate_id"]],
               r["candidate_id"],r["roof_type"],r["terminal_id"],float(r["landsat_minus_sentinel2_minutes"]),
               r["both_inputs_before_matching_cutoff"]=="True",r["matching_release_cutoff_at_utc"],
               r["sentinel2_archive_available_at_utc"],r["sentinel2_geometry_available_at_utc"]) for r in matches))
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("The cross-sensor archive failed integrity validation")
    temporary.replace(DB)


def pan_fit(o,kernel):
    with closing(read_only(DB)) as db:
        rows = db.execute("""SELECT native_line,native_sample,center_east_m,center_north_m,b8_toa_reflectance,
             all_touching_bqa_valid,touching_bqa_flag_bits,view_azimuth_true_deg,view_zenith_deg,solar_azimuth_true_deg,solar_zenith_deg
             FROM pan_pixel_reflectance WHERE pan_observation_id=? ORDER BY native_line,native_sample""",(o["pan_observation_id"],)).fetchall()
    return fit_pan_samples(o, np.asarray(rows, dtype=float), kernel)


def pan_grid(o, m):
    height,width = o["last_native_line"]-o["first_native_line"]+1,o["last_native_sample"]-o["first_native_sample"]+1
    rr,cc = np.indices((height,width)).reshape(2,-1)
    if not np.array_equal(m[:,0],rr+o["first_native_line"]) or not np.array_equal(m[:,1],cc+o["first_native_sample"]):
        raise ValueError(f"Nonrectangular Landsat B8 chip {o['pan_observation_id']}")
    east,north = m[:,2]-o["center_east_m"],m[:,3]-o["center_north_m"]
    radius = o["radius_m"]
    distance = np.hypot(east,north)
    good = np.isfinite(m[:,4])&(m[:,5]==1)&(m[:,6]==0)&(distance<=radius+16)
    weights = np.exp(-0.5*(np.maximum(distance[good]-radius,0)/7)**2)
    return (height,width,(east[0],north[0]),rr[good],cc[good],m[good,4],weights),good


def fit_pan_samples(o, m, kernel):
    grid,good = pan_grid(o,m)
    radius = o["radius_m"]
    count = int(good.sum())
    result = dict(observation_id=o["pan_observation_id"],candidate_id=o["candidate_id"],roof_type=o["roof_type"],
        version_id=o["product_id"],optical_capture_at_utc=o["captured_at_utc"],diameter_m=2*radius,
        sun_elevation_deg=90-o["solar_zenith_deg"],view_zenith_deg=o["view_zenith_deg"],
        shell_above_ground_m=o["shell_above_ground_m"],fitted_pixels=count,fit_status="insufficient_valid_pixels",
        max_roof_view_vector_delta=None,max_roof_sun_vector_delta=None)
    if count<12:
        return result
    north_grid = o["true_north_grid_bearing_deg"]
    sun = bearing_vector(o["solar_zenith_deg"],o["solar_azimuth_true_deg"]+north_grid)
    view = bearing_vector(o["view_zenith_deg"],o["view_azimuth_true_deg"]+north_grid)
    for name,start,reference in (("view",7,view),("sun",9,sun)):
        angles = np.deg2rad(m[good,start]+north_grid)
        vectors = np.tan(np.deg2rad(m[good,start+1]))[:,None]*np.column_stack((np.sin(angles),np.cos(angles)))
        result[f"max_roof_{name}_vector_delta"] = float(np.max(np.linalg.norm(vectors-reference,axis=1)))
    return dict(result,**geometry_fit(grid,radius,o["shell_above_ground_m"],o["roof_type"],sun,view,kernel,
                                      pixel_m=PAN_PIXEL,step_m=PAN_STEP,shifts=PAN_SHIFTS))


def fit_archive(shells):
    with closing(read_only(DB)) as db:
        db.row_factory = sqlite3.Row
        s2,kernels = inputs(db)
        pan = db.execute("""SELECT o.*,s.captured_at_utc FROM pan_observation o JOIN pan_product s USING(product_id)
                            ORDER BY pan_observation_id""").fetchall()
        lsf = np.array([tuple(r) for r in db.execute("SELECT * FROM pan_lsf ORDER BY distance_m")])
        pan_kernels = {product:pan_psf(lsf,bearing) for product,bearing in db.execute("SELECT product_id,along_track_grid_bearing_deg FROM pan_product")}
    def fit_s2(row):
        return estimate(row,shells[row["candidate_id"]],kernels[row["platform"]],calibration_db=DB)
    def fit_pan(row):
        return pan_fit(row,pan_kernels[row["product_id"]])
    results = []
    with ThreadPoolExecutor(max_workers=4) as workers:
        for sensor,func,observations in (("sentinel2_b04",fit_s2,s2),("landsat8_b8",fit_pan,pan)):
            found = {}
            for i,result in enumerate(workers.map(func,observations),1):
                found[result["observation_id"]] = result
                if i%300==0 or i==len(observations):
                    print(f"Fit {i}/{len(observations)} {sensor} observations",flush=True)
            results.append(found)
    return results


def compare_rows(sentinel,pan):
    fields = ("observation_id","version_id","optical_capture_at_utc","sun_elevation_deg","view_zenith_deg","fitted_pixels","fit_status",
        "estimated_gap_m","near_optimal_gap_min_m","near_optimal_gap_max_m","shift_east_m","shift_north_m","pixel_rmse",
        "flat_rmse","reversed_sun_rmse","box_response_gap_m","max_roof_view_vector_delta","max_roof_sun_vector_delta")
    with closing(read_only(DB)) as db:
        db.row_factory = sqlite3.Row
        for match in db.execute("SELECT * FROM sensor_match ORDER BY candidate_id,pan_observation_id"):
            a,b = pan[match["pan_observation_id"]],sentinel[match["sentinel_observation_id"]]
            left,right = a.get("estimated_gap_m"),b.get("estimated_gap_m")
            difference = left-right if left is not None and right is not None else None
            yield dict(match,diameter_m=a["diameter_m"],shell_above_ground_m=a["shell_above_ground_m"],
                **{f"{sensor}_{k}":r.get(k) for sensor,r in (("landsat",a),("sentinel2",b)) for k in fields},
                both_geometry_diagnostics_pass=a["fit_status"]==b["fit_status"]==RESOLVED,
                landsat_minus_sentinel2_gap_m=difference,absolute_gap_disagreement_m=abs(difference) if difference is not None else None)


def consecutive_changes(rows):
    roofs = defaultdict(list)
    for r in rows:
        if r["landsat_minus_sentinel2_gap_m"] is not None:
            roofs[r["candidate_id"]].append(r)
    for candidate,group in sorted(roofs.items()):
        ordered = sorted(group,key=lambda r:r["landsat_optical_capture_at_utc"])
        for a,b in zip(ordered,ordered[1:]):
            earlier,later = (datetime.fromisoformat(r["landsat_optical_capture_at_utc"]) for r in (a,b))
            p = a["landsat_estimated_gap_m"]-b["landsat_estimated_gap_m"]
            s = a["sentinel2_estimated_gap_m"]-b["sentinel2_estimated_gap_m"]
            yield dict(candidate_id=candidate,roof_type=b["roof_type"],terminal_id=b["terminal_id"],
                diameter_m=b["diameter_m"],earlier_landsat_product_id=a["landsat_version_id"],
                later_landsat_product_id=b["landsat_version_id"],earlier_sentinel2_product_id=a["sentinel2_version_id"],
                later_sentinel2_product_id=b["sentinel2_version_id"],earlier_capture_at_utc=a["landsat_optical_capture_at_utc"],
                later_capture_at_utc=b["landsat_optical_capture_at_utc"],landsat_elapsed_days=(later-earlier).total_seconds()/86400,
                landsat_apparent_change_m=p,sentinel2_apparent_change_m=s,landsat_minus_sentinel2_change_m=p-s,
                landsat_both_diagnostics_pass=a["landsat_fit_status"]==b["landsat_fit_status"]==RESOLVED,
                sentinel2_both_diagnostics_pass=a["sentinel2_fit_status"]==b["sentinel2_fit_status"]==RESOLVED,
                both_inputs_available_at_each_matching_cutoff=bool(a["both_inputs_before_matching_cutoff"] and b["both_inputs_before_matching_cutoff"]))


def metrics(rows):
    v = np.array([r["landsat_minus_sentinel2_gap_m"] for r in rows],dtype=float)
    result = dict(pairs=len(rows),landsat_passes=len({r["landsat_version_id"] for r in rows}),unique_tanks=len({r["candidate_id"] for r in rows}))
    if len(v):
        result.update(median_signed_landsat_minus_sentinel2_m=float(np.median(v)),median_absolute_disagreement_m=float(np.median(abs(v))),
                      p90_absolute_disagreement_m=float(np.percentile(abs(v),90)),fraction_within_2m=float(np.mean(abs(v)<=2)))
    return result


def pass_rows(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["landsat_version_id"],r["sentinel2_version_id"],r["roof_type"]].append(r)
    for (p,s,roof),values in sorted(grouped.items(),key=lambda item:item[1][0]["landsat_optical_capture_at_utc"]):
        sample = [r for r in values if r["landsat_minus_sentinel2_gap_m"] is not None]
        passed = [r for r in sample if r["both_geometry_diagnostics_pass"]]
        yield dict(landsat_product_id=p,sentinel2_product_id=s,roof_type=roof,
            landsat_capture_at_utc=values[0]["landsat_optical_capture_at_utc"],
            sentinel2_capture_at_utc=values[0]["sentinel2_optical_capture_at_utc"],paired_tanks=len(values),
            both_inputs_before_cutoff=sum(r["both_inputs_before_matching_cutoff"] for r in values),
            both_numeric=len(sample),both_geometry_diagnostics_pass=len(passed),
            median_absolute_disagreement_m=metrics(sample).get("median_absolute_disagreement_m"),
            median_absolute_disagreement_both_pass_m=metrics(passed).get("median_absolute_disagreement_m"))


def change_metrics(rows):
    days = [r["landsat_elapsed_days"] for r in rows]
    delta = [abs(r["landsat_minus_sentinel2_change_m"]) for r in rows]
    return dict(pairs=len(rows),unique_tanks=len({r["candidate_id"] for r in rows}),
        minimum_elapsed_days=min(days,default=None),median_elapsed_days=float(np.median(days)) if days else None,
        median_absolute_difference_m=float(np.median(delta)) if delta else None,
        p90_absolute_difference_m=float(np.percentile(delta,90)) if delta else None)


def summarize(rows,fixed,open_changes):
    scopes = {"all":rows,"both_sources_historically_available":[r for r in rows if r["both_inputs_before_matching_cutoff"]]}
    result = {}
    for name,values in scopes.items():
        result[name] = {}
        for roof_type in ("EFR","IFR","Domed EFR"):
            selected = [r for r in values if r["roof_type"]==roof_type]
            usable = [r for r in selected if r["landsat_minus_sentinel2_gap_m"] is not None]
            resolved = [r for r in usable if r["both_geometry_diagnostics_pass"]]
            result[name][roof_type] = dict(total=len(selected),all_numeric=metrics(usable),both_diagnostics_pass=metrics(resolved),
                landsat_status=dict(Counter(r["landsat_fit_status"] for r in selected)),sentinel2_status=dict(Counter(r["sentinel2_fit_status"] for r in selected)))
    differences = [abs(r["landsat_estimated_gap_m"]-r["landsat_box_response_gap_m"]) for r in rows if r["landsat_estimated_gap_m"] is not None]
    return dict(by_scope_and_roof=result,fixed_roof_consecutive_pair_count=len(fixed),
        median_absolute_fixed_change_m={k:float(np.median([abs(r[f"{k}_apparent_change_m"]) for r in fixed])) for k in ("landsat","sentinel2")},
        consecutive_open_roof_changes=dict(all_numeric=change_metrics(open_changes),both_diagnostics_pass=change_metrics(
            [r for r in open_changes if r["landsat_both_diagnostics_pass"] and r["sentinel2_both_diagnostics_pass"]])),
        median_absolute_b8_psf_vs_box_difference_m=float(np.median(differences)),
        p90_absolute_b8_psf_vs_box_difference_m=float(np.percentile(differences,90)),
        instrument_source=PSF_DOC,instrument_workbook_sha256=PSF_SHA256,
        scope="Same-day agreement over a development-period archive. Neither satellite supplies ground truth. The B8 kernel uses the measured prelaunch line-spread functions, symmetrized and rotated using the scene-center orbit heading; the effective response of georectified Level-1 products remains approximate.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-pixels",action="store_true")
    args = parser.parse_args()
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        shells = dict(source.execute("SELECT candidate_id,lidar_rim_to_ground_native_z FROM tank_engineering_registry"))
        if not args.reuse_pixels:
            build_pixels(source,read_csv("landsat_sentinel2_pan_calibration_matches"),shells)
    sentinel,pan = fit_archive(shells)
    rows = list(compare_rows(sentinel,pan))
    changes = list(consecutive_changes(rows))
    fixed = [r for r in changes if r["roof_type"]!="EFR"]
    open_changes = [r for r in changes if r["roof_type"]=="EFR"]
    export("landsat_sentinel2_shadow_comparison",rows)
    export("landsat_sentinel2_shadow_by_pass",list(pass_rows(rows)))
    export("landsat_sentinel2_fixed_roof_changes",fixed)
    export("landsat_sentinel2_open_roof_changes",open_changes)
    summary = summarize(rows,fixed,open_changes)
    write(DERIVED/"landsat_sentinel2_shadow_summary.json",json.dumps(summary,indent=2).encode())
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
