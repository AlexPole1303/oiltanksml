"""Evaluate USGS B8 viewing and solar angles at native roof-center pixels."""
import csv
import hashlib
import math
import os
import re
import subprocess
import tarfile
from collections import defaultdict
from contextlib import closing

import numpy as np
import rasterio
from pyproj import Transformer

from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, get, write

TOOL_URL = "https://landsat.usgs.gov/sites/default/files/documents/L8_ANGLES_2_7_0.tgz"
TOOL_SHA256 = "20a8eae551cd48b4ce15ad7ce0991a45c7c1b7f15ac8baa7a3a9ec3aadda5c31"
WORK = ROOT/"tmp/usgs_landsat_angles"
FIELDS = ("native_line","native_sample","angle_status","solar_azimuth_deg","solar_zenith_deg","view_azimuth_deg","view_zenith_deg")


def prepare_tool():
    archive = WORK/"L8_ANGLES_2_7_0.tgz"
    payload = archive.read_bytes() if archive.exists() else get(TOOL_URL)
    if hashlib.sha256(payload).hexdigest()!=TOOL_SHA256:
        raise ValueError("The USGS angle tool archive differs from the verified version")
    if not archive.exists():
        write(archive,payload)
    directory = WORK/"l8_angles"
    if not (directory/"ias_lib/Makefile").is_file():
        with tarfile.open(archive) as source:
            source.extractall(WORK,filter="data")
    library = directory/"ias_lib"
    subprocess.run(["make","-C",str(library),"OPTS=-O2 -D_DARWIN_C_SOURCE"],check=True,capture_output=True)
    tool = WORK/"landsat_b8_angle_points"
    subprocess.run(["cc","-O2","-I",str(library),str(ROOT/"scripts/landsat_b8_angle_points.c"),
                    str(library/"libl8ang.a"),"-lm","-o",str(tool)],check=True)
    return tool


def angle_values(tool, angle_file, points):
    ordered = sorted(set(points))
    run = subprocess.run([str(tool),str(angle_file)],input="".join(f"{r} {c}\n" for r,c in ordered),
                         capture_output=True,text=True,env={**os.environ,"IAS_LOG_CHANNELS":"-IO","IAS_LOG_LEVEL":"ERROR"})
    if run.returncode:
        raise ValueError(f"USGS RPC evaluation failed for {angle_file}: {(run.stderr or run.stdout).strip()}")
    result = {}
    for row in csv.DictReader(run.stdout.splitlines(),fieldnames=FIELDS):
        point = int(row.pop("native_line")),int(row.pop("native_sample"))
        status = row.pop("angle_status")
        values = {k:float(v) if v else None for k,v in row.items()}
        if status=="resolved":
            for key in ("solar_azimuth_deg","view_azimuth_deg"):
                values[key] %= 360
            if not 0<values["solar_zenith_deg"]<90 or not 0<=values["view_zenith_deg"]<20:
                raise ValueError(f"Invalid daylight solar or view geometry for {angle_file} at {point}")
        result[point] = dict(angle_status=status,**values)
    if set(ordered)!=set(result):
        raise ValueError(f"USGS RPC did not return one result per B8 pixel for {angle_file}")
    return result


def source_observations(db):
    rows = db.execute("""SELECT p.product_id,q.candidate_id,t.documented_roof_type,t.documented_terminal_id,
        coalesce(l.center_east_m,c.center_east_m,t.footprint_east_m),coalesce(l.center_north_m,c.center_north_m,t.footprint_north_m),
        CASE WHEN l.center_east_m IS NOT NULL THEN 'lidar_elevated_circle'
             WHEN c.center_east_m IS NOT NULL THEN 'lidar_covered_surface_circle' ELSE 'mapped_area_circle' END
        FROM landsat_local_product p JOIN landsat_tank_scene_quality q USING(product_id)
        JOIN tank_engineering_registry t USING(candidate_id) LEFT JOIN tank_lidar_outline l USING(candidate_id)
        LEFT JOIN enbridge_main_covered_roof_outline c USING(candidate_id)
        WHERE p.source_kind='gcs_collection1_level1_pan' AND q.valid_signal_cells>0 AND t.documented_roof_type IS NOT NULL
        ORDER BY p.product_id,q.candidate_id""")
    groups = defaultdict(list)
    for product,*row in rows:
        groups[product].append(row)
    return groups


def native_origin(directory, angle_file):
    mtl = (directory/"MTL.txt").read_text(encoding="ascii")
    angle = angle_file.read_text(encoding="ascii")
    (x,y) = (float(re.search(rf"CORNER_UL_PROJECTION_{axis}_PRODUCT\s*=\s*([\d.+-]+)",mtl)[1]) for axis in ("X","Y"))
    ang = re.search(r"^\s*UL_CORNER\s*=\s*\(\s*([\d.+-]+),\s*([\d.+-]+)\s*\)",angle,re.M)
    if (x,y)!=tuple(map(float,ang.groups())):
        raise ValueError(f"MTL and ANG upper-left native pixel centers disagree for {directory}")
    return x,y


def roof_angles(tool, asset, roof_centers):
    angle_file = ROOT/asset["local_file"]
    directory = angle_file.parent
    product = asset["product_id"]
    x0,y0 = native_origin(directory,angle_file)
    with rasterio.open(directory/"pan.tif") as raster:
        if (raster.transform.a,raster.transform.b,raster.transform.d,raster.transform.e)!=(15,0,0,-15):
            raise ValueError(f"Clipped B8 grid was resampled for {product}")
        center_x,center_y = raster.xy(0,0)
        raw_line,raw_sample = (y0-center_y)/15,(center_x-x0)/15
        line0,sample0 = round(raw_line),round(raw_sample)
        if not math.isclose(raw_line,line0,rel_tol=0,abs_tol=1e-8) or not math.isclose(raw_sample,sample0,rel_tol=0,abs_tol=1e-8):
            raise ValueError(f"Clipped B8 grid disagrees with ANG and MTL for {product}")
        if not (0<=line0 and line0+raster.height<=int(asset["pan_image_lines"])
                and 0<=sample0 and sample0+raster.width<=int(asset["pan_image_samples"])):
            raise ValueError(f"Clipped B8 grid exceeds ANG dimensions for {product}")
        transform = Transformer.from_crs(32614,raster.crs,always_xy=True)
        data = raster.read(1,masked=True)
        rows,points = [],[]
        for candidate,roof_type,terminal,east,north,basis in roof_centers:
            x,y = transform.transform(east,north)
            clip_row,clip_col = raster.index(x,y)
            line,sample = line0+clip_row,sample0+clip_col
            inside = 0<=clip_row<raster.height and 0<=clip_col<raster.width
            value = data[clip_row,clip_col] if inside else np.ma.masked
            dn = int(value) if not np.ma.is_masked(value) else None
            center_x,center_y = raster.xy(clip_row,clip_col)
            rows.append(dict(product_id=product,capture_at_utc=asset["capture_at_utc"],candidate_id=candidate,
                roof_type=roof_type,terminal_id=terminal,center_basis=basis,crs=raster.crs.to_string(),native_line=line,native_sample=sample,
                native_pixel_center_x_m=center_x,native_pixel_center_y_m=center_y,center_distance_from_pixel_center_m=math.hypot(x-center_x,y-center_y),
                center_pixel_b8_dn=dn,angle_reference_height_m=0,angle_file=asset["local_file"],
                geometry_inputs_available_at_utc=asset["geometry_inputs_available_at_utc"]))
            points.append((line,sample))
    solved = angle_values(tool,angle_file,points)
    for row,point in zip(rows,points,strict=True):
        yield dict(row,**solved[point])


def main():
    tool = prepare_tool()
    with (DERIVED/"landsat_pan_angle_assets.csv").open(newline="",encoding="utf-8") as stream:
        assets = {row["product_id"]:row for row in csv.DictReader(stream)}
    with closing(read_only(SOURCE)) as db:
        groups = source_observations(db)
    rows = []
    for i,(product,centers) in enumerate(groups.items(),1):
        rows.extend(roof_angles(tool,assets[product],centers))
        if i%50==0 or i==len(groups):
            print(f"Evaluated Landsat B8 roof-center angles for {i}/{len(groups)} products",flush=True)
    export("landsat_pan_roof_angles",rows)
    print(f"Stored {len(rows)} original-grid roof-center angle rows",flush=True)


if __name__=="__main__":
    main()
