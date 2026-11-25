"""Reconstruct October 2020 shadow geometry from public north-up aerial sheets."""
import csv
import datetime as dt
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing

import numpy as np
from PIL import Image, ImageDraw
from pyproj import Geod

from build_l1c_quality import candidate_layer
from build_shadow_calibration import SCHEMA, add_psf, add_scene
from build_shadow_change_controls import SELECT_QUALITY, scale
from collect_public import RAW, ROOT, write, write_csv
from fit_shadow_geometry import estimate, inputs, read_only

DERIVED = ROOT / "data" / "derived"
DATABASE = DERIVED / "aerialok_october_satellite.sqlite"
SHEETS = RAW / "aerialok_cushing"
DAYS = ("2020-10-06", "2020-10-16")
SITES = {
    "ngl": (-96.7175314, 35.9412448, 490, 200, 3150, 3546, 4050),
    "enbridge_north": (-96.75182, 35.95450, 530, 200, 2400, 2790, 3180),
    "plains_phase_vi": (-96.73537, 35.93755, 500, 330, 2400, 2790, 3180),
}
GEOD = Geod(ellps="WGS84")


def output(name, rows):
    fields = tuple(rows[0])
    write_csv(DERIVED/f"{name}.csv", fields, ([r.get(f) for f in fields] for r in rows))


def registry(db):
    with (DERIVED/"aerialok_candidate_change_intervals.csv").open(newline="") as stream:
        entry = next(r for r in csv.DictReader(stream) if (r["earlier_flight_date"],r["later_flight_date"])==DAYS)
    ids = sorted(json.loads(entry["same_day_common_efr_candidate_ids_json"]))
    rows = db.execute(f"""SELECT candidate_id,longitude,latitude,lidar_upper_outline_diameter_m AS diameter_m,
        lidar_rim_to_ground_native_z AS shell_m,cohort_id FROM tank_engineering_registry
        WHERE candidate_id IN ({','.join('?' for _ in ids)}) ORDER BY candidate_id""", ids)
    return [dict(r) for r in rows]


def read_plate(site, day, *, sites=SITES, suffix=None):
    lon,lat,alt,scale_ft,width,map_rows,height = sites[site]
    if suffix is None:
        suffix = "_wide" if site == "plains_phase_vi" else ""
    path = SHEETS/f"{site}_{day}{suffix}.png"
    metres_per_px = scale_ft*0.3048/300
    with Image.open(path) as im:
        if im.size!=(width,height) or not math.isclose(im.info["dpi"][0],300,abs_tol=0.01):
            raise ValueError(f"Unexpected printed scale metadata: {path}")
        plate = im.crop((0,0,width,map_rows)).convert("L").resize((round(width*metres_per_px),round(map_rows*metres_per_px)),Image.Resampling.BOX)
    url = f"https://view.aerialok.com/?p=Cushing&d={day}&cam={lon},{lat},{alt},0,-90&list=0"
    return np.asarray(plate,dtype=float),dict(site=site,flight_date=day,source_url=url,source_path=str(path.relative_to(ROOT)),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scale_ft_per_in=scale_ft,dpi=300,metres_per_printed_pixel=metres_per_px,
        sheet_width_px=width,map_height_px=map_rows,sheet_height_px=height,source_time_precision="local_calendar_day")


def sample(image, x, y):
    col,row = np.rint(x).astype(int),np.rint(y).astype(int)
    inside = (col>=0)&(row>=0)&(col<image.shape[1])&(row<image.shape[0])
    return np.where(inside,image[row.clip(0,image.shape[0]-1),col.clip(0,image.shape[1]-1)],np.nan)


def rim_score(image, x, y, radius):
    angles = np.radians(np.arange(0,360,8))
    dx,dy = np.cos(angles),np.sin(angles)
    intensity = [sample(image,x[:,None]+dx*(radius+d),y[:,None]+dy*(radius+d)) for d in (-3,-1,1,3)]
    difference = (intensity[0]+intensity[1]-intensity[2]-intensity[3])/2
    return np.mean(np.nan_to_num(np.maximum(difference,-30),nan=-70),axis=1)


def register(site, image, tanks, *, sites=SITES, coarse_search_m=None, centre_shift_m=None):
    lon,lat,*_ = sites[site]
    centres = []
    for tank in tanks:
        bearing,_,distance = GEOD.inv(lon,lat,tank["longitude"],tank["latitude"])
        east,north = distance*math.sin(math.radians(bearing)),distance*math.cos(math.radians(bearing))
        centres.append((image.shape[1]/2+east,image.shape[0]/2-north))
    if centre_shift_m is not None:
        # An independent exterior registration can anchor the shared centre.
        # Keep a two-metre refinement before the original local rim fit.
        east_axis, south_axis = (round(v) + np.arange(-2, 3, 2) for v in centre_shift_m)
    elif coarse_search_m is None:
        east_axis,south_axis = np.arange(-120,121,2),np.arange(-144,81,2)
    else:
        coarse = np.arange(-coarse_search_m,coarse_search_m+1,8)
        east,south = np.meshgrid(coarse,coarse)
        east,south = east.ravel(),south.ravel()
        anchors = np.unique(np.linspace(0,len(tanks)-1,min(16,len(tanks)),dtype=int))
        scores = [rim_score(image,centres[i][0]+east,centres[i][1]+south,tanks[i]["diameter_m"]/2) for i in anchors]
        best = np.argmax(np.mean(scores,axis=0))
        east_axis,south_axis = (centre+np.arange(-16,17,2) for centre in (east[best],south[best]))
    east,south = np.meshgrid(east_axis,south_axis)
    east,south = east.ravel(),south.ravel()
    scores = [rim_score(image,x+east,y+south,t["diameter_m"]/2) for t,(x,y) in zip(tanks,centres,strict=True)]
    shift = np.argmax(np.mean(scores,axis=0))
    dx,dy = east[shift],south[shift]
    local_e,local_s = np.meshgrid(np.arange(-3,4),np.arange(-3,4))
    local_e,local_s = local_e.ravel(),local_s.ravel()
    for t,(x,y) in zip(tanks,centres,strict=True):
        cx,cy = x+dx+local_e,y+dy+local_s
        values = rim_score(image,cx,cy,t["diameter_m"]/2)
        best = np.argmax(values)
        yield dict(t,plate_x_at_1m=cx[best],plate_y_at_1m=cy[best],group_shift_east_m=int(dx),group_shift_south_m=int(dy),
                   local_shift_east_m=int(local_e[best]),local_shift_south_m=int(local_s[best]),rim_edge_contrast=float(values[best]))


def direction(bearing):
    radians = np.radians(bearing)
    return np.sin(radians),-np.cos(radians)


def scores(signal, shadow):
    count = shadow.sum(axis=1)
    mean_dark = (shadow*signal).sum(axis=1)/np.maximum(count,1)
    mean_lit = (signal.sum()-(shadow*signal).sum(axis=1))/np.maximum(len(signal)-count,1)
    difference = mean_lit-mean_dark
    spread = max(float(np.square(signal-signal.mean()).sum()),1e-8)
    gain = count*(len(signal)-count)/len(signal)*np.square(np.maximum(difference,0))/spread
    gain[(count<8)|(count>len(signal)-8)] = 0
    return gain,difference


def exterior_model(image, tanks, *, bearing_grid=None, ratio_grid=None, outer_reach=40):
    bearings,ratios = np.meshgrid(np.arange(275,361,2) if bearing_grid is None else bearing_grid,
                                  np.arange(0.8,2.61,0.05) if ratio_grid is None else ratio_grid)
    bearings,ratios = bearings.ravel(),ratios.ravel()
    ex,ey = (v[:,None] for v in direction(bearings))
    gains = []
    for tank in tanks:
        r,h = tank["diameter_m"]/2,tank["shell_m"]
        x0,y0 = tank["plate_x_at_1m"],tank["plate_y_at_1m"]
        pos = np.arange(-r-outer_reach-2,r+outer_reach+3,2)
        xx,yy = (v.ravel() for v in np.meshgrid(pos,pos))
        radius = np.hypot(xx,yy)
        valid = (radius>r+3)&(radius<r+outer_reach)&(yy<r*0.65)
        if bearing_grid is None:
            valid &= xx<r*0.65
        for other in tanks:
            if other is not tank:
                distance = np.hypot(x0+xx-other["plate_x_at_1m"],y0+yy-other["plate_y_at_1m"])
                valid &= distance>other["diameter_m"]/2+3
        light = sample(image,x0+xx,y0+yy)
        valid &= np.isfinite(light)
        xx,yy,light = xx[valid],yy[valid],light[valid]
        if len(light)<50:
            continue
        projected = np.clip(ex*xx+ey*yy,0,ratios[:,None]*h)
        shadow = (xx-ex*projected)**2+(yy-ey*projected)**2<=r*r
        gains.append(scores(light,shadow)[0])
    aggregate = np.median(gains,axis=0)
    best = np.argmax(aggregate)
    return float(bearings[best]),float(ratios[best]),float(aggregate[best]),len(gains)


def interior_model(image, tank, bearing, ratio):
    r,h = tank["diameter_m"]/2,tank["shell_m"]
    pos = np.arange(-r,r+1,0.8)
    xx,yy = (v.ravel() for v in np.meshgrid(pos,pos))
    valid = np.hypot(xx,yy)<r-1.7
    light = sample(image,tank["plate_x_at_1m"]+xx,tank["plate_y_at_1m"]+yy)
    valid &= np.isfinite(light)
    xx,yy,light = xx[valid],yy[valid],light[valid]
    gaps = np.arange(0,h+0.01,0.25)
    ex,ey = direction(bearing)
    travel = gaps[:,None]*ratio
    shadow = (xx-ex*travel)**2+(yy-ey*travel)**2>r*r
    gain,contrast = scores(light,shadow)
    best = int(np.argmax(gain))
    near = gaps[gain>=gain[best]-0.02]
    x,y = tank["plate_x_at_1m"],tank["plate_y_at_1m"]
    full_roof = min(x,y,image.shape[1]-x,image.shape[0]-y)>=r+1
    status = "visible_shadow_fit" if full_roof and gain[best]>=0.3 and contrast[best]>=35 and 0<best<len(gaps)-1 else "weak_or_boundary_fit"
    return dict(aerial_gap_m=float(gaps[best]),aerial_fit_status=status,aerial_shadow_explained_fraction=float(gain[best]),
                aerial_lit_minus_shadow_gray=float(contrast[best]),aerial_near_gap_min_m=float(near.min()),aerial_near_gap_max_m=float(near.max()),
                full_roof_inside_plate=bool(full_roof))


def solar_declination(day, latitude, shadow_bearing, ratio):
    date = dt.date.fromisoformat(day)
    days_in_year = (dt.date(date.year+1,1,1)-dt.date(date.year,1,1)).days
    angle = 2*math.pi*(date.timetuple().tm_yday-1)/days_in_year
    # NOAA fractional-year approximation at midday on the displayed calendar date.
    expected = (0.006918-0.399912*math.cos(angle)+0.070257*math.sin(angle)-0.006758*math.cos(2*angle)
                +0.000907*math.sin(2*angle)-0.002697*math.cos(3*angle)+0.00148*math.sin(3*angle))
    latitude = math.radians(latitude)
    sun_azimuth = math.radians((shadow_bearing+180)%360)
    elevation = math.atan2(1,ratio)
    implied = math.asin(math.sin(latitude)*math.sin(elevation)+math.cos(latitude)*math.cos(elevation)*math.cos(sun_azimuth))
    return dict(implied_solar_declination_deg=math.degrees(implied),noaa_midday_solar_declination_deg=math.degrees(expected),
                implied_minus_noaa_declination_deg=math.degrees(implied-expected))


def contact_sheet(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["candidate_id"]][row["flight_date"]] = row
    cell_w,cell_h = 188,208
    sheet = Image.new("RGB",(cell_w*6,cell_h*((len(groups)+2)//3)),(243,242,240))
    draw = ImageDraw.Draw(sheet)
    ordered = sorted(groups.values(),key=lambda g:(g[DAYS[0]]["site"],-g[DAYS[0]]["latitude"],g[DAYS[0]]["longitude"]))
    with ExitStack() as stack:
        plates = {path:stack.enter_context(Image.open(ROOT/path)) for path in sorted({r["source_path"] for r in rows})}
        for index,group in enumerate(ordered):
            for offset,day in enumerate(DAYS):
                r = group[day]
                _,_,_,scale_ft,width,map_rows,_ = SITES[r["site"]]
                metres_per_px = scale_ft*0.3048/300
                px = r["plate_x_at_1m"]*width/round(width*metres_per_px)
                py = r["plate_y_at_1m"]*map_rows/round(map_rows*metres_per_px)
                radius = r["diameter_m"]/2/metres_per_px
                arm = radius*1.65
                tile = plates[r["source_path"]].crop(tuple(round(n) for n in (px-arm,py-arm,px+arm,py+arm)))
                tile = tile.convert("RGB").resize((181,181),Image.Resampling.LANCZOS)
                x,y = (index%3*2+offset)*cell_w,index//3*cell_h
                sheet.paste(tile,(x,y+24))
                draw.text((x+3,y+2),f"{r['candidate_id'][-3:]}  {day}  {r['aerial_gap_m']:.2f} m",fill="black")
                cx,cy = x+90.5,y+114.5
                rr = radius*181/(arm*2)
                draw.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),outline=(223,92,58),width=1)
    sheet.save(DERIVED/"aerialok_october_registered_preview.png")
def aerial_measurements(tanks):
    manifests,rows = [],[]
    for site in SITES:
        selected = [t for t in tanks if t["cohort_id"].startswith(site.split("_")[0])]
        for day in DAYS:
            image,manifest = read_plate(site,day)
            fitted = list(register(site,image,selected))
            bearing,ratio,gain,used = exterior_model(image,fitted)
            manifest.update(exterior_shadow_bearing_deg=bearing,exterior_horizontal_per_vertical=ratio,
                            **solar_declination(day,SITES[site][1],bearing,ratio))
            manifests.append(manifest)
            print(site,day,"shadow",bearing,ratio,round(gain,3),used,flush=True)
            for t in fitted:
                rows.append(dict(site=site,flight_date=day,source_path=manifest["source_path"],exterior_shadow_grid_bearing_deg=bearing,
                    exterior_horizontal_per_vertical=ratio,exterior_median_explained_fraction=gain,exterior_fit_contours=used,
                    **t,**interior_model(image,t,bearing,ratio)))
    output("aerialok_october_export_manifest",manifests)
    output("aerialok_october_roof_measurements",rows)
    contact_sheet(rows)
    return rows


def satellite_measurements(source,tanks, *, days=DAYS, database=DATABASE, workers=1):
    ids = [t["candidate_id"] for t in tanks]
    chosen = source.execute(SELECT_QUALITY+f"""AND substr(q.captured_at_utc,1,10) IN ({','.join('?' for _ in days)})
        AND q.candidate_id IN ({','.join('?' for _ in ids)}) AND optical_cells=valid_cells
        AND roof_clear_by_source_mask AND resolved_b04_detector""",(*days,*ids))
    groups = defaultdict(list)
    for r in chosen:
        groups[r["level"],r["version_id"],r["optical_source_directory"]].append(r)
    candidates = dict((*candidate_layer("osm"), *candidate_layer("supplemental")))
    temporary = database.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    with closing(sqlite3.connect(temporary)) as db,db:
        db.executescript(SCHEMA)
        add_psf(db)
        identities = {}
        for scene,(key,rows) in enumerate(sorted(groups.items()),1):
            add_scene(db,source,scene,key,rows,candidates,identities)
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok":
            raise ValueError("Aerial control optical database integrity check failed")
    temporary.replace(database)
    shells = {t["candidate_id"]:t["shell_m"] for t in tanks}
    results = defaultdict(list)
    with closing(read_only(database)) as db:
        db.row_factory = sqlite3.Row
        observations,kernels = inputs(db)
    def fit(o):
        return estimate(o,shells[o["candidate_id"]],kernels[o["platform"]],calibration_db=database)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(fit, observations):
            if r.get("estimated_gap_m") is not None:
                results[r["candidate_id"],r["optical_capture_at_utc"][:10]].append(r)
    return {key:min(rows,key=lambda r:(-r["fitted_pixels"],r["version_id"])) for key,rows in results.items()}


def main():
    with closing(read_only(DERIVED/"cushing.sqlite")) as source:
        source.row_factory = sqlite3.Row
        tanks = registry(source)
        aerial = aerial_measurements(tanks)
        optical = satellite_measurements(source,tanks)
    paired = defaultdict(dict)
    for r in aerial:
        paired[r["candidate_id"]][r["flight_date"]] = r
    changes = []
    for candidate,rows in sorted(paired.items()):
        first,last = (rows[d] for d in DAYS)
        a,b = (optical[candidate,d] for d in DAYS)
        aerial_change = first["aerial_gap_m"]-last["aerial_gap_m"]
        optical_change = a["estimated_gap_m"]-b["estimated_gap_m"]
        error = optical_change-aerial_change
        changes.append(dict(candidate_id=candidate,site=first["site"],earlier_date=DAYS[0],later_date=DAYS[1],
            earlier_aerial_gap_m=first["aerial_gap_m"],later_aerial_gap_m=last["aerial_gap_m"],aerial_level_change_m=aerial_change,
            earlier_aerial_status=first["aerial_fit_status"],later_aerial_status=last["aerial_fit_status"],
            earlier_optical_gap_m=a["estimated_gap_m"],later_optical_gap_m=b["estimated_gap_m"],optical_level_change_m=optical_change,
            earlier_optical_status=a["fit_status"],later_optical_status=b["fit_status"],earlier_optical_version=a["version_id"],
            later_optical_version=b["version_id"],earlier_optical_captured_at_utc=a["optical_capture_at_utc"],
            later_optical_captured_at_utc=b["optical_capture_at_utc"],optical_minus_aerial_change_m=error,
            optical_minus_aerial_change_equivalent_bbl=error*scale(a)))
    output("aerialok_october_level_change_comparison",changes)
    strong = [r for r in changes if r["earlier_aerial_status"]==r["later_aerial_status"]=="visible_shadow_fit"]
    summary = dict(tanks=len(tanks),aerial_observations=len(aerial),matched_change_rows=len(changes),
        complete_aerial_roof_views=sum(r["full_roof_inside_plate"] for r in aerial),
        visible_shadow_fit_both_dates=len(strong),
        satellite_geometry_resolved_both_dates=sum(r["earlier_optical_status"]==r["later_optical_status"]=="resolved_in_geometry_diagnostic" for r in changes),
        median_abs_optical_minus_aerial_change_m=float(np.median([abs(r["optical_minus_aerial_change_m"]) for r in strong])) if strong else None,
        p90_abs_optical_minus_aerial_change_m=float(np.quantile([abs(r["optical_minus_aerial_change_m"]) for r in strong],0.9)) if strong else None,
        scope="Photogrammetric sheet shadow estimates use a common exterior shadow ratio per plate and LiDAR rim heights. Exact airborne clocks and survey-grade vertical accuracy are unavailable.")
    write(DERIVED/"aerialok_october_change_summary.json",json.dumps(summary,indent=2).encode())
    print(summary,flush=True)


if __name__=="__main__":
    main()
