"""Register the November and December aerial pairs and compare their Sentinel-2 inputs."""
import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing

import numpy as np
from PIL import Image, ImageDraw

from build_aerialok_october_controls import exterior_model, interior_model, output, read_plate, register, satellite_measurements, solar_declination
from collect_public import ROOT, write
from fit_shadow_geometry import read_only

DERIVED = ROOT/"data/derived"
DECEMBER_FLIGHTS = {"2020-12-04":"2020-12-05", "2020-12-10":"2020-12-10"}
NOVEMBER_FLIGHTS = {"2020-10-30":"2020-10-31", "2020-11-06":"2020-11-05"}
DECEMBER_SITES = {
    "dec_north": (-96.75485,36.011,2630,1000,3150,3546,4050),
    "dec_west": (-96.75066,35.942,3100,1320,3150,3546,4050),
    "dec_east": (-96.7166,35.94045,510,330,2400,2790,3180),
}
NOVEMBER_SITES = {site.replace("dec_","nov_"):location for site,location in DECEMBER_SITES.items() if site!="dec_east"}
NORTH = {"enbridge_blueknight","semgroup_cushing"}


def flight_catalog(source, flight_to_satellite, site_prefix):
    matches = defaultdict(list)
    with (DERIVED/"aerialok_sentinel2_flight_matches.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            date = row["flight_date"]
            if (date in flight_to_satellite and row["captured_at_utc"][:10]==flight_to_satellite[date]
                and row["documented_roof_type"]=="EFR" and row["complete_clear_b04_geometry"]=="True"):
                matches[date,row["candidate_id"]].append(row)
    first,last = flight_to_satellite
    ids = sorted({c for day,c in matches if day==first} & {c for day,c in matches if day==last})
    rows = source.execute(f"""SELECT candidate_id,documented_terminal_id,cohort_id,longitude,latitude,
        coalesce(lidar_upper_outline_diameter_m,footprint_diameter_m) AS diameter_m,lidar_rim_to_ground_native_z AS shell_m,
        CASE WHEN lidar_upper_outline_diameter_m IS NULL THEN 'mapped_area_diameter' ELSE 'lidar_upper_outline' END AS diameter_basis
        FROM tank_engineering_registry WHERE candidate_id IN ({','.join('?' for _ in ids)})
        ORDER BY latitude,longitude""",ids)
    tanks = []
    for record in rows:
        row = dict(record)
        terminal = row["documented_terminal_id"]
        side = "north" if terminal in NORTH else "east" if terminal=="ngl_cushing" else "west"
        row["site"] = f"{site_prefix}_{side}"
        block = terminal
        if terminal=="enbridge_cushing":
            block += "_north" if row["latitude"]>=35.948 else "_south"
        if terminal=="enbridge_blueknight":
            block += "_north" if row["latitude"]>=36.015 else "_central"
        if terminal=="magellan_cushing":
            block += "_central" if row["latitude"]>=35.936 else "_south"
        row["shadow_block"] = block
        tanks.append(row)
    if len(tanks)!=len(ids):
        raise ValueError("Missing engineering registration inputs for a common EFR contour")
    return tanks,matches


def plate_measurements(tanks, dates, sites, prefix, diffuse_days):
    manifests,rows,models = [],[],[]
    preview = Image.new("RGB",(len(sites)*470,len(dates)*525),(244,242,239))
    draw = ImageDraw.Draw(preview)
    for x,site in enumerate(sites):
        selected = [tank for tank in tanks if tank["site"]==site]
        for y,day in enumerate(dates):
            image,manifest = read_plate(site,day,sites=sites)
            manifests.append(manifest)
            fitted = list(register(site,image,selected,sites=sites,coarse_search_m=560))
            with Image.open(ROOT/manifest["source_path"]) as source:
                thumb = source.crop((0,0,manifest["sheet_width_px"],manifest["map_height_px"])).convert("RGB")
                thumb.thumbnail((464,490))
            x0,y0 = x*470,y*525+22
            preview.paste(thumb,(x0,y0))
            draw.text((x0+3,y0-18),f"{site} | {day} | {len(fitted)} candidate outlines",fill="black")
            blocks = defaultdict(list)
            for tank in fitted:
                blocks[tank["shadow_block"]].append(tank)
                x1 = x0+tank["plate_x_at_1m"]*thumb.width/image.shape[1]
                y1 = y0+tank["plate_y_at_1m"]*thumb.height/image.shape[0]
                radius = tank["diameter_m"]/2*thumb.width/image.shape[1]
                draw.ellipse((x1-radius,y1-radius,x1+radius,y1+radius),outline="#F78A50",width=1)
            for block,group in sorted(blocks.items()):
                bearing,ratio,gain,support = exterior_model(image,group,bearing_grid=np.arange(280,411,3),
                    ratio_grid=np.arange(1.4,3.61,0.1),outer_reach=60)
                observed_light = "diffuse_no_distinct_cast_shadows" if day in diffuse_days else "directional_cast_shadows_visible"
                models.append(dict(site=site,flight_date=day,shadow_block=block,contours=support,observed_illumination=observed_light,
                    tentative_shadow_bearing_deg=bearing,tentative_horizontal_per_vertical=ratio,median_exterior_explained_fraction=gain,
                    **solar_declination(day,sites[site][1],bearing,ratio)))
                for tank in group:
                    r = tank["diameter_m"]/2
                    cx,cy = tank["plate_x_at_1m"],tank["plate_y_at_1m"]
                    inside = bool(min(cx,cy,image.shape[1]-cx,image.shape[0]-cy)>=r+1)
                    fit = (interior_model(image,tank,bearing,ratio) if day not in diffuse_days and inside else
                           dict(aerial_gap_m=None,aerial_fit_status="diffuse_illumination" if inside else "outline_outside_plate",
                                aerial_shadow_explained_fraction=None,aerial_lit_minus_shadow_gray=None,
                                aerial_near_gap_min_m=None,aerial_near_gap_max_m=None,full_roof_inside_plate=inside))
                    rows.append(dict(flight_date=day,source_path=manifest["source_path"],observed_illumination=observed_light,**tank,**fit))
            print(site,day,"registered",len(fitted),"median rim contrast",round(float(np.median([t["rim_edge_contrast"] for t in fitted])),2),flush=True)
    preview.save(DERIVED/f"{prefix}_registered_maps.png")
    output(f"{prefix}_export_manifest",manifests)
    output(f"{prefix}_aerial_observations",rows)
    output(f"{prefix}_shadow_quality",models)
    return rows,models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("interval",choices=("december","november"))
    interval = parser.parse_args().interval
    is_december = interval=="december"
    flight_to_satellite,sites = (DECEMBER_FLIGHTS,DECEMBER_SITES) if is_december else (NOVEMBER_FLIGHTS,NOVEMBER_SITES)
    diffuse_days = {"2020-12-10"} if is_december else set()
    prefix = f"aerialok_{interval}"
    with closing(read_only(DERIVED/"cushing.sqlite")) as source:
        source.row_factory = sqlite3.Row
        tanks,matches = flight_catalog(source,flight_to_satellite,interval[:3])
        aerial,models = plate_measurements(tanks,flight_to_satellite,sites,prefix,diffuse_days)
        optical = satellite_measurements(source,tanks,days=tuple(flight_to_satellite.values()),database=DERIVED/f"{prefix}_satellite.sqlite")
    by_flight = {(r["candidate_id"],r["flight_date"]):r for r in aerial}
    first,last = flight_to_satellite
    rows = []
    for tank in tanks:
        candidate = tank["candidate_id"]
        data = []
        for day in flight_to_satellite:
            image = by_flight[candidate,day]
            sat = optical.get((candidate,flight_to_satellite[day]))
            match = next((r for r in matches[day,candidate] if sat and r["l1c_product_name"]==sat["version_id"]),None)
            if sat and not match:
                raise ValueError(f"Selected L1C fit outside the matching catalog for {candidate} on {day}")
            data.append((image,sat,match))
        a,b = data
        delta = a[1]["estimated_gap_m"]-b[1]["estimated_gap_m"] if a[1] and b[1] else None
        aerial_delta = (a[0]["aerial_gap_m"]-b[0]["aerial_gap_m"]
                        if a[0]["aerial_fit_status"]==b[0]["aerial_fit_status"]=="visible_shadow_fit" else None)
        difference = delta-aerial_delta if delta is not None and aerial_delta is not None else None
        rows.append(dict(candidate_id=candidate,site=tank["site"],terminal_id=tank["documented_terminal_id"],
            earlier_aerial_date=first,later_aerial_date=last,earlier_aerial_gap_m=a[0]["aerial_gap_m"],
            earlier_aerial_status=a[0]["aerial_fit_status"],later_aerial_gap_m=b[0]["aerial_gap_m"],
            later_aerial_status=b[0]["aerial_fit_status"],aerial_level_change_m=aerial_delta,
            earlier_satellite_version=a[1]["version_id"] if a[1] else None,later_satellite_version=b[1]["version_id"] if b[1] else None,
            earlier_satellite_captured_at_utc=a[1]["optical_capture_at_utc"] if a[1] else None,
            later_satellite_captured_at_utc=b[1]["optical_capture_at_utc"] if b[1] else None,
            earlier_minimum_possible_separation_hours=a[2]["minimum_possible_separation_hours"] if a[2] else None,
            earlier_maximum_possible_separation_hours=a[2]["maximum_possible_separation_hours"] if a[2] else None,
            later_minimum_possible_separation_hours=b[2]["minimum_possible_separation_hours"] if b[2] else None,
            later_maximum_possible_separation_hours=b[2]["maximum_possible_separation_hours"] if b[2] else None,
            earlier_satellite_status=a[1]["fit_status"] if a[1] else None,later_satellite_status=b[1]["fit_status"] if b[1] else None,
            optical_level_change_m=delta,optical_minus_aerial_change_m=difference))
    output(f"{prefix}_level_pairs",rows)
    discrepancies = [abs(r["optical_minus_aerial_change_m"]) for r in rows if r["optical_minus_aerial_change_m"] is not None]
    both_diagnostic = [r for r in rows if r["optical_minus_aerial_change_m"] is not None
                       and r["earlier_satellite_status"]==r["later_satellite_status"]=="resolved_in_geometry_diagnostic"]
    summary = dict(common_contours=len(rows),by_terminal=dict(Counter(r["terminal_id"] for r in rows)),
        aerial_observations=len(aerial),complete_roof_views=sum(r["full_roof_inside_plate"] for r in aerial),
        directional_shadow_fits_by_date={day:sum(r["flight_date"]==day and r["aerial_fit_status"]=="visible_shadow_fit" for r in aerial)
                                        for day in flight_to_satellite},
        optical_change_rows=sum(r["optical_level_change_m"] is not None for r in rows),
        optical_resolved_on_both_dates=sum(r["earlier_satellite_status"]==r["later_satellite_status"]=="resolved_in_geometry_diagnostic" for r in rows),
        aerial_change_rows=sum(r["aerial_level_change_m"] is not None for r in rows),
        matched_aerial_optical_changes=len(discrepancies),matched_changes_with_both_satellite_diagnostics=len(both_diagnostic),
        median_abs_change_discrepancy_m=float(np.median(discrepancies)) if discrepancies else None,
        p90_abs_change_discrepancy_m=float(np.quantile(discrepancies,0.9)) if discrepancies else None,
        block_illumination_checks=models)
    write(DERIVED/f"{prefix}_summary.json",json.dumps(summary,indent=2).encode())
    print({k:v for k,v in summary.items() if k!="block_illumination_checks"},flush=True)


if __name__=="__main__":
    main()
