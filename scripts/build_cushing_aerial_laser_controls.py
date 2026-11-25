"""Reproduce aerial shadow controls on the two dates shared with ICESat-2."""
import csv
import json
import math
from collections import Counter
from datetime import datetime

import numpy as np
from pyproj import Transformer

from audit_shadow_precision import describe
from build_aerialok_concurrent_controls import CENTRAL, daylight_trials, exterior_profile
from build_aerialok_october_controls import interior_model, output, read_plate, register
from build_naip_2025_roof_reference import solar_module
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED
from build_weekly_shadow_validation import legacy_geometry
from collect_public import ROOT, write

STEM = "cushing_aerial_laser"
TO_GEO = Transformer.from_crs(32614,4326,always_xy=True)
PLATES = (
    ("laser_plains","2020-10-09",(-96.7382,35.93175,2940),{"plains_cushing"}),
    ("laser_north","2021-07-09",(-96.75485,36.011,2630),{"enbridge_blueknight"}),
    ("laser_west","2021-07-09",(-96.756,35.947,2800),{"enbridge_cushing"}))
REF_FIELDS = ("granule_start_utc","granule_end_utc","photon_csv","photon_packet","estimated_wall_to_roof_m","beam")


def geometry():
    registry = {r["candidate_id"]:r for r in read_csv("tank_engineering_registry")}
    result = []
    for tank,r in legacy_geometry().items():
        meta = registry[tank]
        lon,lat = TO_GEO.transform(float(r["center_east_m"]),float(r["center_north_m"]))
        result.append(dict(candidate_id=tank,longitude=lon,latitude=lat,diameter_m=float(r["diameter_m"]),
            shell_m=float(r["prior_rim_z_p95_native"])-float(r["prior_ground_z_p50_native"]),
            first_seen=meta["first_seen"],terminal_id=meta["documented_terminal_id"],
            geometry_dataset=r["dataset"],geometry_source_directory=r["source_directory"]))
    return result


def fit_plate(site,day,center,terminals,tanks,targets,spa):
    settings = {site:(*center,660,3150,3546,4050)}
    image,manifest = read_plate(site,day,sites=settings)
    lon,lat,_ = center
    selected = [r for r in tanks if r["terminal_id"] in terminals and r["first_seen"]<=day
        and abs((r["longitude"]-lon)*111_320*math.cos(math.radians(lat)))<image.shape[1]/2-45
        and abs((r["latitude"]-lat)*111_132)<image.shape[0]/2-45]
    fitted = list(register(site,image,selected,sites=settings,coarse_search_m=400))
    times,bearings,ratios = daylight_trials(spa,day,lon,lat)
    donors = {}
    for r in fitted:
        x,y = r["plate_x_at_1m"],r["plate_y_at_1m"]
        if min(x,y,image.shape[1]-x,image.shape[0]-y)<r["diameter_m"]/2+50 or r["rim_edge_contrast"]<12:
            continue
        profile = exterior_profile(image,r,fitted,bearings,ratios)
        if profile is not None:
            donors[r["candidate_id"]] = profile
    out = []
    for r in fitted:
        key = day,r["candidate_id"]
        if key not in targets:
            continue
        p = targets[key]
        if r["geometry_dataset"]!=p["geometry_dataset"] or not math.isclose(
            r["diameter_m"]/2,float(p["geometry_radius_m"]),rel_tol=0,abs_tol=1e-8):
            raise ValueError(f"Different laser and aerial engineering geometry for {key}")
        reference = [v for k,v in donors.items() if k!=key[1]]
        row = dict(r,site=site,flight_date=day,source_path=manifest["source_path"],source_sha256=manifest["sha256"],
            icesat_granule=p["filename"],icesat_cmr_id=p["cmr_id"],**{k:p[k] for k in REF_FIELDS},
            exterior_other_tanks=len(reference),clock_status="insufficient_exterior_donors",exterior_best_utc=None,
            exterior_trial_first_utc=None,exterior_trial_last_utc=None,exterior_near_span_minutes=None,
            exterior_gain=None,exterior_distinct_peak_margin=None,shadow_bearing_deg=None,horizontal_per_vertical=None,
            aerial_gap_m=None,aerial_fit_status="no_checked_exterior_clock",aerial_near_gap_min_m=None,
            aerial_near_gap_max_m=None,aerial_shadow_explained_fraction=None,aerial_lit_minus_shadow_gray=None,
            full_roof_inside_plate=None,maximum_separation_hours=None,aerial_minus_icesat_m=None,
            daylight_trial_max_separation_hours=None,aerial_gap_min_over_clock_profile_m=None,
            aerial_gap_max_over_clock_profile_m=None,all_near_clock_fits_visible=None,
            laser_inside_aerial_clock_and_fit_range=None)
        if len(reference)>=3:
            gain = np.median(reference,axis=0)
            best = int(np.argmax(gain))
            near = np.flatnonzero(gain>=gain[best]-.025)
            margin = gain[best]-max(gain[abs(np.arange(len(times))-best)>=10],default=0)
            good = gain[best]>=.08 and margin>=.025
            row.update(clock_status="shadow_inferred_clock" if good else "ambiguous_exterior_clock",
                exterior_best_utc=times[best],exterior_trial_first_utc=times[near[0]],exterior_trial_last_utc=times[near[-1]],
                exterior_near_span_minutes=(times[near[-1]]-times[near[0]]).total_seconds()/60,exterior_gain=float(gain[best]),
                exterior_distinct_peak_margin=float(margin),shadow_bearing_deg=float(bearings[best]),
                horizontal_per_vertical=float(ratios[best]))
            if good and r["rim_edge_contrast"]>=12:
                row.update(interior_model(image,r,bearings[best],ratios[best]))
                if row["aerial_fit_status"]=="visible_shadow_fit":
                    laser = tuple(datetime.fromisoformat(p[k]) for k in ("granule_start_utc","granule_end_utc"))
                    scenarios = [interior_model(image,r,bearings[i],ratios[i]) for i in near]
                    lower = min(a["aerial_near_gap_min_m"] for a in scenarios)
                    upper = max(a["aerial_near_gap_max_m"] for a in scenarios)
                    row.update(maximum_separation_hours=max(abs((t-edge).total_seconds()) for t in
                        (times[near[0]],times[near[-1]]) for edge in laser)/3600,
                        daylight_trial_max_separation_hours=max(abs((t-edge).total_seconds()) for t in
                            (times[0],times[-1]) for edge in laser)/3600,
                        aerial_gap_min_over_clock_profile_m=lower,aerial_gap_max_over_clock_profile_m=upper,
                        all_near_clock_fits_visible=all(a["aerial_fit_status"]=="visible_shadow_fit" for a in scenarios),
                        laser_inside_aerial_clock_and_fit_range=lower<=float(p["estimated_wall_to_roof_m"])<=upper,
                        aerial_minus_icesat_m=row["aerial_gap_m"]-float(p["estimated_wall_to_roof_m"]))
        out.append(row)
    return manifest,out


def main():
    with (ROOT/"data/reference/aerialok_cushing_flights.csv").open(newline="") as stream:
        dates = {r["flight_date"] for r in csv.DictReader(stream)}
    selected_dates = {p[1] for p in PLATES}
    if not selected_dates<=dates:
        raise ValueError("Unknown flight date in the Aerial Oklahoma calendar")
    targets = {}
    for r in read_csv("cushing_icesat2_roof_profiles"):
        start,end = (datetime.fromisoformat(r[k]).astimezone(CENTRAL) for k in ("granule_start_utc","granule_end_utc"))
        day = start.date().isoformat()
        if r["passes_roof_profile_checks"]=="True" and day in selected_dates and start.date()==end.date():
            key = day,r["candidate_id"]
            if key in targets:
                raise ValueError(f"Multiple laser profiles for one roof and day: {key}")
            targets[key] = r
    tanks = geometry()
    spa = solar_module()
    manifests,rows = [],[]
    for site,day,center,terminals in PLATES:
        manifest,measurements = fit_plate(site,day,center,terminals,tanks,targets,spa)
        manifests.append(manifest)
        rows.extend(measurements)
    if len(rows)!=len(targets) or {(r["flight_date"],r["candidate_id"]) for r in rows}!=set(targets):
        raise ValueError("Aerial export does not cover all targeted laser profiles")
    output(f"{STEM}_sheets",manifests)
    output(f"{STEM}_controls",rows)
    accepted = [r for r in rows if r["aerial_minus_icesat_m"] is not None]
    satellite = {(r["icesat_granule"],r["candidate_id"]):r for r in read_csv("cushing_icesat2_optical_selected")}
    combined = []
    for r in accepted:
        key = r["icesat_granule"],r["candidate_id"]
        other = satellite.get(key,{})
        hours = float(other["max_separation_hours"]) if other else None
        combined.append(dict(icesat_granule=key[0],candidate_id=key[1],terminal_id=r["terminal_id"],
            aerial_sheet=r["source_path"],aerial_inferred_upper_separation_hours=r["maximum_separation_hours"],
            aerial_minus_icesat_m=r["aerial_minus_icesat_m"],satellite_product=other.get("optical_product"),
            satellite_upper_separation_hours=hours,satellite_minus_icesat_m=other.get("optical_minus_laser_m"),
            reduction_in_estimated_upper_separation_hours=hours-r["maximum_separation_hours"] if hours is not None else None))
    output(f"{STEM}_satellite_overlap",combined)
    report = dict(sheets=len(manifests),target_profiles=len(targets),matched_profiles=len(rows),
        clock_status=dict(Counter(r["clock_status"] for r in rows)),
        aerial_fit_status=dict(Counter(r["aerial_fit_status"] for r in rows)),
        accepted_near_clock_fit_stable=sum(r["all_near_clock_fits_visible"] for r in accepted),
        laser_inside_aerial_clock_and_fit_range=sum(r["laser_inside_aerial_clock_and_fit_range"] for r in accepted),
        maximum_inferred_time_separation_hours=max(r["maximum_separation_hours"] for r in accepted),
        shorter_than_existing_satellite_controls=sum(
            r["reduction_in_estimated_upper_separation_hours"] is not None
            and r["reduction_in_estimated_upper_separation_hours"]>0 for r in combined),
        new_control_profiles_without_passing_satellite_fit=sum(r["satellite_product"] is None for r in combined),
        distinct_laser_roof_profiles_with_any_optical_control=len(set(satellite)|{
            (r["icesat_granule"],r["candidate_id"]) for r in accepted}),
        discrepancy=describe([r["aerial_minus_icesat_m"] for r in accepted]),
        interpretation="Aerial times are reconstructed from exterior shadows of other tanks in the same area. "
                       "Laser timestamps are the source granule interval. Measurements are retrospective controls.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
