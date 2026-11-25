"""Compute exact-orbit range geometry for the planned Cushing Sentinel-1 bursts."""
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import timedelta
from functools import cache

import numpy as np
from pyproj import Transformer
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq

from build_cushing_icesat2_controls import identities
from build_s1_slc_2019_reference_plan import WORST_NOMINAL_SLC_SLANT_RESOLUTION_M
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED
from collect_public import ROOT, write, write_csv
from collect_s1_slc_orbits import parse_orbit, utc

STEM = "s1_slc_orbit_geometry"
SIDES = ("previous","current")
ECEF = Transformer.from_crs(4979,4978,always_xy=True)


@cache
def orbit_grid(path,filename,digest):
    payload = gzip.decompress((ROOT/path).read_bytes())
    if hashlib.sha256(payload).hexdigest()!=digest:
        raise ValueError(f"Orbit checksum changed for {filename}")
    data = parse_orbit(payload,filename)
    position = CubicHermiteSpline(data["seconds"],data["state"][:,:3],data["state"][:,3:],extrapolate=False)
    return data["start"],position,position.derivative()


def coordinates(tank,height):
    lon,lat = tank["longitude"],tank["latitude"]
    return np.array(ECEF.transform(lon,lat,height))


def look(tank,height,row):
    origin,position,velocity = orbit_grid(row["local_orbit_gzip"],row["orbit_filename"],row["orbit_sha256"])
    first,last = ((utc(row[k])-origin).total_seconds() for k in ("capture_start_utc","capture_end_utc"))
    target = coordinates(tank,height)
    doppler = lambda t: np.dot(position(t)-target,velocity(t))
    lower,upper = first-10,last+10
    if not doppler(lower)<0<doppler(upper):
        raise ValueError(f"No nearby zero-Doppler solution for {row['product_name']}/{tank['candidate_id']}")
    closest = brentq(doppler,lower,upper,xtol=1e-7)
    look_vector = position(closest)-target
    distance = float(np.linalg.norm(look_vector))
    lat,lon = map(math.radians,(tank["latitude"],tank["longitude"]))
    up = np.array([math.cos(lat)*math.cos(lon),math.cos(lat)*math.sin(lon),math.sin(lat)])
    incidence = math.degrees(math.acos(np.dot(look_vector,up)/distance))
    if not 20<incidence<55 or not 700_000<distance<1_100_000:
        raise ValueError(f"Unexpected Sentinel-1 line-of-sight geometry for {tank['candidate_id']}")
    return (origin+timedelta(seconds=closest)).isoformat(),distance,incidence,first<=closest<=last


@cache
def burst_catalog(path):
    return {r["Id"]:r for r in json.loads((ROOT/path).read_bytes())["value"]}


def tank_geometry(tank,product,burst):
    ground = tank["ground_hae_approx_m"]
    rim = tank["top_hae_approx_m"]
    when,distance,angle,parent = look(tank,rim,product)
    _,ground_distance,_,_ = look(tank,ground,product)
    shell_shift = ground_distance-distance
    if not 0<shell_shift<rim-ground or not math.isclose(shell_shift,(rim-ground)*math.cos(math.radians(angle)),
                                                         rel_tol=0,abs_tol=.01):
        raise ValueError(f"Orbit and local vertical disagree for {tank['candidate_id']}")
    burst_start,burst_stop = (utc(burst[k]) for k in ("BeginningDateTime","EndingDateTime"))
    return dict(rim_zero_doppler_at_utc=when,rim_slant_range_m=distance,rim_incidence_deg=angle,
        rim_zero_doppler_inside_parent=parent,rim_zero_doppler_inside_cdse_catalog_time=burst_start<=utc(when)<=burst_stop,
        rim_seconds_from_cdse_catalog_start=(utc(when)-burst_start).total_seconds(),
        rim_seconds_from_cdse_catalog_end=(utc(when)-burst_stop).total_seconds(),
        rim_to_ground_slant_range_m=shell_shift,maximum_slant_displacement_at_least_3_5m=
            shell_shift>=WORST_NOMINAL_SLC_SLANT_RESOLUTION_M)


def main():
    tanks,_,_ = identities()
    orbits = {r["product_id"]:r for r in read_csv("s1_slc_orbits")}
    orbit_listings = {r["product_id"]:r for r in read_csv("s1_slc_restituted_orbits")}
    resorb = {p:r for p,r in orbit_listings.items() if r["status"]=="verified_resorb"}
    next_cutoff = {r["week_ending"]:r["cutoff_utc"] for r in read_csv("holdout_optical_weeks")}
    staged = {}

    def geometry(tank_id,parent_id,burst_id,path):
        key = tank_id,parent_id,burst_id
        if key not in staged:
            orbit = orbits[parent_id]
            burst = burst_catalog(path)[burst_id]
            if burst["ParentProductName"]!=orbit["product_name"] or burst["ParentProductId"]!=parent_id:
                raise ValueError(f"Burst and orbit refer to different scenes for {tank_id}")
            tank = tanks[tank_id]
            values = tank_geometry(tank,orbit,burst)
            if parent_id in resorb:
                current = orbit | resorb[parent_id]
                _,distance,angle,_ = look(tank,tank["top_hae_approx_m"],current)
                values.update(resorb_minus_poeorb_rim_slant_range_m=distance-values["rim_slant_range_m"],
                              resorb_minus_poeorb_rim_incidence_deg=angle-values["rim_incidence_deg"])
            staged[key] = values
        return staged[key]

    rows = []
    for r in read_csv("s1_holdout_gap_burst_tank_pairs"):
        if r["full_context_in_both_bursts"]!="True":
            continue
        tank_id = r["candidate_id"]
        endpoints = [geometry(tank_id,*(r[f"{side}_{k}"] for k in ("product_id","cdse_uuid","cdse_catalog_file"))) for side in SIDES]
        cutoff = utc(next_cutoff[r["week_ending"]])
        exact_orbits_before = all(utc(orbits[r[f"{side}_product_id"]]["archive_last_modified_at_utc"])<cutoff for side in SIDES)
        quick_orbits_before = all((parent:=r[f"{side}_product_id"]) in resorb and
            utc(resorb[parent]["archive_last_modified_at_utc"])<cutoff for side in SIDES)
        listing_dates = [orbit_listings[r[f"{side}_product_id"]]["asf_directory_modified_at_utc"] for side in SIDES]
        rows.append(dict({k:r[k] for k in ("week_ending","release_date","candidate_id","terminal_id","common_relative_burst_id",
                    "swath","polarization","previous_product_id","current_product_id")},
            forecast_cutoff_utc=cutoff.isoformat(),both_precise_orbit_files_on_aws_before_cutoff=exact_orbits_before,
            both_restituted_orbit_files_asf_before_cutoff=quick_orbits_before,
            both_restituted_orbits_listed_before_cutoff=all(d and utc(d)<cutoff for d in listing_dates),
            both_maximum_slant_displacements_at_least_3_5m=all(g["maximum_slant_displacement_at_least_3_5m"] for g in endpoints),
            **{f"{side}_{k}":v for side,g in zip(SIDES,endpoints,strict=True) for k,v in g.items()}))
    control_inputs = {r["candidate_id"]:r for r in read_csv("s1_naip_2019_calibration_inputs")}
    controls = []
    for r in read_csv("s1_slc_2019_reference_burst_plan"):
        if r["qualified_aerial_shadow_reference"]!="True":
            continue
        tank_id = r["candidate_id"]
        reference = control_inputs[tank_id]
        parent,burst = (r[k] for k in ("original_slc_parent_id","cdse_vv_burst_uuid"))
        g = geometry(tank_id,parent,burst,r["cdse_source_file"]) if tank_id in tanks else {}
        low_shift = None
        if g:
            _,roof_distance,_,_ = look(tanks[tank_id],tanks[tank_id]["top_hae_approx_m"]-float(reference["aerial_gap_low_m"]),orbits[parent])
            low_shift = roof_distance-g["rim_slant_range_m"]
        controls.append(dict(candidate_id=tank_id,parent_product_id=parent,cdse_vv_burst_uuid=burst,
            geometry_status="computed" if g else "missing_dated_ellipsoid_reference",aerial_gap_lower_m=float(reference["aerial_gap_low_m"]),
            lower_slant_shift_from_orbit_m=low_shift,lower_slant_shift_at_least_3_5m=low_shift>=3.5 if low_shift is not None else None,
            incidence_minus_grd_annotation_deg=g["rim_incidence_deg"]-float(reference["incidence_angle_deg"]) if g else None,**g))
    groups = defaultdict(list)
    for r in rows:
        groups[r["week_ending"]].append(r)
    weeks = []
    listings = {r["week_ending"]:r for r in read_csv("s1_slc_restituted_orbits_weeks")}
    for r in read_csv("s1_holdout_gap_burst_weeks"):
        group = groups[r["week_ending"]]
        ids = {p["candidate_id"] for p in group}
        if len(ids)!=int(r["roofs_with_full_context"]):
            raise ValueError(f"Orbit coverage and SLC plan differ on {r['week_ending']}")
        listed = len({p["candidate_id"] for p in group if p["both_restituted_orbits_listed_before_cutoff"]})
        if listed != int(listings[r["week_ending"]]["tank_weeks_with_both_asf_resorb_listed_before_cutoff"]):
            raise ValueError(f"Restituted orbit timing differs for {r['week_ending']}")
        weeks.append(dict(week_ending=r["week_ending"],release_date=r["release_date"],orbit_geometry_tank_weeks=len(ids),
            tank_weeks_with_both_resorb_listed_before_cutoff=listed,
            tank_weeks_with_both_orbits_before_cutoff=len({p["candidate_id"] for p in group if p["both_precise_orbit_files_on_aws_before_cutoff"]}),
            tank_weeks_with_both_resorb_before_cutoff=len({p["candidate_id"] for p in group if p["both_restituted_orbit_files_asf_before_cutoff"]})))
    for label,values in (("pairs",rows),("controls",controls),("weeks",weeks)):
        fields = tuple(dict.fromkeys(k for r in values for k in r))
        write_csv(DERIVED/f"{STEM}_{label}.csv",fields,([r.get(k) for k in fields] for r in values))
    deltas = [abs(g["resorb_minus_poeorb_rim_slant_range_m"]) for g in staged.values() if "resorb_minus_poeorb_rim_slant_range_m" in g]
    report = dict(holdout_candidate_pairs=len(rows),holdout_tank_weeks=sum(r["orbit_geometry_tank_weeks"] for r in weeks),
        holdout_weeks=sum(r["orbit_geometry_tank_weeks"]>0 for r in weeks),
        holdout_tank_weeks_with_both_exact_aws_orbits_before_cutoff=sum(r["tank_weeks_with_both_orbits_before_cutoff"] for r in weeks),
        holdout_tank_weeks_with_both_asf_restituted_orbits_before_cutoff=sum(r["tank_weeks_with_both_resorb_before_cutoff"] for r in weeks),
        holdout_weeks_with_both_asf_restituted_orbits_before_cutoff=sum(r["tank_weeks_with_both_resorb_before_cutoff"]>0 for r in weeks),
        unique_endpoint_geometry_comparisons_resorb_vs_poeorb=len(deltas),
        median_absolute_resorb_minus_poeorb_slant_range_m=float(np.median(deltas)) if deltas else None,
        p90_absolute_resorb_minus_poeorb_slant_range_m=float(np.quantile(deltas,.9)) if deltas else None,
        endpoints_outside_cdse_catalog_time=sum(not r[f"{side}_rim_zero_doppler_inside_cdse_catalog_time"] for r in rows for side in SIDES),
        controls=dict(Counter(r["geometry_status"] for r in controls)),
        controls_lower_slant_shift_at_least_3_5m=sum(r["lower_slant_shift_at_least_3_5m"] is True for r in controls),
        median_absolute_control_incidence_difference_deg=float(np.median([abs(r["incidence_minus_grd_annotation_deg"])
            for r in controls if r["geometry_status"]=="computed"])),
        scope="Earth-fixed POEORB positions and velocities provide zero-Doppler slant geometry. This predicts a geometric "
              "displacement scale. Copernicus catalog times are compared as metadata, without treating them as a "
              "per-pixel validity test. SLC annotations and signal are separate inputs.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
