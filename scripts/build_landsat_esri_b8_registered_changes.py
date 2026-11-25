"""Refit exact B8/B04 pairs with checked exterior offsets and retain offset sensitivity."""

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from itertools import product
from statistics import median

import numpy as np
from rasterio.io import MemoryFile

from audit_landsat_esri_b8_registration import PAIR_ID
from build_cross_sensor_shadow import PAN_PIXEL, PAN_STEP, pan_grid, pan_psf, published_lsf
from build_landsat_c2_b8_measurements import NATIVE, native_pan_samples
from build_registered_shadow_development import registered_pair
from build_s1_reference_matches import export, read
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_panel import DERIVED, RESOLVED, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_landsat_esri_b8 import inputs as landsat_inputs
from collect_public import ROOT, write
from fit_shadow_geometry import (PIXEL_M, STEP_M, bearing_vector, gap_trials, inputs, pixel_grid,
                                 response_profiles)

STEM = "landsat_esri_b8_registered_changes"
SIDES = ("previous","current")
REGISTERED_ID = ("week_or_control_day","candidate_id","earlier_version_id","later_version_id","earlier_source","later_source")
CENTRES = ((0,0),*(p for p in product((-6,0,6),repeat=2) if any(p)))
STRESS_DIRECTIONS = tuple((round(math.cos(k*math.pi/4),12),round(math.sin(k*math.pi/4),12)) for k in range(8))
B04_PACKETS = {"weekly_shadow_holdout_observations.csv":"weekly_shadow_holdout.sqlite",
               "shadow_timing_holdout.sqlite":"shadow_timing_holdout.sqlite"}


def indexed(stem, fields):
    rows = read(stem)
    index = {tuple(r[k] for k in fields):r for r in rows}
    if len(index)!=len(rows):
        raise ValueError(f"Duplicate source identity in {stem}")
    return index


def jobs():
    candidates = indexed("landsat_c2_b8_weekly_pairs",PAIR_ID)
    fits = indexed("landsat_esri_b8_controls_candidate_pairs",PAIR_ID)
    if candidates.keys()!=fits.keys():
        raise ValueError("Candidate and Esri roof ledgers disagree")
    for row in indexed("landsat_esri_b8_registration_pairs",REGISTERED_ID).values():
        if row["registration_status"]!="checked_offset":
            continue
        key = tuple(row[k] for k in REGISTERED_ID)
        base,fit = candidates[key],fits[key]
        if ((base["release_date"],base["forecast_cutoff_utc"],base["route"],base["terminal_id"]) !=
            (row["release_date"],row["forecast_cutoff_utc"],row["comparison"],row["terminal_id"])
            or fit["both_fits_resolved"]!=row["both_roof_fits_resolved"]
            or fit["esri_inputs_usable_for_retrospective_fit"]!="True"):
            raise ValueError(f"Refit uses different source evidence for {key}")
        yield row,base,fit


def identity(base, side):
    return (base[f"{side}_sensor"],base[f"{side}_observation_source"],int(base[f"{side}_observation_id"]))


def prepare(work):
    stages,products,_,_ = landsat_inputs()
    tiles = {r["local_path"]:r for r in read("landsat_esri_b8_tiles")}
    pan_fits = indexed("landsat_esri_b8_controls_fits",("product_id","candidate_id"))
    dated = legacy_geometry()
    needed = {identity(base,s) for _,base,_ in work for s in SIDES}
    pan_rows = {}
    with closing(read_only(NATIVE)) as db:
        db.row_factory = sqlite3.Row
        for r in db.execute("SELECT * FROM roof_window JOIN scene USING(product_id)"):
            key = "B8",NATIVE.name,r["pan_observation_id"]
            if key in needed:
                pan_rows[key] = dict(r)
    sensor_rows,kernels = {},{}
    for source,packet in B04_PACKETS.items():
        path = DERIVED/packet
        with closing(read_only(path)) as db:
            db.row_factory = sqlite3.Row
            rows, kernels[source] = inputs(db)
            scenes = {r["scene_id"]:r for r in db.execute("SELECT * FROM scene WHERE level='L1C'")}
        for r in rows:
            key = "B04",source,r["observation_id"]
            if key not in needed:
                continue
            scene = scenes[r["scene_id"]]
            if scene["crs"]!="EPSG:32614":
                raise ValueError(f"Unexpected Sentinel-2 grid for {key}")
            sensor_rows[key] = (dict(r),scene)
    if pan_rows.keys() | sensor_rows.keys() != needed:
        raise ValueError("Refit sources are missing exact observations")
    lsf = published_lsf()
    pan_kernels = {p:pan_psf(lsf,float(products[p]["along_track_grid_bearing_deg"]))
                   for p in {r["product_id"] for r in pan_rows.values()}}

    @lru_cache(maxsize=160)
    def tile_bytes(path):
        raw = (ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=tiles[path]["sha256"]:
            raise ValueError(f"B8 tile checksum changed for {path}")
        return raw

    @lru_cache(maxsize=len(needed))
    def chip(key):
        band,source,_ = key
        if band=="B8":
            r = pan_rows[key]
            if r["crs"]!="EPSG:32614" or r["geometry_basis"]!="dated_2011_lidar":
                raise ValueError(f"Unexpected B8 geometry for {key}")
            version,tank = r["product_id"],r["candidate_id"]
            prior = pan_fits[version,tank]
            tile = tiles[prior["local_path"]]
            if (prior["raster_sha256"]!=tile["sha256"] or prior["esri_exterior_overlap_status"]!="shared_pixels_match"
                    or int(prior["observation_id"])!=r["pan_observation_id"]):
                raise ValueError(f"B8 provenance changed for {key}")
            origin = int(tile["first_line"]),int(tile["first_sample"])
            with (closing(read_only(NATIVE)) as db,MemoryFile(tile_bytes(tile["local_path"])) as mem,mem.open() as raster):
                samples,problem = native_pan_samples(raster,db,r,origin,stages[version]["b8_saturation_flag_in_scene_mtl"],
                    apparent_reflectance_scale=float(tile["reflectance_scale"]))
            grid,_ = pan_grid(r,samples)
            north = r["true_north_grid_bearing_deg"]
            sun = bearing_vector(r["solar_zenith_deg"],r["solar_azimuth_true_deg"]+north)
            view = bearing_vector(r["view_zenith_deg"],r["view_azimuth_true_deg"]+north)
            kernel,spacing,step = pan_kernels[version],PAN_PIXEL,PAN_STEP
            raster_sha = tile["sha256"]
            if problem or len(grid[-1])!=int(prior["fitted_pixels"]):
                raise ValueError(f"B8 refit changes the screened pixels for {key}")
        else:
            r,_ = sensor_rows[key]
            tank,version = r["candidate_id"],r["version_id"]
            old = dated[tank]
            r = dict(r,center_east_m=float(old["center_east_m"]),center_north_m=float(old["center_north_m"]),
                     radius_m=float(old["diameter_m"])/2,shell_above_ground_m=
                     float(old["prior_rim_z_p95_native"])-float(old["prior_ground_z_p50_native"]))
            with closing(read_only(DERIVED/B04_PACKETS[source])) as db:
                grid = pixel_grid(db,r)
            sun = bearing_vector(r["sun_zenith_deg"],r["sun_azimuth_grid_deg"])
            view = bearing_vector(r["red_view_zenith_deg"],r["red_view_azimuth_grid_deg"])
            kernel,spacing,step = kernels[source][r["platform"]],PIXEL_M,STEP_M
            raster_sha = None
        old = dated[tank]
        if (r["roof_type"]!="EFR" or len(grid[-1])<12 or
            not np.allclose([r["center_east_m"],r["center_north_m"],r["radius_m"]],
                [float(old["center_east_m"]),float(old["center_north_m"]),float(old["diameter_m"])/2],rtol=0,atol=1e-8)):
            raise ValueError(f"Native input lacks the dated tank geometry for {key}")
        return r,grid,sun,view,kernel,spacing,step,raster_sha
    return chip


def profiles(chip, relative_offsets):
    row,grid,sun,view,kernel,spacing,step,_ = chip
    shell = row["shell_above_ground_m"]
    gaps = gap_trials(shell,row["roof_type"])
    offsets = tuple((x+e,y+n) for e,n in relative_offsets for x,y in CENTRES)
    unique_offsets = tuple(dict.fromkeys(offsets))
    coords,loss,_,placebo = response_profiles(grid,row["radius_m"],shell,gaps,sun,view,kernel,
        reversed_sun=True,pixel_m=spacing,step_m=step,centre_offsets=unique_offsets)
    lookup = {xy:i for i,xy in enumerate(coords)}
    count = len(grid[-1])
    weight = float(grid[-1].sum())
    for start in range(0,len(offsets),len(CENTRES)):
        indexes = [lookup[xy] for xy in offsets[start:start+len(CENTRES)]]
        yield CENTRES,gaps,loss[indexes],placebo[indexes],count,weight


def direction(lo,hi):
    return "increase" if lo>0 else "decrease" if hi<0 else "includes_zero"


def scenarios(offset):
    east,north = (float(offset[f"later_minus_earlier_{axis}_m"]) for axis in ("east","north"))
    radius = max(2.0,float(offset["check_median_residual_m"]),float(offset["fit_median_residual_m"]))
    return radius,((0.0,0.0),(east,north),*((east+radius*dx,north+radius*dy) for dx,dy in STRESS_DIRECTIONS))


def matched_refits(chips,settings):
    left,right = (profiles(data,tuple((e*sign/2,n*sign/2) for e,n in settings))
                  for sign,data in zip((-1,1),chips,strict=True))
    return [registered_pair(first,last) for first,last in zip(left,right,strict=True)]


def solve(item, chip):
    offset,base,baseline = item
    chips = [chip(identity(base,s)) for s in SIDES]
    radius,settings = scenarios(offset)
    for side,data in zip(SIDES,chips,strict=True):
        native = data[0]
        version = native.get("product_id",native.get("version_id"))
        if (version!=base[f"{side}_version_id"] or native["candidate_id"]!=base["candidate_id"]
            or datetime.fromisoformat(native["captured_at_utc"])!=datetime.fromisoformat(base[f"{side}_capture_at_utc"])
            or not math.isclose(2*native["radius_m"],float(base[f"{side}_diameter_m"]),rel_tol=0,abs_tol=1e-8)):
            raise ValueError(f"Native fit identity changed for {identity(base,side)}")
    results = matched_refits(chips,settings)
    zero,central,*stress = results
    envelope = min(x["registered_loss_range_min_m"] for x in results[1:]),max(x["registered_loss_range_max_m"] for x in results[1:])
    all_pass = all(x["registered_both_shadow_diagnostics_pass"] for x in results[1:])
    scale = math.pi*chips[0][0]["radius_m"]**2/BARREL_M3
    separate = (float(baseline["previous_gap_m"])-float(baseline["current_gap_m"])
                if baseline["both_fits_resolved"]=="True" else None)
    fields = (*PAIR_ID,"release_date","forecast_cutoff_utc","route","terminal_id","area_m2","capture_interval_hours",
        "current_hours_before_eia_measurement","previous_capture_at_utc","current_capture_at_utc","previous_observation_source",
        "previous_observation_id","current_observation_source","current_observation_id")
    pair = {k:base[k] for k in fields} | {k:offset[k] for k in ("scene_pair_id","later_minus_earlier_east_m",
        "later_minus_earlier_north_m","fit_median_residual_m","check_median_residual_m","cdse_source_bundle_clocks_before_cutoff",
        "hls_exact_scene_lineage_before_cutoff","usgs_original_b8_files_before_cutoff_confirmed","esri_pixels_before_cutoff_verified")}
    return pair | dict(previous_b8_sha256=chips[0][-1],current_b8_sha256=chips[1][-1],
        separate_both_resolved=separate is not None,separate_change_m=separate,
        common_centre_unregistered_change_m=zero["registered_change_m"],**central,
        offset_stress_radius_m=radius,offset_stress_count=len(stress),
        stress_min_m=envelope[0],stress_max_m=envelope[1],all_offset_scenarios_pass=all_pass,
        robust_direction=direction(*envelope) if all_pass else "failed_shadow_diagnostic",
        geometric_change_bbl=scale*central["registered_change_m"],
        geometric_stress_min_bbl=scale*envelope[0],geometric_stress_max_bbl=scale*envelope[1],
        eligible_for_historical_replay=False)


def summarize(pairs):
    calendar = indexed("landsat_esri_b8_registration_weeks",("week_ending",))
    by_tank = defaultdict(list)
    for row in pairs:
        by_tank[row["week_ending"],row["candidate_id"]].append(row)
    selected,by_week = [],defaultdict(list)
    for (week,tank),group in sorted(by_tank.items()):
        central = [r for r in group if r["registered_both_shadow_diagnostics_pass"]]
        stable = [r for r in central if r["all_offset_scenarios_pass"]]
        # A single representative is chosen by capture recency and exterior control error only.
        representative = min(central,key=lambda r:(float(r["current_hours_before_eia_measurement"]),
            float(r["check_median_residual_m"]),float(r["capture_interval_hours"]),r["previous_version_id"],r["current_version_id"])) if central else {}
        lo = min((r["stress_min_m"] for r in central),default=None)
        hi = max((r["stress_max_m"] for r in central),default=None)
        if len(stable)==len(central) and stable:
            status = direction(lo,hi)
        else:
            status = "offset_sensitive_diagnostic" if central else "no_resolved_pair"
        if not math.isclose(float(group[0]["area_m2"]),float(group[-1]["area_m2"]),rel_tol=1e-12):
            raise ValueError(f"Tank area changed across exact pair versions for {week}/{tank}")
        row = dict(week_ending=week,candidate_id=tank,release_date=group[0]["release_date"],
            terminal_id=group[0]["terminal_id"],area_m2=group[0]["area_m2"],checked_pair_options=len(group),
            jointly_resolved_pair_options=len(central),all_offset_diagnostics_pass_pair_options=len(stable),
            all_resolved_pair_options_min_m=lo,all_resolved_pair_options_max_m=hi,direction_across_resolved_options=status,
            representative_previous_version=representative.get("previous_version_id"),
            representative_current_version=representative.get("current_version_id"),
            representative_change_m=representative.get("registered_change_m"),
            representative_geometric_change_bbl=representative.get("geometric_change_bbl"),eligible_for_historical_replay=False)
        selected.append(row)
        by_week[week].append(row)
    weeks = []
    for (week,),meta in calendar.items():
        group = by_week.pop(week,[])
        if sum(r["checked_pair_options"] for r in group)!=int(meta["pairs_with_checked_exterior_offsets"]):
            raise ValueError(f"Refit omitted checked exterior pairs on {week}")
        counts = Counter(r["direction_across_resolved_options"] for r in group)
        weeks.append({k:meta[k] for k in ("week_ending","release_date","forecast_cutoff_utc","b04_status")} |
            dict(tanks_with_checked_exterior_offsets=len(group),tanks_with_jointly_resolved_pair=sum(
                r["jointly_resolved_pair_options"]>0 for r in group),tanks_with_positive_stable_direction=counts["increase"],
                tanks_with_negative_stable_direction=counts["decrease"],tanks_with_bounded_change_including_zero=counts["includes_zero"],
                tanks_with_offset_sensitive_diagnostic=counts["offset_sensitive_diagnostic"],
                historically_admitted_tanks=0))
    if by_week:
        raise ValueError("Unexpected refit week")
    return selected,weeks


def main():
    work = list(jobs())
    chip = prepare(work)
    pairs = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i,row in enumerate(pool.map(lambda item:solve(item,chip),work),1):
            pairs.append(row)
            if i%100==0 or i==len(work):
                print(f"Replayed {i}/{len(work)} exterior-checked shadow pairs",flush=True)
    tank_weeks,weeks = summarize(pairs)
    baseline = [r for r in pairs if r["separate_both_resolved"]]
    pass_joint = [r for r in pairs if r["registered_both_shadow_diagnostics_pass"]]
    common = [r for r in baseline if r["registered_both_shadow_diagnostics_pass"]]
    changes = [abs(r["registered_change_m"]-r["common_centre_unregistered_change_m"]) for r in common]
    directions = Counter(r["direction_across_resolved_options"] for r in tank_weeks)
    summary = dict(checked_exact_pairs=len(pairs),initial_separate_resolved_pairs=len(baseline),jointly_resolved_pairs=len(pass_joint),
        resolved_with_and_without_joint_fit=len(common),newly_resolved_after_joint_fit=len(pass_joint)-len(common),
        no_longer_resolved_after_joint_fit=len(baseline)-len(common),
        common_resolved_median_abs_effect_of_offset_m=median(changes) if changes else None,
        common_resolved_p90_abs_effect_of_offset_m=float(np.percentile(changes,90)) if changes else None,
        pair_stress_direction=Counter(r["robust_direction"] for r in pairs),tank_week_direction=directions,
        tank_weeks_with_joint_fit=sum(r["jointly_resolved_pair_options"]>0 for r in tank_weeks),
        weeks_with_joint_fit=sum(r["tanks_with_jointly_resolved_pair"]>0 for r in weeks),
        weeks_with_stable_direction=sum(r["tanks_with_positive_stable_direction"]+r["tanks_with_negative_stable_direction"]>0 for r in weeks),
        historically_admitted_pairs=0,scope="Retrospective geometric change between actual captures. Both epochs use a common centre "
        "and their original observations, quality masks, angles and sensor response functions. Exterior offsets are applied symmetrically "
        "to the continuous model, with eight perturbations at the larger of 2 m and the exterior fit/check median residuals. "
        "The loss and stress ranges are descriptive and the volume equivalent assumes cylindrical geometry. "
        "No regional inventories, first-publication target values or prices enter this calculation.")
    for label,rows in (("pairs",pairs),("tank_weeks",tank_weeks),("weeks",weeks)):
        export(f"{STEM}_{label}",rows)
    payload = json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
