"""Verify native calibrated GRD coverage and geometric scale on B04 gap weeks."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from statistics import median

import numpy as np

from audit_s1_grd_native_inputs import saved_assets, saved_windows
from build_holdout_optical_inputs import indexed
from build_s1_holdout_gap_burst_plan import roofs
from build_s1_naip_calibration_inputs import GEO, interpolate, native_grid
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import ROOT, write
from collect_s1_grd_roof_inputs import STORE

STEM = "s1_grd_holdout_gap"
RANGE_RESOLUTION_M = 20.0


def verify_grids(products):
    manifests = [json.loads((STORE/p["product_name"].removesuffix(".SAFE")/"native_manifest.json").read_bytes())
                 for p in products.values()]
    clocks,assets = saved_assets(manifests)
    grids = {m["product"]["product_id"]:native_grid(ROOT/m["objects"]["annotation/iw-vv.xml"]["local_file"])
             for m in manifests}
    raw = read(f"{STEM}_native_window_index")
    saved_windows(raw)
    calibrated = {r["native_window_file"]:r for r in read(f"{STEM}_sigma0_window_index")}
    if {r["window_file"] for r in raw}!=calibrated.keys():
        raise ValueError("Native and calibrated GRD windows differ")
    for r in calibrated.values():
        path = ROOT/r["calibrated_window_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r["calibrated_window_sha256"]:
            raise ValueError(f"Calibrated radar window hash differs for {path}")
        with np.load(path,allow_pickle=False) as packet:
            shape = (int(r["window_rows"]),int(r["window_cols"]))
            if any(packet[k].shape!=shape for k in ("sigma0","noise_equivalent_sigma0","valid")) or packet["valid"].dtype!=bool:
                raise ValueError(f"Calibrated radar window schema differs for {path}")
    available = {}
    with closing(read_only(SOURCE)) as db:
        centre = {k:(lon,lat) for k,lon,lat in db.execute("SELECT candidate_id,longitude,latitude FROM tank_engineering_registry")}
    for r in raw:
        if r["polarization"]!="VV" or r["centre_pixel_valid"]!="True":
            continue
        k,tank = r["product_id"],r["candidate_id"]
        line,sample = (float(r[c]) for c in ("native_line","native_sample"))
        grid = grids[k]
        if not grid[0][0]<=line<=grid[0][-1] or not grid[1][0]<=sample<=grid[1][-1]:
            continue
        angle,_,lon,lat,_ = interpolate(grid,line,sample)
        residual = GEO.inv(lon,lat,*centre[tank])[2]
        if not (20<angle<60 and math.isfinite(residual)):
            raise ValueError(f"Invalid native viewing geometry at {k}/{tank}")
        window = calibrated[r["window_file"]]
        available[k,tank] = (float(angle),float(residual),r,window)
    return clocks,assets,available


def main():
    products = {r["product_id"]:r for r in read(f"{STEM}_manifest")}
    clocks,assets,available = verify_grids(products)
    gap_weeks = {w:r for w,r in indexed("holdout_optical_weeks").items() if r["b04_status"]!="measured"}
    dated = roofs()
    rows,weekly = [],[]
    for r in read("s1_cushing_weekly_catalog_coverage"):
        week = r["week_ending"]
        if r["study_phase"]!="holdout" or week not in gap_weeks:
            continue
        cutoff = datetime.fromisoformat(gap_weeks[week]["cutoff_utc"])
        groups = [[products[i] for i in json.loads(r[f"grd_{s}_product_ids_json"]) if i in products] for s in ("previous","current")]
        pairs = [(a,b) for a in groups[0] for b in groups[1] if (a["relative_orbit"],a["direction"])==(b["relative_orbit"],b["direction"])]
        if bool(pairs)!=(r["grd_common_relative_track"]=="True"):
            raise ValueError(f"GRD manifest does not reproduce the catalog on {week}")
        hits = defaultdict(set)
        for a,b in pairs:
            ids = a["product_id"],b["product_id"]
            ready = max(clocks[i] for i in ids)
            for tank,_,_,_,shell in dated[week]:
                name = tank["candidate_id"]
                if not all((i,name) in available for i in ids):
                    continue
                before,after = (available[i,name] for i in ids)
                # Ground-range layover is bounded by the full dated wall height.
                layovers = [shell/math.tan(math.radians(x[0])) for x in (before,after)]
                can_span = min(layovers)>=RANGE_RESOLUTION_M
                hits["valid"].add(name)
                if ready<cutoff:
                    hits["causal"].add(name)
                if can_span:
                    hits["scale"].add(name)
                rows.append(dict(week_ending=week,release_date=tank["release_date"],candidate_id=name,
                    terminal_id=tank["terminal_id"],forecast_cutoff_utc=cutoff.isoformat(),exact_assets_ready_at_utc=ready.isoformat(),
                    exact_asset_version_pre_cutoff=ready<cutoff,rim_to_ground_m=shell,nominal_grd_range_resolution_m=RANGE_RESOLUTION_M,
                    full_shell_layover_at_least_range_resolution_in_both_images=can_span,
                    **{f"{s}_{k}":v for s,t,l,i in zip(("previous","current"),(before,after),layovers,ids,strict=True)
                       for k,v in (("product_id",i),("incidence_angle_deg",t[0]),("native_geolocation_residual_m",t[1]),
                                   ("full_shell_layover_bound_m",l),("native_window_file",t[2]["window_file"]),
                                   ("calibrated_window_file",t[3]["calibrated_window_file"]))}))
        weekly.append(dict(week_ending=week,release_date=gap_weeks[week]["release_date"],same_orbit_grd_pairs=len(pairs),
            dated_roofs_with_valid_vv_pair=len(hits["valid"]),dated_roofs_with_exact_versions_pre_cutoff=len(hits["causal"]),
            dated_roofs_with_maximum_layover_above_nominal_resolution=len(hits["scale"])))
    report = dict(products=len(products),source_metadata_files_verified=assets,
        catalog_weeks=sum(bool(r["same_orbit_grd_pairs"]) for r in weekly),
        weeks_with_valid_vv_pair=sum(bool(r["dated_roofs_with_valid_vv_pair"]) for r in weekly),
        tank_weeks_with_valid_vv_pair=sum(r["dated_roofs_with_valid_vv_pair"] for r in weekly),
        weeks_with_exact_versions_pre_cutoff=sum(bool(r["dated_roofs_with_exact_versions_pre_cutoff"]) for r in weekly),
        tank_weeks_with_exact_versions_pre_cutoff=sum(r["dated_roofs_with_exact_versions_pre_cutoff"] for r in weekly),
        tank_weeks_with_maximum_layover_above_nominal_resolution=sum(r["dated_roofs_with_maximum_layover_above_nominal_resolution"] for r in weekly),
        nominal_range_resolution_m=RANGE_RESOLUTION_M,
        median_full_shell_layover_bound_m=median(min(r[f"{s}_full_shell_layover_bound_m"] for s in ("previous","current")) for r in rows),
        original_product_platforms=dict(Counter(p["satellite"] for p in products.values())),
        interpretation="This is a coverage and theoretical full-shell layover bound. Actual roof displacement can be smaller. "
        "Exact AWS asset timestamps are tested separately from the original product availability. No roof volume estimate is admitted.")
    export(f"{STEM}_coverage_weeks",weekly)
    export(f"{STEM}_tank_pairs",rows)
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_coverage_summary.json",payload.encode())
    print(payload,end="")


if __name__=="__main__":
    main()
