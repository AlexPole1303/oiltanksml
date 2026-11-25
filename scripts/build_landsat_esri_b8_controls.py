"""Use the USGS angles and QA with Esri B8 reflectance for retrospective roof checks."""

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from statistics import median

from rasterio.io import MemoryFile

from audit_shadow_precision import describe
from build_cross_sensor_shadow import pan_psf, published_lsf
from build_landsat_c2_b8_measurements import RESOLVED, control_disagreements, fit_native_crop, sentinel_controls
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, read_only
from collect_landsat_esri_b8 import STEM as SOURCE_STEM, inputs, raster_request
from collect_public import ROOT, write

STEM = "landsat_esri_b8_controls"


def fit_tile(tile, windows, scene, stage, lsf, database):
    r0,r1,c0,c1 = [int(tile[k]) for k in ("first_line","end_line_exclusive","first_sample","end_sample_exclusive")]
    raw = (ROOT / tile["local_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != tile["sha256"]:
        raise ValueError(f"Esri TIFF changed for {tile['local_path']}")
    _,expected = raster_request(scene,int(tile["object_id"]),(r0,r1,c0,c1))
    kernel = pan_psf(lsf,float(scene["along_track_grid_bearing_deg"]))
    with closing(read_only(database)) as db, MemoryFile(raw) as mem, mem.open() as raster:
        db.row_factory = sqlite3.Row
        if raster.transform != expected or raster.shape != (r1-r0,c1-c0):
            raise ValueError(f"Esri pixel window no longer matches B8 for {tile['local_path']}")
        rows = {r["candidate_id"]:r for r in db.execute("SELECT * FROM roof_window JOIN scene USING(product_id) WHERE product_id=?",
                                                        (scene["product_id"],))}
        return [dict(fit_native_crop(raster,db,rows[r["candidate_id"]],(r0,c0),kernel,stage["b8_saturation_flag_in_scene_mtl"],
                                    apparent_reflectance_scale=float(tile["reflectance_scale"])),
                     product_id=scene["product_id"],terminal_id=rows[r["candidate_id"]]["terminal_id"],platform=scene["platform"],
                     geometry_basis=rows[r["candidate_id"]]["geometry_basis"],psf_basis=rows[r["candidate_id"]]["psf_basis"],
                     source_type="esri_apparent_reflectance",local_path=tile["local_path"],raster_sha256=tile["sha256"],
                     usgs_original_verified=False,historical_b8_archive_availability_verified=False) for r in windows]


def laser_controls(fits, targets):
    ledger = read("cushing_icesat2_b8_laser_comparisons")
    if len(ledger)!=len(targets) or {(r["product_id"],r["candidate_id"]) for r in ledger}!=targets.keys():
        raise ValueError("Laser comparison ledger differs from the dated optical targets")
    for r in ledger:
        key = r["product_id"],r["candidate_id"]
        fit = fits.get(key,{})
        if fit and (fit["terminal_id"]!=r["terminal_id"] or fit["diameter_m"]!=float(r["diameter_m"])
                    or fit["roof_type"]!="EFR"):
            raise ValueError(f"Esri fit does not match the ICESat-2 control geometry for {key}")
        gap = fit.get("estimated_gap_m")
        error = gap-float(r["icesat_gap_m"]) if fit.get("fit_status")==RESOLVED else None
        fields = ("product_id","candidate_id","terminal_id","landsat_capture_at_utc","icesat_granule","icesat_cmr_id",
            "icesat_granule_start_utc","icesat_granule_end_utc","maximum_capture_separation_hours","icesat_photon_csv",
            "diameter_m","liquid_volume_equivalent_bbl_per_m","icesat_gap_m")
        yield {k:r[k] for k in fields} | dict(esri_fit_status=fit.get("fit_status","esri_source_missing"),
            esri_gap_m=gap,esri_gap_min_m=fit.get("near_optimal_gap_min_m"),esri_gap_max_m=fit.get("near_optimal_gap_max_m"),
            esri_minus_icesat_gap_m=error,esri_minus_icesat_liquid_volume_equivalent_bbl=
            -error*float(r["liquid_volume_equivalent_bbl_per_m"]) if error is not None else None,
            esri_tiff=fit.get("local_path"),esri_sha256=fit.get("raster_sha256"))


def pairs(fits, overlap):
    context_keys = ("week_or_control_day","candidate_id","earlier_version_id","later_version_id","earlier_source","later_source")
    context = [r for r in read("landsat_c2_context_registration_pairs") if r["study_phase"]=="holdout"]
    registration = {tuple(r[k] for k in context_keys):r for r in context}
    source = read("landsat_c2_b8_weekly_pairs")
    if len(registration)!=len(context) or len(context)!=len(source):
        raise ValueError("Different B8 weekly pairs and independent red-context offsets")
    out = []
    for r in source:
        keys = [(r[f"{s}_version_id"],r["candidate_id"]) for s in ("previous","current")]
        esri = [fits.get(k) if r[f"{s}_sensor"]=="B8" else None for k,s in zip(keys,("previous","current"),strict=True)]
        obtained = all(x is not None or r[f"{side}_sensor"]!="B8" for x,side in zip(esri,("previous","current"),strict=True))
        consistent = obtained and all(overlap[k[0]]=="shared_pixels_match" for k,x in zip(keys,esri,strict=True) if x is not None)
        status = [a["fit_status"] if a is not None else "esri_source_missing" if r[f"{s}_sensor"]=="B8" else r[f"{s}_fit_status"]
                  for a,s in zip(esri,("previous","current"),strict=True)]
        values = [a.get("estimated_gap_m") if a is not None else float(r[f"{s}_gap_m"]) if r[f"{s}_sensor"]!="B8" and r[f"{s}_gap_m"] else None
                  for a,s in zip(esri,("previous","current"),strict=True)]
        reg = registration[(r["week_ending"],r["candidate_id"],keys[0][0],keys[1][0],r["previous_source_file"],r["current_source_file"])]
        if reg["release_date"]!=r["release_date"] or reg["red_context_before_cutoff"] not in ("True","False"):
            raise ValueError(f"Different B8 and red-context release evidence for {keys}")
        out.append({k:r[k] for k in ("week_ending","release_date","forecast_cutoff_utc","candidate_id","route","area_m2",
            "previous_version_id","current_version_id","previous_capture_at_utc","current_capture_at_utc",
            "previous_source_file","current_source_file")} |
            dict(esri_all_required_windows=obtained,esri_inputs_usable_for_retrospective_fit=consistent,
                 previous_fit_status=status[0],current_fit_status=status[1],
                 both_fits_resolved=consistent and all(x==RESOLVED for x in status),previous_gap_m=values[0],current_gap_m=values[1],
                 context_registration_status=reg["registration_status"],
                 checked_red_context_before_cutoff=reg["registration_status"]=="checked_offset" and reg["red_context_before_cutoff"]=="True",
                 archive_before_cutoff_verified=False))
    return out


def weeks(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["week_ending"]].append(r)
    for r in read("landsat_c2_b8_weekly_weeks"):
        group = grouped.pop(r["week_ending"],[])
        identity = ("week_ending","release_date","forecast_cutoff_utc")
        if (len(group)!=int(r["exact_pair_candidates"]) or len({p["candidate_id"] for p in group})!=int(r["candidate_tanks"])
                or any(tuple(p[k] for k in identity)!=tuple(r[k] for k in identity) for p in group)):
            raise ValueError(f"Esri candidate pairs differ from the B8 weekly ledger for {r['week_ending']}")
        staged = [p for p in group if p["esri_all_required_windows"]]
        stable = [p for p in staged if p["esri_inputs_usable_for_retrospective_fit"]]
        resolved = [p for p in stable if p["both_fits_resolved"]]
        checked = [p for p in resolved if p["checked_red_context_before_cutoff"]]
        yield {k:r[k] for k in (*identity,"b04_status","exact_pair_candidates","candidate_tanks")} | dict(
            pairs_with_all_required_windows=len(staged),tanks_with_all_required_windows=len({p["candidate_id"] for p in staged}),
            pairs_with_usable_esri_inputs=len(stable),tanks_with_usable_esri_inputs=len({p["candidate_id"] for p in stable}),
            pairs_with_both_fits_resolved=len(resolved),tanks_with_both_fits_resolved=len({p["candidate_id"] for p in resolved}),
            pairs_resolved_with_checked_red_context_before_cutoff=len(checked),
            tanks_resolved_with_checked_red_context_before_cutoff=len({p["candidate_id"] for p in checked}),
            historical_pairs_verified=0)
    if grouped:
        raise ValueError("Unexpected Esri candidate weeks")


def main():
    stages,products,databases,targets = inputs()
    overlap = {r["product_id"]:r["overlap_status"] for r in read("landsat_esri_b8_registration_sources")}
    by_tile = defaultdict(list)
    for r in read(f"{SOURCE_STEM}_windows"):
        by_tile[r["product_id"],r["block_row"],r["block_col"]].append(r)
    lsf = published_lsf()
    tiles = read(f"{SOURCE_STEM}_tiles")
    jobs = [(r,by_tile[r["product_id"],r["block_row"],r["block_col"]],products[r["product_id"]],stages[r["product_id"]],
             lsf,databases[r["product_id"]]) for r in tiles]
    fits = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,group in enumerate(pool.map(lambda job:fit_tile(*job),jobs),1):
            for r in group:
                key = r["product_id"],r["candidate_id"]
                if key in fits:
                    raise ValueError(f"Duplicate Esri B8 roof fit for {key}")
                r["esri_exterior_overlap_status"] = overlap.get(r["product_id"],"not_checked")
                fits[key] = r
            if i%30==0:
                print(f"Esri shadow fits processed {i}/{len(jobs)} windows of imagery",flush=True)
    controls = list(control_disagreements(fits))
    laser = list(laser_controls(fits,targets))
    s2 = list(sentinel_controls(stages,fits))
    weekly = pairs(fits,overlap)
    calendar = list(weeks(weekly))
    passed = [r for r in weekly if r["both_fits_resolved"]]
    ready = [r for r in weekly if r["esri_all_required_windows"]]
    matched = [r for r in controls if r["b8_minus_reference_gap_m"] is not None]
    by_group = defaultdict(list)
    for r in matched:
        by_group[r["reference_type"]].append(r)
    close_s2 = [r for r in s2 if r["b8_minus_b04_gap_m"] is not None]
    rows = list(fits.values())
    fields = tuple(dict.fromkeys(k for r in rows for k in r))
    export(f"{STEM}_fits",[{k:r.get(k) for k in fields} for r in rows])
    export(f"{STEM}_external",controls)
    export(f"{STEM}_icesat2",laser)
    export(f"{STEM}_sentinel2",s2)
    export(f"{STEM}_candidate_pairs",weekly)
    export(f"{STEM}_weeks",calendar)
    summary = dict(roof_fits=len(fits), roof_fit_status=Counter(r["fit_status"] for r in rows),
        roof_fits_by_exterior_overlap_status=Counter(r["esri_exterior_overlap_status"] for r in rows),
        external_control_matches=len(controls), external_resolved_comparisons=len(matched),
        external_by_reference={kind:dict(rows=len(g),unique_roofs=len({r["candidate_id"] for r in g}),
            difference_m=describe([r["b8_minus_reference_gap_m"] for r in g])) for kind,g in by_group.items()},
        icesat2_control_targets=len(laser),icesat2_fit_status=Counter(r["esri_fit_status"] for r in laser),
        icesat2_difference_m=describe([r["esri_minus_icesat_gap_m"] for r in laser if r["esri_minus_icesat_gap_m"] is not None]),
        resolved_sentinel2_comparisons=len(close_s2), sentinel2_median_absolute_difference_m=median(abs(r["b8_minus_b04_gap_m"])
            for r in close_s2) if close_s2 else None, candidate_pairs_with_all_required_windows=len(ready),
        candidate_weeks_with_all_required_windows=len({r["week_ending"] for r in ready}),
        candidate_pairs_with_usable_esri_inputs=sum(r["pairs_with_usable_esri_inputs"] for r in calendar),
        candidate_tank_weeks_with_usable_esri_inputs=sum(r["tanks_with_usable_esri_inputs"] for r in calendar),
        candidate_weeks_with_usable_esri_inputs=sum(r["tanks_with_usable_esri_inputs"]>0 for r in calendar),
        candidate_pairs_with_both_fits_resolved=len(passed), candidate_tank_weeks_with_both_fits_resolved=len({
            (r["week_ending"],r["candidate_id"]) for r in passed}), candidate_weeks_with_both_fits_resolved=len({r["week_ending"] for r in passed}),
        gap_weeks_in_ledger=len(calendar),candidate_pairs_resolved_with_checked_red_context_before_cutoff=sum(
            r["pairs_resolved_with_checked_red_context_before_cutoff"] for r in calendar),
        candidate_tank_weeks_resolved_with_checked_red_context_before_cutoff=sum(
            r["tanks_resolved_with_checked_red_context_before_cutoff"] for r in calendar),
        candidate_weeks_resolved_with_checked_red_context_before_cutoff=sum(
            r["tanks_resolved_with_checked_red_context_before_cutoff"]>0 for r in calendar),
        admitted_historical_esri_pairs=0,
        scope="Retrospective use of Esri apparent reflectance on the USGS grid. The same source angle and QA records and physical shadow fit are used. "
        "Historical availability of the Esri pixels and weekly cross-date registration are not established by this experiment.")
    payload=json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
