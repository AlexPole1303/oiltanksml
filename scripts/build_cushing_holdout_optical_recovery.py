"""Audit alternative observations on all reporting weeks without reading EIA outcomes."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache
import json
import math
from statistics import median

from audit_sentinel2_alternate_registration import attach_sources, packet_inputs, registration_products, source_packet
from build_registered_shadow_development import registered_pair, registered_profile
from build_holdout_optical_inputs import indexed
from build_shadow_registration_inputs import SEARCH_RADIUS_M, anchor_grid, tank_points
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from build_weekly_shadow_validation import legacy_geometry
from collect_public import ROOT, write
from fit_shadow_registration import tank_offset

STEM = "cushing_holdout_optical_recovery"
DATED = "dated_2011_lidar"
EDGES = ("previous","current")


def timestamp(row,name):
    return datetime.fromisoformat(row[name])


def observation_book():
    book = {}
    for r in read_csv("shadow_timing_holdout_observations"):
        if r["geometry_mode"]!=DATED or r["fit_status"]!="resolved_in_geometry_diagnostic":
            continue
        key = r["candidate_id"],r["version_id"],r["canonical_l2a_id"]
        if key in book:
            raise ValueError(f"Repeated dated scene observation for {key}")
        book[key] = r
    return book


def equal(actual,expected):
    return math.isclose(float(actual),float(expected),rel_tol=1e-10,abs_tol=1e-9)


def verify_bracket(row,week,observations):
    issue = row["release_date"]
    candidate = row["candidate_id"]
    cutoff = timestamp(week,"cutoff_utc")
    if (issue!=week["release_date"] or timestamp(row,"inputs_available_before_utc")!=cutoff
        or any(not row[field] or row[field]>cutoff.date().isoformat() for field in ("lidar_evidence_date","roof_evidence_date"))):
        raise ValueError(f"Late or conflicting roof evidence on {issue}/{candidate}")
    gaps,sources,diameters = {},set(),set()
    for edge in EDGES:
        at = timestamp(week,f"{edge}_measurement_at_utc")
        if timestamp(row,f"{edge}_measurement_at_utc")!=at:
            raise ValueError(f"Timing measurement boundary differs on {issue}/{candidate}")
        dates,measurements = [],[]
        for side in ("before","after"):
            prefix = f"{edge}_{side}"
            key = candidate,row[f"{prefix}_version_id"],row[f"{prefix}_canonical_l2a_id"]
            item = observations[key]
            captured = timestamp(item,"captured_at_utc")
            if (item["terminal_id"]!=row["terminal_id"] or item["measurement_source"]!=row[f"{prefix}_measurement_source"]
                or not equal(item["estimated_gap_m"],row[f"{prefix}_gap_m"])
                or not captured==timestamp(row,f"{prefix}_captured_at_utc")<=timestamp(item,"archive_available_at_utc")
                   <=timestamp(item,"geometry_available_at_utc")<cutoff
                or any(timestamp(item,f"{field}_available_at_utc")!=timestamp(row,f"{prefix}_{field}_available_at_utc")
                       for field in ("archive","geometry"))
                or not (ROOT/item["tile_metadata_path"]).is_file()):
                raise ValueError(f"Unreconciled original timing observation on {issue}/{candidate}/{prefix}")
            dates.append(captured)
            measurements.append(float(item["estimated_gap_m"]))
            diameters.add(float(item["diameter_m"]))
            sources.add(item["tile_metadata_path"])
        before,after = dates
        if not before<=at<after:
            raise ValueError(f"Reporting boundary outside observations on {issue}/{candidate}")
        ratio = (at-before)/(after-before)
        gaps[edge] = measurements[0]+ratio*(measurements[1]-measurements[0])
        if (not equal(ratio,row[f"{edge}_interpolation_weight"]) or not equal(gaps[edge],row[f"{edge}_interpolated_gap_m"])
            or not equal((after-before).total_seconds()/3600,row[f"{edge}_bracket_hours"])):
            raise ValueError(f"Interpolation arithmetic differs on {issue}/{candidate}")
        gaps[f"{edge}_before"] = measurements[0]
    if len(diameters)!=1 or not equal(math.pi*next(iter(diameters))**2/4,row["circular_area_m2"]):
        raise ValueError(f"Geometry differs on {issue}/{candidate}")
    scale = float(row["circular_area_m2"])/BARREL_M3/1000
    amount = (gaps["previous"]-gaps["current"])*scale
    earlier = (gaps["previous_before"]-gaps["current_before"])*scale
    if not equal(amount,row["interpolated_visible_kbbl"]) or not equal(earlier,row["pre_measurement_pair_visible_kbbl"]):
        raise ValueError(f"Interpolated volume differs on {issue}/{candidate}")
    return amount,earlier,"|".join(sorted(sources))


def recovery():
    baseline = indexed("holdout_optical_weeks")
    timing = {r["week_ending"]:r for r in read_csv("shadow_timing_holdout_weeks") if r["geometry_mode"]==DATED}
    if baseline.keys()!=timing.keys():
        raise ValueError("Optical timing and baseline use different release weeks")
    observations = observation_book()
    brackets = defaultdict(list)
    gap_jobs = {}
    used = set()
    for row in read_csv("shadow_timing_holdout_pairs"):
        if row["geometry_mode"]!=DATED or row["timing_status"]!="both_boundaries_interpolated":
            continue
        week = row["week_ending"]
        key = week,row["candidate_id"]
        if key in used:
            raise ValueError(f"Repeated timing bracket {key}")
        used.add(key)
        base = baseline[week]
        amount,earlier,sources = verify_bracket(row,base,observations)
        if base["b04_status"]!="measured":
            for edge in EDGES:
                gap_jobs[f"{week}/{edge}",row["candidate_id"]] = row,[
                    observations[row["candidate_id"],row[f"{edge}_{side}_version_id"],row[f"{edge}_{side}_canonical_l2a_id"]]
                    for side in ("before","after")]
        brackets[week].append({k:row[k] for k in ("week_ending","release_date","candidate_id","terminal_id","inputs_available_before_utc",
            "lidar_evidence_date","roof_evidence_date","circular_area_m2","previous_bracket_hours","current_bracket_hours",
            *(f"{e}_{s}_{k}" for e in EDGES for s in ("before","after") for k in ("version_id","captured_at_utc","geometry_available_at_utc")))} |
            dict(b04_baseline_week_status=base["b04_status"],interpolated_visible_kbbl=amount,
                 matched_pre_measurement_visible_kbbl=earlier,metadata_source_files=sources))
    coverage = []
    for week,o in sorted(baseline.items()):
        t = timing[week]
        sample = brackets[week]
        amount = math.fsum(r["interpolated_visible_kbbl"] for r in sample)
        earlier = math.fsum(r["matched_pre_measurement_visible_kbbl"] for r in sample)
        if (o["release_date"]!=t["release_date"] or int(t["bracketed_pairs"])!=len(sample)
            or (sample and (not equal(amount,t["interpolated_visible_kbbl"]) or not equal(earlier,t["matched_pre_measurement_visible_kbbl"])))):
            raise ValueError(f"Brackets do not reproduce weekly timing file for {week}")
        coverage.append({k:o[k] for k in ("week_ending","release_date","cutoff_utc","b04_status","baseline_b04_tanks",
            "landsat_15m_paired_tanks","landsat_30m_context_only_tanks","sentinel1_slc_common_relative_track",
            "sentinel1_grd_common_relative_track")} | dict(dated_pre_measurement_pairs=int(t["pre_measurement_pairs"]),
                dated_interpolated_tanks=len(sample),dated_interpolated_kbbl=amount if sample else None,
                dated_matched_pre_measurement_kbbl=earlier if sample else None,
                dated_interpolated_tanks_with_no_b04=len(sample) if o["b04_status"]!="measured" else 0))
    return coverage,[r for rows in brackets.values() for r in rows],gap_jobs


def verify_gap_registrations(jobs, *, radius=SEARCH_RADIUS_M, spectral=False):
    if not jobs:
        return ([],[]) if spectral else []
    sources,links = attach_sources(jobs,"holdout")
    tanks, exclusion = tank_points()
    anchors = anchor_grid({key[1]:tanks[key[1]] for key in jobs},exclusion,search_radius_m=radius)
    windows,_,_,profiles,_ = registration_products(links,sources,anchors)
    offsets = {(r["week_or_control_day"],r["candidate_id"]):tank_offset(r,profiles) for r in windows}
    if offsets.keys()!=jobs.keys():
        raise ValueError("Registration must give a status for every gap-week endpoint")
    geometry = legacy_geometry()
    spectral_jobs = []
    manifests = {}

    @lru_cache(maxsize=128)
    def profile(tank,packet,number,east,north):
        path,rows,_,kernels = packet_inputs(packet)
        old = geometry[tank]
        row = dict(rows[number],center_east_m=float(old["center_east_m"]),center_north_m=float(old["center_north_m"]),
                   radius_m=float(old["diameter_m"])/2)
        shell = float(old["prior_rim_z_p95_native"])-float(old["prior_ground_z_p50_native"])
        return registered_profile(path,row,None,shell,kernels[row["platform"]],east,north)

    output = []
    for key,(r,fits) in sorted(jobs.items()):
        edge = key[0].rsplit("/",1)[1]
        offset = offsets[key]
        accepted = offset["registration_status"]=="checked_offset"
        result = {}
        if accepted:
            east,north = (offset[f"later_minus_earlier_{a}_m"]/2 for a in ("east","north"))
            first,last = (profile(key[1],source_packet(fit["measurement_source"]),int(fit["observation_id"]),sign*east,sign*north)
                          for sign,fit in zip((-1,1),fits,strict=True))
            result = registered_pair(first,last)
            if spectral:
                endpoints = [(source_packet(f["measurement_source"]),int(f["observation_id"])) for f in fits]
                for (packet,number),fit in zip(endpoints,fits,strict=True):
                    _, observations, scenes, _ = packet_inputs(packet)
                    scene = scenes[observations[number]["scene_id"]]
                    manifests[fit["version_id"]] = scene["manifest_path"]
                volume = {k:r[k] for k in ("candidate_id","terminal_id","week_ending","release_date")}
                volume.update(study_phase="holdout",cutoff_utc=r["inputs_available_before_utc"],
                    archive_available_at_utc=max(timestamp(f,"archive_available_at_utc") for f in fits).isoformat(),
                    capture_change_kbbl=float(result["registered_change_m"])*float(r["circular_area_m2"])/BARREL_M3/1000,
                    circular_kbbl_per_m=float(r["circular_area_m2"])/BARREL_M3/1000,
                    baseline_red_resolved=result["registered_both_shadow_diagnostics_pass"], alternate_capture_pair=True)
                for side, fit in zip(("earlier","later"),fits,strict=True):
                    volume.update({f"{side}_sentinel2_version":fit["version_id"],f"{side}_capture_at_utc":fit["captured_at_utc"],
                        f"{side}_gap_m":result[f"{side}_registered_gap_m"]})
                spectral_jobs.append((edge,(volume,endpoints,east,north)))
        resolved = result.get("registered_both_shadow_diagnostics_pass",False)
        gap = None
        if resolved:
            before,after = (result[f"{s}_registered_gap_m"] for s in ("earlier","later"))
            gap = before+float(r[f"{edge}_interpolation_weight"])*(after-before)
        output.append({k:r[k] for k in ("week_ending","release_date","candidate_id","terminal_id","inputs_available_before_utc",
                                       "circular_area_m2")} |
            dict(reporting_edge=edge,before_scene_version=fits[0]["version_id"],after_scene_version=fits[1]["version_id"],
                 registration_status=offset["registration_status"],fit_inliers=offset["fit_inliers"],
                 spatial_check_inliers=offset["spatial_check_inliers"],both_shadow_diagnostics_pass=resolved,
                 before_shadow_status=result.get("earlier_registered_status"),after_shadow_status=result.get("later_registered_status"),
                 independently_registered_gap_m=gap))
    spectral_fits = measure_gap_bands(spectral_jobs,manifests,geometry) if spectral_jobs else []
    return (output,spectral_fits) if spectral else output


def measure_gap_bands(spectral_jobs, manifests, geometry):
    from contextlib import closing
    from hashlib import sha256

    from check_multiband_shadow import spectral_kernels
    from check_weekly_multiband_shadow import measure
    from collect_s2_l1c import source_metadata_bytes
    from collect_s2_spectral_detector_masks import collect, jobs
    from fit_shadow_geometry import CALIBRATION, read_only

    with ThreadPoolExecutor(max_workers=4) as pool:
        source_rows = list(pool.map(collect,jobs(manifests)))
    masks = {}
    for r in source_rows:
        source = source_metadata_bytes(ROOT/r["local_file"])
        if len(source)!=int(r["source_bytes"]) or sha256(source).hexdigest()!=r["source_sha256"]:
            raise ValueError(f"Detector footprint changed for {r['version_id']}/{r['band']}")
        masks[r["version_id"],r["band"]] = r
    with closing(read_only(CALIBRATION)) as db:
        kernels = spectral_kernels(db)
    fits = measure("holdout",geometry,kernels,masks,source_jobs=[job for _,job in spectral_jobs])
    output = [dict(reporting_edge=edge,**r) for (edge,_),r in zip(spectral_jobs,fits,strict=True)]
    export(f"{STEM}_wider_gap_spectral_masks",source_rows)
    export(f"{STEM}_wider_gap_spectral_fits",output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wider-gap-check", action="store_true", help="repeat gap registration at a 1,600 m search radius")
    parser.add_argument("--spectral",action="store_true",help="refit wider-registered pairs with B08 and joint bands")
    args = parser.parse_args()
    if args.spectral and not args.wider_gap_check:
        parser.error("--spectral requires --wider-gap-check")
    weeks,tanks,jobs = recovery()
    if args.wider_gap_check:
        checked = verify_gap_registrations(jobs,radius=1600,spectral=args.spectral)
        registration, spectral_fits = checked if args.spectral else (checked,[])
        by_roof = Counter((r["week_ending"],r["candidate_id"]) for r in registration if r["both_shadow_diagnostics_pass"])
        report = dict(search_radius_m=1600, brackets_checked=len(registration),
            registration_statuses=Counter(r["registration_status"] for r in registration),
            brackets_with_passing_shadow_diagnostics=sum(by_roof.values()),
            tank_weeks_passing_at_both_boundaries=sum(n==2 for n in by_roof.values()),
            weeks_passing_at_both_boundaries=sorted({week for (week,_),n in by_roof.items() if n==2}))
        if args.spectral:
            spectral_by_roof = defaultdict(set)
            for row in registration:
                if row["both_shadow_diagnostics_pass"]:
                    spectral_by_roof[row["release_date"],row["candidate_id"]].add(row["reporting_edge"])
            for row in spectral_fits:
                if row["nir_resolved"] or row["joint_resolved"]:
                    spectral_by_roof[row["release_date"],row["candidate_id"]].add(row["reporting_edge"])
            report["spectral"] = dict(
                refitted_brackets=len(spectral_fits),
                passing_b08_brackets=sum(r["nir_resolved"] for r in spectral_fits),
                passing_joint_brackets=sum(r["joint_resolved"] for r in spectral_fits),
                complete_tank_weeks=sum(edges==set(EDGES) for edges in spectral_by_roof.values()))
        export(f"{STEM}_wider_gap_registration",registration)
        write(DERIVED/f"{STEM}_wider_gap_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
        print(json.dumps(report,indent=2))
        return
    gaps = [r for r in weeks if r["b04_status"]!="measured"]
    extra = [r for r in gaps if r["dated_interpolated_tanks"]]
    additions = [r for r in tanks if r["b04_baseline_week_status"]!="measured"]
    registration = verify_gap_registrations(jobs)
    passed = Counter((r["week_ending"],r["candidate_id"]) for r in registration if r["both_shadow_diagnostics_pass"])
    checked = sorted({week for (week,_),n in passed.items() if n==2})
    spans = [float(r[f"{e}_bracket_hours"]) for r in additions for e in EDGES]
    report = dict(releases=len(weeks),b04_gaps=len(gaps),dated_timing_weeks=sum(bool(r["dated_interpolated_tanks"]) for r in weeks),
        verified_dated_timing_tank_weeks=len(tanks),b04_gaps_with_dated_interpolation=len(extra),
        b04_gaps_with_dated_pre_measurement_pair=sum(r["dated_pre_measurement_pairs"]>0 for r in gaps),
        dated_interpolated_tanks_in_b04_gap_weeks=len(additions),b04_gap_weeks_with_dated_interpolation=[r["week_ending"] for r in extra],
        b04_gap_reasons_with_interpolation=dict(Counter(r["b04_status"] for r in extra)),
        interpolated_gap_week_bracket_span_hours=dict(minimum=min(spans),median=median(spans),maximum=max(spans)) if spans else None,
        b04_gap_weeks_with_landsat_15m_pairs=sum(int(r["landsat_15m_paired_tanks"])>0 for r in gaps),
        b04_gap_weeks_with_landsat_30m_context=sum(int(r["landsat_30m_context_only_tanks"])>0 for r in gaps),
        b04_gap_weeks_with_sentinel1_slc_catalog_track=sum(r["sentinel1_slc_common_relative_track"]=="True" for r in gaps),
        b04_gap_weeks_with_sentinel1_grd_catalog_track=sum(r["sentinel1_grd_common_relative_track"]=="True" for r in gaps),
        b04_gap_independent_registration=dict(brackets_checked=len(registration),
            registration_statuses=dict(Counter(r["registration_status"] for r in registration)),
            brackets_with_passing_shadow_diagnostics=sum(passed.values()),
            tank_weeks_passing_at_both_boundaries=sum(n==2 for n in passed.values()),
            weeks_passing_at_both_boundaries=checked),
        interpretation="Dated interpolation uses passing native shadow diagnostics at each observation. It is a separate physical "
        "candidate because it does not require the independently checked B04 pair registration. Radar and Landsat context flags "
        "record availability only. No EIA outcome or market data are read.")
    export(f"{STEM}_weeks",weeks)
    export(f"{STEM}_timing_tank_weeks",tanks)
    if registration:
        export(f"{STEM}_gap_registration",registration)
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__=="__main__":
    main()
