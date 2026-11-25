"""Check dated outer Sentinel-2 pairs against independently registered native pixels."""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import timedelta

from audit_sentinel2_alternate_registration import independent_offsets, refit
from build_cushing_holdout_optical_recovery import DATED, equal, observation_book, timestamp
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_registration_inputs import anchor_grid, tank_points
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_panel import RESOLVED
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write
from fit_cushing_robust_stack import indexed

STEM = "cushing_outer_capture_pairs"
MAX_POST_SURVEY_HOURS = 72
RADIUS_M = 1600


def jobs(selection, occupied, *, internal=False):
    catalogue = observation_book()
    weeks = {}
    batch = {}
    endpoints = ("previous_before","current_after")
    internal_by_issue = defaultdict(list)
    if internal:
        roster = {r["candidate_id"]:r for r in read("sentinel2_space_time_coverage_roof_registry")
                  if r["has_both_dated_sources"]=="True"}
        clocks = {issue:tuple(timestamp(week,k) for k in
                  ("previous_measurement_at_utc","current_measurement_at_utc","cutoff_utc"))
                  for issue,week in selection.items()}
        for obj in catalogue.values():
            if obj["candidate_id"] not in roster:
                continue
            captured = timestamp(obj,"captured_at_utc")
            available = timestamp(obj,"geometry_available_at_utc")
            for issue, (start,end,cutoff) in clocks.items():
                if start<captured<=end and available<cutoff:
                    internal_by_issue[issue,obj["candidate_id"]].append(obj)
        candidates = [selection[issue] | dict(candidate_id=tank,terminal_id=roster[tank]["terminal_id"],
            circular_area_m2=roster[tank]["circular_area_m2"],geometry_mode=DATED,
            lidar_evidence_date=roster[tank]["lidar_source_available_date"],roof_evidence_date=roster[tank]["roof_evidence_date"],
            inputs_available_before_utc=selection[issue]["cutoff_utc"])
            for issue,tank in sorted(internal_by_issue)]
    else:
        candidates = read("shadow_timing_holdout_pairs")
    for r in candidates:
        issue, week, tank = (r[k] for k in ("release_date","week_ending","candidate_id"))
        if issue not in selection or tank in occupied.get(issue, ()) or r["geometry_mode"]!=DATED:
            continue
        calendar = selection[issue]
        start,end,cutoff = (timestamp(calendar,k) for k in ("previous_measurement_at_utc","current_measurement_at_utc","cutoff_utc"))
        if any(timestamp(r,k)!=when for k,when in (
            ("previous_measurement_at_utc",start),("current_measurement_at_utc",end),("inputs_available_before_utc",cutoff)
        )) or week!=calendar["week_ending"]:
            raise ValueError(f"Inconsistent reporting interval on {issue}")
        if internal:
            available = internal_by_issue[issue,tank]
            if len(available) < 2:
                continue
            # Choose the longest available interval. On tied duplicate tiles
            # use the earlier availability, then the source version.
            first_fit = min(available,key=lambda q:(timestamp(q,"captured_at_utc"),
                timestamp(q,"geometry_available_at_utc"),q["version_id"]))
            last_fit = min(available,key=lambda q:(-timestamp(q,"captured_at_utc").timestamp(),
                timestamp(q,"geometry_available_at_utc"),q["version_id"]))
            captures = [first_fit,last_fit]
        else:
            if not all(r[f"{edge}_version_id"] for edge in endpoints):
                continue
            captures = [catalogue[tank,r[f"{edge}_version_id"],r[f"{edge}_canonical_l2a_id"]]
                        for edge in endpoints]
        first,last = (timestamp(q,"captured_at_utc") for q in captures)
        if internal:
            if not start < first < last <= end or last-first < timedelta(hours=48):
                continue
        elif not start-timedelta(days=7)<=first<=start<end<last<cutoff:
            raise ValueError(f"Capture outside valid outer pair on {issue}/{tank}")
        elif (last-end).total_seconds()>MAX_POST_SURVEY_HOURS*3600:
            continue
        for q in captures:
            if (not timestamp(q,"captured_at_utc")<=timestamp(q,"archive_available_at_utc")<=timestamp(q,"geometry_available_at_utc")<cutoff
                or not equal(r["circular_area_m2"],math.pi*float(q["diameter_m"])**2/4)
                or q["fit_status"]!=RESOLVED or r["terminal_id"]!=q["terminal_id"]):
                raise ValueError(f"Invalid native fit on {issue}/{tank}")
        if any(not r[k] or r[k]>=first.date().isoformat() for k in ("lidar_evidence_date","roof_evidence_date")):
            continue
        key = week,tank
        if key in batch:
            raise ValueError(f"Duplicate outer-pair key {key}")
        delta = float(captures[0]["estimated_gap_m"])-float(captures[1]["estimated_gap_m"])
        pair = r | {"pre_measurement_pair_visible_kbbl":delta*float(r["circular_area_m2"])/BARREL_M3/1000}
        batch[key] = pair,captures
        # The fitter takes a closing boundary for scaling. Outer exposures
        # are corrected against the original, actual EIA boundary below.
        weeks[week] = (start,end,issue,cutoff.isoformat())
    return batch,weeks


def build(*, supplement=False, internal=False):
    source = indexed("cushing_recovered_optical_forecasts")
    outer_weeks = {r["release_date"] for r in read(f"{STEM}_fits")
        if r["registered_both_shadow_diagnostics_pass"] == "True"} if internal else set()
    selected = {r["release_date"]:r for r in read("holdout_optical_weeks")
                if bool(int(source[r["release_date"]]["total_recovered_tanks"])) == supplement
                and r["release_date"] not in outer_weeks}
    occupied = {}
    if supplement:
        from fit_cushing_recovered_optical import combined_sources
        occupied = {r["release_date"]:{q["candidate_id"] for q in arms[-1]} for r,arms,_ in combined_sources()
                    if r["release_date"] in selected}
        if occupied.keys() != selected.keys() or any(
            len(v) != int(source[k]["total_recovered_tanks"]) for k,v in occupied.items()
        ):
            raise ValueError("Incomplete original optical coverage")
    batch,weeks = jobs(selected, occupied, internal=internal)
    if not batch:
        return selected,[]
    tanks,exclusion = tank_points()
    anchors = anchor_grid({key[1]:tanks[key[1]] for key in batch},exclusion,search_radius_m=RADIUS_M)
    offsets = independent_offsets(batch,"holdout",anchors)
    if offsets.keys()!=batch.keys():
        raise ValueError("Incomplete independent registration audit")
    calculated = refit(batch,offsets,legacy_geometry(),weeks)
    outputs = []
    for r in calculated:
        week,tank = (r[k] for k in ("week_ending","candidate_id"))
        pair,captures = batch[week,tank]
        first,last = (timestamp(q,"captured_at_utc") for q in captures)
        start,end,_,_ = weeks[week]
        if (r["earlier_version"],r["later_version"])!=tuple(q["version_id"] for q in captures):
            raise ValueError(f"Outer fit used another source on {week}/{tank}")
        overlap = min(last,end) - max(first,start)
        outputs.append(r | dict(previous_measurement_at_utc=start.isoformat(),current_measurement_at_utc=end.isoformat(),
            latest_geometry_available_utc=max(timestamp(q,"geometry_available_at_utc") for q in captures).isoformat(),
            previous_week_fraction=max(0.,(start-first)/timedelta(days=7)),
            following_week_fraction=max(0.,(last-end)/timedelta(days=7)),
            fraction_of_capture_inside_eia_interval=overlap/(last-first),
            fraction_of_eia_interval_observed=overlap/(end-start),
            lidar_evidence_date=pair["lidar_evidence_date"],roof_evidence_date=pair["roof_evidence_date"]))
    return selected,outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--supplement", action="store_true", help="Check missing roofs in weeks with existing measurements")
    mode.add_argument("--internal", action="store_true", help="Check pairs inside the reporting interval in weeks still lacking data")
    args = parser.parse_args()
    selected,rows = build(supplement=args.supplement, internal=args.internal)
    stem = f"{STEM}_supplement" if args.supplement else f"{STEM}_internal" if args.internal else STEM
    passed = [r for r in rows if r["registered_both_shadow_diagnostics_pass"]]
    report = dict(selection="internal_pairs_of_unmeasured_weeks" if args.internal
        else "already_measured_weeks" if args.supplement else "unmeasured_weeks",
        eligible_releases=len(selected),candidate_releases=len({r["release_date"] for r in rows}),
        candidate_pairs=len(rows),max_post_survey_hours=None if args.internal else MAX_POST_SURVEY_HOURS,
        minimum_internal_capture_separation_hours=48 if args.internal else None,search_radius_m=RADIUS_M,
        registration_statuses=Counter(r["registration_status"] for r in rows),
        passed_pairs=len(passed), passed_releases=sorted({r["release_date"] for r in passed}),
        passed_pairs_by_release=Counter(r["release_date"] for r in passed))
    if rows:
        export(f"{stem}_fits",rows)
    else:
        write(DERIVED/f"{stem}_fits.csv",b"registered_both_shadow_diagnostics_pass\n")
    write(DERIVED/f"{stem}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
