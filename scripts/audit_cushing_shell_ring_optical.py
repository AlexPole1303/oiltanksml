"""Check the LiDAR ring inventory against archived B04 reflectance on the same pixels."""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from statistics import mean, median

import numpy as np

from audit_plains_25c6_shadow_support import scores, summarize
from audit_cushing_aerial_shell_rings import STEM as AERIAL_RINGS
from build_cushing_okeis_southbow_controls import STEM as SOUTH_BOW, compare, operator_records
from build_cushing_lidar_shell_rings import STEM as RINGS
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_public import write
from fit_shadow_geometry import (CALIBRATION, PIXEL_M, SHIFTS_M, STEP_M, bearing_vector, inputs, pixel_grid, read_only,
                                 shifted_grid)

STEM = "cushing_lidar_shell_ring_optical"
VARIANTS = ("original_grid_original_geometry", "fine_grid_original_geometry", "fine_grid_candidate_geometry")


def laser_reference(selected, lines, stamp):
    keys = json.loads(selected["selected_flight_and_rank_json"])
    profiles = [lines[selected["candidate_id"], day, source, str(rank)] for day, source, rank in keys]
    maximum_separation = max(abs((stamp-datetime.fromisoformat(r[k])).total_seconds())/3600
                             for r in profiles for k in ("first_lidar_return_at_utc", "last_lidar_return_at_utc"))
    return (median(float(r["ring_gap_m"]) for r in profiles), maximum_separation,
            float(np.ptp([float(r["ring_gap_m"]) for r in profiles])))


def evaluations(db, originals, contours, lines, grounds):
    observations, kernels = inputs(db)
    for o in observations:
        old = originals[str(o["observation_id"])]
        if (old["candidate_id"], old["version_id"]) != (o["candidate_id"], o["version_id"]):
            raise ValueError("Calibration packet differs from the recorded reference fit")
        if o["roof_type"] != "EFR" or not old["estimated_gap_m"]:
            continue
        r = contours.get(o["candidate_id"])
        if r is None or r["status"] != "candidate_ring":
            continue
        original_shell = float(old["shell_above_ground_m"])
        original_gap = float(old["lidar_gap_m"])
        stamp = datetime.fromisoformat(o["captured_at_utc"])
        ring_gap, separation, spread = laser_reference(r, lines, stamp)
        ring_shell = float(r["ring_z_m"])-grounds[o["candidate_id"]]
        dx, dy = (float(r[f"ring_{p}_m"])-o[f"center_{p}_m"] for p in ("east", "north"))
        grid = pixel_grid(db, o)
        candidate_grid = shifted_grid(grid, dx, dy)
        source = dict(row=o, sun=bearing_vector(o["sun_zenith_deg"], o["sun_azimuth_grid_deg"]),
                      view=bearing_vector(o["red_view_zenith_deg"], o["red_view_azimuth_grid_deg"]),
                      kernel=kernels[o["platform"]], step=STEP_M, pixel=PIXEL_M, shifts=SHIFTS_M)
        trials = ((grid, o["radius_m"], original_shell, original_gap, False, 1),
                  (grid, o["radius_m"], original_shell, original_gap, True, 4),
                  (candidate_grid, float(r["ring_diameter_m"])/2, ring_shell, ring_gap, True, 4))
        for variant, (sample, radius, shell, ref, fine, subdiv) in zip(VARIANTS, trials, strict=True):
            gaps, shifts, loss, reverse = scores(source, sample, shell, radius, fine=fine, subdivisions=subdiv)
            f, _ = summarize(gaps, shifts, loss, reverse, sample, ref)
            if variant == VARIANTS[0] and (f["fit_status"], f["gap_m"]) != (old["fit_status"], float(old["estimated_gap_m"])):
                raise ValueError(f"B04 baseline changed at {o['candidate_id']}/{o['version_id']}")
            yield dict(observation_id=o["observation_id"], candidate_id=o["candidate_id"], terminal_id=r["terminal_id"],
                       version_id=o["version_id"], optical_capture_utc=o["captured_at_utc"], variant=variant,
                       ring_selection_basis=r["selection_basis"], original_primary_tile=old["selected_for_pass_metrics"] == "True",
                       diameter_m=2*radius, shell_scenario_m=shell, lidar_reference_gap_m=ref,
                       ring_flight_line_spread_m=spread, maximum_separation_hours=separation,
                       original_lidar_gap_m=original_gap, ring_lidar_gap_m=ring_gap,
                       optical_minus_lidar_gap_m=f["gap_m"]-ref, **f)


def comparisons(rows):
    by_observation = defaultdict(dict)
    for r in rows:
        if r["original_primary_tile"]:
            by_observation[r["observation_id"]][r["variant"]] = r
    groups = defaultdict(list)
    for pair in by_observation.values():
        control = pair[VARIANTS[0]]
        day = control["optical_capture_utc"][:10]
        groups[day, "all"].append(pair)
        groups[day, control["ring_selection_basis"]].append(pair)
    for (day, basis), samples in sorted(groups.items()):
        for variant in VARIANTS:
            paired = [p for p in samples if p[VARIANTS[0]]["fit_status"] == p[variant]["fit_status"] == RESOLVED]
            yield dict(optical_day=day, selection_basis=basis, variant=variant, primary_roofs=len(samples),
                       resolved_roofs=sum(p[variant]["fit_status"] == RESOLVED for p in samples),
                       both_resolved_roofs=len(paired), mean_abs_gap_difference_m=mean(abs(p[variant]["optical_minus_lidar_gap_m"]) for p in paired) if paired else None,
                       original_mean_abs_gap_difference_same_roofs_m=mean(abs(p[VARIANTS[0]]["optical_minus_lidar_gap_m"]) for p in paired) if paired else None,
                       mean_abs_difference_to_upper_ring_reference_m=mean(abs(p[variant]["gap_m"]-p[variant]["ring_lidar_gap_m"]) for p in paired) if paired else None,
                       original_mean_abs_difference_to_upper_ring_same_roofs_m=mean(abs(p[VARIANTS[0]]["gap_m"]-p[variant]["ring_lidar_gap_m"]) for p in paired) if paired else None,
                       separation_min_hours=min(p[variant]["maximum_separation_hours"] for p in samples),
                       separation_max_hours=max(p[variant]["maximum_separation_hours"] for p in samples))


def south_bow_inspections(contours, grounds):
    originals = {r["observation_id"]: r for r in read(f"{SOUTH_BOW}_sentinel_fits")}
    packet = DERIVED / f"{SOUTH_BOW}_sentinel_native.sqlite"
    records, _ = operator_records()
    rows = []
    with closing(read_only(packet)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
        for o in observations:
            old = originals[str(o["observation_id"])]
            candidate = o["candidate_id"]
            r = contours[candidate]
            if (old["candidate_id"], old["version_id"]) != (candidate, o["version_id"]) or r["status"] != "candidate_ring":
                raise ValueError("South Bow case lacks its matching optical and LiDAR geometry")
            if not old["estimated_gap_m"]:
                rows.extend(dict(old, variant=v) for v in VARIANTS)
                continue
            grid = pixel_grid(db, o)
            dx, dy = (float(r[f"ring_{p}_m"])-o[f"center_{p}_m"] for p in ("east", "north"))
            source = dict(row=o, sun=bearing_vector(o["sun_zenith_deg"], o["sun_azimuth_grid_deg"]),
                          view=bearing_vector(o["red_view_zenith_deg"], o["red_view_azimuth_grid_deg"]),
                          kernel=kernels[o["platform"]], step=STEP_M, pixel=PIXEL_M, shifts=SHIFTS_M)
            shell = float(old["shell_above_ground_m"])
            trials = ((grid, o["radius_m"], shell, False, 1), (grid, o["radius_m"], shell, True, 4),
                      (shifted_grid(grid, dx, dy), float(r["ring_diameter_m"])/2, float(r["ring_z_m"])-grounds[candidate], True, 4))
            for v, (sample, radius, height, fine, subdiv) in zip(VARIANTS, trials, strict=True):
                gaps, shifts, loss, reverse = scores(source, sample, height, radius, fine=fine, subdivisions=subdiv)
                reference = mean(r["conditional_design_shell_minus_liquid_m"] for r in records
                                 if r["cohort_id"] == old["cohort_id"])
                result, _ = summarize(gaps, shifts, loss, reverse, sample, reference)
                if v == VARIANTS[0] and (result["fit_status"], result["gap_m"]) != (old["fit_status"], float(old["estimated_gap_m"])):
                    raise ValueError(f"South Bow baseline changed at {candidate}/{o['version_id']}")
                rows.append(dict(old, variant=v, fit_status=result["fit_status"], estimated_gap_m=result["gap_m"],
                                 shell_above_ground_m=height, diameter_m=2*radius,
                                 near_optimal_gap_min_m=result["near_min_m"], near_optimal_gap_max_m=result["near_max_m"],
                                 shift_east_m=result["shift_east_m"], shift_north_m=result["shift_north_m"]))
    # Persist only fields recalculated for every variant, with original source identifiers.
    keys = ("candidate_id", "cohort_id", "product_id", "captured_at_utc", "variant", "fit_status", "estimated_gap_m",
            "shell_above_ground_m", "diameter_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m", "shift_east_m", "shift_north_m")
    groups = [dict(r, variant=v) for v in VARIANTS for r in compare(records, [f for f in rows if f["variant"] == v])]
    return [{k: r[k] for k in keys} for r in rows], groups


def aerial_supported_contours(contours, lines):
    for row in read(f"{AERIAL_RINGS}_two_campaigns"):
        candidate = row["candidate_id"]
        prior = contours[candidate]
        if prior["status"] != "ambiguous_upper_geometry" or row["same_lidar_candidate_both_campaigns"] != "True":
            continue
        keys = json.loads(row["ortho_2025_flights_and_ranks_json"])
        matched = [lines[candidate, day, source, str(rank)] for day, source, rank in keys]
        geometry = {field: median(float(r[field]) for r in matched)
                    for field in ("ring_diameter_m", "ring_z_m", "ring_east_m", "ring_north_m")}
        yield candidate, prior | geometry | dict(status="candidate_ring", selection_basis="two_aerial_campaigns",
                                                  selected_flight_and_rank_json=json.dumps(keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aerial-consensus", action="store_true", help="Check previously ambiguous rings selected on both aerial campaigns")
    args = parser.parse_args()
    originals = {r["observation_id"]: r for r in read("shadow_geometry_estimates")}
    contours = {r["candidate_id"]: r for r in read(f"{RINGS}_contours")}
    lines = {(r["candidate_id"], r["acquisition_date_utc"], r["point_source_id"], r["ring_rank"]): r
             for r in read(f"{RINGS}_flight_lines")}
    if args.aerial_consensus:
        contours = dict(aerial_supported_contours(contours, lines))
    grounds = {r["candidate_id"]: float(r["prior_ground_z_p50_native"]) for r in read("tank_lidar_outlines")
               if r["prior_ground_z_p50_native"]}
    with closing(read_only(CALIBRATION)) as db:
        db.row_factory = sqlite3.Row
        rows = list(evaluations(db, originals, contours, lines, grounds))
    metrics = list(comparisons(rows))
    stem = f"{STEM}_aerial_consensus" if args.aerial_consensus else STEM
    export(f"{stem}_fits", rows)
    export(f"{stem}_comparisons", metrics)
    report = dict(fits=len(rows), unique_contours=len({r["candidate_id"] for r in rows}),
                  resolved_primary_by_variant=Counter(r["variant"] for r in rows if r["original_primary_tile"] and r["fit_status"] == RESOLVED),
                  admitted_synchronized_operator_labels=0)
    if not args.aerial_consensus:
        south_bow, groups = south_bow_inspections(contours, grounds)
        export(f"{STEM}_south_bow_fits", south_bow)
        export(f"{STEM}_south_bow_cohorts", groups)
        report.update(south_bow_inspection_fits=len(south_bow), south_bow_resolved_by_variant=Counter(
            r["variant"] for r in south_bow if r["fit_status"] == RESOLVED), south_bow_complete_cohort_by_variant=Counter(
            r["variant"] for r in groups if r["optical_cohort_mean_gap_m"] is not None))
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
