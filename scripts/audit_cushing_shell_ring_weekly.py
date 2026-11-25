"""Replay weekly roof changes with the same native scenes and dated aerial geometry."""

import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from statistics import median

from audit_cushing_aerial_shell_rings import STEM as AERIAL
from build_cushing_lidar_shell_rings import STEM as RINGS
from build_cushing_shell_ring_source_clocks import STEM as CLOCK
from build_registered_shadow_development import RETROSPECTIVE, observation_sources, prepare_jobs, registered_pair, study_weeks
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_shadow_geometry import bearing_vector, gap_trials, pixel_grid, read_only, response_profiles, shifted_grid

STEM = "cushing_shell_ring_weekly"
MODERN = "cushing_modern_lidar_optical"
BARREL_M3 = .158987294928
FINE_CENTRES = [(float(x), float(y)) for x in range(-6, 7, 2) for y in range(-6, 7, 2)]


def geometry_inputs():
    dependencies = defaultdict(list)
    for r in read(f"{CLOCK}_roof_dependencies"):
        dependencies[r["candidate_id"]].append(r)
    modern = {r["candidate_id"]: r for r in read(f"{MODERN}_registry")}
    lines = {(r["candidate_id"], r["acquisition_date_utc"], r["point_source_id"], r["ring_rank"]): r
             for r in read(f"{RINGS}_flight_lines")}
    for r in read(f"{AERIAL}_two_campaigns"):
        if r["same_lidar_candidate_both_campaigns"] != "True":
            continue
        key = r["candidate_id"]
        stamps = dependencies[key]
        if {x["campaign"] for x in stamps} != {"ok_aerial_2025_rgb", "naip_2023"}:
            raise ValueError(f"Missing publication evidence for {key}")
        rings = [lines[key, day, line, str(rank)] for day, line, rank in json.loads(r["ortho_2025_flights_and_ranks_json"])]
        old = modern[key]
        aerial_date = max(x["date_evidence_for_reconstruction"] for x in stamps)
        dated = [old[k] for k in ("lidar_source_available_date", "roof_evidence_date") if old[k]]
        yield key, dict(candidate_id=key, terminal_id=r["terminal_id"], lidar_status=r["lidar_status"],
                        modern_geometry_status=old["modern_input_status"], lidar_source_available_date=old["lidar_source_available_date"],
                        roof_evidence_date=old["roof_evidence_date"], aerial_source_evidence_date=aerial_date,
                        geometry_source_evidence_date=max(aerial_date, *dated), first_seen=old["first_seen"],
                        ring_east_m=median(float(s["ring_east_m"]) for s in rings), ring_north_m=median(float(s["ring_north_m"]) for s in rings),
                        ring_diameter_m=median(float(s["ring_diameter_m"]) for s in rings),
                        ring_shell_m=median(float(s["ring_z_m"]) for s in rings)-float(old["prior_ground_z_p50_native"]),
                        original_diameter_m=float(old["diameter_m"]),
                        original_shell_m=float(old["prior_rim_z_p95_native"])-float(old["prior_ground_z_p50_native"]),
                        contemporary_use_verified=False)


def replay(job, native, original, ring):
    _, base, _, offset, keys, *_ = job
    result = {k: original[k] for k in ("week_ending", "release_date", "candidate_id", "terminal_id", "earlier_version", "later_version",
              "earlier_capture_at_utc", "later_capture_at_utc", "inputs_available_before_utc", "earlier_archive_available_at_utc",
              "later_archive_available_at_utc", "previous_detector_geometry_available_at_utc", "current_detector_geometry_available_at_utc",
              "fraction_of_eia_interval_observed", "fraction_of_capture_inside_eia_interval", "offset_status", "offset_east_m", "offset_north_m")}
    result.update(geometry_source_evidence_date=ring["geometry_source_evidence_date"], original_diameter_m=ring["original_diameter_m"],
                  candidate_diameter_m=ring["ring_diameter_m"], original_fit_status=original["fit_status"],
                  original_change_m=original["registered_change_m"],
                  original_earlier_status=original["earlier_registered_status"], original_later_status=original["later_registered_status"],
                  original_change_range_min_m=original["registered_loss_range_min_m"], original_change_range_max_m=original["registered_loss_range_max_m"],
                  source_clock_basis="dated_public_sources_reconstructed_retrospectively")
    empty = dict(earlier_registered_gap_m=None, later_registered_gap_m=None, registered_change_m=None, earlier_registered_status=None,
                 later_registered_status=None, registered_both_shadow_diagnostics_pass=False, registered_centre_east_m=None,
                 registered_centre_north_m=None, registered_loss_range_min_m=None, registered_loss_range_max_m=None)
    result.update({f"{prefix}_{k}": v for prefix in ("fine_original", "ring") for k, v in empty.items()})
    if offset["registration_status"] != "checked_offset" or not base["optical_level_change_m"]:
        return result
    profiles = defaultdict(list)
    for side, (packet, _, _, number) in enumerate(keys):
        path, rows, _, kernels = native[packet]
        o = rows[number]
        sign = -.5 if side == 0 else .5
        dx, dy = [sign*float(offset[f"later_minus_earlier_{axis}_m"]) for axis in ("east", "north")]
        with closing(read_only(path)) as db:
            grid = pixel_grid(db, o)
        if not math.isclose(2*o["radius_m"], ring["original_diameter_m"], abs_tol=1e-8):
            raise ValueError(f"Inconsistent native diameter for {o['candidate_id']}")
        sun = bearing_vector(o["sun_zenith_deg"], o["sun_azimuth_grid_deg"])
        view = bearing_vector(o["red_view_zenith_deg"], o["red_view_azimuth_grid_deg"])
        kernel = kernels[o["platform"]]
        old = shifted_grid(grid, dx, dy)
        new = shifted_grid(old, ring["ring_east_m"]-o["center_east_m"], ring["ring_north_m"]-o["center_north_m"])
        for label, sample, radius, shell, centres, subcells in (
                ("original", old, o["radius_m"], ring["original_shell_m"], None, 1),
                ("fine_original", old, o["radius_m"], ring["original_shell_m"], FINE_CENTRES, 4),
                ("ring", new, ring["ring_diameter_m"]/2, ring["ring_shell_m"], FINE_CENTRES, 4)):
            gaps = gap_trials(shell, o["roof_type"])
            shifts, loss, _, reverse = response_profiles(sample, radius, shell, gaps, sun, view, kernel,
                reversed_sun=True, centre_offsets=centres, subcell_samples=subcells)
            profiles[label].append((shifts, gaps, loss, reverse, len(sample[-1]), float(sample[-1].sum())))
    old = registered_pair(*profiles["original"])
    if any(str(old[k]) != original[k] and (not isinstance(old[k], float) or not math.isclose(old[k], float(original[k]), abs_tol=1e-8))
           for k in old):
        raise ValueError(f"Original pair does not reproduce for {result['week_ending']}/{result['candidate_id']}")
    for label in ("fine_original", "ring"):
        result.update({f"{label}_{k}": v for k, v in registered_pair(*profiles[label]).items()})
    return result


def volume(change, diameter):
    return float(change)*math.pi*(diameter/2)**2/BARREL_M3/1000


def panels(rows, weeks):
    groups = defaultdict(list)
    for r in rows:
        groups[r["week_ending"]].append(r)
    for week, (_, _, release, cutoff) in weeks.items():
        candidates = groups[week]
        original = [r for r in candidates if r["original_fit_status"] == "resolved"]
        ring = [r for r in candidates if r["ring_registered_both_shadow_diagnostics_pass"]]
        shared = [r for r in ring if r["original_fit_status"] == "resolved"]
        yield dict(week_ending=week, release_date=release, cutoff_utc=cutoff, eligible_pairs=len(candidates),
                   independently_aligned_pairs=sum(r["offset_status"] == "checked_offset" for r in candidates),
                   original_resolved_pairs=len(original), fine_original_resolved_pairs=sum(r["fine_original_registered_both_shadow_diagnostics_pass"] for r in candidates),
                   ring_resolved_pairs=len(ring), common_resolved_pairs=len(shared),
                   original_common_change_kbbl=sum(volume(r["original_change_m"], r["original_diameter_m"]) for r in shared) if shared else None,
                   ring_common_change_kbbl=sum(volume(r["ring_registered_change_m"], r["candidate_diameter_m"]) for r in shared) if shared else None,
                   original_visible_change_kbbl=sum(volume(r["original_change_m"], r["original_diameter_m"]) for r in original) if original else None,
                   ring_visible_change_kbbl=sum(volume(r["ring_registered_change_m"], r["candidate_diameter_m"]) for r in ring) if ring else None)


def main():
    geometry = dict(geometry_inputs())
    native, saved, _ = observation_sources("holdout")
    weeks = study_weeks("holdout")
    pair = {(r["week_ending"], r["candidate_id"]): r for r in read(f"{MODERN}_pairs")}
    jobs = []
    for job in prepare_jobs(weeks, native, saved, "holdout"):
        mode, base, *_ = job
        key = base["week_ending"], base["candidate_id"]
        ring = geometry.get(key[1])
        if mode != RETROSPECTIVE or not ring or key not in pair:
            continue
        cutoff = datetime.fromisoformat(base["inputs_available_before_utc"])
        if ring["geometry_source_evidence_date"] >= cutoff.date().isoformat():
            continue
        r = pair[key]
        for fields in (("earlier_capture_at_utc", "earlier_archive_available_at_utc", "previous_detector_geometry_available_at_utc"),
                       ("later_capture_at_utc", "later_archive_available_at_utc", "current_detector_geometry_available_at_utc")):
            capture, published, detector = (datetime.fromisoformat(r[k]) for k in fields)
            if not capture <= published <= detector < cutoff:
                raise ValueError(f"Optical data outside the pre-release clock for {key}")
        jobs.append((job, r, ring))
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(lambda x: replay(x[0], native, x[1], x[2]), jobs), 1):
            rows.append(row)
            if i % 250 == 0:
                print(f"{i}/{len(jobs)} weekly geometry pairs", flush=True)
    weekly = list(panels(rows, weeks))
    export(f"{STEM}_geometry", list(geometry.values()))
    export(f"{STEM}_pairs", rows)
    export(f"{STEM}_weeks", weekly)
    report = dict(geometry_contours=len(geometry), dated_geometry_status=Counter(r["modern_geometry_status"] for r in geometry.values()),
                  pairs=len(rows), optical_contours=len({r["candidate_id"] for r in rows}), weeks=len(weekly),
                  first_eligible_release=min(r["release_date"] for r in rows),
                  weeks_with_eligible_pairs=sum(bool(r["eligible_pairs"]) for r in weekly),
                  paired_counts={field: sum(r[field] for r in weekly) for field in ("independently_aligned_pairs", "original_resolved_pairs",
                                 "fine_original_resolved_pairs", "ring_resolved_pairs", "common_resolved_pairs")},
                  weeks_with_common_measurement=sum(bool(r["common_resolved_pairs"]) for r in weekly))
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
