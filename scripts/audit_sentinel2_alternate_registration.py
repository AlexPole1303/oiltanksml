"""Check additional archived Sentinel-2 pairs against independent registration and EIA."""
import argparse
import json
import math
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from functools import lru_cache

from audit_sentinel2_weekly_volume import PHASES, comparison, error_metrics, observations, releases
from audit_shadow_registration_pixels import check_source, coverage_rows, requested_patches
from build_registered_shadow_development import registered_pair, registered_profile, study_weeks
from build_shadow_registration_inputs import anchor_grid, build_windows, describe_source, registration_sources, tank_points
from build_terminal_development_measurements import BARREL_M3, DATED
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, RESOLVED, export, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write
from fit_shadow_geometry import inputs
from fit_shadow_registration import patch_profiles, tank_offset

STEM = "sentinel2_alternate_registration"
ENDPOINTS = ("previous_before", "current_before")


def pair_key(row):
    return row["week_ending"], row["candidate_id"]


def alternatives(phase, weeks, *, excluded_terminals=()):
    original = {pair_key(r): r for r in read_csv(f"weekly_shadow_registered_{phase}_pairs")
                if r["geometry_mode"] == DATED}
    fits = {(r["candidate_id"], r["version_id"], r["canonical_l2a_id"]): r
            for r in read_csv(f"shadow_timing_{phase}_observations")
            if r["geometry_mode"] == DATED and r["fit_status"] == RESOLVED}
    jobs, reused = {}, 0
    for row in read_csv(f"shadow_timing_{phase}_pairs"):
        if (row["geometry_mode"] != DATED or not row["pre_measurement_pair_visible_kbbl"]
            or row["terminal_id"] in excluded_terminals):
            continue
        key = pair_key(row)
        chosen = tuple(row[f"{side}_version_id"] for side in ENDPOINTS)
        prior = original.get(key)
        if prior and chosen == (prior["earlier_version"], prior["later_version"]):
            if prior["baseline_both_shadow_diagnostics_pass"] != "True":
                raise ValueError(f"Unregistered diagnostic differs for identical inputs at {key}")
            reused += 1
            continue
        previous, current, release, cutoff = weeks[key[0]]
        captures = [datetime.fromisoformat(row[f"{side}_captured_at_utc"]) for side in ENDPOINTS]
        if (release, cutoff) != (row["release_date"], row["inputs_available_before_utc"]) or not (
            captures[0] <= previous < captures[1] <= current
        ):
            raise ValueError(f"Alternate pair crosses the wrong reporting boundary at {key}")
        selected = []
        for side, version, capture in zip(ENDPOINTS, chosen, captures, strict=True):
            fit = fits[key[1], version, row[f"{side}_canonical_l2a_id"]]
            fields = ("captured_at_utc", "archive_available_at_utc", "geometry_available_at_utc", "measurement_source")
            if (any(row[f"{side}_{field}"] != fit[field] for field in fields)
                or row[f"{side}_gap_m"] != fit["estimated_gap_m"]
                or not capture <= min(datetime.fromisoformat(fit[field]) for field in fields[1:3])
                or max(datetime.fromisoformat(fit[field]) for field in fields[1:3]) >= datetime.fromisoformat(cutoff)
            ):
                raise ValueError(f"Alternate fit uses different or unavailable data at {key}/{side}")
            selected.append(fit)
        if key in jobs:
            raise ValueError(f"Duplicate alternate tank-week {key}")
        jobs[key] = row, selected
    return jobs, reused


def source_packet(name):
    if name == "core_terminal_legacy_observations.csv":
        return "weekly_shadow_development.sqlite"
    if name.endswith("_observations.csv"):
        return name.removesuffix("_observations.csv") + ".sqlite"
    return name


@lru_cache(maxsize=None)
def packet_inputs(name):
    path = DERIVED / name
    with closing(read_only(path)) as db:
        db.row_factory = sqlite3.Row
        rows, kernels = inputs(db)
        sources = {row["scene_id"]: dict(row) for row in registration_sources(db)}
    return path, {row["observation_id"]: row for row in rows}, sources, kernels


def attach_sources(jobs, phase):
    sources, links = {}, []
    for key, (pair, fits) in jobs.items():
        descriptors = []
        for fit in fits:
            _, rows, scenes, _ = packet_inputs(source_packet(fit["measurement_source"]))
            row = rows[int(fit["observation_id"])]
            scene = scenes[row["scene_id"]]
            identity = fit["version_id"], fit["source_directory"]
            if (row["candidate_id"] != key[1] or (scene["version_id"], scene["source_directory"]) != identity
                or scene["scene_item_id"] != fit["canonical_l2a_id"]
                or datetime.fromisoformat(scene["captured_at_utc"]) != datetime.fromisoformat(fit["captured_at_utc"])
            ):
                raise ValueError(f"Alternate fit and original raster differ for {key}")
            if identity not in sources:
                sources[identity] = describe_source(len(sources)+1, scene)
            descriptors.append(sources[identity])
        links.append(("alternate_weekly", phase, *key, "EFR", pair["terminal_id"],
                      pair["inputs_available_before_utc"], *(fit["archive_available_at_utc"] for fit in fits), *descriptors))
    return {row["source_id"]: row for row in sources.values()}, links


def independent_offsets(jobs, phase, anchors):
    sources, links = attach_sources(jobs, phase)
    screened, _, _, profiles, _ = registration_products(links, sources, anchors)
    return {(row["week_or_control_day"], row["candidate_id"]): tank_offset(row, profiles) for row in screened}


def registration_products(links, sources, anchors):
    pairs, windows, patches, _ = build_windows(links, anchors)
    pairs = {row["scene_pair_id"]: row for row in pairs}
    patches = {row["anchor_id"]: row for row in patches}
    wanted = requested_patches(windows, pairs, patches)
    tasks = [(sources[source], [patches[anchor] for anchor in sorted(anchors)])
             for source, anchors in wanted.items() if anchors]
    with ThreadPoolExecutor(max_workers=4) as pool:
        measured = {(row["source_id"], row["anchor_id"]): row
                    for results in pool.map(lambda task: check_source(*task), tasks) for row in results}
    screened, _ = coverage_rows(windows, pairs, measured)
    profiles, patch_rows = patch_profiles(screened, sources, pairs, patches)
    return screened, pairs, patches, profiles, patch_rows


def refit(jobs, offsets, geometry, weeks):
    @lru_cache(maxsize=1024)
    def profile(tank, name, number, diameter, pixels, gap, east, north):
        path, rows, _, kernels = packet_inputs(name)
        old = geometry[tank]
        row = dict(rows[number], center_east_m=float(old["center_east_m"]),
                   center_north_m=float(old["center_north_m"]), radius_m=float(old["diameter_m"])/2)
        shell = float(old["prior_rim_z_p95_native"])-float(old["prior_ground_z_p50_native"])
        fit = dict(diameter_m=diameter, fitted_pixels=pixels, estimated_gap_m=gap)
        return registered_profile(path, row, fit, shell, kernels[row["platform"]], east, north)

    def solve(item):
        key, (pair, fits) = item
        offset = offsets[key]
        previous, current, _, cutoff = weeks[key[0]]
        before, after = (datetime.fromisoformat(fit["captured_at_utc"]) for fit in fits)
        area = float(pair["circular_area_m2"])
        result = {field: pair[field] for field in ("geometry_mode", "week_ending", "release_date",
                                                   "candidate_id", "terminal_id")}
        result.update(inputs_available_before_utc=cutoff, circular_area_m2=area,
            circular_bbl_per_m=area/BARREL_M3, fraction_of_capture_inside_eia_interval=(after-previous)/(after-before),
            fraction_of_eia_interval_observed=(after-previous)/(current-previous),
            raw_change_kbbl=float(pair["pre_measurement_pair_visible_kbbl"]),
            **{field: offset[field] for field in ("scene_pair_id", "registration_status", "fit_inliers",
                "spatial_check_inliers", "later_minus_earlier_east_m", "later_minus_earlier_north_m")},
            registered_change_m=None, registered_both_shadow_diagnostics_pass=False,
            earlier_registered_status=None, later_registered_status=None, earlier_registered_gap_m=None,
            later_registered_gap_m=None, registered_centre_east_m=None, registered_centre_north_m=None,
            registered_loss_range_min_m=None, registered_loss_range_max_m=None)
        for side, fit in zip(("earlier", "later"), fits, strict=True):
            result[f"{side}_version"] = fit["version_id"]
            result[f"{side}_capture_at_utc"] = fit["captured_at_utc"]
            result[f"{side}_archive_available_at_utc"] = fit["archive_available_at_utc"]
            result[f"{side}_native_packet"] = source_packet(fit["measurement_source"])
            result[f"{side}_native_observation_id"] = fit["observation_id"]
        if offset["registration_status"] == "checked_offset":
            east, north = (float(offset[f"later_minus_earlier_{axis}_m"])/2 for axis in ("east", "north"))
            both = [profile(key[1], source_packet(fit["measurement_source"]), int(fit["observation_id"]),
                            fit["diameter_m"], fit["fitted_pixels"], fit["estimated_gap_m"], sign*east, sign*north)
                    for sign, fit in zip((-1, 1), fits, strict=True)]
            result.update(registered_pair(*both))
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(solve, jobs.items()))


def eia_check(phase, baseline, additions, targets):
    previous = list(comparison(phase, baseline, targets))
    expanded = list(comparison(phase, baseline + additions, targets))
    output = []
    for old, new in zip(previous, expanded, strict=True):
        output.append({field: new[field] for field in ("study_phase", "week_ending", "release_date",
            "first_release_cushing_kbbl", "first_release_change_kbbl", "previous_stock_known_before_issue_kbbl",
            "eia_source_url")} | {
            f"{prefix}_{field}": row[field] for prefix, row in (("original", old), ("expanded", new))
            for field in ("matched_tanks", "satellite_change_kbbl", "satellite_change_stock_error_kbbl",
                          "previous_change_plus_satellite_stock_error_kbbl")})
    scores = {}
    for scope, rows in (
        ("common_weeks", [r for r in output if r["original_matched_tanks"]]),
        ("common_weeks_with_additions", [r for r in output if 0 < r["original_matched_tanks"] < r["expanded_matched_tanks"]]),
        ("new_weeks", [r for r in output if not r["original_matched_tanks"] and r["expanded_matched_tanks"]]),
    ):
        scores[scope] = {}
        for prefix in ("original", "expanded"):
            signed = [r[f"{prefix}_satellite_change_kbbl"]-r["first_release_change_kbbl"]
                      for r in rows if r[f"{prefix}_satellite_change_kbbl"] is not None]
            if signed:
                scores[scope][prefix] = error_metrics(signed)
    return output, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("holdout",), help="Export independently registered alternatives without EIA outcomes")
    args = parser.parse_args()
    phases = (args.phase,) if args.phase else PHASES
    stem = f"{STEM}_{args.phase}" if args.phase else STEM
    geometry = legacy_geometry()
    anchors = anchor_grid(*tank_points())
    all_fits, all_added, all_weeks, summaries = [], [], [], {}
    for phase in phases:
        weeks = study_weeks(phase)
        baseline = list(observations(phase, weeks, geometry))
        # HCS was fitted separately after the original alternate selection was archived.
        jobs, identical = alternatives(phase,weeks,excluded_terminals=("amid_cushing",) if phase=="holdout" else ())
        print(f"{phase}: {len(jobs)} different capture pairs, {identical} identical", flush=True)
        fitted = refit(jobs, independent_offsets(jobs, phase, anchors), geometry, weeks)
        alternate = list(observations(phase, weeks, geometry, fitted))
        original_keys = {pair_key(row) for row in baseline}
        added = [row for row in alternate if pair_key(row) not in original_keys]
        rows, scores = eia_check(phase, baseline, added, releases(phase)) if phase in PHASES else ([], None)
        if len(fitted) != len(jobs) or any(not math.isfinite(r["capture_change_kbbl"]) for r in added):
            raise ValueError(f"Invalid registered alternates in {phase}")
        all_fits.extend(dict(study_phase=phase, **row) for row in fitted)
        all_added.extend(added)
        all_weeks.extend(rows)
        summaries[phase] = dict(identical_unregistered_pairs=identical, alternate_pairs=len(jobs),
            exterior_registration_statuses=dict(Counter(r["registration_status"] for r in fitted)),
            alternate_registered_pairs=len(alternate), added_registered_pairs=len(added),
            original_registered_pairs=len(baseline), expanded_registered_pairs=len(baseline)+len(added),
            original_weeks=len({r["week_ending"] for r in baseline}),
            expanded_weeks=len({r["week_ending"] for r in baseline+added}))
        if scores is not None:
            summaries[phase]["visible_change_vs_eia"] = scores
    if all_fits:
        export(f"{stem}_pairs", all_fits)
    if all_added:
        export(f"{stem}_added_volumes", all_added)
    if all_weeks:
        export(f"{stem}_eia_comparison", all_weeks)
    interpretation = "Keep every original accepted pair. For missing tank-weeks, admit an alternate only after independent " \
        "exterior alignment and both internal shadow diagnostics pass. "
    interpretation += "Holdout labels were not read." if args.phase else "Only open phases are compared with first-release EIA values."
    payload = json.dumps(dict(phases=summaries, interpretation=interpretation), indent=2) + "\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload)


if __name__ == "__main__":
    main()
