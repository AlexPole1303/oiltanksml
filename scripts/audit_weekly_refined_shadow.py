"""Compare original and finer Sentinel-2 alignment on the historical weekly sample."""
import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from statistics import median

from audit_refined_shadow_alignment import OFFSETS, refine_windows
from audit_sentinel2_alternate_registration import packet_inputs, pair_key, registration_products
from audit_sentinel2_weekly_volume import PHASES, comparison, error_metrics, releases
from build_registered_shadow_development import registered_pair, study_weeks
from build_shadow_registration_inputs import anchor_grid, describe_source, tank_points
from build_weekly_shadow_panel import DERIVED, export, read_only
from build_weekly_shadow_validation import legacy_geometry
from check_weekly_multiband_shadow import SIDES, jobs, recovery_jobs
from collect_public import write
from fit_shadow_geometry import bearing_vector, gap_trials, pixel_grid, response_profiles, shifted_grid
from fit_shadow_registration import tank_offset

STEM = "sentinel2_weekly_refined_alignment"
MODES = ("coarse", "local", "scene")
REFINEMENTS = MODES[1:]


def checked_sources(phase, tasks):
    dates, sources, links = study_weeks(phase), {}, []
    seen = set()
    for volume, native, _, _ in tasks:
        key = pair_key(volume)
        previous, current, release, cutoff = dates[key[0]]
        capture = [datetime.fromisoformat(volume[f"{side}_capture_at_utc"]) for side in SIDES]
        available = volume["archive_available_at_utc"]
        if ((release, cutoff) != (volume["release_date"], volume["cutoff_utc"])
            or not capture[0] <= previous < capture[1] <= current
            or not capture[1] <= datetime.fromisoformat(available) < datetime.fromisoformat(cutoff)
        ):
            raise ValueError(f"Unexpected source or EIA calendar at {phase}/{key}")
        descriptors = []
        for side, (name, number), when in zip(SIDES, native, capture, strict=True):
            _, rows, scenes, _ = packet_inputs(name)
            observation = rows[number]
            scene = scenes[observation["scene_id"]]
            version = volume[f"{side}_sentinel2_version"]
            if ((observation["candidate_id"], scene["version_id"]) != (key[1], version)
                or datetime.fromisoformat(scene["captured_at_utc"]) != when or observation["roof_type"] != "EFR"
            ):
                raise ValueError(f"Different native roof, version or time at {phase}/{key}/{side}")
            identity = version, scene["source_directory"]
            if identity not in sources:
                sources[identity] = describe_source(len(sources)+1, scene)
            descriptors.append(sources[identity])
        pair = (*key, *(d["source_id"] for d in descriptors))
        if pair in seen:
            raise ValueError(f"Duplicate native candidate pair at {phase}/{key}")
        seen.add(pair)
        links.append(("weekly", phase, *key, "EFR", volume["terminal_id"], cutoff, available, available, *descriptors))
    return links, {r["source_id"]: r for r in sources.values()}


def alignment_key(row):
    return row["week_or_control_day"], row["candidate_id"], row["scene_pair_id"]


def checked_alignment(phase, tasks, anchors):
    links, sources = checked_sources(phase, tasks)
    windows, pairs, patches, profiles, patch_rows = registration_products(links, sources, anchors)
    if len(tasks) != len(windows):
        raise ValueError(f"Exterior registration lost input pairs in {phase}")
    lookup = {}
    checked = []
    for (volume, native, east, north), window, link in zip(tasks, windows, links, strict=True):
        key = pair_key(volume)
        pair = pairs[window["scene_pair_id"]]
        offset = tank_offset(window, profiles)
        old = tuple(offset[f"later_minus_earlier_{a}_m"] for a in OFFSETS)
        source_ids = tuple(pair[f"{side}_source_id"] for side in SIDES)
        if (key != alignment_key(window)[:2] or source_ids != tuple(s["source_id"] for s in link[-2:])
            or offset["registration_status"] != "checked_offset" or old != (2*east, 2*north)
        ):
            raise ValueError(f"Original 2 m registration differs in {phase}/{key}: {old} vs {(2*east, 2*north)}")
        checked.append(dict(window, roof_type="EFR", east_offset_m=old[0], north_offset_m=old[1]))
        lookup[alignment_key(window)] = volume, native
    original = {(row["scene_pair_id"], row["anchor_id"]): row for row in patch_rows}
    refined, patch_checks = refine_windows(checked, pairs, sources, patches, original)
    if len(refined) != len(lookup):
        raise ValueError(f"Finer registration lost native pairs in {phase}")
    result = [(lookup[alignment_key(row)], row) for row in refined]
    patch_checks = [dict(study_phase=phase, earlier_version=sources[pairs[r["scene_pair_id"]]["earlier_source_id"]]["version_id"],
        later_version=sources[pairs[r["scene_pair_id"]]["later_source_id"]]["version_id"], **r) for r in patch_checks]
    return result, patch_checks


def refit(sample, geometry):
    @lru_cache(maxsize=4096)
    def profile(tank, name, number, east, north):
        path, rows, _, kernels = packet_inputs(name)
        old = geometry[tank]
        row = dict(rows[number], center_east_m=float(old["center_east_m"]),
            center_north_m=float(old["center_north_m"]), radius_m=float(old["diameter_m"])/2)
        shell = float(old["prior_rim_z_p95_native"])-float(old["prior_ground_z_p50_native"])
        with closing(read_only(path)) as db:
            grid = shifted_grid(pixel_grid(db, row), east, north)
        sun = bearing_vector(row["sun_zenith_deg"], row["sun_azimuth_grid_deg"])
        view = bearing_vector(row["red_view_zenith_deg"], row["red_view_azimuth_grid_deg"])
        gaps = gap_trials(shell, row["roof_type"])
        shifts, loss, _, reverse = response_profiles(grid, row["radius_m"], shell if shell >= 5 else 0,
            gaps, sun, view, kernels[row["platform"]], reversed_sun=True)
        return shifts, gaps, loss, reverse, len(grid[-2]), float(grid[-1].sum())

    def solve(item):
        (volume, native), row = item
        fixed = {field: volume[field] for field in ("week_ending", "release_date", "terminal_id", "cutoff_utc",
                                                    "archive_available_at_utc")}
        output = dict(row, **fixed, original_resolved=volume.get("baseline_red_resolved", True),
                      alternate_capture_pair=volume.get("alternate_capture_pair", False),
                      circular_kbbl_per_m=float(volume["circular_kbbl_per_m"]))
        for side, (name, number) in zip(SIDES, native, strict=True):
            output.update({f"{side}_version": volume[f"{side}_sentinel2_version"],
                           f"{side}_capture_at_utc": volume[f"{side}_capture_at_utc"],
                           f"{side}_native_packet": name, f"{side}_native_observation_id": number})
        for mode in MODES:
            fit = None
            if row[f"{mode}_east_m"] is not None:
                east, north = (row[f"{mode}_{axis}_m"]/2 for axis in OFFSETS)
                fit = registered_pair(*(profile(volume["candidate_id"], *source, sign*east, sign*north)
                                        for source, sign in zip(native, (-1, 1), strict=True)))
            passed = fit and fit["registered_both_shadow_diagnostics_pass"]
            if mode == "coarse" and (
                passed != output["original_resolved"] or
                any(fit[f"{side}_registered_gap_m"] != float(volume[f"{side}_gap_m"]) for side in SIDES)
            ):
                raise ValueError(f"Coarse B04 height or diagnostic replay differs at {pair_key(volume)}")
            output.update({f"{mode}_{field}": value for field, value in (
                ("resolved", bool(passed)), ("change_m", fit["registered_change_m"] if fit else None),
                ("change_kbbl", fit["registered_change_m"]*output["circular_kbbl_per_m"] if fit else None),
                ("earlier_gap_m", fit["earlier_registered_gap_m"] if fit else None),
                ("later_gap_m", fit["later_registered_gap_m"] if fit else None),
                ("earlier_fit_status", fit["earlier_registered_status"] if fit else None),
                ("later_fit_status", fit["later_registered_status"] if fit else None))})
            if mode != "coarse":
                output[f"{mode}_can_use"] = bool(passed and
                    row[f"{mode}_check_correlation"] >= row["coarse_check_correlation"]-1e-8)
        return output

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, row in enumerate(pool.map(solve, sample), 1):
            if i % 300 == 0 or i == len(sample):
                print(f"Refitted {i}/{len(sample)} weekly B04 pairs", flush=True)
            yield row


def weekly_sets(rows):
    original = [row for row in rows if row["original_resolved"]]
    fixed = [row for row in original if all(row[f"{mode}_change_kbbl"] is not None for mode in MODES)]
    strict = [row for row in fixed if all(row[f"{mode}_resolved"] for mode in MODES)]
    for scope, sample in (("fixed_tank_weeks", fixed), ("all_modes_resolved", strict)):
        yield scope, {m: [(r, r[f"{m}_change_kbbl"]) for r in sample] for m in MODES}
    guarded = {"coarse": [(r, r["coarse_change_kbbl"]) for r in original]}
    for mode in REFINEMENTS:
        guarded[mode] = [(r, r[f"{mode}_change_kbbl"] if r[f"{mode}_can_use"] else r["coarse_change_kbbl"]) for r in original]
    yield "guarded_fixed_tank_weeks", guarded
    expanded = dict(guarded)
    for mode in REFINEMENTS:
        added = {}
        for row in sorted(rows, key=lambda r: r["alternate_capture_pair"]):
            if not row["original_resolved"] and row[f"{mode}_can_use"]:
                added.setdefault(pair_key(row), row)
        expanded[mode] = guarded[mode] + [(r, r[f"{mode}_change_kbbl"]) for r in added.values()]
    yield "guarded_with_new_tank_weeks", expanded


def metrics(rows, fields, target):
    return {field: error_metrics([row[field]-row[target] for row in rows]) for field in fields}


def eia_panel(phase, rows):
    targets = list(comparison(phase, [], releases(phase)))
    output, summary = [], {}
    for scope, variants in weekly_sets(rows):
        by_week = {mode: defaultdict(list) for mode in MODES}
        for mode, sample in variants.items():
            for row, amount in sample:
                by_week[mode][row["week_ending"]].append((row, amount))
        panel = []
        for target in targets:
            week = target["week_ending"]
            keys = ("study_phase", "week_ending", "release_date", "first_release_cushing_kbbl", "first_release_change_kbbl",
                    "previous_stock_known_before_issue_kbbl", "previous_change_known_before_issue_kbbl", "eia_source_url")
            result = dict(comparison_scope=scope, **{key: target[key] for key in keys})
            anchor, prior = (target[k] for k in keys[5:7])
            result["stock_from_previous_change_kbbl"] = anchor+prior if anchor is not None and prior is not None else None
            for mode in MODES:
                sample = by_week[mode][week]
                if len(sample) != len({row["candidate_id"] for row, _ in sample}) or any(
                    row["release_date"] != target["release_date"] for row, _ in sample):
                    raise ValueError(f"Duplicate or mistimed inventory observation for {phase}/{week}/{scope}")
                amount = sum(value for _, value in sample) if sample else None
                result[f"{mode}_matched_tanks"] = len(sample)
                result[f"{mode}_visible_change_kbbl"] = amount
                result[f"{mode}_stock_with_previous_change_kbbl"] = anchor+prior+amount if None not in (anchor, prior, amount) else None
            panel.append(result)
        output.extend(panel)
        available = [row for row in panel if row["first_release_change_kbbl"] is not None and all(
            row[f"{m}_visible_change_kbbl"] is not None for m in MODES)]
        anchored = [row for row in available if row["stock_from_previous_change_kbbl"] is not None]
        summary[scope] = dict(
            tank_weeks={m: len(sample) for m, sample in variants.items()},
            weeks_by_method={m: sum(bool(row[f"{m}_matched_tanks"]) for row in panel) for m in MODES},
            visible_change_same_weeks=metrics(available, [f"{m}_visible_change_kbbl" for m in MODES], "first_release_change_kbbl"),
            stocks_anchored_same_weeks=metrics(anchored, ["stock_from_previous_change_kbbl"] + [
                f"{m}_stock_with_previous_change_kbbl" for m in MODES], "first_release_cushing_kbbl"))
    return output, summary


def holdout_panel(phase, rows):
    output, summary = [], {}
    for scope,variants in weekly_sets(rows):
        by_week = defaultdict(list)
        for mode,sample in variants.items():
            for row,amount in sample:
                by_week[mode,row["week_ending"]].append((row["candidate_id"],row["release_date"],amount))
        for week,(_,_,release,cutoff) in study_weeks(phase).items():
            result = dict(comparison_scope=scope,study_phase=phase,week_ending=week,release_date=release,cutoff_utc=cutoff)
            for mode in MODES:
                sample = by_week[mode,week]
                if len(sample)!=len({tank for tank,_,_ in sample}) or any(issue!=release for _,issue,_ in sample):
                    raise ValueError(f"Duplicate or mistimed holdout refinement at {week}/{mode}/{scope}")
                result[f"{mode}_matched_tanks"] = len(sample)
                result[f"{mode}_visible_change_kbbl"] = sum(amount for _,_,amount in sample) if sample else None
            output.append(result)
        summary[scope] = {m:len(sample) for m,sample in variants.items()}
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", action="store_true", help="Export geometric refinements without holdout outcomes")
    parser.add_argument("--with-alternates", action="store_true", help="Include verified holdout alternate registrations")
    args = parser.parse_args()
    if args.with_alternates and not args.holdout:
        parser.error("--with-alternates requires --holdout")
    phases = ("holdout",) if args.holdout else PHASES
    stem = f"{STEM}_holdout" if args.holdout else STEM
    geometry, anchors = legacy_geometry(), anchor_grid(*tank_points())
    all_pairs, all_patches, all_weeks, summaries = [], [], [], {}
    for phase in phases:
        include_alternates = not args.holdout or args.with_alternates
        original = list(jobs(phase,include_alternates=include_alternates))
        recovery = list(recovery_jobs(phase,{pair_key(volume) for volume,*_ in original},include_alternates=include_alternates))
        sample = original + recovery
        print(f"{phase}: {len(original)} baseline tank-weeks and {len(recovery)} previously rejected pairs", flush=True)
        refined, patches = checked_alignment(phase, sample, anchors)
        rows = list(refit(refined, geometry))
        panel, scores = holdout_panel(phase,rows) if args.holdout else eia_panel(phase,rows)
        summaries[phase] = dict(original_tank_weeks=len(original), recovery_pairs=len(recovery),
            exterior_patches=len(patches), available_offsets={m: sum(r[f"{m}_change_m"] is not None for r in rows) for m in MODES},
            retained_original_shadow_diagnostics={m: sum(r["original_resolved"] and r[f"{m}_resolved"] for r in rows) for m in MODES},
            exterior_score_change={m: median(r[f"{m}_check_correlation"]-r["coarse_check_correlation"] for r in rows
                                            if r[f"{m}_check_correlation"] is not None) for m in REFINEMENTS}, scores=scores)
        all_pairs.extend(rows)
        all_patches.extend(patches)
        all_weeks.extend(panel)
    export(f"{stem}_pairs", all_pairs)
    export(f"{stem}_patches", all_patches)
    export(f"{stem}_weeks" if args.holdout else f"{stem}_eia_comparison", all_weeks)
    report = dict(phases=summaries, interpretation="Original optical identity and historical availability "
        "are preserved. Both finer offsets use fitting patches only. Guarded runs require both shadow diagnostics and "
        "at least the original median score on independent exterior checks. Recovery pairs are preexisting aligned "
        "B04 failures, standard endpoint pairs take precedence, and no EIA values select fits. "
        "Identical-tank comparisons and coverage-expansion comparisons have different denominators.")
    if args.holdout:
        report["included_alternate_pairs"] = args.with_alternates
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
