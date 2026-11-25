"""Compare the same Grandstaff intervals using Sentinel-2 red, NIR and joint fits."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from statistics import median

from audit_grandstaff_september_sentinel_tiles import STEM as TILE_STEM
from audit_sentinel2_alternate_registration import packet_inputs
from build_grandstaff_okeis_records import LOCAL_TIME, MONTHS, STEM as OPERATOR, TERMINAL
from build_registered_shadow_development import registered_pair, study_weeks
from build_s1_reference_matches import DERIVED, export, read
from build_sentinel2_wider_registration import RADIUS_M, measure as wider_registration, regular_recovery, source_links
from build_weekly_shadow_panel import read_only
from build_weekly_shadow_validation import DATED, legacy_geometry
from check_multiband_shadow import spectral_kernels
from check_weekly_multiband_shadow import METHODS, spectral_profiles
from collect_public import write
from collect_s2_spectral_detector_masks import BANDS, collect, jobs

STEM = "cushing_grandstaff_september_spectral"
PERIOD_STEM = "cushing_grandstaff_monthly_spectral"
WIDER_STEM = "cushing_grandstaff_monthly_spectral_wider_registration"
PAIR_KEY = ("candidate_id", "week_ending", "earlier_version", "later_version")
PASS = "registered_both_shadow_diagnostics_pass"
COMPARED = "all_methods_passed"


def monthly_candidates(pairs):
    groups = defaultdict(list)
    for r in read("weekly_shadow_holdout_pairs"):
        if r["geometry_mode"] != DATED or r["terminal_id"] != TERMINAL:
            continue
        start, end = (datetime.fromisoformat(r[f"{side}_capture_at_utc"]).astimezone(LOCAL_TIME)
                      for side in ("earlier", "later"))
        month = start.strftime("%Y-%m")
        if month not in MONTHS[1:-1] or month != end.strftime("%Y-%m"):
            continue
        if not start < end:
            raise ValueError("The archived optical interval is reversed")
        reg = pairs[tuple(r[k] for k in PAIR_KEY)]
        groups[month, r["candidate_id"]].append((end, start, r, reg))
    candidates = []
    for (month, _), intervals in sorted(groups.items()):
        last = None
        for end, start, r, reg in sorted(intervals, key=lambda x: (x[0], x[1], x[2]["week_ending"])):
            selected = reg["offset_status"] == "checked_offset" and (last is None or last <= start)
            if selected:
                last = end
            candidates.append({k: r[k] for k in PAIR_KEY} | dict(
                report_month=month, cohort_id=r["cohort_id"], earlier_capture_local=start.isoformat(), later_capture_local=end.isoformat(),
                offset_status=reg["offset_status"], registration_source=reg.get("registration_source", "original"),
                selected_without_spectral_fit=selected,
            ))
    return candidates


def recover_unverified_registration(candidates, pairs):
    unresolved = {(r["week_ending"], r["candidate_id"]): tuple(r[k] for k in PAIR_KEY)
                  for r in candidates if r["offset_status"] != "checked_offset"}
    windows = [r for r in read("shadow_registration_tank_windows")
               if (r["comparison"], r["study_phase"], r["terminal_id"]) == ("weekly", "holdout", TERMINAL)
               and (r["week_or_control_day"], r["candidate_id"]) in unresolved]
    if len(windows) != len(unresolved):
        raise ValueError("Exterior registration windows differ from the monthly sample")
    offsets, patches = wider_registration(source_links(windows))
    refits, _ = regular_recovery(study_weeks("holdout"), unresolved.keys(), offsets)
    recovered = {}
    diagnostics = []
    for row in refits:
        week_tank = row["week_ending"], row["candidate_id"]
        key = tuple(row[k] for k in PAIR_KEY)
        if key != unresolved[week_tank]:
            raise ValueError(f"Wider registration changed the selected observations for {week_tank}")
        offset = offsets["weekly", *week_tank]
        diagnostics.append({k: row[k] for k in PAIR_KEY} | dict(
            original_offset_status=pairs[key]["offset_status"], wider_offset_status=row["offset_status"],
            fit_screened_patches=offset["fit_screened_patches"], check_screened_patches=offset["check_screened_patches"],
            offset_east_m=row["offset_east_m"], offset_north_m=row["offset_north_m"],
            check_median_residual_m=offset["check_median_residual_m"],
            b04_fit_status=row["fit_status"], b04_change_m=row["registered_change_m"]))
        if row["offset_status"] == "checked_offset":
            recovered[key] = dict(row, registration_source="wider_exterior_patches")
    return recovered, diagnostics, patches


def measure_intervals(targets, pairs, packet_name):
    path, packet, scenes, _ = packet_inputs(packet_name)
    observations = {(r["candidate_id"], r["version_id"]): r for r in packet.values()}
    by_version = {r["version_id"]: r["manifest_path"] for r in scenes.values()}
    versions = {r[f"{side}_version"] for r in targets for side in ("earlier", "later")}
    masks = [collect(job) for job in jobs({key: by_version[key] for key in versions})]
    files = {(r["version_id"], r["band"]): r["local_file"] for r in masks}
    with closing(read_only(path)) as db:
        kernels = spectral_kernels(db)
    geometry = legacy_geometry()

    def measure(target):
        roof = target["candidate_id"]
        reg = pairs[tuple(target[k] for k in PAIR_KEY)]
        old = geometry[roof]
        shell = float(old["prior_rim_z_p95_native"]) - float(old["prior_ground_z_p50_native"])
        if reg["offset_status"] != "checked_offset":
            raise ValueError(f"Unexpected unverified registration for {roof}")
        shift = [float(reg[f"offset_{axis}_m"]) / 2 for axis in ("east", "north")]
        solved = []
        saved_change = reg["registered_change_m"]
        row = {k: target[k] for k in PAIR_KEY} | dict(registered_b04_pass=reg[PASS] in (True, "True"),
            registered_b04_change_m=float(saved_change) if saved_change not in ("", None) else None,
            circular_bbl_per_m=float(reg["circular_bbl_per_m"]))
        for side, sign in (("earlier", -1), ("later", 1)):
            version = target[f"{side}_version"]
            raw = observations[roof, version]
            scene = scenes[raw["scene_id"]]
            o = dict(raw, center_east_m=float(old["center_east_m"]), center_north_m=float(old["center_north_m"]),
                     radius_m=float(old["diameter_m"]) / 2)
            status, count, profiles, _ = spectral_profiles(path, o, scene, shell, kernels,
                tuple(files[version, band] for band in BANDS), *(sign * value for value in shift))
            solved.append(profiles)
            row.update({f"{side}_pixel_status": status, f"{side}_common_pixels": count})
        for method in METHODS:
            result = registered_pair(*(p[method] for p in solved)) if all(solved) else {}
            for key in ("earlier_registered_gap_m", "later_registered_gap_m", "registered_change_m",
                        "earlier_registered_status", "later_registered_status", "registered_both_shadow_diagnostics_pass"):
                row[f"{method}_{key}"] = result.get(key)
        if row["earlier_pixel_status"] == row["later_pixel_status"] == "compared_full_pixels" and (
            row["red_registered_change_m"] != row["registered_b04_change_m"]
            or row["red_registered_both_shadow_diagnostics_pass"] != row["registered_b04_pass"]
        ):
            raise ValueError(f"Full-pixel B04 replay differs for {roof}/{target['week_ending']}")
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(measure, targets))
    return results, masks


def draw(rows, method):
    return sum(max(0, -r[f"{method}_registered_change_m"]) * r["circular_bbl_per_m"] for r in rows) if rows else None


def monthly_summary(results, candidates):
    reported = [r for r in read(f"{OPERATOR}_optical_throughput_checks") if r["report_month"] in MONTHS[1:-1]]
    monthly = []
    for r in reported:
        month, scope = r["report_month"], r["scope"]
        group = [x for x in candidates if x["report_month"] == month and (scope == "terminal" or x["cohort_id"] == scope)]
        fits = [x for x in results if x["report_month"] == month and (scope == "terminal" or x["cohort_id"] == scope)]
        shared = [x for x in fits if all(x[f"{m}_{PASS}"] for m in METHODS)]
        volumes = {m: draw(shared, m) for m in METHODS}
        published = int(r["reported_numeric_withdrawal_bbl"])
        values = dict(report_month=month, scope=scope, reported_numeric_withdrawal_bbl=published,
                      archived_pairs=len(group), checked_registration_pairs=sum(x["offset_status"] == "checked_offset" for x in group),
                      selected_nonoverlapping_pairs=len(fits), matched_pairs=len(shared),
                      matched_roofs=len({x["candidate_id"] for x in shared}),
                      comparison_status=COMPARED if shared else "no_shared_passing_pair")
        for m in METHODS:
            passing = [x for x in fits if x[f"{m}_{PASS}"]]
            values.update({f"{m}_passing_pairs": len(passing), f"{m}_sum_all_passing_bbl": draw(passing, m),
                           f"{m}_sum_matched_bbl": volumes[m],
                           f"{m}_matched_exceeds_report": volumes[m] > published if shared else None})
        monthly.append(values)
    return monthly


def september_summary(results, masks):
    matched = [r for r in results if all(r[f"{m}_{PASS}"] for m in METHODS)]
    terminal, = [r for r in read(f"{OPERATOR}_optical_throughput_checks")
                 if r["report_month"] == "2026-09" and r["scope"] == "terminal"]
    return dict(tested_intervals=len(results), source_detector_masks=len(masks),
                passing_intervals={m: sum(bool(r[f"{m}_{PASS}"]) for r in results) for m in METHODS},
                all_methods_passing_intervals=len(matched), summed_decreases_on_matched_intervals_bbl={m: draw(matched, m) for m in METHODS},
                reported_terminal_withdrawal_bbl=int(terminal["reported_numeric_withdrawal_bbl"]),
                median_absolute_change_difference_nir_minus_red_m=median(abs(r["nir_registered_change_m"] -
                    r["red_registered_change_m"]) for r in matched) if matched else None,
                independent_acquisitions=False, used_in_frozen_replay=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-full-months", action="store_true")
    parser.add_argument("--recover-registration", action="store_true",
                        help="Retry unverified pairs using a wider exterior search in the monthly diagnostic")
    args = parser.parse_args()
    if args.recover_registration and not args.all_full_months:
        parser.error("--recover-registration requires --all-full-months")
    pairs = {tuple(r[k] for k in PAIR_KEY): r for r in read("weekly_shadow_registered_holdout_pairs")
             if r["geometry_mode"] == DATED and r["terminal_id"] == TERMINAL}
    if args.all_full_months:
        stem = PERIOD_STEM
        candidates = monthly_candidates(pairs)
        if args.recover_registration:
            stem = WIDER_STEM
            recovered, offsets, patches = recover_unverified_registration(candidates, pairs)
            pairs.update(recovered)
            candidates = monthly_candidates(pairs)
            export(f"{stem}_offsets", offsets)
            export(f"{stem}_patches", patches)
        targets = [r for r in candidates if r["selected_without_spectral_fit"]]
        fits, masks = measure_intervals(targets, pairs, "weekly_shadow_holdout.sqlite")
        results = [target | fit for target, fit in zip(targets, fits, strict=True)]
        monthly = monthly_summary(results, candidates)
        export(f"{stem}_candidate_intervals", candidates)
        export(f"{stem}_balance", monthly)
        summary = dict(archived_pairs=len(candidates), checked_registration_pairs=sum(r["offset_status"] == "checked_offset" for r in candidates),
                       selected_nonoverlapping_pairs=len(results), source_detector_masks=len(masks),
                       terminal_months=[r for r in monthly if r["scope"] == "terminal"],
                       independent_acquisitions=False, used_in_frozen_replay=False)
        if args.recover_registration:
            summary.update(wider_search_radius_m=RADIUS_M, retried_registration_pairs=len(offsets),
                           recovered_registration_pairs=len(recovered))
    else:
        stem = STEM
        targets = [r for r in read(f"{OPERATOR}_optical_throughput_intervals") if r["report_month"] == "2026-09"]
        results, masks = measure_intervals(targets, pairs, f"{TILE_STEM}_native.sqlite")
        summary = september_summary(results, masks)
    export(f"{stem}_masks", masks)
    export(f"{stem}_pairs", results)
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
