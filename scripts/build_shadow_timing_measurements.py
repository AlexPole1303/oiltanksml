"""Fit archived L1C alternatives and interpolate roof levels between actual captures."""
import argparse
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from statistics import median

from audit_lidar_tanks import LEGACY_DATASET
from build_core_terminal_legacy_inputs import CORE, roof_clocks
from build_lidar_roof_reference import NEI_ROOF_DATES
from build_registered_shadow_development import observation_sources, study_weeks
from build_shadow_calibration import CONTEXT_M
from build_shadow_timing_candidates import ROLES
from build_terminal_development_measurements import BARREL_M3
from build_terminal_weekly_inputs import TERMINALS, VISUAL_DATE_REQUIRED
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, build_pixels, export, read_only
from build_weekly_shadow_validation import DATED, RETROSPECTIVE
from collect_public import write
from fit_shadow_geometry import estimate, inputs
from terminal_aerial_roof_evidence import hcs_roof_dates

MODES = (RETROSPECTIVE, DATED)


def observation_key(row):
    return row["candidate_id"], row["version_id"], row["canonical_l2a_id"]


def date_evidence():
    lidar = defaultdict(str)
    for row in read_csv("core_terminal_legacy_lidar_source_timing"):
        tank = row["candidate_id"]
        lidar[tank] = max(lidar[tank], row["available_by_date"])
    for row in read_csv("terminal_lidar_source_timing"):
        if row["dataset"] == LEGACY_DATASET:
            tank = row["candidate_id"]
            lidar[tank] = max(lidar[tank], row["source_catalog_date"])
    permits, aerial = roof_clocks()
    with closing(read_only(SOURCE)) as db:
        nei = {(terminal, cohort): day for terminal, cohort, _, day in db.execute(
            NEI_ROOF_DATES + " HAVING documented_roof_type='EFR'")}
    return lidar, permits, aerial, nei, hcs_roof_dates()


def engineering_dates(row, evidence):
    lidar, permits, aerial, nei, hcs = evidence
    tank, terminal, cohort = (row[field] for field in ("candidate_id", "terminal_id", "cohort_id"))
    lidar_day = lidar.get(tank)
    if terminal in CORE:
        doc = row["engineering_source_url"].rsplit("/", 1)[-1].removesuffix(".pdf")
        roof = aerial.get(tank, permits.get(doc, (None, None, None)))[0]
    elif terminal == "amid_cushing" and cohort == "amid_five_2009":
        roof = hcs.get(tank)
    elif terminal in TERMINALS:
        roof = nei.get((terminal, cohort))
        if roof and cohort in VISUAL_DATE_REQUIRED:
            roof = max(roof, row["source_document_date"]) if row["source_document_date"] else None
    else:
        roof = None
    return lidar_day, roof


def dated_source(row, evidence):
    lidar_day, roof = engineering_dates(row, evidence)
    cutoff = row["inputs_available_before_utc"][:10]
    if lidar_day and roof and max(lidar_day, roof) < cutoff:
        return lidar_day, roof
    return None


def candidates(phase, weeks):
    windows = {(row["week_ending"], row["candidate_id"]): row for row in read_csv("shadow_timing_candidates_tanks")
               if row["study_phase"] == phase and row["pair_available_with_post_measurement"] == "True"}
    evidence = date_evidence()
    grouped, dated = defaultdict(lambda: defaultdict(list)), {}
    wanted = {}
    requested_modes = defaultdict(set)
    with (DERIVED / "shadow_timing_candidates_observations.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["study_phase"] != phase:
                continue
            week, tank = row["week_ending"], row["candidate_id"]
            window_key = week, tank
            if window_key not in windows:
                continue
            _, _, release, cutoff = weeks[week]
            if (release, cutoff) != (row["release_date"], row["inputs_available_before_utc"]):
                raise ValueError(f"Timing source uses a different release window for {window_key}")
            if max(datetime.fromisoformat(row[field]) for field in ("captured_at_utc", "archive_available_at_utc",
                   "geometry_available_at_utc")) >= datetime.fromisoformat(cutoff):
                raise ValueError(f"Unavailable capture for {window_key}")
            key = observation_key(row)
            wanted.setdefault(key, row)
            requested_modes[key].add(RETROSPECTIVE)
            if window_key not in dated:
                dated[window_key] = dated_source(row, evidence)
            if dated[window_key]:
                requested_modes[key].add(DATED)
            grouped[window_key][row["endpoint"]].append((key, int(row["context_valid_cells"]), row["overpass_key"]))
    if grouped.keys() != windows.keys():
        raise ValueError("Timing candidates differ from the potential pair ledger")
    print(f"{len(windows)} tank-weeks use {len(wanted)} exact tank/capture observations", flush=True)
    return windows, grouped, dated, wanted, requested_modes


def reusable_estimates(phase):
    native, saved, geometry = observation_sources(phase)
    reusable = {}
    for (stem, mode, number), row in saved.items():
        packet = "weekly_shadow_development" if stem == "core_terminal_legacy" else stem
        _, observations, scenes, _ = native[packet]
        scene = scenes[observations[number]["scene_id"]]
        key = mode, row["candidate_id"], row["version_id"], scene["scene_item_id"]
        if key in reusable:
            raise ValueError(f"Duplicate archived fit for {key}")
        reusable[key] = dict(row, measurement_source=f"{stem}_observations.csv")
    return reusable, geometry


def fit_additional(phase, stem, additional, wanted, geometry, *, reuse_pixels):
    if not additional:
        return {}, None
    packet = DERIVED / f"{stem}.sqlite"
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        shells = dict(db.execute("SELECT candidate_id,lidar_rim_to_ground_native_z FROM tank_engineering_registry"))
        if not reuse_pixels:
            build_pixels(db, [{"endpoint": wanted[key]} for key in sorted(additional)], packet, phase=phase)
    with closing(read_only(packet)) as db:
        db.row_factory = sqlite3.Row
        rows, kernels = inputs(db)
        scenes = {r["scene_id"]: r for r in db.execute("SELECT * FROM scene")}
    observations = {(r["candidate_id"], r["version_id"], scenes[r["scene_id"]]["scene_item_id"]): r for r in rows}
    if observations.keys() != additional.keys() or len(rows) != len(observations):
        raise ValueError("The additional native chip packet does not match the input set")
    jobs = []
    for key, modes in additional.items():
        row = observations[key]
        if (scenes[row["scene_id"]]["crs"] != "EPSG:32614"
                or datetime.fromisoformat(row["captured_at_utc"]) != datetime.fromisoformat(wanted[key]["captured_at_utc"])):
            raise ValueError(f"Wrong source geometry or capture time for {key}")
        for mode in sorted(modes):
            jobs.append((key, mode, row))

    def fit(job):
        key, mode, row = job
        shell = shells[key[0]]
        if mode == DATED:
            old = geometry[key[0]]
            east, north = (float(old[field]) for field in ("center_east_m", "center_north_m"))
            radius = float(old["diameter_m"])/2
            if math.hypot(east-row["center_east_m"], north-row["center_north_m"]) + max(0, radius-row["radius_m"]) > CONTEXT_M-16:
                return dict(fit_status="dated_geometry_outside_chip", diameter_m=radius*2,
                            measurement_source=packet.name, observation_id=row["observation_id"])
            shell = float(old["prior_rim_z_p95_native"])-float(old["prior_ground_z_p50_native"])
            row = dict(row, center_east_m=east, center_north_m=north, radius_m=radius)
        return dict(estimate(row, shell, kernels[row["platform"]], calibration_db=packet), measurement_source=packet.name)

    estimates = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, (job, result) in enumerate(zip(jobs, pool.map(fit, jobs), strict=True), 1):
            key, mode, _ = job
            estimates[mode, *key] = result
            if i % 1000 == 0 or i == len(jobs):
                print(f"Fit {i}/{len(jobs)} additional geometry/capture observations", flush=True)
    return estimates, packet


def prepare(phase, stem, *, reuse_pixels):
    weeks = study_weeks(phase)
    windows, grouped, dated, wanted, requested_modes = candidates(phase, weeks)
    reusable, geometry = reusable_estimates(phase)
    additional = {key: missing for key, modes in requested_modes.items()
                  if (missing := {mode for mode in modes if (mode, *key) not in reusable})}
    print(f"Reusing prior fits, {len(additional)} unique native chips require new measurements", flush=True)
    measured, packet = fit_additional(phase, stem, additional, wanted, geometry, reuse_pixels=reuse_pixels)
    estimates = {(mode, *key): measured.get((mode, *key), reusable.get((mode, *key)))
                 for key, modes in requested_modes.items() for mode in modes}
    if any(value is None for value in estimates.values()):
        raise ValueError("A required timing observation lacks a fit status")
    return weeks, windows, grouped, dated, wanted, estimates, packet


def closest_resolved(options, target, mode, wanted, estimates):
    choices = []
    for key, context_cells, _ in options:
        fit = estimates[mode, *key]
        if fit["fit_status"] == RESOLVED:
            source = wanted[key]
            captured = datetime.fromisoformat(source["captured_at_utc"])
            choices.append((abs(captured-target), -context_cells, key, captured))
    if not choices:
        return None
    _, _, key, captured = min(choices)
    return captured, wanted[key], estimates[mode, *key]


def interpolated_gap(before, after, at):
    first, last = before[0], after[0]
    if not first <= at < last:
        raise ValueError("The target is outside the two capture times")
    weight = (at-first)/(last-first)
    a, b = (float(obs[2]["estimated_gap_m"]) for obs in (before, after))
    return a+(b-a)*weight, weight, (last-first).total_seconds()/3600


def capture_fields(selected):
    fields = {}
    for name, observation in selected.items():
        source, fit = observation[1:] if observation else ({}, {})
        fields.update({f"{name}_{key}": source.get(key) for key in (
            "version_id", "canonical_l2a_id", "captured_at_utc", "archive_available_at_utc", "geometry_available_at_utc")})
        fields.update({f"{name}_gap_m": fit.get("estimated_gap_m"),
                       f"{name}_measurement_source": fit.get("measurement_source")})
    return fields


def timing_rows(weeks, windows, grouped, dated, wanted, estimates):
    for key, window in sorted(windows.items()):
        week, tank = key
        previous, current, release, cutoff = weeks[week]
        before, inside, after = (grouped[key][role] for role in ROLES)
        for mode in MODES if dated[key] else (RETROSPECTIVE,):
            prior = closest_resolved(before, previous, mode, wanted, estimates)
            prior_after = closest_resolved(inside, previous, mode, wanted, estimates)
            recent = closest_resolved(inside, current, mode, wanted, estimates)
            subsequent = closest_resolved(after, current, mode, wanted, estimates)
            selected = dict(previous_before=prior, previous_after=prior_after, current_before=recent, current_after=subsequent)
            resolved = [obs for obs in selected.values() if obs]
            diameters = {float(obs[2]["diameter_m"]) for obs in resolved}
            if len(diameters) > 1:
                raise ValueError(f"The same tank has different diameters for {week}/{tank}/{mode}")
            area = math.pi*next(iter(diameters))**2/4 if diameters else None
            first_gap = last_gap = first_weight = last_weight = first_span = last_span = change = None
            observed = bool(prior and recent)
            bracketed = all(selected.values())
            if bracketed:
                first_gap, first_weight, first_span = interpolated_gap(prior, prior_after, previous)
                last_gap, last_weight, last_span = interpolated_gap(recent, subsequent, current)
                change = first_gap-last_gap
            raw = float(prior[2]["estimated_gap_m"])-float(recent[2]["estimated_gap_m"]) if observed else None
            outer = float(prior[2]["estimated_gap_m"])-float(subsequent[2]["estimated_gap_m"]) if prior and subsequent else None
            if bracketed:
                status = "both_boundaries_interpolated"
            elif observed:
                status = "pre_measurement_pair_only"
            elif outer is not None:
                status = "outer_pair_without_inner_observation"
            else:
                status = "insufficient_resolved_roof_observations"
            lidar_day, roof_day = dated[key] if mode == DATED else (None, None)
            yield dict(week_ending=week, release_date=release, inputs_available_before_utc=cutoff,
                candidate_id=tank, terminal_id=window["terminal_id"], geometry_mode=mode,
                lidar_evidence_date=lidar_day, roof_evidence_date=roof_day, timing_status=status,
                previous_measurement_at_utc=previous.isoformat(), current_measurement_at_utc=current.isoformat(),
                **capture_fields(selected), circular_area_m2=area, previous_interpolation_weight=first_weight,
                current_interpolation_weight=last_weight, previous_bracket_hours=first_span, current_bracket_hours=last_span,
                previous_interpolated_gap_m=first_gap, current_interpolated_gap_m=last_gap,
                interpolated_change_m=change, pre_measurement_pair_change_m=raw, outer_pair_change_m=outer,
                outer_pair_hours=(subsequent[0]-prior[0]).total_seconds()/3600 if outer is not None else None,
                interpolated_visible_kbbl=change*area/BARREL_M3/1000 if bracketed else None,
                pre_measurement_pair_visible_kbbl=raw*area/BARREL_M3/1000 if observed else None)


def fit_rows(wanted, estimates):
    for (mode, *raw_key), estimate_row in sorted(estimates.items()):
        source = wanted[tuple(raw_key)]
        yield dict(geometry_mode=mode, **{field: source[field] for field in (
            "candidate_id", "terminal_id", "canonical_l2a_id", "version_id", "captured_at_utc", "source_directory",
            "tile_metadata_path", "archive_available_at_utc", "geometry_available_at_utc")},
            **{field: estimate_row.get(field) for field in ("measurement_source", "observation_id", "fit_status",
                "diameter_m", "fitted_pixels", "estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m")})


def summarize_window(rows):
    paired = [r for r in rows if r["pre_measurement_pair_change_m"] is not None]
    bracketed = [r for r in paired if r["interpolated_change_m"] is not None]
    return dict(eligible_tank_weeks=len(rows), pre_measurement_pairs=len(paired), bracketed_pairs=len(bracketed),
        new_resolved_outer_pairs=sum(r["timing_status"] == "outer_pair_without_inner_observation" for r in rows),
        bracketed_circle_area_m2=sum(r["circular_area_m2"] for r in bracketed),
        bracketed_median_previous_hours=median(r["previous_bracket_hours"] for r in bracketed) if bracketed else None,
        bracketed_median_current_hours=median(r["current_bracket_hours"] for r in bracketed) if bracketed else None,
        interpolated_visible_kbbl=sum(r["interpolated_visible_kbbl"] for r in bracketed) if bracketed else None,
        matched_pre_measurement_visible_kbbl=sum(r["pre_measurement_pair_visible_kbbl"] for r in bracketed) if bracketed else None)


def output_panels(weeks, rows):
    weekly, terminals = defaultdict(list), defaultdict(list)
    for row in rows:
        key = row["week_ending"], row["geometry_mode"]
        weekly[key].append(row)
        terminals[(*key, row["terminal_id"])].append(row)
    weeks_out = [dict(week_ending=week, release_date=release, inputs_available_before_utc=cutoff,
                      geometry_mode=mode, **summarize_window(weekly[week, mode]))
                 for week, (_, _, release, cutoff) in weeks.items() for mode in MODES]
    terminals_out = [dict(week_ending=week, release_date=weeks[week][2], inputs_available_before_utc=weeks[week][3],
                          geometry_mode=mode, terminal_id=terminal, **summarize_window(items))
                     for (week, mode, terminal), items in sorted(terminals.items())]
    return weeks_out, terminals_out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "validation", "holdout"), default="development")
    parser.add_argument("--reuse-pixels", action="store_true")
    args = parser.parse_args()
    phase = args.phase
    stem = f"shadow_timing_{phase}"
    weeks, windows, grouped, dated, wanted, estimates, packet = prepare(phase, stem, reuse_pixels=args.reuse_pixels)
    pairs = list(timing_rows(weeks, windows, grouped, dated, wanted, estimates))
    weekly, terminals = output_panels(weeks, pairs)
    for suffix, rows in (("observations", list(fit_rows(wanted, estimates))), ("pairs", pairs),
                         ("weeks", weekly), ("terminals", terminals)):
        export(f"{stem}_{suffix}", rows)
    report = dict(study_phase=phase, reporting_weeks=len(weeks), candidate_tank_weeks=len(windows),
        unique_capture_tank_inputs=len(wanted), optical_geometry_observations=len(estimates),
        additional_native_packet=packet.name if packet else None,
        reused_geometry_observations=sum(row["measurement_source"].endswith(".csv") for row in estimates.values()),
        coverage={mode: dict(
            weeks_with_pre_measurement_pair=sum(r["pre_measurement_pairs"] > 0 for r in weekly if r["geometry_mode"] == mode),
            weeks_with_bracketed_pair=sum(r["bracketed_pairs"] > 0 for r in weekly if r["geometry_mode"] == mode),
            timing_status=Counter(r["timing_status"] for r in pairs if r["geometry_mode"] == mode),
            **summarize_window([r for r in pairs if r["geometry_mode"] == mode])) for mode in MODES},
        interpretation="Nearest observations with passing native shadow diagnostics are used on each side. "
        "The two EIA boundaries are interpolated linearly only within observed brackets. Both week-level sums "
        "use identical tanks. No inventory outcomes or equity returns are read.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
