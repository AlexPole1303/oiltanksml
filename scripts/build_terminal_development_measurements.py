"""Measure terminal roof changes with a physical shadow model on development inputs."""
import csv
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from statistics import median

from pyproj import Transformer

from build_terminal_weekly_inputs import STEM, TERMINALS
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, aggregate, build_pixels, export
from collect_public import write, write_csv
from fit_shadow_geometry import FIELDS, add_reference, choose_primary, estimate, inputs, metrics, read_only

STUDY = "weekly_shadow_terminal_development"
PIXELS = DERIVED / f"{STUDY}.sqlite"
CALIBRATION = DERIVED / "shadow_terminal_calibration.sqlite"
REFERENCE_DAY = "2022-12-30"
RETROSPECTIVE = "retrospective_2022_lidar"
DATED = "dated_2011_lidar"
MODES = (RETROSPECTIVE, DATED)
BARREL_M3 = 0.158987294928


def read_csv(name):
    with (DERIVED / f"{name}.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def key(row):
    return row["candidate_id"], row["version_id"], row["canonical_l2a_id"]


def endpoint_pairs():
    pairs = defaultdict(dict)
    for row in read_csv(STEM):
        if row["study_phase"] != "development":
            continue
        identity = row["week_ending"], row["candidate_id"]
        endpoint = row["endpoint"]
        if endpoint in pairs[identity] or row["terminal_id"] not in TERMINALS or row["roof_type"] != "EFR":
            raise ValueError(f"Invalid endpoint for {identity}")
        pairs[identity][endpoint] = row
        cutoff = datetime.fromisoformat(row["inputs_available_before_utc"])
        starts, ends, capture, archive, geometry = (
            datetime.fromisoformat(row[field]) for field in (
                "window_starts_after_utc", "window_ends_by_utc", "captured_at_utc",
                "archive_available_at_utc", "geometry_available_at_utc"))
        if not starts < capture <= ends < cutoff or max(archive, geometry) >= cutoff:
            raise ValueError(f"Endpoint clock mismatch for {identity}")
        if row["week_ending"] >= "2023-01-01" or capture.year >= 2023:
            raise ValueError(f"Endpoint is outside the development period for {identity}")
    for identity, pair in pairs.items():
        if set(pair) != {"previous", "current"}:
            raise ValueError(f"Missing endpoint for {identity}")
    return list(pairs.values())


def dated_pair(pair):
    return all(row["nei_roof_archive_dated_before_cutoff"] == "1"
               and row["roof_evidence_dated_before_cutoff"] == "True"
               and row["lidar_2011_source_dated_before_cutoff"] == "True"
               and row["earliest_nei_roof_archive_date"] < row["inputs_available_before_utc"][:10]
               and row["lidar_2011_source_catalog_date"] < row["inputs_available_before_utc"][:10]
               for row in pair.values())


def run_fits(jobs, kernels, path):
    def fit(job):
        mode, row, shell = job
        result = estimate(row, shell, kernels[row["platform"]], calibration_db=path)
        return dict(result, geometry_mode=mode, center_east_m=row["center_east_m"],
                    center_north_m=row["center_north_m"], outline_fit_status=row["outline_fit_status"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        for number, result in enumerate(pool.map(fit, jobs), 1):
            if number % 200 == 0 or number == len(jobs):
                print(f"Fit {number}/{len(jobs)} observations in {path.name}", flush=True)
            yield result


def weekly_fits(pairs, shells, legacy):
    all_inputs = {key(row): row for pair in pairs for row in pair.values()}
    dated_inputs = {key(row) for pair in pairs if dated_pair(pair) for row in pair.values()}
    with closing(read_only(PIXELS)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
        scenes = {r["scene_id"]: r for r in db.execute("SELECT scene_id,scene_item_id,crs FROM scene")}
    transformers = {crs: Transformer.from_crs(32614, crs, always_xy=True)
                    for crs in {r["crs"] for r in scenes.values()}}
    identities = {}
    unprocessed = set(all_inputs)
    jobs = []
    for row in observations:
        scene = scenes[row["scene_id"]]
        identity = row["candidate_id"], row["version_id"], scene["scene_item_id"]
        if identity not in unprocessed:
            raise ValueError(f"Unexpected or duplicate native chip for {identity}")
        unprocessed.remove(identity)
        endpoint = all_inputs[identity]
        identities[row["observation_id"]] = identity
        if datetime.fromisoformat(row["captured_at_utc"]) != datetime.fromisoformat(endpoint["captured_at_utc"]):
            raise ValueError(f"Native chip capture time mismatch for {identity}")
        if not math.isclose(2 * row["radius_m"], float(endpoint["lidar_2022_diameter_m"])):
            raise ValueError(f"Modern geometry differs from the manifest for {identity}")
        jobs.append((RETROSPECTIVE, row, shells[identity[0]]))
        if identity in dated_inputs:
            geometry = legacy[identity[0]]
            diameter = float(geometry["diameter_m"])
            if (not math.isclose(diameter, float(endpoint["lidar_2011_diameter_m"]))
                    or geometry["dataset"] != endpoint["lidar_dataset_from_dated_source"]):
                raise ValueError(f"Legacy geometry differs from the dated source for {identity}")
            east, north = transformers[scene["crs"]].transform(
                float(geometry["center_east_m"]), float(geometry["center_north_m"]))
            shell = float(geometry["prior_rim_z_p95_native"]) - float(geometry["prior_ground_z_p50_native"])
            jobs.append((DATED, dict(row, center_east_m=east, center_north_m=north, radius_m=diameter / 2,
                                     outline_fit_status=geometry["fit_status"]), shell))
    if unprocessed:
        raise ValueError("Native observations differ from the manifest")

    estimates = {}
    for result in run_fits(jobs, kernels, PIXELS):
        identity = identities[result["observation_id"]]
        endpoint = all_inputs[identity]
        mode = result["geometry_mode"]
        year = 2011 if mode == DATED else 2022
        estimates[mode, identity] = dict(result, terminal_id=endpoint["terminal_id"], canonical_l2a_id=identity[2],
            geometry_source_catalog_date=endpoint[f"lidar_{year}_source_catalog_date"])
    return estimates


def pair_measurements(pairs, estimates):
    for pair in pairs:
        earlier, later = pair["previous"], pair["current"]
        first, last = (datetime.fromisoformat(r["captured_at_utc"]) for r in (earlier, later))
        start, end = (datetime.fromisoformat(later[f"window_{field}_utc"]) for field in ("starts_after", "ends_by"))
        if not first <= start < last <= end:
            raise ValueError(f"Pair is outside the sampling windows for {key(later)}")
        for mode in MODES if dated_pair(pair) else (RETROSPECTIVE,):
            previous, current = (estimates[mode, key(row)] for row in (earlier, later))
            previous_gap, current_gap = (r.get("estimated_gap_m") for r in (previous, current))
            fitted = previous_gap is not None and current_gap is not None
            change = previous_gap - current_gap if fitted else None
            resolved = previous["fit_status"] == current["fit_status"] == RESOLVED
            area = math.pi * current["diameter_m"] ** 2 / 4
            scale = area / BARREL_M3
            yield dict(week_ending=later["week_ending"], release_date=later["release_date"], geometry_mode=mode,
                terminal_id=later["terminal_id"], candidate_id=later["candidate_id"], cohort_id=later["cohort_id"],
                previous_observation_id=previous["observation_id"], current_observation_id=current["observation_id"],
                previous_version=earlier["version_id"], current_version=later["version_id"],
                previous_canonical_l2a_id=earlier["canonical_l2a_id"], current_canonical_l2a_id=later["canonical_l2a_id"],
                previous_capture_at_utc=earlier["captured_at_utc"], current_capture_at_utc=later["captured_at_utc"],
                previous_archive_available_at_utc=earlier["archive_available_at_utc"],
                current_archive_available_at_utc=later["archive_available_at_utc"],
                inputs_available_before_utc=later["inputs_available_before_utc"],
                geometry_source_catalog_date=current["geometry_source_catalog_date"],
                nei_roof_archive_date=later["earliest_nei_roof_archive_date"],
                capture_interval_hours=(last - first).total_seconds() / 3600,
                eia_interval_fraction_observed=(last - start) / (end - start),
                previous_status=previous["fit_status"], current_status=current["fit_status"],
                previous_gap_m=previous_gap, current_gap_m=current_gap, both_shadow_diagnostics_pass=resolved,
                optical_level_change_m=change, circular_area_m2=area, circular_bbl_per_m=scale,
                optical_change_equivalent_bbl=change * scale if fitted else None,
                change_profile_min_m=previous["near_optimal_gap_min_m"] - current["near_optimal_gap_max_m"] if fitted else None,
                change_profile_max_m=previous["near_optimal_gap_max_m"] - current["near_optimal_gap_min_m"] if fitted else None)


def week_tables(source, rows):
    totals = defaultdict(list)
    by_terminal = defaultdict(list)
    for row in rows:
        week, mode, terminal = (row[field] for field in ("week_ending", "geometry_mode", "terminal_id"))
        totals[week, mode].append(row)
        by_terminal[week, mode, terminal].append(row)
    weeks = source.execute("""SELECT week_ending,release_date,inputs_available_before_utc
        FROM eia_satellite_window WHERE endpoint='current' AND study_phase='development' ORDER BY week_ending""").fetchall()
    weekly, terminals = [], []
    for week, release, cutoff in weeks:
        for mode in MODES:
            base = dict(week_ending=week, release_date=release, inputs_available_before_utc=cutoff, geometry_mode=mode)
            weekly.append(dict(base, **aggregate(totals[week, mode])))
            for terminal in TERMINALS:
                terminals.append(dict(base, terminal_id=terminal, **aggregate(by_terminal[week, mode, terminal])))
    return weekly, terminals


def lidar_check(shells):
    with closing(read_only(CALIBRATION)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
        alternative = {row["observation_id"]: row for row in db.execute("""SELECT * FROM b04_view_evidence
            WHERE method='three_native_nodes_inside_triangle' AND archive_single_detector_angle=0""")}
        jobs, sources = [], {}
        for row in observations:
            if row["captured_at_utc"][:10] != REFERENCE_DAY:
                continue
            identity = row["observation_id"]
            angle_source = "archive_bilinear"
            if identity in alternative:
                if row["resolved_b04_detector"]:
                    raise ValueError(f"Inferred view overwrites an archival angle for {identity}")
                angle = alternative[identity]
                row = dict(row, resolved_b04_detector=1, red_detector_id=angle["detector_id"],
                           red_view_zenith_deg=angle["zenith_deg"], red_view_azimuth_grid_deg=angle["azimuth_grid_deg"])
                angle_source = angle["method"]
            sources[identity] = angle_source if row["resolved_b04_detector"] else "unresolved"
            jobs.append((RETROSPECTIVE, row, shells[row["candidate_id"]]))
        estimates = list(run_fits(jobs, kernels, CALIBRATION))
        add_reference(db, estimates)
    for row in estimates:
        row["b04_view_source"] = sources[row["observation_id"]]
    choose_primary(estimates)
    return estimates


def summarize(rows, calibration):
    resolved = [r for r in rows if r["both_shadow_diagnostics_pass"]]
    primary = [r for r in calibration if r["selected_for_pass_metrics"] and r.get("absolute_error_m") is not None]
    by_mode = {}
    for mode in MODES:
        eligible = [r for r in rows if r["geometry_mode"] == mode]
        accepted = [r for r in resolved if r["geometry_mode"] == mode]
        by_mode[mode] = dict(pairs=len(eligible), resolved_pairs=len(accepted),
            weeks=len({r["week_ending"] for r in eligible}), resolved_weeks=len({r["week_ending"] for r in accepted}),
            tanks=len({r["candidate_id"] for r in eligible}), resolved_tanks=len({r["candidate_id"] for r in accepted}),
            pairs_by_terminal=Counter(r["terminal_id"] for r in eligible),
            resolved_pairs_by_terminal=Counter(r["terminal_id"] for r in accepted),
            median_resolved_eia_interval_fraction=median(r["eia_interval_fraction_observed"] for r in accepted) if accepted else None)
    matched = defaultdict(dict)
    for row in rows:
        matched[row["week_ending"], row["candidate_id"]][row["geometry_mode"]] = row
    shared = [pair for pair in matched.values() if len(pair) == len(MODES)]
    both_modes = [pair for pair in shared if all(r["both_shadow_diagnostics_pass"] for r in pair.values())]
    differences = [abs(pair[DATED]["optical_level_change_m"] - pair[RETROSPECTIVE]["optical_level_change_m"])
                   for pair in both_modes]
    return dict(development=by_mode, shared_input_pairs=len(shared), resolved_in_both_modes=len(both_modes),
        median_shared_absolute_change_difference_m=median(differences) if differences else None,
        december_lidar_check=dict(optical_granule_observations=len(calibration),
            all_primary=metrics(primary), resolved_primary=metrics([r for r in primary if r["fit_status"] == RESOLVED]),
            primary_by_b04_view_source=Counter(r["b04_view_source"] for r in primary),
            minimum_time_separation_hours=min((r["minimum_sensor_gap_hours"] for r in primary), default=None),
            maximum_time_separation_hours=max((r["maximum_sensor_gap_hours"] for r in primary), default=None)),
        scope="Development captures through 2022 only. Historical dates qualify the source of geometry and roof type. "
              "Both variants use the same mapped-footprint screening, selected L1C versions and current physical procedure. "
              "Weekly sums describe visible tanks with paired inputs. No EIA outcomes or market returns are joined.")


def main():
    pairs = endpoint_pairs()
    legacy = {r["candidate_id"]: r for r in read_csv("terminal_legacy_lidar_geometry")}
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        shells = dict(source.execute("SELECT candidate_id,lidar_rim_to_ground_native_z FROM tank_engineering_registry"))
        build_pixels(source, pairs, PIXELS)
        estimates = weekly_fits(pairs, shells, legacy)
        rows = list(pair_measurements(pairs, estimates))
        weeks, terminals = week_tables(source, rows)
    calibration = lidar_check(shells)
    geometry_fields = ("geometry_mode", "outline_fit_status", "center_east_m", "center_north_m")
    fields = ("terminal_id", "canonical_l2a_id", "geometry_source_catalog_date", *geometry_fields, *FIELDS)
    write_csv(DERIVED / f"{STUDY}_observations.csv", fields,
              ([r.get(field) for field in fields] for r in estimates.values()))
    fields = ("b04_view_source", *geometry_fields, *FIELDS)
    write_csv(DERIVED / "shadow_terminal_development_lidar_check.csv", fields,
              ([r.get(field) for field in fields] for r in calibration))
    export(f"{STUDY}_pairs", rows)
    export(f"{STUDY}_weeks", weeks)
    export(f"{STUDY}_terminals", terminals)
    report = json.dumps(summarize(rows, calibration), indent=2)
    write(DERIVED / f"{STUDY}_summary.json", (report + "\n").encode())
    print(report, flush=True)


if __name__ == "__main__":
    main()
