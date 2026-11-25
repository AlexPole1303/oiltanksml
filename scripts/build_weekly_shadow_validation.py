"""Apply the fixed optical geometry to archived phase pixels without EIA or price labels."""
import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from statistics import median

from pyproj import Transformer

from build_shadow_calibration import CONTEXT_M
from build_terminal_development_measurements import DATED, key, read_csv, run_fits
from build_weekly_shadow_panel import DERIVED, SOURCE, aggregate, export, pair_rows, read_only
from build_weekly_shadow_phase_inputs import PHASES, audit_packet, engineering_evidence, input_manifest, packet_path, phase_pairs
from collect_public import write
from fit_shadow_geometry import FIT_FIELDS, inputs

PHASE = "validation"
RETROSPECTIVE = "retrospective_registry_geometry"
MODES = (RETROSPECTIVE, DATED)


def legacy_geometry():
    geometries = {}
    for stem in ("core_terminal_legacy_lidar_geometry", "terminal_legacy_lidar_geometry"):
        for row in read_csv(stem):
            candidate = row["candidate_id"]
            if candidate in geometries:
                raise ValueError(f"Duplicate dated LiDAR geometry for {candidate}")
            geometries[candidate] = row
    return geometries


def estimate_pairs(pairs, dated_identities, shells, packet):
    all_endpoints = {key(row): row for pair in pairs.values() for row in pair.values()}
    dated = {key(row) for identity in dated_identities for row in pairs[identity].values()}
    legacy = legacy_geometry()
    with closing(read_only(packet)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
        scenes = {r["scene_id"]: (r["scene_item_id"], r["crs"])
                  for r in db.execute("SELECT scene_id,scene_item_id,crs FROM scene")}
    transforms = {crs: Transformer.from_crs(32614, crs, always_xy=True) for crs in {r[1] for r in scenes.values()}}
    identities, jobs = {}, []
    for row in observations:
        scene, crs = scenes[row["scene_id"]]
        identity = row["candidate_id"], row["version_id"], scene
        endpoint = all_endpoints[identity]
        if not math.isclose(2 * row["radius_m"], float(endpoint["diameter_m"])):
            raise ValueError(f"Registry radius mismatch for {identity}")
        identities[row["observation_id"]] = identity
        jobs.append((RETROSPECTIVE, row, shells[identity[0]]))
        if identity in dated:
            old = legacy[identity[0]]
            radius = float(old["diameter_m"]) / 2
            east, north = transforms[crs].transform(float(old["center_east_m"]), float(old["center_north_m"]))
            shift = math.hypot(east - row["center_east_m"], north - row["center_north_m"])
            if shift + max(0, radius - row["radius_m"]) > CONTEXT_M - 16:
                raise ValueError(f"Dated geometry needs a larger chip for {identity}")
            shell = float(old["prior_rim_z_p95_native"]) - float(old["prior_ground_z_p50_native"])
            jobs.append((DATED, dict(row, center_east_m=east, center_north_m=north, radius_m=radius,
                                    outline_fit_status=old["fit_status"]), shell))

    estimates = {mode: {} for mode in MODES}
    for result in run_fits(jobs, kernels, packet):
        identity = identities[result["observation_id"]]
        estimates[result["geometry_mode"]][identity] = dict(result, canonical_l2a_id=identity[2],
                                                            terminal_id=all_endpoints[identity]["terminal_id"])
    return estimates, legacy


def measurements(pairs, manifest, estimates, legacy, phase):
    rows = []
    for mode in MODES:
        selected = []
        for identity, pair in sorted(pairs.items()):
            if mode == DATED and manifest[identity]["engineering_eligibility"] != "dated_engineering_inputs":
                continue
            if mode == DATED:
                pair = {endpoint: dict(row, diameter_m=float(legacy[identity[1]]["diameter_m"]),
                                       geometry_basis="independent_2011_elevated_circle")
                        for endpoint, row in pair.items()}
            else:
                pair = {endpoint: dict(row, diameter_m=float(row["diameter_m"])) for endpoint, row in pair.items()}
            selected.append(pair)
        for row in pair_rows(selected, estimates[mode], phase=phase):
            evidence = manifest[row["week_ending"], row["candidate_id"]]
            previous, current = (datetime.fromisoformat(row[field])
                                 for field in ("earlier_capture_at_utc", "later_capture_at_utc"))
            start, end = (datetime.fromisoformat(evidence[f"{side}_eia_measurement_at_utc"])
                          for side in ("previous", "current"))
            capture_hours = (current - previous).total_seconds() / 3600
            overlap_hours = (current - start).total_seconds() / 3600
            eia_hours = (end - start).total_seconds() / 3600
            rows.append(dict(
                geometry_mode=mode, **row,
                previous_scene_item_id=evidence["previous_scene_item_id"],
                current_scene_item_id=evidence["current_scene_item_id"],
                previous_archive_available_at_utc=evidence["previous_archive_available_at_utc"],
                current_archive_available_at_utc=evidence["current_archive_available_at_utc"],
                previous_eia_measurement_at_utc=evidence["previous_eia_measurement_at_utc"],
                current_eia_measurement_at_utc=evidence["current_eia_measurement_at_utc"],
                capture_interval_hours=capture_hours, eia_interval_hours=eia_hours, overlap_hours=overlap_hours,
                fraction_of_capture_inside_eia_interval=overlap_hours / capture_hours,
                fraction_of_eia_interval_observed=overlap_hours / eia_hours,
                lidar_2011_source_date=evidence["lidar_2011_source_date"],
                roof_evidence_date=evidence["roof_evidence_date"],
                engineering_eligibility=evidence["engineering_eligibility"]))
    return rows


def weekly_measurements(weeks, rows, terminals):
    totals, groups = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["roof_type"] == "EFR":
            identity = row["week_ending"], row["geometry_mode"]
            totals[identity].append(row)
            groups[(*identity, row["terminal_id"])].append(row)
    weekly, by_terminal = [], []
    for week, release, cutoff in weeks:
        for mode in MODES:
            base = dict(week_ending=week, release_date=release, inputs_available_before_utc=cutoff, geometry_mode=mode)
            weekly.append(dict(base, **aggregate(totals[week, mode])))
            for terminal in terminals:
                by_terminal.append(dict(base, terminal_id=terminal, **aggregate(groups[week, mode, terminal])))
    return weekly, by_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default=PHASE)
    phase = parser.parse_args().phase
    packet = packet_path(phase)
    pairs = phase_pairs(phase)
    audit_packet(packet, pairs)
    manifest = {(row["week_ending"], row["candidate_id"]): row
                for row in input_manifest(pairs, engineering_evidence(phase, pairs))}
    dated = {identity for identity, row in manifest.items() if row["engineering_eligibility"] == "dated_engineering_inputs"}
    with closing(read_only(SOURCE)) as source:
        weeks = source.execute("""SELECT week_ending,release_date,inputs_available_before_utc
            FROM eia_satellite_window WHERE endpoint='current' AND study_phase=? ORDER BY week_ending""", (phase,)).fetchall()
        shells = dict(source.execute("SELECT candidate_id,lidar_rim_to_ground_native_z FROM tank_engineering_registry"))
    estimates, legacy = estimate_pairs(pairs, dated, shells, packet)
    observations = [
        dict(geometry_mode=mode, terminal_id=row["terminal_id"], canonical_l2a_id=row["canonical_l2a_id"],
             center_east_m=row["center_east_m"], center_north_m=row["center_north_m"], outline_fit_status=row["outline_fit_status"],
             **{field: row.get(field) for field in FIT_FIELDS})
        for mode, by_identity in estimates.items() for row in by_identity.values()
    ]
    rows = measurements(pairs, manifest, estimates, legacy, phase)
    terminals = sorted({row["terminal_id"] for row in manifest.values() if row["roof_type"] == "EFR"})
    weekly, by_terminal = weekly_measurements(weeks, rows, terminals)
    phase_summary = {}
    for mode in MODES:
        efr = [r for r in rows if r["geometry_mode"] == mode and r["roof_type"] == "EFR"]
        resolved_fractions = [r["fraction_of_eia_interval_observed"] for r in efr if r["both_shadow_diagnostics_pass"]]
        phase_summary[mode] = dict(
            fit_status_by_observation=dict(sorted(Counter(r["fit_status"] for r in estimates[mode].values()).items())),
            efr_pair_totals=aggregate(efr),
            efr_weeks_with_pairs=sum(r["input_efr_pairs"] > 0 for r in weekly if r["geometry_mode"] == mode),
            efr_weeks_with_resolved_pairs=sum(r["resolved_efr_pairs"] > 0 for r in weekly if r["geometry_mode"] == mode),
            median_eia_interval_fraction_observed=median(resolved_fractions) if resolved_fractions else None)
    summary = dict(
        reporting_weeks=len(weeks), terminals=len(terminals), observation_rows=len(observations), pair_rows=len(rows),
        modes=phase_summary)
    for suffix, data in (("observations", observations), ("pairs", rows), ("weeks", weekly), ("terminals", by_terminal)):
        export(f"weekly_shadow_{phase}_{suffix}", data)
    write(DERIVED / f"weekly_shadow_{phase}_summary.json", json.dumps(summary, indent=2).encode() + b"\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
