"""Package source pixels and audit the frozen validation and holdout sampling windows."""
import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime

from build_terminal_development_measurements import dated_pair, key, read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, build_pixels, export, read_only
from collect_public import write

PHASES = ("validation", "holdout")
BASE = "weekly_shadow_endpoint_manifest"
EXTENSION = "weekly_shadow_terminal_endpoint_manifest"
CORE_LEGACY = "core_terminal_legacy_pair_manifest"


def packet_path(phase):
    return DERIVED / f"weekly_shadow_{phase}.sqlite"


def phase_pairs(phase):
    pairs = {}
    for stem in (BASE, EXTENSION):
        group = defaultdict(dict)
        for row in read_csv(stem):
            if row["study_phase"] != phase:
                continue
            identity = row["week_ending"], row["candidate_id"]
            endpoint = row["endpoint"]
            if identity in pairs or endpoint in group[identity]:
                raise ValueError(f"Duplicate {phase} endpoint for {identity}")
            group[identity][endpoint] = row
        pairs.update(group)
    for identity, pair in pairs.items():
        if set(pair) != {"previous", "current"}:
            raise ValueError(f"Missing {phase} endpoint for {identity}")
        before, after = pair["previous"], pair["current"]
        for field in ("candidate_id", "terminal_id", "cohort_id", "roof_type", "release_date",
                      "inputs_available_before_utc"):
            if before[field] != after[field]:
                raise ValueError(f"Pair disagrees on {field} for {identity}")
        cutoff = datetime.fromisoformat(after["inputs_available_before_utc"])
        for row in pair.values():
            start, stop, capture, archive, geometry = (
                datetime.fromisoformat(row[field]) for field in (
                    "window_starts_after_utc", "window_ends_by_utc", "captured_at_utc",
                    "archive_available_at_utc", "geometry_available_at_utc"))
            if not start < capture <= stop < cutoff or max(capture, archive, geometry) >= cutoff:
                raise ValueError(f"Invalid {phase} input clocks for {identity}")
        if datetime.fromisoformat(before["window_ends_by_utc"]) != datetime.fromisoformat(after["window_starts_after_utc"]):
            raise ValueError(f"Non-adjacent {phase} measurement windows for {identity}")
        if datetime.fromisoformat(before["captured_at_utc"]) >= datetime.fromisoformat(after["captured_at_utc"]):
            raise ValueError(f"Invalid {phase} capture sequence for {identity}")
    return pairs


def engineering_evidence(phase, pairs):
    core = {}
    for row in read_csv(CORE_LEGACY):
        if row["study_phase"] != phase:
            continue
        identity = row["week_ending"], row["candidate_id"]
        if identity in core:
            raise ValueError(f"Duplicate {phase} engineering record for {identity}")
        pair = pairs[identity]
        for name, endpoint in pair.items():
            if (row[f"{name}_scene_item_id"] != endpoint["canonical_l2a_id"]
                    or row[f"{name}_version"] != endpoint["version_id"]):
                raise ValueError(f"Engineering record uses a different {name} product for {identity}")
        core[identity] = row
    return core


def input_manifest(pairs, core):
    for identity, pair in sorted(pairs.items()):
        before, after = pair["previous"], pair["current"]
        legacy = core.get(identity)
        extension = "lidar_2011_source_catalog_date" in after
        if legacy:
            lidar_date = legacy["lidar_source_date"]
            roof_date = legacy["roof_evidence_date"]
            eligibility = legacy["eligibility"]
        elif extension:
            lidar_date = after["lidar_2011_source_catalog_date"]
            roof_date = after["earliest_nei_roof_archive_date"]
            if after["cohort_id"] in ("enterprise_west_owned_efr", "sunoco_little_farm_efr"):
                roof_date = max(roof_date, after["source_document_date"]) if roof_date else ""
            eligibility = "dated_engineering_inputs" if dated_pair(pair) else "dated_engineering_inputs_unconfirmed"
        else:
            lidar_date, roof_date, eligibility = "", "", "outside_dated_efr_cohorts"
        if eligibility == "dated_engineering_inputs" and not (
            lidar_date and roof_date and max(lidar_date, roof_date) < after["inputs_available_before_utc"][:10]
        ):
            raise ValueError(f"Engineering evidence is not dated before the cutoff for {identity}")
        yield dict(
            study_phase=after["study_phase"], week_ending=identity[0], candidate_id=identity[1],
            terminal_id=after["terminal_id"], cohort_id=after["cohort_id"], roof_type=after["roof_type"],
            release_date=after["release_date"], inputs_available_before_utc=after["inputs_available_before_utc"],
            previous_eia_measurement_at_utc=before["window_ends_by_utc"],
            current_eia_measurement_at_utc=after["window_ends_by_utc"],
            previous_scene_item_id=before["canonical_l2a_id"], current_scene_item_id=after["canonical_l2a_id"],
            previous_version=before["version_id"], current_version=after["version_id"],
            previous_capture_at_utc=before["captured_at_utc"], current_capture_at_utc=after["captured_at_utc"],
            previous_archive_available_at_utc=before["archive_available_at_utc"],
            current_archive_available_at_utc=after["archive_available_at_utc"],
            previous_geometry_available_at_utc=before["geometry_available_at_utc"],
            current_geometry_available_at_utc=after["geometry_available_at_utc"],
            registry_geometry_basis=after["geometry_basis"], registry_diameter_m=after["diameter_m"],
            lidar_2011_source_date=lidar_date, roof_evidence_date=roof_date, engineering_eligibility=eligibility,
            source_endpoint_manifest=EXTENSION if extension else BASE)


def audit_packet(path, pairs):
    expected = {key(row): datetime.fromisoformat(row["captured_at_utc"])
                for pair in pairs.values() for row in pair.values()}
    with closing(read_only(path)) as db:
        actual = {}
        for candidate, version, scene, capture in db.execute("""
            SELECT o.candidate_id,s.version_id,s.scene_item_id,s.captured_at_utc
            FROM observation o JOIN scene s USING(scene_id)
        """):
            identity = candidate, version, scene
            if identity in actual:
                raise ValueError(f"Duplicate native observation {identity}")
            actual[identity] = datetime.fromisoformat(capture)
        if actual != expected:
            raise ValueError(f"Native observations differ from the phase manifest in {path.name}")
        return dict(native_observations=len(actual), granules=db.execute("SELECT count(*) FROM scene").fetchone()[0],
                    native_pixel_records=db.execute("SELECT count(*) FROM pixel").fetchone()[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--reuse-pixels", action="store_true")
    args = parser.parse_args()
    phase = args.phase
    pairs = phase_pairs(phase)
    manifest = list(input_manifest(pairs, engineering_evidence(phase, pairs)))
    path = packet_path(phase)
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        if not args.reuse_pixels:
            build_pixels(source, list(pairs.values()), path, phase=phase)
        weeks = {r[0] for r in source.execute(
            "SELECT week_ending FROM eia_satellite_window WHERE study_phase=? AND endpoint='current'", (phase,))}
    covered = {row["week_ending"] for row in manifest}
    if not covered <= weeks:
        raise ValueError(f"Pair weeks outside {phase} window calendar")
    summary = dict(
        study_phase=phase, reporting_weeks=len(weeks), weeks_with_input_pairs=len(covered),
        weeks_without_input_pairs=sorted(weeks - covered), input_pairs=len(manifest),
        pairs_by_roof_type=dict(sorted(Counter(row["roof_type"] for row in manifest).items())),
        pairs_by_engineering_eligibility=dict(sorted(Counter(row["engineering_eligibility"] for row in manifest).items())),
        packet=path.name, packet_bytes=path.stat().st_size, **audit_packet(path, pairs))
    export(f"weekly_shadow_{phase}_input_manifest", manifest)
    write(DERIVED / f"weekly_shadow_{phase}_input_summary.json",
          json.dumps(summary, ensure_ascii=False, indent=2).encode() + b"\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
