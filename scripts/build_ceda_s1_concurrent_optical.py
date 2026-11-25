"""Build Sentinel-2 optical controls selected by time proximity to local CEDA SLC."""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from statistics import median

from audit_s2_sun_clock import xml_time
from build_l1c_quality import candidate_layer
from build_radar_optical_cushing_controls import optical_inventory
from build_s1_reference_matches import export, read
from build_shadow_calibration import SCHEMA, add_psf, add_scene
from build_shadow_change_controls import SELECT_QUALITY
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, read_only
from collect_public import ROOT, write
from collect_s2_l1c import source_metadata_path
from fit_shadow_geometry import estimate, inputs

STEM = "ceda_s1_concurrent_optical"
DATABASE = DERIVED / f"{STEM}.sqlite"
MAX_SECONDS = 24*3600


def radar_index():
    assets = {r["cdse_uuid"]: r for r in read("ceda_s1_slc_regional_assets")}
    layers = defaultdict(dict)
    for row in read("ceda_s1_slc_regional_windows"):
        key = row["product_id"], row["candidate_id"]
        layers[key][row["polarization"]] = row["cdse_uuid"]
    scenes = defaultdict(dict)
    for (product, tank), bands in layers.items():
        if set(bands) == {"VV", "VH"}:
            r = assets[bands["VV"]]
            a, b = map(datetime.fromisoformat, (r["capture_start_utc"], r["capture_end_utc"]))
            scenes[product][tank] = (a, b, bands)
    return scenes


def select(source, radar, registry, max_seconds=MAX_SECONDS):
    groups = defaultdict(dict)
    matches = []
    nearest = timedelta(seconds=max_seconds+120)
    for version, optical in optical_inventory(source).items():
        capture = datetime.fromisoformat(optical["captured_at_utc"])
        products = []
        for identity, tanks in radar.items():
            start, stop, _ = next(iter(tanks.values()))
            if start-nearest <= capture <= stop+nearest:
                products.append(identity)
        if not products:
            continue
        query = SELECT_QUALITY+"AND q.l1c_product_name=? AND q.scene_item_id=? ORDER BY q.candidate_id"
        for source_row in source.execute(query, (version, optical["canonical_l2a_id"])):
            row = dict(source_row)
            tank = row["candidate_id"]
            engineering = registry.get(tank)
            if (not engineering or row["documented_roof_type"] != "EFR"
                    or row["geometry_basis"] != "lidar_elevated_circle"
                    or engineering["has_both_dated_sources"] != "True"):
                continue
            if not (engineering["lidar_source_available_date"] <= capture.date().isoformat()
                    and engineering["roof_evidence_date"] <= capture.date().isoformat()):
                continue
            path = source_metadata_path(ROOT / row["tile_metadata_path"])
            actual = xml_time(str(path.relative_to(ROOT)))
            if abs((actual-capture).total_seconds()) > 120:
                raise ValueError(f"Sentinel-2 clocks disagree for {version}")
            for identity in products:
                if tank not in radar[identity]:
                    continue
                start, stop, bands = radar[identity][tank]
                difference = max(abs((actual-start).total_seconds()), abs((actual-stop).total_seconds()))
                if difference > max_seconds:
                    continue
                valid = (row["valid_cells"]==row["optical_cells"] and row["roof_clear_by_source_mask"]
                         and row["resolved_b04_detector"])
                if valid:
                    key = row["level"], version, row["optical_source_directory"]
                    groups[key][tank] = dict(row, optical_captured_at_utc=actual.isoformat())
                matches.append(dict(radar_product_id=identity, candidate_id=tank, terminal_id=engineering["terminal_id"],
                    radar_vv_burst_id=bands["VV"], radar_vh_burst_id=bands["VH"], radar_start_utc=start,
                    radar_end_utc=stop, optical_version=version, optical_item_id=optical["canonical_l2a_id"],
                    optical_capture_utc=actual, optical_catalog_capture_utc=capture,
                    optical_source_directory=row["optical_source_directory"], optical_source_metadata=str(path.relative_to(ROOT)),
                    maximum_separation_hours=difference/3600, optical_clear=bool(row["roof_clear_by_source_mask"]),
                    optical_single_detector=bool(row["resolved_b04_detector"]), optical_expected_pixels=row["optical_cells"],
                    optical_valid_pixels=row["valid_cells"], optical_source_quality_pass=bool(valid)))
    return matches, groups


def build_pixels(source, groups, database=DATABASE):
    temporary = database.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    candidates = dict((*candidate_layer("osm"), *candidate_layer("supplemental")))
    identities = {}
    try:
        with closing(sqlite3.connect(temporary)) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            db.executescript(SCHEMA)
            add_psf(db)
            for index, (key, rows) in enumerate(sorted(groups.items()), 1):
                add_scene(db, source, index, key, [rows[tank] for tank in sorted(rows)], candidates, identities)
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",) or db.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("Optical control pixel archive failed integrity checks")
        temporary.replace(database)
    finally:
        temporary.unlink(missing_ok=True)


def fits(registry, database=DATABASE):
    with closing(read_only(database)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
        sources = dict(db.execute("SELECT scene_id,source_directory FROM scene"))

    def fit(row):
        tank = row["candidate_id"]
        result = estimate(row, float(registry[tank]["shell_above_ground_m"]), kernels[row["platform"]], calibration_db=database)
        return (tank, row["version_id"], sources[row["scene_id"]]), result

    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(pool.map(fit, observations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hours", type=int, choices=(24,48), default=24)
    args = parser.parse_args()
    stem = STEM if args.max_hours==24 else f"{STEM}_{args.max_hours}h"
    database = DERIVED / f"{stem}.sqlite"
    max_seconds = args.max_hours*3600
    radar = radar_index()
    registry = {r["candidate_id"]: r for r in read("sentinel2_space_time_coverage_roof_registry")}
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        matches, groups = select(source, radar, registry, max_seconds)
        if not groups:
            raise ValueError("No source-qualified Sentinel-2 scenes in the radar control interval")
        build_pixels(source, groups, database)
    estimates = fits(registry, database)
    expected = {(r["candidate_id"], r["optical_version"], r["optical_source_directory"])
                for r in matches if r["optical_source_quality_pass"]}
    if set(estimates) != expected:
        raise ValueError("The stored fit population differs from the concurrent optical selection")
    for r in matches:
        key = r["candidate_id"], r["optical_version"], r["optical_source_directory"]
        f = estimates.get(key, {})
        r.update(optical_fit_status=f.get("fit_status", "source_quality_gate_failed"),
            optical_observation_id=f.get("observation_id"), optical_gap_to_rim_m=f.get("estimated_gap_m"),
            optical_gap_low_m=f.get("near_optimal_gap_min_m"), optical_gap_high_m=f.get("near_optimal_gap_max_m"),
            optical_fit_pass=f.get("fit_status")==RESOLVED)
    export(f"{stem}_pairs", matches)
    resolved = [r for r in matches if r["optical_fit_pass"]]
    unique = {(r["radar_product_id"], r["candidate_id"]) for r in resolved}
    by_tank = defaultdict(list)
    for r in resolved:
        by_tank[r["radar_product_id"], r["candidate_id"]].append(r)
    controls = []
    for (product, tank), values in sorted(by_tank.items()):
        if args.max_hours > 24:
            # Use tiles from the single nearest qualifying optical datatake.
            nearest = min(values, key=lambda r: (r["maximum_separation_hours"], r["optical_version"]))
            datatake = nearest["optical_version"].split("_")[:5]
            values = [r for r in values if r["optical_version"].split("_")[:5]==datatake]
        versions = sorted(values, key=lambda r: r["optical_version"])
        gaps = [r["optical_gap_to_rim_m"] for r in versions]
        controls.append(dict(radar_product_id=product, candidate_id=tank, terminal_id=versions[0]["terminal_id"],
            radar_vv_burst_id=versions[0]["radar_vv_burst_id"], radar_vh_burst_id=versions[0]["radar_vh_burst_id"],
            radar_capture_utc=versions[0]["radar_start_utc"], contributing_optical_tiles=len(versions),
            optical_versions_json=json.dumps([r["optical_version"] for r in versions]),
            optical_gap_to_rim_m=median(gaps), between_tile_gap_spread_m=max(gaps)-min(gaps),
            low_over_all_optical_fits_m=min(r["optical_gap_low_m"] for r in versions),
            high_over_all_optical_fits_m=max(r["optical_gap_high_m"] for r in versions),
            optical_capture_start_utc=min(r["optical_capture_utc"] for r in versions),
            optical_capture_end_utc=max(r["optical_capture_utc"] for r in versions),
            maximum_separation_hours=max(r["maximum_separation_hours"] for r in versions)))
    export(f"{stem}_controls", controls)
    summary = dict(maximum_search_hours=args.max_hours, screened_rows=len(matches),
        radar_scenes_screened=len({r["radar_product_id"] for r in matches}), optical_granules_with_clear_geometry=len(groups),
        source_qualified_observations=len(estimates), fit_status_counts=Counter(r["optical_fit_status"] for r in matches),
        resolved_optical_rows=len(resolved), unique_resolved_radar_tank_pairs=len(unique),
        resolved_radar_scenes=len({key[0] for key in unique}),
        distinct_terminals=len({r["terminal_id"] for r in controls}),
        controls_with_multiple_optical_tiles=sum(r["contributing_optical_tiles"]>1 for r in controls),
        controls_with_gap_agreement_within_1m=sum(r["contributing_optical_tiles"]>1 and
                                                  r["between_tile_gap_spread_m"]<=1 for r in controls),
        closest_resolved_pair_hours=min((r["maximum_separation_hours"] for r in resolved), default=None),
        maximum_resolved_pair_hours=max((r["maximum_separation_hours"] for r in resolved), default=None),
        interpretation="Independent optical shadow measurements. Time separation can include real changes in oil level. "
        "This construction uses satellite observations and dated engineering geometry, without EIA values or prices.")
    payload = json.dumps(summary, indent=2)+"\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
