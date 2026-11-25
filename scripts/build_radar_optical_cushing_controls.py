"""Package nearby radar and Sentinel-2 captures and compare frozen physical fits."""
import argparse
import json
import math
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta

from audit_s2_sun_clock import xml_time
from audit_shadow_precision import describe
from build_l1c_quality import candidate_layer
from build_s1_reference_matches import read
from build_shadow_calibration import SCHEMA, add_psf, add_scene
from build_shadow_change_controls import SELECT_QUALITY
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, export, read_only
from collect_public import ROOT, write
from collect_s2_l1c import parse_product, source_metadata_path
from fit_shadow_geometry import estimate, inputs

DB = DERIVED / "radar_optical_cushing_controls.sqlite"
HOURS = 24
RADAR_PASS = "resolved_arc_diagnostic"


def radar_inputs():
    umbra = read("umbra_cushing_catalog")
    capella = json.loads((DERIVED / "capella_sicd_control_summary.json").read_bytes())["scene"]
    scenes = {r["product_id"]: dict(r, radar_sensor="Umbra") for r in umbra}
    scenes[capella["product_id"]] = dict(capella, radar_sensor="Capella",
        observed_metadata_and_sicd_version_timestamp_utc=capella["sicd_s3_last_modified_utc"])
    measurements = {}
    for fit_name, pixels_name in (("umbra_sicd_all_roof_arc_fits", "umbra_sicd_all_window_index"),
                                  ("capella_sicd_control_arc_fits", "capella_sicd_control_window_index")):
        windows = {(r["product_id"], r["candidate_id"]): r for r in read(pixels_name)}
        for r in read(fit_name):
            key = r["product_id"], r["candidate_id"]
            if key in measurements:
                raise ValueError(f"Duplicate radar observation {key}")
            w = windows[key]
            if (r["roof_type"], r["capture_start_utc"]) != (w["roof_type"], w["capture_start_utc"]):
                raise ValueError(f"Radar fit and native window identity disagree for {key}")
            measurements[key] = dict(w, **r)
    return scenes, measurements


def optical_inventory(source):
    rows = source.execute("""
        SELECT canonical_l2a_id,captured_at_utc,version_id,archive,archive_available_at_utc,
            geometry_available_at_utc,archive_source_url
        FROM sentinel2_l1c_input_inventory WHERE version_id IS NOT NULL AND local_folders_screened>0
          AND calibrated_bands=4
    """)
    inventory = {}
    for row in rows:
        if row["version_id"] in inventory:
            raise ValueError(f"Multiple inventory rows for {row['version_id']}")
        inventory[row["version_id"]] = dict(row)
    return inventory


def time_windows(source):
    return [(row[0], *(datetime.fromisoformat(t) for t in row[1:])) for row in source.execute("""
        SELECT week_ending,window_starts_after_utc,window_ends_by_utc,inputs_available_before_utc
        FROM eia_satellite_window WHERE endpoint='current' AND study_phase='holdout' ORDER BY week_ending
    """)]


def link_observations(source, scenes, radar, hours=HOURS):
    inventory = optical_inventory(source)
    schedule = time_windows(source)
    by_scene = defaultdict(set)
    for product, tank in radar:
        by_scene[product].add(tank)
    matches = []
    wanted = defaultdict(set)
    span = timedelta(hours=hours, minutes=2)
    for scene_id, scene in scenes.items():
        start, stop = map(datetime.fromisoformat, (scene["capture_start_utc"], scene["capture_end_utc"]))
        if not start < stop:
            raise ValueError(f"Invalid radar capture interval for {scene_id}")
        for version, optical in inventory.items():
            stamp = datetime.fromisoformat(optical["captured_at_utc"])
            if start-span <= stamp <= stop+span:
                wanted[version].add(scene_id)
    groups = defaultdict(dict)
    for version, products in sorted(wanted.items()):
        optical = inventory[version]
        capture_key, _, _ = parse_product(version)
        platform, sensing, orbit, _ = capture_key
        datatake = "_".join((platform, sensing, orbit))
        options = source.execute(SELECT_QUALITY+"AND q.l1c_product_name=? AND q.scene_item_id=? ORDER BY q.candidate_id",
                                 (version, optical["canonical_l2a_id"]))
        for raw in options:
            item = dict(raw)
            tank = item["candidate_id"]
            product_ids = [p for p in products if tank in by_scene[p]]
            if not product_ids:
                continue
            catalog_time = datetime.fromisoformat(item["optical_captured_at_utc"])
            if catalog_time != datetime.fromisoformat(optical["captured_at_utc"]):
                raise ValueError(f"Catalog sensing clock differs for {version}/{tank}")
            native_path = str(source_metadata_path(ROOT / item["tile_metadata_path"]).relative_to(ROOT))
            actual_time = xml_time(native_path)
            for product in sorted(product_ids):
                s = scenes[product]
                start, stop = map(datetime.fromisoformat, (s["capture_start_utc"], s["capture_end_utc"]))
                minimum = max(0., (start-actual_time).total_seconds(), (actual_time-stop).total_seconds())
                maximum = max(abs((actual_time-start).total_seconds()), abs((actual_time-stop).total_seconds()))
                if maximum > hours * 3600:
                    continue
                r = radar[product, tank]
                if (r["roof_type"] != item["documented_roof_type"] or item["geometry_basis"] != "lidar_elevated_circle"
                        or not math.isclose(float(r["radius_m"]), item["radius_m"], rel_tol=0, abs_tol=1e-6)):
                    raise ValueError(f"Radar and optical geometry disagree for {product}/{version}/{tank}")
                for axis in ("east", "north"):
                    if not math.isclose(float(r[axis+"_m"]), item["center_"+axis+"_m"], rel_tol=0, abs_tol=1e-6):
                        raise ValueError(f"Radar and optical centers disagree for {tank}")
                week, cutoff = next(((w, t) for w, lo, hi, t in schedule if lo < start <= hi), (None, None))
                radar_copy = datetime.fromisoformat(s["observed_metadata_and_sicd_version_timestamp_utc"])
                optical_clocks = tuple(datetime.fromisoformat(optical[k]) if optical[k] else None
                                       for k in ("archive_available_at_utc", "geometry_available_at_utc"))
                qualified = (item["valid_cells"] == item["optical_cells"] and item["roof_clear_by_source_mask"]
                             and item["resolved_b04_detector"])
                key = item["level"], version, item["optical_source_directory"]
                if qualified:
                    groups[key][tank] = dict(item, optical_captured_at_utc=actual_time.isoformat())
                matches.append(dict(radar_product_id=product, radar_sensor=s["radar_sensor"], candidate_id=tank,
                    terminal_id=r["terminal_id"], roof_type=r["roof_type"], radar_start_utc=start.isoformat(), radar_end_utc=stop.isoformat(),
                    radar_sicd_copy_time_utc=radar_copy.isoformat(), radar_sicd_metadata=r["sicd_metadata_file"],
                    radar_native_window=r["window_file"], radar_view_incidence_deg=r["incidence_deg"],
                    optical_version=version, optical_datatake=datatake, optical_scene_item_id=optical["canonical_l2a_id"],
                    optical_catalog_capture_utc=catalog_time.isoformat(), optical_native_granule_start_utc=actual_time.isoformat(),
                    optical_catalog_minus_native_seconds=(catalog_time-actual_time).total_seconds(),
                    optical_archive_available_at_utc=optical["archive_available_at_utc"],
                    optical_geometry_available_at_utc=optical["geometry_available_at_utc"],
                    optical_archive_source_url=optical["archive_source_url"], optical_native_metadata=native_path,
                    optical_source_directory=item["optical_source_directory"],
                    optical_source_quality_pass=bool(qualified), optical_clear_by_source_mask=bool(item["roof_clear_by_source_mask"]),
                    optical_single_b04_detector=bool(item["resolved_b04_detector"]), optical_roof_cells=item["optical_cells"],
                    optical_valid_four_band_roof_cells=item["valid_cells"],
                    minimum_separation_seconds=round(minimum, 6), maximum_separation_seconds=round(maximum, 6),
                    optical_minus_radar_start_seconds=round((actual_time-start).total_seconds(), 6),
                    radar_capture_reporting_week=week, original_release_cutoff_at_utc=cutoff.isoformat() if cutoff else None,
                    both_copies_available_before_original_cutoff=bool(cutoff and all(optical_clocks) and max(radar_copy, *optical_clocks) < cutoff),
                    shared_lidar_rim_to_ground_m=float(r["lidar_rim_minus_ground_m"]),
                    radar_local_fit_status=r["roof_use_status"], radar_local_height_m=float(r["aligned_height_above_ground_m"]),
                    radar_scene_fit_status=r["scene_registered_use_status"], radar_scene_height_m=(float(r["scene_registered_height_m"])
                        if r["scene_registered_height_m"] else None)))
    return matches, groups


def build_pixels(source, groups, database=DB):
    temporary = database.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    candidates = dict((*candidate_layer("osm"), *candidate_layer("supplemental")))
    identities = {}
    with closing(sqlite3.connect(temporary)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        add_psf(db)
        for index, (key, rows) in enumerate(sorted(groups.items()), 1):
            add_scene(db, source, index, key, [rows[tank] for tank in sorted(rows)], candidates, identities)
            if index % 10 == 0 or index == len(groups):
                print(f"Archived original optical pixels for {index}/{len(groups)} granule windows", flush=True)
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Radar and optical control pixel archive failed validation")
    temporary.replace(database)


def fit_optical(matches, workers, database=DB):
    shells = {r["candidate_id"]: r["shared_lidar_rim_to_ground_m"] for r in matches}
    with closing(read_only(database)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
        directories = dict(db.execute("SELECT scene_id,source_directory FROM scene"))

    def fit(row):
        return estimate(row, shells[row["candidate_id"]], kernels[row["platform"]], calibration_db=database)

    estimates = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, (o, fit_row) in enumerate(zip(observations, pool.map(fit, observations), strict=True), 1):
            key = o["candidate_id"], o["version_id"], directories[o["scene_id"]]
            estimates[key] = fit_row
            if idx % 500 == 0 or idx == len(observations):
                print(f"Fitted {idx}/{len(observations)} optical control observations", flush=True)
    expected = {(r["candidate_id"], r["optical_version"], r["optical_source_directory"])
                for r in matches if r["optical_source_quality_pass"]}
    if set(estimates) != expected:
        raise ValueError("Stored optical pixel identity differs from the matched radar manifest")
    return estimates


def comparison(matches, estimates):
    for r in matches:
        key = r["candidate_id"], r["optical_version"], r["optical_source_directory"]
        fit = estimates.get(key, {})
        gap = fit.get("estimated_gap_m")
        shell = r["shared_lidar_rim_to_ground_m"]
        height = shell - gap if gap is not None and r["roof_type"] == "EFR" else None
        r.update(optical_fit_status=fit.get("fit_status", "source_quality_gate_failed"),
            optical_observation_id=fit.get("observation_id"), optical_fitted_pixels=fit.get("fitted_pixels"),
            optical_sun_elevation_deg=fit.get("sun_elevation_deg"), optical_view_zenith_deg=fit.get("view_zenith_deg"),
            optical_wall_minus_roof_gap_m=gap, optical_height_m=height, selected_granule_for_metrics=False)
        for method in ("local", "scene"):
            radar_height = r[f"radar_{method}_height_m"]
            resolved = r["optical_fit_status"] == RESOLVED and r[f"radar_{method}_fit_status"] == RADAR_PASS
            r[f"both_{method}_optical_resolved"] = resolved
            r[f"radar_{method}_minus_optical_height_m"] = radar_height-height if height is not None and radar_height is not None else None
    by_overlapping_tiles = defaultdict(list)
    for r in matches:
        by_overlapping_tiles[r["radar_product_id"], r["candidate_id"], r["optical_datatake"]].append(r)
    for rows in by_overlapping_tiles.values():
        # Use input QA and context, without selecting a tile by either fitted height.
        best = min(rows, key=lambda r: (not r["optical_source_quality_pass"], -r["optical_valid_four_band_roof_cells"],
                                       r["maximum_separation_seconds"], r["optical_version"]))
        best["selected_granule_for_metrics"] = True
    return matches


def statistics(rows):
    efr = [r for r in rows if r["roof_type"] == "EFR"]
    return dict(observations=len(rows), efr_observations=len(efr), unique_efr_contours=len({r["candidate_id"] for r in efr}),
        optical_source_quality_efr=sum(r["optical_source_quality_pass"] for r in efr),
        optical_fit_resolved_efr=sum(r["optical_fit_status"] == RESOLVED for r in efr),
        radar_local_resolved_efr=sum(r["radar_local_fit_status"] == RADAR_PASS for r in efr),
        radar_scene_resolved_efr=sum(r["radar_scene_fit_status"] == RADAR_PASS for r in efr),
        local_optical_joint=describe([r["radar_local_minus_optical_height_m"] for r in efr if r["both_local_optical_resolved"]]),
        scene_optical_joint=describe([r["radar_scene_minus_optical_height_m"] for r in efr if r["both_scene_optical_resolved"]]),
        both_versions_before_original_release=sum(r["both_copies_available_before_original_cutoff"] for r in rows))


def change_comparisons(selected):
    by_day = defaultdict(list)
    for r in selected:
        if r["roof_type"] == "EFR":
            by_day[r["candidate_id"], r["radar_start_utc"][:10]].append(r)
    by_tank = defaultdict(list)
    for (tank, _), same_day in sorted(by_day.items()):
        best = min(same_day, key=lambda r: (not r["optical_source_quality_pass"], r["maximum_separation_seconds"],
                                          r["radar_product_id"], r["optical_version"]))
        by_tank[tank].append(best)
    for tank, series in by_tank.items():
        for before, after in zip(series, series[1:]):
            optical_elapsed = (datetime.fromisoformat(after["optical_native_granule_start_utc"])
                               - datetime.fromisoformat(before["optical_native_granule_start_utc"])).total_seconds()
            radar_elapsed = (datetime.fromisoformat(after["radar_start_utc"])
                             - datetime.fromisoformat(before["radar_start_utc"])).total_seconds()
            optical_ok = optical_elapsed > 0 and before["optical_datatake"] != after["optical_datatake"]
            optical_earlier, optical_later = (r["optical_height_m"] for r in (before, after))
            optical_change = optical_later-optical_earlier if optical_ok and None not in (optical_earlier, optical_later) else None
            result = dict(candidate_id=tank, terminal_id=after["terminal_id"],
                earlier_radar_product_id=before["radar_product_id"], later_radar_product_id=after["radar_product_id"],
                earlier_radar_start_utc=before["radar_start_utc"], later_radar_start_utc=after["radar_start_utc"],
                earlier_optical_version=before["optical_version"], later_optical_version=after["optical_version"],
                earlier_optical_start_utc=before["optical_native_granule_start_utc"], later_optical_start_utc=after["optical_native_granule_start_utc"],
                radar_elapsed_days=radar_elapsed/86400, optical_elapsed_days=optical_elapsed/86400,
                both_endpoints_maximum_separation_seconds=max(r["maximum_separation_seconds"] for r in (before, after)),
                distinct_ordered_optical_captures=optical_ok,
                earlier_optical_fit_status=before["optical_fit_status"], later_optical_fit_status=after["optical_fit_status"],
                optical_height_change_m=optical_change)
            for method in ("local", "scene"):
                earlier, later = (r[f"radar_{method}_height_m"] for r in (before, after))
                radar_change = later-earlier if None not in (earlier, later) else None
                result.update({f"earlier_radar_{method}_status": before[f"radar_{method}_fit_status"],
                               f"later_radar_{method}_status": after[f"radar_{method}_fit_status"],
                               f"radar_{method}_height_change_m": radar_change,
                               f"all_{method}_optical_fits_resolved": optical_ok and all(
                                   r[f"both_{method}_optical_resolved"] for r in (before, after)),
                               f"radar_{method}_minus_optical_change_m": radar_change-optical_change
                               if None not in (radar_change, optical_change) else None})
            yield result


def change_statistics(rows, method):
    jointly_resolved = [r for r in rows if r[f"all_{method}_optical_fits_resolved"]]
    return dict(candidate_adjacent_intervals=len(rows), resolved_tank_changes=len(jointly_resolved),
        distinct_radar_date_pairs=len({(r["earlier_radar_start_utc"][:10], r["later_radar_start_utc"][:10])
                                      for r in jointly_resolved}),
        resolved_tank_changes_with_both_matches_within_one_hour=sum(
            r["both_endpoints_maximum_separation_seconds"] <= 3600 for r in jointly_resolved),
        disagreement=describe([r[f"radar_{method}_minus_optical_change_m"] for r in jointly_resolved]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-pixels", action="store_true")
    parser.add_argument("--max-hours", type=int, choices=(24, 48), default=HOURS)
    args = parser.parse_args()
    stem = "radar_optical_cushing_control" if args.max_hours == HOURS else f"radar_optical_cushing_control_{args.max_hours}h"
    database = DB if args.max_hours == HOURS else DERIVED / f"{stem}.sqlite"
    radar_scenes, radar_observations = radar_inputs()
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        matches, groups = link_observations(source, radar_scenes, radar_observations, args.max_hours)
        if not args.reuse_pixels:
            build_pixels(source, groups, database)
    fits = fit_optical(matches, args.workers, database)
    comparisons = comparison(matches, fits)
    export(f"{stem}_comparisons", comparisons)
    selected = [r for r in comparisons if r["selected_granule_for_metrics"]]
    changes = list(change_comparisons(selected))
    export(f"{stem}_changes", changes)
    groups = defaultdict(list)
    for r in selected:
        groups[r["radar_product_id"], r["optical_datatake"]].append(r)
    scenes = [dict(radar_product_id=p, radar_sensor=rows[0]["radar_sensor"], radar_start_utc=rows[0]["radar_start_utc"],
        optical_datatake=d, optical_granules=len({r["optical_version"] for r in rows}),
        min_separation_seconds=min(r["minimum_separation_seconds"] for r in rows),
        max_separation_seconds=max(r["maximum_separation_seconds"] for r in rows),
        selected_pairs=len(rows), efr_observations=sum(r["roof_type"] == "EFR" for r in rows),
        optical_quality_passes=sum(r["optical_source_quality_pass"] for r in rows),
        efr_local_optical_joint=sum(r["roof_type"] == "EFR" and r["both_local_optical_resolved"] for r in rows),
        efr_scene_optical_joint=sum(r["roof_type"] == "EFR" and r["both_scene_optical_resolved"] for r in rows))
        for (p, d), rows in sorted(groups.items(), key=lambda item: item[1][0]["radar_start_utc"])]
    export(f"{stem}_capture_pairs", scenes)
    close = [r for r in selected if r["maximum_separation_seconds"] <= 3600]
    closest = min(scenes, key=lambda r: r["max_separation_seconds"])
    summary = dict(maximum_search_hours=args.max_hours, all_granule_tank_matches=len(comparisons), selected_datatake_tank_matches=len(selected),
        radar_captures_checked=len(radar_scenes), radar_captures_with_matched_optical=len({r["radar_product_id"] for r in selected}),
        distinct_radar_optical_capture_pairs=len(groups), closest_radar_optical_capture_pair=closest,
        radar_sensors_without_matched_optical=sorted({r["radar_sensor"] for r in radar_scenes.values()}
                                                     - {r["radar_sensor"] for r in selected}),
        archived_optical_observations=len(fits), by_sensor={sensor: statistics([r for r in selected if r["radar_sensor"] == sensor])
            for sensor in sorted({r["radar_sensor"] for r in selected})},
        within_one_hour=statistics(close),
        within_one_hour_scene_joint_capture_pairs=len({(r["radar_product_id"], r["optical_datatake"]) for r in close
                                                      if r["roof_type"] == "EFR" and r["both_scene_optical_resolved"]}),
        within_twenty_four_hours=statistics([r for r in selected if r["maximum_separation_seconds"] <= 24*3600]),
        adjacent_changes={method: change_statistics(changes, method) for method in ("local", "scene")},
        clocks=dict(granule_tank_matches_with_distinct_xml_time=sum(bool(r["optical_catalog_minus_native_seconds"]) for r in comparisons),
                    maximum_abs_catalog_xml_offset_seconds=max(abs(r["optical_catalog_minus_native_seconds"]) for r in comparisons)),
        interpretation="Retrospective engineering controls using independent sensors and a shared static LiDAR geometry. "
        "Sentinel-2 time is the native first line of the granule, with the radar acquisition interval retained. "
        "Differences include both sensor errors and possible physical movement between captures. "
        "Only reporting schedule timestamps are used. No EIA outcome, price or return data are read.")
    write(DERIVED / f"{stem}_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
