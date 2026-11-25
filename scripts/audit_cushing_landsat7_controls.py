"""Identify dated Sentinel-2 controls and reporting-window coverage for Landsat 7."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import median

from collect_cushing_landsat7 import DERIVED, STEM, export, rows, save_json

CONTROL_LIMIT = timedelta(hours=48)


def sentinel_controls():
    by_roof = defaultdict(dict)
    path = DERIVED / "shadow_timing_development_observations.csv"
    for r in rows(path):
        if (r["geometry_mode"] != "dated_2011_lidar" or r["fit_status"] != "resolved_in_geometry_diagnostic"
            or not "2017" <= r["captured_at_utc"][:4] <= "2021"):
            continue
        time = datetime.fromisoformat(r["captured_at_utc"])
        available = max(datetime.fromisoformat(r[k]) for k in ("archive_available_at_utc", "geometry_available_at_utc"))
        group = by_roof[r["candidate_id"]]
        rank = available, r["version_id"]
        if time not in group or rank < group[time][0]:
            group[time] = rank, r
    return {key: (sorted(group), group) for key, group in by_roof.items()}


def collect_controls(optical, landsat):
    for r in landsat:
        candidate = r["candidate_id"]
        if candidate not in optical or r["angle_valid"] != "True" or not int(r["roof_pixels_passing_qa"]):
            continue
        capture = datetime.fromisoformat(r["captured_at_utc"])
        times, values = optical[candidate]
        offset = bisect_left(times, capture)
        choices = times[max(0, offset - 1):offset + 1]
        nearest = min(choices, key=lambda t: (abs(t - capture), t))
        gap = abs(nearest - capture)
        if gap > CONTROL_LIMIT:
            continue
        (known, _), control = values[nearest]
        all_clear = (int(r["roof_pixels_passing_qa"]) == int(r["roof_pixels"]) > 0 and
                     int(r["context_pixels_passing_qa"]) == int(r["context_pixels"]) > 0)
        yield dict(candidate_id=candidate, terminal_id=r["terminal_id"], landsat_product_id=r["product_id"],
            landsat_tier=r["product_id"].rsplit("_", 1)[-1], landsat_captured_at_utc=capture.isoformat(),
            landsat_archive_assets_created_by_utc=r["archive_assets_created_by_utc"],
            landsat_usable_roof_pixels=int(r["roof_pixels_passing_qa"]),
            landsat_usable_context_pixels=int(r["context_pixels_passing_qa"]),
            landsat_all_roof_and_context_pixel_centers_usable=all_clear,
            sentinel_version_id=control["version_id"], sentinel_canonical_id=control["canonical_l2a_id"],
            sentinel_tile_metadata=control["tile_metadata_path"], sentinel_captured_at_utc=nearest.isoformat(),
            sentinel_inputs_known_at_utc=known.isoformat(), absolute_separation_hours=gap.total_seconds() / 3600,
            sentinel_estimated_gap_m=float(control["estimated_gap_m"]), sentinel_fit_status=control["fit_status"])


def reporting_windows(scenes, overlaps, roof_windows):
    by_scene = defaultdict(set)
    for r in roof_windows:
        if r["angle_valid"] == "True" and int(r["roof_pixels_passing_qa"]) > 0:
            by_scene[r["product_id"]].add(r["candidate_id"])
    valid_scenes = sorted((datetime.fromisoformat(r["captured_at_utc"]), r["product_id"],
                          datetime.fromisoformat(r["archive_assets_created_by_utc"]))
                         for r in scenes if r["product_id"] in by_scene)
    capture_times = [r[0] for r in valid_scenes]
    for r in rows(DERIVED / "sentinel2_space_time_coverage_eia_comparison.csv"):
        week = r["week_ending"]
        if not "2017" <= week[:4] <= "2021" or r["study_phase"] != "development":
            continue
        previous, current, cutoff = (datetime.fromisoformat(r[k]) for k in
            ("previous_measurement_at_utc", "current_measurement_at_utc", "cutoff_utc"))
        candidates = valid_scenes[bisect_right(capture_times, previous):bisect_right(capture_times, current)]
        timely = [p for _, p, known in candidates if known <= cutoff]
        retrospective_roofs = set().union(*(by_scene[p] for _, p, _ in candidates))
        timely_roofs = set().union(*(by_scene[p] for p in timely))
        yield dict(week_ending=week, release_date=r["release_date"], cutoff_utc=cutoff.isoformat(),
            b04_eligible_tanks=int(r["baseline_b04_tanks"]), landsat7_acquired_scenes=len(candidates),
            landsat7_acquired_roofs=len(retrospective_roofs), landsat7_archive_scenes_known_at_cutoff=len(timely),
            landsat7_archive_roofs_known_at_cutoff=len(timely_roofs),
            eia_retro_tank_farm_change_kbbl=overlaps.get(week, ""))


def main():
    optical = sentinel_controls()
    landsat = rows(DERIVED / f"{STEM}_roof_windows.csv")
    scenes = rows(DERIVED / f"{STEM}_native_scenes.csv")
    controls = list(collect_controls(optical, landsat))
    estimates = {r["week_ending"]: r["net_tank_farm_change_kbbl"] for r in
                 rows(DERIVED / "cushing_eia_inventory_controls_weekly_tank_farm_estimates.csv")}
    weeks = list(reporting_windows(scenes, estimates, landsat))
    export(f"{STEM}_sentinel2_controls", controls)
    export(f"{STEM}_reporting_window_coverage", weeks)
    t1 = [r for r in controls if r["landsat_tier"] == "T1"]
    acquired = [r for r in weeks if r["landsat7_acquired_scenes"]]
    summary = dict(stage="engineering_development_2017_2021", closest_sentinel2_pairs_within_48h=len(controls),
        pairs_within_24h=sum(r["absolute_separation_hours"] <= 24 for r in controls),
        pairs_within_3h=sum(r["absolute_separation_hours"] <= 3 for r in controls),
        distinct_matched_roofs=len({r["candidate_id"] for r in controls}),
        distinct_matched_landsat7_scenes=len({r["landsat_product_id"] for r in controls}),
        tier1_pairs_within_48h=len(t1), tier1_pairs_with_all_roof_and_context_pixels_usable=sum(
            r["landsat_all_roof_and_context_pixel_centers_usable"] for r in t1),
        median_pair_separation_hours=median(r["absolute_separation_hours"] for r in controls),
        reporting_weeks=len(weeks), weeks_with_any_usable_landsat7_scene_by_capture=len(acquired),
        weeks_with_landsat7_scene_by_capture_and_no_b04_pair=sum(r["b04_eligible_tanks"] == 0 for r in acquired),
        weeks_with_landsat7_scene_by_capture_and_eia_tank_farm_change=sum(
            bool(r["eia_retro_tank_farm_change_kbbl"]) for r in acquired),
        weeks_with_any_in_week_landsat7_scene_by_archive_clock=sum(r["landsat7_archive_scenes_known_at_cutoff"] > 0 for r in weeks))
    save_json(DERIVED / f"{STEM}_controls_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
