"""Export native radar height-response profiles and measure their repeatability."""
import argparse
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from audit_shadow_precision import describe
from build_radar_optical_cushing_controls import RADAR_PASS, radar_inputs
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from check_umbra_sicd_roof_repeatability import HEIGHT_STEP_M, N_SECTORS, arc_points, fit_inputs, sector_contrast
from collect_public import write


NAME = "cushing_radar_arc_profiles"
SECTOR_BLOCKS = np.array_split(np.arange(N_SECTORS), 4)
REGISTRATIONS = {
    "local": ("native_shift_row_px", "native_shift_column_px", "aligned_height_above_ground_m",
              "roof_use_status"),
    "scene": ("scene_registered_shift_row_px", "scene_registered_shift_column_px", "scene_registered_height_m",
              "scene_registered_use_status"),
}


def profiles(key, measurement, scenes):
    product, candidate = key
    sicd, image, origin, ground, rim, _, _ = fit_inputs(measurement)
    heights = np.arange(.75, rim - ground - .75 + 1e-8, HEIGHT_STEP_M)
    far = arc_points(sicd, measurement, ground + heights, near=False) - origin
    near = arc_points(sicd, measurement, ground + heights, near=True) - origin
    for mode, (dy, dx, prior_height, status) in REGISTRATIONS.items():
        if not measurement[prior_height]:
            continue
        offset = np.array([[float(measurement[dy]), float(measurement[dx])]])
        arc = sector_contrast(image, far, offset)[0]
        placebo = sector_contrast(image, near, offset)[0]
        if not (np.isfinite(arc).all() and np.isfinite(placebo).all()):
            raise ValueError(f"Incomplete height grid for {product}/{candidate}/{mode}")
        height = heights[np.argmax(np.median(arc, axis=-1))]
        if height != float(measurement[prior_height]):
            raise ValueError(f"Native profile disagrees with saved roof fit for {product}/{candidate}/{mode}")
        leave_block_out = []
        single_block = []
        for block in SECTOR_BLOCKS:
            remaining = np.setdiff1d(np.arange(N_SECTORS), block, assume_unique=True)
            leave_block_out.append(float(heights[np.argmax(np.median(arc[:, remaining], axis=-1))]))
            single_block.append(float(heights[np.argmax(np.median(arc[:, block], axis=-1))]))
        yield (dict(product_id=product, candidate_id=candidate, terminal_id=measurement["terminal_id"],
                    sensor=scenes[product]["radar_sensor"], roof_type=measurement["roof_type"],
                    capture_start_utc=measurement["capture_start_utc"], registration=mode,
                    shift_row_px=offset[0, 0], shift_column_px=offset[0, 1], fit_status=measurement[status],
                    lidar_shell_height_m=rim - ground, radius_m=measurement["radius_m"],
                    incidence_deg=measurement["incidence_deg"], height_m=height,
                    omitted_block_min_m=min(leave_block_out), omitted_block_max_m=max(leave_block_out),
                    omitted_block_height_span_m=max(leave_block_out) - min(leave_block_out),
                    omitted_block_max_displacement_m=max(abs(h - height) for h in leave_block_out),
                    omitted_block_heights_m_json=json.dumps(leave_block_out),
                    single_block_heights_m_json=json.dumps(single_block),
                    single_block_height_span_m=max(single_block) - min(single_block),
                    source_metadata_file=measurement["sicd_metadata_file"], source_window_file=measurement["window_file"],
                    source_window_sha256=measurement["window_sha256"]),
               np.column_stack((heights, arc, placebo)).astype(np.float32))


def optical_comparisons(index):
    for reference in read("radar_optical_cushing_control_comparisons"):
        if reference["selected_granule_for_metrics"] != "True" or reference["roof_type"] != "EFR":
            continue
        for mode in REGISTRATIONS:
            if reference[f"both_{mode}_optical_resolved"] != "True":
                continue
            arc = index[reference["radar_product_id"], reference["candidate_id"], mode]
            optical = float(reference["optical_height_m"])
            yield dict(product_id=arc["product_id"], candidate_id=arc["candidate_id"], registration=mode,
                       radar_capture_start_utc=arc["capture_start_utc"], optical_datatake=reference["optical_datatake"],
                       optical_version=reference["optical_version"], optical_observation_id=reference["optical_observation_id"],
                       maximum_separation_seconds=float(reference["maximum_separation_seconds"]),
                       radar_height_m=arc["height_m"], optical_height_m=optical,
                       radar_minus_optical_m=arc["height_m"] - optical,
                       omitted_block_height_span_m=arc["omitted_block_height_span_m"],
                       omitted_block_max_displacement_m=arc["omitted_block_max_displacement_m"],
                       optical_within_omitted_block_extremes=arc["omitted_block_min_m"] <= optical <= arc["omitted_block_max_m"])


def aerial_comparisons(index):
    inputs = (("naip_2025_umbra_roof_comparisons", "both_aerial_radar_scene_resolved"),
              ("ok_aerial_2025_umbra_roof_comparisons", "both_diagnostics_pass"))
    for name, status in inputs:
        for reference in read(name):
            if reference[status] != "True":
                continue
            arc = index[reference["product_id"], reference["candidate_id"], "scene"]
            aerial = float(reference["aerial_roof_height_m"])
            yield dict(product_id=arc["product_id"], candidate_id=arc["candidate_id"],
                       radar_capture_start_utc=arc["capture_start_utc"], reference_kind=reference["reference_kind"],
                       reference_id=reference["reference_id"], reference_day=reference["reference_day"],
                       reference_clock_status=reference.get("exterior_clock_status"),
                       reference_assets_json=reference["reference_assets_json"],
                       maximum_separation_hours=float(reference["maximum_separation_hours"]),
                       radar_height_m=arc["height_m"], aerial_height_m=aerial,
                       radar_minus_aerial_m=arc["height_m"] - aerial,
                       omitted_block_height_span_m=arc["omitted_block_height_span_m"],
                       aerial_within_omitted_block_extremes=arc["omitted_block_min_m"] <= aerial <= arc["omitted_block_max_m"])


def nearest_laser(scenes, measurements):
    radar_by_tank = defaultdict(list)
    capture_times = {key: tuple(datetime.fromisoformat(scene[k]) for k in ("capture_start_utc", "capture_end_utc"))
                     for key, scene in scenes.items()}
    for (product, tank), measurement in measurements.items():
        radar_by_tank[tank].append((product, measurement))
    for laser in read("cushing_icesat2_roof_profiles"):
        start, end = (datetime.fromisoformat(laser[k]) for k in ("granule_start_utc", "granule_end_utc"))
        if end < start:
            raise ValueError(f"Reversed laser acquisition interval {laser['cmr_id']}")
        best = {}
        for product, measurement in radar_by_tank[laser["candidate_id"]]:
            a, b = capture_times[product]
            sensor = scenes[product]["radar_sensor"]
            radar_pass = measurement["scene_registered_use_status"] == RADAR_PASS
            worst = max(abs((a - end).total_seconds()), abs((b - start).total_seconds())) / 3600
            best_case = max(0., (a - end).total_seconds(), (start - b).total_seconds()) / 3600
            key, rank = (sensor, radar_pass), (worst, product)
            if key not in best or rank < best[key][0]:
                best[key] = (rank, dict(product_id=product, candidate_id=laser["candidate_id"], sensor=sensor,
                    radar_capture_start_utc=a.isoformat(), radar_capture_end_utc=b.isoformat(),
                    laser_granule_id=laser["cmr_id"], laser_track=laser["track_id"], laser_beam=laser["beam"],
                    laser_granule_start_utc=start.isoformat(), laser_granule_end_utc=end.isoformat(),
                    laser_profile_pass=laser["passes_roof_profile_checks"] == "True", radar_scene_profile_pass=radar_pass,
                    minimum_separation_hours=best_case, maximum_separation_hours=worst,
                    laser_photon_packet=laser["photon_packet"], radar_source_window=measurement["window_file"]))
        yield from (row for _, row in sorted(best.values(), key=lambda result: result[0]))


def repeat_comparisons(index):
    for pair in read("umbra_sicd_all_roof_arc_close_pairs"):
        if pair["roof_type"] != "EFR":
            continue
        for mode, pass_field in (("local", "both_pass"), ("scene", "scene_registered_both_pass")):
            if pair[pass_field] != "True":
                continue
            first, last = (index[pair[f"{side}_product_id"], pair["candidate_id"], mode] for side in ("first", "last"))
            yield dict(first_product_id=first["product_id"], last_product_id=last["product_id"],
                       candidate_id=pair["candidate_id"], registration=mode, terminal_id=first["terminal_id"],
                       separation_seconds=float(pair["separation_seconds"]), first_height_m=first["height_m"],
                       last_height_m=last["height_m"], last_minus_first_m=last["height_m"] - first["height_m"],
                       max_omitted_block_height_span_m=max(first["omitted_block_height_span_m"], last["omitted_block_height_span_m"]),
                       omitted_block_ranges_overlap=(max(first["omitted_block_min_m"], last["omitted_block_min_m"])
                                                   <= min(first["omitted_block_max_m"], last["omitted_block_max_m"])))


def summary_rows(rows, *, span, error, inside, group):
    subsets = {"all": rows, "span_at_most_0_5m": [r for r in rows if r[span] <= .5],
               "span_above_0_5m": [r for r in rows if r[span] > .5]}
    return {name: dict(describe([r[error] for r in subset]),
                       distinct_capture_pairs=len({group(r) for r in subset}),
                       control_covered_by_profile_ranges=sum(r[inside] for r in subset))
            for name, subset in subsets.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    scenes, measurements = radar_inputs()
    keys = sorted(measurements, key=lambda k: (k[0], measurements[k]["window_file"], k[1]))
    rows, arrays = [], []
    profile_start = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(lambda key: list(profiles(key, measurements[key], scenes)), keys)
        for variants in results:
            for row, array in variants:
                row.update(profile_start_index=profile_start, profile_height_steps=len(array))
                rows.append(row)
                arrays.append(array)
                profile_start += len(array)
    index = {(r["product_id"], r["candidate_id"], r["registration"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate radar profile key")
    optical, repeat = list(optical_comparisons(index)), list(repeat_comparisons(index))
    aerial, laser = list(aerial_comparisons(index)), list(nearest_laser(scenes, measurements))
    export(NAME, rows)
    export(NAME + "_optical_comparisons", optical)
    export(NAME + "_repeat_comparisons", repeat)
    export(NAME + "_aerial_comparisons", aerial)
    export(NAME + "_laser_time_matches", laser)
    matrix = np.concatenate(arrays)
    packed = io.BytesIO()
    np.savez_compressed(packed, height_m=matrix[:, 0], far_arc_db=matrix[:, 1:1 + N_SECTORS],
                       near_arc_db=matrix[:, 1 + N_SECTORS:])
    write(DERIVED / (NAME + ".npz"), packed.getvalue())
    coverage = defaultdict(int)
    for r in rows:
        coverage[(r["sensor"], r["registration"], "EFR" if r["roof_type"] == "EFR" else "covered")] += 1
    summary = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                   input_tank_scene_observations=len(measurements), profile_rows=len(rows),
                   height_grid_rows=len(matrix), coverage={"/".join(k): v for k, v in sorted(coverage.items())},
                   resolved_profiles=sum(r["fit_status"] == RADAR_PASS for r in rows),
                   height_step_m=HEIGHT_STEP_M, sectors=N_SECTORS, omitted_blocks=[b.tolist() for b in SECTOR_BLOCKS],
                   repeat={}, optical={}, aerial={}, laser={},
                   interpretation="Sector omission measures sensitivity within one SAR acquisition. It does not "
                   "estimate uncertainty in geometry, registration, scatterer identity or inter-sensor timing. "
                   "Control subsets reuse capture pairs and are descriptive. The 0.5 m span is an audit stratum, not a selection rule.")
    for mode in REGISTRATIONS:
        summary["repeat"][mode] = summary_rows([r for r in repeat if r["registration"] == mode],
            span="max_omitted_block_height_span_m", error="last_minus_first_m", inside="omitted_block_ranges_overlap",
            group=lambda r: (r["first_product_id"], r["last_product_id"]))
        summary["optical"][mode] = {f"within_{hours}h": summary_rows(
            [r for r in optical if r["registration"] == mode and r["maximum_separation_seconds"] <= hours * 3600],
            span="omitted_block_height_span_m", error="radar_minus_optical_m", inside="optical_within_omitted_block_extremes",
            group=lambda r: (r["product_id"], r["optical_datatake"])) for hours in (1, 24)}
    for kind in sorted({r["reference_kind"] for r in aerial}):
        summary["aerial"][kind] = summary_rows([r for r in aerial if r["reference_kind"] == kind],
            span="omitted_block_height_span_m", error="radar_minus_aerial_m", inside="aerial_within_omitted_block_extremes",
            group=lambda r: (r["product_id"], r["reference_day"]))
    for sensor in sorted({r["sensor"] for r in laser}):
        sample = [r for r in laser if r["sensor"] == sensor]
        accepted = [r for r in sample if r["laser_profile_pass"] and r["radar_scene_profile_pass"]]
        summary["laser"][sensor] = dict(nearest_any_maximum_hours=min(r["maximum_separation_hours"] for r in sample),
            nearest_accepted_maximum_hours=min((r["maximum_separation_hours"] for r in accepted), default=None),
            jointly_accepted_within_24h=sum(r["maximum_separation_hours"] <= 24 for r in accepted))
    write(DERIVED / (NAME + "_summary.json"), (json.dumps(summary, indent=2) + "\n").encode())
    print(f"{len(rows)} profiles, {len(matrix)} grid heights, {len(optical)} optical, {len(aerial)} aerial, "
          f"{len(repeat)} repeat comparisons, {len(laser)} nearest laser matches.", flush=True)


if __name__ == "__main__":
    main()
