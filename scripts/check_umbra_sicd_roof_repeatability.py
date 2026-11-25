"""Measure radar arc visibility and roof-height repeatability without ML or EIA outcomes."""
import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
from scipy.ndimage import map_coordinates
from sarpy.io.complex.sicd_elements.SICD import SICDType

from audit_shadow_precision import describe
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import ROOT, write
from collect_umbra_sicd_windows import project

HEIGHT_STEP_M = 0.25
ARC_FLANK_M = 2.0
N_SECTORS = 12
ANGLE = np.deg2rad(np.linspace(-66, 66, N_SECTORS * 12, endpoint=False) + 132/(N_SECTORS*24))
OFFSET = np.array([(row, col) for row in range(-5, 6) for col in range(-5, 6)])
SCENE_OFFSET = np.array([(row, col) for row in range(-9, 10, 3) for col in range(-9, 10, 3)])
LOCAL_OFFSET = np.array([(row, col) for row in np.arange(-2.5, 2.51, .5)
                        for col in np.arange(-2.5, 2.51, .5)])
MIN_ARC_CONTRAST_DB = 1.25
MIN_PEAK_MARGIN_DB = 0.60
MIN_COMPETING_HEIGHT_M = 1.5
MAX_SECTOR_HEIGHT_SPAN_M = 1.25


@lru_cache(maxsize=18)
def source_image(path):
    with np.load(ROOT / path, allow_pickle=False) as source:
        values = np.abs(source["complex_amplitude"])
    return 20*np.log10(np.maximum(values, max(float(np.quantile(values, .02)), 1e-8)))


@lru_cache(maxsize=6)
def source_metadata(path):
    return SICDType.from_xml_string((ROOT / path).read_bytes())


def arc_points(sicd, tank, heights, near):
    """Project radial flanks and the nominal arc in the original SICD grid."""
    azimuth = np.deg2rad(sicd.SCPCOA.AzimAng) + ANGLE + (0 if near else np.pi)
    radii = float(tank["radius_m"]) + np.array([-ARC_FLANK_M, 0., ARC_FLANK_M])
    east = float(tank["east_m"]) + radii[:, None] * np.sin(azimuth)
    north = float(tank["north_m"]) + radii[:, None] * np.cos(azimuth)
    xyz = np.broadcast_arrays(east, north, heights[:, None, None])
    return project(sicd, *xyz)


def sector_contrast(image, points, offsets):
    shifted = points[None] + offsets[:, None, None, None, :]
    values = map_coordinates(image, np.moveaxis(shifted, -1, 0), order=1,
                             mode="constant", cval=np.nan, prefilter=False)
    difference = values[:, :, 1] - (values[:, :, 0] + values[:, :, 2])/2
    return np.median(difference.reshape(*difference.shape[:2], N_SECTORS, -1), axis=-1)


def fit_roof(image, far, near, heights, offset):
    contrast = sector_contrast(image, far, offset[None])[0]
    placebo = np.median(sector_contrast(image, near, offset[None])[0], axis=-1)
    scores = np.median(contrast, axis=-1)
    best = int(np.argmax(scores))
    competitors = abs(heights - heights[best]) >= MIN_COMPETING_HEIGHT_M
    margin = float(scores[best] - scores[competitors].max())
    sector_heights = [heights[np.argmax(np.median(contrast[:, select], axis=-1))] for select in
                      (slice(0, None, 2), slice(1, None, 2), slice(0, N_SECTORS//2), slice(N_SECTORS//2, None))]
    sector_span = float(np.ptp(sector_heights))
    score, wrong = float(scores[best]), float(placebo.max())
    if best in (0, len(heights) - 1):
        status = "best_height_at_boundary"
    elif score < MIN_ARC_CONTRAST_DB:
        status = "weak_arc"
    elif margin < MIN_PEAK_MARGIN_DB:
        status = "competing_roof_height"
    elif sector_span > MAX_SECTOR_HEIGHT_SPAN_M:
        status = "inconsistent_arc_sectors"
    elif wrong >= score:
        status = "near_arc_placebo_as_strong"
    else:
        status = "resolved_arc_diagnostic"
    return float(heights[best]), score, margin, sector_span, wrong, status


def fit_inputs(tank):
    if tank["native_ground_centre_nonzero"] != "True":
        raise ValueError(f"Missing source pixels at {tank['candidate_id']}")
    sicd = source_metadata(tank["sicd_metadata_file"])
    image = source_image(tank["window_file"])
    origin = np.array([float(tank["window_row_offset"]), float(tank["window_column_offset"])])
    ground, rim = (float(tank[k]) for k in ("ground_hae_approx_m", "top_hae_approx_m"))
    walkway = np.arange(rim-3.0, rim+1.01, HEIGHT_STEP_M)
    near_static = arc_points(sicd, tank, np.r_[ground, walkway], near=True) - origin
    return sicd, image, origin, ground, rim, walkway, near_static


def static_arc_scores(image, points, offsets):
    contrast = np.median(sector_contrast(image, points, offsets), axis=-1)
    if not np.isfinite(contrast).all():
        raise ValueError("Static arc sampling outside the downloaded source window")
    # Require a common displacement for the ground and the outer wall walkway.
    return contrast[:, 0] + contrast[:, 1:].max(axis=1), contrast


def coarse_registration(tank):
    _, image, _, _, _, _, near_static = fit_inputs(tank)
    score, _ = static_arc_scores(image, near_static, SCENE_OFFSET)
    center = float(np.median(score))
    scale = float(np.quantile(score, .9) - center)
    return (score - center) / max(scale, 1.)


def scene_offsets(rows, profiles):
    samples = defaultdict(list)
    for row, profile in zip(rows, profiles, strict=True):
        if row["terminal_id"]:
            samples[row["product_id"], row["terminal_id"]].append(profile)
    by_scene = defaultdict(dict)
    for (product, terminal), sample in samples.items():
        if len(sample) >= 3:
            by_scene[product][terminal] = np.median(sample, axis=0)
    offsets = {}
    for product, terminal in samples:
        donors = [profile for site, profile in by_scene[product].items() if site != terminal]
        if len(donors) >= 3:
            score = np.median(donors, axis=0)
            center = SCENE_OFFSET[int(np.argmax(score))]
            if np.abs(center).max() < np.abs(SCENE_OFFSET).max():
                offsets[product, terminal] = center
    return offsets


def fit_static(image, points, offsets, center):
    score, static_contrast = static_arc_scores(image, points, offsets)
    anchor = int(np.argmax(score))
    offset = offsets[anchor]
    upper = int(np.argmax(static_contrast[anchor, 1:]))
    bottom_score = float(static_contrast[anchor, 0])
    upper_score = float(static_contrast[anchor, 1 + upper])
    anchor_pass = (min(bottom_score, upper_score) >= MIN_ARC_CONTRAST_DB
                   and np.abs(offset - center).max() < np.abs(offsets - center).max())
    return offset, upper, bottom_score, upper_score, anchor_pass


def fit(tank, scene_center):
    sicd, image, origin, ground, rim, walkway, near_static = fit_inputs(tank)
    offset, upper, bottom_score, upper_score, anchor_pass = fit_static(image, near_static, OFFSET, np.zeros(2))
    relative = np.arange(0.75, rim-ground-0.75+1e-8, HEIGHT_STEP_M)
    far = arc_points(sicd, tank, ground+relative, near=False) - origin
    near = arc_points(sicd, tank, ground+relative, near=True) - origin
    aligned = fit_roof(image, far, near, relative, offset)
    baseline = fit_roof(image, far, near, relative, np.zeros(2))
    consensus = dict(scene_registration_available=False, scene_center_row_px=None, scene_center_column_px=None,
        scene_registered_shift_row_px=None, scene_registered_shift_column_px=None, scene_registered_height_m=None,
        scene_registered_far_arc_contrast_db=None, scene_registered_sector_height_span_m=None,
        scene_registered_use_status="no_independent_terminal_offsets")
    if scene_center is not None:
        shift, _, _, _, anchor_ok = fit_static(image, near_static, LOCAL_OFFSET + scene_center, scene_center)
        roof = fit_roof(image, far, near, relative, shift)
        consensus.update(scene_registration_available=True, scene_center_row_px=int(scene_center[0]),
            scene_center_column_px=int(scene_center[1]), scene_registered_shift_row_px=shift[0],
            scene_registered_shift_column_px=shift[1], scene_registered_height_m=roof[0],
            scene_registered_far_arc_contrast_db=roof[1], scene_registered_sector_height_span_m=roof[3],
            scene_registered_use_status=roof[-1] if anchor_ok else "static_anchors_unresolved")
    return dict(candidate_id=tank["candidate_id"], terminal_id=tank["terminal_id"], roof_type=tank["roof_type"],
        product_id=tank["product_id"], capture_start_utc=tank["capture_start_utc"],
        incidence_deg=sicd.SCPCOA.IncidenceAng, near_azimuth_deg=sicd.SCPCOA.AzimAng,
        lidar_rim_minus_ground_m=rim-ground, native_shift_row_px=int(offset[0]), native_shift_column_px=int(offset[1]),
        static_bottom_contrast_db=bottom_score, static_upper_contrast_db=upper_score,
        walkway_minus_lidar_rim_m=float(walkway[upper]-rim), static_anchors_pass=anchor_pass,
        unaligned_height_above_ground_m=baseline[0], unaligned_status=baseline[-1],
        aligned_height_above_ground_m=aligned[0], far_arc_contrast_db=aligned[1], competing_height_margin_db=aligned[2],
        sector_height_span_m=aligned[3], near_arc_placebo_contrast_db=aligned[4], aligned_status=aligned[5],
        roof_use_status=aligned[5] if anchor_pass else "static_anchors_unresolved", **consensus)


def pair_comparisons(fits):
    by_key = {(r["product_id"], r["candidate_id"]): r for r in fits}
    summary, rows = [], []
    for pair in read("umbra_cushing_close_capture_pairs"):
        first, last = (pair[k] for k in ("first_product_id", "last_product_id"))
        ids = sorted({tank for product, tank in by_key if product == first} & {tank for product, tank in by_key if product == last})
        comparisons = []
        for tank in ids:
            before, after = by_key[first, tank], by_key[last, tank]
            if before["roof_type"] != after["roof_type"]:
                raise ValueError(f"Roof type changes across close captures for {tank}")
            scene_pair = before["scene_registration_available"] and after["scene_registration_available"]
            comparisons.append(dict(candidate_id=tank, terminal_id=before["terminal_id"], roof_type=before["roof_type"],
                first_product_id=first, last_product_id=last, separation_seconds=float(pair["separation_seconds"]),
                first_status=before["roof_use_status"], last_status=after["roof_use_status"],
                both_pass=before["roof_use_status"] == after["roof_use_status"] == "resolved_arc_diagnostic",
                aligned_height_difference_m=after["aligned_height_above_ground_m"]-before["aligned_height_above_ground_m"],
                unaligned_height_difference_m=after["unaligned_height_above_ground_m"]-before["unaligned_height_above_ground_m"],
                walkway_height_difference_m=after["walkway_minus_lidar_rim_m"]-before["walkway_minus_lidar_rim_m"],
                both_anchors_pass=before["static_anchors_pass"] and after["static_anchors_pass"],
                scene_registered_both_pass=before["scene_registered_use_status"] == after["scene_registered_use_status"] == "resolved_arc_diagnostic",
                scene_registered_height_difference_m=(after["scene_registered_height_m"]-before["scene_registered_height_m"]
                                                     if scene_pair else None)))
        rows.extend(comparisons)
        efr = [r for r in comparisons if r["roof_type"] == "EFR"]
        passed = [r for r in efr if r["both_pass"]]
        common_scene = [r for r in efr if r["scene_registered_height_difference_m"] is not None]
        passed_scene = [r for r in common_scene if r["scene_registered_both_pass"]]
        summary.append(dict(first_product_id=first, last_product_id=last,
            separation_seconds=float(pair["separation_seconds"]), matched_efr=len(efr), matched_efr_both_pass=len(passed),
            matched_fixed_roofs=len(comparisons)-len(efr), fixed_roofs_both_pass=sum(r["both_pass"] for r in comparisons if r["roof_type"] != "EFR"),
            all_efr_aligned=describe([r["aligned_height_difference_m"] for r in efr]),
            all_efr_unaligned=describe([r["unaligned_height_difference_m"] for r in efr]),
            paired_diagnostic_aligned=describe([r["aligned_height_difference_m"] for r in passed]),
            paired_diagnostic_unaligned=describe([r["unaligned_height_difference_m"] for r in passed]),
            scene_registered_same_tank_sample=describe([r["scene_registered_height_difference_m"] for r in common_scene]),
            local_registered_same_tank_sample=describe([r["aligned_height_difference_m"] for r in common_scene]),
            scene_registered_paired_diagnostic=describe([r["scene_registered_height_difference_m"] for r in passed_scene]),
            local_registered_on_scene_diagnostic_sample=describe([r["aligned_height_difference_m"] for r in passed_scene]),
            unaligned_on_scene_diagnostic_sample=describe([r["unaligned_height_difference_m"] for r in passed_scene]),
            scene_registered_fixed_roofs_both_pass=sum(r["scene_registered_both_pass"] for r in comparisons if r["roof_type"] != "EFR"),
            paired_walkways=describe([r["walkway_height_difference_m"] for r in comparisons if r["both_anchors_pass"]])))
    return rows, summary


def evaluate(source, workers=4):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        centers = scene_offsets(source, list(pool.map(coarse_registration, source)))
        return list(pool.map(lambda row: fit(row, centers.get((row["product_id"], row["terminal_id"]))), source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    index = "umbra_sicd_all" if args.all_scenes else "umbra_sicd_close_pass"
    output = "umbra_sicd_all_roof_arc" if args.all_scenes else "umbra_sicd_roof_arc"
    fits = evaluate(read(index + "_window_index"), args.workers)
    rows, pairs = pair_comparisons(fits)
    export(output + "_fits", fits)
    export(output + "_close_pairs", rows)
    by_scene = defaultdict(list)
    for row in fits:
        by_scene[row["product_id"]].append(row)
    pass_status = "resolved_arc_diagnostic"
    scene_counts = [dict(product_id=product, capture_start_utc=sample[0]["capture_start_utc"],
        incidence_deg=sample[0]["incidence_deg"], near_azimuth_deg=sample[0]["near_azimuth_deg"],
        efr_contours=sum(r["roof_type"] == "EFR" for r in sample),
        efr_local_passes=sum(r["roof_type"] == "EFR" and r["roof_use_status"] == pass_status for r in sample),
        efr_scene_registered_passes=sum(r["roof_type"] == "EFR" and r["scene_registered_use_status"] == pass_status for r in sample),
        covered_roof_contours=sum(r["roof_type"] != "EFR" for r in sample),
        covered_roof_local_false_passes=sum(r["roof_type"] != "EFR" and r["roof_use_status"] == pass_status for r in sample),
        covered_roof_scene_registered_false_passes=sum(r["roof_type"] != "EFR" and r["scene_registered_use_status"] == pass_status for r in sample))
        for product, sample in by_scene.items()]
    export(output + "_scene_coverage", scene_counts)
    result = dict(tank_scene_fits=len(fits), fixed_roof_negative_controls=sum(r["roof_type"] != "EFR" for r in fits),
        fixed_roof_false_passes=sum(r["roof_type"] != "EFR" and r["roof_use_status"] == "resolved_arc_diagnostic" for r in fits),
        scene_registered_fixed_roof_false_passes=sum(r["roof_type"] != "EFR" and r["scene_registered_use_status"] == "resolved_arc_diagnostic" for r in fits),
        status=dict(Counter(r["roof_use_status"] for r in fits)), close_capture_pairs=pairs,
        scene_registered_status=dict(Counter(r["scene_registered_use_status"] for r in fits)),
        scenes=len(scene_counts),
        interpretation="Engineering pilot with fixed geometric templates and locally registered static returns. "
        "The scene registration variant estimates the search center from at least three other terminals, then refines the two static arcs locally. "
        "Height differences on close captures measure repeatability, with possible actual movement over the capture separation. "
        "Outer-wall walkway may lie below the LiDAR rim. Neither an absolute roof-height validation nor an oil-volume estimate. "
        "Source imagery © Umbra Lab, CC BY 4.0.")
    write(DERIVED / (output + "_summary.json"), (json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
