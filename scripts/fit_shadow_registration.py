"""Estimate relative scene offsets from exterior patches and test on separate pixels."""
import argparse
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from audit_shadow_registration_pixels import index, patch_bounds
from build_shadow_registration_inputs import PATCH_M, PADDING_M, PIXEL_M
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import ROOT, write

OFFSETS_M = np.arange(-12, 13, 2)
MIN_PATCH_CORRELATION = .55
MAX_PATCH_DISAGREEMENT_M = 4.
MIN_PEAK_SEPARATION = .02
PEAK_SEPARATION_DISTANCE_M = 6.


def registration_grid():
    east, north = np.meshgrid(OFFSETS_M, OFFSETS_M)
    shifts = np.column_stack((east.ravel(), north.ravel()))
    return shifts, registration_sampling(shifts)


def registration_sampling(shifts, *, patch_m=PATCH_M, padding_m=PADDING_M, pixel_m=PIXEL_M):
    row, col = np.indices((patch_m//pixel_m,)*2) + padding_m//pixel_m
    x = col[None, :, :] + shifts[:, 0, None, None] / (2*pixel_m)
    y = row[None, :, :] - shifts[:, 1, None, None] / (2*pixel_m)
    x0, y0 = x.astype(int), y.astype(int)
    wx, wy = (x-x0).astype("float32"), (y-y0).astype("float32")
    return x0, y0, wx[..., None], wy[..., None]


SHIFTS, SAMPLING = registration_grid()
ZERO = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])


def shifted_gradients(signal, sampling=SAMPLING):
    y_gradient, x_gradient = np.gradient(signal)
    gradient = np.stack((x_gradient, y_gradient), axis=-1)
    x, y, wx, wy = sampling
    lower = (1-wx)*gradient[y, x] + wx*gradient[y, x+1]
    upper = (1-wx)*gradient[y+1, x] + wx*gradient[y+1, x+1]
    shifted = ((1-wy)*lower + wy*upper).reshape(len(x), -1, 2)
    shifted -= shifted.mean(axis=1, keepdims=True)
    shifted = shifted.reshape(len(x), -1)
    norm = np.linalg.norm(shifted, axis=1, keepdims=True)
    return np.divide(shifted, norm, where=norm > 0, out=np.zeros_like(shifted))


def peak(scores, *, shifts=SHIFTS, separation_distance_m=PEAK_SEPARATION_DISTANCE_M):
    best = int(np.argmax(scores))
    offset = shifts[best]
    far = np.linalg.norm(shifts-offset, axis=1) >= separation_distance_m
    separation = float(scores[best]-scores[far].max())
    status = ("weak_similarity" if scores[best] < MIN_PATCH_CORRELATION else
              "search_boundary" if abs(offset).max() == abs(shifts).max() else
              "ambiguous_peak" if separation < MIN_PEAK_SEPARATION else "usable_peak")
    return best, separation, status


def requested_windows(phase, fixed_roofs=False):
    for row in read_csv("shadow_registration_pixel_coverage"):
        key = row["comparison"], row["study_phase"]
        if (phase and key == ("fixed_roof" if fixed_roofs else "weekly", phase)) or (
            not phase and key in (("fixed_roof", "development"), ("tandem", "engineering"))
        ):
            yield row


def patch_profiles(rows, sources, pairs, patches):
    requested = defaultdict(set)
    for row in rows:
        for role in ("fit", "spatial_check"):
            requested[row["scene_pair_id"]].update(filter(None, row[f"{role}_screened_patch_ids"].split("|")))

    rasters = {}

    @lru_cache(maxsize=350)
    def gradients(source_id, anchor_id):
        source, patch = sources[source_id], patches[anchor_id]
        raster = rasters[source_id]
        raw = raster.read(1, window=from_bounds(*patch_bounds(patch), raster.transform))
        expected = (PATCH_M + 2*PADDING_M)//PIXEL_M
        if raw.shape != (expected, expected) or np.isin(raw, [int(source["red_nodata_dn"]),
                                                             int(source["red_saturated_dn"])]).any():
            raise ValueError(f"Unexpected native registration chip: {source_id}/{anchor_id}")
        return shifted_gradients(raw.astype("float32"))

    profiles, observations = {}, []
    for number, (pair_id, ids) in enumerate(sorted(requested.items(), key=lambda kv: int(kv[0])), 1):
        pair = pairs[pair_id]
        if ids:
            with ExitStack() as stack:
                for side in ("earlier", "later"):
                    source_id = pair[f"{side}_source_id"]
                    rasters[source_id] = stack.enter_context(rasterio.open(ROOT / sources[source_id]["red_raster"]))
                for anchor_id in sorted(ids):
                    earlier = gradients(pair["earlier_source_id"], anchor_id)
                    later = gradients(pair["later_source_id"], anchor_id)
                    scores = np.clip(np.einsum("ij,ij->i", earlier[::-1], later), -1, 1)
                    best, separation, status = peak(scores)
                    profiles[pair_id, anchor_id] = scores, best, status
                    east, north = map(int, SHIFTS[best])
                    observations.append(dict(scene_pair_id=pair_id, anchor_id=anchor_id, role=patches[anchor_id]["role"],
                        later_minus_earlier_east_m=east, later_minus_earlier_north_m=north, status=status,
                        peak_correlation=float(scores[best]), zero_correlation=float(scores[ZERO]),
                        peak_separation=separation))
            rasters.clear()
        if number % 50 == 0 or number == len(requested):
            print(f"Matched exterior patches for {number}/{len(requested)} scene pairs", flush=True)
    return profiles, observations


def tank_offset(row, profiles, *, shifts=SHIFTS, zero=ZERO, disagreement_m=MAX_PATCH_DISAGREEMENT_M):
    pair_id = row["scene_pair_id"]
    groups = {}
    for role in ("fit", "spatial_check"):
        ids = filter(None, row[f"{role}_screened_patch_ids"].split("|"))
        groups[role] = [(scores, best) for scores, best, status in (profiles[pair_id, anchor] for anchor in ids)
                        if status == "usable_peak"]
    fit, check = groups["fit"], groups["spatial_check"]
    result = {field: row[field] for field in ("comparison", "study_phase", "week_or_control_day", "candidate_id", "scene_pair_id")}
    result.update(usable_fit_peaks=len(fit), usable_spatial_check_peaks=len(check), fit_inliers=0, spatial_check_inliers=0,
        later_minus_earlier_east_m=None, later_minus_earlier_north_m=None, fit_median_residual_m=None,
        check_median_residual_m=None, check_median_correlation_at_zero=None, check_median_correlation_at_offset=None,
        registration_status="insufficient_fit_peaks")
    if len(fit) < 2:
        return result
    fit_shifts = shifts[[best for _, best in fit]]
    centre = np.median(fit_shifts, axis=0)
    inlier = np.linalg.norm(fit_shifts-centre, axis=1) <= disagreement_m
    result["fit_inliers"] = int(inlier.sum())
    if inlier.sum() < 2 or inlier.sum()*2 <= len(fit):
        return dict(result, registration_status="inconsistent_fit_patches")
    # Choose a tested shift near the fit-only median. Break ties by fit-only patch correlation.
    centre = np.median(fit_shifts[inlier], axis=0)
    distances = np.linalg.norm(shifts-centre, axis=1)
    nearest = np.flatnonzero(distances == distances.min())
    inlier_profiles = np.asarray([scores for (scores, _), ok in zip(fit, inlier, strict=True) if ok])
    estimate = nearest[np.argmax(np.median(inlier_profiles[:, nearest], axis=0))]
    offset = shifts[estimate]
    residual = np.linalg.norm(fit_shifts-offset, axis=1)
    result.update(later_minus_earlier_east_m=int(offset[0]), later_minus_earlier_north_m=int(offset[1]),
                  fit_median_residual_m=float(np.median(residual)), registration_status="insufficient_check_peaks")
    if len(check) < 2:
        return result
    check_shifts = shifts[[best for _, best in check]]
    residual = np.linalg.norm(check_shifts-offset, axis=1)
    check_residual = float(np.median(residual))
    check_inliers = int((residual <= disagreement_m).sum())
    scores = np.asarray([p for p, _ in check])
    zero_score, fitted_score = map(float, np.median(scores[:, (zero, estimate)], axis=0))
    accepted = (check_residual <= disagreement_m and check_inliers >= 2 and
                fitted_score >= MIN_PATCH_CORRELATION and fitted_score >= zero_score)
    return dict(result, spatial_check_inliers=check_inliers, check_median_residual_m=check_residual,
                check_median_correlation_at_zero=zero_score, check_median_correlation_at_offset=fitted_score,
                registration_status="checked_offset" if accepted else "offset_fails_spatial_check")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "validation", "holdout"))
    parser.add_argument("--fixed-roof-controls", action="store_true")
    args = parser.parse_args()
    phase = args.phase
    if args.fixed_roof_controls and not phase:
        parser.error("--fixed-roof-controls requires a study phase")
    scope_name = f"fixed_roof_{phase}" if args.fixed_roof_controls else (phase or "control")
    stem = f"shadow_registration_{scope_name}"
    sources, pairs, patches = (index(table, field) for table, field in (
        ("sources", "source_id"), ("scene_pairs", "scene_pair_id"), ("patches", "anchor_id")))
    rows = list(requested_windows(phase, args.fixed_roof_controls))
    profiles, observations = patch_profiles(rows, sources, pairs, patches)
    offsets = [tank_offset(row, profiles) for row in rows]
    export(f"{stem}_patch_offsets", observations)
    export(f"{stem}_offsets", offsets)
    coverage = defaultdict(Counter)
    for row in offsets:
        coverage[row["comparison"]][row["registration_status"]] += 1
    if args.fixed_roof_controls:
        scope = f"Fixed roof engineering controls in {phase}; no EIA or market outcomes."
    else:
        scope = (f"Frozen weekly input pairs in {phase}." if phase else
                 "Controls comprise fixed roofs in development and the engineering tandem campaigns.")
    report = dict(windows=len(rows), patch_comparisons=len(observations), coverage=coverage,
        search_offset_m=[int(OFFSETS_M[0]), int(OFFSETS_M[-1])], step_m=2,
        minimum_patch_correlation=MIN_PATCH_CORRELATION, minimum_peak_separation=MIN_PEAK_SEPARATION,
        maximum_patch_disagreement_m=MAX_PATCH_DISAGREEMENT_M,
        interpretation="Offsets are later image feature positions minus earlier positions in UTM metres. "
        "The same symmetric interpolation is used for the two images. Check patches never select the fitted offset. "
        + scope)
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
