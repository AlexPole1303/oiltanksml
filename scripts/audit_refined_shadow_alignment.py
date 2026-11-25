"""Check finer exterior alignment on the same native Sentinel-2 control pixels."""
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
from statistics import median

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from audit_shadow_registration_pixels import index, patch_bounds
from build_shadow_registration_inputs import PATCH_M, PADDING_M, PIXEL_M
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from check_spectral_shadow_controls import STEM as CONTROL_STEM, control_key
from collect_public import ROOT, write
from fit_shadow_registration import MAX_PATCH_DISAGREEMENT_M, registration_sampling, shifted_gradients

STEM = "sentinel2_refined_shadow_alignment"
OFFSETS = ("east", "north")
REFINE_STEP = .5
REFINE_REACH = 2.
STEMS = ("control", "fixed_roof_validation", "fixed_roof_holdout")


@lru_cache(maxsize=4096)
def sampling(offsets):
    return registration_sampling(np.asarray(offsets, dtype=float))


def scores(chips, offsets):
    east_north = tuple((float(east), float(north)) for east, north in offsets)
    opposite = tuple((-east, -north) for east, north in east_north)
    earlier = shifted_gradients(chips[0], sampling(opposite))
    later = shifted_gradients(chips[1], sampling(east_north))
    return np.clip(np.einsum("ij,ij->i", earlier, later), -1, 1)


def read_chips(rasters, sources, patch):
    values = []
    expected = (PATCH_M + 2*PADDING_M)//PIXEL_M
    for raster, source in zip(rasters, sources, strict=True):
        window = from_bounds(*patch_bounds(patch), raster.transform)
        samples = raster.read(1, window=window)
        if samples.shape != (expected, expected) or np.isin(samples, [int(source["red_nodata_dn"]),
                                                                     int(source["red_saturated_dn"])]).any():
            raise ValueError(f"Invalid native alignment chip {source['version_id']}/{patch['anchor_id']}")
        values.append(samples.astype("float32"))
    return tuple(values)


def patch_fits(pairs, sources, patches, requested, old):
    trial = np.arange(-REFINE_REACH, REFINE_REACH + REFINE_STEP/2, REFINE_STEP)
    grid = np.column_stack([v.ravel() for v in np.meshgrid(trial, trial)])
    for pair_id, ids in sorted(requested.items(), key=lambda item: int(item[0])):
        pair = pairs[pair_id]
        data = [sources[pair[f"{side}_source_id"]] for side in ("earlier", "later")]
        with ExitStack() as stack:
            rasters = [stack.enter_context(rasterio.open(ROOT / source["red_raster"])) for source in data]
            chips, records = {}, {}
            for anchor in sorted(ids):
                coarse = old[pair_id, anchor]
                if coarse["status"] != "usable_peak":
                    continue
                original = np.asarray([float(coarse[f"later_minus_earlier_{axis}_m"]) for axis in OFFSETS])
                patch = chips[anchor] = read_chips(rasters, data, patches[anchor])
                tested = original + grid
                result = scores(patch, tested)
                centre = len(grid)//2
                if not np.isclose(result[centre], float(coarse["peak_correlation"]), atol=2e-6):
                    raise ValueError(f"Original gradient comparison differs for {pair_id}/{anchor}")
                best = int(np.argmax(result))
                stable = abs(grid[best]).max() < REFINE_REACH
                records[anchor] = dict(scene_pair_id=pair_id, anchor_id=anchor, role=coarse["role"],
                    coarse_east_m=original[0], coarse_north_m=original[1], coarse_correlation=float(result[centre]),
                    refined_east_m=tested[best, 0], refined_north_m=tested[best, 1],
                    refined_correlation=float(result[best]), refinement_status="interior_peak" if stable else "search_boundary")
        yield pair_id, records, chips


def robust_offset(patches, minimum):
    values = np.asarray([[r["refined_east_m"], r["refined_north_m"]] for r in patches
                         if r["refinement_status"] == "interior_peak"])
    if len(values) < minimum:
        return None
    close = np.linalg.norm(values-np.median(values, axis=0), axis=1) <= MAX_PATCH_DISAGREEMENT_M
    if close.sum() < minimum or close.sum()*2 <= len(values):
        return None
    return tuple(np.round(np.median(values[close], axis=0)/REFINE_STEP)*REFINE_STEP)


def refine_windows(windows, pairs, sources, patches, original_patches):
    by_scene = defaultdict(list)
    requested = defaultdict(set)
    for row in windows:
        pair = row["scene_pair_id"]
        fit, check = (tuple(filter(None, row[f"{role}_screened_patch_ids"].split("|"))) for role in ("fit", "spatial_check"))
        by_scene[pair].append((row, fit, check))
        requested[pair].update((*fit, *check))
    output, all_patches = [], []
    for pair, records, chips in patch_fits(pairs, sources, patches, requested, original_patches):
        all_patches.extend(records.values())
        scene = robust_offset([r for r in records.values() if r["role"] == "fit"], 3)

        @lru_cache(maxsize=None)
        def correlation(anchor, offset):
            return float(scores(chips[anchor], (offset,))[0])

        for row, fit, check in by_scene[pair]:
            coarse = tuple(float(row[f"{axis}_offset_m"]) for axis in OFFSETS)
            local = robust_offset([records[a] for a in fit if a in records], 2)
            check = [a for a in check if a in records]
            if len(check) < 2:
                raise ValueError(f"The old checked shift lacks independent patches at {control_key(row)}")
            result = {field: row[field] for field in ("comparison", "study_phase", "week_or_control_day",
                                                      "candidate_id", "scene_pair_id", "roof_type")}
            result.update(check_patches=len(check), coarse_check_correlation=median(correlation(a, coarse) for a in check))
            for name, offset in (("coarse", coarse), ("local", local), ("scene", scene)):
                result.update({f"{name}_{axis}_m": offset[i] if offset is not None else None
                               for i, axis in enumerate(OFFSETS)})
                if name != "coarse":
                    result[f"{name}_check_correlation"] = median(correlation(a, offset) for a in check) if offset is not None else None
            output.append(result)
    return output, all_patches


def comparisons():
    controls = [r for r in read_csv(f"{CONTROL_STEM}_pairs") if r["red_change_m"]]
    windows = {control_key(r): r for r in read_csv("shadow_registration_pixel_coverage")
               if r["comparison"] in ("tandem", "fixed_roof")}
    original = {}
    for stem in STEMS:
        for row in read_csv(f"shadow_registration_{stem}_patch_offsets"):
            key = row["scene_pair_id"], row["anchor_id"]
            if key in original and row != original[key]:
                raise ValueError(f"Stored exterior match differs for {key}")
            original[key] = row
    pairs, sources, patches = (index(stem, field) for stem, field in (
        ("scene_pairs", "scene_pair_id"), ("sources", "source_id"), ("patches", "anchor_id")))
    selected = [dict(windows[control_key(row)], roof_type=row["roof_type"], east_offset_m=row["east_offset_m"],
                     north_offset_m=row["north_offset_m"]) for row in controls]
    return refine_windows(selected, pairs, sources, patches, original)


def main():
    rows, patches = comparisons()
    groups = defaultdict(list)
    for row in rows:
        label = f"tandem_{row['roof_type']}" if row["comparison"] == "tandem" else f"closed_{row['study_phase']}"
        groups[label].append(row)
    summaries = {}
    for label, sample in groups.items():
        summaries[label] = {}
        for name in ("local", "scene"):
            delta = [r[f"{name}_check_correlation"]-r["coarse_check_correlation"] for r in sample
                     if r[f"{name}_check_correlation"] is not None]
            summaries[label][name] = dict(comparisons=len(delta), median_change=median(delta) if delta else None,
                increased=sum(v > 1e-6 for v in delta), decreased=sum(v < -1e-6 for v in delta))
    export(f"{STEM}_pairs", rows)
    export(f"{STEM}_patches", patches)
    payload = json.dumps(dict(compared_controls=len(rows), usable_exterior_patches=len(patches),
        refinement_status=dict(Counter(r["refinement_status"] for r in patches)), groups=summaries,
        interpretation="Subpixel gradient correlation is evaluated on original native samples using symmetric "
        "bilinear interpolation. Offsets use fit patches only. The score for each roof is evaluated on distinct "
        "spatial-check patches. No roof measurements or EIA data choose the candidate offsets."), indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
