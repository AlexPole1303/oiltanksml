"""Compare red, near-infrared and joint shadow fits on native Oklahoma orthophoto."""
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from statistics import fmean, median

import numpy as np

from build_naip_2019_roof_reference import (MAX_CENTER_SHIFT, MAX_NEAR_SPAN, MIN_CONTRAST, MIN_DIRECTION_GAIN,
                                           MIN_GAIN, ROOF_SAMPLE_GAP, profile, signal_setup, sky)
from build_naip_2025_roof_reference import solar_module
from build_ok_aerial_2025_roof_controls import UTM, read_native_bands
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import ROOT, write

STEM = "ok_aerial_2025_spectral_shadow"
BANDS = ("red", "green", "nir")
METHODS = ("red", "nir", "joint")
BLOCK_METRES = 5


def indexed_crops():
    result = defaultdict(list)
    for row in read_csv("ok_aerial_2025_hr4_window_index"):
        for tank in json.loads(row["candidate_ids_json"]):
            key = tank, row["tile_majority_date"], row["tile_nominal_time_utc"]
            result[key].append((row["tile_id"], ROOT / row["crop_path"]))
    return result


def fold_profiles(x, y, channels, radius, bearing, ratio, gaps):
    profiles = {}
    for name, values in zip(BANDS, channels, strict=True):
        setup = signal_setup(x, y, values, radius)
        gain, contrast = profile(x, y, radius, (bearing,), (ratio,), gaps, setup)
        reverse, _ = profile(x, y, radius, (bearing + np.pi,), (ratio,), gaps, setup)
        profiles[name] = gain[0], contrast[0], reverse[0]
    profiles["joint"] = np.mean(list(profiles.values()), axis=0)
    return profiles


def estimate(reference, crop, spa):
    pixels, valid, x, y, _, coverage = crop
    if coverage < .985:
        return "incomplete_raster", None
    x -= float(reference["east_shift_m"])
    y -= float(reference["north_shift_m"])
    radius = float(reference["radius"])
    inside = np.hypot(x, y) <= radius - 1.5
    if np.count_nonzero(inside & ~valid) > .01 * inside.sum():
        return "incomplete_inner_disk", None
    # A regular subset of native samples keeps the fit and check inexpensive.
    keep = (inside & valid)[::3, ::3]
    x, y = x[::3, ::3][keep], y[::3, ::3][keep]
    channels = pixels[:, ::3, ::3][:, keep].astype(float)
    fold = (np.floor(x/BLOCK_METRES) + np.floor(y/BLOCK_METRES)) % 2 == 0
    if min(fold.sum(), (~fold).sum()) < 200:
        return "insufficient_spatial_samples", None
    location = dict(reference, longitude=float(reference["longitude"]), latitude=float(reference["latitude"]))
    _, _, bearings, ratios = sky(spa, location, UTM)
    midpoint = len(bearings)//2
    gaps = np.arange(0, min(24, float(reference["shell_m"])) + 1e-8, ROOF_SAMPLE_GAP)
    curves = [fold_profiles(x[part], y[part], channels[:, part], radius, bearings[midpoint], ratios[midpoint], gaps)
              for part in (fold, ~fold)]
    fields = ("candidate_id", "documented_roof_type", "documented_terminal_id", "reference_day",
              "tile_nominal_time_utc", "exterior_clock_status", "shadow_reference_status", "midpoint_time_gap_m")
    result = {key: reference[key] for key in fields}
    result.update(fit_pixels=int(fold.sum()), check_pixels=int((~fold).sum()), raster_coverage=coverage)
    for name in METHODS:
        gain, contrast, reverse = np.mean([item[name] for item in curves], axis=0)
        best = int(np.argmax(gain))
        near = gaps[gain >= gain[best] - .025]
        chosen = [int(np.argmax(item[name][0])) for item in curves]
        score = fmean(curves[1-index]["joint"][0][trial] for index, trial in enumerate(chosen))
        resolved = (0 < best < len(gaps) - 1 and gain[best] >= MIN_GAIN and contrast[best] >= MIN_CONTRAST
                    and gain[best] - reverse.max() >= MIN_DIRECTION_GAIN and near.max() - near.min() <= MAX_NEAR_SPAN)
        result.update({f"{name}_{field}": value for field, value in (
            ("gap_m", float(gaps[best])), ("gain", float(gain[best])), ("contrast_dn", float(contrast[best])),
            ("reversed_gain", float(reverse.max())), ("gap_range_m", float(near.max()-near.min())),
            ("first_fold_gap_m", float(gaps[chosen[0]])), ("second_fold_gap_m", float(gaps[chosen[1]])),
            ("heldout_joint_gain", score), ("resolved", bool(resolved)))})
    return "compared", result


def metrics(rows):
    return {name: dict(pairs=len(rows), resolved=sum(row[f"{name}_resolved"] for row in rows),
                      median_fold_gap_difference_m=median(abs(row[f"{name}_first_fold_gap_m"]-row[f"{name}_second_fold_gap_m"])
                                                         for row in rows) if rows else None,
                      mean_heldout_joint_gain=fmean(row[f"{name}_heldout_joint_gain"] for row in rows) if rows else None,
                      median_full_gap_difference_to_red_m=median(abs(row[f"{name}_gap_m"]-row["red_gap_m"])
                                                               for row in rows) if rows else None)
            for name in METHODS}


def main():
    references = read_csv("ok_aerial_2025_roof_shadow_reference")
    sources = indexed_crops()
    spa = solar_module()
    rows, coverage = [], Counter()
    with ExitStack() as stack:
        opened = {}
        for reference in references:
            key = tuple(reference[field] for field in ("candidate_id", "reference_day", "tile_nominal_time_utc"))
            if key not in sources or not reference["east_shift_m"]:
                coverage["missing_exact_clock_or_rgb_centre"] += 1
                continue
            tank = {field: float(reference[field]) for field in ("east", "north", "radius")}
            crop = read_native_bands(stack, sources[key], opened, tank, MAX_CENTER_SHIFT+7, (1, 2, 4))
            if crop is None:
                coverage["missing_roof_crop"] += 1
                continue
            status, row = estimate(reference, crop, spa)
            coverage[status] += 1
            if row is not None:
                rows.append(dict(row, source_tiles="|".join(tile for tile, _ in sources[key])))
    efr = [row for row in rows if row["documented_roof_type"] == "EFR"]
    closed = [row for row in rows if row["documented_roof_type"] != "EFR"]
    clock = [row for row in efr if row["exterior_clock_status"] == "exterior_matches_tile_time"]
    report = dict(source_observations=len(references), coverage=dict(coverage), external_roofs=metrics(efr),
        independent_clock_supported_external_roofs=metrics(clock), closed_roof_controls=metrics(closed),
        external_roofs_resolved_by_all=metrics([row for row in efr if all(row[f"{m}_resolved"] for m in METHODS)]),
        interpretation="The exact tile clock and the RGB-derived rim centre are held fixed across all bands. "
        "The optical fit evaluates identical every-third native 15 cm pixels in alternating 5 m blocks. "
        "The held-out score compares the fitted gap on separate roof blocks using the same mean gain in red, green "
        "and near infrared. Fold discrepancies measure spatial repeatability. True roof heights are unobserved.")
    export(STEM, rows)
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
