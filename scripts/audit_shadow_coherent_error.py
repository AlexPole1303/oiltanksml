"""Trace scene-wide volume discrepancies to signed roof errors and model sensitivity."""
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import lru_cache
from statistics import fmean, median

import numpy as np

from audit_refined_shadow_alignment import OFFSETS, STEM as ALIGNMENT
from audit_sentinel2_alternate_registration import packet_inputs
from build_registered_shadow_development import registered_pair
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from check_spectral_shadow_controls import PACKET, STEM as CONTROLS, control_key
from collect_public import write
from fit_shadow_geometry import bearing_vector, gap_trials, pixel_grid, response_profiles, shifted_grid

STEM = "sentinel2_coherent_shadow_error"
ALIGNMENT_METHODS = ("coarse", "local", "scene")
SENSITIVITIES = ("shared_sun", "shared_view", "shared_psf")
METHODS = ALIGNMENT_METHODS + SENSITIVITIES


def observation_vectors(row):
    return tuple(tuple(bearing_vector(row[f"{prefix}_zenith_deg"], row[f"{prefix}_azimuth_grid_deg"]))
                 for prefix in ("sun", "red_view"))


def mean_vector(vectors):
    return tuple(np.mean(vectors, axis=0))


def candidate_pairs():
    alignment = {control_key(row): row for row in read_csv(f"{ALIGNMENT}_pairs")}
    controls = [row for row in read_csv(f"{CONTROLS}_pairs") if row["red_change_m"]]
    if len(controls) != len(alignment) or {control_key(row) for row in controls} != alignment.keys():
        raise ValueError("Alignment and spectral audit describe different control rows")
    for row in controls:
        yield row, alignment[control_key(row)]


def refit(controls, terminals):
    path, native, _, kernels = packet_inputs(PACKET)
    estimates = {int(row["observation_id"]): row for row in read_csv("shadow_change_control_estimates")}

    @lru_cache(maxsize=3072)
    def inputs(number):
        observation = native[number]
        shell_text = estimates[number]["shell_above_ground_m"]
        shell = float(shell_text) if shell_text else None
        gaps = gap_trials(shell, observation["roof_type"])
        with closing(read_only(path)) as db:
            grid = pixel_grid(db, observation)
        return observation, grid, gaps, shell if shell and shell >= 5 else 0

    @lru_cache(maxsize=4096)
    def profile(number, east, north, sun, view, platforms):
        observation, grid, gaps, shell = inputs(number)
        shifted = shifted_grid(grid, east, north)
        kernel = kernels[platforms[0]] if len(platforms) == 1 else np.mean([kernels[p] for p in platforms], axis=0)
        shifts, loss, _, reverse = response_profiles(shifted, observation["radius_m"], shell, gaps,
            np.asarray(sun), np.asarray(view), kernel, reversed_sun=True)
        return shifts, gaps, loss, reverse, len(grid[-2]), float(grid[-1].sum())

    def solve(item):
        source, alignment = item
        sides = ("earlier", "later")
        ids = [int(source[f"{side}_native_observation_id"]) for side in sides]
        observations = [native[number] for number in ids]
        if any((o["candidate_id"], o["version_id"]) != (source["candidate_id"], source[f"{side}_version"])
               for side, o in zip(sides, observations, strict=True)):
            raise ValueError(f"Observation lineage differs at {control_key(source)}")
        if any(source[f"{side}_status"] != "compared_full_pixels" for side in sides):
            raise ValueError("Red-band counterfactuals require the original full sample")
        vectors = [observation_vectors(row) for row in observations]
        suns, views = (tuple(values) for values in zip(*vectors, strict=True))
        platforms = tuple(row["platform"] for row in observations)
        tandem = source["comparison"] == "tandem" and source["roof_type"] == "EFR"
        coarse, local, scene = (
            tuple(float(alignment[f"{mode}_{axis}_m"]) for axis in OFFSETS)
            if alignment[f"{mode}_east_m"] else None for mode in ALIGNMENT_METHODS)
        result = {field: source[field] for field in ("comparison", "study_phase", "week_or_control_day",
            "candidate_id", "scene_pair_id", "roof_type", "earlier_version", "later_version", "elapsed_seconds")}
        result.update(terminal_id=terminals.get(source["candidate_id"]), kbbl_per_m=float(source["kbbl_per_m"]),
            original_all_channels_resolved=all(source[f"{m}_resolved"] == "True" for m in ("red", "nir", "joint")),
            differential_solar_displacement_for_10m_roof_m=float(10*np.linalg.norm(np.subtract(*suns))),
            differential_view_displacement_for_10m_roof_m=float(10*np.linalg.norm(np.subtract(*views))),
            coarse_change_m=float(source["red_change_m"]), coarse_resolved=source["red_resolved"] == "True",
            local_shift_difference_m=float(np.linalg.norm(np.subtract(local, coarse))) if local is not None else None,
            scene_shift_difference_m=float(np.linalg.norm(np.subtract(scene, coarse))) if scene is not None else None)
        original_kernels = tuple((p,) for p in platforms)
        modes = (("local", local, suns, views, original_kernels), ("scene", scene, suns, views, original_kernels),
                 ("shared_sun", coarse if tandem else None, (mean_vector(suns),)*2, views, original_kernels),
                 ("shared_view", coarse if tandem else None, suns, (mean_vector(views),)*2, original_kernels),
                 ("shared_psf", coarse if tandem else None, suns, views, (platforms,)*2))
        for mode, offset, sun, view, kernel in modes:
            if offset is None:
                change, resolved = None, None
            elif offset == coarse and mode in ALIGNMENT_METHODS:
                change, resolved = result["coarse_change_m"], result["coarse_resolved"]
            else:
                e, n = offset
                fitted = registered_pair(*(profile(number, sign*e/2, sign*n/2, s, v, k)
                    for number, sign, s, v, k in zip(ids, (-1, 1), sun, view, kernel, strict=True)))
                change, resolved = fitted["registered_change_m"], fitted["registered_both_shadow_diagnostics_pass"]
            result[f"{mode}_change_m"], result[f"{mode}_resolved"] = change, resolved
        return result

    with ThreadPoolExecutor(max_workers=5) as pool:
        for i, row in enumerate(pool.map(solve, controls), 1):
            if i % 200 == 0 or i == len(controls):
                print(f"Evaluated {i}/{len(controls)} paired roof controls", flush=True)
            yield row


def volume_statistics(rows, mode):
    heights = [row[f"{mode}_change_m"] for row in rows]
    amounts = [h * row["kbbl_per_m"] for h, row in zip(heights, rows, strict=True)]
    total, area = sum(amounts), sum(r["kbbl_per_m"] for r in rows)
    reference = math.sqrt(sum(v*v for v in amounts))
    return dict(observations=len(rows), volume_gradient_kbbl_per_m=area, signed_volume_kbbl=total,
                weighted_height_m=total/area, median_abs_height_m=median(map(abs, heights)),
                p90_abs_height_m=float(np.percentile(np.abs(heights), 90)),
                same_sign_nonzero_roofs=sum(v*total > 0 for v in amounts),
                nonzero_roofs=sum(v != 0 for v in amounts), independent_sign_reference_kbbl=reference,
                observed_to_independent_sign_reference=abs(total)/reference if reference else None)


def tandem_tables(rows):
    by_scene = defaultdict(list)
    for row in rows:
        if row["comparison"] == "tandem" and row["roof_type"] == "EFR":
            by_scene[row["scene_pair_id"]].append(row)
    scenes, terminals = [], []
    for scene, all_rows in sorted(by_scene.items(), key=lambda item: int(item[0])):
        first = all_rows[0]
        info = {field: first[field] for field in ("scene_pair_id", "week_or_control_day", "earlier_version", "later_version",
                                                  "elapsed_seconds")}
        for scope, matched in (("all_common_pixels", all_rows),
                               ("original_all_channels_resolved", [r for r in all_rows if r["original_all_channels_resolved"]])):
            by_terminal = defaultdict(list)
            for row in matched:
                by_terminal[row["terminal_id"]].append(row)
            for mode in METHODS:
                if any(row[f"{mode}_change_m"] is None for row in matched):
                    raise ValueError(f"Missing matched counterfactual at {scene}/{mode}")
                scenes.append(dict(info, scope=scope, method=mode, **volume_statistics(matched, mode)))
                if mode == "coarse":
                    for site, sample in sorted(by_terminal.items(), key=lambda item: str(item[0])):
                        terminals.append(dict(info, scope=scope, terminal_id=site, **volume_statistics(sample, mode)))
    return scenes, terminals


def closed_summary(rows):
    results = {}
    for phase in ("development", "validation", "holdout", "engineering"):
        items = [r for r in rows if r["roof_type"] != "EFR" and r["study_phase"] == phase]
        results[phase] = {}
        for method in ALIGNMENT_METHODS[1:]:
            sample = [row for row in items if row[f"{method}_change_m"] is not None]
            results[phase][method] = dict(pairs=len(sample), methods={
                mode: dict(median_abs_height_m=median(abs(row[f"{mode}_change_m"]) for row in sample),
                           p90_abs_height_m=float(np.percentile([abs(row[f"{mode}_change_m"]) for row in sample], 90)),
                           both_ends_resolved=sum(row[f"{mode}_resolved"] for row in sample))
                for mode in ("coarse", method)})
    return results


def main():
    with closing(read_only(SOURCE)) as db:
        terminals = dict(db.execute("SELECT candidate_id,documented_terminal_id FROM tank_engineering_registry"))
    rows = list(refit(list(candidate_pairs()), terminals))
    scene_rows, terminal_rows = tandem_tables(rows)
    export(f"{STEM}_pairs", rows)
    export(f"{STEM}_scenes", scene_rows)
    export(f"{STEM}_terminals", terminal_rows)
    means = {scope: {method: fmean(abs(r["signed_volume_kbbl"]) for r in scene_rows
             if r["scope"] == scope and r["method"] == method) for method in METHODS}
             for scope in ("all_common_pixels", "original_all_channels_resolved")}
    open_roofs = [r for r in rows if r["comparison"] == "tandem" and r["roof_type"] == "EFR"]
    report = dict(compared_pairs=len(rows), tandem_scene_mean_absolute_volume_kbbl=means,
        tandem_median_differential_horizontal_displacement_m_at_10m={
            k: median(r[f"differential_{k}_displacement_for_10m_roof_m"] for r in open_roofs) for k in ("solar", "view")},
        closed_roof_controls=closed_summary(rows),
        interpretation="All tandem variants compare identical roofs on the same captures. Fine offsets are estimated "
        "solely on exterior fitting patches. Each common-input counterfactual replaces only one varying input "
        "with the average of the two dates and retains the original offsets. They test model sensitivity. "
        "Independent-sign RMS holds individual magnitudes fixed. No inventory or trading outcomes are used.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
