"""Separate geometry rasterization error from alignment on almost simultaneous passes."""
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import lru_cache
from statistics import fmean

import numpy as np

from audit_shadow_coherent_error import STEM as COHERENCE, candidate_pairs, observation_vectors, volume_statistics
from audit_sentinel2_alternate_registration import packet_inputs
from build_registered_shadow_development import registered_pair
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export, read_only
from check_spectral_shadow_controls import PACKET, control_key
from collect_public import write
from fit_shadow_geometry import gap_trials, pixel_grid, response_profiles, shifted_grid

STEM = "sentinel2_shadow_subcell_sampling"
SCHEMES = (1, 2, 4)
ALIGNMENTS = ("coarse", "scene")


def evaluate(pairs, reference):
    path, native, _, kernels = packet_inputs(PACKET)
    estimates = {int(r["observation_id"]): r for r in read_csv("shadow_change_control_estimates")}

    @lru_cache(maxsize=4096)
    def profile(number, east, north, subdivisions):
        observation = native[number]
        original = estimates[number]
        shell = float(original["shell_above_ground_m"]) if original["shell_above_ground_m"] else None
        gaps = gap_trials(shell, observation["roof_type"])
        with closing(read_only(path)) as db:
            grid = shifted_grid(pixel_grid(db, observation), east, north)
        sun, view = map(np.asarray, observation_vectors(observation))
        shifts, loss, _, reverse = response_profiles(grid, observation["radius_m"], shell if shell and shell >= 5 else 0,
            gaps, sun, view, kernels[observation["platform"]], reversed_sun=True, subcell_samples=subdivisions)
        return shifts, gaps, loss, reverse, len(grid[-2]), float(grid[-1].sum())

    def solve(item):
        row, offsets = item
        earlier, later = (int(row[f"{side}_native_observation_id"]) for side in ("earlier", "later"))
        previous = reference[control_key(row)]
        result = {field: previous[field] for field in ("comparison", "study_phase", "week_or_control_day", "candidate_id",
                    "scene_pair_id", "roof_type", "earlier_version", "later_version", "elapsed_seconds",
                    "original_all_channels_resolved")}
        result["kbbl_per_m"] = float(previous["kbbl_per_m"])
        for mode in ALIGNMENTS:
            east, north = (float(offsets[f"{mode}_{axis}_m"])/2 for axis in ("east", "north"))
            for subdivisions in SCHEMES:
                fit = registered_pair(profile(earlier, -east, -north, subdivisions), profile(later, east, north, subdivisions))
                change = fit["registered_change_m"]
                resolved = fit["registered_both_shadow_diagnostics_pass"]
                if subdivisions == 1 and (change != float(previous[f"{mode}_change_m"]) or
                                         str(resolved) != previous[f"{mode}_resolved"]):
                    raise ValueError(f"Original grid replay differs for {control_key(row)}/{mode}")
                result[f"{mode}_{subdivisions}_change_m"] = change
                result[f"{mode}_{subdivisions}_resolved"] = resolved
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, row in enumerate(pool.map(solve, pairs), 1):
            if i % 100 == 0 or i == len(pairs):
                print(f"Compared rasterization grids on {i}/{len(pairs)} tandem control pairs", flush=True)
            yield row


def main():
    pairs = [item for item in candidate_pairs() if item[0]["comparison"] == "tandem"]
    reference = {control_key(r): r for r in read_csv(f"{COHERENCE}_pairs") if r["comparison"] == "tandem"}
    rows = list(evaluate(pairs, reference))
    by_scene = defaultdict(list)
    for r in rows:
        if r["roof_type"] == "EFR":
            by_scene[r["scene_pair_id"]].append(r)
    sums = []
    means = {}
    for scope in ("all_common_pixels", "original_all_channels_resolved"):
        means[scope] = {}
        for mode in ALIGNMENTS:
            means[scope][mode] = {}
            for scheme in SCHEMES:
                volumes = []
                for scene, group in sorted(by_scene.items()):
                    matched = [r for r in group if scope == "all_common_pixels" or r["original_all_channels_resolved"] == "True"]
                    stats = volume_statistics(matched, f"{mode}_{scheme}")
                    volumes.append(abs(stats["signed_volume_kbbl"]))
                    sums.append(dict(scene_pair_id=scene, date=group[0]["week_or_control_day"], scope=scope,
                                    alignment=mode, subdivisions_per_axis=scheme, **stats))
                means[scope][mode][scheme] = fmean(volumes)
    closed = [r for r in rows if r["roof_type"] != "EFR"]
    report = dict(tandem_pairs=len(rows), matched_external_roof_pairs=sum(map(len, by_scene.values())),
        mean_absolute_scene_volume_kbbl=means, closed_tandem_pairs=len(closed),
        closed_false_passes={mode: {n: sum(r[f"{mode}_{n}_resolved"] for r in closed) for n in SCHEMES} for mode in ALIGNMENTS},
        interpretation="Each original two-metre model cell is evaluated at its centre, four subcell midpoints "
        "or sixteen subcell midpoints before applying the same published sensor kernel. Actual image pixels, "
        "angles, geometry and the tested relative alignment are held fixed. Compare scene sums on the same roofs. "
        "This is a numerical rendering sensitivity check on tandem engineering controls.")
    export(f"{STEM}_pairs", rows)
    export(f"{STEM}_scenes", sums)
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
