"""Validate spectral shadow recovery on tandem captures and closed-roof controls."""
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from statistics import median

import numpy as np

from audit_sentinel2_alternate_registration import packet_inputs
from audit_shadow_registration_pixels import index
from build_registered_shadow_development import registered_pair
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export, read_only
from check_multiband_shadow import spectral_kernels
from check_weekly_multiband_shadow import METHODS, SIDES, spectral_profiles
from collect_public import write
from collect_s2_spectral_detector_masks import BANDS, CONTROL_STEM, control_offsets
from fit_shadow_geometry import CALIBRATION

STEM = "sentinel2_spectral_shadow_controls"
PACKET = "shadow_change_controls.sqlite"


def control_key(row):
    return tuple(row[field] for field in ("comparison", "study_phase", "week_or_control_day", "candidate_id", "scene_pair_id"))


def jobs():
    scenes, pairs = index("sources", "source_id"), index("scene_pairs", "scene_pair_id")
    ledger = {control_key(row): row for row in read_csv("shadow_registration_tank_windows")
              if row["comparison"] in ("tandem", "fixed_roof")}
    offsets = list(control_offsets())
    if len(offsets) != len(ledger) or {control_key(row) for row in offsets} != ledger.keys():
        raise ValueError("Registration and control manifests describe different comparisons")
    for row in offsets:
        source = ledger[control_key(row)]
        pair = pairs[row["scene_pair_id"]]
        earlier, later = (scenes[pair[f"{side}_source_id"]] for side in SIDES)
        if earlier["relative_orbit"] != later["relative_orbit"] and row["comparison"] == "tandem":
            raise ValueError("Tandem control uses a different relative orbit")
        yield row, source, earlier, later


def measure(sample, kernels, masks):
    path, native, scenes, _ = packet_inputs(PACKET)
    native = {(row["candidate_id"], row["version_id"]): row for row in native.values()}
    saved = {(row["candidate_id"], row["version_id"]): row for row in read_csv("shadow_change_control_estimates")}
    baseline = {control_key(row): row for row in read_csv("shadow_registered_control_fit")}

    @lru_cache(maxsize=4096)
    def profile(tank, version, east, north):
        observation, original = native[tank, version], saved[tank, version]
        if int(original["observation_id"]) != observation["observation_id"]:
            raise ValueError(f"Observation mismatch for {tank}/{version}")
        scene = scenes[observation["scene_id"]]
        shell = float(original["shell_above_ground_m"]) if original["shell_above_ground_m"] else None
        files = tuple(masks[version, band]["local_file"] for band in BANDS)
        return spectral_profiles(path, observation, scene, shell, kernels, files, east, north)

    def solve(job):
        offset, source, earlier, later = job
        tank = offset["candidate_id"]
        original = [native[tank, s["version_id"]] for s in (earlier, later)]
        names = ("center_east_m", "center_north_m", "radius_m", "roof_type", "geometry_basis", "circle_area_bbl_per_m")
        if any(original[0][name] != original[1][name] for name in names):
            raise ValueError(f"Control geometry differs between scenes for {control_key(offset)}")
        result = dict(zip(("comparison", "study_phase", "week_or_control_day", "candidate_id", "scene_pair_id"),
                          control_key(offset), strict=True))
        elapsed = (datetime.fromisoformat(later["captured_at_utc"])-datetime.fromisoformat(earlier["captured_at_utc"])).total_seconds()
        if elapsed <= 0 or (offset["comparison"] == "tandem" and elapsed > 60):
            raise ValueError(f"Invalid time separation for {control_key(offset)}")
        result.update(roof_type=source["roof_type"], elapsed_seconds=elapsed, registration_status=offset["registration_status"],
                      cutoff_utc=source["inputs_available_before_utc"],
                      east_offset_m=offset["later_minus_earlier_east_m"], north_offset_m=offset["later_minus_earlier_north_m"],
                      kbbl_per_m=original[0]["circle_area_bbl_per_m"]/1000)
        fitted = []
        for side, scene, o, sign in zip(SIDES, (earlier, later), original, (-1, 1), strict=True):
            version = scene["version_id"]
            stored = scenes[o["scene_id"]]
            capture = datetime.fromisoformat(scene["captured_at_utc"])
            optical_archive = datetime.fromisoformat(source[f"{side}_archive_available_at_utc"])
            if (capture != datetime.fromisoformat(o["captured_at_utc"]) or capture > optical_archive
                or scene["canonical_l2a_id"] != stored["scene_item_id"] or scene["red_raster"] != stored["red_raster"]
                or o["roof_type"] != source["roof_type"]):
                raise ValueError(f"Exact source manifest differs for {control_key(offset)}/{side}")
            result.update({f"{side}_{name}": value for name, value in (
                ("version", version), ("captured_at_utc", capture.isoformat()), ("archive_at_utc", optical_archive.isoformat()),
                ("native_observation_id", o["observation_id"]))})
            status, count, profiles, ids, mask_date = "registration_unconfirmed", None, None, {}, None
            if offset["registration_status"] == "checked_offset":
                mask_date = max(datetime.fromisoformat(masks[version, band]["source_created_at_utc"]) for band in BANDS)
                if result["cutoff_utc"] and max(optical_archive, mask_date) >= datetime.fromisoformat(result["cutoff_utc"]):
                    status = "original_inputs_after_cutoff"
                else:
                    east, north = (sign*float(result[f"{axis}_offset_m"])/2 for axis in ("east", "north"))
                    status, count, profiles, ids = profile(tank, version, east, north)
            result.update({f"{side}_{name}": value for name, value in (
                ("status", status), ("common_pixels", count), ("B03_detector", ids.get("B03")),
                ("B08_detector", ids.get("B08")), ("masks_at_utc", mask_date.isoformat() if mask_date else None))})
            fitted.append(profiles)
        matched = all(p is not None for p in fitted)
        for method in METHODS:
            fit = registered_pair(*(p[method] for p in fitted)) if matched else None
            if method == "red" and fit and control_key(offset) in baseline and all(
                result[f"{side}_status"] == "compared_full_pixels" for side in SIDES
            ) and not math.isclose(fit["registered_change_m"], float(baseline[control_key(offset)]["shared_registered_change_m"])):
                raise ValueError(f"Red control replay differs for {control_key(offset)}")
            change = fit["registered_change_m"] if fit else None
            result.update({f"{method}_{name}": value for name, value in (
                ("earlier_gap_m", fit["earlier_registered_gap_m"] if fit else None),
                ("later_gap_m", fit["later_registered_gap_m"] if fit else None),
                ("earlier_fit_status", fit["earlier_registered_status"] if fit else None),
                ("later_fit_status", fit["later_registered_status"] if fit else None),
                ("resolved", fit["registered_both_shadow_diagnostics_pass"] if fit else False),
                ("change_m", change), ("change_kbbl", change*result["kbbl_per_m"] if fit else None))})
        return result

    with ThreadPoolExecutor(max_workers=5) as pool:
        for i, row in enumerate(pool.map(solve, sample), 1):
            if i % 250 == 0 or i == len(sample):
                print(f"Checked {i}/{len(sample)} optical control pairs", flush=True)
            yield row


def differences(rows, method):
    signed = [row[f"{method}_change_m"] for row in rows]
    if not signed:
        return dict(pairs=0)
    absolute = list(map(abs, signed))
    return dict(pairs=len(rows), tanks=len({r["candidate_id"] for r in rows}),
                scene_pairs=len({r["scene_pair_id"] for r in rows}), median_abs_m=median(absolute),
                p90_abs_m=float(np.percentile(absolute, 90)), above_one_m=sum(v > 1 for v in absolute),
                median_signed_m=median(signed))


def summary(rows):
    comparable = [row for row in rows if row["red_change_m"] is not None]
    common = [row for row in comparable if all(row[f"{method}_resolved"] for method in METHODS)]
    rejected = [row for row in comparable if not row["red_resolved"]]
    return dict(pairs=len(rows), registration_passes=sum(r["registration_status"] == "checked_offset" for r in rows),
        comparison_pairs=len(comparable), unique_tanks=len({r["candidate_id"] for r in comparable}),
        red_rejected_pairs=len(rejected),
        on_identical_pixels={m: differences(comparable, m) for m in METHODS},
        all_methods_resolved={m: differences(common, m) for m in METHODS},
        independently_resolved={m: differences([r for r in comparable if r[f"{m}_resolved"]], m) for m in METHODS},
        recovery={m: differences([r for r in rejected if r[f"{m}_resolved"]], m) for m in METHODS[1:]},
        endpoint_statuses=dict(Counter(r[f"{side}_status"] for r in rows for side in SIDES)))


def tandem_scene_sums(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["comparison"] == "tandem" and row["roof_type"] == "EFR" and row["red_change_m"] is not None:
            groups[row["scene_pair_id"]].append(row)
    for scene, rows in sorted(groups.items(), key=lambda item: int(item[0])):
        first = rows[0]
        for scope, sample in (("identical_pixels", rows),
                              ("all_resolved", [r for r in rows if all(r[f"{m}_resolved"] for m in METHODS)])):
            if sample:
                yield dict(scene_pair_id=scene, scope=scope, capture_day=first["week_or_control_day"],
                           earlier_version=first["earlier_version"], later_version=first["later_version"],
                           elapsed_seconds=first["elapsed_seconds"], matched_tanks=len(sample),
                           **{f"{m}_apparent_sum_change_kbbl": sum(r[f"{m}_change_kbbl"] for r in sample) for m in METHODS})


def main():
    with closing(read_only(CALIBRATION)) as db:
        kernels = spectral_kernels(db)
    masks = {(row["version_id"], row["band"]): row for row in read_csv(CONTROL_STEM)}
    rows = list(measure(list(jobs()), kernels, masks))
    groups = defaultdict(list)
    for row in rows:
        if row["roof_type"] != "EFR":
            groups["all_closed_roof_controls"].append(row)
        label = f"tandem_{row['roof_type']}" if row["comparison"] == "tandem" else f"fixed_roof_{row['study_phase']}"
        groups[label].append(row)
    report = dict(groups={label: summary(sample) for label, sample in sorted(groups.items())},
        interpretation="The same independently registered pixels and geometry are fitted in red, NIR and joint "
        "B03/B04/B08. Tandem EFR changes measure repeatability within 60 seconds. Closed roof passes are false "
        "detections of a floating-roof shadow, with repeated measurements per roof. Recovery means both endpoints "
        "passed only after changing the spectral method. This diagnostic uses engineering controls and no EIA or price outcomes.")
    export(f"{STEM}_pairs", rows)
    export(f"{STEM}_tandem_scene_sums", list(tandem_scene_sums(rows)))
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
