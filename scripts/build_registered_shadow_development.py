"""Build dated weekly and terminal shadow inputs after independent scene registration."""
import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from statistics import median

import numpy as np

from audit_shadow_registration_pixels import index
from build_terminal_development_measurements import DATED, RETROSPECTIVE as TERMINAL_RETROSPECTIVE
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, export
from build_weekly_shadow_validation import RETROSPECTIVE, legacy_geometry
from collect_public import write
from fit_shadow_geometry import (bearing_vector, gap_fit_status, gap_trials, inputs, pixel_grid, read_only,
                                 response_profiles, shared_centre_minimum, shifted_grid)

STEM = "weekly_shadow_registered_development"
PRIMARY = "weekly_shadow_development"
TERMINAL = "weekly_shadow_terminal_development"
LEGACY = "core_terminal_legacy"


def study_weeks(phase="development"):
    with closing(read_only(SOURCE)) as db:
        rows = db.execute("""SELECT p.week_ending,p.window_ends_by_utc,c.window_ends_by_utc,c.release_date,
                   c.inputs_available_before_utc FROM eia_satellite_window p JOIN eia_satellite_window c USING(week_ending)
                   WHERE p.endpoint='previous' AND c.endpoint='current' AND c.study_phase=?
                   ORDER BY p.week_ending""", (phase,)).fetchall()
    return {week: (datetime.fromisoformat(first), datetime.fromisoformat(last), release, cutoff)
            for week, first, last, release, cutoff in rows}


def baseline_pairs(phase="development"):
    if phase in ("validation", "holdout"):
        stem = f"weekly_shadow_{phase}"
        for row in read_csv(f"{stem}_pairs"):
            if row["roof_type"] == "EFR":
                yield row["geometry_mode"], stem, stem, row
        return
    for row in read_csv(f"{PRIMARY}_pairs"):
        if row["roof_type"] == "EFR":
            yield RETROSPECTIVE, PRIMARY, PRIMARY, row
    for row in read_csv(f"{LEGACY}_development_pairs"):
        yield DATED, PRIMARY, LEGACY, row
    for raw in read_csv(f"{TERMINAL}_pairs"):
        mode = {TERMINAL_RETROSPECTIVE: RETROSPECTIVE, DATED: DATED}[raw["geometry_mode"]]
        row = dict(raw)
        for normal, original in (("earlier", "previous"), ("later", "current")):
            for field in ("observation_id", "version", "capture_at_utc", "status", "gap_m"):
                row[f"{normal}_{field}"] = raw[f"{original}_{field}"]
        yield mode, TERMINAL, TERMINAL, row


def observation_sources(phase="development"):
    geometry = legacy_geometry()
    sources = {}
    packets = (PRIMARY, TERMINAL) if phase == "development" else (f"weekly_shadow_{phase}",)
    for packet in packets:
        path = DERIVED / f"{packet}.sqlite"
        with closing(read_only(path)) as db:
            db.row_factory = sqlite3.Row
            rows, kernels = inputs(db)
            scenes = {r["scene_id"]: r for r in db.execute("SELECT * FROM scene")}
        if any(row["crs"] != "EPSG:32614" for row in scenes.values()):
            raise ValueError(f"Unexpected native geometry in {packet}")
        sources[packet] = (path, {r["observation_id"]: r for r in rows}, scenes, kernels)
    saved = {}
    for stem in (*packets, LEGACY) if phase == "development" else packets:
        for row in read_csv(f"{stem}_observations"):
            mode = DATED if stem == LEGACY or row.get("geometry_mode") == DATED else RETROSPECTIVE
            key = stem, mode, int(row["observation_id"])
            if key in saved:
                raise ValueError(f"Duplicate original observation: {key}")
            saved[key] = row
    return sources, saved, geometry


def model_inputs(mode, packet, stem, row, offset, scene_pair, sources, native, saved):
    week, tank = row["week_ending"], row["candidate_id"]
    _, observations, scenes, _ = native[packet]
    keys = []
    for side in ("earlier", "later"):
        observation_id = int(row[f"{side}_observation_id"])
        observation = observations[observation_id]
        original = saved[stem, mode, observation_id]
        scene = scenes[observation["scene_id"]]
        source = sources[scene_pair[f"{side}_source_id"]]
        if (observation["candidate_id"] != tank or original["candidate_id"] != tank
                or {original["version_id"], scene["version_id"], source["version_id"]} != {row[f"{side}_version"]}
                or scene["scene_item_id"] != source["canonical_l2a_id"]
                or datetime.fromisoformat(scene["captured_at_utc"]) != datetime.fromisoformat(row[f"{side}_capture_at_utc"])
                or original["estimated_gap_m"] != row[f"{side}_gap_m"]):
            raise ValueError(f"Registered and original inputs differ for {week}/{tank}/{side}/{mode}")
        if datetime.fromisoformat(offset[f"{side}_archive_available_at_utc"]) >= datetime.fromisoformat(row["inputs_available_before_utc"]):
            raise ValueError(f"Registration uses a scene after the release cutoff for {week}/{tank}")
        keys.append((packet, stem, mode, observation_id))
    return keys


def prepare_jobs(weeks, native, saved, phase="development"):
    sources, scene_pairs = (index(table, field) for table, field in (
        ("sources", "source_id"), ("scene_pairs", "scene_pair_id")))
    ledger = {}
    for row in read_csv("shadow_registration_tank_windows"):
        if (row["comparison"], row["study_phase"], row["roof_type"]) != ("weekly", phase, "EFR"):
            continue
        key = row["week_or_control_day"], row["candidate_id"]
        if key in ledger:
            raise ValueError(f"Duplicate EFR scene pair: {key}")
        ledger[key] = row
    offsets = {}
    for row in read_csv(f"shadow_registration_{phase}_offsets"):
        key = row["week_or_control_day"], row["candidate_id"]
        if key in offsets:
            raise ValueError(f"Duplicate {phase} offset: {key}")
        offsets[key] = row
    jobs = []
    seen = set()
    for mode, packet, stem, row in baseline_pairs(phase):
        week, tank = row["week_ending"], row["candidate_id"]
        key = week, tank
        if (mode, key) in seen:
            raise ValueError(f"Duplicate {phase} pair: {mode}/{key}")
        seen.add((mode, key))
        offset, record = offsets[key], ledger[key]
        scene_pair = scene_pairs[record["scene_pair_id"]]
        first, last, release, cutoff = weeks[week]
        earlier, later = (datetime.fromisoformat(row[f"{side}_capture_at_utc"]) for side in ("earlier", "later"))
        if (offset["study_phase"] != phase or offset["scene_pair_id"] != record["scene_pair_id"]
                or release != row["release_date"] or cutoff != row["inputs_available_before_utc"]
                or cutoff != record["inputs_available_before_utc"] or not earlier <= first < later <= last):
            raise ValueError(f"Wrong measurement or registration clock for {key}")
        native_keys = model_inputs(mode, packet, stem, row, record, scene_pair, sources, native, saved)
        jobs.append((mode, row, record, offset, native_keys, (later-earlier).total_seconds()/3600,
                     (later-first)/(later-earlier), (later-first)/(last-first)))
    if {k for mode, k in seen if mode == RETROSPECTIVE} != ledger.keys():
        raise ValueError("The original weekly EFR set differs from the registration ledger")
    return sorted(jobs, key=lambda item: (item[0], item[1]["week_ending"], item[1]["candidate_id"]))


def registered_pair(first, last):
    shifts, before_gaps, before_loss, before_placebo, before_pixels, before_weight = first
    other, after_gaps, after_loss, after_placebo, after_pixels, after_weight = last
    if shifts != other:
        raise ValueError("The common centre trials differ between the two observations")
    left, right = before_loss/before_weight, after_loss/after_weight
    best_shift, i, j, optimum = shared_centre_minimum(left, right)
    # Project the joint descriptive loss contour onto each date while retaining the shared centre.
    limit = optimum*(1+1/max(1,before_pixels+after_pixels-12)) + 1e-10/min(before_weight,after_weight)
    earlier_near = before_gaps[(left + right.min(axis=1)[:, None]).min(axis=0) <= limit]
    later_near = after_gaps[(right + left.min(axis=1)[:, None]).min(axis=0) <= limit]
    early, late = float(before_gaps[i]), float(after_gaps[j])
    early_status = gap_fit_status(before_gaps, early, earlier_near.min(), earlier_near.max(),
        math.sqrt(left[best_shift, i]), math.sqrt(left[:, 0].min()), math.sqrt(before_placebo.min()/before_weight))
    late_status = gap_fit_status(after_gaps, late, later_near.min(), later_near.max(),
        math.sqrt(right[best_shift, j]), math.sqrt(right[:, 0].min()), math.sqrt(after_placebo.min()/after_weight))
    # Keep the shared-centre relation in the envelope of possible paired changes.
    differences = []
    for early_curve, late_curve in zip(left, right, strict=True):
        a, b = np.nonzero(early_curve[:, None] + late_curve[None, :] <= limit)
        if len(a):
            differences.extend(before_gaps[a] - after_gaps[b])
    return dict(earlier_registered_gap_m=early, later_registered_gap_m=late, registered_change_m=early-late,
                earlier_registered_status=early_status, later_registered_status=late_status,
                registered_both_shadow_diagnostics_pass=early_status == late_status == RESOLVED,
                registered_centre_east_m=shifts[best_shift][0], registered_centre_north_m=shifts[best_shift][1],
                registered_loss_range_min_m=min(differences), registered_loss_range_max_m=max(differences))


def registered_profile(path, row, original, shell, kernel, east, north):
    if original is not None and not math.isclose(float(original["diameter_m"]), 2*row["radius_m"]):
        raise ValueError(f"Incorrect geometry mode for observation {row['observation_id']}")
    with closing(read_only(path)) as db:
        grid = shifted_grid(pixel_grid(db, row), east, north)
    count, weight = len(grid[-2]), float(grid[-1].sum())
    if count < 12 or (original is not None and count != int(original["fitted_pixels"])):
        raise ValueError(f"Registered fit uses different native pixels: {row['candidate_id']}/{row['version_id']}")
    gaps = gap_trials(shell, row["roof_type"])
    sun = bearing_vector(row["sun_zenith_deg"], row["sun_azimuth_grid_deg"])
    view = bearing_vector(row["red_view_zenith_deg"], row["red_view_azimuth_grid_deg"])
    shifts, loss, _, placebo = response_profiles(grid, row["radius_m"], shell if shell and shell >= 5 else 0,
                                               gaps, sun, view, kernel, reversed_sun=True)
    if original is not None and east == north == 0:
        best = np.unravel_index(np.argmin(loss), loss.shape)[1]
        if float(gaps[best]) != float(original["estimated_gap_m"]):
            raise ValueError(f"Baseline replay differs for {row['candidate_id']}/{row['version_id']}")
    return shifts, gaps, loss, placebo, count, weight


def refit(jobs, native, saved, geometry):
    @lru_cache(maxsize=4096)
    def profile(key, east, north):
        packet, stem, mode, number = key
        path, observations, _, kernels = native[packet]
        row, original = observations[number], saved[stem, mode, number]
        if mode == DATED:
            old = geometry[row["candidate_id"]]
            row = dict(row, center_east_m=float(old["center_east_m"]), center_north_m=float(old["center_north_m"]),
                       radius_m=float(old["diameter_m"])/2)
        shell = float(original["shell_above_ground_m"]) if original["shell_above_ground_m"] else None
        return registered_profile(path, row, original, shell, kernels[row["platform"]], east, north)

    def solve(job):
        mode, base, ledger, offset, keys, hours, capture_fraction, eia_fraction = job
        fields = ("week_ending", "release_date", "candidate_id", "terminal_id", "earlier_version", "later_version",
                  "earlier_capture_at_utc", "later_capture_at_utc", "inputs_available_before_utc")
        result = dict(geometry_mode=mode, **{k: base[k] for k in fields}, scene_pair_id=offset["scene_pair_id"],
            earlier_archive_available_at_utc=ledger["earlier_archive_available_at_utc"],
            later_archive_available_at_utc=ledger["later_archive_available_at_utc"], capture_interval_hours=hours,
            fraction_of_capture_inside_eia_interval=capture_fraction, fraction_of_eia_interval_observed=eia_fraction,
            circular_area_m2=float(base["circular_area_m2"]), circular_bbl_per_m=float(base["circular_bbl_per_m"]),
            baseline_change_m=float(base["optical_level_change_m"]) if base["optical_level_change_m"] else None,
            baseline_both_shadow_diagnostics_pass=base["both_shadow_diagnostics_pass"] == "True",
            offset_status=offset["registration_status"],
            offset_east_m=offset["later_minus_earlier_east_m"], offset_north_m=offset["later_minus_earlier_north_m"],
            fit_status="offset_unconfirmed", earlier_registered_gap_m=None, later_registered_gap_m=None,
            registered_change_m=None, earlier_registered_status=None, later_registered_status=None,
            registered_both_shadow_diagnostics_pass=False, registered_centre_east_m=None, registered_centre_north_m=None,
            registered_loss_range_min_m=None, registered_loss_range_max_m=None, registered_change_equivalent_bbl=None)
        if offset["registration_status"] != "checked_offset":
            return result
        if result["baseline_change_m"] is None:
            return dict(result, fit_status="missing_usable_native_observation")
        east, north = (float(offset[f"later_minus_earlier_{axis}_m"])/2 for axis in ("east", "north"))
        fit = registered_pair(profile(keys[0], -east, -north), profile(keys[1], east, north))
        result.update(fit)
        result.update(fit_status="resolved" if fit["registered_both_shadow_diagnostics_pass"] else "shadow_diagnostics_failed",
                      registered_change_equivalent_bbl=fit["registered_change_m"]*result["circular_bbl_per_m"])
        return result

    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, row in enumerate(pool.map(solve, jobs), 1):
            if i % 500 == 0 or i == len(jobs):
                print(f"Processed {i}/{len(jobs)} geometry-pairs", flush=True)
            yield row


def aggregate(rows):
    original = [r for r in rows if r["baseline_both_shadow_diagnostics_pass"]]
    registered = [r for r in rows if r["registered_both_shadow_diagnostics_pass"]]
    common = [r for r in registered if r["baseline_both_shadow_diagnostics_pass"]]
    out = dict(input_pairs=len(rows), checked_offsets=sum(r["offset_status"] == "checked_offset" for r in rows),
               registered_numeric_pairs=sum(r["registered_change_m"] is not None for r in rows))
    for name, sample, fields in (("baseline", original, ("baseline_change_m",)),
            ("registered", registered, ("registered_change_m",)),
            ("both", common, ("baseline_change_m", "registered_change_m"))):
        out.update({f"{name}_resolved_pairs": len(sample),
                    f"{name}_resolved_area_m2": sum(r["circular_area_m2"] for r in sample),
                    f"{name}_median_eia_interval_fraction": median(r["fraction_of_eia_interval_observed"] for r in sample) if sample else None})
        for field in fields:
            label = field.removesuffix("_change_m")
            amounts = [r[field]*r["circular_bbl_per_m"] for r in sample]
            out[f"{name}_{label}_visible_kbbl"] = sum(amounts)/1000 if sample else None
            out[f"{name}_{label}_overlap_kbbl"] = sum(a*r["fraction_of_capture_inside_eia_interval"]
                                                    for a, r in zip(amounts, sample, strict=True))/1000 if sample else None
    return out


def panels(rows, weeks):
    by_week, by_terminal = defaultdict(list), defaultdict(list)
    for row in rows:
        key = row["week_ending"], row["geometry_mode"]
        by_week[key].append(row)
        by_terminal[(*key, row["terminal_id"])].append(row)
    weekly = [dict(week_ending=week, geometry_mode=mode, release_date=release, inputs_available_before_utc=cutoff,
                   **aggregate(by_week[week, mode])) for week, (_, _, release, cutoff) in weeks.items()
              for mode in (RETROSPECTIVE, DATED)]
    terminals = [dict(week_ending=week, geometry_mode=mode, terminal_id=terminal, release_date=weeks[week][2],
                      inputs_available_before_utc=weeks[week][3], **aggregate(items))
                 for (week, mode, terminal), items in sorted(by_terminal.items())]
    return weekly, terminals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "validation", "holdout"), default="development")
    phase = parser.parse_args().phase
    stem = f"weekly_shadow_registered_{phase}"
    weeks = study_weeks(phase)
    native, saved, geometry = observation_sources(phase)
    jobs = prepare_jobs(weeks, native, saved, phase)
    rows = list(refit(jobs, native, saved, geometry))
    weekly, terminals = panels(rows, weeks)
    for name, table in (("pairs", rows), ("weeks", weekly), ("terminals", terminals)):
        export(f"{stem}_{name}", table)
    summary = {}
    for mode in (RETROSPECTIVE, DATED):
        items = [r for r in rows if r["geometry_mode"] == mode]
        selected = [r for r in weekly if r["geometry_mode"] == mode]
        summary[mode] = dict(pairs=len(items), fit_status=dict(Counter(r["fit_status"] for r in items)),
            offset_status=dict(Counter(r["offset_status"] for r in items)),
            weeks_with_pairs={key: sum(r[f"{key}_resolved_pairs"] > 0 for r in selected)
                              for key in ("baseline", "registered", "both")},
            pairs_by_method={key: sum(r[f"{key}_resolved_pairs"] for r in selected)
                             for key in ("baseline", "registered", "both")})
    report = dict(study_phase=phase, reporting_weeks=len(weeks), weekly_rows=len(weekly),
        terminal_rows=len(terminals), geometry=summary, interpretation="Registered pairs retain the baseline native pixels and "
        "date-dependent sun and view angles. The two profiles share a centre after the independently checked offset. "
        "Loss ranges are descriptive. Overlap scaling assumes uniform movement between captures. EIA values and equity prices are excluded.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
