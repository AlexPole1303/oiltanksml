"""Compare Sentinel-2 spectral methods on the same aligned historical tank-weeks."""
import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from functools import lru_cache
from statistics import median
from types import SimpleNamespace

import numpy as np
import rasterio

from audit_sentinel2_alternate_registration import packet_inputs, pair_key
from audit_sentinel2_weekly_volume import PHASES, error_metrics, releases
from build_l1c_quality import gml_groups, native_mask, paint
from build_registered_shadow_development import DATED, baseline_pairs, registered_pair, study_weeks
from build_sentinel2_angles import interpolate, stencil
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export, read_only
from build_weekly_shadow_validation import legacy_geometry
from check_multiband_shadow import BANDS, matched_signals, spectral_kernels, tile_metadata
from collect_public import ROOT, write
from collect_s2_spectral_detector_masks import BANDS as NONRED_BANDS, STEM as MASK_INDEX, VOLUMES, recovery_sources
from collect_s2_spectral_detector_masks import HOLDOUT_STEM as HOLDOUT_MASK_INDEX
from fit_shadow_geometry import (CALIBRATION, bearing_vector, gap_trials, pixel_grid, response_profiles, shifted_grid)

STEM = "sentinel2_weekly_multiband"
SIDES = ("earlier", "later")
METHODS = ("red", "nir", "joint")
HOLDOUT_VOLUMES = ("sentinel2_holdout_weekly_tank_volumes", "sentinel2_alternate_registration_holdout_added_volumes")


@lru_cache(maxsize=24)
def detector_planes(raster, xml_path, files):
    with rasterio.open(ROOT / raster) as source:
        ref = SimpleNamespace(bounds=source.bounds, crs=source.crs, transform=source.transform, shape=source.shape)
    metadata = tile_metadata(xml_path)
    planes = {}
    for band, file in zip(NONRED_BANDS, files, strict=True):
        path = ROOT / file
        bits = np.zeros(ref.shape, dtype="uint16")
        if path.name.endswith((".gml", ".gml.gz")):
            groups, _ = gml_groups(path, ref, True, band)
            for detector, polygons in groups.items():
                bits |= paint(polygons, ref).astype("uint16") << (detector-1)
        else:
            codes = native_mask(path, ref, metadata, 1)[0]
            if codes.max() > 12:
                raise ValueError(f"Unexpected detector identifier for {band} in {file}")
            valid = codes > 0
            bits[valid] = np.uint16(1) << (codes[valid].astype("uint16")-1)
        planes[band] = bits
    return ref.transform, planes


def detector_geometry(observation, native, keep, scene, files, *, view_grid=None):
    transform, masks = detector_planes(scene["red_raster"], scene["tile_metadata_path"], files)
    origin, rows, columns = native[2:5]
    x, y = observation["center_east_m"], observation["center_north_m"]
    c, r = (~transform) * (x+origin[0], y+origin[1])
    start_col, start_row = round(c-.5), round(r-.5)
    if max(abs(c-.5-start_col), abs(r-.5-start_row)) > 1e-6:
        raise ValueError(f"Detector plane differs from original pixel grid for {observation['version_id']}")
    i, j = start_row+rows[keep], start_col+columns[keep]
    ids = {}
    for band, mask in masks.items():
        if i.min() < 0 or j.min() < 0 or i.max() >= mask.shape[0] or j.max() >= mask.shape[1]:
            raise ValueError(f"Physical samples fall outside original mask for {observation['version_id']}/{band}")
        bits = np.unique(mask[i, j])
        if len(bits) != 1 or not bits[0] or (int(bits[0]) & (int(bits[0])-1)):
            return "nonunique_or_missing_original_detector", None, ids
        ids[band] = int(bits[0]).bit_length()

    root = tile_metadata(scene["tile_metadata_path"])
    tile = root.find(".//Tile_Geocoding/Geoposition[@resolution='10']")
    left, top = float(tile.findtext("ULX")), float(tile.findtext("ULY"))
    weights = stencil(np.array([(x-left)/5000]), np.array([(top-y)/5000]))
    views = {"B04": bearing_vector(observation["red_view_zenith_deg"], observation["red_view_azimuth_grid_deg"])}
    for band, detector in ids.items():
        band_id = BANDS[band][0]
        grid = root.find(f".//Viewing_Incidence_Angles_Grids[@bandId='{band_id}'][@detectorId='{detector}']")
        if grid is None:
            return "original_detector_without_view_grid", None, ids
        if view_grid is None:
            zenith = interpolate(grid, "Zenith", weights)[0]
            azimuth = interpolate(grid, "Azimuth", weights, True)[0]
        else:
            zenith, azimuth = view_grid(band, detector, grid, x, y, left, top)
        if not np.isfinite(zenith) or not np.isfinite(azimuth):
            return "original_detector_without_centroid_angles", None, ids
        views[band] = bearing_vector(zenith, azimuth+observation["true_north_grid_bearing_deg"])
    return "original_detector_confirmed", views, ids


def spectral_profiles(path, observation, scene, shell, kernels, files, east, north, *, view_grid=None):
    with closing(read_only(path)) as db:
        db.row_factory = sqlite3.Row
        native = pixel_grid(db, observation)
        signals, keep = matched_signals(db, observation, native)
    count = int(keep.sum())
    if count < 12:
        return "insufficient_common_pixels", count, None, {}
    status, views, ids = detector_geometry(observation, native, keep, scene, files, view_grid=view_grid)
    if views is None:
        return status, count, None, ids
    radius = observation["radius_m"]
    gaps = gap_trials(shell, observation["roof_type"])
    model_shell = shell if shell and shell >= 5 else 0
    sun = bearing_vector(observation["sun_zenith_deg"], observation["sun_azimuth_grid_deg"])
    h, w, origin, row, col, _, weight = shifted_grid(native, east, north)
    row, col, weight = row[keep], col[keep], weight[keep]
    weight_sum = float(weight.sum())
    results, normalized = {}, []
    for band, signal in signals.items():
        variance = float(np.dot(weight, (signal-np.average(signal, weights=weight))**2))
        if not np.isfinite(variance) or variance <= 0:
            return "no_band_variation", count, None, ids
        grid = h, w, origin, row, col, signal, weight
        shifts, loss, _, reverse = response_profiles(
            grid, radius, model_shell, gaps, sun, views[band], kernels[observation["platform"], band], reversed_sun=True)
        normalized.append((loss/variance, reverse/variance))
        if band in ("B04", "B08"):
            results[{"B04": "red", "B08": "nir"}[band]] = shifts, gaps, loss, reverse, count, weight_sum
    loss, reverse = np.mean(normalized, axis=0)
    results["joint"] = shifts, gaps, loss, reverse, count, 1.
    return "compared_full_pixels" if keep.all() else "compared_restricted_pixels", count, results, ids


def previous_inputs(phase):
    return {pair_key(row): (packet + ".sqlite", row) for mode, packet, _, row in baseline_pairs(phase) if mode == DATED}


def paired_source(row, previous, alternate):
    if alternate:
        native = [(row[f"{side}_native_packet"], int(row[f"{side}_native_observation_id"])) for side in SIDES]
        east, north = (float(row[f"later_minus_earlier_{axis}_m"])/2 for axis in ("east", "north"))
    else:
        packet, saved = previous[pair_key(row)]
        native = [(packet, int(saved[f"{side}_observation_id"])) for side in SIDES]
        east, north = (float(row[f"offset_{axis}_m"])/2 for axis in ("east", "north"))
    return native, east, north


def jobs(phase, *, include_alternates=True):
    previous = previous_inputs(phase)
    original = {pair_key(r): r for r in read_csv(f"weekly_shadow_registered_{phase}_pairs")
                if r["geometry_mode"] == DATED and r["registered_both_shadow_diagnostics_pass"] == "True"}
    tables = HOLDOUT_VOLUMES if phase == "holdout" else VOLUMES
    alternate_stem = "sentinel2_alternate_registration_holdout_pairs" if phase == "holdout" else "sentinel2_alternate_registration_pairs"
    alternate = {pair_key(r):r for r in read_csv(alternate_stem)
                 if r["study_phase"] == phase and r["registered_both_shadow_diagnostics_pass"] == "True"} if include_alternates else {}
    for table in tables if include_alternates else tables[:1]:
        for volume in read_csv(table):
            if volume["study_phase"] != phase:
                continue
            key = pair_key(volume)
            is_alternate = table != tables[0]
            if is_alternate and key in original:
                raise ValueError(f"An alternate replaces the baseline at {key}")
            row = alternate[key] if is_alternate else original[key]
            if any(row[f"{side}_version"] != volume[f"{side}_sentinel2_version"] for side in SIDES):
                raise ValueError(f"The source pair differs from the volume row at {key}")
            yield dict(volume, alternate_capture_pair=is_alternate), *paired_source(row, previous, is_alternate)


def recovery_jobs(phase, accepted, *, include_alternates=True):
    previous = previous_inputs(phase)
    for alternate, row in recovery_sources(phase, accepted, include_alternates=include_alternates):
        area = float(row["circular_bbl_per_m"])/1000
        volume = {field: row[field] for field in ("week_ending", "release_date", "candidate_id", "terminal_id")}
        volume.update(study_phase=phase, cutoff_utc=row["inputs_available_before_utc"], circular_kbbl_per_m=area,
            capture_change_kbbl=float(row["registered_change_m"])*area, baseline_red_resolved=False,
            alternate_capture_pair=alternate, archive_available_at_utc=max(
                datetime.fromisoformat(row[f"{side}_archive_available_at_utc"]) for side in SIDES).isoformat())
        for side in SIDES:
            volume.update({f"{side}_sentinel2_version": row[f"{side}_version"],
                           f"{side}_capture_at_utc": row[f"{side}_capture_at_utc"],
                           f"{side}_gap_m": row[f"{side}_registered_gap_m"]})
        yield volume, *paired_source(row, previous, alternate)


def measure(phase, geometry, kernels, masks, source_jobs=None, *, native_geometry=False):
    @lru_cache(maxsize=4096)
    def profiles(tank, packet, number, east, north, version):
        path, rows, scenes, _ = packet_inputs(packet)
        o = rows[number]
        if (o["candidate_id"], o["version_id"]) != (tank, version):
            raise ValueError(f"Exact capture identity differs for {tank}/{version}")
        old = geometry[tank]
        shell = float(old["prior_rim_z_p95_native"]) - float(old["prior_ground_z_p50_native"])
        scene = scenes[o["scene_id"]]
        o = dict(o, tile_metadata_path=scene["tile_metadata_path"])
        if not native_geometry:
            o.update(center_east_m=float(old["center_east_m"]),center_north_m=float(old["center_north_m"]),
                     radius_m=float(old["diameter_m"])/2)
        files = tuple(masks[version, band]["local_file"] for band in NONRED_BANDS)
        return spectral_profiles(path, o, scene, shell, kernels, files, east, north)

    def solve(job):
        volume, sources, east, north = job
        result = {field: volume[field] for field in ("study_phase", "week_ending", "release_date", "candidate_id",
                    "terminal_id", "earlier_sentinel2_version", "later_sentinel2_version", "earlier_capture_at_utc",
                    "later_capture_at_utc", "archive_available_at_utc", "cutoff_utc", "capture_change_kbbl")}
        baseline_resolved = volume.get("baseline_red_resolved", True)
        if not baseline_resolved:
            result["alternate_capture_pair"] = volume["alternate_capture_pair"]
        cutoff = datetime.fromisoformat(volume["cutoff_utc"])
        inputs = []
        for side, source, sign in zip(SIDES, sources, (-1, 1), strict=True):
            version = volume[f"{side}_sentinel2_version"]
            available = max(datetime.fromisoformat(masks[version, band]["source_created_at_utc"]) for band in NONRED_BANDS)
            fitted = profiles(volume["candidate_id"], *source, sign*east, sign*north, version) if available < cutoff else (
                "original_detector_masks_after_cutoff", None, None, {})
            status, count, _, ids = fitted
            inputs.append(fitted)
            result.update({f"{side}_{field}": value for field, value in (
                ("status", status), ("common_pixels", count), ("mask_available_at_utc", available.isoformat()),
                ("B03_detector", ids.get("B03")), ("B08_detector", ids.get("B08")))})
        for name in METHODS:
            fit = registered_pair(*(item[2][name] for item in inputs)) if all(item[2] is not None for item in inputs) else None
            if name == "red" and fit and all(item[0] == "compared_full_pixels" for item in inputs):
                if (any(fit[f"{side}_registered_gap_m"] != float(volume[f"{side}_gap_m"]) for side in SIDES)
                    or fit["registered_both_shadow_diagnostics_pass"] != baseline_resolved):
                    raise ValueError(f"Same-pixel B04 replay differs at {pair_key(volume)}")
            result.update({f"{name}_{field}": value for field, value in (
                ("earlier_gap_m", fit["earlier_registered_gap_m"] if fit else None),
                ("later_gap_m", fit["later_registered_gap_m"] if fit else None),
                ("earlier_fit_status", fit["earlier_registered_status"] if fit else None),
                ("later_fit_status", fit["later_registered_status"] if fit else None),
                ("resolved", fit["registered_both_shadow_diagnostics_pass"] if fit else False),
                ("change_kbbl", fit["registered_change_m"]*float(volume["circular_kbbl_per_m"]) if fit else None))})
        return result

    sample = list(jobs(phase) if source_jobs is None else source_jobs)
    with ThreadPoolExecutor(max_workers=5) as pool:
        for i, row in enumerate(pool.map(solve, sample), 1):
            if i % 200 == 0 or i == len(sample):
                print(f"{phase}: checked spectral inputs for {i}/{len(sample)} tank-weeks", flush=True)
            yield row


def weekly_comparison(phase, rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["week_ending"]].append(row)
    metrics, output = {}, []
    targets = releases(phase)
    for scope in ("all_common_pixels", "all_methods_resolved"):
        errors = defaultdict(list)
        for target in targets:
            items = grouped[target["week_ending"]]
            matched = [r for r in items if r["red_change_kbbl"] is not None and (
                scope == "all_common_pixels" or all(r[f"{method}_resolved"] for method in METHODS))]
            if any(r["release_date"] != target["release_date"] for r in matched):
                raise ValueError(f"Mismatched EIA clock in {target['week_ending']}")
            actual = target["change_kbbl_as_known_at_release"]
            amounts = {m: sum(row[f"{m}_change_kbbl"] for row in matched) if matched else None for m in METHODS}
            for method, amount in amounts.items():
                if amount is not None and actual is not None:
                    errors[method].append(amount-actual)
            output.append(dict(study_phase=phase, comparison_scope=scope, week_ending=target["week_ending"],
                release_date=target["release_date"], eia_first_release_change_kbbl=actual, eia_url=target["source_url"],
                matched_tanks=len(matched), **{f"{m}_visible_change_kbbl": a for m, a in amounts.items()}))
        metrics[scope] = {method: error_metrics(signed) for method, signed in errors.items()}
    return output, metrics


def recovery_panel(phase, baseline, rows, calendar):
    original = defaultdict(list)
    for row in baseline:
        original[row["week_ending"]].append(float(row["capture_change_kbbl"]))
    choices = {method: {} for method in METHODS[1:]}
    # The generator yields standard endpoint pairs before alternatives.
    for row in rows:
        for method, selected in choices.items():
            if row[f"{method}_resolved"]:
                selected.setdefault(pair_key(row), row)
    additions = {method: defaultdict(list) for method in choices}
    for method, selected in choices.items():
        for (week, _), row in selected.items():
            additions[method][week].append(row[f"{method}_change_kbbl"])
    output = []
    for target in calendar:
        week = target["week_ending"]
        raw = original[week]
        total = sum(raw)
        row = dict(study_phase=phase, **target, original_tanks=len(raw), original_visible_change_kbbl=total if raw else None)
        for method, added in additions.items():
            sample = added[week]
            row.update({f"{method}_additional_tanks": len(sample),
                        f"{method}_expanded_visible_change_kbbl": total+sum(sample) if raw or sample else None})
        output.append(row)
    for row in rows:
        for method, selected in choices.items():
            row[f"selected_for_{method}_expansion"] = selected.get(pair_key(row)) is row
    return output, choices


def recovery_comparison(phase, baseline, rows):
    calendar = (dict(week_ending=r["week_ending"],release_date=r["release_date"],
                     eia_first_release_change_kbbl=r["change_kbbl_as_known_at_release"], eia_url=r["source_url"])
                for r in releases(phase))
    output, choices = recovery_panel(phase, baseline, rows, calendar)
    summary = {}
    for method, selected in choices.items():
        matched = [r for r in output if r["original_tanks"] and r["eia_first_release_change_kbbl"] is not None]
        changed = [r for r in matched if r[f"{method}_additional_tanks"]]
        new = [r for r in output if not r["original_tanks"] and r[f"{method}_additional_tanks"]]
        scores = {}
        for scope, sample in (("all_original_weeks", matched), ("changed_original_weeks", changed), ("new_weeks", new)):
            fields = ("original_visible_change_kbbl", f"{method}_expanded_visible_change_kbbl")
            scores[scope] = {field: error_metrics([r[field]-r["eia_first_release_change_kbbl"] for r in sample
                                                 if r[field] is not None and r["eia_first_release_change_kbbl"] is not None])
                             for field in fields if any(r[field] is not None and r["eia_first_release_change_kbbl"] is not None
                                                        for r in sample)}
        summary[method] = dict(recovered_tank_weeks=len(selected),
            selected_alternate_pairs=sum(row["alternate_capture_pair"] for row in selected.values()),
            baseline_weeks=sum(bool(r["original_tanks"]) for r in output), added_weeks=len(new), scores=scores)
    return output, summary


def run_recovery(geometry, kernels, masks):
    candidates, weeks, summary = [], [], {}
    volumes = [r for table in VOLUMES for r in read_csv(table)]
    for phase in PHASES:
        baseline = [r for r in volumes if r["study_phase"] == phase]
        accepted = {pair_key(r) for r in baseline}
        rows = list(measure(phase, geometry, kernels, masks, recovery_jobs(phase, accepted)))
        unique = {(*pair_key(r), *(r[f"{side}_sentinel2_version"] for side in SIDES)) for r in rows}
        if len(unique) != len(rows) or any(pair_key(r) in accepted for r in rows):
            raise ValueError(f"Spectral recovery must use distinct pairs outside the accepted set in {phase}")
        panel, metrics = recovery_comparison(phase, baseline, rows)
        summary[phase] = dict(tested_pairs=len(rows), candidate_tank_weeks=len({pair_key(r) for r in rows}),
                              matched_spectral_pairs=sum(r["red_change_kbbl"] is not None for r in rows),
                              baseline_tank_weeks=len(baseline), methods=metrics)
        candidates.extend(rows)
        weeks.extend(panel)
    export(f"{STEM}_recovery_candidates", candidates)
    export(f"{STEM}_recovery_eia_comparison", weeks)
    return dict(phases=summary, interpretation="The original B04 volume series is kept. For absent tank-weeks, "
        "each spectral method may add a measurement if both dates pass the shadow checks. Standard endpoint "
        "pairs are considered first and alternates second. All candidates have independently checked exterior "
        "registration. No EIA outcome selects observations, and the closed period is excluded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--recover", action="store_true", help="Test pairs rejected by the original B04 rule")
    scope.add_argument("--holdout-recover", action="store_true", help="Recover holdout fixed pairs without inventory labels")
    parser.add_argument("--with-alternates", action="store_true", help="Include verified holdout alternate registrations")
    args = parser.parse_args()
    if args.with_alternates and not args.holdout_recover:
        parser.error("--with-alternates requires --holdout-recover")
    geometry = legacy_geometry()
    masks = {(row["version_id"], row["band"]): row for row in read_csv(HOLDOUT_MASK_INDEX if args.holdout_recover else MASK_INDEX)}
    with closing(read_only(CALIBRATION)) as db:
        kernels = spectral_kernels(db)
    if args.holdout_recover:
        phase = "holdout"
        tables = HOLDOUT_VOLUMES if args.with_alternates else HOLDOUT_VOLUMES[:1]
        baseline = [r for table in tables for r in read_csv(table)]
        accepted = {pair_key(r) for r in baseline}
        rows = list(measure(phase, geometry, kernels, masks, recovery_jobs(phase, accepted, include_alternates=args.with_alternates)))
        calendar = (dict(week_ending=week, release_date=release, cutoff_utc=cutoff)
                    for week, (_, _, release, cutoff) in study_weeks(phase).items())
        panel, choices = recovery_panel(phase, baseline, rows, calendar)
        export(f"{STEM}_holdout_recovery_candidates", rows)
        export(f"{STEM}_holdout_recovery_weeks", panel)
        summary = dict(tested_pairs=len(rows), included_alternate_pairs=args.with_alternates,
                       matched_spectral_pairs=sum(r["red_change_kbbl"] is not None for r in rows),
                       recovered_tank_weeks={method:len(selected) for method,selected in choices.items()},
                       new_weeks={method:sum(not r["original_tanks"] and r[f"{method}_additional_tanks"]>0 for r in panel)
                                  for method in choices})
        payload = json.dumps(summary,indent=2)+"\n"
        write(DERIVED/f"{STEM}_holdout_recovery_summary.json",payload.encode())
        print(payload,end="",flush=True)
        return
    if args.recover:
        result = run_recovery(geometry, kernels, masks)
        payload = json.dumps(result, indent=2) + "\n"
        write(DERIVED / f"{STEM}_recovery_summary.json", payload.encode())
        print(payload, end="", flush=True)
        return
    report, measurements, weeks = {}, [], []
    for phase in PHASES:
        rows = list(measure(phase, geometry, kernels, masks))
        if len(rows) != len({pair_key(row) for row in rows}):
            raise ValueError(f"Duplicate tank-weeks in {phase}")
        panel, metrics = weekly_comparison(phase, rows)
        matched = [r for r in rows if r["red_change_kbbl"] is not None]
        report[phase] = dict(input_tank_weeks=len(rows), matched_spectral_tank_weeks=len(matched),
            endpoint_statuses=dict(Counter(r[f"{side}_status"] for r in rows for side in SIDES)),
            resolved_pairs={m: sum(r[f"{m}_resolved"] for r in matched) for m in METHODS},
            median_absolute_change_difference_to_red_kbbl={
                m: median(abs(r[f"{m}_change_kbbl"]-r["red_change_kbbl"]) for r in matched) for m in METHODS[1:]},
            visible_change_vs_first_issue_eia=metrics)
        measurements.extend(rows)
        weeks.extend(panel)
    export(f"{STEM}_tank_weeks", measurements)
    export(f"{STEM}_eia_comparison", weeks)
    payload = json.dumps(dict(phases=report, interpretation="The same preselected capture identities, exterior registration, "
        "geometry and native sample intersection are used for all variants. Original B03/B08 detector masks must "
        "be available before the release cutoff and identify a single detector on every fitted pixel. The view "
        "direction is taken from that detector's angle grid at the dated tank centre. "
        "The joint method averages the per-band loss after normalizing by weighted signal variation. The EIA comparison "
        "uses the same tanks in each week and excludes the closed period."), indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
