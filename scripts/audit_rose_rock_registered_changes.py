"""Test paired optical height changes around the Rose Rock NRC event."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from functools import lru_cache
from itertools import combinations
from statistics import median

from audit_sentinel2_alternate_registration import packet_inputs
from build_cushing_nrc_rose_rock_landsat import STEM as LANDSAT
from build_cushing_nrc_incidents import STEM as NRC
from build_registered_shadow_development import registered_pair
from build_s1_reference_matches import DERIVED, export, read
from build_sentinel2_wider_registration import CHECKED, OFFSETS, RADIUS_M, measure
from build_shadow_registration_inputs import SEARCH_RADIUS_M, describe_source
from build_weekly_shadow_panel import read_only
from check_weekly_multiband_shadow import METHODS, spectral_kernels, spectral_profiles
from collect_public import write
from collect_s2_spectral_detector_masks import BANDS

INPUT = f"{LANDSAT}_lidar_2022"
STEM = "cushing_nrc_rose_rock_registered_changes"
TILES = ("T14SPE", "T14SQE")
PASS = "registered_both_shadow_diagnostics_pass"


def registration_inputs(sources, roofs):
    by_tile = defaultdict(list)
    for number, source in enumerate(sorted(sources.values(), key=lambda r: r["captured_at_utc"]), 1):
        row = describe_source(number, source)
        by_tile["T" + row["mgrs_tile"]].append(row)
    if set(by_tile) != set(TILES) or any(len(group) != 3 for group in by_tile.values()):
        raise ValueError("Expected three source scenes in each of two Sentinel-2 tiles")
    links = []
    for tile, sources in sorted(by_tile.items()):
        for first, last in combinations(sources, 2):
            interval = (first["captured_at_utc"][:10], last["captured_at_utc"][:10])
            ident = "/".join((*interval, tile))
            for roof in sorted(roofs):
                links.append(("nrc_roof_change", "retrospective", ident, roof, "EFR", "semgroup_cushing", "",
                              "", "", first, last))
    return links


def exterior_offsets(links):
    local, local_patches = measure(links, radius=SEARCH_RADIUS_M)
    pending = [r for r in links if local[r[0], r[2], r[3]]["registration_status"] != CHECKED]
    recovered, wide_patches = measure(pending) if pending else ({}, [])
    rows = []
    for key, row in local.items():
        wide = recovered.get(key)
        use_wide = bool(wide and wide["registration_status"] == CHECKED)
        selected = wide if use_wide else row
        rows.append(dict(selected, local_registration_status=row["registration_status"],
                         wider_registration_status=wide["registration_status"] if wide else None,
                         selected_radius_m=RADIUS_M if use_wide else SEARCH_RADIUS_M))
    evidence = [dict(r, search_radius_m=radius) for radius, records in (
        (SEARCH_RADIUS_M, local_patches), (RADIUS_M, wide_patches)) for r in records]
    return rows, evidence


def fit_changes(path, obs, scenes, offsets, geometry, source_rows, masks):
    with closing(read_only(path)) as db:
        kernels = spectral_kernels(db)
    observations = {(r["candidate_id"], r["version_id"]): r for r in obs.values()}

    @lru_cache(maxsize=512)
    def profile(roof, version, east, north):
        row = observations[roof, version]
        source = source_rows[roof, version]
        if int(source["observation_id"]) != row["observation_id"]:
            raise ValueError(f"Saved observation differs from native packet for {roof}/{version}")
        if source["fit_status"] in ("incomplete_optical_geometry_or_quality", "dated_geometry_outside_chip"):
            return source["fit_status"], None, None
        g = geometry[roof]
        shell = float(g["prior_rim_z_p95_native"]) - float(g["prior_ground_z_p50_native"])
        values = dict(row, center_east_m=float(g["center_east_m"]), center_north_m=float(g["center_north_m"]),
                      radius_m=float(g["diameter_m"])/2)
        status, count, profiles, _ = spectral_profiles(path, values, scenes[row["scene_id"]], shell, kernels,
            tuple(masks[version, band] for band in BANDS), east, north)
        return status, count, profiles

    def fit(row, *, apply_offset):
        roof = row["candidate_id"]
        checked = row["registration_status"] == CHECKED
        east, north = (float(row[k])/2 for k in OFFSETS) if checked and apply_offset else (0, 0)
        dates = row["week_or_control_day"].split("/")
        result = dict(candidate_id=roof, earlier_day=dates[0], later_day=dates[1], tile=dates[2],
                      earlier_version=row["earlier_version"], later_version=row["later_version"],
                      registration_status=row["registration_status"], exterior_offset_applied=checked and apply_offset)
        endpoints = []
        for side, direction in (("earlier", -1), ("later", 1)):
            status, count, profiles = (profile(roof, row[f"{side}_version"], direction*east, direction*north)
                                       if checked else ("registration_unverified", None, None))
            result.update({f"{side}_pixel_status": status, f"{side}_common_pixels": count})
            endpoints.append(profiles)
        for method in METHODS:
            values = registered_pair(*(p[method] for p in endpoints)) if all(endpoints) else {}
            for field in ("earlier_registered_status", "later_registered_status", PASS, "registered_change_m",
                          "registered_loss_range_min_m", "registered_loss_range_max_m"):
                result[f"{method}_{field}"] = values.get(field)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        corrected = list(pool.map(lambda r: fit(r, apply_offset=True), offsets))
        zero = list(pool.map(lambda r: fit(r, apply_offset=False), offsets))
    return corrected, zero


def consensus(fits):
    by_roof = defaultdict(dict)
    for r in fits:
        key = r["earlier_day"], r["later_day"], r["candidate_id"]
        by_roof[key][r["tile"]] = r
    results = []
    for (first, last, roof), group in sorted(by_roof.items()):
        if set(group) != set(TILES):
            raise ValueError(f"Incomplete tile comparison for {roof}/{first}/{last}")
        checked = all(r["registration_status"] == CHECKED for r in group.values())
        common = [r for r in group.values() for m in METHODS if r[f"{m}_{PASS}"]]
        row = dict(candidate_id=roof, earlier_day=first, later_day=last, both_tiles_registered=checked,
                   passing_tile_method_fits=len(common), all_six_fits_passed=len(common) == 6)
        for method in METHODS:
            a, b = (group[tile] for tile in TILES)
            passed = bool(a[f"{method}_{PASS}"] and b[f"{method}_{PASS}"])
            lo, hi = (None, None)
            if passed:
                lo = min(x[f"{method}_registered_loss_range_min_m"] for x in (a, b))
                hi = max(x[f"{method}_registered_loss_range_max_m"] for x in (a, b))
            row.update({f"{method}_both_tiles_passed": passed, f"{method}_combined_min_m": lo, f"{method}_combined_max_m": hi,
                        f"{method}_direction": "increase" if passed and lo > 0 else "decrease" if passed and hi < 0 else
                        "includes_zero" if passed else "incomplete"})
        if row["all_six_fits_passed"]:
            low = min(row[f"{m}_combined_min_m"] for m in METHODS)
            high = max(row[f"{m}_combined_max_m"] for m in METHODS)
            direction = "increase" if low > 0 else "decrease" if high < 0 else "includes_zero"
        else:
            low, high, direction = None, None, "incomplete"
        results.append(dict(row, combined_loss_range_min_m=low, combined_loss_range_max_m=high,
                            direction_across_all_six=direction, admitted_nrc_inventory_label=False))
    return results


def interval_counts(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["earlier_day"], row["later_day"]].append(row)
    return [dict(earlier_day=first, later_day=last, candidate_roofs=len(group),
        both_tiles_registered=sum(r["both_tiles_registered"] for r in group),
        both_tiles_passed_by_method={m: sum(r[f"{m}_both_tiles_passed"] for r in group) for m in METHODS},
        all_six_fits_passed=sum(r["all_six_fits_passed"] for r in group),
        stable_sign_all_six=sum(r["direction_across_all_six"] in ("increase", "decrease") for r in group))
        for (first, last), group in groups.items()]


def radar_coverage(roofs):
    event = json.loads((DERIVED / f"{NRC}_summary.json").read_bytes())["rose_rock_inventory_event"]
    event_time = datetime.fromisoformat(event["incident_local_time"])
    rows = []
    for row in read("s1_cushing_catalog"):
        if row["product_type"] not in ("IW_SLC__1S", "IW_GRDH_1S"):
            continue
        start = datetime.fromisoformat(row["capture_start_utc"])
        end = datetime.fromisoformat(row["capture_end_utc"])
        delta = (start + (end-start)/2) - event_time
        if abs(delta) > timedelta(days=7):
            continue
        covered = roofs.intersection(json.loads(row["covered_candidate_ids_json"]))
        if covered:
            rows.append({k: row[k] for k in ("product_id", "product_name", "product_type", "relative_orbit", "direction",
                "polarization", "capture_start_utc", "capture_end_utc", "catalog_url", "source_file")}
                | dict(hours_from_reported_event=delta.total_seconds()/3600, covered_roofs=len(covered),
                       covered_candidate_ids_json=json.dumps(sorted(covered)), native_pixels_measured=False))
    return sorted(rows, key=lambda r: (r["hours_from_reported_event"], r["product_type"]))


def main():
    rows = read(f"{INPUT}_inventory_candidates")
    roofs = {r["candidate_id"] for r in rows if r["capacity_at_least_reported_inventory"] == "True"}
    geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry")
                if r["candidate_id"] in roofs and r["modern_input_status"] == "available"}
    if geometry.keys() != roofs:
        raise ValueError("A capacity-compatible roof lacks the dated geometry")
    path, obs, scenes, _ = packet_inputs(f"{INPUT}_sentinel2_native.sqlite")
    sources = {(r["candidate_id"], r["version_id"]): r for r in read(f"{INPUT}_sentinel2_fits")}
    masks = {(r["version_id"], r["band"]): r["local_file"] for r in read(f"{INPUT}_spectral_masks")}
    offsets, patches = exterior_offsets(registration_inputs(scenes, roofs))
    fits, zero = fit_changes(path, obs, scenes, offsets, geometry, sources, masks)
    results = consensus(fits)
    zero_results = consensus(zero)
    radar = radar_coverage(roofs)
    checks = [r["check_median_residual_m"] for r in offsets if r["registration_status"] == CHECKED]
    summary = dict(capacity_compatible_roofs=len(roofs), offsets=len(offsets), exterior_patch_rows=len(patches),
        registration_status=Counter(r["registration_status"] for r in offsets),
        recovered_with_wider_radius=sum(r["selected_radius_m"] == RADIUS_M for r in offsets),
        median_spatial_check_residual_m=median(checks) if checks else None, intervals=interval_counts(results),
        shared_centre_zero_offset_intervals=interval_counts(zero_results),
        sentinel1_slc_products_within_seven_days=sum(r["product_type"] == "IW_SLC__1S" for r in radar),
        sentinel1_slc_products_within_48h=sum(r["product_type"] == "IW_SLC__1S" and
                                             abs(r["hours_from_reported_event"]) <= 48 for r in radar),
        admitted_nrc_inventory_labels=0, used_in_frozen_replay=False)
    for suffix, rows in (("offsets", offsets), ("patches", patches), ("tile_fits", fits), ("roof_changes", results),
                         ("zero_offset_tile_fits", zero), ("zero_offset_roof_changes", zero_results),
                         ("sentinel1_catalog", radar)):
        export(f"{STEM}_{suffix}", rows)
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
