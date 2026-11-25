"""Run a Landsat 7 shadow inversion and record sensor-response sensitivity."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from statistics import median

import numpy as np
from affine import Affine
from pyproj import Transformer
from rasterio.windows import from_bounds

from audit_shadow_precision import describe
from build_cushing_gedi_native_optical_controls import identities, references
from build_landsat_c2_b8_measurements import roof_touched
from build_sentinel2_angles import ground_coordinates
from build_terminal_development_measurements import BARREL_M3
from collect_cushing_landsat7 import DERIVED, PIXELS, ROOT, STEM, export, legacy_geometry, rows, save_json
from fit_shadow_geometry import bearing_vector, fit_response, gap_fit_status, gap_trials

PIXEL_M, STEP_M = 15, 3
FWHM_M = (18, 21, 24)
CENTRAL_FWHM = 21
MAX_LASER_HOURS = 168
RESOLVED = "resolved_in_geometry_diagnostic"
PAPER = "https://www.usgs.gov/publications/landsat-7-orbit-modulation-transfer-function-estimation"


def response_kernel(fwhm):
    sigma = fwhm / math.sqrt(8 * math.log(2))
    half_width = math.ceil(4 * sigma / STEP_M)
    coordinates = np.arange(-half_width, half_width + 1) * STEP_M
    profile = np.exp(-0.5 * (coordinates / sigma)**2)
    kernel = np.outer(profile, profile)
    return (kernel / kernel.sum()).astype("float32")


def native_chip(data, r, center):
    pan, usable, transform = data
    east, north = center
    radius = float(r["radius_m"])
    pad = radius + 24
    window = from_bounds(east-pad, north-pad, east+pad, north+pad, transform=transform)
    c0, r0 = max(0, math.floor(window.col_off)), max(0, math.floor(window.row_off))
    c1 = min(pan.shape[1], math.ceil(window.col_off + window.width))
    r1 = min(pan.shape[0], math.ceil(window.row_off + window.height))
    if c0 >= c1 or r0 >= r1:
        raise ValueError(f"No Landsat 7 B8 pixels near {r['candidate_id']}")
    iy, ix = np.indices((r1-r0, c1-c0))
    x, y = transform * (ix + c0 + .5, iy + r0 + .5)
    x, y = x-east, y-north
    distance = np.hypot(x, y)
    valid = usable[r0:r1, c0:c1]
    signal = pan[r0:r1, c0:c1]
    roof = roof_touched(x, y, 0, 0, radius)
    good = valid & (distance <= radius+20)
    weights = np.exp(-.5*(np.maximum(distance[good]-radius, 0)/9)**2)
    return (r1-r0, c1-c0, (x[0, 0], y[0, 0]), iy[good], ix[good], signal[good], weights), int(roof.sum()), int((roof&valid).sum())


def shadow_fits(data, r, geometry, item, kernels, offset):
    tank = r["candidate_id"]
    east, north, radius = (float(r[k]) for k in ("center_east_m", "center_north_m", "radius_m"))
    if not math.isclose(2*radius, float(geometry["diameter_m"])):
        raise ValueError(f"The B8 roof geometry differs from the dated LiDAR geometry: {tank}")
    shell = float(geometry["prior_rim_z_p95_native"]) - float(geometry["prior_ground_z_p50_native"])
    if shell < 5 or r["angle_valid"] != "True":
        raise ValueError(f"Inconsistent ETM+ roof or angle geometry for {tank}")
    crs = "EPSG:32614"
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(east, north)
    north_bearing = ground_coordinates(crs, ((lon, lat),))[2][0]
    sun = bearing_vector(float(r["sun_zenith_deg"]), float(r["sun_azimuth_deg"]) + north_bearing)
    view = bearing_vector(float(r["view_zenith_deg"]), float(r["view_azimuth_deg"]) + north_bearing)
    solar_cos = math.cos(math.radians(float(r["sun_zenith_deg"])))
    gaps = gap_trials(shell, "EFR")
    scenarios = {"native_location": ((east, north), 9)}
    if offset and offset["registration_status"] == "checked_offset":
        delta_e, delta_n = (float(offset[f"later_minus_earlier_{axis}_m"]) for axis in ("east", "north"))
        scenarios["relative_offset"] = ((east-delta_e, north-delta_n), 6)
    for basis, (center, span) in scenarios.items():
        raw, cells, good_cells = native_chip(data, r, center)
        reflectance = (raw[-2] * item["b8_reflectance_mult"] + item["b8_reflectance_add"]) / solar_cos
        grid = (*raw[:-2], reflectance, raw[-1])
        count = len(reflectance)
        core = dict(product_id=r["product_id"], candidate_id=tank, basis=basis, terminal_id=r["terminal_id"],
                    captured_at_utc=r["captured_at_utc"], diameter_m=2*radius, shell_above_ground_m=shell,
                    center_east_m=center[0], center_north_m=center[1], fitted_pixels=count,
                    roof_touching_pixels=cells, roof_touching_pixels_usable=good_cells)
        for fwhm, kernel in kernels.items():
            result = dict(core, effective_fwhm_m=fwhm, fit_status="insufficient_native_b8_pixels",
                          estimated_gap_m=None, near_optimal_gap_min_m=None, near_optimal_gap_max_m=None,
                          relative_center_shift_east_m=None, relative_center_shift_north_m=None,
                          shadow_contrast=None, pixel_rmse=None, flat_rmse=None, reversed_sun_rmse=None,
                          center_search_on_boundary=None)
            if count < 12 or good_cells < 4:
                yield result
                continue
            shifts = tuple(range(-span, span+1, 3))
            gap, lower, upper, dx, dy, contrast, rmse, flat, placebo = fit_response(
                grid, radius, shell, gaps, sun, view, kernel, reversed_sun=True,
                pixel_m=PIXEL_M, step_m=STEP_M, shifts=shifts, subcell_samples=3)
            status = gap_fit_status(gaps, gap, lower, upper, rmse, flat, placebo)
            yield dict(result, fit_status=status, estimated_gap_m=float(gap), near_optimal_gap_min_m=lower,
                       near_optimal_gap_max_m=upper, relative_center_shift_east_m=dx, relative_center_shift_north_m=dy,
                       shadow_contrast=-contrast, pixel_rmse=rmse, flat_rmse=flat, reversed_sun_rmse=placebo,
                       center_search_on_boundary=max(abs(dx), abs(dy)) == span)


def native_raster(product):
    item = json.loads((PIXELS / product / "item.json").read_bytes())
    if item["product_id"] != product:
        raise ValueError(f"Landsat 7 packet has the wrong product ID for {product}")
    packet = ROOT / item["native_packet"]
    if hashlib.sha256(packet.read_bytes()).hexdigest() != item["local_sha256"]["native.npz"]:
        raise ValueError(f"Native Landsat 7 packet checksum mismatch for {product}")
    with np.load(packet) as stored:
        if str(stored["crs"]) != "EPSG:32614":
            raise ValueError(f"Unexpected Landsat 7 projection for {product}")
        raster = stored["pan_dn"], stored["pan_usable"], Affine(*stored["pan_transform"])
    return item, raster


def scene(job, legacy, kernels):
    product, roofs = job
    item, raster = native_raster(product)
    return [v for r, offset in roofs for v in shadow_fits(raster, r, legacy[r["candidate_id"]], item, kernels, offset)]


def errors(records):
    report = {}
    for cap in (3, 24, 48):
        subset = [r for r in records if r["absolute_separation_hours"] <= cap]
        report[f"within_{cap}h"] = describe([r["landsat_minus_sentinel_gap_m"] for r in subset]) | {
            "scenes":len({r["landsat_product_id"] for r in subset}), "roofs":len({r["candidate_id"] for r in subset})}
    return report


def laser_references(legacy, candidates):
    icesat = rows(DERIVED / "cushing_icesat2_native_profiles.csv")
    profiles = []
    for r in icesat:
        candidate = r["candidate_id"]
        if candidate not in candidates or r["passes_roof_profile_checks"] != "True":
            continue
        g = legacy[candidate]
        if r["geometry_dataset"] != g["dataset"] or not math.isclose(float(r["wall_navd88_m"]), float(g["prior_rim_z_p95_native"])):
            raise ValueError(f"ICESat-2 uses different wall geometry for {candidate}")
        profiles.append((candidate, "ICESat-2", f"{r['filename']}:{r['beam']}", r["first_roof_signal_photon_utc"],
                         r["last_roof_signal_photon_utc"], float(r["estimated_wall_to_roof_m"])))
    _, groups = references(identities()[0])
    for records in groups.values():
        for r in records:
            candidate = r["candidate_id"]
            if candidate in candidates and r["geometry_dataset"] == legacy[candidate]["dataset"]:
                time = r["observed_at_utc"]
                profiles.append((candidate, "GEDI", f"{r['l2a_granule']}:{r['shot_number']}", time, time, r["gap_m"]))
    return profiles


def laser_controls(legacy, roofs):
    by_roof = defaultdict(list)
    for (product, candidate), r in roofs.items():
        if product.endswith("_T1") and r["angle_valid"] == "True" and int(r["roof_pixels_passing_qa"]) > 0:
            by_roof[candidate].append(r)
    for candidate, sensor, source, first, last, gap in laser_references(legacy, by_roof):
        def separation(roof):
            return max(abs((datetime.fromisoformat(roof["captured_at_utc"]) - datetime.fromisoformat(t)).total_seconds())
                       for t in (first, last)) / 3600
        roof = min(by_roof[candidate], key=lambda r: (separation(r), r["product_id"]))
        hours = separation(roof)
        if hours <= MAX_LASER_HOURS:
            yield dict(candidate_id=candidate, laser_sensor=sensor, laser_source=source, laser_start_utc=first,
                       laser_end_utc=last, laser_gap_m=gap, landsat_product_id=roof["product_id"],
                       landsat_captured_at_utc=roof["captured_at_utc"], maximum_capture_separation_hours=hours)


def rapid_controls(legacy, kernels):
    path = DERIVED / f"{STEM}_rapid_archive_weekly_candidates.csv"
    if not path.is_file():
        return None
    pairs = rows(path)
    pending = defaultdict(set)
    for r in pairs:
        for side in ("previous", "current"):
            pending[r[f"{side}_product_id"]].add(r["candidate_id"])
    fits = {}
    for product, candidates in pending.items():
        item = json.loads((PIXELS / product / "item.json").read_bytes())
        roofs = {r["candidate_id"]: r for r in item["roofs"]}
        batch = [(roofs[key] | dict(product_id=product, captured_at_utc=item["captured_at_utc"], angle_valid="True"), None)
                 for key in sorted(candidates) if roofs[key]["angle_valid"]]
        for r in scene((product, batch), legacy, kernels):
            fits[product, r["candidate_id"], r["effective_fwhm_m"]] = r
    results = []
    for r in pairs:
        candidate = r["candidate_id"]
        width_pairs = [(fits[r["previous_product_id"], candidate, fwhm], fits[r["current_product_id"], candidate, fwhm])
                       for fwhm in FWHM_M]
        before, after = width_pairs[FWHM_M.index(CENTRAL_FWHM)]
        central_ok = all(x["fit_status"] == RESOLVED and not x["center_search_on_boundary"] for x in (before, after))
        stable = all(x["fit_status"] == RESOLVED and not x["center_search_on_boundary"] for pair in width_pairs for x in pair)
        scale = math.pi * (before["diameter_m"]/2)**2 / BARREL_M3
        bounds = [(a["near_optimal_gap_min_m"]-b["near_optimal_gap_max_m"], a["near_optimal_gap_max_m"]-b["near_optimal_gap_min_m"])
                  for a, b in width_pairs] if stable else []
        lowest = min(a for a, _ in bounds) if bounds else None
        highest = max(b for _, b in bounds) if bounds else None
        results.append(r | dict(previous_fit_status=before["fit_status"], current_fit_status=after["fit_status"],
            previous_fitted_pixels=before["fitted_pixels"], current_fitted_pixels=after["fitted_pixels"],
            previous_shift_at_boundary=before["center_search_on_boundary"], current_shift_at_boundary=after["center_search_on_boundary"],
            central_both_height_fits_pass=central_ok, all_response_widths_both_height_fits_pass=stable,
            capture_to_capture_change_bbl=scale*(before["estimated_gap_m"]-after["estimated_gap_m"]) if central_ok else None,
            near_optimal_change_min_m=lowest, near_optimal_change_max_m=highest,
            stable_nonzero_change_sign=bool(stable and (lowest > 0 or highest < 0)), independent_exterior_registration_checked=False))
    if results:
        export(f"{STEM}_rapid_archive_weekly_height_fits", results)
    accepted = [r for r in results if r["central_both_height_fits_pass"]]
    return dict(candidate_pairs=len(results), distinct_endpoint_roof_fits=len(fits)//len(FWHM_M),
                both_height_fits_pass=len(accepted), weeks_with_both_height_fits_pass=len({r["week_ending"] for r in accepted}),
                stable_nonzero_change_sign=sum(r["stable_nonzero_change_sign"] for r in results),
                independently_registered_pairs=0)


def main():
    legacy = legacy_geometry()
    controls = rows(DERIVED / f"{STEM}_sentinel2_controls.csv")
    offsets_file = DERIVED / f"{STEM}_registration_masked_offsets.csv"
    if not offsets_file.is_file():
        offsets_file = DERIVED / f"{STEM}_registration_offsets.csv"
    offsets = {(r["earlier_version_id"], r["later_version_id"], r["candidate_id"]): r for r in rows(offsets_file)}
    roofs = {(r["product_id"], r["candidate_id"]): r for r in rows(DERIVED / f"{STEM}_roof_windows.csv")}
    work = defaultdict(dict)
    for r in controls:
        product, tank = r["landsat_product_id"], r["candidate_id"]
        work[product][tank] = (roofs[product, tank], offsets[product, r["sentinel_version_id"], tank])
    laser = list(laser_controls(legacy, roofs))
    for r in laser:
        product, tank = r["landsat_product_id"], r["candidate_id"]
        work[product].setdefault(tank, (roofs[product, tank], None))
    kernels = {width: response_kernel(width) for width in FWHM_M}
    jobs = [(product, group.values()) for product, group in sorted(work.items())]
    with ThreadPoolExecutor(max_workers=4) as pool:
        estimates = [r for result in pool.map(lambda job: scene(job, legacy, kernels), jobs) for r in result]
    export(f"{STEM}_height_fits", estimates)
    lookup = defaultdict(list)
    for r in estimates:
        lookup[r["product_id"], r["candidate_id"]].append(r)
    for r in laser:
        estimate = next(x for x in lookup[r["landsat_product_id"], r["candidate_id"]]
                        if x["basis"] == "native_location" and x["effective_fwhm_m"] == CENTRAL_FWHM)
        accepted = estimate["fit_status"] == RESOLVED and not estimate["center_search_on_boundary"]
        height = estimate["estimated_gap_m"]
        r.update(landsat_fit_status=estimate["fit_status"], landsat_center_search_on_boundary=estimate["center_search_on_boundary"],
                 landsat_gap_m=height, landsat_fit_passed=accepted, landsat_minus_laser_gap_m=height-r["laser_gap_m"] if accepted else None)
    if laser:
        export(f"{STEM}_height_laser_controls", laser)
    comparisons = []
    for r in controls:
        product, tank = r["landsat_product_id"], r["candidate_id"]
        for x in lookup[product, tank]:
            if x["fit_status"] != RESOLVED or x["center_search_on_boundary"]:
                continue
            baseline = float(r["sentinel_estimated_gap_m"])
            comparisons.append(dict(landsat_product_id=product, candidate_id=tank, sentinel_version_id=r["sentinel_version_id"],
                basis=x["basis"], effective_fwhm_m=x["effective_fwhm_m"], landsat_gap_m=x["estimated_gap_m"],
                sentinel_gap_m=baseline, landsat_minus_sentinel_gap_m=x["estimated_gap_m"]-baseline,
                absolute_separation_hours=float(r["absolute_separation_hours"])))
    export(f"{STEM}_height_comparisons", comparisons)
    central = [r for r in comparisons if r["effective_fwhm_m"] == CENTRAL_FWHM]
    main_stats = {basis: errors([r for r in central if r["basis"] == basis])
                  for basis in ("native_location", "relative_offset")}
    spreads = defaultdict(list)
    for r in comparisons:
        spreads[r["basis"], r["landsat_product_id"], r["candidate_id"]].append(r["landsat_gap_m"])
    spans = [max(g)-min(g) for g in spreads.values() if len(g) == len(FWHM_M)]
    report = dict(input_comparisons=len(controls), input_scenes=len({r["landsat_product_id"] for r in controls}),
        total_scenes_including_laser=len(work),
        registration_source=str(offsets_file.relative_to(ROOT)), fit_rows=len(estimates),
        central_fwhm_m=CENTRAL_FWHM, sensitivity_fwhm_m=FWHM_M, sensor_response_basis=PAPER,
        response_note="Isotropic Gaussian approximations informed by the early-mission along-scan result; no 2017–2021 L1TP PSF calibration.",
        central_fit_status=dict(Counter(f"{r['basis']}/{r['fit_status']}" for r in estimates if r["effective_fwhm_m"] == CENTRAL_FWHM)),
        accepted_comparisons=main_stats, all_widths_resolved_sensitivity_count=len(spans),
        median_all_widths_gap_range_m=median(spans) if spans else None,
        laser_opportunities_within_168h=len(laser), laser_opportunities_within_48h=sum(
            r["maximum_capture_separation_hours"] <= 48 for r in laser),
        laser_accepted_within_48h=describe([r["landsat_minus_laser_gap_m"] for r in laser
            if r["landsat_fit_passed"] and r["maximum_capture_separation_hours"] <= 48]),
        laser_accepted_within_168h=describe([r["landsat_minus_laser_gap_m"] for r in laser if r["landsat_fit_passed"]]),
        rapid_weekly_height_checks=rapid_controls(legacy, kernels))
    save_json(DERIVED / f"{STEM}_height_summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
