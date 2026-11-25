"""Build Landsat 8 real-time roof heights and dated candidate pairs."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache

import numpy as np
from affine import Affine
from pyproj import Transformer

from audit_cushing_landsat8_rapid_archive import DERIVED, ROOT, STEM, STORE, export, legacy_geometry, parse_mtl, rows, save_json
from build_cross_sensor_shadow import PAN_PIXEL, PAN_STEP, PSF_DOC, along_track_bearing, pan_psf, published_lsf
from build_cushing_landsat7_mixed_inputs import Endpoint, calendar, eligible, index, sentinel_endpoints
from build_sentinel2_angles import ground_coordinates
from build_terminal_development_measurements import BARREL_M3
from fit_shadow_geometry import bearing_vector, fit_response, gap_fit_status, gap_trials
from measure_cushing_landsat7 import RESOLVED, native_chip

CRS = "EPSG:32614"
TO_GEOGRAPHIC = Transformer.from_crs(CRS, 4326, always_xy=True)
CENTRE_SHIFTS = (-9, -6, -3, 0, 3, 6, 9)


def verified(path, expected):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")
    return payload


@lru_cache(maxsize=8)
def native_raster(product, store=STORE):
    path = store / product / "native.json"
    item = json.loads(path.read_bytes())
    if item["product_id"] != product:
        raise ValueError(f"Invalid Landsat 8 receipt: {path}")
    packet = ROOT / item["native_packet"]
    with packet.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != item["native_sha256"]:
            raise ValueError(f"Landsat 8 pixel packet checksum mismatch: {product}")
        stream.seek(0)
        with np.load(stream) as data:
            if str(data["crs"]) != CRS:
                raise ValueError(f"Wrong Landsat 8 projection: {product}")
            raster = data["pan_dn"], data["pan_usable"], Affine(*data["pan_transform"])
    angles = ROOT / item["angle_file"]
    verified(angles, item["angle_sha256"])
    mtl = parse_mtl(path.with_name("MTL.txt").read_bytes())
    if mtl["LANDSAT_PRODUCT_ID"] != product:
        raise ValueError(f"Wrong Landsat 8 radiometry metadata: {product}")
    gain, bias = (float(mtl[f"REFLECTANCE_{key}_BAND_8"]) for key in ("MULT", "ADD"))
    return item, raster, angles, gain, bias


def roof_inputs(raster, roof, geometry, gain, bias):
    east, north, radius = (float(roof[k]) for k in ("center_east_m", "center_north_m", "radius_m"))
    expected = (float(geometry["center_east_m"]), float(geometry["center_north_m"]), float(geometry["diameter_m"])/2)
    if not np.allclose((east, north, radius), expected, atol=1e-8, rtol=0):
        raise ValueError(f"Landsat 8 uses different roof geometry: {roof['candidate_id']}")
    shell = float(geometry["prior_rim_z_p95_native"])-float(geometry["prior_ground_z_p50_native"])
    raw, _, good = native_chip(raster, roof, (east, north))
    zenith = float(roof["solar_zenith_deg"])
    signal = (raw[-2]*gain+bias)/math.cos(math.radians(zenith))
    true_north = ground_coordinates(CRS, (TO_GEOGRAPHIC.transform(east, north),))[2][0]
    sun = bearing_vector(zenith, float(roof["solar_azimuth_deg"])+true_north)
    view = bearing_vector(float(roof["view_zenith_deg"]), float(roof["view_azimuth_deg"])+true_north)
    return ((*raw[:-2], signal, raw[-1]), radius, shell, sun, view), good


def fit_roof(inputs, kernel):
    grid, radius, shell, sun, view = inputs
    trial = gap_trials(shell, "EFR")
    gap, lower, upper, dx, dy, _, rmse, flat, placebo = fit_response(grid, radius, shell, trial, sun, view, kernel,
        reversed_sun=True, pixel_m=PAN_PIXEL, step_m=PAN_STEP, shifts=CENTRE_SHIFTS, subcell_samples=3)
    return dict(fitted_pixels=len(grid[-1]), fit_status=gap_fit_status(trial,gap,lower,upper,rmse,flat,placebo),
        center_at_boundary=max(abs(dx),abs(dy))==max(CENTRE_SHIFTS), gap_m=float(gap), gap_min_m=lower, gap_max_m=upper,
        fitted_shift_east_m=dx, fitted_shift_north_m=dy, rmse=rmse, flat_rmse=flat, reversed_sun_rmse=placebo)


def measure_scene(scene, geometry, lsf, *, store=STORE):
    product = scene["product_id"]
    item, raster, angles, gain, bias = native_raster(product, store)
    if item["native_packet"] != scene["native_packet"] or item["native_sha256"] != scene["native_sha256"]:
        raise ValueError(f"Scene manifest and receipt disagree: {product}")
    kernel = pan_psf(lsf, along_track_bearing(angles, CRS))
    result = []
    for r in item["roofs"]:
        tank = r["candidate_id"]
        base = dict(product_id=product, candidate_id=tank, captured_at_utc=scene["captured_at_utc"],
            available_at_utc=scene["archive_bundle_available_utc"], week_ending=scene["week_ending"],
            diameter_m=2*r["radius_m"], native_packet=item["native_packet"], native_sha256=item["native_sha256"],
            angle_file=item["angle_file"], angle_status=r["angle_status"], roof_pixels_passing_qa=r["roof_pixels_passing_qa"],
            fitted_pixels=None, fit_status="no_clear_roof_pixels_or_angles", center_at_boundary=None,
            gap_m=None, gap_min_m=None, gap_max_m=None, fitted_shift_east_m=None, fitted_shift_north_m=None,
            rmse=None, flat_rmse=None, reversed_sun_rmse=None)
        if r["roof_pixels_passing_qa"] <= 0 or r["angle_status"] != "resolved":
            result.append(base)
            continue
        inputs, good = roof_inputs(raster, r, geometry[tank], gain, bias)
        count = len(inputs[0][-1])
        if count < 12 or good < 4:
            result.append(dict(base, fitted_pixels=count, fit_status="insufficient_native_b8_pixels"))
            continue
        result.append(base | fit_roof(inputs,kernel))
    return result


def endpoint_index(fits):
    groups = defaultdict(list)
    for r in fits:
        if r["fit_status"] == RESOLVED and not r["center_at_boundary"]:
            groups[r["candidate_id"]].append(Endpoint(datetime.fromisoformat(r["captured_at_utc"]),
                datetime.fromisoformat(r["available_at_utc"]), r))
    return index(groups)


def endpoint_fields(obs, sensor):
    r = obs.record
    meta = dict(sensor=sensor, captured_at_utc=obs.captured.isoformat(), available_at_utc=obs.available.isoformat(),
                version_id=r["product_id"] if sensor=="B8" else r["version_id"],
                source=r["native_packet"] if sensor=="B8" else r["tile_metadata_path"],
                fitted_pixels=r["fitted_pixels"], gap_m=r["gap_m"] if sensor=="B8" else float(r["estimated_gap_m"]))
    return meta | dict(canonical_id=r.get("canonical_l2a_id"), observation_id=r.get("observation_id"),
                       observation_source=r.get("measurement_source"))


def candidate_pairs(pan, sentinel, registry, reports):
    for report in reports:
        first, last, cutoff = (datetime.fromisoformat(report[k]) for k in ("previous_measurement_at_utc", "current_measurement_at_utc", "cutoff_utc"))
        for tank, group in pan.items():
            g = registry[tank]
            known, exists = max(g["roof_evidence_date"],g["lidar_source_available_date"]), g["first_seen"]
            if known >= cutoff.date().isoformat() or exists > first.date().isoformat():
                continue
            b8_first, b8_last = (eligible(group,t,cutoff,exists) for t in (first,last))
            s2_first, s2_last = (eligible(sentinel.get(tank,([],[])),t,cutoff,exists) for t in (first,last))
            for a, b, s1, s2 in ((b8_first,b8_last,"B8","B8"), (b8_first,s2_last,"B8","B04"), (s2_first,b8_last,"B04","B8")):
                for before in a:
                    for after in b:
                        if not first < after.captured <= last:
                            continue
                        out = dict(week_ending=report["week_ending"], release_date=report["release_date"], cutoff_utc=cutoff.isoformat(),
                            previous_boundary_utc=first.isoformat(), current_boundary_utc=last.isoformat(), candidate_id=tank,
                            terminal_id=g["terminal_id"], route=f"{s1}_to_{s2}", area_m2=float(g["circular_area_m2"]),
                            existing_b04_tanks=int(report["baseline_b04_tanks"]))
                        for label, obs, sensor in (("previous",before,s1),("current",after,s2)):
                            out.update({f"{label}_{k}":v for k,v in endpoint_fields(obs,sensor).items()})
                        out["unregistered_capture_change_bbl"] = (out["previous_gap_m"]-out["current_gap_m"])*out["area_m2"]/BARREL_M3
                        yield out


def main():
    geometry = legacy_geometry()
    registry = {r["candidate_id"]:r for r in rows(DERIVED / "sentinel2_space_time_coverage_roof_registry.csv")}
    scenes = rows(DERIVED / f"{STEM}_native_scenes.csv")
    lsf = published_lsf()
    with ThreadPoolExecutor(max_workers=5) as pool:
        fits = [row for part in pool.map(lambda r:measure_scene(r,geometry,lsf), scenes) for row in part]
    export(f"{STEM}_height_fits",fits)
    pairs = list(candidate_pairs(endpoint_index(fits),sentinel_endpoints(geometry),registry,calendar()))
    if pairs:
        export(f"{STEM}_height_pair_candidates",pairs)
    usable = [r for r in fits if r["fit_status"]==RESOLVED and not r["center_at_boundary"]]
    counts = Counter(r["route"] for r in pairs)
    report = dict(scenes=len(scenes), roof_windows=len(fits), fit_status=dict(Counter(r["fit_status"] for r in fits)),
        roofs_with_passing_height=len(usable), weeks_with_any_passing_height=len({r["week_ending"] for r in usable}),
        passing_height_pairs=len(pairs), pairs_by_route=dict(counts),
        unique_roof_weeks_with_pairs=len({(r["candidate_id"],r["week_ending"]) for r in pairs}),
        weeks_with_pairs=len({r["week_ending"] for r in pairs}),
        weeks_with_pairs_without_b04=len({r["week_ending"] for r in pairs if not r["existing_b04_tanks"]}),
        response_model=PSF_DOC, scope="Collection 1 RT, 2017–2021 development. A pair has two individually resolved heights, not a registered or validated weekly change.")
    save_json(DERIVED / f"{STEM}_height_summary.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__ == "__main__":
    main()
