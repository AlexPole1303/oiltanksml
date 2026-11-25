"""Screen native SLC power for a height-dependent roof return, without EIA labels."""

import gzip
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np
from pyproj import Transformer
from scipy.ndimage import map_coordinates

from audit_shadow_precision import describe
from build_ceda_s1_slc_windows import NativeGeometry, UTM_TO_ECEF
from build_cushing_icesat2_controls import identities
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from calibrate_ceda_s1_slc_windows import STEM as CALIBRATION
from collect_public import ROOT, write

STEM = "ceda_s1_slc_roof_profiles"
ECEF_TO_UTM = Transformer.from_crs(4978, 32614, always_xy=True).transform
SECTORS = 8
ANGLE = np.deg2rad(np.linspace(-64, 64, SECTORS*8, endpoint=False)+1)
SHIFTS = np.array([[0, 0], [0, -1], [0, 1], [-.5, 0], [.5, 0]])
FLANK_M = 6.0
HEIGHT_STEP_M = .5
PASS = "passes_arc_diagnostic"


def geometry(asset, burst_id):
    path = asset["local_annotation"]
    root = ET.fromstring(gzip.decompress((ROOT / path).read_bytes()))
    burst, = [b for b in root.findall("swathTiming/burstList/burst") if b.findtext("burstId") == burst_id]
    info = root.find("imageAnnotation/imageInformation")
    native = NativeGeometry(dict(cdse_uuid=asset["cdse_uuid"], local_annotation_gzip=path,
                                 valid_mask_gzip=asset["local_valid_mask"], slc_burst_azimuth_time_utc=burst.findtext("azimuthTime"),
                                 native_azimuth_interval_seconds=info.findtext("azimuthTimeInterval"),
                                 native_range_pixel_spacing_m=info.findtext("rangePixelSpacing"),
                                 native_slant_range_first_sample_seconds=info.findtext("slantRangeTime")))
    if max(native.grid_errors) > .05:
        raise ValueError(f"SLC grid check failed for {asset['cdse_uuid']}")
    return native


def project(native, xyz, center_time):
    t = np.full(xyz.shape[:-1], center_time)
    acceleration = native.velocity.derivative()
    for _ in range(3):
        pos, vel = native.position(t), native.velocity(t)
        delta = pos-xyz
        t -= np.sum(delta*vel, axis=-1)/np.sum(vel**2+delta*acceleration(t), axis=-1)
    los = native.position(t)-xyz
    velocity = native.velocity(t)
    along_track_m = np.sum(los*velocity, axis=-1)/np.linalg.norm(velocity, axis=-1)
    if not np.isfinite(along_track_m).all() or np.max(np.abs(along_track_m)) > .005:
        raise ValueError("SLC arc zero-Doppler solve did not converge")
    return np.stack(((t-native.start)/native.dt, (np.linalg.norm(los, axis=-1)-native.first_range)/native.dr), axis=0)


def arc_points(native, row, tank):
    ground, rim = (tank[k] for k in ("ground_hae_approx_m", "top_hae_approx_m"))
    gaps = np.arange(.5, rim-ground, HEIGHT_STEP_M)
    east, north, radius = (tank[k] for k in ("east_m", "north_m", "radius_m"))
    center = np.array(UTM_TO_ECEF(east, north, rim))
    expected = native.start+float(row["rim_center_burst_line"])*native.dt
    when, sample = native.locate(center, expected)
    if abs((when-native.start)/native.dt-float(row["rim_center_burst_line"])) > .01 or abs(sample-float(row["rim_center_sample"])) > .01:
        raise ValueError(f"SLC window geometry changed for {row['candidate_id']}")
    towards = ECEF_TO_UTM(*(center + .0001*(native.position(when)-center)))
    bearing = np.arctan2(towards[0]-east, towards[1]-north)
    azimuth = bearing + ANGLE[None, :] + np.array([[0], [np.pi]])
    x, y = east + radius*np.sin(azimuth), north + radius*np.cos(azimuth)
    e, n, h = np.broadcast_arrays(x[None], y[None], rim-gaps[:, None, None])
    xyz = np.stack(UTM_TO_ECEF(e.ravel(), n.ravel(), h.ravel()), axis=-1).reshape(*e.shape, 3)
    points = project(native, xyz, when)
    points -= np.array([float(row["first_line"]), float(row["first_sample"])]).reshape(2, 1, 1, 1)
    return gaps, points, native.origin+timedelta(seconds=when)


def power_profiles(power, noise, points, dr):
    flank_shifts = np.array([0, -FLANK_M/dr, FLANK_M/dr])
    samples = points[:, None, None] + SHIFTS.T[:, :, None, None, None, None]
    samples = np.broadcast_to(samples, (2, len(SHIFTS), 3, *points.shape[1:])).copy()
    samples[1] += flank_shifts[None, :, None, None, None]
    p = map_coordinates(power, samples, order=1, mode="constant", cval=np.nan, prefilter=False)
    n = map_coordinates(noise, samples, order=1, mode="constant", cval=np.nan, prefilter=False)
    if not (np.isfinite(p).all() and np.isfinite(n).all() and (n > 0).all()):
        raise ValueError("SLC profile left valid native context")
    # Compare denoised power in dB with a per-pixel floor from the noise LUT.
    db = 10*np.log10(np.maximum(p-n, n))
    contrast = db[:, 0] - (db[:, 1]+db[:, 2])/2
    return np.median(contrast.reshape(len(SHIFTS), *points.shape[1:-1], SECTORS, -1), axis=-1)


def fit(gaps, profiles):
    # Profiles use azimuth sectors, with a near-side arc as a negative control for the roof return.
    score = np.median(profiles[:, :, 1], axis=-1)
    placebo = np.median(profiles[0, :, 0], axis=-1)
    best = int(np.argmax(score[0]))
    gap = gaps[best]
    score_best = score[0, best]
    rivals = np.abs(gaps-gap) >= 3.5
    margin = score_best-score[0, rivals].max() if rivals.any() else None
    shift_span = np.ptp(gaps[score.argmax(axis=1)])
    sector_span = np.ptp([gaps[np.argmax(np.median(profiles[0, :, 1, i::2], axis=-1))] for i in range(2)])
    if best in (0, len(gaps)-1):
        status = "best_height_at_boundary"
    elif score_best < 1.25:
        status = "weak_arc"
    elif score_best <= placebo.max():
        status = "near_arc_placebo_as_strong"
    elif margin is None:
        status = "insufficient_height_range_for_peak_test"
    elif margin < .60:
        status = "competing_roof_height"
    elif max(shift_span, sector_span) > 2:
        status = "height_unstable_under_shift_or_sector_split"
    else:
        status = PASS
    return dict(gap_m=gap, far_arc_peak_contrast_db=score_best, near_arc_peak_contrast_db=placebo.max(),
                peak_margin_db=margin, gap_span_under_grid_shift_m=shift_span, gap_span_between_sector_halves_m=sector_span,
                status=status)


def asset_profiles(asset, rows, tanks):
    data = (ROOT / asset["calibrated_packet"]).read_bytes()
    if len(data) != int(asset["calibrated_packet_bytes"]) or hashlib.sha256(data).hexdigest() != asset["calibrated_packet_sha256"]:
        raise ValueError(f"Changed calibrated SLC packet for {asset['cdse_uuid']}")
    with np.load(io.BytesIO(data), allow_pickle=False) as source:
        if str(source["cdse_uuid"]) != asset["cdse_uuid"]:
            raise ValueError("Wrong SLC calibrated packet identity")
        power, noise = source["detected_sigma0"], source["noise_sigma0"]
    burst_id, = {r["relative_burst_id"] for r in rows}
    native = geometry(asset, burst_id)
    for row in rows:
        tank = tanks[row["candidate_id"]]
        gaps, points, when = arc_points(native, row, tank)
        a = int(row["offset_complex_samples"])
        b = a+int(row["complex_samples"])
        shape = int(row["height"]), int(row["width"])
        profiles = power_profiles(power[a:b].reshape(shape), noise[a:b].reshape(shape), points, native.dr)
        yield dict(product_id=asset["product_id"], candidate_id=row["candidate_id"], terminal_id=tank["terminal_id"],
                   cdse_uuid=asset["cdse_uuid"], polarization=asset["polarization"], relative_burst_id=burst_id,
                   acquisition_start_utc=asset["capture_start_utc"], rim_zero_doppler_utc=when.isoformat(),
                   native_slant_pixel_spacing_m=native.dr, radius_m=tank["radius_m"],
                   lidar_wall_height_m=tank["top_hae_approx_m"]-tank["ground_hae_approx_m"],
                   calibrated_packet_sha256=asset["calibrated_packet_sha256"], **fit(gaps, profiles)), gaps, profiles


def main():
    tanks, _, _ = identities()
    windows = defaultdict(list)
    for r in read(f"{CALIBRATION}_windows"):
        windows[r["cdse_uuid"]].append(r)
    rows, grids = [], []
    offset = 0
    for asset in read(f"{CALIBRATION}_assets"):
        for row, gaps, profiles in asset_profiles(asset, windows[asset["cdse_uuid"]], tanks):
            rows.append(dict(row, profile_offset=offset, profile_height_steps=len(gaps)))
            columns = np.moveaxis(profiles, 1, 0).reshape(len(gaps), -1)
            grids.append(np.column_stack((gaps, columns)).astype(np.float32))
            offset += len(gaps)
    export(STEM, rows)
    arrays = np.concatenate(grids)
    payload = io.BytesIO()
    np.savez_compressed(payload, gaps_m=arrays[:, 0],
                        contrast_db=arrays[:, 1:].reshape(len(arrays), len(SHIFTS), 2, SECTORS), shifts_row_sample=SHIFTS,
                        sides=np.array(["near", "far"]))
    write(DERIVED / f"{STEM}.npz", payload.getvalue())
    index = {(r["cdse_uuid"], r["candidate_id"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Repeated CEDA SLC profile identity")
    comparisons = []
    for control in read("ceda_s1_concurrent_optical_controls"):
        for pol in ("VV", "VH"):
            row = index.get((control[f"radar_{pol.lower()}_burst_id"], control["candidate_id"]))
            if row is None:
                continue
            optical = float(control["optical_gap_to_rim_m"])
            comparisons.append(dict(row, optical_gap_m=optical, radar_minus_optical_m=row["gap_m"]-optical,
                                    optical_profile_low_m=control["low_over_all_optical_fits_m"],
                                    optical_profile_high_m=control["high_over_all_optical_fits_m"],
                                    maximum_separation_hours=control["maximum_separation_hours"],
                                    optical_versions_json=control["optical_versions_json"]))
    export(f"{STEM}_optical_controls", comparisons)
    aerial = []
    for control in read("ceda_s1_slc_reference_pairs"):
        if control["within_72_hours"] != "True":
            continue
        for pol in ("VV", "VH"):
            row = index[control[f"radar_{pol.lower()}_cdse_uuid"], control["candidate_id"]]
            gap = float(control["reference_gap_to_rim_m"])
            aerial.append(dict(row, aerial_source=control["reference_asset"], aerial_table=control["reference_table"],
                               aerial_start_utc=control["reference_start_utc"], aerial_end_utc=control["reference_end_exclusive_utc"],
                               aerial_clock_status=control["reference_exterior_clock_status"],
                               maximum_separation_hours=control["maximum_capture_separation_hours"],
                               aerial_gap_m=gap, radar_minus_aerial_m=row["gap_m"]-gap))
    export(f"{STEM}_aerial_controls", aerial)
    weekly = []
    for pair in read("ceda_s1_slc_regional_weekly_pairs"):
        for pol in ("VV", "VH"):
            first, last = (index[pair[f"{side}_{pol.lower()}"], pair["candidate_id"]] for side in ("previous", "current"))
            both_pass = first["status"]==last["status"]==PASS
            weekly.append(dict(pair, polarization=pol, previous_status=first["status"], current_status=last["status"],
                               both_pass=both_pass, candidate_rise_m=first["gap_m"]-last["gap_m"] if both_pass else None))
    export(f"{STEM}_weekly", weekly)
    summary = dict(profile_count=len(rows), grid_height_rows=len(arrays),
                   status_by_polarization={p: Counter(r["status"] for r in rows if r["polarization"]==p) for p in ("VV", "VH")},
                   optical_comparisons=len(comparisons),
                   optical_controls_pass_by_polarization={p: describe([r["radar_minus_optical_m"] for r in comparisons
                                                                     if r["polarization"]==p and r["status"]==PASS])
                                                        for p in ("VV", "VH")},
                   aerial_comparisons_within_72h=len(aerial),
                   aerial_controls_pass_by_polarization={p: describe([r["radar_minus_aerial_m"] for r in aerial
                                                                     if r["polarization"]==p and r["status"]==PASS])
                                                        for p in ("VV", "VH")},
                   weekly_pairs=len(weekly), weekly_pairs_both_pass=sum(r["both_pass"] for r in weekly),
                   passes_both_on_b04_gap_week=sum(r["both_pass"] and r["b04_status"]=="no_clear_pair_with_dated_geometry" for r in weekly),
                   scope="Engineering diagnostics with fixed profile settings. A pass requires optical or independent height validation "
                         "before use as a training label. The optical control uses one capture pair.")
    report = json.dumps(summary, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", report.encode())
    print(report, end="")


if __name__ == "__main__":
    main()
