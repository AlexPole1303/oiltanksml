"""Diagnose original GEDI receive waveforms and compare them with the prepared optical controls."""

import json
from collections import defaultdict
from datetime import datetime
from statistics import median

import numpy as np
from scipy.signal import find_peaks

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import write

STEM = "cushing_gedi_waveform_controls"


def main():
    wave = defaultdict(list)
    for r in read("cushing_gedi_native_waveforms"):
        if r["channel"] == "rxwaveform":
            wave[r["shot_number"]].append((int(r["bin"]), float(r["elevation_wgs84_m"]), float(r["adc"])))
    ground = {(r["source_tile_id"], r["extract_row"], r["candidate_id"]): r for r in read("cushing_gedi_olm_roof_candidates")}
    native_ground = {(r["shot_number"],r["candidate_id"]):r for r in read("cushing_gedi_ground_controls")}
    rows = []
    for r in read("cushing_gedi_native_shots"):
        ident = r["exact_shot_number"]
        if not ident:
            continue
        a = np.array(sorted(wave[ident]))
        if len(a) != int(r["rx_sample_count"]) or not np.array_equal(a[:,0], np.arange(len(a))):
            raise ValueError(f"Incomplete receive waveform for {ident}")
        noise = float(r["noise_mean_corrected"])
        sigma = float(r["noise_stddev_corrected"])
        if not sigma > 0:
            raise ValueError(f"Invalid source noise scale for {ident}")
        peaks, _ = find_peaks(a[:,2], height=noise+5*sigma, prominence=3*sigma, distance=4)
        peak = max(peaks, key=lambda p: a[p,2], default=None)
        mode = float(r["elev_lowestmode"])
        peak_m = float(a[peak,1]) if peak is not None else None
        source = ground[(r["source_tile_id"],r["extract_row"],r["candidate_id"])]
        bias = float(source["local_ground_bias_m"]) if source["local_ground_bias_m"] else None
        current = native_ground.get((ident,r["candidate_id"]),{}).get("local_ground_bias_m")
        current = float(current) if current else None
        gap = float(r["olm_wall_ellipsoid_approx_m"]) - mode
        rows.append(dict(candidate_id=r["candidate_id"], shot_number=ident, source_tile_id=r["source_tile_id"],
            extract_row=r["extract_row"], native_capture_utc=r["native_capture_utc"], native_l2a_lowest_mode_m=mode,
            native_footprint_guard_pass=r["native_inside_original_10m_margin"]=="True", peaks_above_noise=len(peaks),
            peak_elevation_m=peak_m, peak_minus_mode_m=peak_m-mode if peak_m is not None else None,
            peak_snr=(float(a[peak,2])-noise)/sigma if peak is not None else None,
            native_wall_to_mode_m=gap, olm_local_ground_bias_m=bias,
            native_local_ground_bias_m=current,
            native_wall_to_mode_with_ground_bias_m=gap+current if current is not None else None))
    by_key = {(r["candidate_id"],r["source_tile_id"],r["extract_row"]):r for r in rows}
    joined = []
    for r in read("cushing_gedi_optical_fits"):
        key = (r["candidate_id"],r["source_tile_id"],r["gedi_extract_row"])
        native = by_key.get(key)
        if not native or r["fit_status"] != "resolved_in_geometry_diagnostic":
            continue
        ref = datetime.fromisoformat(native["native_capture_utc"])
        optical = datetime.fromisoformat(r["sentinel2_xml_capture_utc"])
        gap = float(r["estimated_gap_m"])
        corrected = native["native_wall_to_mode_with_ground_bias_m"]
        joined.append(dict(native, sentinel2_version_id=r["l1c_product_name"], sentinel2_captured_at_utc=optical.isoformat(),
            time_separation_hours=abs((ref-optical).total_seconds())/3600, sentinel2_wall_to_roof_m=gap,
            sentinel2_minus_uncorrected_native_gedi_m=gap-native["native_wall_to_mode_m"],
            sentinel2_minus_ground_corrected_native_gedi_m=gap-corrected if corrected is not None else None))
    export(f"{STEM}_shots", rows)
    export(f"{STEM}_optical", joined)
    unique = {}
    for r in sorted(joined,key=lambda r: (r["time_separation_hours"],r["sentinel2_version_id"])):
        if r["native_footprint_guard_pass"]:
            unique.setdefault((r["shot_number"],r["candidate_id"]),r)
    calibrated = [r for r in unique.values() if r["sentinel2_minus_ground_corrected_native_gedi_m"] is not None]
    summary = dict(waveform_shots=len(rows), single_peak_shots=sum(r["peaks_above_noise"]==1 for r in rows),
        maximum_absolute_peak_to_l2a_mode_m=max(abs(r["peak_minus_mode_m"]) for r in rows if r["peak_minus_mode_m"] is not None),
        optical_tile_comparisons=len(joined), distinct_optical_pairs_inside_margin=len(unique),
        distinct_optical_pairs_within_24h=sum(r["time_separation_hours"]<=24 for r in unique.values()),
        locally_ground_corrected_pairs=len(calibrated),
        median_absolute_ground_corrected_sensor_difference_m=median(abs(r["sentinel2_minus_ground_corrected_native_gedi_m"])
            for r in calibrated) if calibrated else None,
        median_absolute_difference_without_local_ground_correction_m=median(abs(r["sentinel2_minus_uncorrected_native_gedi_m"])
            for r in unique.values()) if unique else None,
        scope="Peak diagnostics use 5 noise standard deviations and a 3-sigma prominence, fixed before evaluating the optical differences. "
        "The Level 2A mode and waveform derive from the same GEDI signal. Optical differences include potential roof movement "
        "over the time between acquisitions. Ground-corrected rows use native same-beam terrain returns.")
    payload = json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
