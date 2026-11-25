"""Archive L1B waveform windows for geometrically contained native L2A roof shots."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.signal import find_peaks

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_cushing_gedi_native import L1B, PACKETS
from collect_public import RAW, write
from sliderule_h5 import h5

STEM = "cushing_gedi_roof_waveforms"
STORE = RAW / STEM


def collect(group):
    resource = group[0]["l1b_granule"] + ".h5"
    beam = int(group[0]["beam"])
    prefix = f"/BEAM{beam:04b}/"
    packets = []

    def source(spec, directory=STORE, cache=True):
        arrays, receipt = h5(directory, "gedil1b", resource, [(prefix+field, a, b, 0) for field,a,b in spec], cache=cache)
        if set(arrays) != {prefix+field for field,_,_ in spec} or receipt["issues"]:
            raise ValueError(f"Incomplete GEDI original {resource}/{beam}: {receipt['issues']}")
        packets.append(receipt)
        return {k.removeprefix(prefix): v for k,v in arrays.items()}

    ids = source([("shot_number",0,-1)], directory=PACKETS, cache=False)
    index = ids["shot_number"]
    if index.dtype != np.dtype("uint64") or not np.all(index[1:] > index[:-1]):
        raise ValueError(f"Invalid GEDI shot order for {resource}/{beam}")
    rows, waves = [], []
    for r in group:
        shot = int(r["shot_number"])
        i = int(np.searchsorted(index, np.uint64(shot)))
        if i >= len(index) or int(index[i]) != shot:
            raise ValueError(f"GEDI L2A shot {shot} absent from L1B")
        info = {k:v[0].item() for k,v in source([(field,i,1) for field in L1B]).items()}
        if info["shot_number"] != shot or abs(info["geolocation/delta_time"]-float(r["delta_time"])) > .001:
            raise ValueError(f"L1B and L2A identities disagree for {shot}")
        setup = [(f"{k}waveform", info[f"{k}_sample_start_index"]-1, info[f"{k}_sample_count"]) for k in ("rx","tx")]
        if any(i < 0 or not 1 <= n <= 4096 for _,i,n in setup):
            raise ValueError(f"Out-of-bounds GEDI waveform indices for {shot}")
        response = source(setup)
        rx = response["rxwaveform"]
        heights = np.linspace(info["geolocation/elevation_bin0"], info["geolocation/elevation_lastbin"], len(rx))
        mean, sigma = info["noise_mean_corrected"], info["noise_stddev_corrected"]
        if not sigma > 0 or any(not np.isfinite(x).all() for x in response.values()):
            raise ValueError(f"Invalid GEDI noise or waveform for {shot}")
        peaks, _ = find_peaks(rx, height=mean+5*sigma, prominence=3*sigma, distance=4)
        peak = max(peaks, key=lambda j: rx[j], default=None)
        peak_z = float(heights[peak]) if peak is not None else None
        rows.append(dict(r, l1b_native_row=i, detected_waveform_peaks=len(peaks), highest_peak_elevation_m=peak_z,
            highest_peak_minus_l2a_mode_m=peak_z-float(r["elev_lowestmode"]) if peak_z is not None else None,
            **{k.replace("/","_"):v for k,v in info.items() if k != "shot_number"}))
        for name, waveform in response.items():
            waves.extend(dict(candidate_id=r["candidate_id"], shot_number=shot, observed_at_utc=r["observed_at_utc"],
                channel=name, bin=j, adc=float(adc), elevation_m=float(heights[j]) if name=="rxwaveform" else None)
                for j,adc in enumerate(waveform))
    return rows,waves,packets


def main():
    candidates = [r for r in read("cushing_gedi_coverage_shots") if r["footprint_inside_with_10m_margin"]=="True"
                  and r["geometrically_plausible_lowest_mode"]=="True"]
    groups = defaultdict(list)
    for row in candidates:
        groups[row["l1b_granule"],row["beam"]].append(row)
    rows, waves, packets, failures = [], [], {}, []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(collect, values): key for key,values in groups.items()}
        for i, future in enumerate(as_completed(jobs),1):
            try:
                new, samples, receipts = future.result()
            except (OSError, ValueError) as error:
                failures.append(dict(resource=jobs[future][0], beam=jobs[future][1], issue=str(error)))
            else:
                rows.extend(new)
                waves.extend(samples)
                packets.update((r["request_sha256"],r) for r in receipts)
            if i%10==0:
                print(f"GEDI waveform groups {i}/{len(groups)}, {len(rows)} shots", flush=True)
    if not rows:
        raise ValueError(f"No GEDI waveforms recovered: {failures[:3]}")
    export(f"{STEM}_shots", sorted(rows,key=lambda r:(r["observed_at_utc"],r["shot_number"])))
    export(f"{STEM}_samples", sorted(waves,key=lambda r:(r["observed_at_utc"],r["shot_number"],r["channel"],r["bin"])))
    export(f"{STEM}_packets", [dict(r, issues=json.dumps(r["issues"])) for r in packets.values()])
    summary = dict(candidate_shots=len(candidates), complete_shot_waveforms=len(rows),
        waveform_shots_by_year=Counter(r["observed_at_utc"][:4] for r in rows),
        distinct_roofs=len({r["candidate_id"] for r in rows}), unique_sources=len({r["l1b_granule"] for r in rows}),
        waveform_samples=Counter(r["channel"] for r in waves), single_peak_shots=sum(r["detected_waveform_peaks"]==1 for r in rows),
        packets=len(packets), unresolved_groups=failures,
        scope="Geometrically contained shots with a plausible L2A return. Waveforms retain original sample values and index links. "
        "They are prospective height controls and require local vertical reference and time-matched imagery for error estimation.")
    payload = json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
