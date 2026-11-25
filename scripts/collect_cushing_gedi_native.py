"""Resolve Cushing roof shots to original GEDI v003 metrics and waveform samples."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
from pyproj import Geod

from build_cushing_icesat2_controls import TO_UTM, identities
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, write
from sliderule_h5 import BASE, h5, version

STEM = "cushing_gedi_native"
PACKETS = RAW / STEM / "packets"
BEAMS = (0, 1, 2, 3, 5, 6, 8, 11)
GEOD = Geod(ellps="WGS84")
L2A = ("shot_number", "delta_time", "lat_lowestmode", "lon_lowestmode", "elev_lowestmode", "elev_highestreturn",
       "num_detectedmodes", "l2a_quality_flag_rel3", "sensitivity", "degrade_flag", "selected_algorithm")
L1B = ("shot_number", "geolocation/delta_time", "rx_sample_start_index", "rx_sample_count", "tx_sample_start_index",
       "tx_sample_count", "geolocation/elevation_bin0", "geolocation/elevation_lastbin", "geolocation/latitude_bin0",
       "geolocation/longitude_bin0", "geolocation/latitude_lastbin", "geolocation/longitude_lastbin",
       "noise_mean_corrected", "noise_stddev_corrected", "stale_return_flag")
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)
GPS_UTC_SECONDS = 18  # No intervening leap second in the study period.


def capture_time(epoch, delta):
    value = GPS_EPOCH + timedelta(seconds=float(epoch) + float(delta) - GPS_UTC_SECONDS)
    if not 2019 <= value.year <= 2026:
        raise ValueError("Unexpected GEDI shot year")
    return value


def collect(group, ground, tanks):
    first = group[0]
    prefix = f"/BEAM{BEAMS[int(first['beamname'])]:04b}/"
    l2 = first["l2a_granule"] + ".h5"
    l1 = first["l1b_granule"] + ".h5"
    receipts = []

    def retrieve(asset, resource, requests):
        values, receipt = h5(PACKETS, asset, resource, [(prefix+name, start, count, col) for name, start, count, col in requests])
        expected = {prefix + field[0] for field in requests}
        if set(values) != expected or receipt["issues"]:
            raise ValueError(f"Incomplete {resource}/{prefix}: {expected-set(values)} {receipt['issues']}")
        receipts.append(receipt)
        return {name.removeprefix(prefix): array for name, array in values.items()}

    orbit = {asset: retrieve(asset, resource, [("shot_number", 0, -1, 0)])["shot_number"]
             for asset, resource in (("gedil2a", l2), ("gedil1b", l1))}
    for key, ids in orbit.items():
        if ids.dtype != np.dtype("uint64") or not len(ids) or not np.all(ids[1:] > ids[:-1]):
            raise ValueError(f"Invalid shot index {key}/{prefix}")
    epoch = retrieve("gedil1b", l1, [("ancillary/master_time_epoch", 0, 1, 0)])["ancillary/master_time_epoch"][0]
    picks, samples = [], []
    for target in group:
        ids = orbit["gedil2a"]
        approximate = int(target["source_shotnumber_rounded"])
        pos = int(np.searchsorted(ids, np.uint64(approximate)))
        start = max(0, pos - 80)
        count = min(len(ids) - start, 161)
        near = retrieve("gedil2a", l2, [(name, start, count, 0) for name in L2A])
        if not np.array_equal(near["shot_number"], ids[start:start+count]):
            raise ValueError(f"GEDI index and original row mismatch for {l2}")
        tlat, tlon = float(target["latitude"]), float(target["longitude"])
        _, _, dist = GEOD.inv(np.full(count, tlon), np.full(count, tlat), near["lon_lowestmode"], near["lat_lowestmode"])
        olmtime = datetime.fromisoformat(target["gedi_approximate_capture_utc"])
        clocks = [capture_time(epoch, delta) for delta in near["delta_time"]]
        possible = [int(i) for i in np.flatnonzero(dist <= 15)
                    if abs((clocks[int(i)] - olmtime).total_seconds()) <= 20]
        record = {k: target[k] for k in ("candidate_id", "source_tile_id", "extract_row", "gedi_acquisition_key",
            "source_shotnumber_rounded", "gedi_approximate_capture_utc", "l1b_concept_id", "l2a_concept_id",
            "l1b_product_created_utc", "l2a_product_created_utc", "l1b_cmr_revision_utc", "l2a_cmr_revision_utc")}
        record.update(beam=prefix.strip("/"), l2a_resource=l2, l1b_resource=l1, position_matches=len(possible),
                      nearest_to_olm_m=float(np.min(dist)), exact_shot_number=None)
        if len(possible) != 1:
            picks.append(record)
            continue
        i, = possible
        shot = near["shot_number"][i]
        ix = int(np.searchsorted(orbit["gedil1b"], shot))
        if ix >= len(orbit["gedil1b"]) or orbit["gedil1b"][ix] != shot:
            raise ValueError(f"Shot {shot} is absent from {l1}")
        info = {k: v[0].item() for k, v in retrieve("gedil1b", l1, [(name, ix, 1, 0) for name in L1B]).items()}
        if info["shot_number"] != shot or abs(info["geolocation/delta_time"] - near["delta_time"][i]) > .001:
            raise ValueError(f"GEDI cross-level identity mismatch for {shot}")
        native = {k: v[i].item() for k, v in near.items()}
        config = [(f"{channel}waveform", info[f"{channel}_sample_start_index"] - 1, info[f"{channel}_sample_count"], 0)
                  for channel in ("rx", "tx")]
        if any(not (start >= 0 and 0 < count <= 4096) for _, start, count, _ in config):
            raise ValueError(f"Invalid original waveform slice for {shot}")
        wave = retrieve("gedil1b", l1, config)
        source = ground[(target["source_tile_id"], target["extract_row"], target["candidate_id"])]
        record.update(exact_shot_number=int(shot), native_capture_utc=clocks[i].isoformat(), source_position_delta_m=float(dist[i]),
                      original_l2a_row=start+i, original_l1b_row=ix,
                      l2a_lowest_mode_delta_m=native["elev_lowestmode"]-float(source["elev_lowestmode"])/100,
                      **{k: v for k, v in native.items() if k != "shot_number"},
                      **{k.replace("/", "_"): v for k, v in info.items() if k != "shot_number"})
        t = tanks[target["candidate_id"]]
        x, y = TO_UTM.transform(native["lon_lowestmode"], native["lat_lowestmode"])
        edge = t["radius_m"] - float(np.hypot(x - t["east_m"], y - t["north_m"]))
        record["native_centre_to_roof_edge_m"] = edge
        record["native_inside_original_10m_margin"] = edge >= 22.5
        record["olm_wall_ellipsoid_approx_m"] = float(source["lidar_wall_ellipsoid_approx_m"])
        picks.append(record)
        for name, _, count, _ in config:
            values = wave[name]
            if len(values) != count or not np.isfinite(values).all():
                raise ValueError(f"Nonfinite or incomplete waveform for {shot}/{name}")
            heights = np.linspace(info["geolocation/elevation_bin0"], info["geolocation/elevation_lastbin"], count) if name=="rxwaveform" else None
            samples.extend(dict(gedi_acquisition_key=target["gedi_acquisition_key"], shot_number=int(shot),
                candidate_id=target["candidate_id"], channel=name, bin=i, adc=float(value),
                elevation_wgs84_m=float(heights[i]) if heights is not None else None)
                for i, value in enumerate(values))
    print(f"GEDI {first['gedi_acquisition_key']} {prefix}: {sum(r['exact_shot_number'] is not None for r in picks)}/{len(group)} matched", flush=True)
    return picks, samples, receipts


def main():
    server = version(PACKETS)
    targets = read("cushing_gedi_olm_v003_original_candidates")
    ground = {(r["source_tile_id"], r["extract_row"], r["candidate_id"]): r for r in read("cushing_gedi_olm_roof_candidates")}
    tanks, _, _ = identities()
    groups = defaultdict(list)
    for target in targets:
        groups[target["gedi_acquisition_key"], target["beamname"]].append(target)
    picks, samples, receipts = [], [], {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for observations, waveforms, metadata in pool.map(lambda rows: collect(rows, ground, tanks), groups.values()):
            picks.extend(observations)
            samples.extend(waveforms)
            receipts.update((r["request_sha256"], r) for r in metadata)
    # Keep a consistent schema when an older footprint cannot be matched uniquely.
    fields = tuple(dict.fromkeys(k for r in picks for k in r))
    picks = [{k: r.get(k) for k in fields} for r in picks]
    export(f"{STEM}_shots", picks)
    export(f"{STEM}_waveforms", samples)
    export(f"{STEM}_packets", [dict(r, issues=json.dumps(r["issues"])) for r in receipts.values()])
    matched = [r for r in picks if r["exact_shot_number"]]
    summary = dict(sliderule_version=server, endpoint=f"{BASE}/h5p", target_rows=len(targets), native_matched_shots=len(matched),
        unique_native_shots=len({r["exact_shot_number"] for r in matched}), distinct_roofs=len({r["candidate_id"] for r in matched}),
        targeted_acquisitions=len({r["gedi_acquisition_key"] for r in picks}), native_margin_passes=sum(r["native_inside_original_10m_margin"] for r in matched),
        waveform_samples_by_channel=Counter(r["channel"] for r in samples), verified_packets=len(receipts),
        native_release3_quality_passes=sum(r["l2a_quality_flag_rel3"] == 1 for r in matched),
        scope="Original GEDI v003 HDF5 fields and slices via SlideRule. Matching uses position, beam, time and uint64 source IDs. "
        "RX sample elevations are interpolated between official waveform endpoints. These samples do not certify roof heights without waveform and ground-context analysis.")
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
