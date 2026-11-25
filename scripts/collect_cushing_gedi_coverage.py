"""Search eight native GEDI v003 beams for individual shots in the Cushing study area."""

import argparse
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from build_cushing_icesat2_controls import TO_UTM, identities
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_cushing_gedi_native import BEAMS, capture_time
from collect_public import RAW, ROOT, write
from sliderule_h5 import BASE, h5, version

STEM = "cushing_gedi_coverage"
STORE = RAW / STEM
FIELDS = ("shot_number", "delta_time", "lat_lowestmode", "lon_lowestmode", "elev_lowestmode", "l2a_quality_flag_rel3")
GEDI_GPS_EPOCH = 1198800018.0


def beam_group(file, beams):
    paths = [f"/BEAM{beam:04b}/{field}" for beam in beams for field in FIELDS]
    arrays, receipt = h5(STORE / "packets", "gedil2a", file, [(path,0,-1,0) for path in paths], cache=False)
    if set(paths) <= arrays.keys():
        yield beams, arrays, receipt
    elif len(beams) > 1:
        for beam in beams:
            yield from beam_group(file, (beam,))
    elif not arrays and not receipt["issues"]:
        yield beams, {}, receipt
    else:
        raise ValueError(f"GEDI missing native arrays {sorted(set(paths)-arrays.keys())} for {file}: {receipt['issues']}")


def scan(record, bounds):
    file = record["l2a_granule"] + ".h5"
    west, south, east, north = bounds
    local = STORE / "tracks" / file.replace(".h5", ".parquet")
    provenance = local.with_suffix(".json")
    if local.is_file() and provenance.is_file():
        body = local.read_bytes()
        receipt = json.loads(provenance.read_bytes())
        if receipt["local_sha256"] != hashlib.sha256(body).hexdigest() or receipt["bounds"] != list(bounds):
            raise ValueError(f"GEDI scan does not match its receipt for {file}")
        return pq.read_table(io.BytesIO(body)), receipt
    columns = defaultdict(list)
    receipts = []
    missing_beams = []
    for beams, arrays, receipt in (group for half in (BEAMS[:4], BEAMS[4:]) for group in beam_group(file, half)):
        receipts.append(receipt)
        if not arrays:
            missing_beams.extend(beams)
            continue
        for beam in beams:
            v = {name: arrays[f"/BEAM{beam:04b}/{name}"] for name in FIELDS}
            ids = v["shot_number"]
            if not len(ids) or ids.dtype != np.dtype("uint64") or not np.all(ids[1:] > ids[:-1]) or any(len(a) != len(ids) for a in v.values()):
                raise ValueError(f"Misaligned GEDI beam {beam}/{file}")
            lat, lon = v["lat_lowestmode"], v["lon_lowestmode"]
            keep = np.flatnonzero((lon >= west) & (lon <= east) & (lat >= south) & (lat <= north))
            columns["beam"].append(np.full(len(keep), beam, dtype=np.uint8))
            columns["native_row"].append(keep.astype(np.uint32))
            for name, array in v.items():
                columns[name].append(array[keep])
    table = pa.table({name: np.concatenate(v) for name, v in columns.items()})
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    body = sink.getvalue().to_pybytes()
    result = dict(resource=file, l2a_concept_id=record["l2a_concept_id"], local_path=str(local.relative_to(ROOT)),
                  local_sha256=hashlib.sha256(body).hexdigest(), bounds=bounds, raw_query_receipts=receipts,
                  beams_without_arrays=missing_beams)
    write(local, body)
    write(provenance, (json.dumps(result,indent=2)+"\n").encode())
    return table, result


def positions(table, source, tanks, receipt):
    if not table.num_rows:
        return
    x, y = TO_UTM.transform(table["lon_lowestmode"].to_numpy(), table["lat_lowestmode"].to_numpy())
    tree = cKDTree(np.column_stack((x, y)))
    begin, end = (datetime.fromisoformat(source[f"l2a_capture_{side}_utc"]) for side in ("start","end"))
    for t in tanks.values():
        centre = t["east_m"], t["north_m"]
        for i in tree.query_ball_point(centre, t["radius_m"] + 12.5):
            row = {k: table[k][i].as_py() for k in table.column_names}
            clock = capture_time(GEDI_GPS_EPOCH, row["delta_time"])
            if not begin <= clock <= end:
                raise ValueError(f"GEDI shot outside the source time for {source['l2a_granule']}")
            if t["first_seen"] > clock.date().isoformat():
                continue
            edge = t["radius_m"] - float(np.hypot(x[i] - centre[0], y[i] - centre[1]))
            depth = t["top_hae_approx_m"] - row["elev_lowestmode"]
            plausible = 0 <= depth <= t["top_hae_approx_m"] - t["ground_hae_approx_m"]
            yield dict(gedi_acquisition_key=source["gedi_acquisition_key"], candidate_id=t["candidate_id"],
                terminal_id=t["terminal_id"], l2a_granule=source["l2a_granule"], l1b_granule=source["l1b_granule"],
                observed_at_utc=clock.isoformat(), centre_to_roof_edge_m=edge, footprint_inside_roof=edge >= 12.5,
                footprint_inside_with_10m_margin=edge >= 22.5, geometrically_plausible_lowest_mode=plausible,
                geometry_dataset=t["geometry_dataset"], wall_to_lowest_mode_approx_m=depth, regional_shots_path=receipt["local_path"], **row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include granules outside the catalogued roof-centre footprints")
    args = parser.parse_args()
    server = version(STORE)
    tanks, _, bounds = identities()
    all_rows = read("cushing_gedi_catalog")
    candidates = [r for r in all_rows if args.all or int(r["catalog_roof_centres"])]
    candidates.sort(key=lambda r: (not bool(int(r["nearby_radar_scenes"])), r["l2a_capture_start_utc"] < "2024"))
    scans, shots, failed = [], [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(scan, row, bounds): row for row in candidates}
        for i, future in enumerate(as_completed(pending), 1):
            record = pending[future]
            try:
                table, receipt = future.result()
                roof_rows = list(positions(table, record, tanks, receipt))
            except (OSError, ValueError) as error:
                failed.append(dict(acquisition=record["gedi_acquisition_key"], issue=str(error)))
            else:
                shots.extend(roof_rows)
                scans.append(dict(gedi_acquisition_key=record["gedi_acquisition_key"], l2a_granule=record["l2a_granule"],
                    covered_catalog_roof_centres=record["catalog_roof_centres"], regional_shots=table.num_rows,
                    local_path=receipt["local_path"], local_sha256=receipt["local_sha256"],
                    beams_without_arrays_json=json.dumps(receipt.get("beams_without_arrays",[]))))
            if i % 10 == 0:
                print(f"GEDI original coverage {i}/{len(candidates)}, {len(failed)} unresolved", flush=True)
    if not scans or not shots:
        raise ValueError(f"No original GEDI coverage: {failed[:3]}")
    export(f"{STEM}_granules", sorted(scans, key=lambda r: r["gedi_acquisition_key"]))
    export(f"{STEM}_shots", sorted(shots, key=lambda r: (r["observed_at_utc"],r["candidate_id"],r["shot_number"])))
    within = [r for r in shots if r["footprint_inside_with_10m_margin"]]
    summary = dict(sliderule_version=server, endpoint=f"{BASE}/h5p", granules_in_catalog=len(all_rows), granules_requested=len(candidates),
        granules_with_regional_subsets=len(scans), verified_all_beam_granules=sum(r["beams_without_arrays_json"]=="[]" for r in scans),
        partial_granules=[dict(acquisition=r["gedi_acquisition_key"], beams_without_arrays=json.loads(r["beams_without_arrays_json"]))
                         for r in scans if r["beams_without_arrays_json"]!="[]"],
        unresolved_granules=failed, regional_shots=sum(r["regional_shots"] for r in scans),
        footprint_roof_intersections=len(shots), fully_inside_roof=sum(r["footprint_inside_roof"] for r in shots),
        fully_inside_with_10m_margin=len(within), margin_and_plausible_lowest_mode=sum(r["geometrically_plausible_lowest_mode"] for r in within),
        fully_inside_with_10m_margin_by_year=Counter(r["observed_at_utc"][:4] for r in within),
        scope="Original GEDI v003 L2A arrays, clipped to the study bounds. Partial granules preserve all beams returned by the service. "
        "Footprints use nominal 25 m diameter. Contained plausible returns are candidates for waveform confirmation.")
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2)+"\n").encode())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
