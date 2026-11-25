"""Collect timestamped ATL03 v7 photon subsets and rebuild Cushing roof controls."""

import hashlib
import io
import json
import re
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from build_cushing_icesat2_controls import (TO_UTM, airborne_matches, export, holdout_calendar, identities,
    measurements, optical_matches, other_structure_context, rows_from_csv, source_catalog)
from build_s1_reference_matches import read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, write
from sliderule_h5 import BASE, message, records, request_packet, version

STEM = "cushing_icesat2_native"
STORE = RAW / STEM
PACKETS = STORE / "packets"
BEAM_NAMES = dict(zip((10, 20, 30, 40, 50, 60), ("gt1l", "gt1r", "gt2l", "gt2r", "gt3l", "gt3r"), strict=True))
PH_FIELDS = ("ph_id_pulse", "ph_id_channel", "ph_id_count", "delta_time")


def decode(body, filename, name):
    issues, alerts = [], []
    payload = bytearray()
    size = None
    complete = False
    for kind, data in records(body):
        if kind == "exceptrec":
            code, level, detail = message(data)
            if code < 0 or level >= 2:
                issues.append(detail)
            alerts.append(detail)
            continue
        if kind not in ("arrowrec.meta", "arrowrec.data", "arrowrec.eof") or len(data) < 128 or complete:
            raise ValueError(f"Unexpected ATL03 data record {kind}")
        if bytes(data[:128]).split(b"\0", 1)[0].decode("ascii") != name:
            raise ValueError("SlideRule returned a different output identity")
        if kind == "arrowrec.meta" and size is None and len(data) == 136:
            size = struct.unpack_from("<q", data, 128)[0]
        elif kind == "arrowrec.data" and size is not None:
            payload.extend(data[128:])
        elif kind == "arrowrec.eof" and size == len(payload) and len(data) == 136:
            complete = True
        else:
            raise ValueError(f"Incomplete ATL03 data stream for {filename}")
    if not complete:
        no_rows = not payload and any(re.search(r"received 0 x 0 dataframe \(eofs=6, dups=0, ptls=0\)", a, re.I)
                                      for a in alerts)
        if no_rows and all("resulted in an invalid dataframe" in m for m in issues) and any(
                f"Successfully completed processing resource [1 out of 1]: {filename}" in a for a in alerts):
            return None, None
        raise ValueError(f"No complete ATL03 response for {filename}: {issues or alerts[-3:]}")
    if any("produced an empty dataframe" not in m for m in issues) or not any(
            re.search(r"eofs=6, dups=0, ptls=0", a) for a in alerts):
        raise ValueError(f"Partial ATL03 response for {filename}: {issues or alerts[-3:]}")
    if not payload.startswith(b"PAR1") or not payload.endswith(b"PAR1"):
        raise ValueError(f"Bad Parquet signature for {filename}")
    table = pq.read_table(io.BytesIO(payload))
    if not table.num_rows and any("produced an empty dataframe" in m for m in issues):
        return None, None
    metadata = json.loads(table.schema.metadata[b"meta"])
    if metadata["endpoint"] != "atl03x" or set(metadata["srctbl"].values()) != {filename}:
        raise ValueError(f"Wrong ATL03 source identity {metadata}")
    return table, bytes(payload)


def collect(source, bounds):
    filename = source["filename"]
    west, south, east, north = bounds
    poly = [dict(lon=x, lat=y) for x, y in ((west,south), (east,south), (east,north), (west,north), (west,south))]
    name = filename.removesuffix(".h5") + ".parquet"
    parms = dict(poly=poly, resources=[filename], srt=0, cnf=[2,3,4], quality_ph=[0,1,2,3],
                 atl03_ph_fields=list(PH_FIELDS), atl03_geo_fields=["segment_id"], timeout=60,
                 output=dict(path=name, format="parquet", with_checksum=True))
    body, receipt = request_packet("atl03x", {"parms": parms}, PACKETS)
    table, payload = decode(body, filename, name)
    note = dict(source, **receipt, photon_count=0, used_photons=0, csv_path=None, packet_path=receipt["response_file"],
                maximum_subsample_stride=1, utc_granule_day=source["granule_start_utc"][:10],
                packet_granules_json=json.dumps([filename]), parquet_sha256=None)
    if table is None:
        return note, []
    cols = {k: table[k].to_numpy() for k in ("gt", "ph_index", "longitude", "latitude", "height", "atl03_cnf",
                                             "quality_ph", "time_ns", "rgt")}
    dates = cols["time_ns"].astype("datetime64[ns]").astype(np.int64)
    lo, hi = [int(datetime.fromisoformat(source[k]).timestamp()*1e9) for k in ("granule_start_utc", "granule_end_utc")]
    if (not table.num_rows or not set(cols["gt"]) <= BEAM_NAMES.keys() or not set(cols["atl03_cnf"]) <= {2,3,4}
        or not set(cols["quality_ph"]) <= {0,1,2,3} or not np.all(cols["rgt"] == int(filename.split("_")[2][:4]))
        or not np.all((dates >= lo) & (dates <= hi)) or not all(np.isfinite(cols[k]).all() for k in ("latitude","longitude","height"))
        or len(set(zip(cols["gt"], cols["ph_index"], strict=True))) != table.num_rows):
        raise ValueError(f"Invalid ATL03 photon values for {filename}")
    # Some SlideRule subsets return a few photons from boundary segments. Clip locally.
    inside = ((cols["longitude"] >= west) & (cols["longitude"] <= east) &
              (cols["latitude"] >= south) & (cols["latitude"] <= north))
    good = np.flatnonzero(inside & (cols["quality_ph"] == 0))
    photons = [{"beam": BEAM_NAMES[cols["gt"][i]], "latitude": float(cols["latitude"][i]),
                "longitude": float(cols["longitude"][i]), "photon height": float(cols["height"][i]),
                "confidence code": int(cols["atl03_cnf"][i]), "time_ns": int(dates[i]), "ph_index": int(cols["ph_index"][i])}
               for i in good]
    path = STORE / "photons" / name
    write(path, payload)
    note.update(photon_count=table.num_rows, photons_inside_query=int(inside.sum()), used_photons=len(photons),
                csv_path=str(path.relative_to(ROOT)), parquet_sha256=hashlib.sha256(payload).hexdigest())
    return note, photons


def observation_bounds(row, photons):
    beam = [p for p in photons if p["beam"] == row["beam"] and p["confidence code"] >= 3]
    x,y = TO_UTM.transform([p["longitude"] for p in beam], [p["latitude"] for p in beam])
    dist = np.hypot(np.asarray(x)-row["geometry_center_east_m"], np.asarray(y)-row["geometry_center_north_m"])
    times = [p["time_ns"] for p,d in zip(beam,dist,strict=True) if d <= row["geometry_radius_m"] - row["roof_inset_m"]]
    if not times:
        return None, None
    return tuple(datetime.fromtimestamp(t / 1e9, timezone.utc).isoformat() for t in (min(times), max(times)))


def compare(photons, old):
    if not old or not old["csv_path"]:
        return 0, 0
    a = rows_from_csv((ROOT / old["csv_path"]).read_bytes())
    by_beam = defaultdict(list)
    for p in photons:
        by_beam[p["beam"]].append((p["longitude"], p["latitude"], p["photon height"]))
    # The public visualization rounds lon/lat and elevation. Match within one metre
    # horizontally and two centimetres vertically without asserting photon identity.
    trees = {name: cKDTree(np.asarray(v) * [90000, 111000, 50]) for name,v in by_beam.items()}
    matched = 0
    for p in a:
        if p["beam"] in trees:
            x,y,z = map(float, (p["longitude"], p["latitude"], p["photon height"]))
            dist, _ = trees[p["beam"]].query([x*90000,y*111000,z*50], p=np.inf, distance_upper_bound=1.00001)
            matched += bool(np.isfinite(dist))
    return len(a), matched


def main():
    server = version(PACKETS)
    tanks, _, bounds = identities()
    sources = list(source_catalog(bounds, False).values())
    old = {}
    for r in read("cushing_icesat2_extracts"):
        if r["filename"] and int(r["photon_count"]) > int(old.get(r["filename"], {}).get("photon_count", 0)):
            old[r["filename"]] = r
    old_profiles = {(r["filename"], r["candidate_id"], r["beam"]): r for table in (
        "cushing_icesat2_roof_profiles", "cushing_icesat2_photon_recovery_profiles") for r in read(table)
        if r["passes_roof_profile_checks"] == "True"}
    dems, structures = read("cushing_icesat2_ground_assets"), other_structure_context()
    manifests, profiles, failed = [], [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(collect, source, bounds) for source in sources]
        for i,(source,future) in enumerate(zip(sources,jobs,strict=True),1):
            try:
                manifest, photons = future.result()
            except (ValueError, OSError) as error:
                failed.append(dict(filename=source["filename"], reason=str(error)))
                continue
            manifest["oa_photons_in_best_extract"], manifest["oa_rows_with_close_native_value"] = compare(photons, old.get(manifest["filename"]))
            manifests.append(manifest)
            if photons:
                manifest["requested_date"] = manifest["utc_granule_day"]
                manifest["track_id"] = int(manifest["filename"].split("_")[2][:4])
                for r in measurements(manifest, photons, tanks, dems, structures):
                    key = tuple(r[k] for k in ("filename","candidate_id","beam"))
                    first,last = observation_bounds(r, photons)
                    prior = old_profiles.get(key)
                    profiles.append(dict(r, first_roof_signal_photon_utc=first, last_roof_signal_photon_utc=last,
                        openaltimetry_previously_qualified=bool(prior),
                        openaltimetry_wall_to_roof_m=float(prior["estimated_wall_to_roof_m"]) if prior else None,
                        new_qualified_control=bool(r["passes_roof_profile_checks"] and not prior)))
            if i % 10 == 0:
                print(f"ATL03 native subsets {i}/{len(sources)}, {len(failed)} unresolved", flush=True)
    passing = [r for r in profiles if r["passes_roof_profile_checks"]]
    timed = [dict(r, granule_start_utc=r["first_roof_signal_photon_utc"], granule_end_utc=r["last_roof_signal_photon_utc"])
             for r in passing]
    if len({(r["filename"], r["beam"], r["candidate_id"]) for r in profiles}) != len(profiles):
        raise ValueError("Duplicate roof profile across native ATL03 subsets")
    optical, airborne = list(optical_matches(timed)), list(airborne_matches(timed))
    weeks = list(holdout_calendar(timed))
    for kind, rows in (("extracts",manifests), ("profiles",profiles), ("optical_controls",optical),
                       ("airborne_controls",airborne), ("holdout_weeks",weeks)):
        export(f"native_{kind}", rows)
    report = dict(sliderule_version=server, endpoint=f"{BASE}/atl03x", catalog_granules=len(sources), verified_subsets=len(manifests),
        unresolved_granules=failed, granules_with_usable_signal=sum(bool(r["used_photons"]) for r in manifests),
        photons_archived=sum(r["photon_count"] for r in manifests), nominal_signal_photons_inside_bounds=sum(r["used_photons"] for r in manifests),
        openaltimetry_photon_rows_compared=sum(r["oa_photons_in_best_extract"] for r in manifests),
        openaltimetry_rows_with_close_native_value=sum(r["oa_rows_with_close_native_value"] for r in manifests),
        profiles=len(profiles), passing_profiles=len(passing), passing_roofs=len({r["candidate_id"] for r in passing}),
        new_qualified_profiles=sum(r["new_qualified_control"] for r in passing),
        optical_controls_within_48h=len(optical), optical_controls_within_24h=sum(r["max_time_separation_hours"]<=24 for r in optical),
        airborne_controls_within_14d=len(airborne), holdout_weeks_with_current_height=sum(bool(r["current_retrospective_roofs"]) for r in weeks),
        holdout_weeks_with_same_roof_on_both_dates=sum(bool(r["same_roofs_with_both_endpoints"]) for r in weeks),
        scope="Original ATL03 v7 photons served by SlideRule. The product has its own land confidence and quality flags. "
        "Profiles apply the existing geometry, ground-alignment and stability requirements. Individual photon timestamps replace "
        "whole-granule bounds for engineering comparisons. These are retrospective height controls.")
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
