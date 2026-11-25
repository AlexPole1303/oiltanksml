"""Recover native SLC roof windows from every available CEDA Sentinel-1A holdout SAFE."""

import gzip
import hashlib
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email import policy
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
from shapely.geometry import Point

from audit_cushing_slc_archive import parent_files, probe
from build_ceda_s1_slc_windows import NativeGeometry
from build_cushing_icesat2_controls import identities
from build_s1_holdout_gap_burst_plan import cdse_records, geometry, roofs
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_ceda_s1_slc_annotations import archive_urls, collect as annotations_for, utc
from collect_ceda_s1_slc_samples import RangeZip, member_offset, native_layout
from collect_public import RAW, ROOT, TLS, write

STEM = "ceda_s1_slc_regional"
STORE = RAW / STEM
PART_COUNT = 48
RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")
POLARIZATIONS = ("VV", "VH")


def ranges(remote, spans):
    if len(spans) == 1:
        start, end = spans[0]
        return {start: remote.slice(start, end-start)}
    header = "bytes=" + ",".join(f"{start}-{end-1}" for start, end in spans)
    request = Request(remote.url, headers={"Range": header, "Accept-Encoding": "identity", "If-Match": remote.etag})
    expected_bytes = sum(end-start for start, end in spans)
    limit = expected_bytes + len(spans)*1024
    for attempt in range(3):
        try:
            with urlopen(request, context=TLS, timeout=45) as response:
                content_type = response.headers["Content-Type"]
                if (response.status != 206 or response.headers["ETag"] != remote.etag
                        or not content_type.startswith("multipart/byteranges;")):
                    raise ValueError("Archive did not return a matching multipart byte-range response")
                payload = response.read(limit+1)
            break
        except OSError as error:
            if attempt == 2 or (isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}):
                raise
            time.sleep(2**attempt)
    if len(payload) > limit:
        raise ValueError("Unexpectedly large multipart response")
    message = BytesParser(policy=policy.default).parsebytes(f"Content-Type: {content_type}\r\n\r\n".encode()+payload)
    expected = dict(spans)
    result = {}
    for item in message.iter_parts():
        match = RANGE.fullmatch(item["Content-Range"])
        first, last, total = map(int, match.groups())
        body = item.get_payload(decode=True)
        if (first in result or expected.get(first) != last+1 or total != remote.size or len(body) != last-first+1):
            raise ValueError("Multipart byte range disagrees with its source request")
        result[first] = body
    if result.keys() != expected.keys():
        raise ValueError("Missing expected radar byte ranges")
    return result


def selection(row, contexts):
    records, path = cdse_records(row)
    footprints = defaultdict(dict)
    for burst in records:
        key = burst["BurstId"], burst["SwathIdentifier"]
        footprints[key][burst["PolarisationChannels"]] = burst, geometry(burst["GeoFootprint"])
    covered = defaultdict(list)
    for key, bands in footprints.items():
        if set(bands) != set(POLARIZATIONS):
            continue
        area = bands["VV"][1].intersection(bands["VH"][1])
        covered[key] = [(tank, area.boundary.distance(context.centroid)) for tank, context in contexts.items()
                        if area.covers(context)]
    best = {}
    for key, tanks in covered.items():
        for tank, margin in tanks:
            if tank not in best or best[tank][0] < margin:
                best[tank] = margin, key
    by_burst = defaultdict(list)
    for tank, (_, key) in best.items():
        by_burst[key].append(tank)
    if not by_burst:
        raise ValueError(f"No roofs with full dual-polarization SLC context in {row['product_name']}")
    queries, roofs_by_uuid = [], {}
    for key, tanks in sorted(by_burst.items()):
        for pol in POLARIZATIONS:
            b, _ = footprints[key][pol]
            queries.append(dict(scope="regional_holdout", product_id=row["product_id"], product_name=row["product_name"],
                cdse_uuid=b["Id"], swath=b["SwathIdentifier"], polarization=pol, weeks="", asf_data_url="",
                asf_annotation_url="", cdse_catalog_file=path, asf_catalog_file=""))
            roofs_by_uuid[b["Id"]] = sorted(tanks)
    return queries, roofs_by_uuid


def merged_rows(requested, image_origin, width):
    by_row = defaultdict(list)
    for _, roi in requested:
        a, b, c, d = (roi[k] for k in ("first_line", "end_line_exclusive", "first_sample", "end_sample_exclusive"))
        for row in range(a, b):
            by_row[row].append((c, d))
    output = []
    for row, cols in sorted(by_row.items()):
        merged = []
        for a, b in sorted(cols):
            if merged and a <= merged[-1][1]:
                merged[-1] = merged[-1][0], max(b, merged[-1][1])
            else:
                merged.append((a, b))
        line_start = image_origin+row*width*4
        output.extend((row, a, b, line_start+a*4, line_start+b*4) for a, b in merged)
    return output


def compact_windows(remote, image_origin, width, endian, requested):
    intervals = merged_rows(requested, image_origin, width)
    rows = defaultdict(list)
    for i in range(0, len(intervals), PART_COUNT):
        group = intervals[i:i+PART_COUNT]
        parts = ranges(remote, [(a, b) for _, _, _, a, b in group])
        for line, left, right, start, _ in group:
            iq = np.frombuffer(parts[start], dtype=f"{endian}i2").reshape(-1, 2)
            if len(iq) != right-left:
                raise ValueError("Radar samples differ from TIFF strip layout")
            rows[line].append((left, right, iq))
    raw, windows = [], []
    offset = 0
    for tank, roi in requested:
        a, b, c, d = (roi[k] for k in ("first_line", "end_line_exclusive", "first_sample", "end_sample_exclusive"))
        pixels = []
        for line in range(a, b):
            ((left, _, iq),) = [r for r in rows[line] if r[0] <= c and d <= r[1]]
            pixels.append(iq[c-left:d-left])
        window = np.stack(pixels)
        if window.shape != (b-a, d-c, 2) or np.mean(np.any(window != 0, axis=2)) < .95:
            raise ValueError(f"Empty or malformed complex window for {tank}")
        raw.append(window.reshape(-1, 2))
        count = (b-a)*(d-c)
        windows.append(dict(candidate_id=tank, offset_complex_samples=offset, complex_samples=count,
                            height=b-a, width=d-c, **roi))
        offset += count
    return np.concatenate(raw).astype("<i2", copy=False), windows, sum(b-a for _, _, _, a, b in intervals), len(intervals)


def collect_scene(row, tanks, contexts):
    queries, roofs_by_uuid = selection(row, contexts)
    requests, files = parent_files((row["product_id"], queries))
    measured = {f["s3_path"]: f["bytes"] for f in files if f["role"] == "measurement"}
    for r in requests:
        r["cdse_parent_swath_bytes"] = measured[r["measurement_s3_path"]]
        for key in ("byte_offset", "relative_burst_id", "cdse_absolute_burst_id", "asf_burst_index"):
            r[key] = "" if r[key] is None else str(r[key])
    parent, annotations, xml_files = annotations_for((row["product_id"], requests))
    if parent["metadata_http_status"] != 200 or parent["full_zip_signature"] != "ZIP":
        raise ValueError(f"CEDA sources changed for {row['product_name']}")
    plan = {r["cdse_uuid"]: r for r in requests}
    url = json.loads(parent["ceda_attempts_json"])[-1][0] + ".zip?download=1"
    name = row["product_name"]
    assets, windows = [], []
    with RangeZip(url) as remote, ZipFile(remote) as archive:
        with archive.open(f"{name}/manifest.safe") as handle:
            if hashlib.sha256(handle.read()).hexdigest() != parent["manifest_sha256"]:
                raise ValueError(f"CEDA manifest differs between archives for {name}")
        for annotation in annotations:
            key = annotation["cdse_uuid"]
            item = archive.getinfo(f"{name}/{annotation['measurement_safe_path']}")
            if item.file_size != int(annotation["native_measurement_bytes"]):
                raise ValueError(f"TIFF member changed for {name}/{key}")
            endian, lines, width, origin = native_layout(remote, member_offset(remote, item), item, annotation, plan[key])
            geolocation = NativeGeometry(annotation)
            if max(geolocation.grid_errors) > .05:
                raise ValueError(f"Product geolocation differs from native model for {name}/{key}")
            near = utc(annotation["slc_burst_azimuth_time_utc"])+timedelta(
                seconds=lines*float(annotation["native_azimuth_interval_seconds"])/2)
            candidates = [(tank, geolocation.window(tanks[tank], near.isoformat())) for tank in roofs_by_uuid[key]]
            candidates = [(tank, roi) for tank, roi in candidates if roi["full_window_valid_by_native_mask"]]
            if not candidates:
                raise ValueError(f"No qualified native windows for {name}/{key}")
            values, blocks, transferred, intervals = compact_windows(remote, origin, width, endian, candidates)
            target = STORE / "iq" / f"{key}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=target.parent) as tmp:
                stage = target.parent / tmp / target.name
                np.savez_compressed(stage, iq_int16=values, cdse_uuid=np.array(key))
                with np.load(stage, allow_pickle=False) as packet:
                    if str(packet["cdse_uuid"]) != key or not np.array_equal(packet["iq_int16"], values):
                        raise OSError(f"Readback changed radar samples for {key}")
                stage.replace(target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            assets.append(dict(product_id=row["product_id"], product_name=name, cdse_uuid=key, swath=annotation["swath"],
                polarization=annotation["polarization"], source_url=url, source_zip_bytes=remote.size,
                source_zip_etag=remote.etag, source_zip_last_modified_utc=parsedate_to_datetime(remote.last_modified),
                first_byte_of_burst_in_zip=origin, lines_per_burst=lines, samples_per_burst=width,
                capture_start_utc=row["capture_start_utc"], capture_end_utc=row["capture_end_utc"],
                catalog_publication_utc=row["catalog_publication_utc"],
                full_measurement_md5=annotation["native_measurement_md5"], full_measurement_zip_crc32=f"{item.CRC:08x}",
                manifest_sha256=parent["manifest_sha256"], local_annotation=annotation["local_annotation_gzip"],
                local_calibration=annotation["local_calibration_gzip"], local_noise=annotation["local_noise_gzip"],
                local_valid_mask=annotation["valid_mask_gzip"], packet_sha256=digest, packet_bytes=target.stat().st_size,
                local_packet=str(target.relative_to(ROOT)), transferred_unique_iq_bytes=transferred, ranges_read=intervals,
                roofs_with_full_valid_windows=len(blocks), max_grid_line_or_sample_error=max(geolocation.grid_errors)))
            windows.extend(dict(product_id=row["product_id"], cdse_uuid=key, polarization=annotation["polarization"],
                                relative_burst_id=plan[key]["relative_burst_id"], **r) for r in blocks)
    return parent, assets, windows, xml_files


def main():
    products = [r for r in read("s1_cushing_catalog") if r["product_type"] == "IW_SLC__1S"
        and r["covers_cushing_reference_point"] == "True" and r["satellite"] == "S1A"
        and "2023-12-22" <= r["capture_start_utc"] < "2026-11-01"]
    jobs = [("ceda_full_zip", r["product_id"], next(archive_urls(r["product_name"]))+".zip?download=1") for r in products]
    with ThreadPoolExecutor(max_workers=6) as pool:
        probes = list(pool.map(probe, jobs))
    status = {r["archive_id"]: r for r in probes}
    export(f"{STEM}_archive_access", [dict(product_id=p["product_id"], product_name=p["product_name"],
        capture_start_utc=p["capture_start_utc"], catalog_publication_utc=p["catalog_publication_utc"],
        **{k: status[p["product_id"]][k] for k in ("url", "status", "source_signature")}) for p in products])
    available = [r for r in products if status[r["product_id"]]["source_signature"] == "ZIP"]
    tanks, _, _ = identities()
    contexts = {key: Point(r["east_m"], r["north_m"]).buffer(r["radius_m"]+120) for key, r in tanks.items()}
    print(f"CEDA confirmed ZIP access for {len(available)}/{len(products)} Sentinel-1A parents", flush=True)
    parents, assets, windows, files = [], [], [], []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(collect_scene, r, tanks, contexts): r["product_name"] for r in available}
        failures = []
        for future in as_completed(pending):
            try:
                parent, new_assets, new_windows, new_files = future.result()
            except Exception as error:
                scene = pending[future]
                print(f"{scene}: {type(error).__name__}: {error}", flush=True)
                failures.append(dict(product_name=scene, error_type=type(error).__name__, detail=str(error)))
                continue
            parents.append(parent)
            assets.extend(new_assets)
            windows.extend(new_windows)
            files.extend(new_files)
            print(f"{parent['product_name']} {len(new_assets)} native layers and {len(new_windows)} roof windows", flush=True)
    for label, rows in (("parents", parents), ("assets", assets), ("windows", windows), ("annotation_files", files)):
        export(f"{STEM}_{label}", sorted(rows, key=lambda r: (r["product_id"], r.get("cdse_uuid", ""),
                                                             r.get("candidate_id", ""), r.get("safe_relative_path", ""))))
    # Use dated geometry and exact weekly endpoints from the existing release calendar.
    registered = {week: {r["candidate_id"] for r, *_ in values} for week, values in roofs().items()}
    layers = {r["cdse_uuid"]: r for r in assets}
    dual = defaultdict(dict)
    for r in windows:
        key = r["product_id"], r["candidate_id"], r["relative_burst_id"]
        dual[key][r["polarization"]] = r["cdse_uuid"]
    coverage = defaultdict(dict)
    for (parent, tank, burst), bands in dual.items():
        if set(bands) == set(POLARIZATIONS):
            coverage[parent, burst][tank] = bands
    ready = lambda uuids: max(max(layers[i]["source_zip_last_modified_utc"], utc(layers[i]["catalog_publication_utc"]))
                              for i in uuids)
    pairs = []
    for r in read("s1_cushing_weekly_catalog_coverage"):
        week = r["week_ending"]
        if r["study_phase"] != "holdout":
            continue
        groups = [[(p, b) for p in json.loads(r[f"slc_{side}_product_ids_json"]) for (a, b) in coverage if a == p]
                  for side in ("previous", "current")]
        for before in groups[0]:
            for after in groups[1]:
                if before[1] != after[1]:
                    continue
                for tank in sorted(registered.get(week, set()) & coverage[before].keys() & coverage[after].keys()):
                    a, b = coverage[before][tank], coverage[after][tank]
                    pairs.append(dict(week_ending=week, candidate_id=tank, relative_burst_id=before[1],
                        previous_vv=a["VV"], previous_vh=a["VH"], current_vv=b["VV"], current_vh=b["VH"],
                        latest_archive_evidence_utc=max(ready(a.values()), ready(b.values()))))
    optics = {r["week_ending"]: r for r in read("holdout_optical_weeks")}
    for r in pairs:
        week = optics[r["week_ending"]]
        r.update(forecast_cutoff_utc=week["cutoff_utc"], b04_status=week["b04_status"],
                 all_versions_before_cutoff=r["latest_archive_evidence_utc"] < utc(week["cutoff_utc"]))
    export(f"{STEM}_weekly_pairs", pairs)
    eligible = {(r["week_ending"], r["candidate_id"]) for r in pairs if r["all_versions_before_cutoff"]}
    gaps = {w for w, _ in eligible if optics[w]["b04_status"] != "measured"}
    report = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), checked_exact_ceda_parent_products=len(products),
        archive_http_status=Counter(r["status"] for r in probes), verified_native_scenes=len(parents),
        collection_failures=failures,
        verified_native_burst_polarizations=len(assets), verified_roof_windows=len(windows),
        local_native_packet_bytes=sum(r["packet_bytes"] for r in assets),
        source_unique_iq_bytes=sum(r["transferred_unique_iq_bytes"] for r in assets),
        pre_cutoff_same_burst_weeks=len({w for w, _ in eligible}), pre_cutoff_same_burst_tank_weeks=len(eligible),
        pre_cutoff_same_burst_optical_gap_weeks=sorted(gaps),
        pre_cutoff_same_burst_optical_gap_tank_weeks=sum(w in gaps for w, _ in eligible),
        interpretation="Both polarizations retain original I/Q values. Window rows and columns come from the embedded "
        "SAFE orbit and every window passes the original sample mask. Dates use the maximum of the exact ZIP object's "
        "Last-Modified and Copernicus parent publication. No roof heights or inventory estimates were inferred.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} online CEDA products need verification")


if __name__ == "__main__":
    main()
