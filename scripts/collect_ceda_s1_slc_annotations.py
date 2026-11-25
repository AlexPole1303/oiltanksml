"""Recover native Sentinel-1 SAFE annotations from CEDA and verify burst identity."""

import gzip
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import PurePosixPath
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np

from audit_cushing_slc_archive import probe
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, TLS, write

STEM = "ceda_s1_slc_annotations"
STORE = RAW / STEM
CEDA = "https://dap.ceda.ac.uk/neodc"
MAX_METADATA_BYTES = 32 * 1024**2
ROLES = ("annotation", "calibration", "noise")


def utc(value):
    stamp = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def archive_urls(name):
    satellite = name[2].lower()
    day = datetime.strptime(name.split("_")[5][:8], "%Y%m%d")
    for version in (3, 2) if day.year <= 2019 else (3,):
        yield f"{CEDA}/sentinel1{satellite}/data/IW/L1_SLC/IPF_v{version}/{day:%Y/%m/%d}/{name.removesuffix('.SAFE')}"


def packet(base):
    url = base + ".metadata_only.zip?download=1"
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=55, context=TLS) as response:
        if response.status != 200:
            raise ValueError(f"Unexpected metadata archive response {response.status} for {base}")
        size = int(response.headers["Content-Length"])
        if not 0 < size <= MAX_METADATA_BYTES:
            raise ValueError(f"Unreasonable metadata archive size {size} for {base}")
        payload = response.read(size+1)
        if len(payload) != size or not payload.startswith(b"PK\x03\x04"):
            raise ValueError(f"Incomplete metadata ZIP for {base}")
        modified = response.headers.get("Last-Modified")
    return payload, parsedate_to_datetime(modified) if modified else None


def manifest_entries(root):
    entries = {}
    for node in root.findall(".//dataObjectSection/dataObject/byteStream"):
        href = node.find("fileLocation").attrib["href"]
        checksum = node.find("checksum")
        digest = checksum.text.lower()
        path = PurePosixPath(href)
        if (not href.startswith("./") or path.is_absolute() or ".." in path.parts
                or path.as_posix() in entries or checksum.attrib["checksumName"].upper() != "MD5"
                or not re.fullmatch(r"[a-f0-9]{32}", digest)):
            raise ValueError(f"Unexpected SAFE manifest entry {href}")
        entries[path.as_posix()] = (int(node.attrib["size"]), digest)
    return entries


def burst_annotation(root, request):
    header = root.find("adsHeader")
    name = request["product_name"]
    tokens = name.split("_")
    if tuple(header.findtext(key) for key in ("missionId", "productType", "mode", "swath", "polarisation")) != (
            tokens[0], "SLC", "IW", request["swath"], request["polarization"]):
        raise ValueError(f"Annotation is for a different band in {name}")
    if (int(header.findtext("absoluteOrbitNumber")) != int(tokens[7])
            or int(header.findtext("missionDataTakeId")) != int(tokens[8], 16)):
        raise ValueError(f"Annotation is for a different acquisition in {name}")
    timing = root.find("swathTiming")
    lines, samples = (int(timing.findtext(k)) for k in ("linesPerBurst", "samplesPerBurst"))
    if (lines, samples) != (int(request["lines_per_burst"]), int(request["samples_per_burst"])):
        raise ValueError(f"Annotation and catalog dimensions differ in {name}")
    bursts = timing.find("burstList")
    if len(bursts) != int(bursts.attrib["count"]):
        raise ValueError(f"Incorrect SAFE burst count in {name}")
    index, burst = next(((i, b) for i, b in enumerate(bursts) if b.findtext("byteOffset") == request["byte_offset"]),
                        (None, None))
    if burst is None or abs((utc(burst.findtext("azimuthTime"))-utc(request["azimuth_time_utc"])).total_seconds()) > .001:
        raise ValueError(f"SAFE burst byte and acquisition time do not match {request['cdse_uuid']}")
    burst_id = burst.find("burstId")
    if burst_id is not None and (burst_id.text, burst_id.get("absolute")) != (
            request["relative_burst_id"], request["cdse_absolute_burst_id"]):
        raise ValueError(f"SAFE burst identifier differs for {request['cdse_uuid']}")
    if request["asf_burst_index"] and int(request["asf_burst_index"]) != index:
        raise ValueError(f"SAFE burst index differs from ASF for {request['cdse_uuid']}")
    bounds = []
    for tag in ("firstValidSample", "lastValidSample"):
        values = burst.find(tag)
        samples_by_line = np.fromstring(values.text, dtype=np.int32, sep=" ")
        if len(samples_by_line) != lines or int(values.attrib["count"]) != lines:
            raise ValueError(f"SAFE mask size differs for {request['cdse_uuid']}")
        bounds.append(samples_by_line)
    first, last = bounds
    invalid = first == -1
    if np.any(invalid != (last == -1)) or np.any((first[~invalid] < 0) | (first[~invalid] > last[~invalid])
                                               | (last[~invalid] >= samples)) or np.all(invalid):
        raise ValueError(f"Invalid per-line SAFE sample bounds for {request['cdse_uuid']}")
    image = root.find("imageAnnotation/imageInformation")
    if (image.findtext("pixelValue") != "Complex" or int(image.findtext("numberOfSamples")) != samples
            or int(image.findtext("numberOfLines")) != len(bursts)*lines):
        raise ValueError(f"Invalid SLC image geometry for {request['cdse_uuid']}")
    return dict(slc_burst_index=index, slc_burst_count=len(bursts), safe_burst_id_present=burst_id is not None,
        slc_burst_azimuth_time_utc=utc(burst.findtext("azimuthTime")), first_native_valid_line=int(np.flatnonzero(~invalid)[0]),
        last_native_valid_line=int(np.flatnonzero(~invalid)[-1]), valid_native_lines=int(np.count_nonzero(~invalid)),
        valid_native_samples=int(np.sum(last[~invalid]-first[~invalid]+1, dtype=np.int64)),
        native_azimuth_interval_seconds=float(image.findtext("azimuthTimeInterval")),
        native_azimuth_pixel_spacing_m=float(image.findtext("azimuthPixelSpacing")),
        native_range_pixel_spacing_m=float(image.findtext("rangePixelSpacing")),
        native_slant_range_first_sample_seconds=float(image.findtext("slantRangeTime")),
        embedded_state_vectors=len(root.findall("./generalAnnotation/orbitList/orbit"))), first, last


def save_xml(name, relative, payload):
    path = STORE / name / f"{relative}.gz"
    write(path, gzip.compress(payload, compresslevel=6, mtime=0))
    if gzip.decompress(path.read_bytes()) != payload:
        raise OSError(f"SAFE file failed readback at {path}")
    return str(path.relative_to(ROOT))


def collect(job):
    parent, requests = job
    name, = {r["product_name"] for r in requests}
    attempts = []
    for base in archive_urls(name):
        try:
            payload, modified = packet(base)
        except HTTPError as exc:
            if exc.code not in {401, 403, 404}:
                raise
            attempts.append((base, exc.code))
        else:
            attempts.append((base, 200))
            break
    else:
        full = probe(("full_s1_slc", parent, base + ".zip?download=1"))
        return dict(product_id=parent, product_name=name, metadata_http_status=attempts[-1][1],
                    ceda_attempts_json=json.dumps(attempts), metadata_bytes=0, metadata_zip_sha256="",
                    metadata_archive_last_modified_utc=None, local_manifest_gzip="", manifest_sha256="",
                    full_zip_http_status=full["status"], full_zip_signature=full["source_signature"]), [], []
    full = probe(("full_s1_slc", parent, base + ".zip?download=1"))
    files = {}
    rows = []
    with ZipFile(io.BytesIO(payload)) as archive:
        paths = archive.namelist()
        if (len(paths) != len(set(paths)) or any(not p.startswith(name+"/") for p in paths)
                or any("/measurement/" in p and p.endswith((".tif", ".tiff")) for p in paths)):
            raise ValueError(f"Unexpected members in metadata-only archive {name}")
        manifest = archive.read(f"{name}/manifest.safe")
        manifest_root = ET.fromstring(manifest)
        entries = manifest_entries(manifest_root)
        local_manifest = save_xml(name, "manifest.safe", manifest)
        for request in requests:
            parts = {role: request[f"{role}_s3_path"].split(f"/{name}/", 1)[1] for role in ROLES}
            raster = request["measurement_s3_path"].split(f"/{name}/", 1)[1]
            size, digest = entries[raster]
            if size != int(request["cdse_parent_swath_bytes"]):
                raise ValueError(f"CEDA manifest and CDSE TIFF size disagree in {name}")
            row = dict(product_id=parent, product_name=name, scope=request["scope"], cdse_uuid=request["cdse_uuid"],
                       swath=request["swath"], polarization=request["polarization"], weeks=request["weeks"],
                       measurement_safe_path=raster, native_measurement_md5=digest, native_measurement_bytes=size)
            for role, relative in parts.items():
                source = archive.read(f"{name}/{relative}")
                length, expected = entries[relative]
                if len(source) != length or hashlib.md5(source).hexdigest() != expected:
                    raise ValueError(f"SAFE member failed manifest verification: {name}/{relative}")
                if relative not in files:
                    files[relative] = dict(product_id=parent, product_name=name, role=role, safe_relative_path=relative,
                        native_bytes=length, manifest_md5=expected, sha256=hashlib.sha256(source).hexdigest(),
                        local_gzip=save_xml(name, relative, source))
                row[f"local_{role}_gzip"] = files[relative]["local_gzip"]
                if role == "annotation":
                    geometry, first, last = burst_annotation(ET.fromstring(source), request)
                    row.update(geometry)
                    mask_path = STORE / name / f"{request['cdse_uuid']}_valid_samples.csv.gz"
                    mask = "".join(f"{i},{lo},{hi}\n" for i, (lo, hi) in enumerate(zip(first, last, strict=True)))
                    write(mask_path, gzip.compress(("burst_line,first_valid_sample,last_valid_sample\n"+mask).encode(), mtime=0))
                    row["valid_mask_gzip"] = str(mask_path.relative_to(ROOT))
            rows.append(row)
    return dict(product_id=parent, product_name=name, metadata_http_status=200,
        ceda_attempts_json=json.dumps(attempts), metadata_bytes=len(payload), metadata_zip_sha256=hashlib.sha256(payload).hexdigest(),
        metadata_archive_last_modified_utc=modified, local_manifest_gzip=local_manifest,
        manifest_sha256=hashlib.sha256(manifest).hexdigest(), full_zip_http_status=full["status"],
        full_zip_signature=full["source_signature"]), rows, list(files.values())


def main():
    groups = defaultdict(list)
    for row in read("cushing_slc_archive_bursts"):
        groups[row["product_id"]].append(row)
    results = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        for parent, bursts, files in pool.map(collect, sorted(groups.items(), key=lambda item: item[1][0]["product_name"])):
            print(parent["product_name"], parent["metadata_http_status"], len(bursts), flush=True)
            results.append((parent, bursts, files))
    parents = [p for p, _, _ in results]
    bursts = [r for _, group, _ in results for r in group]
    files = [r for _, _, group in results for r in group]
    for label, rows in (("parents", parents), ("bursts", bursts), ("files", files)):
        export(f"{STEM}_{label}", rows)
    by_burst = {r["cdse_uuid"] for r in bursts}
    holdout = defaultdict(set)
    for pair in read("s1_holdout_gap_burst_tank_pairs"):
        if pair["full_context_in_both_bursts"] == "True" and {pair["previous_cdse_uuid"], pair["current_cdse_uuid"]} <= by_burst:
            holdout[pair["week_ending"]].add(pair["candidate_id"])
    summary = dict(observed_utc=datetime.now(timezone.utc).isoformat(), requested_parents=len(parents),
        metadata_status=Counter(str(r["metadata_http_status"]) for r in parents), verified_native_annotation_bursts=len(bursts),
        verified_native_annotation_bursts_by_scope=Counter(r["scope"] for r in bursts), verified_native_xml_members=len(files),
        source_metadata_zip_bytes_received=sum(r["metadata_bytes"] for r in parents),
        preserved_local_compressed_bytes=sum(p.stat().st_size for p in STORE.glob("**/*.gz")),
        full_zip_range_http_status=Counter(str(r["full_zip_http_status"]) for r in parents),
        full_zip_range_signatures=Counter(r["full_zip_signature"] for r in parents if r["full_zip_signature"]),
        holdout_weeks_with_both_native_annotations=len(holdout),
        holdout_tank_weeks_with_both_native_annotations=sum(map(len, holdout.values())),
        holdout_weeks_with_both_native_annotations_by_date={w: len(roofs) for w, roofs in sorted(holdout.items())},
        interpretation="The selected XML members and SAFE manifest are retained, with ZIP hash and native checksums. "
        "Measurement size and MD5 are expectations from the manifest; complex pixels were not transferred. "
        "CEDA HTTP Last-Modified is an archive object timestamp, without a historical delivery guarantee.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
