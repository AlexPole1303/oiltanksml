"""Read native complex-int16 bursts by byte range from CEDA's public SAFE ZIPs."""

import gzip
import hashlib
import io
import json
import os
import re
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZIP_STORED, ZipFile

import numpy as np

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, TLS, write

STEM = "ceda_s1_slc_samples"
STORE = RAW / STEM
CHUNK_BYTES = 8 * 1024**2


class RangeZip(io.RawIOBase):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.position = 0
        self.size = None
        self.etag = None
        self.last_modified = None
        self.slice(0, 1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        origin = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        if origin+offset < 0:
            raise ValueError("Negative archive position")
        self.position = origin+offset
        return self.position

    def slice(self, start, count):
        if not 0 < count <= CHUNK_BYTES or start < 0 or (self.size and start+count > self.size):
            raise ValueError("Requested range exceeds archive or transfer bounds")
        end = start+count-1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self.etag:
            headers["If-Match"] = self.etag
        request = Request(self.url, headers=headers)
        for attempt in range(3):
            try:
                with urlopen(request, context=TLS, timeout=45) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    etag = response.headers.get("ETag")
                    if response.status != 206 or not match or not etag or etag.startswith("W/"):
                        raise ValueError(f"Missing exact conditional byte-range response from {self.url}")
                    first, last, size = map(int, match.groups())
                    if (first, last) != (start, end) or (self.size is not None and (size, etag) != (self.size, self.etag)):
                        raise ValueError(f"Archive changed or served a different byte range at {self.url}")
                    payload = response.read(count+1)
                    if len(payload) != count:
                        raise OSError(f"Short byte range {start}-{end}")
                    self.size, self.etag = size, etag
                    self.last_modified = response.headers.get("Last-Modified")
                    return payload
            except (OSError, URLError) as error:
                if attempt == 2 or (isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}):
                    raise
                time.sleep(2**attempt)

    def read(self, count=-1):
        remaining = self.size-self.position
        count = remaining if count < 0 else min(count, remaining)
        if count <= 0:
            return b""
        payload = self.slice(self.position, count)
        self.position += len(payload)
        return payload


def member_offset(remote, entry):
    if entry.compress_type != ZIP_STORED or entry.compress_size != entry.file_size or entry.flag_bits & 1:
        raise ValueError(f"TIFF member is not stored as a seekable byte stream: {entry.filename}")
    header = remote.slice(entry.header_offset, 30)
    fields = struct.unpack("<4s5H3I2H", header)
    filename_length, extra_length = fields[-2:]
    name = remote.slice(entry.header_offset+30, filename_length)
    encoding = "utf-8" if entry.flag_bits & (1 << 11) else "cp437"
    if fields[0] != b"PK\x03\x04" or fields[2:4] != (entry.flag_bits, ZIP_STORED) or name.decode(encoding) != entry.filename:
        raise ValueError(f"Local ZIP header disagrees with central directory: {entry.filename}")
    return entry.header_offset+30+filename_length+extra_length


def native_layout(remote, base, entry, annotation, planned):
    header = remote.slice(base, 8)
    endian = {b"II*\0": "<", b"MM\0*": ">"}.get(header[:4])
    if endian is None:
        raise ValueError(f"Unexpected classic TIFF signature for {entry.filename}")
    offset = struct.unpack_from(endian+"I", header, 4)[0]
    if not 8 <= offset <= entry.file_size-2:
        raise ValueError("TIFF directory lies outside member")
    count = struct.unpack(endian+"H", remote.slice(base+offset, 2))[0]
    size = count*12
    if not 0 < count < 100 or offset+2+size+4 > entry.file_size:
        raise ValueError("Invalid TIFF directory size")
    raw = remote.slice(base+offset+2, size)
    entries = {}
    for i in range(0, len(raw), 12):
        tag, kind, number = struct.unpack_from(endian+"HHI", raw, i)
        entries[tag] = (kind, number, raw[i+8:i+12])

    def read_tag(tag, expected_count):
        kind, number, field = entries[tag]
        code = {3: "H", 4: "I"}[kind]
        if number != expected_count:
            raise ValueError(f"Incorrect element count for TIFF tag {tag}")
        length = struct.calcsize(endian+code)*number
        if length > 4:
            pointer = struct.unpack(endian+"I", field)[0]
            if not 8 <= pointer <= entry.file_size-length:
                raise ValueError(f"Out-of-bounds TIFF tag {tag}")
            field = remote.slice(base+pointer, length)
        return struct.unpack_from(endian+str(number)+code, field)

    lines, width = (int(planned[k]) for k in ("lines_per_burst", "samples_per_burst"))
    height = int(annotation["slc_burst_count"])*lines
    expected = {256: width, 257: height, 258: 32, 259: 1, 277: 1, 278: 1, 284: 1, 339: 5}
    if any(read_tag(tag, 1) != (expected_value,) for tag, expected_value in expected.items()):
        raise ValueError(f"TIFF is not the expected uncompressed complex integer format for {entry.filename}")
    offsets = read_tag(273, height)
    byte_counts = read_tag(279, height)
    start = int(annotation["slc_burst_index"])*lines
    step = width*4
    first = int(planned["byte_offset"])
    if (offsets[start:start+lines] != tuple(range(first, first+lines*step, step))
            or byte_counts[start:start+lines] != (step,)*lines or first+step*lines > entry.file_size):
        raise ValueError(f"TIFF strips and native burst annotation disagree for {planned['cdse_uuid']}")
    return endian, lines, width, base+first


def verify_samples(path, lines, samples, endian, mask_path):
    data = np.memmap(path, dtype=f"{endian}i2", shape=(lines, samples, 2), mode="r")
    with gzip.open(mask_path, "rt") as handle:
        valid = np.loadtxt(handle, delimiter=",", skiprows=1, dtype=np.int32)
    if valid.shape != (lines, 3) or not np.array_equal(valid[:, 0], np.arange(lines)):
        raise ValueError(f"Mask does not match the recovered burst {path}")
    valid_rows = valid[valid[:, 1] >= 0]
    checks = valid_rows[np.linspace(0, len(valid_rows)-1, 64, dtype=int)]
    tile = np.concatenate([data[row, np.linspace(lo, hi, 64, dtype=int)] for row, lo, hi in checks])
    if len(tile) != 4096 or np.mean(np.any(tile != 0, axis=1)) < .9:
        raise ValueError(f"Valid radar samples look empty in {path}")
    return int(np.count_nonzero(np.any(tile != 0, axis=1)))


def collect(job):
    parent, annotation, planned = job
    base_url = json.loads(parent["ceda_attempts_json"])[-1][0]
    url = base_url+".zip?download=1"
    identifier = annotation["cdse_uuid"]
    with RangeZip(url) as remote, ZipFile(remote) as z:
        name = f"{parent['product_name']}/{annotation['measurement_safe_path']}"
        entry = z.getinfo(name)
        if entry.file_size != int(annotation["native_measurement_bytes"]):
            raise ValueError(f"TIFF size changed for {name}")
        with z.open(f"{parent['product_name']}/manifest.safe") as handle:
            manifest_sha256 = hashlib.sha256(handle.read()).hexdigest()
        if manifest_sha256 != parent["manifest_sha256"]:
            raise ValueError(f"Full ZIP and independently preserved manifest disagree for {name}")
        base = member_offset(remote, entry)
        endian, lines, samples, start = native_layout(remote, base, entry, annotation, planned)
        size = lines*samples*4
        if size != int(planned["expected_complex_sample_bytes"]):
            raise ValueError(f"Radar burst byte count differs for {identifier}")
        STORE.mkdir(parents=True, exist_ok=True)
        final = STORE / f"{identifier}.iq16"
        staged = final.with_suffix(".iq16.part")
        digest = hashlib.sha256()
        try:
            with staged.open("wb") as output:
                for offset in range(0, size, CHUNK_BYTES):
                    part = remote.slice(start+offset, min(CHUNK_BYTES, size-offset))
                    digest.update(part)
                    output.write(part)
            with staged.open("rb") as local:
                local_hash = hashlib.file_digest(local, "sha256").hexdigest()
            if staged.stat().st_size != size or digest.hexdigest() != local_hash:
                raise OSError(f"Complex burst failed local readback verification: {identifier}")
            nonzero = verify_samples(staged, lines, samples, endian, ROOT / annotation["valid_mask_gzip"])
            os.replace(staged, final)
        finally:
            staged.unlink(missing_ok=True)
        return dict(cdse_uuid=identifier, product_id=parent["product_id"], product_name=parent["product_name"],
            scope=annotation["scope"], swath=annotation["swath"], polarization=annotation["polarization"],
            source_url=url, source_zip_bytes=remote.size, source_zip_etag=remote.etag,
            source_zip_last_modified_http=remote.last_modified, measurement_safe_path=annotation["measurement_safe_path"],
            full_measurement_expected_md5=annotation["native_measurement_md5"], full_measurement_zip_crc32=f"{entry.CRC:08x}",
            manifest_sha256=manifest_sha256, first_byte_in_zip=start, last_byte_in_zip=start+size-1,
            bytes=size, lines=lines, samples_per_line=samples, complex_layout="row_major_sample_interleaved_i_q_int16",
            endian={"<": "little", ">": "big"}[endian], sha256=local_hash, source_radar_bytes_file=str(final.relative_to(ROOT)),
            valid_mask_gzip=annotation["valid_mask_gzip"], sampled_valid_iq_pixels=4096,
            sampled_valid_nonzero_iq_pixels=nonzero)


def main():
    parents = {r["product_id"]: r for r in read("ceda_s1_slc_annotations_parents")
               if r["full_zip_http_status"] == "206" and r["full_zip_signature"] == "ZIP"}
    plan = {r["cdse_uuid"]: r for r in read("cushing_slc_archive_bursts")}
    jobs = [(parents[r["product_id"]], r, plan[r["cdse_uuid"]])
            for r in read("ceda_s1_slc_annotations_bursts") if r["product_id"] in parents]
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in pool.map(collect, jobs):
            rows.append(row)
            print(f"{row['product_name']} {row['cdse_uuid']} {row['bytes']} verified bytes", flush=True)
    export(f"{STEM}_bursts", rows)
    available = {r["cdse_uuid"] for r in rows}
    by_week = defaultdict(set)
    for row in read("s1_holdout_gap_burst_tank_pairs"):
        if row["full_context_in_both_bursts"] == "True" and {row["previous_cdse_uuid"], row["current_cdse_uuid"]} <= available:
            by_week[row["week_ending"]].add(row["candidate_id"])
    summary = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), verified_native_bursts=len(rows),
        verified_native_bytes=sum(r["bytes"] for r in rows), holdout_weeks_with_both_native_bursts=len(by_week),
        planned_tank_weeks_with_both_native_bursts=sum(map(len, by_week.values())),
        holdout_weeks_with_both_native_bursts_by_date={week: len(tanks) for week, tanks in sorted(by_week.items())},
        interpretation="Bytes from original uncompressed TIFF strips are retained in TIFF byte order with separate line "
        "masks. The remote ZIP version, embedded manifest, all planned strip offsets and the local SHA-256 are checked. "
        "The full TIFF MD5 was not recomputed. Native image coverage and independently measured roof height are separate.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
