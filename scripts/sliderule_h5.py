"""Read typed HDF5 array slices from the public SlideRule h5p endpoint."""

import hashlib
import json
import struct
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from collect_public import ROOT, TLS, get, write

BASE = "https://sliderule.slideruleearth.io/source"
DTYPES = {0: "i1", 1: "<i2", 2: "<i4", 3: "<i8", 4: "u1", 5: "<u2", 6: "<u4", 7: "<u8", 9: "<f4", 10: "<f8"}
MAX_BYTES = 40 * 1024**2


def records(body):
    pos = 0
    while pos < len(body):
        if len(body) - pos < 8:
            raise ValueError("Incomplete SlideRule record header")
        version, name_size, data_size = struct.unpack_from(">HHI", body, pos)
        name_start = pos + 8
        data_start = name_start + name_size
        end = data_start + data_size
        if version != 2 or not name_size or end > len(body) or body[data_start - 1] != 0:
            raise ValueError("Invalid SlideRule record bounds")
        yield body[name_start:data_start-1].decode("ascii"), memoryview(body)[data_start:end]
        pos = end


def message(data):
    code, level = struct.unpack_from("<iI", data)
    return code, level, bytes(data[8:]).split(b"\0", 1)[0].decode("utf-8", errors="replace")


def parse(body):
    arrays, issues = {}, []
    for name, data in records(body):
        if name == "h5file" and len(data) >= 116:
            key = bytes(data[:104]).split(b"\0", 1)[0].decode("ascii")
            dtype, count, size = struct.unpack_from("<III", data, 104)
            if not size:
                issues.append(f"Empty HDF5 field {key}")
            else:
                kind = np.dtype(DTYPES[dtype])
                if key in arrays or size != len(data)-116 or size != count*kind.itemsize:
                    raise ValueError(f"Invalid typed array {key}")
                arrays[key] = np.frombuffer(data, kind, count=count, offset=116)
        elif name == "exceptrec" and len(data) >= 8:
            code, level, detail = message(data)
            if code < 0 or level >= 2:
                issues.append(detail)
        else:
            raise ValueError(f"Unexpected SlideRule record {name}")
    return arrays, issues


def version(directory):
    raw = get(f"{BASE}/version", directory / "sliderule_version.json")
    return json.loads(raw)["server"]["version"]


def request_packet(endpoint, args, path, *, cache=True):
    payload = json.dumps(args, sort_keys=True).encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = path / f"{digest}.bin"
    if path.is_file():
        body = path.read_bytes()
    else:
        request = Request(f"{BASE}/{endpoint}", data=payload, method="POST", headers={"Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urlopen(request, context=TLS, timeout=110) as response:
                    if response.status != 200 or response.headers.get_content_type() != "application/octet-stream":
                        raise ValueError(f"SlideRule returned {response.status}/{response.headers.get_content_type()} for {endpoint}")
                    body = response.read(MAX_BYTES+1)
                if len(body) > MAX_BYTES:
                    raise ValueError(f"SlideRule response exceeded limit for {endpoint}")
                break
            except OSError as error:
                if attempt == 3 or (isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}):
                    raise
                sleep(2**attempt)
    # Validate framing before caching the transport payload.
    for _ in records(body):
        pass
    if not path.is_file() and cache:
        write(path.with_suffix(".json"), payload)
        write(path, body)
    receipt = dict(request_sha256=digest, response_sha256=hashlib.sha256(body).hexdigest(),
                   response_file=str(path.relative_to(ROOT)) if path.is_file() else None)
    return body, receipt


def h5(directory, asset, resource, fields, *, cache=True):
    if not fields or len({r[0] for r in fields}) != len(fields):
        raise ValueError("HDF5 field paths must be distinct in a single request")
    args = {"asset": asset, "resource": resource, "datasets": [
        dict(dataset=path, startrow=start, numrows=count, col=col) for path, start, count, col in fields]}
    body, receipt = request_packet("h5p", args, directory, cache=cache)
    arrays, issues = parse(body)
    for field, _, count, _ in fields:
        if field in arrays and count >= 0 and len(arrays[field]) != count:
            raise ValueError(f"Wrong row count for {resource}/{field}")
    return arrays, dict(receipt, resource=resource, asset=asset, issues=issues)
