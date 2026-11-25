"""Read exact operational HRRR messages for the study calendar."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
from rasterio.io import MemoryFile
from rasterio.windows import Window

from audit_cushing_hrrr_source import RANGE
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_cushing_hrrr import time
from collect_public import RAW, ROOT, TLS, get, write

STEM = "cushing_hrrr_operational"
STORE = RAW / STEM
WANTED = {
    0: {
        ("TMP", "2 m above ground", "anl"): ("temperature_2m_c", "TMP"),
        ("TCDC", "entire atmosphere", "anl"): ("total_cloud_cover_pct", "TCDC"),
        ("DSWRF", "surface", "anl"): ("downward_shortwave_w_m2", "DSWRF"),
    },
    1: {("APCP", "surface", "0-1 hour acc fcst"): ("prior_run_precipitation_1h_mm", "APCP01")},
}


def packet(url, etag, modified, begin, stop):
    request = Request(url, headers={"Range": f"bytes={begin}-{stop-1}", "If-Match": etag, "Accept-Encoding": "identity"})
    for attempt in range(3):
        try:
            with urlopen(request, context=TLS, timeout=40) as response:
                content = RANGE.fullmatch(response.headers.get("Content-Range", ""))
                source_modified = parsedate_to_datetime(response.headers["Last-Modified"])
                if (response.status != 206 or not content or tuple(map(int, content.groups()[:2])) != (begin, stop-1)
                        or response.headers["ETag"] != etag or source_modified != modified):
                    raise ValueError(f"NOAA object changed after header audit at {url}")
                body = response.read(stop-begin+1)
                if (len(body) != stop-begin or not body.startswith(b"GRIB") or not body.endswith(b"7777")
                        or int.from_bytes(body[8:16], "big") != len(body)):
                    raise ValueError(f"Incorrect native GRIB2 message at {url}")
                return body
        except OSError as error:
            if attempt==2 or (isinstance(error, HTTPError) and error.code not in {408, 429, 500, 502, 503, 504}):
                raise


def collect(obj, grid, store=STORE):
    url = obj["source_url"]
    lead = int(obj["forecast_hour"])
    stamp = datetime.fromisoformat(obj["model_valid_hour_utc"])
    first = time(obj["initialized_at_utc"])
    valid = time(obj["model_valid_hour_utc"])
    modified = datetime.fromisoformat(obj["archive_last_modified_utc"])
    expected = WANTED[lead]
    index_path = store / "indexes" / f"{stamp:%Y%m%d_%H}_f{lead:02}.idx"
    try:
        raw = get(url+".idx", index_path, refresh=True)
    except HTTPError as error:
        if error.code==404:
            return [], ["missing_index"]
        raise
    lines = [r.split(":") for r in raw.decode().splitlines()]
    row, col, x, y = (grid[k] for k in ("source_grid_row", "source_grid_col", "x_m", "y_m"))
    first_row, first_col = int(row.min()), int(col.min())
    region = Window(first_col, first_row, int(col.max())-first_col+1, int(row.max())-first_row+1)
    values = []
    missing = set(expected)
    for i, fields in enumerate(lines[:-1]):
        key = tuple(fields[3:6])
        if key not in expected:
            continue
        if key not in missing:
            raise ValueError(f"Duplicate expected native field at {url}")
        missing.remove(key)
        name, element = expected[key]
        begin, stop = int(fields[1]), int(lines[i+1][1])
        if not 0 <= begin < stop <= int(obj["source_bytes"]):
            raise ValueError(f"GRIB2 index extends outside source at {url}")
        body = packet(url, obj["source_etag"], modified, begin, stop)
        with MemoryFile(body) as buffer, buffer.open() as raster:
            tags = raster.tags(1)
            if (raster.count != 1 or tags["GRIB_ELEMENT"] != element or int(tags["GRIB_REF_TIME"]) != first
                    or int(tags["GRIB_VALID_TIME"]) != valid):
                raise ValueError(f"Wrong NOAA model time or field at {url}")
            rx, ry = raster.transform*(col+.5,row+.5)
            if not (np.allclose(rx,x,rtol=0,atol=.001) and np.allclose(ry,y,rtol=0,atol=.001)):
                raise ValueError(f"Native GRIB2 grid disagrees with study coordinates at {url}")
            result = raster.read(1, window=region, masked=True)[row-first_row, col-first_col].filled(np.nan)
        if not np.isfinite(result).all():
            raise ValueError(f"Missing native NOAA cell values for {name} at {url}")
        values.append((dict(model_valid_hour_utc=obj["model_valid_hour_utc"], forecast_hour=lead,
            initialized_at_utc=obj["initialized_at_utc"], archive_last_modified_utc=modified,
            variable=name, source_units=tags["GRIB_UNIT"], source_url=url, source_etag=obj["source_etag"],
            source_index_file=str(index_path.relative_to(ROOT)), source_index_sha256=hashlib.sha256(raw).hexdigest(),
            byte_start=begin, byte_end_exclusive=stop, original_grib2_sha256=hashlib.sha256(body).hexdigest()), result))
    return values, [expected[key][0] for key in sorted(missing)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="read development and validation objects with timely archive clocks")
    training = parser.parse_args().training
    stem = f"{STEM}_training" if training else STEM
    store = RAW / stem
    phases = {"development", "validation"} if training else {"holdout"}
    links = [r for r in read("cushing_hrrr_availability_links") if r["study_phase"] in phases]
    hours = sorted({r["model_valid_hour_utc"] for r in links})
    objects = [r for r in read("cushing_hrrr_availability_objects") if r["model_valid_hour_utc"] in hours]
    if len(objects)!=2*len(hours):
        raise ValueError("Missing NOAA object clock audit")
    eligible = {(r["model_valid_hour_utc"], str(lead)) for r in links
                for lead, prefix in enumerate(("f00_analysis", "f01_previous_run"))
                if r[f"{prefix}_before_cutoff"] == "True"}
    selected = [r for r in objects if r["http_status"] == "200" and
                (not training or (r["model_valid_hour_utc"], r["forecast_hour"]) in eligible)]
    weather = json.loads((DERIVED / "cushing_hrrr_summary.json").read_bytes())
    with np.load(ROOT / weather["coordinates_file"], allow_pickle=False) as saved:
        grid = {name: saved[name] for name in ("source_grid_row", "source_grid_col", "x_m", "y_m")}
    by_key, issues = {}, []
    print(f"Reading {len(selected)} native NOAA objects across {len(hours)} requested hours", flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs = {pool.submit(collect, r, grid, store): r for r in selected}
        for n, future in enumerate(as_completed(jobs), 1):
            obj = jobs[future]
            try:
                packets, missing = future.result()
            except (OSError, ValueError) as error:
                packets, missing = [], [f"{type(error).__name__}: {error}"]
            if missing:
                issues.append(dict(source_url=obj["source_url"], missing=missing))
            for meta, values in packets:
                by_key[meta["model_valid_hour_utc"],meta["variable"]] = (meta, values)
            if n%100==0:
                print(f"Native NOAA source objects {n}/{len(jobs)}", flush=True)
    keys = sorted(by_key)
    metadata = [by_key[k][0] | {"array_row": i} for i, k in enumerate(keys)]
    matrix = np.stack([by_key[k][1] for k in keys])
    path = store / "native_cell_values.npz"
    store.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=matrix, **grid)
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["values"],matrix):
            raise ValueError("Exact NOAA regional values failed readback")
    export(f"{stem}_messages", metadata)
    positions = {(r["model_valid_hour_utc"],r["variable"]): r for r in metadata}
    variables = tuple(name for fields in WANTED.values() for name,_ in fields.values())
    linked = []
    for r in links:
        hour = r["model_valid_hour_utc"]
        cutoff = datetime.fromisoformat(r["forecast_cutoff_utc"])
        ref = datetime.fromisoformat(r["reference_at_utc"])
        detail = {}
        for field in variables:
            row = positions.get((hour,field))
            modified = row["archive_last_modified_utc"] if row else None
            detail.update({f"{field}_array_row":row["array_row"] if row else None,
                           f"{field}_precutoff":modified is not None and modified<cutoff,
                           f"{field}_by_reference":modified is not None and modified<=ref})
        linked.append({k:r[k] for k in ("study_phase", "week_ending", "reference_kind", "endpoint", "source_id",
                       "reference_at_utc", "forecast_cutoff_utc", "model_valid_hour_utc")} | detail)
    export(f"{stem}_links", linked)
    weeks = defaultdict(list)
    for row in linked:
        if row["reference_kind"]=="eia_endpoint":
            weeks[row["week_ending"]].append(row)
    if any(len(group)!=2 for group in weeks.values()):
        raise ValueError("Incomplete EIA endpoint weather rows")
    report = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), holdout_hours=len(hours),
        object_status=Counter(r["http_status"] for r in objects), native_messages=len(metadata),
        native_cells=matrix.shape[1], messages_by_variable=Counter(r["variable"] for r in metadata),
        linked_rows=len(linked), holdout_endpoint_weeks=len(weeks),
        weeks_with_all_four_fields_at_both_endpoints_before_cutoff=sum(
            all(r[f"{name}_precutoff"] for r in group for name in variables) for group in weeks.values()),
        per_source_all_fields_before_cutoff=Counter(r["reference_kind"] for r in linked
                                                   if all(r[f"{name}_precutoff"] for name in variables)),
        incomplete_readable_objects=len(issues), incomplete_objects=issues,
        native_cell_values_file=str(path.relative_to(ROOT)), native_cell_values_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        interpretation="Exact NOAA operational F00 analysis for temperature, cloud and shortwave, and previous-run F01 "
        "one-hour precipitation forecast. Rainfall is in water-equivalent mm. As-of comparisons refer to the current "
        "archive version and preserve separate flags for overpass time and EIA forecast cutoff.")
    if training:
        report["requested_hours"] = report.pop("holdout_hours")
        report["endpoint_weeks"] = report.pop("holdout_endpoint_weeks")
    report["selected_source_objects"] = len(selected)
    report["study_phases"] = sorted(phases)
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
