"""Cross-check Cushing weather fields with original NOAA GRIB2 bytes and clocks."""

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.request import Request, urlopen

import numpy as np
from rasterio.io import MemoryFile
from rasterio.windows import Window

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_cushing_hrrr import file_hash, time, utc
from collect_public import RAW, ROOT, TLS, get, write

STEM = "cushing_hrrr_source_audit"
STORE = RAW / STEM
NOAA = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
FIELDS = {
    ("TMP", "2 m above ground"): ("temperature_2m", .126),
    ("TCDC", "entire atmosphere"): ("total_cloud_cover_atmosphere", .126),
    ("DSWRF", "surface"): ("downward_short_wave_radiation_flux_surface", 4.01),
}
RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+)")


def range_read(url, begin, stop):
    request = Request(url, headers={"Range": f"bytes={begin}-{stop-1}", "Accept-Encoding": "identity"})
    with urlopen(request, context=TLS, timeout=35) as response:
        body = response.read(stop-begin+1)
        content = RANGE.fullmatch(response.headers.get("Content-Range", ""))
        etag = response.headers["ETag"]
        modified = parsedate_to_datetime(response.headers["Last-Modified"])
        if (response.status != 206 or not content or tuple(map(int, content.groups()[:2])) != (begin, stop-1)
                or len(body) != stop-begin or not body.startswith(b"GRIB") or not body.endswith(b"7777")
                or int.from_bytes(body[8:16], "big") != len(body) or not etag):
            raise ValueError(f"Invalid exact GRIB2 range at {url}")
    return body, etag, modified


def noaa_asset(hour):
    return f"{NOAA}/hrrr.{hour:%Y%m%d}/conus/hrrr.t{hour:%H}z.wrfsfcf00.grib2"


def audit(hour, source_rows, source_cols, field_arrays, local_index, xyz):
    stamp = datetime.fromisoformat(utc(hour))
    url = noaa_asset(stamp)
    raw = get(url+".idx", STORE / f"{stamp:%Y%m%d_%H}.idx")
    index = [line.split(":") for line in raw.decode().splitlines()]
    chosen = []
    for i, record in enumerate(index[:-1]):
        key = tuple(record[3:5])
        if key in FIELDS and record[5] == "anl":
            chosen.append((key, int(record[1]), int(index[i+1][1])))
    if len(chosen) != len(FIELDS) or len({k for k,_,_ in chosen}) != len(FIELDS):
        raise ValueError(f"Missing NOAA native variable in {url}")
    results = []
    for key, start, stop in chosen:
        name, tolerance = FIELDS[key]
        body, etag, modified = range_read(url, start, stop)
        path = STORE / f"{stamp:%Y%m%d_%H}_{key[0]}.grib2"
        write(path, body)
        with MemoryFile(body) as memory, memory.open() as ds:
            tags = ds.tags(1)
            if (ds.count != 1 or tags["GRIB_ELEMENT"] != key[0] or int(tags["GRIB_REF_TIME"]) != hour
                    or int(tags["GRIB_VALID_TIME"]) != hour):
                raise ValueError(f"GRIB payload identifies a different NOAA field or time for {url}")
            left, top = int(source_cols.min()), int(source_rows.min())
            win = Window(left, top, int(source_cols.max())-left+1, int(source_rows.max())-top+1)
            data = ds.read(1, window=win, masked=True)
            values = np.asarray(data[source_rows-top,source_cols-left])
            ref_x, ref_y = ds.transform * (source_cols+.5, source_rows+.5)
            if not (np.allclose(ref_x, xyz[0], atol=.001, rtol=0) and np.allclose(ref_y, xyz[1], atol=.001, rtol=0)):
                raise ValueError(f"NOAA GRIB grid disagrees with pinned analysis for {url}")
        local = field_arrays[name][local_index]
        if not (np.isfinite(values).all() and np.isfinite(local).all()):
            raise ValueError(f"Missing sample in cross-checked hour {stamp}/{name}")
        difference = float(np.max(np.abs(local-values)))
        if difference > tolerance:
            raise ValueError(f"Subset differs from NOAA source by {difference} for {name} at {stamp}")
        results.append(dict(model_valid_hour_utc=stamp, variable=name, source_url=url, source_etag=etag,
            source_last_modified_utc=modified, byte_start=start, byte_end_exclusive=stop,
            source_index_sha256=hashlib.sha256(raw).hexdigest(), local_native_grib2=str(path.relative_to(ROOT)),
            native_grib2_sha256=file_hash(path), verified_cells=len(values), max_absolute_difference=difference,
            validation_tolerance=tolerance))
    return results


def main():
    weather = json.loads((DERIVED / "cushing_hrrr_summary.json").read_bytes())
    fields = {r["variable"]: r for r in read("cushing_hrrr_fields")}
    matrices = {name: np.load(ROOT / fields[name]["local_file"], mmap_mode="r", allow_pickle=False)
                for name, _ in FIELDS.values()}
    optical = read("ceda_s1_concurrent_optical_controls")
    clocks = {time(r[k])//3600*3600 for r in optical for k in ("radar_capture_utc", "optical_capture_start_utc")}
    clocks.update(time(x) for x in ("2019-08-12T18:00:00Z", "2025-03-21T17:00:00Z", "2026-10-24T00:00:00Z"))
    with np.load(ROOT / weather["coordinates_file"], allow_pickle=False) as grid:
        epochs = grid["time_epoch_utc"]
        args = [(hour, grid["source_grid_row"], grid["source_grid_col"], matrices,
                 int(np.searchsorted(epochs, hour)), (grid["x_m"], grid["y_m"])) for hour in sorted(clocks)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [r for result in pool.map(lambda job: audit(*job), args) for r in result]
    export(STEM, rows)
    report = dict(hr_model_hours=len(clocks), native_fields=len(rows), cells_verified_per_field=weather["cells_in_region"],
        source_snapshot=weather["snapshot_id"],
        fields_with_archive_mark_after_model_hour=sum(row["model_valid_hour_utc"] < row["source_last_modified_utc"]
                                                      for row in rows),
        checked_variables=Counter(r["variable"] for r in rows),
        max_absolute_difference_by_variable={name: max(r["max_absolute_difference"] for r in rows if r["variable"]==name)
                                             for name in matrices},
        interpretation="Original NOAA GRIB2 byte ranges validate location, time and sampled field values. "
        "The source object Last-Modified is preserved for those hours only.")
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
