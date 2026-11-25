"""Tie exact NOAA HRRR object clocks to EIA endpoints and satellite acquisitions."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from audit_cushing_hrrr_source import NOAA
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_cushing_hrrr import time, utc
from collect_public import RAW, ROOT, TLS, write

STEM = "cushing_hrrr_availability"
STORE = RAW / STEM
HOUR = 3600
SELECTED = ("temperature_2m", "total_cloud_cover_atmosphere", "downward_short_wave_radiation_flux_surface",
            "precipitation_surface")


def ts(value):
    return datetime.fromisoformat(value).timestamp()


def floor_hour(value):
    return int(ts(value))//HOUR*HOUR


def clock_requests():
    requests = set()
    with closing(read_only(SOURCE)) as db:
        for phase, week, side, event, cutoff in db.execute("""
            SELECT study_phase,week_ending,endpoint,window_ends_by_utc,inputs_available_before_utc
            FROM eia_satellite_window"""):
            requests.add((phase, week, "eia_endpoint", side, "", event, cutoff))
            if side=="current":
                # The last hour strictly before the forecast cutoff is eligible for archive inspection.
                requests.add((phase, week, "eia_cutoff_prior_hour", "", "", utc(floor_hour(cutoff)-HOUR), cutoff))
    lookups = {r["week_ending"]: r["forecast_cutoff_utc"] for r in read("cushing_holdout_release_index")}
    for r in read("weekly_shadow_endpoint_manifest"):
        requests.add((r["study_phase"], r["week_ending"], "sentinel2_endpoint_manifest", r["endpoint"],
                      r["version_id"], r["captured_at_utc"], r["inputs_available_before_utc"]))
    for r in read("holdout_optical_tank_weeks"):
        for side in ("earlier", "later"):
            capture = r[f"{side}_capture_at_utc"]
            if capture:
                requests.add(("holdout", r["week_ending"], "sentinel2_b04", side, r[f"{side}_sentinel2_version"],
                              capture, lookups[r["week_ending"]]))
    for table, sensor, key_field, previous_field, current_field in (
            ("ceda_s1_slc_regional_weekly_pairs", "sentinel1_slc", "cdse_uuid", "previous_vv", "current_vv"),
            ("cushing_s1_rtc_weekly_pairs", "sentinel1_rtc", "rtc_id", "previous_rtc_id", "current_rtc_id"),
            ("s1_grd_holdout_gap_tank_pairs", "sentinel1_grd", "product_id", "previous_product_id", "current_product_id")):
        source = {"sentinel1_slc": "ceda_s1_slc_regional_assets", "sentinel1_rtc": "cushing_s1_rtc_scenes",
                  "sentinel1_grd": "s1_cushing_catalog"}[sensor]
        clocks = {r[key_field]: r["capture_start_utc"] for r in read(source)}
        for r in read(table):
            for side, field in (("previous", previous_field), ("current", current_field)):
                identity = r[field]
                requests.add(("holdout", r["week_ending"], sensor, side, identity, clocks[identity],
                              lookups[r["week_ending"]]))
    for phase, week, source, side, identity, event, cutoff in sorted(requests):
        if ts(event) >= ts(cutoff):
            raise ValueError(f"Source timestamp does not precede forecast cutoff for {source}/{identity}/{week}")
        yield dict(study_phase=phase, week_ending=week, reference_kind=source, endpoint=side, source_id=identity,
                   reference_at_utc=event, forecast_cutoff_utc=cutoff, model_valid_hour_utc=utc(floor_hour(event)))


def source_url(valid, lead):
    initialized = datetime.fromisoformat(utc(valid-HOUR*lead))
    return f"{NOAA}/hrrr.{initialized:%Y%m%d}/conus/hrrr.t{initialized:%H}z.wrfsfcf{lead:02}.grib2"


def inspect(job):
    valid, lead = job
    url = source_url(valid, lead)
    base = dict(model_valid_hour_utc=utc(valid), initialized_at_utc=utc(valid-HOUR*lead),
                forecast_hour=lead, source_url=url)
    for attempt in range(3):
        try:
            with urlopen(Request(url, method="HEAD"), context=TLS, timeout=18) as response:
                etag = response.headers["ETag"]
                size = int(response.headers["Content-Length"])
                modified = parsedate_to_datetime(response.headers["Last-Modified"])
                if response.status != 200 or not etag or size <= 0 or modified.timestamp() < valid-HOUR*lead:
                    raise ValueError(f"Invalid NOAA object header at {url}")
                return dict(base, http_status=200, source_etag=etag, source_bytes=size,
                            archive_last_modified_utc=modified.isoformat(),
                            source_version_id=response.headers.get("x-amz-version-id", ""))
        except HTTPError as exc:
            if exc.code == 404:
                return dict(base, http_status=404, source_etag="", source_bytes=None,
                            archive_last_modified_utc="", source_version_id="")
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                raise
        except OSError:
            if attempt == 2:
                raise


def attach(requests, objects, fields, grid):
    start = int(grid["time_epoch_utc"][0])
    cells = {(int(r["source_grid_row"]), int(r["source_grid_col"])) for r in read("cushing_hrrr_tank_grid")}
    col, row = grid["source_grid_col"], grid["source_grid_row"]
    distinct = np.array([i for i, (y,x) in enumerate(zip(row, col, strict=True)) if (int(y), int(x)) in cells])
    summaries = {}
    for r in requests:
        valid = time(r["model_valid_hour_utc"])
        cutoff = ts(r["forecast_cutoff_utc"])
        event = ts(r["reference_at_utc"])
        index = (valid-start)//HOUR
        if index not in range(len(grid["time_epoch_utc"])) or grid["time_epoch_utc"][index] != valid:
            raise ValueError(f"Unknown HRRR snapshot hour {r['model_valid_hour_utc']}")
        if valid not in summaries:
            summaries[valid] = {}
            for name, array in fields.items():
                values = array[index, distinct]
                good = values[np.isfinite(values)]
                summaries[valid][f"{name}_finite_cells"] = len(good)
                summaries[valid][f"{name}_median"] = float(np.median(good)) if len(good) else None
        checks = {}
        for lead, label in ((0,"f00_analysis"), (1,"f01_previous_run")):
            obj = objects[valid, lead]
            modified = ts(obj["archive_last_modified_utc"]) if obj["archive_last_modified_utc"] else None
            checks.update({f"{label}_archive_modified_utc":obj["archive_last_modified_utc"],
                           f"{label}_before_cutoff":modified is not None and modified < cutoff,
                           f"{label}_by_reference":modified is not None and modified <= event,
                           f"{label}_http_status":obj["http_status"]})
        yield dict(r, model_hour_index=index, **checks, **summaries[valid])


def phase_report(rows):
    phases = defaultdict(list)
    for r in rows:
        phases[r["study_phase"]].append(r)
    return {name: dict(release_weeks=len({r["week_ending"] for r in values}),
        object_matched_links=len(values), f00_links_before_cutoff=sum(r["f00_analysis_before_cutoff"] for r in values),
        f01_links_before_cutoff=sum(r["f01_previous_run_before_cutoff"] for r in values),
        both_run_objects_before_cutoff=sum(r["f00_analysis_before_cutoff"] and r["f01_previous_run_before_cutoff"] for r in values),
        cutoff_prior_hour_f00_available_weeks=sum(r["reference_kind"]=="eia_cutoff_prior_hour" and
                                                   r["f00_analysis_before_cutoff"] for r in values)) for name, values in phases.items()}


def main():
    weather = json.loads((DERIVED / "cushing_hrrr_summary.json").read_bytes())
    requests = list(clock_requests())
    hours = sorted({time(r["model_valid_hour_utc"]) for r in requests})
    jobs = [(hour, lead) for hour in hours for lead in (0, 1)]
    print(f"{len(hours)} distinct model hours, {len(jobs)} exact NOAA object headers", flush=True)
    objects = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        pending = {pool.submit(inspect, key): key for key in jobs}
        for n, future in enumerate(as_completed(pending), 1):
            objects[pending[future]] = future.result()
            if n%400==0:
                print(f"NOAA object headers {n}/{len(jobs)}", flush=True)
    rows = [objects[key] for key in jobs]
    snapshot = json.dumps(rows, indent=2)+"\n"
    write(STORE / "noaa_object_headers.json", snapshot.encode())
    fields = {r["variable"]: np.load(ROOT / r["local_file"], mmap_mode="r", allow_pickle=False)
              for r in read("cushing_hrrr_fields") if r["variable"] in SELECTED}
    if fields.keys()!=set(SELECTED):
        raise ValueError("Some required fields from the pinned HRRR analysis are missing")
    with np.load(ROOT / weather["coordinates_file"], allow_pickle=False) as grid:
        links = list(attach(requests, objects, fields, grid))
    export(f"{STEM}_objects", rows)
    export(f"{STEM}_links", links)
    early = [r for r in links if r["study_phase"]=="holdout" and r["reference_kind"]=="eia_endpoint"]
    weeks = defaultdict(list)
    for r in early:
        weeks[r["week_ending"]].append(r)
    if any({r["endpoint"] for r in group}!={"previous","current"} or len(group)!=2 for group in weeks.values()):
        raise ValueError("Incomplete holdout weekly endpoints")
    report = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), snapshot_id=weather["snapshot_id"],
        model_valid_hours=len(hours), queried_noaa_objects=len(jobs), object_http_status=Counter(r["http_status"] for r in rows),
        unique_links=len(links), by_study_phase=phase_report(links),
        holdout_endpoint_weeks=len(weeks),
        holdout_weeks_both_f00_before_cutoff=sum(all(r["f00_analysis_before_cutoff"] for r in group) for group in weeks.values()),
        holdout_weeks_both_f01_before_cutoff=sum(all(r["f01_previous_run_before_cutoff"] for r in group) for group in weeks.values()),
        interpretation="Object Last-Modified is evidence about the exact NOAA archive object seen in this audit. "
        "The meteorological values in the links table come from the separately pinned and rounded HRRR analysis snapshot. "
        "F01 labels the previous model run's forecast, not the F00 analysis. Object presence and timestamps do not prove "
        "per-field equivalence. No EIA outcomes or price values are used.")
    body = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", body.encode())
    print(body, end="", flush=True)


if __name__ == "__main__":
    main()
