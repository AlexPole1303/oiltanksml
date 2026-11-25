"""Archive Cushing airport observations and join pre-reference reports to the study clock."""

import bisect
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pyarrow.parquet as pq

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, TLS, write

STEM = "cushing_ghcnh"
STORE = RAW / STEM
STATION = "USW00053993"
BASE = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year"
FIELDS = ("temperature", "dew_point_temperature", "visibility", "sky_condition", "ceiling_height", "wind_speed")
METAR_CLOUD = re.compile(r"(?:FEW|SCT|BKN|OVC)[0-9]{3}(?:CB|TCU)?|(?:CLR|SKC|NSC|NCD|CAVOK|VV(?:[0-9]{3}|///))")
REPORT_AT_KCUH = re.compile(r"\b(?:METAR|SPECI) KCUH (?P<clock>[0-9]{6})Z\b")


def download(year):
    file = f"GHCNh_{STATION}_{year}.parquet"
    url = f"{BASE}/{year}/parquet/{file}"
    path = STORE / file
    row = dict(year=year, station=STATION, source_url=url, http_status=200, archive_last_modified_utc=None,
               etag=None, bytes=None, sha256=None, local_file=None, observations=0, first_observed_at_utc=None,
               last_observed_at_utc=None)
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), context=TLS, timeout=45) as response:
                body = response.read(12 * 1024 * 1024 + 1)
                if response.status != 200 or len(body) != int(response.headers["Content-Length"]):
                    raise ValueError(f"Incomplete station file {url}")
                modified = response.headers.get("Last-Modified")
                row.update(archive_last_modified_utc=parsedate_to_datetime(modified).isoformat() if modified else None,
                           etag=response.headers.get("ETag"))
            break
        except HTTPError as error:
            if error.code == 404:
                return dict(row, http_status=404), []
            if attempt == 2 or error.code not in {408, 429, 500, 502, 503, 504}:
                raise
        except OSError:
            if attempt == 2:
                raise
    if not body.startswith(b"PAR1") or not body.endswith(b"PAR1"):
        raise ValueError(f"Expected NOAA parquet file at {url}")
    write(path, body)
    columns = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "REM"]
    columns += [name + suffix for name in FIELDS for suffix in ("", "_Quality_Code", "_Source_Code", "_Report_Type")]
    table = pq.read_table(path, columns=columns)
    observations = []
    for values in table.to_pylist():
        at = datetime.fromisoformat(values["DATE"]).replace(tzinfo=timezone.utc)
        if (values["STATION"] != STATION or at.year != year or abs(float(values["LATITUDE"]) - 35.95) > .01
                or abs(float(values["LONGITUDE"]) + 96.7731) > .01):
            raise ValueError(f"Unexpected NOAA station metadata at {url}")
        observations.append((at, values))
    times = [at for at, _ in observations]
    if not times or times != sorted(set(times)):
        raise ValueError(f"Unordered or duplicated observations at {url}")
    row.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest(), local_file=str(path.relative_to(ROOT)),
               observations=len(times), first_observed_at_utc=times[0].isoformat(), last_observed_at_utc=times[-1].isoformat())
    return row, observations


def cloud_report(observation, at):
    raw = observation.get("REM")
    if not raw or at is None:
        return None, None
    start = REPORT_AT_KCUH.search(raw)
    if not start or start["clock"] != at.strftime("%d%H%M"):
        return raw, None
    body = raw[start.end():].split(" RMK", 1)[0].rstrip("=")
    tokens = [token for token in body.split() if METAR_CLOUD.fullmatch(token)]
    return raw, " ".join(tokens) or None


def main():
    refs = read("cushing_hrrr_availability_links")
    years = [datetime.fromisoformat(r["reference_at_utc"]).year for r in refs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, range(min(years), max(years) + 1)))
    manifest = [row for row, _ in results]
    observations = [pair for _, values in results for pair in values]
    times = [at for at, _ in observations]
    rows = []
    for ref in refs:
        stamp = datetime.fromisoformat(ref["reference_at_utc"])
        i = bisect.bisect_right(times, stamp) - 1
        age = (stamp - times[i]).total_seconds() if i >= 0 else None
        at, obs = observations[i] if age is not None and age <= 1800 else (None, {})
        raw, clouds = cloud_report(obs, at)
        rows.append({k: ref[k] for k in ("study_phase", "week_ending", "reference_kind", "endpoint", "source_id",
                     "reference_at_utc", "forecast_cutoff_utc", "model_valid_hour_utc")} |
                    dict(station=STATION, observed_at_utc=at.isoformat() if at else None,
                         observation_age_seconds=age if at else None,
                         raw_report=raw, metar_cloud_groups=clouds,
                         **{name + suffix: obs.get(name + suffix) for name in FIELDS
                            for suffix in ("", "_Quality_Code", "_Source_Code", "_Report_Type")}))
    export(f"{STEM}_sources", manifest)
    export(f"{STEM}_links", rows)
    matched = [r for r in rows if r["observed_at_utc"]]
    report = dict(station=STATION, station_name="CUSHING MUNI AP (KCUH)", source_files=len(manifest),
        available_source_files=sum(r["http_status"]==200 for r in manifest), station_observations=len(times),
        reference_links=len(rows), reference_links_with_report=len(matched),
        reference_links_with_metar_cloud_groups=sum(bool(r["metar_cloud_groups"]) for r in matched),
        matched_by_source=Counter(r["reference_kind"] for r in matched),
        observed_fields=Counter({name: sum(r[name] is not None for r in matched) for name in FIELDS}),
        interpretation="Nearest station report at or before reference, at most 30 minutes old. Original NOAA field values, "
        "source codes and quality flags are preserved. This is a retrospective comparison source. Observation time is "
        "not evidence of public availability of this harmonized file before the study cutoff. Station observations "
        "are point measurements and can overlap with input data assimilated by the weather model.")
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
