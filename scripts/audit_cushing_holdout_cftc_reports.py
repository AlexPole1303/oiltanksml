"""Check the WTI COT factors against the original individual CFTC reports."""
import csv
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import RAW, ROOT, TLS, write

STEM = "cushing_holdout_cftc_audit"
DIRECTORY = RAW / "factors/cftc_weekly_petroleum"
BASE = "https://www.cftc.gov/sites/default/files/files/dea/cotarchives"
EASTERN = ZoneInfo("America/New_York")


def original(report_date):
    observed = date.fromisoformat(report_date)
    filename = f"petroleum_lf{observed:%m%d%y}.htm"
    path = DIRECTORY / str(observed.year) / filename
    meta = path.with_suffix(".json")
    url = f"{BASE}/{observed.year}/futures/{filename}"
    if meta.is_file() and path.is_file():
        metadata = json.loads(meta.read_text())
        body = path.read_bytes()
    else:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=25, context=TLS) as response:
            body = response.read()
            metadata = dict(source_url=response.url, http_last_modified=response.headers.get("Last-Modified"),
                            retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                            sha256=hashlib.sha256(body).hexdigest())
        write(path, body)
        write(meta, (json.dumps(metadata, indent=2) + "\n").encode())
    if metadata["sha256"] != hashlib.sha256(body).hexdigest() or metadata["source_url"] != url:
        raise ValueError(f"Source identity mismatch for {report_date}")
    return body.decode("latin-1"), str(path.relative_to(ROOT)), metadata


def positions(text, report_date):
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if re.search(r"\bCode-067651\s*$", line)]
    if len(matches) != 1:
        raise ValueError(f"Missing or repeated WTI market code on {report_date}")
    line = matches[0]
    heading = lines[line + 1].strip()
    prefix = "Disaggregated Commitments of Traders - Futures Only, "
    if not heading.startswith(prefix) or datetime.strptime(heading.removeprefix(prefix), "%B %d, %Y").date().isoformat() != report_date:
        raise ValueError(f"Incorrect CFTC report date or type on {report_date}")
    totals = [i for i in range(line + 2, min(line + 16, len(lines))) if re.match(r"^All\s*:", lines[i])]
    if len(totals) != 1:
        raise ValueError(f"Missing or repeated WTI totals on {report_date}")
    row = totals[0]
    fields = tuple(int(v.replace(",", "")) for v in re.findall(r"-?\d[\d,]*", lines[row]))
    if len(fields) != 14:
        raise ValueError(f"Wrong CFTC position column count on {report_date}")
    return (fields[0], fields[6], fields[7], fields[1], fields[2]), row + 1


def audit_report(record):
    observed, oi, long, short, producer_long, producer_short, zip_name, basis, scheduled, available = record
    result = dict(report_date=observed, annual_source=f"data/raw/factors/{zip_name}", original_report_path=None,
                  source_url=None, source_sha256=None, source_row=None, source_status=None,
                  http_last_modified_utc=None, http_last_modified_eastern_date=None,
                  report_footer_updated_date=None, saved_clock_basis=basis, documented_or_inferred_date=scheduled,
                  saved_cftc_available_at_utc=available)
    try:
        content, path, metadata = original(observed)
    except (URLError, TimeoutError) as error:
        return result | dict(source_status=f"download_error_{getattr(error, 'code', 'connection')}")
    weekly, line = positions(content, observed)
    footer = re.findall(r">\s*Updated ([A-Za-z]+ \d{1,2}, \d{4})\s*</p>", content)
    if len(footer) != 1:
        raise ValueError(f"Missing or repeated CFTC report update date on {observed}")
    updated = datetime.strptime(footer[0], "%B %d, %Y").date().isoformat()
    if updated <= observed:
        raise ValueError(f"CFTC report updated before observation on {observed}")
    expected = oi, long, short, producer_long, producer_short
    status = "values_match" if weekly == expected else "value_disagreement"
    modified = parsedate_to_datetime(metadata["http_last_modified"]) if metadata["http_last_modified"] else None
    return result | dict(original_report_path=path, source_url=metadata["source_url"], source_sha256=metadata["sha256"],
                         source_row=line, source_status=status, report_footer_updated_date=updated,
                         http_last_modified_utc=modified.astimezone(timezone.utc).isoformat() if modified else None,
                         http_last_modified_eastern_date=modified.astimezone(EASTERN).date().isoformat() if modified else None)


def main():
    with (DERIVED / "cushing_holdout_release_index.csv").open(newline="") as handle:
        indexes = list(csv.DictReader(handle))
    observed = {r["cftc_report_date"] for r in indexes}
    with closing(read_only(SOURCE)) as db:
        records = [r for r in db.execute("""SELECT p.report_date,p.open_interest,p.money_manager_long,p.money_manager_short,
            p.producer_long,p.producer_short,p.source_file,c.timing_basis,c.documented_or_inferred_release_date,
            c.conservative_research_from_utc
            FROM cftc_wti p JOIN cftc_release_clock c USING(report_date) ORDER BY p.report_date""") if r[0] in observed]
    if {r[0] for r in records} != observed or not records:
        raise ValueError("CFTC audit does not cover the holdout report dates")
    with ThreadPoolExecutor(max_workers=6) as pool:
        reports = list(pool.map(audit_report, records))
    by_date = {r["report_date"]: r for r in reports}
    releases = []
    for index in indexes:
        report = by_date[index["cftc_report_date"]]
        cutoff = datetime.fromisoformat(index["forecast_cutoff_utc"])
        saved_clock = datetime.fromisoformat(report["saved_cftc_available_at_utc"])
        modified = report["http_last_modified_utc"]
        updated = report["report_footer_updated_date"]
        if saved_clock > cutoff or saved_clock != datetime.fromisoformat(index["cftc_available_at_utc"]):
            raise ValueError(f"CFTC clock disagrees with the release index for {index['release_date']}")
        releases.append(dict(study_phase=index["study_phase"], release_date=index["release_date"],
                             week_ending=index["week_ending"], forecast_cutoff_utc=index["forecast_cutoff_utc"], **report,
                             report_footer_day_ended_by_cutoff=(
                                 datetime.combine(date.fromisoformat(updated) + timedelta(days=1),
                                                  datetime.min.time(), EASTERN) <= cutoff) if updated else None,
                             http_file_modified_by_cutoff=datetime.fromisoformat(modified) <= cutoff if modified else None))
    dated = [r for r in reports if r["documented_or_inferred_date"] and r["report_footer_updated_date"]]
    summary = dict(report_dates=len(reports), eia_releases=len(releases),
                   report_status=dict(Counter(r["source_status"] for r in reports)),
                   eia_release_status=dict(Counter(r["source_status"] for r in releases)),
                   original_files_modified_by_cutoff=sum(r["http_file_modified_by_cutoff"] is True for r in releases),
                   footer_date_before_cutoff=sum(r["report_footer_day_ended_by_cutoff"] is True for r in releases),
                   release_dates_with_late_footer_date=[r["release_date"] for r in releases
                                                        if r["report_footer_day_ended_by_cutoff"] is False],
                   compared_documented_or_scheduled_release_dates=len(dated),
                   footer_date_differs_from_documented_or_scheduled=[
                       r["report_date"] for r in dated if r["documented_or_inferred_date"] != r["report_footer_updated_date"]],
                   interpretation="The archive report's Updated date and HTTP Last-Modified describe different events. "
                                  "The Updated date has day-level precision. Neither is treated as an intraday release time.")
    export(f"{STEM}_reports", reports)
    export(f"{STEM}_releases", releases)
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
