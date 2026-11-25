"""Archive the public chart series and compare on a common EIA target."""

import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from urllib.parse import urljoin

import numpy as np

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write

STEM = "cushing_watch_benchmark"
STORE = RAW / "documents" / "cushing_watch"
PAGE = "https://cushingwatch.com/analysis"
EXAMPLE = "https://cushingwatch.com/weeks/2026-10-30"


def source_snapshot():
    record_path = STORE / "snapshot.json"
    if record_path.exists():
        record = json.loads(record_path.read_bytes())
        for source in record.values():
            content = (ROOT / source["path"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != source["sha256"]:
                raise ValueError(f"Snapshot bytes changed for {source['path']}")
        return record
    page = get(PAGE)
    match = re.search(rb'<script src="(/assets/cw-data\.js\?v=[a-f0-9]+)"', page)
    if not match:
        raise ValueError("Public Cushing Watch chart data script not found")
    url = urljoin(PAGE, match[1].decode())
    record = {}
    for name, path, source_url, content in (
        ("chart_page", "analysis.html", PAGE, page),
        ("chart_series", "cw-data.js", url, get(url)),
        ("dated_example", "report_2026-10-30.html", EXAMPLE, get(EXAMPLE)),
    ):
        target = STORE / path
        write(target, content)
        record[name] = dict(url=source_url, path=str(target.relative_to(ROOT)), bytes=len(content),
                            sha256=hashlib.sha256(content).hexdigest(),
                            retrieved_at_utc=datetime.now(timezone.utc).isoformat())
    write(record_path, (json.dumps(record, indent=2) + "\n").encode())
    return record


def chart_arrays(record):
    content = (ROOT / record["chart_series"]["path"]).read_text()
    arrays = {}
    for name in ("eia", "pub", "est", "sat"):
        match = re.search(rf"window\.CW_{name}\s*=\s*(\[.*?\]);", content, flags=re.S)
        rows = json.loads(match[1])
        arrays[name] = {date.fromisoformat(row[0]): row[1:] for row in rows}
        if len(arrays[name]) != len(rows):
            raise ValueError(f"Duplicate dates in chart array {name}")
    if arrays["eia"].keys() != arrays["pub"].keys() or arrays["est"].keys() != arrays["sat"].keys():
        raise ValueError("Expected matching dates in both pairs of chart arrays")
    return arrays


def official_eia():
    days = {}
    for row in read("eia_inventory_release_panel"):
        day = date.fromisoformat(row["week_ending"])
        days[day] = dict(stocks=int(row["cushing_kbbl"]), date=date.fromisoformat(row["release_date"]),
                         precision=int(row["cushing_printed_precision_kbbl"]), vintage=row["reported_as"],
                         url=row["source_url"])
    for row in read("eia_wpsr_forward_releases"):
        day = date.fromisoformat(row["latest_week_in_issue"])
        if day not in days:
            days[day] = dict(stocks=int(float(row["cushing_kbbl"])), date=date.fromisoformat(row["release_date"]),
                             precision=int(float(row["printed_precision_kbbl"])), vintage="current_week",
                             url=row["source_url"])
    return days


def validate_eia(chart, official):
    for day, (stock,) in chart["eia"].items():
        issue = official.get(day)
        if issue is None:
            raise ValueError(f"EIA week missing from independent archive: {day}")
        (chart_issue,) = chart["pub"][day]
        yield dict(week_ending=day, chart_eia_kbbl=stock, official_eia_kbbl=issue["stocks"],
                   official_eia_vintage=issue["vintage"], official_precision_kbbl=issue["precision"],
                   stock_delta_kbbl=stock - issue["stocks"], chart_report_date=chart_issue,
                   official_first_known_date=issue["date"],
                   report_date_delta_days=(date.fromisoformat(chart_issue) - issue["date"]).days,
                   official_source_url=issue["url"])


def comparisons(chart, official, record):
    website_weeks = sorted(chart["eia"])
    official_weeks = sorted(official)
    for day, (anchored, fill, ahead) in sorted(chart["est"].items()):
        satellite, satellite_fill, satellite_ahead = chart["sat"][day]
        if (fill, ahead) != (satellite_fill, satellite_ahead):
            raise ValueError(f"Chart metadata differ by method on {day}")
        website_week = min(website_weeks, key=lambda week: abs(week - day))
        site_gap = (website_week - day).days
        site_target = chart["eia"][website_week][0] if abs(site_gap) <= 3 else None
        week = min(official_weeks, key=lambda week: abs(week - day))
        gap = (week - day).days
        issue = official[week]
        eligible = abs(gap) <= 3 and day < issue["date"] and issue["vintage"] == "current_week"
        if abs(gap) > 3:
            issue = {}
        prior = max((w for w, issue in official.items() if issue["date"] < day), default=None)
        prior_issue = official.get(prior, {})
        last = prior_issue.get("stocks")
        yield dict(
            chart_observation_date=day, anchored_estimate_kbbl=anchored, satellite_estimate_kbbl=satellite,
            displayed_fill_pct=fill, provider_ahead_flag=ahead, provider_historical_publication_verified=False,
            provider_source_url=record["chart_series"]["url"], provider_source_sha256=record["chart_series"]["sha256"],
            chart_nearest_week=website_week if site_target is not None else None,
            chart_nearest_eia_kbbl=site_target,
            official_nearest_week=week if issue else None, official_week_minus_observation_days=gap if issue else None,
            official_target_eia_kbbl=issue.get("stocks"), official_target_first_known_date=issue.get("date"),
            official_target_vintage=issue.get("vintage"), official_target_url=issue.get("url"),
            latest_earlier_report_week=prior, latest_earlier_report_date=prior_issue.get("date"),
            latest_earlier_report_kbbl=last, latest_earlier_report_url=prior_issue.get("url"),
            eligible_for_retrospective_paired_comparison=eligible and last is not None,
        )


def score(rows, estimate, target):
    y = np.array([row[target] for row in rows])
    e = np.array([row[estimate] for row in rows]) - y
    relative = np.abs(e / y) * 100
    return dict(
        count=len(rows), mae_kbbl=float(np.mean(np.abs(e))), mean_error_kbbl=float(np.mean(e)),
        mean_absolute_percentage_error=float(np.mean(relative)), median_absolute_percentage_error=float(np.median(relative)),
        p90_absolute_percentage_error=float(np.percentile(relative, 90)),
    )


def one_observation_per_week(rows):
    """Use the date closest to the EIA reference Friday, with earlier ties."""
    by_week = {}
    for row in sorted(rows, key=lambda r: (abs(r["official_week_minus_observation_days"]), r["chart_observation_date"])):
        by_week.setdefault(row["official_nearest_week"], row)
    return list(by_week.values())


def main():
    record = source_snapshot()
    chart = chart_arrays(record)
    official = official_eia()
    eia_checks = list(validate_eia(chart, official))
    pairs = list(comparisons(chart, official, record))
    estimates = ("anchored_estimate_kbbl", "satellite_estimate_kbbl", "latest_earlier_report_kbbl")
    provider_pairs = [r for r in pairs if r["chart_nearest_eia_kbbl"] is not None]
    common = [r for r in pairs if r["eligible_for_retrospective_paired_comparison"]]
    frozen = [r for r in common if date(2024, 1, 1) <= r["official_nearest_week"] <= date(2026, 10, 30)]
    unique_frozen = one_observation_per_week(frozen)
    days = sorted(chart["est"])
    forward = next(r for r in common if r["chart_observation_date"] == date(2026, 11, 5))
    summary = dict(
        archived_observations=len(pairs), first_chart_observation=min(chart["est"]).isoformat(),
        last_chart_observation=max(chart["est"]).isoformat(),
        chart_observation_gaps_days=dict(sorted(Counter((later - earlier).days for earlier, later in
            zip(days, days[1:])).items())),
        chart_eia_week_count=len(eia_checks),
        chart_eia_stock_disagreements=sum(r["stock_delta_kbbl"] != 0 for r in eia_checks),
        chart_eia_report_date_disagreements=sum(r["report_date_delta_days"] != 0 for r in eia_checks),
        provider_ahead_flag_count=sum(r["provider_ahead_flag"] for r in pairs),
        provider_claim_reproduction={e: score(provider_pairs, e, "chart_nearest_eia_kbbl") for e in estimates[:2]},
        independent_eia_matched_retrospective={e: score(common, e, "official_target_eia_kbbl") for e in estimates},
        independent_eia_matched_retrospective_2024_to_2026_10_30={
            e: score(frozen, e, "official_target_eia_kbbl") for e in estimates},
        independent_eia_one_observation_per_week_2024_to_2026_10_30={
            e: score(unique_frozen, e, "official_target_eia_kbbl") for e in estimates},
        forward_example={k: forward[k] for k in ("chart_observation_date", "official_nearest_week",
                         "official_target_first_known_date", "official_target_eia_kbbl", *estimates)},
        scope="Retrospective nearest-week comparison within three calendar days. Official first-release "
              "targets must have the current-week vintage and a release date after observation. "
              "The naive baseline uses the latest reference week published on an earlier calendar date. "
              "The public chart does not establish historical publication timing of its estimates or independent tank levels.",
    )
    for name, rows in (("eia_crosscheck", eia_checks), ("paired_estimates", pairs)):
        export(f"{STEM}_{name}", rows)
    payload = json.dumps(summary, indent=2, default=date.isoformat) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
