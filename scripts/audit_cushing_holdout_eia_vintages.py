"""Trace held-out EIA targets and previously available factors to raw WPSR cells."""
import csv
import hashlib
import io
import json
import sqlite3
from bisect import bisect_left
from collections import Counter
from contextlib import closing
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import cache

import xlrd

from build_eia_market_inputs import METRICS
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import RAW, ROOT, write
from fit_cushing_nowcast_ensemble import METRICS as MODEL_METRICS

STEM = "cushing_holdout_eia_vintage_audit"
WEEKLY = RAW / "eia/cushing_weekly.xls"
ARCHIVE = RAW / "eia/wpsr"
SERIES = "W_EPC0_SAX_YCUOK_MBBL"


def check(condition, location):
    if not condition:
        raise ValueError(f"EIA archive audit failed at {location}")


def report_date(value):
    return datetime.strptime(value, "%m/%d/%y" if len(value.rsplit("/", 1)[-1]) == 2
                             else "%m/%d/%Y").date().isoformat()


@cache
def archive(issue, table, *, directory=ARCHIVE):
    slug = issue.replace("-", "_")
    path = directory / f"{slug}_table{table}.csv"
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("cp1252")
    rows = list(csv.reader(io.StringIO(content)))
    url = path.with_suffix(".source").read_text().strip()
    stubs = sum(cell.startswith("STUB_") for cell in rows[0])
    issue_url = f"https://www.eia.gov/petroleum/supply/weekly/archive/{issue[:4]}/{slug}"
    check(stubs in (1, 2) and url in {f"{issue_url}/{directory}/table{table}.csv"
                                    for directory in ("csv", "download")}, path)
    return rows, stubs, dict(source_path=str(path.relative_to(ROOT)), source_url=url,
                             source_sha256=hashlib.sha256(raw).hexdigest())


def value_and_increment(raw, scale):
    number = Decimal(raw.replace(",", ""))
    return number * scale, Decimal(10) ** number.as_tuple().exponent * scale


@cache
def stocks(issue, *, directory=ARCHIVE):
    rows, stubs, source = archive(issue, 4, directory=directory)
    matches = [(i, row) for i, row in enumerate(rows[1:], 2)
               if row and row[0].strip() in ("Cushing", "Cushing, Oklahoma")]
    check(stubs == 1 and len(matches) == 1 and rows[0][3] == "Difference", issue)
    row_number, row = matches[0]
    week, previous = map(report_date, rows[0][1:3])
    check(date.fromisoformat(week) - date.fromisoformat(previous) == timedelta(days=7)
          and previous < week < issue, f"{issue} reporting dates")
    (current, precision), (prior, prior_precision), (change, _) = (
        value_and_increment(cell, 1000) for cell in row[1:4])
    check(change == current - prior, f"{issue} Table 4 printed change")
    return week, previous, current, prior, change, precision, prior_precision, row_number, source


def weekly_snapshot():
    book = xlrd.open_workbook(WEEKLY)
    sheet = book.sheet_by_name("Data 1")
    check(sheet.cell_value(1, 1) == SERIES, WEEKLY)
    values = {}
    for cell, stock in (row[:2] for row in sheet.get_rows()):
        if cell.ctype == xlrd.XL_CELL_DATE and stock.ctype == xlrd.XL_CELL_NUMBER:
            week = xlrd.xldate_as_datetime(cell.value, book.datemode).date().isoformat()
            check(week not in values, f"{WEEKLY}/{week}")
            values[week] = Decimal(str(stock.value))
    return values


def audit_target(panel, index, clock, prior_issue, snapshot):
    issue, week = index["release_date"], index["week_ending"]
    previous = panel["previous_week_ending"]
    observed, printed_prior, stock, prior, delta, increment, prior_increment, row, source = stocks(issue)
    check((observed, printed_prior) == (week, previous)
          == (clock["latest_week_in_issue"], clock["oldest_week_in_issue"])
          and panel["reported_as"] == index["eia_target_vintage_kind"] == "current_week"
          and panel["previous_week_vintage_date"] == issue
          and panel["source_url"] == index["eia_target_reference"] == source["source_url"]
          and clock["conservative_prerelease_before_utc"] == index["forecast_cutoff_utc"]
          and clock["conservative_postrelease_from_utc"] == index["target_available_at_utc"]
          and (stock, prior, delta, increment, prior_increment) == tuple(panel[k] for k in (
              "cushing_kbbl", "previous_week_kbbl_as_known_at_release", "change_kbbl_as_known_at_release",
              "cushing_printed_precision_kbbl", "previous_week_printed_precision_kbbl")),
          f"{issue} first publication")
    earlier_week, base_week, earlier, _, earlier_change, *_ = stocks(prior_issue)
    check(earlier_week == previous == panel["latest_week_known_before_release"]
          == index["control_observation_week"]
          and base_week == panel["prior_week_of_latest_known"]
          and earlier == panel["latest_kbbl_known_before_release"]
          and earlier_change == panel["latest_weekly_change_known_before_release"]
          and all(panel[k] == prior_issue for k in ("latest_vintage_known_before_release",
                                                   "prior_vintage_for_latest_known_change")),
          f"{issue} previously known stock and change")
    check(week in snapshot and previous in snapshot, f"{issue} weekly workbook coverage")
    return dict(study_phase=index["study_phase"], week_ending=week, release_date=issue,
                forecast_cutoff_utc=index["forecast_cutoff_utc"],
                target_available_at_utc=index["target_available_at_utc"],
                scheduled_release_at_utc=clock["scheduled_table_release_at_utc"],
                timing_basis=clock["timing_basis"], timing_source_url=clock["source_url"],
                source_row=row, current_column=2, prior_column=3,
                difference_column=4, **source, same_issue_prior_week=previous, prior_issue=prior_issue,
                prior_issue_available_at_utc=index["prior_stock_available_at_utc"],
                same_issue_reprint_matches_prior_issue=prior == earlier,
                latest_workbook_level_matches=snapshot[week] == stock,
                latest_workbook_change_matches=snapshot[week] - snapshot[previous] == delta)


def audit_control(index, prior_issue, available, week, role, metric, control):
    issue = index["release_date"]
    location = f"{issue} <- {prior_issue}/{week}/{metric.key}"
    check(control is not None, location)
    rows, stubs, source = archive(prior_issue, metric.table)
    line, column = control["source_row_or_line"], control["source_column_or_pdf_page"]
    check(2 <= line <= len(rows) and column == stubs + (1 if role == "latest" else 2), location)
    row = rows[line - 1]
    label, raw = row[stubs - 1].strip(), row[column - 1].strip()
    group = row[0].strip() if stubs == 2 else None
    labels = (metric.label, "Cushing, Oklahoma") if metric.key == "cushing_crude" else (metric.label,)
    check(report_date(rows[0][column - 1]) == week and control["source_kind"] == "archived_csv"
          and control["table_number"] == metric.table and control["source_url"] == source["source_url"]
          and label in labels and label == control["source_label"]
          and (metric.row is None or line == metric.row) and (metric.group is None or group == metric.group)
          and raw == control["raw_value"] and control["unit"] == metric.unit, location)
    amount, increment = value_and_increment(raw, 1000 if metric.unit == "thousand_bbl" else 1)
    check(control["value"] is not None and control["source_printed_increment"] is not None
          and amount == Decimal(str(control["value"])) and increment == Decimal(str(control["source_printed_increment"])),
          location)
    return dict(release_date=issue, forecast_cutoff_utc=index["forecast_cutoff_utc"], source_issue=prior_issue,
                source_available_at_utc=available, observation_week=week, observation_role=role, metric=metric.key,
                unit=metric.unit, used_by_frozen_ensemble=metric.key in MODEL_METRICS, table=metric.table,
                source_row=line, source_column=column, **source)


def build(db, indexes, replay_clocks):
    db.row_factory = sqlite3.Row
    panel = {(p["week_ending"], p["release_date"]): p for p in db.execute(
        "SELECT * FROM eia_inventory_release_panel WHERE study_phase='holdout'")}
    check(set(panel) == {(r["week_ending"], r["release_date"]) for r in indexes}
          and len(panel) == len(indexes) > 0, "holdout release calendar")
    clocks = {row["release_date"]: row for row in db.execute("SELECT * FROM eia_release_clock ORDER BY release_date")}
    issues = list(clocks)
    controls = {(c["release_date"], c["report_date"], c["metric"]): c for c in db.execute(
        "SELECT * FROM eia_control_vintage")}
    snapshot = weekly_snapshot()
    cushing = next(metric for metric in METRICS if metric.key == "cushing_crude")
    clock_fields = ("latest_week_in_issue", "oldest_week_in_issue", "conservative_postrelease_from_utc")
    releases, features, history = [], [], []
    for index in indexes:
        issue, week = index["release_date"], index["week_ending"]
        cut = datetime.fromisoformat(index["forecast_cutoff_utc"])
        end = bisect_left(issues, issue)
        lags = issues[max(0, end-4):end]
        check(len(lags) == 4 and all(index[k] == lags[-1] for k in
              ("control_vintage", "prior_stock_vintage", "prior_stock_base_vintage"))
              and datetime.fromisoformat(index["target_available_at_utc"]) > cut, f"{issue} vintage selection")
        replay_issues = sorted(d for d in replay_clocks if d < issue)
        check(replay_issues[-4:] == lags and all(
            clocks[d][key] == replay_clocks[d][key] for d in (*lags, issue) for key in clock_fields),
            f"{issue} replay CSV versus database clock")
        releases.append(audit_target(panel[week, issue], index, clocks[issue], lags[-1], snapshot))
        for lag, source_issue in enumerate(reversed(lags), 1):
            clock = clocks[source_issue]
            available = clock["conservative_postrelease_from_utc"]
            newest, oldest, *_, row, source = stocks(source_issue)
            check((newest, oldest) == (clock["latest_week_in_issue"], clock["oldest_week_in_issue"])
                  and datetime.fromisoformat(available) <= cut, f"{issue}/{source_issue} lag availability")
            history.append(dict(release_date=issue, forecast_cutoff_utc=index["forecast_cutoff_utc"], lag=lag,
                                source_issue=source_issue, source_available_at_utc=available, latest_week=newest,
                                previous_week=oldest, source_row=row, current_column=2, prior_column=3,
                                difference_column=4, **source))
            if lag == 1:
                check(available == index["prior_stock_available_at_utc"] == index["prior_stock_base_available_at_utc"],
                      f"{issue} previous issue clock")
            for role, observed in (("latest", newest), ("previous", oldest)):
                for metric in METRICS if lag == 1 else (cushing,):
                    record = audit_control(index, source_issue, available, observed, role, metric,
                                           controls.get((source_issue, observed, metric.key)))
                    if lag == 1:
                        features.append(record)
    return releases, features, history


def main():
    with (DERIVED / "cushing_holdout_release_index.csv").open(newline="") as handle:
        indexes = list(csv.DictReader(handle))
    with (DERIVED / "eia_release_clock.csv").open(newline="") as handle:
        replay_clocks = {r["release_date"]: r for r in csv.DictReader(handle)}
    with closing(read_only(SOURCE)) as db:
        releases, controls, history = build(db, indexes, replay_clocks)
    summary = dict(releases=len(releases), first_release=releases[0]["release_date"],
                   last_release=releases[-1]["release_date"], first_release_levels_and_deltas_verified=len(releases),
                   timing_basis=dict(sorted(Counter(r["timing_basis"] for r in releases).items())),
                   same_issue_reprints_changed=sum(not r["same_issue_reprint_matches_prior_issue"] for r in releases),
                   latest_workbook_level_disagreements=sum(not r["latest_workbook_level_matches"] for r in releases),
                   latest_workbook_change_disagreements=sum(not r["latest_workbook_change_matches"] for r in releases),
                   latest_workbook_path=str(WEEKLY.relative_to(ROOT)),
                   latest_workbook_sha256=hashlib.sha256(WEEKLY.read_bytes()).hexdigest(),
                   latest_workbook_series=SERIES, prior_control_metrics=len(METRICS), prior_control_cells=len(controls),
                   frozen_ensemble_control_cells=sum(r["used_by_frozen_ensemble"] for r in controls),
                   cushing_prior_four_issue_references=len(history), distinct_wpsr_csv_files=archive.cache_info().currsize,
                   interpretation="First-published WPSR targets and historically available feature cells. "
                                  "The DNav workbook is a retrospective cross-check. No target, feature or model score is exported.")
    for suffix, rows in (("releases", releases), ("controls", controls), ("history", history)):
        export(f"{STEM}_{suffix}", rows)
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
