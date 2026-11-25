"""Preserve Cushing tank-farm estimates and dated monthly inventory publications."""

import hashlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from statistics import median

import openpyxl
import xlrd

from audit_cushing_holdout_eia_vintages import stocks
from build_s1_reference_matches import read
from build_weekly_shadow_panel import DERIVED
from collect_eia_monthly_context import STORE, availability, csv_rows, path_data, printed_int
from collect_public import RAW, ROOT, write, write_csv

STEM = "cushing_eia_inventory_controls"
NOTICE = "https://www.eia.gov/petroleum/supply/weekly/wcrudeoilstorage_notice.php"
WEEKLY = "https://www.eia.gov/petroleum/supply/weekly/wcrudeoilstorage.xlsx"
MONTHLY = "https://www.eia.gov/dnav/pet/hist_xls/MCRST_YCUOK_1m.xls"
WEEKLY_RELEASE = datetime(2024, 2, 14).date()


def weekly_controls(raw):
    clocks = read("eia_release_clock")
    first_report = {}
    for r in sorted(clocks, key=lambda r: r["release_date"]):
        for field, column in (("oldest_week_in_issue", 3), ("latest_week_in_issue", 2)):
            first_report.setdefault(r[field], (r["release_date"], column, r["conservative_postrelease_from_utc"]))
    book = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    try:
        sheet = book["Crude oil storage capacity"]
        matrix = list(sheet.values)
    finally:
        book.close()
    header, = [i for i, r in enumerate(matrix) if isinstance(r[0], str) and r[0].startswith("Cushing")
               and r[0].endswith("(subset of PADD 2)")]
    labels = ("Total stocks", "Less:", "Net stocks (held in tank farms)", "Working storage capacity",
              "Net stocks percent of working storage capacity")
    if any(not matrix[header+i][0].startswith(label) for i, label in enumerate(labels, 1)):
        raise ValueError("Unexpected Cushing storage workbook labels")
    previous_day = previous_net = previous_adjustment = None
    for col, value in enumerate(matrix[header][1:], 2):
        if value is None:
            continue
        day = value.date()
        if previous_day and day-previous_day != timedelta(days=7):
            raise ValueError(f"Cushing storage workbook week is out of sequence: {day}")
        total, outside, net, capacity, ratio = [matrix[header+i][col-1] for i in range(1, 6)]
        if total-outside != net or not (0 <= net <= capacity and 0 < outside and math.isclose(net/capacity, ratio, abs_tol=.005)):
            raise ValueError(f"Cushing storage workbook does not reconcile on {day}")
        issue, column, known = first_report[day.isoformat()]
        current_week, prior_week, current, prior, _, _, _, source_row, source = stocks(issue)
        week, first_value = (current_week, current) if column == 2 else (prior_week, prior)
        if week != day.isoformat() or first_value != total:
            raise ValueError(f"Cushing storage workbook and original WPSR disagree on {day}")
        yield dict(week_ending=week, combined_stocks_kbbl=total,
            eia_pipeline_fill_and_in_transit_adjustment_kbbl=outside, estimated_net_tank_farm_stocks_kbbl=net,
            net_tank_farm_change_kbbl=net-previous_net if previous_net is not None else None,
            outside_tank_farm_adjustment_change_kbbl=outside-previous_adjustment if previous_adjustment is not None else None,
            working_storage_capacity_kbbl=capacity, reported_utilization_fraction=ratio, workbook_column=col,
            first_wpsr_issue=issue, first_wpsr_total_known_at_utc=known, first_wpsr_table4_row=source_row,
            first_wpsr_table4_column=column, **{f"first_wpsr_{k}": v for k, v in source.items()},
            storage_workbook_release_date=WEEKLY_RELEASE.isoformat(),
            storage_workbook_known_at_utc=availability(WEEKLY_RELEASE).isoformat())
        previous_day, previous_net, previous_adjustment = day, net, outside


def monthly_snapshot(raw):
    book = xlrd.open_workbook(file_contents=raw)
    sheet = book.sheet_by_name("Data 1")
    if sheet.cell_value(1, 1) != "MCRST_YCUOK_1":
        raise ValueError("Unexpected monthly Cushing series")
    data = {}
    for cells in sheet.get_rows():
        day, value = cells[:2]
        if day.ctype != xlrd.XL_CELL_DATE or value.ctype != xlrd.XL_CELL_NUMBER:
            continue
        period = xlrd.xldate_as_datetime(day.value, book.datemode).date().isoformat()[:7]
        if period in data or value.value != int(value.value):
            raise ValueError(f"Invalid monthly Cushing workbook row for {period}")
        data[period] = int(value.value)
    return data


def monthly_vintages(snapshot):
    for r in read("eia_monthly_context_psm_vintages"):
        path = ROOT / r["balance_path"]
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != r["balance_sha256"]:
            raise ValueError(f"Archived PSM balance checksum changed: {path}")
        (line, row), = [(i, row) for i, row in enumerate(csv_rows(payload), 2)
                        if row["STUB_1"].strip() == "Ending Stocks" and row["STUB_2"].strip() == "Cushing, Oklahoma"]
        printed = printed_int(row["2"])
        if printed is None or printed != printed_int(row["Total"]):
            raise ValueError(f"Archived PSM balance lacks Cushing stocks in {path}")
        month = r["data_month"][:7]
        revised = snapshot[month]
        yield dict(data_month=month, archived_issue=r["issue"], archive_release_date=r["release_date"],
                   archive_known_at_utc=r["available_at_utc"], archived_combined_stock_kbbl=printed,
                   latest_combined_stock_kbbl=revised, latest_minus_archive_kbbl=revised-printed,
                   archive_source_row=line, archive_source_column="2",
                   **{f"archive_{k}": r[f"balance_{k}"] for k in ("url", "path", "sha256")})


def export(name, rows):
    fields = tuple(rows[0])
    write_csv(DERIVED / f"{STEM}_{name}.csv", fields, (tuple(r[k] for k in fields) for r in rows))


def optical_overlap(weeks):
    optical = {}
    for source, field in (("sentinel2_space_time_coverage_eia_comparison", "raw_kbbl"),
                          ("holdout_optical_weeks", "capture_change_kbbl")):
        for row in read(source):
            if row["study_phase"] == "holdout" and field == "raw_kbbl":
                continue
            key = row["week_ending"]
            if key in optical:
                raise ValueError(f"Duplicate optical week {key}")
            optical[key] = (row, row[field], source)
    for r in weeks:
        o, raw, source = optical[r["week_ending"]]
        if r["first_wpsr_issue"] != o["release_date"]:
            raise ValueError(f"Optical week and first EIA issue differ on {r['week_ending']}")
        change = r["net_tank_farm_change_kbbl"]
        measured = float(raw) if raw else None
        yield dict(week_ending=r["week_ending"], release_date=o["release_date"], study_phase=o["study_phase"],
            b04_tanks=int(o["baseline_b04_tanks"]), b04_area_fraction=o["baseline_b04_area_fraction"],
            b04_space_time_fraction=o["baseline_b04_space_time_fraction"], b04_visible_change_kbbl=measured,
            eia_tank_farm_change_kbbl=change,
            eia_outside_tank_farm_adjustment_change_kbbl=r["outside_tank_farm_adjustment_change_kbbl"],
            eia_tank_farm_target_and_b04_available=change is not None and measured is not None,
            optical_source=f"data/derived/{source}.csv")


def main():
    weekly_raw, weekly_source = path_data(WEEKLY, RAW / "eia" / "wcrudeoilstorage.xlsx")
    monthly_raw, monthly_source = path_data(MONTHLY, RAW / "eia" / "cushing_monthly.xls")
    definitions = [
        path_data(url, STORE / "inventory_definitions" / name)[1]
        for name, url in (
            ("eia803_instructions.pdf", "https://www.eia.gov/survey/form/eia_803/instructions.pdf"),
            ("eia813_instructions.pdf", "https://www.eia.gov/survey/form/eia_813/instructions.pdf"),
            ("storage_capacity_notice.html", NOTICE),
        )
    ]
    weeks = list(weekly_controls(weekly_raw))
    overlap = list(optical_overlap(weeks))
    monthly = monthly_snapshot(monthly_raw)
    vintages = list(monthly_vintages(monthly))
    dates = {r["data_month"] for r in vintages}
    if len(vintages) != len(dates):
        raise ValueError("More than one archived monthly balance for a Cushing observation")
    monthly_rows = [dict(data_month=period, latest_combined_stock_kbbl=stock,
                         local_archived_psm_vintage_available=period in dates)
                    for period, stock in sorted(monthly.items())]
    export("weekly_tank_farm_estimates", weeks)
    export("weekly_optical_overlap", overlap)
    export("monthly_snapshot", monthly_rows)
    export("monthly_archive_comparison", vintages)
    differences = [r["latest_minus_archive_kbbl"] for r in vintages]
    summary = dict(
        sources=[weekly_source, monthly_source, *definitions],
        weekly_tank_farm_estimates=len(weeks), first_week=weeks[0]["week_ending"], last_week=weeks[-1]["week_ending"],
        weekly_totals_matching_first_wpsr=len(weeks),
        weekly_changes_with_b04=sum(r["eia_tank_farm_target_and_b04_available"] for r in overlap),
        weekly_net_adjustment_kbbl_counts=Counter(r["eia_pipeline_fill_and_in_transit_adjustment_kbbl"] for r in weeks),
        weeks_when_eia_adjustment_changed=[dict(week_ending=r["week_ending"],
            adjustment_change_kbbl=r["outside_tank_farm_adjustment_change_kbbl"])
            for r in weeks if r["outside_tank_farm_adjustment_change_kbbl"]],
        monthly_snapshot_observations=len(monthly), first_month=min(monthly), last_month=max(monthly),
        monthly_archived_observations=len(vintages), first_month_with_archive=min(dates), last_month_with_archive=max(dates),
        monthly_archive_differences=sum(bool(d) for d in differences),
        maximum_absolute_monthly_revision_kbbl=max(map(abs, differences)),
        median_absolute_nonzero_monthly_revision_kbbl=median(abs(d) for d in differences if d),
        scope="The final weekly utilization workbook supplies retrospectively adjusted tank-farm stocks. Its historical "
              "adjustments use capacity report vintages. The updated monthly workbook is combined stocks and supplies no "
              "first-publication timestamp. Both tables cover the EIA reporting region, not individual tanks.",
    )
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
