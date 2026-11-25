"""Collect archived PSM and STEO vintages and align them to weekly EIA cutoffs."""
import calendar
import csv
import hashlib
import io
import json
import math
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from html import unescape
from urllib.error import HTTPError
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import openpyxl
import xlrd

from build_s1_reference_matches import read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write, write_csv


STEM = "eia_monthly_context"
STORE = RAW / "eia" / "monthly_context"
PSM = "https://www.eia.gov/petroleum/supply/monthly/"
STEO = "https://www.eia.gov/outlooks/steo/outlook.php"
REVISED = "https://www.eia.gov/dnav/pet/xls/PET_MOVE_PTB_A_EPC0_TNR_MBBL_M.xls"
EASTERN = ZoneInfo("America/New_York")
FIRST_ISSUE = "2015-12"
FIRST_REVISED_DATA_MONTH = "2016-01"
LAST_ISSUE = "2026-11"
ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.I | re.S)
CELL = re.compile(r"<td\b[^>]*>(.*?)</td>", re.I | re.S)
LINK = re.compile(r'''href=["']([^"']+)["']''', re.I)
MONTH_NAME = "|".join(calendar.month_name[1:])
PSM_TABLES = {
    "movements": "Movements of Crude Oil by Pipeline, Tanker, Barge, and Rail between PAD Districts",
    "balance": "Crude Oil Supply, Disposition, and Ending Stocks by PAD District",
}
STEO_SERIES = {
    "2tab": {"WTIPUUS": "usd_per_bbl", "BREPUUS": "usd_per_bbl"},
    "3atab": {"papr_world": "million_bbl_per_day", "patc_world": "million_bbl_per_day",
              "papr_opec": "million_bbl_per_day", "papr_ca": "million_bbl_per_day"},
    "4atab": {"COPRPUS": "million_bbl_per_day", "CONIPUS": "million_bbl_per_day",
              "COSQ_DRAW": "million_bbl_per_day", "PAPRPGLF": "million_bbl_per_day",
              "PAPR48NGOM": "million_bbl_per_day"},
}


def label(html):
    return " ".join(unescape(re.sub(r"<[^>]*>", " ", html)).split())


def path_data(url, path):
    payload = get(url, path)
    return payload, dict(url=url, path=str(path.relative_to(ROOT)), bytes=len(payload),
                         sha256=hashlib.sha256(payload).hexdigest())


def provenance(prefix, info):
    return {f"{prefix}_{key}": value for key, value in info.items()}


def export(suffix, rows):
    fields = tuple(dict.fromkeys(k for r in rows for k in r))
    write_csv(DERIVED / f"{STEM}_{suffix}.csv", fields, ([r.get(k) for k in fields] for r in rows))


def month_start(value):
    return datetime.strptime(" ".join(value.split()), "%B %Y").date().replace(day=1)


def availability(day):
    return datetime.combine(day + timedelta(days=1), time.min, EASTERN).astimezone(timezone.utc)


def month_distance(later, earlier):
    return (later.year - earlier.year) * 12 + later.month - earlier.month


def psm_issues():
    url = urljoin(PSM, "archive/")
    index, proof = path_data(url, STORE / "psm_index.html")
    issues = {}
    for row in ROW.findall(index.decode()):
        for cell in CELL.findall(row):
            for link in LINK.findall(cell):
                found = re.search(r"/(20\d{2})/(20\d{2})_(\d{2})/psm_\2_\3\.php$", link)
                if not found or found[1] != found[2]:
                    continue
                key = f"{found[2]}-{found[3]}"
                if FIRST_ISSUE <= key <= LAST_ISSUE:
                    year, month = (int(part) for part in label(cell).split("/")[::-1])
                    issues[key] = (key, date(year, month, 1), urljoin(url, link))
    if not issues:
        raise ValueError("No archived PSM issues were indexed")
    return [issues[k] for k in sorted(issues)], proof


def csv_rows(payload):
    return csv.DictReader(io.StringIO(payload.decode("cp1252")))


def printed_int(value):
    if value is None:
        return None
    raw = value.strip()
    return int(raw.replace(",", "")) if re.fullmatch(r"-?\d[\d,]*", raw) else None


def parse_movements(payload):
    results = []
    for i, row in enumerate(csv_rows(payload), 2):
        match = re.fullmatch(r"From ([1-5]) to", row["STUB_1"].strip(), re.I)
        if not match:
            continue
        origin, destination = int(match[1]), int(row["STUB_2"])
        results.append(dict(source_row=i, origin_padd=origin, destination_padd=destination,
            **{f"{mode}_kbbl": printed_int(row[column]) for mode, column in
               (("pipeline", "Pipeline"), ("tanker_barge", "Tanker and Barge"), ("rail", "Rail"), ("total", "Total"))},
            source_tokens_json=json.dumps(row, ensure_ascii=False)))
    if {(r["origin_padd"], r["destination_padd"]) for r in results} != {(i, j) for i in range(1, 6) for j in range(1, 6) if i != j} or len(results) != 20:
        raise ValueError("Incomplete directed crude oil movements table")
    return results


def parse_balance(payload):
    rows = []
    for i, row in enumerate(csv_rows(payload), 2):
        if not row["STUB_1"].strip():
            continue
        for region in map(str, range(1, 6)):
            token = row[region]
            rows.append(dict(source_row=i, section=row["STUB_1"].strip(), detail=row["STUB_2"].strip(),
                             padd=int(region), value_kbbl=printed_int(token), source_token=token))
    if not any(r["section"] == "Ending Stocks" and r["padd"] == 2 and r["value_kbbl"] is not None for r in rows):
        raise ValueError("No archived PADD 2 ending stocks in PSM balance table")
    return rows


def psm_issue(issue):
    key, observed_month, url = issue
    directory = STORE / "psm" / key
    payload, page = path_data(url, directory / "issue.html")
    html = payload.decode("utf-8")
    content = label(html)
    month = re.search(rf"With Data for ((?:{MONTH_NAME}) \d{{4}})", content)
    day = re.search(rf"(?<!Next )Release Date:\s*((?:{MONTH_NAME}) \d{{1,2}}, \d{{4}})", content)
    if not month or not day or month_start(month[1]) != observed_month:
        raise ValueError(f"PSM archive and issue disagree on the data month in {key}")
    released = datetime.strptime(day[1], "%B %d, %Y").date()
    if month_distance(released, observed_month) < 1:
        raise ValueError(f"Invalid PSM release date in {key}")
    record = dict(issue=key, data_month=observed_month.isoformat(), release_date=released.isoformat(),
                  available_at_utc=availability(released).isoformat(), **provenance("issue", page))
    found = {}
    for row in ROW.findall(html):
        cells = CELL.findall(row)
        for name, title in PSM_TABLES.items():
            if any(label(c).casefold() == title.casefold() for c in cells):
                url_for_csv, = [urljoin(url, v) for v in LINK.findall(row) if re.fullmatch(r"csv/table\d+\.csv", v, re.I)]
                raw, proof = path_data(url_for_csv, directory / f"{name}.csv")
                found[name] = parse_movements(raw) if name == "movements" else parse_balance(raw)
                record.update(provenance(name, proof))
    if found.keys() != PSM_TABLES.keys():
        raise ValueError(f"Missing PSM table in {key}: {PSM_TABLES.keys() - found.keys()}")
    return record, found


def steo_issues():
    payload, proof = path_data(STEO, STORE / "steo_index.html")
    issues = {}
    cells = CELL.findall(payload.decode("utf-8"))
    # The 2019 archive has issue cells outside opening <tr> tags.
    for i, cell in enumerate(cells[:-3]):
        if not re.fullmatch(rf"(?:{MONTH_NAME}) \d{{4}}", label(cell)):
            continue
        key = month_start(label(cell)).isoformat()[:7]
        if not FIRST_ISSUE <= key <= LAST_ISSUE:
            continue
        day = label(cells[i + 1])
        links = [urljoin(STEO, v) for v in LINK.findall(cells[i + 3]) if v.lower().endswith(".xlsx")]
        notice = [urljoin(STEO, v) for v in LINK.findall(cells[i + 4]) if "notice" in v.lower()] if i + 4 < len(cells) else []
        if key in issues:
            raise ValueError(f"Multiple STEO archive entries for {key}")
        issues[key] = key, day, links, notice
    return [issues[k] for k in sorted(issues)], proof


def steo_issue(issue):
    key, day, links, notice = issue
    if not links or not re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", day):
        return dict(issue=key, archive_release_text=day, status="no_dated_workbook"), []
    url, = links
    raw, proof = path_data(url, STORE / "steo" / f"{key}.xlsx")
    released = datetime.strptime(day, "%m/%d/%Y").date()
    checked = released
    notice_info = {}
    if notice:
        notice_url, = notice
        match = re.search(r"(\d{2}_\d{2}_\d{4})\.php$", notice_url)
        if match is None:
            raise ValueError(f"Unrecognized STEO notice date for {key}: {notice_url}")
        checked = max(released, datetime.strptime(match[1], "%m_%d_%Y").date())
        try:
            _, notice_source = path_data(notice_url, STORE / "steo_notices" / f"{key}.html")
            notice_info = provenance("notice", notice_source) | {"notice_status": "archived"}
        except HTTPError as error:
            if error.code != 404:
                raise
            notice_info = {"notice_url": notice_url, "notice_status": "linked_notice_not_found"}
    available = availability(checked)
    result = []
    with io.BytesIO(raw) as stream:
        book = openpyxl.load_workbook(stream, read_only=True, data_only=True)
        try:
            dates = list(book["Dates"].values)
            if month_start(dates[0][3]).isoformat()[:7] != key:
                raise ValueError(f"STEO workbook issue differs from archive index: {key}")
            axis, = [r for r in dates if isinstance(r[1], str) and "AAAA_DATEX" in r[1]]
            historical = next((r[3] for r in dates if str(r[0]).startswith("Last Historical Month")), None)
            for sheet, series in STEO_SERIES.items():
                table = list(book[sheet].values)
                seen = {}
                for row_number, row in enumerate(table, 1):
                    code = row[0]
                    if code not in series:
                        continue
                    values = tuple(row[2:len(axis)])
                    if code in seen:
                        if seen[code] != values:
                            raise ValueError(f"Conflicting STEO values for {key}/{sheet}/{code}")
                        continue
                    seen[code] = values
                    for col, stamp in enumerate(axis[2:], 2):
                        if not isinstance(stamp, int) or not 190001 <= stamp <= 210012 or not 1 <= stamp % 100 <= 12:
                            continue
                        period = date(stamp // 100, stamp % 100, 1)
                        if table[3][col] != calendar.month_abbr[period.month]:
                            raise ValueError(f"STEO worksheet month axis differs from Dates: {key}/{sheet}/{col + 1}")
                        value = row[col] if col < len(row) else None
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                            result.append(dict(issue=key, release_date=released.isoformat(), available_at_utc=available.isoformat(),
                                period_month=period.isoformat(), months_from_release=month_distance(period, released),
                                after_workbook_last_historical_month=(stamp > historical if isinstance(historical, int) else None),
                                series_code=code, series_label=row[1].strip(), unit=series[code], value=value,
                                sheet=sheet, source_row=row_number, source_column=col + 1, **provenance("workbook", proof)))
        finally:
            book.close()
    if {"WTIPUUS", "BREPUUS", "papr_world", "patc_world", "COPRPUS"} - {r["series_code"] for r in result}:
        raise ValueError(f"STEO vintage has missing core series: {key}")
    return dict(issue=key, release_date=released.isoformat(), available_at_utc=available.isoformat(), status="verified",
                observations=len(result), last_historical_month_source=historical,
                latest_archive_notice_or_release_date=checked.isoformat(), **provenance("workbook", proof), **notice_info), result


def revised_movements():
    raw, proof = path_data(REVISED, STORE / "revised_movements.xls")
    book = xlrd.open_workbook(file_contents=raw)
    rows = []
    for sheet in book.sheets()[1:]:
        for col in range(1, sheet.ncols):
            series = sheet.cell_value(1, col)
            description = sheet.cell_value(2, col)
            if not series or not description:
                continue
            route = re.search(r"\(PADD ([1-5])\) Receipts .* from .*\(PADD ([1-5])\) of Crude Oil", description)
            if not route:
                raise ValueError(f"Cannot identify the direction of {series}")
            for i in range(3, sheet.nrows):
                stamp, cell = sheet.cell(i, 0), sheet.cell(i, col)
                if stamp.ctype != xlrd.XL_CELL_DATE or cell.ctype != xlrd.XL_CELL_NUMBER:
                    continue
                month = xlrd.xldate_as_datetime(stamp.value, book.datemode).date().replace(day=1)
                if month.isoformat()[:7] >= FIRST_REVISED_DATA_MONTH:
                    rows.append(dict(series_id=series, description=description, data_month=month.isoformat(),
                                     origin_padd=int(route[2]), destination_padd=int(route[1]),
                                     total_kbbl=int(cell.value), worksheet=sheet.name, source_row=i+1, source_column=col+1,
                                     **provenance("workbook", proof)))
    return rows


def revisions(archived, revised):
    def key(r):
        return r["data_month"], r["origin_padd"], r["destination_padd"]

    lookup = {key(r): r for r in revised}
    if len(lookup) != len(revised):
        raise ValueError("Duplicate movement in the current EIA workbook")
    for old in archived:
        if not (new := lookup.get(key(old))) or old["total_kbbl"] is None:
            continue
        yield dict(data_month=old["data_month"], origin_padd=old["origin_padd"], destination_padd=old["destination_padd"],
            original_issue=old["issue"], original_release_date=old["release_date"], original_csv_row=old["source_row"],
            original_total_kbbl=old["total_kbbl"], revised_series_id=new["series_id"], revised_total_kbbl=new["total_kbbl"],
            revision_kbbl=new["total_kbbl"] - old["total_kbbl"], revised_worksheet=new["worksheet"],
            revised_source_row=new["source_row"], revised_source_column=new["source_column"])


def weekly_alignment(psm, steo):
    psm = sorted(psm, key=lambda r: r["available_at_utc"])
    steo = sorted((r for r in steo if r["status"] == "verified"), key=lambda r: r["available_at_utc"])
    psm_clocks, steo_clocks = ([datetime.fromisoformat(r["available_at_utc"]) for r in x] for x in (psm, steo))
    releases = read("eia_release_clock")
    forward = read("eia_wpsr_forward_releases")
    covered = {r["release_date"] for r in releases}
    releases.extend(r for r in forward if r["release_date"] not in covered)
    for row in sorted(releases, key=lambda r: r["release_date"]):
        cutoff = datetime.fromisoformat(row["conservative_prerelease_before_utc"])
        psm_i, steo_i = bisect_right(psm_clocks, cutoff) - 1, bisect_right(steo_clocks, cutoff) - 1
        psm_data, steo_data = (psm[psm_i] if psm_i >= 0 else {}), (steo[steo_i] if steo_i >= 0 else {})
        yield dict(release_date=row["release_date"], week_ending=row["latest_week_in_issue"], forecast_cutoff_utc=cutoff.isoformat(),
            psm_data_month_age=month_distance(cutoff.astimezone(EASTERN).date(), date.fromisoformat(psm_data["data_month"])) if psm_data else None,
            **{f"{product}_{key}": sample.get(key) for product, sample in (("psm", psm_data), ("steo", steo_data))
               for key in ("issue", "release_date", "available_at_utc")}, psm_data_month=psm_data.get("data_month"))


def weekly_features(weekly, movements, balance, outlook):
    flows = defaultdict(dict)
    for r in movements:
        flows[r["issue"]][r["origin_padd"], r["destination_padd"]] = r
    stocks = defaultdict(dict)
    for r in balance:
        if r["padd"] == 2:
            stocks[r["issue"]][r["section"], r["detail"]] = r["value_kbbl"]
    forecasts = {(r["issue"], r["period_month"], r["series_code"]): r["value"] for r in outlook}
    if len(forecasts) != len(outlook):
        raise ValueError("Duplicate STEO forecast identity")
    for row in weekly:
        release_month = date.fromisoformat(row["release_date"]).replace(day=1)
        next_month = (release_month.replace(day=28) + timedelta(days=4)).replace(day=1)
        issue, forecast_issue = row["psm_issue"], row["steo_issue"]
        # Regional balance figures from an older snapshot remain traceable in the vintage join.
        use_psm = row["psm_data_month_age"] is not None and row["psm_data_month_age"] <= 3
        route = flows.get(issue, {}) if use_psm else {}
        regional = stocks.get(issue, {}) if use_psm else {}
        yield dict(row, psm_used_for_features=use_psm,
            **{f"padd{a}_to_padd{b}_{mode}_kbbl": route.get((a, b), {}).get(f"{mode}_kbbl")
               for a, b in ((2, 3), (3, 2), (2, 4), (4, 2)) for mode in ("pipeline", "total")},
            padd2_refinery_inputs_kbbl=regional.get(("Disposition", "Refinery Inputs")),
            padd2_field_production_kbbl=regional.get(("Supply", "Field Production")),
            padd2_commercial_ending_stocks_kbbl=regional.get(("Ending Stocks", "Commercial")),
            **{f"steo_{code.lower()}_{horizon}": forecasts.get((forecast_issue, month.isoformat(), code))
               for horizon, month in (("current_month", release_month), ("next_month", next_month))
               for codes in STEO_SERIES.values() for code in codes})


def main():
    psm_index, psm_index_proof = psm_issues()
    steo_index, steo_index_proof = steo_issues()
    with ThreadPoolExecutor(max_workers=6) as pool:
        psm, psm_values = zip(*pool.map(psm_issue, psm_index))
        steo, steo_values = zip(*pool.map(steo_issue, steo_index))
    movements, balance = ([dict(issue=i["issue"], data_month=i["data_month"], release_date=i["release_date"],
                               available_at_utc=i["available_at_utc"], **r) for i, data in zip(psm, psm_values, strict=True)
                           for r in data[kind]] for kind in ("movements", "balance"))
    outlook = [r for issue in steo_values for r in issue]
    revised = revised_movements()
    reconciliation = list(revisions(movements, revised))
    weekly = list(weekly_alignment(psm, steo))
    features = list(weekly_features(weekly, movements, balance, outlook))
    for name, values in (("psm_vintages", psm), ("psm_movements", movements), ("psm_balance", balance),
                         ("steo_vintages", steo), ("steo_values", outlook), ("revised_movements", revised),
                         ("weekly_alignment", weekly), ("weekly_features", features), ("psm_revision_comparison", reconciliation)):
        export(name, values)
    summary = dict(psm_index=psm_index_proof, steo_index=steo_index_proof, psm_vintages=len(psm),
        psm_first_data_month=min(r["data_month"] for r in psm), psm_last_data_month=max(r["data_month"] for r in psm),
        psm_movement_rows=len(movements), psm_balance_rows=len(balance), steo_vintage_status=Counter(r["status"] for r in steo),
        steo_notice_status=Counter(r["notice_status"] for r in steo if r.get("notice_status")),
        steo_issues_delayed_by_notice=sum(r.get("latest_archive_notice_or_release_date", "") > r.get("release_date", "") for r in steo),
        steo_series_codes=sorted({r["series_code"] for r in outlook}), steo_value_rows=len(outlook),
        current_revised_movement_rows=len(revised), latest_revised_movement_month=max(r["data_month"] for r in revised),
        movements_with_both_archived_and_revised_value=len(reconciliation),
        movements_with_changed_total=sum(r["revision_kbbl"] != 0 for r in reconciliation),
        weekly_release_rows=len(weekly), weekly_with_psm=sum(bool(r["psm_issue"]) for r in weekly),
        weekly_with_psm_data_at_most_three_months_old=sum(r["psm_data_month_age"] is not None and r["psm_data_month_age"] <= 3 for r in weekly),
        weekly_with_steo=sum(bool(r["steo_issue"]) for r in weekly), weekly_with_current_wti_projection=sum(
            r["steo_wtipuus_current_month"] is not None for r in features),
        availability_basis="Next Eastern midnight after the release date or later dated STEO archive notice. "
                           "Revised PSM history has no historical availability timestamp.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
