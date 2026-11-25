"""Combine WTI sources by dated delivery month and preserve the price basis."""
import argparse
import datetime as dt
import hashlib
import json
import math
import re
from bisect import bisect_left
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as ec

from collect_public import RAW, ROOT, write

ARCHIVE = RAW / "wti_reference" / "tradingcharts"
START = "2016-01-01"
END = "2026-11-11"
MONTH_CODES = "FGHJKMNQUVXZ"
EIA_DEFINITION = "https://www.eia.gov/dnav/pet/tbldefs/pet_pri_fut_tbldef2.asp"
MAY_2020_REPORT = "https://www.cftc.gov/media/5296/InterimStaffReportNYMEX_WTICrudeOil/download"
EXPIRY_EXCEPTIONS = {"202005": ("2020-04-21", "cftc_documented", MAY_2020_REPORT)}
IBKR_DOC = "https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/"

SCHEMA = """
CREATE TABLE wti_tradingcharts_bar (
    delivery_month TEXT, symbol TEXT, session_date TEXT, provider_timestamp TEXT,
    open REAL, high REAL, low REAL, reported_close REAL, volume INTEGER, open_interest INTEGER,
    source_url TEXT, source_file TEXT, source_sha256 TEXT, source_file_mtime_utc TEXT,
    capture_date_chicago TEXT,
    PRIMARY KEY (delivery_month, session_date));
CREATE TABLE wti_delivery_calendar (
    delivery_month TEXT PRIMARY KEY, last_trading_date TEXT, expiry_basis TEXT, source_reference TEXT,
    rule_date TEXT, ibkr_confirmed INTEGER, archive_confirmed INTEGER);
CREATE TABLE wti_eia_dated_close (
    session_date TEXT, curve_rank INTEGER, delivery_month TEXT, official_close REAL,
    source_series TEXT, source_url TEXT, PRIMARY KEY (session_date, curve_rank));
CREATE VIEW wti_close_crosscheck AS
    SELECT e.session_date, e.delivery_month, e.curve_rank, e.official_close AS first_close,
        a.reported_close AS second_close, round(a.reported_close-e.official_close, 6) AS difference,
        'eia_vs_tradingcharts' AS comparison
    FROM wti_eia_dated_close e JOIN wti_tradingcharts_bar a USING (session_date, delivery_month)
    UNION ALL
    SELECT a.session_date, a.delivery_month, NULL, a.reported_close, b.close,
        round(b.close-a.reported_close, 6), 'tradingcharts_vs_ibkr'
    FROM wti_tradingcharts_bar a JOIN tws_futures_closed_sessions b
        ON b.session_date=a.session_date AND b.contract_month=a.delivery_month;
CREATE TABLE wti_nearby_curve (
    session_date TEXT, curve_rank INTEGER, delivery_month TEXT, last_trading_date TEXT, expiry_basis TEXT,
    close_usd_per_bbl REAL, selected_source TEXT, source_reference TEXT, eia_official_close REAL,
    tradingcharts_reported_close REAL, ibkr_trades_close REAL, tradingcharts_volume INTEGER,
    tradingcharts_open_interest INTEGER, ibkr_volume REAL, ibkr_conid INTEGER,
    PRIMARY KEY (session_date, curve_rank));
CREATE TABLE wti_session_basis (
    session_date TEXT PRIMARY KEY, full_cmes_session INTEGER, xnys_session INTEGER,
    cme_no_settlement_notice_url TEXT);
CREATE VIEW wti_curve_daily_coverage AS
    SELECT session_date, count(close_usd_per_bbl) AS priced_maturities,
        count(DISTINCT selected_source) AS price_sources,
        max(CASE WHEN curve_rank=1 THEN close_usd_per_bbl END) AS cl1,
        max(CASE WHEN curve_rank=2 THEN close_usd_per_bbl END) AS cl2,
        max(CASE WHEN curve_rank=3 THEN close_usd_per_bbl END) AS cl3,
        max(CASE WHEN curve_rank=4 THEN close_usd_per_bbl END) AS cl4,
        round(max(CASE WHEN curve_rank=1 THEN close_usd_per_bbl END) -
            max(CASE WHEN curve_rank=2 THEN close_usd_per_bbl END), 4) AS cl1_minus_cl2,
        round(max(CASE WHEN curve_rank=1 THEN close_usd_per_bbl END) -
            max(CASE WHEN curve_rank=4 THEN close_usd_per_bbl END), 4) AS cl1_minus_cl4,
        CASE WHEN b.cme_no_settlement_notice_url IS NOT NULL THEN 'cme_notice_no_settlement'
             WHEN count(close_usd_per_bbl)=4 THEN 'four_prices'
             WHEN b.xnys_session=0 THEN 'incomplete_on_xnys_holiday'
             ELSE 'incomplete_on_xnys_session' END AS coverage_status,
        b.full_cmes_session, b.xnys_session, b.cme_no_settlement_notice_url
    FROM wti_nearby_curve JOIN wti_session_basis b USING (session_date) GROUP BY session_date;
"""


def archive_file(path):
    symbol = path.stem
    month = f"20{symbol[3:]}{MONTH_CODES.index(symbol[2]) + 1:02d}"
    url = f"https://futures.tradingcharts.com/historical/CL_/{month[:4]}/{int(month[4:]):X}/barchart.html"
    raw = path.read_bytes()
    rows = json.loads(raw)
    if not rows or any(row["symbol"] != symbol for row in rows):
        raise ValueError(f"Wrong dated WTI contract in {path}")
    return month, url, hashlib.sha256(raw).hexdigest(), rows


def import_archive(source):
    manifest_path = ARCHIVE / "archive_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())["files"] if manifest_path.is_file() else {}
    imported = 0
    for path in sorted(source.glob("CL*.json")):
        if not re.fullmatch(r"CL[FGHJKMNQUVXZ]\d{2}", path.stem):
            continue
        month, url, digest, rows = archive_file(path)
        timestamp = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        write(ARCHIVE / path.name, path.read_bytes())
        manifest[path.name] = {"delivery_month": month, "chart_url": url, "sha256": digest,
                               "source_file_mtime_utc": timestamp.isoformat(), "rows": len(rows),
                               "imported_from": str(source.resolve())}
        imported += 1
    if not imported:
        raise ValueError(f"No dated CL archives found in {source}")
    write(manifest_path, (json.dumps({"files": manifest}, indent=2, sort_keys=True) + "\n").encode())
    print(f"Imported {imported} dated WTI files, {len(manifest)} in the local manifest")


def add_wti_curve(db):
    manifest_path = ARCHIVE / "archive_manifest.json"
    if not manifest_path.is_file():
        return
    archived = {}
    for filename, entry in json.loads(manifest_path.read_bytes())["files"].items():
        path = ARCHIVE / filename
        month, url, digest, rows = archive_file(path)
        if (month, url, digest, len(rows)) != (entry["delivery_month"], entry["chart_url"],
                                                entry["sha256"], entry["rows"]):
            raise ValueError(f"WTI archive manifest mismatch in {path}")
        archived[month] = max(row["tradingDay"] for row in rows), url
        captured = dt.datetime.fromisoformat(entry["source_file_mtime_utc"])
        capture_day = captured.astimezone(ZoneInfo("America/Chicago")).date().isoformat()
        for row in rows:
            values = [row[field] for field in ("open", "high", "low", "close", "volume", "openInterest")]
            if not all(math.isfinite(value) for value in values) or min(values[4:]) < 0:
                raise ValueError(f"Invalid numeric WTI bar in {path}: {row['tradingDay']}")
            db.execute("INSERT INTO wti_tradingcharts_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (month, row["symbol"], row["tradingDay"], row["timestamp"], *values,
                        url, str(path.relative_to(ROOT)), digest, captured.isoformat(), capture_day))

    broker = dict(db.execute("SELECT DISTINCT contract_month, last_trading_date FROM tws_futures_bar"))
    equity = ec.get_calendar("XNYS", start="2015-12-01", end="2028-01-01")
    business_days = [session.date() for session in equity.sessions]
    for year in range(2016, 2028):
        for number in range(1, 13):
            month = f"{year:04d}{number:02d}"
            reference = dt.date(year - (number == 1), number - 1 or 12, 25)
            index = bisect_left(business_days, reference)
            rule = business_days[index - (3 if business_days[index] == reference else 4)].isoformat()
            last_bar, archive_url = archived.get(month, (None, None))
            last_date, basis, source = EXPIRY_EXCEPTIONS.get(month, (rule, "rule_xnys_proxy", EIA_DEFINITION))
            if month in broker:
                last_date, basis, source = broker[month], "ibkr_contract_metadata", IBKR_DOC
            if last_bar == last_date:
                if basis == "rule_xnys_proxy":
                    basis, source = "rule_and_tradingcharts_last_bar", archive_url
            elif last_bar and last_date <= END:
                raise ValueError(f"WTI {month} archive ends {last_bar}, expected expiry {last_date}")
            db.execute("INSERT INTO wti_delivery_calendar VALUES (?,?,?,?,?,?,?)",
                       (month, last_date, basis, source, rule, int(month in broker), int(last_bar == last_date)))

    calendar = list(db.execute("""SELECT delivery_month, last_trading_date, expiry_basis
                                  FROM wti_delivery_calendar ORDER BY delivery_month"""))
    expiries = [row[1] for row in calendar]
    if any(first >= second for first, second in zip(expiries, expiries[1:])):
        raise ValueError("WTI expiry dates do not increase with delivery month")
    for series, date, value in db.execute("""SELECT series, date, value FROM eia_series
            WHERE series LIKE 'nymex_contract_%_daily' AND date BETWEEN ? AND ?""", (START, END)):
        rank = int(series.split("_")[2])
        month = calendar[bisect_left(expiries, date) + rank - 1][0]
        db.execute("INSERT INTO wti_eia_dated_close VALUES (?,?,?,?,?,?)",
                   (date, rank, month, value, series, EIA_DEFINITION))

    days = {date for date, in db.execute("""SELECT session_date FROM wti_eia_dated_close
        UNION SELECT session_date FROM wti_tradingcharts_bar WHERE volume>0
        UNION SELECT session_date FROM tws_futures_closed_sessions WHERE volume>0""") if START <= date <= END}
    # Include scheduled full sessions even when none of the three providers delivered a price.
    globex = ec.get_calendar("CMES", start=START, end=END)
    early = set(globex.early_closes)
    full_globex_days = {day.date().isoformat() for day in globex.sessions if day not in early}
    equity_days = {day.isoformat() for day in business_days}
    notice_file = ROOT / "data" / "reference" / "wti_no_settlement_notices.json"
    notices = json.loads(notice_file.read_bytes()) if notice_file.is_file() else {}
    days.update(full_globex_days)
    db.executemany("INSERT INTO wti_session_basis VALUES (?,?,?,?)",
                   ((date, int(date in full_globex_days), int(date in equity_days), notices.get(date))
                    for date in sorted(days)))
    keys = [(date, rank, *contract) for date in sorted(days)
            for rank, contract in enumerate(calendar[bisect_left(expiries, date):][:4], 1)]
    db.execute("CREATE TEMP TABLE wti_curve_key (session_date, curve_rank, delivery_month, expiry, expiry_basis)")
    db.executemany("INSERT INTO wti_curve_key VALUES (?,?,?,?,?)", keys)
    db.execute("""INSERT INTO wti_nearby_curve
        SELECT k.*, coalesce(e.official_close, a.reported_close, b.close),
            CASE WHEN e.official_close IS NOT NULL THEN 'eia_official_close'
                 WHEN a.reported_close IS NOT NULL THEN 'tradingcharts_reported_close'
                 WHEN b.close IS NOT NULL THEN 'ibkr_trades_close' END,
            coalesce(e.source_url, a.source_file, b.source_file),
            e.official_close, a.reported_close, b.close, a.volume, a.open_interest, b.volume, b.conid
        FROM wti_curve_key k
        LEFT JOIN wti_eia_dated_close e USING (session_date, curve_rank, delivery_month)
        LEFT JOIN wti_tradingcharts_bar a ON a.session_date=k.session_date
            AND a.delivery_month=k.delivery_month AND a.volume>0
            AND a.session_date<a.capture_date_chicago
        LEFT JOIN tws_futures_closed_sessions b ON b.session_date=k.session_date
            AND b.contract_month=k.delivery_month AND b.volume>0
        ORDER BY k.session_date, k.curve_rank""")
    db.execute("DROP TABLE wti_curve_key")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-archive", type=Path, required=True)
    import_archive(parser.parse_args().import_archive)
