"""Build source-linked fee schedules and financing observations for equity research."""
import csv
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime
from decimal import Decimal

from collect_public import RAW, ROOT
from publication_clocks import DERIVED, export

FIRST_DATE = "2016-01-01"
AS_OF = "2026-11-11"
SNAPSHOT_DATE = "2026-11-12"
SEC_RATE_VERIFIED_THROUGH = "2026-11-12"
SOURCE = RAW / "execution_costs"
FEE_REFERENCE = ROOT / "data/reference/us_equity_regulatory_fee_periods.csv"
IBKR_MONTHLY = "https://www.interactivebrokers.com/en/accounts/fees/monthlyInterestRates.php"
IBKR_COMMISSIONS = "https://www.interactivebrokers.com/en/pricing/commissions-stocks.php"
NY_FED = "https://markets.newyorkfed.org/api/rates"
BOC_CORRA = "https://www.bankofcanada.ca/valet/observations/AVG.INTWO/json"

TABLES = {
    "us_equity_regulatory_fee_period": ("""fee TEXT NOT NULL,effective_from TEXT NOT NULL,effective_to_exclusive TEXT,
        date_basis TEXT NOT NULL,rate REAL NOT NULL,rate_unit TEXT NOT NULL,cap_usd_per_trade REAL,source_url TEXT NOT NULL,
        verified_through_date TEXT NOT NULL,PRIMARY KEY (fee,effective_from)""", "fee,effective_from"),
    "equity_financing_reference_rate": ("""provider TEXT NOT NULL,series TEXT NOT NULL,currency TEXT NOT NULL,
        observation_date TEXT NOT NULL,annual_rate_pct REAL NOT NULL,source_basis TEXT NOT NULL,
        source_value_flag TEXT,source_url TEXT NOT NULL,source_file TEXT NOT NULL,
        PRIMARY KEY (provider,series,currency,observation_date)""", "provider,series,currency,observation_date"),
    "ibkr_benchmark_download_coverage": ("""request_month TEXT PRIMARY KEY,returned_dates INTEGER NOT NULL,
        dates_in_requested_month INTEGER NOT NULL,first_returned_date TEXT,last_returned_date TEXT,
        source_url TEXT NOT NULL,source_file TEXT NOT NULL""", "request_month"),
    "ibkr_equity_commission_snapshot": ("""snapshot_date TEXT NOT NULL,listing_market TEXT NOT NULL,
        currency TEXT NOT NULL,plan TEXT NOT NULL,first_monthly_share INTEGER NOT NULL,last_monthly_share INTEGER,
        rate_per_share REAL NOT NULL,minimum_per_order REAL NOT NULL,maximum_pct_of_trade_value REAL NOT NULL,
        source_url TEXT NOT NULL,source_file TEXT NOT NULL,
        PRIMARY KEY (snapshot_date,listing_market,currency,plan,first_monthly_share)""",
        "snapshot_date,listing_market,currency,plan,first_monthly_share"),
}


def fee_rows():
    schedules = defaultdict(list)
    with FEE_REFERENCE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            schedules[row["fee"]].append(row)
    if set(schedules) != {"SEC_SECTION_31", "FINRA_TAF"}:
        raise ValueError("The SEC and FINRA schedules must both be present")
    for schedule in schedules.values():
        if schedule[0]["effective_from"] > FIRST_DATE:
            raise ValueError("The fee schedule starts after the quote archive")
        for index, row in enumerate(schedule):
            start, end = row["effective_from"], row["effective_to_exclusive"] or None
            date.fromisoformat(start)
            if end and date.fromisoformat(end) <= date.fromisoformat(start):
                raise ValueError(f"Nonpositive fee period: {row}")
            if index and schedule[index - 1]["effective_to_exclusive"] != start:
                raise ValueError(f"Noncontiguous fee periods: {row}")
            rate = Decimal(row["rate"])
            cap = Decimal(row["cap_usd_per_trade"]) if row["cap_usd_per_trade"] else None
            if rate < 0 or (cap is not None and cap < 0):
                raise ValueError(f"Negative fee or cap: {row}")
            yield (row["fee"], start, end, row["date_basis"], str(rate), row["rate_unit"],
                   str(cap) if cap is not None else None, row["source_url"],
                   SEC_RATE_VERIFIED_THROUGH if row["fee"]=="SEC_SECTION_31" else AS_OF)
        if end and end <= AS_OF:
            raise ValueError("The fee schedule does not reach the source cutoff")


def rate_row(provider, series, currency, day, value, basis, flag, url, path):
    rate = Decimal(str(value))
    date.fromisoformat(day)
    if not FIRST_DATE <= day <= AS_OF or not rate.is_finite():
        raise ValueError(f"Invalid reference rate for {provider} {series} on {day}")
    return provider, series, currency, day, str(rate), basis, flag, url, str(path.relative_to(ROOT))


def market_rate_rows():
    for series, family in (("EFFR", "unsecured"), ("OBFR", "unsecured"), ("SOFR", "secured")):
        path = SOURCE / f"nyfed_{series.lower()}.json"
        url = f"{NY_FED}/{family}/{series.lower()}/search.json?startDate={FIRST_DATE}&endDate={AS_OF}&type=rate"
        rows = json.loads(path.read_bytes())["refRates"]
        if not rows or {row["type"] for row in rows} != {series}:
            raise ValueError(f"Missing or mismatched NY Fed rates in {path}")
        for row in rows:
            yield rate_row("NY_FED", series, "USD", row["effectiveDate"], row["percentRate"],
                           "public_overnight_reference", row.get("revisionIndicator") or None, url, path)

    path = SOURCE / "boc_corra.json"
    url = f"{BOC_CORRA}?start_date={FIRST_DATE}&end_date={AS_OF}"
    payload = json.loads(path.read_bytes())
    if "AVG.INTWO" not in payload["seriesDetail"] or not payload["observations"]:
        raise ValueError("No Bank of Canada CORRA observations")
    for row in payload["observations"]:
        yield rate_row("BANK_OF_CANADA", "CORRA", "CAD", row["d"], row["AVG.INTWO"]["v"],
                       "public_overnight_reference", None, url, path)


def ibkr_downloads():
    for path in sorted((SOURCE / "ibkr_benchmark_months").glob("*.csv")):
        month = path.stem
        datetime.strptime(month, "%Y%m")
        url = f"{IBKR_MONTHLY}?date={month}&cvs=1&ib_entity=llc"
        header, *lines = path.read_text().splitlines()
        columns = header.split()
        if columns[0] != "Date" or not {"USD", "CAD"} <= set(columns):
            raise ValueError(f"Unexpected IBKR rate header in {path}")
        days = []
        rows = []
        for line in filter(str.strip, lines):
            fields = line.split()
            if len(fields) != len(columns):
                raise ValueError(f"Mismatched IBKR rate columns in {path}: {line}")
            day = datetime.strptime(fields[0], "%Y%m%d").date().isoformat()
            days.append(day)
            if fields[0][:6] != month or day > AS_OF:
                continue
            for currency in ("CAD", "USD"):
                raw = fields[columns.index(currency)]
                value = raw.rstrip("*%")
                if value == "N/A":
                    continue
                if value.startswith("(") and value.endswith(")"):
                    value = "-" + value[1:-1]
                flag = "asterisk_on_source" if "*" in raw else None
                rows.append(rate_row("IBKR_LLC", "IBKR_BENCHMARK", currency, day, value,
                                     "broker_published_currency_benchmark", flag, url, path))
        if len(set(days)) != len(days):
            raise ValueError(f"Repeated dates in {path}")
        dates_in_month = sum(day[:7].replace("-", "") == month for day in days)
        yield (month, len(days), dates_in_month, min(days, default=None), max(days, default=None),
               url, str(path.relative_to(ROOT))), rows


def commission_rows():
    # Transcribed from the United States and Canada tables in the archived IBKR page.
    source_file = str((SOURCE / "ibkr_stock_commission.html").relative_to(ROOT))
    schedules = (
        ("US", "USD", "0.35", "1.00", "1.0", "0.005", (
            (300_000, "0.0035"), (3_000_000, "0.0020"), (20_000_000, "0.0015"),
            (100_000_000, "0.0010"), (None, "0.0005"))),
        ("CA", "CAD", "1.00", "1.00", "0.5", "0.01", (
            (300_000, "0.008"), (3_000_000, "0.005"), (20_000_000, "0.004"), (None, "0.003"))),
        ("CA", "USD", "0.80", "0.80", "0.4", "0.008", (
            (300_000, "0.006"), (3_000_000, "0.004"), (20_000_000, "0.003"), (None, "0.002"))),
    )
    for market, currency, tier_min, fixed_min, cap, fixed_rate, tiers in schedules:
        yield (SNAPSHOT_DATE, market, currency, "PRO_FIXED", 1, None, fixed_rate, fixed_min, cap,
               IBKR_COMMISSIONS, source_file)
        first = 1
        for last, rate in tiers:
            yield (SNAPSHOT_DATE, market, currency, "PRO_TIERED", first, last, rate, tier_min, cap,
                   IBKR_COMMISSIONS, source_file)
            if last is not None:
                first = last + 1


def add_execution_cost_inputs(db):
    coverage, ibkr_rates = zip(*ibkr_downloads(), strict=True)
    rows_by_table = {
        "us_equity_regulatory_fee_period": list(fee_rows()),
        "equity_financing_reference_rate": [*market_rate_rows(), *(row for month in ibkr_rates for row in month)],
        "ibkr_benchmark_download_coverage": coverage,
        "ibkr_equity_commission_snapshot": list(commission_rows()),
    }
    for table, rows in rows_by_table.items():
        if not rows:
            raise ValueError(f"Missing execution data for {table}")
        schema, _ = TABLES[table]
        db.execute(f"CREATE TABLE IF NOT EXISTS {table} ({schema})")
        db.execute(f"DELETE FROM {table}")
        placeholders = ",".join("?" for _ in rows[0])
        db.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)


def export_execution_cost_inputs(db):
    for table, (_, order_by) in TABLES.items():
        export(db, table, order_by)


def main():
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        db.execute("BEGIN IMMEDIATE")
        with db:
            add_execution_cost_inputs(db)
        export_execution_cost_inputs(db)
        for table in TABLES:
            print(table, db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])


if __name__ == "__main__":
    main()
