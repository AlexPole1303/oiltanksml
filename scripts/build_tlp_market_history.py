"""Verify and publish the observed NYSE TLP listing history and cash events."""
import csv
import datetime as dt
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal
from html import unescape

from build_market_event_panel import add_market_study_quotes
from collect_dolt_equity_history import REF, STORE
from collect_early_equity_history import (
    HISTORICAL_SOURCE_COVERAGE_VIEW, TICKERTECH_HISTORY, TICKERTECH_PATH, add_adjusted_equity,
    nyse_ex_date_from_record, tickertech_date_query, tickertech_quote,
)
from collect_equity_reference import chart_exchange_daily_rows
from collect_public import RAW, ROOT, write, write_csv
from publication_clocks import export

DERIVED = ROOT / "data/derived"
REFERENCE = ROOT / "data/reference"
DOCUMENTS = RAW / "documents"
LISTING = "transmontaigne_partners"
FIRST, LAST = "2016-01-04", "2019-02-26"
CHART_FILE = RAW / "equity_reference/tlp_chartexchange_daily.html"
CHART_URL = "https://chartexchange.com/symbol/nyse-tlp/historical/"
EDGAR = "https://www.sec.gov/Archives/edgar/data/1319229/"
PROXY_URL = EDGAR + "000104746919000232/a2237577zdefm14a.htm"
MERGER_URL = EDGAR + "000110465919010579/a19-5348_1ex99d1.htm"
SEC_DATE = r"[A-Z][a-z]+ \d{1,2}, \d{4}"
DATE_AMOUNT = re.compile(r"declared a distribution of \$(\d+\.\d+) per unit.*?"
                         rf"payable on ({SEC_DATE}) to unitholders of record on ({SEC_DATE})", re.I)

VIEWS = (
    """CREATE VIEW IF NOT EXISTS tlp_market_session_coverage AS
       SELECT c.session_date,c.open_utc,c.close_utc,q.open,q.high,q.low,q.close,q.volume,q.provider,
           q.price_basis,q.source_file,a.adjusted_close AS adjusted_archive_close,a.source_file AS adjusted_archive_file,
           q.issuer_cash_on_ex_date,q.issuer_cash_source_url,
           CASE WHEN q.close IS NOT NULL THEN 'observed_ohlcv'
                WHEN a.adjusted_close IS NOT NULL THEN 'adjusted_archive_only'
                ELSE 'missing' END AS observation_status
       FROM historical_equity_listing_window l JOIN equity_session_calendar c
         ON c.venue=l.exchange AND c.session_date BETWEEN l.first_study_session AND l.last_study_session
       LEFT JOIN market_study_quote q ON q.instrument=l.symbol AND q.session_date=c.session_date
       LEFT JOIN historical_equity_adjusted_archive a ON a.listing_id=l.listing_id AND a.session_date=c.session_date
       WHERE l.listing_id='transmontaigne_partners'""",
    """CREATE VIEW IF NOT EXISTS tlp_equity_total_return AS
       WITH adjacent AS (
           SELECT *,lag(close) OVER (ORDER BY session_date) AS previous_session_close
           FROM tlp_market_session_coverage
       )
       SELECT session_date,open,close,previous_session_close,issuer_cash_on_ex_date,issuer_cash_source_url,
           (close+coalesce(CAST(issuer_cash_on_ex_date AS REAL),0))/nullif(previous_session_close,0)-1
              AS issuer_based_close_to_close_total_return,provider,price_basis
       FROM adjacent WHERE close IS NOT NULL""",
)


def repo_path(path):
    return str(path.relative_to(ROOT))


def dolt_rows(name):
    payload = json.loads((STORE / f"tlp_{name}.json").read_text())
    if payload.get("query_execution_status") != "Success" or REF not in payload["sql_query"]:
        raise ValueError(f"Unverified Dolt result for TLP {name}")
    return payload["rows"]


def tickertech_rows(calendar):
    for path in sorted((TICKERTECH_PATH.parent / "TLP").glob("*.js")):
        if path.stem in calendar:
            values = tickertech_quote(path.read_bytes(), path.stem, "TLP")
            if values is None:
                raise ValueError(f"TLP quote identity mismatch in {path.name}")
            yield path.stem, (*map(lambda value: Decimal(str(value)), values[:4]), values[4]), path


def distributions(calendar):
    filings = json.loads((DOCUMENTS / "tlp_sec_submissions.json").read_text())["filings"]["recent"]
    source = {day: (accession.replace("-", ""), name.replace("_18k.htm", "_1ex99d1.htm"), received)
              for day, accession, name, received, form in zip(*(filings[key] for key in
                  ("filingDate", "accessionNumber", "primaryDocument", "acceptanceDateTime", "form"))) if form == "8-K"}
    vendor = {r["ex_date"]: Decimal(r["amount"]) for r in dolt_rows("dividends")}
    if len(vendor) != 13 or dolt_rows("splits"):
        raise ValueError("Unexpected TLP corporate-action coverage from Dolt")
    calendar_dates = sorted(calendar)
    result = []
    for path in sorted((DOCUMENTS / "tlp_distributions").glob("*.html")):
        declared = path.stem
        accession, document, accepted_at = source[declared]
        text = " ".join(unescape(re.sub("<[^>]+>", " ", path.read_text())).split())
        match = DATE_AMOUNT.search(text)
        if not match:
            raise ValueError(f"TLP distribution could not be parsed for {declared}")
        amount, paid, record = match.groups()
        paid, record = (dt.datetime.strptime(value, "%B %d, %Y").date().isoformat() for value in (paid, record))
        estimated_ex, basis = nyse_ex_date_from_record(calendar_dates, record)
        if vendor.get(estimated_ex) != Decimal(amount) or not declared < estimated_ex < record < paid:
            raise ValueError(f"TLP distribution or calendar mismatch for {declared}")
        result.append(dict(instrument="TLP", ex_date=estimated_ex, record_date=record, declared_date=declared,
                           payable_date=paid, amount_usd=amount, source_url=f"{EDGAR}{accession}/{document}",
                           source_file=repo_path(path), sec_accepted_at_utc=accepted_at,
                           ex_date_source_file=repo_path(STORE / "tlp_dividends.json"),
                           expected_ex_date_basis=basis))
    if {row["ex_date"] for row in result} != set(vendor):
        raise ValueError("Missing SEC distribution evidence for TLP")
    return result


def add_tlp_history(db):
    calendar = {row[0] for row in db.execute("""SELECT session_date FROM equity_session_calendar
                   WHERE venue='XNYS' AND session_date BETWEEN ? AND ?""", (FIRST, LAST))}
    chart, excluded = chart_exchange_daily_rows(CHART_FILE, calendar)
    if not chart or max(chart) != LAST:
        raise ValueError("ChartExchange has no bar for the SEC-confirmed last TLP trading session")
    quotes = {day: (values, "chartexchange_tlp", CHART_FILE, CHART_URL) for day, values in chart.items()}
    comparisons = []
    for day, values, path in tickertech_rows(calendar):
        url = TICKERTECH_HISTORY.replace("ticker=JPEP", "ticker=TLP") + tickertech_date_query(day)
        if day in chart:
            comparisons.append((day, "tickertech", *values, "chartexchange", *chart[day]))
        else:
            quotes[day] = (values, "tickertech_tlp", path, url)
    symbols = dolt_rows("symbol")
    if len(symbols) != 1 or symbols[0]["listing_exchange"] != "NYSE" or "TransMontaigne" not in symbols[0]["security_name"]:
        raise ValueError("Dolt TLP listing identity mismatch")
    dolt_file = STORE / "tlp_quotes_feb2019.json"
    dolt_records = []
    for row in dolt_rows("quotes_feb2019"):
        day = row["date"]
        values = (*[Decimal(row[key]) for key in ("open", "high", "low", "close")], int(row["volume"]))
        if row["act_symbol"] != "TLP" or day not in chart or values[0] != chart[day][0] or values[3] != chart[day][3]:
            raise ValueError(f"Dolt and ChartExchange TLP open or close mismatch on {day}")
        comparisons.append((day, "dolt", *values, "chartexchange", *chart[day]))
        quotes[day] = (values, "dolt_tlp", dolt_file, f"https://www.dolthub.com/repositories/post-no-preference/stocks/data/{REF}/ohlcv")
        dolt_records.append((LISTING, "TLP", day, *map(float, values), "XNYS", "USD", REF, repo_path(dolt_file), MERGER_URL))
    if len(dolt_records) != 17 or quotes[LAST][0][3] != Decimal("40.99"):
        raise ValueError("Unexpected final NYSE TLP quote evidence")

    if quotes["2018-11-23"][0][3] != Decimal("36.40"):
        raise ValueError("TLP closing price conflicts with issuer proxy on 2018-11-23")
    cash = distributions(calendar)
    db.execute("DELETE FROM equity_reference_bar WHERE instrument='TLP'")
    db.execute("DELETE FROM issuer_cash_distribution WHERE instrument='TLP'")
    db.executemany("INSERT OR REPLACE INTO dolt_historical_equity_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", dolt_records)
    for day, (values, provider, path, url) in sorted(quotes.items()):
        opening, high, low, close, volume = values
        if not 0 < low <= min(opening, close) <= max(opening, close) <= high or volume <= 0:
            raise ValueError(f"Invalid TLP OHLCV for {day}")
        db.execute("INSERT INTO equity_reference_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   ("TLP", day, *map(float, values), "USD", provider, "TLP", repo_path(path), url))
    db.executemany("INSERT INTO issuer_cash_distribution VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (("TLP", row["ex_date"], row["record_date"], row["declared_date"], row["payable_date"], row["amount_usd"],
                     None, "USD", "issuer_sec_common_units_ex_date_dolt", row["source_url"]) for row in cash))
    for statement in VIEWS:
        db.execute(statement)
    return calendar, quotes, excluded, cash, comparisons


def selected_reference_rows(filename, key, value):
    with (REFERENCE / filename).open(newline="") as source:
        return [tuple(field or None for field in row.values()) for row in csv.DictReader(source) if row[key] == value]


def main():
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        db.execute("BEGIN IMMEDIATE")
        with db:
            db.execute("DELETE FROM terminal_company_history WHERE terminal_id='transmontaigne_cushing'")
            db.executemany("INSERT INTO terminal_company_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           selected_reference_rows("terminal_company_history.csv", "terminal_id", "transmontaigne_cushing"))
            db.executemany("INSERT OR REPLACE INTO equity_merger_consideration VALUES (?,?,?,?,?,?,?,?)",
                           selected_reference_rows("equity_merger_consideration.csv", "retiring_instrument", "TLP"))
            db.execute("DELETE FROM historical_equity_adjusted_archive WHERE listing_id=?", (LISTING,))
            add_adjusted_equity(db, symbols=("TLP",))
            calendar, quotes, excluded, cash, comparisons = add_tlp_history(db)
            db.execute("INSERT OR IGNORE INTO market_study_instrument VALUES ('TLP','XNYS','USD','historical_operator')")
            db.execute("DELETE FROM market_study_quote WHERE instrument='TLP'")
            add_market_study_quotes(db, "TLP")
            db.execute("DROP VIEW IF EXISTS historical_equity_source_coverage")
            db.execute(HISTORICAL_SOURCE_COVERAGE_VIEW)
            status = dict(db.execute("SELECT observation_status,count(*) FROM tlp_market_session_coverage GROUP BY observation_status"))
            if status.get("observed_ohlcv") != len(quotes):
                raise ValueError("Selected TLP market rows disagree with observed quote count")
        for name in ("tlp_market_session_coverage", "tlp_equity_total_return", "market_study_quote", "eia_equity_horizon_input",
                     "eia_event_operator_exposure", "terminal_data_coverage", "historical_equity_source_coverage"):
            export(db, name)
    write_csv(DERIVED / "tlp_cash_distribution_evidence.csv", cash[0].keys(), (row.values() for row in cash))
    labels = ("provider", "open", "high", "low", "close", "volume")
    write_csv(DERIVED / "tlp_price_crosscheck.csv", ("session_date", *labels, *(f"reference_{label}" for label in labels)),
              sorted(comparisons))
    report = dict(listing_id=LISTING, eligible_sessions=len(calendar), session_status=status,
                  observed_bars_by_provider=dict(Counter(row[1] for row in quotes.values())),
                  excluded_chartexchange_dates=excluded, issuer_cash_events=len(cash),
                  first_observed=min(quotes), last_observed=max(quotes), last_close_usd=float(quotes[LAST][0][3]),
                  cash_at_merger_per_unit_usd=41, merger_source_url=MERGER_URL)
    write(DERIVED / "tlp_market_history_summary.json", json.dumps(report, indent=2).encode() + b"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
