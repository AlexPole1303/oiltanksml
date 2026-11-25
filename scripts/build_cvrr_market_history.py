"""Publish verified historical NYSE CVRR price coverage, cash events, and listing boundaries."""
import datetime as dt
import json
import sqlite3
from contextlib import closing
from decimal import Decimal

from build_market_event_panel import add_market_study_quotes
from build_market_transaction_timeline import reference_rows, source_text
from collect_dolt_equity_history import REF, STORE
from collect_early_equity_history import (
    TICKERTECH_HISTORY, TICKERTECH_PATH, nyse_ex_date_from_record, tickertech_date_query, tickertech_quote,
)
from collect_equity_reference import chart_exchange_daily_rows
from collect_public import RAW, ROOT
from publication_clocks import export

CHART = RAW / "equity_reference" / "cvrr_chartexchange_daily.html"
CHART_URL = "https://chartexchange.com/symbol/nyse-cvrr/historical/"
DOCUMENTS = RAW / "documents"
TICKERTECH = TICKERTECH_PATH.parent / "CVRR"
VIEW = """
CREATE VIEW IF NOT EXISTS cvrr_market_session_coverage AS
SELECT c.session_date,c.open_utc,c.close_utc,q.open,q.high,q.low,q.close,q.volume,
    q.provider,q.price_basis,q.source_file,d.reported_amount AS issuer_cash_on_ex_date,
    d.record_date AS distribution_record_date,d.source_url AS distribution_source_url,
    CASE WHEN q.session_date IS NULL THEN 'missing_observed_bar' ELSE 'observed_ohlcv' END AS observation_status
FROM historical_equity_listing_window l JOIN equity_session_calendar c
    ON c.venue=l.exchange AND c.session_date BETWEEN l.first_study_session AND l.last_study_session
LEFT JOIN market_study_quote q ON q.instrument=l.symbol AND q.session_date=c.session_date
LEFT JOIN issuer_cash_distribution d ON d.instrument=l.symbol AND d.ex_date=c.session_date
WHERE l.listing_id='cvr_refining';
"""


def add_cvrr_history(db):
    first, last = db.execute("""SELECT first_study_session,last_study_session FROM historical_equity_listing_window
        WHERE listing_id='cvr_refining'""").fetchone()
    final = " ".join(source_text(DOCUMENTS / "market_transactions/cvrr_call_completed.html").split())
    if "no longer listed on the NYSE beginning as of January 29, 2019" not in final or "$10.50 per unit" not in final:
        raise ValueError("CVRR closing consideration or NYSE withdrawal is unverified")
    identity = json.loads((STORE / "cvrr_symbol.json").read_text())
    symbol, = identity["rows"]
    if (identity["query_execution_status"] != "Success" or REF not in identity["sql_query"]
            or symbol["act_symbol"] != "CVRR" or symbol["listing_exchange"] != "NYSE"
            or not symbol["security_name"].startswith("CVR Refining, LP Common Units")):
        raise ValueError("CVRR listing identity disagrees with the pinned archive")
    calendar = [day for day, in db.execute("""SELECT session_date FROM equity_session_calendar
        WHERE venue='XNYS' AND session_date BETWEEN ? AND ? ORDER BY session_date""", (first, last))]
    quotes, outside = chart_exchange_daily_rows(CHART, set(calendar))
    if outside or last not in quotes:
        raise ValueError("The CVRR archive does not match the SEC-confirmed listing window")
    observed = {day: (values, "chartexchange_cvrr", CHART, CHART_URL) for day, values in quotes.items()}
    for path in sorted(TICKERTECH.glob("*.js")):
        day = path.stem
        if day not in calendar:
            raise ValueError(f"CVRR archive bar outside the listing calendar for {day}")
        values = tickertech_quote(path.read_bytes(), day, "CVRR")
        if values is None:
            raise ValueError(f"CVRR archive quote has no verified symbol and date for {day}")
        if day in quotes and tuple(map(float, quotes[day][:4])) != tuple(values[:4]):
            raise ValueError(f"CVRR archives disagree on OHLC for {day}")
        url = TICKERTECH_HISTORY.replace("ticker=JPEP", "ticker=CVRR") + tickertech_date_query(day)
        observed.setdefault(day, (values, "tickertech_cvrr", path, url))
    quote_rows = []
    for day, ((opening, high, low, close, volume), provider, path, url) in sorted(observed.items()):
        if not 0 < low <= min(opening, close) <= max(opening, close) <= high or volume <= 0:
            raise ValueError(f"Invalid CVRR OHLCV on {day}")
        quote_rows.append(("CVRR", day, float(opening), float(high), float(low), float(close), volume, "USD",
                           provider, "CVRR", str(path.relative_to(ROOT)), url))
    payload = json.loads((STORE / "cvrr_dividends.json").read_text())
    if payload["query_execution_status"] != "Success" or REF not in payload["sql_query"]:
        raise ValueError("Unverified CVRR dividend archive")
    vendor = {r["ex_date"]: Decimal(r["amount"]) for r in payload["rows"] if r["act_symbol"] == "CVRR"}
    cash_rows = reference_rows("cvrr_cash_distributions")
    if {r["ex_date"]: Decimal(r["reported_amount"]) for r in cash_rows} != vendor:
        raise ValueError("CVRR cash distributions disagree with the pinned archive")
    for row in cash_rows:
        ex_date = row["ex_date"]
        if nyse_ex_date_from_record(calendar, row["record_date"])[0] != ex_date:
            raise ValueError(f"CVRR ex-date does not match the NYSE calendar for {ex_date}")
        body = " ".join(source_text(DOCUMENTS / "cvrr_distributions" / f"{ex_date}.html").split())
        dates = [dt.date.fromisoformat(row[name]) for name in ("declared_date", "record_date", "payable_date")]
        markers = [f"{day:%B} {day.day}, {day.year}" for day in dates]
        markers.append(f"${row['reported_amount']} per common unit")
        if any(marker not in body for marker in markers):
            raise ValueError(f"SEC filing does not corroborate the CVRR cash event on {ex_date}")
    db.execute("DELETE FROM equity_reference_bar WHERE instrument='CVRR'")
    db.execute("DELETE FROM issuer_cash_distribution WHERE instrument='CVRR'")
    db.executemany("INSERT INTO equity_reference_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", quote_rows)
    db.executemany("INSERT INTO issuer_cash_distribution VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (tuple(value or None for value in row.values()) for row in cash_rows))
    db.execute(VIEW)


def main():
    with closing(sqlite3.connect(ROOT / "data/derived/cushing.sqlite")) as db:
        db.execute("BEGIN IMMEDIATE")
        with db:
            for table, name, key in (
                ("historical_equity_listing_window", "historical_equity_listing_windows", "symbol"),
                ("equity_merger_consideration", "equity_merger_consideration", "retiring_instrument"),
                ("terminal_company_history", "terminal_company_history", "ticker"),
            ):
                db.execute(f"DELETE FROM {table} WHERE {key}='CVRR'")
                row, = (r for r in reference_rows(name) if r[key] == "CVRR")
                placeholders = ",".join("?" for _ in row)
                db.execute(f"INSERT INTO {table} VALUES ({placeholders})", tuple(v or None for v in row.values()))
            add_cvrr_history(db)
            db.execute("INSERT OR IGNORE INTO market_study_instrument VALUES ('CVRR','XNYS','USD','historical_operator')")
            db.execute("DELETE FROM market_study_quote WHERE instrument='CVRR'")
            add_market_study_quotes(db, "CVRR")
        for table, order in (
            ("cvrr_market_session_coverage", "session_date"),
            ("market_study_quote", "instrument,session_date"),
            ("eia_equity_horizon_input", "release_date,instrument"),
            ("eia_event_operator_exposure", "release_date,terminal_id,ticker"),
            ("eia_roof_cohort_weekly_coverage", "week_ending,cohort_id,ticker"),
            ("terminal_data_coverage", "terminal_id"),
        ):
            export(db, table, order)
        print(dict(db.execute("""SELECT observation_status,count(*) FROM cvrr_market_session_coverage
            GROUP BY observation_status""")))


if __name__ == "__main__":
    main()
