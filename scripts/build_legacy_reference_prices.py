"""Fill historical quote gaps and verify ETE bars returned under the ET ticker."""
import csv
import sqlite3
from contextlib import closing

from build_market_event_panel import add_market_study_quotes
from collect_early_equity_history import TICKERTECH_HISTORY, TICKERTECH_PATH, tickertech_date_query, tickertech_quote
from collect_equity_reference import ARCHIVE, DERIVED, chart_exchange_daily_rows
from collect_public import ROOT, write_csv
from publication_clocks import export

LISTINGS = {"AMID": "american_midstream", "ETP": "energy_transfer_partners_after_sxl", "TEP": "tallgrass_partners",
            "SXL": "sunoco_logistics", "TEGP": "tallgrass_gp"}
WIDGET_SYMBOLS = ("AMID", "TEP", "SXL", "TEGP")
ISSUER_CLOSES = ROOT / "data/reference/historical_equity_price_anchors.csv"
ETE_TRANSITION = "https://www.sec.gov/Archives/edgar/data/1276187/000119312518303169/d639079dex993.htm"
FIELDS = ("instrument", "session_date", "open", "high", "low", "close", "volume", "currency",
          "provider", "requested_symbol", "source_file", "verification_source_url")


def archived_quotes(db, ticker, listing, issuer_closes):
    first, last = db.execute("""SELECT first_study_session,last_study_session
        FROM historical_equity_listing_window WHERE listing_id=? AND symbol=?""", (listing, ticker)).fetchone()
    calendar = {day for day, in db.execute("""SELECT session_date FROM equity_session_calendar
        WHERE venue='XNYS' AND session_date BETWEEN ? AND ?""", (first, last))}
    anchor = {day: tuple(prices) for day, *prices in db.execute("""SELECT session_date,open,high,low,close
        FROM dolt_historical_equity_quality WHERE listing_id=? AND quality_status='ohlcv_range_valid'""", (listing,))}
    observations = {}
    if ticker in ("ETP", "TEP"):
        path = ARCHIVE / f"{ticker.lower()}_chartexchange_daily.html"
        source = f"https://chartexchange.com/symbol/nyse-{ticker.lower()}/historical/"
        chart, outside = chart_exchange_daily_rows(path, calendar)
        if outside or last not in chart:
            raise ValueError(f"ChartExchange disagrees with the {listing} listing calendar")
        observations.update({day: (values, "chartexchange", path, source) for day, values in chart.items()})

    crosschecks = set()
    widget_files = sorted((TICKERTECH_PATH.parent / ticker).glob("*.js")) if ticker in WIDGET_SYMBOLS else ()
    for path in widget_files:
        day = path.stem
        values = tickertech_quote(path.read_bytes(), day, ticker)
        if day not in calendar or values is None:
            raise ValueError(f"Unverified {ticker} archive bar on {day}")
        if day in observations:
            check_prices(ticker, day, values, observations[day][0])
            crosschecks.add(day)
        url = TICKERTECH_HISTORY.replace("ticker=JPEP", f"ticker={ticker}") + tickertech_date_query(day)
        observations.setdefault(day, (values, f"tickertech_{ticker.lower()}", path, url))

    for day in observations.keys() & anchor.keys():
        check_prices(ticker, day, observations[day][0], anchor[day])
        crosschecks.add(day)
    for _, _, day, expected, _, _ in issuer_closes:
        if day in observations:
            if abs(observations[day][0][3] - float(expected)) > 0.00501:
                raise ValueError(f"Provider disagrees with the issuer-reported close for {ticker} on {day}")
            crosschecks.add(day)
    if not observations and ticker in ("SXL", "TEGP"):
        return
    if len(crosschecks) < 2:
        raise ValueError(f"Too few price anchors for {listing}")
    for day, (values, provider, path, url) in sorted(observations.items()):
        opening, high, low, close, volume = values
        if not 0 < low <= min(opening, close) <= max(opening, close) <= high or volume <= 0:
            raise ValueError(f"Invalid {ticker} OHLCV on {day}")
        if day not in anchor:
            yield (ticker, day, float(opening), float(high), float(low), float(close), volume,
                   "USD", provider, ticker, str(path.relative_to(ROOT)), url)


def check_prices(ticker, day, values, anchor):
    if any(abs(float(left) - float(right)) > 0.00501
           for left, right in zip(values[:4], anchor[:4], strict=True)):
        raise ValueError(f"Price sources disagree for {ticker} on {day}")


def ete_quotes(db):
    first, last = db.execute("""SELECT first_study_session,last_study_session
        FROM historical_equity_listing_window WHERE listing_id='energy_transfer_equity'""").fetchone()
    calendar = {day for day, in db.execute("""SELECT session_date FROM equity_session_calendar
        WHERE venue='XNYS' AND session_date BETWEEN ? AND ?""", (first, last))}
    prices = {day: row for day, *row in db.execute("""SELECT session_date,open,high,low,close,volume,conid,source_file
        FROM tws_equity_research_bar WHERE instrument='ET' AND series='TRADES'
        AND session_date BETWEEN ? AND ?""", (first, last))}
    if prices.keys() != calendar or {row[5] for row in prices.values()} != {37830920}:
        raise ValueError("The IBKR ET price history does not cover the documented ETE listing")
    anchors = list(db.execute("""SELECT session_date,open,high,low,close FROM dolt_historical_equity_quality
        WHERE listing_id='energy_transfer_equity' AND quality_status='ohlcv_range_valid'"""))
    if len(anchors) < 30:
        raise ValueError("Too few independent ETE price anchors")
    for day, *anchor in anchors:
        check_prices("ETE", day, prices[day], anchor)
    for day, (op, high, low, close, volume, _, path) in sorted(prices.items()):
        if not 0 < low <= min(op, close) <= max(op, close) <= high or volume <= 0:
            raise ValueError(f"Invalid ETE quote on {day}")
        yield "ETE", day, op, high, low, close, volume, "USD", "ibkr_tws_ete_reidentified", "ET", path, ETE_TRANSITION


def main():
    with (ROOT / "data/reference/market_study_symbol_history.csv").open(newline="") as stream:
        identities = [tuple(value or None for value in row.values()) for row in csv.DictReader(stream)
                      if row["instrument"] != "DINO"]
    with ISSUER_CLOSES.open(newline="") as stream:
        issuer_closes = [tuple(row.values()) for row in csv.DictReader(stream) if row["price_basis"] == "reported_close"]
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        rows = [row for ticker, listing in LISTINGS.items() for row in
                archived_quotes(db, ticker, listing, [item for item in issuer_closes if item[:2] == (listing, ticker)])]
        rows.extend(ete_quotes(db))
        db.execute("BEGIN IMMEDIATE")
        with db:
            db.executemany("INSERT OR REPLACE INTO market_study_symbol_history VALUES (?,?,?,?,?,?,?)", identities)
            db.executemany("INSERT OR REPLACE INTO historical_equity_price_anchor VALUES (?,?,?,?,?,?)", issuer_closes)
            db.executemany("INSERT OR REPLACE INTO equity_reference_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            for ticker in (*LISTINGS, "ETE", "ET"):
                db.execute("DELETE FROM market_study_quote WHERE instrument=?", (ticker,))
                add_market_study_quotes(db, ticker)
        export(db, "market_study_quote", "instrument,session_date")
        export(db, "eia_equity_horizon_input", "release_date,instrument")
        export(db, "historical_equity_source_coverage", "symbol,listing_id,session_date")
    write_csv(DERIVED / "legacy_equity_reference_bar.csv", FIELDS, rows)
    print({ticker: sum(row[0] == ticker for row in rows) for ticker in (*LISTINGS, "ETE")}, flush=True)


if __name__ == "__main__":
    main()
