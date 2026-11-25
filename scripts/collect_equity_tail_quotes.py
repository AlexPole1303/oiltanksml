"""Close the last five-session study window using overlapping reference quotes."""
import argparse
import datetime as dt
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from build_market_event_panel import COMPARISONS, CONTROLS, CONTROL_CASH_SCHEMA, CORE, add_market_study_quotes
from collect_public import RAW, ROOT, get, write_csv
from publication_clocks import export

DERIVED = ROOT / "data" / "derived"
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/"


def broker_tail(symbol, day, history, yahoo_close):
    path = RAW / "equity_reference" / "ibkr_tail" / f"{symbol}_{day}.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_bytes())
    response = raw["response"]
    if raw["request"]["outside_rth"] or response.get("source") != "Last" or response["chart_step"] != 86400:
        raise ValueError("Unexpected IBKR daily quote basis")
    dates = [dt.datetime.fromisoformat(value).astimezone(ZoneInfo("America/New_York")).date().isoformat()
             for value in response["time"]]
    overlap = history.keys() & set(dates)
    if len(overlap) < 3 or any(abs(response["close"][dates.index(date)]-history[date]) > 0.005 for date in overlap):
        raise ValueError("IBKR connector history differs from the stored TWS series")
    index = dates.index(day.isoformat())
    values = tuple(response[field][index] for field in ("open", "high", "low", "close", "volume"))
    if abs(values[3] - yahoo_close) > 0.005:
        raise ValueError("IBKR and independent close disagree for the requested session")
    return values, path


def quote(symbol, day, history, calendar_close):
    requested = "KEY.TO" if symbol == "KEY" else symbol
    currency = "CAD" if symbol == "KEY" else "USD"
    first = day - dt.timedelta(days=9)
    last = day + dt.timedelta(days=1)
    periods = {f"period{idx}": int(dt.datetime.combine(date, dt.time(), dt.timezone.utc).timestamp())
               for idx, date in enumerate((first, last), 1)}
    source = CHART + requested + "?" + urlencode({**periods, "interval": "1d", "events": "div,splits"})
    path = RAW / "equity_reference" / "yahoo_tail" / f"{requested}_{day}.json"
    result, = json.loads(get(source, path))["chart"]["result"]
    meta = result["meta"]
    if (meta["symbol"], meta["currency"]) != (requested, currency):
        raise ValueError("Unexpected issuer or quote currency")
    if result.get("events", {}).get("splits"):
        raise ValueError("Quote window has a split requiring a separate share-basis check")
    if dt.datetime.now(dt.timezone.utc) <= dt.datetime.fromisoformat(calendar_close):
        raise ValueError("Requested exchange session is still open")
    zone = ZoneInfo(meta["exchangeTimezoneName"])
    dates = [dt.datetime.fromtimestamp(epoch, zone).date().isoformat() for epoch in result["timestamp"]]
    observations, = result["indicators"]["quote"]
    for date, reference in history.items():
        if date not in dates or abs(observations["close"][dates.index(date)] - reference) > 0.005:
            raise ValueError(f"Different closing price on overlapping session {date}")
    if len(history) < 3:
        raise ValueError("Too few overlapping IBKR closes for an independent comparison")
    index = dates.index(day.isoformat())
    opening, high, low, close, volume = (observations[field][index] for field in ("open", "high", "low", "close", "volume"))
    selected = broker_tail(symbol, day, history, close)
    provider = "yahoo_finance_tail" if day == dt.date(2026, 11, 11) else "yahoo_finance_reference_session"
    if selected is not None:
        (opening, high, low, close, volume), path = selected
        provider = "ibkr_connector_tail"
    if not all(math.isfinite(x) for x in (opening, high, low, close, volume)) or not 0 < low <= min(opening, close) <= max(opening, close) <= high or volume <= 0:
        raise ValueError("Invalid regular-session OHLCV")
    return (symbol, day.isoformat(), opening, high, low, close, volume, currency, provider, requested,
            str(path.relative_to(ROOT)), source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    symbols = CORE + CONTROLS + COMPARISONS
    parser.add_argument("--session", default="2026-11-11", type=dt.date.fromisoformat)
    parser.add_argument("--symbol", choices=symbols, action="append")
    parser.add_argument("--output", type=Path, default=DERIVED / "equity_tail_reference.csv")
    parser.add_argument("--refresh-database", action="store_true")
    args = parser.parse_args()
    symbols = args.symbol or symbols
    requests = []
    with sqlite3.connect(DERIVED / "cushing.sqlite") as db:
        for symbol in symbols:
            venue = "XTSE" if symbol == "KEY" else "XNYS"
            calendar_close, = db.execute("SELECT close_utc FROM equity_session_calendar WHERE venue=? AND session_date=?",
                                         (venue, args.session.isoformat())).fetchone()
            history = dict(db.execute("""SELECT session_date,close FROM tws_equity_research_bar
                WHERE instrument=? AND series='TRADES' AND session_date<? ORDER BY session_date DESC LIMIT 4""",
                (symbol, args.session.isoformat())))
            requests.append((symbol, args.session, history, calendar_close))
    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(quote, *params): params[0] for params in requests}
        for future, symbol in jobs.items():
            try:
                row = future.result()
            except (OSError, ValueError, TypeError, KeyError) as error:
                print(f"{symbol}: {type(error).__name__}: {error}")
                continue
            rows.append(row)
    write_csv(args.output, ("instrument", "session_date", "open", "high", "low", "close",
              "volume", "currency", "provider", "requested_symbol", "source_file", "verification_source_url"), sorted(rows))
    if args.refresh_database and rows:
        with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
            db.executescript(CONTROL_CASH_SCHEMA)
            db.execute("BEGIN IMMEDIATE")
            with db:
                for row in rows:
                    symbol, day = row[:2]
                    existing = db.execute("SELECT 1 FROM market_study_quote WHERE instrument=? AND session_date=?",
                                          (symbol, day)).fetchone()
                    if existing is None:
                        db.execute("INSERT OR IGNORE INTO equity_reference_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
                        db.execute("DELETE FROM market_study_quote WHERE instrument=?", (symbol,))
                        add_market_study_quotes(db, symbol)
            export(db, "market_study_quote", "instrument,session_date")
            export(db, "eia_equity_horizon_input", "release_date,instrument")
    print(f"{len(rows)}/{len(symbols)} session quotes passed IBKR overlap and identity checks")


if __name__ == "__main__":
    main()
