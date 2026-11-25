"""Collect NYSE historical bars through the provider's public date selector."""
import argparse
import json
import sqlite3
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from collect_early_equity_history import TICKERTECH_HISTORY, TICKERTECH_PATH, tickertech_date_query, tickertech_quote
from collect_public import ROOT, TLS, write

DB = ROOT / "data" / "derived" / "cushing.sqlite"
WIDGET = "https://www.tickertech.net/etfchannel/cgi/?head=1&js=on&tablewidth=600&a=historical&ticker="
DEFAULT_END = {"JPEP": "2017-03-07", "TLP": "2017-09-12", "CVRR": "2017-08-07",
               "AMID": "2018-09-06", "TEP": "2017-01-10", "SXL": "2017-04-28", "TEGP": "2018-06-29"}


def request_quote(day, symbol):
    params = tickertech_date_query(day)
    source = WIDGET + symbol + params + "&" + urlencode({"action": "https://www.historicalstockprice.com/history/"})
    referrer = TICKERTECH_HISTORY.replace("ticker=JPEP", f"ticker={symbol}") + params
    request = Request(source, headers={"User-Agent": "Mozilla/5.0", "Referer": referrer})
    with urlopen(request, timeout=25, context=TLS) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="JPEP", choices=DEFAULT_END)
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end")
    args = parser.parse_args()
    path = TICKERTECH_PATH.parent / args.symbol
    log_path = ROOT / "data" / "logs" / f"tickertech_{args.symbol.lower()}_daily.jsonl"
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as db:
        days = [day for day, in db.execute("""SELECT c.session_date FROM equity_session_calendar c
            JOIN historical_equity_listing_window l ON l.symbol=?
            WHERE c.venue='XNYS' AND c.session_date BETWEEN l.first_study_session AND l.last_study_session
            AND c.session_date BETWEEN ? AND ? ORDER BY c.session_date DESC""",
            (args.symbol, args.start, args.end or DEFAULT_END[args.symbol]))]
    unavailable = set()
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            row = json.loads(line)
            if row["status"] == "no_data":
                unavailable.add(row["date"])
    wanted = [day for day in days if day not in unavailable and not (path / f"{day}.js").is_file()]
    print(f"{len(wanted)} eligible dates to request", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        for index, day in enumerate(wanted, 1):
            for attempt in range(2):
                try:
                    payload = request_quote(day, args.symbol)
                except HTTPError as error:
                    if error.code != 429:
                        raise
                    payload = b"Server busy"
                if b"Server busy" not in payload:
                    break
                print(f"Provider requested a wait on {day}", flush=True)
                if attempt == 1:
                    raise SystemExit("Provider remained rate limited after one retry")
                time.sleep(61)
            quote = tickertech_quote(payload, day, args.symbol)
            if quote is not None:
                write(path / f"{day}.js", payload)
                status = "ready"
            elif b"heading('Historical quote not available')" in payload:
                status = "no_data"
            else:
                raise ValueError(f"Unexpected public widget response for {day}")
            log.write(json.dumps({"date": day, "status": status}) + "\n")
            log.flush()
            if index == 1 or index % 20 == 0 or index == len(wanted):
                print(f"{index}/{len(wanted)} {day} {status}", flush=True)
            if index < len(wanted):
                time.sleep(15)


if __name__ == "__main__":
    main()
