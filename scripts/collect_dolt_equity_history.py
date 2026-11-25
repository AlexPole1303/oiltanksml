"""Archive daily OHLCV for historical listings using a pinned public Dolt snapshot."""
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

from collect_public import RAW, ROOT, get, write, write_csv

REF = "50h6rm49ibkc0k6j8usqdku139den5ic"
API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/stocks/master?"
DERIVED = ROOT / "data" / "derived"
STORE = RAW / "equity_reference" / "dolt" / REF
PRICE_FIELDS = ("open", "high", "low", "close", "volume")


def pinned_dividend_rows():
    path = STORE / "dividend.json"
    response = json.loads(path.read_text())
    if response.get("query_execution_status") != "Success" or f"dividend AS OF '{REF}'" not in response.get("sql_query", ""):
        raise ValueError("Historical dividend archive is not pinned to the expected revision")
    return response["rows"], path


def query(sql, name):
    source = API + urlencode({"q": sql})
    path = STORE / f"{name}.json"
    payload = get(source, path)
    response = json.loads(payload)
    if response.get("query_execution_status") != "Success":
        path.unlink(missing_ok=True)
        raise ValueError(f"Dolt query {name} failed: {response.get('query_execution_message')}")
    return response["rows"], path


def day_batch(days, symbols):
    dates = ",".join(repr(day) for day in days)
    tickers = ",".join(repr(symbol) for symbol in symbols)
    where = f"date IN ({dates}) AND act_symbol IN ({tickers})"
    return query(f"SELECT * FROM ohlcv AS OF '{REF}' WHERE {where} ORDER BY date,act_symbol",
                 f"ohlcv_{days[0]}_{days[-1]}")


def main():
    with (ROOT / "data" / "reference" / "historical_equity_listing_windows.csv").open() as file:
        listings = list(csv.DictReader(file))
    with (DERIVED / "equity_session_calendar.csv").open() as file:
        calendar = [row["session_date"] for row in csv.DictReader(file) if row["venue"] == "XNYS"]
    symbols = sorted({row["symbol"] for row in listings})
    before = min(row["first_study_session"] for row in listings)
    after = max(row["last_study_session"] for row in listings)
    dates = [day for day in sorted(calendar) if before <= day <= after]
    expected = {(row["symbol"], date): row for row in listings for date in dates
                if row["first_study_session"] <= date <= row["last_study_session"]}
    if len(expected) != sum(row["first_study_session"] <= day <= row["last_study_session"]
                            for row in listings for day in dates):
        raise ValueError("Expected listing windows overlap for the same symbol")
    docs, _ = query("SELECT * FROM dolt_docs WHERE doc_name='LICENSE.md'", "provider_license")
    if docs and docs[0].get("doc_text"):
        write(STORE / "LICENSE.md", docs[0]["doc_text"].encode())
    tickers = ",".join(repr(symbol) for symbol in symbols)
    for table, clock in (("symbol", None), ("dividend", "ex_date"), ("split", "ex_date")):
        condition = f" AND {clock} BETWEEN '{before}' AND '{after}'" if clock else ""
        query(f"SELECT * FROM {table} AS OF '{REF}' WHERE act_symbol IN ({tickers}){condition}", table)
    jobs = [dates[i:i+42] for i in range(0, len(dates), 42)]
    selected = {}
    with ThreadPoolExecutor(max_workers=4) as workers:
        for index, (rows, path) in enumerate(workers.map(lambda days: day_batch(days, symbols), jobs), 1):
            for row in rows:
                key = row["act_symbol"], row["date"]
                if key not in expected:
                    continue
                if key in selected:
                    raise ValueError(f"Duplicate date and symbol from Dolt: {key}")
                selected[key] = (row, path)
            if index % 10 == 0:
                print(f"{index}/{len(jobs)} market-history batches", flush=True)
    fields = ("listing_id", "symbol", "session_date", *PRICE_FIELDS, "exchange", "currency",
              "source_revision", "source_file", "issuer_identity_source")
    rows = [(listing["listing_id"], *key, *(selected[key][0][field] for field in PRICE_FIELDS),
             listing["exchange"], "USD", REF, str(selected[key][1].relative_to(ROOT)), listing["issuer_identity_source"])
            for key, listing in sorted(expected.items()) if key in selected]
    write_csv(DERIVED / "dolt_historical_equity_bars.csv", fields, rows)
    coverage = []
    for listing in listings:
        symbol, identity = listing["symbol"], listing["listing_id"]
        want = [day for day in dates if expected.get((symbol, day), {}).get("listing_id") == identity]
        have = [day for day in want if (symbol, day) in selected]
        coverage.append((identity, symbol, len(want), len(have), min(have, default=None), max(have, default=None),
                         len(want) - len(have), REF, listing["issuer_identity_source"]))
    write_csv(DERIVED / "dolt_historical_equity_coverage.csv", ("listing_id", "symbol", "calendar_sessions", "bars",
               "first_bar", "last_bar", "missing_sessions", "source_revision", "issuer_identity_source"), coverage)
    print(f"{len(rows)} dated equity bars across {len(listings)} listing windows")


if __name__ == "__main__":
    main()
