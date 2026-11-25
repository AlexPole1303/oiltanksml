"""Request only contract details and historical market data from local Trader Workstation."""
import argparse
import dataclasses
import datetime as dt
import json
from zoneinfo import ZoneInfo

from ib_async import Contract, IB, StartupFetch, Stock

from collect_public import RAW, ROOT, write

START = dt.date(2016, 1, 1)
END = dt.date(2026, 11, 10)
SYMBOLS = ("PAA", "ENB", "ET", "OKE", "EPD", "NGL", "AMLP", "ENFR", "XLE", "SPY",
           "DINO", "SOBO", "TRP", "PAGP", "KEY", "CVI", "BKEP", "HEP", "MMP", "SEMG",
           "SXL", "ETP", "ETE", "KMI", "PSX", "TEP", "TGE", "TEGP", "PSXP",
           "USO", "BNO", "XOP", "WMB")
FIRST_LISTED = {"SOBO": dt.date(2024, 10, 1)}


def broker_contract(symbol):
    known_exchange = {"AMLP": "ARCA", "ENFR": "ARCA", "DINO": "NYSE", "SOBO": "NYSE",
                      "USO": "ARCA", "BNO": "ARCA", "XOP": "ARCA", "WMB": "NYSE"}
    if symbol in known_exchange:
        return Stock(symbol, "SMART", "USD", primaryExchange=known_exchange[symbol])
    root = RAW / "ibkr"
    day = root / f"{symbol.lower()}_daily.json"
    request = json.loads(day.read_bytes())["request"] if day.is_file() else {}
    conid = request.get("con_id") or request.get("contract_id")
    if conid:
        return Contract(conId=conid, secType="STK", exchange="SMART")
    if symbol == "KEY":
        return Stock(symbol, "SMART", "CAD", primaryExchange="TSE")
    return Stock(symbol, "SMART", "USD")


def report_to_disk(symbol, kind, end, contract, bars):
    path = RAW / "tws" / symbol / f"{kind}_{end}.json"
    payload = {
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": dataclasses.asdict(contract),
        "what_to_show": kind,
        "regular_hours": True,
        "bars": [dataclasses.asdict(bar) for bar in bars],
    }
    write(path, json.dumps(payload, default=str, separators=(",", ":")).encode())
    return len(bars)


def history(ib, symbol, start, end):
    if symbol == "TGE":
        raise ValueError("Ticker reused after Tallgrass was taken private; historical conId required")
    details = ib.reqContractDetails(broker_contract(symbol))
    if len(details) != 1:
        raise ValueError("TWS could not resolve a unique contract")
    contract = details[0].contract
    if contract.symbol.split(".")[0] != symbol:
        raise ValueError("TWS resolved a different symbol")
    start = max(start, FIRST_LISTED.get(symbol, start))
    count = {}
    for year in range(end.year, start.year - 1, -1):
        last = min(end, dt.date(year, 12, 31))
        path = RAW / "tws" / symbol / f"TRADES_{last}.json"
        if path.exists() and json.loads(path.read_bytes())["bars"]:
            continue
        cutoff = dt.datetime.combine(last, dt.time(23, 59, 59), ZoneInfo("America/New_York"))
        bars = ib.reqHistoricalData(
            contract, cutoff, "1 Y", "1 day", "TRADES", True, timeout=30,
        )
        count[str(last)] = report_to_disk(symbol, "TRADES", last, contract, bars)
        ib.sleep(0.5)
    durations = ("3 Y",) if symbol in FIRST_LISTED else ("11 Y", "10 Y", "5 Y")
    for duration in durations:
        path = RAW / "tws" / symbol / f"ADJUSTED_LAST_{duration.replace(' ', '')}.json"
        if path.exists() and json.loads(path.read_bytes())["bars"]:
            break
        bars = ib.reqHistoricalData(
            contract, None, duration, "1 day", "ADJUSTED_LAST", True, timeout=45,
        )
        count[f"adjusted:{duration}"] = report_to_disk(
            symbol, "ADJUSTED_LAST", duration.replace(" ", ""), contract, bars)
        if bars:
            break
        ib.sleep(0.5)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--symbol", choices=SYMBOLS, action="append")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=START)
    parser.add_argument("--end", type=dt.date.fromisoformat, default=END)
    args = parser.parse_args()
    ib = IB()
    ib.RequestTimeout = 15
    ib.connect("127.0.0.1", args.port, clientId=782, timeout=8, readonly=True,
               fetchFields=StartupFetch(0))
    log = ROOT / "data" / "logs" / "tws_download.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log.open("a") as stream:
            for symbol in args.symbol or SYMBOLS:
                row = {"instrument": symbol}
                try:
                    row["new_bars_by_request"] = history(ib, symbol, args.start, args.end)
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                print(json.dumps(row), file=stream, flush=True)
                print(json.dumps(row), flush=True)
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
