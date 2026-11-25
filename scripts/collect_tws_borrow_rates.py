"""Archive daily indicative IBKR FEE_RATE bars for dated study instruments."""
import argparse
import dataclasses
import datetime as dt
import json
import math
from zoneinfo import ZoneInfo

from ib_async import IB, StartupFetch

from build_market_event_panel import CORE
from collect_public import RAW, write
from collect_tws import broker_contract

STORE = RAW / "tws_execution_inputs"
STUDY_SYMBOLS = ("AMLP", *CORE, "ENFR", "SPY", "XLE", "XOP", "USO", "BNO",
                 "PAGP", "SOBO", "TRP", "DINO", "CVI", "PSX", "KEY", "KMI", "WMB")
EASTERN = ZoneInfo("America/New_York")


def fetch_year(ib, contract, start, end, errors):
    cutoff = dt.datetime.combine(end, dt.time(23, 59, 59), EASTERN)
    errors.clear()
    bars = ib.reqHistoricalData(contract, cutoff, "1 Y", "1 day", "FEE_RATE", useRTH=False, timeout=35)
    retained = [bar for bar in bars if start <= bar.date <= end]
    for bar in retained:
        if (not all(math.isfinite(value) for value in (bar.open, bar.high, bar.low, bar.close))
                or not bar.low <= min(bar.open, bar.close) <= max(bar.open, bar.close) <= bar.high):
            raise ValueError(f"Invalid indicative rates on {bar.date} for {contract.symbol}")
    if len({bar.date for bar in retained}) != len(retained):
        raise ValueError(f"Duplicate FEE_RATE dates for {contract.symbol}")
    return dict(
        downloaded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        contract=dataclasses.asdict(contract), end=cutoff.isoformat(), start_inclusive=start.isoformat(),
        duration="1 Y", bar_size="1 day", what_to_show="FEE_RATE", regular_hours=False,
        errors=list(errors), received_bars=len(bars), bars=[dataclasses.asdict(bar) for bar in retained],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--symbol", choices=STUDY_SYMBOLS, action="append")
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2017, 1, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 11, 11))
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("The requested start is after the requested end")
    ib = IB()
    ib.RequestTimeout = 45
    errors = []
    ib.errorEvent += lambda req, code, msg, contract: errors.append((req, code, msg))
    ib.connect("127.0.0.1", args.port, clientId=786, timeout=8, readonly=True, fetchFields=StartupFetch(0))
    try:
        for symbol in dict.fromkeys(args.symbol or ("AMLP", *CORE)):
            details = ib.reqContractDetails(broker_contract(symbol))
            if len(details) != 1 or details[0].contract.symbol.split(".")[0] != symbol:
                print(f"{symbol}: unique matching contract unavailable", flush=True)
                continue
            contract = details[0].contract
            for year in range(args.end.year, args.start.year - 1, -1):
                start = max(dt.date(year, 1, 1), args.start)
                end = min(dt.date(year, 12, 31), args.end)
                path = STORE / symbol / f"FEE_RATE_{start}_{end}.json"
                if path.is_file():
                    existing = json.loads(path.read_bytes())
                    if existing["contract"]["conId"] == contract.conId and existing["bars"]:
                        continue
                payload = fetch_year(ib, contract, start, end, errors)
                write(path, json.dumps(payload, default=str, separators=(",", ":")).encode())
                print(f"{symbol} {year}: {len(payload['bars'])} bars, codes "
                      f"{sorted({code for _, code, _ in errors if code not in (2104, 2106, 2158, 2174)})}", flush=True)
                ib.sleep(1.5)
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
