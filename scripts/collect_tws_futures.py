"""Download available daily WTI contract bars from the local read-only TWS session."""
import argparse
import dataclasses
import datetime as dt
import json
from zoneinfo import ZoneInfo

from ib_async import Contract, IB, StartupFetch

from collect_public import RAW, ROOT, write


def download(ib, item, as_of, output, duration):
    month = item["contract_month"]
    destination = output / f"CL_{month}.json"
    if destination.is_file():
        cached = json.loads(destination.read_bytes())
        if cached["bars"]:
            return len(cached["bars"])
    candidate = Contract(conId=item["contract_id"], secType="FUT", exchange=item["exchange"],
                         currency="USD", includeExpired=True)
    contracts = ib.qualifyContracts(candidate)
    if (len(contracts) != 1 or contracts[0].symbol != "CL" or contracts[0].conId != item["contract_id"]
        or not contracts[0].lastTradeDateOrContractMonth.startswith(item["last_trading_date"])):
        raise ValueError("TWS did not return the requested CL contract")
    contract = contracts[0]
    expiry = dt.datetime.strptime(item["last_trading_date"], "%Y%m%d").date()
    cutoff = dt.datetime.combine(min(expiry + dt.timedelta(days=1), as_of),
                                 dt.time(23, 59, 59), ZoneInfo("America/Chicago"))
    bars = ib.reqHistoricalData(contract, cutoff, duration, "1 day", "TRADES", False, timeout=30)
    payload = {
        "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": dataclasses.asdict(contract),
        "catalog_contract": item,
        "what_to_show": "TRADES",
        "request_end": cutoff.isoformat(),
        "duration": duration,
        "regular_hours": False,
        "bars": [dataclasses.asdict(bar) for bar in bars],
    }
    write(destination, json.dumps(payload, default=str, separators=(",", ":")).encode())
    return len(bars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--first", default="202412")
    parser.add_argument("--last", default="202712")
    parser.add_argument("--as-of", type=dt.date.fromisoformat, default=dt.date(2026, 11, 10))
    parser.add_argument("--snapshot", type=dt.date.fromisoformat, help="Save a one-week supplement alongside the original files")
    args = parser.parse_args()
    catalog = json.loads((RAW / "ibkr" / "cl_contracts.json").read_bytes())["contracts"]
    contracts = sorted((item for item in catalog if args.first <= item["contract_month"] <= args.last),
                       key=lambda item: item["contract_month"])
    output = RAW / "tws_futures" if args.snapshot is None else RAW / "tws_futures_snapshots" / args.snapshot.isoformat()
    duration = "1 Y" if args.snapshot is None else "1 W"
    ib = IB()
    ib.RequestTimeout = 15
    ib.connect("127.0.0.1", args.port, clientId=783, timeout=8, readonly=True, fetchFields=StartupFetch(0))
    path = ROOT / "data" / "logs" / "tws_futures.jsonl"
    try:
        with path.open("a") as stream:
            for item in contracts:
                row = {"contract_month": item["contract_month"], "conid": item["contract_id"]}
                try:
                    row["bars"] = download(ib, item, args.as_of, output, duration)
                except Exception as error:
                    row["error"] = str(error)
                print(json.dumps(row), file=stream, flush=True)
                print(json.dumps(row), flush=True)
                ib.sleep(0.5)
    finally:
        ib.disconnect()


if __name__ == "__main__":
    main()
