"""Archive five-minute IBKR bid and ask averages near the study's scheduled fills."""
import argparse
import dataclasses
import datetime as dt
import json
import math
from collections import defaultdict
from contextlib import closing

from ib_async import IB, StartupFetch

from build_market_event_panel import CORE
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from collect_tws import broker_contract
from collect_tws_borrow_rates import EASTERN, STORE

SYMBOLS = ("AMLP", *CORE)


def event_targets(symbol):
    months = defaultdict(set)
    with closing(read_only(SOURCE)) as db:
        venue, = db.execute("SELECT venue FROM market_study_instrument WHERE instrument=?", (symbol,)).fetchone()
        limit = db.execute("SELECT first_research_session FROM market_identity_limit WHERE instrument=?", (symbol,)).fetchone()
        for day, ordinal, opened, closed in db.execute("""
            SELECT session_date,ordinal,open_utc,close_utc FROM eia_event_market_session
            WHERE venue=? AND ordinal IN (1,5)
        """, (venue,)):
            if limit and day < limit[0]:
                continue
            instant = (dt.datetime.fromisoformat(opened) if ordinal == 1
                       else dt.datetime.fromisoformat(closed) - dt.timedelta(minutes=5))
            months[day[:7]].add(instant)
    return months


def planned_requests(symbol, repair, start_month, end_month):
    months = {month: targets for month, targets in event_targets(symbol).items()
              if (start_month is None or month >= start_month) and (end_month is None or month <= end_month)}
    if not repair:
        for month, targets in sorted(months.items(), reverse=True):
            yield f"events_{month}", "1 M", targets
        return
    requested, observed = set(), set()
    for path in (STORE / symbol).glob("BID_ASK_5min_*.json"):
        packet = json.loads(path.read_bytes())
        requested.update(dt.datetime.fromisoformat(value) for value in packet["expected_bar_starts_utc"])
        observed.update(dt.datetime.fromisoformat(row["date"]) for row in packet["bars"])
    missing = defaultdict(set)
    for stamp in requested - observed:
        day = stamp.astimezone(EASTERN).date().isoformat()
        if stamp in months.get(day[:7], ()):
            missing[day].add(stamp)
    for day, targets in sorted(missing.items()):
        local_starts = "-".join(t.astimezone(EASTERN).strftime("%H%M") for t in sorted(targets))
        yield f"repair_{day}_{local_starts}", "1 D", targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--symbol", choices=SYMBOLS, action="append")
    parser.add_argument("--repair", action="store_true", help="Request individual sessions omitted from archived responses")
    parser.add_argument("--start-month", type=parse_month)
    parser.add_argument("--end-month", type=parse_month)
    parser.add_argument("--client-id", type=int)
    args = parser.parse_args()
    if args.start_month and args.end_month and args.start_month > args.end_month:
        parser.error("The first month is after the last month")
    ib = IB()
    ib.RequestTimeout = 65
    errors = []
    ib.errorEvent += lambda req, code, msg, contract: errors.append((req, code, msg))
    client_id = args.client_id if args.client_id is not None else (791 if args.repair else 789)
    ib.connect("127.0.0.1", args.port, clientId=client_id, timeout=8,
               readonly=True, fetchFields=StartupFetch(0))
    try:
        for symbol in dict.fromkeys(args.symbol or SYMBOLS):
            detail, = ib.reqContractDetails(broker_contract(symbol))
            contract = detail.contract
            if contract.symbol.split(".")[0] != symbol:
                raise ValueError(f"Unexpected contract for {symbol}")
            for label, duration, target in planned_requests(symbol, args.repair, args.start_month, args.end_month):
                last = max(target).astimezone(EASTERN).date()
                end = dt.datetime.combine(last, dt.time(23, 59, 59), EASTERN)
                path = STORE / symbol / f"BID_ASK_5min_{label}.json"
                if path.is_file():
                    cached = json.loads(path.read_bytes())
                    if (cached["contract"]["conId"] == contract.conId and cached["bars"]
                            and cached["expected_bar_starts_utc"] == sorted(t.isoformat() for t in target)):
                        continue
                errors.clear()
                bars = ib.reqHistoricalData(contract, end, duration, "5 mins", "BID_ASK", True, 2, timeout=55)
                retained = [bar for bar in bars if bar.date in target]
                if len({bar.date for bar in retained}) != len(retained):
                    raise ValueError(f"Duplicate bid/ask bars for {symbol} {label}")
                valid = [bar for bar in retained if all(math.isfinite(n) for n in (bar.open, bar.close))
                         and 0 < bar.open <= bar.close]
                valid_dates = {bar.date for bar in valid}
                payload = dict(
                    contract=dataclasses.asdict(contract), downloaded_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    what_to_show="BID_ASK", duration=duration, bar_size="5 mins", regular_hours=True, end=end.isoformat(),
                    expected_bar_starts_utc=sorted(t.isoformat() for t in target), received_bars=len(bars),
                    bars=[dataclasses.asdict(bar) for bar in valid],
                    invalid_target_bars=[dataclasses.asdict(bar) for bar in retained if bar.date not in valid_dates],
                    errors=list(errors),
                )
                write(path, json.dumps(payload, default=str, separators=(",", ":")).encode())
                print(f"{symbol} {label}: {len(valid)}/{len(target)} scheduled bars", flush=True)
                ib.sleep(1.5)
    finally:
        ib.disconnect()


def parse_month(value):
    try:
        day = dt.date.fromisoformat(value + "-01")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected YYYY-MM") from exc
    return day.isoformat()[:7]


if __name__ == "__main__":
    main()
