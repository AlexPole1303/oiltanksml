"""Verify IBKR indicative borrow-rate observations against the study calendar."""
import datetime as dt
import json
import math
from collections import defaultdict
from contextlib import closing

from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, write
from collect_tws_borrow_rates import STORE


def collect():
    by_symbol = defaultdict(dict)
    blocks = defaultdict(set)
    identities = {}
    for path in sorted(STORE.glob("*/FEE_RATE_*.json")):
        symbol = path.parent.name
        packet = json.loads(path.read_bytes())
        contract = packet["contract"]
        start = dt.date.fromisoformat(packet["start_inclusive"])
        end = dt.datetime.fromisoformat(packet["end"]).date()
        if (packet["what_to_show"], packet["bar_size"], packet["regular_hours"]) != ("FEE_RATE", "1 day", False):
            raise ValueError(f"Invalid request metadata in {path}")
        identity = contract["conId"], contract["currency"], contract["symbol"]
        if identity != identities.setdefault(symbol, identity) or contract["symbol"].split(".")[0] != symbol:
            raise ValueError(f"Contract identity changed for {symbol}")
        blocks[symbol].add((start, end))
        for bar in packet["bars"]:
            day = dt.date.fromisoformat(bar["date"])
            rates = tuple(float(bar[field]) for field in ("open", "high", "low", "close"))
            opening, high, low, close = rates
            if not start <= day <= end or day in by_symbol[symbol] or not all(map(math.isfinite, rates)):
                raise ValueError(f"Unexpected or duplicate observation {symbol} {day}")
            if not low <= min(opening, close) <= max(opening, close) <= high:
                raise ValueError(f"Inconsistent rate range for {symbol} {day}")
            by_symbol[symbol][day] = dict(
                instrument=symbol, session_date=day.isoformat(), conid=contract["conId"], currency=contract["currency"],
                fee_rate_open_native=opening, fee_rate_high_native=high, fee_rate_low_native=low,
                fee_rate_close_native=close, retrieved_at_utc=packet["downloaded_at_utc"],
                raw_file=str(path.relative_to(ROOT)),
            )
    return by_symbol, blocks


def main():
    observed, blocks = collect()
    with closing(read_only(SOURCE)) as db:
        calendars = defaultdict(set)
        for venue, day in db.execute("SELECT venue,session_date FROM equity_session_calendar"):
            calendars[venue].add(dt.date.fromisoformat(day))
        instruments = {symbol: (venue, currency) for symbol, venue, currency in
                       db.execute("SELECT instrument,venue,currency FROM market_study_instrument")}
        limits = {instrument: (start, candidate, url) for instrument, start, candidate, url in db.execute("""
            SELECT instrument,first_research_session,excluded_history_candidate,source_url
            FROM market_identity_limit
        """)}
        symbol_history = defaultdict(list)
        for instrument, symbol, start, end, url in db.execute("""
            SELECT instrument,exchange_symbol,valid_from,valid_to_exclusive,source_url
            FROM market_study_symbol_history ORDER BY instrument,valid_from
        """):
            symbol_history[instrument].append((start, end, symbol, url))
        windows = db.execute("""
            SELECT release_date,instrument,venue,entry_session,fifth_session FROM eia_equity_horizon_input
            ORDER BY release_date,instrument
        """).fetchall()
        event_sessions = defaultdict(list)
        for issue, venue, day in db.execute("""
            SELECT release_date,venue,session_date FROM eia_event_market_session
            WHERE ordinal BETWEEN 1 AND 5 ORDER BY release_date,venue,ordinal
        """):
            event_sessions[issue, venue].append(dt.date.fromisoformat(day))
    coverage = []
    for symbol, requested in sorted(blocks.items()):
        dates = observed[symbol].keys()
        for start, end in sorted(requested):
            expected = {day for day in calendars[instruments[symbol][0]] if start <= day <= end}
            available = {day for day in dates if start <= day <= end}
            coverage.append(dict(
                instrument=symbol, first_requested_session=start, last_requested_session=end,
                exchange_sessions=len(expected), matched_exchange_sessions=len(available & expected),
                other_dated_bars=len(available - expected), missing_exchange_sessions=len(expected - available),
                missing_session_dates="|".join(day.isoformat() for day in sorted(expected - available)),
            ))
    rows = []
    for instrument, rows_by_day in sorted(observed.items()):
        for day, row in sorted(rows_by_day.items()):
            if row["currency"] != instruments[instrument][1]:
                raise ValueError(f"Currency mismatch in broker rates for {instrument}")
            iso = day.isoformat()
            identities = [(symbol, url) for start, end, symbol, url in symbol_history[instrument]
                          if start <= iso and (end is None or iso < end)]
            if len(identities) > 1:
                raise ValueError(f"Conflicting historical symbols for {instrument} on {iso}")
            symbol, url = identities[0] if identities else (instrument, None)
            first, excluded, limit_source = limits.get(instrument, (None, None, None))
            limited = first is not None and iso < first
            rows.append(dict(
                row, historical_exchange_symbol=symbol if not limited else None, historical_symbol_source_url=url,
                identity_status="before_confirmed_broker_series_identity" if limited else "within_broker_series_identity",
                identity_limit_source_url=limit_source, excluded_history_candidate=excluded if limited else None,
            ))
    window_coverage = []
    for issue, instrument, venue, entry, exit_day in windows:
        if instrument not in blocks:
            continue
        sessions = event_sessions[issue, venue]
        if not sessions or sessions[0].isoformat() != entry or (exit_day and sessions[-1].isoformat() != exit_day):
            raise ValueError(f"Borrow calendar disagrees with market window for {issue}/{instrument}")
        missing = [day.isoformat() for day in sessions if day not in observed[instrument]]
        first = limits.get(instrument, (None,))[0]
        outside_identity = [day.isoformat() for day in sessions if first is not None and day.isoformat() < first]
        window_coverage.append(dict(
            release_date=issue, instrument=instrument, entry_session=entry, fifth_session=exit_day,
            expected_session_rates=len(sessions), observed_session_rates=len(sessions)-len(missing),
            missing_session_rates="|".join(missing),
            sessions_before_confirmed_broker_identity="|".join(outside_identity),
            five_exchange_sessions_with_rates=len(sessions) == 5 and not missing,
            five_exchange_sessions_with_identity_and_rates=len(sessions) == 5 and not missing and not outside_identity,
        ))
    export("tws_borrow_rate_daily", rows)
    export("tws_borrow_rate_coverage", coverage)
    export("tws_borrow_window_coverage", window_coverage)
    summary = {
        symbol: dict(bars=len(observed[symbol]), request_blocks=len(blocks[symbol]),
                     first_date=min(observed[symbol]).isoformat() if observed[symbol] else None,
                     last_date=max(observed[symbol]).isoformat() if observed[symbol] else None,
                     missing_exchange_sessions=sum(r["missing_exchange_sessions"] for r in coverage if r["instrument"] == symbol),
                     windows_with_five_rates=sum(r["instrument"] == symbol and r["five_exchange_sessions_with_rates"]
                                                 for r in window_coverage),
                     windows_with_five_rates_within_broker_identity=sum(
                         r["instrument"] == symbol and r["five_exchange_sessions_with_identity_and_rates"] for r in window_coverage))
        for symbol in sorted(blocks)
    }
    write(DERIVED / "tws_borrow_rate_summary.json", json.dumps(summary, indent=2).encode() + b"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
