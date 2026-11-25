"""Normalize bid/ask observations and reconcile them with the study's execution clocks."""
import csv
import datetime as dt
import json
import math
from collections import Counter
from contextlib import closing

from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, write
from collect_tws_event_spreads import SYMBOLS, event_targets
from collect_tws_borrow_rates import EASTERN, STORE

MINUTES = dt.timedelta(minutes=5)


def read_packets(expected):
    observed = {}
    requested = set()
    rejected = set()
    packet_rows = []
    identities = {}
    for path in sorted(STORE.glob("*/BID_ASK_5min_*.json")):
        instrument = path.parent.name
        if instrument not in expected:
            continue
        packet = json.loads(path.read_bytes())
        contract = packet["contract"]
        identity = contract["conId"], contract["currency"], contract["symbol"]
        if (identity != identities.setdefault(instrument, identity)
                or contract["symbol"].split(".")[0] != instrument or contract["currency"] != "USD"):
            raise ValueError(f"Unexpected contract identity in {path}")
        if (packet["what_to_show"], packet["bar_size"], packet["regular_hours"]) != ("BID_ASK", "5 mins", True):
            raise ValueError(f"Unexpected request in {path}")
        wanted = {dt.datetime.fromisoformat(value) for value in packet["expected_bar_starts_utc"]}
        if not wanted or not wanted <= expected[instrument]:
            raise ValueError(f"Unexpected requested timestamps in {path}")
        requested.update((instrument, t) for t in wanted)
        invalid = {dt.datetime.fromisoformat(row["date"]) for row in packet["invalid_target_bars"]}
        rejected.update((instrument, t) for t in invalid)
        seen = set()
        duplicate_sources = 0
        for raw in packet["bars"]:
            start = dt.datetime.fromisoformat(raw["date"])
            key = instrument, start
            bid, ask, low, high = (float(raw[field]) for field in ("open", "close", "low", "high"))
            if (start not in wanted or start in seen
                    or not all(map(math.isfinite, (bid, ask, low, high))) or not 0 < low <= bid <= ask <= high):
                raise ValueError(f"Invalid or duplicate quote observation {key} in {path}")
            seen.add(start)
            if key in observed:
                duplicate_sources += 1
                continue
            observed[key] = dict(
                instrument=instrument, conid=contract["conId"], currency=contract["currency"],
                exchange=contract["exchange"], session_date=start.astimezone(EASTERN).date().isoformat(),
                bar_start_utc=start.isoformat(), bar_end_utc=(start+MINUTES).isoformat(),
                average_bid=bid, average_ask=ask, minimum_bid=low, maximum_ask=high,
                spread_bps=20_000*(ask-bid)/(ask+bid),
                retrieved_at_utc=packet["downloaded_at_utc"], source_file=str(path.relative_to(ROOT)),
            )
        if not invalid <= wanted or invalid & seen:
            raise ValueError(f"Invalid rejected timestamps in {path}")
        codes = sorted({str(code) for _, code, _ in packet["errors"] if code not in (2104, 2106, 2158, 2174)})
        packet_rows.append(dict(
            instrument=instrument, source_file=str(path.relative_to(ROOT)), requested_end=packet["end"],
            target_bars=len(wanted), admitted_bars=len(seen), invalid_target_bars=len(invalid),
            already_observed_from_another_request=duplicate_sources,
            broker_received_bars=packet["received_bars"], error_codes="|".join(codes),
            missing_target_timestamps_utc="|".join(t.isoformat() for t in sorted(wanted-seen)),
        ))
    return observed, requested, rejected, packet_rows


def availability(key, expected, observed, requested, rejected):
    instrument, stamp = key
    if stamp not in expected[instrument]:
        return "outside_confirmed_broker_series"
    if key in observed:
        return "observed"
    if key in rejected:
        return "invalid_quote_range"
    return "missing_from_response" if key in requested else "request_pending"


def window_rows(db, expected, observed, requested, rejected):
    with (DERIVED / "tws_borrow_window_coverage.csv").open(newline="") as stream:
        fee_rows = {(r["release_date"], r["instrument"]): r for r in csv.DictReader(stream)}
    placeholders = ",".join("?" for _ in SYMBOLS)
    for issue, phase, week, instrument, entry, end in db.execute(f"""
        SELECT h.release_date,w.study_phase,w.week_ending,h.instrument,h.entry_open_utc,h.fifth_close_utc
        FROM eia_equity_horizon_input h JOIN eia_release_clock e USING(release_date)
        JOIN eia_satellite_window w ON w.week_ending=e.latest_week_in_issue AND w.endpoint='current'
        WHERE h.instrument IN ({placeholders})
        ORDER BY h.release_date,h.instrument
    """, SYMBOLS):
        entry_start = dt.datetime.fromisoformat(entry)
        exit_start = dt.datetime.fromisoformat(end) - MINUTES
        details = {}
        statuses = {}
        for leg, ticker in (("instrument", instrument), ("amlp", "AMLP")):
            for edge, stamp in (("entry", entry_start), ("exit", exit_start)):
                statuses[f"{leg}_{edge}_spread_status"] = availability(
                    (ticker, stamp), expected, observed, requested, rejected)
            fee = fee_rows.get((issue, ticker), {})
            matched = fee.get("five_exchange_sessions_with_identity_and_rates")
            details[f"{leg}_five_borrow_rate_sessions"] = (matched == "True") if matched is not None else None
        yield dict(
            study_phase=phase, week_ending=week, release_date=issue, instrument=instrument,
            entry_spread_bar_start_utc=entry_start.isoformat(), exit_spread_bar_start_utc=exit_start.isoformat(),
            **statuses, **details,
            both_legs_have_two_spread_observations=all(s == "observed" for s in statuses.values()),
        )


def main():
    expected = {instrument: set().union(*event_targets(instrument).values()) for instrument in SYMBOLS}
    observed, requested, rejected, packets = read_packets(expected)
    with closing(read_only(SOURCE)) as db:
        windows = list(window_rows(db, expected, observed, requested, rejected))
    rows = [row for _, row in sorted(observed.items())]
    export("tws_event_spread_observations", rows)
    export("tws_event_spread_request_coverage", packets)
    export("tws_event_execution_window_coverage", windows)
    summary = dict(
        observations=len(rows), request_packets=len(packets), market_windows=len(windows),
        observations_by_instrument=dict(sorted(Counter(r["instrument"] for r in rows).items())),
        complete_quote_windows_by_instrument={
            symbol: sum(r["instrument"] == symbol and r["both_legs_have_two_spread_observations"] for r in windows)
            for symbol in SYMBOLS
        },
    )
    write(DERIVED / "tws_event_spread_summary.json", json.dumps(summary, indent=2).encode() + b"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
