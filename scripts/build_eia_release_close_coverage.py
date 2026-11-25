"""Inventory inputs for a release-day close scenario using archived publication clocks."""
import json
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta

from build_market_event_panel import CORE
from build_operator_development_inputs import one_per_key
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

STEM = "eia_release_close"
FIVE_MINUTES = timedelta(minutes=5)


def source_rows():
    with closing(read_only(SOURCE)) as db:
        sessions = {(venue, day): (opened, closed, bool(early)) for venue, day, opened, closed, early in db.execute(
            "SELECT venue,session_date,open_utc,close_utc,early_close FROM equity_session_calendar")}
        clocks = {day: (scheduled, basis, conservative) for day, scheduled, basis, conservative in db.execute("""
            SELECT release_date,scheduled_table_release_at_utc,timing_basis,conservative_postrelease_from_utc
            FROM eia_release_clock
        """)}
        symbols = (*CORE, "AMLP")
        quotes = {(ticker, day): (provider, path, basis) for ticker, day, provider, path, basis in db.execute(f"""
            SELECT instrument,session_date,provider,source_file,price_basis FROM market_study_quote
            WHERE instrument IN ({','.join('?' for _ in symbols)}) AND close>0
        """, symbols)}
    return sessions, clocks, quotes


def calendar_rows(sessions, clocks):
    for row in read_csv("eia_phase_market_calendar"):
        issue, venue = row["release_date"], row["calendar_venue"]
        scheduled, basis, conservative = clocks[issue]
        same_day = sessions.get((venue, issue))
        opening, closing, early = same_day if same_day else (None, None, None)
        close_at = datetime.fromisoformat(closing) if closing else None
        scheduled_at = datetime.fromisoformat(scheduled) if scheduled else None
        until_close = (close_at-scheduled_at).total_seconds()/60 if close_at and scheduled_at else None
        conservative_eligible = bool(close_at and datetime.fromisoformat(conservative) < close_at)
        status = ("exchange_closed_on_release_day" if not same_day else "release_clock_unknown" if scheduled_at is None
                  else "scheduled_at_or_after_close" if until_close <= 0 else "scheduled_before_close")
        if row["decision_at_utc"] != conservative:
            raise ValueError(f"Release-date audit disagrees with the frozen decision clock for {issue}")
        if row["fifth_session"] <= issue:
            raise ValueError(f"Fifth session must follow the release for {issue}/{venue}")
        yield dict(study_phase=row["study_phase"], week_ending=row["week_ending"], release_date=issue, calendar_venue=venue,
            release_time_basis=basis, scheduled_table_release_at_utc=scheduled,
            conservative_postrelease_from_utc=conservative, same_day_open_utc=opening, same_day_close_utc=closing,
            same_day_early_close=early, scheduled_minutes_before_close=until_close, schedule_status=status,
            conservative_same_day_eligible=conservative_eligible,
            fifth_session_after_release=row["fifth_session"], fifth_session_close_utc=row["fifth_close_utc"],
            phase_window_eligible=row["calendar_status"] in ("fifth_close_precedes_next_phase_decision", "final_phase"),
            close_window_phase_status=row["calendar_status"],
            entry_spread_bar_start_utc=(close_at-FIVE_MINUTES).isoformat() if close_at else None,
            exit_spread_bar_start_utc=(datetime.fromisoformat(row["fifth_close_utc"])-FIVE_MINUTES).isoformat(),
            release_clock_source_url=row["eia_release_clock_source_url"])


def instrument_rows(calendar, quotes):
    spreads = one_per_key(read_csv("tws_event_spread_observations"), "instrument", "bar_start_utc")
    for row in calendar:
        if row["calendar_venue"] != "XNYS":
            continue
        for instrument in CORE:
            cells = {}
            for leg, symbol in (("instrument", instrument), ("amlp", "AMLP")):
                for edge, day in (("entry", row["release_date"]), ("exit", row["fifth_session_after_release"])):
                    quote = quotes.get((symbol, day))
                    spread = spreads.get((symbol, row[f"{edge}_spread_bar_start_utc"]))
                    provider, path, basis = quote if quote else (None, None, None)
                    if edge == "entry" and not row["same_day_close_utc"] and quote:
                        raise ValueError(f"A daily bar occurs outside the release-day session: {symbol}/{day}")
                    cells.update({f"{leg}_{edge}_close_observed": bool(quote),
                                  f"{leg}_{edge}_close_source": path, f"{leg}_{edge}_close_provider": provider,
                                  f"{leg}_{edge}_close_basis": basis, f"{leg}_{edge}_spread_observed": bool(spread),
                                  f"{leg}_{edge}_spread_source": spread["source_file"] if spread else None})
            close_observed = all(value for key, value in cells.items() if key.endswith("_close_observed"))
            spread_observed = all(value for key, value in cells.items() if key.endswith("_spread_observed"))
            yield dict(study_phase=row["study_phase"], week_ending=row["week_ending"], release_date=row["release_date"],
                instrument=instrument, hedge="AMLP", calendar_venue=row["calendar_venue"], **cells,
                both_legs_have_two_daily_closes=close_observed, both_legs_have_two_indicative_spreads=spread_observed,
                scheduled_close_has_prices=row["phase_window_eligible"] and row["schedule_status"] == "scheduled_before_close" and close_observed,
                conservative_same_day_eligible=row["conservative_same_day_eligible"] and row["phase_window_eligible"])


def main():
    sessions, clocks, quotes = source_rows()
    calendar = list(calendar_rows(sessions, clocks))
    costs = list(instrument_rows(calendar, quotes))
    by_venue = defaultdict(Counter)
    for row in calendar:
        counts = by_venue[f"{row['study_phase']}/{row['calendar_venue']}"]
        counts[row["schedule_status"]] += 1
        counts["conservative_same_day_eligible"] += row["conservative_same_day_eligible"] and row["phase_window_eligible"]
    by_phase = defaultdict(Counter)
    for row in costs:
        counts = by_phase[row["study_phase"]]
        counts["instrument_windows"] += 1
        for field in ("both_legs_have_two_daily_closes", "both_legs_have_two_indicative_spreads", "scheduled_close_has_prices"):
            counts[field] += row[field]
    export(f"{STEM}_calendar", calendar)
    export(f"{STEM}_market_coverage", costs)
    report = dict(calendar_rows=len(calendar), market_rows=len(costs), calendar_coverage=by_venue, market_coverage=by_phase,
        interpretation="Scheduled release timestamps support a sensitivity scenario. The conservative release clock requires "
        "availability by the trade and leaves same-day entry outside the main study. Daily close and indicative spread "
        "presence describe input coverage. The files contain no execution prices or returns.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
