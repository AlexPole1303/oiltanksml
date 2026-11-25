"""Build observed-bar-checked exchange sessions and align WPSR dates to them."""
import datetime as dt
from bisect import bisect_left, bisect_right

import exchange_calendars as ec

from publication_clocks import EASTERN

SOURCES = {
    "XNYS": "https://www.nyse.com/markets/hours-calendars",
    "XTSE": "https://www.tsx.com/en/trading/calendars-and-trading-hours/calendar",
}
CALENDAR_IMPLEMENTATION = "https://pypi.org/project/exchange-calendars/4.13.2/"


def session_rows(db):
    first, last = db.execute("SELECT min(session_date), max(session_date) FROM tws_daily_bar").fetchone()
    if first is None or last is None:
        return
    start = dt.date.fromisoformat(first) - dt.timedelta(days=7)
    end = dt.date.fromisoformat(last) + dt.timedelta(days=14)
    for venue, source in SOURCES.items():
        calendar = ec.get_calendar(venue, start=start, end=end)
        observed = dict(db.execute("""
            SELECT session_date, count(DISTINCT instrument) FROM tws_daily_bar
            WHERE (instrument='KEY')=? AND series='TRADES' GROUP BY session_date
        """, (venue == "XTSE",)))
        expected = {day.date().isoformat() for day in calendar.sessions}
        if observed.keys() - expected:
            raise ValueError(f"TWS returned {venue} bars on unexpected sessions: {sorted(observed.keys() - expected)}")
        for day in calendar.sessions:
            key = day.date().isoformat()
            opened = calendar.session_open(day).to_pydatetime()
            closed = calendar.session_close(day).to_pydatetime()
            yield (venue, key, opened.isoformat(timespec="seconds"), closed.isoformat(timespec="seconds"),
                   int(closed.astimezone(EASTERN).hour < 16), observed.get(key, 0),
                   CALENDAR_IMPLEMENTATION, source)


def release_alignment_rows(db):
    for venue in SOURCES:
        sessions = list(db.execute("""
            SELECT session_date, open_utc, close_utc, observed_instruments FROM equity_session_calendar
            WHERE venue=? ORDER BY session_date
        """, (venue,)))
        dates = [row[0] for row in sessions]
        for date, scheduled, postrelease in db.execute("""
            SELECT release_date, scheduled_table_release_at_utc, conservative_postrelease_from_utc
            FROM eia_release_clock ORDER BY release_date
        """):
            index = bisect_left(dates, date)
            close = sessions[index][2] if index < len(sessions) and dates[index] == date else None
            before_close = int(scheduled < close) if scheduled and close else None
            next_index = bisect_right(dates, date)
            next_row = sessions[next_index] if next_index < len(sessions) else (None, None, None, None)
            if next_row[1] is not None and next_row[1] < postrelease:
                raise ValueError(f"Next session opens before the conservative release boundary: {venue} {date}")
            yield (date, venue, close, before_close, *next_row)
