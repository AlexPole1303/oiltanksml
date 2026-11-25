"""Collect Canadian-dollar reference rates and preserve their publication timing."""
import datetime as dt
import json
from decimal import Decimal
from zoneinfo import ZoneInfo

from collect_public import RAW, ROOT, get, write, write_csv

SERIES = "FXUSDCAD"
FIRST_DATE = "2017-01-01"
LAST_DATE = "2026-11-11"
URL = f"https://www.bankofcanada.ca/valet/observations/{SERIES}/json?start_date={FIRST_DATE}&end_date={LAST_DATE}"
METHOD = "https://www.bankofcanada.ca/rates/exchange/background-information-on-foreign-exchange-rates/"
SOURCE = RAW / "factors" / f"boc_{SERIES}_{FIRST_DATE}_{LAST_DATE}.json"
DERIVED = ROOT / "data" / "derived"
EASTERN = ZoneInfo("America/Toronto")
FIELDS = (
    "observation_date", "base_currency", "quote_currency", "quote_per_base", "provider_series",
    "scheduled_publication_by_utc", "conservative_available_at_utc", "source_url", "method_url", "source_file",
)
SCHEMA = """
CREATE TABLE cad_usd_reference_rate (
    observation_date TEXT PRIMARY KEY,base_currency TEXT,quote_currency TEXT,quote_per_base REAL,provider_series TEXT,
    scheduled_publication_by_utc TEXT,conservative_available_at_utc TEXT,source_url TEXT,method_url TEXT,source_file TEXT);
CREATE VIEW eia_cad_equity_reference_inputs AS
    WITH horizons AS (
        SELECT h.release_date,h.instrument,h.entry_session,h.entry_open_utc,h.fifth_session,h.fifth_close_utc,
            h.first_session_open AS entry_open_cad,h.fifth_session_close AS fifth_close_cad,
            h.amlp_same_date_open,h.amlp_same_date_close,h.amlp_observed_first_five_sessions,
            u.open_utc AS amlp_same_date_open_utc,v.close_utc AS amlp_same_date_close_utc,
            (SELECT max(r.observation_date) FROM cad_usd_reference_rate r
             WHERE julianday(r.conservative_available_at_utc)<=julianday(h.entry_open_utc)) AS entry_fx_date,
            (SELECT max(r.observation_date) FROM cad_usd_reference_rate r
             WHERE julianday(r.conservative_available_at_utc)<=julianday(h.fifth_close_utc)) AS fifth_fx_date
        FROM eia_equity_horizon_input h
        LEFT JOIN equity_session_calendar u ON u.venue='XNYS' AND u.session_date=h.entry_session
        LEFT JOIN equity_session_calendar v ON v.venue='XNYS' AND v.session_date=h.fifth_session
        WHERE h.currency='CAD'
    )
    SELECT h.*,a.quote_per_base AS entry_reference_cad_per_usd,b.quote_per_base AS fifth_reference_cad_per_usd,
        a.conservative_available_at_utc AS entry_fx_available_at_utc,b.conservative_available_at_utc AS fifth_fx_available_at_utc,
        round(24*(julianday(h.entry_open_utc)-julianday(a.scheduled_publication_by_utc)),3) AS entry_fx_hours_since_scheduled,
        round(24*(julianday(h.fifth_close_utc)-julianday(b.scheduled_publication_by_utc)),3) AS fifth_fx_hours_since_scheduled,
        CASE WHEN h.amlp_same_date_open IS NULL OR h.amlp_same_date_close IS NULL THEN 'missing_amlp_endpoint'
             WHEN h.amlp_same_date_open_utc IS NULL OR h.amlp_same_date_close_utc IS NULL THEN 'missing_amlp_clock'
             WHEN julianday(h.entry_open_utc)<>julianday(h.amlp_same_date_open_utc)
                OR julianday(h.fifth_close_utc)<>julianday(h.amlp_same_date_close_utc) THEN 'different_endpoint_clocks'
             WHEN h.amlp_observed_first_five_sessions<5 THEN 'missing_intermediate_amlp_session'
             ELSE 'matching_endpoint_clocks_and_five_sessions' END AS cross_venue_clock_status
    FROM horizons h LEFT JOIN cad_usd_reference_rate a ON a.observation_date=h.entry_fx_date
    LEFT JOIN cad_usd_reference_rate b ON b.observation_date=h.fifth_fx_date;
"""


def observations(payload):
    if SERIES not in payload["seriesDetail"]:
        raise ValueError(f"Bank of Canada did not return {SERIES}")
    dates = set()
    for item in payload["observations"]:
        day = dt.date.fromisoformat(item["d"])
        rate = Decimal(item[SERIES]["v"])
        if not rate.is_finite() or rate <= 0 or day in dates or not FIRST_DATE <= day.isoformat() <= LAST_DATE:
            raise ValueError(f"Invalid Bank of Canada observation for {day}")
        dates.add(day)
        deadline = dt.datetime.combine(day, dt.time(16, 30), EASTERN).astimezone(dt.timezone.utc)
        available = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, EASTERN).astimezone(dt.timezone.utc)
        yield (day.isoformat(), "USD", "CAD", str(rate), SERIES, deadline.isoformat(), available.isoformat(),
               URL, METHOD, str(SOURCE.relative_to(ROOT)))


def add_fx_reference(db):
    if SOURCE.is_file():
        db.executemany("INSERT INTO cad_usd_reference_rate VALUES (?,?,?,?,?,?,?,?,?,?)",
                       observations(json.loads(SOURCE.read_bytes())))


def main():
    data = get(URL)
    rows = sorted(observations(json.loads(data)))
    if not rows:
        raise ValueError("Bank of Canada response has no observations")
    write(SOURCE, data)
    write_csv(DERIVED / "cad_usd_reference_rate.csv", FIELDS, rows)
    print(f"{len(rows)} Bank of Canada FX observations, {rows[0][0]} through {rows[-1][0]}")


if __name__ == "__main__":
    main()
