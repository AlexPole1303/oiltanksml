"""Attach dated transaction disclosures, trading-session clocks and EIA windows."""
import csv
import datetime as dt
import hashlib
import html
import json
import re
import sqlite3
from collections import Counter
from contextlib import closing
from zoneinfo import ZoneInfo

import pdfplumber

from collect_public import RAW, ROOT, write
from publication_clocks import export

REFERENCE = ROOT / "data/reference"
DERIVED = ROOT / "data/derived"
ARCHIVE = RAW / "documents/market_transactions"

SCHEMA = """
CREATE TABLE IF NOT EXISTS market_transaction_event (
    event_id TEXT PRIMARY KEY,transaction_id TEXT,event_type TEXT,occurred_date TEXT,disclosure_date TEXT,
    disclosure_time_local TEXT,disclosure_timezone TEXT,disclosed_from_utc TEXT,disclosed_by_utc TEXT,
    time_precision TEXT,source_url TEXT,archive_file TEXT,archive_sha256 TEXT,archive_url TEXT,notes TEXT);
CREATE INDEX IF NOT EXISTS market_transaction_event_deal_time ON market_transaction_event(transaction_id,disclosed_by_utc);
CREATE TABLE IF NOT EXISTS market_transaction_instrument (
    transaction_id TEXT,transaction_type TEXT,instrument TEXT,participant_role TEXT,terminal_id TEXT,
    PRIMARY KEY (transaction_id,instrument));
CREATE TABLE IF NOT EXISTS market_spin_off_terms (
    transaction_id TEXT PRIMARY KEY,parent_instrument TEXT,distributed_instrument TEXT,
    parent_shares_per_old_share REAL,distributed_shares_per_old_share REAL,record_date TEXT,legal_effective_date TEXT,
    due_bill_first_session TEXT,due_bill_last_session TEXT,parent_ex_distribution_session TEXT,
    distributed_first_tsx_regular TEXT,distributed_first_nyse_regular TEXT,primary_source_url TEXT,trading_mechanics_source_url TEXT);
"""

VIEWS = {
    "market_transaction_session_alignment": """
        CREATE VIEW market_transaction_session_alignment AS
        WITH listing AS (
            SELECT symbol,min(first_study_session) AS first_regular_session,max(last_study_session) AS last_regular_session,
                min(exchange) AS listing_exchange
            FROM historical_equity_listing_window GROUP BY symbol
        ), participants AS (
            SELECT p.*,i.venue,i.currency,coalesce(l.listing_exchange,i.venue) AS listing_exchange,
                coalesce(sp.distributed_first_nyse_regular,
                    CASE WHEN i.study_role='historical_operator' THEN l.first_regular_session END) AS first_regular_session,
                CASE WHEN i.study_role='historical_operator' THEN l.last_regular_session END AS last_regular_session
            FROM market_transaction_instrument p JOIN market_study_instrument i USING(instrument)
            LEFT JOIN listing l ON l.symbol=p.instrument
            LEFT JOIN market_spin_off_terms sp ON sp.transaction_id=p.transaction_id AND sp.distributed_instrument=p.instrument
        )
        SELECT e.*,p.instrument,p.transaction_type,p.participant_role,p.terminal_id,p.venue AS calendar_venue,p.listing_exchange,p.currency,
            p.first_regular_session,p.last_regular_session,
            CASE WHEN p.first_regular_session IS NULL OR e.disclosure_date>=p.first_regular_session THEN (
                SELECT min(c.session_date) FROM equity_session_calendar c WHERE c.venue=p.venue
                    AND c.session_date>=e.disclosure_date AND c.close_utc>e.disclosed_from_utc
                    AND (p.last_regular_session IS NULL OR c.session_date<=p.last_regular_session)
            ) END AS earliest_market_session,
            CASE WHEN p.first_regular_session IS NULL OR e.disclosure_date>=p.first_regular_session THEN (
                SELECT min(c.session_date) FROM equity_session_calendar c WHERE c.venue=p.venue
                    AND c.session_date>=e.disclosure_date AND c.close_utc>=e.disclosed_by_utc
                    AND (p.last_regular_session IS NULL OR c.session_date<=p.last_regular_session)
            ) END AS latest_market_session
        FROM market_transaction_event e JOIN participants p USING(transaction_id);
    """,
    "eia_market_transaction_window": """
        CREATE VIEW eia_market_transaction_window AS
        WITH clocks AS (
            SELECT transaction_id,
                min(CASE WHEN event_type<>'completed' THEN disclosed_from_utc END) AS proposed_from_utc,
                min(CASE WHEN event_type<>'completed' THEN disclosed_by_utc END) AS proposed_by_utc,
                max(CASE WHEN event_type='completed' THEN disclosed_from_utc END) AS completion_disclosed_from_utc,
                max(CASE WHEN event_type='completed' THEN disclosed_by_utc END) AS completion_disclosed_by_utc,
                max(CASE WHEN event_type='completed' THEN occurred_date END) AS legal_completion_date
            FROM market_transaction_event GROUP BY transaction_id
        ), candidate AS (
            SELECT h.release_date,h.instrument,h.venue,h.entry_session,h.entry_open_utc,h.fifth_session,h.fifth_close_utc,
                h.own_quote_status,r.conservative_postrelease_from_utc AS eia_decision_cutoff_utc,
                p.transaction_id,p.transaction_type,p.participant_role,p.terminal_id,
                d.legal_completion_date,
                EXISTS (SELECT 1 FROM market_transaction_event e WHERE e.transaction_id=p.transaction_id
                      AND e.event_type='completed' AND e.disclosed_by_utc<=r.conservative_postrelease_from_utc)
                    AS completion_known_at_cutoff,
                coalesce(d.proposed_by_utc<=r.conservative_postrelease_from_utc
                   AND r.conservative_postrelease_from_utc<d.completion_disclosed_from_utc,0) AS pending_known_at_cutoff,
                coalesce(d.proposed_from_utc<=r.conservative_postrelease_from_utc
                   AND r.conservative_postrelease_from_utc<d.completion_disclosed_by_utc,0) AS pending_possible_at_cutoff,
                d.legal_completion_date BETWEEN h.entry_session AND h.fifth_session AS legal_completion_in_horizon,
                coalesce(sp.parent_instrument=h.instrument AND sp.due_bill_first_session<=h.fifth_session
                   AND sp.due_bill_last_session>=h.entry_session,0) AS spin_off_due_bill_session_in_horizon,
                coalesce(sp.parent_instrument=h.instrument AND sp.parent_ex_distribution_session
                   BETWEEN h.entry_session AND h.fifth_session,0) AS spin_off_ex_distribution_in_horizon,
                EXISTS (SELECT 1 FROM market_transaction_event e WHERE e.transaction_id=p.transaction_id
                      AND e.disclosed_from_utc<h.fifth_close_utc AND e.disclosed_by_utc>h.entry_open_utc)
                    AS possible_disclosure_during_holding,
                EXISTS (SELECT 1 FROM market_transaction_session_alignment e
                      WHERE e.transaction_id=p.transaction_id AND e.instrument=p.instrument
                        AND e.earliest_market_session<=h.fifth_session
                        AND coalesce(e.latest_market_session,e.earliest_market_session)>=h.entry_session)
                    AS possible_first_market_session_in_horizon,
                (SELECT e.event_id FROM market_transaction_event e WHERE e.transaction_id=p.transaction_id
                     AND e.disclosed_by_utc<=r.conservative_postrelease_from_utc
                     ORDER BY e.disclosed_by_utc DESC,e.event_id DESC LIMIT 1) AS latest_event_id_known_at_cutoff
            FROM eia_equity_horizon_input h JOIN eia_release_clock r USING(release_date)
            JOIN market_transaction_instrument p USING(instrument) JOIN clocks d USING(transaction_id)
            LEFT JOIN market_spin_off_terms sp ON sp.transaction_id=p.transaction_id
            WHERE h.fifth_session IS NOT NULL
               AND (p.participant_role<>'distributed_security' OR h.entry_session>=sp.distributed_first_nyse_regular)
        )
        SELECT * FROM candidate WHERE pending_possible_at_cutoff OR legal_completion_in_horizon
            OR spin_off_due_bill_session_in_horizon OR spin_off_ex_distribution_in_horizon
            OR possible_disclosure_during_holding OR possible_first_market_session_in_horizon;
    """,
    "eia_terminal_transfer_disclosure": """
        CREATE VIEW eia_terminal_transfer_disclosure AS
        WITH assets AS (
            SELECT transaction_id,terminal_id,max(CASE WHEN participant_role='seller' THEN instrument END) AS seller,
                max(CASE WHEN participant_role='buyer' THEN instrument END) AS buyer
            FROM market_transaction_instrument WHERE transaction_type='asset_sale'
            GROUP BY transaction_id,terminal_id
        )
        SELECT r.release_date,p.transaction_id,p.terminal_id,p.seller,p.buyer,e.occurred_date AS legal_completion_date,
            r.conservative_postrelease_from_utc AS eia_decision_cutoff_utc,e.disclosed_from_utc,e.disclosed_by_utc,
            r.release_date>=e.occurred_date AS effective_day_or_later,
            e.disclosed_by_utc<=r.conservative_postrelease_from_utc AS completion_known_at_cutoff,
            CASE WHEN r.release_date<e.occurred_date THEN 'before_legal_completion'
                 WHEN e.disclosed_by_utc<=r.conservative_postrelease_from_utc THEN 'completion_disclosed_by_cutoff'
                 ELSE 'completion_not_established_as_public_by_cutoff' END AS transfer_status,
            e.source_url
        FROM assets p JOIN market_transaction_event e USING(transaction_id)
        CROSS JOIN eia_release_clock r WHERE e.event_type='completed'
            AND EXISTS (SELECT 1 FROM eia_inventory_release_panel w WHERE w.release_date=r.release_date);
    """,
}


def reference_rows(name):
    with (REFERENCE / f"{name}.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def source_text(path):
    if path.suffix == ".pdf":
        with pdfplumber.open(path) as pdf:
            return " ".join(page.extract_text() or "" for page in pdf.pages)
    return html.unescape(re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8", errors="replace")))


def publication_window(date, time, timezone):
    zone = ZoneInfo(timezone)
    day = dt.date.fromisoformat(date)
    if time:
        start = dt.datetime.combine(day, dt.time.fromisoformat(time), zone)
        end = start + dt.timedelta(minutes=1)
    else:
        start = dt.datetime.combine(day, dt.time(), zone)
        end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(), zone)
    return tuple(value.astimezone(dt.timezone.utc).isoformat(timespec="seconds") for value in (start, end))


def event_rows():
    alternate = {row["event_id"]: (row["archive_url"], row["archive_marker"])
                 for row in reference_rows("market_transaction_alternate_archives")}
    for row in reference_rows("market_transaction_events"):
        name = row["event_id"]
        path, = ARCHIVE.glob(f"{name}.*")
        archive_url, marker = alternate.get(name, (row["source_url"], row["source_marker"]))
        body = " ".join(source_text(path).split()).casefold()
        if marker.casefold() not in body:
            raise ValueError(f"The archived evidence for {name} does not contain {marker!r}")
        if not dt.date.fromisoformat(row["occurred_date"]) <= dt.date.fromisoformat(row["disclosure_date"]):
            raise ValueError(f"The disclosed historical milestone precedes its occurrence for {name}")
        start, end = publication_window(row["disclosure_date"], row["disclosure_time_local"], row["disclosure_timezone"])
        yield (name, row["transaction_id"], row["event_type"], row["occurred_date"], row["disclosure_date"],
               row["disclosure_time_local"] or None, row["disclosure_timezone"], start, end,
               "minute" if row["disclosure_time_local"] else "day", row["source_url"], str(path.relative_to(ROOT)),
               hashlib.sha256(path.read_bytes()).hexdigest(), archive_url, row["notes"])


def add_market_transaction_timeline(db):
    events = list(event_rows())
    participants = reference_rows("market_transaction_instruments")
    transactions = {row["transaction_id"] for row in participants}
    completions = Counter(row[1] for row in events if row[2] == "completed")
    if transactions != {row[1] for row in events} or completions != dict.fromkeys(transactions, 1):
        raise ValueError("Each transaction needs a complete event history and one completion record")
    instruments = {row[0] for row in db.execute("SELECT instrument FROM market_study_instrument")}
    terminals = {row[0] for row in db.execute("SELECT terminal_id FROM terminal")}
    if any(row["instrument"] not in instruments or (row["terminal_id"] and row["terminal_id"] not in terminals)
           for row in participants):
        raise ValueError("Unknown transaction instrument or terminal")
    for statement in SCHEMA.strip().split(";"):
        if statement.strip():
            db.execute(statement)
    db.execute("DELETE FROM market_transaction_event")
    db.execute("DELETE FROM market_transaction_instrument")
    db.execute("DELETE FROM market_spin_off_terms")
    db.executemany("INSERT INTO market_transaction_event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", events)
    db.executemany("INSERT INTO market_transaction_instrument VALUES (?,?,?,?,?)",
                   (tuple(value or None for value in row.values()) for row in participants))
    db.executemany("INSERT INTO market_spin_off_terms VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (tuple(row.values()) for row in reference_rows("market_spin_off_terms")))
    for name, definition in VIEWS.items():
        db.execute(f"DROP VIEW IF EXISTS {name}")
        db.execute(definition)


def refresh_wildhorse_and_merger_terms(db):
    selected = lambda name, field: (tuple(value or None for value in row.values()) for row in reference_rows(name)
                                    if row[field] == "wildhorse")
    db.execute("DELETE FROM terminal_company_history WHERE terminal_id='wildhorse'")
    db.executemany("INSERT INTO terminal_company_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   selected("terminal_company_history", "terminal_id"))
    db.executemany("INSERT OR REPLACE INTO terminal VALUES (?,?,?,?,?,?,?,?)", selected("terminals", "terminal_id"))
    db.execute("DELETE FROM terminal_parcel_alias WHERE terminal_id='wildhorse'")
    db.executemany("INSERT INTO terminal_parcel_alias VALUES (?,?,?,?,?,?)", selected("terminal_parcel_aliases", "terminal_id"))
    db.executemany("INSERT OR REPLACE INTO terminal_disclosed_capacity VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   selected("cushing_terminal_disclosed_capacity", "terminal_id"))
    db.executemany("INSERT OR REPLACE INTO equity_merger_consideration VALUES (?,?,?,?,?,?,?,?)",
                   (tuple(value or None for value in row.values()) for row in reference_rows("equity_merger_consideration")))


def main():
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        db.execute("BEGIN IMMEDIATE")
        with db:
            refresh_wildhorse_and_merger_terms(db)
            add_market_transaction_timeline(db)
        for name, order in (("market_transaction_event", "disclosed_from_utc,event_id"),
                            ("equity_merger_consideration", "closing_date,retiring_instrument"),
                            ("market_transaction_session_alignment", "disclosed_from_utc,event_id,instrument"),
                            ("eia_market_transaction_window", "release_date,instrument,transaction_id"),
                            ("eia_terminal_transfer_disclosure", "release_date,terminal_id"),
                            ("eia_event_operator_exposure", "release_date,terminal_id,ticker"),
                            ("eia_equity_horizon_input", "release_date,instrument"),
                            ("terminal_data_coverage", "terminal_id"),
                            ("terminal_disclosed_capacity", "terminal_id,scope_key,reference_date")):
            export(db, name, order)
        report = dict(db.execute("""SELECT 'transactions',count(DISTINCT transaction_id) FROM market_transaction_event
            UNION ALL SELECT 'dated_events',count(*) FROM market_transaction_event
            UNION ALL SELECT 'minute_precision_events',count(*) FROM market_transaction_event WHERE time_precision='minute'
            UNION ALL SELECT 'associated_instruments',count(DISTINCT instrument) FROM market_transaction_instrument
            UNION ALL SELECT 'eia_transaction_windows',count(*) FROM eia_market_transaction_window
            UNION ALL SELECT 'asset_transfer_weeks_with_disclosure_delay',count(*) FROM eia_terminal_transfer_disclosure
              WHERE effective_day_or_later=1 AND completion_known_at_cutoff=0"""))
    write(DERIVED / "market_transaction_timeline_summary.json", json.dumps(report, indent=2).encode() + b"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
