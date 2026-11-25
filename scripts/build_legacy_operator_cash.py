"""Publish disclosed historical operator payments with trading-calendar and archive checks."""
import csv
import datetime as dt
import json
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal, ROUND_HALF_UP

from build_operator_issuer_cash import DB, NASDAQ_TRANSITION, NYSE_T2, calendar_ex_date
from collect_dolt_equity_history import REF, pinned_dividend_rows
from collect_issuer_distributions import ARCHIVE, DERIVED, FIELDS, refresh_cash
from collect_public import ROOT, write, write_csv
from publication_clocks import export

INSTRUMENTS = ("AMID", "BKEP", "HEP", "PSXP")
OUTPUT = DERIVED / "legacy_operator_issuer_cash_distributions.csv"


def source_rows():
    for file in (ARCHIVE / "legacy_operator_issuer_notices_extract.csv",
                 ROOT / "data" / "reference" / "jpep_amid_issuer_distributions.csv"):
        with file.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["instrument"] in INSTRUMENTS:
                    yield row


def build_rows(source, calendar, listings, archived, quotes, archive_path):
    seen = set()
    for row in sorted(source, key=lambda r: (r["instrument"], r["record_date"])):
        ticker, record, declared, payable = (row[field] for field in
                                              ("instrument", "record_date", "declared_date", "payable_date"))
        first, last, venue = listings[ticker]
        date = calendar_ex_date(calendar, dt.date.fromisoformat(record))
        amount = Decimal(row["reported_amount"])
        key = ticker, date
        if (not date or not first <= date <= record <= last or not record <= payable
                or (declared and declared > date) or amount <= 0 or key in seen):
            raise ValueError(f"Invalid disclosed cash event for {ticker} on {record}")
        seen.add(key)
        reference = archived.get(key)
        if reference is None:
            basis = "issuer_record_exchange_rule_archive_row_missing"
        elif reference == amount:
            basis = "issuer_record_exchange_rule_dolt_agrees"
        elif reference == amount.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP):
            basis = "issuer_record_exchange_rule_dolt_rounded_to_mills"
        else:
            raise ValueError(f"Issuer and archive amounts disagree for {ticker} on {date}: {amount} versus {reference}")
        cash = (ticker, date, record, declared or None, payable, amount, None, "USD", basis, row["source_url"])
        evidence = dict(instrument=ticker, ex_date=date, record_date=record, announced_date=declared or None,
            payable_date=payable, issuer_amount_usd=amount, pinned_archive_amount_usd=reference,
            archive_minus_issuer_usd=reference - amount if reference is not None else None, ex_date_basis=basis,
            listing_venue=venue, observed_bar_on_ex_date=key in quotes, first_listing_session=first, last_listing_session=last,
            issuer_source_form=row.get("source_form") or "SEC filing", issuer_source_filed_date=row.get("source_filed_date"),
            issuer_source_url=row["source_url"], rule_source_url=NASDAQ_TRANSITION if venue == "XNAS" else NYSE_T2,
            pinned_archive_revision=REF, pinned_archive_file=archive_path)
        yield cash, evidence
    if archived.keys() - seen:
        raise ValueError(f"Archive dividends lack disclosed counterparts: {sorted(archived.keys() - seen)}")


def main():
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as db:
        calendar = [day for day, in db.execute("SELECT session_date FROM equity_session_calendar WHERE venue='XNYS' ORDER BY 1")]
        listings = {t: (a, b, venue) for t, a, b, venue in db.execute("""SELECT symbol,first_study_session,last_study_session,exchange
            FROM historical_equity_listing_window WHERE symbol IN ('AMID','BKEP','HEP','PSXP')""")}
        quotes = set(db.execute("SELECT instrument,session_date FROM market_study_quote WHERE instrument IN ('AMID','BKEP','HEP','PSXP')"))
    if listings.keys() != set(INSTRUMENTS):
        raise ValueError("Historical listing boundaries are incomplete")
    archive, path = pinned_dividend_rows()
    archived, excluded = {}, []
    for r in archive:
        ticker, date = r["act_symbol"], r["ex_date"]
        if ticker not in listings:
            continue
        first, last, _ = listings[ticker]
        if not first <= date <= last:
            excluded.append(dict(r, last_listing_session=last))
        elif (ticker, date) in archived:
            raise ValueError(f"Duplicate archive cash event for {ticker} on {date}")
        else:
            archived[ticker, date] = Decimal(r["amount"])
    rows, checks = zip(*build_rows(source_rows(), calendar, listings, archived, quotes, path.relative_to(ROOT)), strict=True)
    write_csv(OUTPUT, FIELDS, rows)
    write_csv(DERIVED / "legacy_operator_cash_ex_date_evidence.csv", tuple(checks[0]), (r.values() for r in checks))
    refresh_cash(rows, INSTRUMENTS)
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as db:
        export(db, "historical_distribution_ex_date_estimate", "instrument,record_date")
    report = dict(cash_events=len(rows), by_ticker=dict(Counter(r[0] for r in rows)),
        dates_exact_archive_amount=sum(r[8] == "issuer_record_exchange_rule_dolt_agrees" for r in rows),
        dates_rounded_archive_amount=sum(r[8] == "issuer_record_exchange_rule_dolt_rounded_to_mills" for r in rows),
        calendar_only_ex_dates=[r for r in checks if r["pinned_archive_amount_usd"] is None],
        archive_records_outside_listing=excluded, dates_with_observed_bar=sum(r["observed_bar_on_ex_date"] for r in checks))
    write(DERIVED / "legacy_operator_cash_summary.json", (json.dumps(report, indent=2, default=str) + "\n").encode())
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
