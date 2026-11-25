"""Normalize issuer distributions and reconcile U.S. entitlement dates with broker adjustments."""
import csv
import datetime as dt
import json
import re
import sqlite3
from bisect import bisect_right
from collections import Counter
from contextlib import closing
from decimal import Decimal

from collect_dolt_equity_history import pinned_dividend_rows
from collect_issuer_distributions import ARCHIVE, DERIVED, FIELDS, issuer_rows, refresh_cash, source_date
from collect_public import ROOT, get, write, write_csv
from publication_clocks import table_rows

DB = DERIVED / "cushing.sqlite"
CASH_OUTPUT = DERIVED / "operator_issuer_cash_distributions.csv"
AS_OF = dt.date(2026, 11, 11)
FIRST = dt.date(2016, 1, 1)
ET_FIRST = dt.date(2018, 10, 19)
PSX = "https://investor.phillips66.com/stock-information/default.aspx?section=dividend"
TRP = "https://www.tcenergy.com/investors/stock-information/"
NYSE_T2 = "https://www.nyse.com/publicdocs/nyse/regulation/nyse/Changes_Related_to_the_Shortened_Settlement_Cycle_T%2B2.pdf"
NYSE_T1 = "https://www.nyse.com/publicdocs/nyse/markets/nyse/rule-filings/filings/2024/SR-NYSE-2024-19.pdf"
NASDAQ_TRANSITION = "https://listingcenter.nasdaq.com/assets/rulebook/nasdaq/filings/SR-NASDAQ-2023-046.pdf"
PAA_NASDAQ_FIRST = dt.date(2020, 12, 14)
CHECKS = ("instrument", "record_date", "payable_date", "issuer_currency", "issuer_amount", "issuer_ex_date",
          "calendar_candidate_ex_date", "rule_source_url", "broker_reference_instrument", "previous_session",
          "broker_implied_cash_usd", "cash_difference_usd", "admitted_ex_date", "ex_date_basis", "issuer_source_url",
          "pinned_archive_cash_usd", "pinned_archive_file")


def declared_cash():
    path = ROOT / "data" / "reference" / "operator_issuer_explicit_ex_dates.csv"
    with path.open(newline="") as stream:
        releases = {(r["instrument"], r["record_date"]): r for r in csv.DictReader(stream)}
    with (ARCHIVE / "operator_issuer_web_table_extract.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            label = row["source_series"]
            if label not in ("PAA", "ET"):
                raise ValueError(f"Unexpected cash table {label}")
            pattern = "%m/%d/%y" if label == "PAA" else "%m/%d/%Y"
            record, payable = (source_date(row[field], pattern) for field in ("record_date_text", "payable_date_text"))
            ticker = "ETE" if label == "ET" and record < ET_FIRST else label
            amount = Decimal(row["reported_amount_usd"])
            release = releases.pop((ticker, record.isoformat()), None)
            if release:
                if release["payable_date"] != payable.isoformat() or Decimal(release["reported_amount"]) != amount:
                    raise ValueError(f"Distribution table disagrees with the issuer release for {ticker} on {record}")
                ex, declared = (dt.date.fromisoformat(release[field]) for field in ("ex_date", "declared_date"))
                yield (ticker, ex, record, declared, payable, amount, None, "USD", "issuer_ex_date_disclosed", release["source_url"])
            else:
                yield (ticker, None, record, None, payable, amount, None, "USD", "issuer_common_unit_history", row["source_url"])
    if releases:
        raise ValueError("An explicit ex-date does not match an issuer history record")
    for cells in table_rows(get(PSX, ARCHIVE / "PSX_common_distribution_history.html").decode()):
        if len(cells) != 4:
            continue
        amount = re.fullmatch(r"(\$\d+\.\d+|\d+ cents) per share on Phillips 66 common stock", cells[3])
        if amount:
            declared, record, payable = (source_date(value, "%m/%d/%Y") for value in cells[:3])
            original = amount[1]
            value = Decimal(original[1:]) if original.startswith("$") else Decimal(original.split()[0]) / 100
            yield ("PSX", None, record, declared, payable, value, None, "USD", "issuer_common_stock_history", PSX)
    for cells in table_rows(get(TRP, ARCHIVE / "TRP_common_distribution_history.html").decode()):
        if len(cells) != 5 or not re.fullmatch(r"Q[1-4]", cells[0]) or not re.fullmatch(r"\$\d+\.\d+", cells[4]):
            continue
        dates = (value.strip("* ").replace("Sept ", "Sep ") for value in cells[1:4])
        payable, record, ex = (source_date(value, "%B %d, %Y" if len(value.split()[0]) > 3 else "%b %d, %Y") for value in dates)
        yield ("TRP", ex, record, None, payable, Decimal(cells[4][1:]), None, "CAD", "issuer_common_share_ex_date_cad", TRP)
    yield from (row for row in issuer_rows(AS_OF) if row[0] == "NGL")


def calendar_ex_date(sessions, record):
    record = record.isoformat()
    exceptions = {
        "2017-09-05": "2017-08-31",
        "2017-09-06": "2017-09-01",
        "2022-11-14": "2022-11-10",  # FINRA Veteran's Day non-settlement date, November 11.
        "2024-05-28": "2024-05-24",
    }
    if record in exceptions:
        return exceptions[record]
    prior_or_same = bisect_right(sessions, record) - 1
    offset = 2 if record < "2017-09-05" else 1 if record < "2024-05-28" else 0
    if prior_or_same < offset:
        return None
    return sessions[prior_or_same - offset]


def implied_cash(quote, earlier, later):
    if not earlier or not later:
        return None
    p, adjusted, c = quote.get(earlier, (None, None, None))
    q, next_adjusted, next_c = quote.get(later, (None, None, None))
    if not all((p, adjusted, q, next_adjusted)) or not c or c != next_c:
        return None
    return p * (1 - (adjusted / p) / (next_adjusted / q))


def reconcile(source, sessions, quotes, archived, archive_path):
    ticker, ex, record, declared, payable, amount, rebased, currency, basis, url = source
    if not record <= payable or (declared and declared > record) or (ex and ex > record) or amount <= 0:
        raise ValueError(f"Invalid dates or amount for {ticker} on {record}")
    candidate = previous = broker_symbol = cash = difference = rule = archive_amount = None
    admitted = ex
    if currency == "USD":
        candidate = calendar_ex_date(sessions, record)
        archive_amount = archived.get((ticker, candidate))
        if ticker in ("PAA", "PAGP") and record >= PAA_NASDAQ_FIRST:
            rule = NASDAQ_TRANSITION
        else:
            rule = NYSE_T2 if record < dt.date(2024, 5, 28) else NYSE_T1
        broker_symbol = "ET" if ticker == "ETE" else ticker
        if candidate:
            index = sessions.index(candidate)
            previous = sessions[index - 1] if index else None
        cash = implied_cash(quotes.get(broker_symbol, {}), previous, candidate)
        if cash is not None:
            difference = cash - float(amount)
        if ex is not None:
            if difference is not None and abs(difference) > 0.01:
                basis = "issuer_ex_date_disclosed_broker_cash_disagrees"
        elif difference is not None and abs(difference) <= 0.01:
            admitted = dt.date.fromisoformat(candidate)
            basis = "issuer_record_exchange_rule_ibkr_adjustment_agrees"
        elif archive_amount == amount:
            admitted = dt.date.fromisoformat(candidate)
            basis = "issuer_record_exchange_rule_dolt_agrees"
        elif difference is not None:
            basis = "issuer_record_broker_cash_mismatch_ex_date_unverified"
        else:
            basis = "issuer_record_broker_cash_unavailable_ex_date_unverified"
    result = (ticker, admitted, record, declared, payable, amount, rebased, currency, basis, url)
    check = (ticker, record, payable, currency, amount, ex, candidate, rule, broker_symbol, previous, cash, difference,
             admitted, basis, url, archive_amount, archive_path if archive_amount is not None else None)
    return result, check


def main():
    archive, archive_path = pinned_dividend_rows()
    archived = {(r["act_symbol"], r["ex_date"]): Decimal(r["amount"]) for r in archive if r["act_symbol"] == "ETE"}
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as db:
        sessions = [day for day, in db.execute("SELECT session_date FROM equity_session_calendar WHERE venue='XNYS' ORDER BY 1")]
        quotes = {}
        for ticker, day, price, adjusted, conid in db.execute("""SELECT instrument,session_date,close,adjusted_ibkr_close,conid
            FROM market_study_quote WHERE instrument IN ('PAA','ET','NGL','PSX')
            AND price_basis='ibkr_trades_split_adjusted' AND adjusted_ibkr_close>0"""):
            quotes.setdefault(ticker, {})[day] = price, adjusted, conid
    result = sorted((reconcile(row, sessions, quotes, archived, archive_path.relative_to(ROOT))
                     for row in declared_cash() if FIRST <= row[2] <= AS_OF),
                    key=lambda item: (item[0][0], item[0][2]))
    rows, checks = zip(*result, strict=True)
    if len({(r[0], r[2]) for r in rows}) != len(rows):
        raise ValueError("Duplicate distribution record date")
    write_csv(CASH_OUTPUT, FIELDS, rows)
    write_csv(DERIVED / "operator_issuer_cash_ex_date_evidence.csv", CHECKS, checks)
    instruments = {r[0] for r in rows}
    refresh_cash(rows, instruments)
    report = dict(as_of=AS_OF.isoformat(), total=len(rows), by_ticker=dict(Counter(r[0] for r in rows)),
        ex_date_with_broker_agreement=dict(Counter(r[0] for r in rows if r[8] == "issuer_record_exchange_rule_ibkr_adjustment_agrees")),
        issuer_cad_ex_date=dict(Counter(r[0] for r in rows if r[7] == "CAD")),
        issuer_ex_date_with_broker_cash_mismatch=dict(Counter(r[0] for r in rows if r[8] == "issuer_ex_date_disclosed_broker_cash_disagrees")),
        ex_date_with_pinned_archive_agreement=dict(Counter(r[0] for r in rows if r[8] == "issuer_record_exchange_rule_dolt_agrees")),
        unresolved_ex_date=dict(Counter(r[0] for r in rows if r[1] is None)),
        historical_ticker_note="ETE distributions precede the legal ticker transition. IBKR ET back history is only a cross-check.")
    write(DERIVED / "operator_issuer_cash_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
