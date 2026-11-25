"""Normalize issuer cash distributions and check trading dates against two price archives."""
import csv
import datetime as dt
import json
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal

from build_operator_issuer_cash import AS_OF, CHECKS, DB, FIRST, reconcile
from collect_dolt_equity_history import REF, STORE
from collect_issuer_distributions import ARCHIVE, DERIVED, FIELDS, refresh_cash
from collect_public import ROOT, write, write_csv

SOURCE = ARCHIVE / "additional_us_cash_history_extract.csv"
CHECK_FIELDS = (*CHECKS, "issuer_table_amount_basis", "supplemental_issuer_source_url",
                "archive_ex_date_match", "archive_amount_minus_issuer_amount", "issuer_original_pre_split_amount_usd",
                "pre_split_shares_per_post_split_share", "pre_split_amount_source_url", "split_source_url")


def read_issuer_rows():
    with SOURCE.open(newline="") as stream:
        for row in csv.DictReader(stream):
            ticker = row["instrument"]
            ex, record, declared, payable = (
                dt.date.fromisoformat(row[field]) if row[field] else None
                for field in ("ex_date", "record_date", "declared_date", "payable_date")
            )
            if FIRST <= record <= AS_OF:
                amount = Decimal(row["issuer_table_amount_usd"])
                basis = "issuer_ex_date_disclosed" if ex else "issuer_common_share_history"
                yield ((ticker, ex, record, declared, payable, amount, None, "USD", basis, row["source_url"]),
                       row["amount_basis"], row["supplemental_source_url"])


def read_archive(tickers):
    rows, paths = {}, {}
    for ticker in tickers:
        path = STORE / f"{ticker.lower()}_dividend.json"
        payload = json.loads(path.read_text())
        if (payload.get("query_execution_status") != "Success"
                or f"dividend AS OF '{REF}'" not in payload.get("sql_query", "")):
            raise ValueError(f"Unverified archive dividend rows for {ticker}")
        paths[ticker] = path.relative_to(ROOT)
        for row in payload["rows"]:
            key = ticker, row["ex_date"]
            if row["act_symbol"] != ticker or key in rows:
                raise ValueError(f"Invalid or duplicate archive cash row {key}")
            rows[key] = Decimal(row["amount"])
    return rows, paths


def main():
    sources = list(read_issuer_rows())
    tickers = sorted({row[0] for row, _, _ in sources})
    archived, paths = read_archive(tickers)
    with (ARCHIVE / "PAGP_2016_pre_split_cash_extract.csv").open(newline="") as stream:
        originals = {row["record_date"]: row for row in csv.DictReader(stream)}
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as db:
        sessions = [day for day, in db.execute(
            "SELECT session_date FROM equity_session_calendar WHERE venue='XNYS' ORDER BY 1")]
        quotes = {ticker: {} for ticker in tickers}
        placeholders = ",".join("?" for _ in tickers)
        for ticker, day, price, adjusted, conid in db.execute(
                "SELECT instrument,session_date,close,adjusted_ibkr_close,conid FROM market_study_quote "
                f"WHERE instrument IN ({placeholders}) AND currency='USD' "
                "AND price_basis='ibkr_trades_split_adjusted' AND adjusted_ibkr_close>0", tickers):
            quotes[ticker][day] = price, adjusted, conid
    rows, checks = [], []
    for source, amount_basis, extra_url in sources:
        ticker = source[0]
        original = factor = original_url = split_url = None
        if amount_basis == "issuer_split_adjusted":
            native = originals.pop(source[2].isoformat())
            original, factor = (Decimal(native[name]) for name in
                                ("original_amount_usd", "pre_split_shares_per_post_split_share"))
            original_url, split_url = native["original_amount_source_url"], native["split_source_url"]
            if native["payable_date"] != source[4].isoformat() or round(original * factor, 4) != source[5]:
                raise ValueError(f"Split basis does not reconcile for {ticker} on {source[2]}")
            if native["declared_date"]:
                source = (*source[:3], dt.date.fromisoformat(native["declared_date"]), *source[4:])
        row, check = reconcile(source, sessions, quotes, archived, paths[ticker])
        if amount_basis == "issuer_split_adjusted":
            row = (*row[:8], f"{row[8]}_issuer_split_adjusted_amount", row[9])
        # Explicit dates can be independently compared even if the calendar candidate differs.
        comparison_date = row[1].isoformat() if row[1] else check[6]
        archive_cash = archived.get((ticker, comparison_date))
        check = (*check[:13], row[8], check[14], archive_cash, paths[ticker] if archive_cash is not None else None)
        diff = archive_cash - row[5] if archive_cash is not None else None
        rows.append(row)
        checks.append((*check, amount_basis, extra_url, archive_cash is not None, diff,
                       original, factor, original_url, split_url))
    if originals:
        raise ValueError("Unmatched original PAGP cash rows")
    if len({(r[0], r[2]) for r in rows}) != len(rows):
        raise ValueError("Duplicate issuer record dates")
    expected = {"EPD": 44, "KMI": 44, "PAGP": 44, "WMB": 43}
    if dict(Counter(row[0] for row in rows)) != expected:
        raise ValueError(f"Unexpected cash coverage: {Counter(row[0] for row in rows)}")
    write_csv(DERIVED / "additional_us_issuer_cash_distributions.csv", FIELDS, rows)
    write_csv(DERIVED / "additional_us_cash_ex_date_evidence.csv", CHECK_FIELDS, checks)
    refresh_cash(rows, set(tickers))
    audit = [dict(zip(CHECK_FIELDS, check, strict=True)) for check in checks]
    summary = {
        "issuer_rows": expected,
        "admitted_ex_dates": dict(Counter(row[0] for row in rows if row[1])),
        "ibkr_amount_agreement_within_one_cent": dict(Counter(
            row["instrument"] for row in audit if row["cash_difference_usd"] is not None
            and abs(row["cash_difference_usd"]) <= 0.01)),
        "archive_date_match": dict(Counter(row["instrument"] for row in audit if row["archive_ex_date_match"])),
        "archive_amount_exact": dict(Counter(
            row["instrument"] for row in audit if row["archive_amount_minus_issuer_amount"] == 0)),
        "issuer_split_adjusted_amounts": dict(Counter(
            row["instrument"] for row in audit if row["issuer_table_amount_basis"] == "issuer_split_adjusted")),
    }
    write(DERIVED / "additional_us_cash_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
