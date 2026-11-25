"""Recover dated cash distributions for Tallgrass and the two historical ETP listings."""
import datetime as dt
import html
import re
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal

from build_legacy_operator_cash import build_rows
from build_operator_issuer_cash import DB, NYSE_T2, calendar_ex_date
from collect_dolt_equity_history import pinned_dividend_rows
from collect_issuer_distributions import ARCHIVE, DERIVED, FIELDS, refresh_cash
from collect_public import ROOT, write_csv
from publication_clocks import table_rows

TALLGRASS = {"TEP", "TEGP", "TGE"}
ENERGY_TRANSFER = {"ETP", "SXL"}
SEC = "https://www.sec.gov/Archives/edgar/data/"
DATE = r"[A-Z][a-z]+ \d{1,2}, \d{4}"
EXPECTED = {"TEP": 10, "TEGP": 10, "TGE": 6, "ETP": 11, "SXL": 5}
ENERGY_REPORTS = (
    ("legacy_etp_2016_10k", "1012569/000101256917000022/etp12-31x201610k.htm",
     "ETP", "energy_transfer_partners_old", "February 16, 2016"),
    ("sxl_2016_recast", "1161154/000116115417000048/etp12-31x2016xrecastex991.htm",
     "SXL", "sunoco_logistics", "0.4790"),
    ("etp_2017_10k", "1161154/000116115418000018/etp12-31x201710k.htm",
     "ETP", "energy_transfer_partners_after_sxl", "May 16, 2017"),
    ("etp_2018_q3", "1161154/000116115418000068/eto09-30x201810xq.htm",
     "ETP", "energy_transfer_partners_after_sxl", "May 7, 2018"),
)


def iso_date(value):
    return dt.datetime.strptime(value, "%B %d, %Y").date().isoformat()


def tallgrass_sources():
    for path in sorted((ARCHIVE / "tallgrass").glob("*Q[1-4].txt")):
        raw = path.read_text(errors="replace")
        header, _ = raw.split("<DOCUMENT>", 1)
        accession = re.search(r"ACCESSION NUMBER:\s+([\d-]+)", header)[1]
        cik = int(re.search(r"CENTRAL INDEX KEY:\s+(\d+)", header)[1])
        filed = dt.datetime.strptime(re.search(r"FILED AS OF DATE:\s+(\d+)", header)[1], "%Y%m%d").date()
        accepted = dt.datetime.strptime(re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", header)[1], "%Y%m%d%H%M%S")
        declared = iso_date(re.search(r"Date of Report.*?(" + DATE + ")",
                                       " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw[:45000])).split()),
                                       re.I)[1])
        exhibit, = re.findall(r"<DOCUMENT>\s*<TYPE>EX-99\.1\s.*?</DOCUMENT>", raw, re.S)
        filename = re.search(r"<FILENAME>([^\n]+)", exhibit)[1].strip()
        source = f"{SEC}{cik}/{accession.replace('-', '')}/{filename}"
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", exhibit)).split())
        payable = iso_date(re.search(r"will be paid on (?:\w+, )?(" + DATE + ")", text)[1])
        record = iso_date(re.search(r"of record(?: as of the close of business)? on (?:\w+, )?(" + DATE + ")", text)[1])
        amounts = re.findall(r"declared a quarterly cash (?:dividend|distribution) (?:to Class A shareholders )?"
                             r"of \$(\d+\.\d+) per (common unit|Class A share)", text, re.I)
        expected = {"common unit": "TEP", "class a share": "TEGP"} if path.stem <= "2018Q1" else {"class a share": "TGE"}
        if {kind.lower() for _, kind in amounts} != expected.keys() or len(amounts) != len(expected):
            raise ValueError(f"Unexpected classes of shares in {path.name}: {amounts}")
        for amount, kind in amounts:
            yield dict(instrument=expected[kind.lower()], record_date=record, declared_date=declared, payable_date=payable,
                       reported_amount=amount, source_url=source, source_form="8-K Exhibit 99.1",
                       source_filed_date=filed.isoformat(), accepted_at_eastern=accepted.isoformat(),
                       quarter=path.stem, archive_file=str(path.relative_to(ROOT)))


def energy_transfer_rows(calendar, listings, archive, quotes, archive_path):
    ticker_archive = {day: amount for (ticker, day), amount in archive.items() if ticker == "ETP"}
    matched = set()
    for ticker, listing, quarter, record, payable, amount, url, path in energy_transfer_sources(listings):
        first, last, venue = listings[listing]
        ex = calendar_ex_date(calendar, dt.date.fromisoformat(record))
        if not first <= ex <= record <= last or payable < record or amount <= 0:
            raise ValueError(f"Invalid {listing} cash event on {record}")
        archived = ticker_archive.get(ex)
        if amount == archived:
            matched.add(ex)
            check = "dolt_etp_reidentified_as_sxl" if ticker == "SXL" else "dolt_etp_agrees"
        elif listing == "energy_transfer_partners_old":
            check = "dolt_etp_symbol_collision"
        else:
            raise ValueError(f"Cash archive disagrees with the {listing} issuer on {ex}")
        basis = f"issuer_record_exchange_rule_{check}"
        cash = (ticker, ex, record, None, payable, amount, None, "USD", basis, url)
        evidence = dict(instrument=ticker, listing_id=listing, quarter_ending=quarter, ex_date=ex, record_date=record,
                        payable_date=payable, reported_amount=amount,
                        pinned_archive_symbol="ETP" if archived is not None else None, pinned_archive_amount=archived,
                        archive_match=check, listing_venue=venue, observed_bar_on_ex_date=(ticker, ex) in quotes,
                        issuer_source_url=url, issuer_archive_file=path.relative_to(ROOT), rule_source_url=NYSE_T2,
                        pinned_archive_file=archive_path)
        yield cash, evidence
    if set(ticker_archive) != matched:
        raise ValueError(f"Unreconciled Dolt ETP rows: {sorted(set(ticker_archive) - matched)}")


def energy_transfer_sources(listings):
    seen = {}
    for name, document, ticker, listing, anchor in ENERGY_REPORTS:
        path = ARCHIVE / "energy_transfer" / f"{name}.txt"
        filename = document.rsplit("/", 1)[-1]
        raw = path.read_text(errors="replace")
        document_body, = (part for part in raw.split("<DOCUMENT>") if f"<FILENAME>{filename}" in part[:220])
        tables = set()
        for table in re.findall(r"<table\b[^>]*>.*?</table>", document_body.split("</DOCUMENT>", 1)[0], re.I | re.S):
            if "Quarter" not in table or "Record" not in table:
                continue
            rows = [tuple(c for c in r if c and c != "$") for r in table_rows(table) if any(r)]
            if rows and rows[0] == ("Quarter Ended", "Record Date", "Payment Date", "Rate") and any(anchor in r for r in rows):
                tables.add(tuple(rows[1:]))
        table, = tables
        first, last, _ = listings[listing]
        for row in table:
            quarter, record, payable = (iso_date(value) for value in row[:3])
            amount = Decimal(row[3])
            if not first <= record <= last:
                continue
            identity = listing, record
            if identity in seen:
                if seen[identity] != (quarter, payable, amount):
                    raise ValueError(f"Two reports disagree on {listing} for {record}")
                continue
            seen[identity] = (quarter, payable, amount)
            yield ticker, listing, quarter, record, payable, amount, SEC + document, path


def main():
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as db:
        calendar = [day for day, in db.execute("SELECT session_date FROM equity_session_calendar WHERE venue='XNYS' ORDER BY 1")]
        listings = {key: (first, last, venue) for key, first, last, venue in db.execute(
            """SELECT listing_id,first_study_session,last_study_session,exchange FROM historical_equity_listing_window
                WHERE symbol IN ('ETP','SXL')""")}
        tallgrass = {ticker: (first, last, venue) for ticker, first, last, venue in db.execute(
            """SELECT symbol,first_study_session,last_study_session,exchange FROM historical_equity_listing_window
                WHERE symbol IN ('TEP','TEGP','TGE')""")}
        quotes = set(db.execute("""SELECT instrument,session_date FROM market_study_quote
            WHERE instrument IN ('TEP','TEGP','TGE','ETP','SXL')"""))
    archive_rows, archive_path = pinned_dividend_rows()
    archive_path = archive_path.relative_to(ROOT)
    archive = {(r["act_symbol"], r["ex_date"]): Decimal(r["amount"]) for r in archive_rows
               if r["act_symbol"] in TALLGRASS | ENERGY_TRANSFER}
    sources = list(tallgrass_sources())
    by_record = {(r["instrument"], r["record_date"]): r for r in sources}
    events, checks = zip(*build_rows(sources, calendar, tallgrass,
                                     {key: a for key, a in archive.items() if key[0] in TALLGRASS},
                                     quotes, archive_path), strict=True)
    tallgrass_checks = []
    for row in checks:
        source = by_record[row["instrument"], row["record_date"]]
        declared = row.pop("announced_date")
        tallgrass_checks.append({**row, "declared_date": declared,
                                 "edgar_accepted_at_eastern": source["accepted_at_eastern"],
                                 "quarter": source["quarter"], "issuer_archive_file": source["archive_file"]})
    etp, etp_checks = zip(*energy_transfer_rows(calendar, listings, archive, quotes, archive_path), strict=True)
    rows = sorted((*events, *etp))
    if dict(Counter(row[0] for row in rows)) != EXPECTED or len(rows) != len({(r[0], r[1]) for r in rows}):
        raise ValueError("Historical midstream distribution coverage disagrees with the issuer sequence")
    write_csv(DERIVED / "historical_midstream_issuer_cash.csv", FIELDS, rows)
    for name, evidence in (("tallgrass_cash_ex_date_evidence", tallgrass_checks),
                           ("energy_transfer_cash_ex_date_evidence", etp_checks)):
        write_csv(DERIVED / f"{name}.csv", tuple(evidence[0]), (r.values() for r in evidence))
    refresh_cash(rows, TALLGRASS | ENERGY_TRANSFER)
    print(dict(Counter(row[0] for row in rows)), flush=True)


if __name__ == "__main__":
    main()
