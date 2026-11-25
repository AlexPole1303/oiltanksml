"""Preserve primary distribution disclosures and normalize the disclosed cash events."""
import argparse
import datetime as dt
import json
import re
import sqlite3
from contextlib import closing
from decimal import Decimal
from urllib.parse import quote

import pdfplumber

from collect_public import RAW, ROOT, get, write_csv
from publication_clocks import export, table_rows

ARCHIVE = RAW / "issuer_distributions"
DERIVED = ROOT / "data" / "derived"
OKE = "https://ir.oneok.com/stock-information/dividend-history"
OKE_JAN_2016 = "https://ir.oneok.com/news-and-events/press-releases/archive/01-21-2016-212215342"
NGL = "https://www.nglenergypartners.com/investor-relations/distribution-history/"
ALPS = "https://www.alpsfunds.com/exchange-traded-funds/"
ALPS_DIVIDEND = "https://secure.alpsinc.com/MarketingAPI/api/v1/Dividend/"
FUND_CUSIPS = {"AMLP": "00162Q452", "ENFR": "00162Q676"}
FIELDS = ("instrument", "ex_date", "record_date", "declared_date", "payable_date", "reported_amount",
          "amount_on_post_2020_amlp_share_basis", "currency", "source_basis", "source_url")
AMLP_SPLIT = dt.date(2020, 5, 18)
DATE = r"(\d{1,2}/\d{1,2}/(?:20)?\d{2})"
AMOUNT = re.compile(rf"^{DATE}\s+{DATE}\s+{DATE}\s+\$?\s*(\d+\s*\.\s*\d+)")


def source_date(value, pattern):
    return dt.datetime.strptime(value, pattern).date()


def issuer_rows(as_of):
    oneok = get(OKE, ARCHIVE / "OKE.html").decode()
    for cells in table_rows(oneok):
        if len(cells) != 6 or cells[-1] != "U.S. Currency":
            continue
        ex = source_date(cells[0], "%m/%d/%y")
        if dt.date(2016, 1, 1) <= ex <= as_of:
            record, declared, payable = (source_date(value, "%m/%d/%y") for value in cells[1:4])
            source = OKE
            if ex == dt.date(2016, 1, 28):
                get(OKE_JAN_2016, ARCHIVE / "OKE_2016_01_21.html")
                declared, payable = dt.date(2016, 1, 21), dt.date(2016, 2, 12)
                source = OKE_JAN_2016
            if not declared <= ex <= record <= payable:
                raise ValueError(f"Inconsistent ONEOK cash-distribution dates for {ex}")
            yield ("OKE", ex, record, declared, payable, Decimal(cells[4].replace("$", "")),
                   None, "USD", "issuer_cash_history", source)

    ngl = get(NGL, ARCHIVE / "NGL_corporate.html").decode()
    common_units = re.search(r"NGL Common Unit Distribution.*?(<table\b[^>]*>.*?</table>)", ngl, re.I | re.S)
    if common_units is None:
        raise ValueError("Could not locate the NGL common-unit table")
    for row in table_rows(common_units.group(1)):
        cells = [value for value in row if value]
        if len(cells) != 5 or cells[-1] != "U.S. Currency":
            continue
        declared, record, payable = (source_date(value, "%B %d, %Y") for value in cells[:3])
        if dt.date(2016, 1, 1) <= declared <= as_of:
            source = NGL
            if declared == dt.date(2019, 7, 23):
                if record != dt.date(2019, 8, 7):
                    raise ValueError("NGL July 2019 record date disagrees with the 10-Q")
                source = "https://www.sec.gov/Archives/edgar/data/1504461/000150446119000047/ngl-06302019x10q.htm"
            yield ("NGL", None, record, declared, payable, Decimal(cells[3].replace("$", "")),
                   None, "USD", "issuer_cash_history", source)


def fund_rows(as_of):
    for fund in ("AMLP", "ENFR"):
        html = get(ALPS + fund.lower(), ARCHIVE / f"{fund}.html").decode()
        urls = sorted(set(re.findall(r'https://www\.alpsfunds\.com/hubfs/[^"\s]+-8937-\d{8}\.pdf', html)))
        for url in urls:
            suffix = url.rsplit("-", 1)[-1]
            if f"{fund.lower()}-8937" not in url or not suffix.endswith("1130.pdf"):
                continue
            path = ARCHIVE / f"{fund}_8937_{suffix}"
            get(url, path)
            with pdfplumber.open(path) as document:
                lines = [line.strip() for page in document.pages
                         for line in (page.extract_text() or "").splitlines()]
            matched = [match.groups() for line in lines if (match := AMOUNT.match(line))]
            if not matched:
                raise ValueError(f"Could not parse distribution rows in {path.name}")
            for first, second, payment, amount in matched:
                ex_date, record_date = (first, second) if fund == "AMLP" else (second, first)
                ex, record, payable = (source_date(value, "%m/%d/%Y" if len(value.rsplit("/", 1)[-1]) == 4 else "%m/%d/%y")
                                       for value in (ex_date, record_date, payment))
                if not dt.date(2016, 1, 1) <= ex <= as_of:
                    continue
                if not ex <= record <= payable:
                    raise ValueError(f"Date columns in {path.name} disagree with expected order")
                reported = Decimal(amount.replace(" ", ""))
                rebased = reported * (5 if ex < AMLP_SPLIT else 1) if fund == "AMLP" else None
                yield (fund, ex, record, None, payable, reported, rebased, "USD", "issuer_form_8937", url)


def fund_feed(as_of, tax_forms):
    final = {(row[0], row[2]): row for row in tax_forms}
    checked = []
    for fund, cusip in FUND_CUSIPS.items():
        fund_url = "https://www.alpsfunds.com/_hcms/api/getData?api_url=" + quote(ALPS_DIVIDEND + fund, safe="")
        events = json.loads(get(fund_url, ARCHIVE / f"{fund}_official_dividend_feed.json"))
        if not isinstance(events, list) or not events:
            raise ValueError(f"Missing issuer feed for {fund}")
        seen = set()
        for event in events:
            ex = dt.date.fromisoformat(event["exdate"][:10])
            if not dt.date(2016, 1, 1) <= ex <= as_of or event["totalisnull"]:
                continue
            record = dt.date.fromisoformat(event["recorddate"][:10])
            payable = dt.date.fromisoformat(event["payabledate"][:10])
            declared = dt.date.fromisoformat(event["declarationdate"][:10]) if event["declarationdate"] else None
            total = Decimal(event["total"])
            identity = fund, record
            if (event["symbol"] != fund or event["cusip"] != cusip or identity in seen
                    or not ex <= record <= payable or (declared is not None and declared > ex) or total <= 0):
                raise ValueError(f"Invalid issuer-feed cash event for {fund} on {ex}")
            seen.add(identity)
            form = final.get(identity)
            comparison = "feed_only"
            if form:
                form_amount = form[6] if fund == "AMLP" else form[5]
                if form_amount != total or form[4] != payable:
                    raise ValueError(f"Issuer feed and tax form disagree on amount or payment date for {identity}")
                comparison = "matches_form" if form[1] == ex else "ex_date_disagrees_tax_form_used"
            else:
                original = total / 5 if fund == "AMLP" and ex < AMLP_SPLIT else total
                basis = "issuer_fund_feed_total_estimated_flag" if event["estimated"] else "issuer_fund_feed_total"
                final[identity] = (fund, ex, record, declared, payable, original,
                                   total if fund == "AMLP" else None, "USD", basis, fund_url)
            checked.append((fund, ex, record, payable, total, event["estimated"], form[1] if form else None,
                            comparison, form[9] if form else None, fund_url))
    return list(final.values()), checked


def replace_cash_records(db, rows, instruments):
    instruments = tuple(sorted(instruments))
    placeholders = ",".join("?" for _ in instruments)
    db.execute(f"DELETE FROM issuer_cash_distribution WHERE instrument IN ({placeholders})", instruments)
    db.executemany("INSERT INTO issuer_cash_distribution VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (tuple(str(value) if value is not None else None for value in row)
                    for row in rows if row[0] in instruments))


def refresh_cash(rows, instruments):
    instruments = tuple(sorted(instruments))
    placeholders = ",".join("?" for _ in instruments)
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        db.execute("BEGIN IMMEDIATE")
        with db:
            replace_cash_records(db, rows, instruments)
            db.execute(f"""UPDATE market_study_quote AS q SET
                (issuer_cash_on_ex_date,issuer_cash_post_2020_amlp_basis,issuer_cash_source_url)=(
                    SELECT reported_amount,amount_on_post_2020_amlp_share_basis,source_url
                    FROM market_study_issuer_cash_distribution d WHERE d.instrument=q.instrument AND d.ex_date=q.session_date
                    AND d.currency=q.currency
                ) WHERE q.instrument IN ({placeholders})""", instruments)
        export(db, "market_study_quote")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-database", action="store_true")
    args = parser.parse_args()
    as_of = dt.date(2026, 11, 11)
    funds, checks = fund_feed(as_of, list(fund_rows(as_of)))
    rows = sorted([*issuer_rows(as_of), *funds], key=lambda row: (row[0], row[4]))
    keys = {(row[0], row[1] or row[2]) for row in rows}
    if len(rows) != len(keys):
        raise ValueError("Issuer cash records have duplicate instrument and entitlement dates")
    write_csv(DERIVED / "issuer_cash_distributions.csv", FIELDS, rows)
    write_csv(DERIVED / "fund_cash_distribution_crosscheck.csv",
              ("instrument", "feed_ex_date", "record_date", "payable_date", "feed_total", "feed_estimated_flag",
               "form_ex_date", "comparison", "form_source_url", "feed_source_url"), sorted(checks))
    if args.refresh_database:
        refresh_cash(rows, FUND_CUSIPS)
    print(f"{len(rows)} issuer cash distributions")


if __name__ == "__main__":
    main()
