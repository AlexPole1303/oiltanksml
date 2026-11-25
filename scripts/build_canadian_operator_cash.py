"""Build common-share cash ledgers and a separate reference-currency check."""
import csv
import datetime as dt
import json
import re
import sqlite3
from bisect import bisect_right
from calendar import month_name
from collections import Counter
from contextlib import closing
from decimal import Decimal
from html import unescape

import pdfplumber

from build_operator_issuer_cash import AS_OF, DB, NYSE_T1, NYSE_T2, calendar_ex_date, implied_cash
from collect_dolt_equity_history import REF, STORE
from collect_issuer_distributions import ARCHIVE, DERIVED, FIELDS, refresh_cash
from collect_public import ROOT, write, write_csv

NOTICES = ARCHIVE / "enb_notices"
ENB_URL = "https://www.enbridge.com/media-center/news/details?id={}&lang=en"
FINRA_2016 = "https://www.finra.org/sites/default/files/notice_doc_file_ref/Information-Notice_122215.pdf"
FX_POLICIES = {
    "ENB": ("record_date", "https://www.enbridge.com/Investment-Center/FAQs/Dividend-QA.aspx"),
    "TRP": ("payable_date", "https://prep.tcenergy.com/investors/stock-information/frequently-asked-questions/"),
}
DATE = r"\w+ \d{1,2}\s*, \d{4}"
COMMON = re.compile(
    rf"declared a quarterly (?:common share )?dividend of \$(\d+\.\d+) per (?:common )?share,"
    rf"? payable on ({DATE}),? to shareholders of record on ({DATE})", re.I)
DECLARATION = re.compile(
    rf"On ({DATE})\s*, (?:the Enbridge Board of Directors|(?:Enbridge's |the Company's |its |our )?Board of Directors)"
    rf" declared the following quarterly dividends", re.I)
FX_FIELDS = ("instrument", "record_date", "payable_date", "reported_amount_cad", "policy_date_basis",
             "policy_date", "policy_source_url", "fx_observation_date", "fx_date_basis", "cad_per_usd",
             "fx_series", "fx_available_at_utc", "fx_source_url", "gross_usd_reference_per_share",
             "calendar_or_issuer_ex_date", "broker_previous_session", "ibkr_adjustment_in_quote_price_units",
             "ibkr_adjustment_minus_usd_reference", "ibkr_adjustment_minus_reported_cad_number",
             "ibkr_adjustment_divided_by_reported_cad_number",
             "admitted_ex_date", "ex_date_basis", "entitlement_calendar_source_url", "distribution_source_url",
             "dolt_archive_amount_as_reported", "dolt_archive_source_file")
ARCHIVE_FIELDS = ("instrument", "archive_ex_date", "archive_amount_as_reported", "matching_issuer_ex_date",
                  "archive_source_file", "date_status")
KEY_FIELDS = ("table_ex_date", "table_record_date", "payable_date", "reported_amount_cad",
              "used_ex_date", "used_record_date", "ibkr_implied_cash_cad", "source_basis", "source_url")
KEY_EARLY = ROOT / "data/reference/keyera_early_distribution_history.csv"
KEY_TAX = "https://www.keyera.com/assets/Attachments/Tax-Information/"
KEY_EARLY_SOURCES = {
    "tax_2016": KEY_TAX + "2016-Tax-Information-for-Canadian-Residents.pdf",
    "tax_2017": KEY_TAX + "2017-Tax-Information-for-Canadian-Residents.pdf",
    "tax_2018": KEY_TAX + "2018-Tax-information-for-Canadian-Residents.pdf",
    "notice_2018_08": "https://www.keyera.com/news-and-stories/news-releases/keyera-announces-dividend-increase-and-august-dividend/",
    "notice_2018_12": "https://6ix.com/news/keyera-announces-december-2018-dividend",
}
SOBO_FIELDS = ("record_date", "payable_date", "reported_amount_usd", "nyse_calendar_ex_date",
               "broker_previous_session", "ibkr_adjustment_in_quote_price_units", "admitted_ex_date",
               "ex_date_basis", "source_url")


def extract(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def dolt_checks(checks):
    result, unmatched = [], []
    ex_index = FX_FIELDS.index("admitted_ex_date")
    known = {(row[0], row[ex_index]) for row in checks}
    archive = {}
    for ticker in FX_POLICIES:
        path = STORE / f"{ticker.lower()}_dividend.json"
        payload = json.loads(path.read_text())
        if (payload.get("query_execution_status") != "Success"
                or f"dividend AS OF '{REF}'" not in payload.get("sql_query", "")):
            raise ValueError(f"Invalid archive for {ticker}")
        for source in payload["rows"]:
            if source["act_symbol"] != ticker or (ticker, source["ex_date"]) in archive:
                raise ValueError(f"Invalid or duplicate archive cash row for {ticker}")
            day = source["ex_date"]
            amount, file = Decimal(source["amount"]), path.relative_to(ROOT)
            archive[ticker, day] = amount, file
            if (ticker, day) not in known and day <= AS_OF.isoformat():
                unmatched.append((ticker, day, amount, None, file, "archive_date_absent_from_issuer_ledger"))
    for row in checks:
        archived, file = archive.get((row[0], row[ex_index]), (None, None))
        result.append((*row, archived, file))
        if archived is None:
            unmatched.append((row[0], None, None, row[ex_index], None, "issuer_date_absent_from_archive"))
    return result, unmatched


def enb_distributions():
    for path in sorted(NOTICES.glob("release_*.html")):
        body = " ".join(unescape(re.sub(r"<[^>]+>", " ", path.read_text())).split())
        common = COMMON.search(body)
        if common is None:
            raise ValueError(f"Common share declaration missing from {path.name}")
        declaration = DECLARATION.search(body)
        # A regular declaration is announced that day unless a different board date is specified.
        announced = re.search(DATE, body)
        date_text = declaration[1] if declaration else announced[0]
        declared, record, payable = (
            dt.datetime.strptime(re.sub(r"\s+,", ",", value), "%B %d, %Y").date()
            for value in (date_text, common[3], common[2])
        )
        yield (declared, record, payable, Decimal(common[1]), ENB_URL.format(path.stem.split("_")[1]))
    for row in extract(NOTICES / "quarterly_filing_extract.csv"):
        declared = dt.date.fromisoformat(row["declared_date"]) if row["declared_date"] else None
        yield (declared, dt.date.fromisoformat(row["record_date"]), dt.date.fromisoformat(row["payable_date"]),
               Decimal(row["reported_amount_cad"]), row["source_url"])


def key_distributions(sessions, quote):
    yield from early_key_distributions(sessions, quote)
    fixes = {row["payable_date"]: row for row in extract(ARCHIVE / "KEY_release_corrections_extract.csv")}
    for original in extract(ARCHIVE / "KEY_common_distribution_history_extract.csv"):
        payable = original["payable_date"]
        correction = fixes.pop(payable, None)
        row = correction or original
        if row["reported_amount_cad"] != original["reported_amount_cad"]:
            raise ValueError(f"Keyera correction changes the table amount for {payable}")
        ex, record = row["ex_date"], row["record_date"]
        basis = "issuer_release_corrects_history_table" if correction else "issuer_common_share_ex_date_cad"
        admitted = ex if ex in sessions else None
        if admitted is None:
            basis = "issuer_ex_date_not_xtse_session"
        previous = prior_session(sessions, admitted)
        cash = implied_cash(quote, previous, admitted)
        amount = Decimal(row["reported_amount_cad"])
        declared = row.get("declared_date")
        if not ex <= record <= payable or (declared and declared > record) or amount <= 0:
            raise ValueError(f"Invalid Keyera declaration for {payable}")
        yield (("KEY", admitted, record, declared, payable, amount, None, "CAD", basis, row["source_url"]),
               (original["ex_date"], original["record_date"], payable, amount, admitted, record, cash,
                basis, row["source_url"]))
    if fixes:
        raise ValueError("An issuer correction does not match the Keyera table")


def early_key_distributions(sessions, quote):
    with pdfplumber.open(ARCHIVE / "KEY_2018_aif.pdf") as document:
        text = document.pages[61].extract_text()
    amounts = {}
    for month in range(1, 13):
        line = re.search(rf"(?m)^{month_name[month]} \$(\d+\.\d+) \$(\d+\.\d+) \$(\d+\.\d+)$", text)
        if not line:
            raise ValueError(f"Keyera AIF omits {month_name[month]}")
        amounts.update({(year, month): Decimal(value)
                        for year, value in zip((2018, 2017, 2016), line.groups(), strict=True)})
    for row in extract(KEY_EARLY):
        record, payable = (dt.date.fromisoformat(row[name]) for name in ("record_date", "payable_date"))
        amount = Decimal(row["reported_amount_cad"])
        expected = amounts.pop((record.year, record.month))
        candidate = calendar_ex_date(sessions, record)
        disclosed = row["ex_date"] or None
        if (amount != expected or not record < payable or not candidate
                or (disclosed and disclosed != candidate)):
            raise ValueError(f"Keyera's historical sources disagree on {record}")
        cash = implied_cash(quote, prior_session(sessions, candidate), candidate)
        if disclosed:
            admitted, basis = disclosed, "issuer_release_ex_date_cad"
        elif cash is not None and abs(cash - float(amount)) <= 0.01:
            admitted, basis = candidate, "issuer_tax_table_tsx_settlement_ibkr_adjustment_agrees"
        else:
            admitted, basis = None, "issuer_tax_table_tsx_ex_date_unverified"
        source = KEY_EARLY_SOURCES[row["source_document"]]
        record_day, payment_day = record.isoformat(), payable.isoformat()
        yield (("KEY", admitted, record_day, row["declared_date"] or None, payment_day, amount, None, "CAD", basis, source),
               (disclosed, record_day, payment_day, amount, admitted, record_day, cash, basis, source))
    if amounts:
        raise ValueError(f"Keyera historical dividends omit {sorted(amounts)}")


def prior_session(sessions, ex):
    if ex and ex in sessions:
        index = sessions.index(ex)
        return sessions[index - 1] if index else None
    return None


def sobo_distributions(sessions, quote):
    for row in extract(ARCHIVE / "SOBO_common_distribution_notices_extract.csv"):
        declared, record, payable = (dt.date.fromisoformat(row[name]) for name in
                                     ("declared_date", "record_date", "payable_date"))
        amount = Decimal(row["reported_amount_usd"])
        if not declared <= record <= payable or amount <= 0 or record > AS_OF:
            raise ValueError(f"Invalid South Bow declaration for {payable}")
        candidate = calendar_ex_date(sessions, record)
        previous = prior_session(sessions, candidate)
        cash = implied_cash(quote, previous, candidate)
        if cash is not None and abs(cash - float(amount)) <= 0.01:
            admitted, basis = candidate, "issuer_record_exchange_rule_ibkr_adjustment_agrees"
        else:
            admitted, basis = None, "issuer_record_nyse_ex_date_unverified"
        url = row["source_url"]
        yield (("SOBO", admitted, record, declared, payable, amount, None, "USD", basis, url),
               (record, payable, amount, candidate, previous, cash, admitted, basis, url))


def fx_reconciliation(ticker, declared, record, payable, amount, url, issuer_ex, sessions, quote, rates, fx_days):
    date_basis, policy = FX_POLICIES[ticker]
    policy_date = record if date_basis == "record_date" else payable
    date = policy_date.isoformat()
    if date in rates:
        observation, rate_basis = date, "same_observation_date"
    else:
        index = bisect_right(fx_days, date) - 1
        observation = fx_days[index] if index >= 0 else None
        if observation and (policy_date - dt.date.fromisoformat(observation)).days > 4:
            observation = None
        rate_basis = "prior_observation_reference_only" if observation else "missing_fx_reference"
    rate, series, available, fx_url = rates[observation] if observation else (None, None, None, None)
    reference = amount / Decimal(str(rate)) if rate else None
    candidate = issuer_ex or calendar_ex_date(sessions, record)
    calendar_source = url if issuer_ex else (NYSE_T1 if record >= dt.date(2024, 5, 28) else NYSE_T2)
    if ticker == "ENB" and record == dt.date(2016, 11, 15):
        # FINRA lists Nov 9 trading as settling Nov 15 and Nov 10 trading as settling Nov 16.
        candidate, calendar_source = "2016-11-10", FINRA_2016
    previous = prior_session(sessions, candidate)
    cash = implied_cash(quote, previous, candidate)
    difference = cash - float(reference) if cash is not None and reference is not None else None
    cad_difference = cash - float(amount) if cash is not None else None
    cad_ratio = cash / float(amount) if cash is not None else None
    if issuer_ex:
        admitted, basis = issuer_ex, "issuer_common_share_ex_date_cad"
    elif difference is not None and abs(difference) <= 0.01:
        admitted, basis = candidate, "issuer_record_exchange_rule_ibkr_fx_reference_agrees"
    elif cad_difference is not None and abs(cad_difference) <= 0.01:
        admitted, basis = candidate, "issuer_record_exchange_rule_ibkr_cad_numeric_agrees"
    else:
        admitted, basis = None, "issuer_record_nyse_ex_date_unverified"
    fx_row = (ticker, record, payable, amount, date_basis, policy_date, policy, observation, rate_basis, rate, series,
              available, fx_url, reference, candidate, previous, cash, difference, cad_difference, cad_ratio,
              admitted, basis, calendar_source, url)
    return (ticker, admitted, record, declared, payable, amount, None, "CAD", basis, url), fx_row


def main():
    with closing(sqlite3.connect(DB.as_uri() + "?mode=ro", uri=True)) as db:
        venues = {venue: [day for day, in db.execute(
            "SELECT session_date FROM equity_session_calendar WHERE venue=? ORDER BY 1", (venue,))]
                  for venue in ("XNYS", "XTSE")}
        rates = {day: (value, "DEXCAUS", None, "https://fred.stlouisfed.org/series/DEXCAUS")
                 for day, value in db.execute("SELECT date,value FROM macro_series WHERE series='DEXCAUS' "
                                              "AND date<'2017-01-01' AND value>0")}
        rates.update({day: (value, series, available, source) for day, value, series, available, source in db.execute(
            "SELECT observation_date,quote_per_base,provider_series,conservative_available_at_utc,source_url "
            "FROM cad_usd_reference_rate WHERE base_currency='USD' AND quote_currency='CAD' AND quote_per_base>0")})
        quotes = {ticker: {} for ticker in ("ENB", "TRP", "KEY", "SOBO")}
        for ticker, day, price, adjusted, conid in db.execute(
                "SELECT instrument,session_date,close,adjusted_ibkr_close,conid FROM market_study_quote "
                "WHERE price_basis='ibkr_trades_split_adjusted' AND adjusted_ibkr_close>0 "
                "AND ((instrument IN ('ENB','TRP','SOBO') AND currency='USD') OR (instrument='KEY' AND currency='CAD'))"):
            quotes[ticker][day] = price, adjusted, conid
        trp = db.execute("SELECT ex_date,record_date,payable_date,reported_amount,source_url "
                         "FROM issuer_cash_distribution WHERE instrument='TRP' AND currency='CAD' ORDER BY 2").fetchall()
    fx_days = sorted(rates)
    rows, checks = [], []
    enb = sorted((d for d in enb_distributions() if d[1] <= AS_OF), key=lambda d: d[1])
    quarters = Counter((row[2].year, (row[2].month - 1) // 3 + 1) for row in enb)
    expected = {(year, q) for year in range(2016, 2027) for q in range(1, 5)} - {(2026, 4)}
    if set(quarters) != expected or set(quarters.values()) != {1}:
        raise ValueError(f"Enbridge quarterly history is incomplete or duplicated: {quarters}")
    for declared, record, payable, amount, source in enb:
        if not record <= payable or (declared and declared > record) or amount <= 0:
            raise ValueError(f"Invalid Enbridge declaration for {payable}")
        row, check = fx_reconciliation("ENB", declared, record, payable, amount, source, None,
                                       venues["XNYS"], quotes["ENB"], rates, fx_days)
        rows.append(row)
        checks.append(check)
    for ex, record, payable, amount, source in trp:
        _, check = fx_reconciliation("TRP", None, dt.date.fromisoformat(record), dt.date.fromisoformat(payable),
                                     Decimal(amount), source, ex, venues["XNYS"], quotes["TRP"], rates, fx_days)
        checks.append(check)
    key = sorted(key_distributions(venues["XTSE"], quotes["KEY"]), key=lambda pair: pair[0][2])
    rows.extend(row for row, _ in key)
    sobo = list(sobo_distributions(venues["XNYS"], quotes["SOBO"]))
    rows.extend(row for row, _ in sobo)
    if len({(row[0], str(row[2])) for row in rows}) != len(rows):
        raise ValueError("Duplicate common-share dividend record dates")
    checks, archive_exceptions = dolt_checks(checks)
    write_csv(DERIVED / "canadian_operator_issuer_cash_distributions.csv", FIELDS, rows)
    write_csv(DERIVED / "canadian_operator_cash_fx_comparison.csv", FX_FIELDS, checks)
    write_csv(DERIVED / "canadian_operator_cash_archive_exceptions.csv", ARCHIVE_FIELDS, archive_exceptions)
    write_csv(DERIVED / "keyera_cash_ex_date_evidence.csv", KEY_FIELDS, [check for _, check in key])
    write_csv(DERIVED / "south_bow_cash_ex_date_evidence.csv", SOBO_FIELDS, [check for _, check in sobo])
    refresh_cash(rows, {row[0] for row in rows})
    fx_rows = [dict(zip(FX_FIELDS, row, strict=True)) for row in checks]
    summary = {
        "issuer_rows": dict(Counter(row[0] for row in rows)),
        "unverified_ex_date": {ticker: sum(row[0] == ticker and not row[1] for row in rows)
                               for ticker in sorted({row[0] for row in rows})},
        "fx_reference_rows": {ticker: sum(row["instrument"] == ticker and row["gross_usd_reference_per_share"] is not None
                                         for row in fx_rows) for ticker in FX_POLICIES},
        "fx_reference_within_one_cent_of_ibkr": {
            ticker: sum(row["instrument"] == ticker and row["ibkr_adjustment_minus_usd_reference"] is not None
                        and abs(row["ibkr_adjustment_minus_usd_reference"]) <= 0.01 for row in fx_rows)
            for ticker in FX_POLICIES},
        "reported_cad_number_within_one_cent_of_ibkr": {
            ticker: sum(row["instrument"] == ticker and row["ibkr_adjustment_minus_reported_cad_number"] is not None
                        and abs(row["ibkr_adjustment_minus_reported_cad_number"]) <= 0.01 for row in fx_rows)
            for ticker in FX_POLICIES},
        "prior_fx_observation": dict(Counter(row["instrument"] for row in fx_rows
                                            if row["fx_date_basis"] == "prior_observation_reference_only")),
        "dolt_archive_date_match": dict(Counter(row["instrument"] for row in fx_rows
                                               if row["dolt_archive_amount_as_reported"] is not None)),
        "dolt_archive_date_exceptions": len(archive_exceptions),
        "keyera_history_corrections": sum(row[8] == "issuer_release_corrects_history_table" for row, _ in key),
        "keyera_ibkr_agreement": sum(check[6] is not None and abs(check[6] - float(check[3])) <= 0.01
                                     for _, check in key),
    }
    write(DERIVED / "canadian_operator_cash_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
