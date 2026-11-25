"""Archive fund distributions and reconcile them on the IBKR quote share basis."""
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal

from openpyxl import load_workbook

from build_market_event_panel import CONTROL_CASH_SCHEMA, add_market_study_quotes
from build_operator_issuer_cash import AS_OF, FIRST, implied_cash
from collect_issuer_distributions import ARCHIVE, DERIVED, FIELDS, replace_cash_records, source_date
from collect_public import ROOT, get, write_csv
from publication_clocks import export

SPDR = ("SPY", "XLE", "XOP")
WORKBOOK = "spdr-etf-historical-distributions.xlsx"
WORKBOOK_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/" + WORKBOOK
SPECIAL_XLE = "https://www.miaxglobal.com/sites/default/files/alert-files/XLE_Dividend_46261.pdf"
SPLITS = {
    "AMLP": ("2020-05-18", 1, 5, "00162Q866", "00162Q452", "AMLP_8937_20200518.pdf",
             "https://www.alpsfunds.com/hubfs/alps-docs/tax/8937/alerian-mlp-etf-amlp-8937-20200518.pdf"),
    "USO": ("2020-04-29", 1, 8, "91232N108", "91232N207", "USO_2025_10K.pdf",
            "https://www.sec.gov/Archives/edgar/data/1327068/000110465926021501/uso-20251231x10k.htm"),
    "XLE": ("2025-12-05", 2, 1, "81369Y506", "81369Y506", "XLE_2025_split_8937.pdf",
            "https://www.ssga.com/library-content/products/fund-docs/etfs/us/tax-documents/"
            "us-tax-form-state-street-energy-select-sector-spdr-etf-2025.pdf"),
    "XOP": ("2020-03-30", 1, 4, "78464A730", "78468R556", "XOP_2020_split_8937.pdf",
            "https://www.ssga.com/library-content/products/fund-docs/etfs/us/tax-documents/"
            "J942-SPDR-SP-Oil-Gas-Exploration-and-Production-ETF-2020-Form-8937.pdf"),
}
SEC = "https://www.sec.gov/Archives/edgar/data/"
ZERO_DISCLOSURES = {
    "BNO": (SEC + "1472494/000110465926021521/bno-20251231x10k.htm",
            SEC + "1472494/000110465926123993/bno-20260930x10q.htm"),
    "USO": (SEC + "1327068/000110465926021501/uso-20251231x10k.htm",
            SEC + "1327068/000110465926123978/uso-20260930x10q.htm"),
}
COVERAGE_FIELDS = ("instrument", "issuer_confirmed_from", "issuer_confirmed_through",
                   "issuer_distributions", "annual_source_url", "interim_source_url",
                   "annual_source_file", "interim_source_file",
                   "first_paired_broker_session", "last_paired_broker_session", "paired_broker_sessions",
                   "max_close_difference")
BASIS_FIELDS = ("instrument", "ex_date", "amount_on_quote_share_basis")
SPLIT_FIELDS = ("instrument", "first_split_adjusted_session", "new_shares", "old_shares",
                "prior_cusip", "current_cusip", "source_file", "source_url")
EVIDENCE_FIELDS = ("instrument", "ex_date", "record_date", "payable_date", "workbook_row", "workbook_cusip",
                   "ex_date_cusip", "issuer_dividend", "issuer_short_term_capital_gain",
                   "issuer_long_term_capital_gain", "reported_amount", "quote_share_basis_multiplier",
                   "amount_on_quote_share_basis", "previous_broker_session", "ibkr_implied_cash",
                   "absolute_difference_usd", "workbook_source_url", "event_confirmation_url")
HEADER = ("FUND NAME", "TICKER", "CUSIP", "EX-DATE", "RECORD DATE", "PAYABLE DATE", "DIVIDEND ($)",
          "SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)", "FREQUENCY")
TOLERANCE = Decimal("0.01")


def issuer_events():
    path = ARCHIVE / WORKBOOK
    get(WORKBOOK_URL, path)
    with closing(load_workbook(path, read_only=True, data_only=True)) as workbook:
        sheet = workbook["dividend"].values
        if next(sheet) != HEADER:
            raise ValueError("State Street distribution workbook columns have changed")
        for index, row in enumerate(sheet, start=2):
            _, ticker, cusip, ex, record, payable, dividend, short_gain, long_gain, _ = row
            if ticker not in SPDR:
                continue
            ex, record, payable = (source_date(value, "%m/%d/%Y") for value in (ex, record, payable))
            if not FIRST <= ex <= AS_OF:
                continue
            parts = tuple(Decimal(str(value or 0)) for value in (dividend, short_gain, long_gain))
            if not ex <= record <= payable or min(parts) < 0 or sum(parts) <= 0:
                raise ValueError(f"Invalid issuer dates or cash components for {ticker} on {ex}")
            yield ticker, ex.isoformat(), record.isoformat(), payable.isoformat(), index, cusip, parts


def reconcile(events, sessions, quotes):
    cash, basis, evidence = [], [], []
    prior = dict(zip(sessions[1:], sessions[:-1], strict=True))
    for ticker, ex, record, payable, index, cusip, parts in events:
        effective, new, old, before, after, *_ = SPLITS.get(ticker, ("", 1, 1, cusip, cusip))
        if cusip != after:
            raise ValueError(f"Unexpected State Street CUSIP {cusip} for {ticker}")
        earlier = prior[ex]
        factor = Decimal(old) / new if ex < effective else Decimal(1)
        amount = sum(parts)
        normalized = amount * factor
        check = implied_cash(quotes[ticker], earlier, ex)
        difference = abs(Decimal(str(check)) - normalized) if check is not None else None
        if difference is None or difference > TOLERANCE:
            raise ValueError(f"Cash adjustment disagrees for {ticker} on {ex}: {normalized} vs {check}")
        special = SPECIAL_XLE if (ticker, ex) == ("XLE", "2019-12-30") else None
        cash.append((ticker, ex, record, None, payable, amount, None, "USD",
                     "issuer_fund_special_cash_distribution" if special else "issuer_fund_cash_history",
                     WORKBOOK_URL))
        basis.append((ticker, ex, normalized))
        evidence.append((ticker, ex, record, payable, index, cusip, before if ex < effective else after,
                         *parts, amount, factor, normalized, earlier, check, difference, WORKBOOK_URL, special))

    keys = {(row[0], row[1]) for row in basis}
    if len(keys) != len(basis) or {ticker for ticker, _ in keys} != set(SPDR):
        raise ValueError("Issuer history has duplicates or an absent fund")
    for ticker in SPDR:
        for day, earlier in prior.items():
            step = implied_cash(quotes[ticker], earlier, day)
            if step is not None and abs(step) > float(TOLERANCE) and (ticker, day) not in keys:
                raise ValueError(f"IBKR adjustment without issuer cash event for {ticker} on {day}")
    return cash, basis, evidence


def zero_coverage(quotes):
    rows = []
    for ticker, urls in ZERO_DISCLOSURES.items():
        pairs = quotes[ticker]
        difference = max(abs(trades - adjusted) for trades, adjusted, _ in pairs.values())
        if difference > 1e-8:
            raise ValueError(f"Unexpected difference between price and adjusted close for {ticker}")
        annual, interim = ((ARCHIVE / f"{ticker}_{period}.pdf").relative_to(ROOT)
                           for period in ("2025_10K", "2026Q3_10Q"))
        rows.append((ticker, FIRST.isoformat(), "2026-09-30", 0, *urls, str(annual), str(interim),
                     min(pairs), max(pairs), len(pairs), difference))
    return rows


def refresh(db, cash, basis, splits, coverage):
    db.executescript(CONTROL_CASH_SCHEMA)
    db.execute("BEGIN IMMEDIATE")
    with db:
        replace_cash_records(db, cash, SPDR)
        db.executemany("DELETE FROM equity_cash_quote_basis WHERE instrument=?", ((ticker,) for ticker in SPDR))
        db.executemany("INSERT INTO equity_cash_quote_basis VALUES (?,?,?)",
                       ((ticker, ex, str(amount)) for ticker, ex, amount in basis))
        db.executemany("INSERT OR REPLACE INTO equity_share_split_event VALUES (?,?,?,?,?,?,?,?)", splits)
        db.executemany("INSERT OR REPLACE INTO oil_fund_cash_distribution_coverage VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       coverage)
        for ticker in SPDR:
            db.execute("DELETE FROM market_study_quote WHERE instrument=?", (ticker,))
            add_market_study_quotes(db, ticker)


def main():
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        sessions = [day for day, in db.execute("""SELECT session_date FROM equity_session_calendar
            WHERE venue='XNYS' AND session_date BETWEEN ? AND ? ORDER BY session_date""",
                                              (FIRST.isoformat(), AS_OF.isoformat()))]
        quotes = {ticker: {} for ticker in (*SPDR, *ZERO_DISCLOSURES)}
        for ticker, day, price, adjusted, conid in db.execute("""SELECT instrument,session_date,close,
                adjusted_ibkr_close,conid FROM market_study_quote
            WHERE instrument IN ('SPY','XLE','XOP','USO','BNO')
                AND price_basis='ibkr_trades_split_adjusted' AND adjusted_ibkr_close>0"""):
            quotes[ticker][day] = price, adjusted, conid
        cash, basis, evidence = reconcile(sorted(issuer_events()), sessions, quotes)
        coverage = zero_coverage(quotes)
        splits = [(ticker, day, new, old, before, after, str((ARCHIVE / filename).relative_to(ROOT)), url)
                  for ticker, (day, new, old, before, after, filename, url) in SPLITS.items()]
        refresh(db, cash, basis, splits, coverage)
        export(db, "market_study_quote", "instrument,session_date")
        export(db, "eia_equity_horizon_input", "release_date,instrument")
    write_csv(DERIVED / "spdr_issuer_cash_distributions.csv", FIELDS, cash)
    write_csv(DERIVED / "spdr_cash_quote_basis.csv", BASIS_FIELDS, basis)
    write_csv(DERIVED / "spdr_cash_ex_date_evidence.csv", EVIDENCE_FIELDS, evidence)
    write_csv(DERIVED / "oil_fund_cash_distribution_coverage.csv", COVERAGE_FIELDS, coverage)
    write_csv(ROOT / "data" / "reference" / "market_control_share_splits.csv", SPLIT_FIELDS, splits)
    print(dict(Counter(ticker for ticker, *_ in cash)), flush=True)


if __name__ == "__main__":
    main()
