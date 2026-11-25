"""Recover ONEOK's separate pre-merger history and verify the Magellan bars."""
import csv
import datetime as dt
import re
import sqlite3
from decimal import Decimal

from collect_public import RAW, ROOT, get, write_csv
from publication_clocks import table_rows

DERIVED = ROOT / "data" / "derived"
ARCHIVE = RAW / "equity_reference"
ONEOK = "https://finance.yahoo.com/quote/OKE/history/?period1=1451606400&period2=1794355200"
MAGELLAN_DAILY = "https://chartexchange.com/symbol/nyse-mmp/historical/"
MAGELLAN_MONTHLY = "https://www.digrin.com/stocks/detail/MMP/price"
MAGELLAN_CASH = "https://www.digrin.com/stocks/detail/MMP/"
ONEOK_DAILY = "https://chartexchange.com/symbol/nyse-oke/historical/"
FIRST = "2016-01-04"
LAST_MMP = "2023-09-22"
LAST = "2026-11-10"


def money(value):
    return Decimal(value.replace("$", "").replace(",", ""))


def chart_exchange_daily_rows(path, calendar):
    header, *rows = table_rows(path.read_text())
    if header != ["Date(EST)", "Open", "High", "Low", "Close", "Change", "Volume", "ChangeSince"]:
        raise ValueError(f"Unexpected ChartExchange columns in {path.name}")
    accepted, excluded = {}, []
    for day, opening, high, low, close, _, volume, _ in rows:
        if day not in calendar:
            excluded.append(day)
            continue
        if day in accepted:
            raise ValueError(f"Duplicate ChartExchange quote for {day} in {path.name}")
        accepted[day] = (*map(Decimal, (opening, high, low, close)), int(volume.replace(",", "")))
    return accepted, excluded


def yahoo_rows(html):
    prices, cash = {}, {}
    for row in table_rows(html):
        if not row or not re.fullmatch(r"[A-Z][a-z]{2} \d{1,2}, \d{4}", row[0]):
            continue
        date = dt.datetime.strptime(row[0], "%b %d, %Y").date().isoformat()
        if date in prices and len(row) == 7:
            raise ValueError(f"Duplicate Yahoo Finance session {date}")
        if len(row) == 7:
            prices[date] = tuple(money(value) for value in row[1:5]) + (int(row[6].replace(",", "")),)
        elif len(row) == 2 and " Dividend" in row[1]:
            cash[date] = money(row[1].split()[0])
    return prices, cash


def equity_rows(db, instrument, start, end):
    return {date: (op, high, low, close, volume, path) for date, op, high, low, close, volume, path in db.execute("""
        SELECT session_date,open,high,low,close,volume,source_file
        FROM tws_daily_bar WHERE instrument=? AND series='TRADES' AND session_date BETWEEN ? AND ?
        ORDER BY session_date
    """, (instrument, start, end))}


def sessions(db, last):
    return {row[0] for row in db.execute("""
        SELECT session_date FROM equity_session_calendar
        WHERE venue='XNYS' AND session_date BETWEEN ? AND ?
    """, (FIRST, last))}


def main():
    yahoo_path = ARCHIVE / "oke_yahoo_history_2016_2026.html"
    yahoo, yahoo_cash = yahoo_rows(get(ONEOK, yahoo_path).decode())
    daily = list(table_rows(get(MAGELLAN_DAILY, ARCHIVE / "mmp_chartexchange_daily.html").decode()))
    monthly = list(table_rows(get(MAGELLAN_MONTHLY, ARCHIVE / "mmp_digrin_monthly.html").decode()))
    cash = list(table_rows(get(MAGELLAN_CASH, ARCHIVE / "mmp_digrin_cash.html").decode()))
    mmp_cash_dates = {r[0] for r in cash if len(r) == 5 and FIRST <= r[0] <= LAST_MMP
                      and re.fullmatch(r"\d{4}-\d{2}-\d{2}", r[0])}
    with (ROOT / "data" / "reference" / "mmp_issuer_distributions.csv").open() as file:
        expected_cash = {row["ex_date"] for row in csv.DictReader(file)}
    if mmp_cash_dates != expected_cash:
        raise ValueError(f"Magellan cash dates disagree: {sorted(mmp_cash_dates ^ expected_cash)}")

    with sqlite3.connect(f"file:{DERIVED / 'cushing.sqlite'}?mode=ro", uri=True) as db:
        prefix = equity_rows(db, "OKE", FIRST, LAST_MMP)
        suffix = equity_rows(db, "OKE", "2023-09-25", LAST)
        expected_dates = sessions(db, LAST)
        if set(yahoo) != expected_dates or set(prefix) != sessions(db, LAST_MMP):
            raise ValueError("Recovered series do not cover the expected NYSE calendar")
        if set(suffix) != expected_dates - set(prefix):
            raise ValueError("Post-merger ONEOK bars do not cover the expected calendar")
        issuer = {d: Decimal(a) for d, a in db.execute("""
            SELECT ex_date,reported_amount FROM issuer_cash_distribution WHERE instrument='OKE' AND ex_date<=?
        """, (LAST,))}
        if yahoo_cash != issuer:
            raise ValueError("ONEOK Yahoo Finance cash events disagree with issuer disclosures")
        anchors = db.execute("""
            SELECT session_date, requested_security_close, other_security_close
            FROM market_identity_check WHERE requested_instrument='OKE'
        """)
        for date, oneok, other in anchors:
            if float(yahoo[date][3]) != oneok or (other is not None and prefix[date][3] != other):
                raise ValueError(f"Price anchor disagrees with merger disclosure on {date}")

    comparisons = []
    for date, bar in suffix.items():
        comparisons.append(("OKE", date, "yahoo_finance_vs_tws_after_merger", bar[3], yahoo[date][3], ONEOK))
    month_end = {date[:7]: date for date in prefix}
    for row in monthly:
        if len(row) == 3 and re.fullmatch(r"[A-Za-z]+ 20\d{2}", row[0]):
            month = dt.datetime.strptime(row[0], "%B %Y").strftime("%Y-%m")
            if (date := month_end.get(month)) is not None:
                comparisons.append(("MMP", date, "digrin_month_end_vs_tws_prefix", prefix[date][3], money(row[2]),
                                    MAGELLAN_MONTHLY))
    for row in daily:
        if len(row) >= 8 and row[0] in prefix:
            comparisons.append(("MMP", row[0], "chartexchange_daily_vs_tws_prefix", prefix[row[0]][3], money(row[4]),
                                MAGELLAN_DAILY))
    checked_months = {r[1][:7] for r in comparisons if r[2] == "digrin_month_end_vs_tws_prefix"}
    if checked_months != set(month_end):
        raise ValueError("The monthly price check is missing part of Magellan's history")
    differences = [r for r in comparisons if abs(float(r[3]) - float(r[4])) > 0.011]
    if any(r[0] != "OKE" for r in differences):
        raise ValueError(f"Magellan price checks disagree: {differences[:5]}")

    recovered = [(symbol, date, *price, "USD", provider, source_symbol, path, source_url)
                 for date in prefix
                 for symbol, price, provider, source_symbol, path, source_url in (
                     ("OKE", yahoo[date], "yahoo_finance", "OKE", str(yahoo_path.relative_to(ROOT)), ONEOK),
                     ("MMP", prefix[date][:5], "ibkr_tws_history_reidentified", "OKE", prefix[date][5], MAGELLAN_DAILY),
                 )]
    if differences:
        recent = table_rows(get(ONEOK_DAILY, ARCHIVE / "oke_chartexchange_daily.html").decode())
        independent = {r[0]: money(r[4]) for r in recent if len(r) >= 8 and r[0] in suffix}
        for _, date, _, original, price, _ in differences:
            if independent.get(date) != price:
                raise ValueError(f"Unresolved ONEOK closing price disagreement on {date}")
            comparisons.append(("OKE", date, "chartexchange_confirmation_of_yahoo_close", original, price, ONEOK_DAILY))
            recovered.append(("OKE", date, *yahoo[date], "USD", "yahoo_finance", "OKE",
                              str(yahoo_path.relative_to(ROOT)), ONEOK_DAILY))
    write_csv(DERIVED / "equity_reference_bar.csv", ("instrument", "session_date", "open", "high", "low", "close",
               "volume", "currency", "provider", "requested_symbol", "source_file", "verification_source_url"),
               sorted(recovered))
    write_csv(DERIVED / "equity_reference_price_checks.csv", ("instrument", "session_date", "comparison", "tws_close",
               "independent_close", "independent_source_url"), sorted(comparisons))
    print(f"{len(recovered)} recovered price bars and {len(comparisons)} independent price checks")


if __name__ == "__main__":
    main()
