"""Fill the dated SemGroup listing from corroborated public market history."""
import csv
import datetime as dt
import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal

from build_market_event_panel import add_market_study_quotes
from collect_dolt_equity_history import REF, STORE
from collect_early_equity_history import TICKERTECH_HISTORY, TICKERTECH_PATH, nyse_ex_date_from_record, tickertech_date_query, tickertech_quote
from collect_equity_reference import chart_exchange_daily_rows, sessions
from collect_public import RAW, ROOT, write, write_csv
from collect_quandl_wiki_history import DATASET as WIKI_URL, STORE as WIKI
from publication_clocks import export, table_rows

DERIVED = ROOT / "data/derived"
DOCUMENTS = RAW / "documents"
LAST = "2019-12-04"
CHART_FILE = RAW / "equity_reference/semg_chartexchange_daily.html"
CHART_URL = "https://chartexchange.com/symbol/nyse-semg/historical/"
EDGAR = "https://www.sec.gov/Archives/edgar/data/1489136/"
ISSUER_RELEASE = EDGAR + "000148913619000036/exh991q32019earningspr.htm"
STATEMENTS = (
    ("semg_2018_10k.html", EDGAR + "000148913619000011/semg1231201810-k.htm"),
    ("semg_2019-11-07_10-q.html", EDGAR + "000148913619000038/semg9301910-q.htm"),
)
FIELDS = ("open", "high", "low", "close", "volume")


def relative(path):
    return str(path.relative_to(ROOT))


def sec_date(value):
    return dt.datetime.strptime(value, "%B %d, %Y").date().isoformat()


def semg_distributions(calendar):
    path = STORE / "semg_dividends.json"
    archive = json.loads(path.read_text())
    if archive.get("query_execution_status") != "Success" or REF not in archive["sql_query"]:
        raise ValueError("Unverified Dolt result for SEMG distributions")
    ex_dates = {row["ex_date"]: Decimal(row["amount"]) for row in archive["rows"] if row["act_symbol"] == "SEMG"}
    ordered = sorted(calendar)
    cash = {}
    for filename, url in STATEMENTS:
        source = DOCUMENTS / filename
        for row in table_rows(source.read_text()):
            if len(row) != 9 or row[2] != "$" or row[3] not in {"0.45", "0.4725"}:
                continue
            try:
                printed, payable = sec_date(row[6]), sec_date(row[8])
            except ValueError:
                continue
            if printed not in calendar:
                continue
            amount = Decimal(row[3])
            record, declaration, correction = printed, None, None
            if printed == "2019-11-04" and payable == "2019-11-21":
                release = DOCUMENTS / "semg_2019_11_07_earnings_release.html"
                text = release.read_text()
                if not all(piece in text for piece in ("0.4725", "November 21, 2019", "November 14, 2019")):
                    raise ValueError("SEMG issuer release does not support the corrected November record date")
                record, declaration, correction = "2019-11-14", printed, relative(release)
            ex_date, basis = nyse_ex_date_from_record(ordered, record)
            archive_amount = ex_dates.get(ex_date)
            if archive_amount is None or abs(archive_amount-amount) > Decimal("0.0005"):
                raise ValueError(f"SEMG cash event mismatch on {ex_date}")
            cash[ex_date] = dict(instrument="SEMG", ex_date=ex_date, record_date=record, payable_date=payable,
                                 declared_date=declaration, amount_usd=str(amount), archive_amount_usd=str(archive_amount),
                                 filing_record_date=printed, issuer_source_url=ISSUER_RELEASE if correction else url,
                                 filing_source_file=relative(source), correction_source_file=correction,
                                 ex_date_source_file=relative(path), expected_ex_date_basis=basis)
    if len(cash) != 16 or cash.keys() != ex_dates.keys():
        raise ValueError("The SEC schedule and Dolt do not cover the same SEMG ex-date set")
    return list(cash.values())


def add_semg_history(db):
    calendar = sessions(db, LAST)
    archive_path = WIKI / "SEMG.csv"
    wiki_manifest = json.loads((WIKI / "manifest.json").read_text())["symbols"]["SEMG"]
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != wiki_manifest["sha256"]:
        raise ValueError("SEMG source extract does not match the archived WIKI manifest")
    with archive_path.open(newline="") as source:
        raw = {row["date"]: row for row in csv.DictReader(source) if row["ticker"] == "SEMG" and row["date"] in calendar}
    wiki = {day: tuple(Decimal(row[field]) for field in FIELDS) for day, row in raw.items()}
    dolt = {row[0]: tuple(Decimal(str(value)) for value in row[1:]) for row in db.execute("""SELECT session_date,open,high,low,close,volume
        FROM dolt_historical_equity_quality WHERE listing_id='semgroup' AND quality_status='ohlcv_range_valid' AND volume>0""")}
    chart, excluded = chart_exchange_daily_rows(CHART_FILE, calendar)
    overlaps = sorted(wiki.keys() & dolt.keys())
    if len(overlaps) < 50 or any(wiki[day][3] != dolt[day][3] for day in overlaps):
        raise ValueError("WIKI and Dolt SEMG closes failed the listing identity cross-check")
    chart_overlap = sorted(chart.keys() & dolt.keys())
    if len(chart_overlap) < 90 or any(abs(chart[day][3]-dolt[day][3]) > Decimal("0.005") for day in chart_overlap):
        raise ValueError("ChartExchange and Dolt SEMG closes failed the listing identity cross-check")
    if dolt["2019-09-13"][3] != Decimal("10.28"):
        raise ValueError("SEMG close disagrees with the merger announcement anchor")
    sources = {day: (bar, "quandl_wiki_semg", archive_path, WIKI_URL) for day, bar in wiki.items()}
    sources.update((day, (bar, "chartexchange_semg", CHART_FILE, CHART_URL)) for day, bar in chart.items())
    for day in sorted(calendar - (sources.keys() | dolt.keys())):
        path = TICKERTECH_PATH.parent / "SEMG" / f"{day}.js"
        values = tickertech_quote(path.read_bytes(), day, "SEMG")
        if values is None:
            raise ValueError(f"Missing verified SEMG bar on {day}")
        url = TICKERTECH_HISTORY.replace("ticker=JPEP", "ticker=SEMG") + tickertech_date_query(day)
        sources[day] = (tuple(Decimal(str(value)) for value in values), "tickertech_semg", path, url)
    recovered = {day: row for day, row in sources.items() if day not in dolt}
    for day, (bar, _, _, _) in recovered.items():
        opening, high, low, close, volume = bar
        if not 0 < low <= min(opening, close) <= max(opening, close) <= high or volume <= 0:
            raise ValueError(f"Invalid observed SEMG OHLCV for {day}")
    cash = semg_distributions(calendar)
    db.execute("DELETE FROM equity_reference_bar WHERE instrument='SEMG' AND provider IN ('quandl_wiki_semg','chartexchange_semg','tickertech_semg')")
    db.execute("DELETE FROM issuer_cash_distribution WHERE instrument='SEMG' AND source_basis='issuer_sec_common_shares_ex_date_dolt'")
    db.executemany("INSERT INTO equity_reference_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                   (("SEMG", day, *map(float, bar), "USD", provider, "SEMG", relative(path), url)
                    for day, (bar, provider, path, url) in sorted(recovered.items())))
    db.executemany("INSERT INTO issuer_cash_distribution VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (("SEMG", row["ex_date"], row["record_date"], row["declared_date"], row["payable_date"], row["amount_usd"],
                     None, "USD", "issuer_sec_common_shares_ex_date_dolt", row["issuer_source_url"]) for row in cash))
    compared = [(day, provider, *quotes[day], *dolt[day]) for quotes, dates, provider in
                ((wiki, overlaps, "quandl_wiki"), (chart, chart_overlap, "chartexchange")) for day in dates]
    return calendar, recovered, compared, excluded, cash


def main():
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        db.execute("BEGIN IMMEDIATE")
        with db:
            calendar, recovered, compared, excluded, cash = add_semg_history(db)
            with (ROOT / "data/reference/equity_merger_consideration.csv").open(newline="") as source:
                merger, = [tuple(row.values()) for row in csv.DictReader(source) if row["retiring_instrument"] == "SEMG"]
            db.execute("INSERT OR REPLACE INTO equity_merger_consideration VALUES (?,?,?,?,?,?,?,?)", merger)
            db.execute("DELETE FROM market_study_quote WHERE instrument='SEMG'")
            add_market_study_quotes(db, "SEMG")
            rows = db.execute("SELECT * FROM market_study_quote WHERE instrument='SEMG' ORDER BY session_date")
            fields = [column[0] for column in rows.description]
            prices = rows.fetchall()
            dates = {row[1] for row in prices}
            if dates != calendar:
                raise ValueError(f"SEMG study panel coverage differs from the NYSE calendar: {sorted(dates ^ calendar)}")
        for name in ("market_study_quote", "eia_equity_horizon_input", "historical_equity_source_coverage"):
            export(db, name)
    write_csv(DERIVED / "semg_market_history.csv", fields, prices)
    write_csv(DERIVED / "semg_cash_distribution_evidence.csv", cash[0].keys(), (row.values() for row in cash))
    write_csv(DERIVED / "semg_price_crosscheck.csv", ("session_date", "provider", *FIELDS, *(f"dolt_{f}" for f in FIELDS)), compared)
    report = dict(eligible_sessions=len(calendar), observed_sessions=len(prices), recovered_by_provider=dict(Counter(
                  row[1] for row in recovered.values())), overlap_by_provider=dict(Counter(row[1] for row in compared)),
                  excluded_post_listing_chart_rows=len(excluded), confirmed_cash_events=len(cash),
                  wiki_extra_cash_date="2016-01-29", wiki_extra_cash_usd=0.66,
                  wiki_extra_cash_status="absent_from_issuer_common_stock_distributions_and_dolt")
    write(DERIVED / "semg_market_history_summary.json", json.dumps(report, indent=2).encode() + b"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
