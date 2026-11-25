"""Join refiner cash notices, exchange symbols and in-kind consideration to the study data."""
import csv
import datetime as dt
import json
import sqlite3
from collections import Counter
from contextlib import closing
from decimal import Decimal
from zoneinfo import ZoneInfo

from build_additional_us_cash import read_archive
from build_market_event_panel import SYMBOL_HISTORY_SCHEMA
from build_operator_issuer_cash import AS_OF, DB, calendar_ex_date, implied_cash
from collect_dolt_equity_history import REF, STORE
from collect_issuer_distributions import ARCHIVE, DERIVED, FIELDS, replace_cash_records
from collect_public import RAW, ROOT, write, write_csv
from publication_clocks import export

REFERENCE = ROOT / "data" / "reference"
QUOTE_SOURCE = RAW / "equity_reference"
HFC_END = "2022-03-14"
DINO_START = "2022-03-15"
EASTERN = ZoneInfo("America/New_York")
DK_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/DK?period1=1621900800&period2=1625097600&interval=1d"
DINO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/DINO?period1=1647216000&period2=1647648000&interval=1d"
CASH_FIELDS = ("issuer_symbol", "market_instrument", "ex_date", "record_date", "payable_date",
               "regular_cash_usd", "special_cash_usd", "total_cash_usd", "distributed_symbol_if_any",
               "ex_date_basis", "ibkr_adjustment_price_units", "dolt_cash_usd", "dolt_source_file",
               "issuer_source_url", "supplemental_source_url", "alternate_reported_paid_date", "sec_filed_date",
               "special_calendar_source_url", "declared_date", "announcement_date", "declaration_date_basis")
DINO_2023_DECLARATION = "https://www.sec.gov/Archives/edgar/data/1915657/000191565723000015/dino-20221231.htm"


def records(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def chart_rows(path, symbol):
    result, = json.loads(path.read_bytes())["chart"]["result"]
    meta = result["meta"]
    if (meta["symbol"], meta["currency"], meta["exchangeName"], meta["instrumentType"]) != (symbol, "USD", "NYQ", "EQUITY"):
        raise ValueError(f"Unexpected Yahoo listing identity for {symbol}")
    quote, = result["indicators"]["quote"]
    bars = {}
    for stamp, opening, high, low, close, volume in zip(
            result["timestamp"], quote["open"], quote["high"], quote["low"], quote["close"], quote["volume"], strict=True):
        date = dt.datetime.fromtimestamp(stamp, EASTERN).date().isoformat()
        if date in bars or not 0 < low <= min(opening, close) <= max(opening, close) <= high or volume <= 0:
            raise ValueError(f"Invalid {symbol} reference bar on {date}")
        bars[date] = tuple(Decimal(str(v)).quantize(Decimal("0.0001")) for v in (opening, high, low, close)) + (volume,)
    return bars


def quote_identity_checks(quotes):
    path = STORE / "hfc_quote_identity_check.json"
    payload = json.loads(path.read_text())
    if payload["query_execution_status"] != "Success" or f"ohlcv AS OF '{REF}'" not in payload["sql_query"]:
        raise ValueError("HFC comparison archive is unverified")
    hfc = {r["date"]: Decimal(r["close"]) for r in payload["rows"] if r["act_symbol"] == "HFC"}
    if not hfc or max(hfc) != HFC_END:
        raise ValueError("HFC comparison must include the last trading day")
    checks = [("HFC", day, value, path.relative_to(ROOT), None) for day, value in sorted(hfc.items())]
    path = QUOTE_SOURCE / "dino_yahoo_chart_2022-03-14_2022-03-19.json"
    dino = chart_rows(path, "DINO")
    if DINO_START not in dino:
        raise ValueError("The comparison archive lacks the first DINO session")
    checks.extend(("DINO", day, bar[3], path.relative_to(ROOT), DINO_CHART)
                  for day, bar in sorted(dino.items()) if day >= DINO_START)
    for symbol, day, reference_close, file, url in checks:
        ibkr = Decimal(str(quotes[day][0]))
        if abs(ibkr - reference_close) > Decimal("0.005"):
            raise ValueError(f"IBKR history disagrees with {symbol} for {day}")
        yield ("DINO", symbol, day, ibkr, reference_close, file, url)


def cash_rows(sessions, quotes, archived, paths, distribution):
    by_ticker = records(ARCHIVE / "CVI_common_cash_notices_extract.csv")
    notices = [dict(row, issuer_symbol="CVI") for row in by_ticker]
    notices.extend(records(ARCHIVE / "HFC_DINO_common_cash_notices_extract.csv"))
    for row in notices:
        ticker, record, payable, source = (row[name] for name in
                                            ("issuer_symbol", "record_date", "payable_date", "source_url"))
        if record > AS_OF.isoformat():
            continue
        announced = row.get("announcement_date")
        declared = row.get("declared_date") or announced
        declared_basis = "issuer_board_date" if row.get("declared_date") else "issuer_announcement_date"
        supplemental = row.get("supplemental_source_url")
        if ticker == "DINO" and record == "2023-03-07":
            declared, declared_basis = "2023-02-23", "issuer_board_date"
            supplemental = DINO_2023_DECLARATION
        if not declared <= record <= payable or ticker == "HFC" and record > HFC_END or ticker == "DINO" and record < DINO_START:
            raise ValueError(f"Invalid dates for {ticker} on {record}")
        regular, special = (Decimal(row.get(field) or "0") for field in ("regular_cash_usd", "special_cash_usd"))
        amount = regular + special
        explicit = row.get("issuer_ex_date")
        candidate = explicit or calendar_ex_date(sessions, dt.date.fromisoformat(record))
        if amount <= 0 or candidate not in sessions or not sessions.index(candidate):
            raise ValueError(f"No verified entitlement session for {ticker} on {record}")
        previous = sessions[sessions.index(candidate) - 1]
        market = "DINO" if ticker == "HFC" else ticker
        implied = implied_cash(quotes[market], previous, candidate)
        if implied is None or abs(implied - float(amount)) > 0.01:
            raise ValueError(f"IBKR adjustment disagrees for {ticker} on {candidate}: {implied} versus {amount}")
        archive = archived.get((ticker, candidate))
        if archive is not None and abs(archive - amount) > Decimal("0.0005"):
            raise ValueError(f"Dolt dividend disagrees for {ticker} on {candidate}")
        stock = None
        basis = "issuer_record_nyse_rule_ibkr_adjustment_agrees"
        if explicit:
            if (ticker, candidate, record, payable, amount) != (
                    distribution["issuer_symbol"], distribution["ex_date"], distribution["record_date"],
                    distribution["payable_date"], Decimal(distribution["cash_usd_per_issuer_share"])):
                raise ValueError("Cash and non-cash CVI sources disagree")
            stock = distribution["distributed_symbol"]
            basis = "issuer_due_bill_ex_date_cash_component_only"
        elif special:
            basis = "issuer_regular_and_special_cash_same_ex_date_ibkr_adjustment_agrees"
        calendar_source = ("https://www.finra.org/sites/default/files/2021-10/information-notice-101521.pdf"
                           if record == "2022-11-14" else None)
        yield ((ticker, candidate, record, declared, payable, amount, None, "USD", basis, source),
               (ticker, market, candidate, record, payable, regular or None, special or None, amount, stock, basis,
                implied, archive, paths[ticker] if archive is not None else None, source,
                supplemental, row.get("alternate_reported_paid_date"), row.get("sec_filed_date"),
                calendar_source, declared, announced, declared_basis))


def update_database(db, rows, distribution):
    history = [row for row in records(REFERENCE / "market_study_symbol_history.csv") if row["instrument"] == "DINO"]
    merger, = (row for row in records(REFERENCE / "equity_merger_consideration.csv") if row["retiring_instrument"] == "HFC")
    db.executescript(SYMBOL_HISTORY_SCHEMA)
    db.execute("BEGIN IMMEDIATE")
    with db:
        db.execute("DELETE FROM market_study_symbol_history WHERE instrument='DINO'")
        db.executemany("INSERT INTO market_study_symbol_history VALUES (?,?,?,?,?,?,?)",
                       (tuple(value or None for value in row.values()) for row in history))
        db.execute("INSERT OR REPLACE INTO equity_in_kind_distribution VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   tuple(distribution.values()))
        db.execute("INSERT OR REPLACE INTO equity_merger_consideration VALUES (?,?,?,?,?,?,?,?)", tuple(merger.values()))
        replace_cash_records(db, rows, {"CVI", "HFC", "DINO"})
        db.execute("""UPDATE market_study_quote AS q SET
            (issuer_cash_on_ex_date,issuer_cash_post_2020_amlp_basis,issuer_cash_source_url)=(
                SELECT reported_amount,amount_on_post_2020_amlp_share_basis,source_url
                FROM market_study_issuer_cash_distribution d
                WHERE d.instrument=q.instrument AND d.ex_date=q.session_date AND d.currency=q.currency
            ) WHERE q.instrument IN ('CVI','DINO')""")
    export(db, "market_study_quote", order_by="instrument,session_date")
    export(db, "equity_merger_consideration", order_by="closing_date,retiring_instrument")


def main():
    archived, paths = read_archive(("CVI", "HFC", "DINO"))
    distribution, = records(REFERENCE / "equity_in_kind_distributions.csv")
    if not distribution["due_bill_from"] < distribution["record_date"] < distribution["due_bill_through"] == distribution["payable_date"] < distribution["ex_date"]:
        raise ValueError("Invalid CVI due-bill chronology")
    dk_path = QUOTE_SOURCE / "dk_yahoo_chart_2021-05-25_2021-07-01.json"
    dk = chart_rows(dk_path, "DK")
    with closing(sqlite3.connect(DB)) as db:
        sessions = [day for day, in db.execute(
            "SELECT session_date FROM equity_session_calendar WHERE venue='XNYS' ORDER BY 1")]
        if set(dk) != {day for day in sessions if "2021-05-25" <= day <= "2021-06-30"}:
            raise ValueError("Delek reference bars do not cover the specified NYSE window")
        quotes = {ticker: {} for ticker in ("CVI", "DINO")}
        for ticker, day, price, adjusted, conid in db.execute(
                "SELECT instrument,session_date,close,adjusted_ibkr_close,conid FROM market_study_quote "
                "WHERE instrument IN ('CVI','DINO') AND currency='USD' AND price_basis='ibkr_trades_split_adjusted'"):
            quotes[ticker][day] = price, adjusted, conid
        identity = list(quote_identity_checks(quotes["DINO"]))
        rows, checks = zip(*cash_rows(sessions, quotes, archived, paths, distribution), strict=True)
        counts = dict(Counter(row[0] for row in rows))
        if counts != {"CVI": 31, "HFC": 21, "DINO": 18} or len({(row[0], row[1]) for row in rows}) != len(rows):
            raise ValueError(f"Incomplete or duplicated refiner cash ledger: {counts}")
        update_database(db, rows, distribution)
    write_csv(DERIVED / "refiner_issuer_cash_distributions.csv", FIELDS, rows)
    write_csv(DERIVED / "refiner_cash_ex_date_evidence.csv", CASH_FIELDS, checks)
    write_csv(DERIVED / "hfc_dino_quote_identity_checks.csv",
              ("market_instrument", "exchange_symbol", "session_date", "ibkr_close", "reference_close",
               "reference_source_file", "reference_source_url"), identity)
    write_csv(DERIVED / "cvi_delek_distribution_price_reference.csv",
              ("instrument", "session_date", "open", "high", "low", "close", "volume", "currency",
               "source_url", "source_file"),
              (("DK", day, *bar, "USD", DK_CHART, dk_path.relative_to(ROOT)) for day, bar in sorted(dk.items())))
    report = {
        "issuer_cash_rows": counts,
        "archive_cash_date_matches": dict(Counter(row[0] for row in checks if row[11] is not None)),
        "ibkr_adjustment_matches": counts,
        "quote_identity_matches": dict(Counter(row[1] for row in identity)),
        "in_kind_distribution": distribution,
        "delek_reference_sessions": len(dk),
    }
    write(DERIVED / "refiner_issuer_cash_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
