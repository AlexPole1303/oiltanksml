"""Archive the November 2017 adjusted snapshot for historical US listings."""
import csv
import datetime as dt
import hashlib
import io
import math
import re
from decimal import Decimal

from collect_public import RAW, ROOT, get
from publication_clocks import table_rows

ARCHIVE = RAW / "equity_reference" / "kaggle_2017"
DATASET = "https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/Stocks%2F"
SNAPSHOT_END = "2017-11-10"
SYMBOLS = ("AMID", "ETE", "ETP", "PSXP", "SEMG", "TEGP", "TEP", "TLP")
AMID_MONTHLY_URL = "https://www.digrin.com/stocks/detail/AMID/price"
AMID_MONTHLY_PATH = RAW / "equity_reference" / "amid_digrin_monthly.html"
TICKERTECH_PATH = RAW / "equity_reference" / "tickertech" / "JPEP"
TICKERTECH_HISTORY = "https://www.historicalstockprice.com/history/?a=historical&ticker=JPEP"

SCHEMA = """
CREATE TABLE historical_equity_adjusted_archive (
    listing_id TEXT, symbol TEXT, session_date TEXT, adjusted_open REAL, adjusted_high REAL,
    adjusted_low REAL, adjusted_close REAL, reported_volume REAL, source_snapshot_end TEXT,
    source_url TEXT, source_file TEXT, source_sha256 TEXT, PRIMARY KEY (listing_id, session_date));
CREATE TABLE historical_equity_price_anchor (
    listing_id TEXT, symbol TEXT, session_date TEXT, reported_price_usd REAL, price_basis TEXT,
    source_url TEXT, PRIMARY KEY (listing_id, session_date, price_basis));
CREATE TABLE historical_equity_monthly_reference (
    listing_id TEXT, symbol TEXT, month TEXT, provider_real_price REAL, provider_adjusted_price REAL,
    matched_dolt_session TEXT, matched_dolt_close REAL, source_url TEXT, source_file TEXT,
    PRIMARY KEY (listing_id, month));
CREATE TABLE historical_equity_daily_reference (
    listing_id TEXT, symbol TEXT, session_date TEXT, open REAL, high REAL, low REAL,
    close REAL, volume INTEGER, source_url TEXT, source_file TEXT, source_sha256 TEXT,
    PRIMARY KEY (listing_id, session_date));
"""

HISTORICAL_SOURCE_COVERAGE_VIEW = """
CREATE VIEW historical_equity_source_coverage AS
    SELECT l.listing_id, l.symbol, c.session_date,
        CASE WHEN d.quality_status='ohlcv_range_valid' THEN 1 ELSE 0 END AS valid_dolt_bar,
        a.adjusted_close, r.close AS tickertech_close,
        p.reported_price_usd AS issuer_reported_price, p.price_basis AS issuer_price_basis,
        b.close AS reference_close,b.provider AS reference_provider
    FROM historical_equity_listing_window l
    JOIN equity_session_calendar c ON c.venue='XNYS'
        AND c.session_date BETWEEN l.first_study_session AND l.last_study_session
    LEFT JOIN dolt_historical_equity_quality d ON d.listing_id=l.listing_id AND d.session_date=c.session_date
    LEFT JOIN historical_equity_adjusted_archive a ON a.listing_id=l.listing_id AND a.session_date=c.session_date
    LEFT JOIN historical_equity_daily_reference r ON r.listing_id=l.listing_id AND r.session_date=c.session_date
    LEFT JOIN historical_equity_price_anchor p ON p.listing_id=l.listing_id AND p.session_date=c.session_date
    LEFT JOIN equity_reference_bar b ON b.instrument=l.symbol AND b.session_date=c.session_date
        AND b.volume>0 AND b.low>0 AND b.low<=min(b.open,b.close) AND b.high>=max(b.open,b.close);
"""

SCHEMA += HISTORICAL_SOURCE_COVERAGE_VIEW + """
CREATE VIEW historical_distribution_ex_date_estimate AS
    WITH calendar AS (
        SELECT session_date,lag(session_date,1) OVER (ORDER BY session_date) AS previous_session,
            lag(session_date,2) OVER (ORDER BY session_date) AS second_previous_session
        FROM equity_session_calendar WHERE venue='XNYS'
    )
    SELECT d.instrument,d.record_date,d.declared_date,d.payable_date,d.reported_amount,d.currency,
        CASE WHEN d.record_date<'2017-09-05' THEN c.second_previous_session
             WHEN d.record_date>='2017-09-07' THEN c.previous_session END AS estimated_ex_date,
        CASE WHEN c.session_date IS NULL THEN 'record_date_outside_session_calendar'
             WHEN d.record_date<'2017-09-05' THEN 'nyse_t3_estimate'
             WHEN d.record_date>='2017-09-07' THEN 'nyse_t2_estimate'
             ELSE 'transition_date_requires_exchange_confirmation' END AS estimate_basis,
        d.source_url AS distribution_source_url,
        'https://www.nyse.com/publicdocs/nyse/regulation/nyse/Changes_Related_to_the_Shortened_Settlement_Cycle_T%2B2.pdf' AS rule_source_url
    FROM issuer_cash_distribution d LEFT JOIN calendar c ON c.session_date=d.record_date
    WHERE d.instrument IN ('JPEP','AMID') AND d.ex_date IS NULL AND d.source_basis='issuer_sec_common_units';
"""


def file_for(symbol):
    return ARCHIVE / f"{symbol.lower()}.us.txt"


def download_url(symbol):
    return DOWNLOAD + file_for(symbol).name


def read_rows(payload):
    reader = csv.DictReader(io.StringIO(payload.decode()))
    if reader.fieldnames != ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]:
        raise ValueError("Unexpected Kaggle OHLCV schema")
    rows = list(reader)
    dates = [row["Date"] for row in rows]
    if not dates or dates != sorted(set(dates)) or dates[-1] != SNAPSHOT_END:
        raise ValueError("Unexpected daily history in Kaggle 2017 snapshot")
    return rows


def add_adjusted_equity(db, symbols=SYMBOLS):
    for symbol in symbols:
        path = file_for(symbol)
        if not path.is_file():
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        rows = read_rows(raw)
        for identity, first, last in db.execute("""SELECT listing_id, first_study_session, last_study_session
                FROM historical_equity_listing_window WHERE symbol=?""", (symbol,)):
            for row in rows:
                day = row["Date"]
                if not first <= day <= last:
                    continue
                opening, high, low, close, volume = (float(row[k]) for k in ("Open", "High", "Low", "Close", "Volume"))
                if not all(math.isfinite(x) for x in (opening, high, low, close, volume)) or volume < 0:
                    raise ValueError(f"Invalid Kaggle numeric values for {symbol} {day}")
                if not 0 < low <= min(opening, close) <= max(opening, close) <= high:
                    raise ValueError(f"Invalid Kaggle OHLC range for {symbol} {day}")
                db.execute("INSERT INTO historical_equity_adjusted_archive VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (identity, symbol, day, opening, high, low, close, volume, SNAPSHOT_END,
                            download_url(symbol), str(path.relative_to(ROOT)), digest))


def add_amid_monthly_reference(db):
    if not AMID_MONTHLY_PATH.is_file():
        return
    identity = "american_midstream"
    has_dolt_history = db.execute("SELECT 1 FROM dolt_historical_equity_bar WHERE listing_id=? LIMIT 1",
                                  (identity,)).fetchone()
    if not has_dolt_history:
        return
    first, last = db.execute("""SELECT first_study_session, last_study_session
        FROM historical_equity_listing_window WHERE listing_id=?""", (identity,)).fetchone()
    matches = 0
    for row in table_rows(AMID_MONTHLY_PATH.read_text()):
        if len(row) != 3 or not all(price.startswith("$") for price in row[1:]):
            continue
        try:
            month = dt.datetime.strptime(row[0], "%B %Y").date().isoformat()[:7]
        except ValueError:
            continue
        if not first[:7] <= month <= last[:7]:
            continue
        adjusted, real = (float(Decimal(price[1:].replace(",", ""))) for price in row[1:])
        check = db.execute("""SELECT session_date, close FROM dolt_historical_equity_bar
            WHERE listing_id=? AND substr(session_date, 1, 7)=?
            ORDER BY session_date DESC LIMIT 1""", (identity, month)).fetchone()
        if check and abs(check[1] - real) > 0.005:
            raise ValueError(f"AMID price mismatch in Digrin month {month}: {check[1]} versus {real}")
        matches += int(check is not None)
        db.execute("INSERT INTO historical_equity_monthly_reference VALUES (?,?,?,?,?,?,?,?,?)",
                   (identity, "AMID", month, real, adjusted, *(check or (None, None)), AMID_MONTHLY_URL,
                    str(AMID_MONTHLY_PATH.relative_to(ROOT))))
    if not matches:
        raise ValueError("AMID monthly history could not be checked against the historical listing")


def tickertech_quote(payload, day, symbol="JPEP"):
    text = payload.decode()
    display = dt.date.fromisoformat(day).strftime("%m/%d/%Y")
    marker = f"heading('Historical quote for {symbol} on {display}')"
    if marker not in text:
        return None
    section = text.split(marker, 1)[1].split("<form", 1)[0]
    values = re.findall(r'<font color="#222222">([^<]+)</font></td>', section)
    if len(values) != 6 or values[0] != display:
        raise ValueError(f"Unexpected {symbol} quote format for {day}")
    opening, high, low, close, volume = (Decimal(value.replace(",", "")) for value in values[1:])
    if not 0 < low <= min(opening, close) <= max(opening, close) <= high or volume <= 0:
        raise ValueError(f"Invalid or nontraded {symbol} bar for {day}")
    return float(opening), float(high), float(low), float(close), int(volume)


def tickertech_date_query(day):
    date = dt.date.fromisoformat(day)
    return date.strftime("&month=%m&day=%d&year=%Y")


def nyse_ex_date_from_record(sessions, record):
    if "2017-09-05" <= record <= "2017-09-06":
        raise ValueError(f"Ex-date for {record} requires the transition schedule")
    offset = 2 if record < "2017-09-05" else 1
    index = sessions.index(record)
    if index < offset:
        raise ValueError(f"Ex-date for {record} falls outside the source calendar")
    return sessions[index-offset], "nyse_t3" if offset == 2 else "nyse_t2"


def add_tickertech_jpep(db):
    for path in sorted(TICKERTECH_PATH.glob("*.js")):
        day = path.stem
        if not "2016-01-04" <= day <= "2017-03-07":
            continue
        raw = path.read_bytes()
        quote = tickertech_quote(raw, day)
        if quote is None:
            continue
        evidence = db.execute("""SELECT reported_price_usd FROM historical_equity_price_anchor
            WHERE listing_id='jp_energy' AND session_date=? AND price_basis='reported_close'""", (day,)).fetchone()
        if evidence and abs(quote[3] - evidence[0]) > 0.005:
            raise ValueError(f"JPEP close mismatch against issuer price on {day}")
        db.execute("INSERT INTO historical_equity_daily_reference VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   ("jp_energy", "JPEP", day, *quote, TICKERTECH_HISTORY + tickertech_date_query(day),
                    str(path.relative_to(ROOT)), hashlib.sha256(raw).hexdigest()))


if __name__ == "__main__":
    for symbol in SYMBOLS:
        path = file_for(symbol)
        try:
            rows = read_rows(get(download_url(symbol), path))
        except ValueError:
            path.unlink(missing_ok=True)
            raise
        print(f"{symbol}: {len(rows)} original rows through {SNAPSHOT_END}")
    get(AMID_MONTHLY_URL, AMID_MONTHLY_PATH)
