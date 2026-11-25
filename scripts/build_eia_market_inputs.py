"""Normalize published EIA controls and bind market observations to release cutoffs."""
import datetime as dt
import re
from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class Metric:
    key: str
    unit: str
    table: int
    label: str
    row: int | None = None
    group: str | None = None
    pdf_line: int | None = None
    pdf_table: int | None = None
    pdf_label: str | None = None


METRICS = (
    Metric("us_commercial_crude", "thousand_bbl", 4, "Commercial (Excluding SPR)", 3, pdf_line=6),
    Metric("padd1_commercial_crude", "thousand_bbl", 4, "East Coast (PADD 1)", 4, pdf_line=7),
    Metric("padd2_commercial_crude", "thousand_bbl", 4, "Midwest (PADD 2)", 5, pdf_line=8),
    Metric("cushing_crude", "thousand_bbl", 4, "Cushing", 6, pdf_line=9),
    Metric("padd3_commercial_crude", "thousand_bbl", 4, "Gulf Coast (PADD 3)", 7, pdf_line=10),
    Metric("padd4_commercial_crude", "thousand_bbl", 4, "Rocky Mountain (PADD 4)", 8, pdf_line=11),
    Metric("padd5_commercial_crude", "thousand_bbl", 4, "West Coast (PADD 5)", 9, pdf_line=12),
    Metric("us_spr_crude", "thousand_bbl", 1, "Strategic Petroleum Reserve (SPR)", 4),
    Metric("us_gasoline_stocks", "thousand_bbl", 1, "Total Motor Gasoline", 5),
    Metric("us_distillate_stocks", "thousand_bbl", 1, "Distillate Fuel Oil", 11),
    Metric("us_jet_fuel_stocks", "thousand_bbl", 1, "Kerosene-Type Jet Fuel", 10),
    Metric("us_crude_refiner_inputs", "thousand_bbl_per_day", 2, "Crude Oil Inputs", 2, pdf_line=8),
    Metric("padd2_crude_refiner_inputs", "thousand_bbl_per_day", 2, "Midwest (PADD 2)", 4, pdf_line=10),
    Metric("padd3_crude_refiner_inputs", "thousand_bbl_per_day", 2, "Gulf Coast (PADD 3)", 5, pdf_line=11),
    Metric("us_refinery_utilization", "percent", 2, "Percent Utilization", 20, pdf_line=26),
    Metric("padd2_refinery_utilization", "percent", 2, "Midwest (PADD 2)", 22, pdf_line=28),
    Metric("padd3_refinery_utilization", "percent", 2, "Gulf Coast (PADD 3)", 23, pdf_line=29),
    Metric("us_crude_production", "thousand_bbl_per_day", 9, "Domestic Production", group="Crude Oil Production",
           pdf_table=1, pdf_label="(1) Domestic Production"),
    Metric("us_crude_imports_including_spr", "thousand_bbl_per_day", 9, "Total Crude Oil Incl SPR", group="Imports",
           pdf_table=1, pdf_label="(5) Imports"),
    Metric("us_crude_exports", "thousand_bbl_per_day", 9, "Crude Oil", group="Exports",
           pdf_table=1, pdf_label="(9) Exports"),
    Metric("us_crude_net_imports_including_spr", "thousand_bbl_per_day", 9, "Crude Oil", group="Net Imports (Incl SPR)",
           pdf_table=1, pdf_label="(4) Net Imports (Including SPR)"),
    Metric("us_gasoline_supplied", "thousand_bbl_per_day", 9, "Finished Motor Gasoline", group="Product Supplied",
           pdf_table=1, pdf_label="(27) Finished Motor Gasoline"),
    Metric("us_distillate_supplied", "thousand_bbl_per_day", 9, "Distillate Fuel Oil", group="Product Supplied",
           pdf_table=1, pdf_label="(29) Distillate Fuel Oil"),
)

SCHEMA = """
CREATE TABLE eia_inventory_release_panel (
    week_ending TEXT PRIMARY KEY, release_date TEXT, study_phase TEXT, reported_as TEXT,
    cushing_kbbl INTEGER, cushing_printed_precision_kbbl INTEGER,
    previous_week_ending TEXT, previous_week_kbbl_as_known_at_release INTEGER,
    previous_week_vintage_date TEXT, previous_week_printed_precision_kbbl INTEGER,
    change_kbbl_as_known_at_release INTEGER, latest_week_known_before_release TEXT,
    latest_kbbl_known_before_release INTEGER, latest_vintage_known_before_release TEXT,
    latest_weekly_change_known_before_release INTEGER, prior_week_of_latest_known TEXT,
    prior_vintage_for_latest_known_change TEXT, source_url TEXT);
CREATE TABLE eia_control_vintage (
    metric TEXT, unit TEXT, release_date TEXT, report_date TEXT, value REAL, source_printed_increment REAL,
    table_number INTEGER, source_kind TEXT, source_row_or_line INTEGER, source_column_or_pdf_page INTEGER,
    source_label TEXT, raw_value TEXT, source_url TEXT, PRIMARY KEY (release_date, report_date, metric));
CREATE TABLE market_context_clock (
    release_date TEXT, phase TEXT, cutoff_utc TEXT, last_wpsr_issue TEXT, latest_wpsr_observation_week TEXT,
    last_complete_wti_curve_session TEXT, last_cftc_report_date TEXT, last_spot_common_reference_date TEXT,
    PRIMARY KEY (release_date, phase));
CREATE VIEW eia_market_context AS
    SELECT c.release_date,c.phase,c.cutoff_utc,c.last_wpsr_issue,c.latest_wpsr_observation_week,
        v.metric,v.unit,v.value,v.source_printed_increment,v.table_number,v.source_kind,
        v.source_row_or_line,v.source_column_or_pdf_page,v.source_url
    FROM market_context_clock c LEFT JOIN eia_control_vintage v ON v.release_date=c.last_wpsr_issue
        AND v.report_date=c.latest_wpsr_observation_week;
CREATE VIEW oil_market_context AS
    SELECT c.release_date,c.phase,c.cutoff_utc,c.last_complete_wti_curve_session,
        d.cl1,d.cl2,d.cl3,d.cl4,d.cl1_minus_cl2,d.cl1_minus_cl4,
        (SELECT group_concat(delivery_month, '|') FROM (
            SELECT delivery_month FROM wti_nearby_curve WHERE session_date=c.last_complete_wti_curve_session ORDER BY curve_rank
        )) AS wti_delivery_months_in_rank_order,
        c.last_cftc_report_date,t.conservative_research_from_utc AS cftc_conservative_from_utc,
        t.timing_basis AS cftc_timing_basis,p.open_interest,p.money_manager_long,p.money_manager_short,
        p.producer_long,p.producer_short,p.money_manager_long-p.money_manager_short AS money_manager_net,
        c.last_spot_common_reference_date,s.value AS wti_spot_historical_reference,b.value AS brent_spot_historical_reference,
        s.value-b.value AS wti_minus_brent_historical_reference
    FROM market_context_clock c LEFT JOIN wti_curve_daily_coverage d ON d.session_date=c.last_complete_wti_curve_session
    LEFT JOIN cftc_release_clock t ON t.report_date=c.last_cftc_report_date
    LEFT JOIN cftc_wti p ON p.report_date=t.report_date
    LEFT JOIN eia_series s ON s.series='wti_spot_daily' AND s.date=c.last_spot_common_reference_date
    LEFT JOIN eia_series b ON b.series='brent_spot_daily' AND b.date=c.last_spot_common_reference_date;
"""


def metric_values(db, metric):
    for issue, date, raw, row, column, label, source in db.execute("""SELECT release_date,report_date,raw_value,
        source_row,source_column,measure_label,source_url FROM wpsr_vintage_value
        WHERE table_number=? AND (measure_label=? OR (?='Cushing' AND measure_label='Cushing, Oklahoma'))
          AND (? IS NULL OR source_row=?) AND (? IS NULL OR group_label=?) AND release_date<>'2019-07-03'""",
        (metric.table, metric.label, metric.label, metric.row, metric.row, metric.group, metric.group)):
        yield standardized(metric, issue, date, raw, metric.table, "archived_csv", row, column, label, source)
    pdf_table = metric.pdf_table or metric.table
    expected = metric.pdf_label or metric.label
    matches = 0
    for issue, date, raw, line, page, label, source in db.execute("""SELECT release_date,report_date,raw_value,
        source_line,pdf_page,raw_label,source_url FROM wpsr_recovered_value
        WHERE release_date='2019-07-03' AND table_number=? AND pdf_page=1 AND (? IS NULL OR source_line=?)""",
        (pdf_table, metric.pdf_line, metric.pdf_line)):
        if not re.fullmatch(re.escape(expected) + r"(?:\s*\d[\d,]*)?", label):
            continue
        matches += 1
        yield standardized(metric, issue, date, raw, pdf_table, "recovered_issue_pdf", line, page, label, source)
    if matches != 2:
        raise ValueError(f"Expected two recovered issue values for {metric.key}, found {matches}")


def standardized(metric, issue, date, raw, table, kind, row, column, label, source):
    number, increment = None, None
    cleaned = raw.replace(",", "")
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", cleaned):
        value = Decimal(cleaned)
        scale = 1000 if metric.unit == "thousand_bbl" else 1
        number = float(value * scale)
        increment = float(Decimal(10)**value.as_tuple().exponent * scale)
    return metric.key, metric.unit, issue, date, number, increment, table, kind, row, column, label, raw, source


def preceding_week(date):
    return (dt.date.fromisoformat(date) - dt.timedelta(days=7)).isoformat()


def inventory_rows(db):
    vintages = defaultdict(list)
    for week, issue, value, precision, reported, source in db.execute("""SELECT week_ending,release_date,
        cushing_kbbl,precision_kbbl,reported_as,source_url FROM eia_cushing_as_published
        UNION ALL
        SELECT c.report_date,c.release_date,CAST(c.value AS INTEGER),CAST(c.source_printed_increment AS INTEGER),
            'previous_week_recovered_from_pdf',c.source_url
        FROM eia_control_vintage c WHERE c.metric='cushing_crude' AND c.source_kind='recovered_issue_pdf'
            AND NOT EXISTS (SELECT 1 FROM eia_cushing_as_published v WHERE v.release_date=c.release_date
                AND v.week_ending=c.report_date)
        ORDER BY release_date"""):
        vintages[week].append((issue, value, precision, reported, source))
    weeks = sorted(vintages)
    absent = (None, None, None, None, None)

    def latest(week, day, inclusive):
        found = [row for row in vintages.get(week, ()) if row[0] < day or (inclusive and row[0] == day)]
        return found[-1] if found else absent

    for week, issue, phase in db.execute("""SELECT week_ending,release_date,study_phase
        FROM eia_satellite_window WHERE endpoint='current' ORDER BY week_ending"""):
        _, value, precision, reported, source = latest(week, issue, True)
        prior_week = preceding_week(week)
        prior_issue, prior, prior_precision, _, _ = latest(prior_week, issue, True)
        known_week = next((old for old in reversed(weeks[:bisect_left(weeks, week)]) if latest(old, issue, False)[0]), None)
        known_issue, known, _, _, _ = latest(known_week, issue, False)
        earlier_week = preceding_week(known_week) if known_week else None
        earlier_issue, earlier, _, _, _ = latest(earlier_week, issue, False)
        yield (week, issue, phase, reported, value, precision, prior_week, prior, prior_issue, prior_precision,
               value-prior if prior is not None else None, known_week, known, known_issue,
               known-earlier if known is not None and earlier is not None else None, earlier_week, earlier_issue, source)


def context_rows(db):
    wpsr = dict(db.execute("SELECT release_date,latest_week_in_issue FROM eia_release_clock ORDER BY release_date"))
    issues = sorted(wpsr)
    curves = [day for day, in db.execute("""SELECT session_date FROM wti_curve_daily_coverage
        WHERE coverage_status='four_prices' ORDER BY session_date""")]
    spot = [day for day, in db.execute("""SELECT s.date FROM eia_series s JOIN eia_series b USING (date)
        WHERE s.series='wti_spot_daily' AND b.series='brent_spot_daily' ORDER BY s.date""")]
    cot = [(day, dt.datetime.fromisoformat(available)) for day, available in db.execute("""SELECT report_date,
        conservative_research_from_utc FROM cftc_release_clock ORDER BY report_date DESC""")]

    def before(dates, day):
        idx = bisect_left(dates, day)
        return dates[idx - 1] if idx else None

    for issue, pre, post in db.execute("""SELECT c.release_date,c.conservative_prerelease_before_utc,
        c.conservative_postrelease_from_utc FROM eia_release_clock c
        WHERE EXISTS (SELECT 1 FROM eia_inventory_release_panel p WHERE p.release_date=c.release_date)
        ORDER BY c.release_date"""):
        for phase, cutoff in (("pre_release", pre), ("post_release", post)):
            cutoff_date = cutoff[:10]
            wpsr_issue = before(issues, cutoff_date)
            cftc = next((day for day, available in cot if available <= dt.datetime.fromisoformat(cutoff)), None)
            yield (issue, phase, cutoff, wpsr_issue, wpsr.get(wpsr_issue), before(curves, cutoff_date), cftc,
                   before(spot, cutoff_date))


def add_eia_market_inputs(db):
    for metric in METRICS:
        db.executemany("INSERT INTO eia_control_vintage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", metric_values(db, metric))
    db.executemany("INSERT INTO eia_inventory_release_panel VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", inventory_rows(db))
    db.executemany("INSERT INTO market_context_clock VALUES (?,?,?,?,?,?,?,?)", context_rows(db))
