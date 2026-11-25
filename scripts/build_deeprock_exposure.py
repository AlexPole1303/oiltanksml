"""Publish dated Deeprock ownership, listed-parent stakes, and disclosure cutoffs."""
import datetime as dt
import gzip
import html
import re
import sqlite3
from contextlib import closing
from functools import lru_cache
from zoneinfo import ZoneInfo

from build_market_transaction_timeline import publication_window, reference_rows
from collect_public import RAW, ROOT
from publication_clocks import export

DERIVED = ROOT / "data/derived"
ARCHIVE = RAW / "documents/deeprock"
TERMINAL = "deeprock_development"
ORIGINAL = "deeprock_development_before_north"
COMBINED = "deeprock_development_and_north_combined"
NORTH_SOURCE = RAW / "documents/tallgrass_deeprock_north_transaction.html"
NORTH_URL = ("https://www.tallgrass.com/newsroom/press-releases/"
             "tallgrass-energy-partners-announces-crude-oil-terminal-acquisitions-and-commercial-developments-at-pony-express")
SCHEMA = """
CREATE TABLE IF NOT EXISTS deeprock_asset_ownership_period (
    effective_date TEXT PRIMARY KEY, end_date_exclusive TEXT, physical_scope TEXT, gross_storage_bbl INTEGER,
    terminals_stake_pct REAL, incremental_membership_pct REAL, interest_change_basis TEXT,
    filing_date TEXT, sec_accepted_at_utc TEXT, public_by_utc TEXT, source_url TEXT, source_file TEXT,
    supplemental_source_url TEXT, supplemental_source_file TEXT);
CREATE TABLE IF NOT EXISTS deeprock_parent_ownership_observation (
    instrument TEXT, observed_on TEXT, parent_tallgrass_equity_pct REAL,
    tallgrass_equity_tep_units_pct REAL, tep_units_observed_on TEXT,
    separate_gp_and_idr_rights INTEGER, filing_date TEXT, sec_accepted_at_utc TEXT, public_by_utc TEXT,
    source_url TEXT, source_file TEXT, supplemental_source_url TEXT, supplemental_source_file TEXT,
    PRIMARY KEY (instrument, observed_on));
CREATE TABLE IF NOT EXISTS deeprock_parent_structure_event (
    instrument TEXT, effective_date TEXT, event_type TEXT, filing_date TEXT, sec_accepted_at_utc TEXT,
    public_by_utc TEXT, source_url TEXT, source_file TEXT, PRIMARY KEY (instrument, effective_date));
CREATE TABLE IF NOT EXISTS deeprock_other_ownership_evidence (
    evidence_id TEXT PRIMARY KEY, entity TEXT, physical_scope TEXT, stated_pct REAL,
    effective_or_snapshot_date TEXT, date_basis TEXT, public_by_utc TEXT, source_url TEXT, source_file TEXT);
CREATE VIEW IF NOT EXISTS eia_deeprock_equity_exposure AS
    WITH releases AS (
        SELECT c.release_date,c.latest_week_in_issue AS week_ending,p.study_phase,
            c.conservative_postrelease_from_utc AS decision_at_utc,t.ticker AS market_instrument,t.role,
            s.window_starts_after_utc AS previous_window_start_utc
        FROM eia_release_clock c JOIN eia_inventory_release_panel p
            ON p.release_date=c.release_date AND p.week_ending=c.latest_week_in_issue
        JOIN terminal_company_history t ON t.terminal_id='deeprock_development'
            AND t.ticker IN ('TEP','TEGP','TGE') AND c.release_date>=t.valid_from
            AND (t.valid_to_exclusive IS NULL OR c.release_date<t.valid_to_exclusive)
        LEFT JOIN eia_satellite_window s ON s.week_ending=p.week_ending AND s.release_date=p.release_date
            AND s.endpoint='previous'
    )
    SELECT r.*,a.effective_date AS actual_asset_period_start,a.physical_scope AS actual_physical_scope,
        a.terminals_stake_pct AS actual_terminals_stake_pct,a.public_by_utc AS actual_asset_period_public_by_utc,
        k.effective_date AS public_asset_period_start,k.physical_scope AS public_physical_scope,
        k.terminals_stake_pct AS public_terminals_stake_pct,k.gross_storage_bbl AS public_gross_storage_bbl,
        coalesce(k.supplemental_source_url,k.source_url) AS public_asset_source_url,k.public_by_utc AS asset_public_by_utc,
        a.effective_date=k.effective_date AS actual_asset_period_known_at_cutoff,
        CASE WHEN r.previous_window_start_utc IS NOT NULL
            THEN r.previous_window_start_utc>='2018-01-03T06:00:00+00:00' END AS merged_scope_confirmed_for_pair_window,
        b.effective_date AS last_public_parent_structure_change,
        q.observed_on AS parent_ownership_observed_on,q.tep_units_observed_on,
        q.parent_tallgrass_equity_pct,q.tallgrass_equity_tep_units_pct,q.separate_gp_and_idr_rights,
        q.source_url AS parent_source_url,q.public_by_utc AS parent_ownership_public_by_utc,
        CASE WHEN r.market_instrument='TEP' THEN k.terminals_stake_pct
             ELSE k.terminals_stake_pct*q.parent_tallgrass_equity_pct*q.tallgrass_equity_tep_units_pct/10000.0
             END AS ordinary_lookthrough_pct_at_cutoff
    FROM releases r JOIN deeprock_asset_ownership_period a ON r.release_date>=a.effective_date
        AND (a.end_date_exclusive IS NULL OR r.release_date<a.end_date_exclusive)
    LEFT JOIN deeprock_asset_ownership_period k ON k.effective_date=(
        SELECT max(effective_date) FROM deeprock_asset_ownership_period
        WHERE effective_date<=r.release_date AND public_by_utc<=r.decision_at_utc)
    LEFT JOIN deeprock_parent_structure_event b ON b.instrument=r.market_instrument AND b.effective_date=(
        SELECT max(effective_date) FROM deeprock_parent_structure_event WHERE instrument=r.market_instrument
        AND effective_date<=r.release_date AND public_by_utc<=r.decision_at_utc)
    LEFT JOIN deeprock_parent_ownership_observation q ON q.instrument=r.market_instrument AND q.observed_on=(
        SELECT max(observed_on) FROM deeprock_parent_ownership_observation WHERE instrument=r.market_instrument
        AND observed_on<=r.release_date AND public_by_utc<=r.decision_at_utc
        AND (b.effective_date IS NULL OR observed_on>=b.effective_date));
"""


def normalized_body(raw):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", raw)).split()).casefold()


@lru_cache(maxsize=5)
def filing(key):
    path = ARCHIVE / f"{key}.txt.gz" if key != "tge_2019fy" else RAW / "issuer_distributions/tallgrass/2019_10K.txt"
    raw = (gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()).decode(errors="replace")
    head, body = raw.split("<DOCUMENT>", 1)
    accession = re.search(r"ACCESSION NUMBER:\s+([\d-]+)", head)[1]
    cik = int(re.search(r"CENTRAL INDEX KEY:\s+(\d+)", head)[1])
    filename = re.search(r"<FILENAME>([^\n]+)", body)[1].strip()
    filed = dt.datetime.strptime(re.search(r"FILED AS OF DATE:\s+(\d{8})", head)[1], "%Y%m%d").date().isoformat()
    accepted = dt.datetime.strptime(re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", head)[1], "%Y%m%d%H%M%S")
    accepted = accepted.replace(tzinfo=ZoneInfo("America/New_York")).astimezone(dt.timezone.utc).isoformat()
    _, available = publication_window(filed, "", "America/New_York")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filename}"
    return normalized_body(body), (filed, accepted, available, url, str(path.relative_to(ROOT)))


def verified_filing(key, *markers):
    body, info = filing(key)
    if any(marker.casefold() not in body for marker in markers):
        raise ValueError(f"Deeprock evidence is incomplete in {key}: {markers}")
    return info


def asset_periods():
    initial = verified_filing("tep_initial", "effective as of January 1, 2017", "A 20 percent interest", "2.3 million barrels")
    july = verified_filing("tep_july_2017", "On July 20, 2017", "additional 40% membership interest",
                           "On July 21, 2017", "additional 9% membership interest", "to 69%")
    north = verified_filing("tep_fy2017", "On January 2, 2018", "38% membership interest", "approximately 60% membership interest")
    press = normalized_body(NORTH_SOURCE.read_text(errors="replace"))
    if any(marker not in press for marker in ("jan 3, 2018", "60 percent of the combined entity", "4.0 million barrels")):
        raise ValueError("The merger release does not corroborate Deeprock North's scope and size")
    _, north_public = publication_window("2018-01-03", "", "America/Chicago")
    return [
        ("2017-01-01", "2017-07-20", ORIGINAL, 2300000, 20, 20, "TEP_acquired_Tallgrass_Terminals",
         *initial, initial[3].replace("d313884d8k.htm", "d313884dex991.htm"), initial[4]),
        ("2017-07-20", "2017-07-21", ORIGINAL, 2300000, 60, 40, "additional_stake_from_Kinder_Morgan", *july, None, None),
        ("2017-07-21", "2018-01-02", ORIGINAL, 2300000, 69, 9, "additional_stake_from_Deeprock_Energy_Resources", *july, None, None),
        ("2018-01-02", "2020-04-17", COMBINED, 4000000, 60, 38, "north_membership_acquisition_and_approximate_combined_stake",
         *north[:2], north_public, *north[3:], NORTH_URL, str(NORTH_SOURCE.relative_to(ROOT))),
    ]


def parent_observations():
    offer = verified_filing("tegp_2016_offer", "Voting power of TEGP in Tallgrass Equity", "36.94% if the underwriters")
    closed = verified_filing("tegp_2016_offer_close", "has exercised in full", "Offering closed on November 22, 2016")
    q3 = verified_filing("tegp_2016q3", "30.35% controlling membership interest in Tallgrass Equity",
                         "27.42% limited partner interest in TEP at November 2, 2016")
    yield ("TEGP", "2016-11-02", 30.35, 27.42, "2016-11-02", 1, *q3, None, None)
    yield ("TEGP", "2016-11-22", 36.94, 27.42, "2016-11-02", 1, *closed, *offer[3:])
    observations = (
        ("TEGP", "2017-02-15", 36.94, 27.41, "tegp_2016fy", "February 15, 2017"),
        ("TEGP", "2017-05-03", 36.94, 27.30, "tegp_2017q1", "May 3, 2017"),
        ("TEGP", "2017-08-02", 36.94, 27.03, "tegp_2017q2", "August 2, 2017"),
        ("TEGP", "2017-11-02", 36.94, 27.01, "tegp_2017q3", "November 2, 2017"),
        ("TEGP", "2018-02-13", 31.43, 34.60, "tegp_2017fy", "February 13, 2018"),
        ("TEGP", "2018-05-03", 31.43, 34.60, "tegp_2018q1", "May 3, 2018"),
        ("TGE", "2018-07-02", 55.28, 100, "tge_july_2018", "July 2, 2018"),
        ("TGE", "2018-09-30", 55.64, 100, "tge_2018q3", "September 30, 2018"),
        ("TGE", "2018-10-31", 55.79, 100, "tge_2018q3", "October 31, 2018"),
        ("TGE", "2018-12-31", 55.79, 100, "tge_2018fy", "December 31, 2018"),
        ("TGE", "2019-03-31", 63.55, 100, "tge_2019q1", "March 31, 2019"),
        ("TGE", "2019-05-07", 63.70, 100, "tge_2019q1", "May 7, 2019"),
        ("TGE", "2019-06-30", 63.70, 100, "tge_2019q2", "June 30, 2019"),
        ("TGE", "2019-09-30", 63.70, 100, "tge_2019q3", "September 30, 2019"),
        ("TGE", "2019-12-31", 63.75, 100, "tge_2019fy", "December 31, 2019"),
    )
    for ticker, observed, parent, underlying, key, source_day in observations:
        if ticker == "TEGP":
            markers = (f"{parent:.2f}% controlling membership interest in Tallgrass Equity",
                       f"{underlying:.2f}% limited partner interest in TEP at {source_day}")
        else:
            markers = (f"{parent:.2f}%", source_day, "wholly-owned subsidiary")
        evidence = verified_filing(key, *markers)
        if observed > evidence[0]:
            raise ValueError(f"Ownership observation follows the filing in {key}")
        yield ticker, observed, parent, underlying, observed, int(ticker == "TEGP"), *evidence, None, None


def structure_events():
    return [
        ("TEGP", "2018-02-07", "Tallgrass_Development_merger_and_TE_units_issue",
         *verified_filing("tegp_2018_feb_close", "5,619,218", "27,554,785", "February 7, 2018")),
        ("TGE", "2019-03-11", "Class_B_and_TE_units_exchanged_for_Class_A",
         *verified_filing("tge_march_exchange", "March 11, 2019", "were exchanged", "a like number of Class A")),
    ]


def other_ownership_evidence():
    north = "deeprock_north_before_merger"
    kmp = verified_filing("kmi_north_2011", "February 17, 2011", "50% ownership interest", "Deeprock North, LLC")
    july = verified_filing("tep_july_2017", "additional 40% membership interest", "Kinder Morgan Cushing LLC")
    merger = verified_filing("tep_fy2017", "38% membership interest", "Kinder Morgan Deeprock North Holdco")
    q3 = verified_filing("tge_2018q3", "44.36% interest in Tallgrass Equity", "40% membership interest in Deeprock Development")
    exhibit = ARCHIVE / "deeprock_banker_exhibit_99c2.htm"
    if "KMI (11%), Deeprock Energy Resources LLC (20%) and TGE (69%)" not in exhibit.read_text(errors="replace"):
        raise ValueError("The banker exhibit does not corroborate the pre-merger ownership snapshot")
    _, exhibit_public = publication_window("2020-01-21", "", "America/New_York")
    snapshot = (exhibit_public, "https://www.sec.gov/Archives/edgar/data/1633651/000119312520011134/d874077dex99c2.htm",
                str(exhibit.relative_to(ROOT)))
    return [
        ("north_kmp_formation", "Kinder Morgan Energy Partners", north, 50, "2011-02-17",
         "formation_stake_documented_in_annual_filing", *kmp[2:]),
        ("original_kmi_sale", "Kinder Morgan Cushing LLC", ORIGINAL, 40, "2017-07-20",
         "membership_interest_sold_at_close", *july[2:]),
        ("original_kmi_post_purchase_snapshot", "Kinder Morgan", ORIGINAL, 11, None,
         "retrospective_snapshot_after_july_2017_acquisitions_before_north_merger", *snapshot),
        ("original_der_post_purchase_snapshot", "Deeprock Energy Resources", ORIGINAL, 20, None,
         "retrospective_snapshot_after_july_2017_acquisitions_before_north_merger", *snapshot),
        ("north_kmi_sale", "Kinder Morgan Deeprock North Holdco", north, 38, "2018-01-02",
         "membership_interest_sold_at_close", *merger[2:]),
        ("combined_noncontrolling_snapshot", "Noncontrolling members in aggregate", COMBINED, 40, "2018-09-30",
         "reported_approximate_aggregate_noncontrolling_interest", *q3[2:]),
    ]


def add_deeprock_data(db):
    assets, parents, events = asset_periods(), list(parent_observations()), structure_events()
    for statement in SCHEMA.strip().split(";"):
        if statement.strip():
            db.execute(statement)
    for table, rows in (("deeprock_asset_ownership_period", assets), ("deeprock_parent_ownership_observation", parents),
                        ("deeprock_parent_structure_event", events),
                        ("deeprock_other_ownership_evidence", other_ownership_evidence())):
        db.execute(f"DELETE FROM {table}")
        placeholders = ",".join("?" for _ in rows[0])
        db.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)


def export_deeprock_data(db):
    for name, order in (("deeprock_asset_ownership_period", "effective_date"),
                        ("deeprock_parent_ownership_observation", "instrument,observed_on"),
                        ("deeprock_parent_structure_event", "effective_date,instrument"),
                        ("deeprock_other_ownership_evidence", "evidence_id"),
                        ("eia_deeprock_equity_exposure", "release_date,market_instrument")):
        export(db, name, order)


def main():
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        db.execute("BEGIN IMMEDIATE")
        with db:
            db.execute("DELETE FROM terminal_company_history WHERE terminal_id=?", (TERMINAL,))
            db.executemany("INSERT INTO terminal_company_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (tuple(v or None for v in row.values()) for row in reference_rows("terminal_company_history")
                            if row["terminal_id"] == TERMINAL))
            add_deeprock_data(db)
        export_deeprock_data(db)
        export(db, "eia_event_operator_exposure", "release_date,terminal_id,ticker")
        export(db, "eia_equity_horizon_input", "release_date,instrument")
        export(db, "eia_roof_cohort_weekly_coverage", "week_ending,cohort_id,ticker")
        print(dict(db.execute("""SELECT market_instrument,count(*) FROM eia_deeprock_equity_exposure
            GROUP BY market_instrument""")), flush=True)


if __name__ == "__main__":
    main()
