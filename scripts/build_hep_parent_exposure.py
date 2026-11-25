"""Reconstruct the listed parent's dated economic exposure to Cushing Connect."""
import csv
import datetime as dt
import json
import sqlite3
from bisect import bisect_left
from contextlib import closing
from decimal import Decimal

from build_market_transaction_timeline import publication_window, reference_rows
from collect_public import RAW, ROOT, write
from publication_clocks import export

DERIVED = ROOT / "data" / "derived"
ARCHIVE = RAW / "issuer_distributions"
ISSUER_DOCS = ARCHIVE / "sec_index"
EXPOSURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS hep_parent_unit_observation (
    report_period_end TEXT PRIMARY KEY, parent_exchange_symbol TEXT, parent_common_units INTEGER,
    cover_unit_count_date TEXT, total_common_units INTEGER, issuer_reported_parent_pct REAL,
    calculated_parent_pct REAL, filing_date TEXT, sec_accepted_at_utc TEXT, available_by_utc TEXT,
    source_url TEXT, filing_form TEXT);
CREATE TABLE IF NOT EXISTS cushing_connect_parent_equity_regime (
    event_id TEXT PRIMARY KEY, legal_effective_date TEXT, next_effective_date_exclusive TEXT,
    parent_exchange_symbol TEXT, parent_hep_pct REAL, hep_terminal_pct REAL, implied_terminal_pct REAL,
    basis_observation_period_end TEXT, parent_common_units_basis INTEGER, hep_common_units_basis INTEGER,
    new_hep_common_units INTEGER, percentage_basis TEXT, filing_date TEXT, sec_accepted_at_utc TEXT,
    available_by_utc TEXT, source_url TEXT);
CREATE TABLE IF NOT EXISTS cushing_connect_pipeline_contract_term (
    agreement_version INTEGER PRIMARY KEY, agreement_effective_date TEXT, filing_date TEXT,
    sec_accepted_at_utc TEXT, service_provider TEXT, shipper TEXT, minimum_reserved_capacity_bpd INTEGER,
    minimum_throughput_bpd INTEGER, base_usd_per_bbl TEXT, above_minimum_usd_per_bbl TEXT,
    ppi_adjustment_floor_pct REAL, ppi_adjustment_cap_pct REAL, ppi_adjustment_start_date TEXT,
    initial_term_end_date TEXT, automatic_renewal_years INTEGER, source_url TEXT, available_by_utc TEXT);
CREATE VIEW IF NOT EXISTS eia_cushing_connect_parent_equity_exposure AS
    SELECT c.release_date,c.latest_week_in_issue AS week_ending,w.study_phase,
        c.conservative_postrelease_from_utc AS decision_at_utc,'DINO' AS market_instrument,
        r.parent_exchange_symbol,r.event_id,r.legal_effective_date,r.next_effective_date_exclusive,
        r.parent_hep_pct,r.hep_terminal_pct,r.implied_terminal_pct,
        r.available_by_utc<=c.conservative_postrelease_from_utc AS parent_exposure_known_at_cutoff,
        CASE WHEN r.available_by_utc<=c.conservative_postrelease_from_utc THEN r.implied_terminal_pct END
            AS parent_terminal_pct_at_cutoff,
        r.filing_date,r.sec_accepted_at_utc,r.available_by_utc,r.basis_observation_period_end,r.percentage_basis,
        r.source_url
    FROM eia_release_clock c JOIN eia_inventory_release_panel w
        ON w.release_date=c.release_date AND w.week_ending=c.latest_week_in_issue
    JOIN cushing_connect_parent_equity_regime r
        ON c.release_date>=r.legal_effective_date
        AND (r.next_effective_date_exclusive IS NULL OR c.release_date<r.next_effective_date_exclusive);
"""


def read_observations():
    with (ARCHIVE / "HEP_parent_common_unit_observations_extract.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def filings():
    payload = json.loads((ISSUER_DOCS / "HEP_submissions.json").read_bytes())
    if payload["cik"] != "0001283140":
        raise ValueError("HEP filing source has a different registrant")
    recent = payload["filings"]["recent"]
    return {acc.replace("-", ""): (filed, accepted)
            for acc, filed, accepted in zip(recent["accessionNumber"], recent["filingDate"],
                                            recent["acceptanceDateTime"], strict=True)}


def disclosure_time(row, index):
    accession = row["source_url"].split("/")[-2]
    expected = row["filing_date"], row["sec_accepted_at_utc"]
    if index.get(accession) != expected:
        raise ValueError(f"Filing time differs from SEC metadata for {accession}")
    _, end = publication_window(row["filing_date"], "", "America/New_York")
    accepted = dt.datetime.fromisoformat(row["sec_accepted_at_utc"])
    if accepted >= dt.datetime.fromisoformat(end):
        raise ValueError(f"SEC acceptance falls outside the conservative upper bound for {accession}")
    return end


def observations_with_clock(index):
    data = json.loads((ISSUER_DOCS / "HEP_companyfacts.json").read_bytes())
    counts = {r["accn"].replace("-", ""): (r["end"], r["val"])
              for r in data["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]}
    records = []
    for row in sorted(read_observations(), key=lambda r: r["report_period_end"]):
        period, count_date = row["report_period_end"], row["cover_unit_count_date"]
        owned, total = (int(row[name]) for name in ("parent_common_units", "total_common_units"))
        if not period <= count_date <= row["filing_date"] or not 0 < owned <= total:
            raise ValueError(f"Invalid parent unit count for {period}")
        if row["filing_form"] in ("10-K", "10-Q") and counts.get(row["source_url"].split("/")[-2]) != (count_date, total):
            raise ValueError(f"HEP cover unit count differs from the SEC XBRL record for {period}")
        pct = Decimal(owned) * 100 / total
        if round(pct) != int(row["issuer_reported_parent_pct"]):
            raise ValueError(f"HEP unit counts differ from the reported percentage for {period}")
        records.append(dict(row, parent_common_units=owned, total_common_units=total, calculated_parent_pct=pct,
                            available_by_utc=disclosure_time(row, index)))
    if len(records) != 19 or len({row["report_period_end"] for row in records}) != len(records):
        raise ValueError("HEP observation periods are incomplete or duplicated")
    return records


def ownership_regimes(records, index):
    source = list(reference_rows("cushing_connect_parent_equity_events"))
    effective = [row["legal_effective_date"] for row in source]
    if len(source) != 3 or effective != sorted(set(effective)):
        raise ValueError("Cushing Connect ownership events must be ordered and unique")
    dates = [row["filing_date"] for row in records]
    if dates != sorted(dates):
        raise ValueError("HEP periodic filings must be chronological")
    for event, next_effective in zip(source, [*effective[1:], None], strict=True):
        available = disclosure_time(event, index)
        override = event["reported_parent_pct_override"]
        issue = int(event["additional_hep_common_units"] or 0)
        if override:
            period, numerator, denominator = None, None, None
            pct, basis = Decimal(override), "issuer_disclosed_wholly_owned"
        else:
            pos = bisect_left(dates, event["filing_date"]) - 1
            if pos < 0:
                raise ValueError(f"No prior unit count for {event['event_id']}")
            prior = records[pos]
            period = prior["report_period_end"]
            numerator, denominator = prior["parent_common_units"], prior["total_common_units"] + issue
            pct = Decimal(numerator) * 100 / denominator
            basis = "latest_disclosed_parent_and_total_units"
            if issue:
                basis = "prior_disclosed_units_plus_new_issued_units"
                following = records[pos + 1]
                if (numerator, denominator) != (following["parent_common_units"], following["total_common_units"]):
                    raise ValueError("The post-issuance balance differs from the next periodic disclosure")
        terminal_pct = Decimal(event["hep_terminal_pct"])
        yield (event["event_id"], event["legal_effective_date"], next_effective, event["parent_exchange_symbol"],
               str(pct), str(terminal_pct), str(pct * terminal_pct / 100), period, numerator, denominator, issue,
               basis, event["filing_date"], event["sec_accepted_at_utc"], available, event["source_url"])


def add_hep_parent_data(db):
    index = filings()
    observations = observations_with_clock(index)
    regimes = list(ownership_regimes(observations, index))
    links = db.execute("""SELECT valid_from,valid_to_exclusive,interest_pct
        FROM terminal_company_history WHERE terminal_id='cushing_connect' AND ticker='DINO'
        ORDER BY valid_from""").fetchall()
    if len(links) != len(regimes) or any(
            (link[0], link[1]) != (regime[1], regime[2]) or abs(link[2] - float(regime[6])) > 1e-9
            for link, regime in zip(links, regimes, strict=True)):
        raise ValueError("Registry disagrees with the Cushing Connect parent equity history")
    db.execute("DELETE FROM hep_parent_unit_observation")
    db.execute("DELETE FROM cushing_connect_parent_equity_regime")
    db.execute("DELETE FROM cushing_connect_pipeline_contract_term")
    db.executemany("""INSERT INTO hep_parent_unit_observation VALUES
        (:report_period_end,:parent_exchange_symbol,:parent_common_units,:cover_unit_count_date,:total_common_units,
         :issuer_reported_parent_pct,:calculated_parent_pct,:filing_date,:sec_accepted_at_utc,:available_by_utc,
         :source_url,:filing_form)""",
        (dict(row, calculated_parent_pct=str(row["calculated_parent_pct"])) for row in observations))
    db.executemany("INSERT INTO cushing_connect_parent_equity_regime VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", regimes)
    terms = list(reference_rows("cushing_connect_pipeline_contract_terms"))
    if [row["agreement_version"] for row in terms] != ["6", "7", "8", "9"]:
        raise ValueError("Cushing Connect agreement versions are incomplete or unordered")
    for term in terms:
        if (int(term["minimum_throughput_bpd"]) > int(term["minimum_reserved_capacity_bpd"])
                or not Decimal("0") < Decimal(term["above_minimum_usd_per_bbl"]) < Decimal(term["base_usd_per_bbl"])
                or not term["agreement_effective_date"] <= term["filing_date"] < term["initial_term_end_date"]):
            raise ValueError(f"Invalid Cushing Connect terms in version {term['agreement_version']}")
    db.executemany("INSERT INTO cushing_connect_pipeline_contract_term VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   ((*term.values(), disclosure_time(term, index)) for term in terms))


def main():
    with closing(sqlite3.connect(DERIVED / "cushing.sqlite")) as db:
        db.executescript(EXPOSURE_SCHEMA)
        db.execute("BEGIN IMMEDIATE")
        with db:
            db.execute("DELETE FROM terminal_company_history WHERE terminal_id='cushing_connect' AND ticker='DINO'")
            db.executemany("INSERT INTO terminal_company_history VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                           (tuple(value or None for value in r.values()) for r in reference_rows("terminal_company_history")
                            if r["terminal_id"] == "cushing_connect" and r["ticker"] == "DINO"))
            add_hep_parent_data(db)
        for name, order in (("hep_parent_unit_observation", "report_period_end"),
                            ("cushing_connect_parent_equity_regime", "legal_effective_date"),
                            ("cushing_connect_pipeline_contract_term", "agreement_version"),
                            ("eia_cushing_connect_parent_equity_exposure", "release_date"),
                            ("eia_event_operator_exposure", "release_date,terminal_id,ticker"),
                            ("eia_equity_horizon_input", "release_date,instrument"),
                            ("terminal_data_coverage", "terminal_id")):
            export(db, name, order_by=order)
        coverage = dict(db.execute("""SELECT event_id,count(*) FROM eia_cushing_connect_parent_equity_exposure
            GROUP BY event_id"""))
        report = {"exposure_issues_by_regime": coverage,
                  "first_cutoff_pending": [day for day, in db.execute("""SELECT release_date
                      FROM eia_cushing_connect_parent_equity_exposure WHERE parent_exposure_known_at_cutoff=0""")]}
        write(DERIVED / "cushing_connect_parent_exposure_summary.json", (json.dumps(report, indent=2) + "\n").encode())
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
