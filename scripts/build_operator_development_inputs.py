"""Link development-period terminal measurements, EIA issues and operator market windows."""
import csv
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo

from build_terminal_development_measurements import DATED, RETROSPECTIVE, STUDY, TERMINALS
from build_weekly_shadow_development_check import ACTUAL, numeric, read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, write

STEM = "operator_development"
DEVELOPMENT_RELEASES = "SELECT release_date FROM eia_inventory_release_panel WHERE study_phase='development'"
FULL_PRICES = "five_observed_quote_sessions"
USD_ALIGNMENT = "same_date_usd_amlp_endpoints"
LISTING_HISTORY = ROOT / "data" / "reference" / "operator_primary_listing_history.csv"
LITTLE_FARM_SOURCES = ROOT / "data/reference/sunoco_little_farm_operator_parent_sources.csv"
EASTERN = ZoneInfo("America/New_York")


def group(rows, *fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(dict(row))
    return result


def one_per_key(rows, *fields):
    result = group(rows, *fields)
    if any(len(values) != 1 for values in result.values()):
        raise ValueError(f"Duplicate development records for {fields}")
    return {key: values[0] for key, values in result.items()}


def json_values(values):
    return json.dumps(values, ensure_ascii=False, separators=(",", ":")) if values else None


def optical_inputs():
    original = one_per_key(read_csv("weekly_shadow_development_terminals"), "week_ending", "terminal_id")
    additions = group(read_csv(f"{STUDY}_terminals"), "geometry_mode")
    modern = one_per_key(additions[RETROSPECTIVE,], "week_ending", "terminal_id")
    dated = one_per_key(additions[DATED,], "week_ending", "terminal_id")
    core_dated = one_per_key(read_csv("core_terminal_legacy_development_terminals"), "week_ending", "terminal_id")
    stable = one_per_key(read_csv("weekly_shadow_stable_terminals"), "week_ending", "terminal_id")
    if (original.keys() & modern.keys() or modern.keys() != dated.keys()
            or dated.keys() & core_dated.keys() or core_dated.keys() - original.keys()):
        raise ValueError("The terminal panels overlap or use different weeks")
    return original | modern, dated | core_dated, stable


def release_rows(db, *, phase="development"):
    return db.execute(f"""SELECT p.week_ending,p.release_date,p.cushing_kbbl,p.{ACTUAL},
        p.latest_weekly_change_known_before_release,p.source_url AS eia_source_url,
        c.latest_week_in_issue,c.conservative_postrelease_from_utc,c.timing_basis AS eia_timing_basis,
        w.inputs_available_before_utc
        FROM eia_inventory_release_panel p JOIN eia_release_clock c USING(release_date)
        JOIN eia_satellite_window w USING(week_ending,release_date)
        WHERE p.study_phase=? AND w.study_phase=? AND w.endpoint='current'
        ORDER BY p.week_ending""", (phase, phase)).fetchall()


def release_group(db, view, *keys, release_dates=None):
    if release_dates is None:
        return group(db.execute(f"SELECT * FROM {view} WHERE release_date IN ({DEVELOPMENT_RELEASES})"), *keys)
    dates = tuple(sorted(release_dates))
    marks = ",".join("?" for _ in dates)
    return group(db.execute(f"SELECT * FROM {view} WHERE release_date IN ({marks})", dates), *keys)


def phase_market_calendar(db):
    return [dict(row) for row in db.execute("""
        WITH issues AS (
            SELECT c.release_date,c.latest_week_in_issue AS week_ending,p.study_phase,
                c.conservative_postrelease_from_utc AS decision_at_utc,c.source_url
            FROM eia_release_clock c JOIN eia_inventory_release_panel p
                ON p.release_date=c.release_date AND p.week_ending=c.latest_week_in_issue
        ), starts AS (
            SELECT study_phase,min(decision_at_utc) AS first_decision_at_utc
            FROM issues GROUP BY study_phase
        ), cutoffs AS (
            SELECT study_phase,lead(study_phase) OVER by_start AS next_study_phase,
                lead(first_decision_at_utc) OVER by_start AS next_phase_first_decision_at_utc
            FROM starts WINDOW by_start AS (ORDER BY first_decision_at_utc)
        )
        SELECT i.week_ending,i.release_date,i.study_phase,s.venue AS calendar_venue,i.decision_at_utc,
            s.session_date AS entry_session,s.open_utc AS entry_open_utc,
            f.session_date AS fifth_session,f.close_utc AS fifth_close_utc,
            k.next_study_phase,k.next_phase_first_decision_at_utc,
            CASE WHEN k.next_study_phase IS NULL THEN 'final_phase'
                 WHEN f.close_utc IS NULL THEN 'fifth_close_outside_calendar'
                 WHEN f.close_utc>=k.next_phase_first_decision_at_utc THEN 'overlaps_next_phase_decision'
                 ELSE 'fifth_close_precedes_next_phase_decision' END AS calendar_status,
            i.source_url AS eia_release_clock_source_url
        FROM issues i JOIN cutoffs k USING(study_phase)
        JOIN eia_event_market_session s ON s.release_date=i.release_date AND s.ordinal=1
        LEFT JOIN eia_event_market_session f ON f.release_date=i.release_date AND f.venue=s.venue AND f.ordinal=5
        ORDER BY i.release_date,s.venue
    """)]


def primary_listings(db):
    with LISTING_HISTORY.open(newline="") as stream:
        documented = group(csv.DictReader(stream), "instrument")
    historical = group(db.execute("""SELECT symbol AS instrument,first_study_session AS first_session,
        date(last_study_session,'+1 day') AS last_session_exclusive,exchange AS primary_listing_venue,
        issuer_identity_source AS source_url FROM historical_equity_listing_window"""), "instrument")
    return historical | documented


def listing_at(listings, instrument, day):
    matches = [r for r in listings.get((instrument,), ()) if day and r["first_session"] <= day
               and (not r["last_session_exclusive"] or day < r["last_session_exclusive"])]
    if len(matches) > 1:
        raise ValueError(f"Overlapping listing records for {instrument} on {day}")
    return matches[0] if matches else {}


def market_rows(db, releases, relationships, *, include_symbols=()):
    expected = {(r["release_date"], r["ticker"]) for rows in relationships.values() for r in rows if r["ticker"]}
    expected.update((issue, ticker) for issue in releases for ticker in include_symbols)
    dates = tuple(sorted(releases))
    marks = ",".join("?" for _ in dates)
    horizons = db.execute(f"""SELECT * FROM eia_equity_horizon_input WHERE release_date IN ({marks})
        ORDER BY release_date,instrument""", dates)
    cad = release_group(db, "eia_cad_equity_reference_inputs", "release_date", "instrument", release_dates=dates)
    corporate = release_group(db, "eia_market_transaction_window", "release_date", "instrument", release_dates=dates)
    listings = primary_listings(db)
    cash = group(db.execute("""SELECT instrument,issuer_symbol,ex_date,record_date,payable_date,reported_amount,
        amount_on_post_2020_amlp_share_basis,currency AS reported_currency,source_basis,source_url
        FROM market_study_issuer_cash_distribution ORDER BY instrument,record_date"""), "instrument")
    for h in horizons:
        identity = h["release_date"], h["instrument"]
        if identity not in expected:
            continue
        release = releases[h["release_date"]]
        cutoff = datetime.fromisoformat(release["conservative_postrelease_from_utc"])
        entry = datetime.fromisoformat(h["entry_open_utc"])
        if h["entry_session"] <= h["release_date"] or entry < cutoff or h["latest_first_released_week"] != release["week_ending"]:
            raise ValueError(f"Market window precedes its decision or has a different EIA release for {identity}")
        end = h["fifth_session"]
        status = h["own_quote_status"]
        if status == FULL_PRICES:
            status = h["benchmark_alignment"]
        entry_listing = listing_at(listings, h["instrument"], h["entry_session"])
        exit_listing = listing_at(listings, h["instrument"], end)
        row = dict(h, decision_at_utc=release["conservative_postrelease_from_utc"], matched_quote_status=status,
            calendar_venue=h["venue"], primary_listing_venue_at_entry=entry_listing.get("primary_listing_venue"),
            primary_listing_venue_at_exit=exit_listing.get("primary_listing_venue"),
            primary_listing_entry_source_url=entry_listing.get("source_url"),
            primary_listing_exit_source_url=exit_listing.get("source_url"),
            cad_fx_and_clock_inputs_json=json_values(cad[identity]), corporate_windows_json=json_values(corporate[identity]))
        for label, ticker in (("operator", h["instrument"]), ("amlp", "AMLP")):
            events = [r for r in cash[ticker,] if end and r["ex_date"] and h["entry_session"] < r["ex_date"] <= end]
            unresolved = [r for r in cash[ticker,] if end and not r["ex_date"] and h["entry_session"] <= r["record_date"] <= end]
            row[f"{label}_documented_ex_date_cash_in_holding_json"] = json_values(events)
            row[f"{label}_record_dates_without_ex_date_in_window_json"] = json_values(unresolved)
        yield row


def optics(row, legacy, fixed):
    def change(data):
        barrels = numeric(data.get("resolved_visible_change_equivalent_bbl"))
        return barrels / 1000 if barrels is not None else None

    return dict(optical_geometry=row.get("geometry_mode", "original_retrospective") if row else None,
        input_efr_pairs=int(row.get("input_efr_pairs", 0)), resolved_efr_pairs=int(row.get("resolved_efr_pairs", 0)),
        resolved_circle_area_m2=float(row.get("resolved_circle_area_m2", 0)), resolved_visible_change_kbbl=change(row),
        dated_2011_resolved_pairs=int(legacy.get("resolved_efr_pairs", 0)), dated_2011_visible_change_kbbl=change(legacy),
        fixed_cohort_size=numeric(fixed.get("fixed_cohort_size")), fixed_cohort_complete=fixed.get("fixed_cohort_complete"),
        fixed_cohort_resolved_pairs=numeric(fixed.get("fixed_cohort_resolved_pairs")),
        fixed_cohort_visible_change_kbbl=numeric(fixed.get("stable_visible_kbbl")))


def little_farm_parent_evidence(db):
    with LITTLE_FARM_SOURCES.open(newline="") as stream:
        sources = list(csv.DictReader(stream))
    by_role = one_per_key(sources, "evidence_role")
    site, parent, confirmation = (by_role[role,] for role in (
        "site_operator", "listed_operator_parent", "listed_operator_parent_crosscheck"))
    archive, = db.execute("""SELECT archive_version_date FROM nei_roof_equipment_evidence
        WHERE inventory_year=2017 AND cohort_id='sunoco_little_farm_efr' AND equipment_id='74'""").fetchone()
    measured_cohorts = {cohort for cohort, in db.execute("""SELECT DISTINCT cohort_id FROM tank_engineering_registry
        WHERE documented_terminal_id='energy_transfer_cushing' AND documented_roof_type='EFR'""")}
    if (site["source_date"] != archive or site["related_entity"] != parent["subject"]
            or (confirmation["subject"], confirmation["related_entity"]) != (parent["subject"], parent["related_entity"])
            or {row["scope_id"] for row in sources} != {"sunoco_little_farm"}
            or measured_cohorts != {"sunoco_little_farm_efr"}):
        raise ValueError("The Little Farm site operator and listed parent evidence disagree")
    return site, parent


def terminal_rows(releases, terminal_ids, measurements, dated, stable, relationships, market, transfers,
                  parent_exposures, deeprock_exposures, little_farm_sources):
    little_farm_evidence_date = max(row["source_date"] for row in little_farm_sources)
    for release in releases:
        week, issue = release["week_ending"], release["release_date"]
        cutoff = datetime.fromisoformat(release["inputs_available_before_utc"]).astimezone(EASTERN).date().isoformat()
        latest = week == release["latest_week_in_issue"]
        for terminal in terminal_ids:
            key = week, terminal
            optical, historical, fixed = (panel.get(key, {}) for panel in (measurements, dated, stable))
            for row in (optical, historical, fixed):
                if row and row.get("release_date", issue) != issue:
                    raise ValueError(f"Optical panel has a different release for {key}")
                if row and row.get("inputs_available_before_utc", release["inputs_available_before_utc"]) != release["inputs_available_before_utc"]:
                    raise ValueError(f"Optical panel has a different cutoff for {key}")
            base = dict(release, terminal_id=terminal, latest_week_for_market_issue=latest, **optics(optical, historical, fixed))
            active = relationships[issue, terminal]
            for relationship in active or ({},):
                ticker = relationship.get("ticker")
                horizon = market.get((issue, ticker))
                exposure_status = relationship.get("exposure_date_status", "no_registry_relationship")
                status = horizon["matched_quote_status"] if horizon else ("no_market_window" if ticker else "no_listed_symbol")
                parent = parent_exposures.get((issue, ticker), {}) if terminal == "cushing_connect" else {}
                if terminal == "cushing_connect" and ticker == "DINO" and not parent:
                    raise ValueError(f"Missing dated Cushing Connect parent interest for {issue}")
                parent_known = parent.get("parent_exposure_known_at_cutoff")
                if parent_known == 0:
                    exposure_status = "parent_equity_exposure_not_public_by_cutoff"
                deeprock = deeprock_exposures.get((issue, ticker), {}) if terminal == "deeprock_development" else {}
                pending_transfer = [r["transaction_id"] for r in transfers[issue, terminal]
                                    if r["transfer_status"] == "completion_not_established_as_public_by_cutoff"]
                little_farm = terminal == "energy_transfer_cushing" and ticker == "ET" and base["input_efr_pairs"] > 0
                little_farm_known = little_farm_evidence_date < cutoff if little_farm else None
                yield dict(base, issuer=relationship.get("issuer"), ticker=ticker, role=relationship.get("role"),
                    service_agreement_only=relationship.get("role") == "contract_service_operator",
                    interest_pct=relationship.get("interest_pct"), valid_from=relationship.get("valid_from"),
                    valid_to_exclusive=relationship.get("valid_to_exclusive"),
                    date_precision=relationship.get("date_precision"), exposure_date_status=exposure_status,
                    exposure_source_url=relationship.get("source_url"),
                    optical_site_scope_id="sunoco_little_farm" if little_farm else None,
                    optical_site_operator_parent_known_before_cutoff=little_farm_known,
                    optical_site_operator_parent_evidence_date=little_farm_evidence_date if little_farm else None,
                    optical_site_operator_parent_sources_json=json_values(little_farm_sources) if little_farm else None,
                    parent_exchange_symbol=parent.get("parent_exchange_symbol"),
                    parent_equity_exposure_known_at_cutoff=parent_known,
                    parent_terminal_pct_at_cutoff=parent.get("parent_terminal_pct_at_cutoff"),
                    parent_equity_exposure_available_by_utc=parent.get("available_by_utc"),
                    deeprock_public_physical_scope=deeprock.get("public_physical_scope"),
                    deeprock_terminals_stake_pct_at_cutoff=deeprock.get("public_terminals_stake_pct"),
                    deeprock_ordinary_lookthrough_pct_at_cutoff=deeprock.get("ordinary_lookthrough_pct_at_cutoff"),
                    deeprock_asset_period_known_at_cutoff=deeprock.get("actual_asset_period_known_at_cutoff"),
                    deeprock_merged_scope_confirmed_for_pair_window=(
                        deeprock.get("merged_scope_confirmed_for_pair_window") if latest else None),
                    transfer_completion_not_public_by_cutoff=json_values(pending_transfer),
                    matched_quote_status=status,
                    matched_usd_prices_for_latest_week=latest and status == USD_ALIGNMENT,
                    entry_session=horizon["entry_session"] if horizon else None,
                    fifth_session=horizon["fifth_session"] if horizon else None)


def coverage(rows):
    for (terminal, ticker, role), values in sorted(group(rows, "terminal_id", "ticker", "role").items(),
                                                   key=lambda item: tuple(v or "" for v in item[0])):
        latest = [r for r in values if r["latest_week_for_market_issue"]]
        measured = [r for r in latest if r["resolved_visible_change_kbbl"] is not None]
        priced = [r for r in measured if r["matched_usd_prices_for_latest_week"]]
        historical = [r for r in latest if r["matched_usd_prices_for_latest_week"]
                      and r["dated_2011_visible_change_kbbl"] is not None]
        yield dict(terminal_id=terminal, ticker=ticker, role=role, issue_weeks=len(latest), measured_issue_weeks=len(measured),
            measured_and_matched_usd_quote_weeks=len(priced), dated_2011_and_matched_usd_quote_weeks=len(historical),
            both_geometry_modes_and_matched_usd_quote_weeks=sum(
                r["resolved_visible_change_kbbl"] is not None for r in historical),
            little_farm_dated_2011_with_operator_parent_evidence_weeks=sum(
                r["optical_site_operator_parent_known_before_cutoff"] is True for r in historical),
            measured_issues_with_uncertain_exposure_dates=sum(r["exposure_date_status"] in
                ("effective_dates_unspecified", "boundary_month_uncertain", "parent_equity_exposure_not_public_by_cutoff")
                for r in measured),
            first_measured_week=min((r["week_ending"] for r in measured), default=None),
            last_measured_week=max((r["week_ending"] for r in measured), default=None))


def main():
    measurements, dated, stable = optical_inputs()
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        releases = release_rows(db)
        latest = {r["release_date"]: r for r in releases if r["week_ending"] == r["latest_week_in_issue"]}
        if set(latest) != {r["release_date"] for r in releases}:
            raise ValueError("A development issue lacks its latest EIA week")
        relationships = release_group(db, "eia_event_operator_exposure", "release_date", "terminal_id")
        transfers = release_group(db, "eia_terminal_transfer_disclosure", "release_date", "terminal_id")
        parent_exposures = one_per_key(db.execute("""SELECT * FROM eia_cushing_connect_parent_equity_exposure
            WHERE study_phase='development'"""), "release_date", "market_instrument")
        deeprock_exposures = one_per_key(db.execute("""SELECT * FROM eia_deeprock_equity_exposure
            WHERE study_phase='development'"""), "release_date", "market_instrument")
        market = one_per_key(market_rows(db, latest, relationships), "release_date", "instrument")
        little_farm_sources = little_farm_parent_evidence(db)
        calendar = phase_market_calendar(db)
        terminal_ids = {r[0] for r in db.execute("""SELECT terminal_id FROM terminal_company_history UNION
            SELECT documented_terminal_id FROM tank_engineering_registry WHERE documented_terminal_id IS NOT NULL""")}
    terminal_ids |= {terminal for _, terminal in measurements}
    rows = list(terminal_rows(releases, sorted(terminal_ids), measurements, dated, stable, relationships,
                              market, transfers, parent_exposures, deeprock_exposures, little_farm_sources))
    counts = list(coverage(rows))
    export(f"{STEM}_terminal_exposures", rows)
    export(f"{STEM}_market_windows", list(market.values()))
    export(f"{STEM}_coverage", counts)
    export("eia_phase_market_calendar", calendar)
    measured = {(r["release_date"], r["ticker"]) for r in rows
                if r["matched_usd_prices_for_latest_week"] and r["resolved_visible_change_kbbl"] is not None}
    report = dict(reporting_weeks=len(releases), distinct_issues=len(latest), terminals=len(terminal_ids), exposure_rows=len(rows),
        market_windows=len(market), unique_issue_symbols_with_measurement_and_usd_quotes=len(measured),
        cross_phase_calendar_windows=sum(r["calendar_status"] == "overlaps_next_phase_decision" for r in calendar),
        new_terminal_price_overlap=[r for r in counts if r["terminal_id"] in TERMINALS and r["ticker"]],
        scope="Development issues only. Market rows contain quotes and cash source observations. "
              "No investment returns, cash-completeness assumption, or execution costs are calculated. "
              "Exposure dates describe registry evidence. Separate terminal rows sharing one issuer and issue share one market window.")
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
