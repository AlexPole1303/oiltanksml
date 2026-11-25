"""Join the physical, EIA and market calendars without evaluating study outcomes."""
import json
import sqlite3
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime

from build_operator_development_inputs import (
    DATED, FULL_PRICES, USD_ALIGNMENT, group, one_per_key, optical_inputs, phase_market_calendar,
)
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

HORIZONS = """
SELECT release_date, latest_first_released_week, instrument, study_role, venue, currency,
    entry_session, entry_open_utc, fifth_session, fifth_close_utc,
    observed_first_five_sessions, amlp_observed_first_five_sessions,
    first_quote_provider, fifth_quote_provider, first_listing_segment, fifth_listing_segment,
    terminal_relation_boundary_in_window, own_quote_status, benchmark_alignment
FROM eia_equity_horizon_input
ORDER BY release_date, instrument
"""


def physical_coverage():
    coverage = {}
    modern, dated, _ = optical_inputs()
    validation = group(read_csv("weekly_shadow_validation_terminals"), "geometry_mode")
    held_out = group(
        (row for row in read_csv("weekly_shadow_holdout_input_manifest") if row["roof_type"] == "EFR"),
        "week_ending", "terminal_id",
    )
    for phase, registry, legacy in (
        ("development", modern, dated),
        ("validation", one_per_key(validation["retrospective_registry_geometry",], "week_ending", "terminal_id"),
         one_per_key(validation[DATED,], "week_ending", "terminal_id")),
    ):
        for week, terminal in registry.keys() | legacy.keys():
            base = registry.get((week, terminal), {})
            old = legacy.get((week, terminal), {})
            releases = {row["release_date"] for row in (base, old) if "release_date" in row}
            if len(releases) > 1:
                raise ValueError(f"Geometry variants disagree on the issue for {week}/{terminal}")
            coverage[week, terminal] = dict(
                study_phase=phase, release_date=next(iter(releases), None),
                efr_input_pairs=int(base.get("input_efr_pairs", 0)),
                efr_pairs_with_dated_2011_geometry=int(old.get("input_efr_pairs", 0)),
                efr_registry_geometry_resolved_pairs=int(base.get("resolved_efr_pairs", 0)),
                efr_dated_2011_geometry_resolved_pairs=int(old.get("resolved_efr_pairs", 0)),
            )
    for identity, rows in held_out.items():
        releases = {row["release_date"] for row in rows}
        if len(releases) != 1 or identity in coverage:
            raise ValueError(f"Invalid holdout optics calendar for {identity}")
        coverage[identity] = dict(
            study_phase="holdout", release_date=releases.pop(), efr_input_pairs=len(rows),
            efr_pairs_with_dated_2011_geometry=sum(r["engineering_eligibility"] == "dated_engineering_inputs" for r in rows),
            efr_registry_geometry_resolved_pairs=None, efr_dated_2011_geometry_resolved_pairs=None,
        )
    return coverage


def quote_history(db):
    quotes = {(ticker, day): (segment, adjusted)
              for ticker, day, segment, adjusted in db.execute("""
        SELECT instrument,session_date,listing_segment,adjusted_ibkr_close IS NOT NULL
        FROM market_study_quote WHERE close IS NOT NULL
    """)}
    calendars = defaultdict(list)
    clocks = defaultdict(dict)
    for venue, day, close in db.execute("""
        SELECT venue,session_date,close_utc FROM equity_session_calendar ORDER BY venue,session_date
    """):
        clock = datetime.fromisoformat(close)
        calendars[venue].append(clock)
        clocks[venue][day] = clock
    return quotes, calendars, clocks


def prior_close_coverage(horizon, cutoff, history):
    quotes, calendars, clocks = history
    instrument, venue = horizon["instrument"], horizon["venue"]
    sessions = list(clocks[venue])
    last = bisect_left(calendars[venue], datetime.fromisoformat(cutoff))
    last_day = sessions[last - 1] if last else None
    own_recent = quotes.get((instrument, last_day))
    amlp_recent = quotes.get(("AMLP", last_day))
    closes = adjusted = 0
    earliest = None
    for index in range(last - 1, -1, -1):
        day = sessions[index]
        own, amlp = quotes.get((instrument, day)), quotes.get(("AMLP", day))
        if (own is None or amlp is None or own[0] != own_recent[0] or amlp[0] != amlp_recent[0]
                or clocks[venue][day] != clocks["XNYS"].get(day)):
            break
        if adjusted == closes and own[1] and amlp[1]:
            adjusted += 1
        closes += 1
        earliest = day
    return dict(last_scheduled_session_before_decision=last_day, first_contiguous_paired_close_session=earliest,
                contiguous_paired_prior_close_sessions=closes, contiguous_paired_prior_ibkr_adjusted_close_sessions=adjusted)


def instrument_windows(db, calendar):
    listings = group(db.execute("""
        SELECT symbol,listing_id,first_study_session,last_study_session
        FROM historical_equity_listing_window ORDER BY symbol,first_study_session
    """), "symbol")
    fx = one_per_key(db.execute("""
        SELECT release_date,instrument,entry_fx_date,fifth_fx_date,cross_venue_clock_status
        FROM eia_cad_equity_reference_inputs
    """), "release_date", "instrument")
    corporate = group(db.execute("""
        SELECT release_date,instrument,transaction_id,pending_known_at_cutoff,
            legal_completion_in_horizon,spin_off_due_bill_session_in_horizon,spin_off_ex_distribution_in_horizon
        FROM eia_market_transaction_window
        ORDER BY release_date,instrument,transaction_id
    """), "release_date", "instrument")
    history = quote_history(db)
    for raw in db.execute(HORIZONS):
        row = dict(raw)
        issue, ticker = row["release_date"], row["instrument"]
        clock = calendar[issue, row["venue"]]
        for field in ("entry_session", "entry_open_utc", "fifth_session", "fifth_close_utc"):
            if row[field] != clock[field]:
                raise ValueError(f"Different {field} in the market window for {issue}/{ticker}")
        if clock["week_ending"] != row["latest_first_released_week"]:
            raise ValueError(f"Different latest EIA week for {issue}/{ticker}")
        if datetime.fromisoformat(row["entry_open_utc"]) < datetime.fromisoformat(clock["decision_at_utc"]):
            raise ValueError(f"Entry precedes the EIA decision for {issue}/{ticker}")
        fx_row = fx.get((issue, ticker), {})
        usd = row["own_quote_status"] == FULL_PRICES and row["benchmark_alignment"] == USD_ALIGNMENT
        cad = (row["own_quote_status"] == FULL_PRICES
               and fx_row.get("cross_venue_clock_status") == "matching_endpoint_clocks_and_five_sessions"
               and bool(fx_row["entry_fx_date"] and fx_row["fifth_fx_date"]))
        events = corporate[issue, ticker]
        listed_at_entry = [
            record for record in listings[ticker,]
            if record["first_study_session"] <= row["entry_session"] <= record["last_study_session"]
        ]
        if len(listed_at_entry) > 1:
            raise ValueError(f"Conflicting historical listing windows for {issue}/{ticker}")
        listing = listed_at_entry[0] if listed_at_entry else {}
        inside_listing = row["fifth_session"] <= listing["last_study_session"] if listing and row["fifth_session"] else None
        yield dict(
            study_phase=clock["study_phase"], **row, decision_at_utc=clock["decision_at_utc"],
            next_phase_first_decision_at_utc=clock["next_phase_first_decision_at_utc"],
            calendar_status=clock["calendar_status"], matched_usd_quote_window=usd, matched_cad_reference_window=cad,
            cad_clock_status=fx_row.get("cross_venue_clock_status"), entry_fx_date=fx_row.get("entry_fx_date"),
            fifth_fx_date=fx_row.get("fifth_fx_date"),
            historical_entry_listing_id=listing.get("listing_id"),
            historical_listing_active_at_entry=bool(listing) if row["study_role"] == "historical_operator" else None,
            fifth_close_within_historical_entry_listing=inside_listing,
            documented_pending_transactions="|".join(sorted({r["transaction_id"] for r in events if r["pending_known_at_cutoff"]})),
            corporate_action_in_horizon="|".join(sorted({r["transaction_id"] for r in events if (
                r["legal_completion_in_horizon"] or r["spin_off_due_bill_session_in_horizon"]
                or r["spin_off_ex_distribution_in_horizon"])})),
            **prior_close_coverage(row, clock["decision_at_utc"], history),
        )


def terminal_coverage(db, weeks, physical, windows):
    relationships = group(db.execute("""
        SELECT * FROM eia_event_operator_exposure
        ORDER BY release_date,terminal_id,ticker,role,valid_from
    """), "release_date", "terminal_id")
    undisclosed = group(db.execute("""
        SELECT release_date,terminal_id,transaction_id FROM eia_terminal_transfer_disclosure
        WHERE transfer_status='completion_not_established_as_public_by_cutoff'
    """), "release_date", "terminal_id")
    parent = one_per_key(db.execute("""
        SELECT release_date,market_instrument,parent_exposure_known_at_cutoff
        FROM eia_cushing_connect_parent_equity_exposure
    """), "release_date", "market_instrument")
    deeprock = one_per_key(db.execute("""
        SELECT release_date,market_instrument,actual_asset_period_known_at_cutoff,merged_scope_confirmed_for_pair_window
        FROM eia_deeprock_equity_exposure
    """), "release_date", "market_instrument")
    terminals = sorted({terminal for _, terminal in relationships} | {terminal for _, terminal in physical})
    for w in weeks:
        week, issue, phase = (w[field] for field in ("week_ending", "release_date", "study_phase"))
        for terminal in terminals:
            optical = physical.get((week, terminal))
            if optical and (optical["study_phase"] != phase or optical["release_date"] not in (None, issue)):
                raise ValueError(f"Optical and release clocks disagree for {week}/{terminal}")
            data = dict(efr_input_pairs=0, efr_pairs_with_dated_2011_geometry=0,
                        efr_registry_geometry_resolved_pairs=None if phase == "holdout" else 0,
                        efr_dated_2011_geometry_resolved_pairs=None if phase == "holdout" else 0)
            if optical:
                data = {field: optical[field] for field in data}
            for rel in relationships[issue, terminal] or ({},):
                ticker = rel.get("ticker")
                market = windows.get((issue, ticker), {})
                cv = parent.get((issue, ticker), {}) if terminal == "cushing_connect" else {}
                deep = deeprock.get((issue, ticker), {}) if terminal == "deeprock_development" else {}
                yield dict(
                    **w, terminal_id=terminal, **data,
                    issuer=rel.get("issuer"), ticker=ticker, role=rel.get("role"),
                    service_agreement_only=rel.get("role") == "contract_service_operator",
                    valid_from=rel.get("valid_from"), valid_to_exclusive=rel.get("valid_to_exclusive"),
                    date_precision=rel.get("date_precision"), exposure_date_status=rel.get("exposure_date_status"),
                    exposure_source_url=rel.get("source_url"),
                    parent_equity_exposure_known_at_cutoff=cv.get("parent_exposure_known_at_cutoff"),
                    deeprock_asset_period_known_at_cutoff=deep.get("actual_asset_period_known_at_cutoff"),
                    deeprock_merged_scope_confirmed_for_pair_window=deep.get("merged_scope_confirmed_for_pair_window")
                    if w["latest_week_for_market_issue"] else None,
                    legal_transfer_not_public_at_cutoff="|".join(sorted({
                        r["transaction_id"] for r in undisclosed[issue, terminal]})),
                    market_window_status=market.get("own_quote_status", "no_market_window" if ticker else "no_listed_symbol"),
                    matched_usd_quote_window=market.get("matched_usd_quote_window"),
                    matched_cad_reference_window=market.get("matched_cad_reference_window"),
                    market_calendar_status=market.get("calendar_status"),
                )


def issue_coverage(db, weeks, physical):
    controls = one_per_key(db.execute("""
        SELECT release_date,phase,count(DISTINCT CASE WHEN value IS NOT NULL THEN metric END) AS available_eia_controls
        FROM eia_market_context GROUP BY release_date,phase
    """), "release_date", "phase")
    oil = one_per_key(db.execute("""
        SELECT release_date,phase,last_complete_wti_curve_session,last_cftc_report_date,last_spot_common_reference_date
        FROM oil_market_context
    """), "release_date", "phase")
    optics = defaultdict(list)
    for (week, terminal), row in physical.items():
        optics[week].append(row)
    for w in weeks:
        issue = w["release_date"]
        rows = optics[w["week_ending"]]
        data = dict(
            efr_input_pairs=sum(r["efr_input_pairs"] for r in rows),
            terminals_with_efr_input_pairs=sum(r["efr_input_pairs"] > 0 for r in rows),
            efr_pairs_with_dated_2011_geometry=sum(r["efr_pairs_with_dated_2011_geometry"] for r in rows),
            efr_dated_2011_geometry_resolved_pairs=None if w["study_phase"] == "holdout"
            else sum(r["efr_dated_2011_geometry_resolved_pairs"] for r in rows),
        )
        for phase in ("pre_release", "post_release"):
            data[f"{phase}_available_eia_controls"] = controls[issue, phase]["available_eia_controls"]
            data.update({f"{phase}_{field}": oil[issue, phase][field] for field in (
                "last_complete_wti_curve_session", "last_cftc_report_date", "last_spot_common_reference_date")})
        yield dict(**w, **data)


def main():
    physical = physical_coverage()
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        calendar = one_per_key(phase_market_calendar(db), "release_date", "calendar_venue")
        weeks = [dict(row) for row in db.execute("""
            SELECT w.week_ending,w.release_date,w.study_phase,w.inputs_available_before_utc,
                w.window_starts_after_utc AS previous_eia_measurement_at_utc,
                w.window_ends_by_utc AS current_eia_measurement_at_utc,
                c.conservative_postrelease_from_utc AS decision_at_utc,c.source_url AS release_clock_source_url,
                w.week_ending=c.latest_week_in_issue AS latest_week_for_market_issue
            FROM eia_satellite_window w JOIN eia_release_clock c USING(release_date)
            WHERE w.endpoint='current' ORDER BY w.week_ending
        """)]
        instruments = list(instrument_windows(db, calendar))
        windows = one_per_key(instruments, "release_date", "instrument")
        terminals = list(terminal_coverage(db, weeks, physical, windows))
        issues = list(issue_coverage(db, weeks, physical))
    if {week for week, _ in physical} - {r["week_ending"] for r in weeks}:
        raise ValueError("Physical coverage has weeks outside the release calendar")
    export("study_week_input_coverage", issues)
    export("study_instrument_window_coverage", instruments)
    export("study_terminal_input_coverage", terminals)
    report = {}
    for phase in sorted({r["study_phase"] for r in weeks}):
        week_rows = [r for r in issues if r["study_phase"] == phase]
        instrument_rows = [r for r in instruments if r["study_phase"] == phase]
        candidate_windows = [r for r in terminals if r["study_phase"] == phase and r["latest_week_for_market_issue"]
                             and r["ticker"] and not r["service_agreement_only"] and r["efr_pairs_with_dated_2011_geometry"]]
        report[phase] = dict(
            reporting_weeks=len(week_rows), distinct_issues=len({r["release_date"] for r in week_rows}),
            weeks_with_efr_input_pairs=sum(r["efr_input_pairs"] > 0 for r in week_rows),
            weeks_with_dated_engineering_efr_pairs=sum(r["efr_pairs_with_dated_2011_geometry"] > 0 for r in week_rows),
            candidate_issue_instruments_with_dated_efr_inputs_and_usd_quotes=len({
                (r["release_date"], r["ticker"]) for r in candidate_windows if r["matched_usd_quote_window"]}),
            candidate_issue_instruments_with_resolved_dated_efr_and_usd_quotes=None if phase == "holdout" else len({
                (r["release_date"], r["ticker"]) for r in candidate_windows
                if r["matched_usd_quote_window"] and r["efr_dated_2011_geometry_resolved_pairs"]}),
            core_quote_status=dict(sorted(Counter(r["own_quote_status"] for r in instrument_rows
                                                  if r["study_role"] == "core_operator").items())),
        )
    write(DERIVED / "study_input_coverage_summary.json", json.dumps(report, indent=2).encode() + b"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
