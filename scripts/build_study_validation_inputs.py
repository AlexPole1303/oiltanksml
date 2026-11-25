"""Assemble the 2023 physical, EIA and market inputs under the frozen phase clocks."""
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime
from statistics import median
from zoneinfo import ZoneInfo

from build_market_event_panel import CONTROLS, CORE
from build_operator_development_inputs import (
    group, json_values, little_farm_parent_evidence, market_rows, one_per_key,
    phase_market_calendar, release_group, release_rows, terminal_rows,
)
from build_registered_shadow_development import aggregate as registration_aggregate
from build_shadow_timing_measurements import summarize_window as timing_aggregate
from build_weekly_shadow_development_check import numeric, read_csv, seasonal_history
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from build_weekly_shadow_stable_panel import optics, pair_rows
from build_weekly_shadow_validation import DATED, RETROSPECTIVE
from collect_public import write

PHASE = "validation"
PREFIXES = {RETROSPECTIVE: "registry", DATED: "dated_2011"}
STEM = "study_validation"
EASTERN = ZoneInfo("America/New_York")
EIA_TARGET_FIELDS = (
    "reported_as", "cushing_kbbl", "cushing_printed_precision_kbbl", "previous_week_ending",
    "previous_week_kbbl_as_known_at_release", "previous_week_vintage_date", "previous_week_printed_precision_kbbl",
    "change_kbbl_as_known_at_release",
)


def measurement_clocks(db):
    return {
        week: (datetime.fromisoformat(start), datetime.fromisoformat(end), release, cutoff)
        for week, start, end, release, cutoff in db.execute("""
            SELECT p.week_ending,p.window_ends_by_utc,c.window_ends_by_utc,c.release_date,c.inputs_available_before_utc
            FROM eia_satellite_window p JOIN eia_satellite_window c USING(week_ending)
            WHERE p.endpoint='previous' AND c.endpoint='current' AND c.study_phase=?
        """, (PHASE,))
    }


def physical_inputs(clocks, terminals):
    fixed = [dict(r, circular_area_m2=float(r["circular_area_m2"]), in_fixed_cohort=True)
             for r in read_csv("weekly_shadow_stable_cohort") if r["in_fixed_cohort"] == "True"]
    if any(row["selection_last_release"] >= row["selection_release_before"] or
           row["selection_release_before"] >= min(clocks) for row in fixed):
        raise ValueError("Fixed cohort uses evidence after its selection cutoff")
    members = group(fixed, "terminal_id")
    pairs = group(pair_rows(
        (row for row in read_csv("weekly_shadow_validation_pairs") if row["roof_type"] == "EFR"), clocks, fixed
    ), "week_ending", "geometry_mode")
    optics_week, optics_terminal = {}, {}
    for week in sorted(clocks):
        for mode in PREFIXES:
            rows = pairs[week, mode]
            optics_week[week, mode] = optics(rows, fixed)
            by_terminal = group(rows, "terminal_id")
            for terminal in terminals:
                optics_terminal[week, terminal, mode] = optics(by_terminal[terminal,], members[terminal,])
    return optics_week, optics_terminal


def derived_physical_inputs(stem, empty, count_fields, clocks, terminals):
    weekly = one_per_key(read_csv(f"{stem}_weeks"), "week_ending", "geometry_mode")
    terminal = one_per_key(read_csv(f"{stem}_terminals"),
                           "week_ending", "geometry_mode", "terminal_id")
    if weekly.keys() != {(week, mode) for week in clocks for mode in PREFIXES}:
        raise ValueError(f"{stem} weeks differ from the frozen EIA clock")
    if any(week not in clocks or mode not in PREFIXES or site not in terminals for week, mode, site in terminal):
        raise ValueError(f"{stem} terminals are outside the input registry")
    for row in (*weekly.values(), *terminal.values()):
        _, _, release, cutoff = clocks[row["week_ending"]]
        if (row["release_date"], row["inputs_available_before_utc"]) != (release, cutoff):
            raise ValueError(f"{stem} uses a different cutoff for {row['week_ending']}")

    def extract(row):
        return {field: int(row[field]) if field in count_fields else numeric(row[field])
                for field in empty}

    return ({key: extract(row) for key, row in weekly.items()},
            {key: extract(row) for key, row in terminal.items()}, empty)


def eia_weeks(db, clocks, physical, registered, timing):
    history = seasonal_history(db)
    for raw in db.execute("""
        SELECT p.*,c.latest_week_in_issue,c.conservative_prerelease_before_utc,
            c.conservative_postrelease_from_utc,c.timing_basis AS issue_timing_basis
        FROM eia_inventory_release_panel p JOIN eia_release_clock c USING(release_date)
        WHERE p.study_phase=? ORDER BY p.week_ending
    """, (PHASE,)):
        row = dict(raw)
        week, issue = row["week_ending"], row["release_date"]
        previous, current, release, cutoff = clocks[week]
        if issue != release or cutoff != row["conservative_prerelease_before_utc"]:
            raise ValueError(f"Different publication clock for {week}")
        iso = date.fromisoformat(week).isocalendar()
        prior = sorted((w, d, value) for year, w, d, value in history[iso.week]
                       if iso.year-3 <= year < iso.year and w < week and d < issue)
        yield dict(
            row, latest_week_for_market_issue=week == row["latest_week_in_issue"],
            previous_eia_measurement_at_utc=previous.isoformat(), current_eia_measurement_at_utc=current.isoformat(),
            zero_change_kbbl=0, seasonal_median_kbbl=median(r[2] for r in prior) if prior else None,
            seasonal_observations=len(prior), seasonal_source_weeks="|".join(r[0] for r in prior),
            seasonal_source_releases="|".join(r[1] for r in prior),
            **{f"{prefix}_{key}": value for mode, prefix in PREFIXES.items()
               for key, value in physical[week, mode].items()},
            **{f"{prefix}_registration_{key}": value for mode, prefix in PREFIXES.items()
               for key, value in registered[week, mode].items()},
            **{f"{prefix}_timing_{key}": value for mode, prefix in PREFIXES.items()
               for key, value in timing[week, mode].items()},
        )


def public_context(db, view, releases):
    rows = release_group(db, view, "release_date", "phase", release_dates=releases)
    for issue, release in sorted(releases.items()):
        for phase, clock in (("pre_release", release["inputs_available_before_utc"]),
                             ("post_release", release["conservative_postrelease_from_utc"])):
            source = rows[issue, phase]
            if not source or any(row["cutoff_utc"] != clock for row in source):
                raise ValueError(f"Context clock mismatch in {view} for {issue} {phase}")
            if view == "eia_market_context":
                if any(row["last_wpsr_issue"] > issue or (phase == "pre_release" and row["last_wpsr_issue"] == issue)
                       for row in source):
                    raise ValueError(f"EIA context includes a later publication for {issue} {phase}")
            else:
                cutoff = datetime.fromisoformat(clock)
                local_day = cutoff.astimezone(EASTERN).date().isoformat()
                if any(datetime.fromisoformat(row["cftc_conservative_from_utc"]) > cutoff
                       or row["last_complete_wti_curve_session"] >= local_day
                       or row["last_spot_common_reference_date"] >= local_day for row in source):
                    raise ValueError(f"Oil context includes a later observation for {issue} {phase}")
            yield from sorted(source, key=lambda row: row.get("metric", ""))


def main():
    validation_optics = group(read_csv("weekly_shadow_validation_terminals"), "geometry_mode")
    measurements = one_per_key(validation_optics[RETROSPECTIVE,], "week_ending", "terminal_id")
    dated = one_per_key(validation_optics[DATED,], "week_ending", "terminal_id")
    execution = one_per_key(
        (r for r in read_csv("tws_event_execution_window_coverage") if r["study_phase"] == PHASE),
        "release_date", "instrument",
    )
    cost = one_per_key((r for r in read_csv("study_execution_inputs") if r["study_phase"] == PHASE),
                       "release_date", "instrument")
    quote_coverage = one_per_key(
        (r for r in read_csv("study_instrument_window_coverage") if r["study_phase"] == PHASE),
        "release_date", "instrument",
    )
    actions = group((r for r in read_csv("study_equity_window_actions") if r["study_phase"] == PHASE),
                    "release_date", "instrument")
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        releases = release_rows(db, phase=PHASE)
        latest = {r["release_date"]: r for r in releases if r["week_ending"] == r["latest_week_in_issue"]}
        if set(latest) != {r["release_date"] for r in releases}:
            raise ValueError("A validation issue lacks its latest EIA week")
        calendar = group((r for r in phase_market_calendar(db) if r["study_phase"] == PHASE), "release_date")
        safe = {issue: r for issue, r in latest.items() if calendar[issue,] and all(
            w["calendar_status"] == "fifth_close_precedes_next_phase_decision" for w in calendar[issue,])}
        relationships = release_group(db, "eia_event_operator_exposure", "release_date", "terminal_id",
                                      release_dates=latest)
        transfers = release_group(db, "eia_terminal_transfer_disclosure", "release_date", "terminal_id",
                                  release_dates=latest)
        parent = one_per_key(db.execute("""SELECT * FROM eia_cushing_connect_parent_equity_exposure
            WHERE study_phase=?""", (PHASE,)), "release_date", "market_instrument")
        deeprock = one_per_key(db.execute("""SELECT * FROM eia_deeprock_equity_exposure
            WHERE study_phase=?""", (PHASE,)), "release_date", "market_instrument")
        market = one_per_key(market_rows(db, safe, relationships, include_symbols=(*CORE, *CONTROLS)),
                             "release_date", "instrument")
        terminals = sorted({r[0] for r in db.execute("SELECT terminal_id FROM terminal_company_history")}
                           | {terminal for _, terminal in measurements})
        clocks = measurement_clocks(db)
        physical_weeks, physical_terminals = physical_inputs(clocks, terminals)
        registered_empty = registration_aggregate([])
        registered_counts = {field for field in registered_empty if field.endswith("_pairs") or field == "checked_offsets"}
        registered_weeks, registered_terminals, empty_registration = derived_physical_inputs(
            "weekly_shadow_registered_validation", registered_empty, registered_counts, clocks, terminals)
        timing_weeks, timing_terminals, empty_timing = derived_physical_inputs(
            "shadow_timing_validation", timing_aggregate([]),
            {"eligible_tank_weeks", "pre_measurement_pairs", "bracketed_pairs", "new_resolved_outer_pairs"}, clocks, terminals)
        stable = {(week, terminal): physical_terminals[week, terminal, RETROSPECTIVE]
                  for week in clocks for terminal in terminals}
        weeks = list(eia_weeks(db, clocks, physical_weeks, registered_weeks, timing_weeks))
        operators = list(terminal_rows(releases, terminals, measurements, dated, stable, relationships, market,
                                       transfers, parent, deeprock, little_farm_parent_evidence(db)))
        eia = list(public_context(db, "eia_market_context", latest))
        oil = list(public_context(db, "oil_market_context", latest))
    clock_map = {(r["release_date"], r["calendar_venue"]): r for values in calendar.values() for r in values}
    for (issue, ticker), row in market.items():
        clock = clock_map[issue, row["venue"]]
        if row["fifth_close_utc"] != clock["fifth_close_utc"] or (
            datetime.fromisoformat(row["fifth_close_utc"]) >= datetime.fromisoformat(clock["next_phase_first_decision_at_utc"])
        ):
            raise ValueError(f"Window crosses the validation boundary for {issue}/{ticker}")
        spread = execution.get((issue, ticker), {})
        cost_row = cost.get((issue, ticker), {})
        if ticker in CORE and (cost_row.get("phase_window_eligible"), cost_row.get("entry_session"), cost_row.get("fifth_session")) != (
                "True", row["entry_session"], row["fifth_session"]):
            raise ValueError(f"Cost input uses a different validation market window for {issue}/{ticker}")
        coverage = quote_coverage[issue, ticker]
        if any(row[field] != coverage[field] for field in ("entry_open_utc", "fifth_close_utc", "own_quote_status")):
            raise ValueError(f"Quote metadata and coverage disagree for {issue}/{ticker}")
        row.update(
            validation_window_status=clock["calendar_status"],
            contiguous_paired_prior_close_sessions=coverage["contiguous_paired_prior_close_sessions"],
            contiguous_paired_prior_ibkr_adjusted_close_sessions=coverage[
                "contiguous_paired_prior_ibkr_adjusted_close_sessions"],
            cad_reference_coverage=coverage["matched_cad_reference_window"],
            execution_observations_json=json_values(spread),
            observed_cost_inputs_short_instrument=cost_row.get("observed_cost_inputs_short_instrument"),
            observed_cost_inputs_short_amlp=cost_row.get("observed_cost_inputs_short_amlp"),
            operator_action_ids="|".join(r["event_id"] for r in actions[issue, ticker]),
            amlp_action_ids="|".join(r["event_id"] for r in actions[issue, "AMLP"]),
        )
    for row in operators:
        row["market_input_admission"] = ("fifth_close_overlaps_next_phase_decision"
                                         if row["release_date"] not in safe else row["matched_quote_status"])
        week, terminal = row["week_ending"], row["terminal_id"]
        row.update({f"{prefix}_{field}": values
                    for mode, prefix in PREFIXES.items()
                    for field, values in physical_terminals[week, terminal, mode].items()})
        row.update({f"{prefix}_registration_{field}": value
                    for mode, prefix in PREFIXES.items()
                    for field, value in registered_terminals.get((week, mode, terminal), empty_registration).items()})
        row.update({f"{prefix}_timing_{field}": value
                    for mode, prefix in PREFIXES.items()
                    for field, value in timing_terminals.get((week, mode, terminal), empty_timing).items()})
    target_keys = ("week_ending", "release_date", "conservative_postrelease_from_utc", "source_url", *EIA_TARGET_FIELDS)
    targets = [{field: row[field] for field in target_keys} for row in weeks]
    predictors = [{key: value for key, value in row.items() if key not in EIA_TARGET_FIELDS} for row in weeks]
    for suffix, rows in (("eia_weeks", predictors), ("eia_targets", targets),
                         ("terminal_inputs", operators), ("market_windows", list(market.values())),
                         ("eia_context", eia), ("oil_context", oil)):
        export(f"{STEM}_{suffix}", rows)
    summary = dict(
        reporting_weeks=len(weeks), distinct_eia_issues=len(latest), admitted_market_issues=len(safe),
        market_issues_after_boundary=sorted(latest.keys()-safe.keys()), terminal_input_rows=len(operators),
        market_windows=len(market), eia_context_rows=len(eia), oil_context_rows=len(oil),
        weeks_by_dated_pair_status=dict(Counter("resolved_pair" if row["dated_2011_all_resolved_pairs"] > 0
                                               else "no_resolved_pair" for row in weeks)),
        weeks_with_fixed_registry_members_observed=sum(r["registry_fixed_cohort_resolved_pairs"] > 0 for r in weeks),
        weeks_with_full_fixed_registry_cohort=sum(r["registry_fixed_cohort_complete"] for r in weeks),
        weeks_with_registered_pair={prefix: sum(r[f"{prefix}_registration_registered_resolved_pairs"] > 0 for r in weeks)
                                    for prefix in PREFIXES.values()},
        weeks_with_pair_resolved_by_both_methods={prefix: sum(r[f"{prefix}_registration_both_resolved_pairs"] > 0 for r in weeks)
                                                   for prefix in PREFIXES.values()},
        weeks_with_bracketed_timing_pair={prefix: sum(r[f"{prefix}_timing_bracketed_pairs"] > 0 for r in weeks)
                                         for prefix in PREFIXES.values()},
    )
    write(DERIVED / f"{STEM}_summary.json", json.dumps(summary, indent=2).encode() + b"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
