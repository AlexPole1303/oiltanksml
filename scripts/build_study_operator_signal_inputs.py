"""Assemble causal asset and hedge inputs without reading study outcomes."""
import json
import math
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from statistics import covariance, variance

from build_market_event_panel import CORE
from build_operator_development_inputs import EASTERN, group, json_values, little_farm_parent_evidence, one_per_key
from build_study_execution_inputs import BENCHMARK, VENUE
from build_weekly_shadow_development_check import numeric, read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

STEM = "study_operator_signal_inputs"
TERMINALS = "study_operator_terminal_signal_inputs"
OPEN_PHASES = ("development", "validation")
BETA_SESSIONS = (60, 120)
OPEN_LAST_SESSION = "2023-12-31"
ELIGIBLE_WINDOWS = ("fifth_close_precedes_next_phase_decision", "final_phase")
PHYSICAL = "sentinel2_space_time_coverage"
IDENTITY_FIELDS = ("terminal_id", "ticker", "role", "valid_from", "valid_to_exclusive")
EXPOSURE_FIELDS = ("issuer", "role", "valid_from", "valid_to_exclusive", "date_precision", "exposure_date_status",
    "exposure_source_url", "parent_equity_exposure_known_at_cutoff", "deeprock_asset_period_known_at_cutoff",
    "deeprock_merged_scope_confirmed_for_pair_window", "legal_transfer_not_public_at_cutoff")
AREA_FIELDS = ("dated_reference_tanks", "dated_reference_area_m2", "eligible_input_tanks", "eligible_input_area_m2",
    "exterior_checked_tanks", "exterior_checked_area_m2", "baseline_b04_tanks", "baseline_b04_area_m2")
MARKET_FIELDS = ("entry_session", "entry_open_utc", "fifth_session", "fifth_close_utc", "calendar_status",
    "own_quote_status", "matched_usd_quote_window", "matched_cad_reference_window", "cad_clock_status", "entry_fx_date",
    "fifth_fx_date", "historical_entry_listing_id", "historical_listing_active_at_entry", "first_listing_segment",
    "fifth_listing_segment", "terminal_relation_boundary_in_window", "documented_pending_transactions",
    "corporate_action_in_horizon", "last_scheduled_session_before_decision", "contiguous_paired_prior_close_sessions",
    "contiguous_paired_prior_ibkr_adjusted_close_sessions")
COST_SPREADS = ("instrument_entry_spread_bps", "instrument_exit_spread_bps", "amlp_entry_spread_bps", "amlp_exit_spread_bps")
COST_FIELDS = ("both_legs_have_two_spreads", "instrument_five_borrow_rates", "amlp_five_borrow_rates",
    "observed_cost_inputs_short_instrument", "observed_cost_inputs_short_amlp", "commission_scenario_snapshot_date")
CONTEXT_FIELDS = ("pre_release_available_eia_controls", "pre_release_last_complete_wti_curve_session",
    "pre_release_last_cftc_report_date", "pre_release_last_spot_common_reference_date")


def local_day(timestamp):
    return datetime.fromisoformat(timestamp).astimezone(EASTERN).date().isoformat()


def terminal_contributions(*, holdout=False):
    physical = "holdout_optical" if holdout else PHYSICAL
    area = one_per_key(read_csv(f"{physical}_terminal_weeks"), "week_ending", "terminal_id")
    pairs = group((r for r in read_csv(f"{physical}_tank_weeks") if r["baseline_b04_resolved"] == "True"),
                  "week_ending", "terminal_id")
    legal = one_per_key(read_csv("eia_event_operator_exposure"), "release_date", *IDENTITY_FIELDS)
    transfers = group((r for r in read_csv("eia_market_transaction_window") if r["terminal_id"]),
                      "release_date", "instrument", "terminal_id")
    site = {}
    evidence_date = None
    if holdout:
        with closing(read_only(SOURCE)) as db:
            evidence_date = max(r["source_date"] for r in little_farm_parent_evidence(db))
    else:
        for stem in ("operator_development_terminal_exposures", "study_validation_terminal_inputs"):
            site.update(one_per_key((r for r in read_csv(stem) if r["optical_site_scope_id"]), "week_ending", "terminal_id", "ticker"))
    seen_optical = set()
    for (week, terminal, ticker), roles in group(read_csv("study_terminal_input_coverage"), "week_ending", "terminal_id", "ticker").items():
        row = roles[0]
        phase, issue, cutoff, decision = (row[k] for k in ("study_phase", "release_date", "inputs_available_before_utc", "decision_at_utc"))
        if holdout and phase!="holdout":
            continue
        optical = area.get((week, terminal), {})
        samples = pairs[week, terminal]
        seen_optical.add((week, terminal))
        if optical and (optical["study_phase"] != phase or optical["release_date"] != issue
                        or datetime.fromisoformat(optical["cutoff_utc"]) != datetime.fromisoformat(cutoff)):
            raise ValueError(f"Physical and equity calendars differ for {week}/{terminal}")
        if not holdout and phase == "holdout" and (samples or optical.get("capture_change_kbbl")):
            raise ValueError("Closed-period optical outcomes must be absent")
        if (holdout or phase in OPEN_PHASES) and len(samples) != int(optical.get("baseline_b04_tanks", 0)):
            raise ValueError(f"Resolved tank count disagrees for {week}/{terminal}")
        known_at = max((r["archive_available_at_utc"] for r in samples), key=datetime.fromisoformat, default=None)
        if known_at and not datetime.fromisoformat(known_at) < datetime.fromisoformat(cutoff) <= datetime.fromisoformat(decision):
            raise ValueError(f"A physical source arrives outside the cutoff for {week}/{terminal}")
        first_capture = min((r["earlier_capture_at_utc"] for r in samples), key=datetime.fromisoformat, default=None)
        transactions = transfers[issue, ticker, terminal]
        pending_publication = {r["transaction_id"] for r in transactions if r["legal_completion_date"]
            and r["legal_completion_date"] <= issue and r["completion_known_at_cutoff"] == "0"}
        role_details = []
        for rel in roles:
            record = legal.get((issue, *(rel[field] for field in IDENTITY_FIELDS)), {})
            if rel["role"] and not record:
                raise ValueError(f"Cannot trace role to the legal history for {issue}/{terminal}/{ticker}")
            role_details.append({**{field: rel[field] for field in EXPOSURE_FIELDS}, "interest_pct": record.get("interest_pct")})
        asset_roles = [r for r in roles if r["role"] and r["service_agreement_only"] != "True"]
        # Start bounds need to precede the first capture, including conservative month bounds.
        first_day = local_day(first_capture) if first_capture else None
        decision_day = local_day(decision)
        inside_dates = (any(r["valid_from"] and r["valid_from"] < first_day[:len(r["valid_from"])]
                            and (not r["valid_to_exclusive"] or decision_day[:len(r["valid_to_exclusive"])] < r["valid_to_exclusive"])
                            for r in asset_roles) if first_day and asset_roles else None)
        flags = []
        for rel in asset_roles:
            pending_publication.update(filter(None, rel["legal_transfer_not_public_at_cutoff"].split("|")))
            if rel["parent_equity_exposure_known_at_cutoff"] == "0":
                flags.append("parent_interest_unconfirmed_at_cutoff")
            if rel["deeprock_asset_period_known_at_cutoff"] == "0":
                flags.append("deeprock_asset_period_unconfirmed_at_cutoff")
            if rel["deeprock_merged_scope_confirmed_for_pair_window"] == "0":
                flags.append("deeprock_scope_not_confirmed_for_pair")
        if pending_publication:
            flags.append("legal_completion_unconfirmed_at_cutoff")
        little_farm = site.get((week, terminal, ticker), {})
        if holdout and terminal=="energy_transfer_cushing" and ticker=="ET" and samples:
            little_farm = {"optical_site_operator_parent_evidence_date":evidence_date,
                           "optical_site_operator_parent_known_before_cutoff":str(evidence_date<cutoff)}
        if little_farm.get("optical_site_operator_parent_known_before_cutoff") == "False":
            flags.append("little_farm_parent_unconfirmed_at_cutoff")
        yield dict(study_phase=phase, week_ending=week, release_date=issue, decision_at_utc=decision, optical_cutoff_utc=cutoff,
            latest_week_for_market_issue=row["latest_week_for_market_issue"] == "1", terminal_id=terminal, instrument=ticker,
            terminal_in_dated_optical_registry=bool(optical), has_asset_relationship=bool(asset_roles),
            service_only=any(r["service_agreement_only"] == "True" for r in roles) and not asset_roles,
            exposure_roles_json=json_values(role_details),
            capture_pair_inside_documented_asset_dates=inside_dates, known_publication_or_scope_flags="|".join(sorted(set(flags))),
            completion_unconfirmed_transaction_ids="|".join(sorted(pending_publication)),
            little_farm_operator_parent_evidence_date=little_farm.get("optical_site_operator_parent_evidence_date"),
            little_farm_operator_parent_known_before_cutoff=little_farm.get("optical_site_operator_parent_known_before_cutoff"),
            first_paired_capture_at_utc=first_capture,
            last_paired_capture_at_utc=max((r["later_capture_at_utc"] for r in samples), key=datetime.fromisoformat, default=None),
            last_paired_source_available_at_utc=known_at,
            **{k: numeric(optical.get(k, "")) if optical else (None if not holdout and phase == "holdout" and k.startswith("baseline_") else 0)
               for k in AREA_FIELDS},
            **{k: numeric(optical.get(k, "")) for k in ("capture_change_kbbl", "overlap_change_kbbl", "baseline_b04_space_time_fraction")},
            measurement_sources="|".join(sorted({r["measurement_source"] for r in samples})))
    if set(area) - seen_optical:
        raise ValueError("Dated terminal panel has rows missing from the legal roster")


def hedge_history(through=OPEN_LAST_SESSION, *, with_prices=True):
    with closing(read_only(SOURCE)) as db:
        closes = dict(db.execute("""SELECT session_date,close_utc FROM equity_session_calendar
            WHERE venue=? AND session_date<=? ORDER BY session_date""", (VENUE, through)))
        quotes = {(symbol, day): (price, segment) for symbol, day, price, segment in db.execute("""
            SELECT instrument,session_date,CASE WHEN ? THEN adjusted_ibkr_close END,listing_segment FROM market_study_quote
            WHERE venue=? AND currency='USD' AND session_date<=? AND adjusted_ibkr_close>0
        """, (with_prices,VENUE,through))}
    return list(closes), {day: index for index, day in enumerate(closes)}, closes, quotes


def trailing_beta(row, history, length, *, through=OPEN_LAST_SESSION):
    prefix = f"amlp_beta_{length}d"
    value, first, last, last_utc = None, None, None, None
    if history is None:
        status = "closed_period_coverage_only"
    elif row["venue"] != VENUE or row["currency"] != "USD":
        status = "requires_currency_and_clock_aligned_return_series"
    elif row["last_scheduled_session_before_decision"] > through:
        status = "decision_history_extends_after_open_quote_cutoff"
    elif int(row["contiguous_paired_prior_ibkr_adjusted_close_sessions"]) <= length:
        status = "insufficient_contiguous_ibkr_adjusted_closes"
    else:
        sessions, index, closes, quotes = history
        end = index[row["last_scheduled_session_before_decision"]]
        days = sessions[end-length:end+1]
        if len(days) != length+1 or datetime.fromisoformat(closes[days[-1]]) >= datetime.fromisoformat(row["decision_at_utc"]):
            raise ValueError(f"Incomplete claimed adjusted quote window for {row['instrument']}/{row['release_date']}")
        returns = []
        for ticker in (row["instrument"], BENCHMARK):
            prices = [quotes[ticker, day] for day in days]
            if len({segment for _, segment in prices}) != 1 or any(not math.isfinite(p) or p <= 0 for p, _ in prices):
                raise ValueError(f"Noncontinuous adjusted prices for {ticker}/{row['release_date']}")
            returns.append([after[0]/before[0]-1 for before, after in zip(prices, prices[1:])])
        scale = variance(returns[1])
        status = "observed_ibkr_adjusted_close_returns" if scale > 0 else "zero_amlp_return_variance"
        value = covariance(*returns)/scale if scale > 0 else None
        first, last = days[0], days[-1]
        last_utc = closes[last]
    return {prefix: value, f"{prefix}_status": status, f"{prefix}_first_close": first, f"{prefix}_last_close": last,
            f"{prefix}_last_close_utc": last_utc}


def exposure_aggregate(rows, measured_phase, prefix):
    measured = [r for r in rows if r["capture_change_kbbl"] is not None]
    dated = [r for r in measured if r["capture_pair_inside_documented_asset_dates"] and not r["known_publication_or_scope_flags"]]
    counts = {f"{prefix}_{field}": (sum(r[field] for r in rows) if measured_phase or not field.startswith("baseline_") else None)
              for field in AREA_FIELDS}
    area = counts[f"{prefix}_dated_reference_area_m2"]
    return {**counts, f"{prefix}_linked_terminals": len(rows), f"{prefix}_terminal_ids": "|".join(r["terminal_id"] for r in rows),
        f"{prefix}_measured_terminals": len(measured) if measured_phase else None,
        f"{prefix}_measured_terminal_ids": "|".join(r["terminal_id"] for r in measured),
        f"{prefix}_input_area_fraction": counts[f"{prefix}_eligible_input_area_m2"]/area if area else None,
        f"{prefix}_observed_area_fraction": counts[f"{prefix}_baseline_b04_area_m2"]/area if area and measured_phase else None,
        f"{prefix}_space_time_fraction": sum(r["baseline_b04_space_time_fraction"]*r["dated_reference_area_m2"] for r in measured)/area
        if area and measured_phase else None,
        f"{prefix}_capture_change_kbbl": math.fsum(r["capture_change_kbbl"] for r in measured) if measured else None,
        f"{prefix}_overlap_change_kbbl": math.fsum(r["overlap_change_kbbl"] for r in measured) if measured else None,
        f"{prefix}_change_with_documented_pair_dates_kbbl": math.fsum(r["capture_change_kbbl"] for r in dated) if dated else None,
        f"{prefix}_measured_terminals_with_documented_pair_dates": len(dated) if measured_phase else None}


def build(terminal_rows, *, holdout=False):
    issues = one_per_key((r for r in read_csv("study_week_input_coverage") if r["latest_week_for_market_issue"] == "1"), "release_date")
    costs = one_per_key(read_csv("study_execution_inputs"), "release_date", "instrument")
    actions = group(read_csv("study_equity_window_actions"), "release_date", "instrument")
    terminals = group((r for r in terminal_rows if r["latest_week_for_market_issue"] and r["instrument"]), "release_date", "instrument")
    history = None if holdout else hedge_history()
    for row in read_csv("study_instrument_window_coverage"):
        issue, ticker, phase = (row[k] for k in ("release_date", "instrument", "study_phase"))
        if holdout and phase!="holdout":
            continue
        w = issues[issue,]
        if (w["study_phase"], w["week_ending"], w["decision_at_utc"]) != (phase, row["latest_first_released_week"], row["decision_at_utc"]):
            raise ValueError(f"Latest physical and market issue do not match for {issue}/{ticker}")
        decision = datetime.fromisoformat(w["decision_at_utc"])
        if not datetime.fromisoformat(w["inputs_available_before_utc"]) <= decision <= datetime.fromisoformat(row["entry_open_utc"]):
            raise ValueError(f"Optical cutoff, EIA decision and entry are out of order for {issue}/{ticker}")
        cost = costs.get((issue, ticker), {})
        phase_eligible = row["calendar_status"] in ELIGIBLE_WINDOWS
        if cost and (cost["study_phase"], cost["week_ending"], cost["entry_session"], cost["fifth_session"], cost["phase_window_eligible"]) != (
                phase, w["week_ending"], row["entry_session"], row["fifth_session"], str(phase_eligible)):
            raise ValueError(f"Cost inputs refer to a different window for {issue}/{ticker}")
        related = sorted(terminals[issue, ticker], key=lambda r: r["terminal_id"])
        measured_phase = holdout or phase in OPEN_PHASES
        asset = exposure_aggregate([r for r in related if r["has_asset_relationship"]], measured_phase, "asset")
        service = exposure_aggregate([r for r in related if r["service_only"]], measured_phase, "service")
        change = asset["asset_capture_change_kbbl"]
        direction = ("long_instrument_short_amlp" if change > 0 else "short_instrument_long_amlp" if change < 0 else "flat") if change is not None else None
        short_leg = {"long_instrument_short_amlp": "amlp", "short_instrument_long_amlp": "instrument"}.get(direction)
        yield dict(study_phase=phase, week_ending=w["week_ending"], release_date=issue, instrument=ticker,
            study_role=row["study_role"], venue=row["venue"], currency=row["currency"], benchmark=BENCHMARK,
            optical_cutoff_utc=w["inputs_available_before_utc"], decision_at_utc=w["decision_at_utc"], phase_window_eligible=phase_eligible,
            **{k: row[k] for k in MARKET_FIELDS}, **{k: w[k] for k in CONTEXT_FIELDS}, **asset, **service,
            instrument_window_corporate_event_ids="|".join(sorted({a["event_id"] for a in actions[issue, ticker]})),
            amlp_window_corporate_event_ids="|".join(sorted({a["event_id"] for a in actions[issue, BENCHMARK]})),
            candidate_raw_direction=direction, execution_cost_input_row_available=bool(cost),
            **{k: cost.get(k) if phase in OPEN_PHASES else None for k in COST_SPREADS},
            **{k: cost.get(k) for k in COST_FIELDS},
            candidate_direction_cost_inputs_observed=cost.get(f"observed_cost_inputs_short_{short_leg}") if short_leg else None,
            **{k: v for length in BETA_SESSIONS for k, v in trailing_beta(
                row, history if phase in OPEN_PHASES else None, length).items()})


def coverage(rows):
    measured = [r for r in rows if r["asset_capture_change_kbbl"] is not None]
    priced = [r for r in measured if r["phase_window_eligible"] and r["matched_usd_quote_window"] == "True"]
    return dict(windows=len(rows), issues=len({r["release_date"] for r in rows}),
        windows_with_linked_asset_inputs=sum(r["asset_eligible_input_tanks"] > 0 for r in rows),
        windows_with_exterior_checked_asset_inputs=sum(r["asset_exterior_checked_tanks"] > 0 for r in rows),
        measured_asset_windows=len(measured), priced_phase_eligible_measured_asset_windows=len(priced),
        issues_with_priced_phase_eligible_measured_asset_windows=len({r["release_date"] for r in priced}),
        priced_phase_eligible_measured_with_documented_asset_pair_dates=sum(
            r["asset_change_with_documented_pair_dates_kbbl"] is not None for r in priced),
        priced_phase_eligible_measured_with_observed_directional_costs=sum(
            r["candidate_direction_cost_inputs_observed"] == "True" for r in priced),
        issues_with_priced_phase_eligible_measured_asset_and_directional_costs=len({r["release_date"] for r in priced
            if r["candidate_direction_cost_inputs_observed"] == "True"}),
        priced_phase_eligible_measured_with_observed_costs_and_documented_asset_pair_dates=sum(
            r["candidate_direction_cost_inputs_observed"] == "True" and r["asset_change_with_documented_pair_dates_kbbl"] is not None
            for r in priced), **{f"windows_with_{length}d_amlp_beta": sum(r[f"amlp_beta_{length}d"] is not None for r in rows)
                                  for length in BETA_SESSIONS})


def main():
    terminal_rows = list(terminal_contributions())
    rows = list(build(terminal_rows))
    windows = one_per_key(rows, "release_date", "instrument")
    if any(r["instrument"] and (r["release_date"], r["instrument"]) not in windows for r in terminal_rows):
        raise ValueError("Some terminal/equity relations lack a candidate market window")
    core = [r for r in rows if r["instrument"] in CORE]
    report = dict(windows=len(rows), terminal_relationship_weeks=len(terminal_rows), instruments=sorted({r["instrument"] for r in rows}),
        core_instruments=sorted(CORE), core_by_phase={k[0]: coverage(v) for k, v in group(core, "study_phase").items()},
        core_by_instrument_phase={f"{k[0]}:{k[1]}": coverage(v) for k, v in group(core, "study_phase", "instrument").items()},
        interpretation="Each terminal enters an instrument sum once. Asset-linked volumes are physical scope sums in thousand barrels. "
        "They are unweighted by economic stakes and exclude service-only roles. Area denominators cover the dated roof registry. "
        "The documented-pair-date field screens legal bounds and known disclosure or asset-scope gaps. Other registry links may still "
        "require contemporaneous publication evidence. IBKR adjusted daily close returns are used for descriptive 60- and 120-session "
        "hedge coefficients on the open periods, with the final close strictly before the issue decision. Spread and borrow coverage "
        "identifies observation inputs, not executable historical locates. The closed period contains source availability and market "
        "coverage only. No EIA target values, forward returns or P&L are computed.")
    export(TERMINALS, terminal_rows)
    export(STEM, rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2)+"\n").encode())
    print(json.dumps({"windows": len(rows), "terminal_rows": len(terminal_rows), "core_by_phase": report["core_by_phase"]}, indent=2))


if __name__ == "__main__":
    main()
