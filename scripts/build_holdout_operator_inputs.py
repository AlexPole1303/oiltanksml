"""Join holdout physical measurements to as-of operator and hedge inputs."""
import json
from collections import Counter
from contextlib import closing

from audit_terminal_change_repeatability import historical_terminal_measurements, operator_envelopes, registry, terminal_envelopes
from backtest_cushing_operator_strategy import cash_in_holding, corporate_inputs
from build_holdout_equity_price_bridges import effective_basis, load_bridges
from build_holdout_optical_inputs import PHASE
from build_market_event_panel import CORE
from build_operator_development_inputs import one_per_key
from build_study_operator_signal_inputs import BENCHMARK, VENUE, build, terminal_contributions
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

STEM = "cushing_holdout_operator"
VOLUME_FILES = ("sentinel2_holdout_weekly_tank_volumes", "sentinel2_alternate_registration_holdout_added_volumes",
                "hcs_dated_shadow_holdout_volumes")


def accounting_coverage(rows):
    corporate = corporate_inputs((PHASE,))
    bridges = load_bridges()
    with closing(read_only(SOURCE)) as db:
        sessions = [d for d, in db.execute("SELECT session_date FROM equity_session_calendar WHERE venue=? ORDER BY session_date",(VENUE,))]
        quote_identity = {(i,d):(effective_basis(bridges.get((i,d)),provider,reference,basis),segment)
            for i,d,basis,segment,provider,reference in db.execute("""SELECT instrument,session_date,
            price_basis,listing_segment,provider,source_file FROM market_study_quote
            WHERE venue=? AND currency='USD' AND open>0 AND close>0""",(VENUE,))}
    index = {d:i for i,d in enumerate(sessions)}
    for r in rows:
        issues,proof = [],[]
        counts = {}
        if r["phase_window_eligible"] and r["matched_usd_quote_window"]=="True":
            issue,entry,end = (r[k] for k in ("release_date","entry_session","fifth_session"))
            window = sessions[index[entry]:index[end]+1]
            if len(window)!=5:
                raise ValueError(f"Incomplete equity session calendar for {issue}/{r['instrument']}")
            for side,symbol in (("instrument",r["instrument"]),("benchmark",BENCHMARK)):
                proof.extend(f"{symbol}/{day}" for day in window if (symbol,day) in bridges)
                quoted = {quote_identity.get((symbol,day)) for day in window}
                if None in quoted:
                    issues.append(f"{side}:missing_valid_quotes")
                    continue
                if len(quoted)!=1:
                    issues.append(f"{side}:quote_basis_changes_within_window")
                    continue
                basis,_ = next(iter(quoted))
                if basis!="ibkr_trades_split_adjusted":
                    issues.append(f"{side}:unverified_accounting_price_basis")
                    continue
                cash, reason = cash_in_holding(issue,symbol,entry,end,basis,corporate)
                if reason:
                    issues.append(f"{side}:{reason}")
                else:
                    counts[side] = len(cash)
        else:
            issues.append("no_paired_usd_price_window")
        yield dict(r,accounting_inputs_complete=not issues,accounting_input_gaps="|".join(issues),
                   accounting_quote_bridge_keys="|".join(proof),
                   instrument_cash_events_in_holding=counts.get("instrument"),benchmark_cash_events_in_holding=counts.get("benchmark"))


def classify(row):
    if row["asset_capture_change_kbbl"] is None:
        return "no_asset_linked_measurement"
    if not row["phase_window_eligible"] or row["matched_usd_quote_window"]!="True":
        return "no_paired_usd_price_window"
    if row["asset_measured_terminals"]!=row["asset_measured_terminals_with_documented_pair_dates"]:
        return "asset_dates_or_scope_unconfirmed"
    if row["terminal_relation_boundary_in_window"]=="1":
        return "terminal_relationship_changes_within_window"
    if not row["accounting_inputs_complete"]:
        return "accounting_inputs_incomplete"
    return "ready_for_replay"


def main():
    contributions = list(terminal_contributions(holdout=True))
    rows = list(accounting_coverage(build(contributions,holdout=True)))
    inputs = one_per_key(rows,"release_date","instrument")
    if any(r["instrument"] and (r["release_date"],r["instrument"]) not in inputs for r in contributions):
        raise ValueError("Holdout terminal link has no equity calendar entry")
    history = historical_terminal_measurements(registry(),VOLUME_FILES,(PHASE,))
    terminals = list(terminal_envelopes(history,"holdout_optical"))
    volumes = list(operator_envelopes(history,rows,(PHASE,)))
    envelopes = one_per_key(volumes,"release_date","instrument")
    output = []
    for key,r in inputs.items():
        physical = envelopes.get(key,{})
        if (r["asset_capture_change_kbbl"] is not None)!=bool(physical):
            raise ValueError(f"Unreconciled physical input on {key}")
        output.append(dict(r,replay_input_status=classify(r),
            **{k:physical.get(k) for k in ("loss_min_kbbl","loss_max_kbbl","loss_envelope_excludes_zero")}))
    core = [r for r in output if r["instrument"] in CORE]
    ready = [r for r in core if r["replay_input_status"]=="ready_for_replay"]
    report = dict(windows=len(rows),core_windows=len(core),report_weeks=len({r["week_ending"] for r in core}),
        core_windows_with_equity_price_bridges=sum(bool(r["accounting_quote_bridge_keys"]) for r in core),
        core_measured_windows=sum(r["asset_capture_change_kbbl"] is not None for r in core),
        core_input_status=dict(Counter(r["replay_input_status"] for r in core)),
        core_ready_weeks=len({r["week_ending"] for r in ready}),core_ready_by_instrument=dict(Counter(r["instrument"] for r in ready)),
        core_ready_with_observed_directional_costs=sum(r["candidate_direction_cost_inputs_observed"]=="True" for r in ready),
        core_ready_with_loss_envelope_excluding_zero=sum(r["loss_envelope_excludes_zero"] for r in ready),
        interpretation="Coverage uses dated physical asset links and the same accounting basis as the operator replay. "
        "The directional cost flag counts observed inputs. Missing spreads or borrow rates require the registered scenarios. "
        "The physical loss envelope describes fit ambiguity. No EIA target, quote values, returns or P&L are exported.")
    for suffix,records in (("inputs",output),("terminal_links",contributions),("terminal_envelopes",terminals)):
        export(f"{STEM}_{suffix}",records)
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
