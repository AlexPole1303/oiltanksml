"""Compare optical operator selection with uniform selection on the same windows."""

import json
import math
from collections import defaultdict
from statistics import mean

import backtest_cushing_operator_strategy as operator
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from replay_cushing_operator_holdout import PHASE, market_input, read_inputs, settlement_calendar

STEM = "cushing_operator_selection"
PNL = ("gross_pnl_usd", "net_pnl_usd", "stressed_net_pnl_usd")


def frozen_trades():
    ledger = {}
    for name in ("cushing_operator_backtest_trades", "cushing_holdout_operator_replay_trades"):
        for row in read(name):
            if row["strategy"] == "always_long_spread":
                key = row["study_phase"], row["release_date"]
                if key in ledger or row["instrument_side"] != "1":
                    raise ValueError(f"Unexpected frozen trade on {key}")
                ledger[key] = row
    return ledger


def comparisons(ledger):
    open_issues, _ = operator.event_inputs()
    _, _, holdout, public = read_inputs()
    through = max(row["fifth_session"] for sample in public.values() for row, _ in sample)
    holdout_market, _ = market_input(through, public, evaluate=True)
    open_market = operator.market_sources()
    holdout_clearing, _ = settlement_calendar(holdout_market[0])
    fees = operator.execution_sources()
    corporate = operator.corporate_inputs((*operator.PHASES, PHASE))
    terminal_signals = {(r["release_date"],r["week_ending"],r["terminal_id"]):r for r in read("cushing_weekly_change_ml_terminals")}
    for (phase, issue), frozen in sorted(ledger.items(), key=lambda item: item[0][1]):
        if phase == PHASE:
            inputs = [operator.model_input(row, cost, row, 0., 0., None) for row, cost in holdout[issue]]
            market, clearing = holdout_market, holdout_clearing
        else:
            inputs = open_issues[issue]
            market, clearing = open_market, None
        eligible = {}
        for row in inputs:
            legs, reason = operator.price_window(row, market, corporate)
            if not reason:
                eligible[row["instrument"]] = (row, operator.outcome(dict(row, legs=legs), 1, market, fees,
                                                                     settlement_days=clearing))
        symbol = min(eligible, key=lambda ticker: (-abs(eligible[ticker][0]["physical_height_change_m"]), ticker))
        if symbol != frozen["instrument"]:
            raise ValueError(f"Frozen satellite selection differs on {issue}")
        selected = eligible[symbol][1]
        if any(not math.isclose(selected[k], float(frozen[k]), abs_tol=1e-8) for k in PNL):
            raise ValueError(f"Frozen P&L differs on {issue}")
        for row, _ in eligible.values():
            if any(row[k] != frozen[k] for k in ("entry_session", "fifth_session", "decision_at_utc")):
                raise ValueError(f"Comparison does not share the frozen trading window on {issue}")
        height_scores = {}
        for ticker, (row, _) in eligible.items():
            sites = row["asset_measured_terminal_ids"].split("|")
            signals = [terminal_signals[issue,row["week_ending"],site] for site in sites]
            area = math.fsum(float(r["area_m2"]) for r in signals)
            total = math.fsum(float(r["physical"]) for r in signals)
            if (not math.isclose(area,float(row["asset_baseline_b04_area_m2"]),abs_tol=1e-6)
                    or not math.isclose(total,float(row["asset_capture_change_kbbl"]),abs_tol=1e-8)):
                raise ValueError(f"Physical operator input differs from the terminal ledger on {issue}/{ticker}")
            height_scores[ticker] = abs(math.fsum(float(r["attenuated"]) for r in signals)/area)
        attenuated_symbol = min(eligible,key=lambda ticker:(-height_scores[ticker],ticker))
        attenuated = eligible[attenuated_symbol][1]
        uniform = {k: mean(result[k] for _, result in eligible.values()) for k in PNL}
        public_outcomes = {}
        if phase == PHASE:
            for row, cost in public[issue]:
                ticker = row["instrument"]
                if any(row[k] != frozen[k] for k in ("entry_session", "fifth_session", "decision_at_utc")):
                    raise ValueError(f"Public control does not share the frozen trading window on {issue}")
                legs, reason = operator.price_window(row, market, corporate)
                if reason:
                    raise ValueError(f"Unusable public control on {issue}/{ticker}: {reason}")
                public_outcomes[ticker] = operator.outcome(dict(row, legs=legs, cost=cost), 1, market, fees,
                                                            settlement_days=clearing)
            if not eligible.keys() <= public_outcomes.keys() or any(
                not math.isclose(result[k], public_outcomes[ticker][k], abs_tol=1e-8)
                for ticker, (_, result) in eligible.items() for k in PNL
            ):
                raise ValueError(f"Measured and public accounting disagree on {issue}")
        uniform_public = {k: mean(result[k] for result in public_outcomes.values()) if public_outcomes else None for k in PNL}
        yield dict(study_phase=phase, release_date=issue, entry_session=frozen["entry_session"],
                   exit_session=frozen["fifth_session"], selected_operator=symbol,
                   attenuated_operator=attenuated_symbol,
                   eligible_operators="|".join(sorted(eligible)), eligible_operator_count=len(eligible),
                   public_eligible_operators="|".join(sorted(public_outcomes)), public_eligible_count=len(public_outcomes),
                   **{f"{prefix}_{k}": pnl[k] for prefix, pnl in (
                       ("selected", selected), ("attenuated", attenuated), ("uniform", uniform), ("uniform_public", uniform_public)) for k in PNL})


def metrics(rows):
    totals = {f"{prefix}_{k}": math.fsum(row[f"{prefix}_{k}"] for row in rows)
              for prefix in ("selected", "uniform") for k in PNL}
    deltas = [row["selected_net_pnl_usd"] - row["uniform_net_pnl_usd"] for row in rows]
    variable_cost = totals["selected_net_pnl_usd"] - totals["selected_stressed_net_pnl_usd"]
    net_before_variable_cost = totals["selected_net_pnl_usd"] + variable_cost
    return dict(windows=len(rows), windows_with_choice=sum(row["eligible_operator_count"] > 1 for row in rows), **totals,
                selected_minus_uniform_net_usd=math.fsum(deltas),
                selected_minus_uniform_mean_net_95pct_block_bootstrap_usd=block_bootstrap_mean(deltas),
                selected_break_even_variable_cost_multiplier=net_before_variable_cost/variable_cost
                if variable_cost > 0 and net_before_variable_cost >= 0 else None)


def holdout_sensitivity(rows):
    by_year, by_operator = defaultdict(list), defaultdict(list)
    for row in rows:
        by_year[row["release_date"][:4]].append(row)
        by_operator[row["selected_operator"]].append(row)
    profits = sorted((row for row in rows if row["selected_net_pnl_usd"] > 0),
                     key=lambda row: (-row["selected_net_pnl_usd"], row["release_date"]))
    best_three = profits[:3]
    return dict(
        annual={year: metrics(sample) for year, sample in sorted(by_year.items())},
        operators={ticker: dict(windows=len(sample),
                               net_usd=math.fsum(r["selected_net_pnl_usd"] for r in sample),
                               stressed_net_usd=math.fsum(r["selected_stressed_net_pnl_usd"] for r in sample))
                   for ticker, sample in sorted(by_operator.items())},
        three_largest_winners=[dict(release_date=r["release_date"], operator=r["selected_operator"],
                                   net_usd=r["selected_net_pnl_usd"]) for r in best_three],
        net_without_three_largest_winners_usd=math.fsum(r["selected_net_pnl_usd"] for r in rows if r not in best_three),
    )


def public_control(rows):
    totals = {k: math.fsum(r[f"uniform_public_{k}"] for r in rows) for k in PNL}
    selection = [r["selected_net_pnl_usd"] - r["uniform_public_net_pnl_usd"] for r in rows]
    eligibility = [r["uniform_net_pnl_usd"] - r["uniform_public_net_pnl_usd"] for r in rows]
    return dict(windows=len(rows), method="Uniform expected P&L across all publicly eligible operators on the same measured dates, "
                "with the same hedge, accounting and entry and exit times.", **totals,
                measured_selection_minus_public_net_usd=math.fsum(selection),
                measured_selection_minus_public_mean_net_95pct_block_bootstrap_usd=block_bootstrap_mean(selection),
                measured_eligibility_minus_public_net_usd=math.fsum(eligibility),
                measured_eligibility_minus_public_mean_net_95pct_block_bootstrap_usd=block_bootstrap_mean(eligibility))


def attenuated_control(rows):
    totals = {k: math.fsum(r[f"attenuated_{k}"] for r in rows) for k in PNL}
    delta = [r["attenuated_net_pnl_usd"]-r["selected_net_pnl_usd"] for r in rows]
    return dict(windows=len(rows), changed_operator=sum(r["attenuated_operator"]!=r["selected_operator"] for r in rows),
                **totals, delta_net_usd=math.fsum(delta), delta_mean_net_95pct_block_bootstrap_usd=block_bootstrap_mean(delta))


def main():
    rows = list(comparisons(frozen_trades()))
    by_phase = defaultdict(list)
    for row in rows:
        by_phase[row["study_phase"]].append(row)
    results = dict(status="exploratory_comparison_after_holdout_evaluation",
                   comparison="Uniform expected P&L for choosing one operator from the same satellite-eligible set. "
                   "The calendar, direction, gross notional and short AMLP leg match the frozen strategy.",
                   periods={phase: metrics(by_phase[phase]) for phase in (*operator.PHASES, PHASE)},
                   attenuated_selection={phase:attenuated_control(by_phase[phase]) for phase in (*operator.PHASES, PHASE)},
                   holdout_public_control=public_control(by_phase[PHASE]),
                   holdout_sensitivity=holdout_sensitivity(by_phase[PHASE]))
    export(f"{STEM}_windows", rows)
    payload = json.dumps(results, indent=2, allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
