"""Join observed spread and borrow inputs to dated operator-versus-AMLP windows."""
import json
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta

from build_market_event_panel import CORE
from build_operator_development_inputs import group, json_values, one_per_key
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

STEM = "study_execution_inputs"
BENCHMARK = "AMLP"
VENUE = "XNYS"


def event_sessions():
    with closing(read_only(SOURCE)) as db:
        sessions = defaultdict(list)
        for issue, ordinal, day, opening, closing_at in db.execute("""
            SELECT release_date,ordinal,session_date,open_utc,close_utc FROM eia_event_market_session
            WHERE venue=? AND ordinal BETWEEN 1 AND 5 ORDER BY release_date,ordinal
        """, (VENUE,)):
            if ordinal != len(sessions[issue])+1:
                raise ValueError(f"Missing intermediate exchange session for {issue}")
            sessions[issue].append((day, opening, closing_at))
        instruments = {symbol: (venue, currency) for symbol, venue, currency in db.execute(
            "SELECT instrument,venue,currency FROM market_study_instrument")}
    if any(instruments[symbol] != (VENUE, "USD") for symbol in (*CORE, BENCHMARK)):
        raise ValueError("The operator and hedge trade on different configured calendars or currencies")
    return sessions


def dated_fee(rows, day):
    candidates = [r for r in rows if r["effective_from"] <= day <= r["verified_through_date"]
                  and (not r["effective_to_exclusive"] or day < r["effective_to_exclusive"])]
    if len(candidates) != 1 or candidates[0]["date_basis"] != "trade_date":
        raise ValueError(f"No unique trade-date regulatory fee for {day}")
    return candidates[0]


def spread_fields(row, legs, spreads):
    fields = {}
    for leg, symbol in legs:
        for edge in ("entry", "exit"):
            prefix = f"{leg}_{edge}"
            status = row[f"{prefix}_spread_status"]
            stamp = row[f"{edge}_spread_bar_start_utc"]
            quote = spreads.get((symbol, stamp))
            if (status == "observed") != bool(quote):
                raise ValueError(f"Spread observation contradicts its audit for {symbol}/{stamp}")
            if quote and (quote["currency"] != "USD" or not 0 <= float(quote["spread_bps"]) < 20_000):
                raise ValueError(f"Invalid indicative spread for {symbol}/{stamp}")
            fields.update({f"{prefix}_spread_status": status,
                           f"{prefix}_spread_bps": quote["spread_bps"] if quote else None,
                           f"{prefix}_spread_source": quote["source_file"] if quote else None})
    return fields


def borrow_fields(row, legs, sessions, rates):
    fields = {}
    fields_kept = ("fee_rate_open_native", "fee_rate_high_native", "fee_rate_low_native",
                   "fee_rate_close_native", "raw_file", "conid")
    for leg, symbol in legs:
        observations = []
        for day, _, _ in sessions:
            rate = rates.get((symbol, day))
            status = rate["identity_status"] if rate else "rate_missing"
            if rate and rate["currency"] != "USD":
                raise ValueError(f"Borrow currency differs for {symbol}/{day}")
            observations.append(dict(session_date=day, status=status,
                **{field: rate[field] for field in fields_kept} if rate and status == "within_broker_series_identity" else {}))
        complete = len(observations) == 5 and all(
            r["status"] == "within_broker_series_identity" for r in observations)
        if complete != (row[f"{leg}_five_borrow_rate_sessions"] == "True"):
            raise ValueError(f"Borrow coverage disagrees for {symbol}/{row['release_date']}")
        fields.update({f"{leg}_five_borrow_rates": complete,
                       f"{leg}_borrow_sessions_json": json_values(observations)})
    return fields


def build():
    calendar = one_per_key((r for r in read_csv("eia_phase_market_calendar") if r["calendar_venue"] == VENUE), "release_date")
    sessions = event_sessions()
    commission = {r["snapshot_date"] for r in read_csv("ibkr_equity_commission_snapshot")
                  if (r["listing_market"], r["currency"]) == ("US", "USD")}
    if len(commission) != 1:
        raise ValueError("US commission scenario requires a dated source snapshot")
    commission_date = next(iter(commission))
    taf = group(read_csv("us_equity_regulatory_fee_period"), "fee")["FINRA_TAF",]
    spreads = one_per_key(read_csv("tws_event_spread_observations"), "instrument", "bar_start_utc")
    rates = one_per_key((r for r in read_csv("tws_borrow_rate_daily") if r["instrument"] in (*CORE, BENCHMARK)),
                        "instrument", "session_date")
    rows = (r for r in read_csv("tws_event_execution_window_coverage") if r["instrument"] in CORE)
    for row in rows:
        issue, symbol = row["release_date"], row["instrument"]
        clock = calendar[issue,]
        schedule = sessions[issue]
        opening = datetime.fromisoformat(clock["entry_open_utc"])
        closing_at = datetime.fromisoformat(clock["fifth_close_utc"])
        if (len(schedule) != 5 or schedule[0][0] != clock["entry_session"] or schedule[-1][0] != clock["fifth_session"]
                or row["study_phase"] != clock["study_phase"] or row["week_ending"] != clock["week_ending"]
                or datetime.fromisoformat(row["entry_spread_bar_start_utc"]) != opening
                or datetime.fromisoformat(row["exit_spread_bar_start_utc"]) != closing_at-timedelta(minutes=5)):
            raise ValueError(f"Mismatched market window for {symbol}/{issue}")
        legs = (("instrument", symbol), ("amlp", BENCHMARK))
        quotes = spread_fields(row, legs, spreads)
        rates_by_leg = borrow_fields(row, legs, schedule, rates)
        spread_complete = all(quotes[f"{leg}_{edge}_spread_bps"] is not None for leg, _ in legs for edge in ("entry", "exit"))
        if spread_complete != (row["both_legs_have_two_spread_observations"] == "True"):
            raise ValueError(f"Spread completeness disagrees for {symbol}/{issue}")
        entry_fee, exit_fee = (dated_fee(taf, clock[k]) for k in ("entry_session", "fifth_session"))
        yield dict(study_phase=row["study_phase"], week_ending=row["week_ending"], release_date=issue,
            instrument=symbol, benchmark=BENCHMARK, calendar_status=clock["calendar_status"],
            phase_window_eligible=clock["calendar_status"] in ("fifth_close_precedes_next_phase_decision", "final_phase"),
            entry_session=clock["entry_session"], fifth_session=clock["fifth_session"],
            entry_spread_bar_start_utc=row["entry_spread_bar_start_utc"], exit_spread_bar_start_utc=row["exit_spread_bar_start_utc"],
            **quotes, **rates_by_leg, both_legs_have_two_spreads=spread_complete,
            observed_cost_inputs_short_instrument=spread_complete and rates_by_leg["instrument_five_borrow_rates"],
            observed_cost_inputs_short_amlp=spread_complete and rates_by_leg["amlp_five_borrow_rates"],
            finra_entry_rate_per_covered_share_sold=entry_fee["rate"], finra_entry_cap_usd=entry_fee["cap_usd_per_trade"],
            finra_entry_source_url=entry_fee["source_url"], finra_exit_rate_per_covered_share_sold=exit_fee["rate"],
            finra_exit_cap_usd=exit_fee["cap_usd_per_trade"], finra_exit_source_url=exit_fee["source_url"],
            sec_section_31_rate_schedule="us_equity_regulatory_fee_period.csv", sec_section_31_date_basis="sro_charge_date",
            commission_scenario_snapshot_date=commission_date)


def main():
    rows = list(build())
    for issue, records in group(rows, "release_date").items():
        if len(records) != len(CORE) or {row["instrument"] for row in records} != set(CORE):
            raise ValueError(f"The cost input window is missing or duplicating operators for {issue}")
    counts = defaultdict(Counter)
    for row in rows:
        phase = row["study_phase"]
        counts[phase]["windows"] += 1
        if row["phase_window_eligible"]:
            counts[phase]["phase_eligible_windows"] += 1
            for field in ("both_legs_have_two_spreads", "observed_cost_inputs_short_instrument", "observed_cost_inputs_short_amlp"):
                counts[phase][field] += row[field]
    export(STEM, rows)
    report = dict(windows=len(rows), coverage=counts, interpretation="Spreads are indicative bid/ask bar observations. "
        "FEE_RATE entries are native historical broker observations. Only the applicable short leg is required for the corresponding "
        "coverage flag. FINRA terms use trade dates. Section 31 uses the applicable SRO charge date. The commission snapshot "
        "identifies a scenario input. No executions, prices or P&L are calculated.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
