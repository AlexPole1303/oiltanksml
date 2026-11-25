"""Replay the updated Cushing forecast on the paper's dated NYMEX CL spreads."""

import csv
import json
import math
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path

from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import tick_price
from build_cushing_holdout_market_features import numeric_prices
from build_s1_reference_matches import read
from build_weekly_shadow_panel import SOURCE, read_only
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE
from fit_cushing_outer_capture_nowcast import STEM as OPTICAL, SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_volatility_neighbors import volatility_forecast

MODEL = Path(__file__).resolve().parents[1] / "models/cushing_inventory"
# Compare signal combination, cost hurdle, and dated post-publication entry clocks.
RULES = ("updated", "equal_weight", "agreement")
COSTS = (170, 340)


def read_index(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {r["release_date"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate releases in {path}")
    return result


def direction(value, hurdle):
    return int(value > hurdle) - int(value < -hurdle)


def rule_signal(rule, incumbent, updated, hurdle):
    if rule == "updated":
        return updated
    if rule == "equal_weight":
        return (updated + incumbent) / 2
    # Both estimates must clear the threshold in the same direction.
    if direction(incumbent, hurdle) == direction(updated, hurdle):
        return updated
    return 0.


def early_intervals():
    bars = defaultdict(dict)
    for bar in read("wti_cushing_event_bars"):
        bars[bar["release_date"]][int(bar["session_number"]), bar["leg"]] = bar
    through = max(bar["session_date"] for tape in bars.values() for bar in tape.values())
    with closing(read_only(SOURCE)) as db:
        price = numeric_prices(db, {"archive", "ibkr"}, through=through)
    windows = {}
    for date, tape in bars.items():
        if len(tape) != 10:
            continue
        source = next((s for s in ("archive", "ibkr") if all(tape[i, leg][s + "_present"] == "True"
                       for i in range(2, 6) for leg in ("near", "far"))), None)
        if source is None or not all(tape[1, leg][source + "_present"] == "True" for leg in ("near", "far")):
            continue
        marks = []
        for i in (1, 2):
            legs = [tape[i, leg] for leg in ("near", "far")]
            if legs[0]["settlement_reference_time"] != legs[1]["settlement_reference_time"]:
                raise ValueError(f"Asynchronous legs on {date}")
            ticks = [tick_price(price(source, b["session_date"], b["delivery_month"], b[source + "_reference"])) for b in legs]
            marks.append(10 * (ticks[0] - ticks[1]))
        windows[date] = (datetime.fromisoformat(tape[1, "near"]["settlement_reference_time"]),
                         datetime.fromisoformat(tape[2, "near"]["settlement_reference_time"]), marks[1] - marks[0],
                         tape[1, "near"]["delivery_month"], tape[1, "far"]["delivery_month"], source)
    return windows


def replay(intervals):
    balance = read_index(MODEL / "balance_forecasts.csv")
    optical = {r["release_date"]: r for r in read(OPTICAL + "_forecasts")}
    incumbent = {r["release_date"]: r for r in read("cushing_cl_extension_segments_trades")
                 if r["method"] == "curve_post_report_gate"}
    exits = {r["release_date"]: r for r in read("cushing_cl_extension_segments_market_forecasts")
             if r["target"] == "after_report" and r["design"] == "curve"}
    training = []
    early_training = []
    for row, _ in inputs():
        date = row["release_date"]
        prior = optical[date]
        forecast = balance.get(date)
        if forecast:
            if (datetime.fromisoformat(forecast["cutoff"]) != datetime.fromisoformat(prior["cutoff_utc"])
                    or float(forecast["target"]) != row["x"][0]):
                raise ValueError(f"Inventory vintage mismatch on {date}")
            expected = float(forecast["estimate"])
        else:
            if date >= min(balance):
                raise ValueError(f"Missing updated point on {date}")
            expected = float(prior[SUPPLEMENT])
        if not datetime.fromisoformat(prior["cutoff_utc"]) < datetime.fromisoformat(prior["known_utc"]) <= row["decision"] < row["entry"]:
            raise ValueError(f"Inconsistent publication clocks on {date}")
        current = row | dict(x=replace_expectation(row, expected))
        sample = [q for q in training if q["exit_at"] < row["decision"]][-HISTORY:]
        training.append(current)
        extra_signal = None
        route = intervals.get(date)
        if route and row["decision"] < route[0]:
            if route[1] != row["entry"] or tuple(route[3:5]) != (row["near_month"], row["far_month"]):
                raise ValueError(f"Earlier entry shifted a delivery on {date}")
            # Fit the first-to-second-session move separately from the subsequent multi-session move.
            early_row = row | dict(volatility_usd=max(300., row["volatility_usd"] / math.sqrt(3)),
                                   long_gross=route[2], exit_at=route[1])
            early_row["x"] = replace_expectation(early_row, expected)
            old_early = [q for q in early_training if q["exit_at"] < row["decision"]][-HISTORY:]
            early_training.append(early_row)
            if forecast and len(old_early) >= MIN_WINDOWS:
                extra_signal = volatility_forecast(old_early, early_row)
        if not forecast:
            continue
        if len(sample) < MIN_WINDOWS or date not in incumbent:
            raise ValueError(f"Missing training sample or control on {date}")
        updated = volatility_forecast(sample, current)
        ref = incumbent[date]
        if (any(ref[k] != row[k] for k in ("near_month", "far_month")) or
                datetime.fromisoformat(ref["entry_at_utc"]) != row["entry"]):
            raise ValueError(f"Contract or entry mismatch on {date}")
        old = float(ref["expected_long_move_usd"])
        early = row["last_before_next_release"]
        first_close = min(early + 1, 3) if early in (1, 2) else 3
        extension = exits.get(date)
        if extension:
            known = datetime.fromisoformat(extension["holding_decision_utc"])
            if not (datetime.fromisoformat(extension["latest_training_mark_utc"]) < known
                    < row["stamps"][first_close] and first_close < 3):
                raise ValueError(f"Invalid extension timestamp on {date}")
        extension_move = float(extension["expected_usd"]) if extension else 0
        variants = [("previous", HURDLE, old)]
        variants.extend((rule, hurdle, rule_signal(rule, old, updated, hurdle))
                        for rule in RULES for hurdle in COSTS)
        for rule, hurdle, signal in variants:
            side = direction(signal, hurdle)
            end = 3 if ref["extended_forecast_available"] == "True" and side * extension_move > 0 else first_close
            # trade() uses the original strict 170-dollar trigger internally.
            record = trade(row, rule, signal if side else 0., end, training_at=sample[-1]["exit_at"])
            if record["near_side"] != side:
                raise ValueError(f"Position did not reconcile on {date}")
            if rule == "previous" and any(record[k] != int(ref[k]) for k in ("gross_usd", "net_170_usd", "near_side")):
                raise ValueError(f"Archived control P&L did not reconcile on {date}")
            yield {k: record[k] for k in ("release_date", "near_month", "far_month", "decision_at_utc", "entry_at_utc",
                    "exit_at_utc", "latest_training_exit_utc", "near_side", "gross_usd", "net_170_usd",
                    *(f"session_{i}_net_170_usd" for i in range(2, 6)))} | dict(
                    rule=rule, hurdle_usd=hurdle, updated_inventory_estimate_mb=expected,
                    previous_inventory_estimate_mb=float(prior[SUPPLEMENT]), inventory_cutoff=forecast["cutoff"],
                    inventory_known_by=prior["known_utc"],
                    inventory_actual_mb=row["x"][0], previous_spread_signal_usd=old, updated_spread_signal_usd=updated,
                    third_mark=row["stamps"][1].isoformat(), second_to_third_long_usd=row["spread_path_usd"][1],
                    signal_usd=signal, first_interval_forecast_usd=extra_signal, extension_usd=extension_move if extension else None,
                    net_340_usd=record["gross_usd"] - 340 * abs(side))


def entry_variants(records, intervals):
    """Compare the second close, first eligible close, and a learned timing gate."""
    previous_exit = {}
    for r in records:
        yield r | dict(entry_clock="second", session_1_net_170_usd=0, early_entry_used=False)
        if r["rule"] == "previous":
            continue
        route = intervals.get(r["release_date"])
        side = r["near_side"]
        if r["hurdle_usd"] == HURDLE and r["rule"] in ("updated", "equal_weight"):
            deferred = r | dict(entry_clock="third", early_entry_used=False, session_1_net_170_usd=0)
            if datetime.fromisoformat(r["third_mark"]) >= datetime.fromisoformat(r["exit_at_utc"]):
                raise ValueError(f"No interval after third-session entry on {r['release_date']}")
            deferred["entry_at_utc"] = r["third_mark"]
            for name in ("gross_usd", "net_170_usd", "net_340_usd", *(f"session_{i}_net_170_usd" for i in range(3, 6))):
                deferred[name] -= side * r["second_to_third_long_usd"]
            deferred["session_2_net_170_usd"] = 0
            yield deferred
        for clock in ("first_if_available", "predicted_first_interval"):
            key = (r["rule"], r["hurdle_usd"], clock)
            eligible = (route is not None and datetime.fromisoformat(r["decision_at_utc"]) < route[0]
                        and (key not in previous_exit or previous_exit[key] < route[0]))
            if clock == "predicted_first_interval":
                forecast = r["first_interval_forecast_usd"]
                eligible = eligible and forecast is not None and side * forecast > 0
            adjusted = r | dict(entry_clock=clock, early_entry_used=bool(eligible and side), session_1_net_170_usd=0)
            if eligible:
                extra = side * route[2]
                adjusted["entry_at_utc"] = route[0].isoformat()
                adjusted["session_1_net_170_usd"] = -85 * abs(side)
                for field in ("gross_usd", "net_170_usd", "net_340_usd", *(f"session_{i}_net_170_usd" for i in range(2, 6))):
                    adjusted[field] += extra
            if side:
                previous_exit[key] = datetime.fromisoformat(adjusted["exit_at_utc"])
            yield adjusted


def summarize(rows):
    filled = [r for r in rows if r["near_side"]]
    total = high = drawdown = 0
    for r in rows:
        for i in range(1, 6):
            mark = total + r[f"session_{i}_net_170_usd"]
            high = max(high, mark)
            drawdown = max(drawdown, high - mark)
        total += r["net_170_usd"]
    gains = sorted((r["net_170_usd"] for r in filled), reverse=True)
    return dict(windows=len(rows), trades=len(filled), long=sum(r["near_side"] == 1 for r in rows),
                short=sum(r["near_side"] == -1 for r in rows), profitable=sum(v > 0 for v in gains),
                gross_usd=sum(r["gross_usd"] for r in rows), net_170_usd=total,
                net_340_usd=sum(r["net_340_usd"] for r in rows), daily_drawdown_usd=drawdown,
                net_without_three_largest_wins=total - sum(v for v in gains[:3] if v > 0))


def adaptive_book(groups):
    """At each decision, compare three one-spread books on already settled common windows."""
    keys = (("previous", HURDLE, "second"), ("updated", HURDLE, "second"),
            ("updated", HURDLE, "first_if_available"))
    history = []
    last_exit = None
    for options in zip(*(groups[k] for k in keys), strict=True):
        current = options[0]
        now = datetime.fromisoformat(current["decision_at_utc"])
        if any(q["release_date"] != current["release_date"] for q in options):
            raise ValueError("Unequal dates in adaptive book")
        comparable = [batch for settled, batch in history if now - timedelta(weeks=52) <= settled < now]
        active = sum(any(q["near_side"] for q in batch) for batch in comparable)
        choice = 0
        if active >= 6:
            profits = [sum(batch[i]["net_170_usd"] for batch in comparable) for i in range(len(keys))]
            choice = max(range(len(keys)), key=profits.__getitem__)
        q = options[choice]
        if last_exit and q["near_side"] and datetime.fromisoformat(q["entry_at_utc"]) <= last_exit:
            choice = 1
            q = options[choice]
        if q["near_side"]:
            last_exit = datetime.fromisoformat(q["exit_at_utc"])
        yield q | dict(rule="adaptive", entry_clock="past_52_weeks", component=keys[choice][0] + "_" + keys[choice][2])
        history.append((max(datetime.fromisoformat(r["exit_at_utc"]) for r in options), options))


def main():
    with threadpool_limits(limits=1):
        intervals = early_intervals()
        records = list(entry_variants(replay(intervals), intervals))
    groups = defaultdict(list)
    for record in records:
        groups[record["rule"], record["hurdle_usd"], record["entry_clock"]].append(record)
    for record in records:
        record["component"] = ""
    adaptive = list(adaptive_book(groups))
    records.extend(adaptive)
    groups["adaptive", HURDLE, "past_52_weeks"] = adaptive
    for group in groups.values():
        rows = [r for r in group if r["near_side"]]
        if any(datetime.fromisoformat(a["exit_at_utc"]) >= datetime.fromisoformat(b["entry_at_utc"])
               for a, b in zip(rows, rows[1:])):
            raise ValueError("Overlapping contract windows")
    # Exclude any position that was still open at the start of the following year.
    fit = {k: summarize([r for r in rows if r["exit_at_utc"][:4] < "2026"]) for k, rows in groups.items()}
    candidates = [k for k in groups if k[0] not in ("previous", "adaptive")]
    selected = max(candidates, key=lambda k: (fit[k]["net_170_usd"], -fit[k]["daily_drawdown_usd"]))
    report = dict(selected=dict(rule=selected[0], hurdle_usd=selected[1], entry_clock=selected[2]), inventory_release_count=len(read_index(MODEL / "balance_forecasts.csv")),
                  selection_basis="Exploratory historical comparison. Select by realized net P&L on exits in 2024–2025, break ties by lower daily-mark drawdown, then declared order. Show 2026 separately through last observed exit.",
                  selection={f"{k[0]}_{k[1]}_{k[2]}": value for k, value in fit.items()}, results={})
    for name, key in (("previous", ("previous", HURDLE, "second")), ("selected", selected), ("updated_unfiltered", ("updated", HURDLE, "second")),
                      ("learned_timing", ("updated", HURDLE, "predicted_first_interval")),
                      ("adaptive", ("adaptive", HURDLE, "past_52_weeks")),
                      ("equal_second", ("equal_weight", HURDLE, "second"))):
        group = groups[key]
        years = sorted({r["exit_at_utc"][:4] for r in group})
        after = [r for r in group if r["decision_at_utc"][:4] >= "2026"]
        report["results"][name] = dict(all=summarize(group), following_period=summarize(after),
                by_exit_year={year: summarize([r for r in group if r["exit_at_utc"].startswith(year)]) for year in years})
    with (MODEL / "balance_spread_trades.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (MODEL / "balance_spread_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
