"""Replay CL exit decisions using only the previous session's marks."""

import json
import math
from collections import defaultdict
from datetime import datetime, time, timedelta

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_ibkr_intraday import read_packets
from audit_cushing_leader_execution import from_ibkr, tally
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_before_next_release import WEIGHTED
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import RISK_FLOOR_USD, risk_prediction
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_volatility_neighbors import STEM as WEIGHTED_SOURCE
from publication_clocks import EASTERN

STEM = "cushing_cl_mark_to_market_exit"
REFERENCE = "registered_exit"
HORIZONS = (1, 2)
DESIGNS = ("curve", "marked_path")
METHODS = (REFERENCE, "fixed_fifth", "fixed_fourth", "curve_fourth", "marked_fourth", "marked_fourth_guard",
           "marked_two_decisions", "marked_two_decisions_guard")


def holding_inputs(row, anchor, index):
    stamps = row["stamps"]
    decision = datetime.combine(stamps[index].astimezone(EASTERN).date(), time.min, EASTERN)
    latest_price = stamps[index - 1]
    if not row["decision"] < latest_price < decision < stamps[index] < stamps[3]:
        raise ValueError(f"Invalid exit forecast clock for {row['release_date']} session {index+2}")
    volatility = row["volatility_usd"]
    sigma = max(RISK_FLOOR_USD, volatility * math.sqrt((3 - index) / 3))
    original_move = float(anchor["expected_long_move_usd"])
    report = row["next_report_clock"]
    report_in_extension = bool(report and stamps[index] < report <= stamps[3])
    report_since_entry = bool(report and stamps[0] < report <= latest_price)
    target, _, carry, slope, *_ = row["x"]
    core = (carry, slope, target, math.log(sigma / 1000), original_move / volatility,
            float(report_in_extension), float(report_since_entry))
    path = row["spread_path_usd"]
    # The price being selected for execution cannot enter the decision.
    pnl = path[index - 1] / volatility
    return dict(decision=decision, latest_price=latest_price, exit_at=stamps[3], designs=(core, (*core, pnl, pnl * original_move / volatility)),
                volatility_usd=sigma, long_gross=path[3] - path[index], release_date=row["release_date"])


def estimate(current, history):
    past = [q for q in history if q["exit_at"] < current["decision"]][-HISTORY:]
    history.append(current)
    if len(past) < MIN_WINDOWS:
        return {}, None
    result = {}
    for i, (name, x) in enumerate(zip(DESIGNS, current["designs"], strict=True)):
        fitted = [q | dict(x=q["designs"][i]) for q in past]
        result[name] = risk_prediction(fitted, current | dict(x=x), len(x))
    return result, past[-1]["exit_at"]


def exits(predictions, context, side, base):
    selections = {REFERENCE:base, "fixed_fifth":3, "fixed_fourth":2}
    variants = {"curve_fourth":("curve", (2,), 0.), "marked_fourth":("marked_path", (2,), 0.),
                "marked_fourth_guard":("marked_path", (2,), .10), "marked_two_decisions":("marked_path", HORIZONS, 0.),
                "marked_two_decisions_guard":("marked_path", HORIZONS, .10)}
    for name, (design, checks, margin) in variants.items():
        selections[name] = next((i for i in checks if i < base and design in predictions[i]
            and side * predictions[i][design] < -margin * context[i]["volatility_usd"]), base)
    return selections


def replay():
    reference = {r["release_date"]:r for r in read("cushing_cl_extension_segments_trades") if r["method"] == "curve_post_report_gate"}
    original = {q["release_date"]:q for q in read(f"{WEIGHTED_SOURCE}_trades") if q["method"] == WEIGHTED}
    history = {i:[] for i in HORIZONS}
    for row, _ in inputs():
        issue = row["release_date"]
        if issue not in original:
            continue
        anchor, final = original[issue], reference[issue]
        if (any(anchor[k] != final[k] for k in ("near_month", "far_month", "entry_at_utc", "decision_at_utc", "near_side"))
                or any(anchor[k] != row[k] for k in ("near_month", "far_month"))):
            raise ValueError(f"Mismatched original trading records for {issue}")
        contexts = {i:holding_inputs(row, anchor, i) for i in HORIZONS}
        fitted = {i:estimate(context, history[i]) for i,context in contexts.items()}
        predictions = {i:p for i,(p,_) in fitted.items()}
        side, signal = int(anchor["near_side"]), float(anchor["expected_long_move_usd"])
        base = int(final["exit_session_number"]) - 2
        outcomes = []
        for name, index in exits(predictions, contexts, side, base).items():
            t = trade(row, name, signal, index)
            if (t["near_side"] != side or name == REFERENCE and
                    any(t[k] != int(final[k]) for k in ("gross_usd", "net_170_usd", "net_340_usd") if k in t)):
                raise ValueError(f"Original trade did not reproduce on {issue}")
            c, trained = (contexts[index], fitted[index][1]) if index in HORIZONS else (None, None)
            outcomes.append(t | dict(net_340_usd=t["gross_usd"] - 340 * abs(side),
                market_exit_decision_utc=c["decision"].isoformat() if c else None,
                latest_price_mark_utc=c["latest_price"].isoformat() if c else None,
                latest_exit_training_label=trained.isoformat() if trained else None))
        model_rows = []
        for i, (p, trained) in fitted.items():
            for design, point in p.items():
                c = contexts[i]
                model_rows.append(dict(release_date=issue, study_phase=row["study_phase"], session=i+2, design=design,
                    expected_usd=point, actual_usd=c["long_gross"], near_side=side, decision_utc=c["decision"].isoformat(),
                    latest_price_utc=c["latest_price"].isoformat(), training_through_utc=trained.isoformat()))
        yield outcomes, model_rows


def intraday(rows):
    book, _ = read_packets()
    for q in rows:
        if q["study_phase"] != "holdout" or q["method"] != REFERENCE:
            continue
        entry, exit = (datetime.fromisoformat(q[k]) for k in ("entry_at_utc", "exit_at_utc"))
        near, far = q["near_month"], q["far_month"]
        for minutes in (5, 15, 30):
            offset = timedelta(minutes=minutes)
            # Shift only the exit on a common universe, with exact positive-volume legs.
            ends = (entry, exit - offset, exit, exit + offset)
            quotes = [from_ibkr(book, t, near, far, minutes) for t in ends]
            if not all(quotes):
                continue
            for label, t, quote in zip(("early", "same", "late"), ends[1:], quotes[1:], strict=True):
                side = q["near_side"]
                gross = side * (quote[0] - quotes[0][0])
                yield {k:q[k] for k in ("release_date", "near_month", "far_month", "entry_at_utc", "near_side")} | dict(
                    bar_minutes=minutes, timing=label, exit_at_utc=t.isoformat(), gross_usd=gross, net_170_usd=gross - 170 * abs(side),
                    net_340_usd=gross - 340 * abs(side), settlement_reference_net_170_usd=q["net_170_usd"],
                    near_entry_volume=quotes[0][1][0][1], far_entry_volume=quotes[0][1][1][1],
                    near_exit_volume=quote[1][0][1], far_exit_volume=quote[1][1][1],
                    source_sha256="|".join(sorted({bar[2] for pair in (quotes[0],quote) for bar in pair[1]})))


def report_bridge(rows):
    book, _ = read_packets()
    prior = {r["release_date"]:r for r in read("cushing_cl_before_next_release_trades")
             if r["method"] == "weighted_before_next_fixed_side"}
    width = timedelta(minutes=30)
    for q in rows:
        if (q["study_phase"] != "holdout" or q["method"] != REFERENCE or not q["near_side"]
                or not q["crosses_next_eia"] or q["next_report_clock_basis"] == "start_of_release_day"):
            continue
        before = prior[q["release_date"]]
        entry, exit, boundary = (datetime.fromisoformat(q[k]) for k in ("entry_at_utc", "exit_at_utc", "next_issue_at_utc"))
        earlier = datetime.fromisoformat(before["exit_at_utc"])
        ends = (earlier, boundary - width, boundary + width, exit)
        if not entry <= earlier < ends[1] < boundary < ends[2] <= exit or any(
                q[k] != before[k] for k in ("entry_at_utc", "near_month", "far_month")):
            continue
        if any(stamp.timestamp() % 1800 for stamp in ends):
            continue
        quotes = [from_ibkr(book, t, q["near_month"], q["far_month"], 30) for t in ends]
        if not all(quotes):
            continue
        side = q["near_side"]
        moves = [side * (b[0] - a[0]) for a,b in zip(quotes, quotes[1:])]
        if sum(moves) != side * (quotes[-1][0] - quotes[0][0]):
            raise ValueError(f"The EIA price bridge does not reconcile on {q['release_date']}")
        yield {k:q[k] for k in ("release_date", "near_month", "far_month", "near_side", "next_issue_at_utc", "exit_at_utc")} | dict(
            prior_settlement_utc=earlier.isoformat(), before_report_bar_close_utc=ends[1].isoformat(),
            after_report_bar_close_utc=ends[2].isoformat(), before_report_move_usd=moves[0],
            around_report_move_usd=moves[1], after_report_move_usd=moves[2], total_extension_usd=sum(moves),
            daily_reference_extension_usd=q["gross_usd"] - int(before["gross_usd"]),
            source_sha256="|".join(sorted({bar[2] for pair in quotes for bar in pair[1]})))


def summarize(rows, predictions, bars, bridge):
    results = {}
    for phase in PHASES:
        groups = {m:[r for r in rows if (r["study_phase"],r["method"]) == (phase,m)] for m in METHODS}
        base = groups[REFERENCE]
        trading = {}
        for name, group in groups.items():
            if [r["release_date"] for r in group] != [r["release_date"] for r in base]:
                raise ValueError(f"Mismatched exit comparison on {phase}/{name}")
            pairs = list(zip(group, base, strict=True))
            trading[name] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["net_340_usd"] for r in group),
                changed_exits=sum(bool(r["near_side"] and r["exit_at_utc"] != q["exit_at_utc"]) for r,q in pairs),
                delta_95_per_window_usd=block_bootstrap_mean([r["net_170_usd"] - q["net_170_usd"] for r,q in pairs]))
        metrics = {}
        for i in HORIZONS:
            errors = {m:np.array([abs(r["actual_usd"] - r["expected_usd"]) for r in predictions
                      if (r["study_phase"], r["session"], r["design"]) == (phase, i+2, m)]) for m in DESIGNS}
            metrics[i+2] = {m:dict(windows=len(e), mae_usd=float(e.mean()), delta_vs_curve_95_usd=block_bootstrap_mean(e - errors[DESIGNS[0]]))
                            for m,e in errors.items() if len(e)}
        results[phase] = dict(trading=trading, model_errors=metrics)
        print(phase, {m:(r["net_170_usd"],r["net_340_usd"],r["changed_exits"]) for m,r in trading.items()})
    replayed = {}
    for minute in (5, 15, 30):
        group = {k:[r for r in bars if (r["bar_minutes"], r["timing"]) == (minute,k)] for k in ("early", "same", "late")}
        if any([r["release_date"] for r in q] != [r["release_date"] for r in group["same"]] for q in group.values()):
            raise ValueError("Different comparison samples for intraday quotes")
        replayed[minute] = {name:tally(q) | dict(daily_reference_net_170_usd=sum(r["settlement_reference_net_170_usd"] for r in q))
                            for name,q in group.items()}
    bridge_totals = {name:sum(r[name] for r in bridge) for name in ("before_report_move_usd", "around_report_move_usd",
                     "after_report_move_usd", "total_extension_usd", "daily_reference_extension_usd")}
    return dict(status="exploratory_on_reviewed_history", phases=results, ibkr_offset_samples=replayed,
                second_report_bridge=dict(trades=len(bridge), **bridge_totals))


def main():
    with threadpool_limits(limits=1):
        replayed = list(replay())
    rows = [r for trades,_ in replayed for r in trades]
    points = [p for _,forecasts in replayed for p in forecasts]
    bars = list(intraday(rows))
    bridge = list(report_bridge(rows))
    summary = summarize(rows, points, bars, bridge)
    export(f"{STEM}_trades", rows)
    export(f"{STEM}_forecasts", points)
    if bars:
        export(f"{STEM}_intraday", bars)
    if bridge:
        export(f"{STEM}_report_bridge", bridge)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
