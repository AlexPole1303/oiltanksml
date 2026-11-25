"""Test independent optical innovations in the inventory and CL forecasts."""

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed

STEM = "cushing_satellite_innovation"
BASE = "all_recovered"
FORECASTS = (PUBLIC, "carried_scene", BASE, "pooled_gain", "coherent_gain", "optical_direction")
LEADER = "inventory_conditional"
CONTROL = "curve"
FEATURES = ("optical_residual", "optical_agreement", "lagged_residual")
METHODS = (CONTROL, LEADER, "optical_expectation", "coherent_expectation", *FEATURES)


def optical_coverage(pairs, public, cutoff):
    terminal = defaultdict(float)
    latest = None
    roofs = set()
    for pair in pairs:
        stamp = datetime.fromisoformat(pair["available_utc"])
        if pair["candidate_id"] in roofs or stamp >= cutoff or pair["sensitivity"] <= 0:
            raise ValueError(f"Invalid measured roof or timestamp at {cutoff}")
        roofs.add(pair["candidate_id"])
        latest = stamp if latest is None else max(latest, stamp)
        key = pair["terminal_id"]
        terminal[key] += pair["pair_signal"] - pair["prior_component"] - pair["sensitivity"] * public
    totals = list(terminal.values())
    coherence = abs(math.fsum(totals)) / max(math.fsum(map(abs, totals)), 1e-10)
    return coherence, len(totals), latest


def fit_gain(past, row, coherent):
    # The intercept stays fixed at the public forecast. A small prior
    # discourages a gain far from one when historical evidence is weak.
    if len(past) < 32:
        return 1.
    signal = np.array([r["optical_update"] for r in past])
    target = np.array([r["actual"] - r[PUBLIC] for r in past])
    age = np.array([(row["cutoff"] - r["cutoff"]).total_seconds() / 604800 for r in past])
    time_weight = np.exp2(-age / 78)
    if coherent:
        position = np.array([r["coherence"] for r in past])
        similarity = np.exp(-.5 * ((position - row["coherence"]) / .35)**2)
        time_weight *= .35 + .65 * similarity
    strength = .04 * time_weight.sum()
    gain = 1.
    for _ in range(8):
        weight = time_weight / np.maximum(abs(target - gain * signal), .10)
        gain = float(np.clip(((weight * signal) @ target + strength) / ((weight * signal) @ signal + strength), 0., 2.))
    return gain


def inventory_forecasts():
    saved = indexed("cushing_recovered_optical_forecasts")
    history = []
    previous_week = {}
    for state, observations, _ in combined_sources():
        issue, week = (state[k] for k in ("release_date", "week_ending"))
        cutoff, known = (state[k] for k in ("cutoff", "known"))
        r = saved[issue]
        latest_label = r["last_training_label_utc"]
        if (not cutoff < known or cutoff != datetime.fromisoformat(r["cutoff_utc"]) or
                known != datetime.fromisoformat(r["known_utc"]) or float(r["actual"]) != state["actual"] or
                latest_label and datetime.fromisoformat(latest_label) >= cutoff):
            raise ValueError(f"Forecast vintage mismatch on {issue}")
        pairs = observations[-1]
        if len(pairs) != int(r["total_recovered_tanks"]):
            raise ValueError(f"Recovered observations differ on {issue}")
        public, optical = (float(r[k]) for k in (PUBLIC, BASE))
        coherence, terminals, latest = optical_coverage(pairs, public, cutoff)
        update = optical - public
        if not pairs and update:
            raise ValueError(f"Optical forecast changed without a roof measurement on {issue}")
        train = [q for q in history if q["known"] < cutoff][-104:]
        row = dict(release_date=issue, week_ending=week, study_phase=state["study_phase"], actual=state["actual"],
                   cutoff=cutoff, known=known, public_ensemble=public, all_recovered=optical,
                   carried_scene=float(r["carried_scene"]), optical_update=update, coherence=coherence,
                   measured_roofs=len(pairs), measured_terminals=terminals, latest_optical_at=latest)
        pooled, coherent = (fit_gain(train, row, mode) for mode in (False, True))
        predecessor = (date.fromisoformat(week) - timedelta(days=7)).isoformat()
        prior = previous_week.get(predecessor)
        # The placebo uses exactly the previous week and the same current
        # coverage gate. No future label is required to carry a dated feature.
        lagged = prior["optical_update"] if pairs and prior and prior["cutoff"] < cutoff else 0.
        result = row | dict(pooled_gain=public + pooled * update, coherent_gain=public + coherent * update,
                            optical_direction=public + math.copysign(.25, update) if update else public,
                            pooled_beta=pooled, coherent_beta=coherent, lagged_update=lagged, measured_training=len(train),
                            latest_calibration_at=train[-1]["known"] if train else None)
        if pairs:
            history.append(result)
        previous_week[week] = result
        yield result


def market_correction(history, current, field):
    sample = [q for q in history if q[field] != 0][-104:]
    if not current[field] or len(sample) < 26:
        return 0., len(sample)
    values = np.array([q[field] for q in sample])
    design = np.column_stack((values, values * np.array([q["log_vol"] for q in sample])))
    y = np.array([(q["long_gross"] - q["anchor"]) / q["volatility_usd"] for q in sample])
    age = np.array([(current["decision"] - q["decision"]).total_seconds() / 604800 for q in sample])
    time_weight = np.exp2(-age / 78)
    penalty = np.diag((.10, .25)) * time_weight.sum()
    beta = np.zeros(2)
    for _ in range(6):
        weight = time_weight / np.hypot(1., (y - design @ beta) / .8)
        beta = solve(design.T @ (weight[:, None] * design) + penalty, design.T @ (weight * y), assume_a="pos")
    value = current[field] * (beta[0] + beta[1] * current["log_vol"])
    return current["volatility_usd"] * float(np.clip(value, -.35, .35)), len(sample)


def anchors():
    result = defaultdict(dict)
    for row in read("cushing_cl_conditional_response_trades"):
        if row["method"] in (CONTROL, LEADER):
            result[row["release_date"]][row["method"]] = row
    return result


def market_rows(forecasts):
    by_release = {q["release_date"]: q for q in forecasts}
    registered = anchors()
    _, _, cases = market_calendar()
    seen, completed = [], []
    for market in cases:
        issue = market["release_date"]
        f = by_release[issue]
        decision = market["decision"]
        if not f["cutoff"] < f["known"] <= decision < market["entry"] < market["exit_at"] or f["actual"] != market["x"][0]:
            raise ValueError(f"Invalid inventory or settlement clock on {issue}")
        design = [replace_expectation(market, f[k]) for k in (BASE, "coherent_gain")]
        settled = [q for q in seen if q["exit_at"] < decision][-HISTORY:]
        current = market | dict(designs=design)
        seen.append(current)
        if len(settled) < MIN_WINDOWS:
            continue
        references = registered[issue]
        for q in references.values():
            if (any(q[k] != market[k] for k in ("near_month", "far_month")) or
                    any(datetime.fromisoformat(q[k]) != market[key] for k, key in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at"))) or
                    q["latest_calibration_exit_utc"] and datetime.fromisoformat(q["latest_calibration_exit_utc"]) >= decision):
                raise ValueError(f"Unmatched trading timestamp on {issue}")
        public = f[PUBLIC]
        revision = f[BASE] - public
        # Positive values are evidence that the reported move was already
        # visible in the imagery. Negative values represent disagreement.
        agreement = abs(revision) * float(np.clip((f["actual"] - public) * np.sign(revision), -1.5, 1.5))
        anchor = float(references[LEADER]["expected_long_move_usd"])
        current = current | dict(anchor=anchor, log_vol=math.log(market["volatility_usd"] / 1000),
                                 optical_residual=revision, optical_agreement=agreement, lagged_residual=f["lagged_update"])
        past = [q for q in completed if q["exit_at"] < decision]
        completed.append(current)
        estimated = {CONTROL: float(references[CONTROL]["expected_long_move_usd"]), LEADER: anchor}
        samples = {}
        for i, method in enumerate(METHODS[2:4]):
            train = [q | dict(x=q["designs"][i]) for q in settled]
            estimated[method] = risk_prediction(train, current | dict(x=design[i]), len(design[i]))
        for method in FEATURES:
            correction, samples[method] = market_correction(past, current, method)
            estimated[method] = anchor + correction
        for method, estimate in estimated.items():
            side = int(estimate > HURDLE) - int(estimate < -HURDLE)
            gross = side * market["long_gross"]
            if method in references and (side, gross) != tuple(int(references[method][k]) for k in ("near_side", "gross_usd")):
                raise ValueError(f"The control decision changed on {issue}/{method}")
            yield {k: market[k] for k in ("release_date", "study_phase", "near_month", "far_month", "volatility_usd", "alternate_source")} | dict(
                method=method, decision_at_utc=decision.isoformat(), entry_at_utc=market["entry"].isoformat(),
                exit_at_utc=market["exit_at"].isoformat(), latest_market_training_exit_utc=settled[-1]["exit_at"].isoformat(),
                latest_residual_training_exit_utc=past[-1]["exit_at"].isoformat() if past else None,
                forecast_cutoff_utc=f["cutoff"].isoformat(), measured_roofs=f["measured_roofs"],
                optical_revision_million_bbl=revision, residual_nonzero_windows=samples.get(method),
                expected_long_move_usd=estimate, near_side=side, gross_usd=gross,
                alternate_gross_usd=side * market["alternate_long_gross_usd"] if market["alternate_source"] else None,
                **{f"session_{n}_net_170_usd": side * v - (170 if n == 5 else 85) * abs(side)
                   for n, v in enumerate(market["spread_path_usd"], 2)},
                **{f"cost_{c}_usd": c * abs(side) for c in COSTS},
                **{f"net_{c}_usd": gross - c * abs(side) for c in COSTS})


def inventory_metrics(rows):
    report = forecast_metrics(rows, FORECASTS)
    for method in FORECASTS[3:]:
        diff = [1000 * (abs(r["actual"] - r[method]) - abs(r["actual"] - r[BASE])) for r in rows]
        report["models"][method]["delta_vs_incumbent_95_kbbl"] = block_bootstrap_mean(diff)
    measured = [r for r in rows if r["optical_update"]]
    report["optical_residual_diagnostics"] = dict(weeks=len(measured),
        direction_hits=sum((r["actual"] - r[PUBLIC]) * r["optical_update"] > 0 for r in measured),
        correction_helped=sum(abs(r["actual"] - r[BASE]) < abs(r["actual"] - r[PUBLIC]) for r in measured),
        correction_hurt=sum(abs(r["actual"] - r[BASE]) > abs(r["actual"] - r[PUBLIC]) for r in measured))
    return report


def main():
    with threadpool_limits(limits=1):
        forecasts = list(inventory_forecasts())
        trades = list(market_rows(forecasts))
    report = dict(status="exploratory_on_reviewed_history", inventory={}, market={})
    for phase in PHASES:
        sample = [q for q in forecasts if q["study_phase"] == phase]
        measured = [q for q in sample if q["measured_roofs"]]
        report["inventory"][phase] = {group: inventory_metrics(rows) for group, rows in (
            ("all", sample), ("optical", measured), ("large_public_error", [
                q for q in measured if abs(q["actual"] - q[PUBLIC]) >= 1.]))}
        by_method = {m: [q for q in trades if q["study_phase"] == phase and q["method"] == m] for m in METHODS}
        leader = by_method[LEADER]
        comparison = {}
        for method, group in by_method.items():
            if [q["release_date"] for q in group] != [q["release_date"] for q in leader]:
                raise ValueError(f"Different trading cohort on {phase}/{method}")
            changes = [dict(release_date=q["release_date"], old_side=b["near_side"], new_side=q["near_side"],
                            increment_usd=q["net_170_usd"] - b["net_170_usd"])
                       for q, b in zip(group, leader, strict=True) if q["near_side"] != b["near_side"]]
            diff = [q["net_170_usd"] - b["net_170_usd"] for q, b in zip(group, leader, strict=True)]
            forecast_delta = [abs(q["expected_long_move_usd"] - b["expected_long_move_usd"])
                              for q, b in zip(group, leader, strict=True)]
            comparison[method] = robustness(group) | daily_risk(group) | dict(net_delta_vs_leader_usd=sum(diff),
                paired_delta_vs_leader_95_usd=block_bootstrap_mean(diff), changed_decisions=changes,
                forecast_updates=sum(value > 1e-8 for value in forecast_delta),
                largest_forecast_update_usd=max(forecast_delta),
                net_340_usd=sum(q["gross_usd"] - 340 * abs(q["near_side"]) for q in group))
        report["market"][phase] = comparison
        print(phase, {m: round(v["mae_kbbl"], 2) for m, v in report["inventory"][phase]["all"]["models"].items()})
        print(phase, {m: (v["net_170_usd"], v["net_340_usd"], v["daily_drawdown_usd"]) for m, v in comparison.items()})
    export(f"{STEM}_forecasts", [{k: r[k] for k in ("release_date", "week_ending", "study_phase", "actual", "optical_update",
        "coherence", "measured_roofs", "measured_terminals", "pooled_beta", "coherent_beta", "lagged_update", "measured_training",
        *FORECASTS)} | {f"{k}_utc": r[k].isoformat() if r[k] else None for k in ("cutoff", "known", "latest_optical_at",
                                                                             "latest_calibration_at")} for r in forecasts])
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
