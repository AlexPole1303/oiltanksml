"""Stress the trading threshold and the current inventory input without moving the execution window."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_outer_capture_nowcast import SUPPLEMENT
from fit_cushing_volatility_neighbors import STEM as WEIGHTED
from register_cushing_cl_satellite import history as market_history
from register_cushing_leading_models import VOLATILITY_METHOD, prediction

STEM = "cushing_cl_entry_margin"
COSTS = (170, 340)
THRESHOLDS = (120, 145, HURDLE, 195, 220, 340)
SENSITIVITY = (100, 250, 500)  # thousand barrels added and subtracted independently
ERROR_REDUCTION = (.25, .5, 1.)


def side(value, threshold):
    return int(value > threshold) - int(value < -threshold)


def choose_threshold(history, decision):
    past = [q for q in history if q["exit_at"] < decision][-HISTORY:]
    if len(past) < 2 * MIN_WINDOWS:
        return HURDLE, past
    values, actual = [np.asarray([r[k] for r in past]) for k in ("point", "long_gross")]
    steps = np.asarray(THRESHOLDS)
    directions = (values[:, None] > steps).astype(int) - (values[:, None] < -steps).astype(int)
    i = THRESHOLDS.index(HURDLE)
    worst_improvement = np.full(len(steps), np.inf)
    for cost in COSTS:
        payouts = directions * actual[:, None] - cost * abs(directions)
        # Limit the weight of an isolated outcome on threshold selection.
        # Never cap the walk-forward P&L used to evaluate the policy.
        incremental = np.clip(payouts - payouts[:, i, None], -2000., 2000.)
        for half in np.array_split(incremental, 2):
            worst_improvement = np.minimum(worst_improvement, half.mean(axis=0))
    improvement = worst_improvement - 10. * abs(steps - HURDLE) / 50.
    return int(steps[np.argmax(improvement)]) if improvement.max() > 0 else HURDLE, past


def inputs_with_reference():
    references = {r["release_date"]: r for r in read(f"{WEIGHTED}_trades") if r["method"] == VOLATILITY_METHOD}
    cases = {r["release_date"]: r for r, _ in inputs()}
    past, calibrated = [], []
    for h in market_history():
        issue, decision = h["release_date"], h["decision"]
        if issue in references:
            ref, case = references[issue], cases[issue]
            if any(h[k] != case[k] for k in ("near_month", "far_month", "decision", "entry", "exit_at", "long_gross")):
                raise ValueError(f"The two market books disagree on {issue}")
            settled = [r for r in past if r["exit_at"] < decision][-HISTORY:]
            calibration = [r for r in calibrated if r["exit_at"] < decision][-HISTORY:]
            point, revision = prediction(settled, calibration, h)
            value = point + revision
            if (len(settled) < MIN_WINDOWS or not math.isclose(value, float(ref["expected_long_move_usd"]), rel_tol=0, abs_tol=1e-8)
                    or not math.isclose(point, float(ref["base_estimate_usd"]), rel_tol=0, abs_tol=1e-8)
                    or datetime.fromisoformat(ref["latest_training_exit_utc"]) != settled[-1]["exit_at"]
                    or side(value, HURDLE) != int(ref["near_side"])):
                raise ValueError(f"Historical forecast does not reproduce on {issue}")
            yield case, h, ref, value, settled, calibration
            calibrated.append(h | dict(base_estimate=point, response_features=features(
                h["x"][2], h["volatility_usd"], h["x"][0] - h[SUPPLEMENT])))
        past.append(h)


def replay():
    trades, stress, counterfactuals = [], [], []
    history = []
    for r, current, ref, value, settled, calibration in inputs_with_reference():
        d = r["decision"]
        chosen, eligible = choose_threshold(history, d)
        history.append(r | dict(point=value))
        for name, hurdle in [*((f"threshold_{h}", h) for h in THRESHOLDS), ("walk_forward_threshold", chosen)]:
            # The trade constructor implements the standard 170 USD decision
            # hurdle. Express an equivalent side without modifying the tracked model.
            policy_score = math.copysign(max(0., abs(value) + (HURDLE - hurdle)), value)
            row = trade(r, name, policy_score, 3, training_at=settled[-1]["exit_at"],
                        calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if row["near_side"] != side(value, hurdle) or (hurdle == HURDLE and row["net_170_usd"] != int(ref["net_170_usd"])):
                raise ValueError(f"The entry policy changes the reference side on {r['release_date']}/{name}")
            trades.append(row | dict(entry_threshold_usd=hurdle, original_model_move_usd=value,
                threshold_training_windows=len(eligible) if name == "walk_forward_threshold" else 0,
                latest_threshold_outcome_utc=eligible[-1]["exit_at"] if eligible and name == "walk_forward_threshold" else None,
                realized_long_move_usd=r["long_gross"], net_340_usd=row["gross_usd"]-340*abs(row["near_side"])))
        if r["study_phase"] != "holdout":
            continue
        original_side = side(value, HURDLE)
        for fraction in (0., *ERROR_REDUCTION):
            # The reported change was unavailable when the original forecast
            # was issued. These inputs are diagnostic, not trading rules.
            actual, forecast = current["x"][0], current[SUPPLEMENT]
            updated = actual if fraction == 1. else forecast + fraction * (actual - forecast)
            point, correction = prediction(settled, calibration, current | {SUPPLEMENT: updated})
            estimate = point + correction
            new_side = side(estimate, HURDLE)
            gross = new_side * r["long_gross"]
            counterfactuals.append({k: r[k] for k in ("release_date", "near_month", "far_month")} | dict(
                pre_release_error_reduction=fraction, hypothetical_uses_report=fraction != 0,
                actual_kbbl=1000 * actual, inventory_estimate_kbbl=1000 * updated,
                expected_long_move_usd=estimate, near_side=new_side, gross_usd=gross,
                **{f"net_{c}_usd": gross-c*abs(new_side) for c in COSTS}))
        for shift in SENSITIVITY:
            points = []
            for signed_shift in (-shift, shift):
                perturbed = current | {SUPPLEMENT: current[SUPPLEMENT] + signed_shift / 1000.}
                point, correction = prediction(settled, calibration, perturbed)
                points.append(point + correction)
            states = [side(p, HURDLE) for p in points]
            stress.append({k:r[k] for k in ("release_date", "near_month", "far_month")} | dict(
                decision_at_utc=d, last_settled_outcome=settled[-1]["exit_at"], input_shift_kbbl=shift,
                expected_inventory_kbbl=current[SUPPLEMENT]*1000, expected_long_move_usd=value,
                distance_from_threshold_usd=abs(abs(value)-HURDLE), near_side=original_side,
                down_estimate_usd=points[0], up_estimate_usd=points[1], down_side=states[0], up_side=states[1],
                changes_side_at_either_endpoint=any(s != original_side for s in states)))
    return trades, stress, counterfactuals


def summarize(rows, stress, counterfactuals):
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
    phases = {}
    for phase in PHASES:
        reference = groups[phase, f"threshold_{HURDLE}"]
        result = phases[phase] = {}
        for (p,m), arm in groups.items():
            if p != phase:
                continue
            if [r["release_date"] for r in arm] != [r["release_date"] for r in reference]:
                raise ValueError(f"The entry threshold changed the calendar for {m}")
            pairs = list(zip(arm, reference, strict=True))
            base = [r["net_170_usd"]-q["net_170_usd"] for r,q in pairs]
            result[m] = robustness(arm) | daily_risk(arm) | dict(net_340_usd=sum(r["net_340_usd"] for r in arm),
                paired_net_delta_170_usd=sum(base), paired_net_delta_per_window_95_usd=block_bootstrap_mean(base),
                changed_release_dates=[r["release_date"] for r,q in pairs if r["near_side"]!=q["near_side"]],
                windows_by_hurdle={str(h):sum(r["entry_threshold_usd"]==h for r in arm) for h in THRESHOLDS})
        print(phase, {m: (s["trades"],s["net_170_usd"],s["net_340_usd"]) for m,s in result.items()})
    scenarios = {}
    for shift in SENSITIVITY:
        arm = [r for r in stress if r["input_shift_kbbl"] == shift]
        changed = [r for r in arm if r["changes_side_at_either_endpoint"]]
        impact = [max(abs(r["down_estimate_usd"]-r["expected_long_move_usd"]),
                      abs(r["up_estimate_usd"]-r["expected_long_move_usd"])) for r in arm]
        scenarios[shift] = dict(windows=len(arm), affected_windows=len(changed),
            affected_nonzero_positions=sum(bool(r["near_side"]) for r in changed),
            affected_release_dates=[r["release_date"] for r in changed],
            median_maximum_change_usd=float(np.median(impact)), ninety_percentile_maximum_change_usd=float(np.quantile(impact,.90)))
    original = {r["release_date"]: r for r in counterfactuals if r["pre_release_error_reduction"] == 0}
    if sum(r["net_170_usd"] for r in original.values()) != phases["holdout"][f"threshold_{HURDLE}"]["net_170_usd"]:
        raise ValueError("The counterfactual altered its zero-shift reference")
    changed_accuracy = {}
    for f in (0., *ERROR_REDUCTION):
        arm = [q for q in counterfactuals if q["pre_release_error_reduction"] == f]
        changed_accuracy[f] = dict(windows=len(arm), hypothetical_uses_report=f != 0,
            mae_kbbl=float(np.mean([abs(r["actual_kbbl"]-r["inventory_estimate_kbbl"]) for r in arm])),
            trades=sum(bool(r["near_side"]) for r in arm),
            changed_sides=sum(r["near_side"] != original[r["release_date"]]["near_side"] for r in arm),
            **{f"net_{c}_usd":sum(r[f"net_{c}_usd"] for r in arm) for c in COSTS})
    return dict(status="exploratory_on_reviewed_history", phases=phases, current_inventory_input_sensitivity=scenarios,
                hypothetical_error_reduction_with_current_report=changed_accuracy)


def main():
    with threadpool_limits(limits=1):
        rows, stress, counterfactuals = replay()
    result = summarize(rows, stress, counterfactuals)
    export(f"{STEM}_trades", rows)
    export(f"{STEM}_inventory_sensitivity", stress)
    export(f"{STEM}_hypothetical_accuracy", counterfactuals)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
