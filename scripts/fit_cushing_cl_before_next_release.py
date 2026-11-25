"""Fit CL moves ending before the next EIA report and audit the leader's daily path."""

import json
import math
from collections import defaultdict
from datetime import datetime

from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features as response_features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import horizon_outcome, inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_volatility_interactions import features

STEM = "cushing_cl_before_next_release"
BASE = "fifth_public"
WEIGHTED = "outer_supplement:volatility"
WEIGHTED_EXITS = ("weighted_fifth", "weighted_before_next_fixed_side")
METHODS = ("same_side", "curve", "public", "optical", "public_conditional", "optical_conditional")
SEGMENTS = ("before_next_report", "straddling_next_report", "after_next_report")


def estimate_paths(r, options, trained, calibrated):
    end = r["last_before_next_release"]
    if not end:
        return (0.,) * len(METHODS[1:]), None, None
    short = horizon_outcome(r, end)
    # The number of held returns varies with the next release's calendar.
    vectors = tuple((*x, end - 2) for x in options)
    observed = short | dict(vectors=vectors)
    decision = r["decision"]
    past = [q for q in trained if q["exit_at"] < decision][-HISTORY:]
    trained.append(observed)
    if len(past) < MIN_WINDOWS:
        return (0.,) * len(METHODS[1:]), None, None
    values = []
    for i, x in enumerate(vectors):
        sample = [q | dict(x=q["vectors"][i]) for q in past]
        values.append(risk_prediction(sample, short | dict(x=x), len(x)))
    known = []
    for name, point in zip(("public", "optical"), values[1:], strict=True):
        realized = observed | dict(base_estimate=point, response_features=response_features(
            r["x"][2], short["volatility_usd"], r[f"{name}_actual_surprise"]))
        history = calibrated[name]
        settled = [q for q in history if q["exit_at"] < decision][-HISTORY:]
        adjustment, _ = conditional_residual(settled, realized) if len(settled) >= MIN_WINDOWS else (0., 0.)
        history.append(realized)
        values.append(point + adjustment)
        if settled:
            known.append(settled[-1]["exit_at"])
    return values, past[-1]["exit_at"], max(known, default=None)


def components(r, side):
    """Classify whole close-to-close intervals by their relation to the next report."""
    boundary = r["next_report_clock"]
    totals = dict.fromkeys(SEGMENTS, 0)
    for i, (start, end) in enumerate(zip(r["stamps"], r["stamps"][1:]), 1):
        segment = SEGMENTS[0] if end < boundary else SEGMENTS[1] if start < boundary else SEGMENTS[2]
        totals[segment] += side * (r["spread_path_usd"][i] - r["spread_path_usd"][i-1])
    if sum(totals.values()) != side * r["long_gross"]:
        raise ValueError(f"Daily attribution does not reconcile on {r['release_date']}")
    return totals


def replay():
    trained, calibrated = [], defaultdict(list)
    curve = {q["release_date"]: q for q in read("cushing_cl_volatility_interactions_trades")
             if q["method"] == "carry_volatility_context"}
    weighted = {q["release_date"]: q for q in read("cushing_volatility_neighbors_trades")
                if q["method"] == WEIGHTED}
    for r, anchors in inputs():
        issue = r["release_date"]
        boundary = r["next_report_clock"]
        if boundary is None:
            raise ValueError(f"No following report calendar on {issue}")
        models, past, calibration = estimate_paths(r, (features(r)[5], *r["designs"]), trained, calibrated)
        if "inventory_conditional" not in anchors:
            continue
        anchor = anchors["inventory_conditional"]
        carry_anchor = curve[issue]
        weighted_anchor = weighted[issue]
        for reference in (anchor, carry_anchor, weighted_anchor):
            if (any(reference[k] != r[k] for k in ("near_month", "far_month"))
                    or any(datetime.fromisoformat(reference[k]) != r[field] for k, field in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))):
                raise ValueError(f"Mismatched prices or trading clock on {issue}")
        expected = float(anchor["expected_long_move_usd"])
        base = trade(r, BASE, expected, 3)
        if base["gross_usd"] != int(anchor["gross_usd"]) or base["near_side"] != int(anchor["near_side"]):
            raise ValueError(f"The five-session leader changed on {issue}")
        daily = []
        for name, reference in ((BASE, anchor), ("fifth_curve", carry_anchor), (WEIGHTED_EXITS[0], weighted_anchor)):
            side = int(reference["near_side"])
            value = components(r, side)
            if sum(value.values()) != int(reference["gross_usd"]):
                raise ValueError(f"The daily sum differs from the archived payoff on {issue}/{name}")
            daily.append({k:base[k] for k in ("release_date", "study_phase", "near_month", "far_month",
                         "entry_at_utc", "exit_at_utc", "next_issue_at_utc", "next_report_clock_basis")} |
                         dict(method=name, near_side=side, **value, net_170_usd=sum(value.values())-170*abs(side)))
        short_end = r["last_before_next_release"] or 0
        fitted = [base]
        for method, value in zip(METHODS, (expected if short_end else 0., *models), strict=True):
            row = trade(r, method, value, short_end, training_at=past if method != "same_side" else None,
                        calibration_at=calibration if method.endswith("_conditional") else None)
            if row["near_side"] and (row["crosses_next_eia"] or r["stamps"][short_end] >= boundary):
                raise ValueError(f"The shortened trade crosses the next report on {issue}/{method}")
            fitted.append(row)
        point = float(weighted_anchor["expected_long_move_usd"])
        regular = trade(r, WEIGHTED_EXITS[0], point, 3)
        early = trade(r, WEIGHTED_EXITS[1], point if short_end else 0., short_end)
        if (any(regular[k] != int(weighted_anchor[k]) for k in ("near_side", "gross_usd", "net_170_usd"))
                or early["crosses_next_eia"] and early["near_side"]):
            raise ValueError(f"The weighted leader or shortened trading window did not verify on {issue}")
        for row in (regular, early):
            for key in ("latest_training_exit_utc", "latest_calibration_exit_utc"):
                row[key] = weighted_anchor[key] or None
        fitted.extend((regular, early))
        yield fitted, daily


def main():
    with threadpool_limits(limits=1):
        paired = list(replay())
    trades = [r for fitted, _ in paired for r in fitted]
    daily = [r for _, portions in paired for r in portions]
    report = dict(status="exploratory_on_reviewed_history", phases={}, settlement_attribution={})
    for phase in PHASES:
        groups = {m:[r for r in trades if r["study_phase"] == phase and r["method"] == m]
                  for m in (BASE, *METHODS, *WEIGHTED_EXITS)}
        base = groups[BASE]
        for method, group in groups.items():
            if [r["release_date"] for r in group] != [r["release_date"] for r in base]:
                raise ValueError(f"Mismatched calendar on {phase}/{method}")
        result = report["phases"][phase] = {}
        for method, group in groups.items():
            comparator = WEIGHTED_EXITS[1] if method in WEIGHTED_EXITS else "same_side"
            delta = [r["net_170_usd"] - q["net_170_usd"] for r, q in zip(group, groups[comparator], strict=True)]
            result[method] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["gross_usd"] - 340*abs(r["near_side"]) for r in group),
                trades_crossing_next_eia=sum(bool(r["near_side"] and r["crosses_next_eia"]) for r in group),
                delta_vs_fixed_side_short_95_usd=block_bootstrap_mean(delta))
        attribution = report["settlement_attribution"][phase] = {}
        for method in (BASE, "fifth_curve", WEIGHTED_EXITS[0]):
            group = [r for r in daily if r["study_phase"] == phase and r["method"] == method]
            attribution[method] = {k:sum(r[k] for r in group) for k in (*SEGMENTS, "net_170_usd")}
        print(phase, {m:(r["trades"],r["net_170_usd"],r["net_340_usd"]) for m,r in result.items()}, attribution)
    export(f"{STEM}_trades", trades)
    export(f"{STEM}_attribution", daily)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
