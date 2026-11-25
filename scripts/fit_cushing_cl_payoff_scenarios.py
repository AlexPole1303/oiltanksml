"""Forecast both sides of the dated CL spread from settled forecast errors."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_inventory_coverage_distribution import quantile
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_neighbors import STEM as VOLATILITY

STEM = "cushing_cl_payoff_scenarios"
CURVE = "curve_reference"
LEADER = "weighted_satellite_reference"
PUBLIC_REFERENCE = "weighted_public_reference"
MODELS = ("pooled_satellite", "similar_public", "similar_satellite", "similar_with_optical_revision")
WIN_FILTER = "similar_with_optical_revision_win_filter"
POLICIES = (CURVE, LEADER, PUBLIC_REFERENCE, *MODELS,
            "similar_satellite_confirmed", "similar_with_optical_revision_confirmed", WIN_FILTER)
MODEL_INPUT = dict(zip(MODELS, (LEADER, PUBLIC_REFERENCE, LEADER, LEADER), strict=True))


def market_rows():
    inventory = indexed(f"{OUTER}_forecasts")
    anchors = {(r["release_date"], r["method"]): r for r in read(f"{VOLATILITY}_trades")
               if r["method"] in (f"{SUPPLEMENT}:volatility", f"{PUBLIC}:volatility")}
    carry = {r["release_date"]: r for r in read("cushing_cl_volatility_interactions_trades")
             if r["method"] == "carry_volatility_context"}
    for r, _ in inputs():
        issue, decision = r["release_date"], r["decision"]
        if (issue, f"{SUPPLEMENT}:volatility") not in anchors:
            continue
        refs = {CURVE: carry[issue], LEADER: anchors[issue, f"{SUPPLEMENT}:volatility"],
                PUBLIC_REFERENCE: anchors[issue, f"{PUBLIC}:volatility"]}
        f = inventory[issue]
        cutoff = datetime.fromisoformat(f["cutoff_utc"])
        known = datetime.fromisoformat(f["known_utc"])
        if (not cutoff < known <= decision < r["entry"] < r["exit_at"]
                or float(f["actual"]) != r["x"][0]
                or f[f"{SUPPLEMENT}_latest_label_utc"]
                and datetime.fromisoformat(f[f"{SUPPLEMENT}_latest_label_utc"]) >= cutoff):
            raise ValueError(f"Invalid report vintage on {issue}")
        for a in refs.values():
            if (any(a[key] != r[key] for key in ("near_month", "far_month", "study_phase"))
                    or not math.isclose(float(a["volatility_usd"]), r["volatility_usd"])
                    or any(datetime.fromisoformat(a[key]) != r[field] for key, field in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))
                    or any(datetime.fromisoformat(a[key]) >= decision for key in (
                        "latest_training_exit_utc", "latest_calibration_exit_utc") if a.get(key))):
                raise ValueError(f"Unmatched control or unsettled training outcome on {issue}")
        forecasts = {m: float(ref.get("expected_long_move_usd", ref.get("expected_long_gross_usd")))
                     for m, ref in refs.items()}
        sigma = r["volatility_usd"]
        actual, _, carry_level, *_ = r["x"]
        public = float(f[PUBLIC])
        optical = float(f[SUPPLEMENT])
        context = (math.log(sigma / 1000), carry_level, actual - public)
        coordinates = dict(zip(MODELS, ((), (*context, forecasts[PUBLIC_REFERENCE] / sigma),
            (*context, forecasts[LEADER] / sigma), (*context, forecasts[LEADER] / sigma, optical - public)), strict=True))
        yield r | dict(anchors=refs, forecasts=forecasts, coordinates=coordinates)


def distribution(history, row, model):
    anchor = MODEL_INPUT[model]
    sigma = row["volatility_usd"]
    age = np.array([(row["decision"] - p["decision"]).total_seconds() / 604800 for p in history])
    weights = np.exp2(-age / 78)
    current = row["coordinates"][model]
    if current:
        past = np.array([p["coordinates"][model] for p in history])
        widths = np.array((.65, .45, .75, .30, .30))[:len(current)]
        distance = np.square((past - current) / widths).mean(axis=1)
        weights *= .25 + .75 * np.exp(-.5 * distance)
    errors = np.array([(p["long_gross"] - p["forecasts"][anchor]) / p["volatility_usd"] for p in history])
    # Retain historical tail scenarios and shrink the estimated bias toward
    # zero. The mean correction is bounded, while scenario losses are not.
    correction = float(np.clip(weights @ np.clip(errors, -3., 3.) / (weights.sum() + 26.), -.35, .35))
    center = row["forecasts"][anchor] + sigma * correction
    bounds, support, mass = quantile(center + sigma * (errors - np.average(errors, weights=weights)), weights)
    spread = float(mass @ (support * (2 * np.cumsum(mass) - mass - 1)))
    realized = row["long_gross"]
    result = dict(expected_usd=float(mass @ support), median_usd=float(bounds[1]), low_usd=float(bounds[0]),
                  high_usd=float(bounds[2]), crps_usd=float(mass @ abs(support - realized)) - spread,
                  effective_windows=float(weights.sum() ** 2 / (weights @ weights)))
    for name, side in (("long", 1), ("short", -1)):
        payoff = side * support - HURDLE
        wins = float(mass @ np.maximum(payoff, 0))
        losses = float(mass @ np.maximum(-payoff, 0))
        probability = float(np.clip(mass @ (payoff > 0), 0., 1.))
        result.update({f"{name}_win_probability": probability,
                       f"{name}_expected_gain_usd": wins, f"{name}_expected_loss_usd": losses})
        if not math.isclose(wins - losses, side * result["expected_usd"] - HURDLE, abs_tol=1e-8):
            raise ValueError("The probability distribution does not reconcile to the expected payoff")
    return result


def replay():
    history = []
    for r in market_rows():
        settled = [p for p in history if p["exit_at"] < r["decision"]][-HISTORY:]
        history.append(r)
        estimates = {m: distribution(settled, r, m) for m in MODELS} if len(settled) >= MIN_WINDOWS else {}
        forecast = r["forecasts"] | {m: e["expected_usd"] for m, e in estimates.items()}
        for model, estimate in estimates.items():
            diagnostics = {key: r[key] for key in ("release_date", "study_phase", "near_month", "far_month")}
            yield "forecasts", diagnostics | dict(model=model, actual_usd=r["long_gross"],
                decision_at_utc=r["decision"].isoformat(), latest_training_exit_utc=settled[-1]["exit_at"].isoformat(),
                training_windows=len(settled), reference_forecast_usd=r["forecasts"][MODEL_INPUT[model]]) | estimate
        for model in MODELS[2:]:
            value = forecast.get(model, r["forecasts"][LEADER])
            # Require agreement on side with the point model and keep the
            # same entry hurdle for both. This variant only removes trades.
            ref = r["forecasts"][LEADER]
            forecast[f"{model}_confirmed"] = ref if value * ref > 0 and abs(value) > HURDLE else 0.
        # Test whether requiring a majority chance of profit helps even
        # when an asymmetric payoff already has positive expectation.
        forecast[WIN_FILTER] = r["forecasts"][LEADER]
        if estimates:
            scenarios = estimates[MODELS[-1]]
            current = forecast[MODELS[-1]]
            side = "long" if current > 0 else "short"
            forecast[WIN_FILTER] = current if scenarios[f"{side}_win_probability"] > .5 else 0.
        for method in POLICIES:
            anchor = MODEL_INPUT.get(method, method if method in r["anchors"] else LEADER)
            a = r["anchors"][anchor]
            stamp = datetime.fromisoformat(a["latest_training_exit_utc"])
            calibration = a.get("latest_calibration_exit_utc")
            if settled and estimates and method not in r["anchors"]:
                stamp = max(stamp, settled[-1]["exit_at"])
            result = trade(r, method, forecast.get(method, r["forecasts"][anchor]), 3, training_at=stamp,
                           calibration_at=datetime.fromisoformat(calibration) if calibration else None)
            if method in r["anchors"] and (result["near_side"] != int(a["near_side"])
                                           or result["net_170_usd"] != int(a["net_170_usd"])):
                raise ValueError(f"The archived control failed on {r['release_date']}/{method}")
            yield "trades", result | dict(net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def scores(arm, base):
    actual = np.array([q["actual_usd"] for q in arm])
    mean = np.array([q["expected_usd"] for q in arm])
    median = np.array([q["median_usd"] for q in arm])
    error = abs(median - actual)
    anchor_error = abs(np.array([q["reference_forecast_usd"] for q in arm]) - actual)
    crps = np.array([q["crps_usd"] for q in arm])
    pooled_crps = np.array([q["crps_usd"] for q in base])
    brier = {}
    for name, side in (("long", 1), ("short", -1)):
        event = side * actual > HURDLE
        loss = np.square(np.array([q[f"{name}_win_probability"] for q in arm]) - event)
        ref = np.square(np.array([q[f"{name}_win_probability"] for q in base]) - event)
        brier[name] = dict(score=float(loss.mean()), paired_delta_vs_pooled_95=block_bootstrap_mean(loss - ref),
                           actual_profitable_windows=int(event.sum()))
    return dict(windows=len(arm), mae_median_usd=float(error.mean()), mae_anchor_usd=float(anchor_error.mean()),
                mae_mean_usd=float(abs(mean - actual).mean()), mae_median_delta_vs_anchor_95_usd=block_bootstrap_mean(error - anchor_error),
                crps_usd=float(crps.mean()), crps_delta_vs_pooled_95_usd=block_bootstrap_mean(crps - pooled_crps),
                inside_80_percent=int(sum(q["low_usd"] <= q["actual_usd"] <= q["high_usd"] for q in arm)),
                average_80_percent_width_usd=float(np.mean([q["high_usd"] - q["low_usd"] for q in arm])), brier=brier)


def main():
    artifacts = defaultdict(list)
    groups = defaultdict(list)
    for kind, row in replay():
        artifacts[kind].append(row)
        groups[kind, row["study_phase"], row.get("model", row.get("method"))].append(row)
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        pooled = groups["forecasts", phase, MODELS[0]]
        leader = groups["trades", phase, LEADER]
        diagnostics, trading = {}, {}
        for model in MODELS:
            arm = groups["forecasts", phase, model]
            if [q["release_date"] for q in arm] != [q["release_date"] for q in pooled]:
                raise ValueError(f"Unmatched probability comparisons on {phase}/{model}")
            diagnostics[model] = scores(arm, pooled)
        for method in POLICIES:
            arm = groups["trades", phase, method]
            if [q["release_date"] for q in arm] != [q["release_date"] for q in leader]:
                raise ValueError(f"Unmatched trading windows on {phase}/{method}")
            delta = [q["net_170_usd"] - a["net_170_usd"] for q, a in zip(arm, leader, strict=True)]
            trading[method] = robustness(arm) | daily_risk(arm) | dict(net_340_usd=sum(q["net_340_usd"] for q in arm),
                delta_vs_leader_usd=sum(delta), paired_delta_vs_leader_95_usd=block_bootstrap_mean(delta),
                changed_sides=sum(q["near_side"] != a["near_side"] for q, a in zip(arm, leader, strict=True)))
        report["phases"][phase] = dict(forecasts=diagnostics, trades=trading)
        print(phase, {m: (q["trades"], q["net_170_usd"], q["net_340_usd"]) for m, q in trading.items()})
    for kind, rows in artifacts.items():
        export(f"{STEM}_{kind}", rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
