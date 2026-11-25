"""Predict the two parts of holding a dated CL spread through a second EIA report."""

import json
import math
from collections import defaultdict
from datetime import datetime, time

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_before_next_release import WEIGHTED
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_following_report import prerelease
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import RISK_FLOOR_USD, risk_prediction
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from publication_clocks import EASTERN

STEM = "cushing_cl_extension_segments"
BASELINES = ("hold_fifth", "exit_before", "exit_report_close")
DESIGNS = ("curve", "public", "satellite", "calibrated_inventory")
EXIT_RULES = ("total", "segments", "segments_loss_guard", "post_report_gate")
METHODS = (*BASELINES, *(f"{d}_{m}" for d in DESIGNS for m in EXIT_RULES))
TARGETS = ("total", "report_session", "after_report")
INVENTORY = ("public", "satellite", "calibrated_inventory")


def archive():
    forecasts = prerelease()
    for row in read("cushing_prerelease_sensor_forecasts"):
        yield forecasts[row["issue"]] | dict(issue=row["issue"], phase=row["study_phase"], actual=float(row["inventory"]))


def calibrated_inventory(forecasts):
    history = []
    for q in sorted(forecasts, key=lambda r: r["cutoff"]):
        past = [r for r in history if r["known"] < q["cutoff"]][-104:]
        value = q["optical"]
        if len(past) >= MIN_WINDOWS:
            revisions = np.array([r["optical"] - r["public"] for r in past])
            errors = np.array([r["actual"] - r["public"] for r in past])
            weight = np.exp2(-np.arange(len(past) - 1, -1, -1) / 52)
            # Estimate bias and the reliability of the optical innovation.
            # Keep a prior on both parameters, then downweight large misses.
            design = np.column_stack((np.ones(len(past)), revisions))
            regularizer = weight.sum() * np.diag((.75, .25))
            prior = np.array((0., 1.))
            coeff = prior
            for _ in range(6):
                w = weight / np.hypot(1., (errors - design @ coeff) / .75)
                coeff = np.linalg.solve(design.T @ (w[:, None] * design) + regularizer,
                                        design.T @ (w * errors) + regularizer @ prior)
            bias, gain = coeff
            value = q["public"] + float(np.clip(bias, -.4, .4) + np.clip(gain, 0., 1.5) * (q["optical"] - q["public"]))
        yield q | dict(satellite=q["optical"], calibrated_inventory=value,
                       inventory_train_known=past[-1]["known"] if past else None)
        history.append(q)


def holding_context(row, entry):
    early, boundary = row["last_before_next_release"], row["next_report_clock"]
    if early not in (1, 2):
        return None
    step = early + 1
    previous_close = row["stamps"][early - 1]
    decision = datetime.combine(row["stamps"][early].date(), time.min, EASTERN)
    if not previous_close < decision < row["stamps"][early] < boundary <= row["stamps"][step]:
        raise ValueError(f"Invalid order of holding marks on {row['release_date']}")
    sigma = max(RISK_FLOOR_USD, row["volatility_usd"] / math.sqrt(3))
    target, _, carry, slope, *_ = row["x"]
    prior = row["spread_path_usd"][early - 1] / sigma
    vol = math.log(sigma / 1000)
    core = (carry, slope, target, vol, prior, entry / row["volatility_usd"], float(step < 3))
    return dict(decision=decision, previous_close=previous_close, early=early, report_close=step, sigma=sigma, core=core)


def case(row, entry, forecast):
    context = holding_context(row, entry)
    if context is None or forecast is None or forecast["cutoff"] > context["decision"]:
        return None
    if not forecast["cutoff"] < row["next_report_clock"] < forecast["known"]:
        raise ValueError(f"Invalid following-report lineage on {row['release_date']}")
    early, step, core = (context[k] for k in ("early", "report_close", "core"))
    vol = core[3]
    public, satellite, calibrated = (forecast[k] for k in INVENTORY)
    designs = (core, (*core, public, public * vol),
               (*core, public, public * vol, satellite - public),
               (*core, calibrated, calibrated * vol, satellite - public))
    price = row["spread_path_usd"]
    labels = {"total": (price[3] - price[early], row["stamps"][3], 3 - early),
              "report_session": (price[step] - price[early], row["stamps"][step], 1)}
    if step < 3:
        labels["after_report"] = (price[3] - price[step], row["stamps"][3], 3 - step)
    if labels["total"][0] != sum(labels[k][0] for k in labels if k != "total"):
        raise ValueError(f"The extension payoff does not reconcile on {row['release_date']}")
    return context | dict(release_date=row["release_date"], study_phase=row["study_phase"], designs=designs,
                          labels=labels, forecast=forecast)


def predict(history, current):
    results = {}
    for target in current["labels"]:
        samples = [r for r in history[target] if r["exit_at"] < current["decision"]][-HISTORY:]
        change, close, length = current["labels"][target]
        row = current | dict(exit_at=close, volatility_usd=current["sigma"] * math.sqrt(length), long_gross=change)
        history[target].append(row)
        if len(samples) >= MIN_WINDOWS:
            values = {d:risk_prediction([q | dict(x=q["designs"][i]) for q in samples], row | dict(x=x), len(x))
                      for i, (d, x) in enumerate(zip(DESIGNS, row["designs"], strict=True))}
            results[target] = dict(values=values, latest=samples[-1]["exit_at"], samples=len(samples))
    return results


def after_report_curve(forecasts, current):
    initial = {r["release_date"]:r for r in read("cushing_volatility_neighbors_trades") if r["method"] == WEIGHTED}
    by_issue = {r["issue"]:r for r in forecasts}
    history = []
    for row, _ in inputs():
        anchor = initial.get(row["release_date"])
        if anchor is None or row["exit_at"] >= current["decision"]:
            continue
        context = case(row, float(anchor["expected_long_move_usd"]), by_issue.get(row["next_issue"]))
        if context is not None and "after_report" in context["labels"]:
            movement, settled, length = context["labels"]["after_report"]
            history.append(dict(x=context["core"], long_gross=movement, exit_at=settled,
                                volatility_usd=context["sigma"] * math.sqrt(length)))
    sample = history[-HISTORY:]
    if len(sample) < MIN_WINDOWS:
        return None, None
    point = risk_prediction(sample, dict(x=current["core"], volatility_usd=current["sigma"]), len(current["core"]))
    return point, sample[-1]["exit_at"]


def exit_indices(signal, model, side, index, step):
    if not model or not side or "total" not in model or "report_session" not in model:
        return {f"{design}_{rule}":step if rule == "post_report_gate" else 3
                for design in DESIGNS for rule in EXIT_RULES}
    ends = {}
    for name in DESIGNS:
        total = side * model["total"]["values"][name]
        move = side * model["report_session"]["values"][name]
        remaining = model.get("after_report")
        # Shrink the independently estimated sum toward the direct endpoint.
        # When only one price interval remains, both prices are identical.
        full = .5 * (move + side * remaining["values"][name] + total) if remaining else total
        points = {index:0., 3:full}
        if step < 3 and remaining is not None:
            points[step] = move
        ends[f"{name}_total"] = 3 if total > 0 else index
        ends[f"{name}_segments"] = max(points, key=points.get)
        ends[f"{name}_segments_loss_guard"] = max(points, key=points.get) if max(points.values()) - full > .10 * signal else 3
        ends[f"{name}_post_report_gate"] = 3 if remaining and side * remaining["values"][name] > 0 else step
    return ends


def replay(forecasts):
    initial = {r["release_date"]:r for r in read("cushing_volatility_neighbors_trades") if r["method"] == WEIGHTED}
    history = defaultdict(list)
    forecasts_by_issue = {r["issue"]:r for r in forecasts}
    for row, _ in inputs():
        issue = row["release_date"]
        anchor = initial.get(issue)
        if anchor is None:
            continue
        if (any(anchor[k] != row[k] for k in ("near_month", "far_month")) or
                any(datetime.fromisoformat(anchor[k]) != row[col] for k, col in (
                    ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))):
            raise ValueError(f"The original dated contracts or session times changed on {issue}")
        entry = float(anchor["expected_long_move_usd"])
        side = int(anchor["near_side"])
        current = case(row, entry, forecasts_by_issue.get(row["next_issue"]))
        model = predict(history, current) if current else {}
        early = row["last_before_next_release"]
        # These controls retain all entries and shorten only when a strictly
        # later closing price is available before the next report.
        can_shorten = early in (1, 2)
        index = early if can_shorten else 3
        step = min(index + 1, 3)
        ends = dict(zip(BASELINES, (3, index, step), strict=True))
        ends.update(exit_indices(current["sigma"] if current else None, model, side, index, step))
        result = []
        for name, end in ends.items():
            r = trade(row, name, entry, end)
            if r["near_side"] != side or (name == BASELINES[0] and int(anchor["net_170_usd"]) != r["net_170_usd"]):
                raise ValueError(f"Trading reference changed on {issue}/{name}")
            result.append(r | dict(net_340_usd=r["gross_usd"] - 340 * abs(side),
                holding_decision_utc=current["decision"].isoformat() if current else None,
                latest_holding_price_utc=current["previous_close"].isoformat() if current else None,
                following_inventory_cutoff_utc=current["forecast"]["cutoff"].isoformat() if current else None,
                latest_holding_training_exit_utc=max(v["latest"] for v in model.values()).isoformat() if model else None,
                original_latest_training_exit_utc=anchor["latest_training_exit_utc"],
                original_latest_calibration_exit_utc=anchor["latest_calibration_exit_utc"] or None,
                extended_forecast_available="total" in model and "report_session" in model))
        forecasts_out = []
        for target, fitted in model.items():
            actual, close, _ = current["labels"][target]
            for design, value in fitted["values"].items():
                forecasts_out.append(dict(release_date=issue, study_phase=row["study_phase"], target=target,
                    design=design, issuing_side=side, actual_usd=actual, expected_usd=value,
                    holding_decision_utc=current["decision"].isoformat(), target_mark_utc=close.isoformat(),
                    following_issue=current["forecast"]["issue"], latest_training_mark_utc=fitted["latest"].isoformat(),
                    training_windows=fitted["samples"], measured_roofs=current["forecast"]["roofs"]))
        yield result, forecasts_out


def forecast_metrics(groups, actual, get_prediction, scale, unit, baseline):
    if not groups[baseline]:
        return {}
    errors = {m:scale * np.array([abs(get_prediction(q, m) - q[actual]) for q in group])
              for m, group in groups.items()}
    return {m:dict(windows=len(e), **{f"mae_{unit}":float(e.mean()),
        f"paired_delta_95_{unit}":block_bootstrap_mean(e-errors[baseline])}) for m,e in errors.items()}


def summarize(trades, spread, stock):
    result = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        sample = [r for r in trades if r["study_phase"] == phase]
        groups = {m:[r for r in sample if r["method"] == m] for m in METHODS}
        base = groups[BASELINES[0]]
        trading = {}
        for name, group in groups.items():
            if [r["release_date"] for r in group] != [r["release_date"] for r in base]:
                raise ValueError(f"Unmatched holding decisions for {name}/{phase}")
            delta = [q["net_170_usd"] - b["net_170_usd"] for q,b in zip(group, base, strict=True)]
            trading[name] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(q["net_340_usd"] for q in group),
                changed_exits=sum(bool(q["near_side"] and q["exit_at_utc"] != b["exit_at_utc"])
                                  for q,b in zip(group, base, strict=True)),
                through_following_report=sum(bool(q["near_side"] and q["crosses_next_eia"]) for q in group),
                delta_vs_hold_95_usd=block_bootstrap_mean(delta))
        values = [r for r in spread if r["study_phase"] == phase]
        inventory = [r for r in stock if r["phase"] == phase and r["inventory_train_known"] is not None]
        inventory_metrics = forecast_metrics(dict.fromkeys(INVENTORY, inventory), "actual", lambda q,m:q[m], 1000, "kbbl", "satellite")
        market_metrics = {}
        for target in TARGETS:
            for active in (False, True):
                selected = [r for r in values if r["target"] == target and (not active or r["issuing_side"])]
                groups = {m:[r for r in selected if r["design"] == m] for m in DESIGNS}
                dates = [r["release_date"] for r in groups[DESIGNS[0]]]
                if any([r["release_date"] for r in g] != dates for g in groups.values()):
                    raise ValueError(f"Unmatched holding forecasts for {target}/{phase}")
                market_metrics[f"{target}_{'held' if active else 'all'}"] = forecast_metrics(
                    groups, "actual_usd", lambda q,_:q["expected_usd"], 1, "usd", "curve")
        result["phases"][phase] = dict(trading=trading, market_predictions=market_metrics, inventory=inventory_metrics)
        print(phase, {m:(r["net_170_usd"], r["net_340_usd"], r["changed_exits"]) for m,r in trading.items()})
        print(phase, "inventory", inventory_metrics)
    return result


def main():
    with threadpool_limits(limits=1):
        stock = list(calibrated_inventory(archive()))
        comparison = list(replay(stock))
    trades = [r for group,_ in comparison for r in group]
    forecasts = [r for _,group in comparison for r in group]
    report = summarize(trades, forecasts, stock)
    export(f"{STEM}_trades", trades)
    export(f"{STEM}_market_forecasts", forecasts)
    export(f"{STEM}_inventory_forecasts", stock)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
