"""Replay a causal robust stack and its optical ablation on the saved inventory forecasts."""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from backtest_wti_cushing_spread import COSTS, statistics
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_robust_stack"
PUBLIC = ("public_ensemble", "adaptive_public", "prior", "zero")
OPTICAL = (("optical_ensemble", "public_ensemble"), ("stack_multiview", "adaptive_public"),
           ("sensor_only", "prior"))
HALF_LIFE_WEEKS = 78
MIN_HISTORY = 52
MIN_OPTICAL_HISTORY = 26
HUBER_SCALE = 0.6  # million barrels
PUBLIC_PENALTY = 0.08
OPTICAL_PENALTY = 0.15
MAX_OPTICAL_ADJUSTMENT = 1.0  # million barrels
SURPRISE_FILTERS = (0.0, 0.5)  # million barrels


def indexed(stem):
    rows = read(stem)
    result = {r["release_date"]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate release date in {stem}")
    return result


def source_rows():
    open_sensor = indexed("cushing_sensor_market_nowcast_forecasts")
    open_ensemble = indexed("cushing_nowcast_ensemble_forecasts")
    open_stack = indexed("cushing_multiview_nowcast_forecasts")
    closed = indexed("cushing_holdout_replay_forecasts")
    if open_sensor.keys() != open_ensemble.keys() or open_sensor.keys() != open_stack.keys() or open_sensor.keys() & closed.keys():
        raise ValueError("Source forecasts do not share the same release cohorts")
    for issue, sensor in sorted((open_sensor | closed).items()):
        ensemble = open_ensemble.get(issue, sensor)
        stack = open_stack.get(issue, sensor)
        for r in (ensemble, stack):
            if any(r[k] != sensor[k] for k in ("release_date", "week_ending", "study_phase", "forecast_cutoff_utc", "target_known_by_utc")):
                raise ValueError(f"Forecast provenance differs on {issue}")
        target = float(sensor["target"])
        if target != float(ensemble.get("target_million_bbl", sensor["target"])) or target != float(stack["target"]):
            raise ValueError(f"Inventory target differs on {issue}")
        cutoff, known = (datetime.fromisoformat(sensor[k]) for k in ("forecast_cutoff_utc", "target_known_by_utc"))
        upstream = [sensor["latest_training_label_utc"], ensemble.get("latest_training_release_available_utc"),
                    stack.get("latest_stack_training_release_available_utc")]
        if any(datetime.fromisoformat(stamp) >= cutoff for stamp in upstream if stamp) or cutoff >= known:
            raise ValueError(f"Invalid source clock on {issue}")
        forecasts = dict(public_ensemble=float(ensemble["public_ensemble"]), adaptive_public=float(stack["adaptive_public"]),
                         prior=float(ensemble["prior"]), zero=0., optical_ensemble=float(ensemble["optical_ensemble"]),
                         stack_multiview=float(stack["stack_multiview"]), sensor_only=float(sensor["sensor_only"]))
        optical = (np.array([forecasts[variant] - forecasts[base] for variant, base in OPTICAL])
                   if sensor["measured"] == "True" else None)
        yield dict(release_date=issue, study_phase=sensor["study_phase"], week_ending=sensor["week_ending"],
                   cutoff=cutoff, known=known, target=target, forecasts=forecasts,
                   public=np.array([forecasts[model] for model in PUBLIC]), optical=optical)


def fit_weights(history, row, *, feature, targets, anchor, penalty):
    design = np.asarray([r[feature] for r in history])
    actual = np.asarray([targets(r) for r in history])
    ages = np.array([(row["cutoff"] - r["cutoff"]).total_seconds() / 604800 for r in history])
    weights = np.exp2(-ages / HALF_LIFE_WEEKS)
    weights /= weights.sum()

    def objective(beta):
        diff = design @ beta - actual
        normalized = np.hypot(1, diff / HUBER_SCALE)
        offset = beta - anchor
        value = weights @ ((normalized - 1) * HUBER_SCALE**2) + penalty * (offset @ offset)
        gradient = design.T @ (weights * diff / normalized) + 2 * penalty * offset
        return value, gradient

    # A slack weight is equivalent to leaving some of the optical adjustment unused.
    constraints = dict(type="eq" if feature == "public" else "ineq", fun=lambda x: 1 - x.sum(),
                       jac=lambda x: -np.ones(len(x)))
    result = minimize(objective, anchor, method="SLSQP", jac=True, bounds=[(0., 1.)] * len(anchor),
                      constraints=constraints, options={"ftol": 1e-10, "maxiter": 100})
    total = result.x.sum()
    if (not result.success or not np.isfinite(result.x).all() or total > 1 + 1e-8
            or (feature == "public" and abs(total-1) > 1e-8)):
        raise ValueError(f"Invalid {feature} weights for {row['release_date']}: {result.message}")
    return result.x


def replay(rows):
    seen = []
    base_anchor = np.array([0.6, 0.25, 0.15, 0.])
    optical_anchor = np.zeros(len(OPTICAL))
    for row in rows:
        past = [r for r in seen if r["known"] < row["cutoff"]]
        measured = row["optical"] is not None
        public_weights = (fit_weights(past, row, feature="public", targets=lambda r: r["target"],
                                      anchor=base_anchor, penalty=PUBLIC_PENALTY) if len(past) >= MIN_HISTORY else base_anchor)
        row["robust_public"] = float(row["public"] @ public_weights)
        optical_history = [r for r in past if r["optical"] is not None]
        optical_weights = optical_anchor
        correction = 0.
        if measured and len(optical_history) >= MIN_OPTICAL_HISTORY:
            optical_weights = fit_weights(optical_history, row, feature="optical",
                                          targets=lambda r: r["target"] - r["robust_public"],
                                          anchor=optical_anchor, penalty=OPTICAL_PENALTY)
            correction = float(np.clip(row["optical"] @ optical_weights, -MAX_OPTICAL_ADJUSTMENT, MAX_OPTICAL_ADJUSTMENT))
        yield {k: row[k] for k in ("release_date", "study_phase", "week_ending", "target", "robust_public")} | dict(
            forecast_cutoff_utc=row["cutoff"].isoformat(), target_known_by_utc=row["known"].isoformat(),
            last_training_label_utc=max(r["known"] for r in past).isoformat() if past else None,
            public_history=len(past), optical_history=len(optical_history), optical_available=measured,
            public_weights_json=json.dumps(dict(zip(PUBLIC, map(float, public_weights), strict=True))),
            optical_weights_json=json.dumps(dict(zip((k for k, _ in OPTICAL), map(float, optical_weights), strict=True))),
            optical_correction_million_bbl=correction, robust_optical=row["robust_public"] + correction, **row["forecasts"])
        seen.append(row)


def trade_rows(predictions):
    lookup = {r["release_date"]: r for r in predictions}
    sources = (r for r in read("cushing_holdout_replay_trades") if r["method"] == "always_long_spread"
               and r["scope"] == "all")
    last_exit = ""
    for row in sources:
        forecast = lookup[row["release_date"]]
        if row["entry_session"] <= last_exit or forecast["target"] != float(row["published_million_bbl"]):
            raise ValueError(f"Invalid common contract window on {row['release_date']}")
        last_exit = row["exit_session"]
        move = int(row["gross_usd"])
        for model in ("public_ensemble", "robust_public", "robust_optical", "published_change", "always_long_spread", "no_trade"):
            for threshold in SURPRISE_FILTERS if model not in ("always_long_spread", "no_trade") else (0.,):
                expected = 0. if model == "published_change" else forecast.get(model)
                surprise = forecast["target"] - expected if expected is not None else 0.
                side = (1 if model == "always_long_spread" else 0 if abs(surprise) <= threshold else
                        int(surprise < 0) - int(surprise > 0))
                gross = side * move
                yield {k: row[k] for k in ("release_date", "entry_session", "exit_session", "near_month", "far_month", "source")} | dict(
                    method=model, threshold_million_bbl=threshold, near_side=side, expected_million_bbl=expected,
                    published_million_bbl=forecast["target"], gross_usd=gross,
                    **{f"cost_{cost}_usd": abs(side) * cost for cost in COSTS},
                    **{f"net_{cost}_usd": gross - abs(side) * cost for cost in COSTS})


def scores(rows):
    actual = np.asarray([r["target"] for r in rows])
    baseline = np.asarray([r["public_ensemble"] for r in rows])
    absolute_error = np.abs(baseline - actual)
    results = {}
    for method in (*PUBLIC, *(k for k, _ in OPTICAL), "robust_public", "robust_optical"):
        forecast = np.asarray([r[method] for r in rows])
        error = forecast - actual
        results[method] = dict(mae_million_bbl=float(np.abs(error).mean()), rmse_million_bbl=float(np.sqrt(np.mean(error**2))),
                               correct_direction=int(np.count_nonzero(forecast * actual > 0)),
                               mae_delta_vs_public_95pct_ci=block_bootstrap_mean(np.abs(error) - absolute_error))
    optical_delta = np.abs(np.array([r["robust_optical"] for r in rows]) - actual) - np.abs(np.array([r["robust_public"] for r in rows]) - actual)
    return dict(releases=len(rows), methods=results, optical_delta_vs_robust_public_95pct_ci=block_bootstrap_mean(optical_delta))


def main():
    predictions = list(replay(source_rows()))
    phases = {phase: {scope: scores([r for r in predictions if r["study_phase"] == phase and
                                     (scope == "all" or r["optical_available"])]) for scope in ("all", "with_optical")}
              for phase in ("development", "validation", "holdout")}
    trades = list(trade_rows(predictions))
    book = defaultdict(list)
    for r in trades:
        book[r["method"], r["threshold_million_bbl"]].append(r)
    result = dict(status="exploratory_replay_on_previously_examined_history", forecast=phases,
                  trading={f"{method}:{threshold:g}": statistics(rows) for (method, threshold), rows in sorted(book.items())})
    export(f"{STEM}_forecasts", predictions)
    export(f"{STEM}_trades", trades)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    for phase, scopes in phases.items():
        for scope, row in scopes.items():
            print(phase, scope, row["releases"], {k: round(row["methods"][k]["mae_million_bbl"], 4)
                  for k in ("public_ensemble", "robust_public", "robust_optical")})
    for key, row in result["trading"].items():
        print(key, row["trades"], row["net_50_usd"], row["net_170_usd"])


if __name__ == "__main__":
    main()
