"""Compare a small neural correction with the existing Cushing nowcast."""

import json
from collections import deque

import numpy as np
from scipy.optimize import minimize

from build_s1_reference_matches import DERIVED, export, read
from fit_cushing_coverage_nowcast import MIN_MEASURED, coverage_inputs

INPUTS = 5
HIDDEN = 6
WINDOW = 104
ROBUST_SCALE = 0.65
CORRECTION_LIMIT = 0.75
PENALTIES = (0.03, 0.1, 0.3, 1.0)
SEED = 42
MAX_STEPS = 250
STEM = "cushing_mlp_experiment"


def normalized(inputs, scale):
    return np.clip(inputs / scale, -4.0, 4.0)


def objective(parameters, x, target, weights, penalty):
    first = parameters[: INPUTS * HIDDEN].reshape(INPUTS, HIDDEN)
    second = parameters[INPUTS * HIDDEN :]
    hidden = np.tanh(x @ first)
    error = hidden @ second - target
    shape = np.hypot(1.0, error / ROBUST_SCALE)
    # Equivalent to the pseudo-Huber loss, stable near zero residual.
    loss = weights @ (error**2 / (shape + 1.0))
    loss += 0.5 * penalty * (parameters @ parameters)
    output_gradient = weights * error / shape
    second_gradient = hidden.T @ output_gradient
    first_gradient = x.T @ (
        (output_gradient[:, None] * second) * (1.0 - hidden**2)
    )
    gradient = np.r_[first_gradient.ravel(), second_gradient] + penalty * parameters
    return float(loss), gradient


def fit(history, cutoff, penalty):
    x = np.stack([row["x"][:INPUTS] for row in history])
    scale = np.maximum(np.sqrt(np.mean(x**2, axis=0)), 0.08)
    target = np.array([row["target"] - row["public_ensemble"] for row in history])
    age = np.array([(cutoff - row["cutoff"]).total_seconds() / 604800 for row in history])
    weights = np.exp2(-age / 78.0)
    weights /= weights.sum()
    rng = np.random.default_rng(SEED)
    initial = rng.normal(0.0, 0.1, INPUTS * HIDDEN + HIDDEN)
    result = minimize(
        objective,
        initial,
        args=(normalized(x, scale), target, weights, penalty),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": MAX_STEPS, "ftol": 1e-12, "gtol": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"{cutoff}, penalty {penalty}: {result.message}")
    return scale, result


def replay(windows, phase, penalty):
    for row, history in windows:
        if row["study_phase"] != phase:
            continue
        forecast = row["public_ensemble"]
        update = 0.0
        iterations = None
        if row["measured"] and len(history) >= MIN_MEASURED:
            scale, fit_result = fit(history, row["cutoff"], penalty)
            first = fit_result.x[: INPUTS * HIDDEN].reshape(INPUTS, HIDDEN)
            second = fit_result.x[INPUTS * HIDDEN :]
            hidden = np.tanh(normalized(row["x"][:INPUTS], scale) @ first)
            update = float(np.clip(hidden @ second, -CORRECTION_LIMIT, CORRECTION_LIMIT))
            iterations = int(fit_result.nit)
        yield {
            **{key: row[key] for key in ("release_date", "week_ending", "study_phase", "measured", "target")},
            "forecast_cutoff_utc": row["cutoff"].isoformat(),
            "target_known_by_utc": row["known"].isoformat(),
            "latest_training_label_utc": max((q["known"].isoformat() for q in history), default=""),
            "training_weeks": len(history),
            "iterations": iterations,
            "public_ensemble": forecast,
            "coverage_ml": row["coverage_ml"],
            "mlp_correction": update,
            "mlp_forecast": forecast + update,
        }


def make_windows(rows):
    previous = deque()
    for row in rows:
        if not row["cutoff"] < row["known"]:
            raise ValueError(f"Release precedes its cutoff: {row['release_date']}")
        history = tuple(q for q in previous if q["known"] < row["cutoff"])[-WINDOW:]
        yield row, history
        if row["measured"]:
            previous.append(row)


def scores(rows):
    score = {"n": len(rows)}
    observed = np.array([row["target"] for row in rows])
    for key in ("public_ensemble", "coverage_ml", "mlp_forecast"):
        forecast = np.array([row[key] for row in rows])
        residual = forecast - observed
        score[key] = {
            "mae_kbbl": float(1000 * np.mean(np.abs(residual))),
            "rmse_kbbl": float(1000 * np.sqrt(np.mean(residual**2))),
            "direction_accuracy": float(np.mean(np.sign(forecast) == np.sign(observed))),
        }
    return score


def main():
    reference = {row["release_date"]: row for row in read("cushing_coverage_nowcast_forecasts")}
    rows = sorted(coverage_inputs(), key=lambda row: row["cutoff"])
    if len({row["release_date"] for row in rows}) != len(rows):
        raise ValueError("Duplicate release dates")
    for row in rows:
        registered = reference[row["release_date"]]
        if (
            row["cutoff"].isoformat() != registered["forecast_cutoff_utc"]
            or row["target"] != float(registered["target"])
            or row["public_ensemble"] != float(registered["public_ensemble"])
        ):
            raise ValueError(f"Source panel does not reproduce {row['release_date']}")
        row["coverage_ml"] = float(registered["coverage_ml"])

    windows = list(make_windows(rows))
    first_holdout = min(row["cutoff"] for row in rows if row["study_phase"] == "holdout")
    if any(row["known"] >= first_holdout for row in rows if row["study_phase"] == "validation"):
        raise ValueError("Validation results overlap with the holdout cutoff")
    validation = {penalty: list(replay(windows, "validation", penalty)) for penalty in PENALTIES}
    comparison = {penalty: scores(predictions) for penalty, predictions in validation.items()}
    selected = min(PENALTIES, key=lambda penalty: comparison[penalty]["mlp_forecast"]["mae_kbbl"])
    holdout = list(replay(windows, "holdout", selected))
    fitted = [row for row in holdout if row["iterations"] is not None]
    summary = {
        "architecture": [INPUTS, HIDDEN, 1],
        "parameters": INPUTS * HIDDEN + HIDDEN,
        "max_training_weeks": WINDOW,
        "min_training_weeks": MIN_MEASURED,
        "validation": comparison,
        "selected_penalty": selected,
        "validation_fitted_releases": sum(row["iterations"] is not None for row in validation[selected]),
        "holdout_all": scores(holdout),
        "holdout_measured": scores(fitted),
        "holdout_fitted_training_weeks": [min(row["training_weeks"] for row in fitted), max(row["training_weeks"] for row in fitted)],
        "holdout_optimizer_iterations": [min(row["iterations"] for row in fitted), max(row["iterations"] for row in fitted)],
        "holdout_improved_vs_public": sum(abs(row["mlp_forecast"] - row["target"]) < abs(row["public_ensemble"] - row["target"]) for row in fitted),
        "first_holdout_fit": fitted[0],
    }
    export(f"{STEM}_forecasts", holdout)
    (DERIVED / f"{STEM}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
