"""Chronological survey-error and missing-survey networks for weekly Cushing stocks."""

import itertools

import numpy as np
import pandas as pd
from fit_cushing_inventory_network import standardize, transform
from scipy.optimize import minimize

COMMON = (
    "strong_reference",
    "reference_minus_public",
    "lag1",
    "lag2",
    "mean4",
    "vol13",
    "persistence13",
    "midwest_change",
    "gulf_change",
    "basis_move5",
    "level_vs_13",
    "season_sin",
    "season_cos",
    "month_turn",
    "quarter_turn",
    "last_gap",
    "gap4",
    "recent_match",
)
SEEDS = (17, 61, 139)
EPOCHS = 256


def feedback(history, week, cutoff):
    earlier = history[(history.known < cutoff) & (history.week_ending < week) & history.survey.notna()].tail(4)
    if earlier.empty:
        return {"last_gap": 0.0, "gap4": 0.0, "recent_match": 0.0}
    ages = (pd.Timestamp(week) - pd.to_datetime(earlier.week_ending)).dt.days.to_numpy() / 7
    gaps = (earlier.target - earlier.survey).to_numpy()
    recent = ages[-1] == 1
    return {
        "last_gap": float(gaps[-1]) if recent else 0.0,
        "recent_match": float(recent),
        "gap4": float(np.sum(np.exp2(-ages / 2) * gaps) / 2),
    }


def forward(params, x):
    w1, b1, w2, b2, w3, b3 = params
    a = np.tanh(x @ w1 + b1)
    b = np.tanh(a @ w2 + b2)
    return a, b, b @ w3 + b3


def fit_epochs(x, y, sample_weights, seed, epochs, stopping=None):
    rng = np.random.default_rng(seed)
    dims = [x.shape[1], 32, 16, y.shape[1]]
    params = []
    for a, b in itertools.pairwise(dims):
        params += [rng.normal(0, 1 / np.sqrt(a), (a, b)), np.zeros(b)]
    params[-2] *= 0.05
    first = [np.zeros_like(p) for p in params]
    second = [p.copy() for p in first]
    logs = []
    for epoch in range(1, epochs + 1):
        a, b, pred = forward(params, x)
        error = pred - y
        # Pseudo-Huber approaches absolute error in the tails. Missing teacher targets have zero weight.
        grad = sample_weights * error / np.sqrt(0.16**2 + error**2)
        db = (grad @ params[4].T) * (1 - b * b)
        da = (db @ params[2].T) * (1 - a * a)
        gradients = [x.T @ da, da.sum(0), a.T @ db, db.sum(0), b.T @ grad, grad.sum(0)]
        for j, (p, g) in enumerate(zip(params, gradients)):
            if j % 2 == 0:
                g += 0.012 * p
            first[j] = 0.9 * first[j] + 0.1 * g
            second[j] = 0.999 * second[j] + 0.001 * g * g
            p -= 0.0015 * (first[j] / (1 - 0.9**epoch)) / (np.sqrt(second[j] / (1 - 0.999**epoch)) + 1e-8)
        if stopping is not None:
            xs, ys = stopping
            mae = np.mean(np.abs(forward(params, xs)[-1][:, 0] - ys))
            logs.append(float(1000 * mae))
    return params, logs


def fit_head(history, cutoff, present, common=COMMON):
    rows = history[(history.known < cutoff) & history.strong_reference.notna()].sort_values("cutoff")
    if present:
        rows = rows[rows.survey.notna()]
    if len(rows) < 40:
        raise ValueError("Insufficient prior observations to fit a survey correction")
    fields = list(common) + (["survey"] if present else [])
    anchor = rows.survey if present else rows.strong_reference
    y = (rows.target - anchor).to_numpy()[:, None]
    if not present:
        # Shared hidden layers learn to predict the prior surveys as an auxiliary task.
        y = np.column_stack([y, rows.survey - anchor])
    x = rows[fields].to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("Invalid training features")
    n_stop = max(12, min(39, len(rows) // 4))
    start = len(rows) - n_stop
    train = (rows.known < rows.cutoff.iloc[start]).to_numpy()
    weights = np.isfinite(y).astype(float)
    ages = (cutoff - rows.cutoff).dt.total_seconds().to_numpy() / 604800
    weights *= np.exp2(-ages[:, None] / 156.0)

    def normalized(w):
        w = w / np.maximum(w.sum(axis=0), 1)
        if not present:
            w[:, 1] *= 0.25
        return w

    y = np.nan_to_num(y, nan=0.0)
    centre, scale = standardize(x[train])
    z = transform(x, centre, scale)
    stop = (z[start:], y[start:, 0])
    logs = [fit_epochs(z[train], y[train], normalized(weights[train]), seed, EPOCHS, stop)[1] for seed in SEEDS]
    # Same epoch for every seed, selected on the mean stopping loss.
    loss = np.mean(logs, axis=0)
    epoch = int(np.argmin(loss)) + 1
    centre, scale = standardize(x)
    z = transform(x, centre, scale)
    members = []
    for seed in SEEDS:
        params, _ = fit_epochs(z, y, normalized(weights), seed, epoch)
        members.append([p.tolist() for p in params])
    return {
        "features": fields,
        "centre": centre.tolist(),
        "scale": scale.tolist(),
        "members": members,
        "epochs": epoch,
        "training_weeks": len(rows),
        "stopping_weeks": n_stop,
        "latest_label_utc": rows.known.max().isoformat(),
    }, loss


def estimate(head, rows, anchor):
    if len(rows) == 0:
        return np.empty(0)
    if pd.Timestamp(head["latest_label_utc"]) >= rows.cutoff.min():
        raise ValueError("The network contains a label later than the forecast cutoff")
    x = rows[head["features"]].to_numpy()
    if not np.isfinite(x).all():
        raise ValueError("A residual-network input is missing")
    x = transform(x, head["centre"], head["scale"])
    predicted = [forward([np.asarray(p) for p in member], x)[-1][:, 0] for member in head["members"]]
    return np.asarray(anchor) + np.clip(np.mean(predicted, axis=0), -1.5, 1.5)


def feedback_design(rows):
    return np.column_stack([np.ones(len(rows)), rows.strong_reference - rows.survey, -rows.last_gap, rows.gap4])


def fit_feedback_head(history, cutoff):
    rows = history[(history.known < cutoff) & history.strong_reference.notna() & history.survey.notna()]
    if len(rows) < 40:
        raise ValueError("Insufficient settled survey errors")
    x = feedback_design(rows)
    y = (rows.target - rows.survey).to_numpy()
    age = (cutoff - rows.cutoff).dt.total_seconds().to_numpy() / 604800
    w = np.exp2(-age / 104)
    w /= w.sum()
    prior = np.array([0.0, 0.1, 0.25, 0.0])

    def objective(beta):
        err = x @ beta - y
        norm = np.hypot(0.12, err)
        offset = beta - prior
        return w @ norm + 0.025 * (offset @ offset), x.T @ (w * err / norm) + 0.05 * offset

    fit = minimize(
        objective, prior, jac=True, method="L-BFGS-B", bounds=[(-0.25, 0.25), (0, 0.5), (0, 1), (-0.5, 0.5)]
    )
    if not fit.success:
        raise RuntimeError(fit.message)
    return {
        "coefficients": fit.x.tolist(),
        "latest_label_utc": rows.known.max().isoformat(),
        "training_weeks": len(rows),
    }


def feedback_estimate(head, rows):
    if len(rows) and pd.Timestamp(head["latest_label_utc"]) >= rows.cutoff.min():
        raise ValueError("Survey calibration contains an unavailable label")
    return rows.survey.to_numpy() + feedback_design(rows) @ head["coefficients"]
