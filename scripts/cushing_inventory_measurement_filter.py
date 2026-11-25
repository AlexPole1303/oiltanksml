"""Fit robust residual filters to settled weekly inventory measurement discrepancies."""

import numpy as np
import pandas as pd

CORE = ["ref_gap", "state2", "state6", "state16", "last_gap_local"]
RICH = [
    *CORE,
    "positive_quote",
    "negative_quote",
    "prior_error",
    "quarter_turn",
    "year_turn",
    "national",
    "national_present",
    "midwest_change",
    "basis_move5",
    "vol13",
]
RELIABILITY = [
    *RICH,
    "rounded",
    "round_gap",
    "noisy_gap",
    "extreme_gap",
    "state32",
    "stale_weeks",
    "prior_noise",
]
FALLBACK = [
    "strong_reference",
    "national",
    "national_present",
    "lag1",
    "lag2",
    "mean4",
    "midwest_change",
    "gulf_change",
    "vol13",
    "prior_error",
    "state6",
    "quarter_turn",
    "year_turn",
]
SAT = ["radar", "radar_fresh"]
BALANCE_LAGS = [f"error_lag{k}" for k in range(1, 9)]
LAYOUTS = {
    "short_memory": CORE,
    "stock_balance": RICH,
    "random_features": RICH,
    "reliability": RELIABILITY,
    "regional_balance": [
        "ref_gap",
        *BALANCE_LAGS,
        "state16",
        "survey",
        "season_cos",
        "season_sin",
        "level",
        "level_vs_13",
        "midwest_mean4",
        "gulf_mean4",
        "midwest_change",
        "gulf_change",
        "level_quote",
        "basis_quote",
        "national",
        "national_present",
    ],
    "fallback": FALLBACK,
    "radar_sensitivity": [*RICH, *SAT],
    "radar_fallback": [*FALLBACK, *SAT],
}
SEEDS = (71, 137)


def state(history, week, cutoff):
    settled = history[(history.known < cutoff) & (history.week_ending < week) & history.survey.notna()].tail(
        104
    )
    gaps = np.clip((settled.target - settled.survey).to_numpy(), -1.5, 1.5)
    age = (pd.Timestamp(week) - pd.to_datetime(settled.week_ending)).dt.days.to_numpy() / 7
    values = {f"state{k}": float(np.sum(gaps * np.exp2(-(age - 1) / k))) for k in (2, 6, 16, 32)}
    by_week = dict(zip(age, gaps))
    values.update({field: float(by_week.get(k, 0.0)) for k, field in enumerate(BALANCE_LAGS, 1)})
    values["prior_noise"] = float(np.median(np.abs(gaps[-13:]))) if len(gaps) else 0.3
    values["stale_weeks"] = min(8.0, float(age[-1] - 1)) if len(age) else 8.0
    values["last_gap_local"] = float(gaps[-1]) if len(age) and age[-1] == 1 else 0.0
    return values


def covariates(rows):
    rows = rows.copy()
    rows["ref_gap"] = rows.strong_reference - rows.survey
    rows["rounded"] = np.isclose(rows.survey * 10, (rows.survey * 10).round(), rtol=0, atol=1e-7).astype(float)
    rows["round_gap"] = rows.ref_gap * rows.rounded
    rows["noisy_gap"] = rows.ref_gap * (rows.prior_noise - 0.3)
    rows["extreme_gap"] = rows.ref_gap * (rows.survey.abs() / rows.vol13.clip(lower=0.35) - 1).clip(0.0, 2.0)
    rows["positive_quote"] = (rows.survey - 1.0).clip(lower=0.0)
    rows["negative_quote"] = (rows.survey + 1.0).clip(upper=0.0)
    rows["national_present"] = rows.national_kbbl.notna().astype(float)
    rows["national"] = (rows.national_kbbl.fillna(0.0) / 5000).clip(-4, 4)
    rows["level_quote"] = rows.survey * rows.level_vs_13
    rows["basis_quote"] = rows.survey * rows.basis_move5
    rows["radar_fresh"] = rows.radar_weekly_million.notna().astype(float)
    rows["radar"] = (
        rows.radar_weekly_million.fillna(0.0) * np.exp2(-rows.radar_age_days.fillna(99.0) / 7)
    ).clip(-4, 4)
    return rows


def design(x, randomized):
    x = np.clip(x, -5.0, 5.0)
    cols = [np.ones((len(x), 1)), x]
    if randomized:
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            w = rng.normal(0, 1 / np.sqrt(x.shape[1]), (x.shape[1], 24))
            b = rng.normal(0, 0.35, 24)
            cols.append(np.tanh(x @ w + b))
    return np.column_stack(cols)


def fit(rows, cutoff, family, strength, half_life):
    missing = family in ("fallback", "radar_fallback")
    history = rows[(rows.known < cutoff) & rows.strong_reference.notna()]
    if not missing:
        history = history[history.survey.notna()]
    if len(history) < 35:
        raise ValueError("Too few settled dates")
    fields = LAYOUTS[family]
    x = history[fields].to_numpy()
    centre = np.median(x, axis=0)
    scale = np.maximum(np.std(x, axis=0), 0.25)
    z = design((x - centre) / scale, family == "random_features")
    y = (history.target - (history.strong_reference if missing else history.survey)).to_numpy()
    age = (cutoff - history.known).dt.total_seconds().to_numpy() / 604800
    wt = np.exp2(-age / half_life)
    penalty = np.eye(z.shape[1]) * strength * wt.sum()
    penalty[0, 0] *= 0.1
    beta = np.zeros(z.shape[1])
    for _ in range(64):
        w = wt / np.hypot(0.12, z @ beta - y)
        next_beta = np.linalg.solve(z.T @ (w[:, None] * z) + penalty, z.T @ (w * y))
        gap = np.max(np.abs(next_beta - beta))
        beta = next_beta
        if gap < 1e-8:
            break
    return {
        "fields": fields,
        "family": family,
        "centre": centre.tolist(),
        "scale": scale.tolist(),
        "beta": beta.tolist(),
        "training_known_utc": history.known.max().isoformat(),
    }


def predict(model, rows):
    if len(rows) and pd.Timestamp(model["training_known_utc"]) >= rows.cutoff.min():
        raise ValueError("Residual filter trained after decision time")
    z = design(
        (rows[model["fields"]].to_numpy() - model["centre"]) / model["scale"],
        model["family"] == "random_features",
    )
    anchor = rows.strong_reference if model["family"] in ("fallback", "radar_fallback") else rows.survey
    return anchor.to_numpy() + np.clip(z @ model["beta"], -1.7, 1.7)
