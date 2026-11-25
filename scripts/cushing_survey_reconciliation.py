"""Causal stock-level discrepancy estimates and robust conditional medians for Cushing surveys."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# The surveyed weekly difference is a change in a latent discrepancy between two stock levels.
# Each pair specifies weekly persistence and the variance ratio of independent change noise to level noise.
FILTERS = {"quick": (0.0, 0.18), "medium": (0.55, 0.15), "persistent": (0.88, 0.06), "smooth": (0.96, 0.03)}
CORE = ("ref_gap", "quick", "persistent", "peak_quote")
FLOW = (
    "outside_hub",
    "midwest_runs_change",
    "gulf_runs_change",
    "national",
    "national_gap",
    "slope_change",
    "bend_change",
    "thermal",
    "late_week",
    "product_balance",
    "distillate",
    "roof_disagreement",
)
TRUST = (
    "medium",
    "smooth",
    "last_observed_gap",
    "quote_vol",
    "source_round",
    "old_quote",
    "gap_run",
    "weekend",
    "month_turn",
)
LAYOUTS = {
    "level": CORE,
    "outliers": (*CORE, *TRUST),
    "flows": (*CORE, *FLOW),
    "combined": (*CORE, *FLOW, *TRUST),
}
MISSING = (
    "strong_reference",
    "national",
    "outside_hub",
    "midwest_runs_change",
    "gulf_runs_change",
    "slope",
    "slope_change",
    "bend_change",
    "thermal",
    "product_balance",
    "month_turn",
    "lag1",
    "mean4",
    "level_vs_13",
    "quarter_turn",
    "weekend",
)


def states(settled, week, when):
    end = pd.Timestamp(week)
    labels = settled[(settled.known < when) & (settled.week_ending < week) & settled.survey.notna()].tail(160)
    days = pd.to_datetime(labels.week_ending)
    ages = ((end - days).dt.days / 7).to_numpy()
    if len(ages) and (np.any(ages < 1) or np.any(np.diff(ages) >= 0)):
        raise ValueError("Survey observations must be distinct, chronological reporting weeks")
    gaps = (labels.target - labels.survey).to_numpy()
    result = {"previous_survey": float(labels.survey.iloc[-1]) if len(ages) and ages[-1] == 1 else np.nan}
    for name, (phi, ratio) in FILTERS.items():
        m = 0.0
        var = 1 / (1 - phi * phi)
        last = None
        for age, g in zip(ages, gaps):
            if last is not None:
                missed = max(round(last - age) - 1, 0)
                decay = phi**missed
                m *= decay
                var = decay * decay * var + (1 - decay * decay) / (1 - phi * phi)
            # The sign and correlation of change noise matter: both the measurement and the next level contain w.
            s = (phi - 1) ** 2 * var + 1 + ratio
            cov = 1 + phi * (phi - 1) * var
            innovation = np.clip(
                g - (phi - 1) * m, -(1.50 if name == "quick" else 1.10), 1.50 if name == "quick" else 1.10
            )
            m = phi * m + cov / s * innovation
            var = max(0.0, phi * phi * var + 1 - cov * cov / s)
            last = age
        decay = phi ** max(0, round(ages[-1]) - 1) if len(ages) else 1.0
        result[name] = (phi - 1) * decay * m
    result["last_observed_gap"] = float(np.clip(gaps[-1], -1.5, 1.5)) if len(ages) and ages[-1] == 1 else 0.0
    run = 0.0
    for i, (age, g) in enumerate(zip(ages[::-1], gaps[::-1]), 1):
        if age != i or i > 8:
            break
        run += np.clip(g, -1.2, 1.2)
    result["gap_run"] = run
    return result


def features(row, external):
    def num(name):
        v = external.get(name, np.nan)
        return float(v) if pd.notna(v) else 0.0

    ref = float(row["strong_reference"])
    survey = row["survey"]
    q = float(survey) if pd.notna(survey) else ref
    vol = max(float(row["vol13"]), 0.45)
    nat = row.get("national_kbbl", np.nan)
    when = pd.Timestamp(row["cutoff"])
    ts = row.get("survey_time")
    age = (when - pd.Timestamp(ts)).total_seconds() / 3600 if pd.notna(survey) and ts else 0.0
    return {
        "ref_gap": ref - q,
        "peak_quote": np.sign(q) * max(abs(q) - 1.25 * vol, 0.0),
        "quote_vol": q / vol,
        "source_round": float(np.isclose(10 * q, np.rint(10 * q), atol=1e-7, rtol=0)),
        "old_quote": min(max((age - 8) / 18.0, 0.0), 2.0),
        "national": (nat / 5000 if pd.notna(nat) else 0.0),
        "national_gap": (
            (nat / 1000 - num("national_prior")) / 5
            if pd.notna(nat) and pd.notna(external.get("national_prior"))
            else 0.0
        ),
        "outside_hub": num("ex_cushing"),
        "midwest_runs_change": num("midwest_runs_change"),
        "gulf_runs_change": num("gulf_runs_change"),
        "slope": num("slope"),
        "slope_change": num("slope_change"),
        "bend_change": num("bend_change"),
        "thermal": 0.00075 * float(row["level"]) * num("temp_change"),
        "roof_disagreement": (num("optical_move") - q) * min(1.0, 4 * num("optical_coverage"))
        if num("optical_measured")
        else 0.0,
        "weekend": float((pd.Timestamp(row["release_date"]) - pd.Timestamp(row["week_ending"])).days - 5),
        "late_week": float(when.tz_convert("America/New_York").day_of_week > 2),
    }


def fit(rows, cutoff, family, strength, half_life, nonlinear=False, missing=False):
    d = rows[(rows.known < cutoff) & rows.strong_reference.notna()]
    if not missing:
        d = d[d.survey.notna()]
    if len(d) < 35:
        raise ValueError("Insufficient settled survey history")
    fields = MISSING if missing else LAYOUTS[family]
    x = d[list(fields)].to_numpy()
    anchor = d.strong_reference if missing else d.survey
    y = (d.target - anchor).to_numpy()
    ages = (cutoff - d.known).dt.total_seconds().to_numpy() / 604800.0
    wt = np.exp2(-ages / half_life)
    wt /= wt.sum()
    med = np.median(x, axis=0)
    scale = np.maximum(np.quantile(x, 0.85, axis=0) - np.quantile(x, 0.15, axis=0), 0.25)
    z = np.clip((x - med) / scale, -4.0, 4.0)
    design = np.column_stack([np.ones(len(z)), z])
    # Shrink towards a mild survey amplitude correction and the latent measurement estimate.
    prior = np.zeros(design.shape[1])
    if not missing:
        for k, b in (("ref_gap", 0.1), ("quick", 0.3), ("persistent", 0.3)):
            j = fields.index(k)
            prior[j + 1] = b * scale[j]
            prior[0] += b * med[j]

    def objective(beta):
        e = design @ beta - y
        dist = np.hypot(e, 0.08)
        delta = beta - prior
        return wt @ dist + strength * (delta @ delta) / 2, design.T @ (wt * e / dist) + strength * delta

    sol = minimize(objective, prior, jac=True, method="L-BFGS-B", options={"ftol": 1e-12, "maxiter": 300})
    if not sol.success:
        raise RuntimeError(sol.message)
    b = sol.x
    leaves = []
    if nonlinear:
        # Small boosted median stumps around the fitted measurement model.
        residual = y - design @ b

        def median(vals, weights):
            ix = np.argsort(vals, kind="stable")
            return vals[ix[np.searchsorted(np.cumsum(weights[ix]), 0.5 * weights.sum())]]

        for _ in range(12):
            best = (float(wt @ np.abs(residual)), None)
            for j in range(z.shape[1]):
                for cut in np.unique(np.quantile(z[:, j], [0.2, 0.4, 0.6, 0.8])):
                    left = z[:, j] <= cut
                    if min(left.sum(), (~left).sum()) < 18:
                        continue
                    lo = 0.18 * float(median(residual[left], wt[left]))
                    hi = 0.18 * float(median(residual[~left], wt[~left]))
                    loss = float(wt @ np.abs(residual - np.where(left, lo, hi)))
                    if loss < best[0]:
                        best = (loss, (j, float(cut), lo, hi))
            if best[1] is None:
                break
            j, cut, lo, hi = best[1]
            leaves.append(best[1])
            residual -= np.where(z[:, j] <= cut, lo, hi)
    return {
        "fields": list(fields),
        "med": med.tolist(),
        "scale": scale.tolist(),
        "beta": b.tolist(),
        "stumps": leaves,
        "missing": missing,
        "latest": d.known.max().isoformat(),
    }


def predict(model, rows):
    if not len(rows):
        return np.empty(0)
    if pd.Timestamp(model["latest"]) >= rows.cutoff.min():
        raise ValueError("Fit contains a future observation")
    z = np.clip((rows[model["fields"]].to_numpy() - model["med"]) / model["scale"], -4, 4)
    pred = np.column_stack([np.ones(len(z)), z]) @ model["beta"]
    for j, cut, lo, hi in model["stumps"]:
        pred += np.where(z[:, j] <= cut, lo, hi)
    return (rows.strong_reference if model["missing"] else rows.survey).to_numpy() + np.clip(pred, -1.5, 1.5)


OVERLAYS = {
    "slow_stock": ("stock_gap",),
    "stock_and_trust": ("stock_gap", "calibration"),
    "latent_and_trust": ("latent_gap", "calibration"),
    "conditional_stock": ("stock_gap", "calibration", "slope_change", "thermal", "product_balance"),
    "survey_and_roofs": ("stock_gap", "calibration", "roof_disagreement"),
    "weekly_timing": ("stock_gap", "calibration", "survey_acceleration", "last_observed_gap"),
    "variable_gain": ("stock_gap", "stock_recent", "stock_agreement", "survey_acceleration"),
}


def overlay_design(rows, fields):
    stock = np.clip(-rows.state16 / 3.0, -1.1, 1.1)
    recent = np.clip(-rows.state2, -1.1, 1.1)
    covariates = {
        "stock_gap": stock,
        "stock_recent": recent,
        "stock_agreement": stock * np.sign(stock * recent),
        "survey_acceleration": (rows.previous_survey - rows.survey).fillna(0).clip(-2, 2),
        "last_observed_gap": rows.last_observed_gap.clip(-1.0, 1.0),
        "latent_gap": np.clip(0.5 * (-rows.gap_run / 3.0 + rows.persistent), -0.9, 0.9),
        "calibration": np.clip(rows.precise - rows.survey, -0.8, 0.8),
        "roof_disagreement": rows.roof_disagreement.clip(-1.0, 1.0),
        "slope_change": rows.slope_change.clip(-1.0, 1.0),
        "thermal": rows.thermal.clip(-1.0, 1.0),
        "product_balance": rows.product_balance.clip(-2.0, 2.0),
    }
    return np.column_stack([covariates[k] for k in fields])


def fit_overlay(replay_rows, cutoff, name):
    history = replay_rows[(replay_rows.known < cutoff) & replay_rows.survey.notna()]
    fields = OVERLAYS[name]
    if len(history) < 16:
        raise ValueError("Too few past forecasts for residual calibration")
    x = overlay_design(history, fields)
    y = (history.target - history.precise).to_numpy()
    ages = (cutoff - history.known).dt.total_seconds().to_numpy() / 604800.0
    w = np.exp2(-ages / 104.0)
    # The prior shrinks poorly identified conditional effects; the intercept is fixed at zero.
    # All other gains multiply a signed surprise in million barrels.
    constants = {
        "stock_gap": (0.25, 0.0, 1.25),
        "latent_gap": (0.30, 0.0, 1.50),
        "calibration": (0.12, -0.25, 1.0),
        "slope_change": (0.0, -0.25, 0.25),
        "thermal": (0.0, -0.50, 0.50),
        "product_balance": (0.0, -0.15, 0.15),
        "roof_disagreement": (0.0, 0.0, 0.50),
        "stock_recent": (0.0, -0.35, 0.75),
        "stock_agreement": (0.0, -0.50, 0.50),
        "survey_acceleration": (0.0, -0.35, 0.35),
        "last_observed_gap": (0.0, -0.50, 0.50),
    }
    center = np.array([constants[k][0] for k in fields])

    def loss(beta):
        e = x @ beta - y
        dist = np.hypot(e, 0.09)
        dev = beta - center
        return w @ dist + 1.5 * (dev @ dev), x.T @ (w * e / dist) + 3 * dev

    fit = minimize(loss, center, jac=True, method="L-BFGS-B", bounds=[constants[k][1:] for k in fields])
    if not fit.success:
        raise RuntimeError(fit.message)
    return {
        "fields": list(fields),
        "beta": fit.x.tolist(),
        "latest": history.known.max().isoformat(),
        "count": len(history),
    }


def apply_overlay(model, rows):
    if not len(rows):
        return np.empty(0)
    if pd.Timestamp(model["latest"]) >= rows.cutoff.min():
        raise ValueError("Residual overlay trained after decision")
    delta = overlay_design(rows, model["fields"]) @ model["beta"]
    return rows.precise.to_numpy() + np.clip(delta, -0.75, 0.75)
