"""Calibrate the project's dated hybrid inventory forecast with two supervised networks."""
import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fit_cushing_inventory_network import (DERIVED, MODEL_DIR, PUBLIC, REFIT, WIDTHS, direct_prediction, fit_model,
                                          load_panel, metrics, save_json)
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

REFERENCE = "hybrid_positive_transfer_dynamics_only"
BASIS = ("basis_level", "basis_move1", "basis_move5", "basis_move20", "wti_move5", "brent_move5",
         "wti_volatility20", "quote_age_days")
CALENDAR = ("month_turn", "quarter_turn", "year_turn", "thanksgiving_week", "season_sin2", "season_cos2")
ERROR = ("prior_error", "error_mean4", "error_median13", "error_mean26")
DIRECT = (*PUBLIC, *CALENDAR, *BASIS, "vintage_present")
CORRECTION = ("strong_reference", "reference_minus_public", "lag1", "mean4", "vol13", "persistence13",
              "midwest_change", "gulf_change", *CALENDAR, *BASIS, *ERROR)
PACKAGE = MODEL_DIR / "calibrated_weights.json"
REPORT = DERIVED / "cushing_calibrated_network_summary.json"
POINTS = DERIVED / "cushing_calibrated_network_forecasts.csv"


def calendar(week):
    day = date.fromisoformat(week)
    previous, after = day - timedelta(days=7), day + timedelta(days=7)
    # Signed indicators distinguish the week before a boundary from the week after.
    def turn(func):
        return int(func(day) != func(previous)) - int(func(after) != func(day))
    nov1 = date(day.year, 11, 1)
    thanksgiving = nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21)
    angle = 4 * np.pi * day.timetuple().tm_yday / 365.25
    return dict(zip(CALENDAR, [turn(lambda d: d.month), turn(lambda d: (d.month - 1) // 3),
                               turn(lambda d: d.year), int(previous < thanksgiving <= day),
                               np.sin(angle), np.cos(angle)], strict=True))


def past_errors(data, cutoff):
    past = data[(data.known < cutoff) & data.strong_reference.notna()].tail(26)
    e = (past.target - past.strong_reference).to_numpy()
    values = [e[-1], e[-4:].mean(), np.median(e[-13:]), e.mean()] if len(e) else [0.] * 4
    return dict(zip(ERROR, values, strict=True))


def enrich(data):
    if not np.isfinite(data.loc[data.strong_reference.notna(), list(BASIS)]).all().all():
        raise ValueError("Missing archived market input for a dated reference forecast")
    extra = [calendar(r.week_ending) | past_errors(data, r.cutoff) for r in data.itertuples()]
    panel = pd.concat([data, pd.DataFrame(extra, index=data.index)], axis=1)
    panel["vintage_present"] = panel.basis_level.notna().astype(float)
    panel[list(BASIS)] = panel[list(BASIS)].fillna(0.0)
    panel["reference_minus_public"] = panel.strong_reference - panel.public_ensemble
    return panel


def pair(history, cutoff, width, linear=False):
    priced = history[history.strong_reference.notna()]
    return dict(direct=fit_model(history, cutoff, width, features=DIRECT, linear=linear),
                correction=fit_model(priced, cutoff, width, features=CORRECTION, anchor="strong_reference", linear=linear))


def predict_heads(models, rows):
    head = models["direct"]
    direct = direct_prediction(head, rows[head["features"]].to_numpy(), rows.anchor.to_numpy())
    head = models["correction"]
    correction = direct_prediction(head, rows[head["features"]].to_numpy(), 0.)
    return direct, np.clip(correction, -0.75, 0.75)


def replay(data, test, width, linear=False):
    result = []
    for start in range(0, len(test), REFIT):
        rows = test.iloc[start:start + REFIT].copy()
        cutoff = rows.cutoff.iloc[0]
        history = data[(data.cutoff < cutoff) & (data.known < cutoff)]
        models = pair(history, cutoff, width, linear)
        rows["direct"], rows["correction"] = predict_heads(models, rows)
        rows["training_known"] = models["direct"]["latest_training_label_utc"]
        rows["long_training_weeks"] = len(history)
        rows["residual_training_weeks"] = models["correction"]["training_weeks"]
        result.append(rows)
    return pd.concat(result)


def combined(rows, mix):
    return (rows.strong_reference + mix["direct_share"] * (rows.direct - rows.strong_reference)
            + mix["correction_share"] * rows.correction)


def select(pre):
    validation = pre[(pre.release_date >= "2021-01-01") & pre.strong_reference.notna()]
    if validation.empty or pre.release_date.max() >= "2024-01-01" or (validation.phase == "holdout").any():
        raise ValueError("Calibration selection can only see pre-2024 outcomes")
    trials = []
    for width in WIDTHS:
        rows = replay(pre, validation, width)
        for direct in (0.0, 0.25, 0.5, 0.75, 1.0):
            for correction in (0.0, 0.25, 0.5, 0.75, 1.0):
                if direct + correction > 1:
                    continue
                trial = dict(hidden_layers=list(width), direct_share=direct, correction_share=correction)
                ae = 1000 * abs(combined(rows, trial) - rows.target)
                years = ae.groupby(rows.release_date.str[:4]).mean()
                trials.append(trial | dict(year_mae_kbbl=years.to_dict(), mean_year_mae_kbbl=float(years.mean())))
    best = min(trials, key=lambda r: (r["mean_year_mae_kbbl"], r["direct_share"] + r["correction_share"], sum(r["hidden_layers"])))
    return dict(chosen=best, first=validation.release_date.min(), last=validation.release_date.max(),
                validation_weeks=len(validation), trials=trials)


def train():
    data = enrich(load_panel())
    pre = data[data.release_date < "2024-01-01"]
    selection = select(pre)
    mix = selection["chosen"]
    width = tuple(mix["hidden_layers"])
    print("Locked pre-2024:", mix, flush=True)
    test = data[data.phase == "holdout"]
    if pre.known.max() >= test.cutoff.min():
        raise ValueError("Selection overlaps evaluation")
    rows = replay(data, test, width)
    linear = replay(data, test, width, linear=True)
    rows["linear_matched"] = combined(linear, mix)
    rows["calibrated"] = combined(rows, mix)
    first = pd.read_csv(DERIVED / "cushing_inventory_network_forecasts.csv").set_index("release_date").loc[rows.release_date]
    if (not np.allclose(first.target, rows.target, rtol=0, atol=1e-12)
            or not np.array_equal(pd.to_datetime(first.cutoff, utc=True).to_numpy(), rows.cutoff.to_numpy())):
        raise ValueError("First-generation predictions have a different target or cutoff")
    rows["first_generation"] = first.neural_blend.to_numpy()
    methods = ("previous_mlp", "public_ensemble", "first_generation", "strong_reference", "linear_matched", "calibrated")
    gains = 1000 * abs(rows.calibrated - rows.target)
    paired = {k: block_bootstrap_mean(gains - 1000 * abs(rows[k] - rows.target)) for k in methods[:-1]}
    splits = {"all": rows, **{str(y): g for y, g in rows.groupby(rows.release_date.str[:4])},
              "with_b04": rows[rows.measured == 1], "without_b04": rows[rows.measured == 0]}
    report = dict(status="exploratory_previously_examined_holdout", selection=selection, reference=REFERENCE,
                  scores={k: {m: metrics(g, m) for m in methods} for k, g in splits.items()}, paired_delta_95_kbbl=paired)
    cols = ["release_date", "week_ending", "cutoff", "known", "training_known", "long_training_weeks",
            "residual_training_weeks", "target", "direct", "correction", *methods]
    rows[cols].to_csv(POINTS, index=False)
    save_json(REPORT, report)
    ready = data.known.max() + pd.Timedelta(seconds=1)
    save_json(PACKAGE, dict(format_version=1, reference=REFERENCE, mix=mix, **pair(data, ready, width)))
    print(json.dumps(report["scores"]["all"], indent=2))


def predict(record, model):
    row = pd.DataFrame([record])
    cutoff = pd.Timestamp(record["cutoff"])
    stamps = [pd.Timestamp(record[k]) for k in ("latest_public_input", "latest_residual_input", "market_vintage_end_utc")]
    stamps += [pd.Timestamp(model[k]["latest_training_label_utc"]) for k in ("direct", "correction")]
    if cutoff.tzinfo is None or any(t.tzinfo is None or not t < cutoff for t in stamps):
        raise ValueError("Training labels, market vintage and inventory inputs must precede prediction")
    latest_reference = pd.Timestamp(record["reference_available_utc"])
    if latest_reference.tzinfo is None or not latest_reference <= cutoff:
        raise ValueError("The hybrid reference had not been computed by the requested cutoff")
    required = list(dict.fromkeys((*DIRECT, *CORRECTION, "anchor")))
    if not np.isfinite(row[required].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite prediction features")
    row["direct"], row["correction"] = predict_heads(model, row)
    return dict(cutoff=cutoff.isoformat(), forecast_kbbl=float(combined(row, model["mix"]).iloc[0] * 1000),
                reference_kbbl=float(row.strong_reference.iloc[0] * 1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", type=Path, help="JSON object with unlabeled pre-release features")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.predict:
            print(json.dumps(predict(json.loads(args.predict.read_text()), json.loads(PACKAGE.read_text())), indent=2))
        else:
            train()
