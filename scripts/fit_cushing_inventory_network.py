"""Train, evaluate and apply a release-vintage weekly Cushing inventory network."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from collect_public import ROOT, write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

MODEL_DIR = ROOT / "models/cushing_inventory"
PANEL = MODEL_DIR / "weekly_inputs.csv"
WEIGHTS = MODEL_DIR / "weights.json"
DERIVED = ROOT / "data/derived"
PUBLIC = ("lag1", "lag2", "lag3", "mean4", "mean13", "vol13", "persistence13", "persistence52",
          "level", "level_vs_13", "midwest_change", "gulf_change", "midwest_mean4", "gulf_mean4",
          "season_sin", "season_cos", "lag_age_weeks")
OPTICAL = ("raw", "smooth", "moderated", "coverage_weighted", "coherent", "coverage", "measured",
           "supplement_change", "supplement_roofs")
WIDTHS = ((16,), (32, 16), (64, 32))
MIX = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (17, 61, 139)
REFIT = 13
STOP_WEEKS = 39
MAX_EPOCHS = 100


def save_json(path, value):
    write(path, (json.dumps(value, allow_nan=False, indent=2) + "\n").encode())


def load_panel():
    data = pd.read_csv(PANEL)
    for col in ("cutoff", "known", "latest_public_input"):
        data[col] = pd.to_datetime(data[col], utc=True)
    if (data.release_date.duplicated().any() or not data.cutoff.is_monotonic_increasing
            or not (data.latest_public_input < data.cutoff).all() or not (data.cutoff < data.known).all()
            or not np.isfinite(data[[*PUBLIC, *OPTICAL, "target", "anchor"]]).all().all()):
        raise ValueError("The training panel has invalid values or timestamps")
    return data


def standardize(x):
    centre = np.median(x, axis=0)
    scale = np.maximum(np.sqrt(np.mean((x - centre) ** 2, axis=0)), 0.05)
    return centre, scale


def transform(x, centre, scale):
    return np.clip((x - centre) / scale, -4.0, 4.0)


def new_net(width, seed):
    return MLPRegressor(hidden_layer_sizes=width, activation="tanh", solver="adam", batch_size=64,
                        alpha=2.0, learning_rate_init=0.002, random_state=np.random.RandomState(seed))


def fit_member(x, y, ages, width, seed):
    # Select the epoch on the most recent settled block. Fit preprocessing again for the final refit.
    train = slice(None, -STOP_WEEKS)
    stop = slice(-STOP_WEEKS, None)
    centre, scale = standardize(x[train])
    z = transform(x, centre, scale)
    weights = np.exp2(-ages / 156.0)
    target = np.clip(y, -3.0, 3.0)
    model = new_net(width, seed)
    best_loss, best_epoch = np.inf, 1
    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(z[train], target[train], sample_weight=weights[train] / weights[train].mean())
        loss = np.mean(np.abs(model.predict(z[stop]) - y[stop]))
        if loss < best_loss - 0.001:
            best_loss, best_epoch = loss, epoch
        if epoch - best_epoch >= 10:
            break
    centre, scale = standardize(x)
    z = transform(x, centre, scale)
    model = new_net(width, seed)
    for _ in range(best_epoch):
        model.partial_fit(z, target, sample_weight=weights / weights.mean())
    return dict(epochs=best_epoch, weights=[a.tolist() for a in model.coefs_],
                biases=[a.tolist() for a in model.intercepts_])


def fit_model(history, cutoff, width, use_optics=True, linear=False, *, features=None, anchor="anchor"):
    if len(history) < 104 or not (history.known < cutoff).all():
        raise ValueError("Insufficient strictly pre-cutoff training history")
    features = list(PUBLIC + (OPTICAL if use_optics else ())) if features is None else list(features)
    x = history[features].to_numpy()
    y = (history.target - history[anchor]).to_numpy()
    ages = (cutoff - history.cutoff).dt.total_seconds().to_numpy() / 604800
    centre, scale = standardize(x)
    if linear:
        weights = np.exp2(-ages / 156.0)
        ridge = Ridge(alpha=0.15 * weights.sum()).fit(transform(x, centre, scale), np.clip(y, -3, 3), sample_weight=weights)
        members = [dict(weights=[ridge.coef_[:, None].tolist()], biases=[[float(ridge.intercept_)]])]
    else:
        members = [fit_member(x, y, ages, width, seed) for seed in SEEDS]
    return dict(features=features, centre=centre.tolist(), scale=scale.tolist(), members=members,
                latest_training_label_utc=history.known.max().isoformat(), training_weeks=len(history))


def direct_prediction(model, x, anchor):
    z = transform(x, np.asarray(model["centre"]), np.asarray(model["scale"]))
    members = []
    for member in model["members"]:
        h = z
        for i, (weights, bias) in enumerate(zip(member["weights"], member["biases"], strict=True)):
            h = h @ np.asarray(weights) + np.asarray(bias)
            if i + 1 < len(member["weights"]):
                h = np.tanh(h)
        members.append(h[..., 0])
    return anchor + np.clip(np.mean(members, axis=0), -2.5, 2.5)


def replay(data, evaluation, width, *, use_optics=True, linear=False):
    blocks = []
    for start in range(0, len(evaluation), REFIT):
        group = evaluation.iloc[start:start + REFIT].copy()
        cutoff = group.cutoff.iloc[0]
        history = data[(data.cutoff < cutoff) & (data.known < cutoff)]
        model = fit_model(history, cutoff, width, use_optics, linear)
        group["direct"] = direct_prediction(model, group[model["features"]].to_numpy(), group.anchor.to_numpy())
        group["network_training_weeks"] = len(history)
        group["network_training_known"] = model["latest_training_label_utc"]
        blocks.append(group)
    return pd.concat(blocks)


def combine(baseline, direct, share):
    return baseline + share * (direct - baseline)


def select(data):
    validation = data[(data.release_date >= "2021-01-01") & (data.release_date < "2024-01-01") &
                      data.public_ensemble.notna()]
    if validation.empty or (validation.phase == "holdout").any():
        raise ValueError("Validation must precede the project's holdout")
    trials = []
    for width in WIDTHS:
        pred = replay(data, validation, width)
        for share in MIX:
            error = 1000 * abs(combine(pred.optical_reference, pred.direct, share) - pred.target)
            years = error.groupby(pred.release_date.str[:4]).mean()
            trials.append(dict(hidden_layers=list(width), neural_share=share, yearly_mae_kbbl=years.to_dict(),
                               mean_year_mae_kbbl=float(years.mean())))
    selected = min(trials, key=lambda r: (r["mean_year_mae_kbbl"], r["neural_share"], sum(r["hidden_layers"])))
    return dict(criterion="Mean of calendar-year MAEs on pre-2024 rolling forecasts",
                first=validation.release_date.iloc[0], last=validation.release_date.iloc[-1], weeks=len(validation),
                latest_validation_label_utc=validation.known.max().isoformat(), trials=trials, selected=selected)


def metrics(rows, method):
    y = rows.target.to_numpy()
    p = rows[method].to_numpy()
    err = p - y
    return dict(n=len(y), mae_kbbl=float(np.abs(err).mean() * 1000),
                rmse_kbbl=float(np.sqrt(np.mean(err ** 2)) * 1000), correct_sign=int((np.sign(y) == np.sign(p)).sum()))


def train():
    data = load_panel()
    pre = data[data.release_date < "2024-01-01"]
    selection = select(pre)
    chosen = selection["selected"]
    width, share = tuple(chosen["hidden_layers"]), chosen["neural_share"]
    print(f"Locked on {selection['weeks']} pre-2024 forecasts: {width}, neural share {share}", flush=True)
    test = data[data.phase == "holdout"]
    if test.empty or pre.known.max() >= test.cutoff.min():
        raise ValueError("Selection overlaps the holdout")
    pred = replay(data, test, width)
    public_net = replay(data, test, width, use_optics=False)
    linear = replay(data, test, width, linear=True)
    pred["neural_blend"] = combine(pred.optical_reference, pred.direct, share)
    pred["without_satellite"] = combine(pred.public_ensemble, public_net.direct, share)
    pred["linear_blend"] = combine(pred.optical_reference, linear.direct, share)
    methods = ("public_ensemble", "optical_reference", "previous_mlp", "anchor", "linear_blend", "without_satellite",
               "direct", "neural_blend")
    if not np.isfinite(pred[list(methods)]).all().all():
        raise ValueError("Missing comparison on common evaluation dates")
    scores = {k: metrics(pred, k) for k in methods}
    delta = 1000 * abs(pred.neural_blend - pred.target)
    paired = {k: block_bootstrap_mean(delta - 1000 * abs(pred[k] - pred.target)) for k in methods if k != "neural_blend"}
    subgroups = {"with_b04": pred.measured.astype(bool), "without_b04": ~pred.measured.astype(bool)}
    subgroups.update({str(year): pred.release_date.str.startswith(year) for year in sorted(set(pred.release_date.str[:4]))})
    report = dict(status="retrospective_on_previously_reviewed_history", selection=selection, holdout=scores,
                  paired_mae_difference_95_kbbl=paired,
                  subgroups={g: {k: metrics(pred[mask], k) for k in methods} for g, mask in subgroups.items() if mask.any()})
    DERIVED.mkdir(parents=True, exist_ok=True)
    cols = ["release_date", "week_ending", "cutoff", "known", "network_training_known", "network_training_weeks",
            "target", "measured", *methods]
    pred[cols].to_csv(DERIVED / "cushing_inventory_network_forecasts.csv", index=False)
    save_json(DERIVED / "cushing_inventory_network_summary.json", report)
    ready = data.known.max() + pd.Timedelta(seconds=1)
    fitted = fit_model(data, ready, width)
    fitted.update(hidden_layers=list(width), neural_share=share, format_version=1,
                  target_unit="million_barrels", baseline="optical_reference")
    save_json(WEIGHTS, fitted)
    print(json.dumps(scores, indent=2))


def predict(row, model):
    cutoff, latest = pd.Timestamp(row["cutoff"]), pd.Timestamp(model["latest_training_label_utc"])
    source = pd.Timestamp(row["latest_public_input"])
    if cutoff.tzinfo is None or source.tzinfo is None or not max(latest, source) < cutoff:
        raise ValueError("Model weights and published EIA inputs must predate the forecast cutoff")
    x = np.array([row[k] for k in model["features"]], dtype=float)
    baseline, anchor = float(row[model["baseline"]]), float(row["anchor"])
    if not np.isfinite(np.r_[x, baseline, anchor]).all():
        raise ValueError("Incomplete pre-release features")
    direct = float(direct_prediction(model, x, anchor))
    return dict(cutoff=cutoff.isoformat(), estimated_change_kbbl=1000 * float(combine(baseline, direct, model["neural_share"])),
                standalone_network_kbbl=1000 * direct)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", type=Path, metavar="FEATURES.json", help="Apply the saved model to one unlabeled feature row")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.predict:
            print(json.dumps(predict(json.loads(args.predict.read_text()), json.loads(WEIGHTS.read_text())), indent=2))
        else:
            train()


if __name__ == "__main__":
    main()
