"""Compare robust weekly stock-balance, nonlinear and radar models under pre-release cutoffs."""

import argparse
import hashlib
import json
from pathlib import Path

import cushing_inventory_measurement_filter as filters
import fit_cushing_survey_ensemble as v3
import fit_cushing_survey_residuals as v4
import numpy as np
import pandas as pd
from collect_cushing_verified_bulletins import OUTPUT as WIRE
from cushing_survey_residual_network import (
    estimate,
    feedback,
    fit_feedback_head,
    fit_head,
)
from fit_cushing_inventory_network import MODEL_DIR, metrics, save_json
from fit_cushing_network_calibration import PACKAGE
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from prepare_cushing_measurement_inputs import OUTPUT as MEASUREMENTS
from threadpoolctl import threadpool_limits

WEIGHTS = MODEL_DIR / "measurement_weights.json"
NN_FIELDS = (
    "strong_reference",
    "ref_gap",
    "state2",
    "state6",
    "state16",
    "last_gap_local",
    "national",
    "national_present",
    "vol13",
    "midwest_change",
    "gulf_change",
    "quarter_turn",
    "year_turn",
)
REGS = (0.02, 0.12, 0.6)
LIVES = (52, 156)
FAMILIES = (
    "short_memory",
    "stock_balance",
    "reliability",
    "regional_balance",
    "random_features",
    "fallback",
    "radar_sensitivity",
    "radar_fallback",
)


def wire_records(original, filepath=WIRE, cutoff=None):
    updates = v3.survey_records(filepath)
    if cutoff is not None:
        updates = updates[updates.available_utc < pd.Timestamp(cutoff)]
    overlap = updates.index.isin(original.index)
    if not np.array_equal(updates.action, np.where(overlap, "replace", "add")):
        raise ValueError("Inconsistent bulletin action")
    return pd.concat([original.loc[~original.index.isin(updates.index)], updates]).sort_index()


def measurement_at(row, cutoff):
    values = {"national_kbbl": np.nan, "radar_weekly_million": np.nan, "radar_age_days": np.nan}
    if row is None:
        return values
    if pd.notna(row.national_kbbl) and pd.Timestamp(row.national_time) < cutoff:
        values["national_kbbl"] = row.national_kbbl
    if pd.notna(row.radar_weekly_million) and pd.Timestamp(row.radar_available) < cutoff:
        values.update(radar_weekly_million=row.radar_weekly_million, radar_age_days=row.radar_age_days)
    return values


def setup(wire_path=WIRE):
    original = v4.prepared()
    data = v3.add_survey(original, wire_records(v4.sources(), wire_path))
    measurements = pd.read_csv(MEASUREMENTS).set_index("release_date")
    if not measurements.index.is_unique:
        raise ValueError("Duplicate measurement date")
    extra = []
    for r in data.itertuples():
        measure = measurements.loc[r.release_date]
        if measure.week_ending != r.week_ending:
            raise ValueError("Mismatched measurement week")
        extra.append(measurement_at(measure, r.cutoff) | filters.state(data, r.week_ending, r.cutoff))
    updated_feedback = [feedback(data, r.week_ending, r.cutoff) for r in data.itertuples()]
    data = data.assign(**pd.DataFrame(updated_feedback, index=data.index).to_dict("series"))
    return original, filters.covariates(pd.concat([data, pd.DataFrame(extra, index=data.index)], axis=1))


def cfg_name(family, reg, life):
    return f"{family}_{reg:.2f}_{life}"


def all_configs():
    return {cfg_name(f, r, h): (f, r, h) for f in FAMILIES for r in REGS for h in LIVES}


def updated_heads(history, cutoff):
    return {
        "survey": fit_head(history, cutoff, True)[0],
        "fallback": fit_head(history, cutoff, False)[0],
        "feedback_head": fit_feedback_head(history, cutoff),
    }


def replay(original, data, period, configs):
    old = v3.add_survey(original, v3.survey_records())
    v3w = json.loads(v3.MIXTURE.read_text())["mix"]
    net = json.loads(PACKAGE.read_text())["mix"]
    prior = v4.replay(original, original.loc[period.index], v3w, net, old)
    rows = data.loc[period.index].copy()
    rows["previous_model"] = v4.apply(prior, json.loads(v4.WEIGHTS.read_text())["mix"])
    mixture = json.loads(v4.WEIGHTS.read_text())["mix"]
    recalibrated = v4.replay(data, data.loc[period.index], v3w, net, old)
    rows["retrained"] = v4.apply(recalibrated, mixture)
    rows["previous_survey"] = original.loc[rows.index, "survey"]
    changed = rows.survey.notna() & (
        rows.previous_survey.isna() | ((rows.previous_survey - rows.survey).abs() > 1e-9)
    )
    rows["coverage_only"] = rows.previous_model
    rows.loc[changed, "coverage_only"] = (
        0.9 * rows.loc[changed, "survey"] + 0.1 * rows.loc[changed, "strong_reference"]
    )
    for name in [*configs, "neural_filter"]:
        rows[name] = np.nan
    for _, year in rows.groupby(rows.release_date.str[:4]):
        start = year.cutoff.min()
        head, _ = fit_head(data, start, True, common=NN_FIELDS)
        present = year[year.survey.notna()]
        rows.loc[present.index, "neural_filter"] = estimate(head, present, present.survey)
        # Every model here sees only previously settled labels, refreshed quarterly.
        for i in range(0, len(year), 13):
            block = year.iloc[i : i + 13]
            for name, (family, reg, life) in configs.items():
                mod = filters.fit(data, block.cutoff.min(), family, reg, life)
                view = block if family in ("fallback", "radar_fallback") else block[block.survey.notna()]
                rows.loc[view.index, name] = filters.predict(mod, view)
    return rows


def select_group(rows, configs, present, allow_radar):
    families = (
        ["short_memory", "stock_balance", "reliability", "regional_balance", "random_features"]
        if present
        else ["fallback"]
    )
    if allow_radar:
        families += ["radar_sensitivity" if present else "radar_fallback"]
    frame = rows[rows.survey.notna()].copy() if present else rows.copy()
    anchor = "coverage_only" if present else "strong_reference"
    years = frame.release_date.str[:4]
    # Adopt the same published architecture refitted on corrected inputs if it improves
    # both the early calibration year and the following two years as a block.
    if present:
        for_window = [frame[years == years.min()], frame[years != years.min()]]
        if all(
            (s.retrained - s.target).abs().mean() < (s.coverage_only - s.target).abs().mean()
            for s in for_window
        ):
            anchor = "retrained"

    def weights(sample):
        # Teach the fallback on every masked week and weight actually missing surveys equally as a group.
        absent = sample.survey.isna().to_numpy()
        w = np.ones(len(sample)) / len(sample)
        if not present and absent.any():
            w = 0.5 * w + 0.5 * absent / absent.sum()
        return w

    def choose(training):
        ordered = [anchor]
        if present:
            ordered += ["neural_filter"]
        wt = weights(training)
        for f in families:
            names = [name for name, (fam, *_) in configs.items() if fam == f]
            ordered.append(min(names, key=lambda k: np.dot(wt, abs(training[k] - training.target))))
        mix = v3.exact_weights(training[ordered].to_numpy(), training.target.to_numpy(), wt)
        return {k: float(w) for k, w in zip(ordered, mix) if w > 1e-10}

    # Forecast the second and third calibration years using only earlier years' mixture labels.
    # Give the full pre-2024 fit half the deployment weight and each earlier fit one quarter.
    unique = sorted(years.unique())
    if len(unique) != 3:
        raise ValueError("Expected three calibration years")
    cross = pd.Series(np.nan, index=frame.index)
    chosen = {}
    for i, year in enumerate(unique):
        m = choose(frame[years <= year])
        if i + 1 < len(unique):
            later = frame[years == unique[i + 1]]
            cross.loc[later.index] = sum(w * later[col] for col, w in m.items())
        for col, w in m.items():
            chosen[col] = chosen.get(col, 0.0) + (0.5 if i == 2 else 0.25) * w
    valid = frame[cross.notna()]
    wt = weights(valid)
    score = float(wt @ abs(cross.loc[valid.index] - valid.target)) * 1000
    baseline = float(wt @ abs(valid[anchor] - valid.target)) * 1000
    if score >= baseline:
        chosen = {anchor: 1.0}
    return chosen, {
        "cross_year_mae_kbbl": score,
        "baseline_cross_year_mae_kbbl": baseline,
        "anchor": anchor,
        "selected": chosen,
    }


def apply(rows, mix):
    out = np.empty(len(rows))
    for group, mask in [
        ("present", rows.survey.notna().to_numpy()),
        ("missing", rows.survey.isna().to_numpy()),
    ]:
        view = rows[mask]
        if len(view):
            out[mask] = sum(w * view[k].to_numpy() for k, w in mix[group].items())
    return out


def train():
    original, data = setup()
    pre = data[data.release_date < "2024-01-01"]
    period = pre[(pre.release_date >= "2021-01-01") & pre.strong_reference.notna()]
    if pre.known.max() >= data.loc[data.phase == "holdout", "cutoff"].min():
        raise ValueError("Validation overlaps test")
    configs = all_configs()
    val = replay(original.loc[pre.index], pre, period, configs)
    strategies = {}
    selection = {}
    for key, radar in (("strict", False), ("with_reconstructed_radar", True)):
        strategies[key], selection[key] = {}, {}
        for group, present in (("present", True), ("missing", False)):
            strategies[key][group], selection[key][group] = select_group(val, configs, present, radar)
        val[key] = apply(val, strategies[key])
    print("Locked before evaluation", json.dumps(strategies), flush=True)
    rows = replay(original, data, data[data.phase == "holdout"], configs)
    old = pd.read_csv(MODEL_DIR / "residual_forecasts.csv")
    if list(old.release_date) != list(rows.release_date) or not np.allclose(
        rows.previous_model, old.estimate, atol=1e-10, rtol=0
    ):
        raise ValueError("Lost unchanged benchmark")
    for k, m in strategies.items():
        rows[k] = apply(rows, m)
    methods = ["previous_model", "coverage_only", "retrained", *strategies]
    if not np.isfinite(rows[methods].to_numpy()).all():
        raise ValueError("Incomplete forecasts on the common evaluation calendar")
    splits = {
        "all": rows,
        "with_quote": rows[rows.survey.notna()],
        "without_quote": rows[rows.survey.isna()],
        **{year: g for year, g in rows.groupby(rows.release_date.str[:4])},
    }
    scores = {k: {m: metrics(g, m) for m in methods} for k, g in splits.items() if len(g)}
    deltas = {
        k: block_bootstrap_mean(1000 * (abs(rows.strict - rows.target) - abs(rows[k] - rows.target)))
        for k in ("previous_model", "coverage_only")
    }
    ready = data.known.max() + pd.Timedelta(seconds=1)
    strict_used = {k for m in strategies["strict"].values() for k in m}
    fitted = {k: filters.fit(data, ready, *configs[k]) for k in sorted(strict_used) if k in configs}
    head, epochs = fit_head(data, ready, True, common=NN_FIELDS)
    history = data[data.survey.notna()][["week_ending", "known", "target", "survey"]].tail(104).copy()
    history["known"] = history.known.map(lambda t: t.isoformat())
    save_json(
        WEIGHTS,
        {
            "mix": strategies["strict"],
            "filters": fitted,
            "neural": head if "neural_filter" in strict_used else None,
            "settled": history.to_dict(orient="records"),
            "retrained_heads": updated_heads(data, ready) if "retrained" in strict_used else None,
            "v4_sha256": hashlib.sha256(v4.WEIGHTS.read_bytes()).hexdigest(),
        },
    )
    # Diagnostic only. Even an analyst fitting the constant simplex weights to the evaluation
    # outcomes cannot go below this MAE using the enumerated forecasts. Never deploy these weights.
    minimum_abs_error = 0.0
    for present in (True, False):
        view = rows[rows.survey.notna() == present]
        if view.empty:
            continue
        candidates = ["previous_model", "coverage_only", "strong_reference", "retrained"]
        if present:
            candidates += ["survey", "neural_filter"]
        candidates += [
            name
            for name, (family, _, _) in configs.items()
            if present or family in ("fallback", "radar_fallback")
        ]
        x = view[candidates].to_numpy()
        exact = v3.exact_weights(x, view.target.to_numpy(), np.ones(len(view)))
        minimum_abs_error += np.abs(x @ exact - view.target.to_numpy()).sum()
    hindsight_minimum = 1000 * minimum_abs_error / len(rows)
    report = {
        "retrospective_fixed_blend_lower_bound_kbbl": hindsight_minimum,
        "status": "retrospective_previously_examined_calendar",
        "target_mae_kbbl": 200,
        "selection": selection,
        "validation": {k: metrics(val, k) for k in methods},
        "test": scores,
        "paired_difference_95_kbbl": deltas,
        "weeks_with_error_at_most_200": {
            k: int((1000 * abs(rows[k] - rows.target) <= 200).sum()) for k in methods
        },
        "config_count": len(configs),
        "neural_stopping_epoch": head["epochs"],
        "neural_training_weeks": head["training_weeks"],
    }
    save_json(MODEL_DIR / "measurement_evaluation.json", report)
    pd.DataFrame({"epoch": np.arange(1, len(epochs) + 1), "stopping_mae_kbbl": epochs}).to_csv(
        MODEL_DIR / "measurement_epochs.csv", index=False
    )
    rows[
        [
            "release_date",
            "week_ending",
            "cutoff",
            "target",
            "previous_survey",
            "survey",
            "survey_time",
            "national_kbbl",
            *methods,
        ]
    ].to_csv(MODEL_DIR / "measurement_forecasts.csv", index=False)
    print(json.dumps(scores["all"], indent=2), flush=True)


def predict(record, decision, package, old, new, measure, settled=None):
    when = pd.Timestamp(decision)
    deployment = json.loads(v4.WEIGHTS.read_text())
    original = json.loads(v3.MIXTURE.read_text())
    network = json.loads(PACKAGE.read_text())
    if (
        hashlib.sha256(v4.WEIGHTS.read_bytes()).hexdigest() != package["v4_sha256"]
        or hashlib.sha256(v3.MIXTURE.read_bytes()).hexdigest() != deployment["previous_sha256"]
        or hashlib.sha256(PACKAGE.read_bytes()).hexdigest() != original["network_sha256"]
    ):
        raise ValueError("Incompatible model bundles")
    base = v4.predict(record, old, decision, network, original, deployment, settled)
    if measure is not None and measure.week_ending != record["week_ending"]:
        raise ValueError("Mismatched measurement week")
    quote = np.nan
    if new is not None and new.available_utc < when:
        if new.name != record["release_date"] or new.week_ending != record["week_ending"]:
            raise ValueError("Mismatched survey week")
        quote = new.api_change_kbbl / 1000
    row = record | measurement_at(measure, when)
    row["cutoff"] = when
    row["survey"] = quote
    history = v4.settled_history(package["settled"], settled)
    row.update(filters.state(history, record["week_ending"], when))
    baseline = base["estimate_kbbl"] / 1000
    if np.isfinite(quote) and (not base["survey_available"] or abs(old.api_change_kbbl / 1000 - quote) > 1e-9):
        baseline = 0.9 * quote + 0.1 * record["strong_reference"]
    row["coverage_only"] = baseline
    frame = filters.covariates(pd.DataFrame([row]))
    group = "present" if np.isfinite(quote) else "missing"
    if "retrained" in package["mix"][group]:
        updated = {
            "heads": package["retrained_heads"],
            "feedback": package["settled"],
            "mix": deployment["mix"],
        }
        frame["retrained"] = (
            v4.predict(record, new, decision, network, original, updated, settled)["estimate_kbbl"] / 1000
        )
    for name in package["mix"][group]:
        if name in package["filters"]:
            frame[name] = filters.predict(package["filters"][name], frame)
        if name == "neural_filter":
            frame[name] = estimate(package["neural"], frame, frame.survey)
    return {
        "release_date": record["release_date"],
        "as_of": when.isoformat(),
        "survey_available": bool(np.isfinite(quote)),
        "estimate_kbbl": float(1000 * apply(frame, package["mix"])[0]),
        "previous_kbbl": base["estimate_kbbl"],
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predict", type=Path)
    p.add_argument("--as-of")
    p.add_argument("--wire-ledger", type=Path, default=WIRE)
    p.add_argument("--survey-ledger", type=Path, default=v3.SURVEY)
    p.add_argument("--confirmed-ledger", type=Path, default=v4.EXTRA)
    p.add_argument("--measurements", type=Path, default=MEASUREMENTS)
    p.add_argument("--settled-feedback", type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        if args.predict:
            if not args.as_of:
                p.error("Specify --as-of")
            record = json.loads(args.predict.read_text())
            day = record["release_date"]
            full = v4.sources(args.confirmed_ledger, args.survey_ledger)
            old = v4.sources(args.confirmed_ledger, args.survey_ledger, args.as_of)
            # Validate the replacement ledger against the complete baseline before applying the time cutoff.
            new = wire_records(full, args.wire_ledger, args.as_of)
            if day not in new.index or new.loc[day].available_utc >= pd.Timestamp(args.as_of):
                new = old
            ms = pd.read_csv(args.measurements).set_index("release_date")
            if not ms.index.is_unique:
                raise ValueError("Duplicate measurement date")
            get = lambda df: df.loc[day] if day in df.index else None
            settled = pd.read_csv(args.settled_feedback) if args.settled_feedback else None
            print(
                json.dumps(
                    predict(
                        record,
                        args.as_of,
                        json.loads(WEIGHTS.read_text()),
                        get(old),
                        get(new),
                        get(ms),
                        settled,
                    ),
                    indent=2,
                )
            )
        else:
            train()
