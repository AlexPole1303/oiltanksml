"""Compare lagged survey-level error reconciliation and new physical inputs on the fixed Cushing calendar."""

import argparse
import hashlib
import json
from pathlib import Path

import cushing_survey_reconciliation as rec
import fit_cushing_measurement_filters as v5
import numpy as np
import pandas as pd
from cushing_survey_residual_network import feedback
from fit_cushing_measurement_filters import MODEL_DIR, metrics, save_json, v3, v4
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from prepare_cushing_balance_inputs import OUTPUT as EXTERNAL
from prepare_cushing_balance_inputs import stamped_values
from threadpoolctl import threadpool_limits

PRECISION = MODEL_DIR / "precision_observations.csv"
WEIGHTS = MODEL_DIR / "balance_weights.json"
CONFIGS = {}
for fam in rec.LAYOUTS:
    for strength in (0.12, 1.2):
        for life in (52, 156):
            CONFIGS[f"{fam}_{strength}_{life}"] = (fam, strength, life, False, False)
for fam in ("flows", "combined"):
    CONFIGS[f"{fam}_stumps"] = (fam, 0.4, 104, True, False)
for strength in (0.12, 1.2):
    for life in (52, 156):
        CONFIGS[f"missing_{strength}_{life}"] = ("level", strength, life, False, True)


def product_features(rows, week, when):
    values = {"gasoline": 0.0, "distillate": 0.0}
    if rows is not None:
        if rows["product"].duplicated().any() or not (rows.week_ending == week).all():
            raise ValueError("Inconsistent product survey week")
        for r in rows.itertuples():
            stamp = pd.Timestamp(r.available_utc)
            published = pd.Timestamp(r.published_utc)
            if (
                r.product not in values
                or not np.isfinite(r.value_kbbl)
                or stamp.tzinfo is None
                or published.tzinfo is None
                or stamp < published
            ):
                raise ValueError("Invalid dated product observation")
            if stamp < when:
                values[r.product] = r.value_kbbl / 3000.0
    return values | {"product_balance": values["gasoline"] + values["distillate"]}


def setup():
    _, previous = v5.setup()
    sources = v5.wire_records(v4.sources())
    full = v3.add_survey(previous, v5.wire_records(sources, PRECISION))
    extra = pd.read_csv(EXTERNAL).set_index("release_date")
    product_path = MODEL_DIR / "product_observations.csv"
    products = pd.read_csv(product_path)
    groups = {k: g for k, g in products.groupby("release_date")}
    if not extra.index.is_unique:
        raise ValueError("Duplicate measurement week")
    new = []
    for _, r in full.iterrows():
        d = stamped_values(extra.loc[r.release_date], r.cutoff, r.week_ending)
        fuel = product_features(groups.get(r.release_date), r.week_ending, r.cutoff)
        new.append(
            feedback(full, r.week_ending, r.cutoff)
            | v5.filters.state(full, r.week_ending, r.cutoff)
            | rec.states(full, r.week_ending, r.cutoff)
            | rec.features(r, d)
            | fuel
        )
    for col, values in pd.DataFrame(new, index=full.index).items():
        full[col] = values
    return previous, full


def benchmark(data, period):
    old = v3.add_survey(data, v3.survey_records())
    m = json.loads(v3.MIXTURE.read_text())["mix"]
    net = json.loads(v5.PACKAGE.read_text())["mix"]
    hist = v4.replay(data, period, m, net, old)
    return v4.apply(hist, json.loads(v4.WEIGHTS.read_text())["mix"])


def replay(prev, data, period):
    rows = data.loc[period.index].copy()
    rows["baseline"] = benchmark(prev, prev.loc[period.index])
    rows["precise"] = benchmark(data, period)
    rows["original_quote"] = prev.loc[period.index, "survey"]
    for name, cfg in CONFIGS.items():
        rows[name] = np.nan
    for _, year in rows.groupby(rows.release_date.str[:4]):
        for i in range(0, len(year), 8):
            block = year.iloc[i : i + 8]
            for name, cfg in CONFIGS.items():
                fit = rec.fit(data, block.cutoff.min(), *cfg)
                part = block if cfg[-1] else block[block.survey.notna()]
                rows.loc[part.index, name] = rec.predict(fit, part)
    return rows


def overlay_replay(rows, prior):
    history = pd.concat([prior, rows]).sort_values("cutoff")
    if not history.release_date.is_unique:
        raise ValueError("Overlapping calibration forecasts")
    for name in rec.OVERLAYS:
        rows[name] = np.nan
    for _, year in rows.groupby(rows.release_date.str[:4]):
        for start in range(0, len(year), 4):
            part = year.iloc[start : start + 4]
            obs = part[part.survey.notna()]
            for name in rec.OVERLAYS:
                head = rec.fit_overlay(history, part.cutoff.min(), name)
                rows.loc[obs.index, name] = rec.apply_overlay(head, obs)
    return rows


def average_weights(rows, present, overlays):
    base = "precise" if present else "strong_reference"
    names = [base] + [k for k, cfg in CONFIGS.items() if cfg[-1] != present]
    y = rows.target.to_numpy()
    # Equal emphasis per calibration year. Shrink the ensemble towards the source-corrected incumbent.
    years = rows.release_date.str[:4]
    years_weight = years.map(1 / years.value_counts()).to_numpy()
    years_weight = years_weight / years_weight.sum()
    if not present:
        absent = rows.survey.isna().to_numpy()
        if absent.any():
            years_weight = 0.5 * years_weight + 0.5 * absent / absent.sum()
    # Pick one regularization per physical model family on the fitting dates.
    groupings = {}
    for k in names[1:]:
        cfg = CONFIGS[k]
        groupings.setdefault((cfg[0], cfg[3]), []).append(k)
    best = [base] + [min(l, key=lambda k: years_weight @ np.abs(rows[k] - y)) for l in groupings.values()]
    if present:
        best += list(overlays)
    weights = v3.exact_weights(rows[best].to_numpy(), y, years_weight)
    return {k: float(v) for k, v in zip(best, weights) if v > 1e-9}


def mixture(r, w):
    return sum(v * r[k].to_numpy() for k, v in w.items())


def select(val, present, overlays):
    if val.release_date.max() >= "2024-01-01":
        raise ValueError("Selection is restricted to early releases")
    rows = val[val.survey.notna()].copy() if present else val.copy()
    base = "precise" if present else "strong_reference"
    candidates = []
    cross = []
    for year in sorted(rows.release_date.str[:4].unique()):
        train = rows[rows.release_date.str[:4] <= year]
        mix = average_weights(train, present, overlays)
        candidates.append(mix)
        after = rows[rows.release_date.str[:4] == str(int(year) + 1)]
        if not present:
            after = after[after.survey.isna()]
        if len(after):
            new = mixture(after, mix)
            cross.extend(
                (abs(new - after.target.to_numpy()) - abs(after[base] - after.target).to_numpy()).tolist()
            )
    full = {}
    for share, w in zip((0.20, 0.30, 0.50), candidates, strict=True):
        for k, v in w.items():
            full[k] = full.get(k, 0.0) + share * v
    if not cross or np.mean(cross) >= 0:
        full = {base: 1.0}
    return full, {
        "expanding_year_delta_kbbl": 1000 * np.mean(cross),
        "expanding_n": len(cross),
        "weights": full,
    }


def apply(rows, selected):
    forecasts = np.empty(len(rows))
    for group, mask in (
        ("present", rows.survey.notna().to_numpy()),
        ("missing", rows.survey.isna().to_numpy()),
    ):
        if mask.any():
            forecasts[mask] = mixture(rows[mask], selected[group])
    return forecasts


def hindsight_diagnostic(rows):
    # Use revealed outcomes only here to measure the best *fixed* blend
    # in this particular library. The fitted weights never enter forecasting.
    result = {}
    for present in (True, False):
        sample = rows[rows.survey.notna() == present]
        if sample.empty:
            continue
        names = ["baseline", "precise", "strong_reference", "estimate"]
        names += [name for name, cfg in CONFIGS.items() if cfg[-1] != present]
        if present:
            names += ["survey", *rec.OVERLAYS]
        x = sample[names].to_numpy()
        y = sample.target.to_numpy()
        weights = v3.exact_weights(x, y, np.ones(len(sample)) / len(sample))
        result["quoted" if present else "missing"] = {
            "n": len(sample),
            "mae_kbbl": float(1000 * np.abs(x @ weights - y).mean()),
        }
    return sum(r["n"] * r["mae_kbbl"] for r in result.values()) / len(rows)


def calibration_replay(prev, data):
    before = data[data.release_date < "2024-01-01"]
    pre = before[(before.release_date >= "2021-01-01") & before.strong_reference.notna()]
    warm = before[
        (before.release_date >= "2020-07-01")
        & (before.release_date < "2021-01-01")
        & before.strong_reference.notna()
    ].copy()
    warm["precise"] = benchmark(before, warm)
    val = overlay_replay(replay(prev.loc[before.index], before, pre), warm)
    return warm, val


def train(validation_only=False):
    prev, data = setup()
    warm, val = calibration_replay(prev, data)
    chosen = {}
    details = {}
    core = tuple(k for k in rec.OVERLAYS if k not in ("weekly_timing", "variable_gain"))
    families = {"stock": core, "stock_and_timing": tuple(rec.OVERLAYS)}
    trials = {name: select(val, True, overlays) for name, overlays in families.items()}
    best = min(trials, key=lambda name: trials[name][1]["expanding_year_delta_kbbl"])
    chosen["present"], details["present"] = trials[best]
    details["present"]["family"] = best
    details["present"]["candidate_deltas_kbbl"] = {
        name: trial[1]["expanding_year_delta_kbbl"] for name, trial in trials.items()
    }
    chosen["missing"], details["missing"] = select(val, False, ())
    val["estimate"] = apply(val, chosen)
    diag = {
        k: metrics(val[val.survey.notna()], k)["mae_kbbl"]
        for k in ["precise", *[n for n, c in CONFIGS.items() if not c[-1]]]
    }
    print(
        "Calibration",
        json.dumps(details),
        "quoted model MAEs",
        json.dumps({k: round(v, 1) for k, v in diag.items()}),
        flush=True,
    )
    if validation_only:
        return
    test = data[data.phase == "holdout"]
    if data.loc[data.release_date < "2024-01-01", "known"].max() >= test.cutoff.min():
        raise ValueError("Calibration overlaps evaluation")
    rows = overlay_replay(replay(prev, data, test), pd.concat([warm, val]))
    rows["estimate"] = apply(rows, chosen)
    prior = pd.read_csv(MODEL_DIR / "measurement_forecasts.csv")
    if list(rows.release_date) != list(prior.release_date) or not np.allclose(
        rows.baseline, prior.strict, atol=1e-10, rtol=0
    ):
        raise ValueError("Did not reproduce the published model")
    methods = ["baseline", "precise", "estimate"]
    if not np.isfinite(rows[methods]).all().all():
        raise ValueError("Incomplete score")
    groups = {
        "all": rows,
        "original_quoted": rows[rows.original_quote.notna()],
        "original_missing": rows[rows.original_quote.isna()],
        "quoted": rows[rows.survey.notna()],
        "missing": rows[rows.survey.isna()],
        **dict(tuple(rows.groupby(rows.release_date.str[:4]))),
    }
    scores = {g: {k: metrics(r, k) for k in methods} for g, r in groups.items() if len(r)}
    intervals = {
        g: {
            k: block_bootstrap_mean(1000 * (abs(r.estimate - r.target) - abs(r[k] - r.target)))
            for k in methods[:2]
        }
        for g, r in groups.items()
        if len(r)
    }
    result = {
        "target_mae_kbbl": 200,
        "configs": len(CONFIGS) + len(rec.OVERLAYS),
        "selection": details,
        "validation": {k: metrics(val, k) for k in methods},
        "scores": scores,
        "paired_mae_delta_95_kbbl": intervals,
        "within_200": {k: int(((rows[k] - rows.target).abs() * 1000 <= 200).sum()) for k in methods},
        "hindsight_fixed_blend_mae_kbbl": {
            g: hindsight_diagnostic(groups[g]) for g in ("all", "original_quoted")
        },
    }
    readyt = data.known.max() + pd.Timedelta(seconds=1)
    heads = {
        k: rec.fit(data, readyt, *CONFIGS[k]) for k in sorted(set().union(*chosen.values())) if k in CONFIGS
    }
    overlays = {
        k: rec.fit_overlay(pd.concat([warm, val, rows]), readyt, k)
        for k in sorted(set().union(*chosen.values()))
        if k in rec.OVERLAYS
    }
    hist = data[data.survey.notna()][["week_ending", "known", "target", "survey"]].tail(160).copy()
    hist["known"] = hist.known.map(lambda t: t.isoformat())
    pack = {
        "mix": chosen,
        "head": heads,
        "overlays": overlays,
        "updated": v5.updated_heads(data, readyt),
        "settled": hist.to_dict(orient="records"),
        "v5_sha256": hashlib.sha256(v5.WEIGHTS.read_bytes()).hexdigest(),
    }
    save_json(WEIGHTS, pack)
    save_json(MODEL_DIR / "balance_evaluation.json", result)
    rows[
        ["release_date", "week_ending", "cutoff", "target", "original_quote", "survey", "survey_time", *methods]
    ].to_csv(MODEL_DIR / "balance_forecasts.csv", index=False)
    print("Full scores", scores["all"], flush=True)


def dated_quote(day, when, ledgers):
    active = None
    for ledger in ledgers:
        if day in ledger.index:
            q = ledger.loc[day]
            if q.available_utc < when:
                active = q
    return active


def forecast(record, when, package, quote, external, products, settled=None):
    when = pd.Timestamp(when)
    if hashlib.sha256(v5.WEIGHTS.read_bytes()).hexdigest() != package["v5_sha256"]:
        raise ValueError("Mismatched benchmark package")
    previous = json.loads(v5.WEIGHTS.read_text())
    v4weights = json.loads(v4.WEIGHTS.read_text())
    v3weights = json.loads(v3.MIXTURE.read_text())
    if (
        hashlib.sha256(v4.WEIGHTS.read_bytes()).hexdigest() != previous["v4_sha256"]
        or hashlib.sha256(v3.MIXTURE.read_bytes()).hexdigest() != v4weights["previous_sha256"]
        or hashlib.sha256(v5.PACKAGE.read_bytes()).hexdigest() != v3weights["network_sha256"]
    ):
        raise ValueError("Mismatched network dependencies")
    have = quote is not None and quote.available_utc < when
    if have and (quote.name != record["release_date"] or quote.week_ending != record["week_ending"]):
        raise ValueError("Quote refers to a different reporting week")
    updated = {"heads": package["updated"], "feedback": package["settled"], "mix": v4weights["mix"]}
    reference = v4.predict(
        record, quote if have else None, when, json.loads(v5.PACKAGE.read_bytes()), v3weights, updated, settled
    )
    values = dict(
        record,
        cutoff=when,
        survey=quote.api_change_kbbl / 1000 if have else np.nan,
        survey_time=quote.available_utc.isoformat() if have else "",
        precise=reference["estimate_kbbl"] / 1000,
    )
    history = v4.settled_history(package["settled"], settled)
    values.update(v5.filters.state(history, record["week_ending"], when))
    values.update(rec.states(history, record["week_ending"], when))
    values.update(rec.features(values, stamped_values(external, when, record["week_ending"])))
    values.update(product_features(products, record["week_ending"], when))
    frame = pd.DataFrame([values])
    group = "present" if have else "missing"
    for name in package["mix"][group]:
        if name in package["head"]:
            frame[name] = rec.predict(package["head"][name], frame)
        if name in package["overlays"]:
            frame[name] = rec.apply_overlay(package["overlays"][name], frame)
    return {
        "release_date": record["release_date"],
        "as_of": when.isoformat(),
        "survey_available": have,
        "source_corrected_kbbl": reference["estimate_kbbl"],
        "estimate_kbbl": float(1000 * apply(frame, package["mix"])[0]),
    }


def predict_from_files(args):
    if not args.as_of:
        raise ValueError("Prediction requires --as-of")
    when = pd.Timestamp(args.as_of)
    if when.tzinfo is None:
        raise ValueError("Decision time needs a UTC offset")
    record = json.loads(args.predict.read_bytes())
    wk = record["week_ending"]
    day = record["release_date"]
    # Check add/replace actions against complete ledgers, then resolve the active value by publication time.
    checked = v5.wire_records(v4.sources(args.confirmed_ledger, args.survey_ledger), args.wire_ledger)
    v5.wire_records(checked, args.precision_ledger)
    ledgers = [
        v3.survey_records(f)
        for f in (args.survey_ledger, args.confirmed_ledger, args.wire_ledger, args.precision_ledger)
    ]
    quote = dated_quote(day, when, ledgers)
    ms = pd.read_csv(args.measurements).set_index("release_date")
    ext = pd.read_csv(args.balance_inputs).set_index("release_date")
    if not ms.index.is_unique or not ext.index.is_unique:
        raise ValueError("Duplicate measurement dates")
    if day in ms.index and ms.loc[day].week_ending != wk:
        raise ValueError("Wrong national survey week")
    national = v5.measurement_at(ms.loc[day] if day in ms.index else None, when)
    prod = pd.read_csv(args.products)
    settled = pd.read_csv(args.settled_feedback) if args.settled_feedback else None
    return forecast(
        record | national,
        when,
        json.loads(WEIGHTS.read_bytes()),
        quote,
        ext.loc[day] if day in ext.index else None,
        prod[prod.release_date == day],
        settled,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--predict", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument("--settled-feedback", type=Path)
    parser.add_argument("--survey-ledger", type=Path, default=v3.SURVEY)
    parser.add_argument("--confirmed-ledger", type=Path, default=v4.EXTRA)
    parser.add_argument("--wire-ledger", type=Path, default=v5.WIRE)
    parser.add_argument("--precision-ledger", type=Path, default=PRECISION)
    parser.add_argument("--measurements", type=Path, default=v5.MEASUREMENTS)
    parser.add_argument("--balance-inputs", type=Path, default=EXTERNAL)
    parser.add_argument("--products", type=Path, default=MODEL_DIR / "product_observations.csv")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.predict:
            print(json.dumps(predict_from_files(args), indent=2))
        else:
            train(args.validation_only)
