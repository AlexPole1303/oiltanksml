"""Evolve forecast mixtures chronologically and certify the validation MAE minimum with linear programming."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from threadpoolctl import threadpool_limits

from fit_cushing_inventory_network import DERIVED, MODEL_DIR, load_panel, metrics, save_json
from fit_cushing_network_calibration import PACKAGE, enrich, predict_heads, replay
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from cushing_survey_gate import fit_gate, forecast as gate_forecast

SURVEY = MODEL_DIR / "api_observations.csv"
MIXTURE = MODEL_DIR / "survey_mixture.json"
BASE = ["strong_reference", "neural", "public_innovation", "residual_expert"]
WITH = [*BASE, "survey", "survey_debiased", "survey_shrunk", "survey_network"]
MIN_GENERATIONS = 32
MAX_GENERATIONS = 256
POPULATION = 96


def survey_records(path=SURVEY):
    source = pd.read_csv(path)
    if source.release_date.duplicated().any() or not np.isfinite(source.api_change_kbbl).all():
        raise ValueError("Invalid survey ledger")
    for col in ("published_utc", "available_utc"):
        source[col] = pd.to_datetime(source[col], utc=True)
    if not (source.published_utc <= source.available_utc).all():
        raise ValueError("Invalid publication chronology")
    return source.set_index("release_date")


def add_survey(rows, source):
    out = rows.copy()
    out["survey"] = np.nan
    out["survey_time"] = ""
    for i, r in rows.iterrows():
        if r.release_date not in source.index:
            continue
        quote = source.loc[r.release_date]
        if quote.week_ending != r.week_ending:
            raise ValueError(f"Mismatched week for {r.release_date}")
        if quote.available_utc < r.cutoff:
            out.at[i, "survey"] = quote.api_change_kbbl / 1000
            out.at[i, "survey_time"] = quote.available_utc.isoformat()
    return out


def survey_bias(ledger, cutoff):
    settled = ledger[(ledger.known < cutoff) & ledger.survey.notna()].tail(52)
    if len(settled) < 13:
        return 0.
    return float(np.clip(np.median(settled.target - settled.survey), -.25, .25))


def experts(rows, blend):
    r = rows.copy()
    r["neural"] = r.strong_reference + blend["direct_share"] * (r.direct - r.strong_reference) + blend["correction_share"] * r.correction
    r["public_innovation"] = r.strong_reference + np.clip(r.direct - r.public_ensemble, -1.5, 1.5)
    r["residual_expert"] = r.strong_reference + r.correction
    r["survey_debiased"] = r.survey + r.survey_bias
    r["survey_shrunk"] = 0.85 * r.survey
    return r


def exact_weights(x, y, weights):
    # Epigraph of |Xw-y|, w on the probability simplex. HiGHS provides the optimum in this family.
    n, d = x.shape
    eye = -np.eye(n)
    inequalities = np.vstack([np.column_stack([x, eye]), np.column_stack([-x, eye])])
    sol = linprog(np.r_[np.zeros(d), weights], A_ub=inequalities, b_ub=np.r_[y, -y],
                  A_eq=[np.r_[np.ones(d), np.zeros(n)]], b_eq=[1.], bounds=(0., None), method="highs")
    if not sol.success:
        raise RuntimeError(sol.message)
    result = np.clip(sol.x[:d], 0., 1.)
    return result / result.sum()


def minimize_validation(rows):
    rng = np.random.default_rng(17341)
    groups = []
    for present, fields in [(False, BASE), (True, WITH)]:
        view = rows[rows.survey.notna() == present]
        if len(view) < 8:
            raise ValueError("Insufficient validation coverage in a survey availability group")
        x = view[fields].to_numpy()
        y = view.target.to_numpy()
        weights = np.full(len(view), 1000 / len(rows))
        groups.append((x, y, weights, exact_weights(x, y, weights)))
    if any(not np.isfinite(x).all() or not np.isfinite(y).all() for x, y, _, _ in groups):
        raise ValueError("Nonfinite validation data")
    def score(pop):
        losses = np.zeros(len(pop))
        for s, (x, y, w, _) in enumerate(groups):
            part = pop[:, :len(BASE)] if s == 0 else pop[:, len(BASE):]
            losses += w @ np.abs(x @ part.T - y[:, None])
        return losses
    def norm(z):
        for sl in (slice(0, len(BASE)), slice(len(BASE), None)):
            part = np.maximum(z[:, sl], 0.)
            sums = part.sum(axis=1, keepdims=True)
            z[:, sl] = np.divide(part, sums, out=np.full_like(part, 1/part.shape[1]), where=sums > 0)
        return z
    exact = np.concatenate([g[3] for g in groups])
    minimum = float(score(exact[None, :])[0])
    parents = np.column_stack([rng.dirichlet(np.ones(len(fields)), size=POPULATION) for fields in (BASE, WITH)])
    logs = []
    for generation in range(1, MAX_GENERATIONS + 1):
        losses = score(parents)
        elite = parents[np.argsort(losses)[:12]]
        best = float(losses.min())
        logs.append(dict(generation=generation, validation_mae_kbbl=best, gap_to_minimum_kbbl=max(0., best-minimum)))
        if generation == 1 or generation % 16 == 0:
            print(f"Generation {generation:3}: {best:.3f} kbbl, gap {max(0., best-minimum):.3f}", flush=True)
        if generation >= MIN_GENERATIONS and best - minimum <= .01:
            break
        a, b = elite[rng.integers(len(elite), size=(2, POPULATION - len(elite)))]
        fraction = rng.random((len(a), 1))
        children = fraction * a + (1 - fraction) * b
        sigma = max(.001, .16 * .965 ** generation)
        children += rng.normal(0, sigma, children.shape)
        parents = np.vstack([elite, norm(children)])
    mix = dict(without=groups[0][3].tolist(), with_survey=groups[1][3].tolist())
    return mix, logs, minimum


def apply(rows, mix):
    out = rows[BASE].to_numpy() @ np.asarray(mix["without"])
    present = rows.survey.notna().to_numpy()
    out[present] = rows.loc[present, WITH].to_numpy() @ np.asarray(mix["with_survey"])
    return out


def replay_experts(source, period, ledger, blend):
    r = replay(source, period, tuple(blend["hidden_layers"]))
    r["survey"] = ledger.loc[r.index, "survey"]
    r["survey_time"] = ledger.loc[r.index, "survey_time"]
    r["survey_bias"] = [survey_bias(ledger, t) for t in r.cutoff]
    r["survey_network"] = np.nan
    for _, group in r.groupby(r.release_date.str[:4]):
        gate, _ = fit_gate(ledger, group.cutoff.min())
        available = group[group.survey.notna()]
        r.loc[available.index, "survey_network"] = gate_forecast(gate, available)
    return experts(r, blend)


def train():
    data = enrich(load_panel())
    ledger = add_survey(data, survey_records())
    model = json.loads(PACKAGE.read_text())
    blend = model["mix"]
    pre = data[data.release_date < "2024-01-01"]
    validation = pre[(pre.release_date >= "2021-01-01") & pre.strong_reference.notna()]
    test = data[data.phase == "holdout"]
    if pre.known.max() >= test.cutoff.min():
        raise ValueError("Selection labels overlap evaluation")
    val = replay_experts(pre, validation, ledger, blend)
    mix, history, minimum = minimize_validation(val)
    print(f"Validation optimum {minimum:.3f} kbbl on {len(val)} dates; weights frozen.", flush=True)
    rows = replay_experts(data, test, ledger, blend)
    rows["final_forecast"] = apply(rows, mix)
    rows["survey_or_reference"] = rows.survey.fillna(rows.strong_reference)
    methods = ["previous_mlp", "strong_reference", "neural", "survey_or_reference", "final_forecast"]
    splits = {"all": rows, "with_survey": rows[rows.survey.notna()], "without_survey": rows[rows.survey.isna()],
              **{y: g for y, g in rows.groupby(rows.release_date.str[:4])}}
    ae = 1000 * abs(rows.final_forecast - rows.target)
    paired = {k: block_bootstrap_mean(ae - 1000 * abs(rows[k] - rows.target)) for k in methods[:-1]}
    scores = {k: {m: metrics(g, m) for m in methods} for k, g in splits.items() if len(g)}
    gate, gate_log = fit_gate(ledger, data.known.max() + pd.Timedelta(seconds=1))
    result = dict(survey_network_training=gate_log, survey_network_chosen_generation=gate["chosen_generation"], status="retrospective_on_reviewed_calendar", selection=dict(first=val.release_date.min(),
                  last=val.release_date.max(), n=len(val), with_survey=int(val.survey.notna().sum()),
                  generation_count=len(history), validation_minimum_kbbl=minimum, generation_history=history),
                  test=scores, difference_95_kbbl=paired, mix=mix)
    save_json(MODEL_DIR / "evaluation.json", result)
    pd.DataFrame(history).to_csv(MODEL_DIR / "generations.csv", index=False)
    cols = ["release_date", "week_ending", "cutoff", "known", "training_known", "target", "survey", "survey_time", *methods]
    rows[cols].to_csv(MODEL_DIR / "holdout_forecasts.csv", index=False)
    settled = ledger[ledger.survey.notna()][["known","target","survey"]].tail(52).copy()
    settled["known"] = settled.known.map(lambda t: t.isoformat())
    save_json(MIXTURE, dict(without_survey_experts=BASE, with_survey_experts=WITH, mix=mix,
                            survey_gate=gate, network_sha256=hashlib.sha256(PACKAGE.read_bytes()).hexdigest(),
                            survey_feedback=settled.to_dict(orient="records"), minimum_kbbl=minimum))
    print(json.dumps(scores["all"], indent=2))


def predict(record, quote, decision, model, package):
    from fit_cushing_network_calibration import predict as check_prediction
    check_prediction(record, model)
    original = pd.Timestamp(record["cutoff"])
    when = pd.Timestamp(decision)
    if when.tzinfo is None or not original <= when or not when <= pd.Timestamp(record["deadline_utc"]):
        raise ValueError("Decision must follow the underlying forecast and precede the EIA release deadline")
    rows = pd.DataFrame([record])
    rows["direct"], rows["correction"] = predict_heads(model, rows)
    rows["public_ensemble"] = rows.strong_reference - rows.reference_minus_public
    rows["survey"] = np.nan
    if quote is not None and pd.Timestamp(quote.available_utc) < when:
        if quote.week_ending != record["week_ending"] or quote.name != record["release_date"]:
            raise ValueError("Survey and forecast refer to different dates")
        rows["survey"] = quote.api_change_kbbl/1000
    settled = pd.DataFrame(package["survey_feedback"])
    settled["known"] = pd.to_datetime(settled.known, utc=True)
    rows["survey_bias"] = survey_bias(settled, when)
    rows["cutoff"] = when
    rows["survey_network"] = gate_forecast(package["survey_gate"], rows)
    r = experts(rows, model["mix"])
    return dict(release_date=record["release_date"], decision_utc=when.isoformat(), survey_available=bool(r.survey.notna().iloc[0]),
                estimate_kbbl=float(1000 * apply(r, package["mix"])[0]), reference_kbbl=float(1000 * r.strong_reference.iloc[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", type=Path)
    parser.add_argument("--survey-ledger", type=Path, default=SURVEY, help="Dated survey observations for inference")
    parser.add_argument("--as-of", help="Timezone-aware pre-release decision time")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.predict:
            bundle = json.loads(MIXTURE.read_text())
            if hashlib.sha256(PACKAGE.read_bytes()).hexdigest() != bundle["network_sha256"] or not args.as_of:
                parser.error("Use matching network weights and specify --as-of")
            record = json.loads(args.predict.read_text())
            quotes = survey_records(args.survey_ledger)
            quote = quotes.loc[record["release_date"]] if record["release_date"] in quotes.index else None
            print(json.dumps(predict(record, quote, args.as_of, json.loads(PACKAGE.read_text()), bundle), indent=2))
        else:
            train()
