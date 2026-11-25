"""Train, replay and apply the release-vintage survey and fallback residual networks."""

import argparse
import hashlib
import json
from pathlib import Path

import fit_cushing_survey_ensemble as previous
import numpy as np
import pandas as pd
from collect_cushing_survey_confirmation import EXTRA
from cushing_survey_residual_network import (
    EPOCHS,
    estimate,
    feedback,
    feedback_estimate,
    fit_feedback_head,
    fit_head,
)
from fit_cushing_inventory_network import MODEL_DIR, load_panel, metrics, save_json
from fit_cushing_network_calibration import PACKAGE, enrich
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from threadpoolctl import threadpool_limits

WEIGHTS = MODEL_DIR / "residual_mixture.json"
WITH = ["coverage_only", "survey", "survey_head", "survey_feedback_rule", "learned_feedback"]
WITHOUT = ["strong_reference", "fallback_head"]


def sources(extra=EXTRA, original=previous.SURVEY, cutoff=None):
    old = previous.survey_records(original)
    updates = previous.survey_records(extra)
    overlap = updates.index.isin(old.index)
    if not np.array_equal(updates.action, np.where(overlap, "replace", "add")):
        raise ValueError("Inconsistent survey addition or replacement")
    if cutoff is not None:
        updates = updates[updates.available_utc < pd.Timestamp(cutoff)]
    return pd.concat([old.loc[~old.index.isin(updates.index)], updates]).sort_index()


def prepared():
    data = previous.add_survey(enrich(load_panel()), sources())
    updates = [feedback(data, r.week_ending, r.cutoff) for r in data.itertuples()]
    return pd.concat([data, pd.DataFrame(updates, index=data.index)], axis=1)


def give_estimates(rows, heads):
    rows["fallback_head"] = estimate(heads["fallback"], rows, rows.strong_reference)
    present = rows[rows.survey.notna()]
    rows["survey_head"] = np.nan
    rows.loc[present.index, "survey_head"] = estimate(heads["survey"], present, present.survey)
    rows["learned_feedback"] = np.nan
    rows.loc[present.index, "learned_feedback"] = feedback_estimate(heads["feedback_head"], present)
    rows["survey_feedback_rule"] = rows.survey - 0.5 * rows.last_gap
    return rows


def replay(data, period, previous_mix, blend, old_ledger):
    # V3's historic bias and gate stay as originally trained. Only the current quote is expanded.
    rows = previous.replay_experts(data, period, old_ledger, blend)
    rows["old_survey"] = rows.survey
    rows["previous_release_model"] = previous.apply(rows, previous_mix)
    rows["survey"] = data.loc[rows.index, "survey"]
    rows["survey_time"] = data.loc[rows.index, "survey_time"]
    # Compute the gate also on recovered weeks if a later V3 package ever uses it.
    if previous_mix["with_survey"][previous.WITH.index("survey_network")] != 0:
        for _, group in rows.groupby(rows.release_date.str[:4]):
            gate, _ = previous.fit_gate(old_ledger, group.cutoff.min())
            present = group[group.survey.notna()]
            rows.loc[present.index, "survey_network"] = previous.gate_forecast(gate, present)
    else:
        rows["survey_network"] = rows.survey
    rows = previous.experts(rows, blend)
    rows["coverage_only"] = previous.apply(rows, previous_mix)
    parts = []
    for _, group in rows.groupby(rows.release_date.str[:4]):
        cutoff = group.cutoff.min()
        heads = {k: fit_head(data, cutoff, k == "survey")[0] for k in ("fallback", "survey")}
        heads["feedback_head"] = fit_feedback_head(data, cutoff)
        parts.append(give_estimates(group.copy(), heads))
    return pd.concat(parts)


def fit_mix(rows):
    if rows.release_date.max() >= "2024-01-01" or (rows.phase == "holdout").any():
        raise ValueError("Mixture selection requires pre-2024 releases")
    quoted = rows[rows.survey.notna()]
    missing = rows.survey.isna().to_numpy()
    if not missing.any() or len(quoted) < 8:
        raise ValueError("Insufficient coverage for mixture selection")
    # Mask the current survey on all validation weeks to stabilize the sparse fallback cohort.
    # Half the fallback objective uses genuine missing weeks, half uses the complete masked calendar.
    w = 0.5 / len(rows) + 0.5 * missing / missing.sum()
    return {
        "with_survey": previous.exact_weights(
            quoted[WITH].to_numpy(), quoted.target.to_numpy(), np.ones(len(quoted))
        ).tolist(),
        "without": previous.exact_weights(rows[WITHOUT].to_numpy(), rows.target.to_numpy(), w).tolist(),
    }


def apply(rows, mix):
    predictions = rows[WITHOUT].to_numpy() @ mix["without"]
    present = rows.survey.notna().to_numpy()
    predictions[present] = rows.loc[present, WITH].to_numpy() @ mix["with_survey"]
    return predictions


def train():
    data = prepared()
    old = previous.add_survey(data, previous.survey_records())
    old_weights = json.loads(previous.MIXTURE.read_text())["mix"]
    blend = json.loads(PACKAGE.read_text())["mix"]
    pre = data[data.release_date < "2024-01-01"]
    selection = pre[(pre.release_date >= "2021-01-01") & pre.strong_reference.notna()]
    test = data[data.phase == "holdout"]
    if not pre.known.max() < test.cutoff.min():
        raise ValueError("Selection overlaps evaluation")
    val = replay(pre, selection, old_weights, blend, old.loc[pre.index])
    mix = fit_mix(val)
    val["estimate"] = apply(val, mix)
    val_scores = {k: metrics(val, k) for k in ("previous_release_model", "coverage_only", "estimate")}
    print("Locked validation MAE:", {k: v["mae_kbbl"] for k, v in val_scores.items()}, mix, flush=True)
    rows = replay(data, test, old_weights, blend, old)
    original = pd.read_csv(MODEL_DIR / "holdout_forecasts.csv")
    if list(rows.release_date) != list(original.release_date) or not np.allclose(
        rows.previous_release_model, original.final_forecast, rtol=0, atol=1e-10
    ):
        raise ValueError("Previous forecasts differ from the published benchmark")
    rows["estimate"] = apply(rows, mix)
    rows["simple_survey"] = rows.survey.fillna(rows.strong_reference)
    methods = ["previous_release_model", "coverage_only", "simple_survey", "estimate"]
    splits = {
        "all": rows,
        "original_missing": rows[rows.old_survey.isna()],
        "recovered": rows[rows.survey.notna() & rows.old_survey.isna()],
        "with_survey": rows[rows.survey.notna()],
        "without_survey": rows[rows.survey.isna()],
        **{year: group for year, group in rows.groupby(rows.release_date.str[:4])},
    }
    scores = {key: {m: metrics(g, m) for m in methods} for key, g in splits.items() if len(g)}
    difference = {
        m: block_bootstrap_mean(1000 * (abs(rows.estimate - rows.target) - abs(rows[m] - rows.target)))
        for m in methods[:-1]
    }
    ready = data.known.max() + pd.Timedelta(seconds=1)
    training = {}
    trace = {"epoch": np.arange(1, EPOCHS + 1)}
    for k in ("fallback", "survey"):
        head, losses = fit_head(data, ready, k == "survey")
        training[k] = head
        trace[k + "_stopping_mae_kbbl"] = losses
    for epoch in (1, 32, 64, 96, 128, 160, 192, 224, 256):
        print(
            f"Epoch {epoch:3} | survey stopping MAE {trace['survey_stopping_mae_kbbl'][epoch - 1]:.2f} | "
            f"fallback stopping MAE {trace['fallback_stopping_mae_kbbl'][epoch - 1]:.2f} kbbl",
            flush=True,
        )
    training["feedback_head"] = fit_feedback_head(data, ready)
    history = data[data.survey.notna()][["week_ending", "known", "target", "survey"]].tail(4).copy()
    history["known"] = history.known.map(lambda t: t.isoformat())
    save_json(
        WEIGHTS,
        {
            "mix": mix,
            "heads": training,
            "feedback": history.to_dict(orient="records"),
            "with_experts": WITH,
            "without_experts": WITHOUT,
            "previous_sha256": hashlib.sha256(previous.MIXTURE.read_bytes()).hexdigest(),
        },
    )
    summary = {
        "status": "retrospective_on_previously_reviewed_calendar",
        "selection": val_scores,
        "selection_first": val.release_date.min(),
        "selection_last": val.release_date.max(),
        "selection_with": int(val.survey.notna().sum()),
        "test": scores,
        "paired_difference_95_kbbl": difference,
        "mix": mix,
        "final_stopping_epoch": {k: h["epochs"] for k, h in training.items() if "epochs" in h},
        "final_training_weeks": {k: h["training_weeks"] for k, h in training.items()},
    }
    save_json(MODEL_DIR / "residual_evaluation.json", summary)
    pd.DataFrame(trace).to_csv(MODEL_DIR / "residual_epochs.csv", index=False)
    cols = [
        "release_date",
        "week_ending",
        "cutoff",
        "target",
        "old_survey",
        "survey",
        "survey_time",
        *methods,
        "fallback_head",
        "survey_head",
    ]
    rows[cols].to_csv(MODEL_DIR / "residual_forecasts.csv", index=False)
    print(json.dumps(scores["all"], indent=2))


def settled_history(records, extra=None):
    history = pd.DataFrame(records)
    if extra is not None:
        required = ["week_ending", "known", "target", "survey"]
        history = pd.concat([history, extra[required]], ignore_index=True)
    stamps = [pd.Timestamp(t) for t in history.known]
    if any(t.tzinfo is None for t in stamps):
        raise ValueError("A settled EIA release needs its timezone-aware publication time")
    history["known"] = pd.to_datetime(stamps, utc=True)
    if not np.isfinite(history[["target", "survey"]].to_numpy(dtype=float)).all():
        raise ValueError("Invalid settled observation")
    history = history.drop_duplicates()
    if history.week_ending.duplicated().any():
        raise ValueError("Conflicting settled records for one reporting week")
    return history.sort_values("week_ending")


def predict(record, quote, decision, network, old, package, settled=None):
    base = previous.predict(record, quote, decision, network, old)
    when = pd.Timestamp(decision)
    rows = pd.DataFrame([record])
    rows["cutoff"] = when
    rows["survey"] = quote.api_change_kbbl / 1000 if base["survey_available"] else np.nan
    history = settled_history(package["feedback"], settled)
    for feature, value in feedback(history, record["week_ending"], when).items():
        rows[feature] = value
    rows["coverage_only"] = base["estimate_kbbl"] / 1000
    give_estimates(rows, package["heads"])
    return {
        "release_date": record["release_date"],
        "decision_utc": when.isoformat(),
        "survey_available": base["survey_available"],
        "estimate_kbbl": float(1000 * apply(rows, package["mix"])[0]),
        "previous_blend_kbbl": base["estimate_kbbl"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predict", type=Path)
    parser.add_argument("--as-of")
    parser.add_argument(
        "--settled-feedback",
        type=Path,
        help="CSV of subsequently settled survey and EIA changes, in million barrels",
    )
    parser.add_argument("--survey-ledger", type=Path, default=previous.SURVEY)
    parser.add_argument("--confirmed-ledger", type=Path, default=EXTRA)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.predict:
            old = json.loads(previous.MIXTURE.read_text())
            package = json.loads(WEIGHTS.read_text())
            if (
                not args.as_of
                or package["previous_sha256"] != hashlib.sha256(previous.MIXTURE.read_bytes()).hexdigest()
                or old["network_sha256"] != hashlib.sha256(PACKAGE.read_bytes()).hexdigest()
            ):
                parser.error("Use matching model bundles and specify --as-of")
            record = json.loads(args.predict.read_text())
            quotes = sources(args.confirmed_ledger, args.survey_ledger, args.as_of)
            quote = quotes.loc[record["release_date"]] if record["release_date"] in quotes.index else None
            settled = pd.read_csv(args.settled_feedback) if args.settled_feedback else None
            print(
                json.dumps(
                    predict(record, quote, args.as_of, json.loads(PACKAGE.read_text()), old, package, settled),
                    indent=2,
                )
            )
        else:
            train()
