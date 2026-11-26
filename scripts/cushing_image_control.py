"""Retain the public inputs and refit the selected Cushing architecture without any image observations."""

import argparse
import json

import fit_cushing_network_calibration as calibration
import fit_cushing_survey_balance as full
import numpy as np
import pandas as pd
from fit_cushing_cl_regime_strategy import HISTORY
from fit_cushing_inventory_network import MODEL_DIR, OPTICAL, load_panel, save_json
from fit_cushing_storage_state import fit_correction, regional_adjustment
from fit_cushing_volatility_neighbors import REGIONAL, median_residual
from threadpoolctl import threadpool_limits

INPUTS = MODEL_DIR / "image_control_market.csv"
POINTS = MODEL_DIR / "image_control_forecasts.csv"
WEIGHTS = MODEL_DIR / "image_control_weights.json"
REPORT = MODEL_DIR / "image_control_evaluation.json"
REGION = [f"region_{i}" for i in range(6)]
CURVE = [f"curve_{i}" for i in range(6)]
COORD = ["spot_level", "spot_move", "low_stock"]


def export_public_inputs():
    from build_s1_reference_matches import read
    from fit_cushing_cl_curve_shape import COORDINATES
    from fit_cushing_robust_stack import indexed
    from fit_cushing_storage_state import source_rows
    from fit_cushing_volatility_neighbors import archived_coordinates

    spot = indexed("cushing_spot_vintages")
    public = indexed("cushing_public_state_forecasts")
    shapes = {
        r["release_date"]: r
        for r in read("cushing_cl_curve_shape_forecasts")
        if r["channel"] == "inventory"
    }
    records = []
    for row in source_rows():
        day = row["release_date"]
        s, p = shapes[day], public[day]
        last = max(
            pd.Timestamp(s["latest_cl_mark_utc"]),
            pd.Timestamp(row["previous_report_known_utc"]),
        )
        if not last < row["cutoff"] or p["spot_vintage_date"] >= day:
            raise ValueError(f"Late public input for {day}")
        shape = np.tanh(
            np.array([float(s[k]) for k in COORDINATES])
            / [0.7, 0.3, 0.35, 0.20, 0.25, 0.15]
        )
        records.append(
            dict(
                release_date=day,
                week_ending=row["week_ending"],
                cutoff=row["cutoff"],
                known=row["known"],
                target=row["actual"],
                public_ensemble=float(p["public_ensemble"]),
                basis_public=float(p["basis_public"]),
                **dict(zip(REGION, row["regional_state"], strict=True)),
                **dict(zip(CURVE, shape, strict=True)),
                **dict(
                    zip(
                        COORD,
                        archived_coordinates(spot[day], row["state"]),
                        strict=True,
                    )
                ),
            )
        )
    pd.DataFrame(records).to_csv(INPUTS, index=False)


def reference(row, history):
    settled = [q for q in history if q["known"] < row["cutoff"]]
    base = row["basis_public"]
    regional = base + regional_adjustment(settled[-312:], row)
    temp = row | {REGIONAL: regional}
    market = regional + median_residual(settled[-312:], temp, 2)
    shape = fit_correction(
        settled[-HISTORY:],
        row,
        "shape",
        "market_error",
        penalty=(0.25, 0.25, 0.35, 0.35, 0.35, 0.35),
        loss_scale=0.75,
        bound=0.50,
    )
    return market + shape, regional, market


def market_row(record):
    return {
        "cutoff": pd.Timestamp(record["cutoff"]),
        "basis_public": float(record["basis_public"]),
        "regional_state": tuple(float(record[k]) for k in REGION),
        "shape": tuple(float(record[k]) for k in CURVE),
        "inventory_coordinates": tuple(float(record[k]) for k in COORD),
    }


def panel_and_history():
    panel = load_panel()
    public = pd.read_csv(INPUTS)
    eligible = panel[panel.strong_reference.notna()]
    if list(public.release_date) != list(eligible.release_date) or not np.allclose(
        public.target, eligible.target, atol=1e-10
    ):
        raise ValueError("Control calendar or vintage mismatch")
    if not np.array_equal(
        pd.to_datetime(public.cutoff, utc=True), eligible.cutoff
    ) or not np.array_equal(pd.to_datetime(public.known, utc=True), eligible.known):
        raise ValueError("Control chronology mismatch")
    if not np.allclose(
        public.public_ensemble, eligible.public_ensemble, rtol=0, atol=1e-10
    ):
        raise ValueError("Control public ensemble mismatch")
    history, values = [], []
    for d in public.to_dict("records"):
        r = market_row(d)
        val, region, market = reference(r, history)
        y = d["target"]
        history.append(
            r
            | dict(
                known=pd.Timestamp(d["known"]),
                actual=y,
                hybrid_residual=y - r["basis_public"],
                market_error=y - market,
                **{REGIONAL: region},
            )
        )
        values.append(val)
    # Replace all image-containing anchors before enrichment constructs any lagged residuals.
    panel["strong_reference"] = panel.release_date.map(
        dict(zip(public.release_date, values, strict=True))
    )
    panel[list(OPTICAL)] = 0.0
    panel["previous_mlp"] = np.nan
    panel["optical_reference"] = np.nan
    return panel, history


def paired_interval(delta, mask):
    # Sample calendar blocks, keeping missing or unmeasured weeks in their original places.
    x = np.where(mask, delta, np.nan)
    if np.isfinite(x).sum() < 8:
        return None
    rng = np.random.default_rng(74125)
    ix = (
        rng.integers(len(x), size=(4000, int(np.ceil(len(x) / 4))))[:, :, None]
        + np.arange(4)
    ) % len(x)
    sample = x[ix.reshape(4000, -1)[:, : len(x)]]
    valid = np.isfinite(sample).any(axis=1)
    return np.quantile(np.nanmean(sample[valid], axis=1), [0.025, 0.975]).tolist()


def replay_selected(data, locked):
    names = set().union(*locked.values())
    pre = data[data.release_date < "2024-01-01"]
    warm = pre[
        (pre.release_date >= "2020-07-01")
        & (pre.release_date < "2021-01-01")
        & pre.strong_reference.notna()
    ].copy()
    val = pre[(pre.release_date >= "2021-01-01") & pre.strong_reference.notna()].copy()
    test = data[data.phase == "holdout"].copy()
    frames = []
    for period, pool in [(warm, pre), (val, pre), (test, data)]:
        period["precise"] = full.benchmark(pool, period)
        # The published model refits conditional branches by years and eight-release blocks.
        for _, year in period.groupby(period.release_date.str[:4]):
            for j in range(0, len(year), 8):
                block = year.iloc[j : j + 8]
                for name in sorted(names.intersection(full.CONFIGS)):
                    head = full.rec.fit(pool, block.cutoff.min(), *full.CONFIGS[name])
                    period.loc[block.index, name] = full.rec.predict(head, block)
        frames.append(period)
    warm, val, test = frames
    val = full.overlay_replay(val, warm)
    test = full.overlay_replay(test, pd.concat([warm, val]))
    test["no_images"] = full.apply(test, locked)
    return warm, val, test


def train():
    panel, _ = panel_and_history()
    _, data = full.setup(panel=panel, imagery=False)
    locked = json.loads(full.WEIGHTS.read_text())["mix"]
    names = set().union(*locked.values())
    warm, val, test = replay_selected(data, locked)
    old = pd.read_csv(MODEL_DIR / "balance_forecasts.csv")
    _, originals = full.setup()
    _, _, reproduced = replay_selected(originals, locked)
    if not np.allclose(reproduced.no_images, old.estimate, atol=1e-9, rtol=0):
        raise ValueError("Full-arm replay differs from the published 148 predictions")
    if list(old.release_date) != list(test.release_date) or not np.allclose(
        old.target, test.target
    ):
        raise ValueError("Unmatched published forecasts")
    if not np.allclose(old.survey, test.survey, equal_nan=True) or list(
        old.survey_time.fillna("")
    ) != list(test.survey_time):
        raise ValueError("Unmatched survey data")
    original = load_panel().set_index("release_date").loc[old.release_date]
    old["primary_image"] = original.measured.to_numpy().astype(bool)
    old["no_images"] = test.no_images.to_numpy()
    old["matched_fallback"] = old.no_images.where(old.survey.notna(), old.estimate)
    cols = [
        "release_date",
        "week_ending",
        "cutoff",
        "survey_time",
        "primary_image",
        "target",
        "survey",
        "no_images",
        "matched_fallback",
        "estimate",
    ]
    old[cols].to_csv(POINTS, index=False)
    diff = 1000 * (
        (old.no_images - old.target).abs() - (old.estimate - old.target).abs()
    )
    report = {}
    for name, mask in {
        "all": np.ones(len(old), bool),
        "survey": old.survey.notna(),
        "original_survey": old.original_quote.notna(),
        "missing": old.survey.isna(),
        "primary_image": old.primary_image,
        "no_primary_image": ~old.primary_image,
        "survey_and_image": old.survey.notna() & old.primary_image,
    }.items():
        group = old[mask]
        report[name] = {
            "n": len(group),
            "mae_kbbl": {
                k: float(1000 * (group[k] - group.target).abs().mean())
                for k in ("estimate", "no_images", "matched_fallback")
            },
            "api_mae_kbbl": float(1000 * (group.survey - group.target).abs().mean())
            if group.survey.notna().all()
            else None,
            "image_gain_kbbl": float(diff[mask].mean()),
            "paired_95": paired_interval(diff, mask),
        }
    save_json(REPORT, report)
    ready = data.known.max() + pd.Timedelta(seconds=1)
    mix = json.loads(full.v5.PACKAGE.read_text())["mix"]
    net = {"mix": mix, **calibration.pair(data, ready, tuple(mix["hidden_layers"]))}
    survey = json.loads(full.v3.MIXTURE.read_text())
    survey["survey_gate"], _ = full.v3.fit_gate(
        full.v3.add_survey(data, full.v3.survey_records()), ready
    )
    package = {
        "mix": locked,
        "head": {
            k: full.rec.fit(data, ready, *full.CONFIGS[k])
            for k in sorted(names.intersection(full.CONFIGS))
        },
        "overlays": {
            k: full.rec.fit_overlay(pd.concat([warm, val, test]), ready, k)
            for k in sorted(names.intersection(full.rec.OVERLAYS))
        },
        "updated": full.v5.updated_heads(data, ready),
        "settled": json.loads(full.WEIGHTS.read_text())["settled"],
    }
    save_json(
        WEIGHTS,
        {
            "network": net,
            "survey": survey,
            "model": package,
            "residual_mix": json.loads(full.v4.WEIGHTS.read_text())["mix"],
        },
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-inputs", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.export_inputs:
            export_public_inputs()
        train()
