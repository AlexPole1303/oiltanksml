"""Check sensitivity to the retrospectively reprocessed public Cushing Watch satellite-only series."""

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen

import fit_cushing_survey_balance as model
import numpy as np
import pandas as pd
from collect_public import TLS
from threadpoolctl import threadpool_limits

SNAPSHOT_SHA256 = "de8518200a4df2ed3c12520a95c0b7122f72dba138ef2bf708ca364166dea988"
SOURCE = "https://cushingwatch.com/assets/cw-data.js?v=de851820"


def signals(rows, passes):
    values = np.full(len(rows), np.nan)
    for i, week in enumerate(pd.to_datetime(rows.week_ending)):
        current = passes[passes.day.between(week - pd.Timedelta(days=3), week)]
        if current.empty:
            continue
        obs = current.iloc[-1]
        prior = passes[passes.day.between(obs.day - pd.Timedelta(days=15), obs.day - pd.Timedelta(days=5))]
        if not prior.empty:
            last = prior.iloc[-1]
            change = (obs.kbbl - last.kbbl) * 7 / (obs.day - last.day).days / 1000.0
            values[i] = np.clip(change, -4, 4)
    return values


def adjustment(rows, passes):
    sensor = signals(rows, passes)
    delta = np.clip(sensor - rows.estimate.to_numpy(), -2.0, 2.0)
    return np.nan_to_num(delta), int(np.isfinite(sensor).sum())


def audit(raw):
    if hashlib.sha256(raw).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("This audit requires the published snapshot with the recorded SHA-256.")
    # Parse a literal numeric array, without executing third-party JavaScript.
    table = json.loads(re.search(rb"window\.CW_sat = (\[.*?\]);", raw, re.DOTALL)[1])
    passes = pd.DataFrame(table, columns=["day", "kbbl", "fill", "flag"]).assign(
        day=lambda x: pd.to_datetime(x.day)
    )
    if not passes.day.is_unique or not passes.day.is_monotonic_increasing:
        raise ValueError("Expected chronological and distinct satellite passes.")
    _, val = model.calibration_replay(*model.setup())
    mix = json.loads(model.WEIGHTS.read_bytes())["mix"]
    val["estimate"] = model.apply(val, mix)
    delta, n_val = adjustment(val, passes)
    weight = model.v3.exact_weights(
        np.column_stack([val.estimate, val.estimate + delta]),
        val.target.to_numpy(),
        np.ones(len(val)) / len(val),
    )[1]
    holdout = pd.read_csv(model.MODEL_DIR / "balance_forecasts.csv")
    delta, n_test = adjustment(holdout, passes)
    return {
        "snapshot_sha256": SNAPSHOT_SHA256,
        "historical_passes": len(passes),
        "calibration_paired_weeks": n_val,
        "evaluation_paired_weeks": n_test,
        "calibrated_weight": float(weight),
        "evaluation_mae_kbbl": 1000 * float(abs(holdout.estimate + weight * delta - holdout.target).mean()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, help="Local copy of the cited public chart's cw-data.js")
    args = parser.parse_args()
    if args.snapshot:
        raw = args.snapshot.read_bytes()
    else:
        with urlopen(SOURCE, context=TLS, timeout=20) as response:
            raw = response.read()
    with threadpool_limits(limits=1):
        print(json.dumps(audit(raw), indent=2))
