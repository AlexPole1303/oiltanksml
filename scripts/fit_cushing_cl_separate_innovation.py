"""Fit the response to public inventory surprises and optical revisions separately."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast

STEM = "cushing_cl_separate_innovation"
METHODS = ("satellite_reference", "public_reference", "separate_revision", "separate_revision_and_volatility")


def replay():
    inventory = indexed(f"{OUTER}_forecasts")
    baseline_methods = {f"{SUPPLEMENT}:volatility": METHODS[0], f"{PUBLIC}:volatility": METHODS[1]}
    anchors = {(r["release_date"], baseline_methods[r["method"]]): r for r in read(f"{WEIGHTED}_trades")
               if r["method"] in baseline_methods}
    histories = defaultdict(list)
    for r, _ in inputs():
        issue, d = (r[k] for k in ("release_date", "decision"))
        f = inventory[issue]
        cut, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc", "known_utc"))
        pub, sat, actual = (float(f[k]) for k in (PUBLIC, SUPPLEMENT, "actual"))
        if (not cut < known <= d < r["entry"] < r["exit_at"] or actual != r["x"][0]
                or f[f"{SUPPLEMENT}_latest_label_utc"] and datetime.fromisoformat(f[f"{SUPPLEMENT}_latest_label_utc"]) >= cut):
            raise ValueError(f"The inputs use a different inventory vintage on {issue}")
        base, public = (replace_expectation(r, v) for v in (sat, pub))
        revision = sat - pub
        log_sigma = math.log(r["volatility_usd"] / 1000.)
        vectors = (base, public, (*public, revision), (*public, revision, revision * log_sigma))
        # The added feature measures the new information before EIA. The
        # residual stage keeps the public surprise as the market baseline.
        contexts = tuple(features(r["x"][2], r["volatility_usd"], actual - v) for v in (sat, pub, pub, pub))
        for method, x, context in zip(METHODS, vectors, contexts, strict=True):
            history = histories[method]
            train = [q for q in history if q["exit_at"] < d][-HISTORY:]
            current = r | dict(x=x, response_features=context)
            history.append(current)
            if len(train) < MIN_WINDOWS:
                continue
            point = volatility_forecast(train, current)
            current["base_estimate"] = point
            calibration = [q for q in train if "base_estimate" in q]
            correction, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            est = point + correction
            t = trade(r, method, est, 3, training_at=train[-1]["exit_at"],
                      calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if method in METHODS[:2]:
                a = anchors[issue, method]
                if (any(a[k] != t[k] for k in ("near_month", "far_month", "decision_at_utc", "entry_at_utc", "exit_at_utc"))
                        or any(int(a[k]) != t[k] for k in ("net_170_usd", "near_side"))
                        or not math.isclose(est, float(a["expected_long_move_usd"]), rel_tol=0., abs_tol=1e-8)):
                    raise ValueError(f"The comparison failed to reproduce {issue}/{method}")
            yield t | dict(public_surprise_kbbl=1000*(actual-pub), current_optical_revision_kbbl=1000*revision,
                           realized_long_move_usd=r["long_gross"], net_340_usd=t["gross_usd"]-340*abs(t["near_side"]))


def summarize(rows):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["study_phase"], r["method"]].append(r)
    results = {}
    for phase in PHASES:
        results[phase] = output = {}
        ref = by_method[phase, METHODS[0]]
        gains = {r["release_date"]: r for r in ref}
        for m in METHODS:
            sample = by_method[phase, m]
            if [q["release_date"] for q in sample] != list(gains):
                raise ValueError(f"The candidate changed the quote cohort in {phase}/{m}")
            differences = [r["net_170_usd"] - q["net_170_usd"] for r, q in zip(sample, ref, strict=True)]
            error = [abs(r["expected_long_move_usd"]-r["realized_long_move_usd"]) for r in sample]
            original_error = [abs(r["expected_long_move_usd"]-r["realized_long_move_usd"]) for r in ref]
            output[m] = robustness(sample) | daily_risk(sample) | dict(net_340_usd=sum(r["net_340_usd"] for r in sample),
                paired_net_delta_170_usd=sum(differences), paired_net_delta_95_usd=block_bootstrap_mean(differences),
                spread_mae_usd=float(np.mean(error)), spread_mae_delta_95_usd=block_bootstrap_mean(np.array(error)-original_error),
                changed_sides=sum(r["near_side"] != q["near_side"] for r, q in zip(sample,ref,strict=True)))
        print(phase, {m: (v["trades"], v["net_170_usd"], v["net_340_usd"], round(v["spread_mae_usd"], 2)) for m,v in output.items()})
    return dict(status="exploratory_on_reviewed_history", phases=results)


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
    summary = summarize(rows)
    export(f"{STEM}_trades", rows)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(summary, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
