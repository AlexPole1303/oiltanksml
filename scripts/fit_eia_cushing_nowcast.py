"""Compare expanding-window stock-change forecasts before first EIA release."""
import json
import math
from datetime import date, datetime

import numpy as np

from backtest_cushing_operator_strategy import PHASES, ridge
from build_operator_development_inputs import one_per_key
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write

STEM = "cushing_eia_walk_forward"
METHODS = ("previous_change", "previous_plus_optical", "ridge_public", "ridge_public_optical")


def inputs():
    clocks = one_per_key((r for r in read_csv("study_week_input_coverage") if r["study_phase"] in PHASES), "week_ending")
    oil = one_per_key((r for r in read_csv("oil_market_context") if r["phase"] == "pre_release"
                       and r["release_date"] <= "2024-01-05"), "release_date")
    for r in read_csv("sentinel2_space_time_coverage_eia_comparison"):
        if r["study_phase"] not in PHASES or not (r["eia_first_release_change_kbbl"] and r["prior_kbbl"]):
            continue
        week, issue = r["week_ending"], r["release_date"]
        c = clocks[week,]
        cutoff, available = map(datetime.fromisoformat, (r["cutoff_utc"], c["decision_at_utc"]))
        if c["release_date"] != issue or cutoff != datetime.fromisoformat(c["inputs_available_before_utc"]) or available < cutoff:
            raise ValueError(f"Inconsistent pre-release clock for {week}/{issue}")
        context = oil.get((issue,), {})
        if context and datetime.fromisoformat(context["cutoff_utc"]) > cutoff:
            context = {}
        spread = float(context["cl1_minus_cl2"]) if context.get("cl1_minus_cl2") else None
        prior, first_release = (float(r[k])/1000 for k in ("prior_kbbl", "eia_first_release_change_kbbl"))
        optical = float(r["raw_kbbl"])/1000 if r["raw_kbbl"] else None
        seasonal_angle = 2*math.pi*date.fromisoformat(week).timetuple().tm_yday/365.25
        yield dict(study_phase=r["study_phase"], week_ending=week, release_date=issue, forecast_cutoff_utc=cutoff.isoformat(),
            target_known_by_utc=available.isoformat(), target_million_bbl=first_release, prior_million_bbl=prior,
            raw_optical_million_bbl=optical, optical_available=optical is not None,
            eia_features=(prior, math.sin(seasonal_angle), math.cos(seasonal_angle), spread),
            optical_features=(optical if optical is not None else 0., float(r["baseline_b04_space_time_fraction"] or 0),
                              float(optical is not None)), label=first_release)


def predict(rows):
    for row in rows:
        cutoff = datetime.fromisoformat(row["forecast_cutoff_utc"])
        train = [r for r in rows if datetime.fromisoformat(r["target_known_by_utc"]) < cutoff]
        optical_train = sum(r["optical_available"] for r in train)
        if len(train) < 52 or optical_train < 20:
            continue
        yield {k: v for k, v in row.items() if k not in ("eia_features", "optical_features", "label")} | dict(
            training_weeks=len(train), training_weeks_with_optical=optical_train,
            latest_training_release_available_utc=max(r["target_known_by_utc"] for r in train),
            previous_change=row["prior_million_bbl"], previous_plus_optical=row["prior_million_bbl"]+(row["raw_optical_million_bbl"] or 0),
            ridge_public=float(ridge(train, [row], False)[0]), ridge_public_optical=float(ridge(train, [row], True)[0]))


def assess(rows):
    actual = np.array([r["target_million_bbl"] for r in rows])
    if not len(actual):
        return dict(weeks=0)
    comparison = {}
    for method in METHODS:
        estimate = np.array([r[method] for r in rows])
        error = estimate-actual
        comparison[method] = dict(mean_absolute_error_million_bbl=float(np.mean(abs(error))),
            median_absolute_error_million_bbl=float(np.median(abs(error))),
            matching_nonzero_direction=int(np.sum(estimate*actual>0)), mean_signed_error_million_bbl=float(np.mean(error)))
    return dict(weeks=len(rows), methods=comparison)


def main():
    all_rows = list(inputs())
    forecasts = list(predict(all_rows))
    report = dict(source_weeks=len(all_rows), forecasts=len(forecasts), phases={phase:{scope:assess([
            r for r in forecasts if r["study_phase"] == phase and (scope == "all" or r["optical_available"])])
            for scope in ("all", "with_optical")} for phase in PHASES},
        interpretation="A weekly outcome can be used for training only after the conservative first-release clock and strictly "
        "before the target forecast cutoff. No closed-period EIA outcomes are read. Predictive metrics are historical and separate "
        "from trading returns.")
    export(f"{STEM}_forecasts", forecasts)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
