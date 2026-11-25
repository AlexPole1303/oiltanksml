"""Evaluate whether dated CFTC positioning improves the CL release strategy."""

import json
import math
from collections import defaultdict
from datetime import date, datetime

import numpy as np
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_satellite_innovation import CONTROL, LEADER, anchors
from fit_cushing_volatility_interactions import features as market_features

STEM = "cushing_cl_positioning"
METHODS = (CONTROL, "inventory", LEADER, "position_curve", "position_inventory", "position_surprise", "position_optical",
           "position_inventory_conditional", "position_surprise_conditional", "position_optical_conditional")
FULL_METHODS = ("position_curve", "position_inventory", "position_surprise", "position_optical")
CALIBRATION = {"position_inventory_conditional": "position_inventory",
               "position_surprise_conditional": "position_surprise", "position_optical_conditional": "position_optical"}


def positioning():
    panel = {}
    sources = indexed("cushing_holdout_cftc_audit_releases")
    observed = {}
    for r in sorted(read("oil_market_context"), key=lambda q: q["release_date"]):
        if r["phase"] != "pre_release":
            continue
        issue = r["release_date"]
        cutoff, available = (datetime.fromisoformat(r[k]) for k in ("cutoff_utc", "cftc_conservative_from_utc"))
        day = r["last_cftc_report_date"]
        if not date.fromisoformat(day) < cutoff.date() or available > cutoff:
            raise ValueError(f"CFTC positions arrived late on {issue}")
        report = sources.get(issue)
        if report is not None and (report["source_status"] != "values_match" or
                report["report_footer_day_ended_by_cutoff"] != "True" or report["report_date"] != day or
                datetime.fromisoformat(report["saved_cftc_available_at_utc"]) != available):
            raise ValueError(f"The archived CFTC report differs on {issue}")
        interest, longs, shorts = (int(r[k]) for k in ("open_interest", "money_manager_long", "money_manager_short"))
        if interest <= 0 or not 0 <= longs <= interest or not 0 <= shorts <= interest:
            raise ValueError(f"Invalid CFTC position counts on {issue}")
        share = (longs - shorts) / interest
        if day in observed and observed[day] != share:
            raise ValueError(f"Different copies of CFTC report {day}")
        past = [value for old, value in sorted(observed.items()) if old < day][-52:]
        deviation = share - float(np.median(past)) if len(past) >= 20 else 0.
        observed[day] = share
        panel[issue] = dict(position_cutoff=cutoff, position_available=available, report_date=day,
                            money_manager_net_share=share, position_deviation=deviation, position_history=len(past))
    return panel


def replay():
    reported = positioning()
    forecasts = indexed("cushing_recovered_optical_forecasts")
    refs = anchors()
    prior_base = {r["release_date"]: r for r in read("cushing_cl_conditional_response_trades") if r["method"] == "inventory"}
    _, _, cases = market_calendar()
    seen, scored = [], []
    for r in cases:
        issue, decision = r["release_date"], r["decision"]
        position = reported[issue]
        f = forecasts[issue]
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc", "known_utc"))
        actual = float(f["actual"])
        if (not position["position_available"] <= position["position_cutoff"] == cutoff < known <= decision < r["entry"] < r["exit_at"]
            or actual != r["x"][0]):
            raise ValueError(f"Invalid forecast or CFTC clock on {issue}")
        value = position["position_deviation"]
        carry = r["x"][2]
        surprise, optical_surprise = (actual - float(f[k]) for k in (PUBLIC, "all_recovered"))
        curve = market_features(r)[5]
        base, hybrid = (replace_expectation(r, float(f[k])) for k in (PUBLIC, "all_recovered"))
        vectors = dict(position_curve=(*curve, value, carry * value),
                       position_inventory=(*base, value, carry * value),
                       position_surprise=(*base, value, carry * value, surprise * value),
                       position_optical=(*hybrid, value, carry * value, optical_surprise * value))
        train = [q for q in seen if q["exit_at"] < decision][-HISTORY:]
        current = r | position | dict(vectors=vectors, base=base, surprise=surprise, optical_surprise=optical_surprise)
        seen.append(current)
        if len(train) < MIN_WINDOWS:
            continue
        past = [q for q in scored if q["exit_at"] < decision][-HISTORY:]
        anchor = refs[issue] | {"inventory": prior_base[issue]}
        estimates = {m: float(anchor[m]["expected_long_move_usd"]) for m in (CONTROL, "inventory", LEADER)}
        for m in FULL_METHODS:
            sample = [q | dict(x=q["vectors"][m]) for q in train]
            estimates[m] = risk_prediction(sample, current | dict(x=vectors[m]), len(vectors[m]))
        if not math.isclose(estimates["inventory"], risk_prediction([q | dict(x=q["base"]) for q in train],
                current | dict(x=base), len(base)), abs_tol=1e-8):
            raise ValueError(f"Public reference model differs on {issue}")
        for m, arm in CALIBRATION.items():
            field = "optical_surprise" if arm == "position_optical" else "surprise"
            history = [q | dict(response_features=features(q["x"][2], q["volatility_usd"], q[field]),
                                 base_estimate=q["estimates"][arm]) for q in past]
            case = current | dict(response_features=features(carry, r["volatility_usd"], current[field]),
                                  base_estimate=estimates[arm])
            update, _ = conditional_residual(history, case) if len(history) >= MIN_WINDOWS else (0., 0.)
            estimates[m] = estimates[arm] + update
        scored.append(current | dict(estimates=estimates))
        for method, prediction in estimates.items():
            side = int(prediction > HURDLE) - int(prediction < -HURDLE)
            gross = side * r["long_gross"]
            if method in anchor:
                a = anchor[method]
                if (side != int(a["near_side"]) or gross != int(a["gross_usd"]) or
                    any(a[k] != r[k] for k in ("near_month", "far_month")) or
                    any(datetime.fromisoformat(a[k]) != r[field] for k, field in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))):
                    raise ValueError(f"Different source decision on {issue}/{method}")
            yield {k: r[k] for k in ("release_date", "study_phase", "near_month", "far_month", "volatility_usd", "alternate_source")} | dict(
                method=method, decision_at_utc=decision.isoformat(), entry_at_utc=r["entry"].isoformat(),
                exit_at_utc=r["exit_at"].isoformat(), cftc_report_date=position["report_date"],
                cftc_available_utc=position["position_available"].isoformat(), forecast_cutoff_utc=cutoff.isoformat(),
                cftc_net_share=position["money_manager_net_share"], cftc_deviation=value,
                cftc_history=position["position_history"], latest_training_exit_utc=train[-1]["exit_at"].isoformat(),
                latest_calibration_exit_utc=past[-1]["exit_at"].isoformat() if past else None,
                expected_long_move_usd=prediction, near_side=side, gross_usd=gross,
                alternate_gross_usd=side * r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd": side * v - (170 if n == 5 else 85) * abs(side)
                   for n, v in enumerate(r["spread_path_usd"], 2)},
                **{f"cost_{c}_usd": c * abs(side) for c in COSTS},
                **{f"net_{c}_usd": gross - c * abs(side) for c in COSTS})


def main():
    with threadpool_limits(limits=1):
        trades = list(replay())
    results = {}
    for phase in PHASES:
        by_method = {m: [r for r in trades if r["study_phase"] == phase and r["method"] == m] for m in METHODS}
        base = by_method[LEADER]
        group = {}
        for method, rows in by_method.items():
            if [r["release_date"] for r in rows] != [r["release_date"] for r in base]:
                raise ValueError(f"Trading calendars differ on {phase}/{method}")
            delta = [r["net_170_usd"] - q["net_170_usd"] for r, q in zip(rows, base, strict=True)]
            year_delta = defaultdict(int)
            for row, difference in zip(rows, delta, strict=True):
                year_delta[row["release_date"][:4]] += difference
            group[method] = robustness(rows) | daily_risk(rows) | dict(
                net_340_usd=sum(r["gross_usd"] - 340 * abs(r["near_side"]) for r in rows),
                delta_vs_leader_usd=sum(delta), paired_delta_vs_leader_95_usd=block_bootstrap_mean(delta),
                annual_delta_vs_leader_usd=dict(year_delta))
        results[phase] = group
        print(phase, {m: (x["net_170_usd"], x["net_340_usd"], x["daily_drawdown_usd"]) for m, x in group.items()})
    export(f"{STEM}_trades", trades)
    report = dict(status="exploratory_on_reviewed_history", results=results)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
