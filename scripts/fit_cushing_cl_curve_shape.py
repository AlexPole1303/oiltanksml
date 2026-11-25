"""Measure the value of intermediate dated CL contracts before the EIA release."""

import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_cl_equal_model_risk import MATCHED, PUBLIC_ONLY, STEM as MATCHED_STEM, paired_evidence
from backtest_wti_cushing_spread import settlement, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features as response_features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_windows, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_mixed_optical_intervals import STEM as MIXED_STEM
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_optical_error_distribution import inputs as inventory_inputs
from fit_cushing_outer_capture_nowcast import SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import fit_correction
from fit_cushing_volatility_interactions import features as market_features
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from replay_cushing_weighted_inventory_market import replay_weighted_market

STEM = "cushing_cl_curve_shape"
SOURCES = ("official", "archive", "ibkr")
INVENTORY = ("public", "optical")
FORECASTS = ("base", "slope", "shape")
EXTENSION_CONTROLS = {
    "curve_shape": "curve",
    "public_shape": "public",
    "optical_shape": "optical",
    "public_shape_and_inventory": "public_shape",
    "optical_shape_and_inventory": "optical_shape",
}
SATELLITE_CONTROLS = {"optical": "curve", "optical_shape": "public_shape",
                      "optical_shape_and_inventory": "public_shape_and_inventory"}
FALLBACKS = ("curve_gap_slope", "curve_gap_shape", "curve_gap_calibrated")
MIXED = "learned_interval_weight"
MIXED_GAP = "mixed_and_curve_gap_shape"
COORDINATES = ("slope_usd_bbl_month", "slope_change_5_sessions_usd_bbl_month", "bend_usd_bbl",
               "bend_asymmetry_usd_bbl", "bend_change_5_sessions_usd_bbl", "asymmetry_change_5_sessions_usd_bbl")


def delivery_months(window):
    year, month = divmod(int(window["near_month"]), 100)
    indices = range(year * 12 + month - 1, year * 12 + month + 3)
    months = tuple(f"{index // 12:04d}{index % 12 + 1:02d}" for index in indices)
    if months[-1] != window["far_month"]:
        raise ValueError(f"Expected four consecutive delivery months on {window['release_date']}")
    return months


def curve_coordinates(prices):
    front, middle, back = -np.diff(prices) / 100
    return np.array(((front + middle + back) / 3, front - back, front - 2 * middle + back))


def dated_curves(cutoffs):
    windows = indexed("wti_cushing_event_windows")
    with closing(read_only(SOURCE)) as db:
        sessions = [day for day, in db.execute("""SELECT session_date FROM wti_session_basis
            WHERE full_cmes_session=1 AND cme_no_settlement_notice_url IS NULL ORDER BY session_date""")]
        books = {}
        for source in SOURCES:
            book = {}
            for day, month, _, value in price_rows(db, source, numeric_through=max(cutoffs)):
                key, price = (day, month), tick_price(value)
                if book.setdefault(key, price) != price:
                    raise ValueError(f"Conflicting {source} closes for {day}/{month}")
            books[source] = book
    clocks = [settlement(day) for day in sessions]
    for issue, cutoff in sorted(cutoffs.items()):
        end = bisect_left(clocks, cutoff)
        if end < 6 or not clocks[end - 6] < clocks[end - 1] < cutoff:
            raise ValueError(f"Missing eligible pre-release sessions on {issue}")
        previous, latest = sessions[end - 6], sessions[end - 1]
        months = delivery_months(windows[issue])
        keys = [(day, month) for day in (previous, latest) for month in months]
        complete = {source: [book[k] for k in keys] for source, book in books.items() if all(k in book for k in keys)}
        if not complete:
            raise ValueError(f"No source with all eight dated CL marks on {issue}")
        source, marks = next(iter(complete.items()))
        if any(other != marks for other in complete.values()):
            raise ValueError(f"Independent sources disagree on {issue}")
        old, new = (curve_coordinates(marks[i:i + 4]) for i in (0, 4))
        slope = (float(new[0]), float(new[0] - old[0]))
        shape = (*new[1:], *(new - old)[1:])
        yield issue, dict(source=source, crosschecked=len(complete) > 1,
                          previous_session=previous, latest_session=latest, months=months,
                          latest_mark_utc=clocks[end - 1].astimezone(timezone.utc).isoformat(), slope=slope, shape=shape)


def inventory(rows, curves):
    past_by_channel = defaultdict(list)
    mixed = indexed(f"{MIXED_STEM}_forecasts")
    for row in rows:
        issue, cutoff, channel = (row[k] for k in ("release_date", "cutoff", "channel"))
        marks = curves[issue]
        slope = tuple(np.tanh(np.asarray(marks["slope"]) / (.70, .30)))
        shape = (*slope, *np.tanh(np.asarray(marks["shape"]) / (.35, .20, .25, .15)))
        past = past_by_channel[channel]
        settled = [q for q in past if q["known"] < cutoff][-HISTORY:]
        result = row | dict(slope_design=slope, shape_design=shape, mark_source=marks["source"],
                            independent_mark_check=marks["crosschecked"], latest_cl_mark_utc=marks["latest_mark_utc"],
                            previous_cl_session=marks["previous_session"], latest_cl_session=marks["latest_session"],
                            dated_cl_months="/".join(marks["months"]),
                            latest_inventory_label_utc=settled[-1]["known"] if settled else None)
        for name, anchor in zip(INVENTORY, (row["unmeasured"], row["baseline"]), strict=True):
            residual = f"{name}_residual"
            result[residual] = row["actual"] - anchor
            result[f"{name}_base"] = anchor
            for kind, penalty in (("slope", (.25, .25)), ("shape", (.25, .25, .35, .35, .35, .35))):
                delta = fit_correction(settled, result, f"{kind}_design", residual,
                                       penalty=penalty, loss_scale=.75, bound=.50)
                result[f"{name}_{kind}"] = anchor + delta
        anchor = result["optical_base"]
        slope_delta, shape_delta = (result[f"optical_{kind}"] - anchor for kind in ("slope", "shape"))
        # The fallback is eligible only when the satellite branch has no
        # usable roof measurements. It is compared with a fixed zero update.
        gaps = [q for q in settled if not q["roofs"] and q["fitted_curve"]]
        result["fitted_curve"] = len(settled) >= MIN_WINDOWS
        result["curve_gap_average_delta"] = (slope_delta + shape_delta) / 2
        gain = fallback_gain(gaps) if not row["roofs"] else 0.
        result["curve_gap_training_weeks"] = len(gaps) if not row["roofs"] else 0
        result["curve_gap_latest_label_utc"] = gaps[-1]["known"] if gaps and not row["roofs"] else None
        result["curve_gap_gain"] = gain
        result[SUPPLEMENT] = anchor
        result[FALLBACKS[0]] = anchor + (slope_delta if not row["roofs"] else 0.)
        result[FALLBACKS[1]] = anchor + (shape_delta if not row["roofs"] else 0.)
        result[FALLBACKS[2]] = anchor + gain * result["curve_gap_average_delta"]
        intervals = mixed[issue]
        prefix = "long_" if channel == "inventory" else ""
        if (any(datetime.fromisoformat(intervals[k]) != row[k] for k in ("cutoff", "known"))
                or float(intervals["actual"]) != row["actual"] or float(intervals[prefix + SUPPLEMENT]) != anchor
                or any(intervals[k] and datetime.fromisoformat(intervals[k]) >= cutoff
                       for k in ("latest_gain_label_utc", "latest_optical_label_utc"))):
            raise ValueError(f"The archived joint optical forecast differs on {issue}")
        result[MIXED] = float(intervals[prefix + MIXED])
        result["no_mixed_optical"] = not row["roofs"] and not any(int(intervals[k]) for k in (
            "primary_roofs", "internal_roofs", "late_roofs"))
        result[MIXED_GAP] = result[MIXED] + (shape_delta if result["no_mixed_optical"] else 0.)
        past.append(result)
        yield result | dict(zip(COORDINATES, (*marks["slope"], *marks["shape"]), strict=True))


def fallback_gain(gaps):
    if len(gaps) < 26:
        return 0.
    delta = np.array([q["curve_gap_average_delta"] for q in gaps])
    error = np.array([q["optical_residual"] for q in gaps])
    weights = np.exp2(-np.arange(len(gaps) - 1, -1, -1) / 52)
    scale = .75
    prior = 13 * .05 ** 2
    gain = 0.
    for _ in range(6):
        robust = weights / np.hypot(1., (error - gain * delta) / scale)
        gain = float(np.clip((robust @ (error * delta)) / (robust @ (delta ** 2) + prior), 0., 1.))
    return gain


def market(forecasts, curves, windows):
    matched = {MATCHED: "curve", PUBLIC_ONLY: "public"}
    reference = {(r["release_date"], matched[r["method"]]): r for r in read(f"{MATCHED_STEM}_trades")
                 if r["method"] in matched}
    reference.update({(r["release_date"], "optical"): r for r in read(f"{WEIGHTED}_trades")
                      if r["method"] == f"{SUPPLEMENT}:volatility"})
    seen = defaultdict(list)
    for row in windows:
        issue, decision = row["release_date"], row["decision"]
        f = forecasts[issue]
        marks = curves[issue]
        if (not f["cutoff"] < f["known"] <= decision < row["entry"] or f["actual"] != row["x"][0]
                or (marks["months"][0], marks["months"][-1]) != (row["near_month"], row["far_month"])):
            raise ValueError(f"Incompatible inventory vintage or CL months on {issue}")
        curve = market_features(row)[-1]
        # The same four coordinates extend all models. The baseline curve
        # slope and volatility used to weigh training remain unchanged.
        shape = marks["shape"]
        methods = {"curve": (curve, None), "curve_shape": ((*curve, *shape), None)}
        for anchor in INVENTORY:
            for name, value, extra in ((anchor, f[f"{anchor}_base"], ()),
                    (f"{anchor}_shape", f[f"{anchor}_base"], shape),
                    (f"{anchor}_shape_and_inventory", f[f"{anchor}_shape"], shape)):
                methods[name] = ((*replace_expectation(row, value), *extra), f["actual"] - value)
        for name, (design, surprise) in methods.items():
            current = row | dict(x=design, response_features=response_features(row["x"][2], row["volatility_usd"], surprise))
            history = seen[name]
            settled = [q for q in history if q["exit_at"] < decision][-HISTORY:]
            history.append(current)
            if len(settled) < MIN_WINDOWS:
                continue
            point = volatility_forecast(settled, current)
            calibration = [q for q in settled if "base_estimate" in q]
            delta, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            current["base_estimate"] = point
            result = trade(row, name, point + delta, 3, training_at=settled[-1]["exit_at"],
                           calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if name in ("curve", *INVENTORY):
                old = reference[issue, name]
                if (not math.isclose(point + delta, float(old["expected_long_move_usd"]), abs_tol=1e-8, rel_tol=0)
                        or any(result[k] != int(old[k]) for k in ("near_side", "gross_usd", "net_170_usd"))):
                    raise ValueError(f"The archived {name} entry differs on {issue}")
            yield result | dict(long_move_usd=row["long_gross"], mark_source=marks["source"],
                                latest_cl_mark_utc=marks["latest_mark_utc"],
                                net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def inventory_metrics(group):
    forecasts = {f"{a}_{m}": np.array([r[f"{a}_{m}"] for r in group]) for a in INVENTORY for m in FORECASTS}
    target = np.array([r["actual"] for r in group])
    errors = {name: 1000 * np.abs(values - target) for name, values in forecasts.items()}
    metrics = {}
    for name, values in errors.items():
        anchor, kind = name.split("_")
        base = errors[f"{anchor}_base"]
        slope = errors[f"{anchor}_slope"]
        metrics[name] = dict(mae_kbbl=float(values.mean()), rmse_kbbl=float(np.sqrt(np.mean(values**2))),
                             direction_hits=int(np.sum(forecasts[name] * target > 0)),
                             paired_vs_anchor_95_kbbl=block_bootstrap_mean(values - base) if kind != "base" else None,
                             paired_vs_slope_95_kbbl=block_bootstrap_mean(values - slope) if kind == "shape" else None)
    return dict(weeks=len(group), metrics=metrics)


def market_metrics(trades):
    return robustness(trades) | daily_risk(trades) | dict(net_340_usd=sum(r["net_340_usd"] for r in trades))


def fallback_metrics(group, trades):
    missing = {SUPPLEMENT: [i for i, r in enumerate(group) if not r["roofs"]],
               MIXED: [i for i, r in enumerate(group) if r["no_mixed_optical"]]}
    report = {}
    for name, anchor in {**dict.fromkeys(FALLBACKS, SUPPLEMENT), MIXED_GAP: MIXED}.items():
        base = np.array([1000 * abs(r[anchor] - r["actual"]) for r in group])
        gaps = missing[anchor]
        candidate = np.array([1000 * abs(r[name] - r["actual"]) for r in group])
        row = dict(weeks=len(group), changed_weeks=sum(r[name] != r[anchor] for r in group),
                   anchor_mae_kbbl=float(base.mean()), mae_kbbl=float(candidate.mean()),
                   paired_vs_anchor_95_kbbl=block_bootstrap_mean(candidate - base),
                   without_optical=dict(weeks=len(gaps), anchor_mae_kbbl=float(base[gaps].mean()) if gaps else None,
                       mae_kbbl=float(candidate[gaps].mean()) if gaps else None,
                       paired_vs_anchor_95_kbbl=block_bootstrap_mean(candidate[gaps] - base[gaps])), market={})
        for suffix in ("", "_conditional_exit"):
            test = [r for r in trades if r["method"] == name + suffix]
            ref = [r for r in trades if r["method"] == anchor + suffix]
            row["market"]["conditional_exit" if suffix else "fifth_session"] = market_metrics(test) | dict(
                paired_vs_anchor=paired_evidence(test, ref)[0])
        report[name] = row
    return report


def summarize(rows, trades, fallback_trades):
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        group = [r for r in rows if r["study_phase"] == phase and r["channel"] == "inventory"]
        strata = {"all": group, "optical_present": [r for r in group if r["roofs"]],
                  "optical_absent": [r for r in group if not r["roofs"]]}
        trading = defaultdict(list)
        for r in trades:
            if r["study_phase"] == phase:
                trading[r["method"]].append(r)
        performance = {m: market_metrics(sample) | dict(
            mae_usd=float(np.mean([abs(r["expected_long_move_usd"] - r["long_move_usd"]) for r in sample])))
            for m, sample in trading.items()}
        nested = {name: paired_evidence(trading[name], trading[anchor])[0] for name, anchor in EXTENSION_CONTROLS.items()}
        comparisons = {name: paired_evidence(trading[name], trading[anchor])[0] for name, anchor in SATELLITE_CONTROLS.items()}
        fallback = fallback_metrics(group, [r for r in fallback_trades if r["study_phase"] == phase])
        report["phases"][phase] = dict(prices=dict(releases=len(group), sources=dict(Counter(r["mark_source"] for r in group)),
            independently_crosschecked=sum(r["independent_mark_check"] for r in group)),
            inventory={name: inventory_metrics(sample) for name, sample in strata.items()},
            market=performance, market_vs_same_model_without_extension=nested, market_vs_control=comparisons,
            after_review_gap_fallback=fallback)
        print(phase, {m: round(v["mae_kbbl"], 2) for m, v in report["phases"][phase]["inventory"]["all"]["metrics"].items()})
        print(phase, {m: (v["trades"], v["net_170_usd"], v["net_340_usd"]) for m, v in performance.items()})
        print(phase, "gaps", {m: (round(q["mae_kbbl"], 2), q["changed_weeks"], q["market"]["conditional_exit"]["net_170_usd"])
                              for m, q in fallback.items()}, flush=True)
    return report


def main():
    with threadpool_limits(limits=1):
        source = list(inventory_inputs())
        curves = dict(dated_curves({r["release_date"]: r["cutoff"] for r in source if r["channel"] == "inventory"}))
        rows = list(inventory(source, curves))
        forecasts = {r["release_date"]: r for r in rows if r["channel"] == "market"}
        trades = list(market(forecasts, curves, (q for q, _ in market_windows())))
        fallback_trades = list(replay_weighted_market(forecasts, (SUPPLEMENT, *FALLBACKS, MIXED, MIXED_GAP), reference=SUPPLEMENT))
    result = summarize(rows, trades, fallback_trades)
    export(f"{STEM}_forecasts", rows)
    direct = [r | dict(inventory_forecast_kbbl=None, latest_holding_label_utc=None, holding_decision_utc=None) for r in trades]
    gap = [r | dict(long_move_usd=None, mark_source=curves[r["release_date"]]["source"],
                    latest_cl_mark_utc=curves[r["release_date"]]["latest_mark_utc"])
           for r in fallback_trades if not r["method"].startswith((SUPPLEMENT, MIXED))]
    export(f"{STEM}_trades", direct + gap)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
