"""Test dated daily price ranges and trading activity around Cushing releases."""

import json
import math
from bisect import bisect_left
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from statistics import median

import numpy as np

from backtest_wti_cushing_spread import settlement
from build_cushing_holdout_release_index import PRICE_FIELDS
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_reaction_residual import CONTROLS, next_day
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_extended_market_history import STEM as INVENTORY_HISTORY
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import fit_correction
from fit_cushing_volatility_neighbors import STEM as WEIGHTED

STEM = "cushing_cl_daily_activity"
SOURCES = ("archive", "ibkr")
LOOKBACK = 10
INVENTORY_BASE = "long_median_and_regional"
INVENTORY_METHODS = (INVENTORY_BASE, "prior_bar_pressure", "prior_bar_activity")
MARKET_METHODS = ("reference", "report_bar_pressure", "first_full_bar_pressure", "first_full_bar_activity")


def read_activity(db):
    books = {}
    for name in SOURCES:
        table, month, reference, close, eligibility = PRICE_FIELDS[name]
        bars = {}
        for day, contract, opening, high, low, closing, volume, path in db.execute(
                f"SELECT session_date,{month},open,high,low,{close},volume,{reference} "
                f"FROM {table} WHERE {eligibility}"):
            values = (opening, high, low, closing, volume)
            if all(v is not None and math.isfinite(v) for v in values) and low <= min(opening, closing) <= max(opening, closing) <= high:
                bars[day, contract] = (opening, high, low, closing, volume, path)
        books[name] = bars
    return books


def activity(books, days, target, months):
    left = bisect_left(days, target)
    if left < LOOKBACK or left >= len(days) or days[left] != target:
        return None
    preceding = days[left - LOOKBACK:left]
    for source, book in books.items():
        rows = [book.get((day, month)) for day in (*preceding, target) for month in months]
        if any(row is None for row in rows):
            continue
        selected = rows[-2:]
        ranges = [median(r[1] - r[2] for r in rows[leg:-2:2]) for leg in range(2)]
        volumes = [median(r[4] for r in rows[leg:-2:2]) for leg in range(2)]
        widths = [row[1] - row[2] for row in selected]
        if min(*ranges, *volumes, *widths) <= 0.:
            continue
        cloc = tuple((2 * close - high - low) / (high - low) for _, high, low, close, _, _ in selected)
        body = tuple(float(np.clip((row[3] - row[0]) / width, -2., 2.))
                     for row, width in zip(selected, ranges, strict=True))
        spread_location = .5 * (cloc[0] - cloc[1])
        spread_body = .5 * (body[0] - body[1])
        range_change = .5 * sum(log_ratio(w, n) for w, n in zip(widths, ranges, strict=True))
        volume_change = .5 * sum(log_ratio(row[4], n) for row, n in zip(selected, volumes, strict=True))
        return dict(source=source, day=target, first_reference_day=preceding[0],
                    location=spread_location, body=spread_body, range_change=range_change,
                    volume_change=volume_change, paths=tuple(sorted({r[-1] for r in rows})))
    return None


def log_ratio(value, reference):
    return float(np.clip(math.log(value / reference), -2., 2.))


def market_activity(cases):
    events = indexed("wti_cushing_event_windows")
    clocks = indexed("eia_release_clock")
    with closing(read_only(SOURCE)) as db:
        books = read_activity(db)
        days = [r[0] for r in db.execute("SELECT session_date FROM wti_session_basis WHERE full_cmes_session=1 "
                                        "AND cme_no_settlement_notice_url IS NULL ORDER BY session_date")]
    market_cases = {r["release_date"]: r for r in cases}
    for issue, event in sorted(events.items()):
        r = market_cases.get(issue)
        first = event["entry_session"]
        months = (event["near_month"], event["far_month"])
        if r is not None and months != (r["near_month"], r["far_month"]):
            raise ValueError(f"Activity bars use different contracts on {issue}")
        clock = clocks[issue]
        cutoff = datetime.fromisoformat(clock["conservative_prerelease_before_utc"])
        earlier = days[bisect_left(days, issue) - 1]
        # Timestamp the full OHLC and volume conservatively after the trading
        # day ends. The snapshot timestamp is the archive provenance.
        prior_at, report_at, first_at = (next_day(settlement(d)) for d in (earlier, issue, first))
        report_time = clock["scheduled_table_release_at_utc"]
        if r is not None and not prior_at <= cutoff < r["decision"] < first_at < r["entry"]:
            raise ValueError(f"Daily activity violates the forecast clock on {issue}")
        report_ready = bool(r is not None and report_time and cutoff <= datetime.fromisoformat(report_time) < settlement(issue)
                            and report_at <= r["decision"])
        dates = (earlier if prior_at <= cutoff else None, issue if report_ready else None, first if r is not None else None)
        values = tuple(activity(books, days, d, months) if d else None for d in dates)
        yield issue, dict(zip(("before", "report", "first"), values, strict=True)) | dict(
            cutoff=cutoff, prior_at=prior_at, report_at=report_at, first_at=first_at, months=months)


def residual(history, design, cut, end, error, penalty, scale):
    past = [q for q in history if q["known"] < cut][-HISTORY:]
    if design is None:
        return 0., None, 0
    current = dict(cutoff=cut, known=end, design=design, error=error)
    correction = fit_correction(past, current, "design", "error", penalty=penalty, loss_scale=scale, bound=.30)
    history.append(current)
    return correction, past[-1]["known"] if past else None, len(past)


def activity_features(q, anchor):
    return (q["location"], q["body"], anchor * q["range_change"], anchor * q["volume_change"])


def describe(q, available):
    return dict(feature_session=q["day"] if q else None, feature_available_at=available if q else None,
                lookback_start=q["first_reference_day"] if q else None, feature_source=q["source"] if q else None,
                input_files="|".join(q["paths"]) if q else None,
                close_location_pressure=q["location"] if q else None,
                relative_open_close_move=q["body"] if q else None, log_range_ratio=q["range_change"] if q else None,
                log_volume_ratio=q["volume_change"] if q else None)


def inventory(panels):
    histories = defaultdict(list)
    for r in read(f"{INVENTORY_HISTORY}_inventory"):
        issue = r["release_date"]
        cut, known = (datetime.fromisoformat(r[k]) for k in ("cutoff", "known"))
        actual, base = (float(r[k]) for k in ("actual", INVENTORY_BASE))
        panel = panels.get(issue)
        q = panel["before"] if panel else None
        available = panel["prior_at"] if q else None
        if panel and panel["cutoff"] != cut or not cut < known:
            raise ValueError(f"Daily inputs differ from the inventory cutoff on {issue}")
        design = activity_features(q, float(np.clip(base, -2., 2.))) if q else None
        pressure = design[:2] if design is not None else None
        arm = dict(zip(INVENTORY_METHODS[1:], (pressure, design), strict=True))
        for method in INVENTORY_METHODS:
            x = arm.get(method)
            delta, latest, count = residual(histories[method], x, cut, known, actual - base, (.30, .45, .60, .60)[:len(x or ())], .75)
            yield {k: r[k] for k in ("release_date", "week_ending", "study_phase")} | dict(
                method=method, cutoff=cut, known=known, actual=actual, estimate=base + delta,
                correction=delta, latest_training_label=latest, training_releases=count) | describe(q, available)


def market(cases, panels):
    anchors = {(r["release_date"], r["method"]): r for r in read(f"{WEIGHTED}_trades") if r["method"] in CONTROLS.values()}
    forecasts = indexed(f"{OUTER}_forecasts")
    histories = defaultdict(list)
    for r in cases:
        issue = r["release_date"]
        if (issue, CONTROLS["satellite"]) not in anchors:
            continue
        panel = panels[issue]
        f = forecasts[issue]
        sigma = r["volatility_usd"]
        cut, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc", "known_utc"))
        if not cut == panel["cutoff"] < known <= r["decision"] or float(f["actual"]) != r["x"][0]:
            raise ValueError(f"Mismatched inventory release on {issue}")
        for model, control in CONTROLS.items():
            a = anchors[issue, control]
            value = float(a["expected_long_move_usd"])
            train_at = datetime.fromisoformat(a["latest_training_exit_utc"])
            calibration_at = datetime.fromisoformat(a["latest_calibration_exit_utc"]) if a["latest_calibration_exit_utc"] else None
            if (any(a[k] != r[k] for k in ("near_month", "far_month", "study_phase"))
                    or any(datetime.fromisoformat(a[k]) != r[field] for k, field in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))
                    or not train_at < r["decision"] or calibration_at and calibration_at >= r["decision"]):
                raise ValueError(f"Mismatched reference execution on {issue}/{model}")
            surprise = float(np.clip(r["x"][0] - float(f[SUPPLEMENT if model == "satellite" else PUBLIC]), -2., 2.))
            for method, location in zip(MARKET_METHODS, (None, "report", "first", "first"), strict=True):
                q = panel[location] if location else None
                available = panel[f"{location}_at"] if q else None
                decision = max(r["decision"], available) if q else r["decision"]
                if decision >= r["entry"]:
                    raise ValueError(f"Market activity arrived after the entry on {issue}")
                x = None
                if q:
                    anchor = float(np.clip(value / sigma, -2., 2.))
                    x = activity_features(q, anchor)
                    x = (*x, surprise * q["volume_change"]) if method == "first_full_bar_activity" else x[:2]
                # Match the archived information set for the base model and
                # learn the added correction only from already closed windows.
                delta, last, count = residual(histories[model, method], x, decision, r["exit_at"],
                                              (r["long_gross"] - value) / sigma, (.25, .40, .55, .55, .70)[:len(x or ())], 1.)
                row = trade(r | dict(decision=decision), method, value + sigma * delta, 3,
                            training_at=max(train_at, last) if last else train_at, calibration_at=calibration_at)
                if (method == "reference" and any(row[k] != int(a[k]) for k in ("near_side", "gross_usd", "net_170_usd"))):
                    raise ValueError(f"Could not reproduce reference trade on {issue}")
                yield row | dict(model=model, original_prediction_usd=value, prior_signal_at=r["decision"],
                                 realized_long_move_usd=r["long_gross"], newest_activity_outcome_at=last,
                                 activity_training_windows=count, net_340_usd=row["gross_usd"]-340*abs(row["near_side"])) | describe(q, available)


def inventory_summary(rows):
    result = {}
    for phase in PHASES:
        for scope in ("all", "available"):
            subset = [r for r in rows if r["study_phase"] == phase and (scope == "all" or r["feature_source"])]
            errors = {m: np.array([1000 * abs(q["actual"] - q["estimate"]) for q in subset if q["method"] == m])
                      for m in INVENTORY_METHODS}
            baseline = errors[INVENTORY_BASE]
            result[f"{phase}:{scope}"] = {m: dict(releases=len(values), mae_kbbl=float(values.mean()),
                paired_mae_delta_95_kbbl=block_bootstrap_mean(values - baseline)) for m, values in errors.items() if len(values)}
    return result


def trade_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"], r["model"], r["method"]].append(r)
    result = {phase: {} for phase in PHASES}
    for (phase, model, method), group in groups.items():
        base = groups[phase, model, "reference"]
        if [r["release_date"] for r in base] != [r["release_date"] for r in group]:
            raise ValueError(f"Different control calendar on {phase}/{model}/{method}")
        errors = np.array([abs(r["realized_long_move_usd"] - r["expected_long_move_usd"]) for r in group])
        original = np.array([abs(r["realized_long_move_usd"] - r["expected_long_move_usd"]) for r in base])
        pnl = [r["net_170_usd"] - a["net_170_usd"] for r, a in zip(group, base, strict=True)]
        result[phase][f"{model}:{method}"] = robustness(group) | daily_risk(group) | dict(
            net_340_usd=sum(r["net_340_usd"] for r in group), active_features=sum(bool(r["feature_source"]) for r in group),
            changed_sides=sum(r["near_side"] != a["near_side"] for r,a in zip(group, base, strict=True)),
            mae_usd=float(errors.mean()), paired_mae_delta_95_usd=block_bootstrap_mean(errors - original),
            paired_pnl_delta_170_usd=sum(pnl), paired_pnl_delta_per_window_95_usd=block_bootstrap_mean(pnl))
    return result


def main():
    cases = [r for r, _ in inputs()]
    panels = dict(market_activity(cases))
    forecasts, trades = list(inventory(panels)), list(market(cases, panels))
    export(f"{STEM}_forecasts", forecasts)
    export(f"{STEM}_trades", trades)
    summary = dict(status="exploratory_on_reviewed_daily_trade_bars", inventory=inventory_summary(forecasts), trading=trade_summary(trades))
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode())
    for phase, methods in summary["trading"].items():
        print(phase, {m: (q["active_features"], q["trades"], q["net_170_usd"], q["net_340_usd"])
                      for m, q in methods.items()})


if __name__ == "__main__":
    main()
