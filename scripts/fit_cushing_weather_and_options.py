"""Test clock-gated temperature and implied volatility on the Cushing models."""

import csv
import hashlib
import json
import math
from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime

import numpy as np
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import RAW, ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features as response_features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean, ridge_fit, transformed
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_volatility_interactions import features
from publication_clocks import EASTERN

STEM = "cushing_weather_and_options"
OPTICAL = "all_recovered"
FORECASTS = (PUBLIC, "carried_scene", OPTICAL, "public_temperature", "optical_temperature")
LEADER = "inventory_conditional"
METHODS = (
    "curve", "inventory", "optical", "inventory_temperature", "optical_temperature",
    "curve_ovx", "inventory_ovx", "optical_ovx", "inventory_vix",
)
CONDITIONAL = {LEADER: "inventory", "inventory_ovx_conditional": "inventory_ovx"}
WEATHER_STEMS = ("cushing_hrrr_training_features", "cushing_hrrr_holdout_features")
INDEX_FILES = {name: RAW / "factors" / f"{name}_cboe.csv" for name in ("OVX", "VIX")}


def weather_inputs():
    outputs = {}
    for stem in WEATHER_STEMS:
        for row in read(stem):
            key = row["release_date"], row["week_ending"]
            if key in outputs:
                raise ValueError(f"Duplicate weather release {key}")
            outputs[key] = row
    return outputs


def temperature_effect(train, row, base):
    if len(train) < MIN_WINDOWS or row["temperature"] is None:
        return 0.
    y = np.array([q["actual"] - q[base] for q in train])
    seasonal_x, seasonal_z = transformed(train, row, "seasonal")
    weather_x, weather_z = transformed(train, row, "weather")
    seasonal = ridge_fit(seasonal_x, y, seasonal_z, 1., robust=True)
    updated = ridge_fit(weather_x, y, weather_z, 1., robust=True)
    return .5 * float(np.clip(updated - seasonal, -.75, .75))


def forecast_rows():
    weather = weather_inputs()
    seen = []
    for f in read("cushing_recovered_optical_forecasts"):
        issue, week = f["release_date"], f["week_ending"]
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc", "known_utc"))
        w = weather[issue, week]
        if not cutoff < known or cutoff != datetime.fromisoformat(w["forecast_cutoff_utc"]):
            raise ValueError(f"Inventory and NOAA cutoffs differ on {issue}")
        upstream_label = f["last_training_label_utc"]
        if upstream_label and datetime.fromisoformat(upstream_label) >= cutoff:
            raise ValueError(f"A later inventory label entered the forecast on {issue}")
        temperature = None
        for role in ("previous", "current"):
            if datetime.fromisoformat(w[f"{role}_model_valid_hour_utc"]) >= cutoff:
                raise ValueError(f"A weather endpoint is newer than the forecast on {issue}")
        if w["both_endpoints_all_four_before_cutoff"] == "True":
            previous, current = (float(w[f"{role}_temperature_2m_c_median"]) for role in ("previous", "current"))
            # Temperature proxies near-freezing conditions and abrupt changes.
            temperature = (previous, current, float(max(0., -current) - max(0., -previous)))
            if not np.isfinite(temperature).all():
                raise ValueError(f"Invalid temperature endpoint on {issue}")
        angle = 2 * math.pi * date.fromisoformat(week).timetuple().tm_yday / 365.25
        season = (math.sin(angle), math.cos(angle))
        raw = {k: float(f[k]) for k in ("actual", PUBLIC, "carried_scene", OPTICAL)}
        row = raw | dict(release_date=issue, week_ending=week, study_phase=f["study_phase"], cutoff=cutoff, known=known,
                         measured=int(f["total_recovered_tanks"]) > 0, temperature=temperature, seasonal=season,
                         weather=(*season, *temperature) if temperature is not None else None)
        train = [q for q in seen if q["temperature"] is not None and q["known"] < cutoff][-HISTORY:]
        updates = {f"{name}_temperature": row[base] + temperature_effect(train, row, base)
                   for name, base in (("public", PUBLIC), ("optical", OPTICAL))}
        yield row | updates | dict(weather_training_weeks=len(train),
                                   latest_weather_training_label_utc=train[-1]["known"].isoformat() if train else None)
        seen.append(row)


def option_books():
    for name, path in INDEX_FILES.items():
        with path.open() as file:
            field = "OVX" if name == "OVX" else "CLOSE"
            values = sorted((datetime.strptime(q["DATE"], "%m/%d/%Y").date(), float(q[field]))
                            for q in csv.DictReader(file) if q[field])
        days, closes = zip(*values, strict=True)
        if len(set(days)) != len(days) or not all(math.isfinite(x) and x > 0 for x in closes):
            raise ValueError(f"Invalid Cboe closes for {name}")
        yield name, (days, np.log(closes))


def options_state(book, cutoff):
    days, closes = book
    stop = bisect_left(days, cutoff.astimezone(EASTERN).date())
    if stop < 61:
        raise ValueError(f"No prior Cboe history for {cutoff}")
    # Only use a complete calendar day before the inventory forecast, even
    # though the subsequent spread decision is made after the EIA release.
    age = (cutoff.astimezone(EASTERN).date() - days[stop - 1]).days
    if age > 7:
        raise ValueError(f"Stale Cboe close for {cutoff}")
    level = float(closes[stop - 1] - np.median(closes[stop - 61:stop - 1]))
    return float(np.clip(level, -1., 1.)), days[stop - 1].isoformat()


def trades(forecasts):
    books = dict(option_books())
    dates = {r["release_date"]: r for r in forecasts}
    incumbents = {(r["release_date"], r["method"]): r for r in read("cushing_cl_conditional_response_trades")
                  if r["method"] in ("curve", "inventory", LEADER)}
    _, _, cases = market_calendar()
    seen, evaluated = [], []
    for r in cases:
        issue = r["release_date"]
        f = dates[issue]
        if not f["cutoff"] < f["known"] <= r["decision"] < r["entry"] < r["exit_at"] or f["actual"] != r["x"][0]:
            raise ValueError(f"Invalid climate, EIA or CL clocks on {issue}")
        ovx, ovx_day = options_state(books["OVX"], f["cutoff"])
        vix, vix_day = options_state(books["VIX"], f["cutoff"])
        carry = r["x"][2]
        curve = features(r)[5]
        public, optical, warm_public, warm_optical = (replace_expectation(r, f[k]) for k in
                                                      (PUBLIC, OPTICAL, "public_temperature", "optical_temperature"))
        public_surprise = f["actual"] - f[PUBLIC]
        optical_surprise = f["actual"] - f[OPTICAL]
        x = (curve, public, optical, warm_public, warm_optical,
             (*curve, ovx, carry * ovx),
             (*public, ovx, carry * ovx, public_surprise * ovx),
             (*optical, ovx, carry * ovx, optical_surprise * ovx),
             (*public, vix, carry * vix, public_surprise * vix))
        sample = [q for q in seen if q["exit_at"] < r["decision"]][-HISTORY:]
        response_sample = [q for q in evaluated if q["exit_at"] < r["decision"]][-HISTORY:]
        row = r | dict(designs=x, response_features=response_features(carry, r["volatility_usd"], public_surprise))
        seen.append(row)
        if len(sample) < MIN_WINDOWS:
            continue
        estimates = {}
        for i, (method, vector) in enumerate(zip(METHODS, x, strict=True)):
            train = [q | dict(x=q["designs"][i]) for q in sample]
            estimates[method] = risk_prediction(train, row | dict(x=vector), len(vector))
        calibration_date = response_sample[-1]["exit_at"].isoformat() if response_sample else None
        for method, base in CONDITIONAL.items():
            past = [q | dict(base_estimate=q["estimates"][base]) for q in response_sample]
            current = row | dict(base_estimate=estimates[base])
            correction, _ = conditional_residual(past, current) if len(past) >= MIN_WINDOWS else (0., 0.)
            estimates[method] = estimates[base] + correction
        evaluated.append(row | dict(estimates=estimates))
        for method, expected in estimates.items():
            side = int(expected > HURDLE) - int(expected < -HURDLE)
            gross = side * r["long_gross"]
            if (issue, method) in incumbents:
                q = incumbents[issue, method]
                if not math.isclose(expected, float(q["expected_long_move_usd"]), abs_tol=1e-8, rel_tol=1e-10) or gross != int(q["gross_usd"]):
                    raise ValueError(f"The original strategy changed on {issue}/{method}")
            yield {k: r[k] for k in ("release_date", "study_phase", "near_month", "far_month", "volatility_usd", "alternate_source")} | dict(
                method=method, decision_at_utc=r["decision"].isoformat(), entry_at_utc=r["entry"].isoformat(),
                exit_at_utc=r["exit_at"].isoformat(), latest_training_exit_utc=sample[-1]["exit_at"].isoformat(),
                latest_calibration_exit_utc=calibration_date, forecast_cutoff_utc=f["cutoff"].isoformat(),
                ovx_session=ovx_day, vix_session=vix_day, ovx_log_vs_prior_median=ovx, vix_log_vs_prior_median=vix,
                expected_long_move_usd=expected, near_side=side, gross_usd=gross,
                alternate_gross_usd=side * r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd": side * value - (170 if n == 5 else 85) * abs(side)
                   for n, value in enumerate(r["spread_path_usd"], 2)},
                **{f"cost_{cost}_usd": cost * abs(side) for cost in COSTS},
                **{f"net_{cost}_usd": gross - cost * abs(side) for cost in COSTS})


def main():
    forecasts = list(forecast_rows())
    with threadpool_limits(limits=1):
        trading = list(trades(forecasts))
    report = dict(status="exploratory_on_reviewed_history", forecast={}, trading={})
    report["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (*INDEX_FILES.values(), *(DERIVED / f"{stem}.csv" for stem in WEATHER_STEMS))}
    for phase in PHASES:
        sample = [r for r in forecasts if r["study_phase"] == phase]
        phase_forecasts = {}
        report["forecast"][phase] = phase_forecasts
        for scope, group in (("all", sample), ("temperature_available", [r for r in sample if r["temperature"] is not None])):
            result = forecast_metrics(group, FORECASTS)
            phase_forecasts[scope] = result
            for name, base in (("public_temperature", PUBLIC), ("optical_temperature", OPTICAL)):
                delta = [1000 * (abs(q["actual"] - q[name]) - abs(q["actual"] - q[base])) for q in group]
                result["models"][name]["paired_delta_vs_no_temperature_95_kbbl"] = block_bootstrap_mean(delta)
        groups = defaultdict(list)
        for r in trading:
            if r["study_phase"] == phase:
                groups[r["method"]].append(r)
        leader = groups[LEADER]
        comparison = {}
        for name, rows in groups.items():
            if [q["release_date"] for q in rows] != [q["release_date"] for q in leader]:
                raise ValueError(f"Different trading calendar on {phase}/{name}")
            delta = [q["net_170_usd"] - l["net_170_usd"] for q, l in zip(rows, leader, strict=True)]
            years = sorted({q["release_date"][:4] for q in rows})
            comparison[name] = robustness(rows) | daily_risk(rows) | dict(
                net_340_usd=sum(q["gross_usd"] - 340 * abs(q["near_side"]) for q in rows),
                net_delta_vs_leader_usd=sum(delta), paired_delta_vs_leader_95_usd=block_bootstrap_mean(delta),
                yearly_increment_vs_leader_usd={year: sum(d for q, d in zip(rows, delta, strict=True)
                                                          if q["release_date"].startswith(year)) for year in years})
        report["trading"][phase] = comparison
        print(phase, {name: round(q["mae_kbbl"], 2) for name, q in phase_forecasts["all"]["models"].items()})
        print(phase, {name: (q["net_170_usd"], q["net_340_usd"]) for name, q in comparison.items()})
    export(f"{STEM}_forecasts", [
        {k: r[k] for k in ("release_date", "week_ending", "study_phase", "actual", "latest_weather_training_label_utc",
                          "weather_training_weeks", *FORECASTS)} | dict(cutoff_utc=r["cutoff"].isoformat(), known_utc=r["known"].isoformat(),
                             previous_temperature_c=r["temperature"][0] if r["temperature"] is not None else None,
                             current_temperature_c=r["temperature"][1] if r["temperature"] is not None else None)
        for r in forecasts])
    export(f"{STEM}_trades", trading)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
