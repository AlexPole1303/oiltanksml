"""Combine existing inventory forecasts using realized errors at comparable optical coverage."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_extended_market_history import STEM as EXTENDED
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import INTERNAL_CALIBRATION, STEM as OUTER, SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_terminal_consensus import STEM as BREADTH
from fit_cushing_terminal_weekly_transfers import LIMITED_MARKET
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_leading_models import LINEAGE, STEM as LEADER, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_coverage_stack"
BASE = "long_median_and_regional"
EXPERTS = (BASE, LIMITED_MARKET, SUPPLEMENT, INTERNAL_CALIBRATION)
PRIOR_WEIGHTS = np.array((.70, .10, .10, .10))
METHODS = ("fixed_blend", "pooled_stack", "coverage_stack", "coverage_loss_weights")
MARKET_REFERENCE = "weighted_satellite"
MARKET_METHODS = (MARKET_REFERENCE, BASE, *METHODS)
WINDOW = 208


def coverage_group(effective):
    return "no_terminal" if effective == 0 else "below_two" if effective < 2 else "two_or_more"


def source_rows():
    extended, breadth, outer = [indexed(s) for s in (f"{EXTENDED}_inventory", f"{BREADTH}_forecasts", f"{OUTER}_forecasts")]
    if extended.keys() != breadth.keys() or extended.keys() != outer.keys():
        raise ValueError("Inventory models use different release dates")
    for issue, r in sorted(extended.items()):
        q, o = breadth[issue], outer[issue]
        cutoff, known = (datetime.fromisoformat(r[k]) for k in ("cutoff", "known"))
        for record, cut_key, known_key, label_keys in (
            (r, "cutoff", "known", ("latest_training_release",)),
            (q, "cutoff", "known", ("latest_calibration_label_utc",)),
            (o, "cutoff_utc", "known_utc", tuple(f"{k}_latest_label_utc" for k in EXPERTS[2:])),
        ):
            if (datetime.fromisoformat(record[cut_key]) != cutoff or datetime.fromisoformat(record[known_key]) != known
                    or any(r[k] != record[k] for k in ("week_ending", "study_phase"))
                    or float(r["actual"]) != float(record["actual"])
                    or any(datetime.fromisoformat(record[k]) >= cutoff for k in label_keys if record[k])):
                raise ValueError(f"Invalid inventory source on {issue}")
        n = float(q["effective_terminals"])
        values = np.array([float((r if m in EXPERTS[:2] else o)[m]) for m in EXPERTS])
        if not cutoff < known or not 0 <= n <= int(q["terminals"]) + 1e-8 or not np.isfinite(values).all():
            raise ValueError(f"Invalid inventory forecast or terminal count on {issue}")
        yield {k: r[k] for k in ("release_date", "week_ending", "study_phase")} | dict(
            cutoff=cutoff, known=known, actual=float(r["actual"]), effective_terminals=n, estimates=values)


def coverage_weights(history, row):
    past = np.array([r["effective_terminals"] for r in history])
    current = row["effective_terminals"]
    distance = (np.log1p(past) - math.log1p(current)) / .60
    same_availability = (past == 0) == (current == 0)
    return .30 + .70 * np.exp(-.5 * distance * distance) * same_availability


def stack(design, target, mass):
    weights = mass / mass.sum()

    def loss(beta):
        residual = design @ beta - target
        smooth = np.hypot(residual, .15)
        offset = beta - PRIOR_WEIGHTS
        return weights @ smooth + .035 * (offset @ offset), design.T @ (weights * residual / smooth) + .07 * offset

    result = minimize(loss, PRIOR_WEIGHTS, jac=True, method="SLSQP", bounds=[(0., 1.)] * len(EXPERTS),
                      constraints={"type": "eq", "fun": lambda w: w.sum()-1., "jac": np.ones_like},
                      options={"ftol": 1e-10, "maxiter": 100})
    if not result.success or not np.isfinite(result.x).all() or not math.isclose(result.x.sum(), 1., abs_tol=1e-9):
        raise ValueError(f"Coverage forecast weights failed: {result.message}")
    return result.x


def fitted_weights(history, row):
    if len(history) < MIN_WINDOWS:
        return dict.fromkeys(METHODS, PRIOR_WEIGHTS)
    ages = np.array([(row["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in history])
    recency = np.exp2(-ages / 78.)
    localized = recency * coverage_weights(history, row)
    values = np.stack([r["estimates"] for r in history])
    actual = np.array([r["actual"] for r in history])
    # Compare forecast errors on the same weeks and bound the influence of
    # isolated weeks on the loss-based weights.
    error = abs(values - actual[:, None])
    advantage = np.clip(error[:, 0, None] - error, -.5, .5)
    relative = (localized @ advantage) / (16. + localized.sum())
    return dict(fixed_blend=PRIOR_WEIGHTS, pooled_stack=stack(values, actual, recency),
                coverage_stack=stack(values, actual, localized), coverage_loss_weights=softmax(np.log(PRIOR_WEIGHTS) + relative / .035))


def predict(history, row):
    available = [r for r in history if r["known"] < row["cutoff"]][-WINDOW:]
    weights = fitted_weights(available, row)
    estimates = {m: float(w @ row["estimates"]) for m, w in weights.items()}
    return estimates, {m: w.tolist() for m, w in weights.items()}, available


def forecast_rows(history):
    for i, r in enumerate(history):
        estimates, weights, available = predict(history[:i], r)
        yield {k: r[k] for k in ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual", "effective_terminals")} | {
            **dict(zip(EXPERTS, map(float, r["estimates"]), strict=True)), **estimates} | dict(
            coverage_group=coverage_group(r["effective_terminals"]), training_weeks=len(available),
            latest_training_outcome=available[-1]["known"] if available else None,
            weights_json=json.dumps(weights, separators=(",", ":")))


def market(forecasts):
    book = {r["release_date"]: r for r in forecasts}
    reference = {r["release_date"]: r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    histories = defaultdict(list)
    for r, _ in inputs():
        issue, decision = r["release_date"], r["decision"]
        f = book[issue]
        if (not f["cutoff"] < f["known"] <= decision < r["entry"] < r["exit_at"] or f["actual"] != r["x"][0]
                or f["latest_training_outcome"] and f["latest_training_outcome"] >= f["cutoff"]):
            raise ValueError(f"The inventory forecast is unavailable on {issue}")
        for m in MARKET_METHODS:
            expectation = f[SUPPLEMENT if m == MARKET_REFERENCE else m]
            design = replace_expectation(r, expectation)
            context = features(r["x"][2], r["volatility_usd"], r["x"][0] - expectation)
            current = r | dict(x=design, response_features=context)
            history = histories[m]
            settled = [q for q in history if q["exit_at"] < decision][-HISTORY:]
            history.append(current)
            if len(settled) < MIN_WINDOWS:
                continue
            point = volatility_forecast(settled, current)
            current["base_estimate"] = point
            calibration = [q for q in settled if "base_estimate" in q]
            correction, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            value = point + correction
            t = trade(r, m, value, 3, training_at=settled[-1]["exit_at"], calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if m == MARKET_REFERENCE:
                ref = reference[issue]
                if (any(ref[k] != t[k] for k in ("near_month", "far_month", "entry_at_utc", "exit_at_utc"))
                        or any(int(ref[k]) != t[k] for k in ("near_side", "gross_usd", "net_170_usd"))
                        or not math.isclose(value, float(ref["expected_long_move_usd"]), rel_tol=0., abs_tol=1e-8)):
                    raise ValueError(f"The weighted satellite reference changed on {issue}")
            yield t | dict(coverage_group=f["coverage_group"], inventory_estimate_kbbl=1000 * expectation,
                           realized_long_move_usd=r["long_gross"], net_340_usd=t["gross_usd"] - 340 * abs(t["near_side"]))


def report(forecasts, trades):
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"], r["method"]].append(r)
    phases = {}
    for phase in PHASES:
        panel = [r for r in forecasts if r["study_phase"] == phase]
        base = groups[phase, MARKET_REFERENCE]
        inventory, trading = {}, {}
        for scope in ("all", "no_terminal", "below_two", "two_or_more"):
            subset = [r for r in panel if scope == "all" or r["coverage_group"] == scope]
            errors = {m: np.array([1000 * abs(r[m]-r["actual"]) for r in subset]) for m in (*EXPERTS, *METHODS)}
            inventory[scope] = {m: dict(releases=len(loss), mae_kbbl=float(loss.mean()),
                paired_mae_delta_95_kbbl=block_bootstrap_mean(loss - errors[BASE])) for m, loss in errors.items() if len(loss)}
        for m in MARKET_METHODS:
            sample = groups[phase, m]
            if [r["release_date"] for r in sample] != [r["release_date"] for r in base]:
                raise ValueError(f"Trading calendars differ on {phase}/{m}")
            differences = [r["net_170_usd"] - q["net_170_usd"] for r, q in zip(sample, base, strict=True)]
            trading[m] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["net_340_usd"] for r in sample), net_delta_170_usd=sum(differences),
                paired_net_delta_per_window_95_usd=block_bootstrap_mean(differences),
                changed_sides=sum(r["near_side"] != q["near_side"] for r, q in zip(sample, base, strict=True)),
                spread_mae_usd=float(np.mean([abs(r["realized_long_move_usd"]-r["expected_long_move_usd"]) for r in sample])))
        phases[phase] = dict(inventory=inventory, trading=trading)
        print(phase, {m: round(v["mae_kbbl"], 2) for m, v in inventory["all"].items()},
              {m: (v["trades"], v["net_170_usd"], v["net_340_usd"]) for m, v in trading.items()})
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def forward(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    inputs = (DERIVED / f"{EXTENDED}_{issue}_registration.json", DERIVED / f"{LEADER}_{issue}_registration.json",
              original_path.with_name(f"{original_path.stem}_internal_calibration.json"),
              DERIVED / f"{BREADTH}_{issue}_registration.json")
    original = json.loads(original_path.read_bytes())
    records = [json.loads(p.read_bytes()) for p in inputs]
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    extended, leader, internal, breadth = records
    if (any(r[k] != original[k] for k in LINEAGE for r in records)
            or len({extended["measured_roofs"], leader["measured_roofs"], internal["measured_roofs"], breadth["measured_roofs"]}) != 1):
        raise ValueError("The forward inputs use different weeks or satellite measurements")
    for r in records:
        checked_hashes(r["sources_sha256"] if "sources_sha256" in r else r["source_sha256"])
    if not path.exists():
        if issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The mixture must be registered before the first release")
        effective = breadth["terminal_diagnostics"].get("effective_terminals", 0.)
        experts = {m: r["estimates_kbbl"][m] for m, r in zip(EXPERTS, (extended, extended, leader, internal), strict=True)}
        row = dict(cutoff=cutoff, effective_terminals=effective, estimates=np.array(list(experts.values())) / 1000)
        estimates, weights, past = predict(list(source_rows()), row)
        now = datetime.now(timezone.utc)
        if not past or not past[-1]["known"] < cutoff <= now < deadline:
            raise ValueError("The coverage mixture crossed the forecast deadline")
        sources = (original_path, *inputs, Path(__file__).resolve(),
                   *(DERIVED / f"{stem}.csv" for stem in (f"{EXTENDED}_inventory", f"{BREADTH}_forecasts", f"{OUTER}_forecasts")))
        freeze(path, {k: original[k] for k in LINEAGE} | dict(recorded_at_utc=now.isoformat(),
            effective_terminals=effective, coverage_group=coverage_group(effective), measured_roofs=extended["measured_roofs"],
            latest_training_outcome=past[-1]["known"].isoformat(), training_weeks=len(past),
            estimates_kbbl=experts | {m: x * 1000 for m, x in estimates.items()}, weights=weights,
            sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources}))
    frozen = json.loads(path.read_bytes())
    checked_hashes(frozen["sources_sha256"])
    if (any(frozen[k] != original[k] for k in LINEAGE) or not datetime.fromisoformat(frozen["latest_training_outcome"])
            < cutoff <= datetime.fromisoformat(frozen["recorded_at_utc"]) < deadline):
        raise ValueError("The registered coverage mix does not verify")
    outcome = score_release(original_path, original)
    result = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path), estimates_kbbl=frozen["estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_errors_kbbl={m:abs(actual-v) for m,v in frozen["estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(result, indent=2))


def main():
    forecasts = list(forecast_rows(list(source_rows())))
    with threadpool_limits(limits=1):
        trades = list(market(forecasts))
    summary = report(forecasts, trades)
    export(f"{STEM}_forecasts", forecasts)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat, help="Register or score an original EIA release")
    args = parser.parse_args()
    if args.release:
        forward(args.release.isoformat())
    else:
        main()
