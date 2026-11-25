"""Compare curve and inventory forecasts under matched fitting and risk assumptions."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_cl_entry_margin import inputs_with_reference
from audit_cushing_forward_wti_coverage import archives
from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write
from evaluate_cushing_forward_public_cl import ENSEMBLE_INTERACTION_METHOD, REGISTERED, registered_decision
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_robust_stack import indexed
from fit_cushing_cl_volatility import HURDLE
from fit_cushing_volatility_interactions import features as market_features
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_cl_satellite import history as market_history
from register_cushing_leading_models import LINEAGE, RULE, STEM as LEADING, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from score_cushing_forward_pair import score as score_spread

STEM = "cushing_cl_equal_model_risk"
INCUMBENT = "satellite"
MATCHED = "weighted_curve"
PUBLIC_ONLY = "weighted_public"
ORIGINAL_CURVE = "original_curve"
BLENDS = ("equal_public_and_satellite", "learned_public_and_satellite", "learned_curve_and_satellite")
STRATEGIES = (INCUMBENT, MATCHED, PUBLIC_ONLY, ORIGINAL_CURVE, *BLENDS)
COSTS = (170, 340)
FIELDS = ("release_date", "decision_at_utc", "entry_at", "exit_at", "near_month", "far_month")


def blend_weight(past, decision, anchor):
    if len(past) < MIN_WINDOWS:
        return .5
    design = np.array([(r[INCUMBENT] - r[anchor]) / r["sigma"] for r in past])
    target = np.array([(r["move"] - r[anchor]) / r["sigma"] for r in past])
    age = np.array([(decision - r["exit_at"]).total_seconds() / 604800 for r in past])
    decay = np.exp2(-age / 78)
    prior = .020 * decay.sum()
    weight = .5
    for _ in range(8):
        w = decay / np.hypot(target - weight * design, .75)
        weight = float(np.clip((w @ (design * target) + prior * .5) / (w @ (design ** 2) + prior), 0., 1.))
    return weight


def matched_replay():
    sources = {ORIGINAL_CURVE: ("cushing_cl_volatility_interactions_trades", "carry_volatility_context", "expected_long_gross_usd"),
               PUBLIC_ONLY: (f"{WEIGHTED}_trades", f"{PUBLIC}:volatility", "expected_long_move_usd")}
    references = {m: {r["release_date"]: (r, float(r[value])) for r in read(stem) if r["method"] == name}
                  for m, (stem, name, value) in sources.items()}
    history, curve_calibration = [], []
    for case, dated, satellite_row, satellite, settled, _ in inputs_with_reference():
        issue, now, sigma = (case[k] for k in ("release_date", "decision", "volatility_usd"))
        if not settled or sigma <= 0:
            raise ValueError(f"Missing settled risk history on {issue}")
        route = {m: refs[issue] for m, refs in references.items()}
        for row, _ in route.values():
            if (any(row[k] != case[k] for k in ("near_month", "far_month", "study_phase"))
                    or any(datetime.fromisoformat(row[k]) != case[field] for k, field in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))
                    or datetime.fromisoformat(row["latest_training_exit_utc"]) != settled[-1]["exit_at"]
                    or not math.isclose(float(row["volatility_usd"]), sigma)):
                raise ValueError(f"The controls do not share contracts or a training clock on {issue}")
        train = [r | dict(x=market_features(r)[-1]) for r in settled]
        point = curve_point(train, dated)
        q = case | dict(base_estimate=point, response_features=features(case["x"][2], sigma))
        past = [r for r in curve_calibration if r["exit_at"] < now][-HISTORY:]
        residual, _ = conditional_residual(past, q) if len(past) >= MIN_WINDOWS else (0., 0.)
        curve_calibration.append(q)
        estimates = {INCUMBENT: satellite, MATCHED: point + residual} | {m: value for m, (_, value) in route.items()}
        calibration = {INCUMBENT: satellite_row["latest_calibration_exit_utc"], MATCHED: past[-1]["exit_at"] if past else None,
                       PUBLIC_ONLY: route[PUBLIC_ONLY][0]["latest_calibration_exit_utc"], ORIGINAL_CURVE: None}
        past_blend = [r for r in history if r["exit_at"] < now][-HISTORY:]
        history.append(estimates | dict(sigma=sigma, move=case["long_gross"], exit_at=case["exit_at"]))
        weights = (.5, blend_weight(past_blend, now, PUBLIC_ONLY), blend_weight(past_blend, now, MATCHED))
        for name, w, anchor in zip(BLENDS, weights, (PUBLIC_ONLY, PUBLIC_ONLY, MATCHED), strict=True):
            estimates[name] = estimates[anchor] + w * (satellite - estimates[anchor])
            calibration[name] = past_blend[-1]["exit_at"] if name != BLENDS[0] and past_blend else None
        for name, estimate in estimates.items():
            stamp = calibration[name]
            if isinstance(stamp, str):
                stamp = datetime.fromisoformat(stamp) if stamp else None
            if stamp and not stamp < now:
                raise ValueError(f"The residual target was unsettled on {issue}/{name}")
            row = trade(case, name, estimate, 3, training_at=settled[-1]["exit_at"], calibration_at=stamp)
            if name in (INCUMBENT, *route):
                old = satellite_row if name == INCUMBENT else route[name][0]
                if any(row[k] != int(old[k]) for k in ("near_side", "gross_usd", "net_170_usd")):
                    raise ValueError(f"Replayed control is different on {issue}/{name}")
            yield row | dict(long_move_usd=case["long_gross"],
                             curve_base_usd=point if name == MATCHED else None,
                             satellite_weight=weights[BLENDS.index(name)] if name in BLENDS else None,
                             net_340_usd=row["gross_usd"] - 340 * abs(row["near_side"]))


def risk_budget(rows):
    return math.sqrt(math.fsum(float(r["volatility_usd"]) ** 2 for r in rows if r["near_side"]))


def curve_point(sample, current):
    return volatility_forecast(sample, current | dict(x=market_features(current)[-1]))


def common_risk(samples):
    # Fractional rescaling of entire observed paths is an attribution check.
    # It never increases size and does not modify any order or replay.
    requested = {m: risk_budget(group) for m, group in samples.items()}
    budget = min(requested.values())
    if budget <= 0.:
        return {}
    comparisons = {}
    for m, group in samples.items():
        scale = budget / requested[m]
        comparisons[m] = dict(diagnostic_position_multiplier=scale, original_risk_usd=requested[m], common_risk_usd=budget,
            normalized_daily_drawdown_usd=scale * daily_risk(group)["daily_drawdown_usd"],
            **{f"normalized_net_{c}_usd": scale * sum(r[f"net_{c}_usd"] for r in group) for c in COSTS})
    return comparisons


def paired_evidence(later, earlier):
    pairs = list(zip(later, earlier, strict=True))
    if any(a["release_date"] != b["release_date"] for a, b in pairs):
        raise ValueError("Different release windows in the paired sample")
    gains_by_month = defaultdict(int)
    gains_by_side = defaultdict(int)
    differences, after = [], []
    for a, b in pairs:
        day = a["release_date"]
        delta = a["net_170_usd"] - b["net_170_usd"]
        differences.append(delta)
        gains_by_month[day[:7]] += delta
        payload = dict(release_date=day, model=a["method"], control=b["method"], model_side=a["near_side"],
                       control_side=b["near_side"], volatility_usd=a["volatility_usd"], delta_usd=delta)
        after.append(payload)
        change = "same_side" if a["near_side"] == b["near_side"] else "model_only" if not b["near_side"] else (
            "control_only" if not a["near_side"] else "opposite_direction")
        gains_by_side[change] += delta
    total = sum(differences)
    months = sorted(gains_by_month, key=gains_by_month.get, reverse=True)
    leave_out = [(month, total - gains_by_month[month]) for month in months]
    winners = sorted((d for d in differences if d > 0), reverse=True)
    evidence = dict(windows=len(pairs), increment_net_170_usd=total,
                    increment_per_window_95_usd=block_bootstrap_mean(differences),
                    increment_net_340_usd=sum(a["net_340_usd"] - b["net_340_usd"] for a, b in pairs),
                    changed_sides=sum(r["model_side"] != r["control_side"] for r in after),
                    incremental_pnl_by_side=dict(gains_by_side),
                    best_increment_month=months[0], best_month_increment_usd=gains_by_month[months[0]],
                    increment_excluding_best_month_usd=leave_out[0][1],
                    increment_excluding_april_2026_usd=total - gains_by_month.get("2026-04", 0),
                    increment_excluding_three_largest_window_gains_usd=total - sum(winners[:3]),
                    incremental_net_without_best_month_and_at_340_usd=sum(a["net_340_usd"] - b["net_340_usd"]
                        for a, b in pairs if a["release_date"][:7] != months[0]))
    return evidence, after


def curve_forward_prediction(public):
    decision = datetime.fromisoformat(public["decision_at_utc"])
    carry, change, sigma = (float(public[k]) for k in ("carry", "carry_change", "volatility_usd"))
    # The zeroes occupy slots of the shared feature representation. The curve
    # feature slice uses only carry, its change and volatility.
    current = dict(decision=decision, x=(0., 0., carry, change, 0., 0., 0.), volatility_usd=sigma)
    settled = [r for r in market_history() if r["exit_at"] < decision][-HISTORY:]
    if len(settled) < MIN_WINDOWS:
        raise ValueError("The curve control does not have enough settled history")
    stored = {r["release_date"]: r for r in read(f"{STEM}_trades") if r["method"] == MATCHED}
    calibration = []
    for r in settled:
        f = stored.get(r["release_date"])
        if not f or datetime.fromisoformat(f["exit_at_utc"]) != r["exit_at"] or f["near_month"] != r["near_month"] or f["far_month"] != r["far_month"]:
            raise ValueError(f"Missing or different dated curve training window {r['release_date']}")
        calibration.append(r | dict(base_estimate=float(f["curve_base_usd"]),
                                    response_features=features(r["x"][2], r["volatility_usd"])))
    fit = [r | dict(x=market_features(r)[-1]) for r in settled]
    with threadpool_limits(limits=1):
        point = curve_point(fit, current)
        correction, _ = conditional_residual(calibration, current | dict(response_features=features(carry, sigma)))
    return point + correction, settled[-1]["exit_at"]


def forward(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    registration = prefix.with_name(f"{prefix.name}_registration.json")
    receipt = prefix.with_name(f"{prefix.name}_decision.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    leader_path = DERIVED / f"{LEADING}_{issue}_registration.json"
    original, leader = (json.loads(p.read_bytes()) for p in (original_path, leader_path))
    cutoff, deadline = map(datetime.fromisoformat, (leader[k] for k in LINEAGE[-2:]))
    if any(original[k] != leader[k] for k in LINEAGE) or leader["market_rule"] != RULE:
        raise ValueError("The comparison does not share the leader's report and trading rule")
    checked_hashes(leader["sources_sha256"])
    if not registration.exists():
        stamp = datetime.now(timezone.utc)
        if not cutoff <= stamp < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The curve control must be registered before publication")
        sources = (leader_path, original_path, Path(__file__).resolve(), DERIVED / f"{STEM}_trades.csv",
                   *(ROOT / "scripts" / f"{name}.py" for name in (
                       "fit_cushing_volatility_neighbors", "fit_cushing_volatility_interactions",
                       "fit_cushing_cl_conditional_response", "register_cushing_cl_satellite")))
        freeze(registration, {k: leader[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), rule=RULE,
            methods=[INCUMBENT, MATCHED], sources_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources}))
    reg = json.loads(registration.read_bytes())
    checked_hashes(reg["sources_sha256"])
    if (any(reg[k] != leader[k] for k in LINEAGE) or reg["rule"] != RULE or reg["methods"] != [INCUMBENT, MATCHED]
            or not cutoff <= datetime.fromisoformat(reg["recorded_at_utc"]) < deadline):
        raise ValueError("The saved curve control failed its prepublication checks")
    outcome = score_release(original_path, original)
    score = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(registration))
    if outcome["status"] == "scored_first_release":
        now = datetime.now(timezone.utc)
        score["status"] = "awaiting_registered_calendar"
        ledger = json.loads((DERIVED / "cushing_forward_public_cl_summary.json").read_bytes())["release_rows"]
        initial = next((r for r in ledger if r["release_date"] == issue and r["method"] == ENSEMBLE_INTERACTION_METHOD), None)
        calendar_path = REGISTERED / f"{issue}_{ENSEMBLE_INTERACTION_METHOD}.json"
        if initial and initial["eligible_nonoverlap"] and calendar_path.exists():
            public = registered_decision(initial, register=False, now=now)
            decision, entry = map(datetime.fromisoformat, (public[k] for k in ("decision_at_utc", "entry_at")))
            evidence = dict(registration_sha256=digest(registration), calendar_receipt_sha256=digest(calendar_path),
                            first_release_sha256=outcome["first_release_sha256"])
            if public["actual_million_bbl"] != outcome["actual_eia_change_kbbl"] / 1000:
                raise ValueError("The registered calendar does not match the first EIA release")
            score["status"] = "awaiting_curve_decision"
            if not receipt.exists() and decision <= now < entry:
                estimate, label = curve_forward_prediction(public)
                stamp = datetime.now(timezone.utc)
                if not label < decision <= stamp < entry:
                    raise ValueError("The curve decision missed its execution deadline")
                side = int(estimate > HURDLE) - int(estimate < -HURDLE)
                freeze(receipt, {k: public[k] for k in FIELDS} | evidence | dict(recorded_at_utc=stamp.isoformat(),
                       latest_training_exit_at=label.isoformat(), near_side=side, expected_long_move_usd=estimate))
                now = stamp
            if receipt.exists():
                curve = json.loads(receipt.read_bytes())
                known, stamp = map(datetime.fromisoformat, (curve[k] for k in ("latest_training_exit_at", "recorded_at_utc")))
                est = curve["expected_long_move_usd"]
                if (any(curve[k] != public[k] for k in FIELDS) or any(curve[k] != v for k, v in evidence.items())
                        or not (known < decision <= stamp < entry and stamp <= now and math.isfinite(est))
                        or curve["near_side"] != int(est > HURDLE) - int(est < -HURDLE)):
                    raise ValueError("The recorded curve trade decision does not verify")
                leader_decision = DERIVED / f"{LEADING}_{issue}_decision.json"
                score.update(status="awaiting_satellite_decision", curve_decision_sha256=digest(receipt))
                if leader_decision.exists():
                    satellite = json.loads(leader_decision.read_bytes())
                    sat_at = datetime.fromisoformat(satellite["recorded_at_utc"])
                    if (any(satellite[k] != public[k] for k in FIELDS) or satellite["rule"] != RULE
                            or satellite["public_receipt_sha256"] != evidence["calendar_receipt_sha256"]
                            or satellite["first_release_sha256"] != evidence["first_release_sha256"]
                            or satellite["forecast_sha256"] != digest(leader_path) or not decision <= sat_at < entry or sat_at > now):
                        raise ValueError("The two registered decisions use different releases or entry clocks")
                    marks = archives(as_of=now)[1]
                    calendars = {issue: [r for r in read("cushing_forward_wti_coverage_bars")
                                        if r["release_date"] == issue and r["session_index"] != "1"]}
                    comparison, = score_spread({issue: {INCUMBENT: satellite, MATCHED: curve}}, calendars, marks, now,
                                               methods=(INCUMBENT, MATCHED))
                    score.update(comparison=comparison, status="window_scored" if comparison["market_window_complete"] else "awaiting_market_window")
            elif now >= entry:
                score["status"] = "missed_curve_decision_deadline"
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(score, indent=2, allow_nan=False) + "\n").encode())
    print(score)


def main():
    with threadpool_limits(limits=1):
        rows = list(matched_replay())
    report = dict(status="exploratory_on_reviewed_history", phases={})
    comparisons = []
    for phase in PHASES:
        g = {m: [r for r in rows if r["study_phase"] == phase and r["method"] == m] for m in STRATEGIES}
        performance = {m: robustness(group) | daily_risk(group) | dict(
            net_340_usd=sum(r["net_340_usd"] for r in group), exposure_risk_usd=risk_budget(group),
            mae_usd=float(np.mean([abs(r["expected_long_move_usd"] - r["long_move_usd"]) for r in group])))
            for m, group in g.items()}
        pairs = {}
        for m in STRATEGIES[1:]:
            evidence, differences = paired_evidence(g[INCUMBENT], g[m])
            pairs[m] = evidence | dict(matched_risk=common_risk({k: g[k] for k in (INCUMBENT, m)}))
            comparisons.extend(r | dict(study_phase=phase) for r in differences)
        report["phases"][phase] = dict(trades=performance, satellite_vs=pairs)
        print(phase, {m: (p["trades"], p["net_170_usd"], p["net_340_usd"], round(p["mae_usd"], 1)) for m, p in performance.items()})
    export(f"{STEM}_trades", rows)
    export(f"{STEM}_paired", comparisons)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    if args.release:
        forward(args.release.isoformat())
    else:
        main()
