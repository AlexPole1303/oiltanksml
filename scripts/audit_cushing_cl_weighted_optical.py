"""Separate current imagery from the learned inventory history in the WTI leader."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from threadpoolctl import threadpool_limits

from audit_cushing_forward_wti_coverage import archives
from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write
from evaluate_cushing_forward_public_cl import ENSEMBLE_INTERACTION_METHOD, REGISTERED, registered_decision
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_cl_satellite import history as market_history
from register_cushing_leading_models import LINEAGE, RULE, STEM as FORWARD_LEADER, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_inventory
from score_cushing_forward_pair import score as score_spread

STEM = "cushing_cl_weighted_optical"
METHODS = {
    "public": (PUBLIC, PUBLIC),
    "current_optical": (PUBLIC, SUPPLEMENT),
    "historical_optical": (SUPPLEMENT, PUBLIC),
    "full_optical": (SUPPLEMENT, SUPPLEMENT),
}
FIELDS = ("release_date", "decision_at_utc", "entry_at", "exit_at", "near_month", "far_month")


def estimate(sample, past, row, design, context):
    point = volatility_forecast(sample, row | dict(x=design))
    response = row | dict(base_estimate=point, response_features=context)
    correction, _ = conditional_residual(past, response) if len(past) >= MIN_WINDOWS else (0., 0.)
    return response, point + correction


def replay():
    forecasts = indexed(f"{OUTER}_forecasts")
    datasets = (PUBLIC, SUPPLEMENT)
    keys = {f"{m}:volatility": m for m in datasets}
    references = {(q["release_date"], keys[q["method"]]): q for q in read(f"{WEIGHTED}_trades") if q["method"] in keys}
    history, past_points = [], defaultdict(list)
    for row, _ in inputs():
        issue, decision = row["release_date"], row["decision"]
        f = forecasts[issue]
        cut, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc", "known_utc"))
        actual = float(f["actual"])
        label = f[f"{SUPPLEMENT}_latest_label_utc"]
        if (not cut < known <= decision < row["entry"] < row["exit_at"] or actual != row["x"][0]
                or label and datetime.fromisoformat(label) >= cut):
            raise ValueError(f"Invalid input clock on {issue}")
        expectation = {m: float(f[m]) for m in datasets}
        design = {m: replace_expectation(row, expectation[m]) for m in datasets}
        context = {m: features(row["x"][2], row["volatility_usd"], actual - expectation[m]) for m in datasets}
        train = [r for r in history if r["exit_at"] < decision][-HISTORY:]
        history.append(row | dict(design=design))
        if (issue, PUBLIC) not in references:
            continue
        if len(train) < MIN_WINDOWS:
            raise ValueError(f"Insufficient trading history on {issue}")
        samples = {m: [r | dict(x=r["design"][m]) for r in train] for m in datasets}
        past = {m: [r for r in past_points[m] if r["exit_at"] < decision][-HISTORY:] for m in datasets}
        for method, (fit, current) in METHODS.items():
            response, value = estimate(samples[fit], past[fit], row, design[current], context[current])
            point = response["base_estimate"]
            stamp = past[fit][-1]["exit_at"] if past[fit] else None
            t = trade(row, method, value, 3, training_at=train[-1]["exit_at"], calibration_at=stamp)
            if fit == current:
                a = references[issue, current]
                if (any(a[k] != t[k] for k in ("near_month", "far_month", "study_phase"))
                        or any(datetime.fromisoformat(a[k]) != datetime.fromisoformat(t[k]) for k in (
                            "decision_at_utc", "entry_at_utc", "exit_at_utc", "latest_training_exit_utc"))
                        or a["latest_calibration_exit_utc"] != (stamp.isoformat() if stamp else "")
                        or any(int(a[k]) != t[k] for k in ("near_side", "gross_usd", "net_170_usd"))
                        or not math.isclose(point, float(a["base_estimate_usd"]), abs_tol=1e-8, rel_tol=0)
                        or not math.isclose(value, float(a["expected_long_move_usd"]), abs_tol=1e-8, rel_tol=0)):
                    raise ValueError(f"Archived strategy changed on {issue}/{method}")
                past_points[fit].append(response)
            yield t | dict(training_expectation=fit, current_expectation=current, current_eia_kbbl=1000 * actual,
                current_image_revision_kbbl=1000 * (expectation[SUPPLEMENT] - expectation[PUBLIC]),
                observed_roofs=sum(int(f[k]) for k in ("baseline_roofs", "outer_roofs", "supplement_roofs")),
                expected_base_usd=point, realized_long_move_usd=row["long_gross"],
                net_340_usd=t["gross_usd"] - 340 * abs(t["near_side"]))


def paired(group):
    by_date = defaultdict(dict)
    for r in group:
        by_date[r["release_date"]][r["method"]] = r
    for issue, arms in sorted(by_date.items()):
        public, current, historical, full = (arms[k] for k in METHODS)
        revision = full["current_image_revision_kbbl"]
        a, b, c, d = (r["expected_long_move_usd"] for r in (public, current, historical, full))
        if revision == 0 and (not math.isclose(a,b,abs_tol=1e-10) or not math.isclose(c,d,abs_tol=1e-10)):
            raise ValueError(f"Same-week forecasts must match when the optical revision is zero on {issue}")
        yield dict(release_date=issue, study_phase=full["study_phase"], market_decision_at=full["decision_at_utc"],
            current_image_revision_kbbl=revision, observed_roofs=full["observed_roofs"],
            current_image_forecast_contribution_usd=.5*((b-a)+(d-c)),
            historical_image_forecast_contribution_usd=.5*((c-a)+(d-b)),
            **{f"{k}_{field}": arms[k][field] for k in METHODS for field in (
                "near_side", "net_170_usd", "net_340_usd", "expected_long_move_usd")})


def variation(rows, base, candidate):
    changes = [r for r in rows if r[f"{base}_near_side"] != r[f"{candidate}_near_side"]]
    return dict(windows=len(rows), changes=len(changes), change_dates=[r["release_date"] for r in changes],
                same_week_image_on_changed_dates=sum(r["current_image_revision_kbbl"]!=0 for r in changes),
                **{f"increment_net_{cost}_usd":sum(r[f"{candidate}_net_{cost}_usd"]-r[f"{base}_net_{cost}_usd"] for r in rows)
                   for cost in (170,340)},
                net_increment_per_window_95_usd=block_bootstrap_mean(
                    [r[f"{candidate}_net_170_usd"]-r[f"{base}_net_170_usd"] for r in rows]))


def forward_estimates(public, expected, *, recent=None, stored=None):
    decision = datetime.fromisoformat(public["decision_at_utc"])
    current = dict(decision=decision, volatility_usd=public["volatility_usd"],
                   x=(public["actual_million_bbl"], 0., public["carry"], public["carry_change"], 0., 0., 0.))
    past = [r for r in (market_history() if recent is None else recent) if r["exit_at"] < decision]
    settled = past[-HISTORY:]
    if len(settled) < MIN_WINDOWS:
        raise ValueError("Insufficient settled trading windows")
    ref = {(r["release_date"], r["method"]): float(r["base_estimate_usd"]) for r in
           (read(f"{WEIGHTED}_trades") if stored is None else stored)}
    histories, samples = {}, {}
    for m in (PUBLIC, SUPPLEMENT):
        histories[m] = [r | dict(base_estimate=ref[r["release_date"], f"{m}:volatility"], response_features=features(
            r["x"][2], r["volatility_usd"], r["x"][0]-r[m])) for r in past if (r["release_date"], f"{m}:volatility") in ref][-HISTORY:]
        samples[m] = [r | dict(x=replace_expectation(r, r[m])) for r in settled]
    values = {}
    with threadpool_limits(limits=1):
        for m, (training, input_model) in METHODS.items():
            expectation = expected[input_model] / 1000
            design = replace_expectation(current, expectation)
            context = features(current["x"][2], current["volatility_usd"], current["x"][0] - expectation)
            _, values[m] = estimate(samples[training], histories[training], current, design, context)
    return values, settled[-1]["exit_at"]


def forward(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    receipt = prefix.with_name(f"{prefix.name}_decision.json")
    leader = DERIVED / f"{FORWARD_LEADER}_{issue}_registration.json"
    original = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    forecast = json.loads(leader.read_bytes())
    original_record = json.loads(original.read_bytes())
    checked_hashes(forecast["sources_sha256"])
    cutoff, deadline = (datetime.fromisoformat(forecast[k]) for k in LINEAGE[-2:])
    now = datetime.now(timezone.utc)
    method_parameters = {k: list(v) for k,v in METHODS.items()}
    expected = {m: forecast["estimates_kbbl"][m] for m in (PUBLIC, SUPPLEMENT)}
    if (any(forecast[k] != original_record[k] for k in LINEAGE) or forecast["market_rule"] != RULE
            or forecast["market_method"] != f"{SUPPLEMENT}:volatility"):
        raise ValueError("The parent registration uses a different issue or strategy")
    if not path.exists():
        if not cutoff <= now < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The four-way attribution must be registered before EIA publishes")
        sources = (leader, original, DERIVED / f"{WEIGHTED}_trades.csv", Path(__file__).resolve())
        freeze(path, {k: forecast[k] for k in LINEAGE} | dict(recorded_at_utc=now.isoformat(),
            estimates_kbbl=expected, methods=method_parameters, rule=RULE,
            sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    if (any(record[k] != forecast[k] for k in LINEAGE) or record["rule"] != RULE
            or record["methods"] != method_parameters or record["estimates_kbbl"] != expected
            or not cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline):
        raise ValueError("The forward attribution registration has changed")
    outcome = score_inventory(original, original_record)
    result = dict(release_date=issue, registration_sha256=digest(path), status=outcome["status"])
    if outcome["status"] == "scored_first_release":
        result["status"] = "awaiting_registered_market_clock"
        ledger = json.loads((DERIVED / "cushing_forward_public_cl_summary.json").read_bytes())["release_rows"]
        public_path = REGISTERED / f"{issue}_{ENSEMBLE_INTERACTION_METHOD}.json"
        row = next((r for r in ledger if r["release_date"] == issue and r["method"] == ENSEMBLE_INTERACTION_METHOD), None)
        if row and public_path.exists() and row["eligible_nonoverlap"]:
            public = registered_decision(row, register=False, now=now)
            decision, enter = (datetime.fromisoformat(public[k]) for k in ("decision_at_utc", "entry_at"))
            hashes = dict(registration_sha256=digest(path), first_release_sha256=outcome["first_release_sha256"],
                          calendar_receipt_sha256=digest(public_path))
            if public["actual_million_bbl"] != outcome["actual_eia_change_kbbl"] / 1000:
                raise ValueError("The calendar used a different first EIA release")
            if not receipt.exists() and decision <= now < enter:
                values, train_at = forward_estimates(public, record["estimates_kbbl"])
                now = datetime.now(timezone.utc)
                if now >= enter:
                    raise ValueError("The attribution decision finished after entry")
                freeze(receipt, {k: public[k] for k in FIELDS} | hashes | dict(
                    recorded_at_utc=now.isoformat(), latest_training_exit_utc=train_at.isoformat(), estimated_usd=values))
            if receipt.exists():
                saved = json.loads(receipt.read_bytes())
                latest_training, recorded = (datetime.fromisoformat(saved[k]) for k in (
                    "latest_training_exit_utc", "recorded_at_utc"))
                if (any(saved[k] != v for k,v in hashes.items()) or any(saved[k] != public[k] for k in FIELDS)
                        or not latest_training < decision <= recorded < enter or recorded > now
                        or set(saved["estimated_usd"]) != set(METHODS)
                        or not all(math.isfinite(v) for v in saved["estimated_usd"].values())):
                    raise ValueError("The saved attribution decision does not verify")
                models = {m: {k:saved[k] for k in FIELDS} | dict(recorded_at_utc=saved["recorded_at_utc"],
                          near_side=int(v>HURDLE)-int(v< -HURDLE)) for m,v in saved["estimated_usd"].items()}
                marks = archives(as_of=now)[1]
                calendars = {issue:[r for r in read("cushing_forward_wti_coverage_bars")
                                    if r["release_date"] == issue and r["session_index"] != "1"]}
                comparisons = {m:next(score_spread({issue:{m:models[m], "public":models["public"]}},
                        calendars, marks, now, methods=(m,"public"))) for m in METHODS if m!="public"}
                result.update(decision_sha256=digest(receipt), comparisons=comparisons,
                    status="window_scored" if comparisons["full_optical"]["market_window_complete"] else "awaiting_market_window")
            elif now >= enter:
                result["status"] = "missed_decision_deadline"
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(result, indent=2))


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
    differences = list(paired(rows))
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        p = [r for r in differences if r["study_phase"] == phase]
        report["phases"][phase] = dict(
            windows=len(p), nonzero_image_updates=sum(r["current_image_revision_kbbl"] != 0 for r in p),
            methods={},
            full_vs_public=variation(p, "public", "full_optical"),
            current_image_on_public_fit=variation(p, "public", "current_optical"),
            current_image_on_optical_fit=variation(p, "historical_optical", "full_optical"),
            historical_image_at_public_input=variation(p, "public", "historical_optical"),
        )
        for m in METHODS:
            group = [r for r in rows if r["study_phase"] == phase and r["method"] == m]
            report["phases"][phase]["methods"][m] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["net_340_usd"] for r in group))
        armed = report["phases"][phase]["methods"]
        print(phase,{m:(v["trades"],v["net_170_usd"],v["net_340_usd"]) for m,v in armed.items()})
    export(f"{STEM}_trades", rows)
    export(f"{STEM}_decomposition", differences)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat, help="Register or score a new release")
    args = parser.parse_args()
    if args.release:
        forward(args.release.isoformat())
    else:
        main()
