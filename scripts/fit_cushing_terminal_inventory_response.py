"""Learn how Cushing's weekly inventory change is distributed across terminals."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_inputs, trade
from fit_cushing_cl_extension_segments import (
    archive as following_archive,
    calibrated_inventory,
    case as holding_case,
    exit_indices,
    predict as predict_holding,
)
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import estimate, learned_weights, prior_scale, time_covariance
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import EXTENDED, INVENTORY_LEADER, inputs
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL, SCENE_SIGMA_M, scene_design
from fit_cushing_terminal_balance import mapped_areas
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_terminal_inventory_response"
BASE = "uniform_by_area"
MODELS = (BASE, "terminal_slope", "terminal_slope_and_drift", "coverage_weighted_response")
MARKET_MODELS = (*MODELS, *(f"{m}_conditional_exit" for m in MODELS))
HOLDING_RULE = "curve_post_report_gate"
MIN_TERMINAL_WEEKS = 8


def eligible_roster(row, roster):
    if (not math.isclose(math.fsum(roster.values()), row["hub_area"], rel_tol=1e-10)
            or any(roster.get(q["terminal_id"], 0.) <= 0 for q in row["observed"])):
        raise ValueError(f"Satellite measurements and dated terminal roster differ on {row['week_ending']}")


def terminal_evidence(row, model):
    observed = model["observations"]
    tags = np.array([q["terminal_id"] for q in observed])
    design = scene_design(observed, {n:i for i,n in enumerate(sorted(model["scenes"]))})
    cov = model["local"] + SCENE_SIGMA_M**2 * (design @ design.T) + time_covariance([model])
    for site in sorted(set(tags)):
        mask = tags == site
        variance = cov[np.ix_(mask, mask)]
        sensitivity = model["sensitivity"][mask]
        residual = model["residual"][mask]
        residual = np.clip(residual, -3 * np.sqrt(variance.diagonal()), 3 * np.sqrt(variance.diagonal()))
        direction = cho_solve(cho_factor(variance), sensitivity)
        information = float(sensitivity @ direction)
        if not math.isfinite(information) or information <= 0:
            raise ValueError(f"Invalid terminal observation precision for {site}")
        yield site, dict(cutoff=row["cutoff"], known=row["known"], actual=row["actual"],
                         information=information, score=float(direction @ residual))


def parameters(samples, cutoff):
    if len(samples) < MIN_TERMINAL_WEEKS:
        return 0., 0., 0.
    actual, info, score = (np.array([r[k] for r in samples]) for k in ("actual", "information", "score"))
    age = np.array([(cutoff - r["cutoff"]).total_seconds() / 604800 for r in samples])
    weight = np.exp2(-age / 78) / (1 + info)
    data = score / np.sqrt(info)
    regressor = actual * np.sqrt(info)
    slope = 0.
    for _ in range(5):
        influence = weight / np.hypot(1., (data - regressor * slope) / 1.5)
        slope = float(np.clip((influence * regressor) @ data / (influence @ regressor**2 + 16.), -.60, .60))
    # A terminal can have a different throughput and a persistent local bias.
    # Both coefficients describe the residual after the uniform-area model.
    design = np.column_stack((regressor, np.sqrt(info)))
    ridge = np.diag((16., 24.))
    coeff = np.zeros(2)
    for _ in range(5):
        influence = weight / np.hypot(1., (data - design @ coeff) / 1.5)
        coeff = np.clip(np.linalg.solve(design.T @ (influence[:, None] * design) + ridge,
                                        design.T @ (influence * data)), (-.6, -.25), (.6, .25))
    return slope, float(coeff[0]), float(coeff[1])


def response(model, row, target, gains, drifts):
    if not model["observations"]:
        return model
    observations = model["observations"]
    gain = np.array([gains[q["terminal_id"]] for q in observations])
    drift = np.array([drifts[q["terminal_id"]] for q in observations])
    week = date.fromisoformat(row["week_ending"])
    post = week + timedelta(days=7)
    exposure = np.array([q.get("post_fraction", 0.) * q["area"] / row["hub_area"] for q in observations])
    original_sensitivity = model["sensitivity"]
    earlier = np.array([q["prior_component"] for q in observations])
    total_exposure = np.array([q["duration"] * q["area"] / row["hub_area"] for q in observations])
    revised_sensitivity = gain * original_sensitivity
    residual = model["residual"] - (gain - 1) * (earlier + original_sensitivity * target) - total_exposure * drift
    covariance = model["local"]
    if post in model["clocks"]:
        covariance = covariance + PRIOR_INVENTORY_SIGMA_MBBL**2 * (
            np.outer(gain * exposure, gain * exposure) - np.outer(exposure, exposure))
    clocks = {k:(clock, gain * area) for k,(clock,area) in model["clocks"].items()}
    return model | dict(sensitivity=revised_sensitivity, residual=residual, local=covariance, clocks=clocks)


def forecasts(row, roster, state, *, update=True):
    cutoff, base = row["cutoff"], row["public"]
    prediction = dict.fromkeys(MODELS, base)
    archive = state["evidence"]
    params = {}
    trained = None
    if row["observed"]:
        eligible_roster(row, roster)
        zero = dict.fromkeys(roster, 0.)
        for site in roster:
            sample = [r for r in archive[site] if r["known"] < cutoff][-78:]
            params[site] = parameters(sample, cutoff)
            if len(sample) >= MIN_TERMINAL_WEEKS:
                trained = max(trained or sample[-1]["known"], sample[-1]["known"])
        simple = {site:1. + p[0] for site,p in params.items()}
        full = {site:1. + p[1] for site,p in params.items()}
        drift = {site:p[2] for site,p in params.items()}
        covers = math.fsum(q["sensitivity"] for q in row["observed"])
        # When a large fraction of Cushing is observed, the total is less
        # sensitive to how the remaining volume is distributed across sites.
        blended = {site:1. + (full[site] - 1.) * (1. - min(covers, 1.)) for site in roster}
        half_drift = {site:d * (1. - min(covers, 1.)) for site,d in drift.items()}
        coefficients = dict(zip(MODELS[1:], ((simple, zero), (full, drift), (blended, half_drift)), strict=True))
        current = prepared(row, base, row["spanning"])
        actual = prepared(row, row["actual"], row["spanning"]) if update else None
        priors = [q for q in state["errors"] if q["known"] < cutoff]
        sigma = prior_scale(priors)
        for name in MODELS:
            measured = [q for q in state["models"][name] if q["known"] < cutoff]
            recent = [q["model"] for q in measured if cutoff - q["cutoff"] <= MAX_SCENE_AGE]
            forecast, realized = current, actual
            if name != BASE:
                gain, bias = coefficients[name]
                gain_scale = math.fsum(roster[k] * gain[k] for k in roster) / row["hub_area"]
                bias_loc = math.fsum(roster[k] * bias[k] for k in roster) / row["hub_area"]
                gains = {k:v / gain_scale for k,v in gain.items()}
                drifts = {k:v - bias_loc for k,v in bias.items()}
                forecast = response(current, row, base, gains, drifts)
                if update:
                    realized = response(actual, row, row["actual"], gains, drifts)
            fits = estimate(forecast, recent)
            weights = learned_weights(measured)
            increments = np.clip([score / (1 / sigma**2 + info) for _,info,score in fits], -1.25, 1.25)
            prediction[name] = base + float(weights @ increments)
            if update:
                errors = np.abs([base + shift - row["actual"] for shift,_,_ in fits])
                state["models"][name].append(dict(cutoff=cutoff, known=row["known"], model=realized, absolute_errors=errors))
        if update:
            for site, sample in terminal_evidence(row, actual):
                archive[site].append(sample)
    if update:
        state["errors"].append(dict(known=row["known"], error=row["actual"] - base))
    return prediction, params, trained


def start():
    return dict(evidence=defaultdict(list), models=defaultdict(list), errors=[])


def history(state, *, before=None):
    extended = indexed(f"{EXTENDED}_inventory")
    roster = mapped_areas()
    pass_ledger = indexed("cushing_overpass_covariance_forecasts")
    for row in inputs(before=before):
        issue, week, cutoff = (row[k] for k in ("release_date", "week_ending", "cutoff"))
        dated = roster[issue,week]
        if any(clock != cutoff for _,clock in dated.values()):
            raise ValueError(f"Terminal areas differ from the optical cutoff on {issue}")
        prediction, params, latest = forecasts(row, {k:v for k,(v,_) in dated.items()}, state)
        anchor = extended[issue]
        local = pass_ledger[issue]
        if (not math.isclose(prediction[BASE], row["reference"], abs_tol=1e-9, rel_tol=0)
                or any(datetime.fromisoformat(q[k]) != row[k] for q in (anchor, local) for k in ("cutoff", "known"))
                or any(float(q["actual"]) != row["actual"] or q["study_phase"] != row["study_phase"] for q in (anchor,local))):
            raise ValueError(f"The forecast clocks or geometry differ on {issue}")
        baseline = float(anchor[INVENTORY_LEADER]) + float(local["same_week_shared_pass"]) - prediction[BASE]
        fields = ("release_date", "week_ending", "cutoff", "known", "study_phase", "actual")
        yield {k:row[k] for k in fields} | prediction | dict(roofs=len(row["observed"]), support=row["area_fraction"],
            spanning=row["spanning"], terminal_params_json=json.dumps(params), latest_terminal_label_utc=latest,
            **{f"long_{m}":baseline + (p - prediction[BASE]) for m,p in prediction.items()})


def trading(forecasts_by_issue):
    original = {r["release_date"]:r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    references = {r["release_date"]:r for r in read("cushing_cl_extension_segments_trades") if r["method"] == HOLDING_RULE}
    following = {r["issue"]:r for r in calibrated_inventory(following_archive())}
    history_by_model = defaultdict(list)
    history_by_exit = {m:defaultdict(list) for m in MODELS}
    for row, _ in market_inputs():
        issue, decision = row["release_date"], row["decision"]
        f = forecasts_by_issue[issue]
        if not f["cutoff"] < f["known"] <= decision < row["entry"] or f["actual"] != row["x"][0]:
            raise ValueError(f"The inventory vintage is not available at the original decision on {issue}")
        for name in MODELS:
            expected = f[name]
            current = row | dict(x=replace_expectation(row, expected), response_features=features(
                row["x"][2], row["volatility_usd"], f["actual"] - expected))
            seen = history_by_model[name]
            closed = [q for q in seen if q["exit_at"] < decision][-HISTORY:]
            seen.append(current)
            if len(closed) < MIN_WINDOWS:
                continue
            point = volatility_forecast(closed, current)
            calibration = [q for q in closed if "base_estimate" in q]
            correction, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            current["base_estimate"] = point
            timing = dict(training_at=closed[-1]["exit_at"], calibration_at=calibration[-1]["exit_at"] if calibration else None)
            result = trade(row, name, point + correction, 3, **timing)
            old = original[issue]
            if name == BASE and (not math.isclose(point + correction, float(old["expected_long_move_usd"]), abs_tol=1e-8, rel_tol=0)
                    or any(result[k] != int(old[k]) for k in ("near_side", "gross_usd", "net_170_usd"))):
                raise ValueError(f"The original trade differs on {issue}")
            context = holding_case(row, point + correction, following.get(row["next_issue"]))
            holding = predict_holding(history_by_exit[name], context) if context else {}
            index = row["last_before_next_release"]
            early = index if index in (1, 2) else 3
            close = min(early + 1, 3)
            end = exit_indices(context["sigma"] if context else None, holding, result["near_side"], early, close)[HOLDING_RULE]
            change = trade(row, f"{name}_conditional_exit", point + correction, end, **timing)
            if name == BASE:
                paired = references[issue]
                if (any(change[k] != paired[k] for k in ("near_month", "far_month", "entry_at_utc", "exit_at_utc", "decision_at_utc"))
                        or any(change[k] != int(paired[k]) for k in ("near_side", "gross_usd", "net_170_usd"))):
                    raise ValueError(f"The latest execution reference differs on {issue}")
            latest_holding = max((q["latest"] for q in holding.values()), default=None)
            for candidate in (result, change):
                yield candidate | dict(inventory_forecast_kbbl=expected * 1000,
                                        net_340_usd=candidate["gross_usd"] - 340 * abs(candidate["near_side"]),
                                        latest_holding_label_utc=latest_holding if candidate is change else None,
                                        holding_decision_utc=context["decision"] if context and candidate is change else None)


def summarize(rows, trades):
    phases = {}
    for phase in PHASES:
        sample = [r for r in rows if r["study_phase"] == phase]
        groups = {m:[r for r in trades if (r["study_phase"],r["method"]) == (phase,m)] for m in MARKET_MODELS}
        stocks = {}
        for scope in ("all", "measured", "thin", "broad"):
            group = [r for r in sample if scope == "all" or r["roofs"] and (scope == "measured" or (r["support"] >= .15) == (scope == "broad"))]
            stocks[scope] = {}
            if not group:
                continue
            for kind in ("direct", "long_history"):
                prefix = "long_" if kind == "long_history" else ""
                error = {m:np.array([1000 * abs(r[prefix + m] - r["actual"]) for r in group]) for m in MODELS}
                stocks[scope][kind] = {m:dict(weeks=len(e), mae_kbbl=float(e.mean()),
                    paired_delta_95_kbbl=block_bootstrap_mean(e - error[BASE])) for m,e in error.items()}
        pnl = {}
        for m,g in groups.items():
            comparator = f"{BASE}_conditional_exit" if m.endswith("_conditional_exit") else BASE
            pairs = list(zip(g, groups[comparator], strict=True))
            if any(a["release_date"] != b["release_date"] for a,b in pairs):
                raise ValueError(f"The trading calendar changed for {phase}/{m}")
            pnl[m] = robustness(g) | daily_risk(g) | dict(net_340_usd=sum(r["net_340_usd"] for r in g),
                changed_sides=sum(a["near_side"] != b["near_side"] for a,b in pairs),
                changed_exits=sum(bool(a["near_side"] and a["exit_at_utc"] != b["exit_at_utc"]) for a,b in pairs),
                paired_delta_95_usd=block_bootstrap_mean([a["net_170_usd"] - b["net_170_usd"] for a,b in pairs]))
        phases[phase] = dict(inventory=stocks, trading=pnl)
        print(phase, {m:round(stocks["measured"]["long_history"][m]["mae_kbbl"], 2) for m in MODELS},
              {m:(v["net_170_usd"], v["net_340_usd"]) for m,v in pnl.items()})
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def register(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    recovered_path = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
    satellite_path = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    covariance_path = DERIVED / f"cushing_overpass_covariance_{issue}_registration.json"
    native = DERIVED / f"cushing_forward_native_optical_{issue}_summary.json"
    original, recovered, satellite, covariance, geometry = (json.loads(p.read_bytes()) for p in (
        original_path, recovered_path, satellite_path, covariance_path, native))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(r[k] != original[k] for r in (recovered, satellite, covariance) for k in LINEAGE):
        raise ValueError("The frozen issue clocks differ")
    if not path.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The terminal model must be registered before publication")
        for r in (satellite, covariance):
            checked_hashes(r["sources_sha256"])
        checked_hashes({f"data/derived/{name}": checksum for name,checksum in recovered["source_sha256"].items()})
        rosters = defaultdict(float)
        registry = "cushing_modern_lidar_optical_registry"
        preceding = (date.fromisoformat(original["week_ending"]) - timedelta(days=7)).isoformat()
        for row in read(registry):
            if (row["modern_input_status"] == "available" and row["latest_engineering_date"] < cutoff.date().isoformat()
                    and row["first_seen"] <= preceding):
                rosters[row["terminal_id"]] += float(row["circular_area_m2"])
        observed = [r | dict(area=r["volume_per_m"] * BARREL_M3 * 1e6, available_utc=r["source_available_utc"])
                    for r in original["measured_pairs"]] + recovered["recovered_roofs"]
        if (len(observed) != satellite["measured_roofs"] or len({q["candidate_id"] for q in observed}) != len(observed)
                or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observed)
                or satellite["optical_input_policy"] != "frozen_panel_without_eligible_after_target_observations"):
            raise ValueError("The forward roster differs from the optical baseline")
        area = geometry["dated_roof_area_m2"]
        state = start()
        with threadpool_limits(limits=1):
            for _ in history(state, before=cutoff):
                pass
            result, params, latest = forecasts(dict(week_ending=original["week_ending"], cutoff=cutoff, observed=observed,
                hub_area=area, area_fraction=math.fsum(q["area"] for q in observed) / area, spanning=False,
                public=original["public_ensemble_kbbl"] / 1000), rosters, state, update=False)
        estimates = {m:p * 1000 for m,p in result.items()}
        stamp = datetime.now(timezone.utc)
        if (not latest or not latest < cutoff <= stamp < deadline
                or not math.isclose(estimates[BASE], satellite["estimates_kbbl"][SUPPLEMENT], abs_tol=1e-8, rel_tol=0)):
            raise ValueError("The forecast or its reference did not reproduce before the deadline")
        anchor = covariance["long_history_with_additive_pass_delta_kbbl"]["same_week_shared_pass"]
        sources = (original_path, recovered_path, satellite_path, covariance_path, native, Path(__file__).resolve(),
                   *(DERIVED / f"{m}.csv" for m in (registry, f"{STEM}_forecasts", f"{OUTER}_forecasts", f"{EXTENDED}_inventory",
                       "holdout_optical_terminal_weeks", "sentinel2_space_time_coverage_terminal_weeks", "cushing_overpass_covariance_forecasts")),
                   *(ROOT / "scripts" / f"{m}.py" for m in ("fit_cushing_intraweek", "fit_cushing_shared_scene", "fit_cushing_overpass_covariance")))
        freeze(path, {k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), latest_training_label_utc=latest.isoformat(),
            measured_roofs=len(observed), estimates_kbbl=estimates, parameters=params,
            long_estimates_kbbl={m:anchor + p - estimates[BASE] for m,p in estimates.items()},
            sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources}))
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    if (any(record[k] != original[k] for k in LINEAGE) or set(record["estimates_kbbl"]) != set(MODELS)
            or not datetime.fromisoformat(record["latest_training_label_utc"]) < cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline):
        raise ValueError("The terminal registration or clock does not match")
    outcome = score_release(original_path, original)
    score = dict(status=outcome["status"], registration_sha256=digest(path), estimates_kbbl=record["estimates_kbbl"],
                 long_estimates_kbbl=record["long_estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        score.update(actual_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                     long_absolute_errors_kbbl={m:abs(p - actual) for m,p in record["long_estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(score,indent=2,allow_nan=False) + "\n").encode())
    print(score)


def main():
    with threadpool_limits(limits=1):
        predictions = list(history(start()))
        trades = list(trading({r["release_date"]:r for r in predictions}))
    report = summarize(predictions, trades)
    export(f"{STEM}_forecasts", predictions)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report,indent=2,allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    register(args.release.isoformat()) if args.release else main()
