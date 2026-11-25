"""Learn a bounded height response near the rim and test changes on aerial controls."""

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from audit_cushing_weekly_change_controls import aerial
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from calibrate_cushing_height_multisource import learn as learn_midpoint, predict as predict_midpoint
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, VOLUMES
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT, run_forecast
from fit_cushing_overpass_covariance import EXTENDED, INVENTORY_LEADER, inputs
from fit_cushing_robust_stack import indexed
from replay_cushing_weighted_inventory_market import replay_weighted_market
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from replay_cushing_dated_height_calibration import MIDPOINT, calibrated_change, indexed_by_roof, pair_inputs, training_rows

STEM = "cushing_height_response_curve"
MODELS = (SUPPLEMENT, "bounded_height_response", "response_supported_by_labels")
CURRENT_ONLY = "height_response_current_only"
MARKET_MODELS = (*MODELS, CURRENT_ONLY)
KNOTS = (.25, .50, .75)


def basis(fraction):
    # Piecewise-linear changes vanish at the physical limits and retain
    # the ordering of heights under the coefficient bounds in the fit.
    return np.maximum(0., 1. - 4. * np.abs(np.asarray(fraction)[..., None] - KNOTS))


def learn(rows):
    if len(rows) < 20:
        return np.zeros(len(KNOTS)), None
    count = Counter(r["terminal_id"] for r in rows)
    x, y, scale = (np.array([r[key] for r in rows]) for key in ("relative_gap", "reference_gap_m", "shell_m"))
    residual = y - scale * x
    matrix = basis(x) * scale[:, None]
    weight = np.array([1. / count[r["terminal_id"]] for r in rows])
    weight /= weight.sum()

    def objective(beta):
        miss = matrix @ beta - residual
        smooth = np.hypot(miss, .75)
        return float(weight @ smooth + 6. * (beta @ beta)), matrix.T @ (weight * miss / smooth) + 12. * beta

    fit = minimize(objective, np.zeros(len(KNOTS)), jac=True, method="L-BFGS-B", bounds=[(-.08,.08)] * len(KNOTS))
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"The bounded optical response failed: {fit.message}")
    return fit.x, tuple(np.quantile(x, (.10,.90)))


def height(raw, shell, model, *, supported=False):
    beta, limits = model
    fraction = raw / shell
    weight = 1.
    if supported and limits is not None:
        low, high = limits
        weight = max(0., 1. - max(low - fraction, fraction - high, 0.) / .35)
    return float(np.clip(raw + shell * (basis(fraction) @ beta) * weight, 0., shell))


def corrections(issue, observations, red_ids, fits, controls):
    by_site = {site:learn([r for r in controls if r["terminal_id"] != site]) for site in {q["terminal_id"] for q in observations}}
    arms = {name:[] for name in MODELS}
    matched = 0
    for q in observations:
        key = (issue, q["candidate_id"], q["scene_before"], q["scene_after"])
        volume = q["pair_signal"]
        factor = q["volume_per_m"]
        samples = [s for s in fits.get(key, ()) if q["candidate_id"] in red_ids and s[2] > 0
                   and 0 <= min(s[:2]) <= max(s[:2]) <= s[2] and math.isclose((s[0] - s[1]) * factor, volume, abs_tol=1e-10)]
        model = by_site[q["terminal_id"]]
        use = len(samples) == 1 and model[1] is not None
        matched += bool(use)
        for name in MODELS:
            value = volume
            if name != SUPPLEMENT and use:
                first, last, shell, _ = samples[0]
                corrected = [height(v, shell, model, supported=name == MODELS[2]) for v in (first,last)]
                value = (corrected[0] - corrected[1]) * factor
            arms[name].append(q | dict(pair_signal=value))
    return arms, matched


def independent_controls(controls):
    out = []
    by_site = {}
    for site in {q["terminal_id"] for q in controls}:
        sample = [q for q in controls if q["terminal_id"] != site]
        by_site[site] = learn(sample), learn_midpoint(sample)
    for q in controls:
        model, comparison = by_site[q["terminal_id"]]
        results = {m:height(q["raw_gap_m"], q["shell_m"], model, supported=m == MODELS[2]) for m in MODELS[1:]}
        out.append(q | dict(test="lidar_height_other_terminals", actual_m=q["reference_gap_m"],
                            **{SUPPLEMENT:q["raw_gap_m"], MIDPOINT:predict_midpoint(q, comparison)["midpoint_calibration"]}, **results))
    lookup = {(q["candidate_id"], q["flight_date"], q["satellite_version"], q["satellite_captured_at_utc"], q["satellite_sensor"]):q
              for q in read("aerialok_native_optical_pairs")}
    geometry = indexed_by_roof("cushing_modern_lidar_optical_registry")
    for q in aerial():
        first,last = [lookup[q["candidate_id"],q[f"{e}_flight"],q[f"{e}_satellite_version"],q[f"{e}_satellite_capture_utc"],
            q[f"{e}_satellite_sensor"]] for e in ("first","last")]
        asset = geometry[q["candidate_id"]]
        shell = float(asset["prior_rim_z_p95_native"]) - float(asset["prior_ground_z_p50_native"])
        if q["terminal_id"] not in by_site:
            by_site[q["terminal_id"]] = learn(controls), learn_midpoint(controls)
        model, comparison = by_site[q["terminal_id"]]
        endpoints = (float(first["satellite_gap_m"]), float(last["satellite_gap_m"]), shell, float(asset["diameter_m"]))
        results = {}
        for name in MODELS[1:]:
            a,b = (height(float(r["satellite_gap_m"]), shell, model, supported=name == MODELS[2]) for r in (first,last))
            results[name] = a - b
        out.append(dict(test="aerial_change_other_terminals", candidate_id=q["candidate_id"], terminal_id=q["terminal_id"],
                        actual_m=q["target"], **{SUPPLEMENT:q["physical"], MIDPOINT:calibrated_change(*endpoints, comparison, MIDPOINT)},
                        **results))
    return out


def state():
    return dict(models={m:[] for m in MODELS}, errors=[])


def history(controls, book, *, before=None):
    fits = pair_inputs()
    # The B08 and extra radar/optical recovery paths carry their own gap
    # geometry. Only exact matches to the original B04 estimators are eligible.
    red = defaultdict(set)
    for stem in VOLUMES:
        for q in read(stem):
            red[q["release_date"]].add(q["candidate_id"])
    for stem in ("cushing_outer_capture_pairs_fits", "cushing_outer_capture_pairs_supplement_fits", "sentinel2_wider_registration_gap_pairs"):
        for q in read(stem):
            if q["registered_both_shadow_diagnostics_pass"] == "True":
                red[q["release_date"]].add(q["candidate_id"])
    dated = indexed(f"{EXTENDED}_inventory")
    pass_book = indexed("cushing_overpass_covariance_forecasts")
    for row in inputs(before=before):
        issue, cutoff = row["release_date"], row["cutoff"]
        labels = [q for q in controls if q["ready"] < cutoff]
        arms, n = corrections(issue, row["observed"], red[issue], fits, labels)
        settled = [q for q in book["errors"] if q["known"] < cutoff]
        prediction = {m:run_forecast(book["models"][m], settled, row | dict(observed=part), row["spanning"])[0]
                      for m,part in arms.items()}
        book["errors"].append(dict(known=row["known"], error=row["actual"] - row["public"]))
        baseline, covariance = dated[issue], pass_book[issue]
        if (not math.isclose(prediction[SUPPLEMENT], row["reference"], abs_tol=1e-9, rel_tol=0)
                or any(datetime.fromisoformat(q[k]) != row[k] for q in (baseline,covariance) for k in ("cutoff", "known"))
                or any(float(q["actual"]) != row["actual"] for q in (baseline,covariance))):
            raise ValueError(f"The archived inventory forecast differs on {issue}")
        anchor = float(baseline[INVENTORY_LEADER]) + float(covariance["same_week_shared_pass"]) - prediction[SUPPLEMENT]
        yield {k:row[k] for k in ("release_date", "study_phase", "week_ending", "cutoff", "known", "actual")} | prediction | dict(
            measured_roofs=len(row["observed"]), corrected_roofs=n, available_labels=len(labels),
            latest_height_label=max((q["ready"] for q in labels), default=None),
            **{f"long_{m}":anchor + v - prediction[SUPPLEMENT] for m,v in prediction.items()})


def metrics(rows, controls, trades):
    summaries = {}
    for phase in PHASES:
        sample = [q for q in rows if q["study_phase"] == phase]
        arm = {}
        for scope, group in (("all", sample), ("measured", [q for q in sample if q["measured_roofs"]]),
                             ("corrected", [q for q in sample if q["corrected_roofs"]])):
            if not group:
                continue
            e = {m:1000 * np.array([abs(q[f"long_{m}"] - q["actual"]) for q in group]) for m in MODELS}
            arm[scope] = {m:dict(releases=len(v), mae_kbbl=float(v.mean()),
                                 paired_delta_95_kbbl=block_bootstrap_mean(v - e[SUPPLEMENT])) for m,v in e.items()}
        pnl = {}
        for name in (*MARKET_MODELS, *(f"{m}_conditional_exit" for m in MARKET_MODELS)):
            group = [q for q in trades if (q["study_phase"], q["method"]) == (phase,name)]
            pnl[name] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(q["net_340_usd"] for q in group))
        summaries[phase] = dict(inventory=arm, trading=pnl)
        print(phase, {m:round(a["mae_kbbl"], 2) for m,a in arm["measured"].items()},
              {m:v["net_170_usd"] for m,v in pnl.items()})
    physical = {}
    for label in sorted({q["test"] for q in controls}):
        subset = [q for q in controls if q["test"] == label]
        physical[label] = dict(observations=len(subset), terminals=len({q["terminal_id"] for q in subset}),
            mae_m={m:float(np.mean([abs(q[m] - q["actual_m"]) for q in subset])) for m in (*MODELS, MIDPOINT)})
    return dict(status="exploratory_on_reviewed_history", phases=summaries, engineering=physical)


def register(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    recovery_path = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
    sat_path = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    cov_path = DERIVED / f"cushing_overpass_covariance_{issue}_registration.json"
    original,recovery,satellite,covariance = (json.loads(p.read_bytes()) for p in (original_path,recovery_path,sat_path,cov_path))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(r[k] != original[k] for r in (recovery, satellite, covariance) for k in LINEAGE):
        raise ValueError("The height model's original forecast clocks differ")
    if not path.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The height response must be registered before the report")
        for q in (satellite,covariance):
            checked_hashes(q["sources_sha256"])
        checked_hashes({f"data/derived/{k}":v for k,v in recovery["source_sha256"].items()})
        stem = f"cushing_forward_native_optical_{issue}"
        native = DERIVED / f"{stem}_summary.json"
        area = float(json.loads(native.read_bytes())["dated_roof_area_m2"])
        g = indexed_by_roof("cushing_modern_lidar_optical_registry")
        source = defaultdict(set)
        allowed = {r["candidate_id"] for r in original["measured_pairs"]}
        for q in read(f"{stem}_pairs"):
            tank = q["candidate_id"]
            if tank not in allowed or q["registered_both_shadow_diagnostics_pass"] != "True" or q["registration_status"] != "checked_offset":
                continue
            asset = g[tank]
            shell = float(asset["prior_rim_z_p95_native"]) - float(asset["prior_ground_z_p50_native"])
            source[issue,tank,q["earlier_version"],q["later_version"]].add((float(q["earlier_registered_gap_m"]),
                float(q["later_registered_gap_m"]), shell, float(asset["diameter_m"])))
        observed = [q | dict(area=q["volume_per_m"] * BARREL_M3 * 1e6, available_utc=q["source_available_utc"])
                    for q in original["measured_pairs"]] + recovery["recovered_roofs"]
        if (len(observed) != satellite["measured_roofs"] or len({q["candidate_id"] for q in observed}) != len(observed)
                or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observed)
                or satellite["optical_input_policy"] != "frozen_panel_without_eligible_after_target_observations"):
            raise ValueError("Wrong forward B04 measurement set")
        book = state()
        labels = [q for q in training_rows() if q["ready"] < cutoff]
        arms, corrected = corrections(issue, observed, allowed, source, labels)
        current = dict(cutoff=cutoff, known=deadline, week_ending=original["week_ending"], hub_area=area,
                       area_fraction=math.fsum(q["area"] for q in observed) / area, public=original["public_ensemble_kbbl"] / 1000)
        with threadpool_limits(limits=1):
            for _ in history(labels, book, before=cutoff):
                pass
            estimates = {m:1000 * run_forecast(book["models"][m], book["errors"], current | dict(observed=q), False, update=False)[0]
                         for m,q in arms.items()}
        stamp = datetime.now(timezone.utc)
        if (not corrected or not cutoff <= stamp < deadline
                or not math.isclose(estimates[SUPPLEMENT], satellite["estimates_kbbl"][SUPPLEMENT], abs_tol=1e-8, rel_tol=0)):
            raise ValueError("The dated response does not reproduce its forward baseline")
        anchor = covariance["long_history_with_additive_pass_delta_kbbl"]["same_week_shared_pass"]
        sources = (original_path, recovery_path, sat_path, cov_path, native, DERIVED / f"{stem}_pairs.csv", Path(__file__).resolve(),
                   *(DERIVED / f"{m}.csv" for m in (f"{STEM}_forecasts", "cushing_dated_height_calibration_training",
                       "cushing_modern_lidar_optical_registry", "cushing_height_multisource_ml_controls", "shadow_geometry_estimates",
                       "sentinel2_l1c_archive_availability", "sentinel2_space_time_coverage_roof_registry",
                       "cushing_outer_capture_pairs_fits", "cushing_outer_capture_pairs_supplement_fits",
                       "sentinel2_wider_registration_gap_pairs", *VOLUMES)),
                   *(ROOT / "scripts" / f"{m}.py" for m in ("replay_cushing_dated_height_calibration", "fit_cushing_intraweek",
                       "fit_cushing_outer_capture_nowcast", "fit_cushing_overpass_covariance")))
        freeze(path, {k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), measured_roofs=len(observed),
            corrected_roofs=corrected, latest_height_label_utc=max(q["ready"] for q in labels).isoformat(), estimates_kbbl=estimates,
            long_estimates_kbbl={m:anchor + v - estimates[SUPPLEMENT] for m,v in estimates.items()},
            sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    if (any(record[k] != original[k] for k in LINEAGE) or set(record["estimates_kbbl"]) != set(MODELS)
            or not datetime.fromisoformat(record["latest_height_label_utc"]) < cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline):
        raise ValueError("The height response forecast does not match its registration")
    outcome = score_release(original_path, original)
    summary = dict(status=outcome["status"], registration_sha256=digest(path), estimates_kbbl=record["estimates_kbbl"],
                   long_estimates_kbbl=record["long_estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        summary.update(actual_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                       errors_kbbl={m:abs(v - actual) for m,v in record["long_estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode())
    print(summary)


def main():
    controls = list(training_rows())
    with threadpool_limits(limits=1):
        physical = independent_controls(controls)
        rows = list(history(controls, state()))
        book = {r["release_date"]:r | {CURRENT_ONLY:r[MODELS[2]]} for r in rows}
        trades = list(replay_weighted_market(book, MARKET_MODELS, reference=SUPPLEMENT, historical_inputs={CURRENT_ONLY:SUPPLEMENT}))
    report = metrics(rows, physical, trades)
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_trades", trades)
    columns = ("test", "candidate_id", "terminal_id", "actual_m", *MODELS, MIDPOINT)
    export(f"{STEM}_physical_checks", [{k:q[k] for k in columns} for q in physical])
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    print(report["engineering"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    register(args.release.isoformat()) if args.release else main()
