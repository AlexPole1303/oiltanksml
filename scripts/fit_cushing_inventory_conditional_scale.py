"""Forecast inventory error scale before EIA and test standardized surprises in CL."""

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_cases, trade
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_inventory_coverage_distribution import evaluate, quantile
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_optical_error_distribution import STEM as INPUTS, inputs
from fit_cushing_outer_capture_nowcast import SUPPLEMENT
from fit_cushing_robust_stack import indexed
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_inventory_conditional_scale"
METHODS = ("pooled", "scaled", "scaled_and_local")
TRADES = ("registered_exit", "pooled_surprise_gate", "scaled_surprise_gate", "scaled_local_surprise_gate")
EXIT_METHOD = "curve_post_report_gate"
WINDOW = 156
MIN_WEEKS = 52
LOG_FLOOR = .12
LARGE_ERROR = 1.  # million barrels


def design(row):
    point, base = (row[k] for k in ("baseline", "unmeasured"))
    return np.array((min(abs(base), 2.), min(abs(point - base), 1.), math.log1p(row["area_fraction"] / .08),
                     row["posterior_scale"], float(row["roofs"] > 0)))


def scale(history, row):
    if len(history) < MIN_WEEKS:
        return 1.
    x = np.stack([q["features"] for q in history])
    target = np.log(LOG_FLOOR + np.array([abs(q["actual"] - q["baseline"]) for q in history]))
    age = np.array([(row["cutoff"] - q["cutoff"]).total_seconds() / 604800 for q in history])
    weights = np.exp2(-age / 104)
    midpoint = np.average(x, weights=weights, axis=0)
    spread = np.maximum(np.sqrt(np.average((x-midpoint)**2, weights=weights, axis=0)), (.30,.15,.45,.15,.30))
    x = np.clip((x - midpoint) / spread, -3., 3.)
    y = target - np.average(target, weights=weights)
    gamma = np.linalg.solve(x.T @ (weights[:, None] * x) + np.eye(x.shape[1]) * (weights.sum() / 2), x.T @ (weights * y))
    return float(np.exp(np.clip(np.clip((row["features"]-midpoint) / spread, -3., 3.) @ gamma, -.45, .45)))


def calibration(row, history):
    past = [q for q in history if q["known"] < row["cutoff"]][-WINDOW:]
    row = row | dict(features=design(row))
    computed_scale = scale(past, row)
    if len(past) < MIN_WEEKS:
        return row | dict(scale=computed_scale, training_weeks=len(past)), {}
    raw = np.array([q["actual"] - q["baseline"] for q in past])
    standardized = raw / np.array([q["scale"] for q in past]) * computed_scale
    uniform = np.ones(len(past))
    x = np.stack([q["features"] for q in past])
    d = ((x-row["features"]) / (.60,.25,.70,.20,.70)) ** 2
    age = np.array([(row["cutoff"] - q["known"]).total_seconds() / 604800 for q in past])
    weights = np.exp2(-age/104) * (.35 + .65 * np.exp(-.5 * d.mean(axis=1)))
    distributions = dict(zip(METHODS, ((raw, uniform), (standardized,uniform), (standardized,weights)), strict=True))
    return row | dict(scale=computed_scale, training_weeks=len(past), latest_label_utc=past[-1]["known"]), distributions


def replay(source):
    history = {"inventory":[], "market":[]}
    for row in source:
        channel = history[row["channel"]]
        fitted, dist = calibration(row, channel)
        channel.append(fitted)
        if dist:
            outcome = {}
            for m, (residuals, weights) in dist.items():
                outcome[m] = evaluate(residuals, weights, row["baseline"], row["actual"]) | dict(
                    probability_large_error=float(weights @ (abs(residuals) > LARGE_ERROR) / weights.sum()))
            yield fitted | {f"{m}_{k}":v for m,q in outcome.items() for k,v in q.items()}


def decisions(calibrated):
    forecasts = {q["release_date"]:q for q in calibrated if q["channel"] == "market"}
    anchors = {r["release_date"]:r for r in read("cushing_cl_extension_segments_trades") if r["method"] == EXIT_METHOD}
    for row,_ in market_cases():
        issue = row["release_date"]
        if issue not in anchors:
            continue
        ref = anchors[issue]
        f = forecasts.get(issue)
        if f and (not f["latest_label_utc"] < f["cutoff"] < f["known"] <= row["decision"] < row["entry"] or f["actual"] != row["x"][0]):
            raise ValueError(f"The pre-release distribution is unavailable on {issue}")
        estimate = float(ref["expected_long_move_usd"])
        index = int(ref["exit_session_number"]) - 2
        benchmark = trade(row, TRADES[0], estimate, index, training_at=datetime.fromisoformat(ref["original_latest_training_exit_utc"]))
        if (any(benchmark[k] != ref[k] for k in ("entry_at_utc", "exit_at_utc", "near_month", "far_month"))
                or any(benchmark[k] != int(ref[k]) for k in ("near_side", "gross_usd", "net_170_usd"))):
            raise ValueError(f"The frozen market comparison did not reproduce on {issue}")
        for name, m in zip(TRADES, (None,*METHODS), strict=True):
            percentile = f[f"{m}_percentile"] if m and f else None
            # After EIA publishes, trade only when the release lies outside
            # the middle half of the earlier predictive distribution.
            retained = percentile is None or not .25 < percentile < .75
            t = benchmark if name == TRADES[0] else trade(row, name, estimate if retained else 0., index,
                training_at=datetime.fromisoformat(ref["original_latest_training_exit_utc"]))
            yield t | dict(net_340_usd=t["gross_usd"]-340*abs(t["near_side"]), percentile=percentile,
                           inventory_known_utc=f["known"] if f else None, inventory_training_utc=f["latest_label_utc"] if f else None,
                           calendar_reference_side=benchmark["near_side"])


def report(rows, trades):
    phases = {}
    for phase in PHASES:
        forecast_metrics = {}
        for channel in ("inventory","market"):
            outcomes = [r for r in rows if (r["study_phase"],r["channel"]) == (phase,channel)]
            for group, sample in (("all", outcomes), ("measured", [q for q in outcomes if q["roofs"]])):
                actual = np.array([q["actual"] for q in sample])
                pooled = np.array([q["pooled_crps"] for q in sample])
                values = {}
                for name in METHODS:
                    lo, hi, crps = (np.array([q[f"{name}_{k}"] for q in sample]) for k in ("lo", "hi", "crps"))
                    error = abs(np.array([q["baseline"] for q in sample]) - actual)
                    if not len(actual):
                        continue
                    p_tail = np.array([q[f"{name}_probability_large_error"] for q in sample])
                    p_reference = np.array([q["pooled_probability_large_error"] for q in sample])
                    event = error > LARGE_ERROR
                    brier = (p_tail-event)**2
                    values[name] = dict(weeks=len(actual), crps_kbbl=1000 * float(crps.mean()),
                        paired_delta_crps_95_kbbl=block_bootstrap_mean(1000 * (crps-pooled)),
                        interval_score_kbbl=1000 * float(np.mean(hi-lo+10*(np.maximum(lo-actual,0.)+np.maximum(actual-hi,0.)))),
                        covered=int(np.count_nonzero((lo <= actual) & (actual <= hi))),
                        mean_interval_width_kbbl=1000 * float(np.mean(hi-lo)), point_mae_kbbl=1000 * float(error.mean()),
                        large_errors=int(event.sum()), mean_predicted_large_error_probability=float(p_tail.mean()),
                        large_error_brier=float(brier.mean()),
                        large_error_paired_brier_95=block_bootstrap_mean(brier-(p_reference-event)**2))
                forecast_metrics[f"{channel}_{group}"] = values
        p = {m:[q for q in trades if (q["study_phase"],q["method"]) == (phase,m)] for m in TRADES}
        pnl = {}
        for m,g in p.items():
            base = p[TRADES[0]]
            if [q["release_date"] for q in g] != [q["release_date"] for q in base]:
                raise ValueError("Unmatched market samples")
            pairs = list(zip(g,base,strict=True))
            pnl[m] = robustness(g) | daily_risk(g) | dict(net_340_usd=sum(q["net_340_usd"] for q in g),
                changed_trades=sum(a["near_side"] != b["near_side"] for a,b in pairs),
                delta_95_usd=block_bootstrap_mean([a["net_170_usd"]-b["net_170_usd"] for a,b in pairs]))
        phases[phase] = dict(forecast=forecast_metrics, trading=pnl)
        print(phase, {m:(round(q["crps_kbbl"],2),q["covered"],round(q["mean_interval_width_kbbl"],2))
                     for m,q in forecast_metrics["inventory_all"].items()}, {m:q["net_170_usd"] for m,q in pnl.items()})
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def register(issue):
    base_path = DERIVED / f"{INPUTS}_{issue}_registration.json"
    optics_path = DERIVED / f"cushing_overpass_covariance_{issue}_registration.json"
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    base, optics, original = (json.loads(p.read_bytes()) for p in (base_path,optics_path,original_path))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(q[k] != original[k] for q in (base,optics) for k in LINEAGE) or base["measured_roofs"] != optics["measured_roofs"]:
        raise ValueError("The frozen observations do not share a forecast clock")
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    if not path.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The prediction must be registered before the report")
        for q in (base,optics):
            checked_hashes(q["sources_sha256"])
        history = {"inventory":[], "market":[]}
        for q in inputs():
            if q["known"] < cutoff:
                panel = history[q["channel"]]
                panel.append(calibration(q,panel)[0])
        original_direction = DERIVED / f"cushing_optical_direction_{issue}_registration.json"
        direction = json.loads(original_direction.read_bytes())["channels"]["inventory"]["estimates_kbbl"]
        h = base["projections"]["inventory"]["estimates_kbbl"][SUPPLEMENT]
        unmeasured = float(direction["no_revision"])
        profiles = {}
        for channel, point, control in (("inventory",h,unmeasured),
                ("market",base["projections"]["market"]["estimates_kbbl"][SUPPLEMENT],original["public_ensemble_kbbl"])):
            r = dict(cutoff=cutoff, baseline=point/1000, unmeasured=control/1000, roofs=base["measured_roofs"],
                area_fraction=base["area_fraction"], posterior_scale=optics["posterior_scale_kbbl"]["same_week_shared_pass"]/1000)
            fitted, distributions = calibration(r,history[channel])
            values = {}
            for method, (residuals, weights) in distributions.items():
                bounds, _, _ = quantile(residuals, weights)
                values[method] = dict(zip(("lo", "median", "hi"), (point + 1000 * float(q) for q in bounds), strict=True)) | dict(
                    large_error_probability=float(weights @ (abs(residuals) > LARGE_ERROR) / weights.sum()))
            profiles[channel] = dict(scale=fitted["scale"], latest_label_utc=fitted["latest_label_utc"].isoformat(),
                unmeasured_kbbl=control, point_kbbl=point, estimates=values)
        stamp = datetime.now(timezone.utc)
        if not cutoff <= stamp < deadline or any(not math.isfinite(q["point_kbbl"]) for q in profiles.values()):
            raise ValueError("The forward distribution could not be registered in time")
        sources = (base_path,optics_path,original_path,original_direction,Path(__file__).resolve(),
                   *(DERIVED / f"{m}_forecasts.csv" for m in (INPUTS,STEM,"cushing_height_response_curve","cushing_overpass_covariance",
                                                               "cushing_optical_value","cushing_outer_capture_nowcast")))
        freeze(path,{k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), profiles=profiles,
            sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))
    frozen = json.loads(path.read_bytes())
    checked_hashes(frozen["sources_sha256"])
    if (any(frozen[k] != original[k] for k in LINEAGE) or not cutoff <= datetime.fromisoformat(frozen["recorded_at_utc"]) < deadline
            or any(datetime.fromisoformat(q["latest_label_utc"]) >= cutoff for q in frozen["profiles"].values())):
        raise ValueError("The distribution registration does not reproduce")
    outcome = score_release(original_path,original)
    result = dict(status=outcome["status"], registration_sha256=digest(path), profiles=frozen["profiles"])
    if outcome["status"] == "scored_first_release":
        v = outcome["actual_eia_change_kbbl"]
        result.update(actual_kbbl=v, first_release_sha256=outcome["first_release_sha256"], coverage={c:{m:q["lo"] <= v <= q["hi"]
            for m,q in d["estimates"].items()} for c,d in result["profiles"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    print(result)


def main():
    with threadpool_limits(limits=1):
        rows = list(replay(inputs()))
        trades = list(decisions(rows))
    summary = report(rows,trades)
    export(f"{STEM}_forecasts", [{k:v for k,v in q.items() if k != "features"} for q in rows])
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    register(args.release.isoformat()) if args.release else main()
