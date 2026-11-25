"""Calibrate the noise of weekly optical inventory observations against past EIA releases."""

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from audit_cushing_cl_equal_model_risk import paired_evidence
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_curve_shape import MIXED, MIXED_GAP, STEM as GAP_STEM, market_metrics
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import inputs, pass_id
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from replay_cushing_weighted_inventory_market import replay_weighted_market

STEM = "cushing_optical_likelihood"
GAUSSIAN = "fitted_optical_gaussian"
STUDENT = "fitted_optical_student"
FIXED = "fixed_optical_student"
DIRECT = (GAUSSIAN, STUDENT, FIXED)
JOINT = "joint_optical_student"
CURRENT = "optical_student_current_only"
VALIDATED = "evidence_weighted_noise"
VALIDATED_JOINT = "joint_evidence_weighted_noise"
BASES = (SUPPLEMENT, MIXED_GAP)
TRADING = (*BASES, *DIRECT, JOINT, CURRENT, VALIDATED, VALIDATED_JOINT)
GRID = np.linspace(-8, 8, 1601)
DF = 5
MIN_OBSERVATIONS = 32
WINDOW = 104


def timing_summaries(row, history, errors):
    cutoff = row["cutoff"]
    history = [q for q in history if q["known"] < cutoff]
    sigma = prior_scale([q for q in errors if q["known"] < cutoff])
    weights = learned_weights(history)
    current = prepared(row, row["public"], row["spanning"])
    nearby = [q["prepared"] for q in history if cutoff - q["cutoff"] <= MAX_SCENE_AGE]
    values = np.array(estimate(current, nearby))
    information, score = values[:, 1:].T
    if np.any(information <= 0) or not np.isfinite(values).all():
        raise ValueError(f"Invalid optical likelihood on {row['release_date']}")
    point = row["public"] + float(weights @ np.clip(score / (sigma**-2 + information), -1.25, 1.25))
    if not math.isclose(point, row["reference"], abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"The original optical reconstruction differs on {row['release_date']}")
    passes = {pass_id(q[k]) for q in row["observed"] for k in ("scene_before", "scene_after")}
    if len(passes) < 2:
        raise ValueError(f"Weekly optical measurement needs two separate overpasses on {row['release_date']}")
    size = math.log1p(row["area_fraction"] / .10)
    breadth = min(len(passes) - 2, 4) / 4
    design = np.array((1., size - .7, breadth))
    standardized = score / np.sqrt(information)
    return dict(sigma=sigma, weights=weights, information=information, score=score, design=design, passes=len(passes),
                standardized=standardized, unscaled_revision=values[:, 0])


def fit_noise(past, now, *, degrees):
    if len(past) < MIN_OBSERVATIONS:
        return np.zeros(3)
    x = np.stack([r["design"] for r in past])
    e = np.stack([r["optical_residual"] for r in past])
    timing = np.stack([r["weights"] for r in past])
    age = np.array([(now - r["known"]).total_seconds() / 604800 for r in past])
    recency = np.exp2(-age / 78)
    recency /= recency.sum()
    penalty = np.array((.10, .25, .25))

    def objective(beta):
        log_scale = x @ beta
        z2 = np.square(e) * np.exp(-2 * log_scale[:, None])
        if degrees is None:
            loss, slope = .5 * z2, 1. - z2
        else:
            loss = (degrees + 1) / 2 * np.log1p(z2 / degrees)
            slope = 1. - (degrees + 1) * z2 / (degrees + z2)
        mean_loss = recency @ (log_scale + np.sum(timing * loss, axis=1))
        gradient = (recency * np.sum(timing * slope, axis=1)) @ x + 2 * penalty * beta
        return float(mean_loss + penalty @ beta**2), gradient

    fit = minimize(objective, np.zeros(3), jac=True, method="L-BFGS-B", bounds=[(-.7, .7), (-.5, .5), (-.5, .5)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"Optical noise calibration failed: {fit.message}")
    return fit.x


def update(sigma, weights, information, score, scale, *, degrees):
    if degrees is None:
        precision = sigma**-2 + information / scale**2
        terms = (score / scale**2) / precision
        return float(weights @ np.clip(terms, -1.25, 1.25))
    z2 = np.square((score[:, None] / np.sqrt(information[:, None]) - np.sqrt(information[:, None]) * GRID) / scale)
    log_mass = -.5 * (GRID / sigma)**2 - (degrees + 1) / 2 * np.log1p(z2 / degrees)
    mass = np.exp(log_mass - logsumexp(log_mass, axis=1)[:, None])
    # With a fixed book of timing weights, keep its original mixture of
    # component point forecasts. Within each component use a posterior median.
    centres = [np.interp(.5, np.cumsum(p) - .5 * p, GRID) for p in mass]
    return float(weights @ np.clip(centres, -1.25, 1.25))


def learned_blend(history):
    """Shrink toward the original model until past forecast gains justify a deviation."""
    past = [r for r in history if r["noise_trained"]][-78:]
    if len(past) < MIN_OBSERVATIONS:
        return np.zeros(2)
    delta = np.stack([r["optical_deltas"] for r in past])
    target = np.array([r["eia_error"] for r in past])
    recency = np.exp2(-np.arange(len(past) - 1, -1, -1) / 52)
    penalty = 13 * .10**2

    def objective(shares):
        error = target - delta @ shares
        smoothed = np.hypot(error, .20)
        return float(recency @ smoothed + penalty * (shares @ shares)), -(recency * error / smoothed) @ delta + 2 * penalty * shares

    fit = minimize(objective, np.zeros(2), jac=True, method="SLSQP", bounds=[(0, 1), (0, 1)],
                   constraints=[dict(type="ineq", fun=lambda w: 1 - w.sum(), jac=lambda _: -np.ones(2))])
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"The evidence-weighted optical blend did not converge: {fit.message}")
    weights = np.clip(fit.x, 0., 1.)
    return weights / max(1., weights.sum())


def fitted_week(row, history, errors):
    previous = [q for q in history if q["known"] < row["cutoff"]][-WINDOW:]
    effects = dict.fromkeys(DIRECT, 0.)
    scales = dict.fromkeys(DIRECT, 1.)
    diagnostics = dict.fromkeys(("overpasses", "eia_optical_z", "reference_noise_nll", "gaussian_noise_nll", "student_noise_nll"))
    shares = np.zeros(2)
    if row["observed"]:
        fit = timing_summaries(row, history, errors)
        for name, degrees in ((GAUSSIAN, None), (STUDENT, DF), (FIXED, DF)):
            beta = fit_noise(previous, row["cutoff"], degrees=degrees) if name != FIXED else np.zeros(3)
            scale = float(np.exp(np.clip(fit["design"] @ beta, -.7, .7)))
            scales[name] = scale
            fitted = update(*(fit[k] for k in ("sigma", "weights", "information", "score")), scale, degrees=degrees)
            effects[name] = row["public"] + fitted - row["reference"]
        shares = learned_blend(previous)
        diagnostics["overpasses"] = fit["passes"]
        if "actual" in row:
            known = row["known"]
            realized = row["actual"] - row["public"]
            residual = fit["standardized"] - np.sqrt(fit["information"]) * realized
            g, s = scales[GAUSSIAN], scales[STUDENT]
            weights = fit["weights"]
            normal_constant = math.log(2 * math.pi) / 2
            t_constant = math.lgamma(DF / 2) - math.lgamma((DF + 1) / 2) + math.log(DF * math.pi) / 2
            diagnostics.update(eia_optical_z=float(weights @ residual),
                reference_noise_nll=float(weights @ (.5 * residual**2 + normal_constant)),
                gaussian_noise_nll=float(weights @ (.5 * (residual / g)**2 + math.log(g) + normal_constant)),
                student_noise_nll=float(weights @ ((DF + 1) / 2 * np.log1p((residual / s)**2 / DF) + math.log(s) + t_constant)))
            history.append(dict(cutoff=row["cutoff"], known=known, prepared=prepared(row, row["actual"], row["spanning"]),
                absolute_errors=abs(row["public"] + fit["unscaled_revision"] - row["actual"]), weights=weights,
                design=fit["design"], optical_residual=residual, eia_error=row["actual"] - row["reference"],
                optical_deltas=(effects[GAUSSIAN], effects[STUDENT]), noise_trained=len(previous) >= MIN_OBSERVATIONS))
    if "actual" in row:
        errors.append(dict(known=row["known"], error=row["actual"] - row["public"]))
    return effects, scales, shares, diagnostics | dict(optical_calibration_weeks=len(previous),
        latest_calibration_label_utc=previous[-1]["known"] if previous else None)


def candidates(base, joint, effects, shares):
    blend = float(shares @ [effects[GAUSSIAN], effects[STUDENT]])
    return {SUPPLEMENT: base, MIXED_GAP: joint, JOINT: joint + effects[STUDENT], CURRENT: base + effects[STUDENT],
            VALIDATED: base + blend, VALIDATED_JOINT: joint + blend, **{m: base + d for m, d in effects.items()}}


def forecasts():
    archived = {(r["release_date"], r["channel"]): r for r in read(f"{GAP_STEM}_forecasts")}
    history, errors = [], []
    for row in inputs():
        issue, cutoff, known = (row[k] for k in ("release_date", "cutoff", "known"))
        effects, scales, shares, diagnostics = fitted_week(row, history, errors)
        for name in ("market", "inventory"):
            anchor = archived[issue, name]
            if (any(row[k] != datetime.fromisoformat(anchor[k]) for k in ("cutoff", "known"))
                    or any(row[k] != anchor[k] for k in ("week_ending", "study_phase")) or row["actual"] != float(anchor["actual"])
                    or name == "market" and row["reference"] != float(anchor[SUPPLEMENT])):
                raise ValueError(f"Inconsistent publication vintage on {issue}/{name}")
            result = {k: row[k] for k in ("release_date", "study_phase", "week_ending", "cutoff", "known", "actual")}
            base, joint = (float(anchor[k]) for k in BASES)
            yield result | dict(channel=name, primary_roofs=len(row["observed"]), area_fraction=row["area_fraction"],
                gaussian_blend_weight=float(shares[0]), student_blend_weight=float(shares[1]), **diagnostics,
                **{MIXED: float(anchor[MIXED])}, **candidates(base, joint, effects, shares),
                **{f"{m}_scale": v for m, v in scales.items()})


def inventory(group):
    y = np.array([r["actual"] for r in group])
    absolute = {name: 1000 * abs(y - np.array([r[name] for r in group]))
                for name in (*BASES, MIXED, *DIRECT, JOINT, VALIDATED, VALIDATED_JOINT)}
    return dict(weeks=len(group), metrics={name: dict(mae_kbbl=float(e.mean()), rmse_kbbl=float(np.sqrt(np.mean(e**2))),
        delta_vs_paired_baseline_95_kbbl=block_bootstrap_mean(e - absolute[MIXED_GAP if name in (JOINT, VALIDATED_JOINT) else SUPPLEMENT]))
        for name, e in absolute.items()})


def summarize(rows, trades):
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        group = [r for r in rows if r["study_phase"] == phase and r["channel"] == "inventory"]
        panels = {"all": group, "primary_optics": [r for r in group if r["primary_roofs"]],
                  "more_than_two_passes": [r for r in group if r["overpasses"] and r["overpasses"] > 2]}
        comparisons = {}
        for suffix in ("", "_conditional_exit"):
            matched = {m: [r for r in trades if r["study_phase"] == phase and r["method"] == m + suffix] for m in TRADING}
            comparisons[suffix.strip("_") or "fifth_session"] = {m: market_metrics(g) | dict(
                paired_vs_baseline=paired_evidence(g, matched[MIXED_GAP if m in (JOINT, VALIDATED_JOINT) else SUPPLEMENT])[0])
                for m, g in matched.items()}
        panel = {name: inventory(g) for name, g in panels.items() if g}
        measured = panels["primary_optics"]
        loss = {name: float(np.mean([r[f"{name}_noise_nll"] for r in measured])) for name in ("reference", "gaussian", "student")}
        report["phases"][phase] = dict(inventory=panel, market=comparisons, mean_optical_negative_log_likelihood=loss)
        print(phase, {m: round(q["mae_kbbl"], 3) for m, q in panel["all"]["metrics"].items()},
              {m: q["net_170_usd"] for m, q in comparisons["conditional_exit"].items()}, flush=True)
    return report


def main():
    with threadpool_limits(limits=1):
        rows = list(forecasts())
        forecast_book = {r["release_date"]: r for r in rows if r["channel"] == "market"}
        trades = list(replay_weighted_market(forecast_book, TRADING, reference=SUPPLEMENT, historical_inputs={CURRENT: SUPPLEMENT}))
    report = summarize(rows, trades)
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


def register(issue, path, origin, original):
    recovery = origin.with_name(f"{origin.stem}_recovered_optical.json")
    market = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    mixed = DERIVED / f"cushing_mixed_optical_intervals_{issue}_registration.json"
    native = DERIVED / f"cushing_forward_native_optical_{issue}_summary.json"
    recovered, trade, intervals, geometry = (json.loads(p.read_bytes()) for p in (recovery, market, mixed, native))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
        raise ValueError("The optical likelihood must be registered before publication")
    for receipt in (trade, intervals):
        checked_hashes(receipt["sources_sha256"])
    if any(receipt[k] != original[k] for receipt in (recovered, trade, intervals) for k in LINEAGE):
        raise ValueError("The frozen optical forecasts use different publication clocks")
    area = float(geometry["dated_roof_area_m2"])
    observations = [q | dict(area=q["volume_per_m"] * BARREL_M3 * 1e6, available_utc=q["source_available_utc"])
                    for q in original["measured_pairs"]] + recovered["recovered_roofs"]
    if (not observations or intervals["additional_roofs"]
            or not len({q["candidate_id"] for q in observations}) == len(observations) == trade["measured_roofs"] == intervals["measured_roofs"]
            or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observations)):
        raise ValueError("The registered optical likelihood requires the same nonempty pair set as the frozen forecasts")
    expected = trade["estimates_kbbl"][SUPPLEMENT] / 1000
    row = dict(release_date=issue, week_ending=original["week_ending"], cutoff=cutoff, spanning=False, hub_area=area,
               observed=observations, area_fraction=math.fsum(q["area"] for q in observations) / area,
               public=original["public_ensemble_kbbl"] / 1000, reference=expected)
    history, errors = [], []
    with threadpool_limits(limits=1):
        for prior in inputs(before=cutoff):
            fitted_week(prior, history, errors)
        effects, scales, shares, diagnostic = fitted_week(row, history, errors)
    latest = diagnostic["latest_calibration_label_utc"]
    recorded = datetime.now(timezone.utc)
    if latest is None or not latest < cutoff <= recorded < deadline:
        raise ValueError("The trained likelihood missed its data or registration clock")
    points = {}
    for channel, basis in (("market", expected), ("inventory", intervals["estimates_kbbl"][MIXED] / 1000)):
        # The joint fallback activates only on weeks without useful optics.
        projected = candidates(basis, basis, effects, shares)
        points[channel] = {k: 1000 * v for k, v in projected.items() if k != CURRENT}
    sources = (origin, recovery, market, mixed, native, Path(__file__).resolve(), DERIVED / f"{STEM}_forecasts.csv",
               ROOT / "scripts/fit_cushing_intraweek.py", ROOT / "scripts/fit_cushing_overpass_covariance.py")
    freeze(path, {k: original[k] for k in LINEAGE} | dict(recorded_at_utc=recorded.isoformat(),
        latest_calibration_label_utc=latest.isoformat(), optical_calibration_weeks=diagnostic["optical_calibration_weeks"],
        gaussian_blend_weight=float(shares[0]), student_blend_weight=float(shares[1]), optical_noise_scales=scales,
        measured_roofs=len(observations), overpasses=diagnostic["overpasses"], projections_kbbl=points,
        sources_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources}))


def forward(issue):
    origin = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    original = json.loads(origin.read_bytes())
    path = DERIVED / f"{STEM}_{issue}_registration.json"
    if not path.exists():
        register(issue, path, origin, original)
    saved = json.loads(path.read_bytes())
    checked_hashes(saved["sources_sha256"])
    latest, cutoff, recorded, deadline = (datetime.fromisoformat(saved[k]) for k in (
        "latest_calibration_label_utc", "information_cutoff_utc", "recorded_at_utc", "registration_deadline_utc"))
    if any(saved[k] != original[k] for k in LINEAGE) or not latest < cutoff <= recorded < deadline:
        raise ValueError("The saved optical likelihood uses the wrong time or release")
    outcome = score_release(origin, original)
    result = {"release_date": issue, "status": outcome["status"], "registration_sha256": digest(path),
              "projections_kbbl": saved["projections_kbbl"]}
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_errors_kbbl={channel: {m: abs(actual - v) for m, v in values.items()}
                                           for channel, values in saved["projections_kbbl"].items()})
    write(DERIVED / f"{STEM}_{issue}_score.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    forward(args.release.isoformat()) if args.release else main()
