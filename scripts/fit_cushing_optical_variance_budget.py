"""Measure inventory sensitivity to optical noise and unobserved terminal flows."""

import json
import math
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import FIXED_MODELS, estimate, learned_weights, preparation, prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, SCENE_SIGMA_M
from fit_cushing_terminal_balance import geometries, mapped_areas

STEM = "cushing_optical_variance_budget"
BASE = "all_recovered"
BEST_SCENES = "perfect_height_assumption"
FLOW_SCALES = {"terminal_flow_06": .6, "terminal_flow_12": 1.2}
ENDPOINT_SHARES = {"repeat_endpoint_half": .5, "repeat_endpoint_full": 1.}
MODELS = (PUBLIC, "carried_scene", BASE, "reduced_height_noise", BEST_SCENES, *FLOW_SCALES, *ENDPOINT_SHARES)
ARMS = MODELS[2:]
MARKET_MODELS = (PUBLIC, BASE, *FLOW_SCALES, *ENDPOINT_SHARES)


def prepared(row, inventory, arm, transfers):
    # The original diagonal is the sum of the optical fit variance and
    # residual tank activity. Improve the former in the two sensitivity runs.
    observed = row["observed"]
    physical = np.array([(LOCAL_ACTIVITY_M * r["volume_per_m"])**2 * r["duration"] for r in observed])
    if arm in ("reduced_height_noise", BEST_SCENES):
        ratio = .25 if arm == "reduced_height_noise" else 0.
        observed = [r | dict(variance=float(local + ratio * (r["variance"] - local)))
                    for r, local in zip(observed, physical, strict=True)]
    result = preparation(row | dict(observed=observed), inventory)
    if arm in FLOW_SCALES:
        result["local"] += FLOW_SCALES[arm]**2 * transfers
    if arm in ENDPOINT_SHARES:
        movable = np.array([r["variance"] for r in observed])-physical
        if np.any(movable < -1e-12):
            raise ValueError(f"Negative optical fit variance on {row['release_date']}")
        moved = ENDPOINT_SHARES[arm]*np.maximum(movable,0.)
        result["local"] -= np.diag(moved)
        strength = np.sqrt(moved/2)
        # Each measurement loads positively on the initial gap and
        # negatively on the final gap. Variance is unchanged per observation.
        result["endpoint_errors"] = [{(r["candidate_id"],r["scene_before"]):float(sigma),
                                      (r["candidate_id"],r["scene_after"]):-float(sigma)}
                                     for r,sigma in zip(observed,strength,strict=True)]
    return result


def score_prediction(row, past, public_history, arm, preparation_args):
    recent = [r["prepared"] for r in past if row["cutoff"] - r["cutoff"] <= MAX_SCENE_AGE]
    scene_sigma = {BEST_SCENES: 0., "reduced_height_noise": SCENE_SIGMA_M / 2}.get(arm, SCENE_SIGMA_M)
    model = prepared(row, row["public"], arm, preparation_args)
    updates = estimate(model, recent, scene_sigma_m=scene_sigma)
    variance = prior_scale(public_history)**2
    weights = learned_weights(past)
    information = np.array([info for _, info, _ in updates])
    # For the observed footprint, this variance ratio states how uncertain
    # the hub still is even if the height measurement component vanished.
    posterior_sigma = float(weights @ np.sqrt(1 / (1 / variance + information)))
    revisions = np.clip([score / (1 / variance + info) for _, info, score in updates], -1.25, 1.25)
    fixed_errors = np.abs([row["public"] + change - row["actual"] for change, _, _ in updates])
    return row["public"] + float(weights @ revisions), posterior_sigma, fixed_errors


def forecasts():
    areas = mapped_areas()
    original = indexed("cushing_recovered_optical_forecasts")
    seen = {m: [] for m in ARMS}
    errors = []
    for r, arms, _ in combined_sources():
        issue, week, cutoff = (r[k] for k in ("release_date", "week_ending", "cutoff"))
        source = original[issue]
        if (datetime.fromisoformat(source["cutoff_utc"]) != cutoff or
                datetime.fromisoformat(source["known_utc"]) != r["known"] or
                not cutoff < r["known"] or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in arms[-1])):
            raise ValueError(f"Inventory inputs or outcome have an inconsistent timestamp on {issue}")
        footprint = areas[issue, week]
        total = math.fsum(a for a, _ in footprint.values())
        if total <= 0 or any(t != cutoff for _, t in footprint.values()):
            raise ValueError(f"The dated terminal roster differs on {issue}")
        sample = arms[-1]
        fraction = math.fsum(q["area"] for q in sample) / total
        time_share = math.fsum(q["sensitivity"] for q in sample)
        if not 0 <= time_share <= fraction <= 1 + 1e-9 or len(sample) != int(source["total_recovered_tanks"]):
            raise ValueError(f"Inconsistent measured footprint on {issue}")
        state = r | dict(observed=sample, area_fraction=fraction)
        transfers = geometries(footprint, state) if sample else None
        public = r["public"]
        public_history = [q for q in errors if q["known"] < cutoff]
        sigma = prior_scale(public_history)
        values = {}
        for arm in ARMS:
            past = [q for q in seen[arm] if q["known"] < cutoff]
            prediction, posterior = public, sigma
            if sample:
                prediction, posterior, residuals = score_prediction(state, past, public_history, arm, transfers)
                seen[arm].append(dict(cutoff=cutoff, known=r["known"], absolute_errors=residuals,
                                      prepared=prepared(state, r["actual"], arm, transfers)))
            values[arm] = prediction
            values[f"{arm}_posterior_sigma_mbbl"] = posterior
        if not math.isclose(values[BASE], float(source[BASE]), abs_tol=1e-9):
            raise ValueError(f"The original satellite nowcast did not reproduce on {issue}")
        errors.append(dict(error=r["actual"] - public, known=r["known"]))
        yield state | values | dict(public_ensemble=public, carried_scene=float(source["carried_scene"]),
                                     prior_sigma_mbbl=sigma, space_time_fraction=time_share, measured_roofs=len(sample),
                                     latest_training_label_utc=public_history[-1]["known"].isoformat() if public_history else None)


def inventory(group):
    result = forecast_metrics(group, MODELS)
    for m in ARMS[1:]:
        delta = [1000 * (abs(r["actual"] - r[m]) - abs(r["actual"] - r[BASE])) for r in group]
        result["models"][m]["delta_vs_incumbent_95_kbbl"] = block_bootstrap_mean(delta)
    measured = [r for r in group if r["measured_roofs"]]
    if measured:
        result["measured_geometry"] = dict(weeks=len(measured),
            median_area_fraction=float(np.median([r["area_fraction"] for r in measured])),
            median_space_time_fraction=float(np.median([r["space_time_fraction"] for r in measured])),
            median_prior_sigma_mbbl=float(np.median([r["prior_sigma_mbbl"] for r in measured])),
            median_posterior_sigma_mbbl={m: float(np.median([r[f"{m}_posterior_sigma_mbbl"] for r in measured])) for m in ARMS})
    return result


def main():
    with threadpool_limits(limits=1):
        rows = list(forecasts())
        # Hypothetical lower instrument error is a sensitivity diagnostic.
        # The strategy test admits only currently observable inputs.
        trades = list(trading(rows, MARKET_MODELS))
    report = dict(status="exploratory_on_reviewed_history", forecasts={}, trading={})
    for phase in PHASES:
        sample = [r for r in rows if r["study_phase"] == phase]
        report["forecasts"][phase] = dict(all=inventory(sample), measured=inventory([r for r in sample if r["measured_roofs"]]))
        by_method = {m: [r for r in trades if r["study_phase"] == phase and r["method"] == m]
                     for m in ("curve", *MARKET_MODELS)}
        base = by_method[BASE]
        report["trading"][phase] = {}
        for m, group in by_method.items():
            if [r["release_date"] for r in group] != [r["release_date"] for r in base]:
                raise ValueError(f"Different trading cohort on {phase}/{m}")
            delta = [r["net_170_usd"] - q["net_170_usd"] for r, q in zip(group, base, strict=True)]
            report["trading"][phase][m] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["gross_usd"] - 340 * abs(r["near_side"]) for r in group),
                delta_vs_incumbent_95_usd=block_bootstrap_mean(delta))
        print(phase, {m: round(v["mae_kbbl"], 2) for m, v in report["forecasts"][phase]["all"]["models"].items()})
        print(phase, {m: v["net_170_usd"] for m, v in report["trading"][phase].items()})
    fields = ("release_date", "week_ending", "study_phase", "actual", "measured_roofs", "area_fraction", "space_time_fraction",
              "prior_sigma_mbbl", "latest_training_label_utc", *MODELS, *(f"{m}_posterior_sigma_mbbl" for m in ARMS))
    export(f"{STEM}_forecasts", [{k: r[k] for k in fields} | dict(cutoff_utc=r["cutoff"].isoformat(), known_utc=r["known"].isoformat())
                                 for r in rows])
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
