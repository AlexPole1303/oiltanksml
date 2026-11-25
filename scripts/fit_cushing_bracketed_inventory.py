"""Test pre-release observations that bracket the EIA measurement times."""

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from statistics import median

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from build_cushing_holdout_optical_recovery import DATED, EDGES, equal, timestamp
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, NOISE_FLOOR_M, PHASES, PUBLIC, learned_beta
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import SCENE_SIGMA_M, TERMINAL_SIGMA_M

STEM = "cushing_bracketed_inventory"
INCUMBENT = "all_recovered"
MODELS = (PUBLIC, "carried_scene", INCUMBENT, "bracket_fixed", "bracket_precision", "bracket_learned")


def brackets():
    groups = defaultdict(dict)
    for phase in PHASES:
        for r in read(f"shadow_timing_{phase}_pairs"):
            if r["geometry_mode"] != DATED or r["timing_status"] != "both_boundaries_interpolated":
                continue
            key = r["release_date"], r["candidate_id"]
            if key[1] in groups[key[0]]:
                raise ValueError(f"Duplicate timing bracket for {key}")
            groups[key[0]][key[1]] = r | dict(phase=phase)
    return groups


def native_fits(groups):
    needed = {phase: set() for phase in PHASES}
    for group in groups.values():
        for row in group.values():
            for edge in EDGES:
                for side in ("before", "after"):
                    prefix = f"{edge}_{side}"
                    key = row["candidate_id"], row[f"{prefix}_version_id"], row[f"{prefix}_canonical_l2a_id"]
                    needed[row["phase"]].add(key)
    available = {}
    for phase in PHASES:
        with (DERIVED / f"shadow_timing_{phase}_observations.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = row["candidate_id"], row["version_id"], row["canonical_l2a_id"]
                if row["geometry_mode"] != DATED or key not in needed[phase]:
                    continue
                if (phase, *key) in available:
                    raise ValueError(f"Duplicate dated fit for {phase}/{key}")
                available[phase, *key] = row
    if len(available) != sum(map(len, needed.values())):
        raise ValueError("A dated interpolation lacks its native fit")
    return available


def matched_bracket(pair, bracket, cutoff, reporting, fits):
    if (pair["scene_before"], pair["scene_after"]) != tuple(bracket[f"{edge}_before_version_id"] for edge in EDGES):
        return None
    if (not equal(pair["area"], bracket["circular_area_m2"]) or
            timestamp(bracket, "inputs_available_before_utc") != cutoff or any(
                not bracket[k] or bracket[k] > cutoff.date().isoformat()
                for k in ("lidar_evidence_date", "roof_evidence_date")
            )):
        raise ValueError(f"Mismatched dated bracket on {bracket['release_date']}/{bracket['candidate_id']}")
    terms = defaultdict(float)
    delta_terms = defaultdict(float)
    fit_sigma = {}
    snapshot_noise, motion_noise, adjustment_m, native_anchor = 0., 0., 0., 0.
    latest = None
    for edge, sign in zip(EDGES, (1, -1), strict=True):
        centre = timestamp(bracket, f"{edge}_measurement_at_utc")
        if centre != reporting[edge]:
            raise ValueError(f"Different {edge} reporting endpoint on {bracket['release_date']}")
        fields = [f"{edge}_{side}" for side in ("before", "after")]
        capture = [timestamp(bracket, f"{f}_captured_at_utc") for f in fields]
        available = [max(timestamp(bracket, f"{f}_{kind}_available_at_utc") for kind in ("archive", "geometry")) for f in fields]
        if not capture[0] <= centre < capture[1] or any(
            not stamp <= known < cutoff for stamp, known in zip(capture, available, strict=True)
        ):
            raise ValueError(f"Ineligible bracket clock on {bracket['release_date']}")
        edge_latest = max(available)
        latest = edge_latest if latest is None else max(latest, edge_latest)
        span = (capture[1] - capture[0]).total_seconds() / 3600
        weight = (centre - capture[0]) / (capture[1] - capture[0])
        heights = [float(bracket[f"{f}_gap_m"]) for f in fields]
        if not equal(weight, bracket[f"{edge}_interpolation_weight"]) or not equal(span, bracket[f"{edge}_bracket_hours"]):
            raise ValueError(f"Bracket weights differ on {bracket['release_date']}")
        adjustment_m += sign * weight * (heights[1] - heights[0])
        native_anchor += sign * heights[0]
        for field, coefficient, contrast, height in zip(fields, (1 - weight, weight), (-weight, weight), heights, strict=True):
            version = bracket[f"{field}_version_id"]
            native = fits[bracket["phase"], bracket["candidate_id"], version, bracket[f"{field}_canonical_l2a_id"]]
            if (native["fit_status"] != "resolved_in_geometry_diagnostic" or not equal(native["estimated_gap_m"], height)
                or any(timestamp(native, f"{k}_available_at_utc") != timestamp(bracket, f"{field}_{k}_available_at_utc")
                       for k in ("archive", "geometry"))):
                raise ValueError(f"Native fit differs from the bracket on {bracket['release_date']}")
            terms[version] += sign * coefficient
            delta_terms[version] += sign * contrast
            bounds = [float(native[f"near_optimal_gap_{side}_m"]) for side in ("min", "max")]
            if not bounds[0] <= height <= bounds[1]:
                raise ValueError(f"The native fit is outside its diagnostic range on {bracket['release_date']}")
            fit_sigma[version] = max(NOISE_FLOOR_M, (bounds[1] - bounds[0]) / 2)
        motion_noise += LOCAL_ACTIVITY_M**2 * span / 168 * weight * (1 - weight)
        snapshot_noise += NOISE_FLOOR_M**2 * ((1 - weight)**2 + weight**2)
    volume = pair["volume_per_m"]
    # Retain the independently registered pair as the height anchor. Use
    # differences from the common single-scene fits only for the timing update.
    disagreement_m = abs(pair["pair_signal"] / volume - native_anchor)
    precision_variance = volume**2 * (2 * NOISE_FLOOR_M**2 + motion_noise + LOCAL_ACTIVITY_M**2 + disagreement_m**2
                                      + math.fsum((coefficient * fit_sigma[version])**2 for version, coefficient in delta_terms.items()))
    return pair | dict(sensitivity=pair["area"] / reporting["area"], prior_component=0.,
                       pair_signal=pair["pair_signal"] + volume * adjustment_m, anchor_disagreement_m=disagreement_m,
                       variance=volume**2 * (snapshot_noise + motion_noise + LOCAL_ACTIVITY_M**2),
                       precision_variance=precision_variance,
                       contributions={name: coefficient * volume for name, coefficient in terms.items()},
                       available_utc=latest.isoformat())


def satellite_update(pairs, public, sigma, variance_field="variance"):
    s = np.array([r["sensitivity"] for r in pairs])
    y = np.array([r["pair_signal"] - r["prior_component"] for r in pairs]) - s * public
    terminals = np.array([r["terminal_id"] for r in pairs])
    loading = np.array([r["volume_per_m"] * TERMINAL_SIGMA_M for r in pairs])
    noise = np.diag([r[variance_field] for r in pairs]) + np.outer(loading, loading) * (terminals[:, None] == terminals)
    scenes = sorted({k for r in pairs for k in r["contributions"]})
    design = np.array([[r["contributions"].get(scene, 0.) for scene in scenes] for r in pairs])
    cov = noise + SCENE_SIGMA_M**2 * (design @ design.T)
    direction = cho_solve(cho_factor(cov), s)
    clipped = np.clip(y, -3 * np.sqrt(cov.diagonal()), 3 * np.sqrt(cov.diagonal()))
    return float(np.clip(direction @ clipped / (sigma**-2 + direction @ s), -1.25, 1.25))


def replay():
    saved = indexed("cushing_recovered_optical_forecasts")
    by_release = brackets()
    fits = native_fits(by_release)
    calendar = {}
    for stem in ("sentinel2_space_time_coverage_weeks", "holdout_optical_weeks"):
        for r in read(stem):
            calendar[r["release_date"]] = {edge: timestamp(r, f"{edge}_measurement_at_utc") for edge in EDGES} | dict(
                area=float(r["dated_reference_area_m2"]))
    seen = []
    for r, arms, _ in combined_sources():
        current = r["release_date"]
        f = saved[current]
        cutoff, known = r["cutoff"], r["known"]
        upstream_label = f["last_training_label_utc"]
        if (not cutoff < known or cutoff != timestamp(f, "cutoff_utc") or known != timestamp(f, "known_utc")
            or upstream_label and datetime.fromisoformat(upstream_label) >= cutoff):
            raise ValueError(f"Incorrect upstream timestamps on {current}")
        public, incumbent = (float(f[k]) for k in (PUBLIC, INCUMBENT))
        if public != r["public"] or len(arms[-1]) != int(f["total_recovered_tanks"]):
            raise ValueError(f"The original optical forecast uses a different cohort on {current}")
        base, revised = [], []
        sources = by_release[current]
        for pair in arms[-1]:
            tank = pair["candidate_id"]
            if tank not in sources:
                continue
            changed = matched_bracket(pair, sources[tank], cutoff, calendar[current], fits)
            # Require the original independently measured endpoints for each tank.
            if changed is None:
                continue
            volume = pair["volume_per_m"]
            original = pair | dict(contributions={pair["scene_before"]: volume, pair["scene_after"]: -volume})
            base.append(original)
            revised.append(changed)
        past = [q for q in seen if q["known"] < cutoff]
        sigma = prior_scale(past)
        delta = precision_delta = 0.
        if revised:
            reference = satellite_update(base, public, sigma)
            delta = satellite_update(revised, public, sigma) - reference
            precision_delta = satellite_update(revised, public, sigma, "precision_variance") - reference
        matched = [q for q in past if q["bracketed_roofs"]][-104:]
        beta = learned_beta(matched, "delta")
        output = r | dict(public_ensemble=public, all_recovered=incumbent, carried_scene=float(f["carried_scene"]),
                          bracket_fixed=incumbent + .5 * delta, bracket_precision=incumbent + .5 * precision_delta,
                          bracket_learned=incumbent + beta * delta, delta=delta, precision_delta=precision_delta,
                          median_anchor_disagreement_m=median(q["anchor_disagreement_m"] for q in revised) if revised else None,
                          beta=beta, bracketed_roofs=len(revised), original_roofs=len(arms[-1]),
                          latest_bracket_available_utc=max((q["available_utc"] for q in revised), default=None),
                          latest_calibration_label_utc=matched[-1]["known"].isoformat() if matched else None,
                          calibration_weeks=len(matched))
        yield output
        # The incremental coefficient is fitted against residuals of the
        # incumbent forecast that would actually have been available then.
        seen.append(output | dict(public=incumbent, error=r["actual"] - public))


def main():
    with threadpool_limits(limits=1):
        forecasts = list(replay())
        trades = list(trading(forecasts, MODELS))
    report = dict(status="exploratory_on_reviewed_history", forecasts={}, trades={})
    for phase in PHASES:
        samples = [r for r in forecasts if r["study_phase"] == phase]
        selected = [r for r in samples if r["bracketed_roofs"]]
        report["forecasts"][phase] = {}
        for scope, sample in (("all", samples), ("matched_brackets", selected)):
            result = forecast_metrics(sample, MODELS)
            result["matched_tank_dates"] = sum(q["bracketed_roofs"] for q in sample)
            for method in MODELS[3:]:
                difference = [1000 * (abs(r["actual"] - r[method]) - abs(r["actual"] - r[INCUMBENT])) for r in sample]
                result["models"][method]["delta_vs_incumbent_95_kbbl"] = block_bootstrap_mean(difference)
            report["forecasts"][phase][scope] = result
        by_method = {m: [r for r in trades if r["study_phase"] == phase and r["method"] == m] for m in ("curve", *MODELS)}
        baseline = by_method[INCUMBENT]
        report["trades"][phase] = {}
        for m, sample in by_method.items():
            if [q["release_date"] for q in baseline] != [q["release_date"] for q in sample]:
                raise ValueError(f"Different CL trading windows on {phase}/{m}")
            delta = [r["net_170_usd"] - b["net_170_usd"] for r, b in zip(sample, baseline, strict=True)]
            report["trades"][phase][m] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["gross_usd"] - 340 * abs(r["near_side"]) for r in sample),
                paired_delta_vs_incumbent_95_usd=block_bootstrap_mean(delta))
        print(phase, len(selected), {m: round(v["mae_kbbl"], 2) for m, v in report["forecasts"][phase]["all"]["models"].items()})
        print(phase, {m: v["net_170_usd"] for m, v in report["trades"][phase].items()})
    export(f"{STEM}_forecasts", [
        {k: r[k] for k in ("release_date", "week_ending", "study_phase", "actual", "original_roofs", "bracketed_roofs",
                          "delta", "precision_delta", "median_anchor_disagreement_m", "beta", "latest_bracket_available_utc",
                          "latest_calibration_label_utc", "calibration_weeks", *MODELS)}
        | dict(cutoff_utc=r["cutoff"].isoformat(), known_utc=r["known"].isoformat()) for r in forecasts])
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
