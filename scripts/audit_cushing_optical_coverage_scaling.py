"""Estimate the marginal value of measured area on matched Cushing releases."""

import hashlib
import json
import math
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import FIXED_MODELS, predict, preparation
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed

STEM = "cushing_optical_coverage_scaling"
SALTS = ("east", "west", "north", "south")
MODELS = (PUBLIC, "carried_scene", "all_recovered", "split_consensus", "adaptive_consensus")


def order_key(salt, row):
    # The geometry, dates, and price labels must not influence selection.
    return hashlib.sha256(f"{salt}/{row['candidate_id']}".encode()).digest()


def forecast(rows, total_area, r, history, errors):
    fraction = math.fsum(q["area"] for q in rows)/total_area
    state = r | dict(observed=rows, area_fraction=fraction)
    eligible = [q for q in history if q["known"] < r["cutoff"]]
    result = predict(state, eligible, errors)
    if rows:
        history.append(dict(cutoff=r["cutoff"], known=r["known"], prepared=preparation(state, r["actual"]),
                            absolute_errors=np.abs([result[m]-r["actual"] for m in FIXED_MODELS])))
    info = sum(result[f"weight_{m}"]*result[f"information_{m}"] for m in FIXED_MODELS)
    latest = eligible[-1]["known"].isoformat() if eligible else None
    return result["adaptive_prior_and_timing"], fraction, float(info), latest


def replay():
    geometries = {(q["release_date"],q["week_ending"]):q for stem in ("sentinel2_space_time_coverage_weeks", "holdout_optical_weeks")
                  for q in read(stem)}
    original = indexed("cushing_recovered_optical_forecasts")
    histories = {name: [] for name in ("full", *(f"{salt}_{half}" for salt in SALTS for half in (0, 1)))}
    errors, evaluated = [], []
    for r, arms, _ in combined_sources():
        issue, cutoff, known = (r[k] for k in ("release_date", "cutoff", "known"))
        rows = arms[-1]
        g = geometries[issue,r["week_ending"]]
        total = float(g["dated_reference_area_m2"])
        if (not cutoff < known or cutoff != datetime.fromisoformat(g["cutoff_utc"]) or r["week_ending"] != g["week_ending"]
            or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in rows)):
            raise ValueError(f"Inconsistent measurement dates on {issue}")
        public = r["public"]
        past_errors = [q for q in errors if q["known"] < cutoff]
        full, area, info, fitted = forecast(rows, total, r, histories["full"], past_errors)
        if not math.isclose(full, float(original[issue]["all_recovered"]), abs_tol=1e-9):
            raise ValueError(f"The full-footprint forecast differs on {issue}")
        group = []
        for salt in SALTS:
            ordered = sorted(rows, key=lambda q: order_key(salt, q))
            for i in (0, 1):
                subset = ordered[i::2]
                estimate, subset_area, subset_info, latest = forecast(subset, total, r, histories[f"{salt}_{i}"], past_errors)
                group.append(dict(split=f"{salt}_{i}", roofs=len(subset), area_fraction=subset_area, information=subset_info,
                                  latest_optical_calibration_utc=latest, forecast_mbbl=estimate))
        # Split fits retain the actual count and shared-scene structure in
        # each group. If fewer than two roofs exist, use the original update.
        consensus = float(np.median([q["forecast_mbbl"] for q in group])) if len(rows) >= 2 else full
        available = [q for q in evaluated if q["known"] < cutoff][-78:]
        weight = 0.
        if len(available) >= 26:
            differences = [abs(q["consensus"] - q["actual"]) - abs(q["full"] - q["actual"]) for q in available]
            # Keep softness in the choice when the past loss difference is small.
            weight = float(np.clip(-np.mean(differences)/.04, 0., .75))
        adjusted = (1-weight)*full+weight*consensus
        if len(rows) >= 2:
            evaluated.append(dict(known=known, consensus=consensus, full=full, actual=r["actual"]))
        errors.append(dict(known=known, error=r["actual"]-public))
        yield {k:r[k] for k in ("release_date", "week_ending", "study_phase", "actual")} | dict(
            source_cutoff_utc=cutoff.isoformat(), known_utc=known.isoformat(), public_mbbl=public,
            full_mbbl=full, full_area_fraction=area, full_roofs=len(rows), full_information=info,
            full_latest_optical_calibration_utc=fitted, cutoff=cutoff, known=known, public_ensemble=public,
            carried_scene=float(original[issue]["carried_scene"]), all_recovered=full, split_consensus=consensus,
            adaptive_consensus=adjusted, consensus_weight=weight,
            latest_consensus_label_utc=available[-1]["known"].isoformat() if available else None), group


def evaluate(sample):
    n = len(sample)
    if not n:
        return dict(weeks=0)
    public = np.array([1000*abs(q["public_mbbl"]-q["actual"]) for q,_ in sample])
    full = np.array([1000*abs(q["full_mbbl"]-q["actual"]) for q,_ in sample])
    splits = np.array([[1000*abs(v["forecast_mbbl"]-q["actual"]) for v in subs] for q,subs in sample])
    half = splits.mean(axis=1)
    levels = np.array([[v["forecast_mbbl"] for v in subs] for _,subs in sample])
    return dict(weeks=n, public_mae_kbbl=float(public.mean()), full_mae_kbbl=float(full.mean()),
        mean_half_sample_mae_kbbl=float(half.mean()), full_minus_half_mae_95_kbbl=block_bootstrap_mean(full-half),
        median_full_area_pct=100*float(np.median([q["full_area_fraction"] for q,_ in sample])),
        median_half_area_pct=100*float(np.median([v["area_fraction"] for _,subs in sample for v in subs])),
        median_full_information=float(np.median([q["full_information"] for q,_ in sample])),
        median_half_information=float(np.median([v["information"] for _,subs in sample for v in subs])),
        median_between_halves_forecast_difference_kbbl=1000*float(np.median(abs(levels[:,::2]-levels[:,1::2]))))


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        trades = list(trading([r for r,_ in rows], MODELS))
    report = dict(status="retrospective_matched_coverage_sensitivity", phases={}, trading={}, split_pairs=len(SALTS))
    for phase in PHASES:
        cohort = [r for r,_ in rows if r["study_phase"] == phase]
        measured = [(r, sub) for r,sub in rows if r["study_phase"] == phase and r["full_roofs"] >= 2]
        result = dict(matched=evaluate(measured), larger_footprint=evaluate([(r, sub) for r,sub in measured
                      if r["full_area_fraction"] >= .10]), all_forecasts=forecast_metrics(cohort, MODELS))
        base = [q for q in trades if q["study_phase"] == phase and q["method"] == "all_recovered"]
        report["trading"][phase] = {}
        for method in ("curve", *MODELS):
            group = [q for q in trades if q["study_phase"] == phase and q["method"] == method]
            if [q["release_date"] for q in base] != [q["release_date"] for q in group]:
                raise ValueError(f"Unpaired CL forecasts on {phase}/{method}")
            delta = [q["net_170_usd"]-b["net_170_usd"] for q,b in zip(group,base,strict=True)]
            report["trading"][phase][method] = robustness(group) | daily_risk(group) | dict(
                delta_vs_full_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(q["gross_usd"]-340*abs(q["near_side"]) for q in group))
        report["phases"][phase] = result
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in result["all_forecasts"]["models"].items()},
              {m:v["net_170_usd"] for m,v in report["trading"][phase].items()})
    export(f"{STEM}_forecasts", [{k:v for k,v in r.items() if k not in ("cutoff","known")} | q for r,group in rows for q in group])
    export(f"{STEM}_trades", trades)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
