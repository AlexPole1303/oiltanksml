"""Model zero-sum terminal movements jointly across overlapping survey weeks."""

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import SCALES, estimate, learned_weights, prior_scale, time_covariance
from fit_cushing_joint_late_pairs import source_rows
from fit_cushing_late_pair_recovery import COMBINED
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import trading as base_trades
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, prepared
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, scene_design
from fit_cushing_terminal_balance import DEFAULT, mapped_areas, variance_fit
from fit_cushing_volatility_neighbors import STEM as STATE

STEM = "cushing_terminal_weekly_transfers"
REGIONAL = "market_error_median"
FIXED = "terminal_week_fixed"
LEARNED = "terminal_week_learned"
LIMITED = "terminal_week_limited_coverage"
LIMITED_MARKET = "terminal_week_limited_and_market"
MODELS = (PUBLIC, SUPPLEMENT, COMBINED, REGIONAL, FIXED, LEARNED, "terminal_week_and_market", LIMITED, LIMITED_MARKET)


def terminal_design(row, roster, terminal_names):
    """Return roof exposures to latent weekly terminal deviations in million barrels."""
    area = np.array([roster[name][0] if name in roster else 0. for name in terminal_names])
    total = area.sum()
    if (not math.isclose(total, row["hub_area"], rel_tol=1e-10) or np.any(area<0)
            or any(stamp != row["cutoff"] for _,stamp in roster.values())):
        raise ValueError(f"Inconsistent dated terminal area on {row['release_date']}")
    shares = area/total
    balanced = (np.eye(len(shares))-shares[:,None])*np.sqrt(shares)
    if not np.allclose(balanced.sum(axis=0),0.,atol=1e-12):
        raise ValueError("The terminal flow model must conserve the hub total")
    sample = row["observed"]
    terminals = np.array([terminal_names.index(q["terminal_id"]) for q in sample])
    tank_share = np.array([q["area"] for q in sample])/total
    if np.any(shares[terminals]<=0) or np.any(tank_share<=0):
        raise ValueError(f"A measured roof is outside the dated terminal roster on {row['release_date']}")
    observed_week = np.array([q["sensitivity"] for q in sample])/tank_share
    future = np.array([q.get("post_fraction",0.) for q in sample])
    previous = np.array([q["duration"] for q in sample])-observed_week-future
    fractions = np.array((previous,observed_week,future))
    if not np.isfinite(fractions).all() or np.any(fractions< -1e-9) or np.any(fractions>1.+1e-9):
        raise ValueError(f"Invalid terminal week exposure on {row['release_date']}")
    week = date.fromisoformat(row["week_ending"])
    dates = (week-timedelta(days=7),week,week+timedelta(days=7))
    loading = balanced[terminals]*(tank_share/shares[terminals])[:,None]
    return {day:np.clip(fraction,0.,1.)[:,None]*loading for day,fraction in zip(dates,fractions,strict=True)
            if np.any(fraction>1e-9)}


def with_transfers(model, loads, names, scale):
    keys = [(day.isoformat(),name) for day in sorted(loads) for name in names]
    matrix = np.column_stack([loads[day] for day in sorted(loads)])*scale
    return model | dict(endpoint_errors=[{k:float(v) for k,v in zip(keys,r,strict=True) if v}
                                        for r in matrix])


def optical_fit(model, recent, scale, weights):
    fit = estimate(model,recent)
    increment = np.clip([score/(scale**-2+info) for _,info,score in fit],-1.25,1.25)
    return float(weights@increment),float(weights@[q[1] for q in fit]),np.array([q[0] for q in fit])


def excluded_terminals(row, measured, scenes, transfer, sigma, timing_variance):
    """Score satellite observations hidden one terminal at a time, at known hub change."""
    terminals = np.asarray([q["terminal_id"] for q in row["observed"]])
    if len(set(terminals))<2:
        return
    noise = measured["local"]+DEFAULT[1]*(scenes@scenes.T)+timing_variance*time_covariance([measured])
    outcome = measured["residual"]
    for name in sorted(set(terminals)):
        masked = terminals==name
        observed = ~masked
        for method,cov in (("without_transfer",noise),(FIXED,noise+DEFAULT[-1]*transfer),(LEARNED,noise+sigma**2*transfer)):
            keep = cov[np.ix_(observed,observed)]
            cross = cov[np.ix_(observed,masked)]
            factor = cho_factor(keep)
            inverse = cho_solve(factor,np.column_stack((outcome[observed],cross)))
            error = outcome[masked]-cross.T@inverse[:,0]
            predictive = cov[np.ix_(masked,masked)]-cross.T@inverse[:,1:]
            variance = predictive.diagonal()
            scale = np.sqrt(variance)
            # Compare observed roof changes while counting every held-out
            # terminal once in each evaluated release.
            loss = .5*(np.log(2*np.pi*variance)+np.square(error)/variance)
            yield {k:row[k] for k in ("release_date","week_ending","study_phase")} | dict(
                terminal_id=name,held_out_roofs=int(masked.sum()),method=method,
                mean_abs_error_kbbl=1000*float(abs(error).mean()),mean_nll=float(loss.mean()),
                interval_80_coverage=float(np.mean(abs(error)<=1.281551565545*scale)))


def forecasts(diagnostic):
    rosters = mapped_areas()
    names = sorted({name for roster in rosters.values() for name in roster})
    state = indexed(f"{STATE}_forecasts")
    seen, errors = [], []
    for row,_,_,_ in source_rows():
        issue, week, cutoff, known, actual = (row[k] for k in ("release_date","week_ending","cutoff","known","actual"))
        reference = state[issue]
        if (datetime.fromisoformat(reference["cutoff"])!=cutoff or datetime.fromisoformat(reference["known"])!=known
                or float(reference["actual"])!=actual or reference["week_ending"]!=week or row[COMBINED]!=float(reference[COMBINED])
                or any(reference[k] and datetime.fromisoformat(reference[k])>=cutoff for k in (
                    "latest_inventory_training_utc","latest_state_error_known_utc"))):
            raise ValueError(f"Different inventory information on {issue}")
        public, base, market_base = row[PUBLIC],row[COMBINED],float(reference[REGIONAL])
        prior_errors = [p for p in errors if p["known"]<cutoff]
        available = [p for p in seen if p["known"]<cutoff]
        recent = [p for p in available if cutoff-p["cutoff"]<=MAX_SCENE_AGE]
        variance_sample = available[-104:]
        learned_sigma = math.sqrt(DEFAULT[-1])
        ordinary = fixed = learned = 0.
        information = [0.]*3
        if row["observed"]:
            loads = terminal_design(row,rosters[issue,week],names)
            span = any("post_fraction" in q for q in row["observed"])
            current, measured = (prepared(row,target,span) for target in (public,actual))
            sigma = prior_scale(prior_errors)
            weights = learned_weights(available)
            ordinary,information[0],raw = optical_fit(current,[p["prepared"] for p in recent],sigma,weights)
            if not math.isclose(public+ordinary,row[SUPPLEMENT],abs_tol=1e-9):
                raise ValueError(f"Unbalanced optical reference changed on {issue}")
            variance = variance_fit(variance_sample,cutoff)
            learned_sigma = math.sqrt(variance[-1])
            for n,scale in enumerate((math.sqrt(DEFAULT[-1]),learned_sigma),1):
                here = with_transfers(current,loads,names,scale)
                past = [with_transfers(p["prepared"],p["loads"],names,scale) for p in recent]
                revision,info,_ = optical_fit(here,past,sigma,weights)
                information[n] = info
                if n==1:
                    fixed = revision
                else:
                    learned = revision
            scenes = scene_design(row["observed"],{name:i for i,name in enumerate(sorted(measured["scenes"]))})
            covariance = sum(v@v.T for v in loads.values())
            components = np.stack((measured["local"],scenes@scenes.T,covariance))
            cap = 4*np.sqrt(np.diag(np.einsum("i,ijk->jk",DEFAULT,components)))
            diagnostic.extend(excluded_terminals(row,measured,scenes,covariance,learned_sigma,
                                                  float(weights@np.square(SCALES))))
            seen.append(dict(known=known,cutoff=cutoff,prepared=measured,loads=loads,observed=row["observed"],
                             absolute_errors=np.abs(public+raw-actual),precision_inputs=(components,np.clip(measured["residual"],-cap,cap))))
        errors.append(dict(known=known,error=actual-public))
        terminal_count = len({q["terminal_id"] for q in row["observed"]})
        revision = learned-ordinary
        limited_revision = revision if 0<terminal_count<=3 else 0.
        yield {k:row[k] for k in ("release_date","week_ending","study_phase","cutoff","known","actual",PUBLIC,SUPPLEMENT,COMBINED)} | {
            REGIONAL:market_base,FIXED:base+fixed-ordinary,LEARNED:base+learned-ordinary,
            "terminal_week_and_market":market_base+revision,
            LIMITED:base+limited_revision,LIMITED_MARKET:market_base+limited_revision} | dict(
            roofs=len(row["observed"]),terminals=terminal_count,
            latest_training_label_utc=available[-1]["known"] if available else None,
            measured_training_weeks=len(variance_sample),learned_transfer_scale_million_bbl=learned_sigma,
            learned_revision_million_bbl=revision,original_information=information[0],
            fixed_information=information[1],learned_information=information[2])


def metrics(rows):
    if not rows:
        return dict(weeks=0,models={})
    error = {k:np.array([1000*abs(r[k]-r["actual"]) for r in rows]) for k in MODELS}
    return dict(weeks=len(rows), models={k:dict(mae_kbbl=float(e.mean()),rmse_kbbl=float(np.sqrt(np.mean(e**2))),
        delta_vs_combined_95_kbbl=block_bootstrap_mean(e-error[COMBINED]),
        delta_vs_market_calibrated_95_kbbl=block_bootstrap_mean(e-error[REGIONAL])) for k,e in error.items()})


def main():
    diagnostic = []
    with threadpool_limits(limits=1):
        rows = list(forecasts(diagnostic))
        base = list(base_trades(rows,MODELS))
        trades = list(replay_conditional_market(rows,base,MODELS,references={
            PUBLIC:f"{PUBLIC}_conditional",SUPPLEMENT:f"{SUPPLEMENT}_conditional"}))
    report = dict(status="exploratory_on_reviewed_history", inventory={}, excluded_terminal={}, trading={})
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["method"]].append(r)
    for phase in PHASES:
        panel = [r for r in rows if r["study_phase"]==phase]
        report["inventory"][phase] = {scope:metrics(sample) for scope,sample in (
            ("all",panel),("optical",[r for r in panel if r["roofs"]]),
            ("one_to_three_terminals",[r for r in panel if 0<r["terminals"]<=3]))}
        excluded = {}
        for method in ("without_transfer",FIXED,LEARNED):
            sample = [r for r in diagnostic if r["study_phase"]==phase and r["method"]==method]
            excluded[method] = dict(terminal_weeks=len(sample),weeks=len({r["release_date"] for r in sample}),
                **{k:float(np.mean([r[k] for r in sample])) for k in ("mean_abs_error_kbbl","mean_nll","interval_80_coverage")})
        report["excluded_terminal"][phase] = excluded
        trading = report["trading"][phase] = {}
        control = groups[phase,COMBINED]
        for method in MODELS:
            sample = groups[phase,method]
            if [r["release_date"] for r in sample]!=[r["release_date"] for r in control]:
                raise ValueError(f"Different market cohort on {phase}/{method}")
            matched = list(zip(sample,control,strict=True))
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in matched]
            trading[method] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(q["gross_usd"]-340*abs(q["near_side"]) for q in sample),
                delta_vs_combined_per_release_95_usd=block_bootstrap_mean(delta),
                changed_entries=sum(r["near_side"]!=q["near_side"] for r,q in matched))
        print(phase,{k:round(v["mae_kbbl"],2) for k,v in report["inventory"][phase]["all"]["models"].items()},
              {k:(v["trades"],v["net_170_usd"],v["net_340_usd"]) for k,v in trading.items()},flush=True)
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    export(f"{STEM}_terminal_diagnostic",diagnostic)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
