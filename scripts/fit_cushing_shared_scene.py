"""Carry correlated Sentinel-2 endpoint error across successive inventory forecasts."""

import json
import math
from collections import defaultdict
from datetime import timedelta

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC, learned_beta, market_replay, market_summary, replay as baseline_replay
from fit_cushing_robust_stack import indexed

STEM = "cushing_shared_scene"
SCENE_SIGMA_M = .60
TERMINAL_SIGMA_M = .45
PRIOR_INVENTORY_SIGMA_MBBL = .90
MAX_SCENE_AGE = timedelta(days=21)
METHODS = (PUBLIC,"joint_temporal_model","signed_scene","carried_scene","public_robust","carried_scene_robust_public")


def parts(rows, inventory):
    s = np.array([r["sensitivity"] for r in rows])
    residual = np.array([r["pair_signal"]-r["prior_component"] for r in rows])-s*inventory
    var = np.array([r["variance"] for r in rows])
    terminal = np.array([r["terminal_id"] for r in rows])
    u = np.array([r["volume_per_m"]*math.sqrt(r["duration"])*TERMINAL_SIGMA_M for r in rows])
    noise = np.diag(var) + np.outer(u,u)*(terminal[:,None]==terminal[None,:])
    return s,residual,noise


def scene_design(rows, names):
    matrix = np.zeros((len(rows),len(names)))
    for i,r in enumerate(rows):
        for field,sign in (("scene_before",1),("scene_after",-1)):
            matrix[i,names[r[field]]] = sign*r["volume_per_m"]
    return matrix


def estimate(current, baseline, training):
    if not current:
        return 0.,0.,0.,[]
    # Keep the connected component touching current endpoints. This respects
    # correlations between reused scenes even when several weeks share one chain.
    names = {r[k] for r in current for k in ("scene_before","scene_after")}
    connected = []
    for old in reversed(training):
        involved = {q[k] for q in old["observed"] for k in ("scene_before","scene_after")}
        if names & involved:
            connected.append(old)
            names.update(involved)
    by_name = {name:i for i,name in enumerate(sorted(names))}
    precision = np.eye(len(by_name))/SCENE_SIGMA_M**2
    score = np.zeros(len(by_name))
    for old in reversed(connected):
        _,residual,noise = parts(old["observed"], old["actual"])
        residual = np.clip(residual,-3*np.sqrt(noise.diagonal()),3*np.sqrt(noise.diagonal()))
        design = scene_design(old["observed"],by_name)
        update = cho_solve(cho_factor(noise),np.column_stack((design,residual)))
        precision += design.T@update[:,:-1]
        score += design.T@update[:,-1]
    response = scene_design(current,by_name)
    backward = cho_solve(cho_factor(precision),np.column_stack((response.T,score)))
    correction = response@backward[:,-1]
    sensitivity,residual,noise = parts(current,baseline)
    full_cov = noise+response@backward[:,:-1]
    residual = np.clip(residual-correction,-3*np.sqrt(full_cov.diagonal()),3*np.sqrt(full_cov.diagonal()))
    projection = cho_solve(cho_factor(full_cov),sensitivity)
    information = float(sensitivity@projection)
    volume = float(np.clip(projection@residual/(1/PRIOR_INVENTORY_SIGMA_MBBL**2+information),-1.25,1.25))
    return volume,information,float(correction.sum()),connected


def calibrated(observed, cutoff, public, public_robust, past):
    past = [q for q in past if q["observed"] and q["known"]<cutoff]
    recent = [q for q in past if cutoff-q["cutoff"]<=MAX_SCENE_AGE]
    x,info,_,_ = estimate(observed,public,[])
    y,info_reused,scene_total,connected = estimate(observed,public,recent)
    z,_,_,_ = estimate(observed,public_robust,recent)
    if not connected and (not math.isclose(x,y,abs_tol=1e-12) or not math.isclose(info,info_reused,abs_tol=1e-12)):
        raise ValueError(f"Scene correction differs without a reused endpoint at {cutoff}")
    controls = past[-104:]
    signed_weight,carried_weight = (learned_beta(controls,key) for key in ("signed_raw","carried_raw"))
    robust_weight = learned_beta([q | dict(public=q["public_robust"]) for q in controls],"robust_raw")
    latest = max((q["known"] for q in connected),default=None)
    return dict(public_robust=public_robust,signed_scene=public+signed_weight*x,carried_scene=public+carried_weight*y,
        carried_scene_robust_public=public_robust+robust_weight*z,robust_scene_increment=robust_weight*z,
        signed_raw=x,carried_raw=y,robust_raw=z,signed_weight=signed_weight,carried_weight=carried_weight,robust_weight=robust_weight,
        scene_information=info,carried_information=info_reused,sum_scene_correction_kbbl=1000*scene_total,
        previous_roof_pairs_used=sum(len(q["observed"]) for q in connected),
        latest_available_scene_label_utc=latest.isoformat() if latest else None,
        latest_training_label_utc=controls[-1]["known"].isoformat() if controls else None)


def replay():
    robust = indexed("cushing_robust_stack_forecasts")
    seen = []
    for frozen,_,r in baseline_replay():
        observed = r["observed"]
        if r["known"]<=r["cutoff"] or int(frozen["measured_roofs"])!=len(observed):
            raise ValueError(f"Unexpected forecast chronology or coverage on {r['release_date']}")
        public_robust = float(robust[r["release_date"]]["robust_public"])
        result = calibrated(observed,r["cutoff"],r["public"],public_robust,seen)
        state = r | result
        seen.append(state)
        yield frozen | result,state


def run_market(forecasts):
    # Match the existing event model at the same price marks. The inventory surprise
    # is always based on robust_public, so use the learned correction to that forecast.
    rows = [r | dict(joint_temporal_model=r[PUBLIC]+r["robust_scene_increment"],lagged_optical_control=r["signed_scene"])
            for r in forecasts]
    names = {"observed_inventory":"observed_inventory","optical_as_feature":"scene_as_feature",
             "optical_in_surprise":"scene_in_surprise","lagged_optical_as_feature":"signed_scene_as_feature"}
    trades = list(market_replay(rows))
    trading = {p:{names[k]:v for k,v in models.items()} for p,models in market_summary(trades)}
    return [r | dict(method=names[r["method"]]) for r in trades],trading


def main():
    rows = [r for r,_ in replay()]
    forecasts = {p:{scope:forecast_stats([r for r in rows if r["study_phase"]==p and admit(r)])
                     for scope,admit in (("all",lambda r:True),("measured",lambda r:r["measured_roofs"]>0),
                                         ("reused_scene",lambda r:r["previous_roof_pairs_used"]>0))} for p in PHASES}
    trades,trading = run_market(rows)
    report = dict(status="historical_point_in_time_model_comparison",forecasts=forecasts,trading=trading)
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    for p,r in forecasts.items():
        print(p,{s:(q["releases"],{k:round(v["mae_kbbl"],1) for k,v in q["methods"].items()}) for s,q in r.items()})
        print(p,{k:(v["trades"],v["net_170_usd"]) for k,v in trading[p].items()})


def forecast_stats(rows):
    y = np.array([r["actual"] for r in rows])
    preds = {method:np.array([r[method] for r in rows]) for method in METHODS}
    errors = {k:1000*np.abs(y-p) for k,p in preds.items()}
    return dict(releases=len(rows),methods={k:dict(mae_kbbl=float(e.mean()),correct_direction=int(np.count_nonzero(preds[k]*y>0)),
            paired_95_vs_public_kbbl=block_bootstrap_mean(e-errors[PUBLIC])) for k,e in errors.items()},
        carried_minus_signed_95_kbbl=block_bootstrap_mean(errors["carried_scene"]-errors["signed_scene"]),
        carried_minus_previous_95_kbbl=block_bootstrap_mean(errors["carried_scene"]-errors["joint_temporal_model"]),
        robust_carried_minus_robust_public_95_kbbl=block_bootstrap_mean(errors["carried_scene_robust_public"]-errors["public_robust"]))


if __name__ == "__main__":
    main()
