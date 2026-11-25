"""Model the common uncertainty between satellite passes and weekly inventory boundaries."""

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_shared_scene import (
    MAX_SCENE_AGE,
    PRIOR_INVENTORY_SIGMA_MBBL,
    SCENE_SIGMA_M,
    parts,
    replay as shared_replay,
    scene_design,
)

STEM = "cushing_intraweek"
SCALES = (0., 1., 2., 4.)
FIXED_MODELS = ("scene_direct", "intraweek_1", "intraweek_2", "intraweek_4")
MODELS = (PUBLIC, "carried_scene", *FIXED_MODELS, "intraweek_adaptive", "adaptive_prior", "adaptive_prior_and_timing")


def observation_clock(rows, area_fraction, week):
    total = math.fsum(r["area"] for r in rows)/area_fraction
    fractions = np.array([r["area"]/total for r in rows])
    after = np.array([r["sensitivity"] for r in rows])/fractions
    before = 1+after-np.array([r["duration"] for r in rows])
    if np.any((after <= 0) | (after > 1+1e-9) | (before < -1e-9) | (before > 1+1e-9)):
        raise ValueError(f"Invalid weekly observation fractions on {week}")
    days = (week-timedelta(days=7), week)
    return {days[0]:(np.clip(before,0,1),-fractions), days[1]:(np.clip(after,0,1),fractions)}


def preparation(row, inventory):
    observations = row["observed"]
    week = date.fromisoformat(row["week_ending"])
    sensitivity, residual, local = parts(observations, inventory)
    clocks = observation_clock(observations, row["area_fraction"], week)
    scenes = {r[k] for r in observations for k in ("scene_before", "scene_after")}
    return dict(observations=observations, sensitivity=sensitivity, residual=residual, local=local, clocks=clocks, scenes=scenes)


def time_covariance(rows):
    slices = np.cumsum([0,*[len(r["observations"]) for r in rows]])
    covariance = np.zeros((slices[-1],slices[-1]))
    for week in sorted({week for r in rows for week in r["clocks"]}):
        terms = []
        for i, row in enumerate(rows):
            if week not in row["clocks"]:
                continue
            current = row["clocks"][week]
            if isinstance(current[0], np.ndarray):
                current = (current,)
            terms.extend((slice(slices[i],slices[i+1]),time,area) for time,area in current)
        # A pair entirely inside a survey week has two signed endpoint
        # terms. Include both cross terms, even when they occupy one block.
        for i, (block_a, time_a, area_a) in enumerate(terms):
            for j, (block_b, time_b, area_b) in enumerate(terms[i:], i):
                bridge = np.minimum.outer(time_a,time_b)-np.outer(time_a,time_b)
                value = np.outer(area_a,area_b)*bridge
                covariance[block_a,block_b] += value
                if i != j:
                    covariance[block_b,block_a] += value.T
    return covariance


def estimate(current, history, *, scene_sigma_m=SCENE_SIGMA_M, robust=False):
    selected = []
    scenes = set(current["scenes"])
    weeks = set(current["clocks"])
    for past in reversed(history):
        if (scenes & past["scenes"]) or (weeks & past["clocks"].keys()):
            selected.append(past)
            weeks.update(past["clocks"])
            scenes.update(past["scenes"])
    selected.reverse()
    rows = [*selected,current]
    scene_index = {n:i for i,n in enumerate(sorted(scenes))}
    design = np.vstack([scene_design(r["observations"],scene_index) for r in rows])
    base = block_diag(*(r["local"] for r in rows))+scene_sigma_m**2*(design@design.T)
    if "endpoint_errors" in current:
        # Repeated tank/scene measurements can share their roof-level error
        # even after the image-wide shift has been accounted for.
        names = sorted({endpoint for row in rows for pair in row["endpoint_errors"] for endpoint in pair})
        index = {name:i for i,name in enumerate(names)}
        loadings = [pair for row in rows for pair in row["endpoint_errors"]]
        endpoints = np.zeros((len(loadings),len(names)))
        for i, pair in enumerate(loadings):
            for name, loading in pair.items():
                endpoints[i,index[name]] = loading
        base += endpoints@endpoints.T
    timing = time_covariance(rows)
    s = current["sensitivity"]
    split = len(base)-len(s)
    predicted = []
    for sigma in SCALES:
        joint = base+sigma**2*timing
        variance = joint[split:,split:]
        residual = current["residual"]
        if split:
            # Condition on dated optical paths for weeks with known EIA labels.
            earlier = joint[:split,:split]
            limits = 3*np.sqrt(np.concatenate([r["local"].diagonal() for r in selected]))
            y = np.clip(np.concatenate([r["residual"] for r in selected]),-limits,limits)
            cross = joint[:split,split:]
            conditioned = cho_solve(cho_factor(earlier),np.column_stack((cross,y)))
            variance = variance-cross.T@conditioned[:,:-1]
            residual = residual-cross.T@conditioned[:,-1]
        marginal = variance.diagonal()
        if robust:
            # A Student-t influence function on individual residuals. Loading
            # the diagonal preserves the shared error terms for both scenes
            # and terminals while downweighting discordant roofs.
            disagreement = np.square(residual) / marginal
            variance = variance + np.diag(marginal * np.maximum(disagreement - 1., 0.) / 5.)
        y = np.clip(residual,-3*np.sqrt(marginal),3*np.sqrt(marginal))
        direction = cho_solve(cho_factor(variance),s)
        information = float(s@direction)
        revision = float(np.clip(direction@y/(1/PRIOR_INVENTORY_SIGMA_MBBL**2+information),-1.25,1.25))
        predicted.append((revision,information,float(direction@y)))
    return predicted


def learned_weights(history):
    """Use only previously published errors to mix the candidate time scales."""
    prior = np.array([.5,.25,.15,.10])
    available = history[-78:]
    if len(available) < 32:
        return prior
    errors = np.array([q["absolute_errors"] for q in available])
    losses = np.mean(errors,axis=0)
    weights = prior*np.exp(-(losses-losses.min())/.05)
    return weights/weights.sum()


def prior_scale(history):
    # Learn the scale of the public model error without letting an isolated
    # historic miss take over the precision used for the whole hub.
    recent = history[-52:]
    residuals = np.array([q["error"] for q in recent])
    weights = np.exp2(-np.arange(len(recent)-1,-1,-1)/26)
    anchor = PRIOR_INVENTORY_SIGMA_MBBL
    variance = (16*anchor**2+weights@np.square(np.clip(residuals,-3.,3.)))/(16+weights.sum())
    return float(np.clip(math.sqrt(variance),.60,1.50))


def predict(row, measured_history, public_history):
    cutoff = row["cutoff"]
    recent = [q["prepared"] for q in measured_history if cutoff-q["cutoff"] <= MAX_SCENE_AGE]
    observed = row["observed"]
    base = row["public"]
    updates = estimate(preparation(row,base),recent) if observed else [(0.,0.,0.)]*len(SCALES)
    increments = np.array([adjust for adjust,_,_ in updates])
    weights = learned_weights(measured_history)
    scale = prior_scale(public_history)
    revised = np.clip([score/(1/scale**2+info) for _,info,score in updates],-1.25,1.25)
    return dict(public_ensemble=base, **dict(zip(FIXED_MODELS,map(float,base+increments),strict=True)),
                intraweek_adaptive=base+float(weights@increments), adaptive_prior=base+float(revised[0]),
                adaptive_prior_and_timing=base+float(weights@revised), prior_sigma_million_bbl=scale,
                **{f"weight_{n}":float(w) for n,w in zip(FIXED_MODELS,weights,strict=True)},
                **{f"information_{n}":v[1] for n,v in zip(FIXED_MODELS,updates,strict=True)})


def replay():
    seen, errors = [], []
    for _, r in shared_replay():
        cutoff = r["cutoff"]
        past = [q for q in seen if q["known"] < cutoff]
        public_past = [q for q in errors if q["known"] < cutoff]
        observed = r["observed"]
        base = r["public"]
        result = predict(r,past,public_past)
        if not math.isclose(result["scene_direct"]-base,r["carried_raw"],abs_tol=1e-10):
            raise ValueError(f"The zero-variance control does not reproduce on {r['release_date']}")
        errors.append(dict(known=r["known"],error=r["actual"]-base))
        if observed:
            if any(datetime.fromisoformat(q["available_utc"])>=cutoff for q in observed):
                raise ValueError(f"A source arrived after the forecast cutoff on {r['release_date']}")
            values = np.array([result[m] for m in FIXED_MODELS])
            seen.append(r | dict(prepared=preparation(r,r["actual"]),absolute_errors=abs(values-r["actual"])))
        yield r | result | dict(tanks=len(observed), training_weeks=len(past),
                       last_training_label_utc=public_past[-1]["known"].isoformat() if public_past else None)


def forecast_receipt(issue, *, scoring):
    from register_cushing_prerelease_forecast import digest, freeze, score

    original_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    path = original_path.with_name(f"{original_path.stem}_intraweek.json")
    original = json.loads(original_path.read_bytes())
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in ("information_cutoff_utc","registration_deadline_utc"))
    models = (PUBLIC,"scene_direct","intraweek_adaptive","adaptive_prior","adaptive_prior_and_timing")
    if not path.exists() and not scoring:
        if not cutoff <= datetime.now(timezone.utc) < deadline:
            raise ValueError("Forecast registration is outside the original information window")
        observed = [dict(r,area=r["volume_per_m"]*BARREL_M3*1e6) for r in original["measured_pairs"]]
        if len(observed)!=original["measured_roofs"] or any(datetime.fromisoformat(r["source_available_utc"])>=cutoff for r in observed):
            raise ValueError("The frozen satellite cohort differs from the original forecast")
        current = dict(cutoff=cutoff, week_ending=original["week_ending"], public=original["public_ensemble_kbbl"]/1000,
                       observed=observed, area_fraction=original["measured_roof_area_fraction"])
        with threadpool_limits(limits=1):
            history = [r for r in replay() if r["known"] < cutoff]
            measured = [r | dict(prepared=preparation(r,r["actual"]),absolute_errors=np.abs([r[m]-r["actual"] for m in FIXED_MODELS]))
                        for r in history if r["tanks"]]
            all_errors = [dict(error=r["actual"]-r[PUBLIC]) for r in history]
            forecast = predict(current,measured,all_errors)
        now = datetime.now(timezone.utc)
        if not cutoff<=now<deadline or not history:
            raise ValueError("The receipt cannot use this training history or registration clock")
        record = dict(release_date=issue, week_ending=current["week_ending"], information_cutoff_utc=cutoff.isoformat(),
                      registration_deadline_utc=deadline.isoformat(), recorded_at_utc=now.isoformat(), measured_roofs=len(observed),
                      training_label_utc=history[-1]["known"].isoformat(), calibration_public_weeks=len(history[-52:]),
                      calibration_optical_weeks=len(measured[-78:]), prior_sigma_million_bbl=forecast["prior_sigma_million_bbl"],
                      estimates_kbbl={m:1000*forecast[m] for m in models}, model_sha256=digest(ROOT/"scripts/fit_cushing_intraweek.py"),
                      original_forecast_sha256=digest(original_path), history_sha256=digest(DERIVED/f"{STEM}_forecasts.csv"))
        freeze(path,record)
        return record
    record = json.loads(path.read_bytes())
    latest, stamp = (datetime.fromisoformat(record[k]) for k in ("training_label_utc","recorded_at_utc"))
    if (not latest<cutoff<=stamp<deadline or record["original_forecast_sha256"]!=digest(original_path)
        or any(record[k]!=original[k] for k in ("release_date","week_ending","information_cutoff_utc","registration_deadline_utc"))):
        raise ValueError("The intraweek receipt does not match its input clocks or original forecast")
    if not scoring:
        return record
    outcome = score(original_path,original)
    result = dict(release_date=issue,status=outcome["status"],forecast_sha256=digest(path))
    if outcome["status"]=="scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual,first_release_sha256=outcome["first_release_sha256"],
                      absolute_error_kbbl={m:abs(actual-value) for m,value in record["estimates_kbbl"].items()})
    write(path.with_name(f"{path.stem}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--register-release",type=date.fromisoformat)
    command.add_argument("--score-release",type=date.fromisoformat)
    args = parser.parse_args()
    release = args.register_release or args.score_release
    if release:
        print(json.dumps(forecast_receipt(release.isoformat(),scoring=args.score_release is not None),indent=2,allow_nan=False))
        return
    with threadpool_limits(limits=1):
        forecasts = list(replay())
        trades = list(trading(forecasts,MODELS))
    report = dict(status="historical_walk_forward_on_reviewed_period", forecasts={}, trading={})
    for phase in PHASES:
        group = [r for r in forecasts if r["study_phase"]==phase]
        report["forecasts"][phase] = {}
        for scope, sample in (("all",group),("measured",[r for r in group if r["tanks"]])):
            metrics = forecast_metrics(sample,MODELS)
            for m in MODELS:
                deltas = [1000*(abs(r["actual"]-r[m])-abs(r["actual"]-r["scene_direct"])) for r in sample]
                metrics["models"][m]["paired_delta_vs_scene_direct_95_kbbl"] = block_bootstrap_mean(deltas)
            report["forecasts"][phase][scope] = metrics
        base = [r for r in trades if r["study_phase"]==phase and r["method"]==PUBLIC]
        comparison = report["trading"][phase] = {}
        for method in ("curve",*MODELS):
            test = [r for r in trades if r["study_phase"]==phase and r["method"]==method]
            if [r["release_date"] for r in test] != [r["release_date"] for r in base]:
                raise ValueError(f"Different market calendar for {phase}/{method}")
            deltas = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(test,base,strict=True)]
            comparison[method] = robustness(test) | daily_risk(test) | dict(paired_net_170_vs_public_95_usd=block_bootstrap_mean(deltas),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in test))
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in report["forecasts"][phase]["measured"]["models"].items()})
        print(phase,{m:(d["trades"],d["net_170_usd"],d["daily_drawdown_usd"]) for m,d in comparison.items()})
    fields = ("release_date","week_ending","study_phase","actual","tanks","training_weeks","last_training_label_utc",
              "prior_sigma_million_bbl",*MODELS,
              *(f"{key}_{m}" for key in ("weight","information") for m in FIXED_MODELS))
    export(f"{STEM}_forecasts",[{k:r[k] for k in fields} | dict(cutoff_utc=r["cutoff"].isoformat(),known_utc=r["known"].isoformat())
                                for r in forecasts])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
