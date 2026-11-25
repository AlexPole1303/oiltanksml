"""Use heavy-tailed public forecast errors as the prior for optical inventory updates."""

import json
from datetime import datetime

import numpy as np
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import estimate, preparation, prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, replay as shared_replay

STEM = "cushing_inventory_tail_prior"
MODELS = (PUBLIC,"carried_scene","adaptive_gaussian","student_t","empirical","empirical_symmetric")
GRID = np.linspace(-8,8,1601)


def priors(history, sigma):
    # Student t with 5 degrees of freedom, scaled to the same variance as
    # the Gaussian prior. Decreasing its peak scale frees the far tails.
    student = -3*np.log1p(GRID**2/(3*sigma**2))
    student -= logsumexp(student)
    past = history[-104:]
    if len(past)<32:
        return student,student,student
    values = np.clip(np.array([r["error"] for r in past]),-5,5)
    log_weights = -np.log(2)*np.arange(len(past)-1,-1,-1)/52
    # Smooth the empirical support and keep a fixed share of the broad prior.
    # The symmetric control isolates directional bias in the previous errors.
    result = [student]
    for symmetric in (False,True):
        centres = np.r_[values,-values] if symmetric else values
        weights = np.tile(log_weights,2) if symmetric else log_weights
        width = max(.30,.38*sigma)
        kernel = -.5*((GRID[:,None]-centres)/width)**2+weights
        density = logsumexp(kernel,axis=1)
        density -= logsumexp(density)
        result.append(np.logaddexp(np.log(.25)+student,np.log(.75)+density))
    return tuple(result)


def quantiles(log_density):
    masses = np.exp(log_density-logsumexp(log_density))
    return np.interp((.10,.50,.90),np.cumsum(masses)-.5*masses,GRID)


def predict(public, history, info, score):
    sigma = prior_scale(history)
    gaussian = float(np.clip(score/(sigma**-2+info),-1.25,1.25))
    likelihood = score*GRID-.5*info*GRID**2
    result = {}
    for name,density in zip(MODELS[3:],priors(history,sigma),strict=True):
        # A posterior median is the optimal point under absolute error.
        # Subtracting the original prior median isolates the impact of optics.
        offset = quantiles(density)[1]
        low, centre, high = quantiles(density+likelihood)-offset
        result[name] = public+float(np.clip(centre,-1.25,1.25))
        result[f"{name}_p10"] = public+float(low)
        result[f"{name}_p90"] = public+float(high)
    return result | dict(adaptive_gaussian=public+gaussian,prior_sigma_million_bbl=sigma)


def replay():
    past, optical = [], []
    for _,row in shared_replay():
        cutoff, observed, public = (row[k] for k in ("cutoff","observed","public"))
        training = [r for r in past if r["known"] < cutoff]
        recent = [r["prepared"] for r in optical if r["known"] < cutoff and cutoff-r["cutoff"]<=MAX_SCENE_AGE]
        info = score = 0.
        if observed:
            if any(datetime.fromisoformat(o["available_utc"])>=cutoff for o in observed):
                raise ValueError(f"Optical input arrives late on {row['release_date']}")
            _,info,score = estimate(preparation(row,public),recent)[0]
            optical.append(row | dict(prepared=preparation(row,row["actual"])))
        prediction = predict(public,training,info,score)
        past.append(dict(known=row["known"],error=row["actual"]-public))
        yield row | prediction | dict(public_ensemble=public,tanks=len(observed), optical_information=info,optical_score=score,
            last_training_label_utc=training[-1]["known"].isoformat() if training else None)


def metrics(rows):
    report = forecast_metrics(rows,MODELS)
    for m in MODELS[3:]:
        delta = [1000*(abs(r["actual"]-r[m])-abs(r["actual"]-r["adaptive_gaussian"])) for r in rows]
        ranges = [(r[f"{m}_p10"],r[f"{m}_p90"],r["actual"]) for r in rows]
        report["models"][m].update(paired_delta_vs_gaussian_95_kbbl=block_bootstrap_mean(delta),
            interval_covered=sum(lo<=y<=hi for lo,hi,y in ranges),
            interval_width_kbbl=1000*np.mean([hi-lo for lo,hi,_ in ranges]),
            interval_score_kbbl=1000*np.mean([hi-lo+10*(max(lo-y,0)+max(y-hi,0)) for lo,hi,y in ranges]))
    return report


def highest_errors(rows):
    residual = sorted(rows,key=lambda r:abs(r["actual"]-r[PUBLIC]),reverse=True)
    worst = residual[:max(1,round(.1*len(residual)))]
    baseline = sum(abs(r["actual"]-r[PUBLIC]) for r in residual)
    return dict(releases=len(worst), measured_weeks=sum(r["tanks"]>0 for r in worst),
                share_of_public_absolute_error=sum(abs(r["actual"]-r[PUBLIC]) for r in worst)/baseline,
                mae_kbbl={k:1000*np.mean([abs(r["actual"]-r[k]) for r in worst]) for k in MODELS},
                release_dates=[r["release_date"] for r in worst])


def main():
    with threadpool_limits(limits=1):
        forecasts = list(replay())
        trades = list(trading(forecasts,MODELS))
    report = dict(status="historical_walk_forward_on_reviewed_period", forecasts={},trading={})
    for phase in PHASES:
        rows = [r for r in forecasts if r["study_phase"]==phase]
        measured = [r for r in rows if r["tanks"]]
        report["forecasts"][phase] = dict(all=metrics(rows),measured=metrics(measured),largest_public_errors=highest_errors(rows))
        result = report["trading"][phase] = {}
        base = [r for r in trades if r["study_phase"]==phase and r["method"]==PUBLIC]
        for method in ("curve",*MODELS):
            group = [r for r in trades if r["study_phase"]==phase and r["method"]==method]
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Trading dates differ for {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(group,base,strict=True)]
            result[method] = robustness(group) | daily_risk(group) | dict(paired_net_vs_public_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group))
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in report["forecasts"][phase]["measured"]["models"].items()})
        print(phase,{m:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in result.items()})
    coverage = indexed("holdout_optical_weeks")
    widened = indexed("sentinel2_wider_registration_gap_weeks")
    ranked = report["forecasts"]["holdout"]["largest_public_errors"]["release_dates"]
    by_issue = {r["release_date"]:r for r in forecasts}
    gaps = []
    for order,issue in enumerate(ranked,1):
        r, evidence = by_issue[issue], coverage[issue]
        if (r["week_ending"]!=evidence["week_ending"] or r["cutoff"].isoformat()!=evidence["cutoff_utc"]
            or r["tanks"]!=int(evidence["baseline_b04_tanks"])):
            raise ValueError(f"Different optical coverage on {issue}")
        wider = widened.get(issue)
        if wider is not None and any(evidence[k]!=wider[k] for k in ("week_ending","cutoff_utc")):
            raise ValueError(f"Wider registration uses different dates on {issue}")
        gaps.append(dict(rank=order,release_date=issue,week_ending=r["week_ending"],target_kbbl=1000*r["actual"],
            public_error_kbbl=1000*abs(r["actual"]-r[PUBLIC]),measured_tanks=r["tanks"],
            **{k:evidence[k] for k in ("b04_status","dated_bracket_candidate_tanks","landsat_15m_paired_tanks",
                                       "landsat_30m_context_only_tanks","sentinel1_slc_common_relative_track")},
            wider_registration_input_pairs=wider["input_pairs"] if wider else None,
            wider_registration_checked_pairs=wider["checked_candidates"] if wider else None,
            wider_registration_resolved_tanks=wider["resolved_tanks"] if wider else None))
    export(f"{STEM}_largest_errors",gaps)
    fields = ("release_date","study_phase","week_ending","actual","tanks","prior_sigma_million_bbl","optical_information","optical_score",
              "last_training_label_utc",*MODELS,*(f"{m}_p{q}" for m in MODELS[3:] for q in (10,90)))
    export(f"{STEM}_forecasts",[{k:r[k] for k in fields} | dict(cutoff_utc=r["cutoff"].isoformat(),known_utc=r["known"].isoformat())
                                for r in forecasts])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
