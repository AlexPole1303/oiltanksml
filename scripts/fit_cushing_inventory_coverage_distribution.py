"""Calibrate inventory uncertainty by satellite availability using prior residuals."""

import json
from datetime import datetime

import numpy as np

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_robust_stack import indexed

STEM = "cushing_inventory_coverage_distribution"
HYBRID = "basis_when_unmeasured"
METHODS = ("public", "hybrid", "coverage_calibrated", "coverage_median_shrunk")
MAX_HISTORY = 104
MIN_HISTORY = 52


def quantile(errors, weights, levels=(.1,.5,.9)):
    if len(errors)!=len(weights) or not len(errors) or not np.all(weights>0):
        raise ValueError("The calibration sample requires positive weights")
    order = np.argsort(errors,kind="stable")
    errors,weights = errors[order],weights[order]/weights.sum()
    cumulative = np.cumsum(weights)-.5*weights
    return np.interp(levels,cumulative,errors),errors,weights


def evaluate(errors, weights, point, target, *, shrink_median=False):
    values, ordered, mass = quantile(errors,weights)
    if shrink_median:
        # A point correction toward the local median uses a small shift.
        # Shift the entire predictive distribution consistently.
        offset = .25*float(np.clip(values[1],-.4,.4))
        point += offset
    lo,_,hi = point+values
    support = point+ordered
    before = np.cumsum(mass)-mass
    dispersion = np.sum(mass*support*(2*before+mass-1))
    return dict(point=point,lo=float(lo),hi=float(hi),crps=float(mass@abs(support-target)-dispersion),
        percentile=float(mass@(support<target)+.5*(mass@(support==target))))


def replay():
    raw = indexed("cushing_public_state_forecasts")
    history = []
    for issue,q in sorted(raw.items()):
        cutoff,known = (datetime.fromisoformat(q[k]) for k in ("cutoff","known"))
        if not cutoff<known or (q["latest_training_label_utc"]
              and datetime.fromisoformat(q["latest_training_label_utc"])>=cutoff):
            raise ValueError(f"Inconsistent inventory model clocks on {issue}")
        actual,public,hybrid = (float(q[k]) for k in ("actual",PUBLIC,HYBRID))
        measured = q["measured"]=="True"
        train = [r for r in history if r["known"]<cutoff][-MAX_HISTORY:]
        history.append(dict(known=known,cutoff=cutoff,measured=measured,public_error=actual-public,hybrid_error=actual-hybrid))
        if len(train)<MIN_HISTORY:
            continue
        public_error,hybrid_error = (np.array([r[f"{k}_error"] for r in train]) for k in ("public","hybrid"))
        # Partially pool with both coverage groups to limit empty-coverage
        # calibration noise. Recency has a 78-week half-life.
        age = np.array([(cutoff-r["cutoff"]).total_seconds()/604800 for r in train])
        matches = np.array([r["measured"]==measured for r in train])
        local = np.exp2(-age/78)*np.where(matches,1.,.5)
        uniform = np.ones(len(train))
        values = (evaluate(public_error,uniform,public,actual),
                  evaluate(hybrid_error,uniform,hybrid,actual),
                  evaluate(hybrid_error,local,hybrid,actual),
                  evaluate(hybrid_error,local,hybrid,actual,shrink_median=True))
        yield dict(release_date=issue,week_ending=q["week_ending"],study_phase=q["study_phase"],cutoff_utc=cutoff.isoformat(),
            known_utc=known.isoformat(),actual=actual,measured=measured,calibration_weeks=len(train),
            same_coverage_weeks=int(matches.sum()),latest_training_label_utc=train[-1]["known"].isoformat(),
            **{f"{m}_{k}":v for m,prediction in zip(METHODS,values,strict=True) for k,v in prediction.items()})


def summarize(group):
    report = {}
    for m in METHODS:
        target = np.array([r["actual"] for r in group])
        point,lo,hi,crps = (np.array([r[f"{m}_{metric}"] for r in group]) for metric in ("point","lo","hi","crps"))
        base = np.array([r["hybrid_crps"] for r in group])
        report[m] = dict(weeks=len(group),mae_kbbl=float(np.mean(abs(point-target))*1000),
            mean_crps_kbbl=float(np.mean(crps)*1000), coverage_count=int(np.count_nonzero((lo<=target)&(target<=hi))),
            interval_mean_width_kbbl=float(np.mean(hi-lo)*1000),
            interval_score_kbbl=float(np.mean(hi-lo+10*(np.maximum(lo-target,0)+np.maximum(target-hi,0)))*1000),
            paired_crps_delta_vs_hybrid_95_kbbl=block_bootstrap_mean(1000*(crps-base)))
    return report


def main():
    rows = list(replay())
    report = dict(status="exploratory_on_reviewed_history", nominal_interval=.8, phases={})
    for phase in PHASES:
        group = [r for r in rows if r["study_phase"]==phase]
        report["phases"][phase] = {scope:summarize(sample) for scope,sample in (
            ("all",group),("measured",[r for r in group if r["measured"]]),
            ("unmeasured",[r for r in group if not r["measured"]]))}
        print(phase,{k:(v["weeks"],round(v["mae_kbbl"],2),round(v["mean_crps_kbbl"],2),v["coverage_count"])
                     for k,v in report["phases"][phase]["all"].items()})
    export(f"{STEM}_forecasts",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
