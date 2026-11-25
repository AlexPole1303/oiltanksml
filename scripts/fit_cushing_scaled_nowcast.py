"""Forecast weekly Cushing inventory changes with time-varying target scale."""

import json
import math
from datetime import datetime

import numpy as np
from scipy.linalg import solve

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_scaled_nowcast"
PUBLIC = ("public_ensemble", "robust_public", "flow_long", "public_plus_monthly", "prior")
METHODS = ("public_ensemble", "dynamic_public", "dynamic_optical", "dynamic_blend")
PHASES = ("development", "validation", "holdout")


def inputs():
    source = indexed("cushing_robust_stack_forecasts")
    interval = indexed("cushing_interval_nowcast_forecasts")
    flow = indexed("cushing_flow_extensions_forecasts")
    for issue,f in sorted(source.items()):
        a,b = interval[issue],flow[issue]
        cutoff,known = [datetime.fromisoformat(f[k]) for k in ("forecast_cutoff_utc","target_known_by_utc")]
        for r in (a,b):
            if (r["forecast_cutoff_utc"],r["target_known_by_utc"],r["week_ending"]) != (
                    f["forecast_cutoff_utc"],f["target_known_by_utc"],f["week_ending"]):
                raise ValueError(f"Nonmatching forecast vintage on {issue}")
            clock = r["latest_training_label_utc"]
            if clock and datetime.fromisoformat(clock)>=cutoff:
                raise ValueError(f"Unpublished training label on {issue}")
        if not cutoff < known or not float(f["target"]) == float(a["actual"]) == float(b["target"]):
            raise ValueError(f"Invalid forecast target on {issue}")
        yield dict(release_date=issue,study_phase=f["study_phase"],week_ending=f["week_ending"],cutoff=cutoff,known=known,
                   actual=float(f["target"]),public_ensemble=float(f["public_ensemble"]),
                   forecasts=np.asarray([float((f | b)[k]) for k in PUBLIC]),
                   optical=float(a["interval_signal"])-float(a["expected_share"]),
                   covered=float(a["observed_weight"]),measured=a["measured"]=="True")


def target_scale(sample):
    weekly_abs = np.abs([r["actual"] for r in sample[-20:]])
    local = np.mean(np.minimum(weekly_abs,3.)) if len(weekly_abs) else 1.0
    return float(np.clip(local,0.45,2.5))


def robust_residual(history, current, key, *, strength):
    x = np.stack([r[key] for r in history])
    rms = np.maximum(np.sqrt(np.mean(x*x,axis=0)),.1)
    x = np.clip(x/rms,-4,4)
    vector = np.clip(current[key]/rms,-4,4)
    y = np.array([(r["actual"]-r["public_ensemble"])/r["scale"] for r in history])
    age = np.asarray([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in history])
    weights = np.exp2(-age/65)
    weights /= weights.sum()
    penalty = np.eye(x.shape[1])*strength
    w = weights.copy()
    for _ in range(6):
        beta = solve(x.T@(w[:,None]*x)+penalty,x.T@(w*y),assume_a="pos")
        w = weights/np.hypot(1,(y-x@beta)/.6)
    return float(np.clip(vector@beta*current["scale"],-.9,.9))


def replay():
    history = []
    for r in inputs():
        earlier = [q for q in history if q["known"]<r["cutoff"]][-156:]
        r["scale"] = target_scale(earlier)
        scale,public = r["scale"],r["public_ensemble"]
        variability = math.log(scale)
        z = public/scale
        residuals = (r["forecasts"][1:]-public)/scale
        r["x_public"] = np.r_[z,z*variability,residuals]
        signal = r["optical"]/scale
        r["x_optical"] = np.r_[r["x_public"],signal,signal*math.sqrt(r["covered"]),signal*variability]
        optical_train = [q for q in earlier if q["measured"]]
        correction_public = robust_residual(earlier,r,"x_public",strength=.4) if len(earlier)>=52 else 0.
        if r["measured"] and len(optical_train)>=32 and len(earlier)>=52:
            optical = robust_residual(optical_train,r,"x_optical",strength=.7)
            optical -= robust_residual(optical_train,r,"x_public",strength=.7)
            gain = float(np.clip(optical,-.45,.45))
        else:
            gain = 0.
        r["dynamic_public"] = public+correction_public
        r["dynamic_optical"] = r["dynamic_public"]+gain
        recent = earlier[-52:]
        if len(recent)>=26:
            errors = np.asarray([[abs(q[m]-q["actual"]) for m in METHODS[:-1]] for q in recent])
            skill = np.clip(errors,0.,3.).mean(axis=0)
            weights = np.array([.6,.25,.15])*np.exp(-(skill-skill.min())/.1)
        else:
            weights = np.asarray([1.,0.,0.])
        weights /= weights.sum()
        r["dynamic_blend"] = float(weights@[r[m] for m in METHODS[:-1]])
        yield {k:r[k] for k in ("release_date","study_phase","week_ending","actual","measured","scale","covered",*METHODS)} | dict(
            forecast_cutoff_utc=r["cutoff"].isoformat(),target_known_by_utc=r["known"].isoformat(),training_weeks=len(earlier),
            latest_training_label_utc=earlier[-1]["known"].isoformat() if earlier else None,
            optical_increment_million_bbl=gain,blend_weights_json=json.dumps(dict(zip(METHODS[:-1],map(float,weights),strict=True))))
        history.append(r)


def score(rows):
    y = np.asarray([r["actual"] for r in rows])
    v = {method:np.asarray([r[method] for r in rows]) for method in METHODS}
    ae = {m:np.abs(y-x) for m,x in v.items()}
    return dict(weeks=len(rows),methods={m:dict(mae_kbbl=float(a.mean()*1000),rmse_kbbl=float(np.sqrt(np.mean(a*a))*1000),
             correct_direction=int(np.count_nonzero(y*v[m]>0)),paired_mae_delta_95_kbbl=block_bootstrap_mean(1000*(a-ae[METHODS[0]])))
             for m,a in ae.items()}, optical_delta_95_kbbl=block_bootstrap_mean(1000*(ae["dynamic_optical"]-ae["dynamic_public"])))


def main():
    rows = list(replay())
    report = dict(status="exploratory_target_volatility_adjustment", phases={p:{scope:score([r for r in rows if r["study_phase"]==p
                  and (scope=="all" or r["measured"])]) for scope in ("all","measured")} for p in PHASES})
    export(f"{STEM}_forecasts",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    for p,g in report["phases"].items():
        print(p,{s:{m:round(v["mae_kbbl"],2) for m,v in q["methods"].items()} for s,q in g.items()})


if __name__=="__main__":
    main()
