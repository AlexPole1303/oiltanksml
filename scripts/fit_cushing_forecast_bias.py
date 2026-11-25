"""Calibrate recent signed inventory errors as an additional walk-forward forecast."""

import json
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading

STEM = "cushing_forecast_bias"
OPTICAL = "all_recovered"
MODELS = (PUBLIC,"carried_scene",OPTICAL,"public_bias","optical_bias","public_conditional_bias","optical_conditional_bias")
NAMES = (("public_bias",PUBLIC,False),("optical_bias",OPTICAL,False),
         ("public_conditional_bias",PUBLIC,True),("optical_conditional_bias",OPTICAL,True))


def weighted_median(values, weights):
    order = np.argsort(values,kind="stable")
    selected = np.searchsorted(np.cumsum(weights[order]),weights.sum()/2,side="left")
    return float(values[order[selected]])


def correction(rows, current, method, conditional):
    history = rows[-78:]
    if len(history)<26:
        return 0.
    errors = np.array([r["actual"]-r[method] for r in history])
    age = np.array([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in history])
    weights = np.exp2(-age/26.)
    if conditional:
        # Condition on the public forecast magnitude and on whether satellite
        # measurements exist, with a fixed floor to preserve pooled support.
        level = np.array([r[PUBLIC] for r in history])
        match = np.array([r["measured"]==current["measured"] for r in history])
        weights *= .35+.65*np.exp(-.5*((level-current[PUBLIC])/.75)**2)*match
    # Limit how much a recent run of errors can shift the forecast.
    return .5*float(np.clip(weighted_median(errors,weights),-.75,.75))


def replay():
    seen = []
    for r in read("cushing_recovered_optical_forecasts"):
        cutoff,known = [datetime.fromisoformat(r[k]) for k in ("cutoff_utc","known_utc")]
        if not cutoff<known or (r["last_training_label_utc"] and datetime.fromisoformat(r["last_training_label_utc"])>=cutoff):
            raise ValueError(f"Late forecast input on {r['release_date']}")
        current = r | {k:float(r[k]) for k in ("actual",PUBLIC,"carried_scene",OPTICAL)} | dict(
            cutoff=cutoff,known=known,measured=int(r["total_recovered_tanks"])>0)
        eligible = [q for q in seen if q["known"]<cutoff]
        forecasts = {name:current[method]+correction(eligible,current,method,local) for name,method,local in NAMES}
        seen.append(current)
        yield current | forecasts | dict(calibration_weeks=len(eligible[-78:]),
            latest_calibration_label_utc=eligible[-1]["known"].isoformat() if eligible else None)


def main():
    forecasts = list(replay())
    with threadpool_limits(limits=1):
        trades = list(trading(forecasts,MODELS))
    report = dict(status="historical_walk_forward_on_reviewed_period",forecasts={},trading={})
    for phase in PHASES:
        sample = [r for r in forecasts if r["study_phase"]==phase]
        report["forecasts"][phase] = {}
        for scope,rows in (("all",sample),("measured",[r for r in sample if r["measured"]])):
            stats = forecast_metrics(rows,MODELS)
            for name,method,_ in NAMES:
                errors = [1000*(abs(r["actual"]-r[name])-abs(r["actual"]-r[method])) for r in rows]
                stats["models"][name].update(paired_delta_vs_uncalibrated_95_kbbl=block_bootstrap_mean(errors),
                    bias_kbbl=1000*float(np.mean([r[name]-r["actual"] for r in rows])))
            report["forecasts"][phase][scope] = stats
        reference = [r for r in trades if r["study_phase"]==phase and r["method"]==PUBLIC]
        stats = report["trading"][phase] = {}
        for m in ("curve",*MODELS):
            rows = [r for r in trades if r["study_phase"]==phase and r["method"]==m]
            if [r["release_date"] for r in rows]!=[r["release_date"] for r in reference]:
                raise ValueError(f"Different trade calendars in {phase}/{m}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(rows,reference,strict=True)]
            stats[m] = robustness(rows) | daily_risk(rows) | dict(delta_vs_public_usd=sum(delta),paired_delta_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in rows))
        print(phase,{m:round(q["mae_kbbl"],2) for m,q in report["forecasts"][phase]["all"]["models"].items()})
        print(phase,{m:(q["trades"],q["net_170_usd"],q["daily_drawdown_usd"]) for m,q in stats.items()})
    fields = ("release_date","week_ending","study_phase","actual","cutoff_utc","known_utc","measured","calibration_weeks",
              "latest_calibration_label_utc",*MODELS)
    export(f"{STEM}_forecasts",[{k:r[k] for k in fields} for r in forecasts])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
