"""Test dated WTI dynamics and learned optical reliability on pre-release inventory forecasts."""
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from scipy.linalg import solve
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write
from fit_cushing_adaptive_nowcast import FREEZE_DAY
from fit_cushing_multiview_nowcast import STEM as INCUMBENT
from fit_cushing_nowcast_ensemble import MAX_RELEASE, PHASES, block_bootstrap_mean, ridge_fit

STEM = "cushing_sensor_market_nowcast"
EASTERN = ZoneInfo("America/New_York")
MARKET_HORIZONS = (1,3,5,10,20)
MODELS = ("incumbent", "wti_ridge", "wti_boost", "wti_sensor", "sensor_only", "terminal_weighted")


def contract_book():
    book = defaultdict(dict)
    with closing(read_only(SOURCE)) as db:
        for day,month,close in db.execute("""SELECT session_date,delivery_month,official_close
            FROM wti_eia_dated_close WHERE session_date<=? ORDER BY session_date,curve_rank""",(MAX_RELEASE,)):
            book[day][month] = float(close)
    return sorted(day for day,curve in book.items() if len(curve)==4), book


def market_features(days,book,boundaries):
    vector, marks = [], []
    for boundary in boundaries:
        # Reference times use only the previous complete local market date.
        stop = bisect_left(days,boundary.astimezone(EASTERN).date().isoformat())-1
        if stop<0:
            vector.extend([np.nan]*(3+3*len(MARKET_HORIZONS)))
            marks.append(None)
            continue
        day = days[stop]
        marks.append(day)
        vector.extend(curve_features(book[day], (book[days[stop-h]] if stop>=h else {} for h in MARKET_HORIZONS)))
    return np.array(vector), marks


def curve_features(curve, history):
    months = tuple(curve)
    front, second, _, far = curve.values()
    vector = [front,front-second,front-far]
    for past in history:
        old_front, old_second, old_far = (past.get(months[i],np.nan) for i in (0,1,3))
        vector.extend((front-old_front,(front-second)-(old_front-old_second),(front-far)-(old_front-old_far)))
    return vector


def optical_context():
    coverage = {(r["week_ending"],r["release_date"]):r for r in read_csv("sentinel2_space_time_coverage_weeks")
                if r["study_phase"] in PHASES}
    terminals = defaultdict(dict)
    for r in read_csv("sentinel2_space_time_coverage_terminal_weeks"):
        if r["study_phase"] in PHASES and int(r["baseline_b04_tanks"]):
            terminals[r["week_ending"],r["release_date"]][r["terminal_id"]] = float(r["capture_change_kbbl"])/1000
    return coverage, terminals


def inputs():
    days,book = contract_book()
    coverage,terminals = optical_context()
    for r in read_csv(f"{INCUMBENT}_forecasts"):
        key = r["week_ending"],r["release_date"]
        if r["release_date"]>MAX_RELEASE or r["study_phase"] not in PHASES:
            raise ValueError(f"Bad open-period forecast on {key}")
        g = coverage[key]
        boundaries = tuple(datetime.fromisoformat(t) for t in (g["current_measurement_at_utc"],r["forecast_cutoff_utc"]))
        market,marks = market_features(days,book,boundaries)
        yield input_row(r,g,terminals[key],market,marks)


def input_row(r,g,terminal_amounts,market,marks):
    key = r["week_ending"],r["release_date"]
    cutoff, known, boundary = (datetime.fromisoformat(s) for s in (r["forecast_cutoff_utc"],r["target_known_by_utc"],
                                                                     g["current_measurement_at_utc"]))
    if not boundary<cutoff<known or datetime.fromisoformat(g["cutoff_utc"])!=cutoff:
        raise ValueError(f"Bad forecast or observation clock at {key}")
    if r["latest_stack_training_release_available_utc"] and datetime.fromisoformat(r["latest_stack_training_release_available_utc"])>=cutoff:
        raise ValueError(f"Predecessor estimate uses a later label at {key}")
    measured = r["measured"] in (True,"True")
    raw, combined = map(float,(r["raw"],r["combined"]))
    quality, signal = None,None
    if measured:
        groups = np.array(list(terminal_amounts.values()))
        if not len(groups) or not np.isclose(groups.sum(),raw,atol=1e-9,rtol=0):
            raise ValueError(f"Terminal volumes do not reproduce the hub estimate at {key}")
        gross = np.abs(groups).sum()
        coherence = min(1.,abs(raw)/gross) if gross else 0.
        area = float(g["baseline_b04_area_fraction"])
        interval = float(g["baseline_b04_space_time_fraction"])
        overlap = float(g["overlap_change_kbbl"])/1000
        late_gap = float(g["area_weighted_following_hours"])/168
        disagreement = abs(combined-raw)
        quality = np.array([np.log1p(int(g["baseline_b04_tanks"])),area,interval,len(groups),coherence,
                            late_gap,disagreement,abs(raw)])
        signal = np.array([raw,combined-raw,overlap-raw,raw*coherence,raw*np.sqrt(interval),raw*late_gap])
    values = {k:float(r[k]) for k in ("target","prior","prior_plus_raw","previous_optical","stack_multiview")}
    return dict(study_phase=r["study_phase"],week_ending=key[0],release_date=key[1],cutoff=cutoff,known=known,
        measured=measured,quality=quality,signal=signal,market=market,marks=marks,raw=raw,
        terminal_amounts=terminal_amounts if measured else {},incumbent=values["stack_multiview"],**values)


def standardize(x,z):
    observed = np.isfinite(x)
    count = observed.sum(axis=0)
    centre = np.divide(np.nansum(x,axis=0),count,out=np.zeros(x.shape[1]),where=count>0)
    x = np.where(observed,x-centre,0.)
    scale = np.sqrt(np.mean(x*x,axis=0))
    scale[scale==0] = 1.
    z = np.where(np.isfinite(z),z-centre,0.)
    return np.clip(x/scale,-4,4), np.clip(z/scale,-4,4)


def wti_corrections(history,row):
    if len(history)<60:
        return 0.,0.
    sample = history[-156:]
    x = np.array([q["market"] for q in sample])
    y = np.array([q["target"]-q["incumbent"] for q in sample])
    normalized,z = standardize(x,row["market"])
    linear = .55*ridge_fit(normalized,y,z,1.5,robust=True)
    nonlinear = HistGradientBoostingRegressor(loss="absolute_error",max_iter=28,max_leaf_nodes=4,min_samples_leaf=18,
        l2_regularization=4.,learning_rate=.05,early_stopping=False,random_state=2971)
    weights = np.exp2(-np.arange(len(y)-1,-1,-1)/65.)
    nonlinear.fit(x,y,sample_weight=weights)
    boost = .55*float(nonlinear.predict(row["market"][None,:])[0])
    return tuple(float(np.clip(d,-.7,.7)) for d in (linear,boost))


def optical_update(measured,row,anchor):
    sample = measured[-78:]
    if len(sample)<26 or row["signal"] is None:
        return 0., .5
    x = np.array([q["signal"] for q in sample])
    scale = np.maximum(np.sqrt(np.mean(x*x,axis=0)),.05)
    x /= scale
    residual = np.array([q["target"]-q[anchor] for q in sample])
    ages = np.array([(row["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in sample])
    base = np.exp2(-ages/52.)
    weights = base.copy()
    penalty = np.eye(x.shape[1])*.65*weights.sum()
    for _ in range(5):
        beta = solve(x.T@(weights[:,None]*x)+penalty,x.T@(weights*residual),assume_a="pos")
        weights = base*np.minimum(1.,1./np.maximum(np.abs(residual-x@beta),1e-9))
    update = float((row["signal"]/scale)@beta)
    trial = [q for q in sample if abs(q[f"{anchor}_optical_trial"])>1e-9]
    reliability = .5
    if len(trial)>=26:
        context,z = standardize(np.array([q["quality"] for q in trial]),row["quality"])
        design = np.column_stack((np.ones(len(trial)),context))
        labels = np.array([abs(q["target"]-q[anchor]-q[f"{anchor}_optical_trial"])<abs(q["target"]-q[anchor]) for q in trial])
        coeff = np.zeros(design.shape[1])
        ridge = np.eye(len(coeff))*8.
        for _ in range(6):
            predicted = expit(design@coeff)
            weight = predicted*(1-predicted)
            step = solve(design.T@(weight[:,None]*design)+ridge,design.T@(labels-predicted)-ridge@coeff,assume_a="pos")
            coeff += step
        reliability = float(expit(np.r_[1.,z]@coeff))
    limit = min(.8,abs(row["raw"])+abs(row["signal"][1]))
    return float(np.clip(update,-limit,limit)), reliability


def terminal_update(measured,row):
    if not row["measured"] or len(measured)<30:
        return 0.
    sample = measured[-90:]
    terminals = sorted({site for r in sample for site in r["terminal_amounts"]})
    x = np.array([[r["terminal_amounts"].get(site,0.) for site in terminals] for r in sample])
    z = np.array([row["terminal_amounts"].get(site,0.) for site in terminals])
    scale = np.maximum(np.sqrt(np.mean(x*x,axis=0)),.1)
    x /= scale
    y = np.array([r["target"]-r["incumbent"] for r in sample])
    age = np.array([(row["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in sample])
    base = np.exp2(-age/65.)
    weights = base.copy()
    ridge = 1.25*base.sum()*np.eye(len(terminals))
    for _ in range(5):
        beta = solve(x.T@(weights[:,None]*x)+ridge,x.T@(weights*y),assume_a="pos")
        weights = base*np.minimum(1.,1./np.maximum(np.abs(y-x@beta),1e-9))
    # Restrict each site's contribution to a correction on the volume actually observed.
    factors = np.clip(beta/scale,-1.,1.)
    return float(np.clip(z@factors,-.65,.65))


def replay(rows):
    history = []
    for r in rows:
        known = [q for q in history if q["known"]<r["cutoff"]]
        measured = [q for q in known if q["measured"]]
        linear,boost = wti_corrections(known,r)
        r.update(wti_ridge=r["incumbent"]+linear,wti_boost=r["incumbent"]+boost)
        for anchor,model in (("incumbent","sensor_only"),("wti_ridge","wti_sensor")):
            update,reliability = optical_update(measured,r,anchor)
            r[f"{anchor}_optical_trial"] = update
            r[model] = r[anchor]+reliability*update
        r["terminal_weighted"] = r["incumbent"]+terminal_update(measured,r)
        development = [q for q in known if q["release_date"]<FREEZE_DAY and q["measured"]==r["measured"]]
        if len(development)<26:
            chosen = "incumbent"
        else:
            chosen = min(MODELS,key=lambda m:np.mean([abs(q[m]-q["target"]) for q in development]))
        r["development_selected"] = r[chosen]
        history.append(r)
        yield {k:r[k] for k in ("study_phase","week_ending","release_date","target","measured","prior_plus_raw",*MODELS)} | dict(
            forecast_cutoff_utc=r["cutoff"].isoformat(), target_known_by_utc=r["known"].isoformat(),
            latest_training_label_utc=max((q["known"].isoformat() for q in known),default=None),
            wti_latest_before_reporting_mark=r["marks"][0],wti_latest_before_forecast_mark=r["marks"][1],
            market_features_available=int(np.isfinite(r["market"]).sum()), training_weeks=len(known),
            measured_training_weeks=len(measured),optical_coherence=r["quality"][4] if r["measured"] else None,
            optical_reliability_for_wti_update=reliability if r["measured"] else None,
            optical_update_before_reliability=r["wti_ridge_optical_trial"] if r["measured"] else None,
            selected_model=chosen,development_selected=r["development_selected"])


def score(rows):
    target = np.array([r["target"] for r in rows])
    result = {}
    for m in ("prior_plus_raw",*MODELS,"development_selected"):
        estimate = np.array([r[m] for r in rows])
        absolute = np.abs(estimate-target)
        result[m] = dict(mae_million_bbl=float(absolute.mean()),correct_direction=int(np.sum(estimate*target>0)),
            rmse_million_bbl=float(np.sqrt(np.mean((estimate-target)**2))),
            mae_delta_vs_incumbent_95pct_interval=block_bootstrap_mean(absolute-np.abs(np.array([r["incumbent"] for r in rows])-target)))
    return dict(weeks=len(rows),metrics=result,selected_models=dict(Counter(r["selected_model"] for r in rows)))


def main():
    with threadpool_limits(limits=1):
        rows = list(replay(inputs()))
    report = {phase:{scope:score([r for r in rows if r["study_phase"]==phase and (scope=="all" or r["measured"])])
                     for scope in ("all","with_optical")} for phase in PHASES}
    export(f"{STEM}_forecasts",rows)
    result = dict(phases=report,market_horizons_sessions=MARKET_HORIZONS,market_features=2*(3+3*len(MARKET_HORIZONS)),
        interpretation="Dated WTI market changes require the same contract in both sessions. Optical models learn "
        "from errors of prior forecast rows only. Policy selection uses released development targets. 2023 remains "
        "an inspected historical period. Holdout labels are excluded.")
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    for phase,scopes in report.items():
        for scope,s in scopes.items():
            print(phase,scope,s["weeks"],{k:round(v["mae_million_bbl"],4) for k,v in s["metrics"].items()},s["selected_models"],flush=True)


if __name__ == "__main__":
    main()
