"""Walk forward through inventory forecast combinations, residuals and uncertainty."""
import json
import math
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import fit_cushing_adaptive_nowcast as adaptive
import fit_cushing_nowcast_ensemble as ensemble
import fit_cushing_sensor_market_nowcast as sensor
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write

STEM = "cushing_ml_extensions"
BASE = "sensor_only"
EXPERTS = (BASE, "prior_plus_raw", "incumbent", "wti_ridge", "wti_boost", "adaptive_optical",
           "adaptive_public", "public_ensemble", "prior", "zero")
CANDIDATES = (BASE, "registration_residual", "convex_stack", "robust_residual", "boosted_residual", "robust_without_direct_optics")
FREEZE = adaptive.FREEZE_DAY
TARGET_COVERAGE = .80


def by_issue(stem):
    rows = read_csv(f"{stem}_forecasts")
    result = {r["release_date"]:r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate forecast in {stem}")
    return result


def registration_changes():
    changes = defaultdict(list)
    for r in read_csv("shadow_registration_ml_tank_weeks"):
        if r["ml_applied"] == "True":
            cutoff = datetime.fromisoformat(r["cutoff_utc"])
            if not max(datetime.fromisoformat(r[f]) for f in ("archive_available_at_utc", "latest_training_image_at_utc")) < cutoff:
                raise ValueError(f"Late registration result on {r['release_date']}/{r['candidate_id']}")
            changes[r["release_date"], r["week_ending"]].append(
                (float(r["selected_change_kbbl"])-float(r["coarse_change_kbbl"]))/1000)
    return {key:(math.fsum(values), math.fsum(map(abs,values))) for key,values in changes.items()}


def feature_row(r, forecast, public, adapted, registration):
    issue,week = r["release_date"],r["week_ending"]
    for saved,clock in ((forecast,"latest_training_label_utc"),(public,"latest_training_release_available_utc"),
                        (adapted,"latest_training_release_available_utc")):
        trained = saved[clock]
        if ((saved["release_date"],saved["week_ending"]) != (issue,week)
            or datetime.fromisoformat(saved["forecast_cutoff_utc"]) != r["cutoff"]
            or datetime.fromisoformat(saved["target_known_by_utc"]) != r["known"]
            or (trained and datetime.fromisoformat(trained) >= r["cutoff"])):
            raise ValueError(f"Model chronology differs on {issue}")
    source = dict(forecast) | {k:public[k] for k in ("public_ensemble", "prior")} | {
        k:adapted[k] for k in ("adaptive_public", "adaptive_optical")} | dict(zero=0.)
    experts = np.array([float(source[m]) for m in EXPERTS])
    base = experts[0]
    adjustment, impact = registration.get((issue,week),(0.,0.))
    if impact and not r["measured"]:
        raise ValueError(f"Registration corrected a week with no measured B04 at {issue}")
    market = r["market"].reshape(2,-1)
    # Curve levels and same-contract 5- and 20-session moves at both dates.
    common = np.r_[experts, abs(base), experts.std(), market[:, [1,2,9,10,11,15,16,17]].ravel()]
    optics = np.r_[r["signal"], r["quality"]] if r["measured"] else np.full(14,np.nan)
    full = np.r_[common, optics, adjustment, impact, float(r["measured"])]
    return dict(study_phase=r["study_phase"],release_date=issue,week_ending=week,cutoff=r["cutoff"],known=r["known"],
                target=r["target"],measured=r["measured"],experts=experts,sensor_only=base,
                prior_plus_raw=float(source["prior_plus_raw"]),incumbent=float(source["incumbent"]),
                public_features=common,full_features=full,registration_delta=adjustment,
                uncertainty_features=np.r_[abs(base), experts.std(), abs(experts[5]-experts[6]), market[1,9],
                                          market[1,15], optics[-8:], impact])


def open_rows():
    registration = registration_changes()
    predictions, public, adapted = (by_issue(stem) for stem in (sensor.STEM, ensemble.STEM, f"{adaptive.STEM}_extended"))
    for r in sensor.inputs():
        issue = r["release_date"]
        yield feature_row(r, predictions[issue], public[issue], adapted[issue], registration)


def weights(history, row, half_life):
    ages = np.array([(row["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in history])
    return np.exp2(-ages/half_life)


def convex_stack(history,row):
    if len(history)<32:
        return row[BASE]
    x = np.array([q["experts"] for q in history])
    y = np.array([q["target"] for q in history])
    mass = weights(history,row,52)
    prior = np.zeros(len(EXPERTS))
    prior[0] = 1

    def objective(w):
        residual = x@w-y
        smooth = np.sqrt(.18**2+residual**2)
        penalty = .08*np.sum((w-prior)**2)
        return np.average(smooth,weights=mass)+penalty, (x.T@(mass*residual/smooth))/mass.sum()+.16*(w-prior)

    result = minimize(objective,prior,jac=True,method="SLSQP",bounds=[(0.,1.)]*len(prior),
                      constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],options={"ftol":1e-8,"maxiter":100})
    if not result.success or not np.isclose(result.x.sum(),1):
        raise RuntimeError(f"Stack fit failed on {row['release_date']}: {result.message}")
    return float(row["experts"]@result.x)


def bounded_ridge(history,row,key,y,penalty):
    x,z = sensor.standardize(np.array([q[key] for q in history]),row[key])
    return ensemble.ridge_fit(x,y,z,penalty,robust=True)


def correction(history,row):
    if len(history)<12 or not row["registration_delta"]:
        return 0.
    x = np.array([q["registration_delta"] for q in history])
    y = np.array([q["target"]-q[BASE] for q in history])
    mass = weights(history,row,65)
    coefficient = (mass*x)@np.clip(y,-1.5,1.5)/((mass*x)@x+.015*mass.sum())
    return float(np.clip(coefficient,0.,1.)*row["registration_delta"])


def point_predictions(known,row):
    same = [q for q in known if q["measured"] == row["measured"]][-130:]
    adjusted = row[BASE]+correction([q for q in same if q["registration_delta"]],row)
    stack = convex_stack(same,row)
    ridge, boost, no_optics = (row[BASE],)*3
    if len(known)>=60:
        sample = known[-156:]
        y = np.array([q["target"]-q[BASE] for q in sample])
        ridge,no_optics = (row[BASE]+float(np.clip(.65*bounded_ridge(sample,row,key,y,1.15),-.55,.55))
                           for key in ("full_features","public_features"))
        model = HistGradientBoostingRegressor(loss="absolute_error",max_iter=34,max_leaf_nodes=4,min_samples_leaf=18,
            l2_regularization=4.,learning_rate=.045,early_stopping=False,random_state=9021)
        model.fit(np.array([q["full_features"] for q in sample]),y,sample_weight=weights(sample,row,65))
        boost = row[BASE]+float(np.clip(.65*model.predict(row["full_features"][None,:])[0],-.55,.55))
    return dict(zip(CANDIDATES,(row[BASE],adjusted,stack,ridge,boost,no_optics),strict=True))


def choose(known):
    dev = [q for q in known if "2020-01-01" <= q["release_date"] < FREEZE]
    if len(dev)<52:
        return BASE
    return min(CANDIDATES,key=lambda m: np.mean([abs(q["target"]-q[m]) for q in dev]) + (.01 if m!=BASE else 0.))


def quantile(errors):
    k = min(len(errors),math.ceil((len(errors)+1)*TARGET_COVERAGE))-1
    return float(np.partition(errors,k)[k])


def interval(known,row):
    same = [q for q in known if q["measured"]==row["measured"]][-104:]
    sample = same if len(same)>=32 else known[-104:]
    if len(sample)<26:
        return None,None,None
    errors = np.array([abs(q["target"]-q[BASE]) for q in sample])
    relative = [abs(q["target"]-q[BASE])/q["error_scale"] for q in sample if q["error_scale"] is not None]
    scale = float(np.clip(np.exp(bounded_ridge(sample,row,"uncertainty_features",np.log(errors+.2),1.3)),.2,3.))
    flat = quantile(errors)
    return scale,flat,scale*quantile(relative) if len(relative)>=26 else flat


def replay(rows):
    history = []
    for original in rows:
        r = dict(original)
        if not r["cutoff"]<r["known"] or (history and r["cutoff"]<=history[-1]["cutoff"]):
            raise ValueError(f"Bad weekly forecast order on {r['release_date']}")
        known = [q for q in history if q["known"]<r["cutoff"]]
        r.update(point_predictions(known,r))
        r["prior_plus_registered_raw"] = r["prior_plus_raw"]+r["registration_delta"]
        choice = choose(known)
        r["selected"] = r[choice]
        r["error_scale"],flat,conditional = interval(known,r)
        history.append(r)
        yield {k:r[k] for k in ("study_phase","week_ending","release_date","measured","target","prior_plus_raw","prior_plus_registered_raw",
            "incumbent","registration_delta",*CANDIDATES,"selected","error_scale")} | dict(selected_model=choice,
            forecast_cutoff_utc=r["cutoff"].isoformat(),target_known_by_utc=r["known"].isoformat(),
            latest_training_label_utc=max((q["known"].isoformat() for q in known),default=None),
            interval_center_million_bbl=r[BASE],flat_half_width_million_bbl=flat,conditional_half_width_million_bbl=conditional)


def scores(rows):
    target = np.array([r["target"] for r in rows])
    result = {}
    baseline = np.abs(target-np.array([r[BASE] for r in rows]))
    for name in ("prior_plus_raw","prior_plus_registered_raw","incumbent",*CANDIDATES,"selected"):
        point = np.array([r[name] for r in rows])
        error = point-target
        result[name] = dict(mae_kbbl=float(abs(error).mean()*1000),rmse_kbbl=float(np.sqrt(np.mean(error**2))*1000),
                            direction_hits=int(np.sum(point*target>0)),
                            mae_delta_vs_sensor_kbbl_95pct_interval=ensemble.block_bootstrap_mean((abs(error)-baseline)*1000))
    intervals = {}
    for name in ("flat","conditional"):
        sample = [r for r in rows if r[f"{name}_half_width_million_bbl"] is not None]
        half = np.array([r[f"{name}_half_width_million_bbl"] for r in sample])
        error = np.array([abs(r["target"]-r[BASE]) for r in sample])
        decisive = [r for r in sample if abs(r[BASE])>r[f"{name}_half_width_million_bbl"]]
        intervals[name] = dict(weeks=len(sample),covered=int(np.sum(error<=half)),mean_width_kbbl=float(2000*half.mean()),
            interval_score_kbbl=float(1000*np.mean(2*half+2/(1-TARGET_COVERAGE)*np.maximum(error-half,0))),
            direction_calls=len(decisive),correct_calls=sum(bool(r[BASE]*r["target"]>0) for r in decisive))
    raw,registered = (np.array([r[name] for r in rows]) for name in ("prior_plus_raw","prior_plus_registered_raw"))
    optical_delta = (abs(registered-target)-abs(raw-target))*1000
    return dict(weeks=len(rows),metrics=result,intervals=intervals,selected_models=dict(Counter(r["selected_model"] for r in rows)),
                registration_mae_delta_vs_raw_kbbl=float(optical_delta.mean()),
                registration_mae_delta_vs_raw_kbbl_95pct_interval=ensemble.block_bootstrap_mean(optical_delta))


def main():
    with threadpool_limits(limits=1):
        predictions = list(replay(open_rows()))
    report = {}
    for phase,subset in (("development_2020_22",[r for r in predictions if "2020-01-01"<=r["release_date"]<FREEZE]),
                         ("seen_validation_2023",[r for r in predictions if r["study_phase"]=="validation"])):
        cohorts = dict(all=subset, with_b04=[r for r in subset if r["measured"]])
        report[phase] = {scope:scores(rows) for scope,rows in cohorts.items()}
        for scope,s in report[phase].items():
            print(phase,scope,s["weeks"],{m:round(v["mae_kbbl"],1) for m,v in s["metrics"].items()},
                  s["selected_models"],flush=True)
    export(f"{STEM}_forecasts",predictions)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(dict(evaluation=report,target_interval_coverage=TARGET_COVERAGE,interval_center=BASE,
        interpretation="Exploratory 2023 estimates. Model and interval fitting use only earlier released errors. "
        "The automatic point choice uses 2020–2022 and requires a 10 kbbl gain in historical MAE to displace sensor_only. "
        "No actual holdout EIA or trading outcomes were read."),indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
