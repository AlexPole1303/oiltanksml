"""Fit regime-aware pre-release inventory forecasts and replay them chronologically."""
import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timedelta

import numpy as np
from scipy.linalg import solve
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import RAW, ROOT, WPSR, write
from fit_cushing_nowcast_ensemble import PHASES, block_bootstrap_mean, feature_rows, number
from publication_clocks import utc

STEM = "cushing_adaptive_nowcast"
PUBLIC = ("prior", "previous_ensemble", "local_ar_8", "local_ar_26", "state_ar", "state_ar_recent",
          "boosted_residual", "prior_innovation")
OPTICAL = ("no_correction", "quarter_raw", "half_raw", "raw", "overlap", "area_adjusted", "boosted_residual")
POLICIES = {"fast": (5., .10, False), "medium": (13., .15, False), "slow": (26., .22, False),
            "state": (20., .13, True)}
EVALUATION = ("prior", "prior_plus_raw", "previous_public", "previous_optical", "adaptive_public", "adaptive_optical")
FREEZE_DAY = "2023-01-01"


def archived_training_rows():
    directory = RAW/"eia"
    paths = sorted((directory/"wpsr_training_history").glob("*_table4.csv"))
    paths += sorted((directory/"wpsr").glob("2016_*_table4.csv"))
    seen = set()
    for path in paths:
        issue = date.fromisoformat(path.name[:10].replace("_", "-"))
        if not 2012 <= issue.year <= 2016:
            raise ValueError(f"Unexpected training issue {issue}")
        raw = path.read_bytes()
        try:
            decoded = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = raw.decode("cp1252")
        rows = list(csv.reader(decoded.splitlines()))
        header = next(r for r in rows if r and r[0] == "STUB_1")
        week, previous = (datetime.strptime(d, "%m/%d/%y").date() for d in header[1:3])
        source = path.with_suffix(".source").read_text().strip()
        if not source.startswith(f"{WPSR}{issue.year}/{path.name[:10]}/") or not previous < week < issue:
            raise ValueError(f"Invalid vintage on {issue}")
        tank = next(r for r in rows if r and r[0].strip().lower() in {"cushing", "cushing, oklahoma"})
        current, prior, change = (float(x.replace(",", "")) for x in tank[1:4])
        if week in seen or (week-previous).days != 7 or not np.isclose(current-prior,change,atol=.0011,rtol=0):
            raise ValueError(f"Duplicate week or invalid reported change on {issue}")
        seen.add(week)
        yield dict(release_date=issue.isoformat(), week_ending=week.isoformat(), previous_week_ending=previous.isoformat(),
            forecast_cutoff_utc=utc(issue), target_known_by_utc=utc(issue+timedelta(days=1)),
            target_million_bbl=round(current-prior,3), current_level_million_bbl=current,
            previous_level_million_bbl=prior, source_url=source, source_file=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(raw).hexdigest())


def state(history, prior, level):
    recent = np.array([q["label"] for q in history[-8:]])
    if len(recent)<4:
        recent = np.zeros(4) if not len(recent) else np.pad(recent,(4-len(recent),0),mode="edge")
    signed_persistence = float(np.mean(np.sign(recent[1:]*recent[:-1])))
    avg = float(np.mean(recent[-4:]))
    recent_scale = float(np.sqrt(np.mean(recent**2)))
    return np.array([prior, avg, recent_scale, signed_persistence, level/100])


def weighted_ar(history,row,half_life,with_state):
    # Keep estimates regularized even when a fast filter places most weight on a few weeks.
    lag = np.array([q["state"] for q in history])
    now = row["state"]
    if with_state:
        def design(x):
            prior, avg, vol, persistence, level = x.T
            return np.column_stack([np.ones(len(x)),prior,avg,prior*persistence,prior*np.tanh(vol-1),level-.45])
    else:
        def design(x):
            return np.column_stack([np.ones(len(x)),x[:,0]])
    x = design(lag)
    z = design(now[None,:])[0]
    y = np.array([q["label"] for q in history])
    ages = np.array([(row["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in history])
    base = np.exp2(-ages/half_life)
    w = base.copy()
    centre = np.zeros(x.shape[1])
    centre[1] = .65
    penalty = np.diag([1.5,3.,5.,6.,6.,9.][:x.shape[1]])
    for _ in range(5):
        b = solve(x.T@(w[:,None]*x)+penalty,x.T@(w*y)+penalty@centre,assume_a="pos")
        w = base*np.minimum(1.,1.1/np.maximum(np.abs(y-x@b),1e-9))
    return float(z@b)


def tree_prediction(x,y,z,half_life):
    sample_weight = np.exp2(-np.arange(len(y)-1,-1,-1)/half_life)
    model = HistGradientBoostingRegressor(loss="absolute_error",max_iter=36,max_leaf_nodes=4,learning_rate=.065,
        min_samples_leaf=18,l2_regularization=3.,early_stopping=False,random_state=3480)
    model.fit(x,y,sample_weight=sample_weight)
    return float(model.predict(np.asarray(z).reshape(1,-1))[0])


def ensemble(history,row,kind,policy):
    half_life, temperature, conditioned = POLICIES[policy]
    history = [q for q in history if kind in q][-104:]
    if len(history)<12:
        return np.ones(len(row[kind]))/len(row[kind])
    errors = np.abs(np.array([q[kind] for q in history])-np.array([q["label"] for q in history])[:,None])
    age = np.array([(row["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in history])
    weight = np.exp2(-age/half_life)
    if conditioned:
        states = np.array([q["state"] for q in history])
        distance = (states-row["state"])/np.array([2.,1.5,1.,.65,.2])
        weight *= np.exp(-np.minimum(np.sum(distance**2,axis=1),10)/2)
    loss = np.average(errors,axis=0,weights=weight)
    confidence = np.exp(-(loss-loss.min())/temperature)
    return .9*confidence/confidence.sum()+.1/len(confidence)


def public_options(known, active, row, previous):
    prior = row["prior_million_bbl"]
    local = [weighted_ar(known,row,h,False) for h in (8.,26.)]
    state_long = weighted_ar(known,row,104.,True)
    state_short = weighted_ar(active,row,26.,True)
    innovation = [q["label"]-q["state"][0] for q in known[-5:]]
    drift = .25*float(np.median(innovation))
    y = np.array([q["label"]-q["state"][0] for q in active])
    tree = prior+tree_prediction([q["public_features"] for q in active],y,row["public_features"],52.)
    return np.array([prior,previous,*local,state_long,state_short,tree,prior+drift])


def optical_options(history,row,base):
    raw,overlap,_ = row["optical"]
    coverage = row["optical_coverage"]
    learned = 0.5*raw
    if len(history)>=40:
        learned = tree_prediction([q["optical_features"] for q in history],
            [q["label"]-q["adaptive_public"] for q in history],row["optical_features"],44.)
        # Limit the learned correction to the scale and direction of the observed change.
        learned = np.clip(learned,min(0.,2*raw),max(0.,2*raw))
    return base+np.array([0.,.25*raw,.5*raw,raw,overlap,raw*min(2.,1+np.sqrt(coverage)),learned])


def policy_for(known,kind,scope):
    candidates = [q for q in known if q["release_date"] < FREEZE_DAY and kind in q and
                  (scope!="measured" or q["optical"] is not None)]
    if len(candidates)<20:
        return "medium"
    actual = np.array([q["label"] for q in candidates])
    losses = np.mean(np.abs(np.array([q[kind] for q in candidates])-actual[:,None]),axis=0)
    return tuple(POLICIES)[int(np.argmin(losses))]


def replay(archived, study, previous):
    ledger = []
    for r in [*archived,*study]:
        issue = r["release_date"]
        known = [q for q in ledger if q["known_at"]<r["cutoff"]]
        available = r.get("prior_million_bbl",known[-1]["label"] if known else 0.)
        r["state"] = state(known,available,r.get("previous_level_million_bbl",0.))
        ledger.append(r)
        if "wide" not in r:
            continue
        r["public_features"] = np.r_[r["wide"],r["state"][1:4]]
        active = [q for q in known if "public_features" in q]
        old = previous.get(issue)
        if old is None:
            continue
        old_public, old_optical = (float(old[k]) for k in ("public_ensemble","optical_ensemble"))
        if (old["week_ending"]!=r["week_ending"] or datetime.fromisoformat(old["forecast_cutoff_utc"])!=r["cutoff"]):
            raise ValueError(f"Misaligned predecessor forecast on {issue}")
        r["public"] = public_options(known,active,r,old_public)
        r["public_policy_forecasts"] = np.array([ensemble(active,r,"public",p)@r["public"] for p in POLICIES])
        policy = policy_for(active,"public_policy_forecasts","all")
        r["adaptive_public"] = float(r["public_policy_forecasts"][tuple(POLICIES).index(policy)])
        measured = [q for q in active if "optical" in q and q["optical"] is not None and "optical_candidates" in q]
        optical_policy, predicted = None,r["adaptive_public"]
        if r["optical"] is not None:
            r["optical_features"] = np.r_[r["optical"],r["optical_coverage"],r["state"][:4]]
            r["optical_candidates"] = optical_options(measured,r,r["adaptive_public"])
            r["optical_policy_forecasts"] = np.array([ensemble(measured,r,"optical_candidates",p)@r["optical_candidates"] for p in POLICIES])
            optical_policy = policy_for(measured,"optical_policy_forecasts","measured")
            predicted = float(r["optical_policy_forecasts"][tuple(POLICIES).index(optical_policy)])
        past = [q for q in active if "adaptive_optical" in q][-78:]
        interval = np.quantile([q["label"]-q["adaptive_optical"] for q in past],[.10,.90]) if len(past)>=26 else (None,None)
        r["adaptive_optical"] = predicted
        yield {k:r[k] for k in ("study_phase","week_ending","release_date","forecast_cutoff_utc","target_known_by_utc","optical_available")} | dict(
            target_million_bbl=r["label"],prior=available,prior_plus_raw=available+(r["raw_optical_million_bbl"] or 0.),
            previous_public=old_public,previous_optical=old_optical,adaptive_public=r["adaptive_public"],adaptive_optical=predicted,
            recent_sign_persistence=r["state"][3],public_policy=policy,optical_policy=optical_policy,
            published_training_rows=len(known),study_training_rows=len(active),optical_training_rows=len(measured),
            latest_training_release_available_utc=max(q["known_at"] for q in known).isoformat(),
            residual_interval_lower=predicted+interval[0] if interval[0] is not None else None,
            residual_interval_upper=predicted+interval[1] if interval[1] is not None else None,
            **{f"expert_{k}":v for k,v in zip(PUBLIC,r["public"])})


def score(rows):
    y = np.array([r["target_million_bbl"] for r in rows])
    results = {}
    for method in EVALUATION:
        estimate = np.array([r[method] for r in rows])
        results[method] = dict(mae_million_bbl=float(np.mean(abs(estimate-y))),
            rmse_million_bbl=float(np.sqrt(np.mean((estimate-y)**2))), correct_direction=int(np.count_nonzero(estimate*y>0)))
    intervals = [r for r in rows if r["residual_interval_lower"] is not None]
    interval_hits = sum(int(r["residual_interval_lower"]<=r["target_million_bbl"]<=r["residual_interval_upper"]) for r in intervals)
    return dict(weeks=len(rows),methods=results,interval_weeks=len(intervals),interval_hits=interval_hits,
        adaptive_optical_minus_public_mae_interval=block_bootstrap_mean([abs(r["adaptive_optical"]-v)-abs(r["adaptive_public"]-v) for r,v in zip(rows,y)]),
        adaptive_optical_minus_prior_plus_raw_mae_interval=block_bootstrap_mean([abs(r["adaptive_optical"]-v)-abs(r["prior_plus_raw"]-v) for r,v in zip(rows,y)]))


def main():
    raw = list(archived_training_rows())
    if not any(r["release_date"]<"2016-01-01" for r in raw):
        raise ValueError("Download the 2012-2015 WPSR training archive before running the extended model")
    historical = [dict(r,label=r["target_million_bbl"],cutoff=datetime.fromisoformat(r["forecast_cutoff_utc"]),
                       known_at=datetime.fromisoformat(r["target_known_by_utc"])) for r in raw]
    study = [dict(r,previous_level_million_bbl=number(r["core"][4])/1000) for r in feature_rows()]
    if any(not np.isfinite(r["previous_level_million_bbl"]) for r in study):
        raise ValueError("Missing prior Cushing level in study rows")
    previous = {r["release_date"]:r for r in read_csv("cushing_nowcast_ensemble_forecasts")}
    report = dict(training_archive_by_year=dict(Counter(r["release_date"][:4] for r in raw)), variants={},
        candidate_public_models=PUBLIC,candidate_satellite_corrections=OPTICAL,policy_configurations=POLICIES,
        rule="Policy family selection uses released development observations before 2023-01-01. Models and weights update only on published labels. The 2023 sample has been inspected in earlier experiments.")
    export("cushing_wpsr_training_archive",raw)
    with threadpool_limits(limits=1):
        for name, archive in (("extended",historical),("short",[])):
            predictions = list(replay([dict(r) for r in archive],[dict(r) for r in study],previous))
            if len(predictions)!=len(previous):
                raise ValueError("Adaptive forecast sample differs from the prior ensemble")
            export(f"{STEM}_{name}_forecasts",predictions)
            stats = {phase:{scope:score([r for r in predictions if r["study_phase"]==phase and
                     (scope!="with_optical" or r["optical_available"])]) for scope in ("all","with_optical")} for phase in PHASES}
            report["variants"][name] = stats
            for phase,scopes in stats.items():
                for scope, result in scopes.items():
                    print(name,phase,scope,result["weeks"],{k:round(m["mae_million_bbl"],4) for k,m in result["methods"].items()},flush=True)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
