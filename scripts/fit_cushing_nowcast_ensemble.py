"""Replay public and satellite inventory forecasts with release-vintage features."""
import json
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime

import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls
from scipy.spatial.distance import cdist

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write
from fit_eia_cushing_nowcast import PHASES, inputs

STEM = "cushing_nowcast_ensemble"
MAX_RELEASE = "2023-12-31"
METRICS = ("cushing_crude", "padd2_commercial_crude", "padd3_commercial_crude", "us_commercial_crude",
           "padd2_crude_refiner_inputs", "padd3_crude_refiner_inputs", "padd2_refinery_utilization",
           "us_crude_net_imports_including_spr", "us_crude_production", "us_crude_exports")
PUBLIC = ("zero", "prior", "mean4", "ridge_core", "ridge_wide", "ridge_recent", "huber_wide", "kernel_wide")
OPTICAL = ("no_update", "raw_quarter", "raw_half", "raw_full", "overlap_half", "positive_ridge", "positive_ridge_strong")
METHODS = ("zero", "prior", "prior_plus_raw", "legacy_ridge_public", "legacy_ridge_optical", "public_ensemble",
           "optical_ensemble")


def number(value):
    return float(value) if value not in (None, "") else np.nan


def public_vintages(db, through):
    vintage = defaultdict(dict)
    for issue, week, key, value in db.execute("""SELECT release_date,report_date,metric,value
        FROM eia_control_vintage WHERE release_date<=?""", (through,)):
        vintage[issue, week][key] = number(value)
    return vintage


def feature_rows():
    clocks = {r["release_date"]: r for r in read_csv("eia_release_clock") if r["release_date"] <= MAX_RELEASE}
    geometry = {(r["week_ending"], r["release_date"]): r for r in read_csv("sentinel2_space_time_coverage_eia_comparison")}
    with closing(read_only(SOURCE)) as db:
        vintage = public_vintages(db,MAX_RELEASE)
        context = {r[0]: r[1:] for r in db.execute("""SELECT c.release_date,c.cutoff_utc,c.last_wpsr_issue,
            c.latest_wpsr_observation_week,c.last_complete_wti_curve_session,cl1_minus_cl2,cl1_minus_cl4,
            cftc_conservative_from_utc,open_interest,money_manager_net
            FROM oil_market_context JOIN market_context_clock c USING(release_date,phase)
            WHERE c.phase='pre_release' AND c.release_date<=?""", (MAX_RELEASE,))}
    yield from enrich_feature_rows((r for r in inputs() if r["release_date"]<=MAX_RELEASE),geometry,clocks,vintage,context)


def enrich_feature_rows(rows, geometry, clocks, vintage, context):
    source_issues = sorted(clocks)
    for r in rows:
        issue, week = r["release_date"], r["week_ending"]
        clock = clocks[issue]
        if week != clock["latest_week_in_issue"]:
            continue
        cutoff = datetime.fromisoformat(r["forecast_cutoff_utc"])
        context_cutoff, prev, prev_week, curve_day, curve2, curve4, cot_time, interest, net = context[issue]
        if (datetime.fromisoformat(context_cutoff) != cutoff or not prev or prev >= issue
            or datetime.fromisoformat(clocks[prev]["conservative_postrelease_from_utc"]) > cutoff
            or (curve_day and curve_day >= issue) or (cot_time and datetime.fromisoformat(cot_time) > cutoff)):
            raise ValueError(f"Invalid feature clock on {issue}")
        prior_index = source_issues.index(prev)
        prior_releases = source_issues[max(0, prior_index-3):prior_index+1]
        delta = [vintage[p, clocks[p]["latest_week_in_issue"]].get("cushing_crude", np.nan)
                 - vintage[p, clocks[p]["oldest_week_in_issue"]].get("cushing_crude", np.nan) for p in prior_releases]
        known = [d/1000 for d in delta if np.isfinite(d)]
        mean4 = float(np.mean(known)) if known else r["prior_million_bbl"]
        current = vintage[prev, prev_week]
        earlier = vintage[prev, clocks[prev]["oldest_week_in_issue"]]
        change = [current.get(k, np.nan)-earlier.get(k, np.nan) for k in METRICS]
        core = (r["prior_million_bbl"], mean4, *r["eia_features"][1:3], current.get("cushing_crude", np.nan),
                number(curve2), number(curve4))
        wide = (*core, *[current.get(k, np.nan) for k in METRICS[1:]], *change[1:],
                number(net)/interest if interest else np.nan)
        g = geometry[week, issue]
        raw = number(g["raw_kbbl"])/1000
        overlap = number(g["overlap_kbbl"])/1000
        coverage = number(g["baseline_b04_space_time_fraction"])
        if np.isfinite(raw) and (not np.isfinite(overlap) or not 0 < coverage <= 1):
            raise ValueError(f"Incomplete optical features on {issue}")
        optical = (raw, overlap, overlap*np.sqrt(coverage)) if np.isfinite(raw) else None
        yield dict(r, cutoff=cutoff, known_at=datetime.fromisoformat(r["target_known_by_utc"]),
                   prev_wpsr_issue=prev, wti_feature_session=curve_day, cot_feature_available_utc=cot_time,
                   mean4=mean4, core=core, wide=wide, optical=optical, optical_coverage=coverage)


def transformed(rows, row, key):
    x = np.array([r[key] for r in rows])
    observed = np.isfinite(x)
    count = observed.sum(axis=0)
    centre = np.divide(np.nansum(x, axis=0), count, out=np.zeros(x.shape[1]), where=count > 0)
    x = np.where(observed, x-centre, 0.)
    scale = np.sqrt(np.mean(x*x, axis=0))
    scale[scale == 0] = 1
    z = np.array(row[key])
    return np.clip(x/scale, -4, 4), np.clip(np.where(np.isfinite(z), z-centre, 0.)/scale, -4, 4)


def ridge_fit(x, y, z, penalty, robust=False):
    weights = np.ones(len(y))
    design = np.column_stack((np.ones(len(x)), x))
    reg = penalty*len(y)*np.diag(np.r_[0., np.ones(x.shape[1])])
    for _ in range(5 if robust else 1):
        coeff = solve(design.T@(weights[:, None]*design)+reg, design.T@(weights*y), assume_a="pos")
        error = np.abs(y-design@coeff)
        weights = np.minimum(1., 0.9/np.maximum(error, 1e-9))
    return float(np.r_[1., z]@coeff)


def public_predictions(history, row):
    core, core_z = transformed(history, row, "core")
    wide, wide_z = transformed(history, row, "wide")
    recent = history[-104:]
    recent_x, recent_z = transformed(recent, row, "wide")
    y = np.array([r["label"] for r in history])
    # A smooth kernel limits the contribution of isolated market states.
    width = 2*wide.shape[1]
    kernel = np.exp(-cdist(wide, wide, "sqeuclidean")/width)
    kernel.flat[::len(y)+1] += 0.06*len(y)
    response = solve(kernel, y-y.mean(), assume_a="pos")
    nonlinear = float(y.mean()+np.exp(-np.sum((wide-wide_z)**2, axis=1)/width)@response)
    return np.array([0., row["prior_million_bbl"], row["mean4"], ridge_fit(core,y,core_z,0.4),
                     ridge_fit(wide,y,wide_z,1.), ridge_fit(recent_x,y[-len(recent):],recent_z,1.),
                     ridge_fit(wide,y,wide_z,0.5,robust=True), nonlinear])


def optical_predictions(history, row):
    raw, overlap, _ = row["optical"]
    values = [0., .25*raw, .5*raw, raw, .5*overlap]
    x = np.array([r["optical"] for r in history]).reshape(-1, 3)
    y = np.array([r["label"]-r["public_estimate"] for r in history])
    for penalty in (0.03, 0.15):
        coeff = nnls(np.vstack((x, np.eye(3)*np.sqrt(penalty*len(x)))), np.r_[y, np.zeros(3)])[0] if len(x) else np.zeros(3)
        values.append(float(row["optical"]@coeff))
    return np.array(values)


def adaptive_weights(history, values_key, actual_key, warm_start):
    scored = [r for r in history if values_key in r][-78:]
    if len(scored) < 26:
        return np.asarray(warm_start)/sum(warm_start)
    error = np.mean([np.abs(r[values_key]-r[actual_key]) for r in scored], axis=0)
    weight = np.exp(-(error-error.min())/0.10)
    return weight/weight.sum()


def replay(rows):
    history = []
    for r in sorted(rows, key=lambda r: r["cutoff"]):
        known = [q for q in history if q["known_at"] < r["cutoff"]]
        history.append(r)
        if len(known) < 52:
            continue
        r["public_candidates"] = public_predictions(known,r)
        weights = adaptive_weights(known,"public_candidates","label",(1,)*len(PUBLIC))
        r["public_estimate"] = float(weights@r["public_candidates"])
        r["public_error"] = r["label"]-r["public_estimate"]
        correction = 0.
        optical_weights = None
        measured = [q for q in known if "optical_candidates" in q]
        if r["optical"] is not None:
            r["optical_candidates"] = optical_predictions(measured,r)
            optical_weights = adaptive_weights(measured,"optical_candidates","public_error",(2,1,1,0,1,0,0))
            correction = float(optical_weights@r["optical_candidates"])
        yield {k:r[k] for k in ("study_phase","week_ending","release_date","forecast_cutoff_utc","target_known_by_utc",
               "prev_wpsr_issue","wti_feature_session","cot_feature_available_utc","target_million_bbl","optical_available")} | dict(
               training_weeks=len(known), optical_residual_training_weeks=len(measured),
               latest_training_release_available_utc=max(q["known_at"] for q in known).isoformat(),
               optical_space_time_fraction=r["optical_coverage"] if r["optical"] is not None else None,
               zero=0., prior=r["prior_million_bbl"], prior_plus_raw=r["prior_million_bbl"]+(r["raw_optical_million_bbl"] or 0.),
               public_ensemble=r["public_estimate"], optical_ensemble=r["public_estimate"]+correction,
               optical_correction_million_bbl=correction,
               **{f"public_weight_{name}": value for name,value in zip(PUBLIC,weights)},
               **{f"optical_weight_{name}": value for name,value in zip(OPTICAL,optical_weights if optical_weights is not None else [None]*len(OPTICAL))})


def block_bootstrap_mean(values):
    values = np.asarray(values)
    if len(values) < 8:
        return None
    rng = np.random.default_rng(74125)
    starts = rng.integers(len(values),size=(4000,int(np.ceil(len(values)/4))))
    index = ((starts[:,:,None]+np.arange(4))%len(values)).reshape(4000,-1)[:,:len(values)]
    return np.quantile(values[index].mean(axis=1),[.025,.975]).tolist()


def scores(rows):
    y = np.array([r["target_million_bbl"] for r in rows])
    result = {"weeks": len(rows), "methods": {}}
    for method in METHODS:
        selected = np.array([r.get(method) is not None for r in rows])
        estimate = np.array([r[method] for r in rows if r.get(method) is not None])
        if not len(estimate):
            continue
        actual = y[selected]
        error = estimate-actual
        result["methods"][method] = dict(weeks=len(estimate),mae_million_bbl=float(np.mean(abs(error))),
            rmse_million_bbl=float(np.sqrt(np.mean(error**2))), correct_nonzero_direction=int(sum(actual*estimate>0)))
    diff = [abs(r["optical_ensemble"]-t)-abs(r["public_ensemble"]-t) for r,t in zip(rows,y)]
    result["paired_optical_minus_public_mae_95pct_block_bootstrap"] = block_bootstrap_mean(diff)
    return result


def main():
    rows = list(feature_rows())
    forecasts = list(replay(rows))
    legacy = {(r["week_ending"],r["release_date"]):r for r in read_csv("cushing_eia_walk_forward_forecasts")}
    for r in forecasts:
        old = legacy.get((r["week_ending"],r["release_date"]))
        if old and (datetime.fromisoformat(old["forecast_cutoff_utc"]) != datetime.fromisoformat(r["forecast_cutoff_utc"])
                    or float(old["target_million_bbl"]) != r["target_million_bbl"]):
            raise ValueError(f"Legacy baseline misaligned on {r['release_date']}")
        r.update(legacy_ridge_public=float(old["ridge_public"]) if old else None,
                 legacy_ridge_optical=float(old["ridge_public_optical"]) if old else None)
    report = dict(source_issues=len(rows), forecasts=len(forecasts), public_candidates=PUBLIC, optical_candidates=OPTICAL,
        per_issue_candidate_predictions={"public":len(forecasts)*len(PUBLIC),
            "optical":sum(r["optical_available"] for r in forecasts)*len(OPTICAL)},
        evaluation={phase:{scope:scores([r for r in forecasts if r["study_phase"]==phase
            and ("with_optical" not in scope or r["optical_available"])
            and ("legacy_matched" not in scope or r["legacy_ridge_public"] is not None)])
            for scope in ("all","legacy_matched","with_optical","with_optical_legacy_matched")} for phase in PHASES},
        interpretation="One latest reporting week per release. Target and adaptive weights use only already published outcomes. "
        "The 2023 period has been inspected in prior experiments. 2024-2026 targets are excluded.")
    export(f"{STEM}_forecasts", forecasts)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    for phase,scopes in report["evaluation"].items():
        for scope, result in scopes.items():
            print(phase,scope,result["weeks"],{k:round(m["mae_million_bbl"],4) for k,m in result["methods"].items()})


if __name__ == "__main__":
    main()
