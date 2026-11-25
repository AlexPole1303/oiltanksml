"""Walk forward on satellite coverage, terminal composition and inventory surprises."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.linalg import solve

from backtest_wti_cushing_spread import COSTS, statistics
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, inputs as market_inputs, predictions as market_predict
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed
from fit_cushing_weekly_change_ml import trade_rows

STEM = "cushing_coverage_nowcast"
PHASES = ("development", "validation", "holdout")
MIN_MEASURED = 32
FORECAST_FIELDS = ("public_ensemble", "pair_quality_ml", "coverage_ml", "terminal_coverage_ml")
MARKET_FIELDS = ("public_ml", "coverage_ml", "terminal_coverage_ml", "always_long_spread", "no_trade")


def key(r):
    return r["release_date"], r["week_ending"]


def coverage_inputs():
    phase_sources = ("sentinel2_space_time_coverage_terminal_weeks", "holdout_optical_terminal_weeks")
    dated = defaultdict(dict)
    for stem in phase_sources:
        for r in read(stem):
            if (r["study_phase"] == "holdout") != stem.startswith("holdout_"):
                continue
            issue, site = key(r), r["terminal_id"]
            if site in dated[issue]:
                raise ValueError(f"Repeated terminal coverage {issue}/{site}")
            dated[issue][site] = r
    signals = defaultdict(dict)
    for r in read("cushing_weekly_change_ml_terminals"):
        issue,site = key(r),r["terminal_id"]
        if site in signals[issue]:
            raise ValueError(f"Repeated terminal measurement {issue}/{site}")
        signals[issue][site] = r
    sites = sorted({site for group in dated.values() for site in group})
    site_features = [f"adjusted_rate:{site}" for site in sites]
    feature_names = ["raw", "smoothed", "moderated_expansion", "recent_coverage_effect", "coherent_signal"] + site_features
    for r in read("cushing_weekly_change_ml_forecasts"):
        issue, week = key(r)
        control = dated[issue, week]
        measured = signals[issue, week]
        baseline, target = (float(r[k]) for k in ("public_ensemble", "target"))
        cutoff, known = (datetime.fromisoformat(r[k]) for k in ("forecast_cutoff_utc", "target_known_by_utc"))
        if not cutoff < known or bool(measured) != (r["measured"] == "True") or not measured.keys() <= control.keys():
            raise ValueError(f"Invalid coverage cohort or chronology on {issue}")
        anchor = math.fsum(float(v["dated_reference_area_m2"]) for v in control.values())
        if anchor<=0:
            raise ValueError(f"Missing mapped hub footprint on {issue}")
        coverage = np.zeros(len(sites))
        values = np.zeros(len(sites))
        for i, site in enumerate(sites):
            if site not in measured:
                continue
            s, ref = measured[site], control[site]
            if datetime.fromisoformat(s["cutoff_utc"]) != cutoff or datetime.fromisoformat(ref["cutoff_utc"]) != cutoff:
                raise ValueError(f"Different information cutoff on {issue}/{site}")
            if not math.isclose(float(s["area_m2"]), float(ref["baseline_b04_area_m2"]), abs_tol=1e-6):
                raise ValueError(f"Different measured footprint on {issue}/{site}")
            coverage[i] = float(ref["baseline_b04_space_time_fraction"])*float(ref["dated_reference_area_m2"])/anchor
            values[i] = float(s["overlap_attenuated"])/1000
        raw = float(r["physical"])
        smooth = float(r["overlap_attenuated"])
        if not math.isclose(smooth, float(values.sum()), abs_tol=1e-9) or not 0 <= coverage.sum() <= 1.00000001:
            raise ValueError(f"Terminal shares do not reproduce the hub signal on {issue}")
        area = math.fsum(float(s["area_m2"]) for s in measured.values())/anchor
        space_time = float(coverage.sum())
        coherence = abs(smooth)/max(float(np.abs(values).sum()), 1e-9)
        # Shrink the expansion toward the directly measured signal, capping influence when coverage is thin.
        moderated = smooth/max(space_time,0.08)**0.5
        sector_rates = values/np.sqrt(np.maximum(coverage,0.015))
        x = np.r_[raw,smooth,moderated,raw*np.sqrt(space_time),smooth*coherence,sector_rates]
        yield dict(release_date=issue, week_ending=week, study_phase=r["study_phase"], cutoff=cutoff, known=known,
                   target=target, measured=bool(measured), public_ensemble=baseline, pair_quality_ml=float(r["pair_quality_ml"]),
                   x=x, feature_names=feature_names, area_fraction=area, space_time_fraction=space_time, terminals=len(measured))


def correction(history, current, width):
    x = np.array([r["x"][:width] for r in history])
    scale = np.maximum(np.sqrt(np.mean(x*x,axis=0)),0.08)
    x = np.clip(x/scale,-4.,4.)
    y = np.array([r["target"]-r["public_ensemble"] for r in history])
    age = np.array([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in history])
    base = np.exp2(-age/78.)
    penalties = np.r_[np.full(5,0.7),np.full(width-5,2.5)]*base.sum()
    weights = base.copy()
    for _ in range(6):
        beta = solve(x.T@(weights[:,None]*x)+np.diag(penalties),x.T@(weights*y),assume_a="pos")
        weights = base/np.hypot(1.,(y-x@beta)/0.65)
    update = np.clip(current["x"][:width]/scale,-4.,4.)@beta
    return float(np.clip(update,-0.75,0.75))


def forecasts(rows):
    seen = []
    for r in rows:
        train = [q for q in seen if q["measured"] and q["known"]<r["cutoff"]][-104:]
        seen.append(r)
        baseline = r["public_ensemble"]
        widths = (5,len(r["feature_names"]))
        adjusted = [baseline+correction(train,r,n) if len(train)>=MIN_MEASURED and r["measured"] else baseline for n in widths]
        yield {k:r[k] for k in ("release_date", "week_ending", "study_phase", "target", "measured", "public_ensemble", "pair_quality_ml",
                                "area_fraction", "space_time_fraction", "terminals")} | dict(
            forecast_cutoff_utc=r["cutoff"].isoformat(), target_known_by_utc=r["known"].isoformat(),
            latest_training_label_utc=max(q["known"] for q in train).isoformat() if train else None,
            measured_training_weeks=len(train), coverage_ml=adjusted[0], terminal_coverage_ml=adjusted[1])


def direct_market(rows):
    forecast = {r["release_date"]:r for r in rows}
    seen = []
    for r in market_inputs():
        f = forecast[r["release_date"]]
        if datetime.fromisoformat(f["target_known_by_utc"])>r["decision"]:
            raise ValueError(f"Trade uses unpublished inventory on {r['release_date']}")
        sample = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        coverage = f["coverage_ml"]-f["public_ensemble"]
        terminal = f["terminal_coverage_ml"]-f["public_ensemble"]
        width = 5
        current = r | {"x":(*r["x"][:width],coverage,coverage*r["x"][2],terminal-coverage,(terminal-coverage)*r["x"][2])}
        seen.append(current)
        if len(sample)<MIN_WINDOWS:
            continue
        for method,n in zip(MARKET_FIELDS,(5,7,9,0,0),strict=True):
            expected = market_predict(sample,current,n) if n else None
            side = (int(expected>170)-int(expected<-170) if n else int(method=="always_long_spread"))
            gross = r["long_gross"]*side
            yield {k:r[k] for k in ("release_date", "study_phase", "entry_session", "exit_session", "near_month", "far_month", "source")} | dict(
                method=method, near_side=side, expected_long_gross_usd=expected, gross_usd=gross,
                decision_at_utc=r["decision"].isoformat(), latest_training_exit=sample[-1]["exit_at"].isoformat(),
                **{f"cost_{c}_usd":abs(side)*c for c in COSTS}, **{f"net_{c}_usd":gross-abs(side)*c for c in COSTS})


def forecast_score(rows):
    actual = np.asarray([r["target"] for r in rows])
    base = np.abs(actual-np.asarray([r["public_ensemble"] for r in rows]))
    results = {}
    for method in FORECAST_FIELDS:
        value = np.asarray([r[method] for r in rows])
        error = np.abs(actual-value)
        results[method] = dict(mae_million_bbl=float(error.mean()), correct_direction=int(np.count_nonzero(actual*value>0)),
                               delta_mae_ci_95=block_bootstrap_mean(error-base))
    return dict(weeks=len(rows), methods=results)


def main():
    inputs = list(coverage_inputs())
    rows = list(forecasts(inputs))
    spread = list(trade_rows(rows,methods=FORECAST_FIELDS))
    direct = list(direct_market(rows))
    original = {(r["study_phase"],r["release_date"]):r for r in read("cushing_cl_regime_strategy_trades")
                if r["method"]=="public" and r["hurdle_usd"]=="170"}
    rebuilt = [r for r in direct if r["method"]=="public_ml"]
    if len(rebuilt)!=len(original) or any(r["net_170_usd"]!=int(original[r["study_phase"],r["release_date"]]["net_170_usd"]) for r in rebuilt):
        raise ValueError("Original public CL strategy did not reproduce")
    trade_groups = defaultdict(list)
    for r in spread:
        trade_groups[r["scope"],r["method"]].append(r)
    market_groups = defaultdict(list)
    for r in direct:
        market_groups[r["study_phase"],r["method"]].append(r)
    split = float(np.median([r["space_time_fraction"] for r in rows if r["study_phase"]=="development" and r["measured"]]))
    holdout = [r for r in rows if r["study_phase"]=="holdout" and r["measured"]]
    stats = dict(status="exploratory_replay_on_previously_examined_history", forecast={phase:{
        scope:forecast_score([r for r in rows if r["study_phase"]==phase and (scope=="all" or r["measured"])])
        for scope in ("all","with_optical")} for phase in PHASES},
        holdout_coverage_split=dict(development_median_space_time_fraction=split,
            below=forecast_score([r for r in holdout if r["space_time_fraction"]<split]),
            at_or_above=forecast_score([r for r in holdout if r["space_time_fraction"]>=split])),
        surprise_trades={f"{scope}:{method}":statistics(v) for (scope,method),v in sorted(trade_groups.items())},
        direct_trades={phase:{method:statistics(market_groups[phase,method]) for method in MARKET_FIELDS} for phase in PHASES})
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_surprise_trades",spread)
    export(f"{STEM}_direct_trades",direct)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(stats,indent=2,allow_nan=False)+"\n").encode())
    for phase, scores in stats["forecast"].items():
        print(phase,{k:(v["weeks"],{m:round(d["mae_million_bbl"],4) for m,d in v["methods"].items()}) for k,v in scores.items()})
    for phase, scores in stats["direct_trades"].items():
        print(phase,{m:(v["trades"],v["net_50_usd"],v["net_170_usd"]) for m,v in scores.items()})


if __name__=="__main__":
    main()
