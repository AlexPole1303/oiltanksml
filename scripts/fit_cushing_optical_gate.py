"""Estimate an optical correction's reliability from earlier scored releases."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY as MARKET_HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, STEM as OPTICAL_STRATEGIES
from fit_cushing_public_state import OPTICAL
from fit_cushing_robust_stack import indexed

STEM = "cushing_optical_gate"
HYBRID = "basis_when_unmeasured"
FIXED = (PUBLIC, HYBRID, SUPPLEMENT, "half_optical", "public_plus_balance")
LEARNED = ("global_gain", "local_gain", "directional_gain", "learned_mix")
BLENDS = ("correction_blend", "coverage_correction_blend")
TEMPORAL = ("error_memory", "short_error_memory")
MODELS = (*FIXED, *LEARNED, *BLENDS, *TEMPORAL)
VIEWS = {
    # Peel off the fundamental prediction from its optical enhancement.
    # The hybrid already includes an update from a later physical model.
    "cushing_regional_balance_forecasts": ("weekly_monthly", "public"),
    "cushing_inventory_market_bridge_forecasts": ("optical_curve", "all_recovered"),
    "cushing_weather_and_options_forecasts": ("optical_temperature", "all_recovered"),
}
HISTORY = 104
MIN_MEASURED = 40


def source_rows():
    state = indexed("cushing_public_state_forecasts")
    outer = indexed("cushing_outer_capture_nowcast_forecasts")
    variance = indexed("cushing_optical_variance_budget_forecasts")
    stability = indexed("cushing_optical_stability_forecasts")
    views = {name:indexed(name) for name in VIEWS}
    if any(book.keys()!=state.keys() for book in (outer,variance,stability,*views.values())):
        raise ValueError("The optical inputs use different release calendars")
    for issue,q in sorted(state.items()):
        o,v,s,*extra = (book[issue] for book in (outer,variance,stability,*views.values()))
        cutoff, known = (datetime.fromisoformat(q[k]) for k in ("cutoff","known"))
        if (not cutoff < known or any(datetime.fromisoformat(r["cutoff_utc"])!=cutoff or datetime.fromisoformat(r["known_utc"])!=known
                or float(r["actual"])!=float(q["actual"]) or r["study_phase"]!=q["study_phase"] for r in (o,v,s,*extra))
                or any(datetime.fromisoformat(clock) >= cutoff for r in (q,o,v,s,*extra) for k,clock in r.items()
                       if k.endswith(("training_label_utc","latest_label_utc")) and clock)):
            raise ValueError(f"Optical training or forecast clock differs on {issue}")
        actual, public, hybrid, optical, balance = (float(q[k]) for k in ("actual",PUBLIC,HYBRID,OPTICAL,"basis_optical"))
        measured = q["measured"] == "True"
        count, spanning = (int(o[k]) for k in ("baseline_roofs","outer_roofs"))
        if (measured != (count+spanning>0) or float(o[OPTICAL])!=optical or float(o[PUBLIC])!=public
                or measured and not math.isclose(hybrid,optical,abs_tol=1e-10)):
            raise ValueError(f"Matched optical forecast differs on {issue}")
        revision = optical-public
        # A current quality flag summarizes independent tests. The scale of
        # every coordinate is fixed in physical units or in roof counts.
        # The baseline area-time panel does not cover recovered spanning pairs.
        diagnostics = np.array((math.log1p(count+spanning)/2, float(spanning>0),
            float(v["space_time_fraction"])*4, float(s["terminal_deletion_scale"])*4,
            abs(revision)*4, abs(public), abs(balance-optical)*4))
        estimates = {PUBLIC:public,HYBRID:hybrid,SUPPLEMENT:float(o[SUPPLEMENT]),
                     "half_optical":public+.5*revision if measured else hybrid,
                     "public_plus_balance":balance-revision if measured else hybrid}
        corrections = np.array([float(r[signal])-float(r[control])
                                for r,(signal,control) in zip(extra,VIEWS.values(),strict=True)])
        yield {k:q[k] for k in ("release_date","week_ending","study_phase")} | dict(
            actual=actual, cutoff=cutoff, known=known, measured=measured, revision=revision,
            balance_revision=balance-optical, diagnostics=diagnostics, corrections=corrections,
            carried_scene=float(q["carried_scene"]), **estimates)


def fit_gains(design, residual, weight, reference, penalty, *, bounds=(0.,1.75)):
    """Bounded, smoothed absolute-error regression around specified gains."""
    def loss(beta):
        error = design@beta-residual
        scale = np.hypot(.20,error)
        prior = beta-reference
        value = weight@scale + .5*np.dot(penalty,prior**2)
        grad = design.T@(weight*error/scale) + penalty*prior
        return value,grad

    result = minimize(loss,reference,jac=True,bounds=[bounds]*len(reference),method="L-BFGS-B")
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(f"Optical gain fit failed: {result.message}")
    return result.x


def gate(train, current):
    exposure = float(current["revision"])
    balance = float(current["balance_revision"])
    changes = np.array([q["revision"] for q in train])
    balance_changes = np.array([q["balance_revision"] for q in train])
    residual = np.array([q["actual"]-q[PUBLIC] for q in train])
    age = np.array([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in train])
    recency = np.exp2(-age/78)
    distance = np.square(np.stack([q["diagnostics"] for q in train])-current["diagnostics"]).mean(axis=1)
    locality = .35+.65*np.exp(-distance/2)
    one = np.ones(1)
    global_gain, = fit_gains(changes[:,None],residual,recency,one,np.array([4.]))
    local_gain, = fit_gains(changes[:,None],residual,recency*locality,np.array([global_gain]),np.array([4.]))
    sign = (changes>0)==(exposure>0)
    conditional_gain, = fit_gains(changes[:,None],residual,recency*locality*np.where(sign,1.,.15),
                                 np.array([local_gain]),np.array([4.]))
    mix = fit_gains(np.column_stack((changes,balance_changes)),residual,recency*locality,
                    np.array([local_gain,0.]),np.array([4.,8.]))
    revisions = (global_gain*exposure,local_gain*exposure,conditional_gain*exposure,
                 mix@np.array([exposure,balance]))
    return dict(zip(LEARNED,(current[PUBLIC]+float(x) for x in revisions),strict=True)), {
        "global_weight":float(global_gain),"local_weight":float(local_gain),"directional_weight":float(conditional_gain),
        "mix_optical_weight":float(mix[0]),"mix_balance_weight":float(mix[1]),
    }


def replay():
    history, measured = [], []
    for current in source_rows():
        cutoff = current["cutoff"]
        train = [q for q in measured if q["known"] < cutoff][-HISTORY:]
        active = current["measured"] and len(train) >= MIN_MEASURED
        estimates, weights = gate(train,current) if active else (dict.fromkeys(LEARNED,current[HYBRID]),{})
        previous = [q for q in history if q["known"] < cutoff][-156:]
        last = [q["actual"]-q[HYBRID] for q in previous[-4:]]
        current["error_memory_design"] = (last[-1],float(np.mean(last[:-1]))) if len(last)==4 else (0.,0.)
        blend, blend_weights = correction_blends(previous,current)
        temporal = correct_error_memory(previous,current)
        yield current | estimates | weights | blend | blend_weights | temporal | dict(training_weeks=len(train),
            blend_training_weeks=len(previous),latest_blend_training_label_utc=previous[-1]["known"].isoformat() if previous else None,
            latest_training_label_utc=train[-1]["known"].isoformat() if train else None)
        history.append(current)
        if current["measured"]:
            measured.append(current)


def correction_blends(train, current):
    baseline = current[HYBRID]
    if len(train)<52:
        return dict.fromkeys(BLENDS,baseline),{}
    x = np.stack([r["corrections"] for r in train])
    y = np.array([r["actual"]-r[HYBRID] for r in train])
    ages = np.array([(current["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in train])
    recency = np.exp2(-ages/78)
    matching = np.array([r["measured"]==current["measured"] for r in train])
    # A zero-centered penalty keeps the default weight near zero unless the
    # independent adjustment predictably removes residual error.
    penalty = np.array([6.,6.,2.])
    pooled = fit_gains(x,y,recency,np.zeros(3),penalty)
    local = fit_gains(x,y,recency*np.where(matching,1.,.35),pooled,penalty)
    values = dict(zip(BLENDS,(baseline+float(w@current["corrections"]) for w in (pooled,local)),strict=True))
    provenance = {f"{group}_{k}_weight":float(w) for group,weights in zip(("pooled","matched"),(pooled,local),strict=True)
                  for k,w in zip(("regional","curve","weather"),weights,strict=True)}
    return values,provenance


def correct_error_memory(train, current):
    if len(train)<52:
        return dict.fromkeys(TEMPORAL,current[HYBRID])
    x = np.clip(np.asarray([q["error_memory_design"] for q in train]),-2.,2.)
    y = np.array([q["actual"]-q[HYBRID] for q in train])
    ages = np.array([(current["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in train])
    recent = np.clip(current["error_memory_design"],-2.,2.)
    values = []
    for half_life in (78,26):
        recency = np.exp2(-ages/half_life)
        beta = fit_gains(x,y,recency,np.zeros(2),np.array([.3,.5])*recency.sum(),bounds=(-.5,.5))
        values.append(current[HYBRID]+float(np.clip(recent@beta,-.5,.5)))
    return dict(zip(TEMPORAL,values,strict=True))


def forecast_report(sample):
    report = forecast_metrics(sample,(*MODELS,"carried_scene"))
    report["models"].pop("carried_scene")
    errors = np.array([abs(q["actual"]-q[HYBRID]) for q in sample])
    for m, result in report["models"].items():
        difference = np.array([abs(q["actual"]-q[m]) for q in sample])-errors
        result["mean_delta_vs_hybrid_95_kbbl"] = block_bootstrap_mean(difference*1000)
        result["weeks_improved_vs_hybrid"] = int(np.count_nonzero(difference< -1e-12))
        result["weeks_worsened_vs_hybrid"] = int(np.count_nonzero(difference>1e-12))
    return report


def conditional_trading(rows, base):
    forecasts = {r["release_date"]:r for r in rows}
    estimates = {(r["release_date"],r["method"]):r for r in base}
    anchors = {(r["release_date"],r["method"]):r for r in read(f"{OPTICAL_STRATEGIES}_trades")
               if r["method"] in (f"{PUBLIC}_conditional",f"{SUPPLEMENT}_conditional")}
    history = defaultdict(list)
    for case, references in inputs():
        if "inventory_conditional" not in references:
            continue
        issue, decision, sigma = (case[k] for k in ("release_date","decision","volatility_usd"))
        forecast = forecasts[issue]
        if not forecast["cutoff"] < forecast["known"] <= decision or forecast["actual"] != case["x"][0]:
            raise ValueError(f"Bad market or inventory clock on {issue}")
        for method in MODELS:
            original = estimates[issue,method]
            training_at = datetime.fromisoformat(original["settled_training_label_utc"])
            if (not training_at < decision or any(original[k]!=case[k] for k in ("near_month","far_month"))
                    or any(datetime.fromisoformat(original[k])!=case[field] for k,field in (
                    ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))):
                raise ValueError(f"Invalid market calibration row for {issue}/{method}")
            point = original["expected_long_move_usd"]
            row = case | dict(base_estimate=point,response_features=features(case["x"][2],sigma,forecast["actual"]-forecast[method]))
            past = [q for q in history[method] if q["exit_at"]<decision][-MARKET_HISTORY:]
            correction,_ = conditional_residual(past,row) if len(past)>=MIN_WINDOWS else (0.,0.)
            history[method].append(row)
            policy = f"{method}_conditional"
            result = trade(case,policy,point+correction,3,training_at=training_at,
                           calibration_at=past[-1]["exit_at"] if past else None)
            if method in (PUBLIC,SUPPLEMENT):
                anchor = anchors[issue,policy]
                if not math.isclose(result["expected_long_move_usd"],float(anchor["expected_long_move_usd"]),abs_tol=1e-8):
                    raise ValueError(f"Original CL model changed on {issue}/{policy}")
            yield result


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        first_stage = list(trading(rows,MODELS))
        trades = list(conditional_trading(rows, first_stage))
    report = dict(status="exploratory_on_reviewed_history",forecasts={},market={})
    for phase in PHASES:
        sample = [q for q in rows if q["study_phase"]==phase]
        report["forecasts"][phase] = {group:forecast_report(subset) for group,subset in (
            ("all",sample),("measured",[q for q in sample if q["measured"]]),
            ("unmeasured",[q for q in sample if not q["measured"]]))}
        outcomes = defaultdict(list)
        for row in trades:
            if row["study_phase"]==phase:
                outcomes[row["method"]].append(row)
        public = outcomes[f"{PUBLIC}_conditional"]
        hybrid = outcomes[f"{HYBRID}_conditional"]
        market = report["market"][phase] = {}
        for method, group in outcomes.items():
            if [r["release_date"] for r in group] != [r["release_date"] for r in public]:
                raise ValueError(f"Different contracts calendar for {method} in {phase}")
            market[method] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group),
                decisions_different_from_public=sum(q["near_side"]!=r["near_side"] for r,q in zip(group,public,strict=True)),
                decisions_different_from_hybrid=sum(q["near_side"]!=r["near_side"] for r,q in zip(group,hybrid,strict=True)),
                paired_delta_vs_public_95_usd=block_bootstrap_mean([r["net_170_usd"]-q["net_170_usd"]
                                                                   for r,q in zip(group,public,strict=True)]))
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in report["forecasts"][phase]["all"]["models"].items()})
        print(phase,{m:(v["net_170_usd"],v["net_340_usd"]) for m,v in market.items()})
    fields = ("release_date","week_ending","study_phase","actual","cutoff","known","measured","training_weeks",
              "latest_training_label_utc","blend_training_weeks","latest_blend_training_label_utc",
              "revision","balance_revision",*MODELS,"global_weight","local_weight",
              "directional_weight","mix_optical_weight","mix_balance_weight",
              *(f"{group}_{k}_weight" for group in ("pooled","matched") for k in ("regional","curve","weather")))
    export(f"{STEM}_forecasts",[{key:r.get(key) for key in fields} | {
        f"{name}_correction":float(value) for name,value in zip(("regional","curve","weather"),r["corrections"],strict=True)
    } for r in rows])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
