"""Model correlated terminal uncertainty for optical pairs spanning a survey week."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_outer_capture_nowcast import (
    STEM as OUTER, SUPPLEMENT, observation, outer_pairs, prepared,
)
from fit_cushing_public_state import OPTICAL, STEM as PUBLIC_STATE
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE

STEM = "cushing_spanning_pair_uncertainty"
HYBRID = "basis_when_unmeasured"
PHYSICAL = {"span_reference":(0.,False),"span_terminal_noise":(1.5,False),
            "fill_missing_terminal_noise":(1.5,True),"fill_missing_heavy_noise":(3.,True)}
MODELS = (PUBLIC,"carried_scene",OPTICAL,HYBRID,*PHYSICAL)


def geometry_uncertainty(row, inventory, outer, metres):
    model = prepared(row,inventory,outer)
    if not outer or not metres:
        return model
    pairs = model["observations"]
    exposures = defaultdict(lambda:np.zeros((len(pairs),2)))
    for idx,r in enumerate(pairs):
        # The preceding and following portions of a spanning pair can pick up
        # terminal-specific movements unrelated to the target survey week.
        moments = np.sqrt([r.get("pre_fraction",0.),r.get("post_fraction",0.)])
        exposures[r["terminal_id"]][idx] = r["volume_per_m"]*moments
    for loadings in exposures.values():
        model["local"] += metres**2*(loadings@loadings.T)
    return model


def forecast(current, past, errors, metres):
    public = current[PUBLIC]
    if not current["observed"]:
        return public,None
    is_outer = any("post_fraction" in r for r in current["observed"])
    cutoff = current["cutoff"]
    recent = [r["prepared"] for r in past if cutoff-r["cutoff"]<=MAX_SCENE_AGE]
    model = geometry_uncertainty(current,public,is_outer,metres)
    fitted = estimate(model,recent)
    scale = prior_scale(errors)
    increments = np.clip([s/(1/scale**2+i) for _,i,s in fitted],-1.25,1.25)
    # The existing timing-scale mixture scores each candidate with the
    # fixed physical prior. The blended forecast uses the adaptive prior.
    value = public+float(learned_weights(past)@increments)
    metadata = dict(cutoff=cutoff,known=current["known"],
        prepared=geometry_uncertainty(current,current["actual"],is_outer,metres),
        absolute_errors=np.abs([public+increment-current["actual"] for increment,_,_ in fitted]))
    return value,metadata


def replay():
    outer_book = indexed(f"{OUTER}_forecasts")
    state = indexed(f"{PUBLIC_STATE}_forecasts")
    public = indexed("cushing_robust_stack_forecasts")
    geometry = {(r["release_date"],r["week_ending"]):r for name in (
        "sentinel2_space_time_coverage_weeks","holdout_optical_weeks") for r in read(name)}
    spans = outer_pairs("cushing_outer_capture_pairs_fits")
    supplements = outer_pairs("cushing_outer_capture_pairs_supplement_fits")
    histories = {m:[] for m in PHYSICAL}
    errors = []
    for r,views,_ in combined_sources():
        issue, week, cutoff, known = (r[k] for k in ("release_date","week_ending","cutoff","known"))
        baseline = views[-1]
        original = outer_book[issue]
        published = state[issue]
        if (not cutoff < known or datetime.fromisoformat(original["cutoff_utc"])!=cutoff
                or datetime.fromisoformat(published["known"])!=known or float(published["actual"])!=r["actual"]
                or len(baseline)!=int(original["baseline_roofs"])):
            raise ValueError(f"Wrong input history on {issue}")
        total_area = float(geometry[issue,week]["dated_reference_area_m2"])
        prior = float(public[issue]["prior"])
        additions = [observation(o,r,total_area,prior) for o in spans[issue]]
        extra = [observation(o,r,total_area,prior) for o in supplements[issue]]
        history = [e for e in errors if e["known"]<cutoff]
        ordinary = baseline or additions
        if len(baseline)+len(additions)!=len(ordinary):
            raise ValueError(f"Full-week supplements overlap existing roofs on {issue}")
        values = {}
        for m,(metres,fill_only) in PHYSICAL.items():
            pairs = ordinary if fill_only else ordinary+extra
            if len({q["candidate_id"] for q in pairs})!=len(pairs) or any(
                    datetime.fromisoformat(q["available_utc"])>=cutoff for q in pairs):
                raise ValueError(f"Late or duplicate tank measurements on {issue}/{m}")
            row = r | dict(observed=pairs,hub_area=total_area,area_fraction=math.fsum(q["area"] for q in pairs)/total_area,
                           public_ensemble=r["public"])
            prior_scored = [q for q in histories[m] if q["known"]<cutoff]
            value,labels = forecast(row,prior_scored,history,metres)
            if labels:
                histories[m].append(labels)
            if fill_only and not pairs:
                value = float(published[HYBRID])
            values[m] = value
            values[f"{m}_latest_label_utc"] = prior_scored[-1]["known"].isoformat() if prior_scored else None
        if not math.isclose(values["span_reference"],float(original[SUPPLEMENT]),abs_tol=1e-9):
            raise ValueError(f"Original model on {issue} gave {original[SUPPLEMENT]}, replay gave {values['span_reference']}")
        yield {k:r[k] for k in ("release_date","week_ending","study_phase","actual","cutoff","known")} | values | dict(
            **{m:float(published[m]) for m in (PUBLIC,"carried_scene",OPTICAL,HYBRID)},
            baseline_roofs=len(baseline),spanning_roofs=len(additions),supplemental_roofs=len(extra))
        errors.append(dict(known=known,error=r["actual"]-r["public"]))


def inventory_metrics(sample):
    result = forecast_metrics(sample,MODELS)
    for m,stats in result["models"].items():
        delta = [1000*(abs(r["actual"]-r[m])-abs(r["actual"]-r[HYBRID])) for r in sample]
        stats.update(delta_vs_hybrid_95_kbbl=block_bootstrap_mean(delta),
                     weeks_improved_vs_hybrid=sum(v< -1e-9 for v in delta))
    return result


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        base = list(trading(rows,MODELS))
        trades = list(replay_conditional_market(rows,base,(PUBLIC,*PHYSICAL),references={
            PUBLIC:f"{PUBLIC}_conditional","span_reference":f"{SUPPLEMENT}_conditional",
        }))
    report = dict(status="exploratory_on_reviewed_history",forecasts={},market={})
    for phase in PHASES:
        panel = [q for q in rows if q["study_phase"]==phase]
        report["forecasts"][phase] = {scope:inventory_metrics(sample) for scope,sample in (
            ("all",panel),("spanning",[q for q in panel if q["spanning_roofs"]]),
            ("supplement",[q for q in panel if q["supplemental_roofs"]]))}
        market = report["market"][phase] = {}
        benchmark = [r["release_date"] for r in trades if r["study_phase"]==phase and r["method"]==PUBLIC]
        for m in (PUBLIC,*PHYSICAL):
            sample = [r for r in trades if r["study_phase"]==phase and r["method"]==m]
            if not sample or [r["release_date"] for r in sample]!=benchmark:
                raise ValueError(f"Mismatched replay calendar for {phase}/{m}")
            market[m] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample))
        print(phase,{m:round(q["mae_kbbl"],2) for m,q in report["forecasts"][phase]["all"]["models"].items()},
              {m:(q["net_170_usd"],q["net_340_usd"]) for m,q in market.items()})
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
