"""Test previously unused joint-band height pairs in the inventory and CL models."""

import json
import math
from datetime import datetime

from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import FIXED_MODELS, predict, preparation
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed
from fit_cushing_spectral_recovery import source_rows

STEM = "cushing_joint_band_expansion"
BASE = "all_recovered"
MODELS = (PUBLIC,"carried_scene",BASE,"joint_expanded","joint_agreement")
MODE_METHODS = ("joint_expanded","joint_agreement")


def forecast(r, observed, area, history, errors):
    state = r | dict(observed=observed,area_fraction=math.fsum(q["area"] for q in observed)/area)
    if (len({q["candidate_id"] for q in observed})!=len(observed) or
            any(datetime.fromisoformat(q["available_utc"])>=r["cutoff"] for q in observed)):
        raise ValueError(f"Invalid optical input in {r['release_date']}")
    eligible = [q for q in history if q["known"]<r["cutoff"]]
    estimate = predict(state,eligible,errors)
    if observed:
        history.append(dict(cutoff=r["cutoff"],known=r["known"],prepared=preparation(state,r["actual"]),
                      absolute_errors=[abs(estimate[k]-r["actual"]) for k in FIXED_MODELS]))
    return estimate["adaptive_prior_and_timing"], eligible[-1]["known"].isoformat() if eligible else None


def replay():
    geometry = {(q["release_date"],q["week_ending"]):q for stem in ("sentinel2_space_time_coverage_weeks","holdout_optical_weeks")
                for q in read(stem)}
    refs = indexed("cushing_recovered_optical_forecasts")
    history = {k:[] for k in MODE_METHODS}
    errors = []
    measurements = []
    for (r, arms, _), (other, _) in zip(combined_sources(),source_rows("joint"),strict=True):
        issue, cutoff, known = (r[k] for k in ("release_date","cutoff","known"))
        f = refs[issue]
        if any(r[k]!=other[k] for k in ("release_date","week_ending","study_phase","cutoff","known","actual")) or not cutoff<known:
            raise ValueError(f"Joint-band source disagrees with baseline on {issue}")
        if (known!=datetime.fromisoformat(f["known_utc"]) or cutoff!=datetime.fromisoformat(f["cutoff_utc"])
                or r["actual"]!=float(f["actual"])):
            raise ValueError(f"Different archived forecast on {issue}")
        area = float(geometry[issue,r["week_ending"]]["dated_reference_area_m2"])
        base = arms[-1]
        existing = {q["candidate_id"]:q for q in base}
        additions = {q["candidate_id"]:q for q in other["spectral"] if q["candidate_id"] not in existing}
        preferred = {q["candidate_id"] for q in arms[-2]}
        replacements = {}
        for q in other["spectral"]:
            key = q["candidate_id"]
            old = existing.get(key)
            if not old or key in preferred:
                continue
            if any(q[k]!=old[k] for k in ("scene_before","scene_after","area","duration","sensitivity","prior_component")):
                continue
            if q["pair_signal"]*old["pair_signal"]>0:
                replacements[key] = old | dict(
                    pair_signal=.5*(q["pair_signal"]+old["pair_signal"]),
                    available_utc=max(q["available_utc"],old["available_utc"]))
        expanded = base+list(additions.values())
        consensus = [replacements.get(q["candidate_id"],q) for q in expanded]
        past = [q for q in errors if q["known"]<cutoff]
        values = {m:forecast(r,observed,area,history[m],past) for m,observed in zip(MODE_METHODS,(expanded,consensus),strict=True)}
        public = r["public"]
        errors.append(dict(error=r["actual"]-public,known=known))
        for kind, group in (("additional_joint_fit",additions),("same_pair_agreement",replacements)):
            for q in group.values():
                measurements.append(dict(release_date=issue,study_phase=r["study_phase"],method=kind,
                    cutoff_utc=cutoff.isoformat(),**q))
        yield r | dict(public_ensemble=public,carried_scene=float(f["carried_scene"]),all_recovered=float(f[BASE]),
            baseline_roofs=len(base),joint_added_roofs=len(additions),same_pair_agreement_roofs=len(replacements),
            baseline_area_fraction=math.fsum(q["area"] for q in base)/area,
            expanded_area_fraction=math.fsum(q["area"] for q in expanded)/area,
            **{m:value for m,(value,_) in values.items()},
            **{f"{m}_training_label_utc":stamp for m,(_,stamp) in values.items()}), measurements
        measurements = []


def main():
    with threadpool_limits(limits=1):
        series = list(replay())
        trades = list(trading([r for r,_ in series],MODELS))
    report = dict(status="exploratory_on_reviewed_history", phases={}, trading={})
    for phase in PHASES:
        rows = [r for r,_ in series if r["study_phase"]==phase]
        scopes = {"all":rows, "new_roofs":[r for r in rows if r["joint_added_roofs"]],
                  "recovered_week":[r for r in rows if r["joint_added_roofs"] and not r["baseline_roofs"]]}
        result = {}
        for scope, sample in scopes.items():
            stats = forecast_metrics(sample, MODELS)
            for m in MODE_METHODS:
                if sample:
                    diff = [1000*(abs(r["actual"]-r[m])-abs(r["actual"]-r[BASE])) for r in sample]
                    stats["models"][m]["delta_vs_current_95_kbbl"] = block_bootstrap_mean(diff)
            result[scope] = stats
        result["admission"] = dict(
            added_roofs=sum(r["joint_added_roofs"] for r in rows),
            agreed_roofs=sum(r["same_pair_agreement_roofs"] for r in rows),
            new_roof_releases=[r["release_date"] for r in scopes["new_roofs"]],
            newly_measured_releases=[r["release_date"] for r in scopes["recovered_week"]])
        report["phases"][phase] = result
        by_method = {m:[r for r in trades if r["study_phase"]==phase and r["method"]==m] for m in ("curve",*MODELS)}
        base = by_method[BASE]
        report["trading"][phase] = {}
        for m, sample in by_method.items():
            if [r["release_date"] for r in sample]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different trade dates on {phase}/{m}")
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(sample,base,strict=True)]
            report["trading"][phase][m] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample),
                delta_vs_current_95_usd=block_bootstrap_mean(delta),
                changed_decisions=sum(r["near_side"]!=b["near_side"] for r,b in zip(sample,base,strict=True)))
        print(phase,len(scopes["new_roofs"]),len(scopes["recovered_week"]),
            {m:round(q["mae_kbbl"],3) for m,q in result["all"]["models"].items()},
            {m:q["net_170_usd"] for m,q in report["trading"][phase].items()})
    fields = ("release_date","week_ending","study_phase","actual","baseline_roofs","joint_added_roofs","same_pair_agreement_roofs",
              "baseline_area_fraction","expanded_area_fraction",*MODELS,*(f"{m}_training_label_utc" for m in MODE_METHODS))
    export(f"{STEM}_forecasts",[{k:r[k] for k in fields} | dict(cutoff_utc=r["cutoff"].isoformat(),known_utc=r["known"].isoformat()) for r,_ in series])
    export(f"{STEM}_measurements",[q for _,group in series for q in group])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
