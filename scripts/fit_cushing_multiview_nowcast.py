"""Combine pre-release forecasts and optical measurements on matched EIA vintages."""
import json
from datetime import datetime

import numpy as np

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write
from fit_cushing_adaptive_nowcast import FREEZE_DAY, STEM as ADAPTIVE_STEM
from fit_cushing_nowcast_ensemble import PHASES, block_bootstrap_mean

STEM = "cushing_multiview_nowcast"
PUBLIC = ("prior", "previous_public", "adaptive_public")
MEASURED = ("prior_plus_raw", "prior_plus_multiview", "previous_optical", "adaptive_optical", "adaptive_plus_multiview")
METHODS = (*PUBLIC, *MEASURED, "stack_raw", "stack_multiview")
PROFILES = {"short": (5.,.08), "mid": (13.,.12), "long": (26.,.18)}


def optical_views():
    tables = {
        "registered": ("alternate_registration", None, "expanded_matched_tanks", ("expanded_satellite_change_kbbl",)),
        "spectral": ("weekly_multiband_recovery", None, "original_tanks", ("nir_expanded_visible_change_kbbl","joint_expanded_visible_change_kbbl")),
        "alignment": ("weekly_refined_alignment", "all_modes_resolved", "scene_matched_tanks", ("coarse_visible_change_kbbl","local_visible_change_kbbl","scene_visible_change_kbbl")),
    }
    values = {}
    for name,(stem,scope,count,fields) in tables.items():
        seen = set()
        for r in read_csv(f"sentinel2_{stem}_eia_comparison"):
            key = r["release_date"],r["week_ending"]
            if (scope and r["comparison_scope"]!=scope) or not int(r[count]) or not all(r[k] for k in fields):
                continue
            if key in seen:
                raise ValueError(f"Duplicate optical view for {name}/{key}")
            seen.add(key)
            values.setdefault(key,{})[name] = float(np.median([float(r[k])/1000 for k in fields]))
    return values


def features():
    views = optical_views()
    for r in read_csv(f"{ADAPTIVE_STEM}_extended_forecasts"):
        if r["release_date"]>"2023-12-31" or r["study_phase"] not in PHASES:
            raise ValueError(f"Unusable open-period predecessor forecast on {r['release_date']}")
        yield feature_row(r,views.get((r["release_date"],r["week_ending"]),{}))


def feature_row(r, alternatives):
    issue,week = r["release_date"],r["week_ending"]
    cutoff,known = (datetime.fromisoformat(r[k]) for k in ("forecast_cutoff_utc","target_known_by_utc"))
    if not datetime.fromisoformat(r["latest_training_release_available_utc"])<cutoff<known:
        raise ValueError(f"Unusable predecessor forecast on {issue}")
    estimate = {k:float(r[k]) for k in (*PUBLIC,"previous_optical","adaptive_optical","prior_plus_raw")}
    measured = r["optical_available"] in (True,"True")
    raw = estimate["prior_plus_raw"]-estimate["prior"]
    alternatives = alternatives if measured else {}
    combined = float(np.median([raw,*alternatives.values()])) if measured else 0.
    estimate.update(prior_plus_multiview=estimate["prior"]+combined,
                    adaptive_plus_multiview=estimate["adaptive_public"]+combined)
    return dict(study_phase=r["study_phase"],week_ending=week,release_date=issue,cutoff=cutoff,known=known,
        target=float(r["target_million_bbl"]),measured=measured,multi_sources=len(alternatives),raw=raw,combined=combined,**estimate)


def forecasts(known,row,methods):
    values = np.array([row[m] for m in methods])
    historical = [r for r in known if r["measured"]==row["measured"]][-104:]
    errors = np.array([[abs(r[m]-r["target"]) for m in methods] for r in historical])
    age = np.array([(row["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in historical])
    predicted = []
    for half_life, temperature in PROFILES.values():
        if len(historical)<12:
            predicted.append(float(np.mean(values)))
            continue
        loss = np.average(errors,axis=0,weights=np.exp2(-age/half_life))
        score = np.exp(-(loss-loss.min())/temperature)
        predicted.append(float(values@(.92*score/score.sum()+.08/len(score))))
    return predicted


def select_policy(known, key, measured):
    previous = [r for r in known if r["release_date"]<FREEZE_DAY and r["measured"]==measured]
    if len(previous)<20:
        return tuple(PROFILES).index("mid")
    loss = np.mean([np.abs(np.array(r[key])-r["target"]) for r in previous],axis=0)
    return int(np.argmin(loss))


def replay(rows):
    history = []
    for r in rows:
        known = [q for q in history if q["known"]<r["cutoff"]]
        raw_methods = ("prior_plus_raw","previous_optical","adaptive_optical") if r["measured"] else PUBLIC
        r["stack_raw_variants"] = forecasts(known,r,raw_methods)
        r["stack_multiview_variants"] = forecasts(known,r,MEASURED if r["measured"] else PUBLIC)
        for method in ("stack_raw","stack_multiview"):
            key = f"{method}_variants"
            profile = select_policy(known,key,r["measured"])
            r[method] = r[key][profile]
            r[f"{method}_profile"] = tuple(PROFILES)[profile]
        history.append(r)
        yield {k:v for k,v in r.items() if k not in ("stack_raw_variants","stack_multiview_variants","known","cutoff")} | dict(
            forecast_cutoff_utc=r["cutoff"].isoformat(),target_known_by_utc=r["known"].isoformat(),
            latest_stack_training_release_available_utc=max((q["known"].isoformat() for q in known),default=None))


def score(rows):
    actual = np.array([r["target"] for r in rows])
    results = {}
    for method in METHODS:
        estimate = np.array([r[method] for r in rows])
        error = estimate-actual
        results[method] = dict(mae_million_bbl=float(np.mean(abs(error))),rmse_million_bbl=float(np.sqrt(np.mean(error**2))),
                               correct_direction=int(np.sum(estimate*actual>0)))
    difference = [abs(r["stack_multiview"]-r["target"])-abs(r["prior_plus_raw"]-r["target"]) for r in rows]
    return dict(weeks=len(rows),weeks_with_additional_optical_view=sum(r["multi_sources"]>0 for r in rows),methods=results,
        multiview_minus_prior_plus_raw_mae_95pct_interval=block_bootstrap_mean(difference))


def main():
    predictions = list(replay(features()))
    metrics = {phase:{scope:score([r for r in predictions if r["study_phase"]==phase and (scope=="all" or r["measured"])])
               for scope in ("all","with_optical")} for phase in PHASES}
    export(f"{STEM}_forecasts",predictions)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(dict(evaluation=metrics,weight_profiles=PROFILES,
        interpretation="One matching cohort per issue. Optical views come from previously screened pre-release inputs. "
        "Stack weights use older published errors. Weight-profile selection is frozen on releases before 2023."),
        indent=2,allow_nan=False)+"\n").encode())
    for phase,scopes in metrics.items():
        for scope,result in scopes.items():
            print(phase,scope,result["weeks"],{k:round(m["mae_million_bbl"],4) for k,m in result["methods"].items()})


if __name__ == "__main__":
    main()
