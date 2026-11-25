"""Evaluate independently registered gap recovery in the inventory and trading models."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import FIXED_MODELS, predict, preparation
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, NOISE_FLOOR_M, PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_robust_stack import indexed
from fit_cushing_spectral_recovery import SPECTRAL_SIGMA_M, source_rows
from publication_clocks import EASTERN, utc

STEM = "cushing_recovered_optical"
MODELS = (PUBLIC,"carried_scene","adaptive_prior_and_timing","wider_red","wider_multispectral","all_recovered")
SIDES = ("earlier","later")


def widened_candidates():
    red = defaultdict(dict)
    infrared = defaultdict(dict)
    ordered = sorted(read("sentinel2_wider_registration_gap_pairs"),key=lambda r:r["pair_source"]!="primary")
    for r in ordered:
        if r["wider_registration_status"]=="checked_offset" and r["registered_both_shadow_diagnostics_pass"]=="True":
            red[r["release_date"]].setdefault(r["candidate_id"],r)
    ordered = sorted(read("cushing_wider_spectral_pairs"),key=lambda r:r["alternate_capture_pair"]=="True")
    for r in ordered:
        if r["nir_resolved"]=="True" and r["original_red_resolved"]=="False":
            infrared[r["release_date"]].setdefault(r["candidate_id"],r)
    return red,infrared


def recover(row, calendar, before, source, kind):
    cutoff = row["cutoff"]
    red = kind=="red"
    start,end = [datetime.fromisoformat(calendar[k]) for k in ("previous_measurement_at_utc","current_measurement_at_utc")]
    local_start = start.astimezone(EASTERN)
    earlier = datetime.fromisoformat(utc(local_start.date()-timedelta(days=7),local_start.time()))
    scene_fields = [f"{s}_version" if red else f"{s}_sentinel2_version" for s in SIDES]
    dates = [datetime.fromisoformat(source[f"{s}_capture_at_utc"]) for s in SIDES]
    availability = ([source[f"{s}_archive_available_at_utc"] for s in SIDES] if red else
                    [source["archive_available_at_utc"],*(source[f"{s}_mask_available_at_utc"] for s in SIDES)])
    available = max(map(datetime.fromisoformat,availability))
    recorded_cutoff = datetime.fromisoformat(source["inputs_available_before_utc" if red else "cutoff_utc"])
    if cutoff!=recorded_cutoff or not earlier<=dates[0]<=start<dates[1]<=end or not dates[1]<=available<cutoff:
        raise ValueError(f"Invalid times for {kind} recovery on {row['release_date']}/{source['candidate_id']}")
    area = float(source["circular_area_m2"])
    per_m = area/BARREL_M3/1e6
    volume = float(source["capture_change_kbbl" if red else "nir_change_kbbl"])/1000
    gaps = [float(source[f"{s}_registered_gap_m" if red else f"nir_{s}_gap_m"]) for s in SIDES]
    if area<=0 or source[scene_fields[0]]==source[scene_fields[1]] or not math.isclose((gaps[0]-gaps[1])*per_m,volume,abs_tol=1e-10):
        raise ValueError(f"Recovered height and volume differ on {row['release_date']}/{source['candidate_id']}")
    frac_before,frac_after = (start-dates[0])/(start-earlier),(dates[1]-start)/(end-start)
    duration = frac_before+frac_after
    fraction = area/float(calendar["dated_reference_area_m2"])
    width = (float(source["registered_loss_range_max_m"])-float(source["registered_loss_range_min_m"])) if red else 0.
    floor = math.hypot(NOISE_FLOOR_M,width/2) if red else SPECTRAL_SIGMA_M
    return dict(candidate_id=source["candidate_id"],terminal_id=source["terminal_id"],pair_signal=volume,
        prior_component=fraction*frac_before*before,sensitivity=fraction*frac_after,volume_per_m=per_m,area=area,
        variance=per_m**2*(floor**2+LOCAL_ACTIVITY_M**2*duration),duration=duration,
        scene_before=source[scene_fields[0]],scene_after=source[scene_fields[1]],available_utc=available.isoformat())


def combined_sources():
    dates = indexed("holdout_optical_weeks")
    public = indexed("cushing_robust_stack_forecasts")
    red,infrared = widened_candidates()
    for row,_ in source_rows():
        issue = row["release_date"]
        original = row["observed"]
        used = {r["candidate_id"] for r in original}
        if red[issue] or infrared[issue]:
            dated = dates[issue]
            prior = float(public[issue]["prior"])
            red_additions = [recover(row,dated,prior,r,"red") for tank,r in red[issue].items() if tank not in used]
            others = used | {r["candidate_id"] for r in red_additions}
            nir_additions = [recover(row,dated,prior,r,"nir") for tank,r in infrared[issue].items() if tank not in others]
        else:
            red_additions,nir_additions = [],[]
        expanded = original+red_additions+nir_additions
        seen = {r["candidate_id"] for r in expanded}
        # Preserve a single estimate per roof and prefer an independently resolved B04 pair.
        broad = expanded+[r for r in row["spectral"] if r["candidate_id"] not in seen]
        yield row, (original,original+red_additions,expanded,broad), (len(red_additions),len(nir_additions))


def replay():
    baseline = indexed("cushing_intraweek_forecasts")
    geometry = {r["release_date"]:float(r["dated_reference_area_m2"]) for stem in (
        "sentinel2_space_time_coverage_weeks","holdout_optical_weeks") for r in read(stem)}
    book = defaultdict(list)
    errors = []
    for r,arms,additions in combined_sources():
        issue,cutoff,public = (r[k] for k in ("release_date","cutoff","public"))
        historical = [q for q in errors if q["known"]<cutoff]
        values = {}
        for name,observed in zip(MODELS[2:],arms,strict=True):
            if len({q["candidate_id"] for q in observed})!=len(observed):
                raise ValueError(f"Repeated roof observation on {issue}/{name}")
            fraction = math.fsum(q["area"] for q in observed)/geometry[issue]
            current = r | dict(observed=observed,area_fraction=fraction)
            history = [q for q in book[name] if q["known"]<cutoff]
            pred = predict(current,history,historical)
            values[name] = pred["adaptive_prior_and_timing"]
            if observed:
                error = np.abs([pred[m]-r["actual"] for m in FIXED_MODELS])
                book[name].append(current | dict(prepared=preparation(current,r["actual"]),absolute_errors=error))
        base = baseline[issue]
        if not math.isclose(values[MODELS[2]],float(base[MODELS[2]]),abs_tol=1e-10):
            raise ValueError(f"The original adaptive forecast differs on {issue}")
        errors.append(dict(known=r["known"],error=r["actual"]-public))
        yield r | values | dict(public_ensemble=public,baseline_tanks=len(arms[0]),wider_red_added=additions[0],
            wider_nir_added=additions[1],total_recovered_tanks=len(arms[-1]),
            last_training_label_utc=historical[-1]["known"].isoformat() if historical else None)


def forward(issue, *, scoring):
    from fit_cushing_intraweek import replay as intraweek_history
    from register_cushing_prerelease_forecast import digest, freeze, score

    prefix = f"cushing_prerelease_forecast_{issue}"
    original_path = DERIVED/f"{prefix}.json"
    path = DERIVED/f"{prefix}_recovered_optical.json"
    original = json.loads(original_path.read_bytes())
    cutoff,deadline = (datetime.fromisoformat(original[k]) for k in ("information_cutoff_utc","registration_deadline_utc"))
    if not path.exists() and not scoring:
        if datetime.now(timezone.utc)>=deadline:
            raise ValueError("The release has passed its registration deadline")
        coverage_path = DERIVED/f"cushing_forward_optical_coverage_{issue}_summary.json"
        native_path = DERIVED/f"cushing_forward_native_optical_{issue}_summary.json"
        pair_stem = f"cushing_wider_spectral_{issue}_pairs"
        cover = json.loads(coverage_path.read_bytes())
        by_release = indexed("eia_wpsr_forward_releases")
        previous = by_release[original["prior_release"]]
        if datetime.fromisoformat(previous["conservative_postrelease_from_utc"])>=cutoff or previous["source_sha256"]!=original["prior_release_sha256"]:
            raise ValueError("The previous inventory report does not match the frozen vintage")
        total_area = float(json.loads(native_path.read_bytes())["dated_roof_area_m2"])
        calendar = dict(previous_measurement_at_utc=cover["preceding_week_end_utc"],current_measurement_at_utc=cover["target_week_end_utc"],
                        dated_reference_area_m2=total_area)
        observed = [r | dict(area=r["volume_per_m"]*BARREL_M3*1e6) for r in original["measured_pairs"]]
        base = dict(release_date=issue,week_ending=original["week_ending"],cutoff=cutoff,public=original["public_ensemble_kbbl"]/1000,
                    observed=observed,area_fraction=original["measured_roof_area_fraction"])
        selected = {}
        occupied = {r["candidate_id"] for r in observed}
        for candidate in sorted(read(pair_stem),key=lambda r:(r["earlier_sentinel2_version"],r["later_sentinel2_version"])):
            tank = candidate["candidate_id"]
            if candidate["nir_and_joint_agree_in_sign"]=="True" and tank not in occupied and tank not in selected:
                selected[tank] = recover(base,calendar,float(previous["cushing_change_kbbl"])/1000,candidate,"nir")
        augmented = observed+list(selected.values())
        larger = base | dict(observed=augmented,area_fraction=math.fsum(r["area"] for r in augmented)/total_area)
        with threadpool_limits(limits=1):
            history = [r for r in intraweek_history() if r["known"]<cutoff]
            measured = [r | dict(prepared=preparation(r,r["actual"]),absolute_errors=np.abs([r[m]-r["actual"] for m in FIXED_MODELS]))
                        for r in history if r["tanks"]]
            errors = [dict(error=r["actual"]-r[PUBLIC]) for r in history]
            field = "adaptive_prior_and_timing"
            baseline, expanded = [1000*predict(r,measured,errors)[field] for r in (base,larger)]
        prior_path = DERIVED/f"{prefix}_intraweek.json"
        fixed = json.loads(prior_path.read_bytes())
        if not math.isclose(baseline,fixed["estimates_kbbl"][field],abs_tol=1e-8):
            raise ValueError("The original forecast does not reproduce on the frozen observations")
        now = datetime.now(timezone.utc)
        if not cutoff<=now<deadline:
            raise ValueError("The new forecast missed the registration deadline")
        sources = (original_path,prior_path,coverage_path,native_path,DERIVED/f"{pair_stem}.csv",DERIVED/f"cushing_wider_spectral_{issue}_masks.csv")
        record = dict(release_date=issue, week_ending=base["week_ending"], information_cutoff_utc=cutoff.isoformat(),
                      registration_deadline_utc=deadline.isoformat(), recorded_at_utc=now.isoformat(),
                      latest_training_label_utc=history[-1]["known"].isoformat(), baseline_roofs=len(observed),
                      expanded_roofs=len(augmented), recovered_roofs=list(selected.values()),
                      baseline_kbbl=baseline,expanded_kbbl=expanded,public_kbbl=original["public_ensemble_kbbl"],
                      source_sha256={p.name:digest(p) for p in sources})
        freeze(path,record)
        return record
    record = json.loads(path.read_bytes())
    stamp = datetime.fromisoformat(record["recorded_at_utc"])
    if not datetime.fromisoformat(record["latest_training_label_utc"])<cutoff<=stamp<deadline or any(
        digest(DERIVED/name)!=fingerprint for name,fingerprint in record["source_sha256"].items()
    ):
        raise ValueError("The registered spectral forecast has conflicting timestamps or source bytes")
    if not scoring:
        return record
    outcome = score(original_path,original)
    result = dict(status=outcome["status"],release_date=issue,forecast_sha256=digest(path))
    if outcome["status"]=="scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual,first_release_sha256=outcome["first_release_sha256"],
                      absolute_error_kbbl={key:abs(actual-record[f"{key}_kbbl"]) for key in ("public","baseline","expanded")})
    write(path.with_name(f"{path.stem}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--register-release",type=date.fromisoformat)
    mode.add_argument("--score-release",type=date.fromisoformat)
    args = parser.parse_args()
    issue = args.register_release or args.score_release
    if issue:
        record = forward(issue.isoformat(),scoring=args.score_release is not None)
        print(json.dumps({k:v for k,v in record.items() if k!="recovered_roofs"},indent=2,allow_nan=False))
        return
    with threadpool_limits(limits=1):
        forecasts = list(replay())
        trades = list(trading(forecasts,MODELS))
    report = dict(status="historical_walk_forward_on_reviewed_period",forecasts={},trading={})
    for phase in PHASES:
        rows = [r for r in forecasts if r["study_phase"]==phase]
        report["forecasts"][phase] = {}
        for name,sample in (("all",rows),("measured",[r for r in rows if r["baseline_tanks"]]),
                            ("wider_recovery",[r for r in rows if r["wider_red_added"]+r["wider_nir_added"]])):
            metrics = forecast_metrics(sample,MODELS)
            for model,result in metrics["models"].items():
                differences = [1000*(abs(r["actual"]-r[model])-abs(r["actual"]-r[MODELS[2]])) for r in sample]
                result["paired_delta_vs_adaptive_95_kbbl"] = block_bootstrap_mean(differences)
            report["forecasts"][phase][name] = metrics
        base = [r for r in trades if r["study_phase"]==phase and r["method"]==MODELS[2]]
        result = report["trading"][phase] = {}
        for method in ("curve",*MODELS):
            rows = [r for r in trades if r["study_phase"]==phase and r["method"]==method]
            if [r["release_date"] for r in rows]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different comparison calendar for {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(rows,base,strict=True)]
            result[method] = robustness(rows) | daily_risk(rows) | dict(incremental_vs_adaptive_usd=sum(delta),
                paired_net_vs_adaptive_95_usd=block_bootstrap_mean(delta), net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in rows))
        print(phase,{m:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in result.items()})
    fields = ("release_date","study_phase","week_ending","actual","baseline_tanks","wider_red_added","wider_nir_added",
              "total_recovered_tanks","last_training_label_utc",*MODELS)
    export(f"{STEM}_forecasts",[{k:r[k] for k in fields} | dict(cutoff_utc=r["cutoff"].isoformat(),known_utc=r["known"].isoformat())
                                for r in forecasts])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
