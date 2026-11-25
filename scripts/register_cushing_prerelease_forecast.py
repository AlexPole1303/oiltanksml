"""Freeze a public and satellite Cushing forecast before the first EIA publication."""

import argparse
import hashlib
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from tempfile import NamedTemporaryFile

from audit_cushing_forward_wti_coverage import archives
from audit_cushing_forward_optical_coverage import EASTERN
from build_s1_reference_matches import DERIVED, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from evaluate_cushing_forward_public_cl import public_forecasts
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, NOISE_FLOOR_M, learned_beta, physical_update, replay
from fit_cushing_shared_scene import calibrated as shared_scene_forecast, replay as shared_scene_history
from publication_clocks import utc

STEM = "cushing_prerelease_forecast"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected_pairs(issue, summary, actual_cutoff):
    optical_summary = f"cushing_forward_optical_coverage_{issue}"
    native = f"cushing_forward_native_optical_{issue}"
    catalog = {r["l1c_product"]:r for r in read(f"{optical_summary}_tiles")}
    pairs = [r for r in read(f"{native}_pairs") if r["registered_both_shadow_diagnostics_pass"]=="True"]
    pairs.sort(key=lambda r:(float(r["registered_loss_range_max_m"])-float(r["registered_loss_range_min_m"]),
                             r["earlier_version"],r["candidate_id"]))
    selected = {}
    for r in pairs:
        clock = max(datetime.fromisoformat(catalog[r[f"{edge}_version"]]["l1c_with_detector_available_by_utc"]) for edge in ("earlier","later"))
        if clock>=actual_cutoff:
            raise ValueError(f"A selected satellite product became available late on {issue}")
        selected.setdefault(r["candidate_id"],r | dict(source_available_utc=clock.isoformat()))
    if sorted(selected)!=summary["accepted_ids"]:
        raise ValueError("The forecast tank set differs from the physical pipeline")
    return list(selected.values())


def optical_observations(pairs, summary, prior, start, end):
    previous_date = start.astimezone(EASTERN).date()-timedelta(days=7)
    previous = datetime.fromisoformat(utc(previous_date,time(7)))
    for r in pairs:
        first,last = [datetime.fromisoformat(r[f"{edge}_capture"]) for edge in ("earlier","later")]
        if not previous<=first<=start<last<=end:
            raise ValueError(f"Invalid boundary for {r['candidate_id']}")
        before,current = (start-first)/(start-previous),(last-start)/(end-start)
        share = float(r["area_m2"])/summary["dated_roof_area_m2"]
        height = float(r["registered_change_m"])
        volume = float(r["registered_change_kbbl"])/1000
        per_m = float(r["area_m2"])/BARREL_M3/1e6
        if abs(height*per_m-volume)>1e-9:
            raise ValueError(f"Height and barrel conversion differ for {r['candidate_id']}")
        width = (float(r["registered_loss_range_max_kbbl"])-float(r["registered_loss_range_min_kbbl"]))/1000
        duration = before+current
        variance = (NOISE_FLOOR_M*per_m)**2+(width/2)**2+(LOCAL_ACTIVITY_M*per_m)**2*duration
        yield dict(candidate_id=r["candidate_id"],terminal_id=r["terminal_id"],pair_signal=volume,prior_component=share*before*prior,
                   sensitivity=share*current,volume_per_m=per_m,variance=variance,duration=duration,
                   scene_before=r["earlier_version"],scene_after=r["later_version"],source_available_utc=r["source_available_utc"])


def freeze(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=path.parent,prefix=".forecast-",delete=False) as stream:
        temporary = ROOT/stream.name
        stream.write((json.dumps(payload,indent=2,allow_nan=False)+"\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def score(path,record):
    issue = record["release_date"]
    outcomes = {r["release_date"]:r for r in read("eia_wpsr_forward_releases")}
    if issue not in outcomes:
        return dict(status="awaiting_first_release",release_date=issue,forecast_sha256=digest(path))
    result = outcomes[issue]
    recorded,deadline,first_possible = [datetime.fromisoformat(v) for v in (record["recorded_at_utc"],record["registration_deadline_utc"],
                                                                            result["conservative_prerelease_before_utc"])]
    if not recorded < deadline <= first_possible or record["week_ending"]!=result["latest_week_in_issue"]:
        raise ValueError("The registered forecast does not precede its matching EIA release")
    source = ROOT/result["source_path"]
    if digest(source)!=result["source_sha256"]:
        raise ValueError("Original EIA report checksum differs from the archive")
    actual = float(result["cushing_change_kbbl"])
    error = {method:abs(actual-record[f"{method}_kbbl"]) for method in ("public_ensemble","optical_nowcast","robust_public")}
    return dict(status="scored_first_release",release_date=issue,week_ending=record["week_ending"],recorded_at_utc=record["recorded_at_utc"],
                actual_eia_change_kbbl=actual,absolute_error_kbbl=error,
                optical_minus_public_absolute_error_kbbl=error["optical_nowcast"]-error["public_ensemble"],
                forecast_sha256=digest(path),first_release_sha256=result["source_sha256"],first_release_source=result["source_url"])


def create(issue,path):
    historic = read("eia_wpsr_forward_releases")
    cutoff = datetime.fromisoformat(utc(date.fromisoformat(issue)))
    as_of = datetime.now(timezone.utc)
    if as_of>=cutoff or issue<=historic[-1]["release_date"]:
        raise ValueError("Registration must be before the day of the release")
    source = DERIVED/f"cushing_forward_native_optical_{issue}_summary.json"
    coverage_path = DERIVED/f"cushing_forward_optical_coverage_{issue}_summary.json"
    summary,coverage = (json.loads(p.read_bytes()) for p in (source,coverage_path))
    first,end = (datetime.fromisoformat(coverage[k]) for k in ("preceding_week_end_utc","target_week_end_utc"))
    if summary["outcome_available"] or not end<=as_of or summary["week_ending"]!=end.date().isoformat():
        raise ValueError("The upcoming survey window is inconsistent")
    previous = historic[-1]
    if previous["latest_week_in_issue"]!=first.date().isoformat() or datetime.fromisoformat(previous["conservative_postrelease_from_utc"])>=as_of:
        raise ValueError("The previous inventory release is missing at the registration cutoff")
    pending = dict(release_date=issue,latest_week_in_issue=end.date().isoformat(),previous_week_ending=first.date().isoformat(),
                   cushing_change_kbbl=None,conservative_prerelease_before_utc=as_of.isoformat(),
                   conservative_postrelease_from_utc=utc(date.fromisoformat(issue)+timedelta(days=1)))
    forecasts,metadata,*_ = public_forecasts([*historic,pending],pending_as_of=as_of)
    f = forecasts[issue]
    public = float(f["public_ensemble"])
    prior = float(previous["cushing_change_kbbl"])/1000
    pairs = selected_pairs(issue,summary,as_of)
    obs = list(optical_observations(pairs,summary,prior,first,end))
    training = [r for _,_,r in replay() if r["known"]<as_of and r["observed"]][-104:]
    beta = learned_beta(training,"correlated_raw")
    raw,info = physical_update(obs,public,correlated=True)
    updated = public+beta*raw
    metadata = metadata[issue]
    _, marks = archives()
    curve_files = {v[1]:v[2] for (day,_),v in marks.items() if day==metadata["curve_day"] and datetime.fromisoformat(v[3])<=as_of}
    record = dict(release_date=issue,week_ending=pending["latest_week_in_issue"],recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  information_cutoff_utc=as_of.isoformat(),registration_deadline_utc=cutoff.isoformat(),
                  output_unit="thousand_barrels", public_ensemble_kbbl=1000*public, optical_nowcast_kbbl=1000*updated,
                  robust_public_kbbl=1000*float(f["robust_public"]), public_curve_day=metadata["curve_day"],
                  public_curve_source=metadata["curve_source"], optical_update_kbbl=1000*(updated-public),
                  optical_raw_update_kbbl=1000*raw,optical_learned_weight=beta,optical_information=info,
                  measured_roofs=len(obs),terminals=summary["accepted_terminals"],
                  measured_roof_area_fraction=summary["area_fraction"],partial_capture_change_kbbl=summary["measured_partial_change_kbbl"],
                  prior_release=previous["release_date"],prior_release_sha256=previous["source_sha256"],
                  training_measured_releases=len(training),latest_training_label_utc=training[-1]["known"].isoformat(),
                  source_sha256={file.name:digest(file) for file in (source,coverage_path,DERIVED/f"cushing_forward_native_optical_{issue}_pairs.csv",
                                                                  ROOT/metadata["cftc_file"],ROOT/"scripts/fit_cushing_observation_operator.py")},
                  public_curve_archive_sha256=curve_files,
                  measured_pairs=obs)
    if datetime.fromisoformat(record["recorded_at_utc"])>=cutoff:
        raise ValueError("The forecast finished after the registration deadline")
    freeze(path,record)
    return record


def compare_shared_scene(original_path, comparison_path):
    original = json.loads(original_path.read_bytes())
    cutoff = datetime.fromisoformat(original["information_cutoff_utc"])
    deadline = datetime.fromisoformat(original["registration_deadline_utc"])
    if comparison_path.exists():
        return json.loads(comparison_path.read_bytes())
    if datetime.now(timezone.utc)>=deadline:
        raise ValueError("Shared-scene comparison must be registered before the release day")
    public = original["public_ensemble_kbbl"]/1000
    robust = original["robust_public_kbbl"]/1000
    history = [r for _,r in shared_scene_history() if r["cutoff"]<cutoff]
    result = shared_scene_forecast(original["measured_pairs"],cutoff,public,robust,history)
    now = datetime.now(timezone.utc)
    if not cutoff<=now<deadline:
        raise ValueError("Invalid shared-scene forecast receipt time")
    record = {k:original[k] for k in ("release_date","week_ending","registration_deadline_utc","information_cutoff_utc","output_unit",
        "public_ensemble_kbbl","optical_nowcast_kbbl","measured_roofs","measured_roof_area_fraction")} | dict(
        recorded_at_utc=now.isoformat(),original_forecast_sha256=digest(original_path),
        shared_scene_model_sha256=digest(ROOT/"scripts/fit_cushing_shared_scene.py"),
        shared_scene_history_sha256=digest(DERIVED/"cushing_shared_scene_forecasts.csv"),
        shared_scene_kbbl=1000*result["carried_scene"],uncorrelated_across_weeks_kbbl=1000*result["signed_scene"],
        latest_training_label_utc=result["latest_training_label_utc"],
        previous_roof_pairs_used=result["previous_roof_pairs_used"],latest_scene_label_utc=result["latest_available_scene_label_utc"])
    freeze(comparison_path,record)
    return record


def evaluate_shared_scene(path, original_path, original_score):
    comparison = json.loads(path.read_bytes())
    stamp,deadline = (datetime.fromisoformat(comparison[k]) for k in ("recorded_at_utc","registration_deadline_utc"))
    original = json.loads(original_path.read_bytes())
    if (not stamp<deadline or comparison["original_forecast_sha256"]!=digest(original_path)
            or any(comparison[k]!=original[k] for k in ("release_date","week_ending","registration_deadline_utc","information_cutoff_utc"))):
        raise ValueError("Shared-scene forecast receipt differs from the original forecast")
    result = dict(status=original_score["status"],release_date=comparison["release_date"],forecast_sha256=digest(path))
    if original_score["status"]=="scored_first_release":
        actual = original_score["actual_eia_change_kbbl"]
        errors = {k:abs(actual-comparison[f"{k}_kbbl"]) for k in ("public_ensemble","optical_nowcast","shared_scene")}
        result.update(actual_eia_change_kbbl=actual,absolute_error_kbbl=errors,
                      shared_scene_minus_optical_error_kbbl=errors["shared_scene"]-errors["optical_nowcast"],
                      first_release_sha256=original_score["first_release_sha256"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",required=True,type=date.fromisoformat)
    parser.add_argument("--score",action="store_true",help="Compare a registered forecast against its first published outcome")
    parser.add_argument("--compare-shared-scene",action="store_true",help="Register a separate shared-scene forecast on the frozen inputs")
    args = parser.parse_args()
    issue = args.release.isoformat()
    path = DERIVED/f"{STEM}_{issue}.json"
    comparison_path = DERIVED/f"{STEM}_{issue}_shared_scene.json"
    if args.score:
        record = score(path,json.loads(path.read_bytes()))
        destination = DERIVED/f"{STEM}_{issue}_score.json"
        write(destination,(json.dumps(record,indent=2,allow_nan=False)+"\n").encode())
        if comparison_path.exists():
            comparison_score = evaluate_shared_scene(comparison_path,path,record)
            destination = DERIVED/f"{STEM}_{issue}_shared_scene_score.json"
            write(destination,(json.dumps(comparison_score,indent=2,allow_nan=False)+"\n").encode())
            record["shared_scene_comparison"] = comparison_score
    elif args.compare_shared_scene:
        record = compare_shared_scene(path,comparison_path)
    else:
        record = json.loads(path.read_bytes()) if path.exists() else create(issue,path)
    print(json.dumps({k:v for k,v in record.items() if k not in ("measured_pairs","source_sha256")},indent=2))


if __name__ == "__main__":
    main()
