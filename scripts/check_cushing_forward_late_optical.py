"""Stage a post-survey Sentinel-2 pair and apply the frozen weekly timing model."""

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from shapely.geometry import shape
from threadpoolctl import threadpool_limits

from audit_cushing_forward_optical_coverage import scene_coverage
from build_l1c_quality import candidate_layer
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import BBOX, RAW, ROOT, stac, write
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_late_pair_recovery import MAX_DELAY, MIN_OBSERVED_FRACTION, observation
from fit_cushing_late_pair_timing import STEM as TIMING, qualified_gain, ratio_weighted
from fit_cushing_mixed_optical_intervals import model
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import BASE, forecast, inputs
from fit_cushing_repeated_roof_pairs import repeated_errors
from fit_cushing_shared_scene import MAX_SCENE_AGE
from measure_cushing_forward_optical import build_packet, measure, stage
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score

STEM = "cushing_forward_late_optical"


def optics_before_cutoff(issue, week, start, end, as_of, store, roster):
    path = store/"catalog.json"
    if not path.is_file():
        snapshot = stac("sentinel-2-l2a",start.date().isoformat(),as_of.date().isoformat(),bbox=BBOX)
        write(path,(json.dumps(snapshot,separators=(",",":"))+"\n").encode())
    candidates = {f["id"]:f for f in json.loads(path.read_bytes())["features"]}
    footprints = {name:g for name,g in (*candidate_layer("osm"),*candidate_layer("supplemental")) if name in roster}
    if footprints.keys() != roster.keys():
        raise ValueError("Late optical inputs and dated roof geometry disagree")
    earlier = {r["item_id"]:r for r in read(f"cushing_forward_optical_coverage_{issue}_tiles")
               if r["segment"]=="target_week" and int(r["clear_roofs"]) and r["l1c_pre_release"]=="True"}
    records = [dict(q) for q in earlier.values()]
    pending = []
    for q in candidates.values():
        stamp = datetime.fromisoformat(q["properties"]["datetime"])
        if not end < stamp <= min(as_of,end+MAX_DELAY):
            continue
        if not any(q["properties"]["grid:code"] == r["grid"] for r in earlier.values()):
            continue
        if not any(shape(q["geometry"]).intersects(g) for g in footprints.values()):
            continue
        pending.append(q)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r,_ in pool.map(lambda q:scene_coverage(q,footprints,store/"scl",
                            index_store=store/"l1c_indexes",object_index_store=store/"object_indexes"),pending):
            ready = r["l1c_with_detector_available_by_utc"]
            r["segment"] = "after_week"
            r["l1c_pre_release"] = bool(ready and datetime.fromisoformat(ready) < as_of)
            records.append(r)
    if set(earlier)-candidates.keys():
        raise ValueError(f"Target-week optical scenes missing from the new catalogue for {week}")
    return path,records


def select_pairs(rows, tiles, baseline, window):
    archive = {r["l1c_product"]:datetime.fromisoformat(r["l1c_with_detector_available_by_utc"]) for r in tiles}
    start,end,cutoff = (datetime.fromisoformat(window[k])
                        for k in ("previous_measurement_at_utc","current_measurement_at_utc","cutoff_utc"))
    found = {}
    for r in rows:
        first,last = (datetime.fromisoformat(r[f"{edge}_capture"]) for edge in ("earlier","later"))
        if not r["registered_both_shadow_diagnostics_pass"] or not start < first < end < last:
            continue
        if last-end > MAX_DELAY or (end-first)/(end-start) < MIN_OBSERVED_FRACTION:
            continue
        latest = max(archive[r["earlier_version"]],archive[r["later_version"]])
        if not last <= latest < cutoff:
            continue
        if r["candidate_id"] not in baseline:
            continue
        selected = dict(r,
            earlier_capture_at_utc=first.isoformat(),later_capture_at_utc=last.isoformat(),
            available_utc=latest.isoformat(),inputs_available_before_utc=cutoff.isoformat(),
            circular_area_m2=r["area_m2"],release_date=window["release_date"])
        rank = last, float(r["registered_loss_range_max_m"])-float(r["registered_loss_range_min_m"]),r["earlier_version"]
        current = found.get(r["candidate_id"])
        if current is None or rank < current[0]:
            found[r["candidate_id"]] = rank, selected
    return [observation(r,window) for _,r in found.values()]


def update(week, cutoff, original, pairs, hub_area, reference):
    state = dict(measured=[],errors=[])
    for earlier in inputs(before=cutoff):
        forecast(earlier,state)
    prior = [r for r in state["measured"] if r["known"]<cutoff]
    recent = [r["models"][BASE] for r in prior if cutoff-r["cutoff"]<=MAX_SCENE_AGE]
    sigma = prior_scale([r for r in state["errors"] if r["known"]<cutoff])
    weights = learned_weights([dict(absolute_errors=r["errors_by_timing"][BASE]) for r in prior])
    public = original["public_ensemble_kbbl"]/1000
    row = dict(week_ending=week,cutoff=cutoff,observed=pairs,public=public,hub_area=hub_area,
               area_fraction=math.fsum(q["area"] for q in pairs)/hub_area, spanning=False)
    original_model = prepared(row,public,False)

    def project(candidate, history):
        stats = estimate(candidate,history)
        return public+float(weights @ np.clip([entry[2]/(1/sigma**2+entry[1]) for entry in stats],-1.25,1.25))

    anchor = project(original_model,recent)
    if not math.isclose(1000*anchor,reference,abs_tol=1e-8,rel_tol=0):
        raise ValueError("The benchmark does not reproduce with the original data cutoff")

    def marginal(late):
        if not late:
            return 0.,0.
        repeated_history = [repeated_errors(p,row,activity=False) for p in recent]
        base = project(repeated_errors(original_model,row,activity=True),repeated_history)
        deltas = []
        for observations in (late,ratio_weighted(late)):
            joint = model(row,public,original_model,[],observations,1.)
            deltas.append(project(repeated_errors(joint,row,activity=True),repeated_history)-base)
        return tuple(deltas)

    return marginal, prior[-1]["known"]


def main(issue, as_of):
    original_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    recovery_path = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
    anchor_path = DERIVED/f"cushing_pass_reliability_{issue}_registration.json"
    native_path = DERIVED/f"cushing_forward_native_optical_{issue}_summary.json"
    original,recovery,anchor,native = (json.loads(p.read_bytes()) for p in (original_path,recovery_path,anchor_path,native_path))
    checked_hashes(anchor["sources_sha256"])
    if any(q[k] != original[k] for q in (recovery,anchor) for k in LINEAGE):
        raise ValueError("Mismatch in the frozen issue and forecast clock")
    cutoff,deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if not cutoff < as_of < deadline or as_of > datetime.now(timezone.utc):
        raise ValueError("The new optical vintage must be observed between the original cutoff and registration deadline")
    stamp = as_of.strftime("%Y%m%dT%H%M%SZ")
    stem = f"{STEM}_{issue}_{stamp}"
    store = RAW/STEM/issue/stamp
    registration = DERIVED/f"{stem}_registration.json"
    if registration.is_file():
        saved = json.loads(registration.read_bytes())
        checked_hashes(saved["sources_sha256"])
        recorded, trained = (datetime.fromisoformat(saved[k]) for k in ("recorded_at_utc","latest_training_label_utc"))
        if (saved["release_date"] != issue or saved["week_ending"] != original["week_ending"]
                or datetime.fromisoformat(saved["new_cutoff_utc"]) != as_of
                or not trained < cutoff < as_of <= recorded < deadline):
            raise ValueError("The stored forecast does not match the dated registration")
        outcome = score(original_path,original)
        result = dict(status=outcome["status"],registration_sha256=digest(registration),forecasts_kbbl=saved["forecasts_kbbl"])
        if outcome["status"] == "scored_first_release":
            actual = outcome["actual_eia_change_kbbl"]
            result.update(actual_kbbl=actual,first_release_sha256=outcome["first_release_sha256"],
                          absolute_errors_kbbl={k:abs(v-actual) for k,v in saved["forecasts_kbbl"].items()})
        write(DERIVED/f"{stem}_score.json",(json.dumps(result,indent=2)+"\n").encode())
        print(result)
        return
    week = original["week_ending"]
    end = datetime.fromisoformat(f"{week}T12:00:00+00:00")
    start = end-timedelta(days=7)
    before = [q | dict(area=q["volume_per_m"]*BARREL_M3*1e6, available_utc=q["source_available_utc"])
              for q in original["measured_pairs"]] + recovery["recovered_roofs"]
    baseline = {q["candidate_id"] for q in before}
    roster = {q["candidate_id"]:q for q in read("cushing_modern_lidar_optical_registry") if q["candidate_id"] in baseline}
    if baseline != roster.keys() or len(baseline)!=len(before):
        raise ValueError("Wrong set of frozen roof pairs")
    catalog,tiles = optics_before_cutoff(issue,week,start,end,as_of,store,roster)
    tile_path = DERIVED/f"{stem}_tiles.csv"
    fields = dict.fromkeys(k for r in tiles for k in r)
    export(f"{stem}_tiles",[fields | q for q in tiles])
    usable = [r for r in tiles if int(r["clear_roofs"]) and r["l1c_pre_release"] in (True,"True")]
    count = sum(r["segment"]=="after_week" for r in usable)
    if not count:
        report = dict(status="no_timely_locally_clear_late_scene",release_date=issue,as_of_utc=as_of.isoformat(),
                      original_cutoff_utc=cutoff.isoformat(),catalog_sha256=digest(catalog),
                      late_tiles_in_catalog=sum(q["segment"]=="after_week" for q in tiles))
        write(DERIVED/f"{stem}_status.json",(json.dumps(report,indent=2)+"\n").encode())
        print(report)
        return
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError("The window for registering a new optical forecast has closed")
    hub_area = float(native["dated_roof_area_m2"])
    database = DERIVED/f"{stem}.sqlite"
    packet_store = store/"native"
    stage(usable,roster,issue,packet_store,catalog_path=catalog,object_index_store=store/"object_indexes")
    quality = build_packet(database,usable,roster,packet_store)
    release = dict(latest_week_in_issue=week,conservative_prerelease_before_utc=as_of.isoformat())
    with threadpool_limits(limits=1):
        fits = list(measure(database,usable,roster,release,segments=("target_week","after_week")))
        pair_rows = dict(release_date=issue,dated_reference_area_m2=hub_area,cutoff_utc=as_of.isoformat(),
                         previous_measurement_at_utc=start.isoformat(),current_measurement_at_utc=end.isoformat())
        late = select_pairs(fits,usable,baseline,pair_rows)
        projections = anchor["projections"]
        ref = projections["market"]["estimates_kbbl"][SUPPLEMENT]
        marginal, latest = update(week,cutoff,original,before,hub_area,ref)
        delta, weighted_delta = marginal(late)
    timing_path = DERIVED/f"{TIMING}_forecasts.csv"
    timing = []
    for r in read(timing_path.stem):
        known = datetime.fromisoformat(r["known"])
        if r["added_pairs"] == "0" or known >= cutoff:
            continue
        baseline = float(r[f"long_{SUPPLEMENT}"])
        timing.append(dict(known=known, baseline=baseline, actual=float(r["actual"]),
                           tail_share=float(r["tail_share"]), delta=float(r["long_ratio_weighted_pairs"])-baseline))
    tail_share = float(np.median([q["trailing_fraction"]/q["duration"] for q in late])) if late else None
    gain, compatible = qualified_gain(timing,dict(cutoff=cutoff,tail_share=tail_share),weighted_delta) if late else (0.,0)
    fits_path = DERIVED/f"{stem}_fits.csv"
    if fits:
        fields = dict.fromkeys(k for r in fits for k in r)
        export(f"{stem}_fits",[fields | r for r in fits])
    quality_path = DERIVED/f"{stem}_quality.csv"
    export(f"{stem}_quality",quality)
    sources = (original_path,recovery_path,anchor_path,native_path,catalog,tile_path,quality_path,database,timing_path,Path(__file__).resolve(),
               *(ROOT/"scripts"/name for name in ("measure_cushing_forward_optical.py","fit_cushing_repeated_roof_pairs.py",
                                                  "fit_cushing_mixed_optical_intervals.py","fit_cushing_late_pair_timing.py",
                                                  "fit_cushing_intraweek.py","collect_s2_l1c.py",
                                                  "collect_s2_l1c_archive_times.py","audit_cushing_forward_optical_coverage.py")),
               *sorted((store/"object_indexes").glob("*.json")), *((fits_path,) if fits else ()))
    now = datetime.now(timezone.utc)
    if not latest<cutoff<as_of<=now<deadline:
        raise ValueError("The late-pair forecast could not be frozen in time")
    raw = projections["inventory"]["estimates_kbbl"][SUPPLEMENT]
    forecasts = {}
    for name, value in (("inventory",raw),("market",ref)):
        forecasts.update({f"{name}_{method}":value+1000*increment for method,increment in (
            ("reference",0.),("with_late_pair",delta),("ratio_weighted",weighted_delta),("qualified_update",gain*weighted_delta))})
    result = dict(release_date=issue,week_ending=week,original_cutoff_utc=cutoff.isoformat(),
                  new_cutoff_utc=as_of.isoformat(),recorded_at_utc=now.isoformat(),deadline_utc=deadline.isoformat(),
                  latest_training_label_utc=latest.isoformat(),frozen_roofs=len(before),additional_pairs=len(late),
                  additional_pair_ids=sorted(q["candidate_id"] for q in late),late_pair_delta_kbbl=1000*delta,
                  ratio_weighted_delta_kbbl=1000*weighted_delta, median_tail_share=tail_share,
                  qualified_gain=gain, compatible_scored_weeks=compatible,
                  latest_qualification_label_utc=timing[-1]["known"].isoformat() if timing else None, forecasts_kbbl=forecasts,
                  sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources})
    freeze(registration,result)
    print({k:v for k,v in result.items() if k not in ("sources_sha256","additional_pair_ids")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=date.fromisoformat,required=True)
    parser.add_argument("--observed-through",type=datetime.fromisoformat,required=True)
    args = parser.parse_args()
    main(args.release.isoformat(),args.observed_through.astimezone(timezone.utc))
