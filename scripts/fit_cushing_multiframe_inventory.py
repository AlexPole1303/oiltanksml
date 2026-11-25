"""Condition an independently registered tank pair on other prerelease overpasses."""

import argparse
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from audit_sentinel2_alternate_registration import refit, registration_products, attach_sources
from build_cushing_holdout_optical_recovery import DATED
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_registration_inputs import anchor_grid, tank_points
from build_weekly_shadow_panel import RESOLVED
from build_weekly_shadow_validation import legacy_geometry
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, NOISE_FLOOR_M, PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_outer_capture_nowcast import SUPPLEMENT
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import SCENE_SIGMA_M, TERMINAL_SIGMA_M
from fit_shadow_registration import tank_offset

STEM = "cushing_multiframe_inventory"
HYBRID = "basis_when_unmeasured"
METHODS = ("extra_overpasses", "extra_overpasses_shrunk", "extra_overpasses_calibrated", "intraweek_only")
MODELS = (PUBLIC, "carried_scene", HYBRID, SUPPLEMENT, *METHODS)
PASS_SEPARATION = timedelta(minutes=20)


def source_passes():
    for phase in PHASES:
        by_tank = defaultdict(dict)
        for r in read(f"shadow_timing_{phase}_observations"):
            if r["geometry_mode"]!=DATED or r["fit_status"]!=RESOLVED:
                continue
            capture, archive, ready = (datetime.fromisoformat(r[k]) for k in (
                "captured_at_utc","archive_available_at_utc","geometry_available_at_utc"))
            label, low, high = (float(r[k]) for k in (
                "estimated_gap_m","near_optimal_gap_min_m","near_optimal_gap_max_m"))
            if not capture<=archive<=ready or not low<=label<=high:
                raise ValueError(f"Invalid optical provenance on {r['version_id']}")
            key = r["candidate_id"]
            source = dict(scene=r["version_id"],capture=capture,ready=ready,height=label,diameter=float(r["diameter_m"]),
                          mask_width=high-low,sigma=max(NOISE_FLOOR_M,(high-low)/2),terminal=r["terminal_id"],
                          pixels=int(r["fitted_pixels"]),metadata=r["tile_metadata_path"],native=r)
            if source["scene"] in by_tank[key]:
                raise ValueError(f"A single dated scene occurs twice on {key}")
            by_tank[key][source["scene"]] = source
        yield phase,{tank:(scenes,sorted(scenes.values(),key=lambda q:q["capture"])) for tank,scenes in by_tank.items()}


def matching_passes(pair, month, candidates, cutoff, start, stop):
    if pair["candidate_id"] not in candidates:
        return None
    scenes, passes = candidates[pair["candidate_id"]]
    if not all(pair[k] in scenes for k in ("scene_before","scene_after")):
        return None
    anchor = tuple(scenes[pair[k]] for k in ("scene_before","scene_after"))
    if not min(q["capture"] for q in anchor)<=start<max(q["capture"] for q in anchor)<=stop:
        return None
    captures = [q["capture"] for q in passes]
    lower = bisect_left(captures,min(start,anchor[0]["capture"]))
    upper = bisect_right(captures,stop+timedelta(hours=72))
    possible = [q for q in passes[lower:upper] if q["ready"]<cutoff]
    # Duplicate tiles from the same overpass carry nearly identical roof
    # illumination. Preserve the registered primary endpoints on those passes.
    chosen = list(anchor)
    for q in sorted(possible,key=lambda q:(q["mask_width"],-q["pixels"],q["ready"],q["scene"])):
        if all(abs(q["capture"]-p["capture"])>=PASS_SEPARATION for p in chosen):
            chosen.append(q)
    if len(chosen)==2:
        return None
    for q in chosen:
        if (q["ready"]>=cutoff or q["terminal"]!=pair["terminal_id"] or not (ROOT/q["metadata"]).is_file()
                or not math.isclose(math.pi*q["diameter"]**2/4,pair["area"],rel_tol=1e-10)):
            raise ValueError(f"Optical point inconsistent with the registered tank on {month}/{pair['candidate_id']}")
    return anchor,tuple(sorted(chosen,key=lambda q:q["capture"]))


def design_rows(sample, start, stop, public, prior, area, *, selection):
    first, final = (-1.,2.)
    weekly_length = stop-start
    prepared = []
    for pair,anchor,full in sample:
        if selection == "intraweek":
            rows = tuple(q for q in full if q in anchor or start<q["capture"]<=stop)
        else:
            rows = full if selection == "full" else anchor
        times = np.array([(r["capture"]-start)/weekly_length for r in rows])
        if np.any(np.diff(times)<=0) or times.min()<first or times.max()>final:
            raise ValueError(f"Invalid observation interval for {pair['candidate_id']}")
        # Independent height fits share the reference endpoint when taking
        # differences. Work in millions of barrels to retain tank area.
        difference = np.column_stack((-np.ones(len(rows)-1),np.eye(len(rows)-1)))
        conversion = pair["volume_per_m"]
        fraction = pair["area"]/area
        geometry = -conversion*np.array([r["height"] for r in rows])
        exposures = np.column_stack([np.clip(times,week,week+1)-week for week in (-1,0,1)])
        flow = difference@exposures
        weights = difference*conversion
        observation = difference@geometry-fraction*(flow[:,0]*prior+flow[:,1]*public)
        measurement = np.diag([r["sigma"]**2 for r in rows])
        scene_index = {r["scene"]:i for i,r in enumerate(rows)}
        # A reconstructed extra height incorporates the same random error
        # as the anchor plus an independent error in the registered change.
        anchors = np.eye(len(rows))
        for j,r in enumerate(rows):
            if "registration_anchor" in r:
                anchors[j,scene_index[r["registration_anchor"]]] = 1.
        measurement = anchors@measurement@anchors.T
        noise = weights@measurement@weights.T
        brownian = difference@np.minimum.outer(times-times[0],times-times[0])@difference.T
        noise += (LOCAL_ACTIVITY_M*conversion)**2*brownian
        prepared.append((pair["terminal_id"],rows,observation,noise,weights,flow,fraction))
    names = {name:i for i,name in enumerate(sorted({r["scene"] for _,rows,*_ in prepared for r in rows}))}
    terminals = {name:i for i,name in enumerate(sorted({r[0] for r in prepared}))}
    n = sum(len(r[1])-1 for r in prepared)
    scene = np.zeros((n,len(names)))
    terminal = np.zeros((n,3*len(terminals)))
    target = np.zeros(n)
    nuisance = np.zeros(n)
    offset = 0
    for name,rows,observation,_,weights,flow,fraction in prepared:
        idx = slice(offset,offset+len(observation))
        scene[idx,[names[r["scene"]] for r in rows]] = weights
        terminal[idx,3*terminals[name]:3*(terminals[name]+1)] = abs(weights[0,0])*flow
        target[idx] = fraction*flow[:,1]
        nuisance[idx] = fraction*flow[:,2]
        offset += len(observation)
    covariance = block_diag(*(r[3] for r in prepared)) + SCENE_SIGMA_M**2*(scene@scene.T)
    covariance += TERMINAL_SIGMA_M**2*(terminal@terminal.T)+.9**2*np.outer(nuisance,nuisance)
    response = np.concatenate([r[2] for r in prepared])
    return target,response,covariance


def update(sample, row, start, stop, prior, area, sigma, *, selection):
    direction, residual, covariance = design_rows(sample,start,stop,row["public"],prior,area,selection=selection)
    slope = cho_solve(cho_factor(covariance),direction)
    limit = 3*np.sqrt(covariance.diagonal())
    information = float(direction@slope)
    return float(np.clip(slope@np.clip(residual,-limit,limit)/(sigma**-2+information),-1.25,1.25)),information


def dated_roof_panel():
    clocks = {r["release_date"]:r for stem in ("sentinel2_space_time_coverage_weeks","holdout_optical_weeks") for r in read(stem)}
    public_book = indexed("cushing_robust_stack_forecasts")
    hybrid_book = indexed("cushing_public_state_forecasts")
    outer_book = indexed("cushing_outer_capture_nowcast_forecasts")
    catalogue = dict(source_passes())
    for row,arms,_ in combined_sources():
        issue, phase, cutoff, known = (row[k] for k in ("release_date","study_phase","cutoff","known"))
        c, p, h = clocks[issue],public_book[issue],hybrid_book[issue]
        start, stop = (datetime.fromisoformat(c[f"{key}_measurement_at_utc"]) for key in ("previous","current"))
        if (not start<stop<cutoff<known or cutoff!=datetime.fromisoformat(c["cutoff_utc"])
                or cutoff!=datetime.fromisoformat(h["cutoff"]) or known!=datetime.fromisoformat(h["known"])
                or float(h["actual"])!=row["actual"] or float(h[PUBLIC])!=row["public"]):
            raise ValueError(f"Inconsistent report clocks on {issue}")
        selected = []
        total_area = float(c["dated_reference_area_m2"])
        for pair in arms[-1]:
            matches = matching_passes(pair,issue,catalogue[phase],cutoff,start,stop)
            if matches is None:
                continue
            if datetime.fromisoformat(pair["available_utc"])>=cutoff:
                raise ValueError(f"A native registered anchor was unavailable on {issue}")
            selected.append((pair,*matches))
        yield row, h, start, stop, total_area, float(p["prior"]), selected


def registration_jobs(panel):
    jobs, weeks = defaultdict(dict), {}
    dependencies = {}
    for row, _, start, stop, _, _, selected in panel:
        issue, phase, week, cutoff = (row[k] for k in ("release_date","study_phase","week_ending","cutoff"))
        weeks[week] = (start,stop,issue,cutoff.isoformat())
        for pair,anchor,pack in selected:
            for number, extra in enumerate(q for q in pack if q not in anchor):
                primary = min(anchor,key=lambda q:abs(q["capture"]-extra["capture"]))
                first,last = sorted((primary,extra),key=lambda q:q["capture"])
                job = dict(geometry_mode=DATED,week_ending=week,release_date=issue,candidate_id=pair["candidate_id"],
                           terminal_id=pair["terminal_id"],inputs_available_before_utc=cutoff.isoformat(),
                           circular_area_m2=pair["area"],pre_measurement_pair_visible_kbbl=(first["height"]-last["height"])*pair["volume_per_m"]*1000)
                key = week,pair["candidate_id"],number
                jobs[phase][key] = job,[first["native"],last["native"]]
                dependencies[phase,key] = (primary,extra)
    return jobs,weeks,dependencies


def verify_extra_passes():
    jobs,weeks,dependencies = registration_jobs(dated_roof_panel())
    tanks,exclusion = tank_points()
    geometry = legacy_geometry()
    output = []
    for phase,all_jobs in jobs.items():
        print(phase,len(all_jobs),"extra frame pairs",flush=True)
        involved = {key[1]:tanks[key[1]] for key in all_jobs}
        anchors = anchor_grid(involved,exclusion,search_radius_m=1600)
        for number in sorted({k[2] for k in all_jobs}):
            group = {key[:2]:item for key,item in all_jobs.items() if key[2]==number}
            sources,links = attach_sources(group,phase)
            checked,_,_,profiles,_ = registration_products(links,sources,anchors)
            offsets = {(r["week_or_control_day"],r["candidate_id"]):tank_offset(r,profiles) for r in checked}
            if offsets.keys()!=group.keys():
                raise ValueError(f"Missing extra-scene offsets in {phase}")
            additions = refit(group,offsets,geometry,weeks)
            for r in additions:
                primary,extra = dependencies[phase,(r["week_ending"],r["candidate_id"],number)]
                added = r["registered_both_shadow_diagnostics_pass"]
                delta = r["registered_change_m"] if added else None
                if added and r["registration_status"]!="checked_offset":
                    raise ValueError("A verified pair is missing an exterior registration")
                gap = None
                if added:
                    sign = -1 if r["earlier_version"]==primary["scene"] else 1
                    gap = primary["height"]+sign*delta
                output.append(r | dict(study_phase=phase,anchor_scene=primary["scene"],extra_scene=extra["scene"],
                    anchored_extra_height_m=gap,extra_available_utc=extra["ready"].isoformat()))
    if len(output)!=len(dependencies):
        raise ValueError("The registration audit omitted a selected scene")
    export(f"{STEM}_native_registration",output)


def replay(*, registered=False, shared_anchor=True, screening_only=False):
    outer_book = indexed("cushing_outer_capture_nowcast_forecasts")
    errors, fits = [], []
    measurements = []
    registered_book = {}
    if registered:
        records = read(f"{STEM}_native_registration")
        registered_book = {(r["release_date"],r["candidate_id"],r["extra_scene"]):r for r in records}
        if len(registered_book)!=len(records):
            raise ValueError("Repeated registered extra frame")
    for row,h,start,stop,total_area,prior,choices in dated_roof_panel():
        issue,phase,cutoff,known = (row[k] for k in ("release_date","study_phase","cutoff","known"))
        past = [r for r in errors if r["known"]<cutoff]
        available = [r for r in fits if r["known"]<cutoff][-78:]
        selected = []
        for pair,anchor,pack in choices:
            approved = []
            for q in pack:
                if not registered or q in anchor:
                    approved.append(q)
                    continue
                proof = registered_book[issue,pair["candidate_id"],q["scene"]]
                if datetime.fromisoformat(proof["extra_available_utc"])!=q["ready"]:
                    raise ValueError(f"Extra frame provenance differs on {issue}")
                if proof["registered_both_shadow_diagnostics_pass"]=="True":
                    width = float(proof["registered_loss_range_max_m"])-float(proof["registered_loss_range_min_m"])
                    anchor_metadata = dict(registration_anchor=proof["anchor_scene"]) if shared_anchor else {}
                    value = q | dict(height=float(proof["anchored_extra_height_m"]),sigma=max(q["sigma"],width/2)) | anchor_metadata
                    approved.append(q if screening_only else value)
            if len(approved)>2:
                selected.append((pair,anchor,tuple(approved)))
        baseline = float(h[HYBRID])
        old = new = internal = old_info = new_info = internal_info = 0.
        if selected:
            sigma = prior_scale(past)
            kwargs = dict(prior=prior,area=total_area,sigma=sigma)
            old,old_info = update(selected,row,start,stop,**kwargs,selection="anchor")
            new,new_info = update(selected,row,start,stop,**kwargs,selection="full")
            internal,internal_info = update(selected,row,start,stop,**kwargs,selection="intraweek")
        delta = float(np.clip(new-old,-.4,.4))
        internal_delta = float(np.clip(internal-old,-.4,.4))
        weight = 0.
        if len(available)>=12:
            x = np.asarray([q["delta"] for q in available])
            y = np.asarray([q["residual"] for q in available])
            w = 1/np.maximum(abs(y),.15)
            weight = float(np.clip((w*x)@y/((w*x)@x+2.),0.,1.))
        latest = max((q["ready"] for _,_,pack in selected for q in pack),default=None)
        for pair,anchor,pack in selected:
            measurements.append(dict(release_date=issue,study_phase=phase,candidate_id=pair["candidate_id"],
                terminal_id=pair["terminal_id"],passes=len(pack),extra_passes=len(pack)-2,area_m2=pair["area"],
                extra_inside_or_at_boundary=sum(start<q["capture"]<=stop for q in pack if q not in anchor),
                anchor_scenes="|".join(q["scene"] for q in anchor),all_scenes="|".join(q["scene"] for q in pack),
                latest_available_utc=max(r["ready"] for r in pack).isoformat(),cutoff_utc=cutoff.isoformat()))
        result = dict(zip(METHODS,(baseline+delta,baseline+delta/2,baseline+weight*delta,baseline+internal_delta/2),strict=True))
        yield {k:row[k] for k in ("release_date","week_ending","study_phase","actual","cutoff","known")} | result | dict(
            **{PUBLIC:row["public"],"carried_scene":float(h["carried_scene"]),HYBRID:baseline,SUPPLEMENT:float(outer_book[issue][SUPPLEMENT])},
            matched_roofs=len(selected),extra_overpass_count=sum(len(pack)-2 for _,_,pack in selected),
            extra_intraweek_count=sum(start<q["capture"]<=stop for _,anchor,pack in selected for q in pack if q not in anchor),
            area_share=math.fsum(q["area"] for q,_,_ in selected)/total_area,anchor_update=old,all_update=new,
            timing_revision=delta,intraweek_revision=internal_delta,anchor_information=old_info,
            all_information=new_info,intraweek_information=internal_info,calibration_weight=weight,
            latest_optical_at_utc=latest.isoformat() if latest else None,
            latest_training_label_utc=available[-1]["known"].isoformat() if available else None)
        errors.append(dict(known=known,error=row["actual"]-row["public"]))
        if selected:
            fits.append(dict(known=known,delta=delta,residual=row["actual"]-baseline))
    suffix = report_suffix(registered,shared_anchor,screening_only)
    export(f"{STEM}{suffix}_measurements",measurements)


def report_suffix(registered, shared_anchor, screening_only):
    if screening_only:
        return "_screening_only"
    if not registered:
        return ""
    return "_registered" if shared_anchor else "_registered_independent"


def main(*, registered=False, shared_anchor=True, screening_only=False):
    with threadpool_limits(limits=1):
        rows = list(replay(registered=registered,shared_anchor=shared_anchor,screening_only=screening_only))
        first = list(trading(rows,MODELS))
        trades = list(replay_conditional_market(rows,first,(PUBLIC,HYBRID,SUPPLEMENT,*METHODS),references={
            PUBLIC:f"{PUBLIC}_conditional",SUPPLEMENT:f"{SUPPLEMENT}_conditional",
        }))
    report = dict(status="exploratory_on_reviewed_history",
                  method_variant=report_suffix(registered,shared_anchor,screening_only).lstrip("_") or "all_candidates",
                  forecasts={},market={})
    for phase in PHASES:
        group = [r for r in rows if r["study_phase"]==phase]
        report["forecasts"][phase] = {}
        for scope,sample in (("all",group),("matched",[r for r in group if r["matched_roofs"]])):
            metrics = forecast_metrics(sample,MODELS)
            for m,v in metrics["models"].items():
                difference = [1000*(abs(r[m]-r["actual"])-abs(r[HYBRID]-r["actual"])) for r in sample]
                v["paired_delta_vs_hybrid_95_kbbl"] = block_bootstrap_mean(difference)
            metrics["matched_roof_weeks"] = sum(r["matched_roofs"] for r in sample)
            metrics["extra_pass_roof_weeks"] = sum(r["extra_overpass_count"] for r in sample)
            metrics["extra_intraweek_pass_roof_weeks"] = sum(r["extra_intraweek_count"] for r in sample)
            report["forecasts"][phase][scope] = metrics
        report["market"][phase] = {}
        for m in (PUBLIC,HYBRID,SUPPLEMENT,*METHODS):
            sample = [r for r in trades if r["study_phase"]==phase and r["method"]==m]
            report["market"][phase][m] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample))
        print(phase,{m:round(r["mae_kbbl"],2) for m,r in report["forecasts"][phase]["all"]["models"].items()},
              {m:(r["net_170_usd"],r["net_340_usd"]) for m,r in report["market"][phase].items()})
    stem = STEM+report_suffix(registered,shared_anchor,screening_only)
    export(f"{stem}_forecasts",rows)
    export(f"{stem}_trades",trades)
    write(DERIVED/f"{stem}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--verify-extra-frames",action="store_true",help="Recompute native registration evidence")
    modes.add_argument("--registered",action="store_true",help="Use the registered heights and shared height errors")
    modes.add_argument("--registered-independent",action="store_true",help="Treat registered height errors as independent")
    modes.add_argument("--screening-only",action="store_true",help="Use verified pass selection with original height estimates")
    args = parser.parse_args()
    if args.verify_extra_frames:
        verify_extra_passes()
    else:
        main(registered=args.registered or args.registered_independent or args.screening_only,
             shared_anchor=not(args.registered_independent or args.screening_only),screening_only=args.screening_only)
