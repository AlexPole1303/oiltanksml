"""Use prerelease pairs that start during the EIA week and end just after it."""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from audit_sentinel2_alternate_registration import independent_offsets, refit
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_registration_inputs import anchor_grid, tank_points
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import estimate, prior_scale
from fit_cushing_multiframe_inventory import source_passes
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, NOISE_FLOOR_M, PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL, parts

STEM = "cushing_late_pair_recovery"
HYBRID = "basis_when_unmeasured"
METHODS = ("late_pair_full", "late_pair_half", "late_pair_learned")
MULTIFRAME = "registered_multiframe"
COMBINED = "registered_multiframe_and_late_pair"
MODELS = (PUBLIC,"carried_scene",HYBRID,*METHODS,MULTIFRAME,COMBINED)
MAX_DELAY = timedelta(hours=72)
MIN_OBSERVED_FRACTION = .25


def clock(row):
    return tuple(datetime.fromisoformat(row[k]) for k in (
        "previous_measurement_at_utc","current_measurement_at_utc","cutoff_utc"))


def candidates():
    windows = {r["release_date"]:r for stem in ("sentinel2_space_time_coverage_weeks","holdout_optical_weeks") for r in read(stem)}
    state = indexed("cushing_public_state_forecasts")
    roster = {r["candidate_id"]:r for r in read("sentinel2_space_time_coverage_roof_registry") if r["has_both_dated_sources"]=="True"}
    for phase,table in source_passes():
        for issue,p in state.items():
            if p["study_phase"]!=phase:
                continue
            row = windows[issue]
            a,b,cutoff = clock(row)
            if not a<b<cutoff or datetime.fromisoformat(p["cutoff"])!=cutoff:
                raise ValueError(f"Mismatched survey window on {issue}")
            for tank,(_,images) in table.items():
                if tank not in roster:
                    continue
                g = roster[tank]
                boundary = min(b+MAX_DELAY,cutoff)
                available = [q for q in images if a<q["capture"]<=boundary and q["ready"]<cutoff]
                starts = [q for q in available if q["capture"]<=b-(b-a)*MIN_OBSERVED_FRACTION]
                ends = [q for q in available if b<q["capture"]]
                if not starts or not ends:
                    continue
                first = min(starts,key=lambda q:(q["capture"].date(),q["mask_width"],q["ready"],q["scene"]))
                last = min(ends,key=lambda q:(q["capture"].date(),q["mask_width"],q["ready"],q["scene"]))
                area = float(g["circular_area_m2"])
                if any(g[k]>=first["capture"].date().isoformat() for k in ("lidar_source_available_date","roof_evidence_date")):
                    continue
                if any(q["terminal"]!=g["terminal_id"] or not math.isclose(area,math.pi*q["diameter"]**2/4,rel_tol=1e-10)
                       for q in (first,last)):
                    raise ValueError(f"Inconsistent pair geometry on {issue}/{tank}")
                pair = dict(geometry_mode=first["native"]["geometry_mode"],week_ending=row["week_ending"],release_date=issue,
                    candidate_id=tank,terminal_id=g["terminal_id"],inputs_available_before_utc=cutoff.isoformat(),
                    circular_area_m2=area,pre_measurement_pair_visible_kbbl=(first["height"]-last["height"])*area/BARREL_M3/1000)
                yield phase,pair,(a,b,issue,cutoff.isoformat()),first,last


def register_pairs():
    groups,weeks,availability = defaultdict(dict),{},{}
    for phase,pair,times,first,last in candidates():
        key = pair["week_ending"],pair["candidate_id"]
        if key in groups[phase]:
            raise ValueError(f"Duplicate late candidate {key}")
        groups[phase][key] = pair,[first["native"],last["native"]]
        weeks[key[0]] = times
        availability[key] = max(first["ready"],last["ready"])
    tanks,exclusions = tank_points()
    geometry = legacy_geometry()
    fits = []
    for phase,group in groups.items():
        print(phase,len(group),"dated late pairs",flush=True)
        anchors = anchor_grid({k[1]:tanks[k[1]] for k in group},exclusions,search_radius_m=1600)
        offsets = independent_offsets(group,phase,anchors)
        if offsets.keys()!=group.keys():
            raise ValueError(f"Missing independent offset in {phase}")
        for fit in refit(group,offsets,geometry,weeks):
            key = fit["week_ending"],fit["candidate_id"]
            a,b,_,_ = weeks[key[0]]
            fits.append(fit | dict(study_phase=phase,week_start_utc=a.isoformat(),week_end_utc=b.isoformat(),
                                   available_utc=availability[key].isoformat()))
    if len(fits)!=len(availability):
        raise ValueError("A candidate pair is missing a native fit")
    export(f"{STEM}_fits",fits)


def observation(row, panel):
    a,b,cutoff = clock(panel)
    first,last,ready = (datetime.fromisoformat(row[k]) for k in ("earlier_capture_at_utc","later_capture_at_utc","available_utc"))
    if (row["registration_status"]!="checked_offset" or not a<first<b<last<=ready<cutoff
            or last-b>MAX_DELAY or datetime.fromisoformat(row["inputs_available_before_utc"])!=cutoff
            or (b-first)/(b-a)<MIN_OBSERVED_FRACTION):
        raise ValueError(f"Invalid date of accepted pair for {row['release_date']}/{row['candidate_id']}")
    length = b-a
    start, tail = (first-a)/length,(last-b)/length
    area = float(row["circular_area_m2"])
    mapped = area/float(panel["dated_reference_area_m2"])
    conversion = area/BARREL_M3/1e6
    width = float(row["registered_loss_range_max_m"])-float(row["registered_loss_range_min_m"])
    duration = (last-first)/length
    return dict(candidate_id=row["candidate_id"],terminal_id=row["terminal_id"],starting_fraction=start,
        trailing_fraction=tail,mapped_fraction=mapped,scene_before=row["earlier_version"],scene_after=row["later_version"],
        variance=conversion**2*(NOISE_FLOOR_M**2+(width/2)**2+LOCAL_ACTIVITY_M**2*duration),duration=duration,
        volume_per_m=conversion,sensitivity=mapped*(1-start),prior_component=0.,
        pair_signal=conversion*float(row["registered_change_m"]),available_utc=ready,area=area)


def geometry(sample, target, week, *, following_mean=0., following_sigma=PRIOR_INVENTORY_SIGMA_MBBL):
    s,y,local = parts(sample,target)
    start,tail,mapped = (np.array([q[k] for q in sample]) for k in ("starting_fraction","trailing_fraction","mapped_fraction"))
    future = tail*mapped
    # The target week ends at a known clock, but the following week's change
    # is unknown. Factor in both its shared uncertainty and within-week motion.
    y -= following_mean*future
    local += following_sigma**2*np.outer(future,future)
    scenes = {r[k] for r in sample for k in ("scene_before","scene_after")}
    clocks = {week:(start,-mapped),week+timedelta(days=7):(tail,mapped)}
    return dict(observations=sample,sensitivity=s,residual=y,local=local,clocks=clocks,scenes=scenes)


def accepted_pairs():
    windows = {r["release_date"]:r for name in ("sentinel2_space_time_coverage_weeks","holdout_optical_weeks") for r in read(name)}
    fits = defaultdict(list)
    accepted = set()
    for r in read(f"{STEM}_fits"):
        if r["registered_both_shadow_diagnostics_pass"]!="True":
            continue
        issue,tank = r["release_date"],r["candidate_id"]
        if (issue,tank) in accepted:
            raise ValueError(f"Repeated accepted tank on {issue}/{tank}")
        accepted.add((issue,tank))
        fits[issue].append(observation(r,windows[issue]))
    return fits


def replay():
    state = indexed("cushing_public_state_forecasts")
    multi = indexed("cushing_multiframe_inventory_registered_forecasts")
    fits = accepted_pairs()
    history,errors = [],[]
    for issue,q in sorted(state.items()):
        baseline,public,actual = map(float,(q[HYBRID],q[PUBLIC],q["actual"]))
        cutoff,known = (datetime.fromisoformat(q[k]) for k in ("cutoff","known"))
        if not cutoff<known or q["latest_training_label_utc"] and datetime.fromisoformat(q["latest_training_label_utc"])>=cutoff:
            raise ValueError(f"Inconsistent base forecast on {issue}")
        measured = q["measured"]=="True"
        other = multi[issue]
        if (datetime.fromisoformat(other["cutoff"])!=cutoff or datetime.fromisoformat(other["known"])!=known
                or float(other["actual"])!=actual or float(other[HYBRID])!=baseline
                or any(other[k] and datetime.fromisoformat(other[k])>=cutoff
                       for k in ("latest_training_label_utc","latest_optical_at_utc"))):
            raise ValueError(f"Mismatched multiframe baseline on {issue}")
        multiframe = float(other["extra_overpasses"])
        if not measured and multiframe!=baseline:
            raise ValueError(f"Multiframe and late-pair cohorts overlap on {issue}")
        sample = fits[issue]
        delta = weight = information = 0.
        past = [r for r in history if r["known"]<cutoff][-78:]
        current_error = [r for r in errors if r["known"]<cutoff]
        if sample:
            week = datetime.fromisoformat(q["week_ending"]).date()
            prepared = geometry(sample,baseline,week)
            recent = [r["prepared"] for r in past if cutoff-r["cutoff"]<=MAX_SCENE_AGE]
            options = estimate(prepared,recent)
            weights = np.array((.25,.35,.30,.10))
            scale = prior_scale(current_error)
            delta = float(weights@np.clip([score/(1/scale**2+info) for _,info,score in options],-.6,.6))
            information = float(weights@np.asarray([info for _,info,_ in options]))
            if len(past)>=12:
                x,y = (np.array([r[k] for r in past]) for k in ("delta","error"))
                recency = np.exp2(-np.array([(cutoff-r["cutoff"]).total_seconds()/604800 for r in past])/52.)
                weights_fit = recency*np.asarray([1. if r["measured"]==measured else .35 for r in past])/np.maximum(abs(y),.25)
                weight = float(np.clip((weights_fit*x)@y/((weights_fit*x)@x+1.),0.,1.))
            history.append(dict(known=known,cutoff=cutoff,prepared=geometry(sample,actual,week),delta=delta,error=actual-baseline,measured=measured))
        errors.append(dict(known=known,error=actual-public))
        applied = delta if sample and not measured else 0.
        yield dict(release_date=issue,week_ending=q["week_ending"],study_phase=q["study_phase"],
            cutoff=cutoff,known=known,actual=actual,measured=measured,late_roofs=len(sample),
            area_time=math.fsum(r["sensitivity"] for r in sample),information=information,late_pair_delta=delta,weight=weight,
            latest_training_label_utc=past[-1]["known"].isoformat() if past else None,
            latest_optical_at_utc=max((r["available_utc"] for r in sample),default=None),
            **{PUBLIC:public,HYBRID:baseline,"carried_scene":float(q["carried_scene"])},
            **{MULTIFRAME:multiframe,COMBINED:multiframe+applied},
            **dict(zip(METHODS,(baseline+applied,baseline+.5*applied,baseline+weight*applied),strict=True)))


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        base = list(trading(rows,MODELS))
        trades = list(replay_conditional_market(rows,base,(PUBLIC,HYBRID,*METHODS,MULTIFRAME,COMBINED),
                                                references={PUBLIC:f"{PUBLIC}_conditional"}))
    report = dict(status="exploratory_on_reviewed_history",forecasts={},market={})
    for phase in PHASES:
        group = [r for r in rows if r["study_phase"]==phase]
        report["forecasts"][phase] = {}
        for scope,sample in (("all",group),("originally_unmeasured",[r for r in group if not r["measured"]]),
                             ("newly_measured",[r for r in group if r["late_roofs"] and not r["measured"]])):
            result = forecast_metrics(sample,MODELS)
            result["measured_roof_weeks"] = sum(r["late_roofs"] for r in sample)
            for m,v in result["models"].items():
                delta = [1000*(abs(r[m]-r["actual"])-abs(r[HYBRID]-r["actual"])) for r in sample]
                v["paired_delta_vs_hybrid_95_kbbl"] = block_bootstrap_mean(delta)
            report["forecasts"][phase][scope] = result
        market = report["market"][phase] = {}
        compare = [r for r in trades if r["study_phase"]==phase and r["method"]==HYBRID]
        for m in (PUBLIC,HYBRID,*METHODS,MULTIFRAME,COMBINED):
            sample = [r for r in trades if r["study_phase"]==phase and r["method"]==m]
            if [r["release_date"] for r in sample]!=[r["release_date"] for r in compare]:
                raise ValueError(f"Mismatched late-pair market calendar on {phase}/{m}")
            market[m] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample),
                decisions_changed_vs_hybrid=sum(r["near_side"]!=b["near_side"] for r,b in zip(sample,compare,strict=True)))
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in report["forecasts"][phase]["all"]["models"].items()},
              {m:(v["net_170_usd"],v["net_340_usd"]) for m,v in market.items()})
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-pairs",action="store_true",help="Recompute independent registration on original pixels")
    args = parser.parse_args()
    register_pairs() if args.register_pairs else main()
