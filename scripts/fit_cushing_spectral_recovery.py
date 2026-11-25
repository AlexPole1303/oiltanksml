"""Validate additional B08 height pairs as regional inventory and CL features."""

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta

from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, PHASES, PUBLIC, learned_beta
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_regional_balance import vintage_rows
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import estimate, replay as shared_scene_history
from publication_clocks import EASTERN, utc

STEM = "cushing_spectral_recovery"
MODELS = (PUBLIC,"carried_scene","red_joint_recalibrated","spectral_plus_red","spectral_only","spectral_residual")
RECOVERED = ("sentinel2_weekly_multiband_recovery_candidates","sentinel2_weekly_multiband_holdout_recovery_candidates")
SPECTRAL_SIGMA_M = 2.
MAX_SCENE_AGE = timedelta(days=21)


def source_rows(method="nir"):
    if method not in ("nir","joint"):
        raise ValueError(f"Unknown spectral recovery method {method}")
    per_release = defaultdict(list)
    for stem in RECOVERED:
        for r in read(stem):
            if r[f"selected_for_{method}_expansion"]=="True":
                per_release[r["release_date"]].append(r)
    calendar = {(r["release_date"],r["week_ending"]):r for stem in ("sentinel2_space_time_coverage_weeks","holdout_optical_weeks")
                for r in read(stem)}
    roster = {r["candidate_id"]:r for r in read("sentinel2_space_time_coverage_roof_registry")}
    past_reports = {r["week"]:r for r in sorted(vintage_rows(),key=lambda q:q["issue"],reverse=True)}
    public = indexed("cushing_robust_stack_forecasts")
    audit = []
    for ref,state in shared_scene_history():
        issue = state["release_date"]
        found = []
        row = calendar[issue,state["week_ending"]]
        start,end = (datetime.fromisoformat(row[k]) for k in ("previous_measurement_at_utc","current_measurement_at_utc"))
        prior_week = start.astimezone(EASTERN).date()
        earlier = datetime.fromisoformat(utc(prior_week-timedelta(days=7),start.astimezone(EASTERN).time()))
        prior = past_reports.get(prior_week)
        previous_change = prior["change"][0] if prior and prior["known"]<state["cutoff"] else float(public[issue]["prior"])
        candidates = [r for r in per_release[issue] if r["week_ending"]==state["week_ending"]]
        if len({r["candidate_id"] for r in candidates})!=len(candidates):
            raise ValueError(f"Repeated spectral candidate on {issue}")
        already = {r["candidate_id"] for r in state["observed"]}
        for r in candidates:
            roof = roster[r["candidate_id"]]
            g1,g2,archive,mask1,mask2,cutoff = (datetime.fromisoformat(r[k]) for k in (
                "earlier_capture_at_utc","later_capture_at_utc","archive_available_at_utc","earlier_mask_available_at_utc",
                "later_mask_available_at_utc","cutoff_utc"))
            if not (r[f"{method}_resolved"]=="True" and r["red_resolved"]=="False" and
                    r["week_ending"]==state["week_ending"] and r["study_phase"]==state["study_phase"] and
                    cutoff==state["cutoff"] and g1<g2<=archive and max(archive,mask1,mask2)<cutoff and
                    r["terminal_id"]==roof["terminal_id"] and r["earlier_sentinel2_version"]!=r["later_sentinel2_version"]):
                raise ValueError(f"Invalid spectral lineage on {issue}/{r['candidate_id']}")
            disposition = "already_in_red_cohort" if r["candidate_id"] in already else (
                "outside_reporting_geometry" if not earlier<=g1<=start<g2<=end else "admitted")
            if disposition=="admitted" and any(not roof[k] or date.fromisoformat(roof[k])>=g1.date() for k in (
                "lidar_source_available_date","roof_evidence_date"
            )):
                disposition = "geometry_date_unverified"
            area = float(roof["circular_area_m2"])
            volume = float(r[f"{method}_change_kbbl"])/1000
            mbbl_per_m = area/BARREL_M3/1e6
            delta = float(r[f"{method}_earlier_gap_m"])-float(r[f"{method}_later_gap_m"])
            if not math.isclose(volume,delta*mbbl_per_m,abs_tol=1e-10):
                raise ValueError(f"Mismatch between {method} height and volume on {issue}/{r['candidate_id']}")
            audit.append(dict(release_date=issue,study_phase=state["study_phase"],candidate_id=r["candidate_id"],disposition=disposition,
                earlier_scene=r["earlier_sentinel2_version"],later_scene=r["later_sentinel2_version"],
                latest_input_at_utc=max(archive,mask1,mask2).isoformat(),cutoff_utc=cutoff.isoformat(),
                **{f"{method}_change_kbbl":1000*volume}))
            if disposition!="admitted":
                continue
            before,current = (start-g1)/(start-earlier),(g2-start)/(end-start)
            share = area/float(row["dated_reference_area_m2"])
            duration = before+current
            found.append(dict(candidate_id=r["candidate_id"],terminal_id=r["terminal_id"],pair_signal=volume,
                prior_component=share*before*previous_change,sensitivity=share*current,volume_per_m=mbbl_per_m,area=area,
                variance=mbbl_per_m**2*(SPECTRAL_SIGMA_M**2+LOCAL_ACTIVITY_M**2*duration),duration=duration,
                scene_before=r["earlier_sentinel2_version"],scene_after=r["later_sentinel2_version"],
                available_utc=max(archive,mask1,mask2).isoformat()))
        yield state | {"carried_scene":ref["carried_scene"],"spectral":found},audit
        audit = []


def replay(rows):
    seen = []
    for r,audit in rows:
        b04,added = r["observed"],r["spectral"]
        base = r["public"]
        past = [q for q in seen if q["known"]<r["cutoff"]]
        recent = [q for q in past if r["cutoff"]-q["cutoff"]<=MAX_SCENE_AGE]
        # Rerun three matched predictions. Recent observations use the same
        # measurement cohort as the current arm for scene-error estimation.
        arms = (b04,b04+added,added)
        raw = [estimate(observations,base,[q | dict(observed=q["arms"][i]) for q in recent if q["arms"][i]])[0]
               for i,observations in enumerate(arms)]
        names = ("red_raw","expanded_raw","nir_raw","extra_raw")
        raw.append(raw[1]-raw[0])
        train = [q for q in past if q["observed"] or q["spectral"]][-104:]
        with_spectral = [q for q in train if q["spectral"]]
        red_weight,expanded_weight = (learned_beta(train,k) for k in names[:2])
        nir_weight = learned_beta(with_spectral,"nir_raw")
        spectral_weight = learned_beta([q | dict(public=q["carried_scene"]) for q in with_spectral],"extra_raw")
        if not (added or b04) and raw[-1]:
            raise ValueError(f"Computed a spectral change without optical observations on {r['release_date']}")
        result = r | dict(zip(names,raw,strict=True)) | dict(arms=arms,public_ensemble=base,
            red_joint_recalibrated=base+red_weight*raw[0],spectral_plus_red=base+expanded_weight*raw[1],
            spectral_only=base+nir_weight*raw[2],spectral_residual=r["carried_scene"]+(spectral_weight*raw[-1] if added else 0.),
            latest_training_label_utc=train[-1]["known"].isoformat() if train else None)
        seen.append(result)
        yield result,audit


def main():
    rows,audit = [],[]
    for r,tanks in replay(source_rows()):
        rows.append(r)
        audit.extend(tanks)
    trades = list(trading(rows,MODELS))
    report = dict(status="historical_walk_forward_on_reviewed_period",predictions={},trading={})
    for phase in PHASES:
        subset = [r for r in rows if r["study_phase"]==phase]
        report["predictions"][phase] = {scope:forecast_metrics(sample,MODELS) | dict(nir_roof_pairs=sum(len(r["spectral"]) for r in sample))
            for scope,sample in (("all",subset),("with_nir",[r for r in subset if r["spectral"]]),
                ("only_nir_available",[r for r in subset if r["spectral"] and not r["observed"]]))}
        groups = {m:[r for r in trades if r["study_phase"]==phase and r["method"]==m] for m in ("curve",*MODELS)}
        base = groups[PUBLIC]
        result = {}
        for method,group in groups.items():
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different spectral trade windows on {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(group,base,strict=True)]
            result[method] = robustness(group) | daily_risk(group) | dict(increment_usd=sum(delta),paired_delta_95_usd=block_bootstrap_mean(delta))
        report["trading"][phase] = result
        print(phase,{s:(q["weeks"],q["nir_roof_pairs"],{m:round(v["mae_kbbl"],2) for m,v in q["models"].items()})
                     for s,q in report["predictions"][phase].items()})
        print(phase,{m:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in result.items()})
    export(f"{STEM}_forecasts",[{k:r[k] for k in ("release_date","week_ending","study_phase","actual","latest_training_label_utc",*MODELS)} |
        dict(b04_tanks=len(r["observed"]),b08_added_tanks=len(r["spectral"]),cutoff_utc=r["cutoff"].isoformat(),known_utc=r["known"].isoformat())
        for r in rows])
    export(f"{STEM}_measurements",audit)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
