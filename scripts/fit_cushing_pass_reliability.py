"""Estimate the reliability of a weekly optical update from pass deletions."""

import argparse
import json
import math
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import learned_weights, prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_optical_error_distribution import STEM as DISTRIBUTION, inputs as channels
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import BASE, LOCAL, STEM as OVERPASS, forecast, inputs, local_scores, pass_id
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from replay_cushing_weighted_inventory_market import replay_weighted_market

STEM = "cushing_pass_reliability"
MODELS = (SUPPLEMENT, "pooled_gain", "pass_gain", "pass_gain_shrunk")
CURRENT = "pass_gain_current_only"
MIN_WEEKS = 52


def subset(model, keep):
    observations = [r for r, accepted in zip(model["observations"], keep, strict=True) if accepted]
    return dict(observations=observations, scenes={r[k] for r in observations for k in ("scene_before","scene_after")},
                sensitivity=model["sensitivity"][keep], residual=model["residual"][keep],
                local=model["local"][np.ix_(keep,keep)],
                clocks={week:(t[keep],area[keep]) for week,(t,area) in model["clocks"].items()
                        if np.any(area[keep])})


def deletion_diagnostics(row, state):
    if not row["observed"]:
        return dict(passes=0, removable_passes=0, max_pass_share=0., max_shift=0., rms_shift=0.,
                    physical_revision=0., sign_reversals=0), None
    cutoff = row["cutoff"]
    measured = [r for r in state["measured"] if r["known"] < cutoff]
    recent = [r["models"][BASE] for r in measured if cutoff-r["cutoff"] <= MAX_SCENE_AGE]
    scale = prior_scale([r for r in state["errors"] if r["known"] < cutoff])
    timing = learned_weights([dict(absolute_errors=r["errors_by_timing"][BASE]) for r in measured])
    model = prepared(row, row["public"], row["spanning"])
    pass_counts = Counter(pass_id(n) for n in model["scenes"])
    shared = {name for name,n in pass_counts.items() if n > 1}

    def revision(candidate):
        values = local_scores(candidate, recent, shared)
        precision = 1 / scale**2 + np.array([v[1] for v in values])
        return float(timing @ np.clip([v[2] for v in values] / precision, -1.25, 1.25))

    original = revision(model)
    passes = [tuple(pass_id(q[k]) for k in ("scene_before","scene_after")) for q in model["observations"]]
    unique = sorted({p for two in passes for p in two})
    weights = model["sensitivity"] / model["sensitivity"].sum()
    records = []
    for name in unique:
        keep = np.array([name not in pair for pair in passes])
        retained = bool(np.any(keep))
        value = revision(subset(model,keep)) if retained else None
        records.append(dict(pass_id=name, affected_share=float(weights[~keep].sum()),
                            remaining_roofs=int(keep.sum()), deleted_from_revision_mbbl=value))
    values = np.array([q["deleted_from_revision_mbbl"] for q in records if q["remaining_roofs"]])
    shifts = values-original
    return dict(passes=len(unique), removable_passes=len(values), max_pass_share=max(q["affected_share"] for q in records),
                max_shift=float(np.max(abs(shifts),initial=0.)), rms_shift=float(np.sqrt(np.mean(shifts**2))) if len(shifts) else 0.,
                physical_revision=original, sign_reversals=int(np.count_nonzero(values*original < 0))), records


def covariates(row, *, conditional):
    u = row["baseline"]-row["unmeasured"]
    if not conditional:
        return np.array((u,))
    breadth = min(1., row["removable_passes"]/3.)
    stability = 1/(1+row["rms_shift"] / (.05 + abs(row["physical_revision"])))
    return u*np.array((1., breadth*(1-row["max_pass_share"])-.25, breadth*stability-.5))


def gains(history, row):
    base, point = row["unmeasured"], row["baseline"]
    if abs(point-base) <= 1e-10 or len(history) < MIN_WEEKS:
        return (1.,1.,1.)
    target = np.array([q["actual"]-q["baseline"] for q in history])
    age = np.array([(row["cutoff"]-q["known"]).total_seconds()/604800 for q in history])
    weight = np.exp2(-age/104)
    weight /= weight.sum()
    pooled, adapted = 1., 1.
    for conditional in (False, True):
        x = np.stack([covariates(q,conditional=conditional) for q in history])
        z = covariates(row,conditional=conditional)
        penalty = np.array((.04, .07, .07))[:x.shape[1]]

        def loss(beta):
            error = x @ beta-target
            smooth = np.hypot(error,.14)
            return weight @ smooth + .5*penalty @ (beta**2), x.T @ (weight*error/smooth) + penalty*beta

        fit = minimize(loss,np.zeros(len(z)),jac=True,method="L-BFGS-B",bounds=[(-1., 1.)]*len(z))
        if not fit.success or not np.isfinite(fit.x).all():
            raise ValueError(f"Pass response fit failed on {row['release_date']}")
        gain = float(np.clip(1+(z @ fit.x)/(point-base),0.,2.))
        if conditional:
            adapted = gain
        else:
            pooled = gain
    return pooled, adapted, (.5*adapted+.5)


def adjust(row, history):
    previous = [q for q in history if q["known"] < row["cutoff"] and abs(q["baseline"]-q["unmeasured"]) > 1e-10][-156:]
    values = gains(previous,row)
    update = row["baseline"]-row["unmeasured"]
    return row | {SUPPLEMENT:row["baseline"], **{m:row["unmeasured"]+g*update for m,g in zip(MODELS[1:],values,strict=True)}} | dict(
        latest_gain_label_utc=previous[-1]["known"] if previous else None, supervised_weeks=len(previous),
        **{f"{m}_multiplier":g for m,g in zip(MODELS[1:],values,strict=True)})


def replay():
    archive = {(q["release_date"],q["channel"]):q for q in channels()}
    state = dict(measured=[], errors=[])
    history = {"inventory":[], "market":[]}
    for row in inputs():
        issue = row["release_date"]
        diagnostics, removed = deletion_diagnostics(row,state)
        estimates, _, _, _, latest = forecast(row,state)
        if (not math.isclose(estimates[BASE], row["reference"], abs_tol=1e-9, rel_tol=0)
                or not math.isclose(estimates[LOCAL],row["public"]+diagnostics["physical_revision"],abs_tol=1e-9,rel_tol=0)):
            raise ValueError(f"Pass diagnostic does not reproduce frozen physics on {issue}")
        for name in history:
            current = archive[issue,name]
            if any(current[k] != row[k] for k in ("cutoff","known","week_ending","actual","study_phase")):
                raise ValueError(f"Unexpected report provenance on {issue}/{name}")
            current = current | diagnostics | dict(latest_physical_label_utc=latest)
            result = adjust(current,history[name])
            history[name].append(current)
            yield result, removed if name == "inventory" else None


def summarize(rows, trades):
    phases = {}
    for phase in PHASES:
        forecasts = {}
        for channel in ("inventory","market"):
            group = [q for q in rows if (q["study_phase"],q["channel"]) == (phase,channel)]
            measured = [q for q in group if abs(q["baseline"]-q["unmeasured"])>1e-10]
            samples = dict(all=group, measured=measured,
                           without_pass_diagnostic=[q for q in measured if not q["passes"]],
                           fully_dependent=[q for q in measured if q["passes"] and q["max_pass_share"] >= 1-1e-10],
                           distributed=[q for q in measured if q["passes"] and q["max_pass_share"] < 1-1e-10])
            forecasts[channel] = {}
            for name,sample in samples.items():
                if not sample:
                    continue
                actual = np.array([q["actual"] for q in sample])
                errors = {m:1000*abs(np.array([q[m] for q in sample])-actual) for m in ("unmeasured",*MODELS)}
                forecasts[channel][name] = dict(weeks=len(sample),
                    max_pass_deletion_shift_kbbl=1000*max(q["max_shift"] for q in sample),
                    weeks_with_direction_reversal=sum(bool(q["sign_reversals"]) for q in sample),
                    optical_wins=int(np.count_nonzero(errors[SUPPLEMENT] < errors["unmeasured"])),
                    results={m:dict(mae_kbbl=float(e.mean()), paired_delta_95_kbbl=block_bootstrap_mean(e-errors[SUPPLEMENT]))
                             for m,e in errors.items()})
        trade_methods = (*MODELS,CURRENT)
        groups = {m:[q for q in trades if (q["study_phase"],q["method"]) == (phase,f"{m}_conditional_exit")] for m in trade_methods}
        control = groups[SUPPLEMENT]
        trading = {}
        for m,group in groups.items():
            if [q["release_date"] for q in group] != [q["release_date"] for q in control]:
                raise ValueError(f"Different trade calendars in {phase}/{m}")
            trading[m] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(q["net_340_usd"] for q in group),
                changed_sides=sum(q["near_side"] != r["near_side"] for q,r in zip(group,control,strict=True)))
        phases[phase] = dict(forecasts=forecasts,trading=trading)
        print(phase, {m:round(v["mae_kbbl"],2) for m,v in forecasts["inventory"]["all"]["results"].items()},
              {m:v["net_170_usd"] for m,v in trading.items()})
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def forward(issue):
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    original = json.loads(original_path.read_bytes())
    distribution_path = DERIVED / f"{DISTRIBUTION}_{issue}_registration.json"
    overpass_path = DERIVED / f"{OVERPASS}_{issue}_registration.json"
    distribution, overpass = (json.loads(p.read_bytes()) for p in (distribution_path,overpass_path))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(q[k] != original[k] for q in (distribution,overpass) for k in LINEAGE):
        raise ValueError("The frozen inputs disagree on the report clock")
    for q in (distribution,overpass):
        checked_hashes(q["sources_sha256"])
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    if not path.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The pass model must be frozen before EIA publishes")
        recovery = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
        native = DERIVED / f"cushing_forward_native_optical_{issue}_summary.json"
        direction_path = DERIVED / f"cushing_optical_direction_{issue}_registration.json"
        height_path = DERIVED / f"cushing_height_response_curve_{issue}_registration.json"
        angle, height = (json.loads(p.read_bytes()) for p in (direction_path,height_path))
        area = float(json.loads(native.read_bytes())["dated_roof_area_m2"])
        observations = [r | dict(area=r["volume_per_m"] * BARREL_M3 * 1e6, available_utc=r["source_available_utc"])
                        for r in original["measured_pairs"]] + json.loads(recovery.read_bytes())["recovered_roofs"]
        covered = math.fsum(r["area"] for r in observations) / area
        if (len(observations) != distribution["measured_roofs"] or len(observations) != overpass["measured_roofs"]
                or len({r["candidate_id"] for r in observations}) != len(observations)
                or not math.isclose(covered,distribution["area_fraction"],abs_tol=1e-10)
                or any(datetime.fromisoformat(r["available_utc"]) >= cutoff for r in observations)):
            raise ValueError("The frozen set of roof pairs does not reproduce")
        current = dict(release_date=issue, week_ending=original["week_ending"], cutoff=cutoff,
                       public=original["public_ensemble_kbbl"]/1000, observed=observations,
                       hub_area=area, area_fraction=covered, spanning=False)
        state = dict(measured=[],errors=[])
        with threadpool_limits(limits=1):
            for row in inputs(before=cutoff):
                forecast(row,state)
            diagnostics, removed = deletion_diagnostics(current,state)
            archive = list(replay_inputs(cutoff))
            expected = overpass["estimates_kbbl"][LOCAL]/1000 - current["public"]
            if not math.isclose(diagnostics["physical_revision"],expected,abs_tol=1e-10,rel_tol=0):
                raise ValueError("The current full-pass estimate differs from the registered reference")
            projections = {}
            for name, q in distribution["projections"].items():
                estimates = q["estimates_kbbl"]
                point = estimates[SUPPLEMENT]/1000
                gain = q["pooled_gain"]
                if name == "market":
                    base = current["public"]
                else:
                    d = angle["channels"]["inventory"]["estimates_kbbl"]
                    h = height["estimates_kbbl"]
                    # In the long forecast, the unmeasured branch retains
                    # every public correction but none of the current optics.
                    pass_delta = overpass["estimates_kbbl"][LOCAL] - overpass["estimates_kbbl"][BASE]
                    delta = (d["same_week"] - d["no_revision"] + h["response_supported_by_labels"] - h[SUPPLEMENT]
                             + pass_delta)/1000
                    base = point-delta
                if not math.isclose(base+gain*(point-base),estimates["pooled_distribution"]/1000,abs_tol=1e-10,rel_tol=0):
                    raise ValueError(f"The optical-free forecast differs on {issue}/{name}")
                result = adjust(current | diagnostics | dict(baseline=point, unmeasured=base),
                                [r for r in archive if r["channel"] == name])
                latest = result["latest_gain_label_utc"]
                projections[name] = dict(estimates_kbbl={m:1000*result[m] for m in MODELS},
                    reference_without_optics_kbbl=1000*base, latest_label_utc=latest.isoformat(),
                    supervised_weeks=result["supervised_weeks"])
        stamp = datetime.now(timezone.utc)
        sources = (original_path,distribution_path,overpass_path,recovery,native,direction_path,height_path,
                   DERIVED / f"{STEM}_forecasts.csv",Path(__file__).resolve())
        if not cutoff <= stamp < deadline:
            raise ValueError("The pass reliability forecast missed its registration deadline")
        freeze(path,{k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(),
            diagnostics=diagnostics, deleted_passes=removed, projections=projections,
            sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources}))
    saved = json.loads(path.read_bytes())
    checked_hashes(saved["sources_sha256"])
    if (any(saved[k] != original[k] for k in LINEAGE)
            or any(datetime.fromisoformat(q["latest_label_utc"]) >= cutoff for q in saved["projections"].values())
            or not cutoff <= datetime.fromisoformat(saved["recorded_at_utc"]) < deadline):
        raise ValueError("The pass model timestamp does not reproduce")
    outcome = score_release(original_path,original)
    result = dict(status=outcome["status"], registration_sha256=digest(path), projections=saved["projections"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      errors_kbbl={c:{m:abs(v-actual) for m,v in q["estimates_kbbl"].items()} for c,q in saved["projections"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    print(result)


def replay_inputs(cutoff):
    for r in read(f"{STEM}_forecasts"):
        known = datetime.fromisoformat(r["known"])
        if known < cutoff:
            yield r | dict(known=known, cutoff=datetime.fromisoformat(r["cutoff"]),
                           **{k:float(r[k]) for k in ("baseline","unmeasured","actual","rms_shift","physical_revision",
                                                     "removable_passes","max_pass_share")})


def main():
    with threadpool_limits(limits=1):
        rows, audit = [], []
        for r, records in replay():
            rows.append(r)
            if records:
                audit.extend({k:r[k] for k in ("release_date","study_phase","cutoff","physical_revision")} | q for q in records)
        market = {q["release_date"]:q | {CURRENT:q["pass_gain"]} for q in rows if q["channel"] == "market"}
        trades = list(replay_weighted_market(market,(*MODELS,CURRENT), reference=SUPPLEMENT, historical_inputs={CURRENT:SUPPLEMENT}))
    summary = summarize(rows,trades)
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_deletions",audit)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(summary,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    forward(args.release.isoformat()) if args.release else main()
