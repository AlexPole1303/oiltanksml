"""Test persistent zero-sum terminal movements on independent satellite passes."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky
from threadpoolctl import threadpool_limits

from audit_cushing_cl_equal_model_risk import paired_evidence
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_curve_shape import MIXED, MIXED_GAP, STEM as ANCHORS, market_metrics
from fit_cushing_intraweek import SCALES, estimate, learned_weights, prior_scale, time_covariance
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import inputs, pass_id
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, SCENE_SIGMA_M, scene_design
from fit_cushing_terminal_balance import DEFAULT, mapped_areas
from fit_cushing_terminal_weekly_transfers import terminal_design
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from replay_cushing_weighted_inventory_market import replay_weighted_market

STEM = "cushing_transfer_persistence"
INDEPENDENT = "transfer_independent_weeks"
MEAN_REVERTING = "transfer_mean_reversion"
PERSISTENT = "transfer_persistence"
ADAPTIVE = "transfer_learned_persistence"
JOINT = "hybrid_transfer_learned_persistence"
DYNAMIC_ONLY = "hybrid_transfer_dynamics_only"
PERSISTENCE_ONLY = "hybrid_positive_transfer_dynamics_only"
RHOS = {INDEPENDENT: 0., MEAN_REVERTING: -.5, PERSISTENT: .5}
HYBRIDS = (JOINT, DYNAMIC_ONLY, PERSISTENCE_ONLY)
METHODS = (SUPPLEMENT, MIXED_GAP, *RHOS, ADAPTIVE, *HYBRIDS)
TRAINING_WEEKS = 78
MIN_PAIRS = 24
TRANSFER_VARIANCE = DEFAULT[-1]


def temporal_covariance(left, right, rho):
    return sum((rho**(abs((a - b).days) // 7)) * (x @ y.T) for a, x in left.items() for b,y in right.items())


def historical_candidate(current, past):
    return next((row for row in reversed(past)
                 if row["known"] < current["cutoff"] and current["cutoff"] - row["cutoff"] <= MAX_SCENE_AGE
                 and row["sites"] & current["sites"] and not row["passes"] & current["passes"]
                 and max(row["loads"]) < min(current["loads"])), None)


def covariance(model, timing_variance):
    scene = scene_design(model["observations"], {name:i for i,name in enumerate(sorted(model["scenes"]))})
    return model["local"] + SCENE_SIGMA_M**2 * (scene @ scene.T) + timing_variance * time_covariance([model])


def hindsight_scores(current, previous, timing_variance):
    # For measurement validation, condition on the published hub change
    # in each week. These scores enter model selection once the newer label is known.
    now, then = current["prepared"], previous["prepared"]
    cov_now, cov_then = (covariance(q, timing_variance) for q in (now, then))
    y = np.clip(then["residual"], -3 * np.sqrt(cov_then.diagonal()), 3 * np.sqrt(cov_then.diagonal()))
    losses, errors, transfers = [], [], []
    for rho in RHOS.values():
        older = cov_then + TRANSFER_VARIANCE * temporal_covariance(previous["loads"], previous["loads"], rho)
        newer = cov_now + TRANSFER_VARIANCE * temporal_covariance(current["loads"], current["loads"], rho)
        shared = TRANSFER_VARIANCE * temporal_covariance(current["loads"], previous["loads"], rho)
        projected = cho_solve(cho_factor(older), np.column_stack((y, shared.T)))
        shift = shared @ projected[:,0]
        uncertainty = newer - shared @ projected[:,1:]
        error = now["residual"] - shift
        factor = cho_factor(uncertainty)
        nll = .5 * (2 * np.log(factor[0].diagonal()).sum() + error @ cho_solve(factor, error)) / len(error)
        losses.append(float(nll))
        errors.append(1000 * float(np.mean(np.abs(error))))
        transfers.append(1000 * float(np.mean(np.abs(shift))))
    return np.array(losses), errors, transfers


def fit_temporal_models(current, nearby, names, rho):
    group = [*nearby, current]
    days = sorted({d for row in group for d in row["loads"]})
    offsets = np.array([(day - days[0]).days // 7 for day in days])
    correlation = rho**np.abs(offsets[:,None] - offsets[None,:])
    factor = cholesky(correlation, lower=True) * math.sqrt(TRANSFER_VARIANCE)
    keys = [(d.isoformat(), name) for d in days for name in names]
    models = []
    for row in group:
        template = next(iter(row["loads"].values()))
        load = np.stack([row["loads"].get(day, np.zeros_like(template)) for day in days], axis=1)
        rotated = np.einsum("ndk,dt->ntk", load, factor).reshape(len(template), -1)
        endpoint = [{k:float(v) for k,v in zip(keys, obs, strict=True) if v} for obs in rotated]
        # All selected recent releases belong to the joint covariance.
        # This alias carries no pixel-level scene loading.
        model = row["prepared"]
        models.append(model | dict(endpoint_errors=endpoint, scenes=model["scenes"] | {"terminal-week-covariance"}))
    return estimate(models[-1], models[:-1])


def trained_weights(scores, cutoff):
    past = [r for r in scores if r["known"] < cutoff][-TRAINING_WEEKS:]
    if len(past) < MIN_PAIRS:
        return np.array((1.,0.,0.)), past
    age = np.array([(cutoff - r["known"]).total_seconds() / 604800 for r in past])
    age_weights = np.exp2(-age / 52.)
    loss = np.stack([q["nll"] for q in past])
    mean = np.average(loss, weights=age_weights, axis=0)
    # Small likelihood differences stay close to the independent-week model.
    logits = np.log((.70,.15,.15)) - (mean - mean.min()) / .04
    weights = np.exp(logits - logits.max())
    return weights / weights.sum(), past


def transfer_updates(row, recent, ready, errors, loads, names):
    cutoff, public = row["cutoff"], row["public"]
    sigma = prior_scale([q for q in errors if q["known"] < cutoff])
    timing = learned_weights(ready)
    current = prepared(row, public, row["spanning"])
    original = np.array(estimate(current, [r["prepared"] for r in recent]))
    reference = public + float(timing @ np.clip(original[:,2] / (sigma**-2 + original[:,1]), -1.25, 1.25))
    if not math.isclose(reference, row["reference"], abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"The original optical calculation differs on {row['release_date']}")
    updates = {}
    for name, rho in RHOS.items():
        fit = np.array(fit_temporal_models(dict(prepared=current, loads=loads), recent, names, rho))
        point = public + float(timing @ np.clip(fit[:,2] / (sigma**-2 + fit[:,1]), -1.25, 1.25))
        updates[name] = point - reference
    return updates, timing, original[:,0]


def projected(base, joint, updates, mixture):
    by_time = float(mixture @ list(updates.values()))
    return {SUPPLEMENT:base, MIXED_GAP:joint, ADAPTIVE:base+by_time, JOINT:joint+by_time,
            DYNAMIC_ONLY:joint+(by_time-updates[INDEPENDENT]),
            PERSISTENCE_ONLY:joint+(updates[PERSISTENT]-updates[INDEPENDENT]),
            **{k:base+delta for k,delta in updates.items()}}


def forecasts(diagnostics, *, state=None, before=None, additional_sites=()):
    anchors = {(r["release_date"], r["channel"]):r for r in read(f"{ANCHORS}_forecasts")}
    roster = mapped_areas()
    names = sorted({name for r in roster.values() for name in r} | set(additional_sites))
    seen, errors, scored = [], [], []
    for row in inputs():
        issue, cutoff, actual, public = (row[k] for k in ("release_date", "cutoff", "actual", "public"))
        if before is not None and row["known"] >= before:
            break
        ready = [r for r in seen if r["known"] < cutoff]
        recent = [r for r in ready if cutoff-r["cutoff"] <= MAX_SCENE_AGE]
        updates = dict.fromkeys(RHOS, 0.)
        mixture, basis = trained_weights(scored, cutoff)
        previous = None
        if row["observed"]:
            known = row["known"]
            loads = terminal_design(row, roster[issue,row["week_ending"]], names)
            realised = prepared(row, actual, row["spanning"])
            updates, timing, original = transfer_updates(row, recent, ready, errors, loads, names)
            data = dict(cutoff=cutoff, known=known, loads=loads, prepared=realised,
                        sites={q["terminal_id"] for q in row["observed"]},
                        passes={pass_id(q[k]) for q in row["observed"] for k in ("scene_before", "scene_after")},
                        absolute_errors=abs(public + original - actual))
            previous = historical_candidate(data, recent)
            if previous:
                losses, errors_kbbl, transfer = hindsight_scores(data, previous, float(timing @ np.square(SCALES)))
                scored.append(dict(known=known, nll=losses))
                for i, name in enumerate(RHOS):
                    diagnostics.append(dict(release_date=issue, study_phase=row["study_phase"], method=name,
                        label_known_utc=known, predecessor_known_utc=previous["known"], roofs=len(row["observed"]),
                        elapsed_days=(cutoff-previous["cutoff"]).total_seconds()/86400,
                        mean_error_kbbl=errors_kbbl[i], nll_per_roof=losses[i], mean_transfer_correction_kbbl=transfer[i]))
            seen.append(data)
        errors.append(dict(known=row["known"], error=actual-public))
        identity = {k:row[k] for k in ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual")}
        for channel in ("inventory", "market"):
            anchor = anchors[issue,channel]
            if (any(datetime.fromisoformat(anchor[k]) != row[k] for k in ("cutoff", "known"))
                    or float(anchor["actual"]) != actual or anchor["week_ending"] != row["week_ending"]):
                raise ValueError(f"The transfer forecast and original labels differ on {issue}")
            base, joint = (float(anchor[m]) for m in (SUPPLEMENT, MIXED_GAP))
            yield identity | dict(channel=channel, roofs=len(row["observed"]), independent_predecessor=bool(previous),
                latest_transfer_training_label_utc=basis[-1]["known"] if basis else None, transfer_training_pairs=len(basis),
                **projected(base, joint, updates, mixture), **{f"{m}_weight":float(w) for m,w in zip(RHOS,mixture,strict=True)})
    if state is not None:
        state.update(seen=seen, errors=errors, scored=scored, terminal_names=names)


def summarize(rows, diagnostics, trades):
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        release = [r for r in rows if r["study_phase"] == phase and r["channel"] == "inventory"]
        inventory = {}
        for key,sample in (("all", release), ("independent_pass_predecessor", [r for r in release if r["independent_predecessor"]])):
            errors = {m:1000*np.array([abs(r[m]-r["actual"]) for r in sample]) for m in METHODS}
            inventory[key] = {m:dict(weeks=len(e), mae_kbbl=float(e.mean()), paired_95_kbbl=block_bootstrap_mean(
                e-errors[MIXED_GAP if m in (MIXED_GAP, *HYBRIDS) else SUPPLEMENT]))
                              for m,e in errors.items()}
        validation = {}
        independent = [r for r in diagnostics if r["study_phase"] == phase and r["method"] == INDEPENDENT]
        for m in RHOS:
            q = [r for r in diagnostics if r["study_phase"] == phase and r["method"] == m]
            if [r["release_date"] for r in q] != [r["release_date"] for r in independent]:
                raise ValueError(f"Incompatible diagnostic dates for {phase}/{m}")
            validation[m] = dict(weeks=len(q), **{k:float(np.mean([r[k] for r in q])) for k in (
                "mean_error_kbbl", "mean_transfer_correction_kbbl", "nll_per_roof")},
                paired_95_kbbl=block_bootstrap_mean([r["mean_error_kbbl"]-b["mean_error_kbbl"] for r,b in zip(q,independent,strict=True)]))
        market = {}
        for suffix in ("", "_conditional_exit"):
            books = {m:[r for r in trades if r["study_phase"] == phase and r["method"] == m+suffix] for m in METHODS}
            for m,g in books.items():
                baseline = MIXED_GAP if m in (MIXED_GAP, *HYBRIDS) else SUPPLEMENT
                market[m+suffix] = market_metrics(g) | dict(paired_vs_anchor=paired_evidence(g, books[baseline])[0])
        report["phases"][phase] = dict(inventory=inventory, independent_pass_validation=validation, market=market)
        print(phase, {m:round(v["mae_kbbl"],3) for m,v in inventory["all"].items()},
              {m:market[m+"_conditional_exit"]["net_170_usd"] for m in METHODS},flush=True)
    return report


def main():
    diagnostics = []
    with threadpool_limits(limits=1):
        rows = list(forecasts(diagnostics))
        trades = list(replay_weighted_market({r["release_date"]:r for r in rows if r["channel"] == "market"}, METHODS, reference=SUPPLEMENT))
    report = summarize(rows, diagnostics, trades)
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_independent_pass", diagnostics)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())


def register(issue, path, origin, original):
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
        raise ValueError("The terminal forecast must be registered before EIA publication")
    recovery = origin.with_name(f"{origin.stem}_recovered_optical.json")
    market = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    mixed = DERIVED / f"cushing_mixed_optical_intervals_{issue}_registration.json"
    native = DERIVED / f"cushing_forward_native_optical_{issue}_summary.json"
    coverage = DERIVED / f"cushing_forward_optical_coverage_{issue}_summary.json"
    recovered, trade, intervals, geometry, window = (json.loads(p.read_bytes()) for p in (recovery, market, mixed, native, coverage))
    for record in (trade, intervals):
        checked_hashes(record["sources_sha256"])
    if any(r[k] != original[k] for r in (trade, intervals, recovered) for k in LINEAGE):
        raise ValueError("The frozen transfer inputs cover different report periods")
    observations = [q | dict(area=q["volume_per_m"] * BARREL_M3 * 1e6, available_utc=q["source_available_utc"])
                    for q in original["measured_pairs"]] + recovered["recovered_roofs"]
    if (not len({q["candidate_id"] for q in observations}) == len(observations) == trade["measured_roofs"] == intervals["measured_roofs"]
            or not observations or intervals["additional_roofs"] or any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observations)):
        raise ValueError("The transfer forecast requires the original nonempty optical panel")
    roster_stem = "cushing_modern_lidar_optical_registry"
    roster = defaultdict(float)
    preceding = window["preceding_week_end_utc"][:10]
    for q in read(roster_stem):
        if q["modern_input_status"] == "available" and q["latest_engineering_date"] < cutoff.date().isoformat() and q["first_seen"] <= preceding:
            roster[q["terminal_id"]] += float(q["circular_area_m2"])
    area = geometry["dated_roof_area_m2"]
    price_base = trade["estimates_kbbl"][SUPPLEMENT] / 1000
    physical_base = intervals["estimates_kbbl"][MIXED] / 1000
    row = dict(release_date=issue, week_ending=original["week_ending"], cutoff=cutoff, public=original["public_ensemble_kbbl"] / 1000,
               observed=observations, area_fraction=math.fsum(r["area"] for r in observations) / area,
               hub_area=area, reference=price_base, spanning=False)
    state = {}
    with threadpool_limits(limits=1):
        for _ in forecasts([], state=state, before=cutoff, additional_sites=roster):
            pass
        names = state["terminal_names"]
        loads = terminal_design(row, {k:(v,cutoff) for k,v in roster.items()}, names)
        ready = state["seen"]
        recent = [r for r in ready if cutoff-r["cutoff"] <= MAX_SCENE_AGE]
        updates, _, _ = transfer_updates(row, recent, ready, state["errors"], loads, names)
        weights, basis = trained_weights(state["scored"], cutoff)
    stamp = datetime.now(timezone.utc)
    if not basis or not basis[-1]["known"] < cutoff <= stamp < deadline:
        raise ValueError("The trained terminal projection missed the report cutoff")
    predictions = {channel:{k:1000*v for k,v in projected(base, base, updates, weights).items()}
                   for channel,base in (("market",price_base), ("inventory",physical_base))}
    sources = (origin, recovery, market, mixed, native, coverage, DERIVED/f"{roster_stem}.csv", DERIVED/f"{STEM}_forecasts.csv",
               DERIVED/f"{STEM}_independent_pass.csv", Path(__file__).resolve(), ROOT/"scripts/fit_cushing_intraweek.py",
               ROOT/"scripts/fit_cushing_terminal_weekly_transfers.py")
    freeze(path, {k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), measured_roofs=len(observations),
        transfer_training_pairs=len(basis), latest_training_label_utc=basis[-1]["known"].isoformat(),
        weights=dict(zip(RHOS, (float(v) for v in weights), strict=True)), estimates_kbbl=predictions,
        sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))


def forward(issue):
    origin = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    original = json.loads(origin.read_bytes())
    path = DERIVED / f"{STEM}_{issue}_registration.json"
    if not path.exists():
        register(issue, path, origin, original)
    saved = json.loads(path.read_bytes())
    checked_hashes(saved["sources_sha256"])
    latest, cutoff, stamp, deadline = (datetime.fromisoformat(saved[k]) for k in (
        "latest_training_label_utc", "information_cutoff_utc", "recorded_at_utc", "registration_deadline_utc"))
    if not latest < cutoff <= stamp < deadline or any(saved[k] != original[k] for k in LINEAGE):
        raise ValueError("The frozen terminal forecast clock does not reproduce")
    outcome = score_release(origin, original)
    result = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path), estimates_kbbl=saved["estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_errors_kbbl={k:{m:abs(v-actual) for m,v in row.items()} for k,row in saved["estimates_kbbl"].items()})
    write(DERIVED/f"{STEM}_{issue}_score.json", (json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    forward(args.release.isoformat()) if args.release else main()
