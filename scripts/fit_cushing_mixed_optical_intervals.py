"""Fuse verified within-week and late optical pairs with the established observation panel."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_height_response_curve import STEM as HEIGHT
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_late_pair_recovery import accepted_pairs
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, observation, outer_pairs, prepared
from fit_cushing_overpass_covariance import inputs
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL, parts
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from replay_cushing_weighted_inventory_market import replay_weighted_market

STEM = "cushing_mixed_optical_intervals"
MODES = (SUPPLEMENT, "internal_only", "joint_intervals", "conservative_intervals", "learned_interval_weight", "weight_by_footprint")
KINDS = MODES[:4]


def admissible(row, internal, late):
    occupied = {q["candidate_id"] for q in row["observed"]}
    inside = [q for q in internal if q["candidate_id"] not in occupied]
    occupied.update(q["candidate_id"] for q in inside)
    afterward = [q for q in late if q["candidate_id"] not in occupied]
    merged = inside + afterward + row["observed"]
    if len({q["candidate_id"] for q in merged}) != len(merged):
        raise ValueError(f"Multiple optical intervals for one roof on {row['release_date']}")
    for q in merged:
        ready = q["available_utc"]
        if (datetime.fromisoformat(ready) if isinstance(ready, str) else ready) >= row["cutoff"]:
            raise ValueError(f"A late optical pair entered the dated forecast on {row['release_date']}")
    return inside, afterward


def model(row, target, original, internal, late, inflation):
    if not internal and not late:
        return original
    primary = original["observations"] if original is not None else []
    sample = primary + [q | dict(variance=q["variance"] * inflation) for q in internal + late]
    if not sample:
        return None
    n, m, k = len(primary), len(internal), len(late)
    calendar = defaultdict(list)

    def extend(day, times, areas, lead, tail):
        calendar[day].append((np.pad(times, (lead, tail)), np.pad(areas, (lead, tail))))

    if original:
        for day, (t,a) in original["clocks"].items():
            extend(day, t, a, 0, m+k)
    week = date.fromisoformat(row["week_ending"])
    next_week = week + timedelta(days=7)
    if internal:
        area = np.array([q["mapped_hub_fraction"] for q in internal])
        for edge, sign in (("first", -1), ("last", 1)):
            extend(week, np.array([q[f"internal_{edge}"] for q in internal]), sign * area, n, k)
    if late:
        area = np.array([q["mapped_fraction"] for q in late])
        extend(week, np.array([q["starting_fraction"] for q in late]), -area, n+m, 0)
        extend(next_week, np.array([q["trailing_fraction"] for q in late]), area, n+m, 0)
    sensitivity, residual, local = parts(sample, target)
    # Both pair types ending beyond the survey boundary contain the same
    # unknown following-week change, with separate exposure for each roof.
    future = np.r_[[q.get("post_fraction", 0.) * q.get("mapped_hub_fraction", 0.) for q in primary], np.zeros(m),
                   [q["trailing_fraction"] * q["mapped_fraction"] for q in late]]
    local += PRIOR_INVENTORY_SIGMA_MBBL**2 * np.outer(future, future)
    clocks = {d:tuple(v) for d,v in calendar.items()}
    return dict(observations=sample, sensitivity=sensitivity, residual=residual, local=local, clocks=clocks,
                scenes={q[key] for q in sample for key in ("scene_before", "scene_after")})


def fitted(row, current, actual, history, sigma):
    if current is None:
        return row["public"], None
    known = [q for q in history if q["known"] < row["cutoff"]]
    recent = [q["prepared"] for q in known if row["cutoff"] - q["cutoff"] <= MAX_SCENE_AGE]
    values = estimate(current, recent)
    projected = np.clip([s / (1 / sigma**2 + i) for _,i,s in values], -1.25, 1.25)
    term = float(learned_weights(known) @ projected)
    history.append(dict(cutoff=row["cutoff"], known=row["known"], prepared=actual,
                        absolute_errors=np.abs([row["public"] + p[0] - row["actual"] for p in values])))
    return row["public"] + term, known[-1]["known"] if known else None


def calibrated_weight(history, cutoff, *, anchor=.5):
    previous = [q for q in history if q["known"] < cutoff][-104:]
    if len(previous) < 24:
        return anchor, previous[-1]["known"] if previous else None
    delta = np.array([q["delta"] for q in previous])
    target = np.array([q["actual"] - q["baseline"] for q in previous])
    age = np.array([(cutoff-q["cutoff"]).total_seconds() / 604800 for q in previous])
    decay = np.exp2(-age / 78)
    penalty = 8 * .15**2
    gain = anchor
    for _ in range(6):
        weight = decay / np.hypot(1., (target - delta * gain) / .60)
        gain = float(np.clip(((weight * delta) @ target + anchor * penalty) / (weight @ delta**2 + penalty), 0., 1.))
    return gain, previous[-1]["known"]


def forecast_rows():
    archive = indexed(f"{HEIGHT}_forecasts")
    late = accepted_pairs()
    interior = outer_pairs("cushing_outer_capture_pairs_internal_fits")
    histories = defaultdict(list)
    errors, revisions = [], []
    footprint_residuals = defaultdict(list)
    audit = []
    for row in inputs():
        release, cutoff, known, actual = (row[k] for k in ("release_date", "cutoff", "known", "actual"))
        anchor = archive[release]
        if (any(datetime.fromisoformat(anchor[k]) != row[k] for k in ("cutoff", "known")) or float(anchor["actual"]) != actual
                or any(anchor[k] != row[k] for k in ("week_ending", "study_phase"))):
            raise ValueError(f"Different original forecast date or target on {release}")
        inside, after = admissible(row, [observation(q,row,row["hub_area"],0.,internal=True) for q in interior[release]], late[release])
        gain, training = calibrated_weight(revisions, cutoff)
        main_estimate = float(anchor["long_response_supported_by_labels"])
        footprint = bool(row["observed"])
        footprint_gain, footprint_label = calibrated_weight(footprint_residuals[footprint], cutoff, anchor=0.)
        sigma = prior_scale([q for q in errors if q["known"] < cutoff])
        targets = (row["public"], actual)
        original = tuple(prepared(row,t,row["spanning"]) if row["observed"] else None for t in targets)
        estimates, labels = {}, {}
        for name, (normal, trailing, noise) in zip(KINDS, (([], [], 1.), (inside, [], 1.), (inside, after, 1.), (inside, after, 2.)), strict=True):
            pair = tuple(model(row, t, p, normal, trailing, noise) for t,p in zip(targets,original,strict=True))
            estimates[name], labels[name] = fitted(row, *pair, histories[name], sigma)
        if not math.isclose(estimates[SUPPLEMENT], row["reference"], abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"The original satellite forecast failed to reproduce on {release}")
        baseline = estimates[SUPPLEMENT]
        delta = estimates["joint_intervals"] - baseline
        estimates["learned_interval_weight"] = baseline + gain * delta
        estimates["weight_by_footprint"] = baseline + footprint_gain * delta
        if inside or after:
            revisions.append(dict(cutoff=cutoff, known=known, baseline=baseline, delta=delta, actual=actual))
            footprint_residuals[footprint].append(dict(cutoff=cutoff, known=known, baseline=main_estimate, delta=delta, actual=actual))
        errors.append(dict(known=known, error=actual-row["public"]))
        for kind, rows in (("within_week", inside), ("ending_after_week", after)):
            for q in rows:
                audit.append(dict(release_date=release, study_phase=row["study_phase"], kind=kind, cutoff=cutoff,
                    candidate_id=q["candidate_id"], terminal_id=q["terminal_id"], scene_before=q["scene_before"],
                    scene_after=q["scene_after"], available_utc=q["available_utc"], area_m2=q["area"], sensitivity=q["sensitivity"]))
        fields = ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual")
        yield {k:row[k] for k in fields} | estimates | dict(primary_roofs=len(row["observed"]), internal_roofs=len(inside),
            late_roofs=len(after), added_area_fraction=math.fsum(q["area"] for q in inside+after) / row["hub_area"],
            learned_weight=gain, latest_gain_label_utc=training, latest_optical_label_utc=max((q for q in labels.values() if q), default=None),
            footprint_weight=footprint_gain, latest_footprint_label_utc=footprint_label,
            **{f"long_{m}":main_estimate + v - baseline for m,v in estimates.items()}), audit
        audit = []


def inventory(sample):
    if not sample:
        return {}
    actual = np.array([r["actual"] for r in sample])
    losses = {m:1000. * abs(np.array([r[f"long_{m}"] for r in sample]) - actual) for m in MODES}
    return {m:dict(releases=len(e), mae_kbbl=float(e.mean()), delta_vs_reference_95_kbbl=block_bootstrap_mean(e - losses[SUPPLEMENT]))
            for m,e in losses.items()}


def summarize(forecasts, trades):
    report = {}
    for phase in PHASES:
        sample = [r for r in forecasts if r["study_phase"] == phase]
        groups = {"all":sample, "new_on_blank":[r for r in sample if not r["primary_roofs"] and r["internal_roofs"]+r["late_roofs"]],
                  "more_on_measured":[r for r in sample if r["primary_roofs"] and r["internal_roofs"]+r["late_roofs"]],
                  "internal":[r for r in sample if r["internal_roofs"]], "late":[r for r in sample if r["late_roofs"]]}
        trades_by_method = defaultdict(list)
        for q in trades:
            if q["study_phase"] == phase:
                trades_by_method[q["method"]].append(q)
        metrics = {}
        for name, group in trades_by_method.items():
            baseline = trades_by_method[f"{SUPPLEMENT}_conditional_exit" if name.endswith("_conditional_exit") else SUPPLEMENT]
            if [q["release_date"] for q in group] != [q["release_date"] for q in baseline]:
                raise ValueError("A timing variant changed the comparison calendar")
            increments = [r["net_170_usd"] - q["net_170_usd"] for r,q in zip(group,baseline,strict=True)]
            metrics[name] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(q["net_340_usd"] for q in group),
                delta_vs_reference_95_usd=block_bootstrap_mean(increments))
        report[phase] = dict(inventory={k:inventory(q) for k,q in groups.items()}, trading=metrics,
                             internal_roofs=sum(r["internal_roofs"] for r in sample), late_roofs=sum(r["late_roofs"] for r in sample))
        print(phase, {k:{m:round(v["mae_kbbl"], 2) for m,v in q.items()} for k,q in report[phase]["inventory"].items()},
              {m:v["net_170_usd"] for m,v in metrics.items()})
    return dict(status="exploratory_on_reviewed_history", phases=report)


def main():
    with threadpool_limits(limits=1):
        pairs = list(forecast_rows())
        rows = [r for r,_ in pairs]
        trades = list(replay_weighted_market({q["release_date"]:q for q in rows}, MODES, reference=SUPPLEMENT))
    result = summarize(rows, trades)
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_trades", trades)
    export(f"{STEM}_admitted_pairs", [q for _,a in pairs for q in a])
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())


def forward(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    origin = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    geometry = DERIVED / f"cushing_forward_optical_coverage_{issue}_summary.json"
    heights = DERIVED / f"{HEIGHT}_{issue}_registration.json"
    optical = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    original, coverage, h, s = (json.loads(q.read_bytes()) for q in (origin, geometry, heights, optical))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(q[k] != original[k] for q in (h,s) for k in LINEAGE) or h["measured_roofs"] != s["measured_roofs"]:
        raise ValueError("The forward observations have mismatched clocks or roof counts")
    if not path.exists():
        stamp = datetime.now(timezone.utc)
        if not cutoff <= stamp < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The expanded forecast must be recorded before publication")
        for q in (h,s):
            checked_hashes(q["sources_sha256"])
        passes = [q for q in coverage["passes"] if q["distinct_clear_roofs"]]
        if (coverage["release_date"] != issue or any(q["segment"] == "after_target" for q in passes)
                or len({q["date"] for q in passes if q["segment"] == "target_week"}) != 1
                or s["optical_input_policy"] != "frozen_panel_without_eligible_after_target_observations"):
            raise ValueError("Additional timely optical intervals require fitting before registration")
        # These are same-panel current-pair deltas. With no new pairs the
        # forecasts carry the frozen baseline, including its original fit.
        sources = (origin, geometry, heights, optical, ROOT / coverage["source_catalog"], Path(__file__).resolve(),
                   DERIVED / f"{STEM}_forecasts.csv")
        freeze(path, {k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), measured_roofs=h["measured_roofs"],
            additional_roofs=0, forecast_basis="same_frozen_observations_without_additional_pairs",
            estimates_kbbl=dict.fromkeys(MODES, h["long_estimates_kbbl"]["response_supported_by_labels"]),
            sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))
    reg = json.loads(path.read_bytes())
    checked_hashes(reg["sources_sha256"])
    if (any(reg[k] != original[k] for k in LINEAGE) or not cutoff <= datetime.fromisoformat(reg["recorded_at_utc"]) < deadline
            or set(reg["estimates_kbbl"]) != set(MODES)):
        raise ValueError("The saved mixed-interval forecast failed its time or model checks")
    outcome = score_release(origin, original)
    result = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path), additional_roofs=reg["additional_roofs"],
                  estimates_kbbl=reg["estimates_kbbl"], forecast_basis=reg["forecast_basis"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      errors_kbbl={m:abs(v - actual) for m,v in reg["estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    forward(args.release.isoformat()) if args.release else main()
