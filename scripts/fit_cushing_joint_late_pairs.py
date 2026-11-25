"""Fuse independently registered late pairs with an existing optical week."""

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_late_pair_recovery import COMBINED, HYBRID, STEM as LATE, accepted_pairs
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT, observation, outer_pairs, prepared
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL, parts

STEM = "cushing_joint_late_pairs"
LATE_METHODS = ("joint_late_full", "joint_late_half", "joint_late_calibrated")
CARRY_METHODS = ("late_carry_full", "late_carry_half", "late_carry_calibrated")
METHODS = (*LATE_METHODS, *CARRY_METHODS)
MODELS = (PUBLIC, SUPPLEMENT, COMBINED, *METHODS)


def joint_model(main, extra, target, week, *, carried):
    """Retain common scene, terminal and survey-clock errors across both sets."""
    ordinary = main["observations"]
    sample = ordinary + extra
    sensitivity, residual, local = parts(sample, target)
    n, m = len(ordinary), len(extra)
    start, tail, mapped = (np.array([q[k] for q in extra])
                           for k in ("starting_fraction", "trailing_fraction", "mapped_fraction"))
    early_week = week - timedelta(days=7) if carried else week
    late_week = early_week + timedelta(days=7)
    clocks = {day: ((np.r_[times, np.zeros(m)], np.r_[areas, np.zeros(m)]),)
              for day, (times, areas) in main["clocks"].items()}
    for day, elapsed, signed_area in ((early_week, start, -mapped), (late_week, tail, mapped)):
        clocks[day] = (*clocks.get(day, ()), (np.r_[np.zeros(n), elapsed], np.r_[np.zeros(n), signed_area]))
    # The same unobserved following-week change affects all pairs that end
    # after the target boundary. In carried pairs this endpoint is earlier.
    ordinary_exposure = [q.get("post_fraction", 0.) * q.get("mapped_hub_fraction", 0.) for q in ordinary]
    exposure = np.r_[ordinary_exposure, np.zeros(m) if carried else tail*mapped]
    local += PRIOR_INVENTORY_SIGMA_MBBL**2 * np.outer(exposure, exposure)
    scenes = {q[k] for q in sample for k in ("scene_before", "scene_after")}
    return dict(observations=sample, sensitivity=sensitivity, residual=residual, local=local,
                clocks=clocks, scenes=scenes)


def source_rows():
    fit = indexed(f"{LATE}_forecasts")
    forecasts = indexed(f"{OUTER}_forecasts")
    registry = {(q["release_date"], q["week_ending"]): q for name in (
        "sentinel2_space_time_coverage_weeks", "holdout_optical_weeks") for q in read(name)}
    past_report = indexed("cushing_robust_stack_forecasts")
    spans = outer_pairs("cushing_outer_capture_pairs_fits")
    supplements = outer_pairs("cushing_outer_capture_pairs_supplement_fits")
    late = accepted_pairs()
    previous_weeks = {}
    for r, views, _ in combined_sources():
        issue, week, cutoff, known = (r[k] for k in ("release_date", "week_ending", "cutoff", "known"))
        f, o = fit[issue], forecasts[issue]
        for q, cutoff_key, known_key in ((f, "cutoff", "known"), (o, "cutoff_utc", "known_utc")):
            if (datetime.fromisoformat(q[cutoff_key]) != cutoff or datetime.fromisoformat(q[known_key]) != known
                    or float(q["actual"]) != r["actual"] or float(q[PUBLIC]) != r["public"]
                    or q["week_ending"] != week or q["study_phase"] != r["study_phase"]
                    or any(datetime.fromisoformat(stamp) >= cutoff for key, stamp in q.items()
                           if stamp and key.endswith(("latest_label_utc", "latest_training_label_utc", "latest_optical_at_utc")))):
                raise ValueError(f"Invalid forecast lineage on {issue}")
        area = float(registry[issue, week]["dated_reference_area_m2"])
        prior = float(past_report[issue]["prior"])
        ordinary = views[-1]
        outside = [observation(q, r, area, prior) for q in spans[issue]]
        extra = [observation(q, r, area, prior) for q in supplements[issue]]
        main = ordinary + outside + extra
        ids = {q["candidate_id"] for q in main}
        additional = [q for q in late[issue] if q["candidate_id"] not in ids] if main else []
        as_week = date.fromisoformat(week)
        previous = previous_weeks.get(as_week - timedelta(days=7))
        carried = []
        carry_label = None
        if previous and previous["known"] < cutoff and main:
            occupied = ids | previous["main_ids"] | {q["candidate_id"] for q in additional}
            carried = [q | dict(sensitivity=q["mapped_fraction"] * q["trailing_fraction"],
                                prior_component=q["sensitivity"] * previous["actual"])
                       for q in previous["pairs"] if q["candidate_id"] not in occupied]
            if carried:
                carry_label = previous["known"]
        previous_weeks[as_week] = dict(known=known, actual=r["actual"], main_ids=ids, pairs=late[issue])
        rows = main + additional + carried
        if (not cutoff < known or bool(main) != (f["measured"] == "True")
                or len(main) != len(ids) or len(late[issue]) != int(f["late_roofs"])
                or any((t if isinstance(t, datetime) else datetime.fromisoformat(t)) >= cutoff
                       for t in (q["available_utc"] for q in rows))):
            raise ValueError(f"Future or duplicated optical observation on {issue}")
        main_exposure = math.fsum(q["area"] for q in main) / area
        extra_exposure = math.fsum(q["sensitivity"] for q in additional)
        accounting = dict(main_roofs=len(main), late_roofs_available=len(late[issue]),
                          added_roofs=len(additional), main_area_fraction=main_exposure,
                          added_area_time=extra_exposure,
                          added_terminals=len({q["terminal_id"] for q in additional} - {q["terminal_id"] for q in main}),
                          carried_roofs=len(carried), carried_area_time=math.fsum(q["sensitivity"] for q in carried),
                          carried_from_label_utc=carry_label)
        yield r | {PUBLIC: r["public"], SUPPLEMENT: float(o[SUPPLEMENT]), HYBRID: float(f[HYBRID]),
                   COMBINED: float(f[COMBINED])} | dict(observed=main, area_fraction=main_exposure, hub_area=area,
                                                       carried_scene=float(f["carried_scene"])), additional, carried, accounting


def gain(past, row):
    if not past:
        return .5
    revision, error = (np.array([q[k] for q in past]) for k in ("revision", "reference_error"))
    age = np.array([(row["cutoff"] - q["cutoff"]).days / 7 for q in past])
    recency = np.exp2(-age / 78)
    # Regularize toward half use and fit on previously scored incremental revisions.
    penalty = 8 * .20**2
    slope = .5
    for _ in range(6):
        weights = recency / np.hypot(1, (error - slope*revision) / .60)
        slope = float(np.clip(((weights*revision)@error + .5*penalty) /
                              ((weights*revision)@revision + penalty), 0, 1))
    return slope


def replay():
    history, errors = [], []
    revisions = defaultdict(list)
    for row, late, carried, accounting in source_rows():
        cutoff, known, actual = (row[k] for k in ("cutoff", "known", "actual"))
        public, base = row[PUBLIC], row[COMBINED]
        eligible = [q for q in history if q["known"] < cutoff]
        recent = [q["prepared"] for q in eligible if cutoff - q["cutoff"] <= MAX_SCENE_AGE]
        past = {name: [q for q in revisions[name] if q["known"] < cutoff][-78:] for name in ("late", "carry")}
        gains = {name: gain(sample, row) for name, sample in past.items()}
        updates = dict.fromkeys(past, 0.)
        information = dict.fromkeys(past)
        baseline_info = None
        if row["observed"]:
            spans = any("post_fraction" in q for q in row["observed"])
            model = prepared(row, public, spans)
            fits = estimate(model, recent)
            sigma = prior_scale([q for q in errors if q["known"] < cutoff])
            weights = learned_weights(eligible)
            corrections = np.clip([score / (sigma**-2 + info) for _, info, score in fits], -1.25, 1.25)
            original = float(weights @ corrections)
            if not math.isclose(public + original, row[SUPPLEMENT], abs_tol=1e-9):
                raise ValueError(f"Underlying optical reference changed on {row['release_date']}")
            week = date.fromisoformat(row["week_ending"])
            baseline_info = float(weights @ [info for _, info, _ in fits])
            for name, new in (("late", late), ("carry", carried)):
                if not new:
                    continue
                fused = estimate(joint_model(model, new, public, week, carried=name == "carry"), recent)
                corrections = np.clip([score / (sigma**-2 + info) for _, info, score in fused], -1.25, 1.25)
                updates[name] = float(weights @ corrections) - original
                information[name] = float(weights @ [info for _, info, _ in fused])
                if information[name] + 1e-8 < baseline_info:
                    raise ValueError(f"The marginal pair reduced joint information on {row['release_date']}")
                revisions[name].append(dict(cutoff=cutoff, known=known, revision=updates[name], reference_error=actual-base))
            history.append(dict(cutoff=cutoff, known=known, prepared=prepared(row, actual, spans),
                                absolute_errors=np.abs(public + np.array([p[0] for p in fits]) - actual)))
        elif row[SUPPLEMENT] != public:
            raise ValueError(f"Optical reference without observations on {row['release_date']}")
        errors.append(dict(known=known, error=actual-public))
        estimates = {}
        for name, methods in (("late", LATE_METHODS), ("carry", CARRY_METHODS)):
            revision = updates[name]
            estimates.update(dict(zip(methods, (base+revision, base+.5*revision, base+gains[name]*revision), strict=True)))
            estimates.update({f"{name}_latest_gain_training_utc": max((q["known"] for q in past[name]), default=None),
                              f"{name}_training_weeks": len(past[name]), f"{name}_revision_million_bbl": revision,
                              f"{name}_revision_gain": gains[name], f"{name}_joint_information": information[name]})
        yield {k: row[k] for k in ("release_date", "week_ending", "study_phase", "actual", "cutoff", "known",
                                  "carried_scene", PUBLIC, SUPPLEMENT, COMBINED)} | accounting | dict(
            latest_reference_training_utc=max((q["known"] for q in eligible), default=None),
            main_information=baseline_info) | estimates


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        base = list(trading(rows, MODELS))
        trades = list(replay_conditional_market(rows, base, MODELS, references={
            PUBLIC: f"{PUBLIC}_conditional", SUPPLEMENT: f"{SUPPLEMENT}_conditional"}))
    report = dict(status="exploratory_on_reviewed_history", forecasts={}, trading={})
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"], r["method"]].append(r)
    for phase in PHASES:
        panel = [q for q in rows if q["study_phase"] == phase]
        revised = [q for q in panel if q["added_roofs"]]
        report["forecasts"][phase] = {}
        for scope, sample in (("all", panel), ("existing_optical", [r for r in panel if r["main_roofs"]]),
                              ("added_pairs", revised), ("carried_pairs", [r for r in panel if r["carried_roofs"]])):
            metrics = forecast_metrics(sample, ("carried_scene", *MODELS))
            for name, v in metrics["models"].items():
                delta = [1000*(abs(r[name]-r["actual"]) - abs(r[COMBINED]-r["actual"])) for r in sample]
                v.update(delta_vs_combined_95_kbbl=block_bootstrap_mean(delta),
                         mean_signed_error_kbbl=float(np.mean([1000*(r[name]-r["actual"]) for r in sample])))
            report["forecasts"][phase][scope] = metrics | dict(extra_roofs=sum(r["added_roofs"] for r in sample),
                carried_roofs=sum(r["carried_roofs"] for r in sample),
                new_terminal_weeks=sum(r["added_terminals"] > 0 for r in sample))
        control = groups[phase, COMBINED]
        result = report["trading"][phase] = {}
        for name in MODELS:
            group = groups[phase, name]
            if not group or [q["release_date"] for q in group] != [q["release_date"] for q in control]:
                raise ValueError(f"Mismatched dated market panel on {phase}/{name}")
            delta = [r["net_170_usd"] - q["net_170_usd"] for r, q in zip(group, control, strict=True)]
            result[name] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(q["gross_usd"] - 340*abs(q["near_side"]) for q in group),
                incremental_net_170_95_usd=block_bootstrap_mean(delta),
                changed_entries=sum(q["near_side"] != r["near_side"] for r, q in zip(group, control, strict=True)))
        print(phase, len(revised), 'revised weeks', {m: round(q["mae_kbbl"], 2) for m, q in report["forecasts"][phase]["all"]["models"].items()},
              {m: (q["net_170_usd"], q["net_340_usd"]) for m, q in result.items()}, flush=True)
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
