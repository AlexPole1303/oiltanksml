"""Move independently registered optical observations to the tradable cutoff."""

import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

import numpy as np
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, risk_prediction
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_joint_late_pairs import joint_model
from fit_cushing_late_pair_recovery import accepted_pairs
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PUBLIC
from fit_cushing_outer_capture_nowcast import observation, outer_pairs, prepared
from fit_cushing_prerelease_cl import cases
from fit_cushing_prerelease_sensor import event_response, source_book, visible
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE

STEM = "cushing_early_registered_pairs"
RECOVERED = "early_recovered_scene"
POINT = (RECOVERED, "registered_fallback", "registered_all")
MODELS = ("early_public_ml", *POINT)
TRADE_METHODS = ("curve", "public_ml", "recovered_scene_measured_only", "registered_fallback_ml", "registered_all_ml",
                 "registered_fallback_response", "registered_all_response", "recovered_response",
                 "recovered_fit_fallback_input", "fallback_fit_recovered_input")


def dated_sources():
    windows = {r["release_date"]: r for name in ("sentinel2_space_time_coverage_weeks", "holdout_optical_weeks")
               for r in read(name)}
    outer = {name: outer_pairs(f"cushing_outer_capture_pairs{suffix}_fits") for name, suffix in (
        ("spanning", ""), ("supplemental", "_supplement"), ("internal", "_internal"))}
    late = accepted_pairs()
    clocks = indexed("eia_release_clock")
    prior_reports = {}
    for q in read("eia_inventory_release_panel"):
        issue = q["release_date"]
        if issue not in clocks or q["reported_as"] != "current_week" or not q["change_kbbl_as_known_at_release"]:
            continue
        stamp = datetime.fromisoformat(clocks[issue]["conservative_postrelease_from_utc"])
        previous = prior_reports.get(q["week_ending"])
        if previous is None or stamp < previous[0]:
            prior_reports[q["week_ending"]] = stamp, float(q["change_kbbl_as_known_at_release"])/1000
    source = source_book()
    stored = {r["issue"]: r for r in read("cushing_prerelease_sensor_forecasts")}
    for r in cases()[0]:
        issue, cutoff, known = (r[k] for k in ("issue", "decision", "known"))
        ref = stored[issue]
        if (datetime.fromisoformat(ref["cutoff_utc"]) != cutoff or datetime.fromisoformat(ref["known_utc"]) != known
                or float(ref["early_public_ml"]) != r["early_public_ml"] or float(ref["inventory"]) != r["inventory"]
                or not cutoff < r["entry"] < r["reported_at"] < r["exit_at"] < known):
            raise ValueError(f"Different early release or contract window on {issue}")
        total, original, widened = visible(r, source[issue])
        window = windows[issue]
        report_cutoff = datetime.fromisoformat(window["cutoff_utc"])
        previous_week = (date.fromisoformat(r["week_ending"])-timedelta(days=7)).isoformat()
        prior_clock, vintage = prior_reports.get(previous_week, (None, 0.))
        reference = dict(r, cutoff=report_cutoff, release_date=issue)
        # Archived fits were selected before the report-day cutoff. Use only
        # the subset whose inputs and required preceding label were already known.
        candidates = {}
        for name, panel in outer.items():
            candidates[name] = [observation(q, reference, total, vintage, internal=name == "internal") for q in panel[issue]
                                if datetime.fromisoformat(q["latest_geometry_available_utc"]) < cutoff
                                and (name == "internal" or prior_clock is not None and prior_clock < cutoff)]
        candidates["late"] = [q for q in late[issue] if q["available_utc"] < cutoff]
        baseline = original + widened
        public = r["early_public_ml"]
        available = dict(original=baseline, **candidates)
        if not math.isclose(float(window["dated_reference_area_m2"]), total, rel_tol=1e-12):
            raise ValueError(f"Inconsistent dated roof area on {issue}")
        yield r | dict(cutoff=cutoff, public=public, actual=r["inventory"], hub_area=total, reference=ref,
                       preceding_report_known_utc=prior_clock if prior_clock and prior_clock < cutoff else None), available


def selected_sources(available, method):
    baseline = available["original"]
    if method == RECOVERED or (method == "registered_fallback" and baseline):
        return baseline, [], []
    main = list(baseline)
    used = {q["candidate_id"] for q in main}
    for kind in ("spanning", "supplemental"):
        for q in available[kind]:
            if q["candidate_id"] not in used:
                used.add(q["candidate_id"])
                main.append(q)
    late = [q for q in available["late"] if q["candidate_id"] not in used]
    if method == "registered_fallback" and main:
        late = []
    internal = available["internal"] if not main and not late else []
    return main, late, internal


def observations(row, main, late, internal, target):
    primary = main or internal
    if not primary and not late:
        return None
    area = math.fsum(q["area"] for q in primary) / row["hub_area"]
    spanning = any("post_fraction" in q for q in primary)
    if not primary:
        core = dict(observations=[], clocks={})
    else:
        core = prepared(row | dict(observed=primary, area_fraction=area), target, spanning)
    if not late:
        return core
    return joint_model(core, late, target, date.fromisoformat(row["week_ending"]), carried=False)


def optical_forecast(row, selection, past, scale):
    public = row[PUBLIC]
    model = observations(row, *selection, public)
    if model is None:
        return public, None, 0.
    available = [q for q in past if q["known"] < row["cutoff"]]
    recent = [q["prepared"] for q in available if row["cutoff"]-q["cutoff"] <= MAX_SCENE_AGE]
    fits = estimate(model, recent)
    updates = np.clip([score/(scale**-2+info) for _, info, score in fits], -1.25, 1.25)
    weights = learned_weights(available)
    fit = dict(known=row["known"], cutoff=row["cutoff"], prepared=observations(row, *selection, row["actual"]),
               absolute_errors=np.abs([public+revision-row["actual"] for revision, _, _ in fits]))
    return public+float(weights@updates), fit, float(weights@[info for _, info, _ in fits])


def forecast_rows():
    errors, histories = [], defaultdict(list)
    for row, available in dated_sources():
        row[PUBLIC] = row["early_public_ml"]
        cutoff = row["cutoff"]
        known = [q for q in errors if q["known"] < cutoff]
        sigma = prior_scale(known)
        metadata = dict(latest_public_error_utc=known[-1]["known"] if known else None)
        for method in POINT:
            selection = selected_sources(available, method)
            sample = [q for group in selection for q in group]
            clocks = [q["available_utc"] for q in sample]
            clocks = [v if isinstance(v, datetime) else datetime.fromisoformat(v) for v in clocks]
            if len(sample) != len({q["candidate_id"] for q in sample}) or any(v >= cutoff for v in clocks):
                raise ValueError(f"Duplicate or late geometry on {row['issue']}/{method}")
            value, fit, information = optical_forecast(row, selection, histories[method], sigma)
            row[method] = value
            metadata.update({f"{method}_roofs": len(sample), f"{method}_latest_input_utc": max(clocks, default=None),
                             f"{method}_latest_training_utc": max((q["known"] for q in histories[method] if q["known"] < cutoff), default=None),
                             f"{method}_information": information})
            if fit is not None:
                histories[method].append(fit)
        if not math.isclose(row[RECOVERED], float(row["reference"][RECOVERED]), abs_tol=1e-10):
            raise ValueError(f"The recovered early forecast failed to reproduce on {row['issue']}")
        errors.append(dict(known=row["known"], error=row["actual"]-row[PUBLIC]))
        yield row | metadata | dict(**{f"available_{name}_roofs": len(sample) for name, sample in available.items()})


def trading(rows):
    anchors = {(r["release_date"], r["method"]): r for r in read("cushing_prerelease_sensor_trades")}
    seen = []
    for f in rows:
        issue, cutoff = (f[k] for k in ("issue", "cutoff"))
        curve, _, _, _, public, _ = f["design"]
        designs = (curve, public, *(public+(f[m]-f[PUBLIC], (f[m]-f[PUBLIC])*curve[-1]) for m in POINT))
        current = f | dict(designs=designs, curve_estimate=None)
        past = [r for r in seen if r["exit_at"] < cutoff][-HISTORY:]
        seen.append(current)
        if len(past) < MIN_WINDOWS:
            continue
        points = [risk_prediction([r | dict(x=r["designs"][i]) for r in past], current | dict(x=x), len(x))
                  for i, x in enumerate(designs)]
        curve_estimate, public_estimate, *satellite = points
        current["curve_estimate"] = curve_estimate
        satellite = [v if f[f"{m}_roofs"] else public_estimate for m, v in zip(POINT, satellite, strict=True)]
        changes = [event_response(past, current, curve_estimate, f[m]-f[PUBLIC], symmetric=True)
                   for m in (*POINT[1:], RECOVERED)]
        swapped = []
        for fit_index, input_index in ((2, 3), (3, 2)):
            count_field = f"{POINT[input_index-2]}_roofs"
            x = designs[input_index]
            training = [r | dict(x=r["designs"][fit_index]) for r in past]
            swapped.append(risk_prediction(training, current | dict(x=x), len(x)) if f[count_field] else public_estimate)
        for method, expected in zip(TRADE_METHODS, (curve_estimate, public_estimate, *satellite, *changes, *swapped), strict=True):
            side = int(expected > HURDLE)-int(expected < -HURDLE)
            gross = side*f["long_gross"]
            if method in TRADE_METHODS[:3]:
                anchor = anchors[issue, method]
                if ((side, gross) != (int(anchor["near_side"]), int(anchor["gross_usd"]))
                        or not math.isclose(expected, float(anchor["expected_long_move_usd"]), abs_tol=1e-8)):
                    raise ValueError(f"Early control {method} changed on {issue}")
            yield {k:f[k] for k in ("study_phase", "near_month", "far_month", "source")} | dict(
                release_date=issue, method=method, cutoff_utc=cutoff.isoformat(), entry_at_utc=f["entry"].isoformat(),
                report_at_utc=f["reported_at"].isoformat(), exit_at_utc=f["exit_at"].isoformat(),
                latest_training_exit_utc=past[-1]["exit_at"].isoformat(), expected_long_move_usd=expected,
                long_spread_gross_usd=f["long_gross"], near_side=side, gross_usd=gross,
                **{f"cost_{c}_usd": c*abs(side) for c in (*COSTS, 340)},
                **{f"net_{c}_usd": gross-c*abs(side) for c in (*COSTS, 340)})


def point_metrics(sample):
    if not sample:
        return dict(weeks=0, models={})
    errors = {m: np.asarray([1000*abs(r[m]-r["actual"]) for r in sample]) for m in MODELS}
    return dict(weeks=len(sample), models={m: dict(mae_kbbl=float(e.mean()), rmse_kbbl=float(np.sqrt(np.mean(np.square(e)))),
        direction_matches=sum(r[m]*r["actual"]>0 for r in sample), delta_vs_recovered_95_kbbl=block_bootstrap_mean(e-errors[RECOVERED]))
        for m, e in errors.items()})


def main():
    with threadpool_limits(limits=1):
        forecasts = list(forecast_rows())
        trades = list(trading(forecasts))
    result = dict(status="exploratory_on_reviewed_history", forecasts={}, trading={})
    groups = defaultdict(list)
    for q in trades:
        groups[q["study_phase"], q["method"]].append(q)
    for phase in PHASES:
        panel = [r for r in forecasts if r["study_phase"] == phase]
        new_measurement = [r for r in panel if not r[f"{RECOVERED}_roofs"] and r["registered_fallback_roofs"]]
        new_variant = [r for r in panel if r["registered_all"] != r[RECOVERED]]
        info = result["forecasts"][phase] = {name: point_metrics(sample) for name, sample in (
            ("all", panel), ("previously_unmeasured", new_measurement), ("all_variant_changed", new_variant))}
        info["coverage_weeks"] = dict(Counter(k for r in panel for k in POINT if r[f"{k}_roofs"]))
        info["fill_roofs"] = sum(r["registered_fallback_roofs"] for r in new_measurement)
        control = groups[phase, "curve"]
        recovered = groups[phase, "recovered_scene_measured_only"]
        output = result["trading"][phase] = {}
        for m in TRADE_METHODS:
            sample = groups[phase, m]
            if [q["release_date"] for q in sample] != [q["release_date"] for q in control]:
                raise ValueError(f"Different early windows for {phase}/{m}")
            delta = [q["net_170_usd"]-r["net_170_usd"] for q, r in zip(sample, control, strict=True)]
            changed = [dict(release_date=q["release_date"], new_side=q["near_side"], recovered_side=r["near_side"],
                            net_170_delta_usd=q["net_170_usd"]-r["net_170_usd"], net_340_delta_usd=q["net_340_usd"]-r["net_340_usd"])
                       for q, r in zip(sample, recovered, strict=True) if q["near_side"] != r["near_side"]]
            output[m] = robustness(sample) | dict(net_340_usd=sum(q["net_340_usd"] for q in sample),
                  changed_vs_curve=sum(q["near_side"] != r["near_side"] for q, r in zip(sample, control, strict=True)),
                  incremental_net_vs_curve_95_usd=block_bootstrap_mean(delta), changes_vs_recovered=changed)
        print(phase, info["coverage_weeks"], {m: round(q["mae_kbbl"], 2) for m, q in info["all"]["models"].items()},
              {m: (v["net_170_usd"], v["net_340_usd"]) for m, v in output.items()}, flush=True)
    fixed = ("issue", "study_phase", "week_ending", "actual", "cutoff", "known", *MODELS,
             "latest_public_error_utc", "preceding_report_known_utc")
    fields = (*fixed, *(k for k in forecasts[0] if k.startswith(("available_", "early_recovered_scene_", "registered_")) and k not in fixed))
    export(f"{STEM}_forecasts", [{k:r[k] for k in fields} for r in forecasts])
    export(f"{STEM}_trades", trades)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
