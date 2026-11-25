"""Compare constant and weekday-dependent inventory movement between survey marks."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from functools import cache
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import EXTENDED, INVENTORY_LEADER, inputs
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from publication_clocks import EASTERN
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_weekend_timing"
BASE = "uniform_time"
RATES = {BASE: 1., "slower_weekend": .5, "faster_weekend": 1.5}
ADAPTIVE = "adaptive_weekend_mix"
METHODS = (*RATES, ADAPTIVE)
MIX_PRIOR = np.array((.6, .2, .2))


@cache
def weekend_bounds(week):
    if week.weekday() != 4:
        raise ValueError(f"The reporting week must end on a Friday: {week}")
    days = (week - timedelta(days=7), week - timedelta(days=6), week - timedelta(days=4), week)
    hours = (7, 0, 0, 7)
    stamps = [datetime.combine(d, time(h), EASTERN).timestamp() for d, h in zip(days, hours, strict=True)]
    start, _, _, end = stamps
    return tuple((v - start) / (end - start) for v in stamps[1:3])


def weighted_time(t, week, weekend_rate):
    lower, upper = weekend_bounds(week)
    duration = np.clip(t - lower, 0, upper - lower)
    return (t + (weekend_rate - 1) * duration) / (1 + (weekend_rate - 1) * (upper - lower))


def adjusted(model, row, inventory, rate):
    if rate == 1.:
        return model
    observations = model["observations"]
    weeks = {week: (weighted_time(clock, week, rate), area) for week, (clock, area) in model["clocks"].items()}
    end = date.fromisoformat(row["week_ending"])
    earlier = end - timedelta(days=7)
    # The previous week's target is already embedded in each observation.
    # Rescale its contribution by the ratio of weighted to elapsed time.
    fraction = 1 - model["clocks"][earlier][0]
    progress = 1 - weeks[earlier][0]
    ratio = np.divide(progress, fraction, out=np.ones_like(fraction), where=fraction > 1e-12)
    preceding = np.array([q["prior_component"] for q in observations])
    sensitivity = model["sensitivity"].copy()
    if end in weeks:
        elapsed, mask = model["clocks"][end]
        sensitivity += (weeks[end][0] - elapsed) * mask
    residual = model["residual"] - (ratio - 1) * preceding - (sensitivity - model["sensitivity"]) * inventory
    covariance = model["local"]
    later = end + timedelta(days=7)
    if later in weeks:
        p, a = model["clocks"][later]
        q, _ = weeks[later]
        covariance = covariance + PRIOR_INVENTORY_SIGMA_MBBL**2 * (np.outer(a*q, a*q) - np.outer(a*p, a*p))
    return model | dict(clocks=weeks, local=covariance, residual=residual, sensitivity=sensitivity)


def forecast(row, state, *, update=True):
    base = row["public"]
    measured = [r for r in state["measured"] if r["known"] < row["cutoff"]]
    settled = [r for r in state["errors"] if r["known"] < row["cutoff"]]
    values = dict.fromkeys(RATES, base)
    weights = MIX_PRIOR
    if len(measured) >= 32:
        errors = np.array([r["error"] for r in measured[-78:]]).mean(axis=0)
        weights = MIX_PRIOR * np.exp(-(errors - errors.min()) / .025)
        weights /= weights.sum()
    if row["observed"]:
        current = prepared(row, base, row["spanning"])
        available = [r for r in measured if row["cutoff"] - r["cutoff"] <= MAX_SCENE_AGE]
        scale = prior_scale(settled)
        timing_errors = {}
        for method, rate in RATES.items():
            model = adjusted(current, row, base, rate)
            options = estimate(model, [r["models"][method] for r in available])
            timing = learned_weights([dict(absolute_errors=r["timing"][method]) for r in measured])
            increments = np.clip([s / (1 / scale**2 + i) for _, i, s in options], -1.25, 1.25)
            values[method] = base + float(timing @ increments)
            if update:
                timing_errors[method] = np.abs([base + q[0] - row["actual"] for q in options])
        if update:
            target = row["actual"]
            realized = prepared(row, target, row["spanning"])
            state["measured"].append(dict(cutoff=row["cutoff"], known=row["known"], timing=timing_errors,
                models={m: adjusted(realized, row, target, rate) for m, rate in RATES.items()},
                error=[abs(v - target) for v in values.values()]))
    if update:
        state["errors"].append(dict(known=row["known"], error=row["actual"] - base))
    values[ADAPTIVE] = float(weights @ list(values.values())) if row["observed"] else base
    return values, weights, measured[-1]["known"] if measured else None


def history(state, *, before=None):
    extended = indexed(f"{EXTENDED}_inventory")
    for row in inputs(before=before):
        values, weights, latest = forecast(row, state)
        if not math.isclose(values[BASE], row["reference"], abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"The uniform time baseline differs on {row['release_date']}")
        anchor = extended[row["release_date"]]
        if (any(datetime.fromisoformat(anchor[k]) != row[k] for k in ("cutoff", "known"))
                or float(anchor["actual"]) != row["actual"] or anchor["study_phase"] != row["study_phase"]):
            raise ValueError("The extended projection does not share its issue clock")
        fields = ("release_date", "study_phase", "week_ending", "cutoff", "known", "actual")
        yield {k:row[k] for k in fields} | values | dict(measured_roofs=len(row["observed"]),
                latest_optical_label=latest, spanning=row["spanning"],
                **{f"long_{m}":float(anchor[INVENTORY_LEADER]) + (v - values[BASE]) for m, v in values.items()},
                **{f"weight_{m}":float(v) for m, v in zip(RATES, weights, strict=True)})


def market(inventory):
    forecasts = {r["release_date"]:r for r in inventory}
    anchors = {r["release_date"]:r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    history = defaultdict(list)
    for row, _ in market_inputs():
        issue, decision = (row[k] for k in ("release_date", "decision"))
        f = forecasts[issue]
        if not f["cutoff"] < f["known"] <= decision < row["entry"] or f["actual"] != row["x"][0]:
            raise ValueError(f"Inventory and market clocks differ on {issue}")
        for method in METHODS:
            expected = f[method]
            current = row | dict(x=replace_expectation(row, expected), response_features=features(
                row["x"][2], row["volatility_usd"], f["actual"] - expected))
            settled = [r for r in history[method] if r["exit_at"] < decision][-HISTORY:]
            history[method].append(current)
            if len(settled) < MIN_WINDOWS:
                continue
            point = volatility_forecast(settled, current)
            calibration = [r for r in settled if "base_estimate" in r]
            delta, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            current["base_estimate"] = point
            result = trade(row, method, point + delta, 3, training_at=settled[-1]["exit_at"],
                           calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if method == BASE:
                a = anchors[issue]
                if (any(result[k] != a[k] for k in ("near_month", "far_month", "entry_at_utc", "exit_at_utc", "decision_at_utc"))
                        or any(result[k] != int(a[k]) for k in ("near_side", "gross_usd", "net_170_usd"))
                        or not math.isclose(result["expected_long_move_usd"], float(a["expected_long_move_usd"]), abs_tol=1e-8)):
                    raise ValueError(f"The trading baseline did not reproduce on {issue}")
            yield result | dict(inventory_kbbl=expected * 1000, net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def metrics(rows, *, prefix=""):
    if not rows:
        return {}
    actual = np.array([r["actual"] for r in rows])
    errors = {m:1000 * np.abs(np.array([r[prefix + m] for r in rows]) - actual) for m in METHODS}
    return {m:dict(releases=len(e), mae_kbbl=float(e.mean()), paired_delta_95_kbbl=block_bootstrap_mean(e-errors[BASE]))
            for m,e in errors.items()}


def summarize(inventory, trades):
    phases = {}
    for phase in PHASES:
        rows = [r for r in inventory if r["study_phase"] == phase]
        groups = {m: [r for r in trades if (r["study_phase"], r["method"]) == (phase,m)] for m in METHODS}
        pnl = {}
        for method, group in groups.items():
            paired = list(zip(group, groups[BASE], strict=True))
            if any(a["release_date"] != b["release_date"] for a,b in paired):
                raise ValueError(f"Different trade calendar in {phase}/{method}")
            pnl[method] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["net_340_usd"] for r in group),
                changed_sides=sum(a["near_side"] != b["near_side"] for a,b in paired),
                paired_net_delta_95_usd=block_bootstrap_mean([a["net_170_usd"] - b["net_170_usd"] for a,b in paired]))
        scopes = {"all": rows, "measured": [r for r in rows if r["measured_roofs"]], "spanning": [r for r in rows if r["spanning"]]}
        phases[phase] = dict(inventory={k:metrics(q) for k,q in scopes.items()},
                             long_calibration={k:metrics(q, prefix="long_") for k,q in scopes.items()}, trading=pnl)
        print(phase,{m:(round(phases[phase]["long_calibration"]["all"][m]["mae_kbbl"], 3), pnl[m]["net_170_usd"],
                        pnl[m]["net_340_usd"]) for m in METHODS})
    return dict(status="exploratory_on_reviewed_history", weekend_relative_rates=RATES, phases=phases)


def prerelease(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    recovery_path = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
    leader_path = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    extended_path = DERIVED / f"{EXTENDED}_{issue}_registration.json"
    base_input, recovery, leader, extended = (json.loads(p.read_bytes()) for p in (original_path, recovery_path, leader_path, extended_path))
    cutoff, deadline = (datetime.fromisoformat(base_input[k]) for k in LINEAGE[-2:])
    if any(r[k] != base_input[k] for r in (recovery, leader, extended) for k in LINEAGE):
        raise ValueError("The registered optical panels do not share clocks")
    for r in (leader, extended):
        checked_hashes(r["sources_sha256"])
    checked_hashes({f"data/derived/{name}": checksum for name, checksum in recovery["source_sha256"].items()})
    if not path.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The timing alternative must be registered before publication")
        native = DERIVED / f"cushing_forward_native_optical_{issue}_summary.json"
        area = json.loads(native.read_bytes())["dated_roof_area_m2"]
        obs = [r | dict(area=r["volume_per_m"] * BARREL_M3 * 1e6, available_utc=r["source_available_utc"])
               for r in base_input["measured_pairs"]] + recovery["recovered_roofs"]
        if (len(obs) != leader["measured_roofs"] or len({r["candidate_id"] for r in obs}) != len(obs)
                or any(datetime.fromisoformat(r["available_utc"]) >= cutoff for r in obs)
                or leader["optical_input_policy"] != "frozen_panel_without_eligible_after_target_observations"):
            raise ValueError("The current optical roster does not verify")
        state = dict(measured=[], errors=[])
        with threadpool_limits(limits=1):
            list(history(state, before=cutoff))
            values, weights, latest = forecast(dict(cutoff=cutoff, week_ending=base_input["week_ending"], observed=obs,
                area_fraction=math.fsum(r["area"] for r in obs) / area, hub_area=area, spanning=False,
                public=base_input["public_ensemble_kbbl"] / 1000), state, update=False)
        stamp = datetime.now(timezone.utc)
        if not latest < cutoff <= stamp < deadline or not math.isclose(values[BASE] * 1000,
                leader["estimates_kbbl"][SUPPLEMENT], abs_tol=1e-8, rel_tol=0):
            raise ValueError("The uniform model or timing of the registration did not verify")
        sources = (original_path, recovery_path, leader_path, extended_path, native, Path(__file__).resolve(),
                   *(ROOT / "scripts" / f"{name}.py" for name in ("fit_cushing_intraweek", "fit_cushing_overpass_covariance",
                      "fit_cushing_outer_capture_nowcast", "fit_cushing_shared_scene")),
                   *(DERIVED / f"{name}.csv" for name in (f"{STEM}_forecasts", f"{EXTENDED}_inventory", f"{OUTER}_forecasts",
                      "holdout_optical_weeks", "sentinel2_space_time_coverage_weeks", "cushing_robust_stack_forecasts",
                      "cushing_outer_capture_pairs_fits", "cushing_outer_capture_pairs_supplement_fits")))
        actual_scale = {m:1000 * v for m,v in values.items()}
        reference = extended["estimates_kbbl"][INVENTORY_LEADER]
        freeze(path, {k:base_input[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(),
            measured_roofs=len(obs), latest_optical_label=latest.isoformat(), estimates_kbbl=actual_scale,
            long_estimates_kbbl={m:reference + (v - actual_scale[BASE]) for m,v in actual_scale.items()},
            weights=dict(zip(RATES,map(float,weights),strict=True)),
            sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))
    reg = json.loads(path.read_bytes())
    checked_hashes(reg["sources_sha256"])
    if (any(reg[k] != base_input[k] for k in LINEAGE) or reg["measured_roofs"] != leader["measured_roofs"]
            or not datetime.fromisoformat(reg["latest_optical_label"]) < cutoff <= datetime.fromisoformat(reg["recorded_at_utc"]) < deadline
            or set(reg["estimates_kbbl"]) != set(METHODS) or set(reg["long_estimates_kbbl"]) != set(METHODS)):
        raise ValueError("The registered timing result did not verify")
    target = score_release(original_path, base_input)
    score = dict(release_date=issue, status=target["status"], registration_sha256=digest(path),
                 estimates_kbbl=reg["estimates_kbbl"], long_estimates_kbbl=reg["long_estimates_kbbl"])
    if target["status"] == "scored_first_release":
        v = target["actual_eia_change_kbbl"]
        score.update(actual_eia_change_kbbl=v, first_release_sha256=target["first_release_sha256"],
                     errors_kbbl={m:abs(q-v) for m,q in reg["estimates_kbbl"].items()},
                     long_errors_kbbl={m:abs(q-v) for m,q in reg["long_estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(score, indent=2, allow_nan=False) + "\n").encode())
    print(score)


def main():
    with threadpool_limits(limits=1):
        inventory = list(history(dict(measured=[], errors=[])))
        trades = list(market(inventory))
    report = summarize(inventory, trades)
    export(f"{STEM}_forecasts", inventory)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report,indent=2,allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    if args.release:
        prerelease(args.release.isoformat())
    else:
        main()
