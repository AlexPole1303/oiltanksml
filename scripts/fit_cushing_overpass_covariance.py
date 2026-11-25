"""Test a shared scene error across overlapping tiles of one Sentinel-2 pass."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from collect_s2_l1c import parse_product
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_inventory_seasonal_residual import BASE as INVENTORY_LEADER, EXTENDED
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT, observation, outer_pairs, prepared
from fit_cushing_recovered_optical import combined_sources
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, SCENE_SIGMA_M
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_overpass_covariance"
FRACTIONS = {"independent_tiles": 0., "half_shared": .5, "shared_pass": 1.}
BASE = next(iter(FRACTIONS))
ADAPTIVE = "learned_mix"
LOCAL = "same_week_shared_pass"
METHODS = (*FRACTIONS, ADAPTIVE, LOCAL)
MODEL_PRIOR = np.array((.50, .35, .15))


@cache
def pass_id(name):
    (platform, timestamp, orbit, _), _, _ = parse_product(name)
    return f"pass:{platform}:{timestamp}:{orbit}"


def with_pass_covariance(model, fraction):
    if not fraction:
        return model
    shared_sd = SCENE_SIGMA_M * math.sqrt(fraction)
    loadings = []
    for r in model["observations"]:
        first, last = (pass_id(r[k]) for k in ("scene_before", "scene_after"))
        if first == last:
            raise ValueError("The temporal change uses the same overpass twice")
        coefficient = r["volume_per_m"] * shared_sd
        loadings.append({(first,): coefficient, (last,): -coefficient})
    aliases = {pass_id(n) for n in model["scenes"]}
    # Use shared pass IDs to connect weeks. Only the extra loadings carry
    # a pass-level error. The pre-existing design still uses original tiles.
    return model | dict(scenes=model["scenes"] | aliases, endpoint_errors=loadings)


def local_scores(model, prior, local_passes):
    # Keep original time/shared-scene conditioning, and merge the original
    # scene-error columns only for the currently verified multi-tile passes.
    def group(row):
        observations = [{**r, **{k: pass_id(r[k]) if pass_id(r[k]) in local_passes else r[k]
                                 for k in ("scene_before", "scene_after")}} for r in row["observations"]]
        scenes = {r[k] for r in observations for k in ("scene_before", "scene_after")}
        return row | dict(observations=observations, scenes=scenes)

    return estimate(group(model), [group(r) for r in prior])


def inputs(*, before=None):
    geometry = {(q["release_date"], q["week_ending"]): q for stem in (
                "sentinel2_space_time_coverage_weeks", "holdout_optical_weeks") for q in read(stem)}
    archive = indexed(f"{OUTER}_forecasts")
    prior = indexed("cushing_robust_stack_forecasts")
    first, extra = (outer_pairs(f"cushing_outer_capture_pairs{suffix}_fits") for suffix in ("", "_supplement"))
    for row, arms, _ in combined_sources():
        issue, week, cutoff, known = (row[k] for k in ("release_date", "week_ending", "cutoff", "known"))
        if before is not None and known >= before:
            break
        ref = archive[issue]
        if (datetime.fromisoformat(ref["cutoff_utc"]) != cutoff or datetime.fromisoformat(ref["known_utc"]) != known
                or not cutoff < known or row["actual"] != float(ref["actual"])):
            raise ValueError(f"Unmatched inventory data and report clocks on {issue}")
        area = float(geometry[issue, week]["dated_reference_area_m2"])
        outside = first[issue] + extra[issue]
        sample = arms[-1] + [observation(q, row, area, float(prior[issue]["prior"])) for q in outside]
        if (len({r["candidate_id"] for r in sample}) != len(sample)
                or any(datetime.fromisoformat(r["available_utc"]) >= cutoff for r in sample)):
            raise ValueError(f"Duplicate or late optical measurements on {issue}")
        yield row | dict(observed=sample, hub_area=area, area_fraction=math.fsum(r["area"] for r in sample) / area,
                         spanning=bool(outside), reference=float(ref[SUPPLEMENT]))


def ensemble_weights(past):
    recent = past[-78:]
    if len(recent) < 32:
        return MODEL_PRIOR
    errors = np.array([r["errors"] for r in recent])
    # Each week scores once. The prior keeps the observed differences from
    # making a small, persistent covariance choice look certain.
    means = errors.mean(axis=0)
    weights = MODEL_PRIOR * np.exp(-(means - means.min()) / .025)
    return weights / weights.sum()


def forecast(row, state, *, update=True):
    cutoff, base = (row[k] for k in ("cutoff", "public"))
    measured = [r for r in state["measured"] if r["known"] < cutoff]
    errors = [r for r in state["errors"] if r["known"] < cutoff]
    scale = prior_scale(errors)
    weights = ensemble_weights(measured)
    predictions = dict.fromkeys(FRACTIONS, base)
    posterior = dict.fromkeys(FRACTIONS, scale)
    redundant_tiles = 0
    if row["observed"]:
        model = prepared(row, base, row["spanning"])
        groups = defaultdict(set)
        for n in model["scenes"]:
            groups[pass_id(n)].add(n)
        shared = {name for name, versions in groups.items() if len(versions) > 1}
        redundant_tiles = len(model["scenes"]) - len(groups)
        ready = [r for r in measured if cutoff - r["cutoff"] <= MAX_SCENE_AGE]
        absolute_errors = {}
        for name, fraction in FRACTIONS.items():
            current = with_pass_covariance(model, fraction)
            prior = [r["models"][name] for r in ready]
            values = estimate(current, prior, scene_sigma_m=SCENE_SIGMA_M * math.sqrt(1 - fraction))
            local_weights = learned_weights([dict(absolute_errors=r["errors_by_timing"][name]) for r in measured])
            info = np.array([i for _, i, _ in values])
            precision = 1 / scale**2 + info
            shifts = np.clip([q[2] for q in values] / precision, -1.25, 1.25)
            predictions[name] = base + float(local_weights @ shifts)
            posterior[name] = float(local_weights @ (1 / np.sqrt(precision)))
            if update:
                absolute_errors[name] = np.abs([base + v[0] - row["actual"] for v in values])
        if update:
            realized = prepared(row, row["actual"], row["spanning"])
            state["measured"].append(dict(cutoff=cutoff, known=row["known"], errors_by_timing=absolute_errors,
                models={m: with_pass_covariance(realized, f) for m, f in FRACTIONS.items()},
                errors=[abs(predictions[m] - row["actual"]) for m in FRACTIONS]))
    if update:
        state["errors"].append(dict(known=row["known"], error=row["actual"] - base))
    predictions[ADAPTIVE] = float(weights @ list(predictions.values())) if row["observed"] else base
    predictions[LOCAL], posterior[LOCAL] = predictions[BASE], posterior[BASE]
    if row["observed"] and shared:
        values = local_scores(model, [r["models"][BASE] for r in ready], shared)
        timing = learned_weights([dict(absolute_errors=r["errors_by_timing"][BASE]) for r in measured])
        precision = 1 / scale**2 + np.array([v[1] for v in values])
        predictions[LOCAL] = base + float(timing @ np.clip([v[2] for v in values] / precision, -1.25, 1.25))
        posterior[LOCAL] = float(timing @ (1 / np.sqrt(precision)))
    return predictions, posterior, weights, redundant_tiles, measured[-1]["known"] if measured else None


def history(state, *, before=None):
    for row in inputs(before=before):
        values, dispersion, weights, mixed, current = forecast(row, state)
        if not math.isclose(values[BASE], row["reference"], abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"The dated baseline differs on {row['release_date']}")
        fields = ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual", "area_fraction")
        yield {k: row[k] for k in fields} | values | dict(redundant_tile_ids=mixed,
            measured_roofs=len(row["observed"]), last_optical_label=current,
            **{f"{m}_posterior_scale_mbbl": q for m, q in dispersion.items()},
            **{f"{m}_ensemble_weight": float(v) for m, v in zip(FRACTIONS, weights, strict=True)})


def market(forecasts):
    book = {r["release_date"]: r for r in forecasts}
    archived = {r["release_date"]: r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    seen = defaultdict(list)
    for row, _ in market_inputs():
        issue, decision = (row[k] for k in ("release_date", "decision"))
        f = book[issue]
        if not f["cutoff"] < f["known"] <= decision < row["entry"] or f["actual"] != row["x"][0]:
            raise ValueError(f"Unmatched market and inventory clock on {issue}")
        for m in METHODS:
            current = row | dict(x=replace_expectation(row, f[m]), response_features=features(
                row["x"][2], row["volatility_usd"], f["actual"] - f[m]))
            settled = [r for r in seen[m] if r["exit_at"] < decision][-HISTORY:]
            seen[m].append(current)
            if len(settled) < MIN_WINDOWS:
                continue
            base = volatility_forecast(settled, current)
            calibration = [r for r in settled if "base_estimate" in r]
            correction, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            current["base_estimate"] = base
            result = trade(row, m, base + correction, 3, training_at=settled[-1]["exit_at"],
                           calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if m == BASE:
                old = archived[issue]
                if (any(result[k] != old[k] for k in ("decision_at_utc", "entry_at_utc", "exit_at_utc", "near_month", "far_month"))
                        or any(result[k] != int(old[k]) for k in ("near_side", "gross_usd", "net_170_usd"))
                        or not math.isclose(result["expected_long_move_usd"], float(old["expected_long_move_usd"]), abs_tol=1e-8)):
                    raise ValueError(f"The original trading result differs on {issue}")
            yield result | dict(inventory_kbbl=1000 * f[m], redundant_tile_ids=f["redundant_tile_ids"],
                                net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def inventory_metrics(rows):
    if not rows:
        return {}
    labels = np.array([r["actual"] for r in rows])
    losses = {m: 1000 * abs(np.array([r[m] for r in rows]) - labels) for m in METHODS}
    return {m: dict(mae_kbbl=float(v.mean()), paired_delta_95_kbbl=block_bootstrap_mean(v - losses[BASE])) for m, v in losses.items()}


def report(forecasts, trades):
    phases = {}
    extended = indexed(f"{EXTENDED}_inventory")
    for phase in PHASES:
        rows = [r for r in forecasts if r["study_phase"] == phase]
        groups = {m: [r for r in trades if r["study_phase"] == phase and r["method"] == m] for m in METHODS}
        market_results = {}
        for m, group in groups.items():
            pairs = list(zip(group, groups[BASE], strict=True))
            if any(a["release_date"] != b["release_date"] for a, b in pairs):
                raise ValueError(f"Different market windows on {phase}/{m}")
            market_results[m] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(r["net_340_usd"] for r in group),
                changed_sides=sum(a["near_side"] != b["near_side"] for a, b in pairs),
                paired_net_170_delta_95_usd=block_bootstrap_mean([a["net_170_usd"] - b["net_170_usd"] for a,b in pairs]))
        mixed = [r for r in rows if r["redundant_tile_ids"]]
        measured = [r for r in rows if r["measured_roofs"]]
        def overlay(r):
            anchor = extended[r["release_date"]]
            if (any(datetime.fromisoformat(anchor[k]) != r[k] for k in ("cutoff", "known"))
                    or float(anchor["actual"]) != r["actual"] or anchor["study_phase"] != phase):
                raise ValueError("The long-history expectation and geometry use different dates")
            forecast = float(anchor[INVENTORY_LEADER])
            return r | {m: forecast + (r[m] - r[BASE]) for m in METHODS}

        phases[phase] = dict(releases=len(rows), mixed_tile_releases=len(mixed), measured_releases=len(measured),
            changed_forecasts={m: sum(abs(r[m] - r[BASE]) > 1e-9 for r in rows) for m in METHODS},
            inventory=dict(all=inventory_metrics(rows), measured=inventory_metrics(measured), mixed=inventory_metrics(mixed)),
            long_history_with_additive_pass_delta=dict(all=inventory_metrics([overlay(r) for r in rows]),
                                                       mixed=inventory_metrics([overlay(r) for r in mixed])),
            trading=market_results)
        print(phase, len(mixed), {m: (round(p["mae_kbbl"], 3), market_results[m]["net_170_usd"])
                                 for m, p in phases[phase]["inventory"]["all"].items()})
    return dict(status="exploratory_on_reviewed_history", shared_variance_fraction=FRACTIONS, phases=phases)


def forward(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    recovered_path = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
    leader_path = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    extended_path = DERIVED / f"{EXTENDED}_{issue}_registration.json"
    original, recovered, leader, extended = (json.loads(p.read_bytes()) for p in (original_path, recovered_path, leader_path, extended_path))
    if any(r[k] != original[k] for k in LINEAGE for r in (recovered, leader, extended)):
        raise ValueError("The frozen satellite inputs have different release dates")
    for source in (leader, extended):
        checked_hashes(source["sources_sha256"])
    checked_hashes({f"data/derived/{name}": checksum for name, checksum in recovered["source_sha256"].items()})
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if not path.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The overpass model must be frozen before the EIA release")
        native = DERIVED / f"cushing_forward_native_optical_{issue}_summary.json"
        area = float(json.loads(native.read_bytes())["dated_roof_area_m2"])
        observations = [r | dict(area=r["volume_per_m"] * BARREL_M3 * 1e6, available_utc=r["source_available_utc"])
                        for r in original["measured_pairs"]] + recovered["recovered_roofs"]
        if (len(observations) != leader["measured_roofs"] or len({r["candidate_id"] for r in observations}) != len(observations)
                or any(datetime.fromisoformat(r["available_utc"]) >= cutoff for r in observations)):
            raise ValueError("The frozen roof set is incomplete or late")
        row = dict(week_ending=original["week_ending"], cutoff=cutoff, public=original["public_ensemble_kbbl"] / 1000,
                   observed=observations, hub_area=area, area_fraction=math.fsum(r["area"] for r in observations) / area, spanning=False)
        state = dict(measured=[], errors=[])
        with threadpool_limits(limits=1):
            for _ in history(state, before=cutoff):
                pass
            estimates, dispersion, weights, mixed, latest = forecast(row, state, update=False)
        stamp = datetime.now(timezone.utc)
        if (not latest < cutoff <= stamp < deadline
                or not math.isclose(1000 * estimates[BASE], leader["estimates_kbbl"][SUPPLEMENT], abs_tol=1e-8, rel_tol=0)):
            raise ValueError("The original forecast did not reproduce before the deadline")
        projection = {m: 1000 * v for m, v in estimates.items()}
        anchor = extended["estimates_kbbl"][INVENTORY_LEADER]
        overlay = {m: anchor + (v - projection[BASE]) for m, v in projection.items()}
        sources = (original_path, recovered_path, leader_path, extended_path, native, Path(__file__).resolve(),
                   *(DERIVED / f"{m}.csv" for m in (f"{STEM}_forecasts", f"{OUTER}_forecasts", f"{EXTENDED}_inventory",
                       "holdout_optical_weeks", "sentinel2_space_time_coverage_weeks", "cushing_robust_stack_forecasts",
                       "cushing_outer_capture_pairs_fits", "cushing_outer_capture_pairs_supplement_fits")),
                   *(ROOT / "scripts" / f"{m}.py" for m in ("fit_cushing_intraweek", "fit_cushing_shared_scene",
                                                             "fit_cushing_outer_capture_nowcast")))
        freeze(path, {k: original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(),
            measured_roofs=len(observations), redundant_tile_ids=mixed, latest_optical_label_at=latest.isoformat(),
            estimates_kbbl=projection, long_history_with_additive_pass_delta_kbbl=overlay,
            posterior_scale_kbbl={m: 1000 * v for m, v in dispersion.items()},
            ensemble_weights=dict(zip(FRACTIONS, map(float, weights), strict=True)),
            sources_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources}))
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    if (any(record[k] != original[k] for k in LINEAGE) or record["measured_roofs"] != leader["measured_roofs"]
            or set(record["estimates_kbbl"]) != set(METHODS)
            or set(record["long_history_with_additive_pass_delta_kbbl"]) != set(METHODS)
            or not datetime.fromisoformat(record["latest_optical_label_at"]) < cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline):
        raise ValueError("The saved overpass forecasts do not verify")
    outcome = score_release(original_path, original)
    score = dict(release_date=issue, status=outcome["status"], estimates_kbbl=record["estimates_kbbl"],
                 long_history_with_additive_pass_delta_kbbl=record["long_history_with_additive_pass_delta_kbbl"], registration_sha256=digest(path))
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        score.update(actual_eia_change_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                     errors_kbbl={m: abs(v - actual) for m, v in record["estimates_kbbl"].items()},
                     long_history_errors_kbbl={m: abs(v - actual) for m, v in record["long_history_with_additive_pass_delta_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(score, indent=2, allow_nan=False) + "\n").encode())
    print(score)


def main():
    with threadpool_limits(limits=1):
        forecasts = list(history(dict(measured=[], errors=[])))
        trades = list(market(forecasts))
    result = report(forecasts, trades)
    export(f"{STEM}_forecasts", forecasts)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    if args.release:
        forward(args.release.isoformat())
    else:
        main()
