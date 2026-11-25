"""Reduce optical revisions that depend on a single terminal or pair of scenes."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC, learned_beta
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL, estimate, replay as shared_replay

STEM = "cushing_optical_stability"
MODELS = (PUBLIC, "carried_scene", "scene_direct", "stability_adjusted", "stability_calibrated")


def deletion_scale(observed, public, recent, full, fields):
    groups = defaultdict(list)
    for row in observed:
        groups[tuple(row[k] for k in fields)].append(row)
    n = len(groups)
    if n < 2:
        return 0., n
    changes = []
    for withheld in groups:
        subset = [r for group, rows in groups.items() if group != withheld for r in rows]
        value, *_ = estimate(subset, public, recent)
        changes.append((value-full)**2)
    return math.sqrt((n-1)/n*math.fsum(changes)), n


def stability_adjustment(observed, public, recent, full, information):
    fields = (("terminal_id",), ("scene_before", "scene_after"))
    (terminal, n_terminals), (pair, n_pairs) = [deletion_scale(observed, public, recent, full, keys) for keys in fields]
    # Propagate the modelled measurement covariance through the linear
    # update. Deletion sensitivity reflects the unresolved footprint instead.
    noise = math.sqrt(information)/(1/PRIOR_INVENTORY_SIGMA_MBBL**2+information)
    scale = math.hypot(noise, max(terminal, pair))
    if scale == 0:
        weight = 0.
    else:
        weight = full**2/(full**2+scale**2)
    return dict(stable_raw=full*weight, stability_weight=weight, model_noise=noise,
                terminal_deletion_scale=terminal, pair_deletion_scale=pair, observed_terminals=n_terminals, observed_scene_pairs=n_pairs)


def replay():
    history = []
    for _, r in shared_replay():
        cutoff, observed, public = r["cutoff"], r["observed"], r["public"]
        if any(datetime.fromisoformat(q["available_utc"]) >= cutoff for q in observed):
            raise ValueError(f"Optical observation arrived late on {r['release_date']}")
        eligible = [q for q in history if q["known"] < cutoff]
        recent = [q for q in eligible if cutoff-q["cutoff"] <= MAX_SCENE_AGE]
        training = eligible[-104:]
        raw = r["carried_raw"]
        result = stability_adjustment(observed, public, recent, raw, r["carried_information"])
        beta = learned_beta(training, "stable_raw")
        result.update(public_ensemble=public, scene_direct=public+raw,
                      stability_adjusted=public+result["stable_raw"], stability_calibrated=public+beta*result["stable_raw"],
                      stability_beta=beta, tanks=len(observed), latest_training_label_utc=training[-1]["known"].isoformat() if training else None,
                      training_measured_weeks=len(training))
        state = r | result
        if observed:
            history.append(state)
        yield state


def register(release, path, original_path, original):
    from register_cushing_prerelease_forecast import digest, freeze

    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in ("information_cutoff_utc","registration_deadline_utc"))
    if datetime.now(timezone.utc) >= deadline:
        raise ValueError("The optical forecast is past its registration deadline")
    public = original["public_ensemble_kbbl"]/1000
    observed = original["measured_pairs"]
    if len(observed) != original["measured_roofs"] or any(
        datetime.fromisoformat(r["source_available_utc"]) >= cutoff for r in observed
    ):
        raise ValueError("Frozen observations differ from the original information cutoff")
    with threadpool_limits(limits=1):
        training = [r for _,r in shared_replay() if r["observed"] and r["known"] < cutoff]
        recent = [r for r in training if cutoff-r["cutoff"] <= MAX_SCENE_AGE]
        raw, information, *_ = estimate(observed,public,recent)
        result = stability_adjustment(observed,public,recent,raw,information)
    forecast = {PUBLIC:1000*public, "scene_direct":1000*(public+raw),
                "stability_adjusted":1000*(public+result["stable_raw"])}
    now = datetime.now(timezone.utc)
    if not cutoff <= now < deadline:
        raise ValueError("Forecast registration falls outside the original information window")
    sources = (original_path, DERIVED/"cushing_shared_scene_forecasts.csv", DERIVED/"cushing_observation_operator_measurements.csv")
    record = dict(release_date=release, week_ending=original["week_ending"], recorded_at_utc=now.isoformat(),
                  information_cutoff_utc=cutoff.isoformat(), registration_deadline_utc=deadline.isoformat(),
                  forecast_kbbl=forecast, measured_roofs=len(observed), stability=result, training_releases=len(training),
                  latest_training_label_utc=training[-1]["known"].isoformat() if training else None,
                  model_sha256=digest(ROOT/"scripts/fit_cushing_optical_stability.py"),
                  source_sha256={p.name:digest(p) for p in sources})
    freeze(path,record)
    return record


def forward(release, *, scoring):
    from register_cushing_prerelease_forecast import digest, score

    original_path = DERIVED/f"cushing_prerelease_forecast_{release}.json"
    path = original_path.with_name(f"{original_path.stem}_optical_stability.json")
    original = json.loads(original_path.read_bytes())
    if not scoring and not path.exists():
        return register(release,path,original_path,original)
    record = json.loads(path.read_bytes())
    cutoff, stamp, deadline = (datetime.fromisoformat(record[k]) for k in (
        "information_cutoff_utc","recorded_at_utc","registration_deadline_utc"))
    latest = record["latest_training_label_utc"]
    if (not cutoff <= stamp < deadline or (latest and datetime.fromisoformat(latest) >= cutoff)
        or any(record[k] != original[k] for k in ("release_date","week_ending","information_cutoff_utc","registration_deadline_utc"))
        or any(digest(DERIVED/k) != h for k,h in record["source_sha256"].items())):
        raise ValueError("The stability forecast or its frozen inputs differ")
    if not scoring:
        return record
    outcome = score(original_path,original)
    result = dict(status=outcome["status"], release_date=release, forecast_sha256=digest(path))
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual, report_sha256=outcome["first_release_sha256"],
                      absolute_error_kbbl={k:abs(actual-v) for k,v in record["forecast_kbbl"].items()})
    write(path.with_name(f"{path.stem}_score.json"),(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group()
    command.add_argument("--register-release",type=date.fromisoformat)
    command.add_argument("--score-release",type=date.fromisoformat)
    args = parser.parse_args()
    release = args.score_release or args.register_release
    if release is not None:
        result = forward(release.isoformat(), scoring=args.score_release is not None)
        print(json.dumps(result,indent=2,allow_nan=False))
        return
    with threadpool_limits(limits=1):
        forecasts = list(replay())
        trades = list(trading(forecasts, MODELS))
    report = dict(status="historical_walk_forward_on_reviewed_period", forecasts={}, trading={})
    for phase in PHASES:
        sample = [r for r in forecasts if r["study_phase"] == phase]
        measured = [r for r in sample if r["tanks"]]
        report["forecasts"][phase] = {
            scope: forecast_metrics(rows, MODELS) for scope, rows in (
                ("all", sample), ("measured", measured), ("multiple_scene_pairs", [r for r in measured if r["observed_scene_pairs"] > 1])
            )
        }
        report["forecasts"][phase]["variability"] = dict(
            measured_weeks=len(measured), median_update_weight=float(np.median([r["stability_weight"] for r in measured])),
            single_terminal_weeks=sum(r["observed_terminals"] == 1 for r in measured),
            single_scene_pair_weeks=sum(r["observed_scene_pairs"] == 1 for r in measured),
        )
        results = report["trading"][phase] = {}
        base = [r for r in trades if r["study_phase"] == phase and r["method"] == PUBLIC]
        for method in ("curve", *MODELS):
            group = [r for r in trades if r["study_phase"] == phase and r["method"] == method]
            if [r["release_date"] for r in group] != [r["release_date"] for r in base]:
                raise ValueError(f"Different trading calendar in {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(group,base,strict=True)]
            results[method] = robustness(group) | daily_risk(group) | dict(
                changed_directions=sum(r["near_side"] != q["near_side"] for r,q in zip(group,base,strict=True)),
                increment_usd=sum(delta), paired_increment_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group),
            )
        print(phase, {m: round(v["mae_kbbl"],2) for m,v in report["forecasts"][phase]["measured"]["models"].items()})
        print(phase, {m: (v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in results.items()})
    fields = ("release_date", "week_ending", "study_phase", "actual", "tanks", "carried_raw", "stable_raw", "stability_beta",
              "stability_weight", "model_noise", "terminal_deletion_scale", "pair_deletion_scale", "observed_terminals",
              "observed_scene_pairs", "latest_training_label_utc", "training_measured_weeks", *MODELS)
    export(f"{STEM}_forecasts", [{k: r[k] for k in fields} | dict(cutoff_utc=r["cutoff"].isoformat(), known_utc=r["known"].isoformat())
                                 for r in forecasts])
    export(f"{STEM}_trades", trades)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
