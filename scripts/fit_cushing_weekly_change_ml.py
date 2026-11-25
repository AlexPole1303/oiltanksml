"""Estimate the incremental value of uncertainty-aware weekly tank changes."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares

from backtest_wti_cushing_spread import COSTS, statistics
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_weekly_change_ml"
VOLUMES = ("sentinel2_weekly_tank_volumes", "sentinel2_holdout_weekly_tank_volumes",
           "sentinel2_alternate_registration_added_volumes", "sentinel2_alternate_registration_holdout_added_volumes",
           "hcs_dated_shadow_volumes", "hcs_dated_shadow_holdout_volumes")
MIN_WEEKS = 26
LOOKBACK = 104
HALF_LIFE = 52.
FEATURES = ("physical", "attenuated", "loss_bound", "overlap_attenuated")
METHODS = ("public_ensemble", "public_plus_physical", "physical_ml", "pair_quality_ml")


def attenuate(change, low, high):
    # Descriptive fit width serves as a feature, without treating it as a confidence interval.
    halfwidth = (high - low) / 2
    return change / (1 + (halfwidth / 1.0)**2)


def measurements():
    seen = set()
    for stem in VOLUMES:
        for r in read(stem):
            issue, tank = r["release_date"], r["candidate_id"]
            key = issue, r["week_ending"], tank
            if key in seen:
                raise ValueError(f"Repeated tank measurement on {key}")
            seen.add(key)
            earlier, report_start, later, report_end, published, cutoff = (datetime.fromisoformat(r[k]) for k in (
                "earlier_capture_at_utc", "previous_measurement_at_utc", "later_capture_at_utc", "current_measurement_at_utc",
                "archive_available_at_utc", "cutoff_utc"))
            if not earlier <= report_start < later <= report_end or not later <= published < cutoff:
                raise ValueError(f"Ineligible measurement clock on {key}")
            factor = float(r["circular_kbbl_per_m"])
            change, low, high, certain, overlap = (float(r[k]) for k in (
                "capture_change_kbbl", "joint_loss_min_change_kbbl", "joint_loss_max_change_kbbl",
                "minimum_magnitude_change_kbbl", "overlap_fraction"))
            if not (factor > 0 and low-1e-9 <= change <= high+1e-9 and 0 < overlap <= 1
                    and math.isclose(change, (float(r["earlier_gap_m"])-float(r["later_gap_m"]))*factor, abs_tol=1e-9)
                    and math.isclose(certain, max(low, min(0., high)), abs_tol=1e-9)):
                raise ValueError(f"Invalid roof change or joint loss range on {key}")
            adjusted = attenuate(change/factor, low/factor, high/factor)*factor
            yield dict(release_date=issue, week_ending=r["week_ending"], study_phase=r["study_phase"],
                       cutoff_utc=r["cutoff_utc"], candidate_id=tank, terminal_id=r["terminal_id"], source=stem,
                       earlier_sentinel2_version=r["earlier_sentinel2_version"], later_sentinel2_version=r["later_sentinel2_version"],
                       archive_available_at_utc=r["archive_available_at_utc"], circular_area_m2=float(r["circular_area_m2"]),
                       physical=change, attenuated=adjusted, loss_bound=certain, overlap_attenuated=overlap*adjusted,
                       descriptive_range_contains_zero=low<=0<=high, loss_min_kbbl=low, loss_max_kbbl=high)


def summarize(rows, *, terminals=False):
    groups = defaultdict(list)
    for r in rows:
        key = (r["release_date"], r["week_ending"], r["terminal_id"]) if terminals else (r["release_date"], r["week_ending"])
        groups[key].append(r)
    for key, sample in sorted(groups.items()):
        first = sample[0]
        yield {k:first[k] for k in ("release_date", "study_phase", "week_ending", "cutoff_utc")} | (
            {"terminal_id":key[2]} if terminals else {}) | dict(
                tanks=len(sample), zero_plausible_tanks=sum(r["descriptive_range_contains_zero"] for r in sample),
                area_m2=math.fsum(r["circular_area_m2"] for r in sample),
                **{field:math.fsum(r[field] for r in sample) for field in FEATURES})


def week_inputs(weekly):
    by_week = {(r["release_date"],r["week_ending"]):r for r in weekly}
    previous = indexed("cushing_robust_stack_forecasts")
    original = {(r["release_date"],r["week_ending"]):r for r in read("sentinel2_space_time_coverage_weeks") if r["study_phase"] != "holdout"}
    original.update({(r["release_date"],r["week_ending"]):r for r in read("holdout_optical_weeks")})
    for issue, f in sorted(previous.items()):
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("forecast_cutoff_utc", "target_known_by_utc"))
        key = issue, f["week_ending"]
        r, control = by_week.get(key), original[key]
        measured = f["optical_available"] == "True"
        if measured != bool(r) or cutoff != datetime.fromisoformat(control["cutoff_utc"]) or not cutoff < known:
            raise ValueError(f"Weekly signal coverage or chronology differs on {issue}")
        if r and (r["week_ending"] != f["week_ending"] or r["study_phase"] != f["study_phase"]
                  or datetime.fromisoformat(r["cutoff_utc"]) != cutoff or r["tanks"] != int(control["baseline_b04_tanks"])
                  or not math.isclose(r["physical"],float(control["capture_change_kbbl"]),abs_tol=1e-8)
                  or not math.isclose(r["area_m2"],float(control["baseline_b04_area_m2"]),abs_tol=1e-6)):
            raise ValueError(f"Weekly optical features do not reproduce the original sample on {issue}")
        yield dict(release_date=issue, week_ending=f["week_ending"], study_phase=f["study_phase"], cutoff=cutoff, known=known,
                   target=float(f["target"]), public_ensemble=float(f["public_ensemble"]),
                   measured=measured, tanks=r["tanks"] if r else 0,
                   **{key:float(r[key])/1000 if r else 0. for key in FEATURES})


def learn(history, current, fields):
    values = np.array([[r[field] for field in fields] for r in history])
    y = np.array([r["target"]-r["public_ensemble"] for r in history])
    age = np.array([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in history])
    w = np.exp2(-age/HALF_LIFE)
    w /= w.sum()
    # Fit a zero-centred correction in the original units, with soft-L1 observation loss.
    # Positive weights allow the history to choose which attenuated measurement earns a place.
    root = np.sqrt(w)
    ridge = math.sqrt(0.15*len(fields))
    result = least_squares(lambda beta: np.r_[root*(values @ beta-y), ridge*beta], np.full(len(fields),0.25),
                           bounds=(0., 2.), loss="soft_l1", f_scale=0.65)
    if not result.success:
        raise ValueError(f"Weekly change fit failed on {current['release_date']}: {result.message}")
    return np.where(result.active_mask == -1, 0., result.x)


def predict(rows):
    seen = []
    for r in rows:
        available = [q for q in seen if q["known"] < r["cutoff"]][-LOOKBACK:]
        measured = [q for q in available if q["measured"]]
        f = r["public_ensemble"]
        raw_weight = 0.
        quality_weights = np.zeros(len(FEATURES))
        if len(measured) >= MIN_WEEKS:
            raw_weight, = learn(measured, r, ("physical",))
            quality_weights = learn(measured, r, FEATURES)
        yield {k:r[k] for k in ("release_date", "week_ending", "study_phase", "target", "measured", "tanks", *FEATURES)} | dict(
            forecast_cutoff_utc=r["cutoff"].isoformat(), target_known_by_utc=r["known"].isoformat(),
            latest_training_label_utc=max(q["known"] for q in measured).isoformat() if measured else None,
            measured_training_weeks=len(measured), public_ensemble=f, public_plus_physical=f+r["physical"],
            physical_ml=f+float(np.clip(raw_weight*r["physical"],-1.,1.)),
            pair_quality_ml=f+float(np.clip(quality_weights @ np.asarray([r[k] for k in FEATURES]),-1.,1.)),
            physical_ml_weight=raw_weight, pair_quality_ml_weights_json=json.dumps(dict(zip(FEATURES,map(float,quality_weights),strict=True))))
        seen.append(r)


def scores(rows):
    y = np.array([r["target"] for r in rows])
    baseline = np.abs(y-np.array([r["public_ensemble"] for r in rows]))
    results = {}
    for method in METHODS:
        pred = np.array([r[method] for r in rows])
        errors = np.abs(y-pred)
        results[method] = dict(mae_million_bbl=float(errors.mean()), correct_direction=int(np.count_nonzero(y*pred>0)),
                               paired_mae_delta_ci_95=block_bootstrap_mean(errors-baseline))
    return dict(releases=len(rows), methods=results)


def trade_rows(forecasts, methods=METHODS):
    mapping = {r["release_date"]:r for r in forecasts}
    calendar = indexed("cushing_holdout_release_index")
    for r in read("cushing_holdout_replay_trades"):
        if r["method"] != "always_long_spread":
            continue
        f = mapping[r["release_date"]]
        window = calendar[r["release_date"]]
        move = int(r["gross_usd"])
        if (f["target"] != float(r["published_million_bbl"]) or
                any(r[k]!=window[k] for k in ("entry_session","exit_session","near_month","far_month")) or
                not datetime.fromisoformat(f["target_known_by_utc"])<=datetime.fromisoformat(window["decision_at_utc"])
                <datetime.fromisoformat(window["entry_mark"])):
            raise ValueError(f"Market calendar differs on {r['release_date']}")
        for method in (*methods, "published_change", "always_long_spread", "no_trade"):
            expected = f.get(method,0.)
            surprise = f["target"]-expected
            side = (1 if method == "always_long_spread" else 0 if method == "no_trade" else int(surprise<0)-int(surprise>0))
            gross = side*move
            yield {k:r[k] for k in ("release_date", "scope", "source", "entry_session", "exit_session", "near_month", "far_month")} | dict(
                method=method, near_side=side, gross_usd=gross,
                **{f"cost_{c}_usd":abs(side)*c for c in COSTS}, **{f"net_{c}_usd":gross-abs(side)*c for c in COSTS})


def main():
    rows = list(measurements())
    weekly = list(summarize(rows))
    forecast = list(predict(week_inputs(weekly)))
    trades = list(trade_rows(forecast))
    by_market = defaultdict(list)
    for r in trades:
        by_market[r["scope"],r["method"]].append(r)
    summary = dict(status="exploratory_replay_on_previously_examined_history", measured_tank_weeks=len(rows),
                   forecast={phase:{scope:scores([r for r in forecast if r["study_phase"] == phase and (scope == "all" or r["measured"])])
                                    for scope in ("all", "with_optical")} for phase in ("development", "validation", "holdout")},
                   trading={f"{scope}:{method}":statistics(sample) for (scope,method),sample in sorted(by_market.items())})
    for suffix, sample in (("measurements",rows), ("weeks",weekly), ("terminals",list(summarize(rows,terminals=True))),
                           ("forecasts",forecast), ("trades",trades)):
        export(f"{STEM}_{suffix}",sample)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(summary,indent=2,allow_nan=False)+"\n").encode())
    for phase, scopes in summary["forecast"].items():
        for scope, r in scopes.items():
            print(phase, scope, r["releases"], {k:round(v["mae_million_bbl"],4) for k,v in r["methods"].items()})
    for method, r in summary["trading"].items():
        print(method, r["net_50_usd"], r["net_170_usd"])


if __name__ == "__main__":
    main()
