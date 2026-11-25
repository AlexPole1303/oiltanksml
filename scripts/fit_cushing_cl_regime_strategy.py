"""Walk forward on the net economics of trading dated WTI calendar spreads."""

import json
from collections import defaultdict
from contextlib import closing
from datetime import datetime

import numpy as np
from sklearn.linear_model import HuberRegressor

from backtest_wti_cushing_spread import COSTS, statistics
from build_cushing_holdout_market_features import make_plan, numeric_prices, read_protocol, vector_from_marks
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed
from fit_cushing_sensor_market_nowcast import MARKET_HORIZONS, contract_book, market_features

STEM = "cushing_cl_regime_strategy"
MIN_WINDOWS = 52
HISTORY = 156
FEATURES = ("published", "public_surprise", "carry", "carry_week_change", "surprise_x_carry",
            "optical_available", "optical_revision", "optical_x_carry")
RULES = ("public", "public_optical")
HURDLES = (50, 170)
CARRY = 1
CARRY_CHANGE_5D = 3 + 3 * MARKET_HORIZONS.index(5) + CARRY


def market(known):
    days, prices = contract_book()
    plan = list(make_plan(read_protocol()))
    with closing(read_only(SOURCE)) as db:
        closes = numeric_prices(db, {source for _, source, _, _ in plan},
                                through=max(r["observed_session"] for _, _, marks, _ in plan for r in marks))
    for row, source, marks, _ in plan:
        vector, sessions = vector_from_marks(marks, source, closes)
        issue = row["release_date"]
        if issue not in known or row["forecast_cutoff_utc"] != known[issue]["forecast_cutoff_utc"] or sessions[1] >= issue:
            raise ValueError(f"Invalid market feature cutoff on {issue}")
        pre_release = vector[len(vector)//2:]
        yield issue, (pre_release[CARRY], pre_release[CARRY_CHANGE_5D], sessions[1])
    for issue, row in known.items():
        if row["study_phase"] == "holdout":
            continue
        vector, sessions = market_features(days, prices, (datetime.fromisoformat(row["forecast_cutoff_utc"]),))
        if not sessions[0] or sessions[0] >= issue:
            raise ValueError(f"Invalid open-period market feature cutoff on {issue}")
        yield issue, (vector[CARRY], vector[CARRY_CHANGE_5D], sessions[0])


def inputs():
    forecasts = indexed("cushing_robust_stack_forecasts")
    context = dict(market(forecasts))
    windows = indexed("wti_cushing_event_windows")
    earlier = [r for r in read("wti_cushing_event_trades") if r["scope"] == "sensor_all" and r["source"] == "archive"
               and r["method"] == "always_long_spread"]
    later = [r for r in read("cushing_holdout_replay_trades") if r["scope"] == "all" and r["method"] == "always_long_spread"]
    previous_exit = None
    for r in sorted(earlier + later, key=lambda item: item["release_date"]):
        issue = r["release_date"]
        w, f = windows[issue], forecasts[issue]
        carry, carry_delta, source_session = context[issue]
        target = float(f["target"])
        decision, entry, exit_at = (datetime.fromisoformat(w[k]) for k in ("decision_at_utc", "entry_mark", "exit_mark"))
        if any(r[k] != w[k] for k in ("entry_session", "exit_session", "near_month", "far_month")) or target != float(r["published_million_bbl"]):
            raise ValueError(f"Contract, calendar or vintage mismatch on {issue}")
        if not datetime.fromisoformat(f["target_known_by_utc"]) <= decision < entry < exit_at:
            raise ValueError(f"Trade decision does not precede entry on {issue}")
        if previous_exit is not None and previous_exit >= entry:
            raise ValueError(f"Overlapping spread windows on {issue}")
        previous_exit = exit_at
        public_surprise = target-float(f["robust_public"])
        optical_revision = float(f["robust_optical"])-float(f["robust_public"])
        x = (target, public_surprise, carry, carry_delta, public_surprise*carry,
             float(f["optical_available"] == "True"), optical_revision, optical_revision*carry)
        yield dict(r, decision=decision, entry=entry, exit_at=exit_at, market_feature_session=source_session,
                   x=x, long_gross=int(r["gross_usd"]))


def predictions(history, row, width):
    x = np.array([r["x"][:width] for r in history])
    valid = np.isfinite(x)
    n = valid.sum(axis=0)
    centre = np.divide(np.where(valid,x,0.).sum(axis=0), n, out=np.zeros(width), where=n>0)
    x = np.where(valid,x-centre,0.)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.
    z = np.asarray(row["x"][:width])-centre
    z = np.where(np.isfinite(z),z/scale,0.)
    y = np.array([r["long_gross"] for r in history])/1000
    weights = np.exp2(-np.arange(len(history)-1,-1,-1)/78)
    model = HuberRegressor(epsilon=1.5, alpha=10., max_iter=400).fit(np.clip(x/scale,-4,4), y, sample_weight=weights)
    return 1000*float(model.predict(np.clip(z,-4,4)[None,:])[0])


def replay(rows):
    seen = []
    for row in rows:
        prior = [r for r in seen if r["exit_at"] < row["decision"]][-HISTORY:]
        seen.append(row)
        if len(prior) < MIN_WINDOWS:
            continue
        estimate = {rule: predictions(prior, row, width) for rule, width in zip(RULES, (5, len(FEATURES)), strict=True)}
        for rule, expected in estimate.items():
            for hurdle in HURDLES:
                side = int(expected > hurdle)-int(expected < -hurdle)
                gross = side*row["long_gross"]
                yield {k:row[k] for k in ("release_date", "study_phase", "entry_session", "exit_session", "near_month", "far_month",
                                           "source", "market_feature_session", "long_gross")} | dict(
                    method=rule, hurdle_usd=hurdle, expected_long_gross_usd=expected, near_side=side, gross_usd=gross,
                    decision_at_utc=row["decision"].isoformat(), training_windows=len(prior),
                    latest_training_exit=prior[-1]["exit_at"].isoformat(),
                    **{f"cost_{c}_usd": c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    rows = list(replay(inputs()))
    by_rule = defaultdict(list)
    for r in rows:
        by_rule[r["study_phase"], r["method"], r["hurdle_usd"]].append(r)
    report = {}
    for phase in ("development", "validation", "holdout"):
        out = {f"{rule}:{hurdle}": statistics(sample) for (p, rule, hurdle), sample in sorted(by_rule.items()) if p == phase}
        for hurdle in HURDLES:
            public = by_rule[phase,"public",hurdle]
            optical = by_rule[phase,"public_optical",hurdle]
            if [r["release_date"] for r in public] != [r["release_date"] for r in optical]:
                raise ValueError(f"Different public and optical calendar in {phase}")
            delta = [b["net_50_usd"]-a["net_50_usd"] for a,b in zip(public,optical,strict=True)]
            out[f"optical_minus_public:{hurdle}"] = dict(net_usd=sum(delta), mean_95pct_ci_usd=block_bootstrap_mean(delta))
        report[phase] = out
    candidate = by_rule["holdout", "public", 170]
    stress = [r["net_170_usd"] for r in candidate]
    best = max(candidate, key=lambda r: r["net_170_usd"])
    result = dict(status="exploratory_replay_on_previously_examined_history", phases=report,
                  public_170_holdout_sensitivity=dict(
                      mean_stressed_net_per_window_95pct_block_bootstrap_usd=block_bootstrap_mean(stress),
                      largest_winning_release=best["release_date"], largest_win_after_stress_usd=best["net_170_usd"],
                      net_without_largest_win_after_stress_usd=sum(stress)-best["net_170_usd"]))
    export(f"{STEM}_trades", rows)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False)+"\n").encode())
    for phase, rules in report.items():
        print(phase, {key:(row["trades"], row["net_50_usd"], row["net_170_usd"]) for key, row in rules.items() if "trades" in row})


if __name__ == "__main__":
    main()
