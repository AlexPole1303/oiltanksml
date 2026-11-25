"""Compare fixed CL decision clocks and execution windows with horizon-specific training."""

import json
from collections import defaultdict
from math import sqrt

from audit_cushing_cl_horizons import paths
from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, inputs
from fit_cushing_cl_volatility import HURDLE, PHASES, RISK_FLOOR_USD, risk_inputs, risk_prediction, summarize
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_execution_windows"
WINDOWS = ((1, 5), (2, 5), (1, 4), (2, 4))
METHODS = ("public_volatility", "coverage_volatility", "carry_only_volatility")


def replay():
    forecasts = indexed("cushing_coverage_nowcast_forecasts")
    history = defaultdict(list)
    for row, days, times, moves in paths(list(risk_inputs(list(inputs())))):
        f = forecasts[row["release_date"]]
        revision = float(f["coverage_ml"])-float(f["public_ensemble"])
        public = row["x"][:5]
        features = (public, (*public, revision, revision*public[2]), public[2:4])
        for first, last in WINDOWS:
            if not row["decision"] < times[first-1] < times[last-1]:
                raise ValueError(f"Execution clock outside the window on {row['release_date']}")
            realized = moves[last-1]-moves[first-1]
            sigma = max(RISK_FLOOR_USD, row["volatility_usd"]*sqrt((last-first)/4))
            for method, x in zip(METHODS, features, strict=True):
                key = first, last, method
                current = dict(x=x, long_gross=realized, volatility_usd=sigma, exit_at=times[last-1])
                train = [r for r in history[key] if r["exit_at"] < row["decision"]][-HISTORY:]
                history[key].append(current)
                if len(train) < MIN_WINDOWS:
                    continue
                estimate = risk_prediction(train, current, len(x))
                side = int(estimate > HURDLE)-int(estimate < -HURDLE)
                gross = side*realized
                yield {k:row[k] for k in ("study_phase", "release_date", "near_month", "far_month", "source")} | dict(
                    method=method, first_session_number=first, last_session_number=last, entry_session=days[first-1],
                    exit_session=days[last-1], decision_at_utc=row["decision"].isoformat(),
                    latest_training_exit=train[-1]["exit_at"].isoformat(), volatility_usd=sigma,
                    expected_long_gross_usd=estimate, near_side=side, gross_usd=gross,
                    **{f"cost_{c}_usd":c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def robustness(rows):
    outcome = summarize(rows)
    months = defaultdict(int)
    for r in rows:
        months[r["release_date"][:7]] += r["net_170_usd"]
    return outcome | dict(stress_net_without_best_month_usd=outcome["net_170_usd"]-max(months.values()),
                          stress_net_without_april_2026_usd=outcome["net_170_usd"]-months.get("2026-04", 0))


def main():
    rows = list(replay())
    previous = {(r["release_date"], r["method"]):r for r in read("cushing_cl_volatility_trades") if r["method"] in METHODS}
    baseline = [r for r in rows if (r["first_session_number"], r["last_session_number"]) == (1, 5)]
    if len(baseline) != len(previous) or any(r["net_170_usd"] != int(previous[r["release_date"], r["method"]]["net_170_usd"]) for r in baseline):
        raise ValueError("The baseline five-session variant failed to reproduce")
    groups = defaultdict(list)
    for r in rows:
        key = r["first_session_number"], r["last_session_number"], r["method"]
        groups[r["study_phase"], key].append(r)
    keys = [(first, last, method) for first, last in WINDOWS for method in METHODS]
    reports, paired = {}, {}
    for phase in PHASES:
        base_calendar = [r["release_date"] for r in groups[phase, keys[0]]]
        reports[phase], paired[phase] = {}, {}
        for key in keys:
            first, last, method = key
            group = groups[phase, key]
            if [r["release_date"] for r in group] != base_calendar:
                raise ValueError(f"Calendars differ in {phase} for {key}")
            name = f"{first}-{last}:{method}"
            reports[phase][name] = robustness(group)
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(group,groups[phase, (1, 5, method)], strict=True)]
            paired[phase][name] = dict(net_increment_usd=sum(delta), mean_ci_95_usd=block_bootstrap_mean(delta))
    def selection_score(name):
        first, second = [reports[p][name]["net_170_usd"] for p in PHASES[:2]]
        return min(first, second), first+second

    early_choice = max(reports[PHASES[0]], key=selection_score)
    report = dict(status="exploratory_replay_on_previously_examined_history", phases=reports, paired_vs_same_model_1_5=paired,
                  chosen_using_development_and_2023=early_choice)
    payload = json.dumps(report, indent=2)+"\n"
    export(f"{STEM}_trades", rows)
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(f"Pre-2024 choice: {early_choice}")
    for phase, group in reports.items():
        print(phase, {k:(v["trades"], v["net_170_usd"], v["stress_net_without_best_month_usd"]) for k,v in group.items()})


if __name__ == "__main__":
    main()
