"""Compare CL exits before and across the following EIA release."""

import json
import math
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime

from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features as response_features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, RISK_FLOOR_USD, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_exit_horizon"
HORIZONS = ("third", "fourth")
REFERENCES = ("inventory_conditional", "optical_conditional")


def inputs():
    forecasts = indexed("cushing_recovered_optical_forecasts")
    clocks = indexed("eia_release_clock")
    # A final trade in the research sample can exit before a subsequent
    # published release outside the sample. Import its date, never its value.
    for row in read("eia_wpsr_forward_releases"):
        issue = row["release_date"]
        if issue not in clocks:
            clocks[issue] = dict(timing_basis=row["timing_basis"],
                                 scheduled_table_release_at_utc=row["scheduled_release_at_utc"],
                                 conservative_prerelease_before_utc=row["conservative_prerelease_before_utc"])
    release_days = sorted(clocks)
    schedule = {(r["release_date"], int(r["session_number"])): datetime.fromisoformat(r["settlement_reference_time"])
                for r in read("wti_cushing_event_bars") if r["leg"] == "near"}
    anchors = {(r["release_date"], r["method"]): r for r in read("cushing_cl_conditional_response_trades")
               if r["method"] in REFERENCES}
    _, _, cases = market_calendar()
    for r in cases:
        issue = r["release_date"]
        stamp = [schedule[issue, n] for n in range(2, 6)]
        if not stamp[0] == r["entry"] < stamp[1] < stamp[2] < stamp[3] == r["exit_at"]:
            raise ValueError(f"The delayed market calendar differs on {issue}")
        next_index = bisect_right(release_days, issue)
        next_issue = release_days[next_index] if next_index < len(release_days) else None
        source = clocks.get(next_issue, {})
        event = source.get("scheduled_table_release_at_utc") or source.get("conservative_prerelease_before_utc")
        clock_basis = source.get("timing_basis") if source.get("scheduled_table_release_at_utc") else "start_of_release_day"
        next_report_clock = datetime.fromisoformat(event) if event else None
        last_before = max((offset for offset, moment in enumerate(stamp) if next_report_clock and moment < next_report_clock), default=None)
        forecast = forecasts[issue]
        cutoff, known = (datetime.fromisoformat(forecast[k]) for k in ("cutoff_utc", "known_utc"))
        actual = float(forecast["actual"])
        if not cutoff < known <= r["decision"] < r["entry"] or actual != r["x"][0]:
            raise ValueError(f"Wrong EIA vintage or timing on {issue}")
        public, optical = (float(forecast[k]) for k in (PUBLIC, "all_recovered"))
        designs = tuple(replace_expectation(r, value) for value in (public, optical))
        references = {k: anchors[issue, k] for k in REFERENCES if (issue, k) in anchors}
        yield r | dict(stamps=stamp, next_issue=next_issue, next_report_clock=next_report_clock,
                       next_report_clock_basis=clock_basis if next_report_clock else None,
                       last_before_next_release=last_before, public_actual_surprise=actual-public,
                       optical_actual_surprise=actual-optical, designs=designs), references


def horizon_outcome(row, index):
    # Three price changes in the original four-mark holding window.
    sigma = max(RISK_FLOOR_USD, row["volatility_usd"] * math.sqrt(index / 3))
    return row | dict(volatility_usd=sigma, long_gross=row["spread_path_usd"][index], exit_at=row["stamps"][index])


def strategies():
    history = []
    fitted = defaultdict(list)
    for r, anchors in inputs():
        history.append(r)
        if len(anchors) != len(REFERENCES):
            continue
        for method, anchor in anchors.items():
            if (any(anchor[k] != r[k] for k in ("near_month", "far_month")) or
                any(datetime.fromisoformat(anchor[k]) != r[name] for k, name in (
                     ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))):
                raise ValueError(f"Original trading window differs on {r['release_date']}")
            forecast = float(anchor["expected_long_move_usd"])
            side = int(forecast > HURDLE) - int(forecast < -HURDLE)
            if side != int(anchor["near_side"]) or side*r["long_gross"] != int(anchor["gross_usd"]):
                raise ValueError(f"Original strategy differs on {r['release_date']}")
            kind = "optical" if method.startswith("optical") else "public"
            yield trade(r, "fifth_"+kind, forecast, 3)
            for i, horizon in enumerate(HORIZONS, 1):
                yield trade(r, f"{horizon}_{kind}_fixed_side", forecast, i)
            # This reuses the original side and studies the cutoff on the
            # archived release calendar. It is marked separately from fixed exits.
            if r["next_report_clock"] is not None:
                cutoff = r["last_before_next_release"]
                yield trade(r, f"before_next_{kind}", forecast if cutoff else 0., cutoff or 0)
        for i, horizon in enumerate(HORIZONS, 1):
            train = [horizon_outcome(p, i) for p in history if p["stamps"][i] < r["decision"]][-HISTORY:]
            if len(train) < MIN_WINDOWS:
                raise ValueError(f"Insufficient settled training windows on {r['release_date']}/{horizon}")
            actual = horizon_outcome(r, i)
            for j, kind in enumerate(("public", "optical")):
                key = horizon, kind
                x = r["designs"][j]
                context = response_features(r["x"][2], actual["volatility_usd"], r[f"{kind}_actual_surprise"])
                sample = [p | dict(x=p["designs"][j]) for p in train]
                prediction = risk_prediction(sample, actual | dict(x=x), len(x))
                previous = [p for p in fitted[key] if p["exit_at"] < r["decision"]][-HISTORY:]
                current = actual | dict(response_features=context, base_estimate=prediction)
                adjustment, _ = conditional_residual(previous, current) if len(previous) >= MIN_WINDOWS else (0., 0.)
                fitted[key].append(current)
                yield trade(r, f"{horizon}_{kind}_refit", prediction + adjustment, i,
                            training_at=train[-1]["exit_at"], calibration_at=previous[-1]["exit_at"] if previous else None)


def trade(r, method, expected, exit_index, *, training_at=None, calibration_at=None):
    side = int(expected > HURDLE) - int(expected < -HURDLE)
    exit_at = r["stamps"][exit_index]
    gross = side * r["spread_path_usd"][exit_index]
    risk_path = [side * r["spread_path_usd"][min(i, exit_index)] - (170 if i >= exit_index else 85) * abs(side)
                 for i in range(4)]
    return {k: r[k] for k in ("release_date", "study_phase", "near_month", "far_month", "volatility_usd",
                              "next_issue", "next_report_clock_basis")} | dict(
        method=method, decision_at_utc=r["decision"].isoformat(), entry_at_utc=r["entry"].isoformat(), exit_at_utc=exit_at.isoformat(),
        latest_training_exit_utc=training_at.isoformat() if training_at else None,
        latest_calibration_exit_utc=calibration_at.isoformat() if calibration_at else None,
        next_issue_at_utc=r["next_report_clock"].isoformat() if r["next_report_clock"] else None,
        crosses_next_eia=(r["entry"] < r["next_report_clock"] <= exit_at) if r["next_report_clock"] else None,
        exit_session_number=exit_index+2, expected_long_move_usd=expected, near_side=side, gross_usd=gross,
        alternate_source=r["alternate_source"] if exit_index == 3 else None,
        alternate_gross_usd=side*r["alternate_long_gross_usd"] if exit_index == 3 and r["alternate_source"] else None,
        **{f"session_{i}_net_170_usd":value for i,value in enumerate(risk_path, 2)},
        **{f"cost_{c}_usd":c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    with threadpool_limits(limits=1):
        rows = list(strategies())
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        groups = defaultdict(list)
        for r in rows:
            if r["study_phase"] == phase:
                groups[r["method"]].append(r)
        base = {r["release_date"]: r for r in groups["fifth_public"]}
        group = {}
        for method, sample in groups.items():
            if not method.startswith("before_next_") and list(base) != [r["release_date"] for r in sample]:
                raise ValueError(f"Different trading calendar on {phase}/{method}")
            delta = [r["net_170_usd"]-base[r["release_date"]]["net_170_usd"] for r in sample]
            group[method] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample),
                trades_crossing_next_eia=sum(bool(r["near_side"] and r["crosses_next_eia"]) for r in sample),
                delta_vs_matched_fifth_public_usd=sum(delta), delta_per_window_95_usd=block_bootstrap_mean(delta))
        before = {r["release_date"]: r for r in groups["before_next_public"]}
        matched = [r for issue, r in base.items() if issue in before]
        report["phases"][phase] = group | dict(attribution=dict(
            matched_windows=len(matched), original_trades_crossing=sum(r["near_side"] != 0 and r["crosses_next_eia"] for r in matched),
            original_net_170_usd=sum(r["net_170_usd"] for r in matched),
            last_close_before_next_net_170_usd=sum(r["net_170_usd"] for r in before.values()),
            held_past_last_close_net_increment_usd=sum(r["net_170_usd"]-before[r["release_date"]]["net_170_usd"] for r in matched),
            missing_next_report_clock=[issue for issue in base if issue not in before]))
        print(phase, {m:(x["windows"],x["trades"],x["net_170_usd"],x["net_340_usd"]) for m,x in group.items()})
    export(f"{STEM}_trades", rows)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
