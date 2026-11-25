"""Use forecasts available before the following EIA report to decide whether to hold."""

import json
import math
from collections import defaultdict
from datetime import datetime, time

from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, RISK_FLOOR_USD, risk_prediction
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from publication_clocks import EASTERN

STEM = "cushing_cl_following_report"
METHODS = ("hold_fifth", "exit_before_report", "curve_gate", "public_gate", "optical_gate", "optical_revision_gate")


def prerelease():
    forecasts = {}
    for q in read("cushing_prerelease_sensor_forecasts"):
        issue = q["issue"]
        cutoff, known = (datetime.fromisoformat(q[k]) for k in ("cutoff_utc", "known_utc"))
        if issue in forecasts or not cutoff < known or any(
            q[k] and datetime.fromisoformat(q[k]) >= cutoff
            for k in ("latest_inventory_training_utc", "latest_optical_input_utc")
        ):
            raise ValueError(f"Invalid early forecast lineage on {issue}")
        forecasts[issue] = dict(cutoff=cutoff, known=known, public=float(q["early_public_ml"]),
                                optical=float(q["early_recovered_scene"]),
                                roofs=int(q["measured_roofs"])+int(q["recovered_roofs"]))
    return forecasts


def context(r, following):
    index = r["last_before_next_release"]
    if following is None or index not in (1, 2):
        return None
    exit_cutoff = datetime.combine(r["stamps"][index].date(), time.min, EASTERN)
    if not r["entry"] < exit_cutoff < r["stamps"][index] < r["next_report_clock"] <= r["exit_at"]:
        raise ValueError(f"Invalid holding decision time on {r['release_date']}")
    if following["cutoff"] > exit_cutoff:
        return None
    if following["known"] <= r["stamps"][index] or following["known"].astimezone(EASTERN).date().isoformat() <= r["next_issue"]:
        raise ValueError(f"Next inventory report was prematurely admitted on {r['release_date']}")
    sigma = max(RISK_FLOOR_USD, r["volatility_usd"]*math.sqrt((3-index)/3))
    target, _, carry, _, *_ = r["x"]
    vol = math.log(sigma/1000)
    # The return from the close being decided on is unavailable when the
    # choice is made. Only the previous session's movement enters the model.
    known_path = r["spread_path_usd"][index-1]/sigma
    core = (carry, vol, target, known_path)
    public, optical = (following[k] for k in ("public", "optical"))
    revision = optical-public
    designs = (core, (*core, public, public*vol), (*core, optical, optical*vol),
               (*core, public, public*vol, revision, revision*vol))
    return r | dict(hold_decision=exit_cutoff, volatility_usd=sigma, designs=designs,
                    long_gross=r["spread_path_usd"][-1]-r["spread_path_usd"][index], following=following)


def replay():
    forecasts = prerelease()
    seen = []
    for r, anchors in inputs():
        f = forecasts.get(r["next_issue"])
        row = context(r, f)
        past = []
        if row is not None:
            decision = row["hold_decision"]
            past = [q for q in seen if q["exit_at"] < decision][-HISTORY:]
            seen.append(row)
        if "inventory_conditional" not in anchors:
            continue
        expected = float(anchors["inventory_conditional"]["expected_long_move_usd"])
        side = int(expected > HURDLE)-int(expected < -HURDLE)
        index = r["last_before_next_release"]
        early = (index or 0) if r["next_report_clock"] is not None else 3
        entry_expected = expected if early else 0.
        estimates = []
        if len(past) >= MIN_WINDOWS:
            for j, x in enumerate(row["designs"]):
                estimates.append(risk_prediction([q | dict(x=q["designs"][j]) for q in past], row | dict(x=x), len(x)))
        else:
            estimates = [None]*4
        forecast_fields = dict(following_forecast_cutoff_utc=None, hold_decision_utc=None, following_forecast_roofs=None)
        if row is not None:
            forecast_fields.update(following_forecast_cutoff_utc=row["following"]["cutoff"].isoformat(),
                                   hold_decision_utc=row["hold_decision"].isoformat(),
                                   following_forecast_roofs=row["following"]["roofs"])
        for m, move in zip(METHODS, (None, None, *estimates), strict=True):
            end = 3 if m == "hold_fifth" or (move is not None and side*move > 0) else early
            result = trade(r, m, expected if m == "hold_fifth" else entry_expected, end)
            yield result | forecast_fields | dict(
                expected_long_move_if_held_usd=move, extension_training_windows=len(past),
                latest_extension_outcome_utc=past[-1]["exit_at"].isoformat() if past else None,
                moved_exit_from_fifth=bool(result["near_side"] and end != 3))


def main():
    with threadpool_limits(limits=1):
        trades = list(replay())
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        groups = defaultdict(list)
        for q in trades:
            if q["study_phase"] == phase:
                groups[q["method"]].append(q)
        base = groups["hold_fifth"]
        curve = groups["curve_gate"]
        public = groups["public_gate"]
        report["phases"][phase] = {}
        for method, rows in groups.items():
            if [q["release_date"] for q in rows] != [q["release_date"] for q in base]:
                raise ValueError(f"Unmatched holding windows on {phase}/{method}")
            delta = [q["net_170_usd"]-b["net_170_usd"] for q, b in zip(rows, base, strict=True)]
            curve_delta = [q["net_170_usd"]-c["net_170_usd"] for q, c in zip(rows, curve, strict=True)]
            public_delta = [q["net_170_usd"]-p["net_170_usd"] for q, p in zip(rows, public, strict=True)]
            report["phases"][phase][method] = robustness(rows) | daily_risk(rows) | dict(
                net_340_usd=sum(q["gross_usd"]-340*abs(q["near_side"]) for q in rows),
                windows_with_next_forecast=sum(q["hold_decision_utc"] is not None for q in rows),
                traded_windows_with_next_forecast=sum(bool(q["near_side"] and q["hold_decision_utc"]) for q in rows),
                traded_windows_with_next_optical=sum(bool(q["near_side"] and q["following_forecast_roofs"]) for q in rows),
                held_across_next_release=sum(bool(q["near_side"] and q["crosses_next_eia"]) for q in rows),
                delta_vs_hold_usd=sum(delta), delta_vs_hold_95_usd=block_bootstrap_mean(delta),
                delta_vs_curve_gate_usd=sum(curve_delta), delta_vs_curve_gate_95_usd=block_bootstrap_mean(curve_delta),
                delta_vs_public_gate_usd=sum(public_delta), delta_vs_public_gate_95_usd=block_bootstrap_mean(public_delta),
                traded_exits_different_from_curve=sum(bool(q["near_side"] and q["exit_at_utc"] != c["exit_at_utc"])
                                                     for q, c in zip(rows, curve, strict=True)))
        print(phase,{m:(v["net_170_usd"],v["net_340_usd"],v["held_across_next_release"])
                     for m,v in report["phases"][phase].items()})
    export(f"{STEM}_trades", trades)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
