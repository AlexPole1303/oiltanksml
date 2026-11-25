"""Blend frozen short- and long-memory WTI forecasts using settled comparisons."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_volatility import HURDLE, PHASES
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_extended_market_history import STEM as EXTENDED
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_cl_memory_ensemble"
SHORT = "short_reference"
LONG = "long_reference"
FIXED = {SHORT: 0., "quarter_long": .25, "half_long": .5, LONG: 1.}
METHODS = (*FIXED, "settled_error_weight", "settled_trade_weight")


def weights(previous, decision):
    if len(previous) < 52:
        return 0., 0.
    fractions = np.array(list(FIXED.values()))
    ages = np.array([(decision - q["decision"]).total_seconds() / 604800 for q in previous])
    w = np.exp2(-ages / 52)
    short, long, actual, vol = np.array([[q[k] for k in ("short", "long", "long_gross", "volatility_usd")]
                                        for q in previous]).T
    predicted = short[:, None] + (long - short)[:, None] * fractions
    loss = np.minimum(3., abs(predicted - actual[:, None]) / vol[:, None])
    sides = (predicted > HURDLE).astype(int) - (predicted < -HURDLE).astype(int)
    # Score after round-trip costs. Cap single-week influence and shrink
    # the fitted weights toward the short forecast in a small panel.
    gains = np.clip(sides * actual[:, None] - HURDLE * abs(sides), -2000, 2000)
    trimmed_error = w @ (loss[:, 0, None] - loss) / (w.sum() + 16) - .02 * fractions
    trimmed_pnl = w @ (gains - gains[:, 0, None]) / (w.sum() + 16) - 15 * fractions
    return float(fractions[np.argmax(trimmed_error)]), float(fractions[np.argmax(trimmed_pnl)])


def replay():
    labels = {"optical_reference": "short", "optical_long_since_2018": "long"}
    archive = {(r["release_date"], labels[r["method"]]): r for r in read(f"{EXTENDED}_trades") if r["method"] in labels}
    history = []
    for r, _ in inputs():
        issue, decision = r["release_date"], r["decision"]
        if (issue, "short") not in archive:
            continue
        rows = {key: archive[issue, key] for key in labels.values()}
        for q in rows.values():
            if (any(q[k] != r[k] for k in ("near_month", "far_month", "study_phase"))
                    or not math.isclose(float(q["volatility_usd"]), r["volatility_usd"])
                    or int(q["realized_long_move_usd"]) != r["long_gross"]
                    or any(datetime.fromisoformat(q[k]) != r[field] for k, field in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))
                    or any(datetime.fromisoformat(q[k]) >= decision
                           for k in ("latest_training_exit_utc", "latest_calibration_exit_utc") if q[k])):
                raise ValueError(f"Forecast or market calendar differs on {issue}")
        latest_train = max(datetime.fromisoformat(q["latest_training_exit_utc"]) for q in rows.values())
        latest_calibration = max((datetime.fromisoformat(q["latest_calibration_exit_utc"]) for q in rows.values()
                                  if q["latest_calibration_exit_utc"]), default=None)
        current = r | {k: float(q["expected_long_move_usd"]) for k, q in rows.items()}
        eligible = [q for q in history if q["exit_at"] < decision][-156:]
        adaptive = weights(eligible, decision)
        history.append(current)
        for m, w in zip(METHODS, (*FIXED.values(), *adaptive), strict=True):
            estimate = (1 - w) * current["short"] + w * current["long"]
            stamp = latest_train if m in FIXED or not eligible else max(latest_train, eligible[-1]["exit_at"])
            q = trade(r, m, estimate, 3, training_at=stamp, calibration_at=latest_calibration)
            if m in (SHORT, LONG):
                ref = rows["short" if m == SHORT else "long"]
                if any(q[k] != int(ref[k]) for k in ("near_side", "gross_usd", "net_170_usd")):
                    raise ValueError(f"The frozen reference differs on {issue}/{m}")
            yield q | dict(long_weight=w, settled_blend_windows=len(eligible),
                latest_blend_label_utc=eligible[-1]["exit_at"].isoformat() if m not in FIXED and eligible else None,
                realized_long_move_usd=r["long_gross"], net_340_usd=q["gross_usd"] - 340 * abs(q["near_side"]))


def report(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"], r["method"]].append(r)
    phases = {}
    for phase in PHASES:
        base = groups[phase, SHORT]
        result = phases[phase] = {}
        for m in METHODS:
            group = groups[phase, m]
            if [q["release_date"] for q in group] != [q["release_date"] for q in base]:
                raise ValueError(f"The forecast ensemble changed the calendar on {phase}/{m}")
            pairs = list(zip(group, base, strict=True))
            error, ref = (np.array([abs(q[i]["realized_long_move_usd"] - q[i]["expected_long_move_usd"]) for q in pairs])
                          for i in (0, 1))
            delta = [q["net_170_usd"] - a["net_170_usd"] for q, a in pairs]
            result[m] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(r["net_340_usd"] for r in group),
                long_weight_mean=float(np.mean([q["long_weight"] for q in group])), price_move_mae_usd=float(error.mean()),
                paired_error_delta_95_usd=block_bootstrap_mean(error - ref), gain_vs_short_usd=sum(delta),
                paired_gain_per_window_95_usd=block_bootstrap_mean(delta),
                changed_sides=sum(q["near_side"] != a["near_side"] for q, a in pairs))
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def main():
    rows = list(replay())
    result = report(rows)
    export(f"{STEM}_trades", rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    for phase, section in result["phases"].items():
        print(phase, {m: (r["trades"], r["net_170_usd"], r["net_340_usd"], round(r["price_move_mae_usd"], 2))
                      for m, r in section.items()})


if __name__ == "__main__":
    main()
