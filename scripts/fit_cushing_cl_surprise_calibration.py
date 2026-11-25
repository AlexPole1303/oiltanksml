"""Walk-forward correction to the original CL signal using EIA percentiles."""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_inventory_conditional_scale import EXIT_METHOD, METHODS, STEM as DISTRIBUTIONS
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_storage_state import fit_correction

STEM = "cushing_cl_surprise_calibration"
REFERENCE = "registered_exit"


def replay():
    forecasts = {q["release_date"]: q for q in read(f"{DISTRIBUTIONS}_forecasts") if q["channel"] == "market"}
    anchors = {q["release_date"]: q for q in read("cushing_cl_extension_segments_trades") if q["method"] == EXIT_METHOD}
    history = defaultdict(list)
    for row, _ in inputs():
        issue = row["release_date"]
        if issue not in anchors:
            continue
        ref = anchors[issue]
        estimate = float(ref["expected_long_move_usd"])
        original_training = datetime.fromisoformat(ref["original_latest_training_exit_utc"])
        index = int(ref["exit_session_number"]) - 2
        baseline = trade(row, REFERENCE, estimate, index, training_at=original_training)
        if (any(baseline[k] != ref[k] for k in ("entry_at_utc", "exit_at_utc", "near_month", "far_month"))
                or any(baseline[k] != int(ref[k]) for k in ("near_side", "gross_usd", "net_170_usd"))
                or not original_training < row["decision"] < row["entry"] < row["stamps"][index]):
            raise ValueError(f"The frozen market comparison did not reproduce on {issue}")
        sigma = row["volatility_usd"]
        realized = row["spread_path_usd"][index]
        yield with_diagnostics(baseline, estimate, realized)
        f = forecasts.get(issue)
        if f:
            label, cutoff, known = (datetime.fromisoformat(f[k]) for k in ("latest_label_utc", "cutoff", "known"))
            if not label < cutoff < known <= row["decision"] or float(f["actual"]) != row["x"][0]:
                raise ValueError(f"Wrong inventory information set on {issue}")
        for method in METHODS:
            correction = 0.
            previous = [q for q in history[method] if q["known"] < row["decision"]][-HISTORY:]
            if f:
                percentile = 2 * float(f[f"{method}_percentile"]) - 1
                strength = float(np.clip(estimate / sigma, -2., 2.))
                current = dict(cutoff=row["decision"], known=row["stamps"][index],
                               design=(percentile, strength * (abs(percentile) - .5)),
                               residual=(realized - estimate) / sigma)
                correction = sigma * fit_correction(previous, current, "design", "residual",
                                                     penalty=(.25, .35), loss_scale=1., bound=.25)
                history[method].append(current)
            adjusted = estimate + correction
            # The cutoff and possible extension were specified for the original
            # side. This overlay can retain that side or abstain.
            admitted = adjusted if adjusted * baseline["near_side"] > 0 else 0.
            candidate = trade(row, method, admitted, index, training_at=original_training,
                              calibration_at=max(q["known"] for q in previous) if f and previous else None)
            if candidate["near_side"] not in (0, baseline["near_side"]):
                raise ValueError(f"The overlay changed the original side on {issue}")
            yield with_diagnostics(candidate, adjusted, realized, correction)


def with_diagnostics(record, estimate, realized, correction=0.):
    return record | dict(adjusted_long_move_usd=estimate, realized_long_move_usd=realized,
                         adjustment_usd=correction, net_340_usd=record["gross_usd"]-340*abs(record["near_side"]))


def summarize(rows):
    phases = {}
    for phase in PHASES:
        by_method = {m: [q for q in rows if (q["study_phase"], q["method"]) == (phase, m)]
                     for m in (REFERENCE, *METHODS)}
        base = by_method[REFERENCE]
        if not base:
            continue
        result = {}
        for name, group in by_method.items():
            if [q["release_date"] for q in group] != [q["release_date"] for q in base]:
                raise ValueError("Unmatched market samples")
            absolute_error = [abs(q["adjusted_long_move_usd"] - q["realized_long_move_usd"]) for q in group]
            result[name] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(q["net_340_usd"] for q in group),
                spread_mae_usd=float(np.mean(absolute_error)),
                changed_trades=sum(q["near_side"] != b["near_side"] for q, b in zip(group, base, strict=True)),
                delta_95_usd=block_bootstrap_mean([q["net_170_usd"] - b["net_170_usd"]
                                                for q, b in zip(group, base, strict=True)]))
        phases[phase] = result
        print(phase, {m: (q["net_170_usd"], q["net_340_usd"], round(q["spread_mae_usd"], 2),
                          q["changed_trades"]) for m, q in result.items()})
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
    summary = summarize(rows)
    export(f"{STEM}_trades", rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
