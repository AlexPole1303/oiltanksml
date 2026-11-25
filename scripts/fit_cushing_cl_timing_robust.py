"""Train on a mixture of plausible CL entry/exit marks and replay frozen sessions."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_leader_execution import COSTS, SCENARIOS as CLOCKS, anchors, settlement_paths, tally
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast

STEM = "cushing_cl_timing_robust"
ORIGINAL = "weighted_reference"
METHODS = (ORIGINAL, "half_weight_on_timing_mixture", "timing_mixture")
BLEND = dict(zip(METHODS, (0., .5, 1.), strict=True))


def replay():
    f = indexed(f"{OUTER}_forecasts")
    ref = {r["release_date"]: r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    past = {m: [] for m in METHODS}
    for r, _ in inputs():
        issue, decision = r["release_date"], r["decision"]
        x = f[issue]
        cutoff, known = (datetime.fromisoformat(x[k]) for k in ("cutoff_utc", "known_utc"))
        target = float(x["actual"])
        inventory = float(x[SUPPLEMENT])
        if (not cutoff < known <= decision < r["entry"] < r["exit_at"] or target != r["x"][0]
                or x[f"{SUPPLEMENT}_latest_label_utc"]
                and datetime.fromisoformat(x[f"{SUPPLEMENT}_latest_label_utc"]) >= cutoff):
            raise ValueError(f"Inconsistent EIA clock on {issue}")
        path = r["spread_path_usd"]
        outcomes = {m: path[b - 2] - path[a - 2] for m, (a, b) in CLOCKS.items()}
        if path[0] != 0 or path[-1] != r["long_gross"]:
            raise ValueError(f"Invalid historical price path on {issue}")
        timing_mixture = float(np.mean(list(outcomes.values())))
        design = replace_expectation(r, inventory)
        context = features(r["x"][2], r["volatility_usd"], target - inventory)
        for m, previous in past.items():
            # Even an outcome defined using an earlier mark is admitted only
            # after the last scheduled mark. Every variant sees the same history.
            settled = [q for q in previous if q["exit_at"] < decision][-HISTORY:]
            response = (1 - BLEND[m]) * r["long_gross"] + BLEND[m] * timing_mixture
            row = r | dict(x=design, long_gross=response, response_features=context)
            previous.append(row)
            if len(settled) < MIN_WINDOWS:
                continue
            point = volatility_forecast(settled, row)
            calibrated = [q for q in settled if "base_estimate" in q]
            residual, _ = conditional_residual(calibrated, row) if len(calibrated) >= MIN_WINDOWS else (0., 0.)
            row["base_estimate"] = point
            estimate = point + residual
            result = trade(r, m, estimate, 3, training_at=settled[-1]["exit_at"],
                           calibration_at=calibrated[-1]["exit_at"] if calibrated else None)
            if m == ORIGINAL and (any(result[k] != int(ref[issue][k]) for k in ("near_side", "gross_usd", "net_170_usd"))
                    or not math.isclose(estimate, float(ref[issue]["expected_long_move_usd"]), abs_tol=1e-8)):
                raise ValueError(f"The original weighted rule changed on {issue}")
            side = result["near_side"]
            yield result | dict(training_windows=len(settled), mixture_long_move_usd=timing_mixture,
                unadjusted_long_move_usd=r["long_gross"], net_340_usd=result["gross_usd"] - 340 * abs(side),
                **{f"{scenario}_gross_usd": side * gross for scenario, gross in outcomes.items()})


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"], r["method"]].append(r)
    output = {}
    for phase in ("development", "validation", "holdout"):
        base = groups[phase, ORIGINAL]
        phase_report = {}
        for m in METHODS:
            arm = groups[phase, m]
            if not arm or [r["release_date"] for r in arm] != [q["release_date"] for q in base]:
                raise ValueError(f"Missing matching trade windows on {phase}/{m}")
            stress = {}
            for scenario in CLOCKS:
                repriced, comparison = (after_cost(sample, scenario) for sample in (arm, base))
                delta = [r["net_170_usd"] - q["net_170_usd"] for r, q in zip(repriced, comparison, strict=True)]
                stress[scenario] = tally(repriced) | dict(paired_delta_vs_reference_95_usd=block_bootstrap_mean(delta))
            point = np.array([r["expected_long_move_usd"] for r in arm])
            anchor = np.array([r["expected_long_move_usd"] for r in base])
            errors = {}
            for label in ("unadjusted_long_move_usd", "mixture_long_move_usd"):
                target = np.array([r[label] for r in arm])
                e = abs(point - target)
                errors[label] = dict(mae_usd=float(e.mean()),
                                     paired_mae_delta_vs_reference_95_usd=block_bootstrap_mean(e - abs(anchor - target)))
            phase_report[m] = dict(changed_sides=sum(r["near_side"] != q["near_side"] for r, q in zip(arm, base, strict=True)),
                **daily_risk(arm), payoff_error=errors, stress=stress)
        output[phase] = phase_report
    return output


def after_cost(rows, scenario):
    key = f"{scenario}_gross_usd"
    return [r | dict(gross_usd=r[key], **{f"net_{cost}_usd":r[key] - cost * abs(r["near_side"])
                                        for cost in COSTS}) for r in rows]


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
    phases = summarize(rows)
    priced = {i: marks[source] for i, _, _, source, marks in settlement_paths(anchors())}
    for r in rows:
        if r["study_phase"] != "holdout":
            continue
        p = priced[r["release_date"]]
        for name, (a, b) in CLOCKS.items():
            if r[f"{name}_gross_usd"] != r["near_side"] * (p[b - 2] - p[a - 2]):
                raise ValueError(f"Timing stress failed independent price check for {r['release_date']}/{name}")
    result = dict(status="exploratory_on_reviewed_history", phases=phases)
    export(f"{STEM}_trades", rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    for phase, policies in phases.items():
        print(phase, {m: {s: (q["trades"], q["net_170_usd"], q["net_340_usd"]) for s, q in d["stress"].items()}
                      for m, d in policies.items()})


if __name__ == "__main__":
    main()
