"""Replay inventory variants through common weighted CL entries and exits."""

import math
from collections import defaultdict

from build_s1_reference_matches import read
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_exit_horizon import inputs as market_inputs, trade
from fit_cushing_cl_extension_segments import (
    archive as following_archive,
    calibrated_inventory,
    case as holding_case,
    exit_indices,
    predict as predict_holding,
)
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_outer_capture_nowcast import SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast

HOLDING_RULE = "curve_post_report_gate"


def replay_weighted_market(forecasts_by_issue, models, *, reference, historical_inputs=None):
    historical_inputs = historical_inputs or {}
    if any(m not in models or source not in models or m == source for m,source in historical_inputs.items()):
        raise ValueError("The fixed trading history must refer to a different model in the same run")
    original = {r["release_date"]:r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    references = {r["release_date"]:r for r in read("cushing_cl_extension_segments_trades") if r["method"] == HOLDING_RULE}
    following = {r["issue"]:r for r in calibrated_inventory(following_archive())}
    history_by_model = defaultdict(list)
    history_by_exit = {m:defaultdict(list) for m in models}
    for row, _ in market_inputs():
        issue, decision = row["release_date"], row["decision"]
        f = forecasts_by_issue[issue]
        if not f["cutoff"] < f["known"] <= decision < row["entry"] or f["actual"] != row["x"][0]:
            raise ValueError(f"The inventory vintage is not available at the original decision on {issue}")
        for name in models:
            expected = f[name]
            current = row | dict(x=replace_expectation(row, expected), response_features=features(
                row["x"][2], row["volatility_usd"], f["actual"] - expected))
            source = historical_inputs.get(name, name)
            seen = history_by_model[source]
            closed = [q for q in seen if q["exit_at"] < decision][-HISTORY:]
            if source == name:
                seen.append(current)
            if len(closed) < MIN_WINDOWS:
                continue
            point = volatility_forecast(closed, current)
            calibration = [q for q in closed if "base_estimate" in q]
            correction, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            current["base_estimate"] = point
            timing = dict(training_at=closed[-1]["exit_at"], calibration_at=calibration[-1]["exit_at"] if calibration else None)
            result = trade(row, name, point + correction, 3, **timing)
            old = original[issue]
            if name == reference and (not math.isclose(point + correction, float(old["expected_long_move_usd"]), abs_tol=1e-8, rel_tol=0)
                    or any(result[k] != int(old[k]) for k in ("near_side", "gross_usd", "net_170_usd"))):
                raise ValueError(f"The original trade differs on {issue}")
            context = holding_case(row, point + correction, following.get(row["next_issue"]))
            holding = predict_holding(history_by_exit[name], context) if context else {}
            index = row["last_before_next_release"]
            early = index if index in (1, 2) else 3
            close = min(early + 1, 3)
            end = exit_indices(context["sigma"] if context else None, holding, result["near_side"], early, close)[HOLDING_RULE]
            change = trade(row, f"{name}_conditional_exit", point + correction, end, **timing)
            if name == reference:
                paired = references[issue]
                if (any(change[k] != paired[k] for k in ("near_month", "far_month", "entry_at_utc", "exit_at_utc", "decision_at_utc"))
                        or any(change[k] != int(paired[k]) for k in ("near_side", "gross_usd", "net_170_usd"))):
                    raise ValueError(f"The latest execution reference differs on {issue}")
            latest_holding = max((q["latest"] for q in holding.values()), default=None)
            for candidate in (result, change):
                yield candidate | dict(inventory_forecast_kbbl=expected * 1000,
                                        net_340_usd=candidate["gross_usd"] - 340 * abs(candidate["near_side"]),
                                        latest_holding_label_utc=latest_holding if candidate is change else None,
                                        holding_decision_utc=context["decision"] if context and candidate is change else None)

