"""Test delivery-location contrasts in units of their optical estimation error."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_cl_entry_margin import inputs_with_reference
from audit_cushing_cl_equal_model_risk import paired_evidence
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_curve_shape import MIXED_GAP, market_metrics
from fit_cushing_cl_exit_horizon import trade
from fit_cushing_cl_extension_segments import STEM as EXIT_STEM
from fit_cushing_cl_regime_strategy import HISTORY
from fit_cushing_delivery_location import ORIGINAL, STEM as LOCATION_STEM
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import SUPPLEMENT
from fit_cushing_storage_state import fit_correction
from replay_cushing_weighted_inventory_market import HOLDING_RULE, replay_weighted_market

STEM = "cushing_delivery_standardized"
INVENTORY = "standardized_inventory"
HYBRID = "standardized_hybrid"
MARKET = ("standardized_contrast", "standardized_surprise_interaction")


def inventory():
    seen = defaultdict(list)
    for q in read(f"{LOCATION_STEM}_forecasts"):
        info = float(q["contrast_information"])
        z = float(q["raw_contrast_million_bbl"]) * math.sqrt(info) if info else 0.
        standardized = float(np.clip(z, -2.5, 2.5))
        row = {k:q[k] for k in ("release_date", "week_ending", "study_phase", "channel", "near_month", "latest_scene_ready_utc")}
        row |= dict(cutoff=datetime.fromisoformat(q["cutoff"]), known=datetime.fromisoformat(q["known"]), actual=float(q["actual"]),
                    matched_roofs=int(q["matched_roofs"]), contrast_z=z, regressor=standardized, design=(standardized,),
                    **{m:float(q[m]) for m in (SUPPLEMENT, MIXED_GAP)})
        history = seen[row["channel"]]
        past = [r for r in history if r["known"] < row["cutoff"]][-HISTORY:]
        for method, anchor in ((INVENTORY, SUPPLEMENT), (HYBRID, MIXED_GAP)):
            target = f"{anchor}_error"
            row[target] = row["actual"] - row[anchor]
            correction = fit_correction(past, row, "design", target, penalty=(1.,), loss_scale=.75, bound=.25)
            row[method] = row[anchor] + correction
        if info:
            history.append(row)
        yield row | dict(latest_training_label_utc=past[-1]["known"] if past else None)


def direct(book):
    reference = {r["release_date"]:r for r in read(f"{EXIT_STEM}_trades") if r["method"] == HOLDING_RULE}
    seen = []
    for r, _, _, expected, settled, calibrated in inputs_with_reference():
        date = r["release_date"]
        row, anchor = book[date], reference[date]
        decision, entry, sigma = (r[k] for k in ("decision", "entry", "volatility_usd"))
        if (not row["cutoff"] < row["known"] <= decision < entry or row["actual"] != r["x"][0]
                or row["near_month"] != r["near_month"] or abs(expected - float(anchor["expected_long_move_usd"])) > 1e-8):
            raise ValueError(f"The standardized observation and archived strategy disagree on {date}")
        z = row["regressor"]
        surprise = float(np.clip(row["actual"] - row[SUPPLEMENT], -2., 2.))
        target = (r["long_gross"] - expected) / sigma
        current = dict(cutoff=decision, exit_at=r["exit_at"], target=target, **dict(zip(MARKET, ((z,), (z,z*surprise)), strict=True)))
        past = [q for q in seen if q["exit_at"] < decision][-HISTORY:]
        seen.append(current)
        index = int(anchor["exit_session_number"]) - 2
        timing = dict(training_at=settled[-1]["exit_at"], calibration_at=calibrated[-1]["exit_at"] if calibrated else None)
        for name in (ORIGINAL, *MARKET):
            delta = 0. if name == ORIGINAL else sigma * fit_correction(past, current, name, "target",
                penalty=(1., 2.)[:len(current[name])], loss_scale=1., bound=.25)
            for suffix, exit_index in (("", 3), ("_conditional_exit", index)):
                result = trade(r, name + suffix, expected + delta, exit_index, **timing)
                if name == ORIGINAL and suffix and int(anchor["net_170_usd"]) != result["net_170_usd"]:
                    raise ValueError(f"The original rule failed to reproduce on {date}")
                yield result | dict(inventory_forecast_kbbl=1000*row[SUPPLEMENT], adjustment_usd=delta, contrast_z=row["contrast_z"],
                    latest_location_training_exit_utc=past[-1]["exit_at"] if past and name != ORIGINAL else None,
                    net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def summarize(rows, trades):
    summary = dict(status="exploratory_on_reviewed_history_after_raw_contrast", phases={})
    for phase in PHASES:
        group = [r for r in rows if r["channel"] == "inventory" and r["study_phase"] == phase]
        panel = {}
        for key, releases in (("all", group), ("same_pass", [r for r in group if r["matched_roofs"]])):
            errors = {m:1000*np.array([abs(r[m]-r["actual"]) for r in releases]) for m in (SUPPLEMENT, MIXED_GAP, INVENTORY, HYBRID)}
            panel[key] = {m:dict(mae_kbbl=float(e.mean()), weeks=len(e), paired_95_kbbl=block_bootstrap_mean(
                e-errors[MIXED_GAP if m in (MIXED_GAP, HYBRID) else SUPPLEMENT])) for m,e in errors.items()}
        book = defaultdict(list)
        for r in trades:
            if r["study_phase"] == phase:
                book[r["method"]].append(r)
        result = {}
        for index in ("", "_conditional_exit"):
            for name, baseline in ((INVENTORY, SUPPLEMENT), (HYBRID, MIXED_GAP), *((m, ORIGINAL) for m in MARKET)):
                g = book[name + index]
                result[name + index] = market_metrics(g) | dict(paired_vs_baseline=paired_evidence(g, book[baseline + index])[0])
        summary["phases"][phase] = dict(inventory=panel, market=result)
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in panel["all"].items()},
              {m:v["net_170_usd"] for m,v in result.items()},flush=True)
    return summary


def main():
    with threadpool_limits(limits=1):
        forecasts = list(inventory())
        book = {r["release_date"]:r for r in forecasts if r["channel"] == "market"}
        decisions = list(direct(book))
        replay = list(replay_weighted_market(book, (SUPPLEMENT, MIXED_GAP, INVENTORY, HYBRID), reference=SUPPLEMENT))
    joined = decisions + [r | dict(adjustment_usd=None, contrast_z=book[r["release_date"]]["contrast_z"], latest_location_training_exit_utc=None)
                          for r in replay]
    aggregate = summarize(forecasts, joined)
    export(f"{STEM}_forecasts", forecasts)
    export(f"{STEM}_trades", joined)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(aggregate, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
