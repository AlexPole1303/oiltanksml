"""Test the date-specific direction of optical revisions and learn separate gains."""

import argparse
import json
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_cl_entry_margin import inputs_with_reference
from build_s1_reference_matches import DERIVED, export
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import trade
from fit_cushing_cl_regime_strategy import MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_inventory_seasonal_residual import BASE, EXTENDED, inputs
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_value import STEM as VALUE
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_robust_stack import indexed
from register_cushing_leading_models import LINEAGE, STEM as LEADING, checked_hashes, prediction
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_optical_direction"
INCUMBENT = "same_week"
POOLED = "pooled_gain"
ADAPTIVE = "agreement_gain"
MODES = (INCUMBENT, "no_revision", "reversed_direction", "earlier_direction", POOLED, ADAPTIVE)
COSTS = (170, 340)


def gain(sample, cutoff, prior):
    """Learn a nonnegative scale with a robust loss and a prior on the scale."""
    if len(sample) < MIN_WINDOWS:
        return prior
    update = np.array([p["update"] for p in sample])
    target = np.array([p["actual"] - p["base"] for p in sample])
    age = np.array([(cutoff - p["known"]).total_seconds() / 604800 for p in sample])
    recency = np.exp2(-age / 104)
    penalty = .04 * recency.sum()
    value = prior
    for _ in range(10):
        weight = recency / np.hypot(target - value * update, .25)
        value = float(np.clip((weight @ (update * target) + penalty * prior)
                              / (weight @ np.square(update) + penalty), 0., 2.))
    return value


def variants(row, history, earlier):
    cutoff, base, update = (row[k] for k in ("cutoff", "base", "update"))
    if not update:
        return dict.fromkeys(MODES, row[INCUMBENT]), (1., 1.), 0, 0, None
    data = [p for p in history if p["known"] < cutoff and p["update"]][-156:]
    same_sign = (base * update > 0)
    matching = [p for p in data if (p["base"] * p["update"] > 0) == same_sign]
    pooled = gain(data, cutoff, 1.)
    scoped = gain(matching, cutoff, pooled)
    lagged = math.copysign(abs(update), earlier["update"]) if earlier and earlier["update"] else None
    values = (row[INCUMBENT], base, base - update, base + lagged if lagged is not None else row[INCUMBENT],
              base + pooled * update, base + scoped * update)
    return dict(zip(MODES, values, strict=True)), (pooled, scoped), len(data), len(matching), data[-1]["known"] if data else None


def inventory():
    optics, values = (indexed(f"{stem}_forecasts") for stem in (OUTER, VALUE))
    history = {"inventory": [], "trading": []}
    by_week = {}
    for source in inputs():
        issue, week = (source[k] for k in ("release_date", "week_ending"))
        optical, value = optics[issue], values[issue]
        if (any(source[k] != value[k] for k in ("week_ending", "study_phase"))
                or any(datetime.fromisoformat(value[k]) != source[k] for k in ("cutoff", "known"))
                or abs(float(value["actual"]) - source["actual"]) > 1e-10
                or source[SUPPLEMENT] != float(optical[SUPPLEMENT])):
            raise ValueError(f"Inputs do not match on {issue}")
        prior_week = (date.fromisoformat(week) - timedelta(weeks=4)).isoformat()
        earlier = by_week.get(prior_week, {})
        if earlier and any(p["cutoff"] >= source["cutoff"] for p in earlier.values()):
            raise ValueError(f"A reference week violates the cutoff on {issue}")
        estimate = source[BASE]
        update = float(value["optical_update"])
        public = float(optical[PUBLIC])
        targets = {"inventory": (estimate - update, estimate), "trading": (public, source[SUPPLEMENT])}
        base_rows = {}
        for channel, (base, original) in targets.items():
            row = source | dict(channel=channel, base=base, same_week=original, update=original - base)
            earlier_row = earlier.get(channel)
            estimates, (pooled, scoped), n, subgroup, latest = variants(row, history[channel], earlier_row)
            base_rows[channel] = row
            history[channel].append(row)
            yield row | estimates | dict(pooled_multiplier=pooled, agreement_multiplier=scoped,
                training_weeks=n, agreement_training_weeks=subgroup, latest_training_at=latest,
                earlier_week=prior_week if earlier_row and earlier_row["update"] and row["update"] else None)
        by_week[week] = base_rows


def market(forecasts):
    references = {r["release_date"]: r for r in forecasts if r["channel"] == "trading"}
    for case, current, archive, base, settled, calibration in inputs_with_reference():
        issue = case["release_date"]
        f = references[issue]
        if (not f["cutoff"] < f["known"] <= case["decision"] or current[SUPPLEMENT] != f[INCUMBENT]
                or f["actual"] != current["x"][0] or f["study_phase"] != case["study_phase"]):
            raise ValueError(f"Archived optical forecast differs on {issue}")
        for method in MODES:
            expected = f[method]
            value = base
            if expected != f[INCUMBENT]:
                value = sum(prediction(settled, calibration, current | {SUPPLEMENT: expected}))
            result = trade(case, method, value, 3, training_at=settled[-1]["exit_at"],
                           calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if method == INCUMBENT and any(int(archive[k]) != result[k] for k in ("near_side", "net_170_usd")):
                raise ValueError(f"Original decision changed on {issue}")
            yield result | dict(earlier_week=f["earlier_week"], sample_with_optical_revision=bool(f["update"]),
                expected_inventory_kbbl=1000 * expected, realized_long_move_usd=case["long_gross"],
                net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def inventory_metrics(rows):
    losses = {m: np.array([1000 * abs(r["actual"] - r[m]) for r in rows]) for m in MODES}
    return {m: dict(releases=len(rows), mae_kbbl=float(v.mean()),
                    delta_vs_same_week_95_kbbl=block_bootstrap_mean(v - losses[INCUMBENT]))
            for m, v in losses.items()} if rows else {}


def market_metrics(rows):
    by_method = {m: [r for r in rows if r["method"] == m] for m in MODES}
    reference = by_method[INCUMBENT]
    result = {}
    for method, group in by_method.items():
        if not group:
            continue
        paired = list(zip(group, reference, strict=True))
        if any(a["release_date"] != b["release_date"] for a, b in paired):
            raise ValueError("The market comparison used a different release calendar")
        changed = [dict(release_date=a["release_date"], near_side=a["near_side"], reference_side=b["near_side"],
                        delta_net_170_usd=a["net_170_usd"] - b["net_170_usd"])
                   for a, b in paired if a["near_side"] != b["near_side"]]
        result[method] = robustness(group) | daily_risk(group) | dict(
            net_340_usd=sum(q["net_340_usd"] for q in group), changed=changed,
            delta_vs_same_week_95_usd=block_bootstrap_mean([a["net_170_usd"] - b["net_170_usd"] for a, b in paired]),
            mean_absolute_error_usd=float(np.mean([abs(q["expected_long_move_usd"] - q["realized_long_move_usd"])
                                                   for q in group])))
    return result


def sign_stress(weeks, current, reversed_values):
    # Conditional sensitivity with fixed magnitudes and fixed fitted histories.
    # Do not interpret the frequencies as p-values after model selection.
    dates = np.array([(date.fromisoformat(w) - date.fromisoformat(weeks[0])).days // 7 for w in weeks])
    alternative = np.asarray(reversed_values) - np.asarray(current)
    result = {}
    for span in (4, 13):
        block_delta = np.bincount(dates // span, weights=alternative)
        active = block_delta[block_delta != 0]
        if len(active) <= 16:
            flips = ((np.arange(2 ** len(active))[:, None] >> np.arange(len(active))) & 1)
            method = "enumerated"
        else:
            flips = np.random.default_rng(67412 + span).integers(0, 2, size=(16000, len(active)))
            method = "simulated"
        totals = math.fsum(current) + flips @ active
        result[str(span)] = dict(method=method, active_blocks=len(active), scenarios=len(totals),
                                 total_quantiles=np.quantile(totals, [.05, .5, .95]).tolist(),
                                 observed_total=math.fsum(current), minimum=float(totals.min()), maximum=float(totals.max()))
    return result


def stress_report(forecasts, trades):
    inventory = [r for r in forecasts if r["study_phase"] == "holdout" and r["channel"] == "inventory"]
    all_weeks = {r["release_date"]: r["week_ending"] for r in inventory}
    book = {m: {r["release_date"]: r for r in trades if r["study_phase"] == "holdout" and r["method"] == m}
            for m in (INCUMBENT, "reversed_direction")}
    outputs = {"inventory_total_absolute_error_kbbl": sign_stress(
        list(all_weeks.values()), *([1000 * abs(r[m] - r["actual"]) for r in inventory] for m in book))}
    for cost in COSTS:
        outputs[f"net_{cost}_usd"] = sign_stress(list(all_weeks.values()), *(
            [book[m].get(issue, {}).get(f"net_{cost}_usd", 0) for issue in all_weeks] for m in book))
    return outputs


def forward(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    leader_path = DERIVED / f"{LEADING}_{issue}_registration.json"
    extended_path = DERIVED / f"{EXTENDED}_{issue}_registration.json"
    original, leader, extended = (json.loads(p.read_bytes()) for p in (original_path, leader_path, extended_path))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    for q in (leader, extended):
        checked_hashes(q["sources_sha256"])
        if any(q[k] != original[k] for k in LINEAGE) or q["measured_roofs"] != leader["measured_roofs"]:
            raise ValueError("The prerelease records do not describe the same optical panel")
    if not path.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The comparison must be registered before publication")
        measurement = leader["inventory_components_kbbl"]
        if measurement["additional_passes"] or measurement["terminal_limited_coverage"]:
            raise ValueError("The optical component has additional terms. Recompute the optical-free estimate.")
        public, optical = (leader["estimates_kbbl"][m] / 1000 for m in (PUBLIC, SUPPLEMENT))
        inventory_total = extended["estimates_kbbl"][BASE] / 1000
        prior_week = (date.fromisoformat(original["week_ending"]) - timedelta(weeks=4)).isoformat()
        past = list(inventory())
        channels = {}
        for name, initial, point in (("inventory", inventory_total - (measurement["optical_base"] / 1000 - public), inventory_total),
                                     ("trading", public, optical)):
            history = [r for r in past if r["channel"] == name and r["known"] < cutoff]
            earlier = next((r for r in history if r["week_ending"] == prior_week), None)
            if not earlier:
                raise ValueError("The four-week baseline is missing from the dated history")
            values, gains, n, subgroup, latest = variants(dict(cutoff=cutoff, base=initial, same_week=point,
                                                               update=point - initial), history, earlier)
            channels[name] = dict(estimates_kbbl={m: 1000 * v for m, v in values.items()}, pooled_multiplier=gains[0],
                                  agreement_multiplier=gains[1], training_weeks=n, agreement_training_weeks=subgroup,
                                  latest_training_at=latest.isoformat() if latest else None,
                                  earlier_week=prior_week, earlier_update_kbbl=1000 * earlier["update"])
        sources = (original_path, leader_path, extended_path, Path(__file__).resolve(),
                   ROOT / "scripts/fit_cushing_inventory_seasonal_residual.py", ROOT / "scripts/fit_cushing_optical_value.py",
                   DERIVED / "cushing_terminal_consensus_forecasts.csv",
                   *(DERIVED / f"{name}.csv" for name in (f"{OUTER}_forecasts", f"{VALUE}_forecasts", f"{EXTENDED}_inventory")))
        stamp = datetime.now(timezone.utc)
        if not cutoff <= stamp < deadline:
            raise ValueError("The prerelease comparison missed its deadline")
        freeze(path, {k: original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), channels=channels,
               measured_roofs=leader["measured_roofs"], sources_sha256={str(q.relative_to(ROOT)): digest(q) for q in sources}))
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    if (any(record[k] != original[k] for k in LINEAGE)
            or not cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline
            or record["measured_roofs"] != leader["measured_roofs"]
            or set(record["channels"]) != {"inventory", "trading"}
            or any(set(q["estimates_kbbl"]) != set(MODES) for q in record["channels"].values())
            or any(q["latest_training_at"] and datetime.fromisoformat(q["latest_training_at"]) >= cutoff
                   for q in record["channels"].values())):
        raise ValueError("The prerelease comparison failed its cutoff checks")
    outcome = score_release(original_path, original)
    result = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path), channels=record["channels"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_error_kbbl={k: {m: abs(v - actual) for m, v in q["estimates_kbbl"].items()}
                                           for k, q in record["channels"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(result["status"], {k: {m: round(v, 2) for m, v in q["estimates_kbbl"].items()}
                             for k, q in record["channels"].items()})


def main():
    with threadpool_limits(limits=1):
        forecasts = list(inventory())
        trades = list(market(forecasts))
    report = dict(status="exploratory_on_reviewed_history", lag_calendar_weeks=4, phases={},
                  hypothetical_block_sign_reversal=stress_report(forecasts, trades))
    for phase in PHASES:
        metrics = {}
        for channel in ("inventory", "trading"):
            rows = [r for r in forecasts if (r["study_phase"], r["channel"]) == (phase, channel)]
            metrics[channel] = {"all": inventory_metrics(rows), "current_and_lagged": inventory_metrics(
                [r for r in rows if r["earlier_week"]])}
        rows = [r for r in trades if r["study_phase"] == phase]
        metrics["market"] = {"all": market_metrics(rows), "current_and_lagged": market_metrics(
            [r for r in rows if r["earlier_week"]])}
        report["phases"][phase] = metrics
        print(phase, {m: (round(v["mae_kbbl"], 2), metrics["market"]["all"][m]["net_170_usd"])
                      for m, v in metrics["inventory"]["all"].items()})
    export(f"{STEM}_forecasts", forecasts)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    if args.release:
        forward(args.release.isoformat())
    else:
        main()
