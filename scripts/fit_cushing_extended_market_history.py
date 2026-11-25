"""Carry the actual 2017 CL contracts into causal training and test longer memory."""

import argparse
import hashlib
import json
import math
from bisect import bisect_right
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import tick_price
from audit_cushing_spot_vintages import inspect as inspect_spot
from build_cushing_holdout_market_features import numeric_prices
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import ROOT, write
from fit_cushing_adaptive_nowcast import archived_training_rows, state, weighted_ar
from fit_cushing_cl_conditional_response import conditional_residual, features as response_features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import CARRY, CARRY_CHANGE_5D, HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import RISK_FLOOR_USD, risk_inputs
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_public_state import released_balances
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_sensor_market_nowcast import contract_book, market_features
from fit_cushing_storage_state import COMBINED, regional_adjustment, source_rows as inventory_sources, storage_coordinates
from fit_cushing_terminal_weekly_transfers import LIMITED_MARKET, STEM as TERMINAL
from fit_cushing_volatility_interactions import features as market_features_with_interactions
from fit_cushing_volatility_neighbors import REGIONAL, STEM as WEIGHTED, archived_coordinates, median_residual, volatility_forecast
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from register_cushing_public_fallback import latest_forward_balances

STEM = "cushing_extended_market_history"
BOUNDARY = "2018-01-10"
INVENTORY_MODELS = (LIMITED_MARKET, "long_median", "long_median_and_regional")
MODELS = {
    "optical_reference": (SUPPLEMENT, HISTORY, False),
    "optical_earlier_start": (SUPPLEMENT, HISTORY, True),
    "optical_long_since_2018": (SUPPLEMENT, 312, False),
    "optical_long_history": (SUPPLEMENT, 312, True),
    "public_reference": (PUBLIC, HISTORY, False),
    "public_long_history": (PUBLIC, 312, True),
    "curve_reference": (None, HISTORY, False),
    "curve_long_history": (None, 312, True),
    "inventory_reference": (LIMITED_MARKET, HISTORY, False),
    "inventory_long_calibration": (INVENTORY_MODELS[-1], HISTORY, False),
}
COMPARISONS = {m: f"{m.split('_')[0]}_reference" for m in MODELS}


def early_inventory(clocks):
    hist = []
    for r in archived_training_rows():
        cutoff, known = (datetime.fromisoformat(r[k]) for k in ("forecast_cutoff_utc", "target_known_by_utc"))
        precursors = [q for q in hist if q["known"] < cutoff]
        last = precursors[-1] if precursors else {"label": 0., "level": r["previous_level_million_bbl"]}
        hist.append(dict(cutoff=cutoff, known=known, label=r["target_million_bbl"], level=r["current_level_million_bbl"],
                         state=state(precursors, last["label"], last["level"])))
    source = defaultdict(dict)
    for r in read("eia_control_vintage"):
        if "2017-01-01" <= r["release_date"] < BOUNDARY and r["metric"] == "cushing_crude":
            source[r["release_date"]][r["report_date"]] = r
    for issue, rows in sorted(source.items()):
        c = clocks[issue]
        week, previous = c["latest_week_in_issue"], c["oldest_week_in_issue"]
        fresh, prior = rows[week], rows[previous]
        if fresh["source_url"] != prior["source_url"] or fresh["source_kind"] != "archived_csv":
            raise ValueError(f"Inconsistent 2017 first-release source on {issue}")
        path = ROOT / "data/raw/eia/wpsr" / f"{issue.replace('-', '_')}_table4.csv"
        if path.with_suffix(".source").read_text().strip() != fresh["source_url"]:
            raise ValueError(f"Missing original 2017 table on {issue}")
        cutoff, known = (datetime.fromisoformat(c[k]) for k in
                         ("conservative_prerelease_before_utc", "conservative_postrelease_from_utc"))
        if not cutoff < known:
            raise ValueError(f"Invalid 2017 publication clock on {issue}")
        settled = [r for r in hist if r["known"] < cutoff]
        current, previous_level = (float(q["value"]) / 1000 for q in (fresh, prior))
        row = dict(release_date=issue, week_ending=week, study_phase="archive_2017", cutoff=cutoff, known=known,
                   actual=round(current - previous_level, 3), source_url=fresh["source_url"],
                   source_file=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        # Train the same small autoregression on the original inventory issues.
        # This is a public proxy for 2017, where the incumbent has no signals.
        last = settled[-1]
        row["state"] = state(settled, last["label"], last["level"])
        expected = weighted_ar(settled[-HISTORY:], row, 26., False)
        row.update(label=row["actual"], level=current)
        hist.append(row)
        yield {k: v for k, v in row.items() if k not in ("state", "label", "level")} | {
            PUBLIC: expected, SUPPLEMENT: expected, "prior_change": last["label"], "latest_inventory_training_utc": last["known"]}


def archived_candidates(forecasts, clocks):
    book = {r["release_date"]: r for r in forecasts}
    days, curve = contract_book()
    by_issue = defaultdict(list)
    release_days = sorted(clocks)
    for b in read("wti_cushing_event_bars"):
        if b["release_date"] in book and int(b["session_number"]) >= 2:
            by_issue[b["release_date"]].append(b)
    for w in read("wti_cushing_event_windows"):
        issue = w["release_date"]
        if issue not in book or w["phase_ok"] != "True":
            continue
        f = book[issue]
        decision = datetime.fromisoformat(w["decision_at_utc"])
        vector, sessions = market_features(days, curve, (f["cutoff"],))
        bars = sorted(by_issue[issue], key=lambda b: (int(b["session_number"]), b["leg"] != "near"))
        leg_keys = [(int(b["session_number"]), b["leg"], b["delivery_month"]) for b in bars]
        if leg_keys != [(i, k, w[f"{k}_month"]) for i in range(2, 6) for k in ("near", "far")]:
            raise ValueError(f"Different dated contract legs on {issue}")
        stamps = [datetime.fromisoformat(b["settlement_reference_time"]) for b in bars]
        source = next((s for s in ("archive", "ibkr") if all(b[f"{s}_present"] == "True" for b in bars)), None)
        if (stamps[::2] != stamps[1::2] or not f["cutoff"] < f["known"] <= decision < stamps[0] < stamps[-1]
                or not sessions[0] < issue or f["week_ending"] != w["week_ending"]):
            raise ValueError(f"Invalid archived trade clock on {issue}")
        next_issue = release_days[bisect_right(release_days, issue)]
        next_clock = clocks[next_issue]
        event = next_clock["scheduled_table_release_at_utc"] or next_clock["conservative_prerelease_before_utc"]
        carry, change = map(float, vector[[CARRY, CARRY_CHANGE_5D]])
        surprise = f["actual"] - f[PUBLIC]
        yield dict(w, release_date=issue, study_phase="archive_2017", decision=decision, entry=stamps[0], exit_at=stamps[-1],
                   stamps=stamps[::2], source=source, points=bars, next_issue=next_issue, next_report_clock=datetime.fromisoformat(event),
                   next_report_clock_basis=next_clock["timing_basis"], x=(f["actual"], surprise, carry, change, surprise*carry, 0., 0.))


def archived_windows(forecasts, clocks, first_base_entry):
    candidates = list(risk_inputs(list(archived_candidates(forecasts, clocks)), allow_missing=True))
    through = max(b["session_date"] for r in candidates for b in r["points"])
    with closing(read_only(SOURCE)) as db:
        price = numeric_prices(db, ("archive", "ibkr"), through=through)
    outcomes, coverage = [], []
    last = None
    for r in candidates:
        source = r["source"]
        reason = ("missing_marks" if not source or r["volatility_usd"] is None else
                  "overlaps_incumbent" if r["exit_at"] >= first_base_entry else
                  "overlaps_archive" if last is not None and r["entry"] <= last else "admitted")
        coverage.append({k: r[k] for k in ("release_date", "near_month", "far_month", "decision", "entry", "exit_at")} |
                        dict(status=reason, source=source, risk_source=r.get("volatility_source")))
        if reason != "admitted":
            continue
        last = r["exit_at"]
        leg = [tick_price(price(source, b["session_date"], b["delivery_month"], b[f"{source}_reference"])) for b in r["points"]]
        path = [10 * (a - b - leg[0] + leg[1]) for a, b in zip(leg[::2], leg[1::2], strict=True)]
        sigma = max(RISK_FLOOR_USD, r["volatility_usd"] * math.sqrt(.75))
        outcomes.append(r | dict(spread_path_usd=path, long_gross=path[-1], volatility_usd=sigma, alternate_source=None,
                                 alternate_long_gross_usd=None))
    return outcomes, coverage


def baseline_inventory():
    for issue, r in sorted(indexed(f"{OUTER}_forecasts").items()):
        cutoff, known = (datetime.fromisoformat(r[k]) for k in ("cutoff_utc", "known_utc"))
        prior = r[f"{SUPPLEMENT}_latest_label_utc"]
        if not cutoff < known or prior and datetime.fromisoformat(prior) >= cutoff:
            raise ValueError(f"The historical forecast was unavailable on {issue}")
        yield issue, dict(cutoff=cutoff, known=known, actual=float(r["actual"]), **{m: float(r[m]) for m in (PUBLIC, SUPPLEMENT)})


def inventory_memory(*, before=None, state=None):
    anchor = indexed(f"{TERMINAL}_forecasts")
    archived = indexed(f"{WEIGHTED}_forecasts")
    spot = indexed("cushing_spot_vintages")
    history, scored = [], {m: [] for m in INVENTORY_MODELS}
    for r in inventory_sources():
        issue, cutoff, known = (r[k] for k in ("release_date", "cutoff", "known"))
        if before is not None and known >= before:
            break
        a, s, spot_row = anchor[issue], archived[issue], spot[issue]
        if (not cutoff < known or any(datetime.fromisoformat(q["cutoff"]) != cutoff for q in (a, s, spot_row))
                or not spot_row["archived_reference_date"] <= spot_row["vintage_date"] < issue
                or any(datetime.fromisoformat(q["known"]) != known or float(q["actual"]) != r["actual"] for q in (a, s))):
            raise ValueError(f"Inconsistent archived inventory panel on {issue}")
        for q in (a, s):
            if any(datetime.fromisoformat(v) >= cutoff for k, v in q.items()
                   if v and k in ("latest_training_label_utc", "latest_inventory_training_utc", "latest_state_error_known_utc")):
                raise ValueError(f"Unavailable archived inventory calibration on {issue}")
        past = [q for q in history if q["known"] < cutoff][-312:]
        coord = archived_coordinates(spot_row, r["state"])
        current = r | dict(inventory_coordinates=coord)
        corrections = (regional_adjustment(past[-HISTORY:], r), regional_adjustment(past, r))
        regional = tuple(r[COMBINED] + delta for delta in corrections)
        if not math.isclose(regional[0], float(s[REGIONAL]), abs_tol=1e-10):
            raise ValueError(f"The archived regional calibration did not reproduce on {issue}")
        next_rows = [current | {REGIONAL: x} for x in (regional[0], *regional)]
        previous = [[p for p in scored[m] if p["known"] < cutoff][-n:]
                    for m, n in zip(INVENTORY_MODELS, (HISTORY, 312, 312), strict=True)]
        point = [q[REGIONAL] + median_residual(pool, q, 2) for q, pool in zip(next_rows, previous, strict=True)]
        if not math.isclose(point[0], float(s["market_error_median"]), abs_tol=1e-10):
            raise ValueError(f"The archived inventory median did not reproduce on {issue}")
        # Hold the original terminal and optics components fixed when
        # changing the training horizon of the public calibration.
        terminal = float(a[LIMITED_MARKET]) - point[0]
        history.append(r)
        for m, q in zip(INVENTORY_MODELS, next_rows, strict=True):
            scored[m].append(q)
        yield {k: r[k] for k in ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual")} | dict(
            **{m: x + terminal for m, x in zip(INVENTORY_MODELS, point, strict=True)},
            available_training_weeks=len(past), latest_training_release=max((q["known"] for q in past), default=None))
    if state is not None:
        state.update(history=history, scored=scored)


def inventory_metrics(rows):
    result = {}
    for phase in PHASES:
        active = [r for r in rows if r["study_phase"] == phase]
        errors = {m: np.array([1000 * abs(r["actual"] - r[m]) for r in active]) for m in INVENTORY_MODELS}
        result[phase] = dict(releases=len(active), models={m: dict(mae_kbbl=float(e.mean()),
            paired_delta_vs_reference_95_kbbl=block_bootstrap_mean(e - errors[LIMITED_MARKET])) for m, e in errors.items()})
    return result


def replay(windows, forecasts):
    seen = {m: [] for m in MODELS}
    anchor = {(r["release_date"], r["inventory_model"]): r for r in read(f"{WEIGHTED}_trades")
              if r["method"] in (f"{PUBLIC}:volatility", f"{SUPPLEMENT}:volatility")}
    for r in windows:
        issue, decision = r["release_date"], r["decision"]
        f = forecasts[issue]
        if not f["cutoff"] < f["known"] <= decision < r["entry"] < r["exit_at"] or f["actual"] != r["x"][0]:
            raise ValueError(f"The first release does not match the trading case on {issue}")
        for m, (source, limit, admit_2017) in MODELS.items():
            if issue < BOUNDARY and not admit_2017:
                continue
            if source is None:
                design = market_features_with_interactions(r)[5]
                context = response_features(r["x"][2], r["volatility_usd"])
            else:
                design = replace_expectation(r, f[source])
                context = response_features(r["x"][2], r["volatility_usd"], f["actual"] - f[source])
            history = seen[m]
            settled = [q for q in history if q["exit_at"] < decision][-limit:]
            current = r | dict(x=design, response_features=context)
            history.append(current)
            if len(settled) < MIN_WINDOWS:
                continue
            point = volatility_forecast(settled, current)
            # Keep the residual stage on the same 156-window policy in
            # every arm. The experiment changes the point model's memory.
            local = [q for q in settled if "base_estimate" in q][-HISTORY:]
            residual, _ = conditional_residual(local, current) if len(local) >= MIN_WINDOWS else (0., 0.)
            current["base_estimate"] = point
            estimate = point + residual
            result = trade(r, m, estimate, 3, training_at=settled[-1]["exit_at"],
                           calibration_at=local[-1]["exit_at"] if local else None)
            if m in ("public_reference", "optical_reference"):
                a = anchor[issue, source]
                if (any(a[k] != result[k] for k in ("near_month", "far_month", "entry_at_utc", "exit_at_utc"))
                        or not math.isclose(float(a["expected_long_move_usd"]), estimate, abs_tol=1e-8)):
                    raise ValueError(f"The incumbent forecast did not reproduce on {issue}/{m}")
            yield result | dict(training_windows=len(settled), early_training_windows=sum(q["release_date"] < BOUNDARY for q in settled),
                earliest_training_release=settled[0]["release_date"], realized_long_move_usd=r["long_gross"],
                net_340_usd=result["gross_usd"] - 340*abs(result["near_side"]))


def evaluate(rows):
    indexed_rows = {(r["release_date"], r["method"]): r for r in rows}
    if len(indexed_rows) != len(rows):
        raise ValueError("Repeated historical market decision")
    results = {}
    for phase in PHASES:
        phases = [r for r in rows if r["study_phase"] == phase]
        policies = {}
        for m in MODELS:
            subset = [r for r in phases if r["method"] == m]
            matched = [(r, indexed_rows[r["release_date"], COMPARISONS[m]]) for r in subset
                       if (r["release_date"], COMPARISONS[m]) in indexed_rows]
            pnl = [r["net_170_usd"] - ref["net_170_usd"] for r, ref in matched]
            observed = np.array([r["realized_long_move_usd"] for r, _ in matched])
            error, base_error = (abs(np.array([float(r[index]["expected_long_move_usd"]) for r in matched]) - observed)
                                 for index in (0, 1))
            policies[m] = robustness(subset) | daily_risk(subset) | dict(
                net_340_usd=sum(r["net_340_usd"] for r in subset), earlier_windows=len(subset)-len(matched),
                paired_windows=len(matched), mae_usd=float(error.mean()), mae_reference_usd=float(base_error.mean()),
                paired_mae_delta_95_usd=block_bootstrap_mean(error - base_error), paired_net_delta_usd=sum(pnl),
                paired_net_delta_95_usd=block_bootstrap_mean(pnl),
                paired_net_delta_340_usd=sum(r["net_340_usd"] - q["net_340_usd"] for r, q in matched),
                windows_with_2017_training=sum(r["early_training_windows"] > 0 for r in subset))
        results[phase] = policies
    a = {r["release_date"]: r for r in rows if r["study_phase"] == "holdout" and r["method"] == "optical_long_history"}
    b = {r["release_date"]: r for r in rows if r["study_phase"] == "holdout" and r["method"] == "optical_long_since_2018"}
    if a.keys() != b.keys():
        raise ValueError("The two long training panels use different market calendars")
    deltas = [q["net_170_usd"] - b[issue]["net_170_usd"] for issue, q in a.items()]
    results["archive_increment"] = dict(paired_windows=len(deltas), net_delta_add_2017_usd=sum(deltas),
        paired_delta_95_usd=block_bootstrap_mean(deltas), changed_sides=sum(q["near_side"] != b[i]["near_side"] for i, q in a.items()))
    return results


def register_forecast(issue, *, scoring):
    path = DERIVED / f"{STEM}_{issue}_registration.json"
    main_path = DERIVED / f"cushing_forward_leading_models_{issue}_registration.json"
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    main_record, original = (json.loads(p.read_bytes()) for p in (main_path, original_path))
    cutoff, deadline = (datetime.fromisoformat(main_record[k]) for k in LINEAGE[-2:])
    if any(main_record[k] != original[k] for k in LINEAGE):
        raise ValueError("The frozen inventory control uses a different release")
    checked_hashes(main_record["sources_sha256"])
    if not path.exists():
        if scoring or not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("Register the extended inventory forecast before release")
        target_week = main_record["week_ending"]
        forward, evidence = latest_forward_balances(cutoff, target_week, original["prior_release"])
        known_balances = [q for q in released_balances() if q["known"] < cutoff and q["week"] < target_week]
        if not forward or known_balances[-1]["week"] >= forward[0]["week"]:
            raise ValueError("The later regional balance duplicates a training release")
        coords = storage_coordinates(known_balances + forward)
        spot = inspect_spot(dict(release_date=issue, study_phase="forward", cutoff_utc=cutoff.isoformat(),
                                 spot_reference_day=main_record["archived_spot_reference_date"]))
        if (spot["vintage_date"], spot["archived_reference_date"]) != (main_record["archived_spot_vintage"],
                                                                    main_record["archived_spot_reference_date"]):
            raise ValueError("The market archive differs from the control")
        for key in ("wti", "brent"):
            evidence[spot[f"{key}_path"]] = spot[f"{key}_sha256"]
        current = coords | dict(cutoff=cutoff, inventory_coordinates=archived_coordinates(spot, coords["state"]))
        past = {}
        with threadpool_limits(limits=1):
            for _ in inventory_memory(before=cutoff, state=past):
                pass
            sizes = (HISTORY, 312, 312)
            history = past["history"]
            original_regional = regional_adjustment(history[-HISTORY:], current)
            regional = (original_regional, original_regional, regional_adjustment(history[-312:], current))
            median = [median_residual(past["scored"][m][-size:], current, 2)
                      for m, size in zip(INVENTORY_MODELS, sizes, strict=True)]
        anchor = main_record["inventory_components_kbbl"]
        original_adjustment = (anchor["regional_levels"], anchor["basis_error_median"])
        if (not math.isclose(1000 * regional[0], original_adjustment[0], abs_tol=1e-8)
                or not math.isclose(1000 * median[0], original_adjustment[1], abs_tol=1e-8)):
            raise ValueError("The existing 156-release forecast did not reproduce")
        optical = sum(anchor[k] for k in ("optical_base", "additional_passes", "terminal_limited_coverage"))
        estimates = {m: optical + 1000 * (a + b) for m, a, b in zip(INVENTORY_MODELS, regional, median, strict=True)}
        if not math.isclose(estimates[LIMITED_MARKET], main_record["estimates_kbbl"][LIMITED_MARKET], abs_tol=1e-8):
            raise ValueError("The forward control point estimate differs from the registered forecast")
        wanted = (main_path, original_path, Path(__file__).resolve(), DERIVED / f"{STEM}_inventory.csv",
                  DERIVED / f"{WEIGHTED}_forecasts.csv", DERIVED / f"{TERMINAL}_forecasts.csv", DERIVED / "cushing_spot_vintages.csv",
                  DERIVED / "cushing_late_pair_recovery_forecasts.csv")
        evidence.update({str(p.relative_to(ROOT)): digest(p) for p in wanted})
        checked_hashes(evidence)
        stamp = datetime.now(timezone.utc)
        if not history[-1]["known"] < cutoff <= stamp < deadline:
            raise ValueError("The extended calibration missed the registration deadline")
        freeze(path, {k: main_record[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), estimates_kbbl=estimates,
            regional_adjustment_kbbl={m: 1000 * a for m, a in zip(INVENTORY_MODELS, regional, strict=True)},
            market_median_adjustment_kbbl={m: 1000 * a for m, a in zip(INVENTORY_MODELS, median, strict=True)},
            latest_training_label_utc=history[-1]["known"].isoformat(), measured_roofs=main_record["measured_roofs"],
            training_windows=dict(zip(INVENTORY_MODELS, sizes, strict=True)), sources_sha256=evidence))
    record = json.loads(path.read_bytes())
    if (any(record[k] != main_record[k] for k in LINEAGE) or not datetime.fromisoformat(record["latest_training_label_utc"])
            < cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline):
        raise ValueError("The archived forward forecast is inconsistent")
    checked_hashes(record["sources_sha256"])
    if not scoring:
        return record
    outcome = score_release(original_path, original)
    result = dict(release_date=issue, status=outcome["status"], estimates_kbbl=record["estimates_kbbl"], registration_sha256=digest(path))
    if outcome["status"] == "scored_first_release":
        if datetime.fromisoformat(record["recorded_at_utc"]) >= deadline:
            raise ValueError("A late forecast cannot be scored")
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_error_kbbl={m: abs(actual - v) for m, v in record["estimates_kbbl"].items()})
    write(path.with_name(f"{STEM}_{issue}_score.json"), (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


def main():
    clocks = indexed("eia_release_clock")
    # Preserve the following-report clock on the archived windows and
    # verify later windows from the original calendar rather than rebuilding.
    earlier = list(early_inventory(clocks))
    primary = [r for r, _ in inputs()]
    archive, coverage = archived_windows(earlier, clocks, primary[0]["entry"])
    forecasts = dict(baseline_inventory()) | {r["release_date"]: r for r in earlier}
    with threadpool_limits(limits=1):
        inventory = list(inventory_memory())
        for r in inventory:
            f = forecasts[r["release_date"]]
            if any(f[k] != r[k] for k in ("actual", "cutoff", "known")):
                raise ValueError(f"Forecast arms do not match on {r['release_date']}")
            f.update({m: r[m] for m in INVENTORY_MODELS})
        rows = list(replay(archive + primary, forecasts))
    report = dict(status="exploratory_earlier_calibration_and_training_memory", early_eia_releases=len(earlier),
                  early_candidate_windows=len(coverage), early_admitted_windows=len(archive),
                  early_public_mae_kbbl=1000 * float(np.mean([abs(r[PUBLIC] - r["actual"]) for r in earlier])),
                  early_prior_mae_kbbl=1000 * float(np.mean([abs(r["prior_change"] - r["actual"]) for r in earlier])),
                  phases=evaluate(rows), inventory=inventory_metrics(inventory))
    export(f"{STEM}_early_inventory", earlier)
    export(f"{STEM}_early_coverage", coverage)
    export(f"{STEM}_inventory", inventory)
    export(f"{STEM}_trades", rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    print("archive", len(earlier), len(coverage), len(archive))
    for phase in PHASES:
        models = report["phases"][phase]
        print(phase, {m: (v["trades"], v["net_170_usd"], v["net_340_usd"], round(v["mae_usd"], 2)) for m, v in models.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.score and not args.release:
        parser.error("--score requires --release")
    if args.release:
        result = register_forecast(args.release, scoring=args.score)
        print(json.dumps({k: v for k, v in result.items() if k != "sources_sha256"}, indent=2))
    else:
        main()
