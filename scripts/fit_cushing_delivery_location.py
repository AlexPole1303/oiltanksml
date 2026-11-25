"""Compare the optical signal at named CL delivery hubs with synchronous nearby storage."""

import argparse
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from audit_cushing_cl_entry_margin import inputs_with_reference
from audit_cushing_cl_equal_model_risk import paired_evidence
from backtest_wti_cushing_spread import full_sessions
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import ROOT, write
from fit_cushing_cl_curve_shape import MIXED_GAP, STEM as GAP_STEM, market_metrics
from fit_cushing_cl_exit_horizon import trade
from fit_cushing_cl_extension_segments import STEM as EXIT_STEM
from fit_cushing_cl_regime_strategy import HISTORY
from fit_cushing_intraweek import time_covariance
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import inputs, pass_id
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import SCENE_SIGMA_M, scene_design
from fit_cushing_storage_state import fit_correction
from fit_cushing_terminal_balance import mapped_areas
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from replay_cushing_weighted_inventory_market import HOLDING_RULE, replay_weighted_market

STEM = "cushing_delivery_location"
ORIGINAL = "original_location_control"
LOCATION = "named_delivery_location"
INTERACTION = "named_delivery_and_surprise"
LAG = "prior_location_control"
INVENTORY = "inventory_location"
HYBRID = "hybrid_location"
MARKET = (ORIGINAL, LOCATION, INTERACTION, LAG)
FORWARD_POLICY = "same_pass_named_hub_contrast"


def named_sites(issue, month):
    # CME SER-8872 dates the change and specifies the first affected maturity.
    sites = {"enbridge_cushing", "enterprise_cushing"}
    if issue >= "2021-11-08" and month >= "202202":
        sites.add("plains_cushing")
    return sites


def location(row, roster, month):
    sites = named_sites(row["release_date"], month)
    buckets = defaultdict(list)
    for q in row["observed"]:
        if datetime.fromisoformat(q["available_utc"]) >= row["cutoff"] or q["terminal_id"] not in roster:
            raise ValueError(f"Undated or unmatched reservoir on {row['release_date']}")
        pair = tuple(pass_id(q[k]) for k in ("scene_before", "scene_after"))
        buckets[pair].append(q)
    common = [group for group in buckets.values() if {q["terminal_id"] in sites for q in group} == {False, True}]
    sample = [q for group in common for q in group]
    core = np.array([q["terminal_id"] in sites for q in sample], dtype=bool)
    share = math.fsum(roster.get(k, 0.) for k in sites) / row["hub_area"]
    if not math.isclose(math.fsum(roster.values()), row["hub_area"], rel_tol=1e-10):
        raise ValueError(f"Mismatched contemporaneous terminal area on {row['release_date']}")
    output = dict(near_month=month, named_hubs="/".join(sorted(sites)), named_hub_area_share=share,
                  matched_pass_pairs=len(common), matched_roofs=len(sample), named_hub_roofs=int(core.sum()),
                  matched_terminals=len({q["terminal_id"] for q in sample}), contrast_information=0., contrast_million_bbl=0.,
                  raw_contrast_million_bbl=None, latest_scene_ready_utc=None)
    if not sample:
        return output
    cohort = row | dict(observed=sample, area_fraction=math.fsum(q["area"] for q in sample) / row["hub_area"])
    model = prepared(cohort, row["public"], any("post_fraction" in q for q in sample))
    s = model["sensitivity"]
    # Apply the dated hub shares to a zero-sum transfer between location
    # groups, then eliminate the nuisance parameter for all-Cushing change.
    locations = s * (core.astype(float) - share)
    scene = scene_design(model["observations"], {v: i for i, v in enumerate(sorted(model["scenes"]))})
    cov = model["local"] + SCENE_SIGMA_M**2 * (scene @ scene.T) + time_covariance([model])
    residual = np.clip(model["residual"], -3 * np.sqrt(cov.diagonal()), 3 * np.sqrt(cov.diagonal()))
    result = cho_solve(cho_factor(cov), np.column_stack((s, locations, residual)))
    u, v, z = result.T
    scale = float((s @ v) / (s @ u))
    info = float(locations @ v - scale * (s @ v))
    score = float(locations @ z - scale * (s @ z))
    if not math.isfinite(info) or info <= 0 or not math.isfinite(score):
        raise ValueError(f"Delivery-group contrast is unidentifiable on {row['release_date']}")
    return output | dict(contrast_information=info, raw_contrast_million_bbl=score / info,
                         contrast_million_bbl=float(np.clip(score / (2. + info), -1.25, 1.25)),
                         latest_scene_ready_utc=max(datetime.fromisoformat(q["available_utc"]) for q in sample))


def inventory_correction(history, row, baseline):
    past = [r for r in history if r["known"] < row["cutoff"] and r["matched_roofs"]][-HISTORY:]
    delta = fit_correction(past, row, "location_design", f"{baseline}_error", penalty=(.18, .4, .6), loss_scale=.75, bound=.30)
    return delta, past[-1]["known"] if past else None, len(past)


def location_design(features):
    x = features["contrast_million_bbl"]
    breadth = min(1., (features["matched_terminals"] - 1) / 6) if features["matched_roofs"] else 0.
    return x, x * breadth, abs(x)


def forecasts():
    rosters = mapped_areas()
    months = indexed("wti_cushing_event_windows")
    anchor = {(r["release_date"], r["channel"]): r for r in read(f"{GAP_STEM}_forecasts")}
    histories = defaultdict(list)
    for row in inputs():
        issue, week, cutoff = (row[k] for k in ("release_date", "week_ending", "cutoff"))
        roster = rosters[issue, week]
        if any(stamp != cutoff for _, stamp in roster.values()):
            raise ValueError(f"The location roster is unavailable at the cutoff on {issue}")
        evidence = location(row, {k: area for k, (area, _) in roster.items()}, months[issue]["near_month"])
        for channel in ("inventory", "market"):
            ref = anchor[issue, channel]
            if (any(datetime.fromisoformat(ref[k]) != row[k] for k in ("cutoff", "known"))
                    or float(ref["actual"]) != row["actual"] or ref["week_ending"] != week):
                raise ValueError(f"Different inventory release on {issue}")
            q = {k: row[k] for k in ("release_date", "study_phase", "week_ending", "cutoff", "known", "actual")} | evidence
            q |= dict(channel=channel, location_design=location_design(evidence), **{m: float(ref[m]) for m in (SUPPLEMENT, MIXED_GAP)})
            past = histories[channel]
            for name, base in ((INVENTORY, SUPPLEMENT), (HYBRID, MIXED_GAP)):
                value, latest, count = inventory_correction(past, q, base)
                q |= {name: q[base] + value, f"{base}_error": q["actual"] - q[base],
                      f"{name}_latest_label_utc": latest, f"{name}_training_weeks": count}
            past.append(q)
            yield q


def direct_market(forecasts_by_issue):
    reference = {r["release_date"]: r for r in read(f"{EXIT_STEM}_trades") if r["method"] == HOLDING_RULE}
    issues = sorted(forecasts_by_issue)
    history = []
    for row, _, _, point, settled, calibration in inputs_with_reference():
        issue, now, sigma = (row[k] for k in ("release_date", "decision", "volatility_usd"))
        q = forecasts_by_issue[issue]
        ref = reference[issue]
        if (not q["cutoff"] < q["known"] <= now < row["entry"] or q["near_month"] != row["near_month"]
                or q["actual"] != row["x"][0]):
            raise ValueError(f"The location forecast is not aligned to the trade on {issue}")
        # Direct correction uses the matched fifth-session target. Evaluate
        # both that exit and the existing conditional exit schedule.
        current = q["contrast_million_bbl"]
        index = bisect_left(issues, issue)
        previous = forecasts_by_issue[issues[index - 1]] if index else None
        lag = previous["contrast_million_bbl"] if previous and previous["known"] < q["cutoff"] else 0.
        surprise = float(np.clip(q["actual"] - q[SUPPLEMENT], -2., 2.))
        designs = {LOCATION: (current, abs(current)), INTERACTION: (current, abs(current), current * surprise),
                   LAG: (lag, abs(lag), lag * surprise)}
        r = dict(cutoff=now, exit_at=row["exit_at"],
                 error=(row["long_gross"] - point) / sigma, **designs)
        past = [z for z in history if z["exit_at"] < now][-HISTORY:]
        history.append(r)
        timing = dict(training_at=settled[-1]["exit_at"], calibration_at=calibration[-1]["exit_at"] if calibration else None)
        exit_index = int(ref["exit_session_number"]) - 2
        baseline = trade(row, ORIGINAL, point, exit_index, **timing)
        if (any(baseline[k] != ref[k] for k in ("near_month", "far_month", "exit_at_utc", "entry_at_utc"))
                or any(baseline[k] != int(ref[k]) for k in ("gross_usd", "net_170_usd", "near_side"))):
            raise ValueError(f"The historical leader failed to reproduce on {issue}")
        for name in MARKET:
            adjustment = 0. if name == ORIGINAL else fit_correction(past, r, name, "error",
                penalty=(.20, .45, .65)[:len(designs[name])], loss_scale=1., bound=.30)
            for suffix, end in (("", 3), ("_conditional_exit", exit_index)):
                t = trade(row, name + suffix, point + sigma * adjustment, end, **timing)
                yield t | dict(inventory_forecast_kbbl=1000*q[SUPPLEMENT], net_340_usd=t["gross_usd"] - 340 * abs(t["near_side"]),
                               location_adjustment_usd=sigma * adjustment, matched_roofs=q["matched_roofs"],
                               latest_location_training_exit_utc=past[-1]["exit_at"] if past and name != ORIGINAL else None)


def summarize(rows, trades):
    result = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        observations = [r for r in rows if r["channel"] == "inventory" and r["study_phase"] == phase]
        measured = [r for r in observations if r["matched_roofs"]]
        metrics = {}
        for label, group in (("all", observations), ("same_pass", measured)):
            scores = {m:1000*np.array([abs(r[m] - r["actual"]) for r in group]) for m in (SUPPLEMENT, MIXED_GAP, INVENTORY, HYBRID)}
            metrics[label] = {m:dict(mae_kbbl=float(error.mean()), weeks=len(group), paired_95_kbbl=block_bootstrap_mean(
                error - scores[MIXED_GAP if m in (MIXED_GAP, HYBRID) else SUPPLEMENT])) for m, error in scores.items()}
        books = defaultdict(list)
        for r in trades:
            if r["study_phase"] == phase:
                books[r["method"]].append(r)
        pnl = {}
        for method in books:
            name, suffix = (method.removesuffix("_conditional_exit"), "_conditional_exit") if method.endswith("_conditional_exit") else (method, "")
            baseline = MIXED_GAP if name in (MIXED_GAP, HYBRID) else ORIGINAL if name in MARKET else SUPPLEMENT
            group = books[method]
            pnl[method] = market_metrics(group) | dict(paired_vs_anchor=paired_evidence(group, books[baseline + suffix])[0])
        result["phases"][phase] = dict(inventory=metrics, trading=pnl, matched_weeks=len(measured),
            median_contrast_information=float(np.median([r["contrast_information"] for r in measured])))
        print(phase, {k:round(v["mae_kbbl"], 2) for k,v in metrics["all"].items()},
              {k:v["net_170_usd"] for k,v in pnl.items() if k.endswith("_conditional_exit")}, flush=True)
    return result


def main():
    with threadpool_limits(limits=1):
        rows = list(forecasts())
        book = {r["release_date"]:r for r in rows if r["channel"] == "market"}
        trades = list(direct_market(book))
        fixed = list(replay_weighted_market(book, (SUPPLEMENT, MIXED_GAP, INVENTORY, HYBRID), reference=SUPPLEMENT))
    combined = trades + [r | dict(location_adjustment_usd=None, matched_roofs=book[r["release_date"]]["matched_roofs"],
                                   latest_location_training_exit_utc=None) for r in fixed]
    report = summarize(rows, combined)
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_trades", combined)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


def register(issue, path, origin, original):
    recovery = origin.with_name(f"{origin.stem}_recovered_optical.json")
    trade_path = DERIVED / f"cushing_forward_cl_satellite_{issue}_inventory.json"
    mixed_path = DERIVED / f"cushing_mixed_optical_intervals_{issue}_registration.json"
    native = DERIVED / f"cushing_forward_native_optical_{issue}_summary.json"
    coverage = DERIVED / f"cushing_forward_optical_coverage_{issue}_summary.json"
    recovered, trade_record, mixed, geometry, optical = (json.loads(p.read_bytes()) for p in (recovery, trade_path, mixed_path, native, coverage))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
        raise ValueError("The delivery contrast must be registered before publication")
    for r in (trade_record, mixed):
        checked_hashes(r["sources_sha256"])
    if any(r[k] != original[k] for r in (recovered, trade_record, mixed) for k in LINEAGE):
        raise ValueError("The delivery contrast uses inconsistent forecast clocks")
    roster_source = "cushing_modern_lidar_optical_registry"
    roster = defaultdict(float)
    preceding = optical["preceding_week_end_utc"][:10]
    for r in read(roster_source):
        if r["modern_input_status"] == "available" and r["latest_engineering_date"] < cutoff.date().isoformat() and r["first_seen"] <= preceding:
            roster[r["terminal_id"]] += float(r["circular_area_m2"])
    observations = [q | dict(area=q["volume_per_m"] * BARREL_M3 * 1e6, available_utc=q["source_available_utc"])
                    for q in original["measured_pairs"]] + recovered["recovered_roofs"]
    if (len({q["candidate_id"] for q in observations}) != len(observations) or len(observations) != trade_record["measured_roofs"]
            or mixed["additional_roofs"] or mixed["measured_roofs"] != len(observations)
            or optical["release_date"] != issue or optical["target_week_end_utc"][:10] != original["week_ending"]):
        raise ValueError("The registered location comparison requires the original roof panel and roster")
    with closing(read_only(SOURCE)) as db:
        calendar = full_sessions(db, issue)
        expiries = list(db.execute("SELECT delivery_month,last_trading_date FROM wti_delivery_calendar ORDER BY delivery_month"))
    guard = calendar[bisect_right(calendar, issue) + 6]
    months = [month for month, expiry in expiries if expiry > guard]
    row = dict(release_date=issue, week_ending=original["week_ending"], cutoff=cutoff, observed=observations,
               public=original["public_ensemble_kbbl"] / 1000, hub_area=geometry["dated_roof_area_m2"])
    evidence = location(row, roster, months[0])
    forecasts_path = DERIVED / f"{STEM}_forecasts.csv"
    historical = [r | dict(cutoff=datetime.fromisoformat(r["cutoff"]), known=datetime.fromisoformat(r["known"]),
                    matched_roofs=int(r["matched_roofs"]),
                    location_design=location_design(dict(matched_roofs=int(r["matched_roofs"]), matched_terminals=int(r["matched_terminals"]),
                        contrast_million_bbl=float(r["contrast_million_bbl"]))),
                    **{f"{m}_error": float(r[f"{m}_error"]) for m in (SUPPLEMENT, MIXED_GAP)})
                  for r in read(f"{STEM}_forecasts")]
    current = row | evidence | dict(location_design=location_design(evidence))
    estimates, clocks = {}, []
    for channel, name, base, point in (("market", INVENTORY, SUPPLEMENT, trade_record["estimates_kbbl"][SUPPLEMENT]),
            ("inventory", HYBRID, MIXED_GAP, mixed["estimates_kbbl"]["learned_interval_weight"])):
        delta, latest, count = inventory_correction([r for r in historical if r["channel"] == channel], current, base)
        estimates[channel] = {base: point, name: point + 1000 * delta}
        clocks.append(latest)
    stamp = datetime.now(timezone.utc)
    if not all(k and k < cutoff <= stamp < deadline for k in clocks):
        raise ValueError("The location research breached the registration or data cutoff")
    evidence["latest_scene_ready_utc"] = evidence["latest_scene_ready_utc"].isoformat() if evidence["latest_scene_ready_utc"] else None
    sources = (origin, recovery, trade_path, mixed_path, native, coverage, forecasts_path, Path(__file__).resolve(),
               DERIVED / f"{roster_source}.csv", ROOT / "scripts/fit_cushing_intraweek.py")
    freeze(path, {k: original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(), policy=FORWARD_POLICY,
        latest_label_utc=max(clocks).isoformat(), training_weeks=count, estimates_kbbl=estimates,
        location=evidence, dates=dict(near_month=months[0], far_month=months[3], expiry_guard=guard),
        sources_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources}))


def forward(issue):
    origin = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    original = json.loads(origin.read_bytes())
    path = DERIVED / f"{STEM}_{issue}_registration.json"
    if not path.exists():
        register(issue, path, origin, original)
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    latest, cutoff, stamped, deadline = (datetime.fromisoformat(record[k]) for k in (
        "latest_label_utc", "information_cutoff_utc", "recorded_at_utc", "registration_deadline_utc"))
    if (any(record[k] != original[k] for k in LINEAGE) or not latest < cutoff <= stamped < deadline
            or record["policy"] != FORWARD_POLICY):
        raise ValueError("Invalid delivery location registration")
    outcome = score_release(origin, original)
    result = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path), estimates_kbbl=record["estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_errors_kbbl={k:{m:abs(v - actual) for m, v in f.items()} for k, f in record["estimates_kbbl"].items()})
    write(DERIVED / f"{STEM}_{issue}_score.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    forward(args.release.isoformat()) if args.release else main()
