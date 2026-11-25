"""Register the exit policy before EIA and score it on the original dated CL entry."""

import argparse
import json
from bisect import bisect_right
from contextlib import closing
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from threadpoolctl import threadpool_limits

from audit_cushing_forward_wti_coverage import archives
from backtest_wti_cushing_spread import full_sessions, settlement
from build_s1_reference_matches import DERIVED, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import RAW, ROOT, write
from evaluate_cushing_forward_public_cl import ENSEMBLE_INTERACTION_METHOD, REGISTERED
from fit_cushing_cl_extension_segments import STEM, after_report_curve, archive, calibrated_inventory, holding_context
from fit_cushing_robust_stack import indexed
from publication_clocks import EASTERN, table_rows
from register_cushing_leading_models import STEM as LEADER, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze
from score_cushing_forward_pair import ROUND_TRIP_COSTS, window_path

STRATEGY = "curve_post_report_gate"


def following_clock(schedule, issue, week_ending):
    week = date.fromisoformat(week_ending) + timedelta(days=7)
    special = []
    for cells in table_rows(schedule):
        if len(cells) != 5 or cells[0] != f"{week:%B} {week.day}, {week.year}":
            continue
        day = datetime.strptime(cells[1], "%B %d, %Y").date()
        clock = datetime.strptime(cells[3].replace(".", ""), "%I:%M %p").time()
        special.append(datetime.combine(day, clock, EASTERN))
    result = special[0] if special else datetime.combine(week + timedelta(days=5), time(10, 30), EASTERN)
    if len(special) > 1 or result.date().isoformat() <= issue:
        raise ValueError("The next EIA date does not match the weekly calendar")
    return result


def register(issue, path, leader_path, now):
    leader = json.loads(leader_path.read_bytes())
    checked_hashes(leader["sources_sha256"])
    cutoff, deadline = (datetime.fromisoformat(leader[k]) for k in ("information_cutoff_utc", "registration_deadline_utc"))
    if leader["release_date"] != issue or not cutoff <= now < deadline or issue in indexed("eia_wpsr_forward_releases"):
        raise ValueError("The exit rule must be recorded before the initial report")
    schedule = (RAW / "eia" / "forward_release_schedule.html").read_bytes()
    boundary = following_clock(schedule.decode("utf-8"), issue, leader["week_ending"])
    schedule_snapshot = RAW / "eia" / "wpsr_forward_schedules" / f"{issue}_following.html"
    write(schedule_snapshot, schedule)
    with closing(read_only(SOURCE)) as db:
        sessions = full_sessions(db, issue)
    days = sessions[bisect_right(sessions, issue):][:5]
    marks = list(map(settlement, days[1:]))
    if len(marks) != 4:
        raise ValueError("The forward calendar lacks full settlements")
    report_close = next((i for i, stamp in enumerate(marks) if boundary <= stamp), 3)
    sources = (leader_path, schedule_snapshot, Path(__file__).resolve(), ROOT / "scripts" / f"fit_{STEM}.py",
               ROOT / "scripts" / "fit_cushing_cl_regime_strategy.py", ROOT / "scripts" / "fit_cushing_cl_exit_horizon.py",
               ROOT / "scripts" / "fit_cushing_cl_volatility.py", DERIVED / f"{STEM}_market_forecasts.csv",
               DERIVED / "cushing_volatility_neighbors_trades.csv", DERIVED / "cushing_prerelease_sensor_forecasts.csv")
    freeze(path, dict(release_date=issue, strategy=STRATEGY, recorded_at_utc=now.isoformat(), deadline_utc=deadline.isoformat(),
        following_report_at=boundary.isoformat(), sessions=[s.isoformat() for s in marks], report_close_index=report_close,
        sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))


def score(record, prefix, leader_decision, decision_path, now):
    issue = record["release_date"]
    boundary = datetime.fromisoformat(record["following_report_at"])
    follow = indexed("eia_wpsr_forward_releases").get(boundary.date().isoformat())
    if follow and (not follow["scheduled_release_at_utc"] or datetime.fromisoformat(follow["scheduled_release_at_utc"]) != boundary):
        raise ValueError("The next release clock differs from the registered calendar")
    stamps = list(map(datetime.fromisoformat, record["sessions"]))
    if (datetime.fromisoformat(leader_decision["entry_at"]) != stamps[0]
            or datetime.fromisoformat(leader_decision["exit_at"]) != stamps[3]):
        raise ValueError("The registered entry or original exit differs")
    points = [r for r in read("cushing_forward_wti_coverage_bars") if r["release_date"] == issue and r["session_index"] != "1"]
    _, marks = archives(as_of=now)
    path, count = window_path(leader_decision, points, marks, now)
    spread = [v for _,v in path]
    public_receipt = REGISTERED / f"{issue}_{ENSEMBLE_INTERACTION_METHOD}.json"
    if digest(public_receipt) != leader_decision["public_receipt_sha256"]:
        raise ValueError("The original public trading inputs changed")
    side = leader_decision["near_side"]
    step = record["report_close_index"]
    result = dict(status="awaiting_market_window", observed_leg_sessions=count, near_side=side,
                  near_month=leader_decision["near_month"], far_month=leader_decision["far_month"])
    if step == 2 and side:
        early = step - 1
        cutoff = datetime.combine(stamps[early].date(), time.min, EASTERN)
        receipt = prefix.with_name(f"{prefix.name}_decision.json")
        if not receipt.is_file():
            if now < cutoff:
                return result | dict(status="awaiting_holding_decision", holding_decision_utc=cutoff.isoformat())
            if now >= stamps[early]:
                return result | dict(status="holding_decision_window_closed")
            _, cutoff_marks = archives(as_of=cutoff)
            cutoff_path, _ = window_path(leader_decision, points, cutoff_marks, cutoff)
            prior_path = [v for _,v in cutoff_path[:early]]
            if any(v is None for v in prior_path):
                return result | dict(status="awaiting_holding_input_prices", holding_decision_utc=cutoff.isoformat())
            public = json.loads(public_receipt.read_bytes())["decision"]
            prior = prior_path[-1] - prior_path[0]
            market = dict(release_date=issue, stamps=stamps, next_report_clock=boundary, last_before_next_release=early,
                volatility_usd=public["volatility_usd"], spread_path_usd=(prior,),
                x=(public["actual_million_bbl"], 0., public["carry"], public["carry_change"]))
            context = holding_context(market, leader_decision["expected_long_move_usd"])
            with threadpool_limits(limits=1):
                expected, latest = after_report_curve(calibrated_inventory(archive()), context)
            stamp = datetime.now(timezone.utc)
            if stamp >= stamps[early] or latest and latest >= cutoff:
                raise ValueError("The holding decision used a late price or label")
            freeze(receipt, dict(recorded_at_utc=stamp.isoformat(), holding_decision_utc=cutoff.isoformat(), near_side=side,
                original_decision_sha256=digest(decision_path), expected_long_move_after_report_usd=expected,
                latest_training_exit_utc=latest.isoformat() if latest else None,
                last_price_mark_utc=stamps[early - 1].isoformat(), exit_index=3 if expected is not None and side * expected > 0 else step))
        frozen = json.loads(receipt.read_bytes())
        value = frozen["expected_long_move_after_report_usd"]
        selected = 3 if value is not None and side * value > 0 else step
        if (frozen["original_decision_sha256"] != digest(decision_path) or frozen["near_side"] != side
                or frozen["exit_index"] != selected or not cutoff <= datetime.fromisoformat(frozen["recorded_at_utc"]) < stamps[early]):
            raise ValueError("The holding decision receipt does not verify")
        result.update(frozen)
    else:
        selected = step
    result.update(exit_at=stamps[selected].isoformat(), original_exit_at=stamps[3].isoformat(),
                  fixed_report_close_exit_at=stamps[step].isoformat())
    if all(v is not None for v in spread):
        result["status"] = "scored_market_window"
        for method, i in ((STRATEGY, selected), ("original", 3), ("fixed_report_close", step)):
            gross = side * (spread[i] - spread[0])
            result[method] = dict(gross_usd=gross, **{f"net_{c}_usd":gross - c * abs(side) for c in ROUND_TRIP_COSTS})
    return result


def run(issue):
    now = datetime.now(timezone.utc)
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    leader = DERIVED / f"{LEADER}_{issue}_registration.json"
    if not path.exists():
        register(issue, path, leader, now)
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    if (record["release_date"] != issue or record["strategy"] != STRATEGY
            or not datetime.fromisoformat(record["recorded_at_utc"]) < datetime.fromisoformat(record["deadline_utc"])):
        raise ValueError("The registered exit rule does not verify")
    decision = DERIVED / f"{LEADER}_{issue}_decision.json"
    result = dict(release_date=issue, status="awaiting_original_entry_decision", registration_sha256=digest(path),
                  following_report_at=record["following_report_at"], sessions=record["sessions"],
                  report_close_index=record["report_close_index"])
    if decision.exists():
        entry = json.loads(decision.read_bytes())
        if (entry["release_date"] != issue or entry["forecast_sha256"] != digest(leader)
                or entry["near_side"] not in (-1, 0, 1)):
            raise ValueError("The original entry receipt does not match the leader")
        result.update(score(record, prefix, entry, decision, now))
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True, type=date.fromisoformat)
    run(parser.parse_args().release.isoformat())
