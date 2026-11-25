"""Record the conditional CL research candidate against the existing forward leader."""

import argparse
import json
from datetime import date, datetime, timezone

from audit_cushing_forward_wti_coverage import archives
from build_s1_reference_matches import DERIVED, read
from collect_public import ROOT, write
from evaluate_cushing_forward_public_cl import ENSEMBLE_INTERACTION_METHOD, REGISTERED, registered_decision
from fit_cushing_cl_conditional_response import base_inputs, conditional_residual, features
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE
from fit_cushing_delayed_calendar import replay as market_calendar
from register_cushing_prerelease_forecast import digest, freeze
from score_cushing_forward_pair import score, window_move, window_path

METHOD = "inventory_conditional_second_session"
CONTROL = ENSEMBLE_INTERACTION_METHOD
SOURCE_FILES = ("scripts/fit_cushing_cl_conditional_response.py", "data/derived/cushing_optical_stability_trades.csv",
                "data/derived/cushing_optical_stability_forecasts.csv")


def context(row):
    return features(row["carry"],row["volatility_usd"],row["actual_million_bbl"]-row["forecast_million_bbl"])


def previous_forward_rows(ledger, issue, decision, calendars):
    previous = []
    _,marks = archives(as_of=decision)
    for row in ledger:
        day = row["release_date"]
        if (row["method"] != CONTROL or day >= issue or not row["eligible_nonoverlap"]
            or not (REGISTERED/f"{day}_{CONTROL}.json").exists() or datetime.fromisoformat(row["exit_at"]) >= decision):
            continue
        verified = registered_decision(row,register=False,now=decision)
        if datetime.fromisoformat(verified["recorded_at_utc"]) >= decision:
            continue
        move,_ = window_move(verified,calendars[day],marks,decision)
        if move is not None:
            previous.append(dict(decision=datetime.fromisoformat(row["decision_at_utc"]),exit_at=datetime.fromisoformat(row["exit_at"]),
                                 base_estimate=row["estimated_long_move_usd"],volatility_usd=row["volatility_usd"],
                                 response_features=context(row),long_gross=move))
    return previous


def register(path, baseline_path, control, ledger, calendars):
    decision, deadline = [datetime.fromisoformat(control[k]) for k in ("decision_at_utc","registration_deadline_utc")]
    if not decision <= datetime.now(timezone.utc) < deadline:
        raise ValueError("The candidate must be registered before its entry mark")
    _,_,cases = market_calendar()
    history = [rows[1] for _,rows in base_inputs(cases) if rows[1]["exit_at"] < decision]
    history.extend(previous_forward_rows(ledger,control["release_date"],decision,calendars))
    history = sorted(history,key=lambda r:r["decision"])[-HISTORY:]
    if len(history)<MIN_WINDOWS or any(r["exit_at"]>=decision for r in history):
        raise ValueError("The settled calibration history is incomplete or contains a future exit")
    current = dict(decision=decision,response_features=context(control),volatility_usd=control["volatility_usd"])
    adjustment, support = conditional_residual(history,current)
    estimate = control["estimated_long_move_usd"]+adjustment
    stamp = datetime.now(timezone.utc)
    if not datetime.fromisoformat(control["recorded_at_utc"]) <= stamp < deadline:
        raise ValueError("The registration finished after the entry mark")
    fields = ("release_date","decision_at_utc","registration_deadline_utc","entry_at","exit_at","near_month","far_month")
    candidate = {k:control[k] for k in fields} | dict(method=METHOD, recorded_at_utc=stamp.isoformat(),
                estimated_long_move_usd=estimate, near_side=int(estimate>HURDLE)-int(estimate<-HURDLE))
    record = dict(decision=candidate, control_receipt_sha256=digest(baseline_path), base_estimate_usd=control["estimated_long_move_usd"],
                  conditional_adjustment_usd=adjustment, similar_report_weight=support, calibration_windows=len(history),
                  latest_calibration_exit_utc=max(r["exit_at"] for r in history).isoformat(),
                  source_sha256={source:digest(ROOT/source) for source in SOURCE_FILES})
    freeze(path,record)
    return record


def following_release_check(path, issue, candidate, points, marks, now):
    """Score a fixed control using the already registered following EIA date."""
    prerelease_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    forecast = json.loads(prerelease_path.read_bytes())
    received = datetime.fromisoformat(forecast["recorded_at_utc"])
    boundary = datetime.fromisoformat(forecast["registration_deadline_utc"])
    stamp = datetime.now(timezone.utc)
    if (forecast["release_date"] != issue or not received < boundary or
        not datetime.fromisoformat(candidate["recorded_at_utc"]) <= stamp < datetime.fromisoformat(candidate["entry_at"]) or
        boundary >= datetime.fromisoformat(candidate["exit_at"])):
        raise ValueError("The pre-report comparison needs a date and a timely original decision")
    marks_at, _ = window_path(candidate, points, marks, now)
    before = [when for when,_ in marks_at[1:] if when < boundary]
    if not before:
        raise ValueError("No closing session before the next reported week")
    result = dict(recorded_at_utc=stamp.isoformat(), next_release_date=issue, next_release_day_start_utc=boundary.isoformat(),
                  comparison_exit_at=before[-1].isoformat(), original_receipt_sha256=digest(path),
                  prerelease_forecast_sha256=digest(prerelease_path))
    return result


def compare_following_release(issue, candidate, path, points, marks, now, *, register_next_issue=None):
    comparison_path = path.with_name(f"{path.stem}_before_following_release.json")
    if register_next_issue and not comparison_path.exists():
        freeze(comparison_path, following_release_check(path, register_next_issue, candidate, points, marks, now))
    if not comparison_path.exists():
        return None
    record = json.loads(comparison_path.read_bytes())
    if register_next_issue and register_next_issue != record["next_release_date"]:
        raise ValueError("The dated comparison is already registered")
    forecast_path = DERIVED/f"cushing_prerelease_forecast_{record['next_release_date']}.json"
    recorded, exit_at, boundary = (datetime.fromisoformat(record[k]) for k in (
        "recorded_at_utc", "comparison_exit_at", "next_release_day_start_utc"))
    entry = datetime.fromisoformat(candidate["entry_at"])
    path_marks, _ = window_path(candidate, points, marks, now)
    known_marks = dict(path_marks)
    if (digest(path) != record["original_receipt_sha256"] or digest(forecast_path) != record["prerelease_forecast_sha256"] or
        not datetime.fromisoformat(candidate["recorded_at_utc"]) <= recorded <= datetime.now(timezone.utc) or
        not recorded < entry < exit_at < boundary < datetime.fromisoformat(candidate["exit_at"]) or
        exit_at != max(when for when in known_marks if when < boundary)):
        raise ValueError(f"The dated comparison no longer matches its source on {issue}")
    start, before, after = (known_marks[t] for t in (entry, exit_at, datetime.fromisoformat(candidate["exit_at"])))
    side = candidate["near_side"]
    complete = all(value is not None for value in (start, before, after))
    return record | dict(comparison_complete=complete, observed_paired_marks=sum(v is not None for v in (start, before, after)),
                         net_before_following_release_170_usd=side*(before-start)-170*abs(side) if complete else None,
                         net_original_exit_170_usd=side*(after-start)-170*abs(side) if complete else None,
                         increment_from_holding_usd=side*(after-before) if complete else None)


def run(issue, *, scoring, register_next_issue=None):
    ledger = json.loads((DERIVED/"cushing_forward_public_cl_summary.json").read_bytes())["release_rows"]
    row, = (r for r in ledger if r["release_date"]==issue and r["method"]==CONTROL)
    now = datetime.now(timezone.utc)
    control = registered_decision(row,register=False,now=now)
    baseline_path = REGISTERED/f"{issue}_{CONTROL}.json"
    if not control["recorded_at_utc"] or not control["eligible_nonoverlap"]:
        raise ValueError("A timely baseline decision is required for this candidate")
    path = REGISTERED/f"{issue}_{METHOD}.json"
    calendars = {}
    for p in read("cushing_forward_wti_coverage_bars"):
        if p["session_index"]!="1":
            calendars.setdefault(p["release_date"],[]).append(p)
    if not scoring and not path.exists():
        return register(path,baseline_path,control,ledger,calendars)
    record = json.loads(path.read_bytes())
    candidate = record["decision"]
    fields = ("release_date","decision_at_utc","registration_deadline_utc","entry_at","exit_at","near_month","far_month")
    if (any(candidate[k]!=control[k] for k in fields) or record["control_receipt_sha256"]!=digest(baseline_path)
        or candidate["method"]!=METHOD or any(digest(ROOT/name)!=value for name,value in record["source_sha256"].items())
        or not datetime.fromisoformat(record["latest_calibration_exit_utc"])<datetime.fromisoformat(control["decision_at_utc"])
        or not datetime.fromisoformat(control["recorded_at_utc"])<=datetime.fromisoformat(candidate["recorded_at_utc"])<=now
        or datetime.fromisoformat(candidate["recorded_at_utc"])>=datetime.fromisoformat(control["registration_deadline_utc"])):
        raise ValueError("The candidate no longer matches its registered baseline or cutoff")
    if not scoring:
        return record
    _,marks = archives(as_of=now)
    result, = score({issue:{METHOD:candidate,CONTROL:control}},calendars,marks,now,methods=(METHOD,CONTROL))
    summary = dict(status="window_scored" if result["market_window_complete"] else "awaiting_market_window",
                   leader=METHOD, control=CONTROL, receipt_sha256=digest(path), result=result)
    comparison = compare_following_release(issue,candidate,path,calendars[issue],marks,now,register_next_issue=register_next_issue)
    if comparison is not None:
        summary["before_following_release"] = comparison
    write(DERIVED/f"cushing_cl_conditional_forward_{issue}_score.json",(json.dumps(summary,indent=2,allow_nan=False)+"\n").encode())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--register-release",type=date.fromisoformat)
    command.add_argument("--score-release",type=date.fromisoformat)
    parser.add_argument("--register-following-release", type=date.fromisoformat)
    args = parser.parse_args()
    if args.register_following_release and not args.score_release:
        parser.error("--register-following-release requires --score-release")
    release = args.score_release or args.register_release
    following = args.register_following_release
    print(json.dumps(run(release.isoformat(),scoring=args.score_release is not None,
                         register_next_issue=following.isoformat() if following else None),indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
