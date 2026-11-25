"""Score the registered research leader and the curve control on identical CL windows."""

import json
from collections import defaultdict
from datetime import datetime, timezone

from audit_cushing_forward_wti_coverage import archives
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from evaluate_cushing_forward_public_cl import (
    CARRY_INTERACTION_METHOD,
    ENSEMBLE_INTERACTION_METHOD,
    REGISTERED,
    STEM as DECISION_STEM,
    registered_decision,
)
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_forward_cl_pair"
METHODS = (ENSEMBLE_INTERACTION_METHOD, CARRY_INTERACTION_METHOD)
ROUND_TRIP_COSTS = (170, 340)


def window_path(leader, points, marks, as_of):
    issue = leader["release_date"]
    entry, exit_at = (datetime.fromisoformat(leader[k]) for k in ("entry_at", "exit_at"))
    expected = [(str(i), leg) for i in range(2, 6) for leg in ("near", "far")]
    if [(p["session_index"], p["leg"]) for p in points] != expected or any(
        p["contract_month"] != leader[f"{p['leg']}_month"] for p in points
    ):
        raise ValueError(f"Invalid delivery months or sessions on {issue}")
    dates = [datetime.fromisoformat(p["settlement_at"]) for p in points]
    paired_dates = dates[::2]
    if (paired_dates != dates[1::2] or paired_dates[0] != entry or paired_dates[-1] != exit_at
        or sorted(set(paired_dates)) != paired_dates):
        raise ValueError(f"Different settlement path on {issue}")
    count = 0
    spreads = []
    for near, far, stamp in zip(points[::2], points[1::2], paired_dates, strict=True):
        prices = []
        for point in (near, far):
            if stamp.date().isoformat() != point["session_date"]:
                raise ValueError(f"Settlement date does not match the trading session on {issue}")
            mark = marks.get((point["session_date"], point["contract_month"]))
            if mark and stamp < datetime.fromisoformat(mark[3]) <= as_of:
                prices.append(mark[0])
        count += len(prices)
        spreads.append(10*(prices[0]-prices[1]) if len(prices)==2 else None)
    return list(zip(paired_dates, spreads, strict=True)), count


def window_move(leader, points, marks, as_of):
    path, count = window_path(leader, points, marks, as_of)
    move = path[-1][1]-path[0][1] if count == len(points) else None
    return move, count


def score(candidates, calendars, marks, as_of, *, methods=METHODS):
    previous_exit = None
    for issue, models in sorted(candidates.items()):
        if set(models) != set(methods):
            continue
        leader, control = (models[m] for m in methods)
        fields = ("decision_at_utc", "entry_at", "exit_at", "near_month", "far_month")
        if any(leader[k] != control[k] for k in fields):
            raise ValueError(f"Different decision or execution windows on {issue}")
        decision, entry, exit_at = (datetime.fromisoformat(leader[k]) for k in fields[:3])
        if not (decision < entry < exit_at and (previous_exit is None or previous_exit < entry)):
            raise ValueError(f"Invalid or overlapping registered CL window on {issue}")
        previous_exit = exit_at
        points = calendars[issue]
        long_gross, mark_count = window_move(leader, points, marks, as_of)
        complete = long_gross is not None
        sides = (leader["near_side"], control["near_side"])
        if any(s not in (-1, 0, 1) for s in sides):
            raise ValueError(f"Invalid trade direction on {issue}")
        result = dict(
            release_date=issue, **{k: leader[k] for k in fields},
            leader_recorded_at_utc=leader["recorded_at_utc"], control_recorded_at_utc=control["recorded_at_utc"],
            leader_near_side=sides[0], control_near_side=sides[1], observed_leg_sessions=mark_count,
            expected_leg_sessions=len(points), market_window_complete=complete, long_spread_gross_usd=long_gross,
        )
        for cost in ROUND_TRIP_COSTS:
            leader_net, control_net = (s*long_gross-cost*abs(s) if complete else None for s in sides)
            result.update({f"leader_net_{cost}_usd": leader_net, f"control_net_{cost}_usd": control_net,
                           f"increment_net_{cost}_usd": leader_net-control_net if complete else None})
        yield result


def main():
    as_of = datetime.now(timezone.utc)
    ledger = json.loads((DERIVED/f"{DECISION_STEM}_summary.json").read_bytes())["release_rows"]
    candidates, calendars = defaultdict(dict), defaultdict(list)
    for row in ledger:
        issue, method = row["release_date"], row["method"]
        if method not in METHODS or not (REGISTERED/f"{issue}_{method}.json").is_file():
            continue
        verified = registered_decision(row, register=False, now=as_of)
        if not verified["eligible_nonoverlap"] or datetime.fromisoformat(verified["recorded_at_utc"]) > as_of or method in candidates[issue]:
            raise ValueError(f"Ineligible or duplicate registration on {issue}/{method}")
        candidates[issue][method] = verified
    for row in read("cushing_forward_wti_coverage_bars"):
        if row["session_index"] != "1":
            calendars[row["release_date"]].append(row)
    _, marks = archives()
    rows = list(score(candidates, calendars, marks, as_of))
    closed = [r for r in rows if r["market_window_complete"]]
    report = dict(as_of_utc=as_of.isoformat(),leader=METHODS[0],control=METHODS[1],
                  registered_pairs=len(rows),completed_pairs=len(closed),
                  completed_pairs_with_different_sides=sum(r["leader_near_side"]!=r["control_near_side"] for r in closed),
                  net_usd={},releases=rows)
    for cost in ROUND_TRIP_COSTS:
        fields = [f"{m}_net_{cost}_usd" for m in ("leader","control","increment")]
        report["net_usd"][cost] = {k:sum(r[k] for r in closed) if closed else None for k in fields} | dict(
            mean_increment_95_usd=block_bootstrap_mean([r[fields[2]] for r in closed]))
    if rows:
        export(f"{STEM}_releases", rows)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())
    print(f"Registered pairs: {len(rows)}. Completed pairs: {len(closed)}.")


if __name__ == "__main__":
    main()
