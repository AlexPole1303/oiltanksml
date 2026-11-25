"""Test three entry clocks on identical dated CL contracts and report windows."""

import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_ibkr_intraday import read_packets
from audit_cushing_leader_execution import from_ibkr
from backtest_wti_cushing_spread import COSTS, settlement, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, RISK_FLOOR_USD
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as INVENTORY, SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features
from fit_cushing_volatility_neighbors import volatility_forecast

STEM = "cushing_cl_earlier_entry"
ENTRIES = ("report_day", "first_session", "second_session")
MODELS = ("curve", "public", "satellite")
FIXED_SIGNAL = "satellite_second_session_signal"
LEADER = "curve_post_report_gate"
LAG = timedelta(minutes=30)


def matched_windows(counts, *, short=False):
    cases = list(inputs())
    clocks = indexed("eia_release_clock")
    forecast = indexed(f"{INVENTORY}_forecasts")
    original_windows = indexed("wti_cushing_event_windows")
    anchors = {r["release_date"]:r for r in read("cushing_cl_extension_segments_trades") if r["method"] == LEADER}
    first = defaultdict(dict)
    for r in read("wti_cushing_event_bars"):
        if r["session_number"] == "1":
            first[r["release_date"]][r["leg"]] = r
    through = max(r["exit_at"].date().isoformat() for r, _ in cases)
    with closing(read_only(SOURCE)) as db:
        prices = {s: {(d, m):(tick_price(p), ref) for d,m,ref,p in price_rows(db,s,numeric_through=through)}
                  for s in ("archive", "ibkr", "official")}
        full_days = {r[0] for r in db.execute("SELECT session_date FROM wti_session_basis WHERE full_cmes_session=1 "
                                             "AND cme_no_settlement_notice_url IS NULL")}
    previous_exit = None
    for row, _ in cases:
        issue, phase = row["release_date"], row["study_phase"]
        counts[phase]["candidate_releases"] += 1
        event = clocks[issue]
        release = event["scheduled_table_release_at_utc"]
        day_mark = settlement(issue)
        decision = day_mark - LAG
        if not release or issue not in full_days or datetime.fromisoformat(release) + LAG > decision:
            counts[phase]["missing_scheduled_release_or_full_session"] += 1
            continue
        f = forecast[issue]
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc", "known_utc"))
        scheduled = datetime.fromisoformat(release)
        # The archived conservative timestamp is next-day. For this replay
        # use a separate schedule-based assumption with a 30-minute buffer.
        assumed_known = scheduled + LAG
        last_label = f[f"{SUPPLEMENT}_latest_label_utc"]
        if (not cutoff < scheduled < known <= row["decision"] or float(f["actual"]) != row["x"][0]
                or last_label and datetime.fromisoformat(last_label) >= cutoff):
            raise ValueError(f"Mismatched inventory source on {issue}")
        near, far = (row[k] for k in ("near_month", "far_month"))
        legs = first[issue]
        first_mark = datetime.fromisoformat(legs["near"]["settlement_reference_time"])
        marks = (day_mark, first_mark, *row["stamps"][:3])
        if ([legs[k]["delivery_month"] for k in ("near","far")] != [near,far]
                or legs["near"]["settlement_reference_time"] != legs["far"]["settlement_reference_time"]
                or not assumed_known <= decision < marks[0] < marks[1] < marks[2] < marks[3] < marks[4]):
            raise ValueError(f"Invalid dated session sequence on {issue}")
        if short:
            marks = marks[:2]
        keys = [(t.date().isoformat(), m) for t in marks for m in (near,far)]
        sources = [s for s, book in prices.items() if all(k in book for k in keys)]
        if not sources:
            counts[phase]["missing_complete_dated_price_path"] += 1
            continue
        if previous_exit is not None and marks[0] <= previous_exit:
            counts[phase]["overlapping_earliest_entry"] += 1
            continue
        previous_exit = marks[-1]
        counts[phase]["admitted_releases"] += 1
        source = sources[0]
        counts[phase][f"source_{source}"] += 1
        quotes = [prices[source][k] for k in keys]
        path = tuple(10 * (a[0]-b[0]) for a,b in zip(quotes[::2],quotes[1::2],strict=True))
        if not short and tuple(v-path[2] for v in path[2:]) != tuple(row["spread_path_usd"][:3]):
            raise ValueError(f"Earlier-entry endpoint differs from the archived spread on {issue}")
        if issue in anchors:
            anchor = anchors[issue]
            if any(row[k] != anchor[k] for k in ("near_month","far_month","study_phase")):
                raise ValueError(f"Different benchmark contracts on {issue}")
        else:
            anchor = None
        expectations = (float(f[PUBLIC]), float(f[SUPPLEMENT]))
        base_window = original_windows[issue]
        if not (marks[-1].date().isoformat() < base_window["near_expiry"] < base_window["far_expiry"]):
            raise ValueError(f"A contract expires inside the comparison window on {issue}")
        yield row | dict(marks=marks, marks_usd=path, exit_at=marks[-1], decision=decision, expectations=expectations,
                         source=source, price_references="|".join(dict.fromkeys(ref for _,ref in quotes)),
                         scheduled_report_utc=scheduled.isoformat(), assumed_report_known_utc=assumed_known.isoformat(),
                         release_time_basis=event["timing_basis"], anchor=anchor)


def risk(path, side, entry_index):
    return [side * (mark-path[entry_index]) - abs(side) * (170 if i == len(path)-1 else 85)
            for i, mark in enumerate(path) if i >= entry_index]


def decisions(windows, *, short=False):
    history = defaultdict(list)
    for row in windows:
        times, path = row["marks"], row["marks_usd"]
        clocks = ENTRIES[:1] if short else ENTRIES
        models = (*MODELS, "always_long") if short else (*MODELS, FIXED_SIGNAL, "always_long")
        estimates = {}
        training_at = None
        for i, clock in enumerate(clocks):
            sigma = max(RISK_FLOOR_USD, row["volatility_usd"]*math.sqrt((len(times)-1-i)/3))
            # Build carry interactions with the same horizon-specific risk
            # estimate used to scale the price target.
            design_row = row | dict(volatility_usd=sigma)
            designs = (features(design_row)[-1], *(replace_expectation(design_row, v) for v in row["expectations"]))
            for model, x in zip(MODELS, designs, strict=True):
                series = history[clock, model]
                current = dict(x=x, volatility_usd=sigma, long_gross=path[-1]-path[i], exit_at=times[-1])
                train = [q for q in series if q["exit_at"] < row["decision"]][-HISTORY:]
                series.append(current)
                if len(train) >= MIN_WINDOWS:
                    estimates[clock, model] = volatility_forecast(train, current)
                    training_at = train[-1]["exit_at"]
        if training_at is None:
            continue
        ref = row["anchor"]
        for i, clock in enumerate(clocks):
            for model in models:
                if model == "always_long":
                    expected, side = None, 1
                else:
                    key = ("second_session", "satellite") if model == FIXED_SIGNAL else (clock, model)
                    expected = estimates[key]
                    side = int(expected > HURDLE)-int(expected < -HURDLE)
                gross = side*(path[-1]-path[i])
                yield {k:row[k] for k in ("release_date","study_phase","near_month","far_month","source",
                                          "scheduled_report_utc","assumed_report_known_utc","release_time_basis",
                                          "price_references")} | dict(
                    method=f"{clock}:{model}", decision_at_utc=row["decision"].isoformat(),
                    entry_at_utc=times[i].isoformat(), exit_at_utc=times[-1].isoformat(),
                    latest_training_exit_utc=training_at.isoformat(), expected_long_move_usd=expected,
                    near_side=side, long_move_usd=path[-1]-path[i], gross_usd=gross,
                    entry_spread_usd=path[i], exit_spread_usd=path[-1],
                    daily_path_usd=json.dumps(risk(path,side,i)), leader_near_side=int(ref["near_side"]) if ref else None,
                    leader_net_170_usd=int(ref["net_170_usd"]) if ref else None,
                    **{f"cost_{c}_usd":c*abs(side) for c in COSTS},
                    **{f"net_{c}_usd":gross-c*abs(side) for c in (*COSTS,340)})


def summarize(group):
    equity = peak = drawdown = 0
    for q in group:
        for move in json.loads(q["daily_path_usd"]):
            value = equity+move
            peak = max(peak,value)
            drawdown = max(drawdown,peak-value)
        equity += q["net_170_usd"]
    fitted = [q for q in group if q["expected_long_move_usd"] is not None]
    return robustness(group) | dict(
        net_340_usd=sum(q["net_340_usd"] for q in group), daily_drawdown_usd=drawdown,
        spread_mae_usd=float(np.mean([abs(q["long_move_usd"]-q["expected_long_move_usd"]) for q in fitted])) if fitted else None)


def mark_check(rows):
    book,_ = read_packets()
    sample = []
    for r in rows:
        if r["study_phase"] != "holdout" or not r["near_side"] or not r["method"].endswith(":satellite"):
            continue
        near, far = r["near_month"], r["far_month"]
        start, end = (datetime.fromisoformat(r[k]) for k in ("entry_at_utc","exit_at_utc"))
        entry, exit_mark = (from_ibkr(book, t, near, far, 30) for t in (start,end))
        if entry is None or exit_mark is None:
            continue
        gross = r["near_side"]*(exit_mark[0]-entry[0])
        sample.append(dict(release_date=r["release_date"],method=r["method"],
                           settlement_net_170_usd=r["net_170_usd"],trade_bars_net_170_usd=gross-170,
                           deviation_usd=gross-r["gross_usd"]))
    return sample


def combined_book(early):
    original = [q for q in read("cushing_cl_extension_segments_trades") if q["method"] == LEADER]
    later_rows = [dict(q, clock="second_session", near_side=int(q["near_side"]), gross_usd=int(q["gross_usd"]),
                       latest_training_exit_utc=q["original_latest_training_exit_utc"],
                       daily_path_usd=json.dumps([int(q[f"session_{i}_net_170_usd"]) for i in range(2,6)]),
                       **{f"net_{cost}_usd":int(q[f"net_{cost}_usd"]) for cost in (*COSTS,340)}) for q in original]
    later_entries = [datetime.fromisoformat(q["settlement_reference_time"])
                     for q in read("wti_cushing_event_bars") if (q["session_number"],q["leg"]) == ("2","near")]
    methods = (*MODELS, "always_long")
    for m in ("leader", *methods):
        shorts = [q | dict(clock="report_day") for q in early if q["method"] == f"report_day:{m}"]
        candidates = sorted([*later_rows, *shorts],key=lambda q:datetime.fromisoformat(q["entry_at_utc"]))
        reserved_until = None
        for q in candidates:
            decision, entry, exit_at = (datetime.fromisoformat(q[k])
                                       for k in ("decision_at_utc","entry_at_utc","exit_at_utc"))
            if not decision < entry < exit_at:
                raise ValueError(f"Invalid combined-book clock on {q['release_date']}")
            side = q["near_side"]
            # A known open position blocks entry. The calendar also keeps
            # the short horizon away from the already defined later entry.
            early_entry = q["clock"] == "report_day"
            overlaps = bool(side and ((reserved_until and reserved_until >= (decision if early_entry else entry))
                            or (early_entry and any(entry <= t <= exit_at for t in later_entries))))
            if overlaps and not early_entry:
                raise ValueError(f"An early leg overlaps the later rule on {q['release_date']}")
            if side and not overlaps:
                reserved_until = exit_at
            side = 0 if overlaps else side
            gross = q["gross_usd"] if side else 0
            path = q["daily_path_usd"] if side else "[]"
            if side and json.loads(path)[-1] != gross-170:
                raise ValueError(f"Unreconciled book on {q['release_date']}")
            yield {k:q[k] for k in ("release_date","study_phase","clock","near_month","far_month","latest_training_exit_utc",
                                    "decision_at_utc","entry_at_utc","exit_at_utc")} | dict(
                method=m, near_side=side, gross_usd=gross, blocked_early_entry=overlaps, daily_path_usd=path,
                expected_long_move_usd=None,
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},
                **{f"net_{c}_usd":gross-c*abs(side) for c in (*COSTS,340)})


def main():
    counts = defaultdict(Counter)
    short_counts = defaultdict(Counter)
    grouped = defaultdict(list)
    with threadpool_limits(limits=1):
        rows = list(decisions(matched_windows(counts)))
        short = list(decisions(matched_windows(short_counts, short=True), short=True))
    combined = list(combined_book(short))
    for r in rows:
        grouped[r["study_phase"],r["method"]].append(r)
    phases = {}
    for phase in PHASES:
        methods = {m:summarize(g) for (p,m),g in grouped.items() if p == phase}
        frozen = grouped[phase,"second_session:satellite"]
        if any([q["release_date"] for q in g] != [q["release_date"] for q in frozen]
               for (p,_),g in grouped.items() if p == phase):
            raise ValueError(f"Unmatched execution calendars in {phase}")
        comparisons = {}
        fixed = grouped[phase,f"second_session:{FIXED_SIGNAL}"]
        for clock in ENTRIES:
            s, c = (grouped[phase,f"{clock}:{m}"] for m in ("satellite","curve"))
            delta = [q["net_170_usd"]-b["net_170_usd"] for q,b in zip(s,c,strict=True)]
            same_signals = grouped[phase,f"{clock}:{FIXED_SIGNAL}"]
            if any(q["near_side"] != b["near_side"] for q, b in zip(same_signals, fixed, strict=True)):
                raise ValueError(f"Changed side during matched reprice in {phase}/{clock}")
            clock_delta = [q["net_170_usd"]-b["net_170_usd"] for q,b in zip(same_signals, fixed, strict=True)]
            comparisons[clock] = dict(increment_vs_curve_usd=sum(delta),paired_mean_95_usd=block_bootstrap_mean(delta),
                fixed_signal_entry_increment_usd=sum(clock_delta),fixed_signal_entry_paired_95_usd=block_bootstrap_mean(clock_delta))
        live = [q for q in frozen if q["leader_net_170_usd"] is not None]
        book_report = {}
        for method in ("leader", *MODELS, "always_long"):
            book_rows = [q for q in combined if (q["study_phase"],q["method"]) == (phase,method)]
            if not book_rows:
                continue
            # Aggregate P&L by EIA issue before calculating the confidence
            # interval. Two entries in one week are not independent.
            early_by_issue = {q["release_date"]:q for q in book_rows if q["clock"] == "report_day"}
            later = [q for q in book_rows if q["clock"] == "second_session"]
            increments = [early_by_issue.get(q["release_date"], {}).get("net_170_usd",0) for q in later]
            if set(early_by_issue)-{q["release_date"] for q in later}:
                raise ValueError(f"An early decision has no aligned later week in {phase}")
            total = summarize(book_rows)
            book_report[method] = total | dict(windows=len(later), extra_net_170_usd=sum(increments),
                paired_extra_95_usd=block_bootstrap_mean(increments),
                early_trades=sum(bool(q["near_side"]) for q in early_by_issue.values()),
                blocked_early_trades=sum(q["blocked_early_entry"] for q in early_by_issue.values()))
        phases[phase] = dict(coverage=counts[phase],short_coverage=short_counts[phase],
                             results=methods,paired_satellite_vs_curve=comparisons,combined=book_report,
                             archived_leader_on_same_dates=dict(releases=len(live),
                                 trades=sum(bool(q["leader_near_side"]) for q in live),
                                 net_170_usd=sum(q["leader_net_170_usd"] for q in live)))
        print(phase,dict(counts[phase]),{m:(r["trades"],r["net_170_usd"],r["net_340_usd"],
                                            r["stress_net_without_april_2026_usd"]) for m,r in methods.items()})
        print("combined", {m:(q["trades"],q["net_170_usd"],q["net_340_usd"],q["daily_drawdown_usd"])
                           for m,q in book_report.items()})
    sample = mark_check(rows)
    export(f"{STEM}_trades",rows)
    export(f"{STEM}_short_trades",short)
    export(f"{STEM}_combined_trades",combined)
    report = dict(status="schedule_based_sensitivity_on_reviewed_history",
                  clock_assumption="Release time from EIA schedule plus 30 minutes for table availability",
                  phases=phases, trade_bar_check=sample)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
