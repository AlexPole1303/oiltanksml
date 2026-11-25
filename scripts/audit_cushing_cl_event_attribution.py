"""Compare fixed CL decisions with same-length windows before the EIA release."""

import json
from bisect import bisect_left
from contextlib import closing
from datetime import datetime

from backtest_wti_cushing_spread import full_sessions, settlement, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_delayed_calendar import replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_event_attribution"
METHODS = ("curve", "inventory_conditional", "optical_conditional")
COST = 170


def matched_prices(cases, clocks):
    with closing(read_only(SOURCE)) as db:
        end = cases[-1]["exit_at"].date().isoformat()
        sessions = full_sessions(db, end)
        sources = {source: {(day, month): tick_price(value) for day, month, _, value in
                            price_rows(db, source, numeric_through=end)}
                   for source in ("archive", "ibkr")}
    release_clocks = sorted(datetime.fromisoformat(q["scheduled_table_release_at_utc"]) for q in clocks.values()
                            if q["scheduled_table_release_at_utc"])
    for r in cases:
        issue = r["release_date"]
        event = clocks[issue]["scheduled_table_release_at_utc"]
        if not event:
            yield issue, None, "missing_report_clock"
            continue
        post_start = bisect_left(sessions, r["entry"].date().isoformat())
        post = sessions[post_start:post_start+4]
        if len(post) != 4 or settlement(post[0]) != r["entry"] or settlement(post[-1]) != r["exit_at"]:
            raise ValueError(f"The actual holding period differs on {issue}")
        pre_end = bisect_left(sessions, issue)
        pre = sessions[pre_end-4:pre_end]
        if len(pre) != 4:
            yield issue, None, "insufficient_prior_sessions"
            continue
        report = datetime.fromisoformat(event)
        starts = [settlement(days[0]) for days in (pre, post)]
        ends = [settlement(days[-1]) for days in (pre, post)]
        if not starts[0] < ends[0] < report < r["decision"] < starts[1] < ends[1]:
            raise ValueError(f"Invalid market or report clock on {issue}")
        other_releases = [sum(start < clock <= end for clock in release_clocks)
                          for start, end in zip(starts, ends, strict=True)]
        months = (r["near_month"], r["far_month"])
        keys = [(day, month) for day in pre + post for month in months]
        available = [name for name, book in sources.items() if all(k in book for k in keys)]
        if not available:
            yield issue, None, "no_common_price_source"
            continue
        outcomes = {}
        for name in available:
            book = sources[name]
            spreads = [10 * (book[d, months[0]] - book[d, months[1]]) for d in pre + post]
            outcomes[name] = (spreads[3] - spreads[0], spreads[7] - spreads[4])
        source = available[0]
        before, after = outcomes[source]
        if after != r["long_gross"]:
            raise ValueError(f"Same-contract outcome differs from original P&L on {issue}")
        crosscheck = available[1] if len(available) == 2 else None
        if crosscheck and outcomes[crosscheck] != (before, after):
            raise ValueError(f"Common sources disagree on {issue}")
        yield issue, dict(release_at_utc=report.isoformat(), pre_start_utc=starts[0].isoformat(),
                          pre_end_utc=ends[0].isoformat(), post_start_utc=starts[1].isoformat(),
                          post_end_utc=ends[1].isoformat(), price_source=source, alternate_source=crosscheck,
                          pre_other_eia_releases=other_releases[0], post_other_eia_releases=other_releases[1],
                          pre_long_move_usd=before, post_long_move_usd=after), None


def summarize(sample):
    groups = {method: [r for r in sample if r["method"] == method] for method in METHODS}
    curve = groups[METHODS[0]]
    result = {}
    for method, group in groups.items():
        pre = [r["pre_net_usd"] for r in group]
        post = [r["post_net_usd"] for r in group]
        difference = [y - x for x, y in zip(pre, post, strict=True)]
        advantage_pre = [x - q["pre_net_usd"] for x, q in zip(pre, curve, strict=True)]
        advantage_post = [x - q["post_net_usd"] for x, q in zip(post, curve, strict=True)]
        excess = [x - y for x, y in zip(advantage_post, advantage_pre, strict=True)]
        result[method] = dict(windows=len(group), trades=sum(bool(r["near_side"]) for r in group),
            pre_net_usd=sum(pre), post_net_usd=sum(post), post_minus_pre_usd=sum(difference),
            post_minus_pre_per_window_95_usd=block_bootstrap_mean(difference),
            pre_advantage_vs_curve_usd=sum(advantage_pre), post_advantage_vs_curve_usd=sum(advantage_post),
            advantage_difference_per_window_95_usd=block_bootstrap_mean(excess),
            cross_checked_windows=sum(bool(r["alternate_source"]) for r in group))
    return result


def main():
    references = {(q["release_date"], q["method"]): q for q in read("cushing_cl_conditional_response_trades")
                  if q["method"] in METHODS}
    _, _, cases = market_calendar()
    used = [r for r in cases if (r["release_date"], METHODS[0]) in references]
    excluded, results = [], []
    for r, (issue, prices, reason) in zip(used, matched_prices(used, indexed("eia_release_clock")), strict=True):
        if r["release_date"] != issue:
            raise ValueError(f"Unexpected release date {issue}")
        if reason:
            excluded.append(dict(release_date=issue, study_phase=r["study_phase"], reason=reason))
            continue
        for method in METHODS:
            ref = references[issue, method]
            side = int(ref["near_side"])
            before = side * prices["pre_long_move_usd"] - COST * abs(side)
            after = side * prices["post_long_move_usd"] - COST * abs(side)
            if after != int(ref["net_170_usd"]) or any(ref[k] != r[k] for k in ("near_month", "far_month")):
                raise ValueError(f"Original side or price path differs on {issue}/{method}")
            results.append({k: r[k] for k in ("release_date", "study_phase", "near_month", "far_month", "volatility_usd")} |
                           prices | dict(method=method, near_side=side, pre_net_usd=before, post_net_usd=after))
    report = dict(status="retrospective_attribution_not_tradable", round_trip_cost_usd=COST, excluded=excluded, phases={})
    for phase in PHASES:
        group = [r for r in results if r["study_phase"] == phase]
        scopes = {"all_matched": group,
                  "no_other_eia_release": [r for r in group if not (r["pre_other_eia_releases"] or r["post_other_eia_releases"])],
                  "same_other_release_count": [r for r in group if r["pre_other_eia_releases"] == r["post_other_eia_releases"]]}
        report["phases"][phase] = {k: summarize(v) for k, v in scopes.items()}
        for scope, output in report["phases"][phase].items():
            print(phase, scope, {m: (r["windows"], r["pre_net_usd"], r["post_net_usd"]) for m, r in output.items()})
    export(f"{STEM}_windows", results)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
