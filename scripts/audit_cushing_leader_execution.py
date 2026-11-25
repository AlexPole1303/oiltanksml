"""Reprice frozen CL decisions from dated daily marks and aligned trade bars."""

import json
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta

from audit_cushing_ibkr_intraday import read_packets
from backtest_wti_cushing_spread import tick_price
from build_cushing_holdout_market_features import numeric_prices
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_leader_execution"
MODELS = {
    "weighted_optical": ("cushing_volatility_neighbors_trades", "outer_supplement:volatility"),
    "optical": ("cushing_outer_capture_nowcast_trades", "outer_supplement_conditional"),
    "public": ("cushing_cl_conditional_response_trades", "inventory_conditional"),
    "curve": ("cushing_cl_volatility_interactions_trades", "carry_volatility_context"),
}
REFERENCE = "weighted_optical"
PROVIDERS = ("archive", "ibkr", "official")
SCENARIOS = {"original": (2, 5), "entry_one_later": (3, 5), "exit_one_earlier": (2, 4), "both": (3, 4)}
COSTS = (170, 340)
BAR_SCENARIOS = ("ibkr_bar_exact_5m", "ibkr_bar_exact_15m", "ibkr_bar_exact_30m")


def anchors():
    out = defaultdict(dict)
    for name, (stem, method) in MODELS.items():
        for q in read(stem):
            if q["study_phase"] != "holdout" or q["method"] != method:
                continue
            issue = q["release_date"]
            if name in out[issue]:
                raise ValueError(f"Duplicate frozen signal on {issue}/{name}")
            out[issue][name] = q
    if any(found.keys() != MODELS.keys() for found in out.values()):
        raise ValueError("The strategies do not share the same frozen calendar")
    return out


def settlement_paths(signals):
    group = defaultdict(dict)
    for r in read("wti_cushing_event_bars"):
        issue = r["release_date"]
        session = int(r["session_number"])
        if issue not in signals or session < 2:
            continue
        key = session, r["leg"]
        if key in group[issue]:
            raise ValueError(f"Duplicate settlement leg on {issue}/{key}")
        group[issue][key] = r
    if group.keys() != signals.keys():
        raise ValueError("A decision has no dated settlement rows")
    through = max(r["session_date"] for a in group.values() for r in a.values())
    with closing(read_only(SOURCE)) as db:
        price = numeric_prices(db, PROVIDERS, through=through)
    dates = (2, 3, 4, 5)
    for issue, model in sorted(signals.items()):
        legs = [group[issue][n, side] for n in dates for side in ("near", "far")]
        anchor = model[REFERENCE]
        for row in model.values():
            if any(row[k] != anchor[k] for k in ("near_month", "far_month", "decision_at_utc", "entry_at_utc", "exit_at_utc")):
                raise ValueError(f"The dated strategy windows differ on {issue}")
        mid_months = [r["delivery_month"] for r in legs]
        if mid_months != [anchor["near_month"], anchor["far_month"]] * 4:
            raise ValueError(f"The month rolled within {issue}")
        stamps = tuple(datetime.fromisoformat(r["settlement_reference_time"]) for r in legs)
        if (stamps[::2] != stamps[1::2] or not datetime.fromisoformat(anchor["decision_at_utc"]) < stamps[0] < stamps[-1]
                or stamps[0] != datetime.fromisoformat(anchor["entry_at_utc"])
                or stamps[-1] != datetime.fromisoformat(anchor["exit_at_utc"])):
            raise ValueError(f"The settlement schedule differs on {issue}")
        paths = {}
        for source in PROVIDERS:
            if not all(r[f"{source}_present"] == "True" for r in legs):
                continue
            ticks = [tick_price(price(source, r["session_date"], r["delivery_month"], r[f"{source}_reference"])) for r in legs]
            paths[source] = tuple(10 * (a - b) for a, b in zip(ticks[::2], ticks[1::2], strict=True))
        primary = next((k for k in PROVIDERS[:2] if k in paths), None)
        if primary is None:
            raise ValueError(f"No full four-session source on {issue}")
        coded = paths[primary]
        for name, row in model.items():
            side = int(row["near_side"])
            for n, spread in zip(dates, coded, strict=True):
                expected = side * (spread - coded[0]) - (170 if n == 5 else 85) * abs(side)
                if expected != int(row[f"session_{n}_net_170_usd"]):
                    raise ValueError(f"Historical price path fails to reconcile on {issue}/{name}/session {n}")
        yield issue, model, stamps[::2], primary, paths


def from_ibkr(book, stamp, near, far, minutes):
    width = timedelta(minutes=minutes)
    bars = [book.get((month, minutes * 60, stamp - width)) for month in (near, far)]
    if not all(bars):
        return None
    return 10 * (bars[0][0] - bars[1][0]), bars


def replay():
    book, _ = read_packets()
    signals = anchors()
    for issue, models, stamps, primary, paths in settlement_paths(signals):
        reference = models[REFERENCE]
        near, far = reference["near_month"], reference["far_month"]
        for name, (a, b) in SCENARIOS.items():
            for m, r in models.items():
                side = int(r["near_side"])
                trusted = paths[primary]
                gross = side * (trusted[b - 2] - trusted[a - 2])
                if name == "original" and (gross != int(r["gross_usd"]) or gross - 170 * abs(side) != int(r["net_170_usd"])):
                    raise ValueError(f"The frozen contractual payoff differs on {issue}/{m}")
                yield dict(release_date=issue, method=m, scenario=name, price_basis=primary, near_month=near, far_month=far,
                    entry_at_utc=stamps[a - 2].isoformat(), exit_at_utc=stamps[b - 2].isoformat(), near_side=side,
                    gross_usd=gross, **{f"net_{cost}_usd":gross - cost * abs(side) for cost in COSTS},
                    original_mark_gross_usd=int(r["gross_usd"]),
                    **{f"{s}_gross_usd":side * (paths[s][b - 2] - paths[s][a - 2]) if s in paths else None for s in PROVIDERS},
                    **bar_lineage())
        # The trade bar must close at the same timestamp as the settlement
        # reference in every sensitivity row. No earlier-print filling.
        for minutes in (5, 15, 30):
            snapshots = [from_ibkr(book, t, near, far, minutes) for t in (stamps[0], stamps[-1])]
            if not all(snapshots):
                continue
            entry, exit = snapshots
            move = exit[0] - entry[0]
            marks = bar_lineage(entry[1], exit[1])
            for m, q in models.items():
                side = int(q["near_side"])
                gross = side * move
                yield dict(release_date=issue, method=m, scenario=f"ibkr_bar_exact_{minutes}m", price_basis="ibkr_last_trade_bar",
                    near_month=near, far_month=far, entry_at_utc=stamps[0].isoformat(), exit_at_utc=stamps[-1].isoformat(), near_side=side,
                    gross_usd=gross, **{f"net_{cost}_usd":gross - cost * abs(side) for cost in COSTS},
                    original_mark_gross_usd=int(q["gross_usd"]), archive_gross_usd=None, ibkr_gross_usd=None, official_gross_usd=None,
                    **marks)


def bar_lineage(entry=(None, None), exit=(None, None)):
    return {f"{leg}_{phase}_{field}": bar[index] if bar is not None else None
            for phase, pair in (("entry", entry), ("exit", exit)) for leg, bar in zip(("near", "far"), pair, strict=True)
            for index, field in enumerate(("ticks", "volume", "source_sha256"))}


def tally(rows):
    trades = [q for q in rows if q["near_side"]]
    by_year, by_month = defaultdict(int), defaultdict(int)
    for q in rows:
        by_year[q["release_date"][:4]] += q["net_170_usd"]
        by_month[q["release_date"][:7]] += q["net_170_usd"]
    net = sum(q["net_170_usd"] for q in trades)
    gross = sum(q["gross_usd"] for q in trades)
    equity = peak = dd = 0
    for q in trades:
        equity += q["net_170_usd"]
        peak = max(peak, equity)
        dd = max(dd, peak - equity)
    top_three = sorted((q["net_170_usd"] for q in trades if q["net_170_usd"] > 0), reverse=True)[:3]
    by_cost = {f"net_{c}_usd":sum(q[f"net_{c}_usd"] for q in trades) for c in COSTS}
    return dict(windows=len(rows), trades=len(trades), gross_usd=gross, **by_cost, closed_trade_drawdown_170_usd=dd,
        cost_break_even_usd=gross / len(trades) if trades else None, net_170_without_top_three_usd=net - sum(top_three),
        net_170_without_april_2026_usd=net - by_month["2026-04"], net_170_by_year=dict(by_year))


def main():
    rows = list(replay())
    group = defaultdict(list)
    for q in rows:
        group[q["scenario"], q["method"]].append(q)
    summary = dict(status="historical_frozen_decisions", scenarios={}, ibkr_same_clock={})
    for scenario in (*SCENARIOS, *BAR_SCENARIOS):
        ref = group[scenario, REFERENCE]
        section = summary["scenarios" if scenario in SCENARIOS else "ibkr_same_clock"][scenario] = {}
        for m in MODELS:
            chosen = group[scenario, m]
            if [q["release_date"] for q in chosen] != [q["release_date"] for q in ref]:
                raise ValueError(f"Different stress sample on {scenario}/{m}")
            direction = [q["net_170_usd"] - r["net_170_usd"] for q, r in zip(chosen, ref, strict=True)]
            section[m] = tally(chosen) | dict(delta_vs_weighted_170_usd=sum(direction),
                paired_delta_per_window_95_usd=block_bootstrap_mean(direction))
            if scenario in BAR_SCENARIOS:
                used = [q for q in chosen if q["near_side"]]
                gap = [q["gross_usd"] - q["original_mark_gross_usd"] for q in used]
                section[m].update(paired_original_marks_gross_usd=sum(q["original_mark_gross_usd"] for q in used),
                    paired_gross_change_usd=sum(gap), largest_absolute_paired_change_usd=max(map(abs, gap), default=None))
    live = group["original", REFERENCE]
    cash = [q for q in live if q["near_side"]]
    comparison = [q for q in cash if q["archive_gross_usd"] is not None and q["ibkr_gross_usd"] is not None]
    summary["settlement_source_comparison"] = dict(windows_with_both=sum(q["archive_gross_usd"] is not None and
            q["ibkr_gross_usd"] is not None for q in live), trades_with_both=len(comparison),
            april_trades_with_both=sum(q["release_date"].startswith("2026-04") for q in comparison),
            trades_official_available=sum(q["official_gross_usd"] is not None for q in cash),
            largest_abs_provider_delta_usd=max((abs(q["archive_gross_usd"] - q["ibkr_gross_usd"]) for q in comparison), default=None))
    export(f"{STEM}_trades", rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode())
    for scenario, q in (summary["scenarios"] | summary["ibkr_same_clock"]).items():
        print(scenario, {m: (v["trades"], v["net_170_usd"], v["net_340_usd"]) for m, v in q.items()})


if __name__ == "__main__":
    main()
