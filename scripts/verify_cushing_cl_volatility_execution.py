"""Reconcile CL model trades against daily marks and the second price provider."""

import json
from collections import defaultdict
from contextlib import closing
from datetime import datetime

from audit_cushing_cl_horizons import paths
from backtest_wti_cushing_spread import COSTS, tick_price
from build_cushing_holdout_market_features import numeric_prices
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_regime_strategy import inputs

STEM = "cushing_cl_volatility_execution_audit"
METHODS = ("coverage_volatility", "public_volatility", "public_volatility_second_session",
           "carry_only_volatility", "always_long_spread")


def audit():
    trade_rows = read("cushing_cl_volatility_trades")
    selected = [r | dict(first_session_number="1") for r in trade_rows if r["study_phase"] == "holdout" and r["method"] in METHODS]
    selected.extend(r | dict(method="public_volatility_second_session") for r in read("cushing_cl_execution_windows_trades")
                    if r["study_phase"] == "holdout" and r["method"] == "public_volatility"
                    and (r["first_session_number"], r["last_session_number"]) == ("2", "5"))
    by_issue = defaultdict(list)
    for row in selected:
        by_issue[row["release_date"]].append(row)
    source = [r for r in inputs() if r["release_date"] in by_issue]
    with closing(read_only(SOURCE)) as db:
        price = numeric_prices(db, {"archive", "ibkr"}, through=max(r["exit_session"] for r in source))
    bar_refs = defaultdict(dict)
    for bar in read("wti_cushing_event_bars"):
        if bar["release_date"] in by_issue:
            key = bar["session_date"], bar["delivery_month"]
            bar_refs[bar["release_date"]][key] = bar
    for anchor, days, times, moves in paths(source):
        issue = anchor["release_date"]
        primary = anchor["source"]
        second = "ibkr" if primary == "archive" else "archive"
        refs = bar_refs[issue]
        legs = [(day, anchor[leg]) for day in days for leg in ("near_month", "far_month")]
        alternative = None
        if all(refs[leg][f"{second}_present"] == "True" for leg in legs):
            alternative = [tick_price(price(second, day, month, refs[day, month][f"{second}_reference"])) for day, month in legs]
        for row in by_issue[issue]:
            start = int(row["first_session_number"])-1
            if (row["entry_session"] != days[start] or row["exit_session"] != days[-1]
                    or any(row[k] != anchor[k] for k in ("near_month", "far_month", "source"))):
                raise ValueError(f"Window differs from replay on {issue}")
            if not datetime.fromisoformat(row["latest_training_exit"]) < datetime.fromisoformat(row["decision_at_utc"]) < times[start]:
                raise ValueError(f"Invalid training or entry order on {issue}")
            side = int(row["near_side"])
            trajectory = [0]*start+[side*(move-moves[start]) for move in moves[start:]]
            gross = trajectory[-1]
            alt_gross = (10*side*(alternative[-2]-alternative[-1]-alternative[2*start]+alternative[2*start+1])
                         if alternative is not None else None)
            if int(row["gross_usd"]) != gross or any(int(row[f"net_{c}_usd"]) != gross-c*abs(side) for c in COSTS):
                raise ValueError(f"P&L does not reconcile on {issue}")
            # Charge half the round-trip cost at entry and the rest at exit.
            path = [value-(170 if n == len(days)-1 else 85 if n >= start else 0)*abs(side) for n, value in enumerate(trajectory)]
            yield dict(release_date=issue, method=row["method"], near_month=row["near_month"], far_month=row["far_month"],
                       near_side=side, entry_session=days[start], exit_session=days[-1], primary_source=primary,
                       gross_usd=gross, net_170_usd=path[-1], adverse_settlement_move_usd=min(trajectory),
                       net_170_entry_delayed_one_session_usd=side*(moves[-1]-moves[start+1])-170*abs(side),
                       net_170_exit_advanced_one_session_usd=trajectory[-2]-170*abs(side),
                       best_settlement_move_usd=max(trajectory),
                       secondary_source=second if alt_gross is not None else None, secondary_gross_usd=alt_gross,
                       **{f"session_{n}_net_usd": value for n, value in enumerate(path, 1)})


def summarize(rows):
    equity, peak, drawdown = 0, 0, 0
    traded = [r for r in rows if r["near_side"]]
    secondary = [r for r in traded if r["secondary_source"]]
    months = defaultdict(int)
    for r in rows:
        for n in range(1, 6):
            level = equity+r[f"session_{n}_net_usd"]
            peak = max(peak, level)
            drawdown = max(drawdown, peak-level)
        equity += r["net_170_usd"]
        months[r["release_date"][:7]] += r["net_170_usd"]
    best_month = max(months, key=months.__getitem__)
    return dict(windows=len(rows), trades=len(traded), net_170_usd=equity, daily_settlement_drawdown_170_usd=drawdown,
                net_170_entry_delayed_one_session_usd=sum(r["net_170_entry_delayed_one_session_usd"] for r in rows),
                net_170_exit_advanced_one_session_usd=sum(r["net_170_exit_advanced_one_session_usd"] for r in rows),
                best_release_month=best_month, best_release_month_net_170_usd=months[best_month],
                net_170_excluding_best_release_month_usd=equity-months[best_month],
                worst_adverse_settlement_move_usd=min(r["adverse_settlement_move_usd"] for r in traded),
                trades_with_both_providers=len(secondary),
                max_abs_pnl_difference_usd=max((abs(r["gross_usd"]-r["secondary_gross_usd"]) for r in secondary), default=None),
                largest_wins=[{k:r[k] for k in ("release_date", "gross_usd", "secondary_gross_usd", "adverse_settlement_move_usd")}
                              for r in sorted(traded, key=lambda row: row["gross_usd"], reverse=True)[:3]])


def main():
    rows = list(audit())
    groups = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    calendar = [r["release_date"] for r in groups[METHODS[0]]]
    if calendar != sorted(set(calendar)) or any([r["release_date"] for r in g] != calendar for g in groups.values()):
        raise ValueError("Missing or duplicated audited windows")
    export(f"{STEM}_trades", rows)
    report = {method: summarize(group) for method, group in groups.items()}
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
