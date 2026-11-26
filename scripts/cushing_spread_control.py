"""Replay curve-only and inventory-conditioned spread decisions with matched contracts, clocks and costs."""

import argparse
import json
from collections import defaultdict
from datetime import datetime

import backtest_cushing_balance_spread as study
import pandas as pd
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_inventory_network import MODEL_DIR, save_json
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_volatility_interactions import features
from fit_cushing_volatility_neighbors import volatility_forecast
from threadpoolctl import threadpool_limits

LEDGER = MODEL_DIR / "spread_control_windows.csv"
SUMMARY = MODEL_DIR / "spread_control_evaluation.json"


def signals(row, history, *, with_actual=False):
    actual, _surprise, carry, change, _interaction, _optical, _other = row["x"]
    market = row | {
        "x": (actual if with_actual else 0.0, 0.0, carry, change, 0.0, 0.0, 0.0)
    }
    market["x"] = features(market)[3]
    past = [r for r in history if r["exit_at"] < row["decision"]][-HISTORY:]
    history.append(market)
    if len(past) < MIN_WINDOWS:
        return None
    return volatility_forecast(past, market), past[-1]["exit_at"]


MARKET = MODEL_DIR / "spread_control_market.jsonl"
CLOCKS = ("decision", "entry", "exit_at", "next_report_clock")
FIELDS = (
    "release_date",
    "study_phase",
    "near_month",
    "far_month",
    "volatility_usd",
    "next_issue",
    "next_report_clock_basis",
    "last_before_next_release",
    "spread_path_usd",
    "long_gross",
    "alternate_source",
    "alternate_long_gross_usd",
)


def export_market():
    routes = study.early_intervals()
    rows = []
    for row, _ in inputs():
        item = {key: row[key] for key in FIELDS}
        item.update({k: row[k].isoformat() if row[k] else None for k in CLOCKS})
        item["stamps"] = [t.isoformat() for t in row["stamps"]]
        # Only the realized first release, carry and carry change enter either control.
        actual, _, carry, change, *_ = row["x"]
        item["x"] = [actual, 0, carry, change, 0, 0, 0]
        route = routes.get(row["release_date"])
        item["early"] = (
            [route[0].isoformat(), route[1].isoformat(), route[2]] if route else None
        )
        rows.append(
            json.dumps(
                item, ensure_ascii=False, allow_nan=False, default=lambda v: v.item()
            )
        )
    MARKET.write_text("\n".join(rows) + "\n")


def read_market():
    rows = []
    routes = {}
    for line in MARKET.read_text().splitlines():
        item = json.loads(line)
        item.update(
            {k: datetime.fromisoformat(item[k]) if item[k] else None for k in CLOCKS}
        )
        item["stamps"] = [datetime.fromisoformat(v) for v in item["stamps"]]
        early = item.pop("early")
        if early:
            routes[item["release_date"]] = (
                datetime.fromisoformat(early[0]),
                datetime.fromisoformat(early[1]),
                early[2],
            )
        if (
            not item["decision"]
            < item["entry"]
            == item["stamps"][0]
            < item["stamps"][-1]
            == item["exit_at"]
        ):
            raise ValueError("Unordered market input clocks")
        rows.append(item)
    return rows, routes


def main():
    published = pd.read_csv(
        MODEL_DIR / "balance_spread_trades.csv",
        dtype={"near_month": str, "far_month": str},
    )
    one = published[
        (published.rule == "updated")
        & (published.hurdle_usd == 170)
        & (published.entry_clock == "second")
    ].set_index("release_date")
    adaptive = published[published.rule == "adaptive"].set_index("release_date")
    rows, routes = read_market()
    estimates, strict, alternatives, history, actual_history = [], [], [], [], []
    for row in rows:
        signal = signals(row, history)
        actual_signal = signals(row, actual_history, with_actual=True)
        day = row["release_date"]
        if day not in one.index:
            continue
        if signal is None:
            raise ValueError("Incomplete curve-only history")
        curve, known = signal
        ref = one.loc[day]
        if (
            row["near_month"] != ref.near_month
            or row["far_month"] != ref.far_month
            or row["entry"] != datetime.fromisoformat(ref.entry_at_utc)
        ):
            raise ValueError("Different trading instruments or entry marks")
        early = row["last_before_next_release"]
        first_close = min(early + 1, 3) if early in (1, 2) else 3
        c = trade(row, "curve", curve, first_close, training_at=known)
        a_only = trade(
            row,
            "realized_eia",
            actual_signal[0],
            first_close,
            training_at=actual_signal[1],
        )
        f = trade(
            row,
            "inventory",
            ref.updated_spread_signal_usd,
            first_close,
            training_at=datetime.fromisoformat(ref.latest_training_exit_utc),
        )
        if (
            c["exit_at_utc"] != f["exit_at_utc"]
            or c["entry_at_utc"] != f["entry_at_utc"]
        ):
            raise ValueError("Unmatched counterfactual")
        strict.append(
            {
                "release_date": day,
                "near_month": row["near_month"],
                "far_month": row["far_month"],
                "decision_at_utc": c["decision_at_utc"],
                "entry_at_utc": c["entry_at_utc"],
                "exit_at_utc": c["exit_at_utc"],
                "curve_signal": curve,
                "inventory_signal": ref.updated_spread_signal_usd,
                "curve_side": c["near_side"],
                "inventory_side": f["near_side"],
                "curve_net": c["net_170_usd"],
                "inventory_net": f["net_170_usd"],
                "actual_only_signal": actual_signal[0],
                "actual_only_side": a_only["near_side"],
                "actual_only_net": a_only["net_170_usd"],
            }
        )
        # Independent adaptive control. Both second-session books receive the identical market-only forecast.
        # The extension forecast in the published experiment itself uses only the CL curve.
        extension = ref.extension_usd
        end = (
            3 if pd.notna(extension) and c["near_side"] * extension > 0 else first_close
        )
        rec = trade(row, "curve", curve, end, training_at=known)
        rec["net_340_usd"] = rec["gross_usd"] - 340 * abs(rec["near_side"])
        for rule in ("previous", "updated"):
            estimates.append(
                rec
                | {
                    "rule": rule,
                    "hurdle_usd": 170,
                    "signal_usd": curve,
                    "first_interval_forecast_usd": None,
                    "third_mark": row["stamps"][1].isoformat(),
                    "second_to_third_long_usd": row["spread_path_usd"][1],
                }
            )
        # Attribution on the exact observed adaptive entry and exit, including windows without an observed position.
        a = adaptive.loc[day]
        start = datetime.fromisoformat(a.entry_at_utc)
        stop = datetime.fromisoformat(a.exit_at_utc)
        end = row["stamps"].index(stop)
        if start == row["entry"]:
            earlier = 0
        else:
            route = routes[day]
            if start != route[0] or route[1] != row["entry"]:
                raise ValueError("Incompatible adaptive entry")
            earlier = route[2]
        delta = earlier + row["spread_path_usd"][end]
        if a.near_side * delta - 170 * abs(a.near_side) != a.net_170_usd:
            raise ValueError("Failed to reconcile adaptive record")
        side = c["near_side"]
        alternatives.append(
            {
                "release_date": day,
                "adaptive_entry": a.entry_at_utc,
                "adaptive_exit": a.exit_at_utc,
                "adaptive_side": int(a.near_side),
                "adaptive_net": int(a.net_170_usd),
                "curve_on_adaptive_clocks_side": side,
                "curve_on_adaptive_clocks_net": side * delta - 170 * abs(side),
                "actual_on_adaptive_clocks_side": a_only["near_side"],
                "actual_on_adaptive_clocks_net": a_only["near_side"] * delta
                - 170 * abs(a_only["near_side"]),
            }
        )
    groups = defaultdict(list)
    for r in study.entry_variants(estimates, routes):
        groups[r["rule"], 170, r["entry_clock"]].append(r)
    control = list(study.adaptive_book(groups))
    if list(one.index) != [r["release_date"] for r in strict] or list(
        adaptive.index
    ) != list(one.index):
        raise ValueError("144-window comparison changed dates")
    book = pd.DataFrame(strict).merge(pd.DataFrame(alternatives), validate="one_to_one")
    if list(book.release_date) != [r["release_date"] for r in control]:
        raise ValueError("Independent adaptive dates changed")
    book["curve_adaptive_net"] = [r["net_170_usd"] for r in control]
    book["curve_adaptive_side"] = [r["near_side"] for r in control]
    book["curve_adaptive_entry"] = [r["entry_at_utc"] for r in control]
    book["curve_adaptive_exit"] = [r["exit_at_utc"] for r in control]
    book.to_csv(LEDGER, index=False)
    specs = (
        ("fixed", "inventory", "curve"),
        ("fixed_forecast_only", "inventory", "actual_only"),
        ("same_clocks_forecast_only", "adaptive", "actual_on_adaptive_clocks"),
        ("same_adaptive_clocks", "adaptive", "curve_on_adaptive_clocks"),
        ("independent_adaptive", "adaptive", "curve_adaptive"),
    )
    report = {}
    for name, left, right in specs:
        a, b = book[left + "_side"], book[right + "_side"]
        av, bv = book[left + "_net"], book[right + "_net"]
        delta = av - bv
        report[name] = {
            "windows": len(book),
            "inventory_trades": int(a.ne(0).sum()),
            "curve_trades": int(b.ne(0).sum()),
            "inventory_net": int(av.sum()),
            "curve_net": int(bv.sum()),
            "net_difference": int(delta.sum()),
            "changed_sides": int((a != b).sum()),
            "opposite": int((a * b == -1).sum()),
            "inventory_only": int(((a != 0) & (b == 0)).sum()),
            "curve_only": int(((a == 0) & (b != 0)).sum()),
            "difference_on_changed_sides": int(delta[a != b].sum()),
            "difference_on_same_sides": int(delta[a == b].sum()),
            "mean_difference_per_window_95": block_bootstrap_mean(delta),
            "by_year": {
                str(y): [int(av[ix].sum()), int(bv[ix].sum())]
                for y, ix in book.groupby(book.release_date.str[:4]).groups.items()
            },
        }
    for prefix, start, end in [
        ("inventory", "entry_at_utc", "exit_at_utc"),
        ("curve", "entry_at_utc", "exit_at_utc"),
        ("curve_adaptive", "curve_adaptive_entry", "curve_adaptive_exit"),
    ]:
        active = book[book[prefix + "_side"] != 0]
        if any(x >= y for x, y in zip(active[end].iloc[:-1], active[start].iloc[1:])):
            raise ValueError("Overlapping standalone positions")
    save_json(SUMMARY, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-inputs", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.export_inputs:
            export_market()
        main()
