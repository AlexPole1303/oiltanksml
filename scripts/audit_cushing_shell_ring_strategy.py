"""Compare fixed strategies and walk-forward roof corrections on common CL windows."""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from audit_cushing_shell_ring_weekly import STEM as WEEKLY
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_shell_ring_strategy"
VOLUME_FIELDS = ("original_common_change_kbbl", "ring_common_change_kbbl")
MIN_TRAIN = 12
RIDGE = 1.0


def metrics(rows):
    net = np.array([r["net_usd"] for r in rows])
    wealth = np.r_[0, np.cumsum(net)]
    return dict(windows=len(rows), trades=sum(r["side"] != 0 for r in rows), gross_usd=sum(r["gross_usd"] for r in rows),
                net_usd=float(net.sum()), stressed_net_usd=sum(r["stressed_net_usd"] for r in rows),
                closed_trade_max_drawdown_usd=float(np.max(np.maximum.accumulate(wealth)-wealth)),
                mean_net_usd_ci_95=block_bootstrap_mean(net))


def frozen_rankings():
    groups = defaultdict(list)
    for r in read("cushing_holdout_replay_trades"):
        groups["CL", r["scope"], r["method"]].append(dict(release_date=r["release_date"], side=int(r["near_side"]),
            gross_usd=float(r["gross_usd"]), net_usd=float(r["net_50_usd"]), stressed_net_usd=float(r["net_170_usd"])))
    equity = defaultdict(dict)
    for r in read("cushing_holdout_operator_replay_trades"):
        equity[r["strategy"]][r["release_date"]] = dict(release_date=r["release_date"], side=int(r["instrument_side"]),
            gross_usd=float(r["gross_pnl_usd"]), net_usd=float(r["net_pnl_usd"]), stressed_net_usd=float(r["stressed_net_pnl_usd"]))
    no_trade = lambda issue: dict(release_date=issue, side=0, gross_usd=0., net_usd=0., stressed_net_usd=0.)
    calendar = sorted(equity["always_long_spread"])
    for method, rows in equity.items():
        if not rows.keys() <= set(calendar):
            raise ValueError(f"Equity strategy outside the common calendar: {method}")
        groups["operators_vs_AMLP", "common", method] = [rows.get(issue, no_trade(issue)) for issue in calendar]
    for market, scope in sorted({key[:2] for key in groups}):
        issue_sets = {tuple(sorted(r["release_date"] for r in rows)) for key, rows in groups.items() if key[:2] == (market, scope)}
        dates, = issue_sets
        groups[market, scope, "no_trade"] = [no_trade(issue) for issue in dates]
    scores = [dict(market=market, scope=scope, method=method, **metrics(sorted(rows, key=lambda r: r["release_date"])))
              for (market, scope, method), rows in groups.items()]
    return sorted(scores, key=lambda r: (r["market"], r["scope"], -r["net_usd"], r["method"]))


def correction(history, field, current):
    x = np.array([float(r[field])/1000 for r in history])
    residual = np.array([float(r["target"])-float(r["public_ensemble"]) for r in history])
    # Penalize the one optical coefficient. Public forecasts supply the intercept.
    return float((x @ residual)/(x @ x+RIDGE)) * float(current[field])/1000


def walk_forward():
    by_release = {r["release_date"]: r for r in read(f"{WEEKLY}_weeks")}
    frozen = {r["release_date"]: r for r in read("cushing_holdout_replay_trades")
              if r["scope"] == "all" and r["method"] == "always_long_spread"}
    history, forecasts, trades = [], [], []
    for f in sorted(read("cushing_holdout_replay_forecasts"), key=lambda r: r["release_date"]):
        issue = f["release_date"]
        m = by_release[issue]
        if not int(m["common_resolved_pairs"]):
            continue
        row = f | {k: m[k] for k in VOLUME_FIELDS}
        cutoff = datetime.fromisoformat(f["forecast_cutoff_utc"])
        if cutoff != datetime.fromisoformat(m["cutoff_utc"]):
            raise ValueError(f"Physical inputs use a different release cutoff on {issue}")
        prior = [r for r in history if datetime.fromisoformat(r["target_known_by_utc"]) < cutoff]
        history.append(row)
        if len(prior) < MIN_TRAIN:
            continue
        expected = float(f["public_ensemble"])
        values = dict(public=expected, original_geometry=expected+correction(prior, VOLUME_FIELDS[0], row),
                      aerial_geometry=expected+correction(prior, VOLUME_FIELDS[1], row))
        actual = float(f["target"])
        stamp = max(r["target_known_by_utc"] for r in prior)
        for method, prediction in values.items():
            forecasts.append(dict(release_date=issue, forecast_cutoff_utc=cutoff.isoformat(), method=method, training_releases=len(prior),
                                  latest_training_label_utc=stamp, common_roofs=int(m["common_resolved_pairs"]),
                                  expected_million_bbl=prediction, actual_million_bbl=actual, absolute_error_million_bbl=abs(prediction-actual)))
        if issue not in frozen:
            continue
        trade = frozen[issue]
        if float(trade["published_million_bbl"]) != actual:
            raise ValueError(f"Trade and forecast vintages differ on {issue}")
        for method, prediction in (list(values.items()) + [("published_change", 0.), ("always_long_spread", None), ("no_trade", actual)]):
            side = 1 if prediction is None else int(actual < prediction)-int(actual > prediction)
            gross = side*float(trade["gross_usd"])
            trades.append(dict(release_date=issue, entry_session=trade["entry_session"], exit_session=trade["exit_session"],
                               near_month=trade["near_month"], far_month=trade["far_month"], price_source=trade["source"], method=method,
                               side=side, expected_million_bbl=prediction, actual_million_bbl=actual, gross_usd=gross,
                               net_usd=gross-50*abs(side), stressed_net_usd=gross-170*abs(side)))
    return forecasts, trades


def main():
    ranking = frozen_rankings()
    forecasts, trades = walk_forward()
    export(f"{STEM}_fixed_rules", ranking)
    export(f"{STEM}_forecasts", forecasts)
    export(f"{STEM}_trades", trades)
    by_method = defaultdict(list)
    for r in forecasts:
        by_method[r["method"]].append(r["absolute_error_million_bbl"])
    changes = np.asarray(by_method["aerial_geometry"])-np.asarray(by_method["public"])
    results = dict(minimum_training_releases=MIN_TRAIN, ridge_penalty=RIDGE,
                   comparison_status="exploratory_replay_after_holdout_results_were_known",
                   forecast_metrics={method: dict(releases=len(errors), mae_million_bbl=float(np.mean(errors)))
                                     for method, errors in sorted(by_method.items())},
                   aerial_minus_public_mae_95pct_block_bootstrap_million_bbl=block_bootstrap_mean(changes),
                   trading_metrics={method: metrics([r for r in trades if r["method"] == method]) for method in sorted({r["method"] for r in trades})})
    payload = json.dumps(results, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
