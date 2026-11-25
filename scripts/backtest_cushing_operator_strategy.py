"""Replay the frozen operator-versus-AMLP research strategies on open phases."""
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta
from fractions import Fraction

import numpy as np

from build_market_event_panel import CORE
from build_operator_development_inputs import group, one_per_key
from build_study_execution_inputs import BENCHMARK, VENUE
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

STEM = "cushing_operator_backtest"
PHASES = ("development", "validation")
SYMBOLS = (*CORE, BENCHMARK)
GROSS_NOTIONAL = 100_000
MISSING_SPREAD = 25/10_000
MISSING_BORROW = 0.05
RETURN_HURDLE = 0.0025
NET_RETURN_HURDLE = 0.0015
RIDGE_PENALTY = 0.5
STRATEGIES = ("always_long_spread", "eia_sign", "physical_sign", "physical_geometry_filter", "ridge_eia", "ridge_eia_optical",
              "ridge_directional_net")
MAX_QUOTE_DATE = "2023-12-31"


def market_sources(through=MAX_QUOTE_DATE):
    calendar_through = max("2024-01-12", (date.fromisoformat(through)+timedelta(days=12)).isoformat())
    with closing(read_only(SOURCE)) as db:
        days = [r[0] for r in db.execute("""SELECT session_date FROM equity_session_calendar WHERE venue=? AND
            session_date BETWEEN '2016-01-01' AND ? ORDER BY session_date""", (VENUE,calendar_through))]
        params = (*SYMBOLS, through)
        quotes = {(ticker, day): dict(open=opening, close=close, basis=basis, segment=segment) for
                  ticker, day, opening, close, basis, segment in db.execute(f"""
            SELECT instrument,session_date,open,close,price_basis,listing_segment FROM market_study_quote
            WHERE instrument IN ({','.join('?' for _ in SYMBOLS)}) AND currency='USD' AND session_date<=?""", params)}
    return days, {day: index for index, day in enumerate(days)}, quotes


def model_input(row, cost, envelope, current, prior, wti):
    if (cost["entry_session"], cost["fifth_session"]) != (row["entry_session"], row["fifth_session"]):
        raise ValueError(f"Execution cost window differs for {row['release_date']}/{row['instrument']}")
    magnitude = float(row["asset_capture_change_kbbl"])
    area_factor = float(row["asset_baseline_b04_area_m2"])/BARREL_M3/1000
    halfwidth = (float(envelope["loss_max_kbbl"])-float(envelope["loss_min_kbbl"]))/2
    return dict(row, magnitude_kbbl=magnitude, physical_height_change_m=magnitude/area_factor,
        loss_excludes_zero=envelope["loss_envelope_excludes_zero"] == "True", eia_current_kbbl=current,
        eia_features=(current/1000, prior/1000, wti), optical_features=(magnitude/area_factor,
        halfwidth/area_factor, float(row["asset_observed_area_fraction"])), cost=cost)


def event_inputs():
    eia = one_per_key(read_csv("sentinel2_space_time_coverage_eia_comparison"), "week_ending", "release_date")
    oil = one_per_key((r for r in read_csv("oil_market_context") if r["phase"] == "pre_release"
                       and r["release_date"] <= MAX_QUOTE_DATE), "release_date")
    envelopes = one_per_key(read_csv("terminal_change_repeatability_operator_envelopes"), "release_date", "instrument")
    costs = one_per_key((r for r in read_csv("study_execution_inputs") if r["study_phase"] in PHASES), "release_date", "instrument")
    data = defaultdict(list)
    excluded = Counter()
    for r in read_csv("study_operator_signal_inputs"):
        if r["study_phase"] not in PHASES or r["instrument"] not in CORE or not r["asset_capture_change_kbbl"]:
            continue
        phase, issue, ticker = (r[k] for k in ("study_phase", "release_date", "instrument"))
        if r["phase_window_eligible"] != "True" or r["matched_usd_quote_window"] != "True" or r["fifth_session"] > MAX_QUOTE_DATE:
            excluded[phase, "window_or_prices_unavailable"] += 1
            continue
        if r["asset_measured_terminals"] != r["asset_measured_terminals_with_documented_pair_dates"]:
            excluded[phase, "measured_asset_dates_or_scope_incomplete"] += 1
            continue
        if r["terminal_relation_boundary_in_window"] == "1":
            excluded[phase, "terminal_relationship_changes_within_window"] += 1
            continue
        ref = eia[r["week_ending"], issue]
        if not ref["eia_first_release_change_kbbl"] or not ref["prior_kbbl"]:
            excluded[phase, "missing_eia_context"] += 1
            continue
        features = oil.get((issue,), {})
        if features and datetime.fromisoformat(features["cutoff_utc"]) > datetime.fromisoformat(r["decision_at_utc"]):
            raise ValueError(f"Oil context after decision for {issue}")
        eia_current, eia_prior = (float(ref[k]) for k in ("eia_first_release_change_kbbl", "prior_kbbl"))
        wti = float(features["cl1_minus_cl2"]) if features.get("cl1_minus_cl2") else None
        data[issue].append(model_input(r, costs[issue,ticker], envelopes[issue,ticker], eia_current, eia_prior, wti))
    return data, excluded


def corporate_inputs(phases=PHASES):
    refs = one_per_key(read_csv("study_equity_action_events"), "event_id")
    windows = group((r for r in read_csv("study_equity_window_actions") if r["study_phase"] in phases
                     and r["instrument"] in SYMBOLS), "release_date", "instrument")
    fx = one_per_key((r for r in read_csv("canadian_operator_cash_fx_comparison") if r["instrument"] == "ENB"),
                     "instrument", "record_date")
    splits = defaultdict(list)
    for row in refs.values():
        if row["event_type"] == "share_split" and row["instrument"] in SYMBOLS:
            terms = json.loads(row["issuer_terms_json"])
            splits[row["instrument"]].append((row["first_relevant_date"], Fraction(terms["old_shares"], terms["new_shares"])))
    return refs, windows, fx, splits


def cash_in_holding(issue, symbol, entry, end, basis, corporate):
    refs, windows, fx, _ = corporate
    cash = []
    for link in windows[issue, symbol]:
        event_id, kind = link["event_id"], link["event_type"]
        if kind == "share_split" and basis == "ibkr_trades_split_adjusted":
            continue
        if kind != "cash_distribution":
            return None, f"unsupported_{kind}"
        terms = json.loads(refs[event_id,]["issuer_terms_json"])
        ex_day = terms["ex_date"] or terms["estimated_ex_date"]
        if not ex_day:
            return None, "unresolved_ex_date"
        if not entry < ex_day <= end:
            continue
        if terms["currency"] == "CAD" and symbol == "ENB":
            usd = fx.get((symbol, terms["record_date"]), {})
            if not usd.get("gross_usd_reference_per_share") or usd["admitted_ex_date"] != ex_day:
                return None, "unresolved_dividend_fx"
            value = usd["gross_usd_reference_per_share"]
        elif terms["currency"] == "USD":
            value = (terms["amount_on_post_2020_amlp_share_basis"] if symbol == "AMLP" else
                     terms["amount_on_quote_share_basis"] or terms["reported_amount"])
        else:
            return None, "unsupported_dividend_currency"
        if value is None:
            return None, "unresolved_dividend_share_basis"
        cash.append(dict(event_id=event_id, ex_date=ex_day, amount_per_quote_share=float(value)))
    return cash, None


def factor_on(day, splits):
    factor = math.prod(f for effective, f in splits if day < effective)
    if factor != int(factor):
        raise ValueError(f"Noninteger historical-to-quote share factor on {day}")
    return int(factor)


def price_window(row, market, corporate):
    days, index, quotes = market
    entry, end = row["entry_session"], row["fifth_session"]
    window = days[index[entry]:index[end]+1]
    if len(window) != 5 or datetime.fromisoformat(row["decision_at_utc"]) > datetime.fromisoformat(row["entry_open_utc"]):
        raise ValueError(f"Unusable chronology for {row['release_date']}/{row['instrument']}")
    legs = {}
    for field, symbol in (("instrument", row["instrument"]), ("amlp", BENCHMARK)):
        series = [quotes[symbol, d] for d in window]
        if len({(q["basis"], q["segment"]) for q in series}) != 1:
            return None, "quote_basis_changes_within_window"
        basis = series[0]["basis"]
        if basis != "ibkr_trades_split_adjusted":
            return None, "unverified_accounting_price_basis"
        distributions, reason = cash_in_holding(row["release_date"], symbol, entry, end, basis, corporate)
        if reason:
            return None, reason
        opening, close = float(series[0]["open"]), float(series[-1]["close"])
        if min(opening, close) <= 0 or not math.isfinite(opening+close):
            raise ValueError(f"Invalid trading price for {symbol}/{entry}")
        splits = corporate[3][symbol]
        entry_factor, exit_factor = (factor_on(day, splits) for day in (entry, end))
        lot = max(1, entry_factor//exit_factor)
        quantity = math.floor((GROSS_NOTIONAL/2)/(opening/entry_factor)/lot)*lot
        shares = quantity/entry_factor
        cash_per_quote_share = math.fsum(event["amount_per_quote_share"] for event in distributions)
        legs[field] = dict(symbol=symbol, open=opening, close=close, cash_per_share=cash_per_quote_share, shares=shares,
            entry_shares=quantity, exit_shares=round(shares*exit_factor), gross_return=(close+cash_per_quote_share)/opening-1,
            dividends=distributions, splits=splits, sessions=window)
        if not math.isclose(shares*exit_factor, legs[field]["exit_shares"]):
            raise ValueError(f"Unhandled fractional split for {symbol}/{entry}")
    return legs, None


def execution_sources():
    commissions = [r for r in read_csv("ibkr_equity_commission_snapshot")
                   if (r["listing_market"], r["currency"], r["plan"]) == ("US", "USD", "PRO_FIXED")]
    if len(commissions) != 1:
        raise ValueError("Ambiguous commission snapshot")
    return commissions[0], [r for r in read_csv("us_equity_regulatory_fee_period") if r["fee"] == "SEC_SECTION_31"]


def settlement_day(day, settlement_days):
    lag = 3 if day < "2017-09-05" else 2 if day < "2024-05-28" else 1
    index = bisect_right(settlement_days,day)+lag-1
    if index >= len(settlement_days):
        raise ValueError(f"Settlement calendar ends before T+{lag} for {day}")
    return settlement_days[index]


def sec_rate(day, settlement_days, fee_periods):
    settlement = settlement_day(day,settlement_days)
    periods = [r for r in fee_periods if r["effective_from"] <= settlement <= r["verified_through_date"]
               and (not r["effective_to_exclusive"] or settlement < r["effective_to_exclusive"])]
    if len(periods) != 1 or periods[0]["date_basis"] != "sro_charge_date":
        raise ValueError(f"Missing unique SEC charge-date rate for {day}/{settlement}")
    return float(periods[0]["rate"])/1_000_000


def borrow_cost(leg, quote, market):
    days, index, prices = market
    rates = {r["session_date"]: r for r in json.loads(quote)}
    total, missing = 0., 0
    for start, finish in zip(leg["sessions"], leg["sessions"][1:]):
        elapsed = (date.fromisoformat(finish)-date.fromisoformat(start)).days
        raw = rates.get(start, {}).get("fee_rate_open_native")
        annual = float(raw) if raw not in (None, "") else MISSING_BORROW
        missing += elapsed if raw in (None, "") else 0
        previous = prices.get((leg["symbol"], days[index[start]-1]), {})
        mark = float(previous["close"]) if previous.get("close") else leg["open"]
        factor = factor_on(start, leg["splits"])
        collateral = leg["shares"]*factor*math.ceil(mark/factor*1.02)
        total += collateral*annual*elapsed/360
    return total, missing


def outcome(row, sign, market, fees, *, settlement_days=None):
    commission, sec_fees = fees
    costs = row["cost"]
    amounts = defaultdict(float)
    spreads_missing = 0
    for field, leg_sign in (("instrument", sign), ("amlp", -sign)):
        leg = row["legs"][field]
        q = leg["shares"]
        amounts["gross_price_pnl_usd"] += leg_sign*q*(leg["close"]-leg["open"])
        amounts["cash_distributions_pnl_usd"] += leg_sign*q*leg["cash_per_share"]
        for edge, price_key, count, day in (("entry", "open", leg["entry_shares"], row["entry_session"]),
                                            ("exit", "close", leg["exit_shares"], row["fifth_session"])):
            notional = q*leg[price_key]
            spread_bps = costs[f"{field}_{edge}_spread_bps"]
            spreads_missing += not bool(spread_bps)
            amounts["spread_cost_usd"] += notional*(float(spread_bps)/10_000 if spread_bps else MISSING_SPREAD)/2
            amounts["commission_usd"] += min(max(count*float(commission["rate_per_share"]), float(commission["minimum_per_order"])),
                                              notional*float(commission["maximum_pct_of_trade_value"])/100)
            if (leg_sign < 0 and edge == "entry") or (leg_sign > 0 and edge == "exit"):
                amounts["regulatory_usd"] += notional*sec_rate(day, market[0] if settlement_days is None else settlement_days, sec_fees)+min(
                    count*float(costs[f"finra_{edge}_rate_per_covered_share_sold"]), float(costs[f"finra_{edge}_cap_usd"]))
    short = "amlp" if sign > 0 else "instrument"
    borrow, missing_days = borrow_cost(row["legs"][short], costs[f"{short}_borrow_sessions_json"], market)
    gross = amounts["gross_price_pnl_usd"]+amounts["cash_distributions_pnl_usd"]
    variable = amounts["spread_cost_usd"]+borrow
    fixed = amounts["commission_usd"]+amounts["regulatory_usd"]
    return dict(**amounts, borrow_usd=borrow, gross_pnl_usd=gross, net_pnl_usd=gross-variable-fixed,
                stressed_net_pnl_usd=gross-2*variable-fixed, missing_spread_edges=spreads_missing,
                missing_borrow_calendar_days=missing_days, both_cost_series_observed=not (spreads_missing or missing_days))


def vector(row, optical):
    return (*row["eia_features"], *row["optical_features"]) if optical else row["eia_features"]


def ridge(train, rows, optical, label="label"):
    x = np.array([vector(r, optical) for r in train], dtype=float)
    observed_count = np.isfinite(x).sum(axis=0)
    centre = np.divide(np.nansum(x, axis=0), observed_count, out=np.zeros(x.shape[1]), where=observed_count>0)
    x = np.where(np.isfinite(x), x, centre)-centre
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    x = x/scale
    y = np.array([r[label] for r in train])
    weights = np.linalg.solve(x.T@x+len(x)*RIDGE_PENALTY*np.eye(x.shape[1]), x.T@(y-y.mean()))
    z = np.array([vector(r, optical) for r in rows], dtype=float)
    z = (np.where(np.isfinite(z), z, centre)-centre)/scale
    return z@weights+y.mean()


def picks(rows, prior):
    best = min(rows, key=lambda r: (-abs(r["physical_height_change_m"]), r["instrument"]))
    robust = [r for r in rows if r["loss_excludes_zero"]]
    yield "always_long_spread", best, 1.0
    yield "eia_sign", best, best["eia_current_kbbl"]
    yield "physical_sign", best, best["magnitude_kbbl"]
    if robust:
        top = min(robust, key=lambda r: (-abs(r["physical_height_change_m"]), r["instrument"]))
        yield "physical_geometry_filter", top, top["magnitude_kbbl"]
    if len(prior) < 40 or len({r["release_date"] for r in prior}) < 20:
        return
    for strategy in ("ridge_eia", "ridge_eia_optical"):
        predictions = ridge(prior, rows, strategy == "ridge_eia_optical")
        prediction, top = min(zip(predictions, rows, strict=True), key=lambda item: (-abs(item[0]), item[1]["instrument"]))
        if abs(prediction) >= RETURN_HURDLE:
            yield strategy, top, float(prediction)
    directional = [(float(value), sign, row) for sign, label in ((1, "net_long_label"), (-1, "net_short_label"))
                   for value, row in zip(ridge(prior, rows, True, label=label), rows, strict=True)]
    expected, sign, selected = min(directional, key=lambda r: (-r[0], r[2]["instrument"], -r[1]))
    if expected >= NET_RETURN_HURDLE:
        yield "ridge_directional_net", selected, sign*expected


def replay(issues, market, fees, corporate, excluded, *, settlement_days=None, common_holdout_issues=None):
    history, trades, usable = [], [], []
    last_exit = {}
    for issue, candidates in sorted(issues.items()):
        admitted = []
        for r in candidates:
            legs, reason = price_window(r, market, corporate)
            if reason:
                excluded[r["study_phase"], reason] += 1
                continue
            row = dict(r, legs=legs, label=(legs["instrument"]["gross_return"]-legs["amlp"]["gross_return"])/2)
            accounting = {sign: outcome(row, sign, market, fees, settlement_days=settlement_days) for sign in (-1, 1)}
            admitted.append(dict(row, accounting=accounting, net_long_label=accounting[1]["net_pnl_usd"]/GROSS_NOTIONAL,
                                 net_short_label=accounting[-1]["net_pnl_usd"]/GROSS_NOTIONAL))
        if not admitted:
            continue
        first = admitted[0]
        usable.append((first["study_phase"], issue))
        decision = datetime.fromisoformat(first["decision_at_utc"])
        prior = [r for r in history if datetime.fromisoformat(r["fifth_close_utc"]) < decision]
        for strategy, row, score in picks(admitted, prior):
            opening = datetime.fromisoformat(row["entry_open_utc"])
            if common_holdout_issues is not None and row["study_phase"]=="holdout" and issue not in common_holdout_issues:
                continue
            if score == 0 or (strategy in last_exit and opening <= last_exit[strategy]):
                continue
            sign = 1 if score > 0 else -1
            last_exit[strategy] = datetime.fromisoformat(row["fifth_close_utc"])
            trades.append(dict(strategy=strategy, study_phase=row["study_phase"], release_date=issue, instrument=row["instrument"],
                entry_session=row["entry_session"], fifth_session=row["fifth_session"], decision_at_utc=row["decision_at_utc"],
                entry_open_utc=row["entry_open_utc"], fifth_close_utc=row["fifth_close_utc"], instrument_side=sign,
                selection_score=score, physical_change_kbbl=row["magnitude_kbbl"], eia_change_kbbl=row["eia_current_kbbl"],
                prior_training_windows=len(prior), prior_training_issues=len({r["release_date"] for r in prior}),
                latest_training_close_utc=max((r["fifth_close_utc"] for r in prior), default=None),
                instrument_entry_shares=row["legs"]["instrument"]["entry_shares"], amlp_entry_shares=row["legs"]["amlp"]["entry_shares"],
                instrument_entry_quote=row["legs"]["instrument"]["open"], instrument_exit_quote=row["legs"]["instrument"]["close"],
                amlp_entry_quote=row["legs"]["amlp"]["open"], amlp_exit_quote=row["legs"]["amlp"]["close"],
                **row["accounting"][sign]))
        history.extend(admitted)
    return trades, history, usable


def statistics(rows, issues):
    by_issue = {r["release_date"]: r for r in rows}
    pnl = np.array([by_issue.get(day, {}).get("net_pnl_usd", 0.) for day in issues])/GROSS_NOTIONAL
    equity = np.r_[1., np.cumprod(1+pnl)]
    realized_dd = min(equity/np.maximum.accumulate(equity)-1)
    rng = np.random.default_rng(2301)
    if len(pnl) >= 4:
        start = rng.integers(0, len(pnl), (3000, math.ceil(len(pnl)/4)))
        index = ((start[:, :, None]+np.arange(4))%len(pnl)).reshape(3000, -1)[:, :len(pnl)]
        low, high = np.quantile(np.prod(1+pnl[index], axis=1)-1, (0.025, 0.975))
    else:
        low, high = None, None
    observed = [r for r in rows if r["both_cost_series_observed"]]
    return dict(trades=len(rows), unique_eligible_issues=len(issues), net_positive_trades=sum(r["net_pnl_usd"] > 0 for r in rows),
        gross_simple_return_pct=100*math.fsum(r["gross_pnl_usd"] for r in rows)/GROSS_NOTIONAL,
        net_simple_return_pct=100*pnl.sum(), net_compounded_return_pct=100*(equity[-1]-1),
        net_compounded_ci_95_pct=[100*low,100*high] if low is not None else None, max_realized_drawdown_pct=100*realized_dd,
        stressed_simple_return_pct=100*math.fsum(r["stressed_net_pnl_usd"] for r in rows)/GROSS_NOTIONAL,
        trades_with_observed_spread_and_used_borrow=len(observed),
        observed_cost_subset_net_simple_return_pct=100*math.fsum(r["net_pnl_usd"] for r in observed)/GROSS_NOTIONAL)


def main():
    issues, excluded = event_inputs()
    trades, eligible, usable = replay(issues, market_sources(), execution_sources(), corporate_inputs(), excluded)
    performance = {phase:{strategy:statistics([r for r in trades if r["study_phase"]==phase and r["strategy"]==strategy],
                    [issue for p, issue in usable if p==phase]) for strategy in STRATEGIES} for phase in PHASES}
    positive_both = [s for s in STRATEGIES if all(performance[p][s]["net_simple_return_pct"] > 0 for p in PHASES)]
    report = dict(eligible_measured_instrument_windows=len(eligible), eligible_by_phase=dict(Counter(r["study_phase"] for r in eligible)),
        excluded_by_phase_and_reason={f"{phase}:{reason}": count for (phase, reason), count in sorted(excluded.items())},
        periods=performance, strategies_positive_after_costs_in_both_open_periods=positive_both,
        trading_research_status="candidates_require_independent_test" if positive_both else "no_tested_candidate_positive_in_both_periods",
        added_after_initial_replay=["ridge_directional_net"],
        interpretation="Open-period exploratory strategy replay. Two legs, next eligible open to fifth session close. "
        "Gross includes documented cash distributions. The stated expense assumptions apply to the 100,000 USD gross test size. "
        "No held-out market outcomes or EIA targets are loaded into the model. Borrow availability and actual fills are unverified.")
    export(f"{STEM}_trades", trades)
    export(f"{STEM}_eligible", [dict(**{k: r[k] for k in ("study_phase", "week_ending", "release_date", "instrument", "decision_at_utc",
        "entry_open_utc", "fifth_close_utc", "magnitude_kbbl", "physical_height_change_m", "loss_excludes_zero", "eia_current_kbbl")},
        prior_eia_change_million_bbl=r["eia_features"][1], wti_prompt_calendar_spread=r["eia_features"][2],
        physical_height_halfwidth_m=r["optical_features"][1], observed_area_fraction=r["optical_features"][2],
        gross_spread_return_per_gross_notional=r["label"], net_long_instrument_per_gross_notional=r["net_long_label"],
        net_short_instrument_per_gross_notional=r["net_short_label"]) for r in eligible])
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
