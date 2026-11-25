"""Verify or evaluate the frozen operator-versus-AMLP holdout inputs."""
import argparse
import csv
import io
import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import backtest_cushing_operator_strategy as operator
import fit_cushing_strategy_residual_ml as residual
from build_cushing_holdout_market_features import make_plan, numeric_prices, read_protocol, synthetic_close, vector_from_marks
from build_holdout_equity_price_bridges import effective_basis, load_bridges
from build_holdout_optical_inputs import PHASE
from build_operator_development_inputs import one_per_key
from build_study_operator_signal_inputs import BETA_SESSIONS, hedge_history, trailing_beta
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_holdout_operator_replay"
PROTOCOL = ROOT/"data/reference/cushing_holdout_operator_protocol.json"
SETTLEMENT_EXCEPTIONS = ROOT/"data/reference/us_equity_holdout_settlement_exceptions.csv"
BOUNDARIES = ("entry_session", "fifth_session")
SIDES = ("instrument", "amlp")


def protocol():
    config = json.loads(PROTOCOL.read_text())
    expected = dict(symbols=sorted(operator.CORE), benchmark=operator.BENCHMARK, market=operator.VENUE, currency="USD",
        gross_notional_usd=operator.GROSS_NOTIONAL, missing_full_spread_bps=10_000*operator.MISSING_SPREAD,
        missing_annual_borrow_rate=operator.MISSING_BORROW, ridge_penalty=operator.RIDGE_PENALTY,
        gross_return_hurdle=operator.RETURN_HURDLE, net_return_hurdle=operator.NET_RETURN_HURDLE)
    names = (config["primary_candidate"],config["primary_control"],*config["secondary_candidates"])
    if (any(config[k]!=v for k,v in expected.items()) or sorted(names)!=sorted(operator.STRATEGIES)
        or tuple(config["residual_model"]["penalties"])!=residual.PENALTIES):
        raise ValueError("Operator replay rules differ from the registered policy")
    return config


def settlement_calendar(days):
    with SETTLEMENT_EXCEPTIONS.open(newline="") as stream:
        exceptions = one_per_key(csv.DictReader(stream),"session_date")
    clearing = set(days)
    for (day,),r in exceptions.items():
        open_day = r["regular_way_settlement_day"]=="True"
        if not r["source_url"].startswith("https://www.dtcc.com/") or (day in clearing)==open_day:
            raise ValueError(f"Invalid or redundant clearing exception {day}")
        (clearing.add if open_day else clearing.remove)(day)
    return sorted(clearing),exceptions


def market_input(through, candidates, *, evaluate):
    market = operator.market_sources(through) if evaluate else operator.market_sources()
    with closing(read_only(SOURCE)) as db:
        days = [d for d, in db.execute("""SELECT session_date FROM equity_session_calendar WHERE venue=?
            AND session_date BETWEEN '2016-01-01' AND ? ORDER BY session_date""",
            (operator.VENUE,(date.fromisoformat(through)+timedelta(days=12)).isoformat()))]
        index = {d:i for i,d in enumerate(days)}
        identity = {(s,d):dict(basis=b,segment=l,provider=p,reference=f,conid=c) for s,d,b,l,p,f,c in db.execute("""
            SELECT instrument,session_date,price_basis,listing_segment,provider,source_file,conid FROM market_study_quote
            WHERE instrument IN (?,?,?,?,?,?,?) AND venue=? AND currency='USD' AND session_date BETWEEN '2023-12-20' AND ?
              AND open>0 AND close>0""",(*operator.SYMBOLS,operator.VENUE,through))}
    held,previous = quote_keys(candidates,days,index)
    marks = {key:identity[key] for key in held|previous}
    bridges = load_bridges()
    for (symbol,day),r in marks.items():
        proof = bridges.get((symbol,day))
        r["basis"] = effective_basis(proof,r["provider"],r["reference"],r["basis"])
        broker = r["basis"]=="ibkr_trades_split_adjusted" and r["provider"].startswith("ibkr_tws") and r["conid"]
        if not (broker or proof) or not r["reference"]:
            raise ValueError(f"Wrong equity source at {symbol}/{day}")
        path = Path(r["reference"])
        if not (ROOT/path).is_file():
            raise ValueError(f"Missing source file for {symbol}/{day}: {path}")
    prices = market[2].copy()
    if evaluate:
        for key,r in marks.items():
            prices[key] = dict(prices[key],basis=r["basis"])
    else:
        for (symbol,day),r in marks.items():
            if (symbol,day) not in prices:
                level = 20+5*operator.SYMBOLS.index(symbol)
                tick = date.fromisoformat(day).toordinal()
                prices[symbol,day] = dict(r,open=level*(1+.04*math.sin(tick/17.)),close=level*(1+.05*math.cos(tick/21.)))
    return (days,index,prices),marks


def read_inputs():
    rows = [r for r in read_csv("cushing_holdout_operator_inputs") if r["instrument"] in operator.CORE]
    plans = {r["release_date"]:(r,source,marks) for r,source,marks,_ in make_plan(read_protocol())}
    costs = one_per_key((r for r in read_csv("study_execution_inputs") if r["study_phase"]==PHASE),"release_date","instrument")
    if {(r["release_date"],r["instrument"]) for r in rows}!={(issue,s) for issue in plans for s in operator.CORE} or len(rows)!=len(costs):
        raise ValueError("Closed equity calendar, cost inputs, or EIA targets have different coverage")
    candidates,public = defaultdict(list),defaultdict(list)
    for r in rows:
        issue,symbol = r["release_date"],r["instrument"]
        p,_,_ = plans[issue]
        cost = costs[issue,symbol]
        if (r["week_ending"]!=p["week_ending"] or any(r[k]!=cost[k] for k in BOUNDARIES)
            or r["execution_cost_input_row_available"]!="True"):
            raise ValueError(f"Misaligned release, instrument or costs on {issue}/{symbol}")
        measured = r["replay_input_status"]=="ready_for_replay"
        eligible = (r["phase_window_eligible"]=="True" and r["matched_usd_quote_window"]=="True"
                    and int(r["asset_linked_terminals"])>0 and r["terminal_relation_boundary_in_window"]!="1"
                    and r["accounting_inputs_complete"]=="True")
        if measured and not eligible:
            raise ValueError(f"Measured candidate is outside the public training cohort on {issue}/{symbol}")
        if not eligible:
            continue
        cutoff,decision,opening,exit_at = (datetime.fromisoformat(r[k]) for k in
            ("optical_cutoff_utc","decision_at_utc","entry_open_utc","fifth_close_utc"))
        if not cutoff==datetime.fromisoformat(p["forecast_cutoff_utc"])<datetime.fromisoformat(p["target_available_at_utc"])<=decision<opening<exit_at:
            raise ValueError(f"Invalid release-to-trade chronology on {issue}/{symbol}")
        if any(p[k]!="1" for k in ("eia_target_present","eia_prior_change_present")):
            raise ValueError(f"Missing EIA vintage on {issue}/{symbol}")
        public[issue].append((r,cost))
        if measured:
            candidates[issue].append((r,cost))
    return rows,plans,candidates,public


def common_windows(candidates):
    output,last = [],""
    for issue,sample in sorted(candidates.items()):
        first = sample[0][0]
        if any(any(r[k]!=first[k] for k in (*BOUNDARIES,"decision_at_utc","entry_open_utc","fifth_close_utc")) for r,_ in sample):
            raise ValueError(f"Mixed equity calendars on {issue}")
        if first["entry_session"]>last:
            output.append(issue)
            last = first["fifth_session"]
    return output


def quote_keys(candidates,days,index):
    held,previous = set(),set()
    for sample in candidates.values():
        for r,_ in sample:
            entry,end = (index[r[k]] for k in BOUNDARIES)
            if end-entry!=4:
                raise ValueError(f"Wrong session count for {r['release_date']}/{r['instrument']}")
            for symbol in (r["instrument"],operator.BENCHMARK):
                held.update((symbol,d) for d in days[entry:end+1])
                previous.add((symbol,days[entry-1]))
    return held,previous


def numeric_inputs(plans, *, evaluate):
    through = max(plans)
    if evaluate:
        with closing(read_only(SOURCE)) as db:
            targets = {(i,w):(y,p) for i,w,y,p in db.execute("""SELECT release_date,week_ending,
                change_kbbl_as_known_at_release,latest_weekly_change_known_before_release
                FROM eia_inventory_release_panel WHERE study_phase=? AND release_date<=?""",(PHASE,through))}
            get = numeric_prices(db,{source for _,source,_ in plans.values()},
                                 through=max(m["observed_session"] for _,_,marks in plans.values() for m in marks))
    else:
        get = lambda source,day,month,reference: synthetic_close(day,month)
    for i,(issue,(r,source,marks)) in enumerate(sorted(plans.items())):
        values,days = vector_from_marks(marks,source,get)
        if days[1]!=r["pre_release_reference_session"]:
            raise ValueError(f"Wrong WTI context session on {issue}")
        y,p = targets[issue,r["week_ending"]] if evaluate else (1900*math.sin(i/5),1900*math.sin((i-1)/5))
        yield issue,(float(y),float(p),round(float(values[len(values)//2+1]),4))


def cash_clock(row,corporate):
    cash_ids = {}
    for side,symbol in (("instrument",row["instrument"]),("benchmark",operator.BENCHMARK)):
        cash,reason = operator.cash_in_holding(row["release_date"],symbol,row["entry_session"],row["fifth_session"],
                                               "ibkr_trades_split_adjusted",corporate)
        if reason or len(cash)!=int(row[f"{side}_cash_events_in_holding"]):
            raise ValueError(f"Changed cash-action coverage on {row['release_date']}/{symbol}: {reason}")
        cash_ids[side] = "|".join(event["event_id"] for event in cash)
    return cash_ids


def verify_costs(row,cost,market,clearing,exceptions,fees):
    issue,symbol = row["release_date"],row["instrument"]
    days,index,_ = market
    sessions = days[index[row["entry_session"]]:index[row["fifth_session"]]+1]
    result = {}
    for edge,day in zip(("entry","exit"),(sessions[0],sessions[-1]),strict=True):
        charge_day = operator.settlement_day(day,clearing)
        result[f"{edge}_section31_charge_date"] = charge_day
        result[f"{edge}_section31_usd_per_million"] = operator.sec_rate(day,clearing,fees[1])*1_000_000
        result[f"{edge}_clearing_exception_sources"] = "|".join(e["source_url"] for (d,),e in exceptions.items() if day<d<=charge_day)
        if not all(cost[f"finra_{edge}_{k}"]!="" for k in ("rate_per_covered_share_sold","cap_usd","source_url")):
            raise ValueError(f"Missing FINRA source on {issue}/{symbol}/{edge}")
    for side in SIDES:
        rates = json.loads(cost[f"{side}_borrow_sessions_json"])
        if [r["session_date"] for r in rates]!=sessions:
            raise ValueError(f"Unaligned borrow sessions on {issue}/{symbol}/{side}")
        if any(rate.get("fee_rate_open_native") and not (ROOT/rate["raw_file"]).is_file() for rate in rates):
            raise ValueError(f"Missing borrow source on {issue}/{symbol}/{side}")
        if any(cost[f"{side}_{edge}_spread_source"] and not (ROOT/cost[f"{side}_{edge}_spread_source"]).is_file()
               for edge in ("entry","exit")):
            raise ValueError(f"Missing spread source on {issue}/{symbol}/{side}")
        result[f"{side}_borrow_marks_used"] = sum(r.get("fee_rate_open_native") not in ("",None) for r in rates[:-1])
        result[f"{side}_spread_edges_observed"] = sum(bool(cost[f"{side}_{edge}_spread_source"]) for edge in ("entry","exit"))
    return result


def prepare(candidates,plans,values,market,marks,clearing,exceptions,fees,corporate,common):
    inputs,manifest = {},[]
    for issue,sample in sorted(candidates.items()):
        p,source,_ = plans[issue]
        inputs[issue] = []
        for r,cost in sample:
            symbol = r["instrument"]
            days,index,_ = market
            session_dates = days[index[r["entry_session"]]-1:index[r["fifth_session"]]+1]
            sources = {side:"|".join(marks[t,d]["reference"] for d in session_dates)
                       for side,t in (("operator",symbol),("amlp",operator.BENCHMARK))}
            cash = cash_clock(r,corporate)
            check = verify_costs(r,cost,market,clearing,exceptions,fees)
            inputs[issue].append(operator.model_input(r,cost,r,*values[issue]))
            manifest.append({k:r[k] for k in ("release_date","week_ending","instrument","asset_measured_terminal_ids",
                "optical_cutoff_utc","decision_at_utc","entry_session","entry_open_utc","fifth_session","fifth_close_utc",
                "accounting_quote_bridge_keys")} |
                dict(common_nonoverlapping_issue=issue in common,eia_target_reference=p["eia_target_reference"],
                     eia_target_available_at_utc=p["target_available_at_utc"],prior_stock_vintage=p["prior_stock_vintage"],
                     wti_source=source,wti_session=p["pre_release_reference_session"],sessions_with_previous_close="|".join(session_dates),
                     operator_quote_references=sources["operator"],amlp_quote_references=sources["amlp"],
                     operator_cash_events=cash["instrument"],amlp_cash_events=cash["benchmark"],**check))
    return inputs,manifest


def reproduce_csv(rows, name):
    original = (DERIVED/f"{name}.csv").read_bytes()
    open_rows = [r for r in rows if r["study_phase"] in operator.PHASES]
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf,fieldnames=tuple(open_rows[0]))
    writer.writeheader()
    writer.writerows(open_rows)
    if buf.getvalue().encode()!=original:
        raise ValueError(f"The original open-period results do not reproduce for {name}")
    return len(open_rows)


def verify_replay(trades,history,candidates,common,open_excluded,combined_excluded):
    verified = reproduce_csv(trades,f"{operator.STEM}_trades")
    closed = [r for r in trades if r["study_phase"]==PHASE]
    admitted = {(r["release_date"],r["instrument"]) for r in history if r["study_phase"]==PHASE}
    if (admitted!={(issue,r["instrument"]) for issue,sample in candidates.items() for r,_ in sample}
        or open_excluded!=combined_excluded):
        raise ValueError("A previously eligible price, action, or cost input was rejected")
    last = {}
    for r in closed:
        s = r["strategy"]
        entry,exit_at,training,decision = (datetime.fromisoformat(r[k]) for k in
            ("entry_open_utc","fifth_close_utc","latest_training_close_utc","decision_at_utc"))
        costs = sum(r[k] for k in ("spread_cost_usd","commission_usd","regulatory_usd","borrow_usd"))
        if (r["release_date"] not in common or training>=decision or decision>=entry or entry>=exit_at
            or (s in last and entry<=last[s]) or not math.isclose(r["gross_pnl_usd"]-costs,r["net_pnl_usd"],abs_tol=1e-8)):
            raise ValueError(f"Invalid chronology or net accounting on {r['release_date']}/{s}")
        last[s] = exit_at
    return verified,closed


def compare(trades,common,config):
    by_method = {m:{r["release_date"]:r["net_pnl_usd"] for r in trades if r["strategy"]==m} for m in operator.STRATEGIES}
    a,b = (by_method[config[k]] for k in ("primary_candidate","primary_control"))
    differences = np.array([a.get(d,0.)-b.get(d,0.) for d in common])
    return dict(primary_mean_delta_net_usd=float(differences.mean()),primary_delta_ci_95_usd=block_bootstrap_mean(differences),
        methods={m:operator.statistics([r for r in trades if r["strategy"]==m],common) for m in operator.STRATEGIES})


def beta_inputs(public, *, evaluate):
    through = max(r["last_scheduled_session_before_decision"] for sample in public.values() for r,_ in sample)
    days,index,closes,prices = hedge_history(through,with_prices=evaluate)
    if not evaluate:
        prices = {(s,d):(50+6*math.sin(date.fromisoformat(d).toordinal()/19.+sum(map(ord,s))),segment)
                  for (s,d),(_,segment) in prices.items() if s in operator.SYMBOLS}
    with closing(read_only(SOURCE)) as db:
        refs = {(s,d):f for s,d,f in db.execute("""SELECT instrument,session_date,adjusted_ibkr_source_file FROM market_study_quote
            WHERE instrument IN (?,?,?,?,?,?,?) AND venue=? AND currency='USD' AND session_date<=?
              AND adjusted_ibkr_close>0""",(*operator.SYMBOLS,operator.VENUE,through))}
    return (days,index,closes,prices),refs,through


def prepare_residual(public,measured,plans,values,market,marks,clearing,exceptions,fees,corporate,*,evaluate):
    hedge,adjusted_refs,through = beta_inputs(public,evaluate=evaluate)
    optics = {(r["release_date"],r["instrument"]):r["optical_features"] for sample in measured.values() for r in sample}
    inputs,manifest = defaultdict(list),[]
    for issue,sample in sorted(public.items()):
        plan,source,_ = plans[issue]
        for r,cost in sample:
            ticker = r["instrument"]
            key = issue,ticker
            beta = {k:v for length in BETA_SESSIONS for k,v in trailing_beta(r,hedge,length,through=through).items()}
            refs = {}
            for length in BETA_SESSIONS:
                prefix = f"amlp_beta_{length}d"
                first,last = (beta[f"{prefix}_{k}_close"] for k in ("first","last"))
                for leg,symbol in (("operator",ticker),("amlp",operator.BENCHMARK)):
                    days,index,*_ = hedge
                    files = sorted({adjusted_refs[symbol,d] for d in days[index[first]:index[last]+1]}) if first else []
                    if any(not ref or not (ROOT/ref).is_file() for ref in files):
                        raise ValueError(f"Missing adjusted-price source on {issue}/{symbol}")
                    refs[f"{prefix}_{leg}_source_files"] = "|".join(files)
            legs,reason = operator.price_window(r,market,corporate)
            if reason:
                raise ValueError(f"Public training accounting coverage differs on {issue}/{ticker}: {reason}")
            cash = cash_clock(r,corporate)
            cost_evidence = verify_costs(r,cost,market,clearing,exceptions,fees)
            days,index,_ = market
            dates = days[index[r["entry_session"]]-1:index[r["fifth_session"]]+1]
            sources = {f"{side}_quote_references":"|".join(marks[s,d]["reference"] for d in dates)
                       for side,s in (("operator",ticker),("amlp",operator.BENCHMARK))}
            window = dict(r,legs=legs,cost=cost)
            outcomes = {side:operator.outcome(window,side,market,fees,settlement_days=clearing) for side in residual.SIDES}
            inputs[issue].append(dict(r,public=residual.public_vector(ticker,*values[issue],beta["amlp_beta_60d"]),
                optical=optics.get(key),target=np.array([outcomes[s]["net_pnl_usd"]/operator.GROSS_NOTIONAL for s in residual.SIDES]),
                realized_accounting=outcomes,known_at=datetime.fromisoformat(r["fifth_close_utc"])))
            manifest.append({k:r[k] for k in ("release_date","week_ending","instrument","asset_terminal_ids","decision_at_utc",
                "entry_session","entry_open_utc","fifth_session","fifth_close_utc","accounting_quote_bridge_keys")} |
                dict(optical_residual_eligible=key in optics,operator_cash_events=cash["instrument"],amlp_cash_events=cash["benchmark"]) |
                dict(eia_target_reference=plan["eia_target_reference"],prior_stock_vintage=plan["prior_stock_vintage"],
                     wti_source=source,wti_session=plan["pre_release_reference_session"],sessions_with_previous_close="|".join(dates)) | sources |
                {k:v for k,v in beta.items() if k not in ("amlp_beta_60d","amlp_beta_120d")} | refs | cost_evidence)
    return inputs,manifest


def verify_residual(measured,public,inputs,common):
    earlier,_ = residual.examples()
    with threadpool_limits(limits=1):
        history,predictions,trades = residual.replay(earlier|inputs,common_holdout_issues=set(common))
    open_predictions = reproduce_csv(predictions,f"{residual.STEM}_forecasts")
    open_trades = reproduce_csv(trades,f"{residual.STEM}_trades")
    closed_predictions = [r for r in predictions if r["study_phase"]==PHASE]
    closed_trades = [r for r in trades if r["study_phase"]==PHASE]
    expected = {(i,r["instrument"]) for i,sample in measured.items() for r,_ in sample}
    if ({(r["release_date"],r["instrument"]) for r in closed_predictions}!=expected
        or len(closed_predictions)!=len(expected)
        or sum(r["study_phase"]==PHASE for r in history)!=sum(map(len,public.values()))
        or any(datetime.fromisoformat(r["last_training_outcome_utc"])>=datetime.fromisoformat(r["decision_at_utc"])
               for r in closed_predictions)
        or any(r["release_date"] not in common for r in closed_trades)):
        raise ValueError("Residual ML replay changed the paired cohort or used a future outcome")
    return open_predictions,open_trades,closed_predictions,closed_trades


def compare_residual(predictions,trades,common):
    pairs = {m:np.array([[abs(r[f"{m}_{side}"]-r[f"actual_{side}"]) for side in ("short","long")] for r in predictions])
             for m in ("public","hybrid")}
    issue_errors = defaultdict(list)
    for r,delta in zip(predictions,(pairs["hybrid"]-pairs["public"]).mean(axis=1),strict=True):
        issue_errors[r["release_date"]].append(delta*10_000)
    issue_delta = [np.mean(values) for values in issue_errors.values()]
    by_model = {m:[r for r in trades if r["model"]==m] for m in residual.MODELS}
    net = {m:{r["release_date"]:r["net_pnl_usd"] for r in rows} for m,rows in by_model.items()}
    trade_delta = [net["hybrid"].get(d,0.)-net["public"].get(d,0.) for d in common]
    return dict(mae_bp={m:float(errors.mean()*10_000) for m,errors in pairs.items()},
        paired_issue_mean_mae_delta_bp=float(np.mean(issue_delta)),paired_issue_mae_delta_ci_95_bp=block_bootstrap_mean(issue_delta),
        paired_mean_net_delta_usd=float(np.mean(trade_delta)),paired_net_delta_ci_95_usd=block_bootstrap_mean(trade_delta),
        methods={m:operator.statistics(rows,common) for m,rows in by_model.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate",action="store_true",help="Read the actual holdout targets and equity outcomes")
    args = parser.parse_args()
    config = protocol()
    rows,plans,candidates,public = read_inputs()
    through = max(r["fifth_session"] for sample in public.values() for r,_ in sample)
    market,marks = market_input(through,public,evaluate=args.evaluate)
    clearing,exceptions = settlement_calendar(market[0])
    fees = operator.execution_sources()
    common = common_windows(candidates)
    corporate = operator.corporate_inputs((*operator.PHASES,PHASE))
    values = dict(numeric_inputs(plans,evaluate=args.evaluate))
    inputs,manifest = prepare(candidates,plans,values,market,marks,clearing,exceptions,fees,corporate,set(common))
    earlier,excluded = operator.event_inputs()
    combined_excluded = excluded.copy()
    with threadpool_limits(limits=1):
        trades,history,_ = operator.replay(earlier|inputs,market,fees,corporate,combined_excluded,
            settlement_days=clearing,common_holdout_issues=set(common))
    open_rows,closed = verify_replay(trades,history,candidates,set(common),excluded,combined_excluded)
    residual_inputs,residual_manifest = prepare_residual(public,inputs,plans,values,market,marks,clearing,exceptions,fees,corporate,
                                                         evaluate=args.evaluate)
    checked_predictions,checked_trades,predictions,residual_trades = verify_residual(candidates,public,residual_inputs,common)
    report = dict(evaluated=args.evaluate,open_trade_rows_reproduced=open_rows,holdout_core_windows=len(rows),
        holdout_release_dates=len(plans),input_status=dict(Counter(r["replay_input_status"] for r in rows)),
        ready_windows=len(manifest),ready_release_dates=len(candidates),paired_nonoverlapping_release_dates=len(common),
        quote_provenance_keys=len(marks),settlement_exception_edges_in_measured_cohort=sum(bool(r[f"{edge}_clearing_exception_sources"])
            for r in manifest for edge in ("entry","exit")),settlement_exception_edges_in_public_cohort=sum(
            bool(r[f"{edge}_clearing_exception_sources"]) for r in residual_manifest for edge in ("entry","exit")),
        legs_with_cash_actions=sum(bool(r[f"{side}_cash_events"])
            for r in manifest for side in ("operator","amlp")),residual_model=dict(
        open_predictions_reproduced=checked_predictions,open_trades_reproduced=checked_trades,
        public_windows=len(residual_manifest),public_release_dates=len(public),paired_optical_predictions=len(predictions),
        paired_optical_release_dates=len({r["release_date"] for r in predictions}),
        windows_using_equity_price_bridges=sum(bool(r["accounting_quote_bridge_keys"]) for r in residual_manifest),
        paired_adjusted_close_coverage={str(n):sum(r[f"amlp_beta_{n}d_first_close"] is not None
            for r in residual_manifest) for n in BETA_SESSIONS}))
    if args.evaluate:
        report.update(compare(closed,common,config))
        export(f"{STEM}_trades",closed)
        report["residual_model"].update(compare_residual(predictions,residual_trades,common))
        export(f"{STEM}_residual_forecasts",predictions)
        if residual_trades:
            export(f"{STEM}_residual_trades",residual_trades)
    else:
        report["interpretation"] = "Actual identities, cash actions, cost inputs and clocks. Holdout equity prices, WTI prices "
        report["interpretation"] += "and EIA targets are synthetic. Holdout scores and trade outcomes are discarded."
        export(f"{STEM}_manifest",manifest)
        export(f"{STEM}_residual_manifest",residual_manifest)
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_{'summary' if args.evaluate else 'dry_run'}.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
