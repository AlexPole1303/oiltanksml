"""Compare settlement-to-settlement CL spreads across delivery-month gaps."""

import json
import math
from bisect import bisect_left
from collections import defaultdict
from contextlib import closing
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS, full_sessions, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features as response_context
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, RISK_FLOOR_USD, RISK_LOOKBACK, risk_prediction, spread_volatility
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features

STEM = "cushing_cl_tenors"
TENORS = (1,2,3,4,5,6)
METHODS = ("curve","inventory","optical","conditional")
SELECTIONS = ("largest_expected_net", "largest_expected_net_per_risk")


def panel(cases):
    end = cases[-1]["release_date"]
    with closing(read_only(SOURCE)) as db:
        sessions = full_sessions(db,end)
        months = [m for m, in db.execute("SELECT delivery_month FROM wti_delivery_calendar ORDER BY delivery_month")]
        books = {source:{(d,m):tick_price(value) for d,m,_,value in price_rows(db,source,numeric_through=cases[-1]["exit_at"].date().isoformat())}
                 for source in ("archive","ibkr")}
    for r in cases:
        issue = r["release_date"]
        start = bisect_left(sessions,r["entry"].date().isoformat())
        stop = bisect_left(sessions,issue)
        backward = sessions[stop-RISK_LOOKBACK-1:stop]
        forward = sessions[start:start+4]
        if len(backward)!=RISK_LOOKBACK+1 or backward[-1]>=issue or forward[-1]!=r["exit_at"].date().isoformat():
            raise ValueError(f"Incorrect dated spread schedule on {issue}")
        near = r["near_month"]
        far = {n:months[months.index(near)+n] for n in TENORS}
        if far[3]!=r["far_month"]:
            raise ValueError(f"Original three-month contract differs on {issue}")
        book_months = (near,*far.values())
        keys = {name:[(day,m) for day in days for m in book_months] for name,days in (("history",backward),("outcome",forward))}
        providers = {key:[name for name,book in books.items() if all(k in book for k in legs)] for key,legs in keys.items()}
        if not all(providers.values()):
            yield r, None
            continue
        risk_source = providers["history"][0]
        source,*alternate = providers["outcome"]
        spreads = {}
        for key, selected in (("history",risk_source),("outcome",source)):
            prices = np.array([books[selected][k] for k in keys[key]]).reshape(-1,len(book_months))
            spreads[key] = 10*(prices[:,0,None]-prices[:,1:])
        outcome = spreads["outcome"]
        paths = outcome-outcome[0]
        if paths[-1,2]!=r["long_gross"] or list(paths[:,2])!=r["spread_path_usd"]:
            raise ValueError(f"The common-source three-month outcome changed on {issue}")
        if alternate:
            book = books[alternate[0]]
            change = np.array([10*((book[forward[-1],near]-book[forward[-1],m])-(book[forward[0],near]-book[forward[0],m])) for m in far.values()])
        else:
            change = [None]*len(far)
        windows = []
        for i,(n,m) in enumerate(far.items()):
            sigma = max(RISK_FLOOR_USD,spread_volatility(spreads["history"][:,i])*math.sqrt(3/4))
            windows.append(r | dict(tenor=n, far_month=m, volatility_usd=sigma, long_gross=int(paths[-1,i]),
                spread_path_usd=list(map(int,paths[:,i])), source=source, risk_source=risk_source,
                last_risk_session=backward[-1], alternate_source=alternate[0] if alternate else None,
                alternate_long_gross_usd=int(change[i]) if alternate else None))
        if not math.isclose(windows[2]["volatility_usd"],r["volatility_usd"],abs_tol=1e-10):
            raise ValueError(f"The common-source three-month risk changed on {issue}")
        yield r, windows


def replay(cases):
    forecasts = indexed("cushing_intraweek_forecasts")
    history, residual_history = defaultdict(list), defaultdict(list)
    unavailable = []
    trades = []
    for anchor,windows in panel(cases):
        issue = anchor["release_date"]
        if windows is None:
            unavailable.append(issue)
            continue
        forecast = forecasts[issue]
        cutoff, known = [datetime.fromisoformat(forecast[k]) for k in ("cutoff_utc","known_utc")]
        actual, public, optical = (float(forecast[k]) for k in ("actual",PUBLIC,"adaptive_prior_and_timing"))
        if not cutoff<known<=anchor["decision"] or actual!=anchor["x"][0]:
            raise ValueError(f"The EIA issue differs on {issue}")
        for r in windows:
            n = r["tenor"]
            prior = [q for q in history[n] if q["exit_at"]<r["decision"]][-HISTORY:]
            vectors = (features(r)[5],replace_expectation(r,public),replace_expectation(r,optical))
            history[n].append(r | dict(vectors=vectors))
            if len(prior)<MIN_WINDOWS:
                continue
            base = [risk_prediction([q | dict(x=q["vectors"][i]) for q in prior],r | dict(x=x),len(x)) for i,x in enumerate(vectors)]
            context = response_context(r["x"][2],r["volatility_usd"],actual-public)
            current = r | dict(response_features=context,base_estimate=base[1])
            settled = [q for q in residual_history[n] if q["exit_at"]<r["decision"]][-HISTORY:]
            residual_history[n].append(current)
            adjustment,_ = conditional_residual(settled,current) if len(settled)>=MIN_WINDOWS else (0.,0.)
            for method,estimate in zip(METHODS,(*base,base[1]+adjustment),strict=True):
                side = int(estimate>HURDLE)-int(estimate<-HURDLE)
                gross = side*r["long_gross"]
                trades.append({k:r[k] for k in ("release_date","study_phase","near_month","far_month","tenor","source","risk_source",
                    "volatility_usd","last_risk_session","alternate_source")} | dict(method=method,decision_at_utc=r["decision"].isoformat(),
                    entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),latest_training_exit_utc=prior[-1]["exit_at"].isoformat(),
                    latest_response_exit_utc=settled[-1]["exit_at"].isoformat() if settled else None,
                    response_training_windows=len(settled),expected_long_move_usd=estimate,near_side=side,gross_usd=gross,
                    alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                    **{f"session_{i}_net_170_usd":side*v-(170 if i==5 else 85)*abs(side) for i,v in enumerate(r["spread_path_usd"],2)},
                    **{f"cost_{c}_usd":c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS}))
    return trades, unavailable


def select_one(trades):
    by_issue = defaultdict(list)
    for r in trades:
        if r["method"]=="conditional":
            by_issue[r["release_date"]].append(r)
    for issue,rows in by_issue.items():
        if tuple(r["tenor"] for r in rows)!=TENORS:
            raise ValueError(f"Incomplete set of tenor estimates on {issue}")
        for method in SELECTIONS:
            def score(r):
                net = abs(r["expected_long_move_usd"])-HURDLE
                return net/r["volatility_usd"] if method=="largest_expected_net_per_risk" else net

            candidate = max(rows,key=score)
            # All positive decisions were made before the shared entry mark.
            # A booked row with a zero side has exactly zero costs and P&L.
            yield candidate | dict(selection=method, selection_score=score(candidate))


def main():
    with threadpool_limits(limits=1):
        _,_,cases = market_calendar()
        trades, unavailable = replay(cases)
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["tenor"],r["method"]].append(r)
    selections = list(select_one(trades))
    report = dict(status="common_calendar_on_reviewed_history",unavailable_releases=unavailable,phases={},selection={})
    for phase in PHASES:
        baseline = groups[phase,3,"conditional"]
        result = report["phases"][phase] = {}
        for n in TENORS:
            result[n] = {}
            for method in METHODS:
                rows = groups[phase,n,method]
                if [r["release_date"] for r in rows]!=[r["release_date"] for r in baseline]:
                    raise ValueError(f"Comparison calendar differs for {phase}/{n}/{method}")
                delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(rows,baseline,strict=True)]
                result[n][method] = robustness(rows) | daily_risk(rows) | dict(incremental_net_170_usd=sum(delta),
                    paired_net_vs_three_month_95_usd=block_bootstrap_mean(delta),net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in rows))
        report["selection"][phase] = {}
        for method in SELECTIONS:
            rows = [r for r in selections if r["study_phase"]==phase and r["selection"]==method]
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(rows,baseline,strict=True)]
            report["selection"][phase][method] = robustness(rows) | daily_risk(rows) | dict(
                incremental_net_170_usd=sum(delta), paired_net_vs_three_month_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in rows),
                trades_by_tenor={n:sum(r["near_side"]!=0 and r["tenor"]==n for r in rows) for n in TENORS})
        print(phase,{n:{m:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in arm.items()} for n,arm in result.items()})
    effects = []
    for phase in PHASES:
        for n in TENORS:
            inventory, optical = (groups[phase,n,m] for m in ("inventory","optical"))
            delta = [o["net_170_usd"]-b["net_170_usd"] for o,b in zip(optical,inventory,strict=True)]
            report["phases"][phase][n]["optical"]["paired_vs_same_tenor_public_95_usd"] = block_bootstrap_mean(delta)
            report["phases"][phase][n]["optical"]["net_vs_same_tenor_public_usd"] = sum(delta)
            effects.extend(dict(release_date=o["release_date"],study_phase=phase,tenor=n, public_side=b["near_side"],
                optical_side=o["near_side"], public_net_usd=b["net_170_usd"],optical_net_usd=o["net_170_usd"],
                optical_increment_usd=value) for value,o,b in zip(delta,optical,inventory,strict=True) if o["near_side"]!=b["near_side"])
    export(f"{STEM}_trades",trades)
    export(f"{STEM}_selection_trades",selections)
    export(f"{STEM}_optical_trade_changes",effects)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
