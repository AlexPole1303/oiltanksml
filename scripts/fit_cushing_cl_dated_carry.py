"""Use the curve of the exact dated contracts underlying each CL spread trade."""

import json
import math
from bisect import bisect_left
from collections import defaultdict
from contextlib import closing
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import COSTS, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, RISK_FLOOR_USD, RISK_LOOKBACK, risk_prediction, spread_volatility
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PUBLIC
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features

STEM = "cushing_cl_dated_carry"
METHODS = ("curve", "inventory", "exact_curve", "exact_inventory", "exact_optical", "both_inventory")


def dated_marks(cases):
    with closing(read_only(SOURCE)) as db:
        sessions = [day for day, in db.execute("SELECT session_date FROM wti_session_basis WHERE full_cmes_session=1 "
                                               "AND cme_no_settlement_notice_url IS NULL ORDER BY session_date")]
        books = {source:{(day,month):tick_price(value) for day,month,_,value in
                         price_rows(db,source,numeric_through=cases[-1]["release_date"])} for source in ("archive","ibkr")}
    for r in cases:
        issue = r["release_date"]
        end = bisect_left(sessions,issue)
        days = sessions[end-RISK_LOOKBACK-1:end]
        keys = [(day,r[leg]) for day in days for leg in ("near_month","far_month")]
        source = next((name for name,book in books.items() if all(k in book for k in keys)),None)
        if source is None or len(days)!=RISK_LOOKBACK+1 or days[-1]>=issue:
            raise ValueError(f"Missing eligible dated CL prices on {issue}")
        prices = np.array([books[source][k] for k in keys]).reshape(-1,2)
        spread = prices[:,0]-prices[:,1]
        sigma = max(RISK_FLOOR_USD,spread_volatility(10*spread)*math.sqrt(3/4))
        if not math.isclose(sigma,r["volatility_usd"],abs_tol=1e-10):
            raise ValueError(f"Different risk source or fixed contracts on {issue}")
        yield r, source, days[-1], days[-6], float(spread[-1]/300), float((spread[-1]-spread[-6])/300)


def replay(cases):
    forecasts = indexed("cushing_intraweek_forecasts")
    incumbent = {r["release_date"]:r for r in read("cushing_regional_balance_trades") if r["method"]=="public_expectation"}
    seen = []
    for r, source, current_day, prior_day, carry, change in dated_marks(cases):
        issue = r["release_date"]
        f = forecasts[issue]
        public = float(f[PUBLIC])
        cutoff,known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc","known_utc"))
        if not cutoff < known <= r["decision"] or float(f["actual"])!=r["x"][0]:
            raise ValueError(f"Forecast cutoff or EIA label differs on {issue}")
        target, surprise, old_carry, old_change, _, *extra = r["x"]
        fixed = r | dict(x=(target,surprise,carry,change,surprise*carry,*extra))
        prior_curve = features(r)[5]
        public_design = replace_expectation(r,public)
        exact_curve = features(fixed)[5]
        exact_inventory = replace_expectation(fixed,public)
        exact_optical = replace_expectation(fixed,float(f["intraweek_adaptive"]))
        both = (*public_design, carry-old_carry, change-old_change)
        vectors = (prior_curve,public_design,exact_curve,exact_inventory,exact_optical,both)
        history = [q for q in seen if q["exit_at"] < r["decision"]][-HISTORY:]
        seen.append(r | dict(vectors=vectors))
        if len(history) < MIN_WINDOWS:
            continue
        for i, (method, x) in enumerate(zip(METHODS,vectors,strict=True)):
            estimate = risk_prediction([q | dict(x=q["vectors"][i]) for q in history],r | dict(x=x),len(x))
            if method=="inventory" and not math.isclose(estimate,float(incumbent[issue]["expected_long_move_usd"]),abs_tol=1e-8):
                raise ValueError(f"Base model differs on {issue}")
            side = int(estimate>HURDLE)-int(estimate<-HURDLE)
            gross = side*r["long_gross"]
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd","alternate_source")} | dict(
                method=method, decision_at_utc=r["decision"].isoformat(), entry_at_utc=r["entry"].isoformat(), exit_at_utc=r["exit_at"].isoformat(),
                latest_training_exit_utc=history[-1]["exit_at"].isoformat(), dated_source=source, latest_dated_session=current_day,
                previous_dated_session=prior_day, exact_carry_usd_bbl_per_month=carry, exact_change_usd_bbl_per_month=change,
                expected_long_move_usd=estimate, near_side=side, gross_usd=gross,
                alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*v-(170 if n==5 else 85)*abs(side) for n,v in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    with threadpool_limits(limits=1):
        _,_,cases = market_calendar()
        trades = list(replay(cases))
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["method"]].append(r)
    report = dict(status="historical_walk_forward_on_reviewed_period", phases={})
    for phase in PHASES:
        base = groups[phase,"inventory"]
        result = report["phases"][phase] = {}
        for method in METHODS:
            test = groups[phase,method]
            if [r["release_date"] for r in test]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different trading windows for {phase}/{method}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(test,base,strict=True)]
            result[method] = robustness(test) | daily_risk(test) | dict(
                incremental_net_170_usd=sum(delta), paired_increment_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in test))
        print(phase,{m:(d["trades"],d["net_170_usd"],d["daily_drawdown_usd"]) for m,d in result.items()})
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
