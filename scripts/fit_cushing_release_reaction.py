"""Learn delayed CL spread moves using the first post-report price response."""

import json
import math
from collections import defaultdict
from contextlib import closing

from audit_cushing_cl_horizons import paths
from backtest_wti_cushing_spread import COSTS, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, inputs
from fit_cushing_cl_volatility import HURDLE, PHASES, RISK_FLOOR_USD, risk_inputs, risk_prediction
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_release_reaction"
METHODS = ("public", "market_reaction_only", "public_reaction", "public_reaction_optical", "public_reaction_confirmed")


def replay():
    optical = indexed("cushing_coverage_nowcast_forecasts")
    anchors = list(risk_inputs(list(inputs())))
    with closing(read_only(SOURCE)) as db:
        through = max(r["exit_session"] for r in anchors)
        prices = {s:{(d,m):tick_price(p) for d,m,_,p in price_rows(db,s,numeric_through=through)} for s in ("archive","ibkr")}
    earlier = {r["release_date"]:r for r in read("cushing_cl_execution_windows_trades") if r["method"]=="public_volatility"
               and r["first_session_number"]=="2" and r["last_session_number"]=="5"}
    seen = defaultdict(list)
    for r,days,times,moves in paths(anchors):
        issue = r["release_date"]
        first_mark = days[0]
        risk_date = r["risk_last_session"]
        sigma = max(RISK_FLOOR_USD,r["volatility_usd"]*math.sqrt(3/4))
        move = moves[-1]-moves[1]
        public = r["x"][:5]
        revision = float(optical[issue]["coverage_ml"])-float(optical[issue]["public_ensemble"])
        months = r["near_month"],r["far_month"]
        keys = [(day,month) for day in (risk_date,first_mark) for month in months]
        source = next((s for s in (r["volatility_source"],r["source"]) if all(k in prices[s] for k in keys)),None)
        if source is None:
            raise ValueError(f"Missing both exact spread marks on {issue}")
        before_near,before_far,after_near,after_far = [prices[source][key] for key in keys]
        reaction = (10*(after_near-after_far-before_near+before_far))/sigma
        oil_return = (after_near-before_near)/max(abs(before_near),1000)
        x_market = (*public[2:4],reaction,oil_return)
        x_public = (*public,reaction,oil_return,reaction*public[1])
        xs = (public,x_market,x_public,(*x_public,revision,revision*reaction),x_public)
        if not r["decision"] < times[0] < times[1] < times[-1]:
            raise ValueError(f"Second-session decision chronology failed on {issue}")
        for method,x in zip(METHODS,xs,strict=True):
            current = dict(x=x,long_gross=move,volatility_usd=sigma,exit_at=times[-1])
            # Retain the prior cutoffs so every arm trains on the same settled weeks.
            train = [q for q in seen[method] if q["exit_at"]<r["decision"]][-HISTORY:]
            seen[method].append(current)
            if len(train)<MIN_WINDOWS:
                continue
            estimate = risk_prediction(train,current,len(x))
            side = int(estimate>HURDLE)-int(estimate<-HURDLE)
            if method=="public_reaction_confirmed" and reaction*side<0:
                side = 0
            gross = side*move
            if method=="public" and gross-HURDLE*abs(side)!=int(earlier[issue]["net_170_usd"]):
                raise ValueError(f"Different historical baseline on {issue}")
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month")} | dict(method=method,
                feature_mark_session=first_mark,feature_mark_time=times[0].isoformat(),reaction_source=source,
                entry_session=days[1],exit_session=days[-1],latest_training_exit=train[-1]["exit_at"].isoformat(),
                reaction_over_sigma=reaction,near_oil_return=oil_return,expected_long_gross_usd=estimate,near_side=side,gross_usd=gross,
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    rows = list(replay())
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["study_phase"],r["method"]].append(r)
    phases = {}
    for phase in PHASES:
        base = grouped[phase,"public"]
        phases[phase] = {}
        for m in METHODS:
            group = grouped[phase,m]
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different market calendars in {phase}/{m}")
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(group,base,strict=True)]
            phases[phase][m] = robustness(group) | dict(delta_vs_public_170_usd=sum(delta),paired_delta_95_usd=block_bootstrap_mean(delta))
        print(phase,{m:(r["trades"],r["net_170_usd"],r["stress_net_excluding_three_largest_wins_usd"])
                     for m,r in phases[phase].items()})
    export(f"{STEM}_trades",rows)
    report = dict(status="exploratory_replay_on_previously_examined_history", phases=phases)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2)+"\n").encode())


if __name__=="__main__":
    main()
