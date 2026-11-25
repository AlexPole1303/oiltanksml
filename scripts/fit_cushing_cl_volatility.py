"""Compare risk-scaled CL forecasts on the same dated contract and entry windows."""

import json
import math
from bisect import bisect_left
from collections import defaultdict
from contextlib import closing
from datetime import datetime

import numpy as np

from backtest_wti_cushing_spread import COSTS, statistics, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, inputs, predictions
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_volatility"
HURDLE = 170
RISK_LOOKBACK = 10
RISK_FLOOR_USD = 300
MODELS = ("public", "coverage", "public_volatility", "coverage_volatility", "carry_only_volatility",
          "always_long_spread", "no_trade")
PHASES = ("development", "validation", "holdout")


def spread_volatility(near_minus_far_usd):
    daily = np.diff(near_minus_far_usd)
    if len(daily)!=RISK_LOOKBACK or not np.isfinite(daily).all():
        raise ValueError("The spread volatility estimate requires ten complete price changes")
    return max(RISK_FLOOR_USD,float(np.std(daily,ddof=1)*math.sqrt(4)))


def risk_inputs(rows, *, allow_missing=False):
    with closing(read_only(SOURCE)) as db:
        sessions = [d for d, in db.execute("SELECT session_date FROM wti_session_basis WHERE full_cmes_session=1 "
                                           "AND cme_no_settlement_notice_url IS NULL ORDER BY session_date")]
        quotes = {source:{(d,m):(tick_price(value),ref) for d,m,ref,value in price_rows(db,source,numeric_through=rows[-1]["release_date"])}
                  for source in ("archive","ibkr")}
    for r in rows:
        stop = bisect_left(sessions,r["release_date"])
        dates = sessions[stop-RISK_LOOKBACK-1:stop]
        legs = [(d,r[k]) for d in dates for k in ("near_month","far_month")]
        if len(dates)!=RISK_LOOKBACK+1 or dates[-1]>=r["release_date"]:
            raise ValueError(f"Invalid pre-release calendar on {r['release_date']}")
        source = next((name for name, marks in quotes.items() if all(leg in marks for leg in legs)),None)
        if source is None:
            if allow_missing:
                yield r | dict(volatility_usd=None, volatility_source=None)
                continue
            raise ValueError(f"No complete pre-release risk source on {r['release_date']}")
        book = quotes[source]
        spread = np.asarray([10*(book[d,r["near_month"]][0]-book[d,r["far_month"]][0]) for d in dates])
        sigma = spread_volatility(spread)
        yield r | dict(volatility_usd=sigma, volatility_source=source, risk_first_session=dates[0], risk_last_session=dates[-1],
                       risk_source_references="|".join(sorted({book[leg][1] for leg in legs})))


def risk_prediction(rows, current, width):
    scaled = [r | {"long_gross":1000*r["long_gross"]/r["volatility_usd"]} for r in rows]
    return predictions(scaled,current,width)*current["volatility_usd"]/1000


def replay():
    optical = indexed("cushing_coverage_nowcast_forecasts")
    seen = []
    for r in risk_inputs(list(inputs())):
        f = optical[r["release_date"]]
        if not datetime.fromisoformat(f["forecast_cutoff_utc"]) < datetime.fromisoformat(f["target_known_by_utc"]) <= r["decision"]:
            raise ValueError(f"Inventory inputs are outside the release clock on {r['release_date']}")
        revision = float(f["coverage_ml"])-float(f["public_ensemble"])
        row = r | {"x":(*r["x"][:5],revision,revision*r["x"][2])}
        train = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        seen.append(row)
        if len(train)<MIN_WINDOWS:
            continue
        fitted = [model(train,row,width) for model in (predictions,risk_prediction) for width in (5,7)]
        carry_train = [q | {"x":q["x"][2:4]} for q in train]
        carry_only = risk_prediction(carry_train,row | {"x":row["x"][2:4]},2)
        for method, value in zip(MODELS,(*fitted,carry_only,None,None),strict=True):
            side = int(value>HURDLE)-int(value<-HURDLE) if value is not None else int(method=="always_long_spread")
            gross = r["long_gross"]*side
            yield {k:r[k] for k in ("release_date","study_phase","entry_session","exit_session","near_month","far_month","source",
                                   "volatility_usd","volatility_source","risk_first_session","risk_last_session","risk_source_references")} | dict(
                method=method, decision_at_utc=r["decision"].isoformat(), latest_training_exit=train[-1]["exit_at"].isoformat(),
                expected_long_gross_usd=value, near_side=side, gross_usd=gross,
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def summarize(rows):
    values = [r["net_170_usd"] for r in rows]
    best = sorted((v for v in values if v>0),reverse=True)
    years = defaultdict(list)
    for row in rows:
        years[row["release_date"][:4]].append(row)
    return statistics(rows) | dict(stressed_mean_net_ci_95=block_bootstrap_mean(values),
        stress_net_excluding_largest_win_usd=sum(values)-sum(best[:1]),
        stress_net_excluding_three_largest_wins_usd=sum(values)-sum(best[:3]),
        stress_net_excluding_2026_04_01_usd=sum(r["net_170_usd"] for r in rows if r["release_date"]!="2026-04-01"),
        by_year={year:dict(trades=sum(r["near_side"]!=0 for r in sample),stress_net_usd=sum(r["net_170_usd"] for r in sample))
                 for year,sample in sorted(years.items())})


def main():
    rows = list(replay())
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
    reference = {(r["study_phase"],r["release_date"],r["method"]):r for r in read("cushing_coverage_nowcast_direct_trades")}
    for r in rows:
        mapping = {"public":"public_ml","coverage":"coverage_ml"}
        if r["method"] in mapping and r["net_170_usd"]!=int(reference[r["study_phase"],r["release_date"],mapping[r["method"]]]["net_170_usd"]):
            raise ValueError(f"Previous model did not reproduce on {r['release_date']}")
    paired = {}
    for p in PHASES:
        paired[p] = {}
        for a,b in (("coverage_volatility","public_volatility"),("coverage_volatility","coverage"),
                    ("public_volatility","public"),("public_volatility","carry_only_volatility")):
            left,right = groups[p,a],groups[p,b]
            if [r["release_date"] for r in left]!=[r["release_date"] for r in right]:
                raise ValueError(f"Different calendars for {a}/{b}")
            delta = [x["net_170_usd"]-y["net_170_usd"] for x,y in zip(left,right,strict=True)]
            paired[p][f"{a}_vs_{b}"] = dict(net_usd=sum(delta),mean_net_ci_95=block_bootstrap_mean(delta))
    report = dict(status="exploratory_replay_on_previously_examined_history", phases={p:{m:summarize(groups[p,m]) for m in MODELS}
                  for p in PHASES}, paired_stress=paired)
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    for p,g in report["phases"].items():
        print(p,{m:(r["trades"],r["net_50_usd"],r["net_170_usd"],r["stress_net_excluding_largest_win_usd"]) for m,r in g.items()})


if __name__=="__main__":
    main()
