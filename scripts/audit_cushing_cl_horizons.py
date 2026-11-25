"""Compare CL holding periods and isolate the value of weekly roof measurements."""

import json
from collections import defaultdict
from contextlib import closing
from datetime import datetime

from backtest_wti_cushing_spread import COSTS, statistics, tick_price
from build_cushing_holdout_market_features import numeric_prices
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, inputs, predictions
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_horizons"
SESSIONS = (2, 3, 5)
TRADE_HURDLE = 170
STRATEGIES = ("public_ml", "weekly_optical_ml", "always_long_spread", "no_trade")


def paths(rows):
    stored = defaultdict(list)
    for r in read("wti_cushing_event_bars"):
        stored[r["release_date"]].append(r)
    with closing(read_only(SOURCE)) as db:
        value = numeric_prices(db,{r["source"] for r in rows},through=max(r["exit_session"] for r in rows))
    for r in rows:
        days = defaultdict(dict)
        stamps = {}
        source = r["source"]
        for bar in stored[r["release_date"]]:
            session, month, leg = (bar[k] for k in ("session_date", "delivery_month", "leg"))
            if month != r[f"{leg}_month"] or leg in days[session]:
                raise ValueError(f"Mixed contract months in {r['release_date']}")
            stamp = datetime.fromisoformat(bar["settlement_reference_time"])
            if stamps.setdefault(session,stamp)!=stamp:
                raise ValueError(f"Mixed settlement clocks in {r['release_date']}")
            days[session][leg] = tick_price(value(source,session,month,bar[f"{source}_reference"]))
        calendar = sorted(days)
        if (len(calendar)!=5 or calendar[0]!=r["entry_session"] or calendar[-1]!=r["exit_session"]
                or stamps[calendar[0]]!=r["entry"] or stamps[calendar[-1]]!=r["exit_at"]):
            raise ValueError(f"Different window than the five-session replay on {r['release_date']}")
        spread = [10*(days[day]["near"]-days[day]["far"]) for day in calendar]
        moves = [mark-spread[0] for mark in spread]
        if moves[-1]!=r["long_gross"]:
            raise ValueError(f"Five-session spread move differs on {r['release_date']}")
        yield r, calendar, [stamps[d] for d in calendar], moves


def replay(source):
    weekly = indexed("cushing_weekly_change_ml_forecasts")
    predictions_at = defaultdict(list)
    for r, calendar, stamps, moves in paths(source):
        f = weekly[r["release_date"]]
        if not datetime.fromisoformat(f["forecast_cutoff_utc"]) < datetime.fromisoformat(f["target_known_by_utc"]) <= r["decision"]:
            raise ValueError(f"Invalid optical information clock on {r['release_date']}")
        raw, smooth = (float(f[k]) for k in ("physical", "attenuated"))
        if f["measured"] == "False" and (raw or smooth):
            raise ValueError(f"Unmeasured week has an optical signal on {r['release_date']}")
        public = r["x"][:5]
        optical = (float(f["measured"] == "True"), smooth, raw-smooth, smooth*public[2])
        for n in SESSIONS:
            gross = moves[n-1]
            previous = [q for q in predictions_at[n] if q["exit_at"] < r["decision"]][-HISTORY:]
            exit_at = stamps[n-1]
            predictions_at[n].append(dict(x=(*public,*optical), long_gross=gross, exit_at=exit_at))
            if len(previous)<MIN_WINDOWS:
                continue
            row = predictions_at[n][-1]
            expected = [predictions(previous,row,width) for width in (len(public),len(row["x"]))]
            for method, mean_move in zip(STRATEGIES,(*expected,None,None),strict=True):
                side = (1 if method=="always_long_spread" else 0 if method=="no_trade" else
                        int(mean_move>TRADE_HURDLE)-int(mean_move<-TRADE_HURDLE))
                result = side*gross
                yield {k:r[k] for k in ("release_date", "study_phase", "entry_session", "near_month", "far_month", "source")} | dict(
                    exit_session=calendar[n-1],decision_at_utc=r["decision"].isoformat(),entry_at_utc=r["entry"].isoformat(),
                    exit_at_utc=exit_at.isoformat(),latest_training_exit=previous[-1]["exit_at"].isoformat(),
                    sessions=n,method=method,expected_long_gross_usd=mean_move,near_side=side,gross_usd=result,
                    **{f"cost_{c}_usd":abs(side)*c for c in COSTS},**{f"net_{c}_usd":result-abs(side)*c for c in COSTS})


def summarize(rows):
    base = statistics(rows)
    stress = [r["net_170_usd"] for r in rows]
    best = max(stress)
    return base | dict(net_without_best_trade_stressed_usd=sum(stress)-best,
                       stressed_mean_net_ci_95=block_bootstrap_mean(stress))


def main():
    rows = list(replay(list(inputs())))
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"],r["sessions"]].append(r)
    phases = ("development", "validation", "holdout")
    totals = {phase:{f"{method}:{n}":summarize(groups[phase,method,n]) for method in STRATEGIES for n in SESSIONS} for phase in phases}
    # Select only on development and validation, using the weaker stressed result.
    contenders = [(method,n) for method in STRATEGIES[:2] for n in SESSIONS]
    def rank(rule):
        method,n = rule
        pnl = [totals[p][f"{method}:{n}"]["net_170_usd"] for p in phases[:2]]
        return min(pnl),sum(pnl),-n,method

    method,n = max(contenders,key=rank)
    result = dict(status="exploratory_replay_on_previously_examined_history", trade_hurdle_usd=TRADE_HURDLE,
                  phases=totals, choice_using_open_periods=dict(method=method,sessions=n,positive_in_both=rank((method,n))[0]>0))
    benchmark = {(r["study_phase"],r["release_date"]):r for r in read("cushing_cl_regime_strategy_trades")
                 if r["method"]=="public" and r["hurdle_usd"]==str(TRADE_HURDLE)}
    reference = [r for r in rows if r["method"]=="public_ml" and r["sessions"]==5]
    if len(reference)!=len(benchmark) or any(r["net_170_usd"]!=int(benchmark[r["study_phase"],r["release_date"]]["net_170_usd"])
                                            for r in reference):
        raise ValueError("Five-session public result does not reproduce the earlier replay")
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    for phase, group in totals.items():
        print(phase,{k:(v["trades"],v["net_50_usd"],v["net_170_usd"]) for k,v in group.items()})
    print(result["choice_using_open_periods"])


if __name__ == "__main__":
    main()
