"""Trade delayed CL windows on their actual nonoverlapping execution calendar."""

import json
import math
from collections import defaultdict
from contextlib import closing
from datetime import datetime

from backtest_wti_cushing_spread import COSTS, tick_price
from build_cushing_holdout_market_features import numeric_prices
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, market
from fit_cushing_cl_volatility import HURDLE, PHASES, RISK_FLOOR_USD, risk_inputs, risk_prediction
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed

STEM = "cushing_cl_delayed_calendar"
METHODS = ("public", "coverage_optical", "carry_only", "public_risk_3000", "always_long")


def candidates():
    forecasts = indexed("cushing_robust_stack_forecasts")
    optical = indexed("cushing_coverage_nowcast_forecasts")
    context = dict(market(forecasts))
    grouped = defaultdict(list)
    for r in read("wti_cushing_event_bars"):
        if int(r["session_number"])>=2:
            grouped[r["release_date"]].append(r)
    for r in sorted(read("wti_cushing_event_windows"),key=lambda q:q["release_date"]):
        issue = r["release_date"]
        if issue not in forecasts or r["phase_ok"]!="True":
            continue
        f = forecasts[issue]
        decision = datetime.fromisoformat(r["decision_at_utc"])
        if not datetime.fromisoformat(f["forecast_cutoff_utc"]) < datetime.fromisoformat(f["target_known_by_utc"]) <= decision:
            raise ValueError(f"Inventory unavailable on {issue}")
        points = sorted(grouped[issue],key=lambda q:(q["session_number"],q["leg"]!="near"))
        if len(points)!=8 or any((b["session_number"],b["delivery_month"])!=(str(i),r[f"{leg}_month"])
                                for b,(i,leg) in zip(points,((i,k) for i in range(2,6) for k in ("near","far")),strict=True)):
            raise ValueError(f"Incorrect legs in delayed window on {issue}")
        source = next((s for s in ("archive","ibkr") if all(b[f"{s}_present"]=="True" for b in points)),None)
        target = float(f["target"])
        surprise = target-float(f["robust_public"])
        carry,change,_ = context[issue]
        x = (target,surprise,carry,change,surprise*carry)
        revision = float(optical[issue]["coverage_ml"])-float(optical[issue]["public_ensemble"])
        start,end = [datetime.fromisoformat(points[i]["settlement_reference_time"]) for i in (0,-1)]
        if not decision < start < end:
            raise ValueError(f"Invalid delayed execution clock on {issue}")
        yield r | dict(source=source,points=points,decision=decision,entry=start,exit_at=end,x=x,optical_revision=revision)


def replay():
    eligible = list(risk_inputs(list(candidates()),allow_missing=True))
    with closing(read_only(SOURCE)) as db:
        price = numeric_prices(db,{"archive","ibkr"},through=max(r["exit_session"] for r in eligible))
    seen = []
    last = None
    excluded = defaultdict(lambda:defaultdict(int))
    output = []
    for r in eligible:
        issue,phase = r["release_date"],r["study_phase"]
        if r["source"] is None or r["volatility_usd"] is None:
            excluded[phase]["missing_marks"]+=1
            continue
        if last is not None and r["entry"]<=last:
            excluded[phase]["overlap"]+=1
            continue
        last = r["exit_at"]
        train = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        source = r["source"]
        legs = [tick_price(price(source,b["session_date"],b["delivery_month"],b[f"{source}_reference"])) for b in r["points"]]
        path = [10*(near-far-legs[0]+legs[1]) for near,far in zip(legs[::2],legs[1::2],strict=True)]
        move = path[-1]
        alternate = next((s for s in ("archive","ibkr") if s != source and all(b[f"{s}_present"]=="True" for b in r["points"])),None)
        if alternate:
            values = [tick_price(price(alternate,b["session_date"],b["delivery_month"],b[f"{alternate}_reference"])) for b in r["points"]]
            alternate_move = 10*(values[-2]-values[-1]-values[0]+values[1])
        sigma = max(RISK_FLOOR_USD,r["volatility_usd"]*math.sqrt(3/4))
        revision = r["optical_revision"]
        current = dict(x=(*r["x"],revision,revision*r["x"][2]),long_gross=move,volatility_usd=sigma,exit_at=r["exit_at"],
                       release_date=issue,study_phase=phase,decision=r["decision"],entry=r["entry"],near_month=r["near_month"],far_month=r["far_month"],
                       spread_path_usd=path,alternate_source=alternate,alternate_long_gross_usd=alternate_move if alternate else None)
        seen.append(current)
        if len(train)<MIN_WINDOWS:
            excluded[phase]["initial_training"]+=1
            continue
        p = risk_prediction(train,current,5)
        o = risk_prediction(train,current,7)
        carry_train = [q | dict(x=q["x"][2:4]) for q in train]
        c = risk_prediction(carry_train,current | dict(x=current["x"][2:4]),2)
        for method,expected in zip(METHODS,(p,o,c,p,None),strict=True):
            side = int(expected>HURDLE)-int(expected<-HURDLE) if expected is not None else 1
            if method=="public_risk_3000" and sigma>3000:
                side = 0
            gross = side*move
            output.append(dict(release_date=issue,study_phase=phase,method=method,near_month=r["near_month"],far_month=r["far_month"],
                source=source,decision_at_utc=r["decision"].isoformat(),entry_mark=r["entry"].isoformat(),exit_mark=r["exit_at"].isoformat(),
                latest_training_exit_utc=train[-1]["exit_at"].isoformat(),training_windows=len(train),volatility_usd=sigma,
                expected_long_gross_usd=expected,near_side=side,gross_usd=gross,alternate_source=alternate,
                alternate_gross_usd=side*alternate_move if alternate else None,
                **{f"session_{n}_net_170_usd":side*value-(170 if n==5 else 85)*abs(side) for n,value in enumerate(path,2)},
                **{f"cost_{cost}_usd":cost*abs(side) for cost in COSTS},**{f"net_{cost}_usd":gross-cost*abs(side) for cost in COSTS}))
    return output, excluded, seen


def daily_risk(group):
    equity = peak = max_drawdown = 0
    for r in group:
        for n in range(2,6):
            value = equity+r[f"session_{n}_net_170_usd"]
            peak = max(peak,value)
            max_drawdown = max(max_drawdown,peak-value)
        equity += r["net_170_usd"]
    crosschecked = [r for r in group if r["near_side"] and r["alternate_source"]]
    return dict(daily_drawdown_usd=max_drawdown, trades_with_both_sources=len(crosschecked),
                largest_source_difference_usd=max((abs(r["gross_usd"]-r["alternate_gross_usd"]) for r in crosschecked),default=None))


def main():
    rows,excluded,_ = replay()
    original = {r["release_date"]:r for r in read("cushing_cl_execution_windows_trades") if r["method"]=="public_volatility"
                and r["first_session_number"]=="2" and r["last_session_number"]=="5"}
    phases = {}
    for phase in PHASES:
        grouped = {m:[r for r in rows if r["study_phase"]==phase and r["method"]==m] for m in METHODS}
        phases[phase] = dict(excluded=excluded[phase],results={m:robustness(g) | daily_risk(g) for m,g in grouped.items()})
        matched = [r for r in grouped["public"] if r["release_date"] in original]
        delta = [r["net_170_usd"]-int(original[r["release_date"]]["net_170_usd"]) for r in matched]
        phases[phase]["matching_original_public"] = dict(windows=len(matched),original_net_170_usd=sum(
            int(original[r["release_date"]]["net_170_usd"]) for r in matched),expanded_net_170_usd=sum(r["net_170_usd"] for r in matched),
            paired_delta_ci_95_usd=block_bootstrap_mean(delta))
        print(phase,len(grouped["public"]),[(m,v["trades"],v["net_170_usd"],v["stress_net_excluding_three_largest_wins_usd"])
              for m,v in phases[phase]["results"].items()])
    export(f"{STEM}_trades",rows)
    report = dict(status="exploratory_actual_delayed_execution_calendar",phases=phases)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2)+"\n").encode())


if __name__=="__main__":
    main()
