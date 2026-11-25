"""Fit public and incremental optical net-return models with causal labels."""
import json
import math
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from backtest_cushing_operator_strategy import (CORE, GROSS_NOTIONAL, MAX_QUOTE_DATE, NET_RETURN_HURDLE, PHASES,
    corporate_inputs, event_inputs, execution_sources, market_sources, outcome, price_window, statistics)
from build_operator_development_inputs import one_per_key
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write

STEM = "cushing_residual_strategy_ml"
PENALTIES = (0.1, 0.5, 2.)
MODELS = ("public", "hybrid", "consensus")
SIDES = (-1, 1)


def public_vector(symbol, current, prior, wti, beta):
    return (current/1000,prior/1000,wti,float(beta) if beta not in ("",None) else math.nan,
            *(float(symbol==s) for s in CORE))


def examples():
    measured, _ = event_inputs()
    measured = {(issue,r["instrument"]):r for issue,rows in measured.items() for r in rows}
    eia = one_per_key(read_csv("sentinel2_space_time_coverage_eia_comparison"), "week_ending", "release_date")
    oil = one_per_key((r for r in read_csv("oil_market_context") if r["phase"] == "pre_release" and
                       r["release_date"] <= MAX_QUOTE_DATE), "release_date")
    cost = one_per_key((r for r in read_csv("study_execution_inputs") if r["study_phase"] in PHASES), "release_date", "instrument")
    market, corp, fees = market_sources(), corporate_inputs(), execution_sources()
    excluded = Counter()
    result = defaultdict(list)
    for r in read_csv("study_operator_signal_inputs"):
        phase, issue, symbol = (r[k] for k in ("study_phase", "release_date", "instrument"))
        if (phase not in PHASES or symbol not in CORE or r["phase_window_eligible"] != "True"
            or r["matched_usd_quote_window"] != "True" or r["fifth_session"] > MAX_QUOTE_DATE
            or not int(r["asset_linked_terminals"]) or r["terminal_relation_boundary_in_window"] == "1"):
            continue
        e = eia[r["week_ending"],issue]
        if not (e["eia_first_release_change_kbbl"] and e["prior_kbbl"]):
            excluded[phase,"missing_eia"] += 1
            continue
        o = oil.get((issue,),{})
        if o and datetime.fromisoformat(o["cutoff_utc"]) > datetime.fromisoformat(r["decision_at_utc"]):
            raise ValueError(f"Oil context after decision for {issue}")
        x = public_vector(symbol,float(e["eia_first_release_change_kbbl"]),float(e["prior_kbbl"]),
                          float(o["cl1_minus_cl2"]) if o.get("cl1_minus_cl2") else math.nan,r["amlp_beta_60d"])
        legs, reason = price_window(r, market, corp)
        if reason:
            excluded[phase,reason] += 1
            continue
        c = cost[issue,symbol]
        if (c["entry_session"],c["fifth_session"]) != (r["entry_session"],r["fifth_session"]):
            raise ValueError(f"Cost rows are not aligned for {issue}/{symbol}")
        inputs = dict(r, legs=legs, cost=c)
        accounting = {side:outcome(inputs,side,market,fees) for side in SIDES}
        v = measured.get((issue,symbol))
        result[issue].append(dict(r, public=x, optical=v["optical_features"] if v else None,
            target=np.array([accounting[side]["net_pnl_usd"]/GROSS_NOTIONAL for side in SIDES]),
            realized_accounting=accounting, known_at=datetime.fromisoformat(r["fifth_close_utc"])))
    return result, excluded


def fitted_ridge(x, y, z, penalties, weeks):
    sample = Counter(weeks)
    weights = np.array([1/sample[w] for w in weeks])
    present = np.isfinite(x)
    denominator = present.T@weights
    centre = np.divide(np.nansum(x*weights[:,None],axis=0), denominator, out=np.zeros(x.shape[1]), where=denominator>0)
    x = np.where(present,x,centre)-centre
    scale = np.sqrt(np.average(x*x,axis=0,weights=weights))
    scale[scale==0]=1
    x = np.clip(x/scale,-4,4)
    z = np.clip((np.where(np.isfinite(z),z,centre)-centre)/scale,-4,4)
    mean_y = np.average(y,axis=0,weights=weights)
    cross = x.T@(weights[:,None]*(y-mean_y))
    gram = x.T@(weights[:,None]*x)
    return [z@np.linalg.solve(gram+penalty*weights.sum()*np.eye(x.shape[1]),cross)+mean_y for penalty in penalties]


def forecast(history, current):
    observed_issues = [r["release_date"] for r in history]
    if len(history)<150 or len(set(observed_issues))<52:
        return
    px, actual, next_x = (np.asarray(v) for v in ([r["public"] for r in history],
                                    [r["target"] for r in history], [r["public"] for r in current]))
    baseline = fitted_ridge(px,actual,next_x,PENALTIES,observed_issues)
    for i,r in enumerate(current):
        r["public_prediction"] = np.array([values[i] for values in baseline])
    old = [r for r in history if r["optical"] is not None and "public_prediction" in r]
    new = [r for r in current if r["optical"] is not None]
    old_issues = [r["release_date"] for r in old]
    if len(old)<40 or len(set(old_issues))<20 or not new:
        return
    old_x = np.array([r["optical"] for r in old])
    new_x = np.array([r["optical"] for r in new])
    for index, penalty in enumerate(PENALTIES):
        residual = np.array([r["target"]-r["public_prediction"][index] for r in old])
        change, = fitted_ridge(old_x,residual,new_x,(penalty,),old_issues)
        for i,row in enumerate(new):
            row.setdefault("hybrid_prediction",[]).append(row["public_prediction"][index]+change[i])


def penalty_choice(completed, model):
    key = f"{model}_prediction"
    errors = defaultdict(list)
    for r in completed:
        if "hybrid_prediction" in r:
            errors[r["release_date"]].append(np.mean((r[key]-r["target"])**2,axis=1))
    if len(errors)<20:
        return PENALTIES.index(0.5)
    return int(np.argmin(np.mean([np.mean(values,axis=0) for values in errors.values()],axis=0)))


def trading_choices(rows):
    for model in MODELS:
        candidates = []
        for r in rows:
            for index,side in enumerate(SIDES):
                public,hybrid = r["public_active"][index],r["hybrid_active"][index]
                expected = {"public":public,"hybrid":hybrid,"consensus":min(public,hybrid)}[model]
                if expected>=NET_RETURN_HURDLE:
                    candidates.append((expected,side,r))
        if candidates:
            yield model,min(candidates,key=lambda t:(-t[0],t[2]["instrument"],-t[1]))


def replay(book, *, common_holdout_issues=None):
    history, predictions, trades = [], [], []
    last_close = {}
    for issue,rows in sorted(book.items()):
        decision = datetime.fromisoformat(rows[0]["decision_at_utc"])
        completed = [r for r in history if r["known_at"]<decision]
        forecast(completed,rows)
        candidates = [r for r in rows if "hybrid_prediction" in r]
        p,h = (penalty_choice(completed,m) for m in ("public","hybrid"))
        for r in candidates:
            r.update(public_active=r["public_prediction"][p],hybrid_active=r["hybrid_prediction"][h])
            predictions.append(dict(**{k:r[k] for k in ("study_phase","release_date","instrument","decision_at_utc","fifth_close_utc")},
                completed_training_rows=len(completed),completed_training_issues=len({q["release_date"] for q in completed}),
                last_training_outcome_utc=max(q["fifth_close_utc"] for q in completed),public_penalty=PENALTIES[p],hybrid_penalty=PENALTIES[h],
                public_short=r["public_active"][0],public_long=r["public_active"][1],hybrid_short=r["hybrid_active"][0],
                hybrid_long=r["hybrid_active"][1],actual_short=r["target"][0],actual_long=r["target"][1]))
        for model,(score,side,r) in trading_choices(candidates):
            if common_holdout_issues is not None and r["study_phase"]=="holdout" and issue not in common_holdout_issues:
                continue
            if model in last_close and datetime.fromisoformat(r["entry_open_utc"])<=last_close[model]:
                continue
            last_close[model]=r["known_at"]
            trades.append(dict(model=model,study_phase=r["study_phase"],release_date=issue,instrument=r["instrument"],
                instrument_side=side,expected_net_pct=100*score,entry_open_utc=r["entry_open_utc"],fifth_close_utc=r["fifth_close_utc"],
                **r["realized_accounting"][side]))
        history.extend(rows)
    return history,predictions,trades


def main():
    book, excluded = examples()
    history,predictions,trades = replay(book)
    report = dict(penalties=list(PENALTIES),candidate_rows=len(history),candidate_issues=len(book),
        measured_rows=sum(r["optical"] is not None for r in history),common_prediction_rows=len(predictions),
        excluded={f"{a}:{b}":n for (a,b),n in excluded.items()},periods={},
        interpretation="The first stage uses all linked, priced open-period windows. The optical stage learns residuals of "
        "causal public predictions only on measured rows. Hyperparameters use previously completed shared predictions. "
        "This experiment was developed after the original backtest. The 2024-2026 outcomes remain sealed.")
    for phase in PHASES:
        forecasts=[r for r in predictions if r["study_phase"]==phase]
        issues=sorted({r["release_date"] for r in forecasts})
        report["periods"][phase]=dict(predictions=len(forecasts),issues=len(issues),net_prediction_mae_bp={model:
            10_000*float(np.mean([abs(r[f"{model}_{side}"]-r[f"actual_{side}"]) for r in forecasts for side in ("short","long")]))
            for model in ("public","hybrid")},trading={model:statistics([r for r in trades if r["study_phase"]==phase and r["model"]==model],issues)
            for model in MODELS})
    export(f"{STEM}_forecasts",predictions)
    if trades:
        export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
