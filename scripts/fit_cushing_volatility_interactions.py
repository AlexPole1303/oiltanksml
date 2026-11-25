"""Test whether observed CL volatility changes the value of the inventory surprise."""

import json
import math
from collections import defaultdict

import numpy as np

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_cl_volatility_interactions"
MODELS = ("public", "carry", "volatility_context", "inventory_interaction", "inventory_interaction_optical",
          "carry_volatility_context", "settled_performance_blend")


def features(r):
    target,surprise,carry,carry_delta,interaction,optical,_ = r["x"]
    vol = math.log(r["volatility_usd"]/1000)
    public = (target,surprise,carry,carry_delta,interaction)
    curve = (carry,carry_delta)
    context = (*public,vol,carry*vol,carry_delta*vol)
    inventory = (*context,target*vol,surprise*vol)
    curve_context = (*curve,vol,carry*vol,carry_delta*vol)
    return public,curve,context,inventory,(*inventory,optical,optical*vol),curve_context


def trade(estimate, actual):
    side = int(estimate>HURDLE)-int(estimate<-HURDLE)
    return side,side*actual


def replay(original, cases):
    anchor = {r["release_date"]:r for r in original if r["method"]=="public"}
    seen = []
    evaluated = []
    for r in cases:
        if not r["decision"]<r["entry"]<r["exit_at"]:
            raise ValueError(f"Invalid trading clocks for {r['release_date']}")
        current = r | dict(vector=features(r))
        training = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        seen.append(current)
        if len(training)<MIN_WINDOWS:
            continue
        values = []
        for j,x in enumerate(current["vector"]):
            values.append(risk_prediction([q | dict(x=q["vector"][j]) for q in training],current | dict(x=x),len(x)))
        signal_check = anchor[r["release_date"]]["expected_long_gross_usd"]
        if not math.isclose(values[0],signal_check,abs_tol=1e-9):
            raise ValueError(f"Original delayed model did not reproduce on {r['release_date']}")
        p = np.asarray([trade(value,r["long_gross"]) for value in values[:5]])
        recent = [pnl for stamp,pnl in evaluated if stamp<r["decision"]][-52:]
        # Weight arms by prior stressed P&L. Cap every historical scoring observation
        # so the choice is not set by a single large outcome.
        weights = np.array([.5,.2,.15,.1,.05])
        if len(recent)>=26:
            mean = np.mean(recent,axis=0)
            weights *= np.exp(mean/125)
        weights /= weights.sum()
        values.append(float(weights@values[:5]))
        evaluated.append((r["exit_at"],np.clip(p[:,1]-HURDLE*np.abs(p[:,0]),-1500,1500)))
        for method,value in zip(MODELS,values,strict=True):
            side,gross = trade(value,r["long_gross"])
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd")} | dict(
                method=method,decision_at_utc=r["decision"].isoformat(),entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),
                training_windows=len(training),latest_training_exit_utc=training[-1]["exit_at"].isoformat(),expected_long_gross_usd=value,
                side_policy_weights_json=json.dumps(dict(zip(MODELS[:5],map(float,weights),strict=True))) if method==MODELS[-1] else None,
                near_side=side,gross_usd=gross,alternate_source=r["alternate_source"],
                alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*v-(170 if n==5 else 85)*abs(side) for n,v in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},
                **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    old,_,history = calendar()
    rows = list(replay(old,history))
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
    phases, ablation = {}, {}
    for p in PHASES:
        base = groups[p,"public"]
        phases[p] = {}
        for m in MODELS:
            g = groups[p,m]
            if [r["release_date"] for r in g]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different calendar in {p}/{m}")
            delta = [r["net_170_usd"]-ref["net_170_usd"] for r,ref in zip(g,base,strict=True)]
            phases[p][m] = robustness(g) | daily_risk(g) | dict(delta_vs_public_usd=sum(delta),paired_delta_ci_95_usd=block_bootstrap_mean(delta))
        ablation[p] = {}
        for a,b in (("inventory_interaction","carry_volatility_context"),("inventory_interaction","volatility_context"),
                    ("inventory_interaction_optical","inventory_interaction")):
            delta = [r["net_170_usd"]-ref["net_170_usd"] for r,ref in zip(groups[p,a],groups[p,b],strict=True)]
            ablation[p][f"{a}_vs_{b}"] = dict(total_usd=sum(delta),mean_95_usd=block_bootstrap_mean(delta))
    export(f"{STEM}_trades",rows)
    report = dict(status="exploratory_regime_replay",phases=phases,matched_information_ablations=ablation)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2)+"\n").encode())
    for p,g in phases.items():
        print(p,{m:(r["trades"],r["net_170_usd"],r["stress_net_excluding_three_largest_wins_usd"],r["stress_net_without_best_month_usd"])
                 for m,r in g.items()})


if __name__=="__main__":
    main()
