"""Test whether a given EIA surprise matters more when Cushing stocks are low."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features, replay as original_replay

STEM = "cushing_cl_inventory_level"
METHODS = ("inventory", "level_and_curve", "inventory_with_level", "inventory_with_scarcity")


def replay(cases):
    vintages = {(r["release_date"],r["week_ending"]):r for r in read("eia_inventory_release_panel")}
    forecasts = indexed("cushing_robust_stack_forecasts")
    clocks = indexed("eia_release_clock")
    seen = []
    for r in cases:
        issue = r["release_date"]
        cur = vintages[issue,forecasts[issue]["week_ending"]]
        known = datetime.fromisoformat(clocks[issue]["conservative_postrelease_from_utc"])
        if not known<=r["decision"]<r["entry"]<r["exit_at"] or (
            float(cur["change_kbbl_as_known_at_release"])/1000!=r["x"][0]
        ):
            raise ValueError(f"Inventory release mismatch on {issue}")
        level = float(cur["cushing_kbbl"])/1000
        log_level = math.log(level/40.)
        # Scale the reported surprise by the previous week's same-vintage stock level,
        # so the transformation does not itself move with the current inventory shock.
        previous_level = float(cur["previous_week_kbbl_as_known_at_release"])/1000
        scarcity = r["x"][1]*(40./previous_level-1.)
        _,_,_,inventory,_,curve = features(r)
        vectors = (inventory,(*curve,log_level),(*inventory,log_level),(*inventory,log_level,scarcity,
                   scarcity*math.log(r["volatility_usd"]/1000)))
        current = r | dict(vectors=vectors)
        train = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
        seen.append(current)
        if len(train)<MIN_WINDOWS:
            continue
        for i,(method,x) in enumerate(zip(METHODS,vectors,strict=True)):
            estimate = risk_prediction([q | dict(x=q["vectors"][i]) for q in train],current | dict(x=x),len(x))
            side = int(estimate>HURDLE)-int(estimate<-HURDLE)
            gross = side*r["long_gross"]
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","volatility_usd","alternate_source")} | dict(
                method=method,level_million_bbl=level,previous_level_million_bbl=previous_level,
                eia_source_url=cur["source_url"],published_known_at_utc=known.isoformat(),decision_at_utc=r["decision"].isoformat(),
                entry_at_utc=r["entry"].isoformat(),exit_at_utc=r["exit_at"].isoformat(),latest_training_exit=train[-1]["exit_at"].isoformat(),
                expected_move_usd=estimate,near_side=side,gross_usd=gross,
                alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                **{f"session_{n}_net_170_usd":side*v-(170 if n==5 else 85)*abs(side) for n,v in enumerate(r["spread_path_usd"],2)},
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    original,_,cases = calendar()
    reference = {r["release_date"]:r for r in original_replay(original,cases) if r["method"]=="inventory_interaction"}
    rows = list(replay(cases))
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"],r["method"]].append(r)
        if r["method"]=="inventory" and (r["net_170_usd"]!=reference[r["release_date"]]["net_170_usd"] or
            not np.isclose(r["expected_move_usd"],reference[r["release_date"]]["expected_long_gross_usd"],atol=1e-8,rtol=1e-10)):
            raise ValueError(f"Original inventory model changed on {r['release_date']}")
    report = dict(status="exploratory_replay_on_reviewed_history",phases={})
    for p in PHASES:
        report["phases"][p] = {}
        base = groups[p,"inventory"]
        for method in METHODS:
            group = groups[p,method]
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different calendar in {p}/{method}")
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(group,base,strict=True)]
            report["phases"][p][method] = robustness(group) | daily_risk(group) | dict(
                increment_vs_inventory_usd=sum(delta),increment_per_release_95_usd=block_bootstrap_mean(delta),
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group))
        print(p,{m:(v["windows"],v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in report["phases"][p].items()})
    export(f"{STEM}_trades",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
