"""Evaluate non-overlapping early and delayed trades at the original settlement marks."""

import json
from collections import defaultdict
from datetime import datetime

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_volatility import PHASES
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_cl_two_windows"
EARLY = "cushing_cl_prerelease_trades"
DELAYED = "cushing_cl_volatility_interactions_trades"
INVENTORY = "inventory_interaction"
CURVE = "carry_volatility_context"
POLICIES = {"delayed_inventory": (None,INVENTORY), "delayed_curve": (None,CURVE), "early_curve": ("curve_only",None),
            "early_curve_and_delayed_inventory": ("curve_only",INVENTORY), "early_curve_and_delayed_curve": ("curve_only",CURVE),
            "early_public_and_delayed_inventory": ("early_public_ml",INVENTORY),
            "early_optical_and_delayed_inventory": ("public_and_optical",INVENTORY)}
COSTS = (50,90,170,250,340)


def source_rows():
    for leg,stem in enumerate((EARLY,DELAYED)):
        clock = ("previous_session","report_day")[leg]
        wanted = {pair[leg] for pair in POLICIES.values()}
        for r in read(stem):
            if r["method"] not in wanted:
                continue
            decision,entry,exit_at = (datetime.fromisoformat(r[k]) for k in (
                ("model_information_at_utc","entry_at_utc","exit_at_utc") if leg==0 else
                ("decision_at_utc","entry_at_utc","exit_at_utc")))
            training = datetime.fromisoformat(r["latest_training_exit_utc"])
            side,gross = (int(r[k]) for k in ("near_side","gross_usd"))
            if not training<decision<entry<exit_at or side not in (-1,0,1) or gross and not side:
                raise ValueError(f"Invalid source trade on {r['release_date']}/{clock}")
            if int(r["net_170_usd"])!=gross-170*abs(side):
                raise ValueError(f"Source costs differ on {r['release_date']}/{clock}")
            path = [int(r[f"session_{i}_net_170_usd"]) for i in range(2,6)] if leg else [-85*abs(side),gross-170*abs(side)]
            yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month")} | dict(
                clock=clock,leg=leg,source_model=r["method"],decision=decision,entry=entry,exit_at=exit_at,
                near_side=side,gross_usd=gross,path_170=path)


def book(candidates, portfolio, methods):
    occupied_until = None
    for r in candidates:
        if r["source_model"]!=methods[r["leg"]]:
            continue
        side = r["near_side"]
        conflict = bool(side and occupied_until and r["entry"]<=occupied_until)
        if conflict:
            side = 0
        if side:
            occupied_until = r["exit_at"]
        gross = r["gross_usd"] if side else 0
        path = r["path_170"] if side else [0]*len(r["path_170"])
        if path[-1]!=gross-170*abs(side):
            raise ValueError(f"Inconsistent exit accounting on {r['release_date']}/{portfolio}")
        yield r | dict(portfolio=portfolio,near_side=side,blocked_by_open_spread=conflict,gross_usd=gross,path_170=path,
            **{f"net_{cost}_usd":gross-cost*abs(side) for cost in COSTS})


def results(rows, releases):
    equity = peak = drawdown = 0
    by_issue = dict.fromkeys(releases,0)
    by_month = defaultdict(int)
    realized = []
    for r in rows:
        for p in r["path_170"]:
            peak = max(peak,equity+p)
            drawdown = max(drawdown,peak-equity-p)
        value = r["net_170_usd"]
        equity += value
        by_issue[r["release_date"]] += value
        by_month[r["release_date"][:7]] += value
        if r["near_side"]:
            realized.append(value)
    best = sorted((x for x in realized if x>0),reverse=True)
    return dict(releases=len(releases),windows=len(rows),trades=len(realized),wins=sum(r>0 for r in realized),
        early_trades=sum(r["leg"]==0 and r["near_side"]!=0 for r in rows),blocked_by_open_spread=sum(r["blocked_by_open_spread"] for r in rows),
        daily_drawdown_170_usd=drawdown,gross_usd=sum(r["gross_usd"] for r in rows),worst_trade_170_usd=min(realized,default=0),
        **{f"net_{c}_usd":sum(r[f"net_{c}_usd"] for r in rows) for c in COSTS},
        net_170_without_three_best_usd=equity-sum(best[:3]),net_170_without_april_2026_usd=equity-by_month["2026-04"],
        net_170_without_best_month_usd=equity-max(by_month.values(),default=0),
        net_170_by_year={year:sum(v for d,v in by_issue.items() if d.startswith(year)) for year in sorted({d[:4] for d in releases})}),by_issue


def main():
    candidates = sorted(source_rows(),key=lambda r:(r["entry"],r["leg"],r["release_date"]))
    if len({(r["release_date"],r["clock"],r["source_model"]) for r in candidates})!=len(candidates):
        raise ValueError("Repeated source trade")
    forecasts = read("cushing_robust_stack_forecasts")
    rows = [r for name,methods in POLICIES.items() for r in book(candidates,name,methods)]
    report = dict(status="exploratory_replay_on_reviewed_history",max_overlapping_positions=1,phases={})
    for p in PHASES:
        releases = sorted(r["release_date"] for r in forecasts if r["study_phase"]==p)
        groups = {m:results([r for r in rows if r["portfolio"]==m and r["study_phase"]==p],releases) for m in POLICIES}
        out = report["phases"][p] = {}
        for name,(stats,net) in groups.items():
            out[name] = stats | {f"delta_vs_{control}":dict(
                total_usd=sum(delta),paired_mean_95_usd=block_bootstrap_mean(delta))
                for control in ("delayed_inventory","early_curve_and_delayed_curve")
                for delta in [[net[k]-groups[control][1][k] for k in releases]]}
        print(p,{m:(v["trades"],v["net_170_usd"],v["daily_drawdown_170_usd"],v["blocked_by_open_spread"]) for m,v in out.items()})
    export(f"{STEM}_trades",[{k:v for k,v in r.items() if k not in ("path_170","leg")} |
        dict(path_170_json=json.dumps(r["path_170"])) for r in rows])
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
