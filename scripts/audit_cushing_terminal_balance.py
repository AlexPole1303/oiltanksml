"""Separate the effect of learned error variances, terminal balance, and shrinking the optical signal."""

import json

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_robust_stack import indexed
from fit_cushing_terminal_balance import MAX_SCENE_AGE, source_rows, update

STEM = "cushing_terminal_balance_ablation"
MODELS = (PUBLIC,"carried_scene","terminal_learned_direct","scene_direct","terminal_fixed_direct","without_transfer_covariance")


def replay():
    recorded = indexed("cushing_terminal_balance_forecasts")
    past = []
    for r,old in source_rows():
        stamp = r["cutoff"].isoformat()
        saved = recorded[r["release_date"]]
        if saved["cutoff_utc"]!=stamp or float(saved["actual"])!=r["actual"] or int(saved["tanks"])!=len(r["observed"]):
            raise ValueError(f"Terminal forecast inputs changed on {r['release_date']}")
        variances = np.square([float(saved[k]) for k in ("noise_scale","shared_scene_sigma_m","terminal_transfer_sigma_mbbl")])
        available = [q for q in past if q["known"]<r["cutoff"] and r["cutoff"]-q["cutoff"]<=MAX_SCENE_AGE]
        without_transfer = 0.
        if r["observed"]:
            variances[-1] = 0.
            without_transfer,_ = update(r,available,variances)
            past.append(r)
        base = r[PUBLIC]
        yield r | dict(cutoff_utc=stamp,known_utc=r["known"].isoformat(),tanks=len(r["observed"]),
                carried_scene=old["carried_scene"],terminal_learned_direct=float(saved["terminal_learned_direct"]),
                scene_direct=base+old["carried_raw"],terminal_fixed_direct=base+float(saved["fixed_raw"]),
                without_transfer_covariance=base+without_transfer)


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        trades = list(trading(rows,MODELS))
    report = dict(status="historical_ablation_on_reviewed_period",forecasts={},trading={})
    for phase in PHASES:
        sample = [r for r in rows if r["study_phase"]==phase]
        measured = [r for r in sample if r["tanks"]]
        report["forecasts"][phase] = {}
        for scope,test in (("all",sample),("measured",measured)):
            stats = forecast_metrics(test,MODELS)
            changes = [1000*(abs(r["actual"]-r["terminal_learned_direct"])-abs(r["actual"]-r["without_transfer_covariance"])) for r in test]
            report["forecasts"][phase][scope] = stats | dict(covar_swap_delta_mae_kbbl=float(np.mean(changes)),
                                                           covar_swap_delta_95_kbbl=block_bootstrap_mean(changes))
        base = [r for r in trades if r["study_phase"]==phase and r["method"]==PUBLIC]
        by_method = {}
        for method in ("curve",*MODELS):
            group = [r for r in trades if r["study_phase"]==phase and r["method"]==method]
            if [r["release_date"] for r in group]!=[r["release_date"] for r in base]:
                raise ValueError(f"Different market calendar on {phase}/{method}")
            delta = [r["net_170_usd"]-ref["net_170_usd"] for r,ref in zip(group,base,strict=True)]
            by_method[method] = robustness(group) | daily_risk(group) | dict(paired_vs_public_95_usd=block_bootstrap_mean(delta))
        report["trading"][phase] = by_method
        print(phase,{m:round(report["forecasts"][phase]["measured"]["models"][m]["mae_kbbl"],2) for m in MODELS})
        print(phase,{m:(v["trades"],v["net_170_usd"],v["daily_drawdown_usd"]) for m,v in by_method.items()})
    export(f"{STEM}_forecasts",[{k:r[k] for k in ("release_date","study_phase","actual","cutoff_utc","known_utc","tanks",*MODELS)}
                                for r in rows])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
