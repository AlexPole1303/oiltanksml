"""Measure a fixed pre-EIA trading rule's sensitivity to hypothetical forecast accuracy."""

import json
import math
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, risk_prediction
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_prerelease_cl import cases
from fit_cushing_prerelease_sensor import event_response, replay

STEM = "cushing_prerelease_accuracy_value"
SHARES = (.25,.5,1.)
TRADES = "cushing_prerelease_sensor_trades"


def scenarios():
    sibling = {(r["release_date"],r["method"]):r for r in read(TRADES)}
    with threadpool_limits(limits=1):
        marked,_ = cases()
        rows = list(replay(marked))
    history = []
    for f,available in rows:
        issue = f["issue"]
        if not available:
            history.append(f)
            continue
        q = sibling[issue,"curve"]
        r = sibling[issue,"symmetric_release_response"]
        start,report,end = (datetime.fromisoformat(q[k]) for k in ("entry_at_utc","scheduled_report_at_utc","exit_at_utc"))
        cutoff = f["decision"]
        if not cutoff<start<report<end<f["known"] or q["study_phase"]!=f["study_phase"]:
            raise ValueError(f"Mismatched early decision clocks on {issue}")
        early,actual,optical = (float(f[k]) for k in ("early_public_ml","inventory","early_recovered_scene"))
        base = float(q["expected_long_move_usd"])
        past = [p for p in history if p["exit_at"]<cutoff and p["known"]<cutoff][-HISTORY:]
        if len(past)<MIN_WINDOWS:
            raise ValueError(f"The early experiment lacks settled release data on {issue}")
        slope = event_response(past,f,base,1.,symmetric=True)-base
        if not math.isclose(base+slope*(f["early_adaptive_scene"]-early),float(r["expected_long_move_usd"]),abs_tol=1e-8):
            raise ValueError(f"Original early event-response model changed on {issue}")
        movement = f["long_gross"]
        paths = [("public",early),("satellite",optical),*((f"hypothetical_{share:g}",
                 (1-share)*optical+share*actual) for share in SHARES)]
        for scenario,point in paths:
            yield payoff(q,scenario,point,actual,base,slope,base+slope*(point-early),movement)
        # Fit an unrestricted risk-normalized market response to these same
        # oracle forecasts. Old actuals are usable at each train cutoff, while
        # the today's perfect or half-error point still requires a future label.
        for share in (0.,.5,1.):
            train = [p | dict(x=direct_design(p,(1-share)*p["early_recovered_scene"]+share*p["inventory"])) for p in past]
            point = (1-share)*optical+share*actual
            design = direct_design(f,point)
            move = risk_prediction(train,f | dict(x=design),len(design))
            name = "refit_satellite" if share==0. else f"hypothetical_refit_{share:g}"
            yield payoff(q,name,point,actual,base,None,move,movement)
        history.append(f)


def direct_design(row, point):
    curve = row["design"][0]
    surprise = point-row["early_public_ml"]
    return (*curve,point,surprise,surprise*curve[-1])


def payoff(row, scenario, point, actual, base, slope, move, movement):
    side = int(move>HURDLE)-int(move< -HURDLE)
    gross = side*movement
    return {k:row[k] for k in ("release_date","study_phase","near_month","far_month","entry_at_utc","exit_at_utc")} | dict(
        scenario=scenario,hypothetical=scenario.startswith("hypothetical_"),
        forecast_kbbl=1000*point,actual_kbbl=1000*actual,curve_move_usd=base,estimated_move_usd=move,
        inventory_sensitivity_usd_per_million_bbl=slope,near_side=side,gross_usd=gross,
        net_170_usd=gross-170*abs(side),net_340_usd=gross-340*abs(side))


def main():
    rows = list(scenarios())
    report = dict(status="untradable_counterfactual_accuracy_test",scenario_uses_future_label={
        name:name.startswith("hypothetical_") for name in dict.fromkeys(r["scenario"] for r in rows)},phases={})
    for phase in PHASES:
        names = report["scenario_uses_future_label"]
        groups = {m:[r for r in rows if r["study_phase"]==phase and r["scenario"]==m] for m in names}
        satellite = groups["satellite"]
        block = report["phases"][phase] = {}
        for name,group in groups.items():
            if not group or [r["release_date"] for r in group]!=[r["release_date"] for r in satellite]:
                raise ValueError(f"The accuracy experiment uses a different calendar for {phase}/{name}")
            pnl = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(group,satellite,strict=True)]
            block[name] = dict(windows=len(group),trades=sum(abs(r["near_side"]) for r in group),
                mae_kbbl=float(np.mean([abs(r["forecast_kbbl"]-r["actual_kbbl"]) for r in group])),
                gross_usd=sum(r["gross_usd"] for r in group),net_170_usd=sum(r["net_170_usd"] for r in group),
                net_340_usd=sum(r["net_340_usd"] for r in group),mean_increment_vs_satellite_95_usd=block_bootstrap_mean(pnl))
        print(phase,{m:(v["trades"],round(v["mae_kbbl"],2),v["net_170_usd"],v["net_340_usd"]) for m,v in block.items()})
    export(f"{STEM}_scenarios",rows)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
