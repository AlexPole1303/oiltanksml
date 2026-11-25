"""Forecast whether a current satellite correction reduces absolute error."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_windows, trade
from fit_cushing_cl_volatility import HURDLE
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_late_pair_recovery import COMBINED, STEM as LATE
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import trading as base_trades
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_robust_stack import indexed
from fit_cushing_terminal_weekly_transfers import LIMITED_MARKET as BASE, STEM as TERMINAL
from fit_cushing_volatility_neighbors import STEM as MARKET

STEM = "cushing_optical_value"
NO_OPTICS = "remove_current_optical"
PARTIAL = "half_current_optical"
METHODS = ("average_value_admission", "coverage_value_admission", "coverage_risk_blend", "pooled_median_gain", "coverage_median_gain")
MODELS = (PUBLIC, SUPPLEMENT, BASE, NO_OPTICS, PARTIAL, *METHODS)
MARKET_REFERENCE = "volatility_reference"
MARKET_METHODS = (MARKET_REFERENCE, "model_median", "model_direction_agreement", "model_cost_agreement", "model_range_penalty")
FRACTIONS = np.asarray((0.,.5,1.))
MIN_MEASURED = 40
LOOKBACK = 104


def dated_inputs():
    sources = {name:indexed(f"{name}_forecasts") for name in (TERMINAL,LATE,OUTER,MARKET,"cushing_optical_stability")}
    baseline = sources[TERMINAL]
    if any(book.keys() != baseline.keys() for book in sources.values()):
        raise ValueError("Optical forecasts do not use the same release panel")
    for issue,q in sorted(baseline.items()):
        cutoff,known = (datetime.fromisoformat(q[k]) for k in ("cutoff","known"))
        latest = sources[LATE][issue]
        expanded = sources[OUTER][issue]
        calibration = sources[MARKET][issue]
        stability = sources["cushing_optical_stability"][issue]
        for row in (q,latest,expanded,calibration,stability):
            if (datetime.fromisoformat(row.get("cutoff",row.get("cutoff_utc"))) != cutoff
                    or datetime.fromisoformat(row.get("known",row.get("known_utc"))) != known
                    or row["study_phase"] != q["study_phase"] or float(row["actual"]) != float(q["actual"])
                    or any(datetime.fromisoformat(v)>=cutoff for k,v in row.items() if v and k.endswith((
                        "training_label_utc","latest_label_utc","latest_optical_at_utc","latest_inventory_training_utc",
                        "latest_state_error_known_utc")))):
                raise ValueError(f"Inconsistent source cutoff on {issue}")
        if not cutoff < known or float(q[COMBINED]) != float(latest[COMBINED]):
            raise ValueError(f"The reference forecast differs on {issue}")
        observed_roofs = int(q["roofs"])
        late_roofs = int(latest["late_roofs"])
        optical_input = observed_roofs > 0 or late_roofs > 0
        public = float(q[PUBLIC])
        optical_control = public if observed_roofs else float(latest["basis_when_unmeasured"])
        satellite_update = float(q[COMBINED])-optical_control+float(q[BASE])-float(q["market_error_median"])
        if (bool(observed_roofs) != (latest["measured"] == "True") or not optical_input and abs(satellite_update)>1e-12
                or float(q["market_error_median"])!=float(calibration["market_error_median"])):
            raise ValueError(f"Missing contemporaneous optical signal on {issue}")
        satellite = float(q[BASE])
        control = satellite-satellite_update
        values = control+FRACTIONS*satellite_update
        actual = float(q["actual"])
        optical_type = "native" if observed_roofs else "late_only" if late_roofs else "unmeasured"
        info = float(q["learned_information"] if observed_roofs else latest["information"])
        diagnostics = np.array((math.log1p(observed_roofs or late_roofs)/4., int(q["terminals"])/5.,
            math.log1p(info), abs(satellite_update)/.25, abs(control)/.75,
            float(stability["terminal_deletion_scale"])/.25, float(int(expanded["outer_roofs"])>0), float(optical_type=="late_only")))
        if not np.isfinite(diagnostics).all():
            raise ValueError(f"Invalid optical diagnostics on {issue}")
        yield dict(release_date=issue,week_ending=q["week_ending"],study_phase=q["study_phase"],cutoff=cutoff,known=known,
                   actual=actual,optical_type=optical_type,optical_roofs=observed_roofs or late_roofs,optical_update=satellite_update,
                   diagnostics=diagnostics,available_forecasts=values,loss=np.abs(values-actual),
                   **{PUBLIC:public,SUPPLEMENT:float(q[SUPPLEMENT]),BASE:satellite,NO_OPTICS:control,PARTIAL:float(values[1])})


def estimated_loss(history, now, *, conditional):
    """Model incremental absolute loss for half and full current corrections."""
    targets = np.stack([q["loss"][1:]-q["loss"][0] for q in history])
    time = np.asarray([(now["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in history])
    # A near-zero prior treats extra optical signal as unproven on a small panel.
    weight = np.exp2(-time/78.)
    if conditional:
        design = np.stack([q["diagnostics"] for q in history])
        # Use fixed units and penalized slopes so rare coverage patterns do
        # not determine the prediction from one or two weeks.
        center = np.average(design, axis=0, weights=weight)
        z = now["diagnostics"]-center
        x = np.column_stack((np.ones(len(history)), design-center))
        penalty = np.diag([8., *([18.]*len(z))])
        scales = np.array((.45,.5,.45,1.,1.,.6,1.,1.))
        distance = np.square(np.clip((design-now["diagnostics"])/scales,-4.,4.)).mean(axis=1)
        w = weight*(.35+.65*np.exp(-.5*distance))
        params = solve(x.T@(w[:,None]*x)+penalty, x.T@(w[:,None]*targets), assume_a="pos")
        difference = np.r_[1.,z]@params
    else:
        difference = weight@targets/(weight.sum()+8.)
    return np.r_[0.,difference]


def median_gain(history, now, *, conditional):
    updates = np.asarray([q["optical_update"] for q in history])
    residuals = np.asarray([q["actual"]-q[NO_OPTICS] for q in history])
    elapsed = np.asarray([(now["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in history])
    weights = np.exp2(-elapsed/78.)
    if conditional:
        design = np.stack([q["diagnostics"] for q in history])
        distances = np.square((design-now["diagnostics"])/np.array((.45,.5,.45,1.,1.,.6,1.,1.))).mean(axis=1)
        weights *= .35+.65*np.exp(-.5*distances)
    # |residual - gain*update| equals |update| times the absolute
    # distance to the implied gain. Add prior mass at the original gain 1.
    valid = abs(updates)>1e-10
    targets = np.r_[residuals[valid]/updates[valid],1.]
    mass = np.r_[abs(updates[valid])*weights[valid],8*.20]
    order = np.argsort(targets,kind="stable")
    position = np.searchsorted(np.cumsum(mass[order]),mass.sum()/2)
    return float(np.clip(targets[order[position]],0.,1.5))


def forecasts():
    history = []
    for r in dated_inputs():
        prior = [q for q in history if q["known"]<r["cutoff"]][-LOOKBACK:]
        all_loss = local_loss = np.zeros(len(FRACTIONS))
        weights = (1.,)*len(METHODS)
        if r["optical_roofs"] and len(prior)>=MIN_MEASURED:
            all_loss = estimated_loss(prior,r,conditional=False)
            local_loss = estimated_loss(prior,r,conditional=True)
            # Expected improvements under five thousand barrels give weak
            # support for a switch, measured in millions of barrels here.
            switching = np.asarray((.005,.0025,0.))
            advantage = local_loss+switching
            policy = np.exp(-(advantage-advantage.min())/.04)*np.array((1.,1.,2.))
            weights = (float(FRACTIONS[np.argmin(all_loss+switching)]), float(FRACTIONS[np.argmin(advantage)]),
                       float((policy@FRACTIONS)/policy.sum()),
                       *(median_gain(prior,r,conditional=flag) for flag in (False,True)))
        if r["optical_roofs"]:
            history.append(r)
        output = {m:r[NO_OPTICS]+w*r["optical_update"] for m,w in zip(METHODS,weights,strict=True)}
        yield {k:r[k] for k in ("release_date","week_ending","study_phase","cutoff","known","actual",
                                 "optical_type","optical_roofs","optical_update",*MODELS[:5])} | output | dict(
            training_weeks=len(prior), latest_training_outcome_utc=prior[-1]["known"] if prior else None,
            expected_full_vs_no_optical_kbbl=1000*float(local_loss[-1]),
            **{f"{m}_weight":w for m,w in zip(METHODS,weights,strict=True)})


def metrics(panel):
    if not panel:
        return dict(weeks=0,models={})
    abs_error = {m:np.array([1000*abs(r[m]-r["actual"]) for r in panel]) for m in MODELS}
    return dict(weeks=len(panel),models={m:dict(mae_kbbl=float(a.mean()),better_than_reference=int(np.sum(a<abs_error[BASE])),
        mae_delta_vs_reference_95_kbbl=block_bootstrap_mean(a-abs_error[BASE]),
        mae_delta_vs_current_input_removed_95_kbbl=block_bootstrap_mean(a-abs_error[NO_OPTICS])) for m,a in abs_error.items()})


def disagreement_trades(refits):
    archive = {q["release_date"]:q for q in read(f"{MARKET}_trades") if q["method"] == f"{SUPPLEMENT}:volatility"}
    fresh = {(q["release_date"],q["method"]):q for q in refits if q["method"] in (PUBLIC,BASE)}
    for window,_ in market_windows():
        day = window["release_date"]
        if day not in archive:
            continue
        matching = [archive[day],fresh[day,PUBLIC],fresh[day,BASE]]
        for q in matching:
            if (any(q[k]!=window[k] for k in ("near_month","far_month")) or any(
                    datetime.fromisoformat(q[k])!=window[field] for k,field in (
                        ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at"))) or any(
                    datetime.fromisoformat(q[k])>=window["decision"] for k in (
                        "latest_training_exit_utc","latest_calibration_exit_utc") if q[k])):
                raise ValueError(f"Inconsistent disagreement inputs on {day}")
        values = np.asarray([float(q["expected_long_move_usd"]) for q in matching])
        reference = float(values[0])
        direction = float(np.sign(reference))
        low = min(direction*values)
        span = float(np.ptp(values))
        choices = (reference,float(np.median(values)),reference if low>0 else 0.,
                   reference if low>HURDLE else 0., direction*max(0.,abs(reference)-.25*span))
        training = max(datetime.fromisoformat(q["latest_training_exit_utc"]) for q in matching)
        calibration = max((datetime.fromisoformat(q["latest_calibration_exit_utc"]) for q in matching if q["latest_calibration_exit_utc"]),default=None)
        for m,value in zip(MARKET_METHODS,choices,strict=True):
            r = trade(window,m,value,3,training_at=training,calibration_at=calibration)
            if m==MARKET_REFERENCE and (str(r["near_side"])!=matching[0]["near_side"] or str(r["gross_usd"])!=matching[0]["gross_usd"]):
                raise ValueError(f"The volatility reference differs on {day}")
            yield r | dict(public_estimated_move_usd=float(values[1]),hybrid_estimated_move_usd=float(values[2]),forecast_range_usd=span)


def main():
    with threadpool_limits(limits=1):
        rows = list(forecasts())
        base = list(base_trades(rows,MODELS))
        trades = list(replay_conditional_market(rows,base,MODELS,references={PUBLIC:f"{PUBLIC}_conditional",SUPPLEMENT:f"{SUPPLEMENT}_conditional"}))
        disagreement = list(disagreement_trades(trades))
    report = dict(status="exploratory_on_reviewed_history",inventory={},trading={})
    groups = defaultdict(list)
    for r in (*trades,*disagreement):
        groups[r["study_phase"],r["method"]].append(r)
    for phase in PHASES:
        panel = [r for r in rows if r["study_phase"]==phase]
        opt = [r for r in panel if r["optical_roofs"]]
        report["inventory"][phase] = {k:metrics(g) for k,g in (("all",panel),("optical",opt),
            ("native",[r for r in opt if r["optical_type"]=="native"]),
            ("late_only",[r for r in opt if r["optical_type"]=="late_only"]))}
        report["inventory"][phase]["optical_weights"] = {m:dict(mean=float(np.mean([r[f"{m}_weight"] for r in opt])),
            cut_weeks=sum(r[f"{m}_weight"]<1. for r in opt), amplified_weeks=sum(r[f"{m}_weight"]>1. for r in opt)) for m in METHODS}
        result = report["trading"][phase] = {}
        for m in (*MODELS,*MARKET_METHODS):
            reference = groups[phase,MARKET_REFERENCE if m in MARKET_METHODS else BASE]
            sample = groups[phase,m]
            if not sample or [r["release_date"] for r in sample]!=[q["release_date"] for q in reference]:
                raise ValueError(f"Market observations differ on {phase}/{m}")
            delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(sample,reference,strict=True)]
            result[m] = robustness(sample) | daily_risk(sample) | dict(net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in sample),
                net_delta_per_release_95_usd=block_bootstrap_mean(delta),changed_entries=sum(r["near_side"]!=q["near_side"] for r,q in zip(sample,reference,strict=True)))
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in report["inventory"][phase]["all"]["models"].items()},
              {m:(v["trades"],v["net_170_usd"],v["net_340_usd"],v["daily_drawdown_usd"]) for m,v in result.items()},flush=True)
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    export(f"{STEM}_disagreement_trades",disagreement)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
