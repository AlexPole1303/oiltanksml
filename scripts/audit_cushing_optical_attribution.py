"""Isolate today's optical revision from changes in the fitted CL model."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_public_state import OPTICAL
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed

STEM = "cushing_optical_attribution"
HYBRID = "basis_when_unmeasured"
SWAP = {
    "public": (PUBLIC, PUBLIC),
    "satellite": (SUPPLEMENT, SUPPLEMENT),
    "public_fit_satellite_input": (PUBLIC, SUPPLEMENT),
    "satellite_fit_public_input": (SUPPLEMENT, PUBLIC),
    "hybrid": (HYBRID, HYBRID),
    "public_fit_hybrid_input": (PUBLIC, HYBRID),
    "hybrid_fit_public_input": (HYBRID, PUBLIC),
}
EXTRA = ("half_current_image", "satellite_after_agreement")
MODELS = (*SWAP, *EXTRA)


def inventory():
    state = indexed("cushing_public_state_forecasts")
    outer = indexed(f"{OUTER}_forecasts")
    space = indexed("cushing_optical_variance_budget_forecasts")
    supplement = defaultdict(float)
    for row in read("cushing_outer_capture_pairs_supplement_fits"):
        if row["registered_both_shadow_diagnostics_pass"]=="True":
            issue = row["release_date"]
            if datetime.fromisoformat(row["latest_geometry_available_utc"])>=datetime.fromisoformat(state[issue]["cutoff"]):
                raise ValueError(f"Supplementary image unavailable on {issue}")
            supplement[issue] += float(row["circular_area_m2"])
    for issue,q in sorted(state.items()):
        r, v = outer[issue], space[issue]
        cutoff, known = (datetime.fromisoformat(q[k]) for k in ("cutoff", "known"))
        actual = float(q["actual"])
        if (not cutoff<known or any(datetime.fromisoformat(item["cutoff_utc"])!=cutoff
                or datetime.fromisoformat(item["known_utc"])!=known or float(item["actual"])!=actual for item in (r,v))
                or any(datetime.fromisoformat(value)>=cutoff for item in (r,q,v) for key,value in item.items()
                       if key.endswith(("latest_label_utc","training_label_utc")) and value)):
            raise ValueError(f"Mismatched inventory vintage on {issue}")
        public = float(q[PUBLIC])
        if public!=float(r[PUBLIC]) or float(q[OPTICAL])!=float(r[OPTICAL]):
            raise ValueError(f"Mismatched inventory comparator on {issue}")
        base_roofs, outer_roofs, extra_roofs = (int(r[k]) for k in ("baseline_roofs","outer_roofs","supplement_roofs"))
        satellite = float(r[SUPPLEMENT])
        if bool(extra_roofs)!=bool(supplement[issue]) or (not (base_roofs or outer_roofs or extra_roofs) and satellite!=public):
            raise ValueError(f"Optical observations differ from forecast coverage on {issue}")
        yield dict(release_date=issue, week_ending=q["week_ending"], study_phase=q["study_phase"],
            cutoff=cutoff, known=known, actual=actual, baseline_roofs=base_roofs, spanning_roofs=outer_roofs,
            supplemental_roofs=extra_roofs, supplemental_area_m2=supplement[issue],
            baseline_area_time=float(v["space_time_fraction"]),
            **{PUBLIC:public,SUPPLEMENT:satellite,HYBRID:float(q[HYBRID]),OPTICAL:float(q[OPTICAL])})


def base_estimates():
    sources = {PUBLIC: f"{OUTER}_trades", SUPPLEMENT: f"{OUTER}_trades", HYBRID: "cushing_public_state_trades"}
    source = {name:read(name) for name in set(sources.values())}
    anchor = {}
    baseline = {}
    for m,name in sources.items():
        for r in source[name]:
            if r["method"]==m:
                baseline[r["release_date"],m] = r
        conditional = source[name] if m!=HYBRID else read("cushing_public_state_conditional_trades")
        key = f"{m}_conditional" if m!=HYBRID else f"conditional_{m}"
        for r in conditional:
            if r["method"]==key:
                anchor[r["release_date"],m] = r
    return baseline, anchor


def estimate(history, current, fit, model):
    expectation = current["inventory"][model]
    x = replace_expectation(current,expectation)
    trained = [r | dict(x=r["design"][fit]) for r in history]
    point = risk_prediction(trained,current | dict(x=x),len(x))
    calibration = [r | dict(base_estimate=r["estimates"][fit],response_features=r["response"][fit])
                   for r in history if fit in r["estimates"]]
    context = current | dict(response_features=current["response"][model])
    adjustment,_ = conditional_residual(calibration,context) if len(calibration)>=MIN_WINDOWS else (0.,0.)
    return point, adjustment


def replay_conditional_market(rows, base_rows, methods, *, references):
    forecasts = {r["release_date"]:r for r in rows}
    base = {(r["release_date"],r["method"]):r for r in base_rows}
    anchors = {(r["release_date"],r["method"]):r for r in read(f"{OUTER}_trades")
               if r["method"] in (f"{PUBLIC}_conditional",f"{SUPPLEMENT}_conditional")}
    histories = defaultdict(list)
    for case, controls in inputs():
        if "inventory_conditional" not in controls:
            continue
        issue, decision = (case[k] for k in ("release_date","decision"))
        f = forecasts[issue]
        if not f["cutoff"]<f["known"]<=decision<case["entry"] or f["actual"]!=case["x"][0]:
            raise ValueError(f"Unmatched inventory or trading calendar on {issue}")
        for m in methods:
            source = base[issue,m]
            latest_base = datetime.fromisoformat(source["settled_training_label_utc"])
            if (not latest_base<decision or any(source[k]!=case[k] for k in ("near_month","far_month"))
                    or any(datetime.fromisoformat(source[k])!=case[field] for k,field in (
                    ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))):
                raise ValueError(f"Invalid ongoing trading record on {issue}/{m}")
            point = float(source["expected_long_move_usd"])
            current = case | dict(base_estimate=point,response_features=features(
                case["x"][2],case["volatility_usd"],f["actual"]-f[m]))
            past = [r for r in histories[m] if r["exit_at"]<decision][-HISTORY:]
            adjustment,_ = conditional_residual(past,current) if len(past)>=MIN_WINDOWS else (0.,0.)
            histories[m].append(current)
            estimate = point+adjustment
            ref = references.get(m)
            if ref and not math.isclose(estimate,float(anchors[issue,ref]["expected_long_move_usd"]),abs_tol=1e-8):
                raise ValueError(f"Conditional market control changed on {issue}/{m}")
            yield trade(case,m,estimate,3,training_at=latest_base,calibration_at=past[-1]["exit_at"] if past else None)


def market(forecasts):
    base, anchors = base_estimates()
    history = []
    for case, _ in inputs():
        issue, decision = (case[k] for k in ("release_date","decision"))
        f = forecasts[issue]
        if not f["cutoff"] < f["known"] <= decision < case["entry"] or f["actual"]!=case["x"][0]:
            raise ValueError(f"Unmatched EIA or market inputs on {issue}")
        inventory = {m:f[m] for m in (PUBLIC,SUPPLEMENT,HYBRID)}
        response = {m:features(case["x"][2],case["volatility_usd"],f["actual"]-v) for m,v in inventory.items()}
        design = {m:replace_expectation(case,v) for m,v in inventory.items()}
        estimates = {m:float(base[issue,m]["expected_long_move_usd"]) for m in inventory if (issue,m) in base}
        row = case | dict(inventory=inventory,design=design,response=response,estimates=estimates)
        past = [r for r in history if r["exit_at"]<decision][-HISTORY:]
        history.append(row)
        if (issue,PUBLIC) not in anchors:
            continue
        if len(past)<MIN_WINDOWS or any(
                datetime.fromisoformat(base[issue,m]["settled_training_label_utc"])>=decision
                or any(quote[k]!=case[k] for k in ("near_month","far_month"))
                or any(datetime.fromisoformat(quote[k])!=case[field] for k,field in (
                    ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))
                for m in inventory for quote in (base[issue,m],anchors[issue,m])):
            raise ValueError(f"Unexpected market training sample on {issue}")
        points = {m:estimate(past,row,*fields) for m,fields in SWAP.items()}
        values = {m:point+adjustment for m,(point,adjustment) in points.items()}
        for method, m in (("public",PUBLIC),("satellite",SUPPLEMENT),("hybrid",HYBRID)):
            if (not math.isclose(points[method][0],estimates[m],abs_tol=1e-8)
                    or not math.isclose(values[method],float(anchors[issue,m]["expected_long_move_usd"]),abs_tol=1e-8)):
                raise ValueError(f"Market attribution failed to reproduce its control on {issue}/{method}")
        public, satellite, current = (values[k] for k in ("public","satellite","public_fit_satellite_input"))
        same_side = (public>HURDLE and satellite>HURDLE) or (public< -HURDLE and satellite< -HURDLE)
        values.update(half_current_image=(public+current)/2,
                      satellite_after_agreement=satellite if same_side else public)
        for method in MODELS:
            yield trade(case,method,values[method],3,training_at=past[-1]["exit_at"]) | dict(
                optical_revision_kbbl=1000*(f[SUPPLEMENT]-f[PUBLIC]), supplemental_roofs=f["supplemental_roofs"],
                actual_inventory_kbbl=1000*f["actual"], measured=bool(f["baseline_roofs"]+f["spanning_roofs"]),
                base_estimate_usd=points[method][0] if method in points else None,
                similar_release_correction_usd=points[method][1] if method in points else None)


def inventory_metrics(rows):
    if not rows:
        return dict(weeks=0)
    public = np.array([abs(r["actual"]-r[PUBLIC]) for r in rows])*1000
    result = {}
    for m in (PUBLIC,OPTICAL,HYBRID,SUPPLEMENT):
        error = np.array([abs(r["actual"]-r[m]) for r in rows])*1000
        delta = error-public
        result[m] = dict(mae_kbbl=float(error.mean()),median_kbbl=float(np.median(error)),
                         delta_vs_public_95_kbbl=block_bootstrap_mean(delta),
                         improved=int(np.count_nonzero(delta<0)),worsened=int(np.count_nonzero(delta>0)))
    return dict(weeks=len(rows),forecasts=result)


def attribute_market(group):
    rows = {(r["release_date"],r["method"]):r for r in group}
    sides = []
    for q in group:
        if q["method"]!="public":
            continue
        issue = q["release_date"]
        a,b,c,d = [rows[issue,m] for m in ("public","public_fit_satellite_input","satellite_fit_public_input","satellite")]
        vals = [r["expected_long_move_usd"] for r in (a,b,c,d)]
        a0,b0,c0,d0 = vals
        sides.append(dict(release_date=issue,study_phase=q["study_phase"],
            same_week_optical_revision_kbbl=d["optical_revision_kbbl"],supplemental_roofs=d["supplemental_roofs"],
            public_move_usd=a0,current_measurement_only_usd=b0,refit_only_usd=c0,combined_move_usd=d0,
            current_measurement_contribution_usd=(b0-a0+d0-c0)/2,
            refit_contribution_usd=(c0-a0+d0-b0)/2,interaction_usd=d0-b0-c0+a0,
            public_side=a["near_side"],current_measurement_only_side=b["near_side"],refit_only_side=c["near_side"],
            combined_side=d["near_side"],extra_pnl_170_usd=d["net_170_usd"]-a["net_170_usd"]))
    return sides


def main():
    forecasts = {r["release_date"]:r for r in inventory()}
    with threadpool_limits(limits=1):
        trades = list(market(forecasts))
    report = dict(status="exploratory_on_reviewed_history", forecasts={}, market={})
    changes = attribute_market(trades)
    report["holdout_changed_satellite_decisions"] = [r for r in changes if r["study_phase"]=="holdout" and
                                                    r["combined_side"]!=r["public_side"]]
    by_method = defaultdict(list)
    for t in trades:
        by_method[t["study_phase"],t["method"]].append(t)
    for phase in PHASES:
        rows = [r for r in forecasts.values() if r["study_phase"]==phase]
        measured = [r for r in rows if r["baseline_roofs"]+r["spanning_roofs"]]
        report["forecasts"][phase] = {key:inventory_metrics(q) for key,q in (
            ("all",rows),("measured",measured),("unmeasured",[r for r in rows if not r["baseline_roofs"]+r["spanning_roofs"]]),
            ("one_or_two_roofs",[r for r in measured if r["baseline_roofs"]+r["spanning_roofs"]<=2]),
            ("three_or_more_roofs",[r for r in measured if r["baseline_roofs"]+r["spanning_roofs"]>=3]),
            ("with_supplement",[r for r in rows if r["supplemental_roofs"]]),
            ("spanning_without_baseline",[r for r in rows if r["spanning_roofs"] and not r["baseline_roofs"]]))}
        control = by_method[phase,"public"]
        report["market"][phase] = {}
        for m in MODELS:
            group = by_method[phase,m]
            if [t["release_date"] for t in group]!=[t["release_date"] for t in control]:
                raise ValueError(f"Mismatched trading dates on {phase}/{m}")
            pairs = list(zip(group,control,strict=True))
            diffs = [r["net_170_usd"]-b["net_170_usd"] for r,b in pairs]
            report["market"][phase][m] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in group),
                mean_delta_vs_public_95_usd=block_bootstrap_mean(diffs),
                changed_decisions=[dict(release_date=r["release_date"],side=r["near_side"],public_side=b["near_side"],
                    net_difference_usd=diff) for (r,b),diff in zip(pairs,diffs,strict=True) if r["near_side"]!=b["near_side"]])
        print(phase,{m:(v["net_170_usd"],v["net_340_usd"],len(v["changed_decisions"])) for m,v in report["market"][phase].items()})
    export(f"{STEM}_forecasts",list(forecasts.values()))
    export(f"{STEM}_trades",trades)
    export(f"{STEM}_decomposition",changes)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
