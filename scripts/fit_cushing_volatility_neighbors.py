"""Reweight settled observations by market state for inventory and CL forecasts."""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import HuberRegressor
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_windows, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import risk_prediction
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_late_pair_recovery import COMBINED
from fit_cushing_nowcast_ensemble import block_bootstrap_mean, transformed
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import inventory_forecasts
from publication_clocks import EASTERN

STEM = "cushing_volatility_neighbors"
REGIONAL = "hybrid_storage_and_region"
INVENTORY_MODELS = (COMBINED, REGIONAL, "market_error_median", "market_storage_error_median")
MARKET_INPUTS = (PUBLIC, SUPPLEMENT, REGIONAL, "market_error_median")
ARCHIVED = (PUBLIC, SUPPLEMENT)
REWEIGHT = ("pooled", "volatility", "volatility_and_curve")


def kernel(history, current, widths):
    standardized = (np.asarray(history)-current)/widths
    return .25+.75*np.exp(-.5*np.mean(np.square(standardized), axis=1))


def median_residual(history, row, dimensions):
    if len(history) < MIN_WINDOWS:
        return 0.
    features_now = row["inventory_coordinates"][:dimensions]
    features_past = [q["inventory_coordinates"][:dimensions] for q in history]
    similarities = kernel(features_past, features_now, np.array((.55,.4,.45))[:dimensions])
    weights = similarities*np.exp2(-np.arange(len(history)-1,-1,-1)/78)
    errors = np.asarray([q["actual"]-q[REGIONAL] for q in history])
    order = np.argsort(errors, kind="stable")
    midpoint = int(np.searchsorted(np.cumsum(weights[order]), weights.sum()/2))
    return .5*float(np.clip(errors[order[midpoint]], -.50, .50))


def archived_coordinates(archived, storage):
    spot = tuple(float(np.tanh(float(archived[f"archived_basis_{field}"])/scale))
                 for field,scale in (("level",2.),("move_5",1.)))
    return (*spot,float(storage[0]))


def inventory():
    spot = indexed("cushing_spot_vintages")
    history = []
    for r in inventory_forecasts():
        q = spot[r["release_date"]]
        local_day = r["cutoff"].astimezone(EASTERN).date()
        if (datetime.fromisoformat(q["cutoff"]) != r["cutoff"] or q["study_phase"] != r["study_phase"]
                or not q["archived_reference_date"] <= q["vintage_date"] == (local_day-timedelta(days=1)).isoformat()):
            raise ValueError(f"Unexpected archived crude-price vintage on {r['release_date']}")
        # Archived relative price level and change describe the regime at
        # forecast time without using revisions to the current inventory.
        row = r | dict(inventory_coordinates=archived_coordinates(q,r["state"]),spot_vintage_date=q["vintage_date"])
        past = [v for v in history if v["known"]<r["cutoff"]][-HISTORY:]
        estimates = [r[REGIONAL]+median_residual(past,row,n) for n in (2,3)]
        history.append(row)
        yield row | dict(zip(INVENTORY_MODELS[2:],estimates,strict=True)) | dict(
            latest_state_error_known_utc=max((v["known"] for v in past),default=None))


def settlement_weights(history, current):
    log_vol = np.array([math.log(q["volatility_usd"]/1000.) for q in history])
    carry = np.array([q["x"][2] for q in history])
    current_vol = math.log(current["volatility_usd"]/1000.)
    scale = np.array((.65,.45))
    return (None, kernel(log_vol[:,None], np.array((current_vol,)), scale[:1]),
            kernel(np.column_stack((log_vol,carry)), np.array((current_vol,current["x"][2])), scale))


def weighted_spread(sample, current, similarity):
    design, prediction = transformed(sample,current,"x")
    recency = np.exp2(-np.arange(len(sample)-1,-1,-1)/78)
    weights = recency*similarity
    weights *= recency.sum()/weights.sum()
    response = np.asarray([q["long_gross"]/q["volatility_usd"] for q in sample])
    model = HuberRegressor(epsilon=1.5, alpha=10., max_iter=400).fit(design,response,sample_weight=weights)
    return float(model.predict(prediction[None,:])[0])*current["volatility_usd"]


def volatility_forecast(history,current):
    similarity = settlement_weights(history,current)[1]
    return weighted_spread(history,current,similarity)


def market(forecasts):
    book = {r["release_date"]:r for r in forecasts}
    metadata = read(f"{OUTER}_trades")
    refs = {(q["release_date"],q["method"]):q for q in metadata
            if q["method"] in (*ARCHIVED, *(f"{m}_conditional" for m in ARCHIVED))}
    history, calibrated = [], defaultdict(list)
    for r,_ in market_windows():
        issue, cutoff = r["release_date"],r["decision"]
        f = book[issue]
        if not f["cutoff"] < f["known"] <= cutoff < r["entry"] < r["exit_at"] or f["actual"] != r["x"][0]:
            raise ValueError(f"Different inventory vintage on {issue}")
        design = {m:replace_expectation(r,f[m]) for m in MARKET_INPUTS}
        current = r | dict(design=design)
        train = [q for q in history if q["exit_at"] < cutoff][-HISTORY:]
        history.append(current)
        if len(train) < MIN_WINDOWS:
            continue
        weights = settlement_weights(train,current)
        for model in MARKET_INPUTS:
            archived = refs.get((issue,model))
            conditional = refs.get((issue,f"{model}_conditional"))
            for q in (archived,conditional) if model in ARCHIVED else ():
                if (any(q[k] != r[k] for k in ("near_month","far_month"))
                        or any(datetime.fromisoformat(q[k]) != r[field] for k,field in (
                            ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))
                        or datetime.fromisoformat(q["settled_training_label_utc"]) >= cutoff):
                    raise ValueError(f"Mismatched market anchors on {issue}/{model}")
            sample = [q | dict(x=q["design"][model]) for q in train]
            x = design[model]
            design_row = current | dict(x=x)
            predictions = [risk_prediction(sample,design_row,len(x))]
            predictions.extend(weighted_spread(sample,design_row,w) for w in weights[1:])
            if archived and not math.isclose(predictions[0],float(archived["expected_long_move_usd"]),abs_tol=1e-8):
                raise ValueError(f"The pooled market fit changed on {issue}/{model}")
            context = features(r["x"][2],r["volatility_usd"],f["actual"]-f[model])
            for style,value,w in zip(REWEIGHT,predictions,weights,strict=True):
                method = f"{model}:{style}"
                context_row = r | dict(response_features=context,base_estimate=value)
                past = [q for q in calibrated[method] if q["exit_at"] < cutoff][-HISTORY:]
                calibrated[method].append(context_row)
                adjustment, support = conditional_residual(past,context_row) if len(past)>=MIN_WINDOWS else (0.,0.)
                result = trade(r,method,value+adjustment,3,training_at=train[-1]["exit_at"],
                               calibration_at=past[-1]["exit_at"] if past else None)
                if style == "pooled" and conditional and (result["near_side"] != int(conditional["near_side"])
                        or result["gross_usd"] != int(conditional["gross_usd"])
                        or not math.isclose(value+adjustment,float(conditional["expected_long_move_usd"]),abs_tol=1e-8)):
                    raise ValueError(f"The conditional market fit changed on {issue}/{model}")
                age = np.exp2(-np.arange(len(train)-1,-1,-1)/78)
                mass = age if w is None else age*w
                yield result | dict(inventory_model=model, method=method, training_windows=len(train),
                    effective_training_windows=float(mass.sum()**2/(mass@mass)), similar_report_weight=support,
                    base_estimate_usd=value,
                    realized_long_move_usd=r["long_gross"],
                    net_340_usd=result["gross_usd"]-340*abs(result["near_side"]))


def score_inventory(rows):
    error = {m:np.asarray([1000*abs(r[m]-r["actual"]) for r in rows]) for m in INVENTORY_MODELS}
    return dict(weeks=len(rows), models={m:dict(mae_kbbl=float(v.mean()),
        delta_vs_regional_95_kbbl=block_bootstrap_mean(v-error[REGIONAL])) for m,v in error.items()}) if rows else dict(weeks=0,models={})


def trading_summary(sample, reference):
    if not sample or [q["release_date"] for q in sample] != [q["release_date"] for q in reference]:
        raise ValueError("The weighted comparison changed the market cohort")
    increments = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(sample,reference,strict=True)]
    error = np.array([abs(r["realized_long_move_usd"]-r["expected_long_move_usd"]) for r in sample])
    control = np.array([abs(r["realized_long_move_usd"]-r["expected_long_move_usd"]) for r in reference])
    conditions = (lambda v:v<1000,lambda v:1000<=v<=3000,lambda v:v>3000)
    cells = {}
    for name, test in zip(("below_1000","1000_to_3000","above_3000"),conditions,strict=True):
        indexes = [i for i,q in enumerate(sample) if test(q["volatility_usd"])]
        if indexes:
            cells[name] = dict(windows=len(indexes),net_170_usd=sum(sample[i]["net_170_usd"] for i in indexes),
                delta_vs_pooled_usd=sum(increments[i] for i in indexes),mae_usd=float(error[indexes].mean()),
                effective_training_windows=float(np.mean([sample[i]["effective_training_windows"] for i in indexes])))
    return robustness(sample) | daily_risk(sample) | dict(net_340_usd=sum(q["net_340_usd"] for q in sample),
        price_move_mae_usd=float(error.mean()),price_mae_delta_vs_pooled_95=block_bootstrap_mean(error-control),
        delta_vs_pooled_usd=sum(increments),delta_per_report_vs_pooled_95=block_bootstrap_mean(increments),
        changed_directions=sum(a["near_side"] != b["near_side"] for a,b in zip(sample,reference,strict=True)),by_volatility=cells)


def main():
    with threadpool_limits(limits=1):
        forecasts = list(inventory())
        trades = list(market(forecasts))
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["method"]].append(r)
    report = dict(status="exploratory_on_reviewed_history",inventory={},trading={})
    for phase in PHASES:
        rows = [r for r in forecasts if r["study_phase"] == phase]
        report["inventory"][phase] = score_inventory(rows)
        report["trading"][phase] = {f"{m}:{style}":trading_summary(groups[phase,f"{m}:{style}"],groups[phase,f"{m}:pooled"])
                                    for m in MARKET_INPUTS for style in REWEIGHT}
        report.setdefault("inventory_input_ablation", {})[phase] = {
            style: trading_summary(groups[phase,f"market_error_median:{style}"],groups[phase,f"{REGIONAL}:{style}"])
            for style in REWEIGHT}
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in report["inventory"][phase]["models"].items()},
              {m:(v["trades"],v["net_170_usd"],v["net_340_usd"],round(v["price_move_mae_usd"],2))
               for m,v in report["trading"][phase].items()},flush=True)
    fields = ("release_date","study_phase","week_ending","cutoff","known","actual","spot_vintage_date",
              "latest_inventory_training_utc","latest_state_error_known_utc",*INVENTORY_MODELS)
    export(f"{STEM}_forecasts",[{k:r[k] for k in fields} for r in forecasts])
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
