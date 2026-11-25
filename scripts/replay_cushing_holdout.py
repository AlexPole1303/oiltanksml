"""Exercise or evaluate the frozen holdout forecast and calendar-spread replay."""
import argparse
import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime

import numpy as np
from threadpoolctl import threadpool_limits

import fit_cushing_adaptive_nowcast as adaptive
import fit_cushing_multiview_nowcast as multiview
import fit_cushing_nowcast_ensemble as ensemble
import fit_cushing_sensor_market_nowcast as sensor
from backtest_wti_cushing_spread import COSTS, settlement_trade, statistics, tick_price
from build_cushing_holdout_market_features import (make_plan, numeric_prices, read_protocol, synthetic_close,
                                                   vector_from_marks)
from build_cushing_holdout_release_index import validate_trade
from build_holdout_optical_inputs import PHASE, indexed
from build_operator_development_inputs import group
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

STEM = "cushing_holdout_replay"
VIEW_NAMES = ("registered", "spectral", "refined_alignment")


def closed_inputs(plan, views, *, evaluate):
    through = plan[-1][0]["release_date"]
    clocks = {r["release_date"]:r for r in read_csv("eia_release_clock") if r["release_date"]<=through}
    source_priority = {source for _,source,_,_ in plan}
    with closing(read_only(SOURCE)) as db:
        vintage = ensemble.public_vintages(db,through if evaluate else ensemble.MAX_RELEASE)
        if evaluate:
            targets = {(d,w):(y,p) for d,w,y,p in db.execute("""SELECT release_date,week_ending,
                change_kbbl_as_known_at_release,latest_weekly_change_known_before_release
                FROM eia_inventory_release_panel WHERE study_phase=? AND release_date<=?""",(PHASE,through))}
            positions = {d:(interest,long-short) for d,interest,long,short in db.execute("""SELECT report_date,
                open_interest,money_manager_long,money_manager_short FROM cftc_wti WHERE report_date<?""",(through,))
                if interest and long is not None and short is not None}
            values = numeric_prices(db,source_priority,through=max(row["observed_session"] for _,_,marks,_ in plan for row in marks))
        else:
            values = lambda source,day,month,ref:synthetic_close(day,month)
    if not evaluate:
        templates = [issue for issue,c in sorted(clocks.items()) if issue<=ensemble.MAX_RELEASE and all(
            (issue,c[f"{edge}_week_in_issue"]) in vintage for edge in ("oldest","latest"))][-25:]
        for i,(issue,c) in enumerate((item for item in sorted(clocks.items()) if item[0]>ensemble.MAX_RELEASE)):
            sample = templates[i%len(templates)]
            for edge in ("oldest","latest"):
                vintage[issue,c[f"{edge}_week_in_issue"]] = vintage[sample,clocks[sample][f"{edge}_week_in_issue"]].copy()
    geometry, context, vectors = {}, {}, {}
    rows = []
    for i,(r,source,marks,metadata) in enumerate(plan):
        issue,week = r["release_date"],r["week_ending"]
        cutoff,known = r["forecast_cutoff_utc"],r["target_available_at_utc"]
        if evaluate:
            target,prior = (float(n)/1000 for n in targets[issue,week])
            interest,net = positions[r["cftc_report_date"]]
        else:
            target,prior = (1.8*math.sin(j/4.1) + .6*math.cos(j/2.9) for j in (i,i-1))
            interest,net = 100_000,20_000*math.sin(i/11.)
        vector,anchors = vector_from_marks(marks,source,values)
        if int(np.isfinite(vector).sum())!=metadata["forecast_features_present"]:
            raise ValueError(f"Wrong numeric feature coverage on {issue}")
        v = views[week]
        if v["release_date"]!=issue or datetime.fromisoformat(v["cutoff_utc"])!=datetime.fromisoformat(cutoff):
            raise ValueError(f"Wrong optical feature chronology on {issue}")
        measured = v["measured_b04"]=="True"
        raw = float(v["b04_capture_change_kbbl"])/1000 if measured else None
        angle = 2*math.pi*date.fromisoformat(week).timetuple().tm_yday/365.25
        geometry[week,issue] = dict(raw_kbbl=v["b04_capture_change_kbbl"],overlap_kbbl=v["b04_overlap_change_kbbl"],
                                   baseline_b04_space_time_fraction=v["b04_space_time_fraction"])
        tail = len(vector)//2
        context[issue] = (cutoff,r["control_vintage"],r["control_observation_week"],anchors[1],round(float(vector[tail+1]),4),
                          round(float(vector[tail+2]),4),r["cftc_available_at_utc"],interest,net)
        vectors[issue] = vector,anchors
        rows.append(dict(study_phase=PHASE,week_ending=week,release_date=issue,forecast_cutoff_utc=cutoff,
            target_known_by_utc=known,target_million_bbl=target,prior_million_bbl=prior,
            raw_optical_million_bbl=raw,optical_available=measured,label=target,
            eia_features=(prior,math.sin(angle),math.cos(angle),context[issue][4])))
    return list(ensemble.enrich_feature_rows(rows,geometry,clocks,vintage,context)),vectors


def reproduce_open(predicted, stem, fields):
    observed = {r["release_date"]:r for r in read_csv(f"{stem}_forecasts")}
    historical = {r["release_date"]:r for r in predicted if r["study_phase"]!=PHASE}
    if historical.keys()!=observed.keys():
        raise ValueError(f"Changed open-period sample for {stem}")
    for issue,r in historical.items():
        for field in fields:
            if r[field] is None and observed[issue][field]=="":
                continue
            if r[field] is None or observed[issue][field]=="":
                raise ValueError(f"Changed missing open-period forecast on {issue}/{stem}/{field}")
            if not math.isclose(float(r[field]),float(observed[issue][field]),rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError(f"Changed open-period forecast on {issue}/{stem}/{field}")
    return len(historical)


def forecast_replay(closed,vectors,views,plan,*,check_ml_extensions=False):
    study = list(ensemble.feature_rows())+closed
    history = [dict(r,label=r["target_million_bbl"],cutoff=datetime.fromisoformat(r["forecast_cutoff_utc"]),
        known_at=datetime.fromisoformat(r["target_known_by_utc"])) for r in adaptive.archived_training_rows()]
    ensemble_rows = list(ensemble.replay([dict(r) for r in study]))
    checked = {ensemble.STEM:reproduce_open(ensemble_rows,ensemble.STEM,("public_ensemble","optical_ensemble"))}
    print("Reproduced open public and optical ensembles.",flush=True)
    predecessors = {r["release_date"]:r for r in ensemble_rows}
    adaptive_rows = list(adaptive.replay(history,[dict(r,previous_level_million_bbl=ensemble.number(r["core"][4])/1000)
                                               for r in study],predecessors))
    checked[adaptive.STEM] = reproduce_open(adaptive_rows,f"{adaptive.STEM}_extended",("adaptive_public","adaptive_optical"))
    print("Reproduced open adaptive models.",flush=True)
    original_views = multiview.optical_views()
    stack_rows = []
    for r in adaptive_rows:
        key = r["release_date"],r["week_ending"]
        if r["study_phase"]==PHASE:
            v = views[key[1]]
            alternatives = {name:float(v[f"{name}_variant_kbbl"])/1000 for name in VIEW_NAMES if v[f"{name}_variant_kbbl"]}
        else:
            alternatives = original_views.get(key,{})
        stack_rows.append(multiview.feature_row(r,alternatives))
    stacked = list(multiview.replay(stack_rows))
    checked[multiview.STEM] = reproduce_open(stacked,multiview.STEM,("stack_raw","stack_multiview"))
    print("Reproduced open multiview stack.",flush=True)
    geometry, terminals = sensor.optical_context()
    official_days,official_book = sensor.contract_book()
    geometry.update({(r["week_ending"],r["release_date"]):r for r in indexed("holdout_optical_weeks").values()})
    for r in read_csv("holdout_optical_terminal_weeks"):
        if int(r["baseline_b04_tanks"]):
            terminals[r["week_ending"],r["release_date"]][r["terminal_id"]] = float(r["capture_change_kbbl"])/1000
    sensor_rows = []
    for r in stacked:
        issue,key = r["release_date"],(r["week_ending"],r["release_date"])
        g = geometry[key]
        if r["study_phase"]==PHASE:
            market,marks = vectors[issue]
        else:
            times = tuple(datetime.fromisoformat(g[name]) for name in ("current_measurement_at_utc","cutoff_utc"))
            market,marks = sensor.market_features(official_days,official_book,times)
        sensor_rows.append(sensor.input_row(r,g,terminals[key],market,marks))
    calibrated = list(sensor.replay(sensor_rows))
    checked[sensor.STEM] = reproduce_open(calibrated,sensor.STEM,(*sensor.MODELS,"development_selected"))
    print("Reproduced open sensor and WTI corrections.",flush=True)
    model_fields = ("prior","public_ensemble","optical_ensemble")
    adapt = {r["release_date"]:r for r in adaptive_rows}
    extension_check = None
    if check_ml_extensions:
        import fit_cushing_ml_extensions as extension

        corrections = extension.registration_changes()
        extra = list(extension.replay(extension.feature_row(state,prediction,predecessors[state["release_date"]],
                    adapt[state["release_date"]],corrections) for state,prediction in zip(sensor_rows,calibrated,strict=True)))
        fields = (*extension.CANDIDATES,"selected","prior_plus_registered_raw","flat_half_width_million_bbl",
                  "conditional_half_width_million_bbl")
        checked[extension.STEM] = reproduce_open(extra,extension.STEM,fields)
        closed_extra = [r for r in extra if r["study_phase"]==PHASE]
        if (len(closed_extra)!=len(plan) or any(not datetime.fromisoformat(r["latest_training_label_utc"])<
                datetime.fromisoformat(r["forecast_cutoff_utc"]) for r in closed_extra)):
            raise ValueError("Unexpected extended-model calendar or training timestamps")
        extension_check = dict(synthetic_releases=len(closed_extra),
            with_nonzero_registration_adjustment=sum(bool(r["registration_delta"]) for r in closed_extra),
            with_forecast_interval=sum(r["conditional_half_width_million_bbl"] is not None for r in closed_extra))
        print("Reproduced open ML challengers and exercised their holdout feature path.",flush=True)
    market_sources = {r["release_date"]:metadata["predictor_source"] for r,_,_,metadata in plan}
    output = []
    for r in calibrated:
        if r["study_phase"]!=PHASE:
            continue
        issue = r["release_date"]
        output.append(r | {k:predecessors[issue][k] for k in model_fields} | {k:adapt[issue][k] for k in
            ("adaptive_public","adaptive_optical")} | dict(stack_multiview=r["incumbent"],predictor_source=market_sources[issue]))
    return output,checked,extension_check


def trades(plan, forecasts, protocol, *, evaluate):
    bars = group((r for r in read_csv("wti_cushing_event_bars") if r["study_phase"]==PHASE),"release_date")
    windows = indexed("wti_cushing_event_windows")
    predicted = {r["release_date"]:r for r in forecasts}
    if tuple(protocol["execution_cost_scenarios_usd"])!=COSTS or protocol["execution_quote_source_priority"]!=["archive","ibkr"]:
        raise ValueError("Changed execution rule or cost scenarios")
    if evaluate:
        with closing(read_only(SOURCE)) as db:
            get = numeric_prices(db,protocol["execution_quote_source_priority"],through=max(r["exit_session"] for r,_,_,_ in plan))
    else:
        get = lambda source,day,month,ref:round(synthetic_close(day,month)+.33*math.sin(
            date.fromisoformat(day).toordinal()/9.+int(month[-2:])*.7),2)
    methods = (protocol["execution_primary_candidate"],protocol["execution_primary_control"],*protocol["execution_secondary_candidates"])
    for r,_,_,_ in plan:
        issue,source = r["release_date"],r["wti_replay_source"]
        f,w,points = predicted[issue],windows[r["week_ending"]],bars[issue,]
        validate_trade(issue,w,source,points)
        if datetime.fromisoformat(f["target_known_by_utc"])>datetime.fromisoformat(w["decision_at_utc"]):
            raise ValueError(f"Target was not published at the decision on {issue}")
        values = {(int(p["session_number"]),p["leg"]):tick_price(get(source,p["session_date"],p["delivery_month"],
                  p[f"{source}_reference"])) for p in points}
        spread = [10*(values[i,"near"]-values[i,"far"]) for i in range(1,6)]
        changes = [p-spread[0] for p in spread]
        for cohort,column in (("all","nonoverlapping_window_all"),("with_b04","nonoverlapping_window_b04")):
            if r[column]!="True":
                continue
            for method in methods:
                expected = None if method=="always_long_spread" else 0. if method=="published_change" else f[method]
                yield settlement_trade(w,cohort,source,method,f["target"],expected,changes)


def outcomes(forecasts, entries, plan, protocol):
    measured = {r["release_date"] for r,_,_,_ in plan if r["optical_views_available"]=="3"}
    candidates = (protocol["forecast_primary_candidate"],protocol["forecast_primary_control"],*protocol["forecast_secondary_candidates"])
    predictions = {}
    for scope, rows in (("all",forecasts),("with_b04",[r for r in forecasts if r["measured"]]),
                        ("with_b04_and_all_three_views",[r for r in forecasts if r["release_date"] in measured])):
        actual = np.array([r["target"] for r in rows])
        errors = {method:np.array([r[method] for r in rows])-actual for method in candidates}
        delta = abs(errors[candidates[0]])-abs(errors[candidates[1]])
        predictions[scope] = dict(releases=len(rows),primary_mae_delta_million_bbl=float(delta.mean()),
            primary_mae_delta_95pct_interval=ensemble.block_bootstrap_mean(delta),metrics={method:dict(
                mae_million_bbl=float(abs(err).mean()),rmse_million_bbl=float(np.sqrt((err**2).mean())),
                correct_nonzero_direction=int(np.sum((actual+err)*actual>0))) for method,err in errors.items()})
    trading = {}
    for (scope,), cohort in group(entries,"scope").items():
        by_method = group(cohort,"method")
        base,candidate = (by_method[protocol[f"execution_primary_{name}"],] for name in ("control","candidate"))
        if [r["release_date"] for r in base]!=[r["release_date"] for r in candidate]:
            raise ValueError(f"Unpaired trading result for {scope}")
        delta = np.array([b["net_50_usd"]-a["net_50_usd"] for a,b in zip(base,candidate,strict=True)])
        trading[scope] = dict(primary_net_delta_usd_per_window=float(delta.mean()),
            primary_net_delta_95pct_interval=ensemble.block_bootstrap_mean(delta),
            metrics={method:statistics(rows) for (method,),rows in by_method.items()})
    return dict(forecast=predictions,trading=trading)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--evaluate",action="store_true",help="Open the holdout and publish scores and trades")
    mode.add_argument("--check-ml-extensions",action="store_true",help="Exercise the exploratory ML models with synthetic holdout inputs")
    args = parser.parse_args()
    protocol = read_protocol()
    plan = list(make_plan(protocol))
    views = indexed("holdout_optical_expanded_view_features")
    closed,vectors = closed_inputs(plan,views,evaluate=args.evaluate)
    with threadpool_limits(limits=1):
        forecasts,checked,extension_check = forecast_replay(closed,vectors,views,plan,check_ml_extensions=args.check_ml_extensions)
    if len(forecasts)!=len(plan) or any(not datetime.fromisoformat(r["latest_training_label_utc"])<
            datetime.fromisoformat(r["forecast_cutoff_utc"]) for r in forecasts):
        raise ValueError("Holdout forecast coverage or training clock failed")
    entries = list(trades(plan,forecasts,protocol,evaluate=args.evaluate))
    report = dict(evaluation=args.evaluate,releases=len(forecasts),open_forecasts_reproduced=checked,
        predictor_sources=dict(Counter(r["predictor_source"] for r in forecasts)),
        paired_trading_windows={scope:len({r["release_date"] for r in entries if r["scope"]==scope}) for scope in protocol["execution_cohorts"]})
    results = outcomes(forecasts,entries,plan,protocol)
    if args.evaluate:
        report.update(results)
        export(f"{STEM}_forecasts",forecasts)
        export(f"{STEM}_trades",entries)
    else:
        report["interpretation"] = "Only synthetic holdout EIA labels, public data and WTI prices were used. Synthetic scores are discarded."
    if extension_check is not None:
        report["ml_extension_checks"] = extension_check
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_{'summary' if args.evaluate else 'dry_run'}.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
