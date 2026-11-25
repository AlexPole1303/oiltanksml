"""Evaluate point-in-time regional flow and inventory forecasts on the full release history."""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from backtest_wti_cushing_spread import statistics
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_public_flows_nowcast import archived_training_rows, dataset, forecast, vintages
from fit_cushing_robust_stack import indexed
from fit_cushing_weekly_change_ml import trade_rows

STEM = "cushing_flow_extensions"
PSM = ("padd2_to_padd3_pipeline_kbbl", "padd3_to_padd2_pipeline_kbbl", "padd2_to_padd4_pipeline_kbbl",
       "padd4_to_padd2_pipeline_kbbl", "padd2_refinery_inputs_kbbl", "padd2_field_production_kbbl",
       "padd2_commercial_ending_stocks_kbbl")
STEO = ("papr_world", "patc_world", "coprpus", "conipus", "cosq_draw", "wtipuus")
METHODS = ("public_ensemble", "flow_long", "flow_short", "public_plus_flow", "public_plus_monthly", "monthly_optical")


def number(text):
    return float(text) if text else np.nan


def features(rows):
    monthly = indexed("eia_monthly_context_weekly_features")
    for r in rows:
        issue = r["release_date"]
        m = monthly.get(issue)
        if not m:
            yield r | dict(monthly=np.full(len(PSM)+len(STEO)*2,np.nan), psm_issue=None,steo_issue=None)
            continue
        for source in ("psm","steo"):
            stamp = m[f"{source}_available_at_utc"]
            if stamp and datetime.fromisoformat(stamp)>r["cutoff"]:
                raise ValueError(f"Late {source} information on {issue}")
        psm = tuple(number(m[k]) if m["psm_used_for_features"]=="True" else np.nan for k in PSM)
        current = tuple(number(m[f"steo_{k}_current_month"]) for k in STEO)
        spread = tuple(number(m[f"steo_{k}_next_month"])-v for k,v in zip(STEO,current,strict=True))
        yield r | dict(monthly=np.array((*psm,*current,*spread)),psm_issue=m["psm_issue"],steo_issue=m["steo_issue"])


def fit_residual(train, current, *, include_monthly):
    def vector(row):
        return np.r_[row["predictors"],row["monthly"]] if include_monthly else row["predictors"]

    x = np.array([vector(r) for r in train])
    y = np.array([r["target"]-r["public_ensemble"] for r in train])
    age = np.array([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in train])
    fit = HistGradientBoostingRegressor(loss="absolute_error",max_iter=45,max_leaf_nodes=5,min_samples_leaf=20,
                                       l2_regularization=4.,learning_rate=0.045,early_stopping=False,random_state=1706)
    fit.fit(x,y,sample_weight=np.exp2(-age/104.))
    return float(np.clip(fit.predict(vector(current)[None,:])[0],-1.,1.))*0.5


def experiments():
    archive = list(archived_training_rows())
    stacked = indexed("cushing_robust_stack_forecasts")
    source = indexed("cushing_holdout_release_index")
    clocks = indexed("eia_release_clock")
    weekly = indexed("cushing_weekly_change_ml_forecasts")
    additional = []
    for issue, f in sorted(stacked.items()):
        if f["study_phase"]!="holdout":
            continue
        s = source[issue]
        cutoff,known = (datetime.fromisoformat(f[k]) for k in ("forecast_cutoff_utc","target_known_by_utc"))
        control_clock = datetime.fromisoformat(clocks[s["control_vintage"]]["conservative_postrelease_from_utc"])
        if not control_clock<=cutoff<known or cutoff!=datetime.fromisoformat(s["forecast_cutoff_utc"]):
            raise ValueError(f"Incorrect flow release clock on {issue}")
        additional.append(dict(week_ending=f["week_ending"],release_date=issue,study_phase="holdout",cutoff=cutoff,known=known,
                               target=float(f["target"]),clock_prior_issue=s["control_vintage"],reported_prior=float(f["prior"])))
    values,_ = vintages(archive,through=max(stacked))
    rows = list(features(dataset(archive,values,additional_rows=additional)))
    frozen = indexed("cushing_public_flows_nowcast_forecasts")
    seen = []
    rebuilt = 0
    for r in rows:
        issue = r["release_date"]
        prior = [q for q in seen if q["known"]<r["cutoff"]]
        seen.append(r)
        if issue not in stacked:
            continue
        f = stacked[issue]
        if (r["week_ending"]!=f["week_ending"] or r["target"]!=float(f["target"])
                or r["cutoff"]!=datetime.fromisoformat(f["forecast_cutoff_utc"])):
            raise ValueError(f"Different forecast vintage on {issue}")
        public = float(f["public_ensemble"])
        r["public_ensemble"] = public
        short = [q for q in prior if q["study_phase"]!="archive"]
        long_value,short_value = forecast(prior,r),forecast(short,r)
        if issue in frozen:
            if any(not math.isclose(value,float(frozen[issue][method]),abs_tol=1e-9,rel_tol=1e-9)
                   for method,value in (("flow_long",long_value),("flow_short",short_value))):
                raise ValueError(f"Prior published flow replay differs on {issue}")
            rebuilt += 1
        train = [q for q in prior if "public_ensemble" in q][-156:]
        flow,monthly = (fit_residual(train,r,include_monthly=flag) if len(train)>=78 else 0. for flag in (False,True))
        quality = float(weekly[issue]["pair_quality_ml"])-public
        yield dict(release_date=issue,week_ending=r["week_ending"],study_phase=r["study_phase"],
                   forecast_cutoff_utc=r["cutoff"].isoformat(),target_known_by_utc=r["known"].isoformat(),
                   latest_training_label_utc=max(q["known"] for q in prior).isoformat(),previous_wpsr_issue=r["previous_issue"],
                   psm_issue=r["psm_issue"],steo_issue=r["steo_issue"],training_weeks=len(train),target=r["target"],
                   measured=f["optical_available"]=="True",public_ensemble=public,flow_long=long_value,flow_short=short_value,
                   public_plus_flow=public+flow,public_plus_monthly=public+monthly,monthly_optical=public+monthly+quality)
    if rebuilt!=len(frozen):
        raise ValueError(f"Reproduced {rebuilt}/{len(frozen)} open-period forecasts")


def score(rows):
    actual = np.asarray([r["target"] for r in rows])
    baseline = np.abs(actual-np.asarray([r["public_ensemble"] for r in rows]))
    result = {}
    for method in METHODS:
        estimate = np.asarray([r[method] for r in rows])
        error = np.abs(actual-estimate)
        result[method] = dict(mae_million_bbl=float(error.mean()),correct_direction=int(np.count_nonzero(estimate*actual>0)),
                              paired_mae_delta_ci_95=block_bootstrap_mean(error-baseline))
    return dict(releases=len(rows),models=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-forecasts", action="store_true", help="Score the existing forecast ledger")
    args = parser.parse_args()
    if args.saved_forecasts:
        rows = [r | {k:float(r[k]) for k in ("target",*METHODS)} | {"measured":r["measured"]=="True"}
                for r in read(f"{STEM}_forecasts")]
    else:
        with threadpool_limits(limits=1):
            rows = list(experiments())
        export(f"{STEM}_forecasts",rows)
    base = indexed("cushing_robust_stack_forecasts")
    if (len(rows)!=len(base) or {r["release_date"] for r in rows}!=base.keys()
            or any(r["target"]!=float(base[r["release_date"]]["target"])
                   or r["forecast_cutoff_utc"]!=base[r["release_date"]]["forecast_cutoff_utc"]
                   or not datetime.fromisoformat(r["latest_training_label_utc"])<datetime.fromisoformat(r["forecast_cutoff_utc"])
                   for r in rows)):
        raise ValueError("Incomplete or inconsistent regional flow forecast history")
    trades = list(trade_rows(rows, methods=METHODS))
    by_trade = defaultdict(list)
    for r in trades:
        by_trade[r["scope"],r["method"]].append(r)
    summary = dict(status="exploratory_replay_on_previously_examined_history",phases={phase:{
        scope:score([r for r in rows if r["study_phase"]==phase and (scope=="all" or r["measured"])])
        for scope in ("all","with_optical")} for phase in ("development","validation","holdout")},
        trading={f"{scope}:{method}":statistics(sample) for (scope,method),sample in sorted(by_trade.items())})
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(summary,indent=2,allow_nan=False)+"\n").encode())
    for phase,scopes in summary["phases"].items():
        for scope,r in scopes.items():
            print(phase,scope,r["releases"],{k:round(v["mae_million_bbl"],4) for k,v in r["models"].items()})


if __name__ == "__main__":
    main()
