"""Replay inventory-flow forecasts using archived WPSR observations since 2012."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date, datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from build_eia_market_inputs import METRICS, standardized
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import RAW, ROOT, WPSR, write
from fit_cushing_adaptive_nowcast import archived_training_rows
from fit_cushing_multiview_nowcast import STEM as INCUMBENT
from fit_cushing_nowcast_ensemble import MAX_RELEASE, PHASES, block_bootstrap_mean, feature_rows

STEM = "cushing_public_flows_nowcast"
TABLES = (2,4,9)
STOCKS = ("cushing_crude", "padd2_commercial_crude", "padd3_commercial_crude", "us_commercial_crude")
FLOWS = ("padd2_crude_refiner_inputs", "padd3_crude_refiner_inputs", "us_crude_refiner_inputs",
         "us_crude_production", "us_crude_net_imports_including_spr", "padd2_refinery_utilization",
         "padd3_refinery_utilization", "us_crude_exports")
METHODS = ("incumbent", "prior_plus_raw", "flow_short", "flow_long", "incumbent_with_short_flows", "incumbent_with_long_flows")


def early_table(path,table,release,expected_dates):
    body = path.read_bytes()
    try:
        text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = body.decode("cp1252")
    header,*rows = list(csv.reader(text.splitlines()))
    stubs = sum(k.startswith("STUB_") for k in header)
    dates = [datetime.strptime(d,"%m/%d/%y").date().isoformat() for d in header[stubs:stubs+2]]
    url = path.with_suffix(".source").read_text().strip()
    if dates!=expected_dates or not url.startswith(f"{WPSR}{release[:4]}/{release.replace('-','_')}/"):
        raise ValueError(f"Source vintage or date does not match Table 4 for {path.name}")
    records = {}
    for metric in METRICS:
        if metric.table!=table or metric.key not in (*STOCKS,*FLOWS):
            continue
        for num,row in enumerate(rows,start=2):
            if len(row)<stubs+2 or (metric.row is not None and metric.row!=num):
                continue
            label = row[stubs-1].strip()
            if label!=metric.label or (metric.group and row[0].strip()!=metric.group):
                continue
            records[metric.key] = tuple(standardized(metric,release,d,raw.strip(),table,"archived_csv",num,
                stubs+i+1,label,url)[4] for i,(d,raw) in enumerate(zip(dates,row[stubs:stubs+2],strict=True)))
    receipt = dict(release_date=release,table=table,source_url=url,source_file=str(path.relative_to(ROOT)),
                   bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),model_metrics=len(records))
    return records,receipt


def vintages(history, *, through=MAX_RELEASE):
    values = defaultdict(lambda:defaultdict(dict))
    receipts = []
    early = RAW/"eia"/"wpsr_training_history"
    for r in history:
        issue = r["release_date"]
        if issue>="2016-01-01":
            continue
        dates = [r["week_ending"],r["previous_week_ending"]]
        for table in TABLES:
            path = early/f"{issue.replace('-','_')}_table{table}.csv"
            if not path.is_file():
                raise ValueError(f"Missing historical WPSR source {path.name}")
            records,receipt = early_table(path,table,issue,dates)
            for name,pair in records.items():
                for week,value in zip(dates,pair,strict=True):
                    values[issue][week][name] = value
            receipts.append(receipt)
    with closing(read_only(SOURCE)) as db:
        for issue,week,metric,value in db.execute("""SELECT release_date,report_date,metric,value FROM eia_control_vintage
            WHERE release_date BETWEEN '2016-01-01' AND ?""",(through,)):
            if metric in (*STOCKS,*FLOWS):
                values[issue][week][metric] = value
    return values,receipts


def num(value):
    return float(value) if value is not None else np.nan


def state(previous):
    days = sorted(previous,reverse=True)
    if len(days)<2:
        return np.full(2*(len(STOCKS)+len(FLOWS)),np.nan)
    current,prior = (previous[d] for d in days[:2])
    series = np.array([[num(row.get(k)) for k in (*STOCKS,*FLOWS)] for row in (current,prior)])
    # Separate the rest of the Midwest from Cushing and the Gulf Coast from the national total.
    series[:,1] -= series[:,0]
    series[:,3] -= series[:,1]+series[:,0]+series[:,2]
    return np.r_[series[0],series[0]-series[1]]


def dataset(raw,values, *, additional_rows=()):
    history = [dict(week_ending=r["week_ending"],release_date=r["release_date"],study_phase="archive",
        cutoff=datetime.fromisoformat(r["forecast_cutoff_utc"]),known=datetime.fromisoformat(r["target_known_by_utc"]),
        target=r["target_million_bbl"],clock_prior_issue=None,reported_prior=None) for r in raw]
    study = [dict(week_ending=r["week_ending"],release_date=r["release_date"],study_phase=r["study_phase"],
        cutoff=r["cutoff"],known=r["known_at"],target=r["label"],clock_prior_issue=r["prev_wpsr_issue"],
        reported_prior=r["prior_million_bbl"]) for r in feature_rows()]
    study.extend(additional_rows)
    known, pending = [], []
    for row in history+study:
        known.extend(r for r in pending if r["known"]<row["cutoff"])
        pending = [r for r in pending if r["known"]>=row["cutoff"]]
        if known:
            previous = row["clock_prior_issue"] or known[-1]["release_date"]
            if previous>row["cutoff"].date().isoformat() or previous>=row["release_date"]:
                raise ValueError(f"Later input vintage on {row['release_date']}")
            change = row["reported_prior"] if row["reported_prior"] is not None else known[-1]["target"]
            recent = [r["target"] for r in known[-4:]]
            angle = 2*math.pi*date.fromisoformat(row["week_ending"]).timetuple().tm_yday/365.25
            signal = state(values[previous])
            if np.isfinite(signal[0]):
                yield dict(row,previous_issue=previous,prior=change,
                           predictors=np.r_[change,np.mean(recent),math.sin(angle),math.cos(angle),signal])
        pending.append(row)


def forecast(known,row):
    if len(known)<48:
        return row["prior"]
    x = np.array([q["predictors"] for q in known])
    y = np.array([q["target"]-q["prior"] for q in known])
    age = np.array([(row["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in known])
    model = HistGradientBoostingRegressor(loss="absolute_error",max_iter=48,max_leaf_nodes=5,min_samples_leaf=22,
        l2_regularization=3.,learning_rate=.055,early_stopping=False,random_state=7801)
    model.fit(x,y,sample_weight=np.exp2(-age/140.))
    correction = float(model.predict(row["predictors"][None,:])[0])
    return row["prior"]+.75*float(np.clip(correction,-2.5,2.5))


def replay(features, incumbents):
    past = []
    for r in features:
        previous = [q for q in past if q["known"]<r["cutoff"]]
        past.append(r)
        f = incumbents.get(r["release_date"])
        if not f:
            continue
        if (r["week_ending"]!=f["week_ending"] or r["target"]!=float(f["target"])
            or r["cutoff"]!=datetime.fromisoformat(f["forecast_cutoff_utc"])):
            raise ValueError(f"Flow features and incumbent cohorts disagree on {r['release_date']}")
        short = [q for q in previous if q["study_phase"]!="archive"]
        long_value,short_value = (forecast(sample,r) for sample in (previous,short))
        public,base = map(float,(f["adaptive_public"],f["stack_multiview"]))
        yield {k:r[k] for k in ("study_phase","week_ending","release_date","target","previous_issue")} | dict(
            forecast_cutoff_utc=r["cutoff"].isoformat(),latest_training_label_utc=max(q["known"] for q in previous).isoformat(),
            measured=f["measured"]=="True",long_training_rows=len(previous),short_training_rows=len(short),
            flow_features_available=int(np.isfinite(r["predictors"]).sum()),incumbent=base,prior_plus_raw=float(f["prior_plus_raw"]),
            flow_long=long_value,flow_short=short_value,
            incumbent_with_long_flows=base+.35*(long_value-public),incumbent_with_short_flows=base+.35*(short_value-public))


def assess(rows):
    actual = np.array([r["target"] for r in rows])
    before = np.abs(np.array([r["incumbent"] for r in rows])-actual)
    results = {}
    for method in METHODS:
        forecast_values = np.array([r[method] for r in rows])
        abs_error = np.abs(forecast_values-actual)
        results[method] = dict(mae_million_bbl=float(abs_error.mean()),direction=int(np.sum(actual*forecast_values>0)),
                             paired_mae_difference_interval=block_bootstrap_mean(abs_error-before))
    return dict(weeks=len(rows),metrics=results)


def main():
    raw = list(archived_training_rows())
    values, receipts = vintages(raw)
    features = list(dataset(raw,values))
    incumbents = {r["release_date"]:r for r in read_csv(f"{INCUMBENT}_forecasts")}
    with threadpool_limits(limits=1):
        predictions = list(replay(features,incumbents))
    if len(predictions)!=len(incumbents):
        raise ValueError("The flow forecast comparison has incomplete shared coverage")
    export(f"{STEM}_archived_sources",receipts)
    export(f"{STEM}_forecasts",predictions)
    metrics = {phase:{scope:assess([r for r in predictions if r["study_phase"]==phase and (scope=="all" or r["measured"])])
               for scope in ("all","with_optical")} for phase in PHASES}
    result = dict(archive_tables=dict(Counter(r["table"] for r in receipts)),feature_rows=len(features),
        archive_feature_rows=sum(r["study_phase"]=="archive" for r in features),phases=metrics,
        interpretation="Only previously released vintages enter the flow features. Missing historical series stay missing. "
        "The same booster is replayed with and without the earlier archive. 2023 is an inspected historical period.")
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())
    for phase,scopes in metrics.items():
        for scope,m in scopes.items():
            print(phase,scope,m["weeks"],{k:round(v["mae_million_bbl"],4) for k,v in m["metrics"].items()},flush=True)


if __name__ == "__main__":
    main()
