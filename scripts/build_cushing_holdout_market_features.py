"""Choose holdout WTI feature sources by coverage and verify the open-period bridge."""
import json
import math
from collections import Counter
from contextlib import closing
from datetime import date, datetime

import numpy as np

from build_cushing_holdout_release_index import price_coverage, price_rows, predictor_coverage
from build_holdout_optical_inputs import indexed
from build_operator_development_inputs import group
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, write
from fit_cushing_nowcast_ensemble import MAX_RELEASE
from fit_cushing_sensor_market_nowcast import (EASTERN, MARKET_HORIZONS, contract_book, curve_features,
                                               market_features)

STEM = "cushing_holdout_market_feature"
PROTOCOL = ROOT/"data/reference/cushing_holdout_protocol.json"
ANCHORS = ("report_week_end", "pre_release")
LAGS = (0,*MARKET_HORIZONS)


def read_protocol():
    return json.loads(PROTOCOL.read_text())


def synthetic_close(day, month):
    return 65. + 6.*math.sin(date.fromisoformat(day).toordinal()/32.) - .11*int(month[-2:])


def vector_from_marks(rows, source, value):
    values, dates = [], []
    by_anchor = group(rows,"boundary")
    if set(by_anchor)!={(name,) for name in ANCHORS}:
        raise ValueError("Incomplete predictor anchors")
    for name in ANCHORS:
        by_lag = group(by_anchor[name,],"lag_sessions")
        if set(by_lag)!={(lag,) for lag in LAGS}:
            raise ValueError(f"Incomplete predictor lags at {name}")
        curves = []
        for lag in LAGS:
            curve = {}
            marks = sorted(by_lag[lag,],key=lambda r:r["rank_on_reference_session"])
            ranks = (1,2,3,4) if not lag else (1,2,4)
            if tuple(r["rank_on_reference_session"] for r in marks)!=ranks:
                raise ValueError(f"Invalid contract ranks at {name}/{lag}")
            for r in marks:
                day, month = r["observed_session"],r["delivery_month"]
                boundary = datetime.fromisoformat(r["cutoff_utc"]).astimezone(EASTERN).date().isoformat()
                if not day<=r["reference_session"]<boundary or (lag==0 and day!=r["reference_session"]):
                    raise ValueError(f"Quote after the reference clock at {r['release_date']}/{name}/{lag}")
                if r["within_first_four_on_observed_session"] and r[f"{source}_reference"]:
                    curve[month] = value(source,day,month,r[f"{source}_reference"])
            curves.append(curve)
        if len(curves[0])!=4:
            raise ValueError(f"Source {source} lacks a complete reference curve at {name}")
        values.extend(curve_features(curves[0],curves[1:]))
        dates.append(by_lag[0,][0]["reference_session"])
    return np.array(values), dates


def numeric_prices(db, sources, *, through):
    prices = {source:{(d,m):(ref,float(p)) for d,m,ref,p in price_rows(db,source,numeric_through=through)} for source in sources}

    def get(source,day,month,reference):
        ref, price = prices[source][day,month]
        if ref!=reference or not math.isfinite(price):
            raise ValueError(f"Wrong numeric price provenance for {source}/{day}/{month}")
        return price

    return get


def prepared_marks():
    return group((dict(r, lag_sessions=int(r["lag_sessions"]),rank_on_reference_session=int(r["rank_on_reference_session"]),
                       within_first_four_on_observed_session=r["within_first_four_on_observed_session"]=="True")
                  for r in read_csv("cushing_holdout_release_predictor_marks")),"release_date")


def make_plan(protocol):
    marks = prepared_marks()
    issues = sorted(indexed("cushing_holdout_release_index").values(),key=lambda r:r["release_date"])
    if len(issues)!=protocol["holdout_releases"] or tuple(protocol["forecast_lags_sessions"])!=MARKET_HORIZONS:
        raise ValueError("Prepared forecast inputs differ from the fixed release count or lags")
    for r in issues:
        issue = r["release_date"]
        source = next((s for s in protocol["forecast_quote_source_priority"] if all(
            r[f"{anchor}_{s}_four_leg_snapshot"]=="True" for anchor in ANCHORS)),None)
        if source is None:
            raise ValueError(f"No single source can supply both predictor anchors on {issue}")
        spec = marks[issue,]
        vector,days = vector_from_marks(spec,source,lambda s,d,m,ref:synthetic_close(d,m))
        count = sum(int(r[f"{anchor}_{source}_features"]) for anchor in ANCHORS)
        if count!=np.isfinite(vector).sum() or days!=[r[f"{anchor}_reference_session"] for anchor in ANCHORS]:
            raise ValueError(f"Selected source coverage does not reconcile on {issue}")
        yield r,source,spec,dict(study_phase=r["study_phase"],week_ending=r["week_ending"],release_date=issue,
            forecast_cutoff_utc=r["forecast_cutoff_utc"],predictor_source=source,forecast_features_present=count,
            report_week_end_reference_session=days[0],pre_release_reference_session=days[1],execution_source=r["wti_replay_source"])


def open_period_bridge(db):
    refs,curves,days = price_coverage(db)
    days = [d for d in days if d<=MAX_RELEASE]
    official_days,official_book = contract_book()
    if days!=official_days:
        raise ValueError("Prepared calendar fails open-period parity")
    get = numeric_prices(db,refs,through=MAX_RELEASE)
    base = {(r["week_ending"],r["release_date"]):r for r in read_csv("sentinel2_space_time_coverage_weeks")
            if r["study_phase"] in ("development","validation")}
    comparisons = {s:Counter(releases=0, shared_features=0, different_shared_features=0, different_missing_masks=0)
                   for s in refs}
    for r in read_csv("cushing_sensor_market_nowcast_forecasts"):
        issue = r["release_date"]
        g = base[r["week_ending"],issue]
        boundaries = tuple(datetime.fromisoformat(s) for s in (g["current_measurement_at_utc"],r["forecast_cutoff_utc"]))
        prior_vector, prior_dates = market_features(official_days,official_book,boundaries)
        marks, stats = zip(*(predictor_coverage(issue,n,b,refs,curves,days) for n,b in zip(ANCHORS,boundaries,strict=True)))
        joined = [row for anchor in marks for row in anchor]
        for source,total in comparisons.items():
            if not all(item[f"{source}_four_leg_snapshot"] for item in stats):
                continue
            vector,dates = vector_from_marks(joined,source,get)
            if dates!=prior_dates:
                raise ValueError(f"Different open-period WTI clock on {issue}/{source}")
            shared = np.isfinite(vector)&np.isfinite(prior_vector)
            total.update(releases=1,shared_features=int(shared.sum()),
                different_shared_features=int(np.count_nonzero(vector[shared]!=prior_vector[shared])),
                different_missing_masks=int(not np.array_equal(np.isfinite(vector),np.isfinite(prior_vector))))
    if any(comparisons["official"][k] for k in ("different_shared_features","different_missing_masks")):
        raise ValueError("The shared feature builder does not reproduce the original EIA inputs")
    return comparisons


def main():
    protocol = read_protocol()
    plan = [record for _,_,_,record in make_plan(protocol)]
    with closing(read_only(SOURCE)) as db:
        bridge = open_period_bridge(db)
    report = dict(releases=len(plan),predictor_sources=dict(Counter(r["predictor_source"] for r in plan)),
        execution_sources=dict(Counter(r["execution_source"] for r in plan)),
        features_present=dict(sorted(Counter(r["forecast_features_present"] for r in plan).items())),
        open_period_bridge=bridge, interpretation="Holdout quotes and EIA outcomes were not read. "
        "Synthetic prices only exercised the selected coverage mask. Open-period comparisons used observed prices.")
    export(f"{STEM}_plan",plan)
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
