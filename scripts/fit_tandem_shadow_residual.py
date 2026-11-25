"""Learn per-frame optical corrections from almost-synchronous satellite pairs."""
import json
import math
from collections import defaultdict
from contextlib import closing
from statistics import median

import numpy as np

from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from calibrate_shadow_height_ml import clip_gap, field_vector
from collect_public import write

STEM = "shadow_tandem_residual_ml"
PENALTIES = (0.1, 0.5, 2.0, 8.0)
MAX_CORRECTION_M = 2.0
FEATURES = ("profile_span_m", "gain_over_flat", "contrast_to_residual", "box_gap_minus_raw_gap_m", "east_shift_m",
            "north_shift_m", "sun_elevation_deg", "view_zenith_deg", "fixed_roof_median_gap_m")


def measured_pairs():
    with closing(read_only(SOURCE)) as db:
        sites = dict(db.execute("SELECT candidate_id,documented_terminal_id FROM tank_engineering_registry"))
    estimates = {r["observation_id"]:r for r in read_csv("shadow_change_control_estimates")}
    fixed = defaultdict(list)
    for r in estimates.values():
        terminal = sites.get(r["candidate_id"])
        if r["roof_type"] == "IFR" and r["estimated_gap_m"] and terminal:
            fixed[r["version_id"]].append((terminal,float(r["estimated_gap_m"])))
    roof = {r["candidate_id"]:r for r in read_csv("sentinel2_space_time_coverage_roof_registry")}
    for r in read_csv("shadow_tandem_pass_pairs"):
        if (r["selected_for_summary"] != "True" or r["both_resolved"] != "True" or r["roof_type"] != "EFR"
                or r["candidate_id"] not in roof or abs(float(r["other_minus_a_seconds"]))>60):
            continue
        terminal = roof[r["candidate_id"]]["terminal_id"]
        endpoints = []
        for side in ("a","other"):
            frame = estimates[r[f"{side}_observation_id"]]
            if frame["candidate_id"] != r["candidate_id"] or frame["version_id"] != r[f"{side}_version"]:
                raise ValueError("Different sensor sources for tandem residual label")
            donors = [gap for site,gap in fixed[frame["version_id"]] if site!=terminal]
            _, _, span, gain, contrast, box = field_vector(frame)
            x = (span,gain,contrast,box,*(float(frame[k]) for k in ("shift_east_m","shift_north_m",
                "sun_elevation_deg","view_zenith_deg")),median(donors) if len(donors)>=3 else math.nan)
            endpoints.append(dict(x=x,gap=float(frame["estimated_gap_m"]),shell=float(frame["shell_above_ground_m"]),version=frame["version_id"]))
        if endpoints[1]["gap"]-endpoints[0]["gap"] != float(r["other_minus_a_gap_m"]):
            raise ValueError("Stored repeatability label does not reproduce")
        yield dict(capture_day=r["capture_day"],candidate_id=r["candidate_id"],terminal_id=terminal,other_sensor=r["other_sensor"],
            factor=float(roof[r["candidate_id"]]["circular_area_m2"])/BARREL_M3/1000,endpoints=endpoints)


def design(values, centre, scale):
    x = np.array(values,dtype=float)
    return np.clip((np.where(np.isfinite(x),x,centre)-centre)/scale,-3.,3.)


def fit(train, penalty):
    sample = np.array([[e["x"] for e in r["endpoints"]] for r in train])
    flattened = sample.reshape(-1,sample.shape[-1])
    centre = np.nanmean(flattened,axis=0)
    centre = np.where(np.isfinite(centre),centre,0.)
    scale = np.nanstd(flattened,axis=0)
    scale = np.where(np.isfinite(scale)&(scale>0),scale,1.)
    transformed = design(sample,centre,scale)
    x = transformed[:,1]-transformed[:,0]
    y = -np.array([r["endpoints"][1]["gap"]-r["endpoints"][0]["gap"] for r in train])
    by_day = defaultdict(int)
    for r in train:
        by_day[r["capture_day"]]+=1
    w = np.array([1/by_day[r["capture_day"]] for r in train])
    weights = np.linalg.solve(x.T@(w[:,None]*x)+penalty*w.sum()*np.eye(x.shape[1]),x.T@(w*y))
    return dict(centre=centre.tolist(),scale=scale.tolist(),weights=weights.tolist(),penalty=penalty)


def predict(row, model):
    raw = [e["gap"] for e in row["endpoints"]]
    corrected = [clip_gap(e["gap"]+np.clip(d @ np.asarray(model["weights"]),-MAX_CORRECTION_M,MAX_CORRECTION_M),e["shell"])
                 for e,d in zip(row["endpoints"],design([e["x"] for e in row["endpoints"]],model["centre"],model["scale"]),strict=True)]
    return raw, corrected


def evaluate(rows, model, scope):
    for r in rows:
        raw,adjusted = predict(r,model)
        yield {k:r[k] for k in ("capture_day","candidate_id","terminal_id","other_sensor","factor")} | dict(scope=scope,
            first_version=r["endpoints"][0]["version"],second_version=r["endpoints"][1]["version"],
            raw_first_gap_m=raw[0],raw_second_gap_m=raw[1],ml_first_gap_m=adjusted[0],ml_second_gap_m=adjusted[1],
            raw_difference_m=raw[1]-raw[0],ml_difference_m=adjusted[1]-adjusted[0])


def day_metrics(rows):
    return dict(pairs=len(rows), **{name:dict(mean_absolute_gap_difference_m=float(np.mean([abs(r[f"{name}_difference_m"]) for r in rows])),
        within_one_m=sum(abs(r[f"{name}_difference_m"])<=1 for r in rows),
        median_absolute_gap_difference_m=median(abs(r[f"{name}_difference_m"]) for r in rows)) for name in ("raw","ml")})


def volume_checks(rows, training_days, penalty):
    base = [r for r in read_csv("terminal_change_repeatability_tank_controls") if r["elapsed_days"]=="10" and r["all_four_resolved"]=="True"]
    pairs = {(r["capture_day"],r["candidate_id"],r["other_sensor"]):r for r in rows}
    groups = defaultdict(list)
    for r in base:
        groups[r["earlier_day"],r["later_day"],r["terminal_id"]].append(r)
    for (a,b,terminal),members in sorted(groups.items()):
        # Both endpoints of the evaluated interval are withheld together.
        independent = [r for r in rows if r["capture_day"] in training_days and r["capture_day"] not in (a,b)]
        model = fit(independent,penalty)
        values = defaultdict(list)
        for r in members:
            before,after = (predict(pairs[d,r["candidate_id"],r["other_sensor"]],model) for d in (a,b))
            for index,method in enumerate(("raw","ml")):
                values[method].append([(before[index][side]-after[index][side])*float(r["kbbl_per_m"]) for side in (0,1)])
        result = dict(earlier_day=a,later_day=b,terminal_id=terminal,tanks=len(members),training_days="|".join(sorted({r["capture_day"] for r in independent})))
        for method,measures in values.items():
            first,second = (math.fsum(v[i] for v in measures) for i in (0,1))
            result.update({f"{method}_first_change_kbbl":first,f"{method}_second_change_kbbl":second,
                           f"{method}_absolute_disagreement_kbbl":abs(first-second),f"{method}_opposite_sign":first*second<0})
        yield result


def main():
    rows = list(measured_pairs())
    *training, test_day = sorted({r["capture_day"] for r in rows})
    train = [r for r in rows if r["capture_day"] in training]
    scores = {}
    for penalty in PENALTIES:
        scores[penalty] = [day_metrics(list(evaluate([r for r in train if r["capture_day"]==day],
              fit([r for r in train if r["capture_day"]!=day],penalty),"withheld_day")))["ml"]["mean_absolute_gap_difference_m"] for day in training]
    chosen = min(PENALTIES,key=lambda p:float(np.mean(scores[p])))
    model = fit(train,chosen)
    test = list(evaluate([r for r in rows if r["capture_day"]==test_day],model,"later_date"))
    # The paired-volume replay uses a fixed penalty independently of the two omitted dates.
    volumes = list(volume_checks(rows,training,0.5))
    report = dict(training_dates=training,test_date=test_day,training_pairs=len(train),training_day_cross_validation=scores,
        selected_penalty=chosen,later_date=day_metrics(test),ten_day_cross_fitted_terminal_changes={name:dict(
            median_absolute_disagreement_kbbl=median(r[f"{name}_absolute_disagreement_kbbl"] for r in volumes),
            opposite_sign=sum(r[f"{name}_opposite_sign"] for r in volumes),intervals=len(volumes)) for name in ("raw","ml")},
        interpretation="The ML model minimizes disagreement between almost-synchronous satellite measurements. "
        "No raw gap is included as a predictor. Diagnostics and fixed-roof observations are evaluated separately per frame. "
        "Held-out tandem dates measure repeatability, not absolute volume. Ten-day checks exclude both endpoints from training. "
        "No EIA or trading outcomes are read.")
    export(f"{STEM}_later_date",test)
    export(f"{STEM}_terminal_changes",volumes)
    model.update(feature_names=list(FEATURES),training_dates=training,max_correction_m=MAX_CORRECTION_M)
    for suffix,data in (("summary",report),("model",model)):
        write(DERIVED/f"{STEM}_{suffix}.json",(json.dumps(data,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
