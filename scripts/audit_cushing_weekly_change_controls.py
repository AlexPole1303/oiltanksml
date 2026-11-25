"""Check pair-level shrinkage against near-dated aerial and tandem measurements."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import least_squares

from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import write
from fit_cushing_weekly_change_ml import attenuate

STEM = "cushing_weekly_change_controls"
METHODS = ("physical", "attenuated", "learned")


def aerial():
    fields = ("candidate_id", "flight_date", "satellite_version", "satellite_captured_at_utc", "satellite_sensor")
    references = {tuple(r[k] for k in fields): r for r in read("aerialok_native_optical_pairs")}
    groups = defaultdict(list)
    for r in read("aerialok_native_optical_changes"):
        sides = ("first", "last")
        if r["both_aerial_stable"] != "True" or not all(r[f"{side}_satellite_sensor"].startswith("Sentinel-2 B04") for side in sides):
            continue
        first, last = (references[r["candidate_id"],r[f"{side}_flight"],r[f"{side}_satellite_version"],
                                  r[f"{side}_satellite_capture_utc"],r[f"{side}_satellite_sensor"]] for side in sides)
        change, observed = (float(r[k]) for k in ("satellite_rise_m", "aerial_rise_m"))
        if change != float(first["satellite_gap_m"])-float(last["satellite_gap_m"]):
            raise ValueError(f"Source pair does not reproduce {r['candidate_id']}")
        low = float(first["satellite_gap_min_m"])-float(last["satellite_gap_max_m"])
        high = float(first["satellite_gap_max_m"])-float(last["satellite_gap_min_m"])
        groups[r["candidate_id"],r["first_flight"],r["last_flight"]].append(dict(r,physical=change,target=observed,
                      low=low,high=high,attenuated=attenuate(change,low,high),
                      max_gap_hours=max(float(r[f"{side}_max_hours"]) for side in sides)))
    return [min(rows,key=lambda r:(r["max_gap_hours"],r["first_satellite_version"],r["last_satellite_version"]))
            for _,rows in sorted(groups.items())]


def train(rows):
    x = np.asarray([[r["physical"],r["attenuated"]] for r in rows])
    y = np.asarray([r["target"] for r in rows])
    size = len(rows)
    centre = np.array([0.5,0.25])
    fit = least_squares(lambda beta:np.r_[(x@beta-y)/math.sqrt(size),math.sqrt(0.2)*(beta-centre)],
                        centre,bounds=(0.,1.),loss="soft_l1",f_scale=0.5)
    if not fit.success:
        raise ValueError(fit.message)
    return np.where(fit.active_mask == -1, 0., fit.x)


def score(rows):
    return {m:dict(mae_m=float(np.mean([abs(r[m]-r["target"]) for r in rows])),
                   mae_cylindrical_bbl=float(np.mean([abs(r[m]-r["target"])*r["bbl_per_m"] for r in rows])),
                   same_nonzero_direction=sum(r[m]*r["target"]>0 for r in rows),
                   opposite_sign=sum(r[m]*r["target"]<0 for r in rows)) for m in METHODS}


def independent_tandem(fit):
    raw = {r["observation_id"]:r for r in read("shadow_change_control_estimates")}
    passes = {(r["capture_day"],r["candidate_id"],r["other_sensor"]):r for r in read("shadow_tandem_pass_pairs")
              if r["selected_for_summary"] == r["both_resolved"] == "True"}
    by_terminal = defaultdict(list)
    for r in read("terminal_change_repeatability_tank_controls"):
        if r["all_four_resolved"] != "True" or r["elapsed_days"] != "10":
            continue
        first,last = (passes[r[f"{k}_day"],r["candidate_id"],r["other_sensor"]] for k in ("earlier","later"))
        first_stamp,last_stamp = (datetime.fromisoformat(q["a_capture_at_utc"]) for q in (first,last))
        if first_stamp >= last_stamp or any(abs(float(q["other_minus_a_seconds"]))>60 for q in (first,last)):
            raise ValueError("Tandem timestamps differ")
        predictions = defaultdict(list)
        factor = float(r["kbbl_per_m"])
        for side, output in (("a","first"),("other","second")):
            before,after = (raw[q[f"{side}_observation_id"]] for q in (first,last))
            change = float(before["estimated_gap_m"])-float(after["estimated_gap_m"])
            low = float(before["near_optimal_gap_min_m"])-float(after["near_optimal_gap_max_m"])
            high = float(before["near_optimal_gap_max_m"])-float(after["near_optimal_gap_min_m"])
            adjusted = attenuate(change,low,high)
            if not math.isclose(change*factor,float(r[f"{output}_platform_change_kbbl"]),abs_tol=1e-8):
                raise ValueError("Tandem change does not reproduce")
            for name,value in zip(METHODS,(change,adjusted,float(fit@[change,adjusted])),strict=True):
                predictions[name].append(value*factor)
        by_terminal[r["earlier_day"],r["later_day"],r["terminal_id"]].append(predictions)
    for (start,stop,terminal), rows in sorted(by_terminal.items()):
        record = dict(earlier_day=start,later_day=stop,terminal_id=terminal,tanks=len(rows))
        for method in METHODS:
            first,last = map(math.fsum,zip(*(r[method] for r in rows),strict=True))
            record.update({f"{method}_first_kbbl":first,f"{method}_second_kbbl":last,
                           f"{method}_disagreement_kbbl":abs(first-last),f"{method}_opposite_sign":first*last<0})
        yield record


def main():
    reference = aerial()
    terminals = {r["terminal_id"] for r in reference}
    by_terminal = {t:train([q for q in reference if q["terminal_id"] != t]) for t in terminals}
    rows = [dict(r,learned=float(by_terminal[r["terminal_id"]]@[r["physical"],r["attenuated"]]),
                 bbl_per_m=float(r["roof_area_m2"])/BARREL_M3) for r in reference]
    fit = train(reference)
    tandem = list(independent_tandem(fit))
    summary = dict(aerial=dict(roofs=len(rows),terminals=len({r["terminal_id"] for r in rows}),
                               distinct_flight_intervals=len({(r["first_flight"],r["last_flight"]) for r in rows}),
                               metrics=score(rows)),
                   aerial_trained_weights=dict(zip(METHODS[:2],map(float,fit),strict=True)),
                   tandem={method:dict(intervals=len(tandem),median_disagreement_kbbl=float(np.median(
                       [r[f"{method}_disagreement_kbbl"] for r in tandem])),opposite_sign=sum(r[f"{method}_opposite_sign"] for r in tandem),
                       median_average_absolute_signal_kbbl=float(np.median([
                           (abs(r[f"{method}_first_kbbl"])+abs(r[f"{method}_second_kbbl"]))/2 for r in tandem])))
                           for method in METHODS})
    export(f"{STEM}_aerial",rows)
    export(f"{STEM}_tandem",tandem)
    payload = json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
