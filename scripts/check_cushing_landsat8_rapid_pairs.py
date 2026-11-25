"""Check common-centre roof changes for independently registered Landsat 8 RT pairs."""

import json
from collections import Counter, defaultdict
from datetime import datetime
from functools import lru_cache
from statistics import median, quantiles

from build_cross_sensor_shadow import PAN_PIXEL, PAN_STEP, along_track_bearing, pan_psf, published_lsf
from build_registered_shadow_development import registered_pair
from build_terminal_development_measurements import BARREL_M3
from measure_cushing_landsat7_mixed import CRS, DIRECTIONS, OFFSETS, profiles, sentinel_chip
from measure_cushing_landsat8_rapid import DERIVED, STEM, export, legacy_geometry, native_raster, roof_inputs, rows, save_json

SIDES = ("previous", "current")


def candidate_key(row):
    return row["week_ending"], row["candidate_id"], *(row[f"{s}_version_id"] for s in SIDES)


def registration_key(row):
    return row["week_or_control_day"], row["candidate_id"], row["earlier_version_id"], row["later_version_id"]


def roof_parameters(g):
    return (float(g["center_east_m"]), float(g["center_north_m"]), float(g["diameter_m"])/2,
            float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"]))


def prepare(legacy, lsf):
    @lru_cache(maxsize=128)
    def b8_chip(product, tank, source, count):
        item, raster, angles, gain, bias = native_raster(product)
        roofs = {r["candidate_id"]:r for r in item["roofs"]}
        chip, _ = roof_inputs(raster, roofs[tank], legacy[tank], gain, bias)
        if source != item["native_packet"] or count != len(chip[0][-1]):
            raise ValueError(f"B8 independent measurement does not reproduce: {product}/{tank}")
        return chip, pan_psf(lsf,along_track_bearing(angles,CRS)), PAN_PIXEL, PAN_STEP, 3

    def endpoint(row, side):
        src = {k.removeprefix(side+"_"):v for k,v in row.items() if k.startswith(side+"_")}
        tank = row["candidate_id"]
        if src["sensor"] == "B8":
            return b8_chip(src["version_id"],tank,src["source"],int(src["fitted_pixels"]))
        data = {f"sentinel_{k}":src[k] for k in ("version_id","canonical_id","captured_at_utc","observation_id",
                                                "observation_source","fitted_pixels","gap_m")}
        data.update(candidate_id=tank, week_ending=row["week_ending"], landsat_product_id=next(
            row[f"{s}_version_id"] for s in SIDES if row[f"{s}_sensor"]=="B8"))
        chip, kernel = sentinel_chip(data, roof_parameters(legacy[tank]))
        return chip, kernel, 10, 2, 1

    return endpoint


def solve(row, offset, endpoint):
    dx,dy = (float(offset[f"later_minus_earlier_{k}_m"]) for k in ("east","north"))
    radius = max(2.,float(offset["check_median_residual_m"]),float(offset["fit_median_residual_m"]))
    shifts = [(dx,dy), *((dx+x*radius,dy+y*radius) for x,y in DIRECTIONS)]
    profiles_by_side = []
    for side, sign in zip(SIDES,(-.5,.5),strict=True):
        chip,kernel,pixel,step,subcells = endpoint(row,side)
        profiles_by_side.append(profiles(chip,kernel,[(sign*x,sign*y) for x,y in shifts],pixel,step,subcells))
    out = []
    for i,(first,last) in enumerate(zip(*profiles_by_side,strict=True)):
        fit = registered_pair(first,last)
        inside = max(abs(fit[f"registered_centre_{axis}_m"]) for axis in ("east","north")) < max(OFFSETS)
        out.append(dict(fit, offset_scenario=i, offset_east_m=shifts[i][0], offset_north_m=shifts[i][1],
                        fit_pass=bool(inside and fit["registered_both_shadow_diagnostics_pass"])))
    return radius,out


def check_pairs(candidates, offsets, endpoint):
    for row in candidates:
        key = candidate_key(row)
        offset = offsets[key]
        if (row["cutoff_utc"] is None or offset["comparison"]!=row["route"]
                or offset["release_date"]!=row["release_date"]):
            raise ValueError(f"Pair differs from the registration ledger: {key}")
        checked = offset["registration_status"]=="checked_offset" and offset["red_context_before_cutoff"]=="True"
        result = dict(row, exterior_registration_status=offset["registration_status"], registration_before_cutoff=checked,
            joint_central_fit_pass=False, all_offset_scenarios_pass=False, robust_nonzero_change=False,
            offset_stress_radius_m=None, central_capture_change_bbl=None, stress_capture_change_min_bbl=None,
            stress_capture_change_max_bbl=None, reporting_interval_observed_fraction=None, capture_within_reporting_fraction=None)
        if not checked:
            yield result,[]
            continue
        radius,fits = solve(row,offset,endpoint)
        central = fits[0]
        stable = all(f["fit_pass"] for f in fits)
        scale = float(row["area_m2"])/BARREL_M3
        lower = min(f["registered_loss_range_min_m"] for f in fits)*scale if stable else None
        upper = max(f["registered_loss_range_max_m"] for f in fits)*scale if stable else None
        a,b,t0,t1 = (datetime.fromisoformat(row[k]) for k in ("previous_boundary_utc","current_boundary_utc","previous_captured_at_utc","current_captured_at_utc"))
        if not t0<=a<t1<=b:
            raise ValueError(f"Invalid capture windows for {key}")
        result.update(joint_central_fit_pass=central["fit_pass"], all_offset_scenarios_pass=stable,
            robust_nonzero_change=bool(stable and (lower>0 or upper<0)), offset_stress_radius_m=radius,
            central_capture_change_bbl=central["registered_change_m"]*scale if central["fit_pass"] else None,
            stress_capture_change_min_bbl=lower, stress_capture_change_max_bbl=upper,
            reporting_interval_observed_fraction=(t1-a)/(b-a), capture_within_reporting_fraction=(t1-a)/(t1-t0))
        fields = ("week_ending","candidate_id","route","previous_version_id","current_version_id")
        yield result,[{k:row[k] for k in fields}|fit for fit in fits]


def weekly(rows_by_pair):
    grouped = defaultdict(list)
    for row in rows_by_pair:
        if row["robust_nonzero_change"]:
            grouped[row["week_ending"],row["candidate_id"]].append(row)
    for group in grouped.values():
        # Select closest captures; do not rank by estimated barrels or the EIA result.
        yield min(group,key=lambda r:((datetime.fromisoformat(r["previous_boundary_utc"])-datetime.fromisoformat(r["previous_captured_at_utc"]))
            +(datetime.fromisoformat(r["current_boundary_utc"])-datetime.fromisoformat(r["current_captured_at_utc"])),
            r["previous_version_id"],r["current_version_id"]))


def sensor_difference(rows_by_pair):
    controls = rows(DERIVED/f"{STEM}_height_sentinel_controls.csv")
    result = []
    for pair in rows_by_pair:
        if not pair["robust_nonzero_change"] or pair["route"] == "B8_to_B8":
            continue
        target = pair["week_ending"][:4]
        sample = [r for r in controls if r["landsat_captured_at_utc"][:4]!=target
                  and float(r["capture_separation_hours"])<=3 and r["signed_gap_difference_m"]]
        if len(sample)<2:
            continue
        offsets = [float(r["signed_gap_difference_m"]) for r in sample]
        percentiles = quantiles(offsets,n=20,method="inclusive")
        lo,hi = percentiles[0],percentiles[-1]
        scale = float(pair["area_m2"])/BARREL_M3
        low,high = float(pair["stress_capture_change_min_bbl"]),float(pair["stress_capture_change_max_bbl"])
        # The B8-minus-B04 level difference enters with opposite signs in the two routes.
        sign = 1 if pair["route"]=="B8_to_B04" else -1
        stress_low = low-max(sign*lo,sign*hi)*scale
        stress_high = high-min(sign*lo,sign*hi)*scale
        result.append({k:pair[k] for k in ("week_ending","candidate_id","route","existing_b04_tanks","reporting_interval_observed_fraction")}
            | dict(training_control_pairs=len(sample), training_landsat_scenes=len({r["landsat_product_id"] for r in sample}),
                training_control_years="|".join(sorted({r["landsat_captured_at_utc"][:4] for r in sample})),
                median_b8_minus_b04_m=median(offsets), empirical_p05_b8_minus_b04_m=lo, empirical_p95_b8_minus_b04_m=hi,
                stressed_min_bbl=stress_low, stressed_max_bbl=stress_high, empirical_offset_range_excludes_zero=stress_low>0 or stress_high<0))
    return result


def main():
    candidates = rows(DERIVED / f"{STEM}_height_pair_candidates.csv")
    offsets = {registration_key(r):r for r in rows(DERIVED / f"{STEM}_registration_offsets.csv")}
    if offsets.keys() != {candidate_key(r) for r in candidates} or len(offsets)!=len(candidates):
        raise ValueError("Candidate and registration ledgers are inconsistent")
    results,cases = [],[]
    for r,fits in check_pairs(candidates,offsets,prepare(legacy_geometry(),published_lsf())):
        results.append(r)
        cases.extend(fits)
    export(f"{STEM}_registered_pairs",results)
    if cases:
        export(f"{STEM}_registered_stress_cases",cases)
    chosen = list(weekly(results))
    if chosen:
        export(f"{STEM}_selected_roof_weeks",chosen)
    difference = sensor_difference(chosen)
    if difference:
        export(f"{STEM}_cross_sensor_difference_sensitivity",difference)
    report = dict(candidate_pairs=len(results), exterior_registration_before_cutoff=sum(r["registration_before_cutoff"] for r in results),
        joint_central_fit_pairs=sum(r["joint_central_fit_pass"] for r in results), checked_offset_scenarios=len(cases),
        all_offset_scenarios_pass_pairs=sum(r["all_offset_scenarios_pass"] for r in results),
        robust_nonzero_capture_change_pairs=sum(r["robust_nonzero_change"] for r in results), selected_roof_weeks=len(chosen),
        selected_by_route=dict(Counter(r["route"] for r in chosen)), weeks_with_robust_change=len({r["week_ending"] for r in chosen}),
        weeks_without_existing_b04=len({r["week_ending"] for r in chosen if r["existing_b04_tanks"]=="0"}),
        median_fraction_of_eia_interval_observed=median(r["reporting_interval_observed_fraction"] for r in chosen) if chosen else None,
        mixed_roof_weeks_tested_against_cross_sensor_difference=len(difference),
        mixed_roof_weeks_with_sign_after_cross_sensor_stress=sum(r["empirical_offset_range_excludes_zero"] for r in difference),
        scope="Exploratory 2017–2021 capture-to-capture changes on a common centre. Offset scenarios use independent exterior checks. Sensor-response model and cross-sensor height bias remain uncertain.")
    save_json(DERIVED / f"{STEM}_registered_summary.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
