"""Extend the independent roof-change comparison to October 2020 aerial plans."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import combinations
from statistics import fmean, median

import cv2
import numpy as np
from audit_aerialok_native_optical import NEW_SITES, SITES, engineering, pair_observations, plate, solar_module
from build_aerialok_october_controls import SHEETS
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write

STEM = "cushing_autumn_aerial_intervals"
FLIGHTS = ("2020-10-06", "2020-10-13", "2020-10-20")
SIDES = ("north", "west")
MAX_HOURS = 30.
REFERENCE_DAY = "2020-09-07"
IMAGE_SCALE = .42


def features(path, zones=()):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.shape != (4050, 3150):
        raise ValueError(f"Invalid aerial plan {path}")
    gray = cv2.resize(gray[:3546], None, fx=IMAGE_SCALE, fy=IMAGE_SCALE)
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    for x, y, radius in zones:
        cv2.circle(mask, (round(x*IMAGE_SCALE), round(y*IMAGE_SCALE)), round(radius*IMAGE_SCALE), 0, -1)
    return cv2.SIFT_create(nfeatures=5500).detectAndCompute(gray, mask)


def exterior_translation(day, side, reference, roof_rows):
    source = ROOT/reference["source_path"]
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != reference["sha256"]:
            raise ValueError(f"Reference plan changed on disk for {side}")
    scale_m = float(reference["metres_per_printed_pixel"])
    centres = [(float(r["plate_x_at_1m"])/scale_m, float(r["plate_y_at_1m"])/scale_m,
                (float(r["diameter_m"])/2+60)/scale_m) for r in roof_rows]
    key, descriptor = features(source, centres)
    target = SHEETS/f"oct{day[-2:]}_{side}_{day}.png"
    match, packet = features(target)
    if descriptor is None or packet is None or len(packet) < 2:
        raise ValueError(f"Insufficient native terrain detail on {target.name}")
    accepted = [a for a,b in cv2.BFMatcher().knnMatch(descriptor,packet,k=2) if a.distance < .68*b.distance]
    first = np.array([key[r.queryIdx].pt for r in accepted])/IMAGE_SCALE
    second = np.array([match[r.trainIdx].pt for r in accepted])/IMAGE_SCALE
    if len(first) < 40:
        raise ValueError(f"Insufficient exterior matches on {target.name}")
    tile = np.floor(first/360.).astype(int)
    calibration = tile.sum(axis=1)%2==0
    if min(int(calibration.sum()), int((~calibration).sum())) < 15:
        raise ValueError(f"Cannot withhold independent ground patches on {target.name}")
    matrix, inliers = cv2.estimateAffinePartial2D(first[calibration],second[calibration],method=cv2.RANSAC,ransacReprojThreshold=4.)
    if matrix is None or inliers is None or int(inliers.sum()) < 15:
        raise ValueError(f"No independent terrain registration on {target.name}")
    residual = np.linalg.norm(first[~calibration]@matrix[:,:2].T+matrix[:,2]-second[~calibration],axis=1)*scale_m
    center = np.array((3150/2,3546/2))
    travel = matrix[:,:2]@center+matrix[:,2]-center
    # The public export sets a fixed printed scale. Verify the actual image
    # stayed at that scale before using a translation-only rim registration.
    transform_scale = float(np.linalg.norm(matrix[:,0]))
    rotation = math.degrees(math.atan2(matrix[1,0],matrix[0,0]))
    if (abs(transform_scale-1.)>.002 or abs(rotation)>.12 or float(np.median(residual))>3
            or not np.isfinite(travel).all()):
        raise ValueError(f"The plan is not aligned at its printed scale on {target.name}")
    shift = np.array([float(roof_rows[0][f"group_shift_{direction}_m"]) for direction in ("east","south")])+travel*scale_m
    return shift, dict(fit_inliers=int(inliers.sum()), independent_matches=len(residual),
        independent_median_residual_m=float(np.median(residual)), reference_sheet=reference["source_path"],
        reference_sheet_sha256=reference["sha256"], transform_scale=transform_scale, rotation_degrees=rotation,
        exterior_east_m=float(shift[0]), exterior_south_m=float(shift[1]))


def comparisons(sources):
    endpoints = {}
    for r in sources:
        if (not r["both_pass_at_inferred_best"] or not r["all_near_clock_fits_visible"]
                or r["maximum_inferred_separation_hours"] > MAX_HOURS):
            continue
        if r.get("roof_type", "EFR") != "EFR":
            continue
        sensor = r["satellite_sensor"].removesuffix(" direct")
        if sensor == "Landsat 8 RT B8":
            continue
        key = (r["candidate_id"], r["flight_date"], sensor)
        rank = (r["maximum_inferred_separation_hours"], r["maximum_calendar_day_separation_hours"], r["satellite_version"])
        if key not in endpoints or rank < endpoints[key][0]:
            endpoints[key] = (rank, r)
    by_tank = defaultdict(list)
    for (tank, day, sensor), (_, r) in endpoints.items():
        by_tank[tank].append((day, sensor, r))
    for tank, records in sorted(by_tank.items()):
        for (first, a, p), (last, b, q) in combinations(sorted(records, key=lambda u: u[:2]), 2):
            start, stop = (datetime.fromisoformat(r["satellite_captured_at_utc"]) for r in (p,q))
            days = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days
            hours = (stop - start).total_seconds()/3600
            if not 5 <= days <= 9 or not 5*24 <= hours <= 9*24:
                continue
            if p["terminal_id"] != q["terminal_id"] or p["diameter_m"] != q["diameter_m"]:
                raise ValueError(f"The paired aerial heights use different geometry for {tank}")
            physical = p["satellite_gap_m"] - q["satellite_gap_m"]
            aerial = p["aerial_gap_m"] - q["aerial_gap_m"]
            lower = p["satellite_gap_min_m"]-q["satellite_gap_max_m"]-(p["aerial_gap_max_m"]-q["aerial_gap_min_m"])
            upper = p["satellite_gap_max_m"]-q["satellite_gap_min_m"]-(p["aerial_gap_min_m"]-q["aerial_gap_max_m"])
            if not lower-1e-8 <= physical-aerial <= upper+1e-8:
                raise ValueError(f"The signed change lies outside the reconstructed range for {tank}")
            bbl_m = math.pi*(p["diameter_m"]/2)**2/BARREL_M3
            yield dict(candidate_id=tank, terminal_id=p["terminal_id"], first_flight=first, last_flight=last, first_sensor=a,
                last_sensor=b, satellite_interval_hours=hours, first_satellite_version=p["satellite_version"],
                last_satellite_version=q["satellite_version"], first_aerial=p["aerial_source"], last_aerial=q["aerial_source"],
                first_aerial_sha256=p["aerial_sha256"], last_aerial_sha256=q["aerial_sha256"],
                first_hours=p["maximum_inferred_separation_hours"], last_hours=q["maximum_inferred_separation_hours"],
                maximum_separation_hours=max(p["maximum_inferred_separation_hours"],q["maximum_inferred_separation_hours"]),
                first_calendar_hours=p["maximum_calendar_day_separation_hours"], last_calendar_hours=q["maximum_calendar_day_separation_hours"],
                satellite_rise_m=physical, aerial_rise_m=aerial, satellite_minus_aerial_rise_m=physical-aerial,
                difference_min_m=lower, difference_max_m=upper, difference_range_excludes_zero=lower>0 or upper<0,
                cylindrical_bbl_per_m=bbl_m, satellite_minus_aerial_cylindrical_bbl=bbl_m*(physical-aerial))


def statistics(rows):
    if not rows:
        return dict(comparisons=0, distinct_roofs=0)
    signed = [r["satellite_minus_aerial_rise_m"] for r in rows]
    return dict(comparisons=len(rows), distinct_roofs=len({r["candidate_id"] for r in rows}),
        distinct_intervals=len({(r["first_flight"],r["last_flight"]) for r in rows}),
        max_inferred_separation_hours=max(r["maximum_separation_hours"] for r in rows),
        mean_absolute_difference_m=fmean(map(abs,signed)), median_absolute_difference_m=median(map(abs,signed)),
        mean_signed_difference_m=fmean(signed),
        same_nonzero_sign=sum(r["satellite_rise_m"]*r["aerial_rise_m"]>0 for r in rows),
        opposite_sign=sum(r["satellite_rise_m"]*r["aerial_rise_m"]<0 for r in rows))


def transfer_check(fall, december):
    # Fit on comparable December roofs and evaluate on a different month.
    # The estimate is a retrospective transfer test, not a dated forecast.
    landsat = {"Landsat 7 T1 B8", "Landsat 8 T1 B8"}
    eligible = [r for r in december if {r["first_sensor"],r["last_sensor"]}==landsat and r["maximum_separation_hours"]<=24]
    later = [r for r in fall if {r["first_sensor"],r["last_sensor"]}==landsat and r["maximum_separation_hours"]<=24]
    if not eligible or not later:
        return dict(december_roofs=len(eligible), october_roofs=len(later)), []
    direction = lambda r: 1 if r["first_sensor"] == "Landsat 7 T1 B8" else -1
    # At the same true height, reversing the sensor order also reverses an
    # additive instrument bias. This stress tests that transfer assumption.
    by_terminal = {site:median(direction(r)*r["satellite_minus_aerial_rise_m"] for r in eligible if r["terminal_id"] == site)
                   for site in sorted({r["terminal_id"] for r in eligible})}
    if len(by_terminal) < 2:
        raise ValueError("Cross-terminal transfer requires at least two December terminals")
    offset = median(by_terminal.values())
    evaluated = []
    for r in later:
        held_out_offset = median(value for site,value in by_terminal.items() if site != r["terminal_id"])
        signed = r["satellite_minus_aerial_rise_m"]
        evaluated.append(r | dict(december_landsat_offset_m=offset, december_without_terminal_offset_m=held_out_offset,
            adjusted_minus_aerial_rise_m=signed-direction(r)*offset,
            adjusted_without_terminal_minus_aerial_rise_m=signed-direction(r)*held_out_offset))
    return dict(december_roofs=len(eligible), december_terminals=len(by_terminal), october_roofs=len(later),
        estimated_offset_l7_minus_l8_m=offset, unadjusted_mae_m=fmean(abs(r["satellite_minus_aerial_rise_m"]) for r in evaluated),
        adjusted_mae_m=fmean(abs(r["adjusted_minus_aerial_rise_m"]) for r in evaluated),
        adjusted_without_terminal_mae_m=fmean(abs(r["adjusted_without_terminal_minus_aerial_rise_m"]) for r in evaluated),
        by_terminal_december_offset_m=by_terminal), evaluated


def main():
    stations = {f"oct{day[-2:]}_{side}": NEW_SITES[side] for day in FLIGHTS for side in SIDES}
    SITES.update(stations)
    roofs, spa = list(engineering()), solar_module()
    jobs = [(day, f"oct{day[-2:]}_{side}", side) for day in FLIGHTS for side in SIDES]
    ref = {r["site"].removeprefix("sep07_"):r for r in read("aerialok_native_optical_sheets") if r["flight_date"]==REFERENCE_DAY}
    by_side = defaultdict(list)
    for r in read("aerialok_native_optical_aerial"):
        if r["flight_date"]==REFERENCE_DAY:
            by_side[r["side"]].append(r)

    def fit(job):
        day,_,side = job
        shift, proof = exterior_translation(day,side,ref[side],by_side[side])
        manifest, estimates = plate(job, roofs, spa, centre_shift_m=shift)
        return manifest | proof, estimates

    with ThreadPoolExecutor(max_workers=2) as pool:
        rasters = list(pool.map(fit, jobs))
    aerial = [r for _, rows in rasters for r in rows if r["roof_type"] == "EFR"]
    candidates = [r | dict(roof_type="EFR") for r in pair_observations(aerial)]
    selected = list(comparisons(candidates))
    efr_roofs = {r["candidate_id"] for r in roofs if r["roof_type"] == "EFR"}
    truth_fields = ("satellite_gap_m", "satellite_gap_min_m", "satellite_gap_max_m", "aerial_gap_m", "aerial_gap_min_m",
                    "aerial_gap_max_m", "maximum_inferred_separation_hours", "maximum_calendar_day_separation_hours", "diameter_m")
    dec_endpoints = [r | {k:float(r[k]) for k in truth_fields} | dict(both_pass_at_inferred_best=True, all_near_clock_fits_visible=True)
                     for r in read("aerialok_native_optical_pairs") if r["flight_date"] in ("2020-12-01", "2020-12-08")
                     and r["candidate_id"] in efr_roofs and r["both_pass_at_inferred_best"]==r["all_near_clock_fits_visible"]=="True"]
    transfer, comparison = transfer_check(selected, list(comparisons(dec_endpoints)))
    group = defaultdict(list)
    for r in selected:
        group[" / ".join(r[k] for k in ("first_flight","last_flight","first_sensor","last_sensor"))].append(r)
    physical = [r for r in aerial if r["aerial_fit_status"] == "visible_shadow_fit" and r["all_near_clock_fits_visible"]]
    verified_endpoints = [r for r in candidates if r["both_pass_at_inferred_best"] and r["all_near_clock_fits_visible"]]
    summary = dict(source_sheets=len(rasters), inspected_roof_observations=len(aerial),
        max_independent_registration_median_residual_m=max(r["independent_median_residual_m"] for r,_ in rasters),
        aerial_fit_passes_by_day=Counter(r["flight_date"] for r in physical),
        joint_endpoint_passes_by_day=Counter(r["flight_date"] for r in verified_endpoints),
        within_30h=statistics(selected), within_24h=statistics([r for r in selected if r["maximum_separation_hours"]<=24]),
        within_12h=statistics([r for r in selected if r["maximum_separation_hours"]<=12]),
        by_interval_and_sensor={k:statistics(v) for k,v in sorted(group.items())},
        december_sensor_offset_tested_on_october=transfer,
        scope="Aerial times inferred from exterior shadows independent of the target. Aerial and satellite shadow estimates may share geometry errors. This is a retrospective engineering comparison.")
    export(f"{STEM}_sheets", [r for r,_ in rasters])
    export(f"{STEM}_paired_endpoints", candidates)
    if selected:
        export(f"{STEM}_roof_changes", selected)
    if comparison:
        export(f"{STEM}_offset_transfer", comparison)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(summary,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
