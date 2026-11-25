"""Measure the two 25C-6 candidates in archived LiDAR and nearby optical passes."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from audit_plains_25c6_detector_angles import recovered_profiles
from audit_sentinel2_alternate_registration import packet_inputs
from build_cushing_inspection_b8 import recover
from build_cushing_okeis_plains_control import STEM as PLAINS
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_landsat import LANDSAT, catalog
from collect_public import write
from collect_s2_spectral_detector_masks import collect, jobs

STEM = f"{PLAINS}_lidar_reference"
CALIBRATION = "shadow_terminal_calibration.sqlite"
LANDSAT_PRODUCT = "LC08_L1TP_027035_20221230_20230110_02_T1"
OPTICAL_DATES = {"2022-12-30", "2023-01-04"}


def inputs():
    ground, = read(f"{PLAINS}_ground_record")
    roofs = set(json.loads(ground["possible_candidate_ids_json"]))
    geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry") if r["candidate_id"] in roofs
                and r["modern_input_status"] == "available"}
    references = {r["candidate_id"]: r for r in read("lidar_roof_elevation_reference") if r["candidate_id"] in roofs
                  and r["dataset"] == "OK_Statewide_2_D22" and r["acquisition_date_utc"] == "2022-12-31"}
    if not geometry.keys() == references.keys() == roofs:
        raise ValueError("A possible 25C-6 contour lacks the dated LiDAR profile")
    return geometry, references


def sentinel(geometry, references):
    path, observations, scenes, _ = packet_inputs(CALIBRATION)
    selected = {k: r for k, r in observations.items() if r["candidate_id"] in geometry and r["captured_at_utc"][:10] in OPTICAL_DATES}
    scenes = {i: scenes[i] for i in {r["scene_id"] for r in selected.values()}}
    if len(scenes) != 4 or len(selected) != 4*len(geometry):
        raise ValueError("Unexpected Sentinel-2 coverage around the December 2022 LiDAR")
    with ThreadPoolExecutor(max_workers=4) as pool:
        masks = list(pool.map(collect, jobs({r["version_id"]: r["manifest_path"] for r in scenes.values()})))
    files = {(r["version_id"], r["band"]): r["local_file"] for r in masks}
    fits, angles, _, _ = recovered_profiles(path, selected, scenes, files, geometry,
                                             {k: float(r["rim_minus_center_plane_native_z"]) for k, r in references.items()})
    renamed = {"conditional_operator_nominal_gap_m": "lidar_reference_trial_gap_m",
               "conditional_trial_to_best_loss_ratio": "lidar_reference_trial_to_best_loss_ratio",
               "admitted_synchronized_ground_label": "admitted_synchronized_lidar_label"}
    rows = [{renamed.get(k, k): v for k, v in r.items()} for r in fits]
    return rows, angles, masks


def landsat(geometry):
    item, = [r for r in catalog(2022, "usgs") if r["id"] == LANDSAT_PRODUCT]
    scenes, fits, _ = recover({LANDSAT_PRODUCT: set(geometry)}, {LANDSAT_PRODUCT: item}, store=LANDSAT / STEM,
                              native=DERIVED / f"{STEM}_landsat_native.sqlite", purpose="retrospective_2022_lidar_same_contours",
                              dated_geometry=geometry, geometry_basis="dated_2022_lidar")
    export(f"{STEM}_landsat_sources", scenes)
    return [dict(candidate_id=r["candidate_id"], version_id=r["product_id"], capture_at_utc=r["optical_capture_at_utc"],
                 method="landsat_b8", fit_status=r["fit_status"], gap_m=r.get("estimated_gap_m"),
                 near_min_m=r.get("near_optimal_gap_min_m"), near_max_m=r.get("near_optimal_gap_max_m")) for r in fits]


def comparisons(references, fits):
    for fit in fits:
        r = references[fit["candidate_id"]]
        t = datetime.fromisoformat(fit["capture_at_utc"])
        first, last = (datetime.fromisoformat(r[f"{side}_return_at_utc"]) for side in ("first", "last"))
        gap = float(r["rim_minus_center_plane_native_z"])
        yield dict(candidate_id=fit["candidate_id"], version_id=fit["version_id"], optical_capture_utc=t.isoformat(),
                   lidar_first_return_utc=first.isoformat(), lidar_last_return_utc=last.isoformat(),
                   optical_minus_lidar_last_hours=(t-last).total_seconds()/3600,
                   maximum_possible_separation_hours=max(abs((t-x).total_seconds()) for x in (first, last))/3600,
                   method=fit["method"], fit_status=fit["fit_status"], optical_gap_m=fit["gap_m"],
                   optical_near_min_m=fit["near_min_m"], optical_near_max_m=fit["near_max_m"], lidar_gap_m=gap,
                   lidar_center_plane_residual_p90_minus_p10_m=float(r["center_plane_residual_p90_minus_p10_native"]),
                   optical_minus_lidar_gap_m=float(fit["gap_m"])-gap if fit["fit_status"] == RESOLVED else None,
                   measurement_same_instant=first <= t <= last, equipment_id_known_for_contour=False)


def main():
    geometry, references = inputs()
    with ThreadPoolExecutor(max_workers=2) as pool:
        optical = pool.submit(sentinel, geometry, references)
        b8 = pool.submit(landsat, geometry)
        fits, angles, masks = optical.result()
        pan = b8.result()
    rows = list(comparisons(references, fits+pan))
    for suffix, records in (("lidar", list(references.values())), ("sentinel_fits", fits), ("sentinel_angles", angles),
                            ("sentinel_masks", masks), ("comparisons", rows)):
        export(f"{STEM}_{suffix}", records)
    before = [r for r in rows if r["maximum_possible_separation_hours"] <= 24]
    passed = [r for r in before if r["optical_minus_lidar_gap_m"] is not None]
    errors = [abs(r["optical_minus_lidar_gap_m"]) for r in passed]
    report = dict(lidar_roof_profiles=len(references), comparisons=len(rows), comparisons_within_24h=len(before),
                  qualified_fits_within_24h_by_method=Counter(r["method"] for r in passed),
                  within_24h_abs_gap_difference_m=[min(errors), max(errors)] if errors else None,
                  synchronized_observations=sum(r["measurement_same_instant"] for r in rows))
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
