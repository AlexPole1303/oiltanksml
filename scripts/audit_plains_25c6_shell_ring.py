"""Separate the upper LiDAR ring from the exterior envelope and replay optical fits."""

import json
from collections import Counter
from datetime import date

import numpy as np

from audit_lidar_tanks import DATASET, GPS_EPOCH, SURVEYS
from audit_plains_25c6_lidar_reference import CALIBRATION, STEM as HISTORIC
from audit_plains_25c6_shadow_support import landsat_inputs, scores, sentinel_inputs, summarize
from build_cushing_okeis_plains_control import STEM as PLAINS
from build_lidar_radial_profiles import Circle, select_returns
from build_s1_reference_matches import DERIVED, export, read
from build_tank_lidar_outlines import circle
from build_weekly_shadow_panel import RESOLVED
from collect_public import ROOT, write
from fit_shadow_geometry import shifted_grid

STEM = f"{PLAINS}_shell_ring"
WINDOWS = ((0.6, 1.6), (0.75, 1.45), (0.5, 1.75))
RADIAL_LIMITS = (28., 32.)


def points_for_terminal():
    registry = {r["candidate_id"]: r for r in read("tank_engineering_registry")
                if r["documented_terminal_id"] == "cushing_connect"}
    refs = sorted((r for r in read("lidar_roof_elevation_reference") if r["candidate_id"] in registry
                   and r["dataset"] == DATASET and r["acquisition_date_utc"] == "2022-12-31"),
                  key=lambda r: r["candidate_id"])
    if len(refs) != len(registry) or len({r["source_directory"] for r in refs}) != 1:
        raise ValueError("The terminal does not have one dated LiDAR reference per contour")
    circles = [Circle(r["candidate_id"], r["geometry_basis"], r["outline_fit_status"],
                      *(float(r[k]) for k in ("center_east_m", "center_north_m", "radius_m"))) for r in refs]
    folder = ROOT / refs[0]["source_directory"]
    nodes = [folder / f"{node}.laz" for node in sorted({n for r in refs for n in json.loads(r["source_node_ids_json"])})]
    selected = select_returns(circles, SURVEYS[DATASET], nodes, include_xy=True)
    for i, (r, outline) in enumerate(zip(refs, circles, strict=True)):
        a = np.concatenate(selected[i])
        day = (date.fromisoformat(r["acquisition_date_utc"]) - GPS_EPOCH.date()).days
        a = a[(a[:, 0] == day) & (a[:, 1] == int(r["point_source_id"])) & (a[:, 6] == 0)]
        yield r, outline, a, nodes


def upper_rings(design_diameter):
    geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry")}
    for ref, outline, a, nodes in points_for_terminal():
        z0 = float(ref["rim_sector_p95_median_native"])
        roof_z = float(ref["center_plane_z_native"])
        ground = float(geometry[outline.candidate]["prior_ground_z_p50_native"])
        r = np.hypot(a[:, 8], a[:, 9])
        near = (r > RADIAL_LIMITS[0]) & (r < RADIAL_LIMITS[1])
        for lo, hi in WINDOWS:
            chosen = a[near & (a[:, 5] > z0 + lo) & (a[:, 5] < z0 + hi)]
            sectors = chosen[:, 3].astype(int)
            counts = np.bincount(sectors, minlength=8)
            row = dict(candidate_id=outline.candidate, dataset=ref["dataset"], acquisition_date_utc=ref["acquisition_date_utc"],
                       point_source_id=ref["point_source_id"], reference_ring_z_m=z0, selection_low_above_reference_m=lo,
                       selection_high_above_reference_m=hi, point_count=len(chosen), sector_counts_json=json.dumps(counts.tolist()),
                       envelope_diameter_m=2*outline.radius, operator_25c6_diameter_m=design_diameter,
                       lidar_center_plane_z_m=roof_z, lidar_surrounding_ground_p50_m=ground,
                       source_node_ids_json=json.dumps([nodes[i].stem for i in np.unique(chosen[:, 7]).astype(int)]))
            fields = dict(upper_ring_east_m=None, upper_ring_north_m=None, upper_ring_z_m=None, upper_ring_diameter_m=None,
                          upper_ring_to_center_plane_m=None, upper_ring_to_surrounding_ground_m=None,
                          p90_abs_radial_residual_m=None, omit_one_sector_diameter_min_m=None, omit_one_sector_diameter_max_m=None)
            status = "insufficient_angular_support"
            if counts.min() >= 3:
                xy = chosen[:, 8:10]
                centre, radius = circle(xy)
                z = float(np.median(chosen[:, 5]))
                residual = float(np.percentile(abs(np.linalg.norm(xy-centre, axis=1)-radius), 90))
                omitted = [2*circle(xy[sectors != k])[1] for k in range(8)]
                status = "coherent_upper_ring" if residual <= .35 else "diffuse_upper_returns"
                fields = dict(upper_ring_east_m=float(outline.east+centre[0]), upper_ring_north_m=float(outline.north+centre[1]),
                              upper_ring_z_m=z, upper_ring_diameter_m=2*radius, upper_ring_to_center_plane_m=z-roof_z,
                              upper_ring_to_surrounding_ground_m=z-ground, p90_abs_radial_residual_m=residual,
                              omit_one_sector_diameter_min_m=min(omitted), omit_one_sector_diameter_max_m=max(omitted))
            yield row | dict(status=status) | fields


def optical_sources():
    yield from sentinel_inputs()
    yield from landsat_inputs()
    yield from sentinel_inputs(CALIBRATION, HISTORIC, "sentinel_fits", subset=True)
    baselines = {(r["candidate_id"], r["version_id"]): r for r in read(f"{HISTORIC}_comparisons") if r["method"] == "landsat_b8"}
    yield from landsat_inputs(HISTORIC, 2022, baselines={key: dict(gap_m=r["optical_gap_m"], fit_status=r["fit_status"])
                                                         for key, r in baselines.items()})


def optical_fits(sources, rings, nominal_gap):
    timing = {(r["candidate_id"], r["version_id"]): float(r["maximum_possible_separation_hours"])
              for r in read(f"{HISTORIC}_comparisons") if r["method"] in ("red", "landsat_b8")}
    for s in sources:
        candidate, version = s["key"]
        r = rings[candidate]
        separation = timing.get(s["key"])
        historic = separation is not None
        prior = s["row"]
        dx, dy = (r[f"upper_ring_{new}_m"]-prior[old] for new, old in (("east", "center_east_m"), ("north", "center_north_m")))
        grid = shifted_grid(s["grid"], dx, dy)
        diameter = r["upper_ring_diameter_m"]
        shell = r["upper_ring_to_surrounding_ground_m"]
        ref_gap = r["upper_ring_to_center_plane_m"] if historic else nominal_gap
        tests = (("baseline", s["grid"], prior["radius_m"], s["shell"], False, 1, None),
                 ("fine_grid_original_geometry", s["grid"], prior["radius_m"], s["shell"], True, 4, None),
                 ("fine_grid_upper_ring_xy", grid, diameter/2, s["shell"], True, 4, None),
                 ("fine_grid_upper_ring_xy_z_original_limit", grid, diameter/2, shell, True, 4, s["shell"]),
                 ("fine_grid_upper_ring_xy_z", grid, diameter/2, shell, True, 4, None))
        for test, samples, radius, height, fine, subdivisions, gap_limit in tests:
            gaps, shifts, loss, reverse = scores(s, samples, height, radius, fine=fine, subdivisions=subdivisions, gap_limit=gap_limit)
            fit, _ = summarize(gaps, shifts, loss, reverse, samples, ref_gap)
            if test == "baseline":
                baseline = s["baseline"]
                if (fit["gap_m"], fit["fit_status"]) != (float(baseline.get("gap_m") or baseline["estimated_gap_m"]), baseline["fit_status"]):
                    raise ValueError(f"Original result differs for {candidate}/{version}")
            yield dict(candidate_id=candidate, version_id=version, band=s["band"], capture_at_utc=s["capture"],
                       test=test, radius_m=radius, shell_height_scenario_m=height, reference_gap_m=ref_gap,
                       reference_kind="dated_lidar_upper_ring_to_roof" if historic else "conditional_operator_liquid_level",
                       maximum_lidar_separation_hours=separation,
                       fitted_minus_reference_gap_m=fit["gap_m"]-ref_gap if fit["fit_status"] == RESOLVED else None,
                       original_pixel_weights_held_fixed=True, **fit)


def main():
    ground, = read(f"{PLAINS}_ground_record")
    diameter = float(ground["operator_diameter_ft"])*.3048
    nominal_gap = float(ground["operator_shell_height_ft"])*.3048 - float(ground["liquid_level_m"])
    possible = set(json.loads(ground["possible_candidate_ids_json"]))
    profiles = list(upper_rings(diameter))
    rings = {r["candidate_id"]: r for r in profiles if r["selection_low_above_reference_m"] == WINDOWS[0][0]
             and r["candidate_id"] in possible and r["status"] == "coherent_upper_ring"}
    if rings.keys() != possible:
        raise ValueError("Both candidate circles must support an upper ring")
    fits = list(optical_fits(optical_sources(), rings, nominal_gap))
    export(f"{STEM}_lidar", profiles)
    export(f"{STEM}_optical", fits)
    passed = Counter((r["capture_at_utc"][:4], r["test"]) for r in fits if r["fit_status"] == RESOLVED)
    close = [r for r in fits if r["test"] == "fine_grid_upper_ring_xy_z" and r["maximum_lidar_separation_hours"] is not None
             and r["maximum_lidar_separation_hours"] <= 24 and r["fit_status"] == RESOLVED]
    differences = [abs(r["fitted_minus_reference_gap_m"]) for r in close]
    report = dict(lidar_selection_variants=len(profiles), optical_fits=len(fits),
                  passed_by_capture_year={y: {t: passed[y, t] for t in sorted({r["test"] for r in fits})}
                                          for y in sorted({r["capture_at_utc"][:4] for r in fits})},
                  full_upper_ring_fits_within_24h_by_band=Counter(r["band"] for r in close),
                  full_upper_ring_abs_gap_difference_within_24h_m=[min(differences), max(differences)] if differences else None,
                  admitted_synchronized_ground_labels=0)
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
