"""Inventory upper circular LiDAR returns for every documented open floating roof."""

import json
from collections import Counter, defaultdict
from datetime import date
from itertools import product
from statistics import median

import numpy as np

from audit_lidar_tanks import DATASET, GPS_EPOCH, SURVEYS
from build_lidar_radial_profiles import Circle, select_returns
from build_s1_reference_matches import DERIVED, export, read
from build_tank_lidar_outlines import circle
from collect_public import ROOT, write

STEM = "cushing_lidar_shell_rings"
SECTORS = 8
Z_HALF_M = 0.30
R_HALF_M = 0.45
MIN_PER_SECTOR = 3


def references():
    return [r for r in read("lidar_roof_elevation_reference") if r["dataset"] == DATASET
            and r["geometry_basis"] == "lidar_elevated_circle"]


def dated_returns(refs):
    directories = defaultdict(list)
    for r in refs:
        directories[r["source_directory"]].append(r)
    for directory, group in sorted(directories.items()):
        contours = {r["candidate_id"]: Circle(r["candidate_id"], r["geometry_basis"], r["outline_fit_status"],
                    *(float(r[k]) for k in ("center_east_m", "center_north_m", "radius_m"))) for r in group}
        indexes = {candidate: i for i, candidate in enumerate(contours)}
        nodes = sorted((ROOT / directory).glob("*.laz"))
        returns = select_returns(list(contours.values()), SURVEYS[DATASET], nodes, include_xy=True)
        for r in group:
            outline = contours[r["candidate_id"]]
            batches = returns.get(indexes[outline.candidate], [])
            a = np.concatenate(batches) if batches else np.empty((0, 10))
            day = (date.fromisoformat(r["acquisition_date_utc"]) - GPS_EPOCH.date()).days
            keep = (a[:, 0] == day) & (a[:, 1] == int(r["point_source_id"])) & (a[:, 6] == 0)
            yield r, outline, a[keep], nodes


def ring_candidates(a, outline, low_z, high_z):
    xy = a[:, 8:10]
    radial = np.hypot(xy[:, 0], xy[:, 1])
    lower_radius = max(outline.radius-5, outline.radius*.8)
    mask = ((a[:, 5] >= low_z) & (a[:, 5] <= high_z) & (radial >= lower_radius) & (radial <= outline.radius+1.25))
    a, xy, radial = a[mask], xy[mask], radial[mask]
    z, sectors = a[:, 5], a[:, 3].astype(int)
    if len(a) < SECTORS*MIN_PER_SECTOR:
        return []

    candidates = []
    for seed_z in np.arange(low_z, high_z+.01, .20):
        height_indices = np.flatnonzero(abs(z-seed_z) <= Z_HALF_M)
        if len(height_indices) < SECTORS*MIN_PER_SECTOR:
            continue
        for seed_r in np.arange(lower_radius, outline.radius+1.26, .30):
            chosen = height_indices[abs(radial[height_indices]-seed_r) <= R_HALF_M]
            counts = np.bincount(sectors[chosen], minlength=SECTORS)
            if counts.min() < MIN_PER_SECTOR:
                continue
            centre, radius = circle(xy[chosen])
            level = float(np.median(z[chosen]))
            residual = abs(np.linalg.norm(xy-centre, axis=1)-radius)
            chosen = np.flatnonzero((residual <= R_HALF_M) & (abs(z-level) <= Z_HALF_M))
            counts = np.bincount(sectors[chosen], minlength=SECTORS)
            if counts.min() < MIN_PER_SECTOR:
                continue
            # Equalize angular coverage by fitting each sector's median representative.
            ring_xy = np.asarray([np.median(xy[chosen[sectors[chosen] == k]], axis=0) for k in range(SECTORS)])
            centre, _ = circle(ring_xy)
            # The sector median vectors lie slightly inside an arc. Recover radius from the original points.
            distances = np.linalg.norm(xy[chosen]-centre, axis=1)
            radius = float(np.median([np.median(distances[sectors[chosen] == k]) for k in range(SECTORS)]))
            residual = abs(distances-radius)
            p90 = float(np.percentile(residual, 90))
            heights = [np.median(z[chosen[sectors[chosen] == k]]) for k in range(SECTORS)]
            if p90 > .35 or np.ptp(heights) > .38 or np.linalg.norm(centre) > 1.25:
                continue
            level = float(np.median(heights))
            score = (int(np.minimum(counts, 15).sum()), int(counts.min()), -p90)
            candidates.append((score, centre, radius, level, p90, counts, chosen))

    unique = []
    for candidate in sorted(candidates, key=lambda row: row[0], reverse=True):
        _, _, radius, level, *_ = candidate
        if not any(abs(other[2]-radius) < .50 and abs(other[3]-level) < .40 for other in unique):
            unique.append(candidate)
    return [(a, *row) for row in unique]


def fit_rings(refs, registry):
    for r, outline, a, nodes in dated_returns(refs):
        candidate = r["candidate_id"]
        tank = registry[candidate]
        roof = float(r["center_plane_z_native"]) if r["center_plane_z_native"] else None
        former_rim = float(r["rim_sector_p95_median_native"]) if r["rim_sector_p95_median_native"] else None
        ground = tank["lidar_rim_to_ground_native_z"]
        evidence = dict(candidate_id=candidate, terminal_id=tank["documented_terminal_id"], cohort_id=tank["cohort_id"],
                        dataset=r["dataset"], acquisition_date_utc=r["acquisition_date_utc"], point_source_id=r["point_source_id"],
                        first_lidar_return_at_utc=r["first_return_at_utc"], last_lidar_return_at_utc=r["last_return_at_utc"],
                        original_outline_status=r["outline_fit_status"], original_east_m=outline.east, original_north_m=outline.north,
                        original_diameter_m=2*outline.radius, original_rim_reference_z_m=former_rim, roof_plane_z_m=roof,
                        original_rim_to_ground_m=float(ground) if ground else None, permit_diameter_m=tank["permit_diameter_m"],
                        input_points=len(a), source_directory=r["source_directory"])
        rings = [] if roof is None or former_rim is None else ring_candidates(a, outline, max(roof+.55, former_rim-1.5), former_rim+2.5)
        if not rings:
            rings = [(a, None, None, None, None, None, np.zeros(SECTORS, dtype=int), np.array([], dtype=int))]
        for i, (points, score, centre, radius, z, p90, counts, chosen) in enumerate(rings, 1):
            yield evidence | dict(status="coherent_circle" if score else "no_supported_circle", ring_rank=i if score else None,
                                  rank_score=score[0] if score else None, ring_east_m=float(outline.east+centre[0]) if score else None,
                                  ring_north_m=float(outline.north+centre[1]) if score else None,
                                  ring_diameter_m=2*radius if score else None, ring_z_m=z, ring_gap_m=z-roof if score else None,
                                  p90_abs_radial_residual_m=p90, sector_counts_json=json.dumps(counts.tolist()), ring_points=len(chosen),
                                  source_node_ids_json=json.dumps([nodes[k].stem for k in np.unique(points[chosen, 7]).astype(int)]))


def contour_rows(rows, registry):
    groups = defaultdict(list)
    for r in rows:
        groups[r["candidate_id"]].append(r)
    for candidate, tank in sorted(registry.items()):
        group = groups[candidate]
        flight_keys = {(r["acquisition_date_utc"], r["point_source_id"]) for r in group}
        best = [r for r in group if r["ring_rank"] == 1]
        permit = float(tank["permit_diameter_m"]) if tank["permit_diameter_m"] else None
        basis = "documented_diameter_constraint" if permit else "lidar_only"
        by_flight = defaultdict(list)
        for r in group:
            if r["ring_rank"] is None:
                continue
            if permit is None or abs(r["ring_diameter_m"]-permit) <= max(.6, .01*permit):
                by_flight[r["acquisition_date_utc"], r["point_source_id"]].append(r)
        shortlist = [[r for r in options if r["rank_score"] >= .90*options[0]["rank_score"]]
                     for options in by_flight.values()]
        compatible = []
        if shortlist and len(shortlist) == len(best):
            for assignment in product(*shortlist):
                spreads = np.ptp([[r[key] for key in ("ring_diameter_m", "ring_z_m", "ring_east_m", "ring_north_m")]
                                  for r in assignment], axis=0)
                if np.all(spreads <= (.75, .60, .60, .60)):
                    compatible.append(assignment)
        supported = compatible[0] if len(compatible) == 1 else ()
        if not group:
            status = "no_dated_lidar_reference"
        elif not best:
            status = "no_supported_circle"
        elif permit and not by_flight:
            status = "no_ring_matches_documented_diameter"
        elif not supported:
            status = "ambiguous_upper_geometry"
        else:
            status = "candidate_ring"
        usable = status == "candidate_ring"
        d = median(r["ring_diameter_m"] for r in supported) if usable else None
        z = median(r["ring_z_m"] for r in supported) if usable else None
        original = tank["lidar_upper_outline_diameter_m"]
        yield dict(candidate_id=candidate, terminal_id=tank["documented_terminal_id"], cohort_id=tank["cohort_id"], status=status,
                   selection_basis=basis, lidar_flight_lines=len(flight_keys), lines_with_ring=len(best),
                   compatible_ring_assignments=len(compatible), selected_flight_and_rank_json=json.dumps(
                       [[r["acquisition_date_utc"], r["point_source_id"], r["ring_rank"]] for r in supported] if usable else []),
                   original_diameter_m=float(original) if original else None, permit_diameter_m=permit,
                   ring_diameter_m=d, ring_z_m=z,
                   ring_east_m=median(r["ring_east_m"] for r in supported) if usable else None,
                   ring_north_m=median(r["ring_north_m"] for r in supported) if usable else None,
                   ring_minus_original_diameter_m=d-float(original) if usable and original else None,
                   ring_minus_permit_diameter_m=d-permit if usable and permit else None)


def main():
    registry = {r["candidate_id"]: r for r in read("tank_engineering_registry") if r["documented_roof_type"] == "EFR"}
    refs = [r for r in references() if r["candidate_id"] in registry]
    rows = list(fit_rings(refs, registry))
    contours = list(contour_rows(rows, registry))
    export(f"{STEM}_flight_lines", rows)
    export(f"{STEM}_contours", contours)
    documented = [r for r in contours if r["permit_diameter_m"] and r["ring_diameter_m"]]
    report = dict(documented_efr_contours=len(registry), dated_flight_line_references=len(refs), ring_rows=len(rows),
                  contour_status=Counter(r["status"] for r in contours),
                  contour_status_with_documented_diameter=Counter(r["status"] for r in contours if r["permit_diameter_m"]),
                  permit_constrained_diameter_comparisons=len(documented),
                  permit_constrained_mean_abs_diameter_difference_m=float(np.mean([abs(r["ring_minus_permit_diameter_m"]) for r in documented])) if documented else None,
                  original_diameter_mae_same_contours_m=float(np.mean([abs(r["original_diameter_m"]-r["permit_diameter_m"])
                                                                     for r in documented])) if documented else None)
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
