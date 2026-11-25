"""Measure circular edges on archived 15 cm aerial tiles and compare LiDAR alternatives."""

import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from build_cushing_lidar_shell_rings import STEM as LIDAR
from build_naip_2019_roof_reference import native_crop, project
from build_ok_aerial_2025_roof_controls import TO_STATE, read_crop, rgb_tiles
from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write

STEM = "cushing_aerial_shell_rings"
ANGLES = np.arange(144)*np.pi/72
DIRECTIONS = np.stack((np.sin(ANGLES), np.cos(ANGLES)))
SPACING = .15
MIN_COVERAGE = .98
STRENGTH_FRACTIONS = (.55, .65, .75, .85)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def angular_edges(gray, affine, centre, radii, half_span):
    positions = centre[:, :, None, None] + radii[None, None, :, None]*DIRECTIONS[None, :, None, :]
    col, row = (~affine)*(positions[:, 0], positions[:, 1])
    samples = map_coordinates(gray, [row-.5, col-.5], order=1, prefilter=False, mode="constant", cval=np.nan)
    # Compare brightness on both sides of each radius over 0.60 m.
    stride = 2*half_span
    change = np.abs(samples[:, stride:]-samples[:, :-stride])
    by_sector = change.reshape(len(centre), -1, 8, 18)
    valid = np.isfinite(by_sector).sum(axis=-1) >= 14
    safe = np.where(np.isfinite(by_sector), by_sector, 0.)
    sector = np.where(valid, np.median(safe, axis=-1), 0.)
    return np.quantile(sector, .25, axis=-1), sector, valid


def measure(gray, affine, east, north, radius, *, spacing=SPACING):
    axis = np.arange(-6, 6.01, 1.)
    x, y = np.meshgrid(axis, axis)
    offsets = np.column_stack((x.ravel(), y.ravel()))
    offsets = offsets[np.linalg.norm(offsets, axis=1) <= 6]
    centre = np.array([east, north])
    radii = radius+np.arange(-6., 3.16, spacing)
    half_span = round(.30/spacing)
    strength, _, _ = angular_edges(gray, affine, centre+offsets, radii, half_span)
    i, j = np.unravel_index(strength.argmax(), strength.shape)
    seed = centre+offsets[i]
    small = np.arange(-.8, .81, .20)
    x, y = np.meshgrid(small, small)
    trials = seed+np.column_stack((x.ravel(), y.ravel()))
    strength, sectors, valid = angular_edges(gray, affine, trials, radii, half_span)
    i, _ = np.unravel_index(strength.argmax(), strength.shape)
    # A single translated centre is held fixed across all radial alternatives.
    profile = strength[i]
    peaks, _ = find_peaks(profile, distance=3, prominence=2.0)
    if not len(peaks):
        return []
    found = []
    for j in sorted(peaks, key=lambda j: (-profile[j], j)):
        if not valid[i, j].all() or profile[j] < 8:
            continue
        q = sectors[i, j]
        found.append(dict(aerial_edge_radius_m=float(radii[j+half_span]), aerial_edge_p25_dn=float(profile[j]),
                          aerial_edge_sector_medians_dn_json=json.dumps(q.tolist()),
                          aerial_east_m=float(trials[i, 0]), aerial_north_m=float(trials[i, 1]),
                          aerial_translation_m=float(np.linalg.norm(trials[i]-centre))))
    return found


def centre_and_radius(tank):
    x, y = (float(tank[f"footprint_{k}_m"]) for k in ("east", "north"))
    diameter = tank["lidar_upper_outline_diameter_m"] or tank["footprint_diameter_m"]
    return *tank.get("outline_centre", (x, y)), float(diameter)/2


def evidence(registry, contours):
    coverage = defaultdict(list)
    for r in read("ok_aerial_tank_image_coverage"):
        if r["candidate_id"] in contours:
            coverage[r["candidate_id"]].append(r)
    tiles = rgb_tiles()
    verified = {}
    with ExitStack() as stack:
        sources = {}
        for candidate, r in sorted(contours.items()):
            t = registry[candidate]
            primary = max(coverage[candidate], key=lambda x: (int(x["valid_pixels"]), x["tile_id"]))
            day, clock = primary["tile_majority_date"], primary["tile_time_raw"]
            east, north, radius = centre_and_radius(t)
            options = tiles[day, clock]
            # Only same-day, same-clock tiles contribute to each raster window.
            x, y = TO_STATE.transform(east, north)
            extent = radius+16
            used = []
            for identity, path in options:
                if path not in sources:
                    sources[path] = stack.enter_context(rasterio.open(path))
                left, bottom, right, top = sources[path].bounds
                if left < x+extent and right > x-extent and bottom < y+extent and top > y-extent:
                    used.append((identity, path))
                    if path not in verified:
                        verified[path] = digest(path)
            crop = read_crop(stack, used, sources, dict(east=east, north=north, radius=radius), 16.)
            support = float(crop[-1]) if crop else 0.
            base = dict(candidate_id=candidate, terminal_id=r["terminal_id"], cohort_id=r["cohort_id"], lidar_status=r["status"],
                        aerial_day=day, aerial_tile_nominal_time_utc=clock, aerial_primary_tile=primary["tile_id"],
                        source_paths_sha256_json=json.dumps({str(p.relative_to(ROOT)): verified[p] for _, p in used}),
                        reference_radius_m=radius, crop_valid_fraction=support, permit_diameter_m=r["permit_diameter_m"])
            edges = measure(crop[0], crop[4], east, north, radius) if support >= MIN_COVERAGE else []
            yield base, edges


def historical_evidence(registry, contours):
    by_candidate = defaultdict(list)
    for r in read("naip_tank_source_clock"):
        if r["acquisition_year"] == "2023" and r["candidate_id"] in contours and float(r["contour_fraction"]) >= .98:
            by_candidate[r["candidate_id"]].append(r)
    verified = {}
    with ExitStack() as stack:
        sources = {}
        for candidate, rows in sorted(by_candidate.items()):
            east, north, radius = centre_and_radius(registry[candidate])
            clock = min(rows, key=lambda r: (-float(r["contour_fraction"]), r["source_polygon_id"]))
            options = []
            for path in json.loads(clock["dated_asset_paths_json"]):
                if path not in sources:
                    sources[path] = stack.enter_context(rasterio.open(ROOT/path))
                source = sources[path]
                if source.crs.to_epsg() != 26914 or source.res != (.3, .3):
                    raise ValueError(f"Unexpected NAIP 2023 CRS or spacing for {path}")
                x, y = project("EPSG:32614", source.crs)(east, north)
                crop = native_crop(source, x, y, radius)
                if crop:
                    options.append((crop[-1], path, crop, x, y))
            if not options:
                continue
            coverage, path, (gray, _, _, _, affine, _), x, y = max(options, key=lambda r: (r[0], r[1]))
            if path not in verified:
                verified[path] = digest(ROOT/path)
            edges = measure(gray, affine, x, y, radius, spacing=.30) if coverage >= MIN_COVERAGE else []
            for edge in edges:
                edge["aerial_east_m"], edge["aerial_north_m"] = project("EPSG:26914", "EPSG:32614")(
                    edge["aerial_east_m"], edge["aerial_north_m"])
            r = contours[candidate]
            base = dict(candidate_id=candidate, terminal_id=r["terminal_id"], cohort_id=r["cohort_id"], lidar_status=r["status"],
                        aerial_day=clock["source_acquisition_date"], aerial_tile_nominal_time_utc=None,
                        aerial_primary_tile=Path(path).parent.name, source_paths_sha256_json=json.dumps({path: verified[path]}),
                        reference_radius_m=radius, crop_valid_fraction=float(coverage), permit_diameter_m=r["permit_diameter_m"])
            yield base, edges


def compare(base, edges, alternatives):
    groups = defaultdict(list)
    for r in alternatives:
        groups[r["point_source_id"]].append(r)
    out = []
    for rank, edge in enumerate(edges, 1):
        radius = edge["aerial_edge_radius_m"]
        for rows in groups.values():
            for ring in rows:
                delta = 2*radius-float(ring["ring_diameter_m"])
                out.append(base | edge | dict(aerial_edge_rank=rank, lidar_point_source_id=ring["point_source_id"],
                    lidar_acquisition_date_utc=ring["acquisition_date_utc"],
                    lidar_ring_rank=int(ring["ring_rank"]), lidar_ring_diameter_m=float(ring["ring_diameter_m"]),
                    lidar_ring_elevation_m=float(ring["ring_z_m"]), lidar_ring_east_m=float(ring["ring_east_m"]),
                    lidar_ring_north_m=float(ring["ring_north_m"]), aerial_minus_lidar_diameter_m=delta))
    return out


def agreement(base, matches, lidar_contour):
    def flight(r):
        return r["lidar_acquisition_date_utc"], r["lidar_point_source_id"]

    max_strength = max((r["aerial_edge_p25_dn"] for r in matches), default=0.)
    flights = {flight(r) for r in matches}
    results, details = [], []
    for fraction in STRENGTH_FRACTIONS:
        by_flight = defaultdict(dict)
        for r in matches:
            diameter = r["lidar_ring_diameter_m"]
            if (r["aerial_edge_p25_dn"] >= max(35, fraction*max_strength)
                    and abs(r["aerial_minus_lidar_diameter_m"]) <= max(.6, .01*diameter)):
                by_flight[flight(r)].setdefault(r["lidar_ring_rank"], r)
        one_per_flight = bool(by_flight) and by_flight.keys() == flights and all(len(r) == 1 for r in by_flight.values())
        selected = [next(iter(by_flight[f].values())) for f in sorted(by_flight)] if one_per_flight else []
        if selected:
            spread = np.ptp([[r[f"lidar_ring_{field}_m"] for field in ("diameter", "elevation", "east", "north")]
                             for r in selected], axis=0)
            if np.any(spread > (.75, .60, .60, .60)):
                selected = []
        keys = [(*flight(r), r["lidar_ring_rank"]) for r in selected]
        details.append(dict(relative_strength_threshold=fraction, candidate_rings_by_flight=[
            [*key, sorted(options)] for key, options in sorted(by_flight.items())], cross_line_consistent=bool(selected)))
        results.append((keys, selected))
    stable = results[0][1] if results[0][0] and all(key == results[0][0] for key, _ in results) else []
    permit = float(base["permit_diameter_m"]) if base["permit_diameter_m"] else None
    diameter_conflict = bool(stable and permit and any(abs(r["lidar_ring_diameter_m"]-permit) > max(.6, .01*permit) for r in stable))
    prior = [tuple(k) for k in json.loads(lidar_contour["selected_flight_and_rank_json"])]
    new = results[0][0] if stable else []
    if diameter_conflict:
        status = "conflicts_with_documented_diameter"
    elif stable and prior and sorted(prior) != new:
        status = "conflicts_with_lidar_selection"
    elif stable:
        status = "one_cross_sensor_geometry_candidate"
    elif not matches:
        status = "no_aerial_lidar_pair"
    else:
        status = "unresolved_aerial_lidar_pair"
    return base | dict(status=status, lidar_lines_with_ring_candidates=len(flights),
                       sensitivity_json=json.dumps(details, separators=(",", ":")),
                       stable_flights_and_ranks_json=json.dumps(new),
                       stable_median_diameter_m=float(np.median([r["lidar_ring_diameter_m"] for r in stable])) if stable else None,
                       stable_aerial_edge_ranks_json=json.dumps([r["aerial_edge_rank"] for r in stable]),
                       stable_aerial_minus_lidar_diameter_max_abs_m=max(abs(r["aerial_minus_lidar_diameter_m"]) for r in stable) if stable else None,
                       confirmed_shadow_casting_shell=False)


def run(records, contours, lidar, stem):
    ring_rows, edge_rows, checks, pairs = [], [], [], []
    for i, (base, edges) in enumerate(records, 1):
        matches = compare(base, edges, lidar[base["candidate_id"]])
        ring_rows.extend(matches)
        edge_rows.extend(base | edge | dict(aerial_edge_rank=j) for j, edge in enumerate(edges, 1))
        checks.append(base | dict(aerial_edges=len(edges), lidar_alternatives=len(lidar[base["candidate_id"]])))
        pairs.append(agreement(base, matches, contours[base["candidate_id"]]))
        if i % 50 == 0:
            print(f"{stem}: {i} orthophoto windows", flush=True)
    for suffix, rows in (("coverage", checks), ("edges", edge_rows), ("matches", ring_rows), ("comparisons", pairs)):
        export(f"{stem}_{suffix}", rows)
    report = dict(contours=len(checks), complete_aerial_windows=sum(r["crop_valid_fraction"] >= MIN_COVERAGE for r in checks),
                  with_measured_aerial_edges=sum(r["aerial_edges"] > 0 for r in checks), aerial_edges=len(edge_rows), lidar_matches=len(ring_rows),
                  comparison_status=Counter(r["status"] for r in pairs),
                  previously_ambiguous_with_one_geometry_candidate=sum(r["status"] == "one_cross_sensor_geometry_candidate"
                      and r["lidar_status"] == "ambiguous_upper_geometry" for r in pairs),
                  documented_diameter_status=Counter(r["status"] for r in pairs if r["permit_diameter_m"]))
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="")
    return {r["candidate_id"]: r for r in pairs}


def two_campaigns(recent, early):
    accepted = "one_cross_sensor_geometry_candidate"
    for k in sorted(recent):
        a, b = recent[k], early.get(k)
        same = bool(b and a["status"] == b["status"] == accepted
                    and a["stable_flights_and_ranks_json"] == b["stable_flights_and_ranks_json"])
        yield dict(candidate_id=k, terminal_id=a["terminal_id"], cohort_id=a["cohort_id"], lidar_status=a["lidar_status"],
                   ortho_2025_status=a["status"], naip_2023_status=b["status"] if b else "no_matched_2023_coverage",
                   ortho_2025_flights_and_ranks_json=a["stable_flights_and_ranks_json"],
                   naip_2023_flights_and_ranks_json=b["stable_flights_and_ranks_json"] if b else None,
                   same_lidar_candidate_both_campaigns=same,
                   matched_diameter_m=a["stable_median_diameter_m"] if same else None,
                   confirmed_shadow_casting_shell=False)


def main():

    contours = {r["candidate_id"]: r for r in read(f"{LIDAR}_contours")}
    registry = {r["candidate_id"]: r for r in read("tank_engineering_registry")
                if r["candidate_id"] in contours}
    for r in read("tank_lidar_outlines"):
        if r["candidate_id"] in registry and r["diameter_m"]:
            registry[r["candidate_id"]]["outline_centre"] = tuple(float(r[k]) for k in ("center_east_m", "center_north_m"))
    lidar = defaultdict(list)
    for r in read(f"{LIDAR}_flight_lines"):
        if r["ring_rank"]:
            lidar[r["candidate_id"]].append(r)
    recent = run(evidence(registry, contours), contours, lidar, STEM)
    early = run(historical_evidence(registry, contours), contours, lidar, f"{STEM}_naip_2023")
    joined = list(two_campaigns(recent, early))
    export(f"{STEM}_two_campaigns", joined)
    print(dict(same_lidar_candidate_both_campaigns=Counter(r["lidar_status"] for r in joined
               if r["same_lidar_candidate_both_campaigns"])), flush=True)


if __name__ == "__main__":
    main()
