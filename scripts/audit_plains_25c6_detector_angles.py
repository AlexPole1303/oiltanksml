"""Recover inspection-window viewing angles from the original same-detector grids."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from itertools import combinations

import numpy as np

from audit_rose_rock_registered_changes import PASS, consensus, exterior_offsets
from audit_sentinel2_alternate_registration import packet_inputs
from build_cushing_okeis_plains_control import STEM as PLAINS
from build_registered_shadow_development import registered_pair
from build_s1_reference_matches import DERIVED, export, read
from build_sentinel2_angles import array, interpolate, stencil
from build_sentinel2_wider_registration import CHECKED, OFFSETS
from build_shadow_registration_inputs import describe_source
from build_terminal_shadow_inputs import mask_detector
from build_weekly_shadow_panel import RESOLVED, read_only
from check_multiband_shadow import tile_metadata
from check_weekly_multiband_shadow import METHODS, spectral_kernels, spectral_profiles
from collect_public import ROOT, write
from collect_s2_l1c import source_metadata_bytes
from collect_s2_spectral_detector_masks import BANDS, collect, jobs
from fit_shadow_geometry import bearing_vector, estimate, gap_fit_status

STEM = f"{PLAINS}_detector_recovery"
MESH_M = 5000
RADIUS_CELLS = 2.1


def circular_mean(angles, weights):
    directions = np.radians(angles)
    return math.degrees(math.atan2(weights @ np.sin(directions), weights @ np.cos(directions))) % 360


def triangle_angles(zenith, azimuth, x, y):
    nodes = np.argwhere(np.isfinite(zenith) & np.isfinite(azimuth))
    distance = np.hypot(nodes[:, 1]-x, nodes[:, 0]-y)
    nearby = nodes[distance <= RADIUS_CELLS]
    candidates = []
    for trial in combinations(nearby, 3):
        vertices = np.asarray(trial)
        delta = vertices[:, ::-1] - [x, y]
        design = np.vstack((delta.T, np.ones(3)))
        if abs(np.linalg.det(design)) < 1e-9:
            continue
        weights = np.linalg.solve(design, [0, 0, 1])
        if weights.min() < -1e-9:
            continue
        indices = tuple(vertices.T)
        angles = float(weights @ zenith[indices]), circular_mean(azimuth[indices], weights)
        distances = np.linalg.norm(delta, axis=1)
        # Prefer local triangles by their farthest node, then weighted squared distance.
        key = float(distances.max()), float(weights @ distances**2), tuple(vertices.ravel())
        candidates.append((key, vertices, weights, angles))
    return sorted(candidates, key=lambda r: r[0])


def angle_evidence(grid, east, north, left, top):
    x, y = (east-left)/MESH_M, (top-north)/MESH_M
    w = stencil(np.array([x]), np.array([y]))
    z, a = (array(grid, name) for name in ("Zenith", "Azimuth"))
    zenith = float(interpolate(grid, "Zenith", w)[0])
    azimuth = float(interpolate(grid, "Azimuth", w, True)[0])
    if np.isfinite([zenith, azimuth]).all():
        nodes = [(int(r), int(c)) for r, c in zip(w[0][:, 0], w[1][:, 0], strict=True)]
        return dict(method="source_bilinear", zenith_deg=zenith, azimuth_true_deg=azimuth, candidate_triangles=0,
                    nodes_json=json.dumps(nodes), weights_json=json.dumps(w[2][:, 0].tolist()),
                    selected_node_angle_pairs_json=json.dumps([[float(z[r, c]), float(a[r, c])] for r, c in nodes]),
                    farthest_selected_node_m=MESH_M*max(math.hypot(c-x, r-y) for r, c in nodes),
                    maximum_alternative_horizontal_shift_for_15m_m=0.)
    trials = triangle_angles(z, a, x, y)
    if not trials:
        raise ValueError("The roof centre is outside the supported local detector triangles")
    key, nodes, weights, selected = trials[0]
    v = bearing_vector(*selected)
    span = max(float(np.linalg.norm(bearing_vector(*angles)-v)) for _, _, _, angles in trials)
    return dict(method="local_triangle_interpolation", zenith_deg=selected[0], azimuth_true_deg=selected[1],
                candidate_triangles=len(trials), nodes_json=json.dumps(nodes.tolist()), weights_json=json.dumps(weights.tolist()),
                selected_node_angle_pairs_json=json.dumps([[float(z[r, c]), float(a[r, c])] for r, c in nodes]),
                farthest_selected_node_m=key[0]*MESH_M, maximum_alternative_horizontal_shift_for_15m_m=15*span)


def node_validation(scenes, angles):
    rows = []
    numbers = {"B03": 2, "B04": 3, "B08": 7}
    fields = {(r["version_id"], r["band"], r["detector_id"]) for r in angles}
    scene_by_version = {r["version_id"]: r for r in scenes.values()}
    for version, band, detector in sorted(fields):
        scene = scene_by_version[version]
        root = tile_metadata(scene["tile_metadata_path"])
        grid = root.find(f".//Viewing_Incidence_Angles_Grids[@bandId='{numbers[band]}'][@detectorId='{detector}']")
        z, a = (array(grid, name) for name in ("Zenith", "Azimuth"))
        for y, x in np.argwhere(np.isfinite(z) & np.isfinite(a)):
            saved = float(z[y, x]), float(a[y, x])
            hidden = z.copy()
            hidden[y, x] = np.nan
            trials = triangle_angles(hidden, a, x, y)
            if trials:
                estimated = trials[0][-1]
                shift = 15 * float(np.linalg.norm(bearing_vector(*saved)-bearing_vector(*estimated)))
                rows.append(dict(version_id=version, band=band, detector_id=detector, row=int(y), column=int(x),
                                 source_zenith_deg=saved[0], source_azimuth_deg=saved[1], held_out_zenith_deg=estimated[0],
                                 held_out_azimuth_deg=estimated[1], horizontal_shift_for_15m_m=shift))
    return rows


def recovered_profiles(path, observations, scenes, masks, geometry, reference_gaps, *, originals=None):
    fits, angles, profiles = [], [], {}
    with closing(read_only(path)) as db:
        kernels = spectral_kernels(db)
        detectors = {i: mask_detector(db, i) for i in observations}
    for i, original in sorted(observations.items()):
        roof, version = original["candidate_id"], original["version_id"]
        scene = scenes[original["scene_id"]]
        if originals is not None and (originals[roof, version]["fit_status"] != "incomplete_optical_geometry_or_quality"
                                      or original["resolved_b04_detector"]):
            raise ValueError(f"Unexpected unrecovered reference status for {roof}/{version}")
        detector = detectors[i]
        if detector is None or not original["roof_clear_by_source_mask"] or (
                original["roof_cells_expected"] != original["roof_cells_valid_all_bands"]):
            raise ValueError(f"Incomplete native B04 pixels or nonunique detector for {roof}/{version}")
        g = geometry[roof]
        x, y = (float(g[k]) for k in ("center_east_m", "center_north_m"))
        radius = float(g["diameter_m"])/2
        shell = float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"])
        root = tile_metadata(scene["tile_metadata_path"])
        tile = root.find(".//Tile_Geocoding/Geoposition[@resolution='10']")
        left, top = float(tile.findtext("ULX")), float(tile.findtext("ULY"))
        digest = hashlib.sha256(source_metadata_bytes(ROOT / scene["tile_metadata_path"])).hexdigest()
        metadata = {}

        def view_grid(band, det, grid, east, north, left, top):
            evidence = angle_evidence(grid, east, north, left, top)
            metadata[band] = dict(candidate_id=roof, version_id=version, band=band, detector_id=det, **evidence,
                                  tile_metadata_file=scene["tile_metadata_path"], tile_metadata_sha256=digest)
            return evidence["zenith_deg"], evidence["azimuth_true_deg"]

        red = root.find(f".//Viewing_Incidence_Angles_Grids[@bandId='3'][@detectorId='{detector}']")
        zenith, azimuth = view_grid("B04", detector, red, x, y, left, top)
        row = dict(original, center_east_m=x, center_north_m=y, radius_m=radius, resolved_b04_detector=1,
                   red_detector_id=detector, red_view_zenith_deg=zenith,
                   red_view_azimuth_grid_deg=(azimuth + original["true_north_grid_bearing_deg"]) % 360)
        file_list = tuple(masks[version, band] for band in BANDS)
        status, count, curves, _ = spectral_profiles(path, row, scene, shell, kernels, file_list, 0, 0, view_grid=view_grid)
        fits.extend(fit_rows(row, status, count, curves, estimate(row, shell, kernels[row["platform"], "B04"], calibration_db=path),
                             reference_gaps[roof]))
        angles.extend(metadata.values())
        profiles[roof, version] = (row, scene, shell, file_list)
    return fits, angles, profiles, kernels


def fit_rows(row, status, count, profiles, red, nominal_gap):
    for method in METHODS:
        gap = lower = upper = trial_gap = loss_ratio = None
        diagnosis = status
        if profiles:
            _, gaps, loss, reverse, cells, weight = profiles[method]
            score = loss / weight
            _, index = np.unravel_index(score.argmin(), score.shape)
            minimum = score.min()
            trial = int(np.argmin(abs(gaps-nominal_gap)))
            trial_gap = float(gaps[trial])
            loss_ratio = float(score[:, trial].min()/minimum) if minimum > 0 else None
            near = gaps[score.min(axis=0) <= minimum*(1+1/max(1, cells-6)) + 1e-10/weight]
            gap, lower, upper = map(float, (gaps[index], near.min(), near.max()))
            diagnosis = gap_fit_status(gaps, gap, lower, upper, math.sqrt(minimum), math.sqrt(score[:, 0].min()),
                                       math.sqrt(reverse.min()/weight))
        if method == "red" and status == "compared_full_pixels" and (gap, diagnosis) != (red.get("estimated_gap_m"), red["fit_status"]):
            raise ValueError(f"Full-pixel multispectral B04 differs from the native refit for {row['candidate_id']}")
        yield dict(candidate_id=row["candidate_id"], version_id=row["version_id"], capture_at_utc=row["captured_at_utc"],
                   method=method, pixel_status=status, common_pixels=count, fit_status=diagnosis, gap_m=gap,
                   near_min_m=lower, near_max_m=upper, conditional_operator_nominal_gap_m=nominal_gap,
                   closest_model_grid_gap_m=trial_gap, conditional_trial_to_best_loss_ratio=loss_ratio,
                   admitted_synchronized_ground_label=False)


def paired_fits(path, scenes, profiles, kernels):
    by_tile = defaultdict(list)
    for i, scene in enumerate(sorted(scenes.values(), key=lambda r: r["captured_at_utc"]), 1):
        entry = describe_source(i, scene)
        by_tile["T"+entry["mgrs_tile"]].append(entry)
    links = []
    for tile, (first, last) in sorted(by_tile.items()):
        for roof in sorted({k[0] for k in profiles}):
            label = "/".join((first["captured_at_utc"][:10], last["captured_at_utc"][:10], tile))
            links.append(("plains_ground_level", "retrospective", label, roof, "EFR", "cushing_connect", "", "", "", first, last))
    offsets, patches = exterior_offsets(links)

    def view_grid(_, __, grid, x, y, left, top):
        r = angle_evidence(grid, x, y, left, top)
        return r["zenith_deg"], r["azimuth_true_deg"]

    results = []
    for r in offsets:
        earlier, later, tile = r["week_or_control_day"].split("/")
        roof = r["candidate_id"]
        checked = r["registration_status"] == CHECKED
        dx, dy = (float(r[k])/2 for k in OFFSETS) if checked else (0., 0.)
        row = dict(candidate_id=roof, earlier_day=earlier, later_day=later, tile=tile, registration_status=r["registration_status"])
        endpoints = []
        if checked:
            for side, sign in (("earlier", -1), ("later", 1)):
                o, scene, shell, files = profiles[roof, r[f"{side}_version"]]
                _, _, fit, _ = spectral_profiles(path, o, scene, shell, kernels, files, sign*dx, sign*dy, view_grid=view_grid)
                endpoints.append(fit)
        for method in METHODS:
            result = registered_pair(*(x[method] for x in endpoints)) if endpoints and all(endpoints) else {}
            row.update({f"{method}_{k}": result.get(k) for k in (PASS, "registered_change_m", "earlier_registered_gap_m",
                         "later_registered_gap_m", "earlier_registered_status", "later_registered_status",
                         "registered_loss_range_min_m", "registered_loss_range_max_m")})
        results.append(row)
    return offsets, patches, results


def main():
    packet, observations, scenes, _ = packet_inputs(f"{PLAINS}_sentinel_native.sqlite")
    record, = read(f"{PLAINS}_ground_record")
    nominal_gap = float(record["operator_shell_height_ft"])*.3048 - float(record["liquid_level_m"])
    originals = {(r["candidate_id"], r["version_id"]): r for r in read(f"{PLAINS}_sentinel_fits")}
    geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry") if r["candidate_id"] in {k[0] for k in originals}}
    with ThreadPoolExecutor(max_workers=4) as pool:
        masks = list(pool.map(collect, jobs({r["version_id"]: r["manifest_path"] for r in scenes.values()})))
    files = {(r["version_id"], r["band"]): r["local_file"] for r in masks}
    fits, angles, profiles, kernels = recovered_profiles(packet, observations, scenes, files, geometry,
                                                         dict.fromkeys(geometry, nominal_gap), originals=originals)
    validation = node_validation(scenes, angles)
    offsets, patches, pairs = paired_fits(packet, scenes, profiles, kernels)
    consensus_rows = [{k: v for k, v in r.items() if k != "admitted_nrc_inventory_label"} for r in consensus(pairs)]
    for suffix, rows in (("angle_nodes", angles), ("held_out_nodes", validation), ("masks", masks), ("single_scene_fits", fits),
                         ("registration", offsets), ("registration_patches", patches), ("paired_fits", pairs), ("two_tile_changes", consensus_rows)):
        export(f"{STEM}_{suffix}", rows)
    passed = [r for r in fits if r["fit_status"] == RESOLVED]
    ratios = [r["conditional_trial_to_best_loss_ratio"] for r in passed if r["conditional_trial_to_best_loss_ratio"] is not None]
    report = dict(original_roof_windows=len(observations), recovered_b04_angles=sum(r["band"] == "B04" for r in angles),
                  angle_methods=Counter(r["method"] for r in angles),
                  maximum_alternative_triangle_horizontal_shift_for_15m_m=max(r["maximum_alternative_horizontal_shift_for_15m_m"] for r in angles),
                  held_out_nodes=len(validation), maximum_held_out_horizontal_shift_for_15m_m=max(r["horizontal_shift_for_15m_m"] for r in validation),
                  single_scene_passes=Counter(r["method"] for r in passed),
                  conditional_operator_trial_to_optical_best_loss_ratio_range=[min(ratios), max(ratios)] if ratios else None,
                  registered_roof_tile_pairs=sum(r["registration_status"] == CHECKED for r in offsets),
                  roof_pairs_passing_all_methods_and_tiles=sum(r["all_six_fits_passed"] for r in consensus_rows),
                  admitted_synchronized_ground_labels=0)
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
