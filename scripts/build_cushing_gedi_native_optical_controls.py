"""Collect exact Sentinel-2 B04 roof measurements around native GEDI shots."""

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from statistics import median

from shapely import transform
from shapely.geometry import Point, shape
from shapely.ops import unary_union

from audit_s2_sun_clock import xml_time
from build_cushing_icesat2_controls import TO_GEO, identities
from build_cushing_icesat2_optical_controls import query_catalog
from build_l1c_quality import QUALITY_FIELDS, scene_rows
from build_s1_reference_matches import read
from build_weekly_shadow_panel import DERIVED
from collect_cushing_gedi_optical_windows import CONTEXT_M, fit_geometry, kernels
from collect_l1c_catalog_gaps import acquire_aws
from collect_public import RAW, ROOT, write, write_csv
from collect_s2_l1c import download, earliest_products, local_granule_files, parse_product
from train_shadow_quality import RESOLVED

STEM = "cushing_gedi_native_optical"
STORE = RAW / STEM
HOURS = 48


def references(tanks):
    ground = {r["shot_number"]: r for r in read("cushing_gedi_ground_controls")}
    shots = read("cushing_gedi_roof_waveforms_shots")
    if {r["shot_number"] for r in shots} != ground.keys() or len(shots) != len(ground):
        raise ValueError("Native GEDI waveforms and ground controls differ")
    selected = defaultdict(list)
    for r in shots:
        g = ground[r["shot_number"]]
        tank = tanks[r["candidate_id"]]
        stamp = datetime.fromisoformat(r["observed_at_utc"])
        if (any(g[k] != r[k] for k in ("candidate_id", "observed_at_utc", "gedi_acquisition_key", "beam"))
                or r["geometry_dataset"] != tank["geometry_dataset"]):
            raise ValueError(f"GEDI source identity or geometry changed on {r['shot_number']}")
        if (r["l2a_quality_flag_rel3"] != "1" or r["detected_waveform_peaks"] != "1"
                or r["footprint_inside_with_10m_margin"] != "True" or g["within_physical_wall_limits"] != "True"
                or tank["first_seen"] > stamp.date().isoformat()):
            continue
        selected[r["gedi_acquisition_key"]].append(dict(r, gap_m=float(g["wall_to_mode_ground_corrected_m"])))
    return shots, selected


def catalog(group, tanks):
    name, shots = group
    times = [r["observed_at_utc"] for r in shots]
    window = {"granule_start_utc": min(times), "granule_end_utc": max(times)}
    roofs = [Point(tanks[r["candidate_id"]]["east_m"], tanks[r["candidate_id"]]["north_m"])
             .buffer(tanks[r["candidate_id"]]["radius_m"] + CONTEXT_M) for r in shots]
    bounds = transform(unary_union(roofs), TO_GEO.transform, interleaved=False).bounds
    receipt, _ = query_catalog(name, {name: window}, store=STORE / "catalog", bounds=bounds)
    path = ROOT / receipt["earth_search_snapshot"]
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != receipt["earth_search_sha256"]:
        raise ValueError(f"Sentinel-2 catalog checksum differs for {name}")
    return receipt, json.loads(body)["features"]


def opportunities(groups, snapshots, tanks):
    features, links = {}, defaultdict(dict)
    for (name, shots), (_, candidates) in zip(groups.items(), snapshots, strict=True):
        for feature in candidates:
            props = feature["properties"]
            key = parse_product(props["s2:product_uri"])[0]
            footprint = shape(feature["geometry"])
            for r in shots:
                tank = tanks[r["candidate_id"]]
                stamp = datetime.fromisoformat(props["datetime"])
                distance = abs(stamp - datetime.fromisoformat(r["observed_at_utc"]))
                if distance > timedelta(hours=HOURS) or not footprint.covers(Point(tank["longitude"], tank["latitude"])):
                    continue
                choice = (props["created"], feature["id"])
                if key not in features or choice < features[key][0]:
                    features[key] = choice, feature
                links[key][r["shot_number"]] = r
    return features, links


def scene(key, feature, shots, tanks, versions):
    circles = {}
    for r in shots.values():
        tank = tanks[r["candidate_id"]]
        circles[r["candidate_id"]] = Point(tank["east_m"], tank["north_m"]).buffer(tank["radius_m"])
    footprint = unary_union([p.buffer(CONTEXT_M) for p in circles.values()])
    geographic = transform(footprint, TO_GEO.transform, interleaved=False)
    bounds = geographic.intersection(shape(feature["geometry"])).bounds
    out = STORE / "sentinel2"
    stamp = feature["properties"]["datetime"]
    directory = out / "granules" / stamp[:4] / feature["id"]
    if key in versions:
        route = "gcs"
        status = download(feature, versions, out, bounds, footprint, verify=True, pack_text=True)["status"]
    else:
        route = "aws"
        directory = out / "alternate_aws" / "granules" / stamp[:4] / feature["id"]
        version = feature["properties"]["s2:product_uri"]
        if (directory / "item.json").is_file():
            local_granule_files(directory, feature["id"], version)
        else:
            acquire_aws(feature, feature, bounds, footprint, out=out)
        status = "ready"
    manifest = directory / "item.json"
    if status != "ready":
        raise ValueError(f"Unavailable Sentinel-2 pixels for {feature['id']}: {status}")
    record = json.loads(manifest.read_bytes())
    if parse_product(record["product_name"])[0] != key:
        raise ValueError(f"Unexpected Sentinel-2 product for {feature['id']}")
    polygons = [(k, transform(p, TO_GEO.transform, interleaved=False)) for k, p in circles.items()]
    points = {k: (tanks[k]["longitude"], tanks[k]["latitude"]) for k in circles}
    observed = {v[0]: dict(zip(QUALITY_FIELDS, v, strict=True)) for v in scene_rows(manifest, polygons, points, {}, None)}
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}
    return dict(product_key="_".join(key), earth_search_item_id=feature["id"], archive=route,
                version_id=record["product_name"], manifest=str(manifest.relative_to(ROOT)),
                local_file_sha256_json=json.dumps(digests, sort_keys=True)), observed


def compare(sources, links, tanks):
    psf = kernels({r["version_id"][:3] for _, (r, _) in sources})
    for key, (source, optical) in sources:
        for shot in links[key].values():
            tank = tanks[shot["candidate_id"]]
            row = optical.get(shot["candidate_id"])
            time = datetime.fromisoformat(shot["observed_at_utc"])
            captured = xml_time(row["tile_metadata_path"]) if row else None
            hours = (captured - time).total_seconds() / 3600 if captured else None
            failure = "roof_outside_raster"
            if row:
                expected = row["optical_touched_cells_expected"]
                full = expected > 0 and row["optical_cells_valid_in_all_bands"] == expected
                clear = row["cloud_mask_basis"] != "missing" and not any(
                    row[k] for k in ("opaque_roof_cells", "cirrus_roof_cells", "snow_ice_roof_cells"))
                detector = row["red_detector_covered_roof_cells"] == expected and not row["red_detector_ambiguous_roof_cells"] and (
                    len(json.loads(row["red_detector_with_view_options_json"])) == 1)
                failure = next((reason for accepted, reason in (
                    (abs(hours) <= HOURS, "outside_48h_by_tile_clock"),
                    (full, "incomplete_valid_pixels"),
                    (clear, "source_cloud_mask_or_obscured_roof"),
                    (detector, "ambiguous_b04_detector"),
                ) if not accepted), None)
                fit = fit_geometry(row, tank, psf[source["version_id"][:3]]) if failure is None else {}
            else:
                full = clear = detector = False
                fit = {}
            estimate = fit.get("estimated_gap_m")
            yield dict(candidate_id=shot["candidate_id"], terminal_id=shot["terminal_id"],
                gedi_shot_number=shot["shot_number"], gedi_l1b_granule=shot["l1b_granule"],
                gedi_l2a_granule=shot["l2a_granule"], gedi_capture_utc=time.isoformat(),
                gedi_corrected_wall_to_roof_m=shot["gap_m"], geometry_dataset=shot["geometry_dataset"],
                sentinel2_version_id=source["version_id"], sentinel2_manifest=source["manifest"],
                sentinel2_tile_metadata=row["tile_metadata_path"] if row else None,
                sentinel2_capture_utc=captured.isoformat() if captured else None,
                sentinel2_minus_gedi_time_hours=hours, valid_roof_pixels=full, clear_roof=clear, b04_detector_resolved=detector,
                fit_status=fit.get("fit_status", failure), b04_wall_to_roof_m=estimate,
                b04_minus_gedi_m=estimate-shot["gap_m"] if estimate is not None else None,
                b04_profile_min_m=fit.get("near_optimal_gap_min_m"), b04_profile_max_m=fit.get("near_optimal_gap_max_m"))


def export(suffix, rows):
    fields = tuple(rows[0]) if rows else ("candidate_id", "gedi_shot_number", "sentinel2_version_id")
    write_csv(DERIVED / f"{STEM}_{suffix}.csv", fields, (tuple(r[k] for k in fields) for r in rows))


def laser_repeatability(groups):
    by_roof = defaultdict(list)
    for shots in groups.values():
        for r in shots:
            stamp = datetime.fromisoformat(r["observed_at_utc"])
            by_roof[r["candidate_id"]].append((stamp, stamp, "GEDI", r["shot_number"], r["gap_m"]))
    for r in read("cushing_icesat2_native_profiles"):
        if r["passes_roof_profile_checks"] == "True":
            start, end = (datetime.fromisoformat(r[k]) for k in ("first_roof_signal_photon_utc", "last_roof_signal_photon_utc"))
            by_roof[r["candidate_id"]].append((start, end, "ICESat-2", f"{r['filename']}:{r['beam']}",
                                               float(r["estimated_wall_to_roof_m"])))
    adjacent = []
    for tank, rows in by_roof.items():
        ordered = sorted(rows)
        for before, after in zip(ordered, ordered[1:]):
            if after[0] <= before[1]:
                raise ValueError(f"Overlapping independent laser controls for {tank}")
            elapsed = (after[0] - before[1]).total_seconds() / 86400
            adjacent.append(dict(candidate_id=tank, previous_sensor=before[2], current_sensor=after[2],
                previous_source=before[3], current_source=after[3], previous_start_utc=before[0].isoformat(),
                previous_end_utc=before[1].isoformat(), current_start_utc=after[0].isoformat(),
                current_end_utc=after[1].isoformat(), minimum_days_between=elapsed,
                previous_gap_m=before[4], current_gap_m=after[4]))
    export("adjacent_laser_profiles", sorted(adjacent, key=lambda r: (r["minimum_days_between"], r["candidate_id"])))
    return dict(qualified_gedi_and_icesat2_profiles=sum(map(len, by_roof.values())), distinct_laser_roofs=len(by_roof),
                adjacent_pairs=len(adjacent), shortest_gap_days=min(r["minimum_days_between"] for r in adjacent),
                adjacent_pairs_within_14_days=sum(r["minimum_days_between"] <= 14 for r in adjacent))


def main():
    tanks, _, _ = identities()
    shots, groups = references(tanks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        snapshots = list(pool.map(lambda g: catalog(g, tanks), groups.items()))
    features, links = opportunities(groups, snapshots, tanks)
    versions = earliest_products([feature for _, feature in features.values()], 3)
    with ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(lambda k: scene(k, features[k][1], links[k], tanks, versions), sorted(features)))
    sources = list(zip(sorted(features), completed, strict=True))
    rows = list(compare(sources, links, tanks))
    selected = {}
    for r in sorted((r for r in rows if r["fit_status"] == RESOLVED),
                    key=lambda r: (abs(r["sentinel2_minus_gedi_time_hours"]), r["sentinel2_version_id"])):
        selected.setdefault(r["gedi_shot_number"], r)
    old = {r["shot_number"] for r in read("cushing_gedi_waveform_controls_optical")
           if r["native_footprint_guard_pass"] == "True" and r["native_local_ground_bias_m"]}
    for suffix, data in (("catalog", [r for r, _ in snapshots]), ("sources", [r for r, _ in completed]),
                         ("fits", rows), ("nearest", list(selected.values()))):
        export(suffix, data)
    within_day = [r for r in selected.values() if abs(r["sentinel2_minus_gedi_time_hours"]) <= 24]
    summary = dict(native_waveform_shots=len(shots), quality_and_local_ground_pass=sum(map(len, groups.values())),
        compared_shots=len({r["gedi_shot_number"] for r in rows}), unique_granules=len(sources),
        source_archives=Counter(r["archive"] for r, _ in completed), fit_status=Counter(r["fit_status"] for r in rows),
        passing_nearest_within_48h=len(selected), passing_nearest_within_24h=len(within_day),
        newly_matched_shots=len(selected.keys() - old),
        median_abs_nearest_difference_m=median(abs(r["b04_minus_gedi_m"]) for r in selected.values()) if selected else None,
        median_abs_within_24h_difference_m=median(abs(r["b04_minus_gedi_m"]) for r in within_day) if within_day else None,
        combined_spaceborne_laser_repeats=laser_repeatability(groups),
        scope="Retrospective direct comparison on dated roof geometry. GEDI uses guarded native waveforms and local terrain alignment. "
              "Differences include actual motion between acquisitions. No inventory targets or prices used.")
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
