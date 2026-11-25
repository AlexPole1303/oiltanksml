"""Fit a common roof height across same-pass B04 tiles with duplicate-pixel weights."""

import argparse
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime
from statistics import mean, median
from zlib import crc32

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from audit_cushing_weekly_change_controls import aerial
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_calibration import CONTEXT_M
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_panel import RESOLVED, SOURCE, build_pixels
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write
from collect_s2_l1c import parse_product
from fit_shadow_geometry import CALIBRATION, bearing_vector, gap_fit_status, gap_trials, inputs, pixel_grid, read_only, response_profiles

STEM = "cushing_overlapping_tile_fit"
MAX_AERIAL_HOURS = 36.
METHODS = ("primary", "joint", "equal_average")


def overpass(version):
    ((platform, take, orbit, _), baseline, _) = parse_product(version)
    return platform, take, orbit, baseline


def aerial_inputs(geometry):
    groups = defaultdict(list)
    for r in read("aerialok_native_optical_pairs"):
        if (r["candidate_id"] in geometry and r["satellite_sensor"].startswith("Sentinel-2 B04")
                and r["aerial_fit_status"] == "visible_shadow_fit" and r["all_near_clock_fits_visible"] == "True"
                and r["maximum_inferred_separation_hours"] and float(r["maximum_inferred_separation_hours"]) <= MAX_AERIAL_HOURS):
            key = r["candidate_id"], r["flight_date"], overpass(r["satellite_version"])
            groups[key].append(r)
    references, expected = {}, set()
    for (tank, day, capture), variants in sorted(groups.items()):
        versions = tuple(sorted({r["satellite_version"] for r in variants}))
        if len({v.split("_")[5] for v in versions}) != 2:
            continue
        if len(versions) != 2 or len({r["aerial_gap_m"] for r in variants}) != 1:
            raise ValueError(f"Conflicting versions or control height on {tank}/{day}")
        references[tank, day, capture] = (float(variants[0]["aerial_gap_m"]),
                                          max(float(r["maximum_inferred_separation_hours"]) for r in variants),
                                          variants[0]["terminal_id"], variants[0]["aerial_sha256"])
        expected.update((tank, v) for v in versions)
    return references, expected


def stage_aerial(expected, packet):
    by_version = defaultdict(set)
    for tank, version in expected:
        by_version[version].add(tank)
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        selections = [dict(row) for version, roofs in by_version.items() for row in db.execute(
            """SELECT candidate_id, l1c_product_name AS version_id, scene_item_id AS canonical_l2a_id
                FROM sentinel2_l1c_tank_quality WHERE l1c_product_name=?""", (version,)) if row["candidate_id"] in roofs]
        if len(selections) != len(expected) or {(r["candidate_id"], r["version_id"]) for r in selections} != expected:
            raise ValueError("The aerial controls do not map to exact native tiles")
        build_pixels(db, [{"endpoint": dict(r, study_phase="development")} for r in selections], packet, phase="development")


def coords(grid):
    x = grid[2][0] + 10 * grid[4]
    y = grid[2][1] - 10 * grid[3]
    return [(int(a), int(b)) for a, b in zip(np.rint(x * 1000), np.rint(y * 1000), strict=True)]


def render(o, grid, shell, kernel):
    gaps = gap_trials(shell, "EFR")
    sun, view = (bearing_vector(o[f"{k}_zenith_deg"], o[f"{k}_azimuth_grid_deg"]) for k in ("sun", "red_view"))
    _, loss, _, reverse = response_profiles(grid, o["radius_m"], shell if shell >= 5 else 0., gaps,
                                            sun, view, kernel, reversed_sun=True, pixel_losses=True)
    return gaps, loss, reverse


def fit(gaps, loss, reverse, weights):
    profiled = loss.min(axis=0)
    index = int(profiled.argmin())
    score = profiled[index]
    near = gaps[profiled <= score * (1 + 1 / max(1, len(weights) - 6)) + 1e-10]
    total = float(sum(weights))
    rmse, flat, placebo = (math.sqrt(v / total) for v in (score, profiled[0], reverse.min()))
    gap = float(gaps[index])
    status = gap_fit_status(gaps, gap, near[0], near[-1], rmse, flat, placebo)
    return gap, status, float(near[-1] - near[0])


def fit_group(packet, kernels, rows, shell):
    rows = sorted(rows, key=lambda o: o["version_id"])
    captures = [datetime.fromisoformat(r["captured_at_utc"]) for r in rows]
    seconds = (max(captures) - min(captures)).total_seconds()
    if seconds > 10 or len({r["radius_m"] for r in rows}) != 1 or any(
        rows[0][k] != r[k] for r in rows[1:] for k in ("center_east_m", "center_north_m")
    ):
        raise ValueError("The same-pass controls have incompatible capture times or centres")
    items = []
    with closing(read_only(packet)) as db:
        for o in rows:
            if not (o["resolved_b04_detector"] and o["roof_clear_by_source_mask"]
                    and o["roof_cells_expected"] == o["roof_cells_valid_all_bands"]):
                continue
            grid = pixel_grid(db, o)
            if len(grid[-1]) < 12:
                continue
            pixel_keys = coords(grid)
            gaps, profile, reversed_profile = render(o, grid, shell, kernels[o["platform"]])
            estimate, status, span = fit(gaps, profile.sum(axis=2), reversed_profile.sum(axis=2), grid[-1])
            items.append(dict(version=o["version_id"], grid=grid, pixels=pixel_keys, loss=profile,
                              reverse=reversed_profile, estimate=estimate, status=status, span=span))
    if len(items) != 2:
        return None, len(items)
    counts = Counter(k for q in items for k in q["pixels"])
    if max(counts.values()) > 2:
        raise ValueError("A chip repeats the same native coordinates")
    joint_loss, joint_reverse = (sum(q[field] @ np.array([1 / counts[k] for k in q["pixels"]]) for q in items)
                                  for field in ("loss", "reverse"))
    weights = {k: float(w) for q in items for k, w in zip(q["pixels"], q["grid"][-1], strict=True)}
    joint, joint_status, joint_span = fit(gaps, joint_loss, joint_reverse, list(weights.values()))
    primary = min(items, key=lambda q: (-len(q["pixels"]), q["version"]))
    reflectance = [dict(zip(q["pixels"], q["grid"][-2], strict=True)) for q in items]
    common = set(reflectance[0]) & set(reflectance[1])
    differences = [abs(reflectance[0][k] - reflectance[1][k]) for k in common]
    record = dict(candidate_id=rows[0]["candidate_id"], capture_at_utc=min(captures), capture_span_seconds=seconds,
                  radius_m=rows[0]["radius_m"], version_1=items[0]["version"], version_2=items[1]["version"],
                  tile_1_status=items[0]["status"], tile_2_status=items[1]["status"],
                  tile_1_m=items[0]["estimate"], tile_2_m=items[1]["estimate"],
                  tile_1_pixels=len(items[0]["pixels"]), tile_2_pixels=len(items[1]["pixels"]),
                  distinct_pixels=len(counts), shared_pixels=len(common),
                  same_reflectance_pixels=sum(bool(v < 1e-7) for v in differences),
                  reflectance_median_abs_difference=float(median(differences)) if differences else None,
                  primary_version=primary["version"], primary=primary["estimate"], primary_status=primary["status"],
                  primary_span_m=primary["span"], joint=joint, joint_status=joint_status, joint_span_m=joint_span,
                  equal_average=mean(q["estimate"] for q in items))
    return record, len(items)


def replay(packet, *, geometry=None, fit_reference=None, roster=None):
    with closing(read_only(packet)) as db:
        db.row_factory = sqlite3.Row
        rows, kernels = inputs(db)
        crs = {r[0] for r in db.execute("SELECT DISTINCT crs FROM scene")}
    if crs != {"EPSG:32614"}:
        raise ValueError(f"Native grids require a shared projection: {packet}")
    groups = defaultdict(list)
    for row in rows:
        if row["roof_type"] == "EFR":
            groups[row["candidate_id"], overpass(row["version_id"])].append(dict(row))

    def f(job):
        key, variants = job
        if roster is not None:
            g = roster[key[0]]
            shell = float(g["prior_rim_z_p95_native"]) - float(g["prior_ground_z_p50_native"])
        elif geometry is None:
            heights = {float(fit_reference[str(r["observation_id"])]["shell_above_ground_m"]) for r in variants}
            shell, = heights
        else:
            g = geometry[key[0]]
            shell = float(g["prior_rim_z_p95_native"]) - float(g["prior_ground_z_p50_native"])
            for row in variants:
                east, north = (float(g[k]) for k in ("center_east_m", "center_north_m"))
                radius = float(g["diameter_m"]) / 2
                if math.hypot(east - row["center_east_m"], north - row["center_north_m"]) + max(0, radius - row["radius_m"]) > CONTEXT_M - 16:
                    return key, None, 0
                row.update(center_east_m=east, center_north_m=north, radius_m=radius)
        record, count = fit_group(packet, kernels, variants, shell)
        if record and fit_reference is not None:
            for i, row in enumerate(sorted(variants, key=lambda q: q["version_id"]), 1):
                ref = fit_reference[str(row["observation_id"])]
                if record[f"tile_{i}_m"] != float(ref["estimated_gap_m"]) or record[f"tile_{i}_status"] != ref["fit_status"]:
                    raise ValueError(f"Baseline physics did not reproduce on {key}/{row['version_id']}")
        return key, record, count

    jobs = [(k, v) for k, v in sorted(groups.items()) if len(v) == 2 and len({q["version_id"].split("_")[5] for q in v}) == 2]
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=5) as pool:
        for i, output in enumerate(pool.map(f, jobs), 1):
            if i % 200 == 0 or i == len(jobs):
                print(packet.name, i, len(jobs), flush=True)
            yield output


def controls(*, reuse_pixels=False):
    lidar = {r["observation_id"]: r for r in read("shadow_geometry_estimates") if r["roof_type"] == "EFR"}
    roof_meta = {r["candidate_id"]: r["documented_terminal_id"] for r in read("tank_engineering_registry")}
    laser_references = {(r["candidate_id"], r["version_id"]): r for r in lidar.values()}
    geometry = legacy_geometry()
    aerial_refs, wanted = aerial_inputs(geometry)
    packet = DERIVED / f"{STEM}_aerial.sqlite"
    if not reuse_pixels:
        stage_aerial(wanted, packet)
    with closing(read_only(packet)) as db:
        staged = list(db.execute("SELECT candidate_id,version_id FROM observation JOIN scene USING(scene_id)"))
    if set(staged) != wanted or len(staged) != len(wanted):
        raise ValueError("Native chips have a different aerial control population")
    laser, optical, aerial_endpoints = [], [], {}
    populations = Counter()
    for key, row, n in replay(CALIBRATION, fit_reference=lidar):
        populations[f"lidar_{n}_usable"] += 1
        if not row:
            continue
        a, b = (laser_references[key[0], row[f"version_{i}"]] for i in (1, 2))
        if a["lidar_gap_m"] != b["lidar_gap_m"] or not a["lidar_gap_m"]:
            continue
        laser.append(row | dict(terminal_id=roof_meta[key[0]], reference_m=float(a["lidar_gap_m"]),
                                reference_max_separation_hours=max(float(a["maximum_sensor_gap_hours"]),
                                                                   float(b["maximum_sensor_gap_hours"]))))
    for key, row, n in replay(packet, geometry=geometry):
        populations[f"aerial_{n}_usable"] += 1
        if row:
            aerial_endpoints[key] = row
    for (roof, day, capture), (gap, hours, terminal, sha) in aerial_refs.items():
        estimate = aerial_endpoints.get((roof, capture))
        if estimate:
            optical.append(estimate | dict(terminal_id=terminal, reference_m=gap, flight_date=day,
                                           reference_max_separation_hours=hours, reference_sha256=sha))
    return laser, optical, aerial_endpoints, populations


def describe(rows):
    return {m: dict(observations=len(rows), mae_m=mean(abs(r[m] - r["reference_m"]) for r in rows),
                    median_absolute_error_m=median(abs(r[m] - r["reference_m"]) for r in rows),
                    mean_abs_equivalent_bbl=mean(abs(r[m] - r["reference_m"]) * math.pi * r["radius_m"] ** 2 / BARREL_M3
                                                for r in rows)) for m in METHODS} if rows else {}


def with_errors(rows):
    return [r | {f"{m}_abs_error_m": abs(r[m]-r["reference_m"]) for m in METHODS} for r in rows]


def forward_controls(issue):
    base = f"cushing_forward_native_optical_{issue}"
    receipt = json.loads((DERIVED / f"cushing_prerelease_forecast_{issue}.json").read_bytes())
    cutoff = datetime.fromisoformat(receipt["information_cutoff_utc"])
    tile_book = {r["l1c_product"]: r for r in read(f"cushing_forward_optical_coverage_{issue}_tiles")
                 if r["l1c_with_detector_available_by_utc"]}
    roster = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry")}
    endpoints = {}
    rows = []
    for key, data, _ in replay(DERIVED / f"{base}.sqlite", roster=roster):
        if not data:
            continue
        tiles = [tile_book[data[f"version_{i}"]] for i in (1, 2)]
        segment, = set(r["segment"] for r in tiles)
        available = max(datetime.fromisoformat(r["l1c_with_detector_available_by_utc"]) for r in tiles)
        if not max(datetime.fromisoformat(r["captured_at_utc"]) for r in tiles) <= available < cutoff:
            raise ValueError(f"Late or invalid input for {key}")
        record = data | dict(segment=segment, latest_source_available_at_utc=available, terminal_id=roster[key[0]]["terminal_id"])
        endpoints[key] = record
        rows.append(record)
    pairs = []
    for reference in receipt["measured_pairs"]:
        a, b = (endpoints.get((reference["candidate_id"], overpass(reference[k]))) for k in ("scene_before", "scene_after"))
        if a is None or b is None:
            continue
        if not a["capture_at_utc"] < b["capture_at_utc"] < cutoff or (a["segment"], b["segment"]) != ("preceding_week", "target_week"):
            raise ValueError("The common-pixel check used a different reporting interval")
        difference = (a["joint"] - b["joint"]) - (a["primary"] - b["primary"])
        pairs.append(dict(candidate_id=reference["candidate_id"], terminal_id=reference["terminal_id"],
                          first_capture=a["capture_at_utc"], last_capture=b["capture_at_utc"],
                          jointly_resolved=all(q["joint_status"] == RESOLVED for q in (a, b)),
                          both_primary_resolved=all(q["primary_status"] == RESOLVED for q in (a, b)),
                          primary_change_m=a["primary"] - b["primary"], joint_change_m=a["joint"] - b["joint"],
                          joint_minus_primary_change_m=difference,
                          joint_minus_primary_cylindrical_change_kbbl=difference * reference["volume_per_m"] * 1000))
    result = dict(release_date=issue, status="pre_registration_geometry_diagnostic", paired_tile_endpoints=len(rows),
                  identical_pixel_sets=sum(r["distinct_pixels"] == r["shared_pixels"] for r in rows),
                  median_shared_fraction=median(r["shared_pixels"] / r["distinct_pixels"] for r in rows),
                  changed_endpoint_heights=sum(r["joint"] != r["primary"] for r in rows),
                  registered_b04_roofs=len(receipt["measured_pairs"]), registered_roofs_with_both_overlap_endpoints=len(pairs),
                  changed_unregistered_roof_differences=sum(r["joint_minus_primary_change_m"] != 0 for r in pairs),
                  both_methods_resolved=sum(r["jointly_resolved"] and r["both_primary_resolved"] for r in pairs),
                  changed_where_both_resolved=sum(r["joint_minus_primary_change_m"] != 0 and r["jointly_resolved"] and
                                                   r["both_primary_resolved"] for r in pairs),
                  primary_to_joint_volume_difference_kbbl=sum(r["joint_minus_primary_cylindrical_change_kbbl"] for r in pairs))
    for tag, data in (("endpoints", rows), ("pairs", pairs)):
        export(f"{STEM}_{issue}_{tag}", data)
    write(DERIVED / f"{STEM}_{issue}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(result, flush=True)


def tandem_controls():
    packet = DERIVED / "shadow_change_controls.sqlite"
    saved = {r["observation_id"]: r for r in read("shadow_change_control_estimates") if r["roof_type"] == "EFR"}
    fits = {key: r for key, r, _ in replay(packet, fit_reference=saved) if r}
    vintage = defaultdict(set)
    for r in read("sentinel2_l1c_input_inventory"):
        if r["archive_available_at_utc"] and r["geometry_available_at_utc"]:
            vintage[r["version_id"]].add(max(datetime.fromisoformat(r[k]) for k in ("archive_available_at_utc", "geometry_available_at_utc")))
    passes, intervals, index = [], [], {}
    for r in read("shadow_tandem_pass_pairs"):
        if (r["selected_for_summary"] != "True" or r["both_resolved"] != "True" or r["roof_type"] != "EFR"
                or abs(float(r["other_minus_a_seconds"])) > 60):
            continue
        keys = [(r["candidate_id"], overpass(r[f"{arm}_version"])) for arm in ("a", "other")]
        if keys[0][1][0] == keys[1][1][0]:
            raise ValueError("Tandem observations came from the same instrument")
        before, after = (fits.get(k) for k in keys)
        if before is None or after is None:
            continue
        row = {k: r[k] for k in ("capture_day", "candidate_id", "other_sensor", "a_version", "other_version")}
        base_time = max(datetime.fromisoformat(r[f"{part}_{kind}_available_at_utc"])
                        for part in ("a", "other") for kind in ("archive", "geometry"))
        tiles = [q[f"version_{i}"] for q in (before, after) for i in (1, 2)]
        # Exact version can be available in several archives. Use the latest
        # recorded readiness when the pixel packet does not identify which one.
        available = max((base_time, *(max(vintage[v]) for v in tiles))) if all(vintage[v] for v in tiles) else None
        if available is not None and available < max(q["capture_at_utc"] for q in (before, after)):
            raise ValueError("Tandem training label is dated before observation")
        row.update(reference_m=0., radius_m=before["radius_m"], original=float(r["other_minus_a_gap_m"]),
                   tandem_label_available_at_utc=available,
                   a_primary_span_m=before["primary_span_m"], a_joint_span_m=before["joint_span_m"],
                   a_native_pixels=before["distinct_pixels"], a_gap_m=before["primary"],
                   a_tile_gap_spread_m=abs(before["tile_1_m"] - before["tile_2_m"]),
                   a_tile_reflectance_delta=before["reflectance_median_abs_difference"],
                   resolved=all(q[f"{m}_status"] == RESOLVED for q in (before, after) for m in ("joint", "primary")))
        for m in METHODS:
            row.update({m: after[m] - before[m], f"a_{m}": before[m], f"other_{m}": after[m]})
        passes.append(row)
        identity = r["capture_day"], r["candidate_id"], r["other_sensor"]
        if identity in index:
            raise ValueError(f"Repeat tandem match on {identity}")
        index[identity] = row
    for r in read("terminal_change_repeatability_tank_controls"):
        if r["all_four_resolved"] != "True" or r["elapsed_days"] != "10":
            continue
        a, b = (index.get((r[f"{edge}_day"], r["candidate_id"], r["other_sensor"])) for edge in ("earlier", "later"))
        if a is None or b is None:
            continue
        factor = float(r["kbbl_per_m"])
        row = {k: r[k] for k in ("earlier_day", "later_day", "candidate_id", "terminal_id", "other_sensor")}
        row.update(resolved=bool(a["resolved"] and b["resolved"]), kbbl_per_m=factor)
        for m in METHODS:
            for arm in ("a", "other"):
                row[f"{arm}_{m}_change_kbbl"] = (a[f"{arm}_{m}"] - b[f"{arm}_{m}"]) * factor
        intervals.append(row)
    groups = defaultdict(list)
    for r in intervals:
        if r["resolved"]:
            groups[r["earlier_day"], r["later_day"], r["terminal_id"], r["other_sensor"]].append(r)
    terminals = []
    for (start, end, terminal, sensor), rows in sorted(groups.items()):
        result = dict(earlier_day=start, later_day=end, terminal_id=terminal, other_sensor=sensor, tanks=len(rows))
        for m in METHODS:
            a, b = (math.fsum(r[f"{arm}_{m}_change_kbbl"] for r in rows) for arm in ("a", "other"))
            result.update({f"{m}_difference_kbbl": abs(a - b), f"{m}_opposite_sign": a * b < 0})
        terminals.append(result)
    report = dict(passes=describe(passes), common_resolved_passes=describe([r for r in passes if r["resolved"]]),
                  resolved_terminal_intervals=len(terminals), terminal_intervals_with_five_roofs=sum(r["tanks"] >= 5 for r in terminals),
                  repeat_error_kbbl={m: median(r[f"{m}_difference_kbbl"] for r in terminals) for m in METHODS},
                  sign_disagreements={m: sum(r[f"{m}_opposite_sign"] for r in terminals) for m in METHODS})
    resolved = [r for r in passes if r["resolved"]]
    report["clock_valid_ranking_days"] = [day for day in sorted({r["capture_day"] for r in resolved})
        if len({r["capture_day"] for r in resolved if r["tandem_label_available_at_utc"] is not None
                and r["tandem_label_available_at_utc"] < datetime.fromisoformat(day + "T00:00:00+00:00")}) >= 2]
    scoring, report["retrospective_ranking"] = rank_tandem(resolved)
    for name, rows in (("tandem_passes", with_errors(passes)), ("tandem_changes", intervals), ("tandem_terminals", terminals),
                       ("tandem_ranking", scoring)):
        export(f"{STEM}_{name}", rows)
    write(DERIVED / f"{STEM}_tandem_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    print(report, flush=True)


def rank_tandem(rows):
    # This cross-date engineering comparison uses recovered imagery that was
    # not always available on the historical dates. There is no causal replay.
    days = sorted({r["capture_day"] for r in rows})
    core = ("a_primary_span_m", "a_native_pixels", "a_gap_m", "radius_m")
    overlap = (*core, "a_tile_gap_spread_m", "a_joint_span_m", "a_tile_reflectance_delta")
    result, summary = [], {}

    def group(row):
        return crc32(row["candidate_id"].encode()) % 5

    for day in days:
        now = [r for r in rows if r["capture_day"] == day]
        scored = []
        for fold in range(5):
            before = [r for r in rows if r["capture_day"] != day and group(r) != fold]
            test = [r for r in now if group(r) == fold]
            if not test:
                continue
            y = np.array([abs(r["primary"]) <= 2 for r in before])
            if len({r["capture_day"] for r in before}) < 2 or len(np.unique(y)) != 2:
                raise ValueError("Too few independent tandem dates or roof observations for retrospective ranking")
            estimates = []
            day_counts = Counter(r["capture_day"] for r in before)
            weights = [len(before) / (len(day_counts) * day_counts[r["capture_day"]]) for r in before]
            for cols in (core, overlap):
                x, z = (np.array([[r[k] for k in cols] for r in arm]) for arm in (before, test))
                centre = np.median(x, axis=0)
                scale = np.maximum(x.std(axis=0), np.array([.25, 5., .5, 1., .5, .25, .0001])[:len(cols)])
                clf = LogisticRegression(C=.05, max_iter=500).fit(np.clip((x - centre) / scale, -4, 4), y, sample_weight=weights)
                estimates.append(clf.predict_proba(np.clip((z - centre) / scale, -4, 4))[:, 1])
            scored.extend(r | dict(consistent=abs(r["primary"]) <= 2, core_score=float(a), overlap_score=float(b),
                                   roof_fold=fold, training_observations=len(before), training_dates="|".join(sorted(day_counts)))
                          for r, a, b in zip(test, *estimates, strict=True))
        quota = len(scored) // 2
        sorted_by = (("physics", lambda r: (r["a_primary_span_m"], -r["a_native_pixels"])),
                     ("base_ml", lambda r: (-r["core_score"],)), ("overlap_ml", lambda r: (-r["overlap_score"],)))
        def identifier(row):
            return row["candidate_id"], row["other_sensor"]

        picks = {m: {identifier(r) for r in sorted(scored, key=lambda r: (*order(r), *identifier(r)))[:quota]}
                 for m, order in sorted_by}
        if len({identifier(r) for r in scored}) != len(scored) or any(len(p) != quota for p in picks.values()):
            raise ValueError(f"Repeated tandem roof identity in the fixed-coverage check on {day}")
        for r in scored:
            result.append(r | {f"{m}_selected": identifier(r) in keys for m, keys in picks.items()})
        summary[day] = dict(observations=len(now), selected_per_model=quota, training_capture_dates=[d for d in days if day != d],
                            consistent_among_selected={m: sum(r["consistent"] and identifier(r) in p for r in scored)
                                                       for m, p in picks.items()})
    return result, summary


def main(*, reuse_pixels=False):
    laser, photo, endpoints, counts = controls(reuse_pixels=reuse_pixels)
    changes = []
    for row in aerial():
        versions = [row[f"{k}_satellite_version"] for k in ("first", "last")]
        a, b = (endpoints.get((row["candidate_id"], overpass(v))) for v in versions)
        if not a or not b:
            continue
        changes.append(dict(candidate_id=row["candidate_id"], terminal_id=row["terminal_id"],
            clock_separation_hours=row["max_gap_hours"], first_flight=row["first_flight"], last_flight=row["last_flight"],
            radius_m=a["radius_m"], reference_m=row["target"], archived_physical_m=row["physical"],
            all_four_resolved=all(q[f"tile_{i}_status"]==RESOLVED for q in (a,b) for i in (1,2)),
            both_joint_and_primary_resolved=all(q[f"{m}_status"]==RESOLVED for q in (a,b) for m in ("joint","primary")),
            **{m:a[m]-b[m] for m in METHODS}))
    result = dict(status="engineering_control_comparison", populations=dict(counts), summaries={})
    for label, data in (("lidar_levels", laser), ("aerial_levels", photo), ("aerial_changes", changes)):
        common = ([r for r in data if r["both_joint_and_primary_resolved"] and r["all_four_resolved"]] if label == "aerial_changes" else
                  [r for r in data if all(r[k] == RESOLVED for k in ("joint_status", "tile_1_status", "tile_2_status"))])
        report = dict(all=describe(data), all_models_resolved=describe(common))
        if label != "aerial_changes":
            report.update(identical_pixel_sets=sum(r["shared_pixels"] == r["distinct_pixels"] for r in data),
                          identical_shared_reflectances=sum(r["shared_pixels"] == r["same_reflectance_pixels"] for r in data),
                          median_shared_fraction=median(r["shared_pixels"] / r["distinct_pixels"] for r in data),
                          common_resolved_joint_changed=sum(r["primary"] != r["joint"] for r in common),
                          joint_resolved_primary_rejected=sum(r["joint_status"] == RESOLVED and r["primary_status"] != RESOLVED for r in data),
                          by_reference_day={day: describe([r for r in common if r.get("flight_date", str(r["capture_at_utc"])[:10]) == day])
                            for day in sorted({r.get("flight_date", str(r["capture_at_utc"])[:10]) for r in common})})
        result["summaries"][label] = report
        if data:
            export(f"{STEM}_{label}", with_errors(data))
        print(label, report["all"], report["all_models_resolved"], flush=True)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-pixels", action="store_true")
    parser.add_argument("--release", type=date.fromisoformat)
    parser.add_argument("--tandem", action="store_true")
    args = parser.parse_args()
    if args.release:
        forward_controls(args.release.isoformat())
    elif args.tandem:
        tandem_controls()
    else:
        main(reuse_pixels=args.reuse_pixels)
