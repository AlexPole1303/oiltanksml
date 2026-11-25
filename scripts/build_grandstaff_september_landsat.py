"""Collect independent Landsat observations for the September Grandstaff balance check."""

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from statistics import median

from shapely.geometry import Point, shape

from build_cushing_inspection_b8 import recover
from build_grandstaff_okeis_records import LOCAL_TIME, STEM as OPERATOR, TERMINAL
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_landsat import LANDSAT, catalog
from collect_public import write

STEM = "cushing_grandstaff_september_landsat"
STORE = LANDSAT / STEM
MONTH = "2026-09"
TOLERANCE = timedelta(hours=48)


def inputs():
    intervals = [r for r in read(f"{OPERATOR}_optical_throughput_intervals") if r["report_month"] == MONTH]
    roofs = {r["candidate_id"] for r in intervals}
    points = {r["candidate_id"]: Point(float(r["longitude"]), float(r["latitude"]))
              for r in read("tank_engineering_registry") if r["candidate_id"] in roofs}
    scenes, coverage = {}, {}
    for item in catalog(2026, "usgs"):
        props = item["properties"]
        stamp = datetime.fromisoformat(props["datetime"]).astimezone(LOCAL_TIME)
        if stamp.strftime("%Y-%m") != MONTH or props["view:sun_elevation"] <= 0:
            continue
        footprint = shape(item["geometry"])
        covered = {roof for roof, point in points.items() if footprint.covers(point)}
        if covered:
            scenes[item["id"]] = item
            coverage[item["id"]] = covered
    return intervals, scenes, coverage


def cross_sensor(intervals, scenes, fits, coverage):
    b04 = {}
    roofs = {r["candidate_id"] for r in intervals}
    for row in read("weekly_shadow_holdout_observations"):
        if row["terminal_id"] == TERMINAL and row["geometry_mode"] == "dated_2011_lidar" and row["candidate_id"] in roofs:
            b04[row["candidate_id"], row["version_id"]] = row
    b8 = {(r["candidate_id"], r["product_id"]): r for r in fits}
    scene_lookup = {r["product_id"]: r for r in scenes}
    times = {product: datetime.fromisoformat(r["captured_at_utc"]) for product, r in scene_lookup.items()}
    comparisons, pairs = {}, []
    for interval in intervals:
        tank = interval["candidate_id"]
        chosen = []
        for side in ("earlier", "later"):
            version = interval[f"{side}_version"]
            source = b04[tank, version]
            stamp = datetime.fromisoformat(interval[f"{side}_capture_local"])
            if datetime.fromisoformat(source["optical_capture_at_utc"]) != stamp or source["fit_status"] != RESOLVED:
                raise ValueError(f"Expected exact resolved B04 version {tank}/{version}")
            candidates = []
            for product, tanks in coverage.items():
                delta = times[product] - stamp
                if tank not in tanks or abs(delta) > TOLERANCE:
                    continue
                fit = b8.get((tank, product), {})
                status = fit.get("fit_status") or scene_lookup[product]["status"]
                gap = fit["estimated_gap_m"] if status == RESOLVED else None
                if gap is not None and abs(float(source["diameter_m"]) - fit["diameter_m"]) > 1e-6:
                    raise ValueError(f"Dated diameters differ for {tank}/{product}/{version}")
                record = dict(candidate_id=tank, b04_version_id=version, b04_capture_at_utc=source["optical_capture_at_utc"],
                              b04_gap_m=float(source["estimated_gap_m"]), b8_product_id=product,
                              b8_capture_at_utc=scene_lookup[product]["captured_at_utc"],
                              b8_minus_b04_hours=delta.total_seconds() / 3600, b8_fit_status=status, b8_gap_m=gap,
                              b8_minus_b04_gap_m=gap - float(source["estimated_gap_m"]) if gap is not None else None)
                comparisons[tank, version, product] = record
                if gap is not None:
                    candidates.append((abs(delta), product, record))
            chosen.append(min(candidates)[2] if candidates else None)
        first, last = chosen
        change = None
        if first and last and times[first["b8_product_id"]] < times[last["b8_product_id"]]:
            change = first["b8_gap_m"] - last["b8_gap_m"]
        pairs.append({k: interval[k] for k in ("candidate_id", "cohort_id", "earlier_capture_local", "later_capture_local",
                                               "earlier_version", "later_version", "satellite_net_change_equivalent_bbl",
                                               "registered_change_equivalent_bbl")} | dict(
            b04_roof_height_change_m=float(b04[tank, interval["earlier_version"]]["estimated_gap_m"])
            - float(b04[tank, interval["later_version"]]["estimated_gap_m"]),
            earlier_b8_id=first["b8_product_id"] if first else None,
            earlier_b8_minus_b04_hours=first["b8_minus_b04_hours"] if first else None,
            later_b8_id=last["b8_product_id"] if last else None,
            later_b8_minus_b04_hours=last["b8_minus_b04_hours"] if last else None,
            b8_roof_height_change_m=change,
        ))
    return list(comparisons.values()), pairs


def laser_coverage(tanks):
    sources = []
    for dataset in ("cushing_icesat2_native_profiles", "cushing_icesat2_photon_recovery_profiles"):
        for r in read(dataset):
            if datetime.fromisoformat(r["granule_start_utc"]).astimezone(LOCAL_TIME).strftime("%Y-%m") != MONTH:
                continue
            sources.append({k: r[k] for k in ("candidate_id", "terminal_id", "granule_start_utc", "filename", "beam",
                                             "estimated_wall_to_roof_m", "passes_roof_profile_checks", "photon_packet")} |
                           dict(dataset=dataset, covers_comparison_roof=r["candidate_id"] in tanks))
    return sources


def main():
    intervals, scenes, coverage = inputs()
    native = DERIVED / f"{STEM}_native.sqlite"
    scene_rows, fits, stats = recover(coverage, scenes, store=STORE, native=native,
                                     purpose="retrospective_grandstaff_balance")
    comparisons, pairs = cross_sensor(intervals, scene_rows, fits, coverage)
    laser = laser_coverage({r["candidate_id"] for r in intervals})
    for suffix, rows in (("scenes", scene_rows), ("fits", fits), ("height_controls", comparisons),
                         ("interval_controls", pairs), ("laser_coverage", laser)):
        export(f"{STEM}_{suffix}", rows)
    valid = [r for r in comparisons if r["b8_fit_status"] == RESOLVED]
    both = [r for r in pairs if r["b8_roof_height_change_m"] is not None]
    passing_laser = sum(r["covers_comparison_roof"] and r["passes_roof_profile_checks"] == "True" for r in laser)
    summary = dict(stats, covering_usgs_products=len(scenes), downloaded_esri_products=sum(r["esri_exact_product_available"]
                   for r in scene_rows), independent_roof_fits_resolved=sum(r["fit_status"] == RESOLVED for r in fits),
                   b8_fit_status=dict(Counter(r["fit_status"] for r in fits)),
                   cross_sensor_window_hours=TOLERANCE.total_seconds() / 3600,
                   b04_b8_height_candidates=len(comparisons), b04_b8_height_controls=len(valid),
                   independent_landsat_products_in_height_controls=len({r["b8_product_id"] for r in valid}),
                   unique_landsat_roof_heights_in_controls=len({(r["b8_product_id"], r["candidate_id"]) for r in valid}),
                   median_absolute_height_disagreement_m=median(abs(r["b8_minus_b04_gap_m"]) for r in valid) if valid else None,
                   intervals_with_two_b8_measurements=len(both), september_laser_profiles_on_comparison_roofs=passing_laser,
                   original_b8_verified=0, scope="Retrospective Esri apparent reflectance on the USGS B8 grid. Sensor pairs may span real roof movement.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
