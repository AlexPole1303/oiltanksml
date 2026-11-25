"""Check the exact real-time Landsat 7 files against historical EIA cutoffs."""

import argparse
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from collect_cushing_landsat7 import (ASSETS, DERIVED, PIXELS, ROOT, STEM, UTC, catalog, collection1, export,
                                     legacy_geometry, parse_mtl, prepare_tool, rows, save_json, source_bytes, stage)
from collect_landsat import GCS_OBJECT, gcs_index
from collect_landsat_c2_registration_context import gcs_version


def exact_bundle(job):
    product, prefix, reference, _ = job
    directory = PIXELS / product
    objects = {Path(r["name"]).name: r for r in gcs_index(prefix, directory / "objects.json")}
    assets = {key: objects.get(f"{product}_{suffix}") for key, suffix in ASSETS.items()}
    if any(a is None for a in assets.values()):
        return dict(product_id=product, capture_day=reference["acquired_date"], captured_at_utc=None,
                    all_assets_created_by_utc=None, missing_assets="|".join(k for k, a in assets.items() if not a),
                    source_objects=str((directory / "objects.json").relative_to(ROOT)), b8_archive_generation=None)
    meta = parse_mtl(source_bytes(assets["mtl"], directory / "MTL.txt"))
    if (meta["LANDSAT_PRODUCT_ID"] != product or meta["DATE_ACQUIRED"] != reference["acquired_date"]
            or meta["SPACECRAFT_ID"] != "LANDSAT_7" or meta["SENSOR_ID"] != "ETM"):
        raise ValueError(f"Wrong real-time product metadata: {product}")
    observed = datetime.fromisoformat(f"{meta['DATE_ACQUIRED']}T{meta['SCENE_CENTER_TIME']}")
    start, stop = (datetime.fromisoformat(reference[f"c2_{field}_utc"]).replace(tzinfo=UTC) for field in ("start", "stop"))
    if not start - timedelta(seconds=1) <= observed <= stop + timedelta(seconds=1):
        raise ValueError(f"Collection 1 and 2 clocks differ for {product}")
    known = max(datetime.fromisoformat(asset["timeCreated"]) for asset in assets.values())
    if known < observed:
        raise ValueError(f"Real-time archive files predate their acquisition: {product}")
    pan = assets["pan"]
    generation = gcs_version(GCS_OBJECT + pan["name"], pan)
    return dict(product_id=product, capture_day=reference["acquired_date"], captured_at_utc=observed.isoformat(),
        all_assets_created_by_utc=known.isoformat(), missing_assets="",
        source_objects=str((directory / "objects.json").relative_to(ROOT)), b8_archive_generation=generation)


def reporting_pairs(usable, calendar):
    for report in calendar:
        previous, current, cutoff = (datetime.fromisoformat(report[k]) for k in
            ("previous_measurement_at_utc", "current_measurement_at_utc", "cutoff_utc"))
        for candidate, observations in sorted(usable.items()):
            selected = []
            for endpoint in (previous, current):
                options = [(observed, roof, item) for observed, roof, item in observations
                           if endpoint-timedelta(days=7) < observed <= endpoint
                           and max(roof["roof_evidence_date"], roof["lidar_available_date"]) < report["release_date"]
                           and datetime.fromisoformat(item["archive_assets_created_by_utc"]) < cutoff]
                if not options:
                    break
                selected.append(max(options, key=lambda x: (x[0], x[1]["roof_pixels_passing_qa"], x[2]["product_id"])))
            if len(selected) != 2 or selected[0][0] >= selected[1][0]:
                continue
            result = dict(week_ending=report["week_ending"], release_date=report["release_date"],
                          forecast_cutoff_utc=cutoff.isoformat(), candidate_id=candidate,
                          terminal_id=selected[0][1]["terminal_id"], b04_week_tanks=int(report["baseline_b04_tanks"]))
            for side, endpoint, (captured, roof, item) in zip(("previous", "current"), (previous, current), selected, strict=True):
                result.update({f"{side}_product_id": item["product_id"], f"{side}_captured_at_utc": captured.isoformat(),
                    f"{side}_archive_at_utc": item["archive_assets_created_by_utc"], f"{side}_native_packet": item["native_packet"],
                    f"{side}_roof_pixels_passing_qa": roof["roof_pixels_passing_qa"],
                    f"{side}_hours_before_eia_measurement": (endpoint-captured).total_seconds() / 3600})
            yield result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    legacy = legacy_geometry()
    _, primary_jobs = catalog(legacy)
    variants = collection1()
    jobs, primary = [], {}
    for selected, _, record, roofs in primary_jobs:
        key = record["track"], record["acquired_date"].replace("-", "")
        for kind, tier, _, product, prefix in variants[key]:
            if kind == "L1TP" and tier == 2:
                jobs.append((product, prefix, record, roofs))
                primary[product] = selected
    if len(primary) != len(jobs):
        raise ValueError("Repeated Landsat 7 real-time product")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        bundles = list(pool.map(exact_bundle, jobs))

    calendar = [r for r in rows(DERIVED / "sentinel2_space_time_coverage_eia_comparison.csv") if r["study_phase"] == "development"]
    current = [datetime.fromisoformat(r["current_measurement_at_utc"]) for r in calendar]
    if current != sorted(set(current)):
        raise ValueError("EIA development weeks do not have a unique sorted measurement time")
    evidence = {r["candidate_id"]: r for r in rows(DERIVED / "sentinel2_space_time_coverage_roof_registry.csv")}
    main_records = rows(DERIVED / f"{STEM}_reporting_window_coverage.csv")
    baseline_by_week = {r["week_ending"]: int(r["landsat7_archive_scenes_known_at_cutoff"]) for r in main_records}
    result, timely = [], []
    for bundle, job in zip(bundles, jobs, strict=True):
        decision = dict(week_ending=None, release_date=None, cutoff_utc=None, archive_before_cutoff=False)
        if bundle["captured_at_utc"]:
            observed = datetime.fromisoformat(bundle["captured_at_utc"])
            index = bisect_left(current, observed)
            report = calendar[index]
            previous = datetime.fromisoformat(report["previous_measurement_at_utc"])
            cutoff = datetime.fromisoformat(report["cutoff_utc"])
            if not previous < observed <= current[index]:
                raise ValueError(f"Collection 1 RT acquisition is outside the reporting window: {job[0]}")
            eligible = datetime.fromisoformat(bundle["all_assets_created_by_utc"]) < cutoff
            decision.update(week_ending=report["week_ending"], release_date=report["release_date"],
                            cutoff_utc=cutoff.isoformat(), archive_before_cutoff=eligible)
            if eligible:
                timely.append(job)
        result.append(bundle | decision | dict(selected_version_id=primary[job[0]],
                                               is_alternative_version=primary[job[0]] != job[0]))
    export(f"{STEM}_rapid_archive_bundles", result)
    usable_by_week, usable_by_roof, stage_rows = defaultdict(set), defaultdict(list), []
    indexed = {r["product_id"]: r for r in result}
    if timely:
        tool = prepare_tool()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            staged = list(pool.map(lambda job: stage(job, legacy, evidence, tool), timely))
        for item in staged:
            product = item["product_id"]
            decision = indexed[product]
            if datetime.fromisoformat(item["archive_assets_created_by_utc"]) != datetime.fromisoformat(decision["all_assets_created_by_utc"]):
                raise ValueError(f"Staged files changed the real-time bundle clock: {product}")
            qualified = [r for r in item["roofs"] if r["angle_valid"] and r["roof_pixels_passing_qa"] > 0]
            cutoff_day = decision["release_date"]
            documented = [r for r in qualified if max(r["roof_evidence_date"], r["lidar_available_date"]) < cutoff_day]
            capture = datetime.fromisoformat(item["captured_at_utc"])
            for roof in qualified:
                usable_by_roof[roof["candidate_id"]].append((capture, roof, item))
            if documented:
                usable_by_week[decision["week_ending"]].add(product)
            stage_rows.append(dict(product_id=product, week_ending=decision["week_ending"], cutoff_utc=decision["cutoff_utc"],
                captured_at_utc=item["captured_at_utc"], archive_at_utc=item["archive_assets_created_by_utc"],
                roof_windows=len(item["roofs"]), roofs_with_usable_pixels_and_angles=len(qualified),
                roofs_also_documented_before_cutoff=len(documented), native_packet=item["native_packet"],
                receipt=str((PIXELS / product / "item.json").relative_to(ROOT))))
        export(f"{STEM}_rapid_archive_native_scenes", stage_rows)
    candidates = list(reporting_pairs(usable_by_roof, [r for r in calendar if "2017" <= r["week_ending"][:4] <= "2021"]))
    if candidates:
        export(f"{STEM}_rapid_archive_weekly_candidates", candidates)
    by_week = Counter(r["week_ending"] for r in candidates)
    summary = dict(l1tp_rt_products=len(jobs), alternative_to_selected_versions=sum(r["is_alternative_version"] for r in result),
        complete_bundles=sum(not r["missing_assets"] for r in result), files_before_eia_cutoff=len(timely),
        weeks_with_files_before_cutoff=len({r["week_ending"] for r in result if r["archive_before_cutoff"]}),
        staged_native_products=len(stage_rows), scenes_with_usable_roof_pixels_and_angles=sum(
            r["roofs_with_usable_pixels_and_angles"] > 0 for r in stage_rows),
        weeks_with_usable_pixels_and_dated_roofs=len(usable_by_week),
        additional_weeks_with_usable_pixels_and_dated_roofs=sum(not baseline_by_week.get(w, 0) for w in usable_by_week),
        roof_windows_with_usable_pixels_and_dated_geometry=sum(r["roofs_also_documented_before_cutoff"] for r in stage_rows),
        same_roof_two_endpoint_candidates=len(candidates), weeks_with_same_roof_two_endpoint_candidates=len(by_week),
        weeks_with_same_roof_candidates_and_no_b04_pair=len({r["week_ending"] for r in candidates if r["b04_week_tanks"] == 0}),
        two_endpoint_candidate_scope="One RT capture per endpoint within the preceding seven days, both bundles and geometry dated before the current release. Requires separate registration and height validation.")
    save_json(DERIVED / f"{STEM}_rapid_archive_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
