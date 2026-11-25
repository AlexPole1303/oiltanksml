"""Build the dated Landsat 7 RT / Sentinel-2 B04 development input inventory."""

import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta

from collect_cushing_landsat7 import DERIVED, ROOT, STEM, export, legacy_geometry, rows, save_json
from measure_cushing_landsat7 import CENTRAL_FWHM, FWHM_M, RESOLVED, response_kernel, scene

MIXED = f"{STEM}_mixed"
MAX_AGE = timedelta(days=7)


@dataclass(frozen=True)
class Endpoint:
    captured: datetime
    available: datetime
    record: dict


def version_identity(row):
    return row["version_id"], row["canonical_l2a_id"], row["tile_metadata_path"]


def rapid_endpoints(geometry):
    observations, packets = defaultdict(list), {}
    for r in rows(DERIVED / f"{STEM}_rapid_archive_native_scenes.csv"):
        item = json.loads((ROOT / r["receipt"]).read_bytes())
        product = item["product_id"]
        captured, available = (datetime.fromisoformat(item[k]) for k in ("captured_at_utc", "archive_assets_created_by_utc"))
        if (r["product_id"] != product or product in packets or not product.endswith("_RT")
            or datetime.fromisoformat(r["captured_at_utc"]) != captured or datetime.fromisoformat(r["archive_at_utc"]) != available):
            raise ValueError(f"Inconsistent archive manifest for {product}")
        packets[product] = r["receipt"]
        for roof in item["roofs"]:
            tank = roof["candidate_id"]
            if tank not in geometry or not roof["angle_valid"] or not roof["roof_pixels_passing_qa"]:
                continue
            if not math.isclose(2*roof["radius_m"], float(geometry[tank]["diameter_m"])):
                raise ValueError(f"Landsat RT uses a different dated diameter for {tank}")
            record = roof | dict(product_id=product, captured_at_utc=item["captured_at_utc"], angle_valid="True",
                                 native_packet=item["native_packet"], native_receipt=r["receipt"])
            observations[tank].append(Endpoint(captured, available, record))
    return index(observations)


def sentinel_endpoints(geometry):
    observations = defaultdict(dict)
    for r in rows(DERIVED / "shadow_timing_development_observations.csv"):
        tank = r["candidate_id"]
        if tank not in geometry or r["geometry_mode"] != "dated_2011_lidar" or r["fit_status"] != RESOLVED:
            continue
        captured = datetime.fromisoformat(r["captured_at_utc"])
        if not 2017 <= captured.year <= 2021:
            continue
        if not math.isclose(float(r["diameter_m"]), float(geometry[tank]["diameter_m"])):
            raise ValueError(f"Sentinel-2 uses a different dated diameter for {tank}")
        available = max(datetime.fromisoformat(r[k]) for k in ("archive_available_at_utc", "geometry_available_at_utc"))
        key = version_identity(r)
        if key in observations[tank]:
            raise ValueError(f"Duplicate Sentinel-2 endpoint for {tank}/{key}")
        observations[tank][key] = Endpoint(captured, available, r)
    return index({key: group.values() for key, group in observations.items()})


def index(observations):
    grouped = {}
    for key, group in observations.items():
        ordered = sorted(group, key=lambda r: r.captured)
        grouped[key] = [r.captured for r in ordered], ordered
    return grouped


def eligible(group, endpoint, cutoff, first_seen):
    times, observations = group
    start = bisect_right(times, endpoint-MAX_AGE)
    stop = bisect_right(times, endpoint)
    return [r for r in observations[start:stop] if r.available < cutoff and r.captured.date().isoformat() >= first_seen]


def calendar():
    path = DERIVED / "sentinel2_space_time_coverage_eia_comparison.csv"
    return [r for r in rows(path) if r["study_phase"] == "development" and "2017" <= r["week_ending"][:4] <= "2021"]


def eligible_pairs(pan, sentinel, registry, reports):
    candidates = sorted(pan.keys() & sentinel.keys() & registry.keys())
    for report in reports:
        a, b, cutoff = (datetime.fromisoformat(report[k]) for k in ("previous_measurement_at_utc", "current_measurement_at_utc", "cutoff_utc"))
        for tank in candidates:
            g = registry[tank]
            first_seen = g["first_seen"]
            if (g["has_both_dated_sources"] != "True" or first_seen > a.date().isoformat()
                or max(g["roof_evidence_date"], g["lidar_source_available_date"]) >= cutoff.date().isoformat()):
                continue
            for pan_date, sentinel_date, b8_first in ((a, b, True), (b, a, False)):
                for p in eligible(pan[tank], pan_date, cutoff, first_seen):
                    for s in eligible(sentinel[tank], sentinel_date, cutoff, first_seen):
                        first, last = (p, s) if b8_first else (s, p)
                        if not first.captured <= a < last.captured <= b:
                            continue
                        if p.record["terminal_id"] != s.record["terminal_id"] or p.record["terminal_id"] != g["terminal_id"]:
                            raise ValueError(f"Terminal identity differs between sensors for {tank}")
                        yield report, g, p, s, first, last


def fit_rapid(options, geometry):
    pending = defaultdict(dict)
    for _, _, p, _, _, _ in options:
        roof = p.record
        pending[roof["product_id"]][roof["candidate_id"]] = (roof, None)
    kernels = {f: response_kernel(f) for f in FWHM_M}
    jobs = [(name, group.values()) for name, group in sorted(pending.items())]
    with ThreadPoolExecutor(max_workers=5) as pool:
        estimates = [row for group in pool.map(lambda job: scene(job, geometry, kernels), jobs) for row in group]
    by_key = {(r["product_id"], r["candidate_id"], r["effective_fwhm_m"]): r for r in estimates}
    if len(by_key) != len(estimates):
        raise ValueError("Duplicate Landsat RT fit")
    return estimates, by_key


def pair_record(option, fits, baseline_tanks):
    report, g, p, s, first, last = option
    a, b = (datetime.fromisoformat(report[k]) for k in ("previous_measurement_at_utc", "current_measurement_at_utc"))
    b8, b04 = p.record, s.record
    name, tank = b8["product_id"], b8["candidate_id"]
    fit = fits[name, tank, CENTRAL_FWHM]
    status = fit["fit_status"]
    center_at_edge = bool(fit["center_search_on_boundary"])
    return dict(week_ending=report["week_ending"], release_date=report["release_date"], forecast_cutoff_utc=report["cutoff_utc"],
        previous_measurement_at_utc=a.isoformat(), current_measurement_at_utc=b.isoformat(),
        candidate_id=tank, terminal_id=g["terminal_id"], diameter_m=g["diameter_m"], area_m2=g["circular_area_m2"],
        roof_evidence_date=g["roof_evidence_date"], lidar_source_available_date=g["lidar_source_available_date"],
        b04_baseline_week_tanks=int(report["baseline_b04_tanks"]), b04_baseline_covers_this_roof=(report["week_ending"],tank) in baseline_tanks,
        route="B8_RT_to_B04" if first is p else "B04_to_B8_RT", previous_capture_utc=first.captured.isoformat(),
        current_capture_utc=last.captured.isoformat(), previous_capture_age_hours=(a-first.captured).total_seconds()/3600,
        current_capture_age_hours=(b-last.captured).total_seconds()/3600,
        landsat_product_id=name, landsat_captured_at_utc=p.captured.isoformat(), landsat_available_at_utc=p.available.isoformat(),
        landsat_packet=b8["native_packet"], landsat_receipt=b8["native_receipt"], landsat_fit_status=status,
        landsat_center_search_at_boundary=center_at_edge, landsat_fitted_pixels=fit["fitted_pixels"],
        landsat_native_height_diagnostic_pass=status==RESOLVED and not center_at_edge, landsat_gap_m=fit["estimated_gap_m"],
        sentinel_version_id=b04["version_id"], sentinel_canonical_id=b04["canonical_l2a_id"],
        sentinel_tile_metadata=b04["tile_metadata_path"], sentinel_captured_at_utc=s.captured.isoformat(),
        sentinel_available_at_utc=s.available.isoformat(), sentinel_observation_source=b04["measurement_source"],
        sentinel_observation_id=b04["observation_id"], sentinel_fitted_pixels=int(b04["fitted_pixels"]),
        sentinel_gap_m=float(b04["estimated_gap_m"]))


def main():
    geometry = legacy_geometry()
    registry = {r["candidate_id"]: r for r in rows(DERIVED / "sentinel2_space_time_coverage_roof_registry.csv")}
    baseline_tanks = {(r["week_ending"], r["candidate_id"]) for r in rows(DERIVED / "sentinel2_space_time_coverage_tank_weeks.csv")
                      if r["study_phase"] == "development" and r["measurement_source"]}
    reports = calendar()
    options = list(eligible_pairs(rapid_endpoints(geometry), sentinel_endpoints(geometry), registry, reports))
    if not options:
        raise ValueError("No Landsat 7 and Sentinel-2 endpoints pass the dated-input rules")
    fits, by_key = fit_rapid(options, geometry)
    options = [pair_record(option, by_key, baseline_tanks) for option in options]
    export(f"{MIXED}_endpoint_fits", fits)
    export(f"{MIXED}_candidate_pairs", options)
    passed = [r for r in options if r["landsat_native_height_diagnostic_pass"]]
    report = dict(phase="development_2017_2021", weekly_release_windows=len(reports), dated_candidate_pairs=len(options),
        distinct_dated_roof_weeks=len({(r["week_ending"],r["candidate_id"]) for r in options}),
        weeks_with_candidates=len({r["week_ending"] for r in options}), landsat_rt_endpoint_fits=len(fits)//len(FWHM_M),
        landsat_rt_endpoint_status=dict(Counter(r["fit_status"] for r in fits if r["effective_fwhm_m"]==CENTRAL_FWHM)),
        pairs_passing_independent_height_diagnostics=len(passed),
        distinct_roof_weeks_passing_independent_height_diagnostics=len({(r["week_ending"],r["candidate_id"]) for r in passed}),
        weeks_passing_independent_height_diagnostics=len({r["week_ending"] for r in passed}),
        weeks_passing_independent_height_diagnostics_without_b04_pair=len({r["week_ending"] for r in passed if not r["b04_baseline_week_tanks"]}),
        description="Both acquisitions precede their own boundary by at most seven days. The current image is after the prior boundary. Exact source versions and engineering geometry precede the EIA cutoff. No EIA values enter selection.")
    save_json(DERIVED / f"{MIXED}_input_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
