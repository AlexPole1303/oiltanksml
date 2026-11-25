"""Match external chart dates to catalog acquisitions and verify native radar controls."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from urllib.parse import urlencode
from zoneinfo import ZoneInfo

from shapely import make_valid, points
from shapely.geometry import shape

from audit_cushing_2020_radar_week import PASS, fit_scene
from audit_s1_grd_native_inputs import saved_assets, saved_windows, source_sample
from audit_cushing_watch_benchmark import official_eia, score
from build_cushing_icesat2_controls import identities
from build_s1_reference_matches import export, read
from calibrate_s1_grd_native_windows import read_band, window
from collect_public import BBOX, RAW, ROOT, get, write
from collect_s1_grd_roof_inputs import collect
from collect_s1_roof_catalog import CATALOG, COG, DERIVED, GRD, SLC, product_rows

STEM = "cushing_watch_satellite_alignment"
STORE = RAW / STEM
CHICAGO = ZoneInfo("America/Chicago")
RECENT = {"2026-09-30", "2026-10-12", "2026-10-24", "2026-11-05"}
OPTICAL = {
    "sentinel_2_l1c": "stac_supplemental/sentinel-2-l1c",
    "sentinel_2_c1_l2a": "stac_supplemental/sentinel-2-c1-l2a",
    "landsat_8_9_l1": "landsat/usgs_catalog",
}
LOCAL_GRD = ("s1_grd_native_window_index", "s1_grd_holdout_gap_native_window_index",
             "cushing_2020_radar_week_native_windows", f"{STEM}_s1_native_windows")


def november_catalog():
    west, south, east, north = BBOX.split(",")
    polygon = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    query = (f"Collection/Name eq 'SENTINEL-1' and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') "
             "and ContentDate/Start ge 2026-11-01T00:00:00.000Z and ContentDate/Start lt 2026-11-10T00:00:00.000Z "
             "and (contains(Name,'_IW_SLC__1S') or contains(Name,'_IW_GRDH_1S'))")
    url = CATALOG + "?" + urlencode({"$filter": query, "$expand": "Attributes", "$count": "true",
                                     "$orderby": "ContentDate/Start asc", "$top": 100})
    path = STORE / "s1_november_2026_odata.json"
    payload = get(url, path)
    result = json.loads(payload)
    if len(result["value"]) != result["@odata.count"] or result.get("@odata.nextLink"):
        raise ValueError("Incomplete Sentinel-1 November query")
    relative = str(path.relative_to(ROOT))
    evidence = dict(url=url, source_file=relative, sha256=hashlib.sha256(payload).hexdigest(),
                    product_count=len(result["value"]))
    return list(product_rows([(row, relative) for row in result["value"]])), evidence


def radar_catalog_products(supplement):
    result = {}
    for row in read("s1_cushing_catalog") + supplement:
        row["relative_orbit"] = int(row["relative_orbit"])
        row["absolute_orbit"] = int(row["absolute_orbit"])
        row["covers_cushing_reference_point"] = row["covers_cushing_reference_point"] in (True, "True")
        row["covered_registry_centroids"] = int(row["covered_registry_centroids"])
        if row["product_id"] in result:
            raise ValueError(f"Repeated exact radar version {row['product_id']}")
        result[row["product_id"]] = row
    return result


def collect_recent(products, registry, refresh, verify_source):
    recent = [row for row in products.values() if row["capture_start_utc"][:10] in RECENT
              and row["relative_orbit"] == 34 and row["direction"] == "ASCENDING"
              and row["product_type"] == GRD and row["covers_cushing_reference_point"]]
    if {row["capture_start_utc"][:10] for row in recent} != RECENT or len(recent) != len(RECENT):
        raise ValueError("Expected exactly one covering original GRD per selected recent date")
    tanks = {key: tuple(float(r[f]) for f in ("longitude", "latitude", "footprint_diameter_m"))
             for key, r in registry.items()}
    store = STORE / "grd_native"
    if refresh:
        with ThreadPoolExecutor(max_workers=3) as pool:
            outputs = list(pool.map(lambda r: collect(dict(r, selection_reasons="cushing_watch_recent_asc_34_control"),
                                                       tanks, store=store), recent))
        export(f"{STEM}_s1_native_windows", [r for _, rows in outputs for r in rows])
    manifests = [json.loads((store / row["product_name"].removesuffix(".SAFE") /
                             "native_manifest.json").read_bytes()) for row in recent]
    rows = read(f"{STEM}_s1_native_windows")
    clocks, asset_count = saved_assets(manifests)
    windows = saved_windows(rows)
    checked = source_sample(windows, per_product=True) if verify_source else None
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["product_id"]].append(row)
    sources = []
    for manifest in manifests:
        product = manifest["product"]
        identity = product["product_id"]
        good = Counter(row["candidate_id"] for row in grouped[identity] if row["centre_pixel_valid"] == "True")
        sources.append(dict(product_id=identity, product_name=product["product_name"],
                            radar_capture_utc=product["capture_start_utc"],
                            native_files=len({r["window_file"] for r in grouped[identity]}),
                            native_rows=len(grouped[identity]), roofs_with_both_valid_centres=sum(n == 2 for n in good.values()),
                            archive_exact_inputs_ready_utc=clocks[identity].isoformat(),
                            native_manifest=str((store / product["product_name"].removesuffix(".SAFE") /
                                                 "native_manifest.json").relative_to(ROOT))))
    return manifests, rows, sources, dict(local_windows_verified=len(windows), source_metadata_verified=asset_count,
                                          source_windows_read_back=checked)


def optical_catalog(radar_times, registry):
    locations = points([(float(r["longitude"]), float(r["latitude"])) for r in registry.values()])
    for source, directory in OPTICAL.items():
        for path in sorted((RAW / directory).glob("*.geojson")):
            if path.stem < "2018":
                continue
            raw = json.loads(path.read_bytes())
            for item in raw["features"]:
                meta = item["properties"]
                stamp = datetime.fromisoformat(meta["datetime"])
                if stamp.date() not in radar_times:
                    continue
                coverage = int(make_valid(shape(item["geometry"])).covers(locations).sum())
                if not coverage:
                    continue
                granule = meta.get("s2:product_uri", item["id"])
                datatake = meta.get("s2:datatake_id", meta.get("landsat:scene_id", item["id"]))
                yield dict(chart_observation_date=stamp.date(), optical_catalog=source, item_id=item["id"],
                           product_version=granule, acquisition_key=datatake, platform=meta["platform"],
                           capture_utc=stamp.isoformat(), same_day_radar_utc=radar_times[stamp.date()].isoformat(),
                           optical_minus_radar_hours=(stamp - radar_times[stamp.date()]).total_seconds() / 3600,
                           footprint_covers_efr_centres=coverage,
                           catalog_scene_cloud_pct=meta.get("eo:cloud_cover"), source_file=str(path.relative_to(ROOT)),
                           catalog_item_url=next((link["href"] for link in item["links"] if link["rel"] == "self"), None))


def inventory(days, products, registry):
    grd_by_day, slc_by_day = defaultdict(set), defaultdict(set)
    for stem in LOCAL_GRD:
        for row in read(stem):
            if row["candidate_id"] in registry and row["centre_pixel_valid"] == "True" and (ROOT / row["window_file"]).is_file():
                grd_by_day[row["product_id"]].add(row["candidate_id"])
    slc_assets = {r["cdse_uuid"]: r for r in read("ceda_s1_slc_regional_assets") if (ROOT / r["local_packet"]).is_file()}
    for row in read("ceda_s1_slc_regional_windows"):
        asset = slc_assets.get(row["cdse_uuid"])
        if asset and row["candidate_id"] in registry and row["full_window_valid_by_native_mask"] == "True":
            if row["product_id"] != asset["product_id"]:
                raise ValueError(f"SLC parent does not reconcile for {row['cdse_uuid']}")
            slc_by_day[row["product_id"]].add(row["candidate_id"])
    radar_by_day = defaultdict(list)
    for row in products.values():
        if row["covers_cushing_reference_point"]:
            radar_by_day[date.fromisoformat(row["capture_start_utc"][:10])].append(row)
    for day in days:
        candidates = radar_by_day[day]
        track = [r for r in candidates if r["relative_orbit"] == 34 and r["direction"] == "ASCENDING"]
        track_key = {(r["satellite"], r["absolute_orbit"]) for r in track}
        primary = min(track, key=lambda r: (r["product_type"] != SLC, r["capture_start_utc"], r["product_id"]), default=None)
        if primary is None:
            raise ValueError(f"No Sentinel-1 ascending orbit 34 product covers the reference point on {day}")
        timestamp = datetime.fromisoformat(primary["capture_start_utc"])
        efr_grd = set().union(*(grd_by_day[r["product_id"]] for r in track))
        efr_slc = set().union(*(slc_by_day[r["product_id"]] for r in track))
        yield dict(
            chart_observation_date=day, radar_utc_capture_date=timestamp.date(),
            radar_cushing_local_date=timestamp.astimezone(CHICAGO).date(), orbit34_passes=len(track_key),
            other_sentinel_1_passes=len({(r["satellite"], r["absolute_orbit"]) for r in candidates} - track_key),
            satellite=primary["satellite"], absolute_orbit=primary["absolute_orbit"],
            relative_orbit=primary["relative_orbit"], direction=primary["direction"],
            radar_start_utc=primary["capture_start_utc"], radar_catalog_url=primary["catalog_url"],
            radar_source_file=primary["source_file"],
            slc_products_json=json.dumps(sorted(r["product_name"] for r in track if r["product_type"] == SLC)),
            original_grd_products_json=json.dumps(sorted(r["product_name"] for r in track if r["product_type"] == GRD)),
            cog_grd_products_json=json.dumps(sorted(r["product_name"] for r in track if r["product_type"] == COG)),
            local_grd_efr_count=len(efr_grd), local_slc_efr_count=len(efr_slc),
            provider_source_product_verified=False,
        )


def recent_diagnostics(manifests, windows):
    geometry, _, _ = identities()
    unique = saved_windows(windows)
    index = {m["product"]["product_id"]: m for m in manifests}
    bands = {(key, pol): read_band(m, pol) for key, m in index.items() for pol in m["bands"]}
    sigma_rows = [window(r, bands[r["product_id"], r["polarization"]], output_root=STORE / "sigma0")
                  for r in unique.values()]
    sigma = {r["native_window_file"]: r for r in sigma_rows}
    fit_windows = [r for r in windows if r["candidate_id"] in geometry]
    fits = [r for m in manifests for r in fit_scene(m, fit_windows, sigma, geometry)]
    grouped = defaultdict(dict)
    for row in fits:
        grouped[row["radar_start_utc"][:10]][row["candidate_id"]] = row
    changes = []
    for first, last in zip(sorted(grouped), sorted(grouped)[1:]):
        for tank in sorted(grouped[first].keys() & grouped[last].keys()):
            a, b = grouped[first][tank], grouped[last][tank]
            valid = a["status"] == b["status"] == PASS
            changes.append(dict(
                first_capture_day=first, last_capture_day=last,
                capture_days=(date.fromisoformat(last) - date.fromisoformat(first)).days, candidate_id=tank,
                first_product_id=a["product_id"], last_product_id=b["product_id"],
                first_fit_status=a["status"], last_fit_status=b["status"],
                first_gap_m=a["geometry_template_gap_m"], last_gap_m=b["geometry_template_gap_m"],
                both_pass_fixed_grd_diagnostic=valid,
                roof_height_change_m=a["geometry_template_gap_m"] - b["geometry_template_gap_m"] if valid else None,
            ))
    export(f"{STEM}_s1_sigma0", sigma_rows)
    export(f"{STEM}_s1_diagnostics", fits)
    export(f"{STEM}_s1_changes", changes)
    return dict(fit_count=len(fits), fit_status_counts=dict(Counter(r["status"] for r in fits)),
                repeat_pairs=len(changes), repeat_pairs_both_pass=sum(r["both_pass_fixed_grd_diagnostic"] for r in changes))


def acquisition_clock_checks(chart, matches):
    official = official_eia()
    clock = {r["release_date"]: r for r in read("eia_release_clock")}
    forward = {r["release_date"]: r for r in read("eia_wpsr_forward_releases")}
    schedule = {}
    for week, row in official.items():
        issue = row["date"].isoformat()
        if issue in forward:
            record = forward[issue]
            scheduled = record["scheduled_release_at_utc"]
        else:
            record = clock[issue]
            scheduled = record["scheduled_table_release_at_utc"]
        conservative = datetime.fromisoformat(record["conservative_postrelease_from_utc"])
        schedule[week] = (datetime.fromisoformat(scheduled) if scheduled else conservative, conservative)
    chart_by_day = {r["chart_observation_date"]: r for r in chart}
    official_weeks = sorted(official)
    for row in matches:
        day = row["chart_observation_date"]
        record = chart_by_day[day.isoformat()]
        acquired = datetime.fromisoformat(row["radar_start_utc"])
        nearest_local = min(official_weeks, key=lambda week: abs(week - row["radar_cushing_local_date"]))
        target = date.fromisoformat(record["official_nearest_week"])
        previous = date.fromisoformat(record["latest_earlier_report_week"]) if record["latest_earlier_report_week"] else None
        scheduled = max((week for week, pair in schedule.items() if pair[0] < acquired), default=None)
        conservative = max((week for week, pair in schedule.items() if pair[1] < acquired), default=None)
        yield dict(
            chart_observation_date=day, radar_start_utc=row["radar_start_utc"],
            radar_cushing_local_date=row["radar_cushing_local_date"], nearest_chart_week=target,
            nearest_radar_local_week=nearest_local, nearest_week_differs=nearest_local != target,
            chart_date_baseline_week=previous, scheduled_clock_baseline_week=scheduled,
            conservative_clock_baseline_week=conservative,
            scheduled_clock_differs_from_chart_date=scheduled != previous,
            conservative_clock_differs_from_chart_date=conservative != previous,
            scheduled_clock_baseline_kbbl=official[scheduled]["stocks"] if scheduled else None,
            conservative_clock_baseline_kbbl=official[conservative]["stocks"] if conservative else None,
            anchored_estimate_kbbl=int(record["anchored_estimate_kbbl"]),
            satellite_estimate_kbbl=int(record["satellite_estimate_kbbl"]),
            official_target_eia_kbbl=int(record["official_target_eia_kbbl"]),
            prior_retrospective_comparison_eligible=record["eligible_for_retrospective_paired_comparison"] == "True",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-recent", action="store_true")
    parser.add_argument("--verify-remote-windows", action="store_true")
    args = parser.parse_args()
    chart = read("cushing_watch_benchmark_paired_estimates")
    days = sorted(date.fromisoformat(r["chart_observation_date"]) for r in chart)
    registry = {r["candidate_id"]: r for r in read("tank_engineering_registry") if r["documented_roof_type"] == "EFR"}
    supplement, evidence = november_catalog()
    products = radar_catalog_products(supplement)
    manifests, windows, recent, verified = collect_recent(products, registry, args.collect_recent, args.verify_remote_windows)
    matched = list(inventory(days, products, registry))
    radar_times = {r["chart_observation_date"]: datetime.fromisoformat(r["radar_start_utc"]) for r in matched}
    optical = list(optical_catalog(radar_times, registry))
    optical_acquisitions = defaultdict(set)
    for row in optical:
        optical_acquisitions[row["chart_observation_date"], row["optical_catalog"]].add(row["acquisition_key"])
    optical_fields = {"sentinel_2_l1c": "same_utc_day_sentinel2_l1c_passes",
                      "sentinel_2_c1_l2a": "same_utc_day_sentinel2_c1_l2a_passes",
                      "landsat_8_9_l1": "same_utc_day_landsat_passes"}
    for row in matched:
        row.update({field: len(optical_acquisitions[row["chart_observation_date"], source])
                    for source, field in optical_fields.items()})
    clock = list(acquisition_clock_checks(chart, matched))
    clock_subset = [r for r in clock if r["prior_retrospective_comparison_eligible"]
                    and r["scheduled_clock_baseline_kbbl"] is not None and r["conservative_clock_baseline_kbbl"] is not None]
    clock_scores = {k: score(clock_subset, k, "official_target_eia_kbbl") for k in (
        "anchored_estimate_kbbl", "satellite_estimate_kbbl", "scheduled_clock_baseline_kbbl",
        "conservative_clock_baseline_kbbl")}
    diagnostics = recent_diagnostics(manifests, windows)
    catalog_days = {r["capture_start_utc"][:10] for r in products.values()
                    if days[0].isoformat() <= r["capture_start_utc"][:10] <= days[-1].isoformat()
                    and r["covers_cushing_reference_point"] and r["relative_orbit"] == 34
                    and r["direction"] == "ASCENDING" and r["product_type"] == SLC}
    chart_days = {day.isoformat() for day in days}
    summary = dict(
        chart_observations=len(days), chart_days_with_one_asc_orbit34_pass=sum(r["orbit34_passes"] == 1 for r in matched),
        chart_days_with_other_sentinel1_pass=sum(r["other_sentinel_1_passes"] > 0 for r in matched),
        ascending_orbit34_slc_catalog_days_in_window=len(catalog_days),
        chart_dates_missing_from_asc_orbit34_slc=sorted(chart_days - catalog_days),
        asc_orbit34_slc_dates_missing_from_chart=sorted(catalog_days - chart_days),
        chart_day_differs_from_radar_cushing_local_day=sum(r["chart_observation_date"] != r["radar_cushing_local_date"] for r in matched),
        matches_by_platform=dict(sorted(Counter(r["satellite"] for r in matched).items())),
        chart_days_same_utc_day_sentinel2_l1c=sum(r["same_utc_day_sentinel2_l1c_passes"] > 0 for r in matched),
        chart_days_same_utc_day_sentinel2_c1_l2a=sum(r["same_utc_day_sentinel2_c1_l2a_passes"] > 0 for r in matched),
        chart_days_same_utc_day_landsat=sum(r["same_utc_day_landsat_passes"] > 0 for r in matched),
        chart_days_local_grd=sum(r["local_grd_efr_count"] > 0 for r in matched),
        chart_days_local_slc=sum(r["local_slc_efr_count"] > 0 for r in matched),
        chart_days_with_local_grd_or_slc=sum(r["local_grd_efr_count"] > 0 or r["local_slc_efr_count"] > 0 for r in matched),
        radar_clock_sensitivity=dict(nearest_week_changes_with_local_calendar=sum(r["nearest_week_differs"] for r in clock),
                                     scheduled_baseline_week_changes=sum(r["scheduled_clock_differs_from_chart_date"] for r in clock),
                                     scheduled_clock_uses_conservative_when_release_time_unknown=True,
                                     conservative_baseline_week_changes=sum(r["conservative_clock_differs_from_chart_date"] for r in clock),
                                     original_eia_targets_retained_for_baseline_scores=clock_scores),
        november_catalog=evidence, recent_source_verification=verified, recent_fixed_grd_diagnostics=diagnostics,
        scope="Date-set correspondence is evidence for an acquisition track. Provider product IDs, processing and "
              "historical forecast availability are unverified. GRD diagnostics are retrospective.",
    )
    for name, rows in (("s1_recent_sources", recent), ("optical_catalog_matches", optical), ("dates", matched),
                       ("eia_clock_sensitivity", clock)):
        export(f"{STEM}_{name}", rows)
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
