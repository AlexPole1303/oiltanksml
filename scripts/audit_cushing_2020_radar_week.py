"""Preserve Sentinel-1 inputs near the December 2020 aerial roof controls."""

import argparse
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time, timedelta, timezone
from itertools import combinations
from zoneinfo import ZoneInfo

import numpy as np

from audit_s1_grd_native_inputs import saved_assets, saved_windows, source_sample
from audit_shadow_precision import describe
from build_cushing_icesat2_controls import identities
from build_s1_naip_calibration_inputs import interpolate, native_grid
from build_s1_reference_matches import export, read
from build_weekly_shadow_validation import legacy_geometry
from calibrate_s1_grd_native_windows import read_band, window
from check_s1_grd_roof_resolvability import RANGE_RESOLUTION_M, SAMPLE_SPACING_M, tank_fit
from collect_public import RAW, ROOT, write
from collect_s1_grd_roof_inputs import DERIVED, collect


STEM = "cushing_2020_radar_week"
STORE = RAW / "sentinel1_grd_native" / STEM
LOCAL_ZONE = ZoneInfo("America/Chicago")
RADAR_DATES = {"2020-11-28", "2020-12-04", "2020-12-10"}
AERIAL_DATES = {"2020-12-01", "2020-12-04", "2020-12-08", "2020-12-10"}
PASS = "passes_single_scene_template_diagnostic"


def local_day(value):
    return datetime.fromisoformat(value).astimezone(LOCAL_ZONE).date()


def radar_sources():
    products = [r for r in read("s1_cushing_catalog") if r["relative_orbit"] == "34"
                and r["covers_cushing_reference_point"] == "True" and r["product_type"] in {"IW_GRDH_1S", "IW_SLC__1S"}
                and local_day(r["capture_start_utc"]).isoformat() in RADAR_DATES]
    return sorted(products, key=lambda r: (r["capture_start_utc"], r["product_id"]))


def collect_grd(products, geometry):
    engineering = {k: (g["longitude"], g["latitude"], 2*g["radius_m"]) for k, g in geometry.items()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        retrieved = list(pool.map(lambda p: collect(dict(p, selection_reasons="aerialok_december_2020_control"),
                                                   engineering, store=STORE), products))
    manifest = [m for m, _ in retrieved]
    windows = [r for _, group in retrieved for r in group]
    clocks, metadata_count = saved_assets(manifest)
    unique = saved_windows([{k: str(v) if k == "centre_pixel_valid" else v for k, v in r.items()} for r in windows])
    verified = source_sample(unique, per_product=True)
    export(f"{STEM}_native_windows", windows)
    sources = [dict(product_id=m["product"]["product_id"], product_name=m["product"]["product_name"],
                    source_manifest=str((STORE/m["product"]["product_name"].removesuffix(".SAFE")/"native_manifest.json").relative_to(ROOT)),
                    native_windows=sum(r["native_windows"] for r in m["bands"].values()),
                    archive_inputs_ready_utc=clocks[m["product"]["product_id"]].isoformat()) for m in manifest]
    export(f"{STEM}_grd_sources", sources)
    return manifest, windows, unique, dict(source_metadata_verified=metadata_count, source_windows_read_back=verified)


def calibrate(manifests, windows):
    by_product = {m["product"]["product_id"]: m for m in manifests}
    grouped = defaultdict(dict)
    for r in windows:
        grouped[r["product_id"], r["polarization"]][r["window_file"]] = r
    rows = []
    for (p, pol), sources in sorted(grouped.items()):
        parameters = read_band(by_product[p], pol)
        rows.extend(window(r, parameters, output_root=STORE/"sigma0") for r in sources.values())
    export(f"{STEM}_sigma0", rows)
    return {r["native_window_file"]: r for r in rows}


def fit_scene(manifest, windows, sigma, geometry):
    p = manifest["product"]
    group = {r["candidate_id"]: r for r in windows if r["product_id"] == p["product_id"] and r["polarization"] == "VV"}
    positions = {k: (float(r["native_sample"]), float(r["native_line"])) for k, r in group.items()}
    disks = {k: (r["radius_m"], r["lidar_top_navd88_m"]-r["lidar_ground_navd88_m"]) for k, r in geometry.items()}
    path = manifest["objects"]["annotation/iw-vv.xml"]["local_file"]
    grid = native_grid(ROOT/path)
    if not np.all(np.diff(grid[2][:, :, 1], axis=1) > 0):
        raise ValueError(f"Unexpected direction of increasing slant range: {p['product_name']}")
    for roof, source in sorted(group.items()):
        x, y = positions[roof]
        incidence, slant, lon, lat, terrain = interpolate(grid, y, x)
        tank = geometry[roof]
        row = dict(candidate_id=roof, terminal_id=tank["terminal_id"], product_id=p["product_id"], product_name=p["product_name"],
                   radar_local_date=local_day(p["capture_start_utc"]).isoformat(), radar_start_utc=p["capture_start_utc"],
                   radar_end_utc=p["capture_end_utc"], source_annotation=path, approximate_native_line=y,
                   approximate_native_sample=x, nominal_rim_diameter_m=2*tank["radius_m"], incidence_angle_deg=float(incidence),
                   slant_range_time_s=float(slant), qualified_aerial_shadow_reference="False",
                   maximum_wall_roof_displacement_m=disks[roof][1]/math.tan(math.radians(incidence)))
        yield row | tank_fit(sigma[source["window_file"]], row, disks, positions)


def aerial_matches(fits):
    aerial = defaultdict(list)
    for r in read("aerialok_native_optical_aerial"):
        if r["flight_date"] in AERIAL_DATES:
            aerial[r["candidate_id"]].append(r)
    for r in fits:
        start, stop = (datetime.fromisoformat(r[k]) for k in ("radar_start_utc", "radar_end_utc"))
        for a in aerial[r["candidate_id"]]:
            day = datetime.fromisoformat(a["flight_date"]).date()
            bounds = [datetime.combine(day+timedelta(days=i), time(), LOCAL_ZONE).astimezone(timezone.utc) for i in (0,1)]
            maximum = max(abs((s-t).total_seconds()) for s in (start, stop) for t in bounds)/3600
            if maximum > 96:
                continue
            infer = a["clock_status"] == "shadow_inferred_clock"
            clock = [datetime.fromisoformat(a[k]) for k in ("exterior_trial_first_utc", "exterior_trial_last_utc")] if infer else []
            inferred = max(abs((s-t).total_seconds()) for s in (start, stop) for t in clock)/3600 if infer else None
            aerial_ok = a["aerial_fit_status"] == "visible_shadow_fit"
            gap = float(a["aerial_gap_m"]) if aerial_ok else None
            displacement = gap/math.tan(math.radians(r["incidence_angle_deg"])) if aerial_ok else None
            radar_ok = r["status"] == PASS
            yield dict(candidate_id=r["candidate_id"], terminal_id=r["terminal_id"], aerial_date=a["flight_date"],
                       aerial_source=a["source_path"], aerial_sha256=a["source_sha256"], aerial_status=a["aerial_fit_status"],
                       radar_product_id=r["product_id"], radar_product_name=r["product_name"], radar_local_date=r["radar_local_date"],
                       radar_start_utc=r["radar_start_utc"], radar_end_utc=r["radar_end_utc"], radar_fit_status=r["status"],
                       maximum_calendar_day_separation_hours=maximum, maximum_inferred_separation_hours=inferred,
                       aerial_gap_m=gap, radar_gap_m=r["geometry_template_gap_m"], expected_roof_rim_displacement_m=displacement,
                       radar_minus_aerial_gap_m=r["geometry_template_gap_m"]-gap if aerial_ok and radar_ok else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--reuse-grd", action="store_true")
    args = parser.parse_args()
    legacy = legacy_geometry()
    tanks, _, _ = identities()
    geometry = {k: tanks[k] for k in legacy}
    products = radar_sources()
    if args.reuse_grd:
        manifests = [json.loads((ROOT/r["source_manifest"]).read_bytes()) for r in read(f"{STEM}_grd_sources")]
        raw = read(f"{STEM}_native_windows")
        _, metadata_count = saved_assets(manifests)
        unique = saved_windows(raw)
        evidence = dict(source_metadata_verified=metadata_count, source_windows_read_back=source_sample(unique, per_product=True))
    else:
        grd = [p for p in products if p["product_type"] == "IW_GRDH_1S"]
        manifests, raw, unique, evidence = collect_grd(grd, geometry)
    write(DERIVED/f"{STEM}_collection.json", (json.dumps(evidence, indent=2)+"\n").encode())
    if args.collect_only:
        return
    sigma = calibrate(manifests, raw)
    fits = [r for m in manifests for r in fit_scene(m, raw, sigma, geometry)]
    export(f"{STEM}_fits", fits)
    comparisons = list(aerial_matches(fits))
    export(f"{STEM}_aerial_pairs", comparisons)
    radar_by_scene = defaultdict(set)
    for r in fits:
        if r["status"] == PASS:
            radar_by_scene[r["product_name"]].add(r["candidate_id"])
    days = defaultdict(list)
    for r in comparisons:
        days[r["aerial_date"], r["radar_local_date"]].append(r)
    separations = {key: [r["maximum_inferred_separation_hours"] for r in rows
                         if r["maximum_inferred_separation_hours"] is not None] for key, rows in days.items()}
    report = dict(grd_scenes=len(manifests), native_windows=len(unique), native_polarization_rows=len(raw), calibrated_windows=len(sigma),
                  vv_roof_fits=len(fits), fit_status=Counter(r["status"] for r in fits),
                  radar_roofs_passing_both_scenes={f"{a} / {b}": len(radar_by_scene[a] & radar_by_scene[b])
                                                 for a, b in combinations(sorted({r["product_name"] for r in fits}), 2)},
                  aerial_comparisons={f"{a} / {b}": dict(rows=len(rows),
                      aerial_resolved=sum(r["aerial_gap_m"] is not None for r in rows),
                      inferred_maximum_separation_hours=dict(observations=len(separations[a,b]),
                          minimum=min(separations[a,b], default=None), maximum=max(separations[a,b], default=None)),
                      joint=describe([r["radar_minus_aerial_gap_m"] for r in rows if r["radar_minus_aerial_gap_m"] is not None]),
                      aerial_displacement_at_least_one_native_pixel=sum((r["expected_roof_rim_displacement_m"] or 0)>=SAMPLE_SPACING_M for r in rows),
                      aerial_displacement_at_least_nominal_resolution=sum((r["expected_roof_rim_displacement_m"] or 0)>=RANGE_RESOLUTION_M for r in rows))
                      for (a,b), rows in sorted(days.items())},
                  **evidence,
                  scope="Retrospective native GRD intensity experiment using the 2019 diagnostic thresholds. "
                        "A passing radar arc does not establish an independent roof height. Optical clocks and heights are estimated from aerial shadows.")
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2)+"\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
