"""Inventory independent Sentinel-1 burst data for the September Grandstaff balance check."""

import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from shapely.geometry import Point

from audit_cushing_slc_archive import parent_files, probe
from audit_cushing_2020_radar_week import PASS, fit_scene
from audit_s1_grd_native_inputs import saved_assets, saved_windows, source_sample
from build_grandstaff_okeis_records import LOCAL_TIME
from build_grandstaff_september_landsat import MONTH, inputs
from build_cushing_icesat2_controls import identities
from build_s1_holdout_gap_burst_plan import CONTEXT_M, bursts
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_validation import legacy_geometry
from calibrate_s1_grd_native_windows import read_band, window
from collect_public import RAW, ROOT, write
from collect_s1_grd_roof_inputs import collect

STEM = "cushing_grandstaff_september_s1"
STORE = RAW / STEM


def burst_inputs():
    intervals, _, _ = inputs()
    roofs = {r["candidate_id"] for r in intervals}
    centres = legacy_geometry()
    windows = {roof: Point(float(centres[roof]["center_east_m"]), float(centres[roof]["center_north_m"])).buffer(
        float(centres[roof]["diameter_m"]) / 2 + CONTEXT_M) for roof in roofs}
    parents = [r for r in read("s1_cushing_catalog") if r["product_type"] == "IW_SLC__1S"
               and datetime.fromisoformat(r["capture_start_utc"]).astimezone(LOCAL_TIME).strftime("%Y-%m") == MONTH
               and roofs.intersection(json.loads(r["covered_candidate_ids_json"]))]
    with ThreadPoolExecutor(max_workers=3) as pool:
        candidates = list(pool.map(bursts, parents))
    groups, coverage = [], {}
    for product in candidates:
        ident, name = (product[k] for k in ("product_id", "product_name"))
        requests = []
        for b in product["bursts"]:
            covered = sorted(roof for roof, window in windows.items() if b["footprint"].covers(window))
            if not covered or not b["cdse_uuid"]:
                continue
            keys = ("cdse_uuid", "swath", "polarization", "asf_data_url", "asf_annotation_url", "cdse_catalog_file", "asf_catalog_file")
            request = {k: b[k] for k in keys} | dict(scope="grandstaff_september_balance", product_id=ident, product_name=name, weeks="")
            requests.append(request)
            coverage[b["cdse_uuid"]] = covered
        if requests:
            groups.append((ident, requests))
    return intervals, parents, groups, coverage


def grd_controls(parents, targets):
    absolute_orbits = {r["absolute_orbit"] for r in parents}
    registry = read("tank_engineering_registry")
    tanks = {r["candidate_id"]: tuple(float(r[k]) for k in ("longitude", "latitude", "footprint_diameter_m"))
             for r in registry if r["footprint_diameter_m"]}
    # Retain adjacent contours for the radar layover mask.
    tanks = {r["candidate_id"]: tanks[r["candidate_id"]] for r in registry if r["candidate_id"] in tanks
             and (r["documented_terminal_id"] == "enbridge_blueknight" or any(
                 abs(tanks[r["candidate_id"]][0] - tanks[t][0]) < .004
                 and abs(tanks[r["candidate_id"]][1] - tanks[t][1]) < .003 for t in targets))}
    products = [r for r in read("s1_cushing_catalog") if r["product_type"] == "IW_GRDH_1S"
                and not r["product_name"].endswith("_COG.SAFE") and r["absolute_orbit"] in absolute_orbits]
    if len(products) != len(absolute_orbits):
        raise ValueError("Expected one original GRD for each SLC pass")
    with ThreadPoolExecutor(max_workers=3) as pool:
        packets = list(pool.map(lambda r: collect(dict(r, selection_reasons="grandstaff_september_balance"), tanks,
                                                   store=STORE / "grd"), products))
    manifests = [manifest for manifest, _ in packets]
    raw = [row for _, rows in packets for row in rows]
    export(f"{STEM}_grd_native_windows", raw)
    _, metadata_count = saved_assets(manifests)
    unique = saved_windows(read(f"{STEM}_grd_native_windows"))
    matched = source_sample(unique, per_product=True)
    calibrated, fits = [], []
    geometry, _, _ = identities()
    for manifest, rows in packets:
        product = manifest["product"]["product_id"]
        bands = {pol: read_band(manifest, pol) for pol in manifest["bands"]}
        sigma = {path: window(r, bands[r["polarization"]], output_root=STORE / "sigma0")
                 for path, r in unique.items() if r["product_id"] == product}
        calibrated.extend(sigma.values())
        group = fit_scene(manifest, [r for r in rows if r["candidate_id"] in geometry], sigma, geometry)
        fits.extend(r for r in group if r["candidate_id"] in targets)
    by_date = defaultdict(dict)
    for r in fits:
        day = datetime.fromisoformat(r["radar_start_utc"]).astimezone(LOCAL_TIME).date().isoformat()
        by_date[day][r["candidate_id"]] = r
    changes = []
    for first, last in zip(sorted(by_date), sorted(by_date)[1:]):
        for tank in sorted(by_date[first].keys() & by_date[last].keys()):
            a, b = by_date[first][tank], by_date[last][tank]
            valid = a["status"] == b["status"] == PASS
            changes.append(dict(candidate_id=tank, earlier_local_date=first, later_local_date=last,
                                earlier_fit_status=a["status"], later_fit_status=b["status"],
                                roof_height_change_m=a["geometry_template_gap_m"] - b["geometry_template_gap_m"] if valid else None))
    for suffix, rows in (("grd_calibrated_windows", calibrated), ("grd_roof_fits", fits), ("grd_changes", changes)):
        export(f"{STEM}_{suffix}", rows)
    sources = [{"product_id": r["product_id"], "product_name": r["product_name"],
                "local_manifest": str((STORE / "grd" / r["product_name"].removesuffix(".SAFE") / "native_manifest.json").relative_to(ROOT))}
               for r in products]
    export(f"{STEM}_grd_sources", sources)
    return dict(grd_original_products=len(products), grd_native_files=len(unique), grd_metadata_files_verified=metadata_count,
                grd_source_windows_read_back=matched, grd_target_roof_fits=len(fits),
                grd_fit_status=dict(Counter(r["status"] for r in fits)),
                grd_changes_with_resolved_heights=sum(r["roof_height_change_m"] is not None for r in changes))


def main():
    intervals, parents, groups, coverage = burst_inputs()
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(lambda job: parent_files(job, store=STORE), groups))
    candidates = [r | dict(covered_candidate_ids_json=json.dumps(coverage[r["cdse_uuid"]])) for rows, _ in result for r in rows]
    assets = [r for _, rows in result for r in rows]
    jobs = [(label, r["cdse_uuid"], r[key]) for r in candidates for label, key in (
        ("cdse_burst", "cdse_burst_download_url"), ("cdse_direct_burst", "cdse_direct_burst_url"),
        ("asf_tiff", "asf_data_url")) if r[key]]
    jobs.extend((f"cdse_{r['role']}", r["s3_path"], r["download_url"]) for r in assets if r["role"] == "measurement")
    with ThreadPoolExecutor(max_workers=5) as pool:
        probes = list(pool.map(probe, jobs))
    clocks = {}
    for r in candidates:
        if r["polarization"] == "VV":
            for roof in coverage[r["cdse_uuid"]]:
                clocks.setdefault(roof, []).append((datetime.fromisoformat(r["azimuth_time_utc"]), r["cdse_uuid"]))
    timing = []
    for roof, timestamp in sorted({(r["candidate_id"], r[f"{side}_capture_local"])
                                   for r in intervals for side in ("earlier", "later")}):
        optical = datetime.fromisoformat(timestamp)
        radar, burst = min(clocks[roof], key=lambda item: (abs(item[0] - optical), item[1]))
        hours = (radar - optical).total_seconds() / 3600
        timing.append(dict(candidate_id=roof, optical_capture_local=timestamp, nearest_vv_burst_id=burst,
                           burst_azimuth_time_utc=radar.isoformat(), radar_minus_optical_hours=hours,
                           within_48h=abs(hours) <= 48))
    nearby = {(r["candidate_id"], r["optical_capture_local"]) for r in timing if r["within_48h"]}
    both = sum(all((r["candidate_id"], r[f"{side}_capture_local"]) in nearby for side in ("earlier", "later")) for r in intervals)
    for suffix, rows in (("bursts", candidates), ("assets", assets), ("probes", probes), ("endpoint_timing", timing)):
        export(f"{STEM}_{suffix}", rows)
    report = dict(local_month=MONTH, covering_slc_parents=len(parents), parents_with_covering_bursts=len(groups),
                  covering_bursts=len(candidates), covered_roofs=len({roof for roofs in coverage.values() for roof in roofs}),
                  relative_orbits=sorted({r["relative_orbit"] for r in candidates}),
                  routes_and_responses={label: dict(Counter(p["status"] for p in probes if p["route"] == label))
                                        for label in sorted({p["route"] for p in probes})},
                  verified_anonymous_tiff_responses=sum(p["source_signature"] == "TIFF" for p in probes),
                  slc_roof_heights_verified=0, optical_endpoints_with_slc_within_48h=len(nearby),
                  optical_intervals_with_slc_within_48h_of_both_endpoints=both,
                  **grd_controls(parents, {r["candidate_id"] for r in intervals}))
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
