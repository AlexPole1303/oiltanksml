"""Verify CEDA access for Sentinel-1 passes close to qualified laser roof controls."""

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from audit_cushing_slc_archive import probe
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_ceda_s1_slc_annotations import archive_urls, utc
from collect_public import write

STEM = "ceda_s1_laser_access"
MAX_HOURS = 7*24


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="check the close exact-photon control scenes")
    args = parser.parse_args()
    if args.native:
        native = read("cushing_slc_laser_controls_native")
        parents = {r["product_id"]:r["product_name"] for r in native}
        jobs = [("ceda_exact_laser_zip", key, url+".zip?download=1") for key,name in parents.items() for url in archive_urls(name)]
        asf = {r[f"{pol}_cdse_uuid"]:r[f"{pol}_asf_data_url"] for r in native for pol in ("vv","vh")}
        jobs.extend(("asf_burst_tiff",key,url) for key,url in asf.items() if url)
        with ThreadPoolExecutor(max_workers=4) as pool:
            probes = list(pool.map(probe,jobs))
        export(f"{STEM}_native_probes",probes)
        report = dict(parent_scenes=len(parents), exact_control_rows=len(native), requested_routes=len(probes),
            by_route_and_status=Counter(f"{r['route']} {r['status']}" for r in probes),
            readable_native_responses=sum(r["source_signature"] in ("ZIP","TIFF") for r in probes))
        payload = json.dumps(report,indent=2)+"\n"
        write(DERIVED/f"{STEM}_native_summary.json",payload.encode())
        print(payload,end="",flush=True)
        return
    profiles = read("cushing_icesat2_roof_profiles")+read("cushing_icesat2_photon_recovery_profiles")
    accepted = {tuple(r[k] for k in ("cmr_id","candidate_id","beam")): r for r in profiles
                if r["passes_roof_profile_checks"]=="True"}
    scenes = [r for r in read("s1_cushing_catalog") if r["product_type"]=="IW_SLC__1S"
              and r["covers_cushing_reference_point"]=="True"]
    exact = {r["product_id"] for r in read("cushing_slc_laser_controls")}
    products = {}
    pair_rows = []
    for r in accepted.values():
        start, stop = (utc(r[k]) for k in ("granule_start_utc","granule_end_utc"))
        for p in scenes:
            begin, end = (utc(p[k]) for k in ("capture_start_utc","capture_end_utc"))
            distance = max(abs((start-end).total_seconds()), abs((stop-begin).total_seconds()))/3600
            if distance > MAX_HOURS or r["candidate_id"] not in json.loads(p["covered_candidate_ids_json"]):
                continue
            identity = p["product_id"]
            products[identity] = p
            pair_rows.append(dict(product_id=identity, product_name=p["product_name"], candidate_id=r["candidate_id"],
                icesat2_cmr_id=r["cmr_id"], laser_beam=r["beam"], laser_start_utc=start, laser_end_utc=stop,
                radar_start_utc=begin, radar_end_utc=end, maximum_separation_hours=distance,
                in_existing_24h_burst_control_plan=identity in exact))
    jobs = [("ceda_full_slc_zip", identity, base+".zip?download=1") for identity, p in products.items()
            if p["satellite"] in {"S1A","S1B"} for base in archive_urls(p["product_name"])]
    with ThreadPoolExecutor(max_workers=6) as pool:
        probes = list(pool.map(probe, jobs))
    by_parent = defaultdict(list)
    for r in probes:
        by_parent[r["archive_id"]].append(r)
    for r in pair_rows:
        checks = by_parent[r["product_id"]]
        r["ceda_zip_probes"] = len(checks)
        r["ceda_zip_available"] = any(a["source_signature"]=="ZIP" for a in checks)
    export(f"{STEM}_pairs", sorted(pair_rows, key=lambda r: (r["maximum_separation_hours"], r["candidate_id"])))
    export(f"{STEM}_probes", probes)
    report = dict(qualified_laser_profiles=len(accepted), maximum_hours=MAX_HOURS, catalog_parent_scenes=len(products),
        matched_catalog_roof_pairs=len(pair_rows), ceda_s1a_s1b_routes=len(probes),
        ceda_response_status=Counter(r["status"] for r in probes), parents_with_readable_zip=sum(
            any(r["source_signature"]=="ZIP" for r in group) for group in by_parent.values()),
        interpretation="Parent footprint and capture clock screen for additional controls. CEDA requests are limited "
        "to the Sentinel-1A/B archive paths. File access for the close burst plan remains tracked separately.")
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
