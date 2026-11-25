"""Rank the exact USGS originals for checking the Esri results and recovering weekly inputs."""

import json
from collections import defaultdict

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import write

STEM = "landsat_esri_b8_original_priority"


def main():
    original = read("cushing_b8_download_manifest")
    mirror = {r["product_id"]:r for r in read("cushing_b8_archive_files")}
    esri = {r["product_id"]:r for r in read("landsat_esri_b8_scenes")}
    source = {r["product_id"]:r for r in read("landsat_esri_b8_registration_sources")}
    names = {r["product_id"] for r in original}
    if not len(original)==len(names) or not names==mirror.keys()==esri.keys() or not source.keys()<=names:
        raise ValueError("Originals, mirror inventory and Esri product IDs disagree")
    robust = {(r["week_ending"],r["candidate_id"]) for r in read("landsat_esri_b8_registered_changes_tank_weeks")
              if r["direction_across_resolved_options"] in ("increase","decrease")}
    weekly,stable,clocked,solo = (defaultdict(set) for _ in range(4))
    for r in read("landsat_esri_b8_registered_changes_pairs"):
        if r["registered_both_shadow_diagnostics_pass"]!="True":
            continue
        key = r["week_ending"],r["candidate_id"]
        needed = {r[f"{s}_version_id"] for s in ("previous","current") if r[f"{s}_b8_sha256"]}
        if not needed or not needed<=names:
            raise ValueError(f"Unidentified original B8 requirement for {key}")
        timing = r["cdse_source_bundle_clocks_before_cutoff"]=="True" or r["hls_exact_scene_lineage_before_cutoff"]=="True"
        for n in needed:
            weekly[n].add(key)
            if key in robust:
                stable[n].add(key)
            if timing:
                clocked[n].add(key)
                if len(needed)==1:
                    solo[n].add(key)
    controls = defaultdict(set)
    for r in read("landsat_esri_b8_tandem_pairs"):
        if r["registered_joint_diagnostics_pass"]=="True" and r["independent_b8_minus_b04_m"]:
            controls[r["landsat_product_id"]].add(r["candidate_id"])
    order = {"lidar_2022":1,"state_aerial_2025":2,"icesat2_laser_reference":3}
    rows = []
    for r in original:
        n = r["product_id"]
        m = mirror[n]
        if (m["original_b8_filename"],m["usgs_expected_b8_sha512"])!=(r["original_b8_filename"],r["expected_original_b8_sha512"]):
            raise ValueError(f"Mirror filename or checksum differs for {n}")
        rows.append(dict(product_id=n,purpose=r["purpose"],captured_at_utc=r["captured_at_utc"],
            priority_before_esri=int(r["priority"]),b8_status=r["b8_status"],original_b8_filename=r["original_b8_filename"],
            expected_original_b8_sha512=r["expected_original_b8_sha512"],b8_download_url=r["b8_download_url"],
            cdse_b8_download_url=m["cdse_b8_download_url"],cdse_b8_bytes=int(m["cdse_b8_bytes"]),
            exact_product_in_esri=esri[n]["exact_product_in_esri"],esri_overlap_status=source.get(n,{}).get("overlap_status"),
            external_laser_or_aerial_control_observations=int(r["external_control_observations"]),
            matched_tandem_control_roofs=len(controls[n]),jointly_resolved_tank_weeks_involving_this_product=len(weekly[n]),
            direction_stable_tank_weeks_involving_this_product=len(stable[n]),
            jointly_resolved_tank_weeks_with_bundle_or_lineage_clock=len(clocked[n]),
            jointly_resolved_mixed_tank_weeks_with_bundle_or_lineage_clock=len(solo[n]),
            additional_full_qa_tank_weeks_if_staged_next=int(r["additional_candidate_tank_weeks_if_staged_next"]),
            original_file_clock_established=False))
    rows.sort(key=lambda r:(order.get(r["purpose"],4),-r["jointly_resolved_mixed_tank_weeks_with_bundle_or_lineage_clock"],
        -r["direction_stable_tank_weeks_involving_this_product"],-r["matched_tandem_control_roofs"],
        -r["additional_full_qa_tank_weeks_if_staged_next"],r["priority_before_esri"]))
    rows = [dict(download_priority=i,**r) for i,r in enumerate(rows,1)]
    export(STEM,rows)
    report = dict(originals=len(rows),awaiting_original=sum(r["b8_status"]=="awaiting_original_b8" for r in rows),
        total_bytes=sum(r["cdse_b8_bytes"] for r in rows),external_control_originals=sum(r["purpose"] in order for r in rows),
        first_weekly_originals=[{k:r[k] for k in ("product_id","jointly_resolved_mixed_tank_weeks_with_bundle_or_lineage_clock",
             "direction_stable_tank_weeks_involving_this_product","cdse_b8_bytes")} for r in rows if r["purpose"] not in order][:5],
        scope="Download order for verification. The overlap of different originals is expected. Counts describe existing retrospective "
        "Esri observations linked to exact USGS product IDs. No count establishes that a USGS B8 file was available by an EIA cutoff.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
