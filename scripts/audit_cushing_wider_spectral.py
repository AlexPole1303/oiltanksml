"""Reprocess red-band gaps in other bands after independently checked wider registration."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime
from hashlib import sha256

from audit_sentinel2_alternate_registration import packet_inputs
from build_s1_reference_matches import DERIVED, export, read
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_validation import legacy_geometry
from check_multiband_shadow import spectral_kernels
from check_weekly_multiband_shadow import NONRED_BANDS, measure, pair_key, previous_inputs
from collect_public import ROOT, write
from collect_s2_l1c import source_metadata_bytes
from collect_s2_spectral_detector_masks import collect, jobs as detector_jobs
from fit_shadow_geometry import CALIBRATION, read_only

STEM = "cushing_wider_spectral"
SIDES = ("earlier","later")


def historical_inputs():
    primary = previous_inputs("holdout")
    alternate = {pair_key(r):r for r in read("sentinel2_alternate_registration_holdout_pairs")}
    for r in read("sentinel2_wider_registration_gap_pairs"):
        if r["wider_registration_status"]!="checked_offset":
            continue
        key = pair_key(r)
        alternative = r["pair_source"]=="alternate"
        if alternative:
            saved = alternate[key]
            endpoints = [(saved[f"{side}_native_packet"],int(saved[f"{side}_native_observation_id"])) for side in SIDES]
        else:
            packet,saved = primary[key]
            endpoints = [(packet,int(saved[f"{side}_observation_id"])) for side in SIDES]
        manifests = {}
        for side,(packet,number) in zip(SIDES,endpoints,strict=True):
            _,observations,scenes,_ = packet_inputs(packet)
            obs = observations[number]
            scene = scenes[obs["scene_id"]]
            version = r[f"{side}_version"]
            if (obs["candidate_id"],obs["version_id"],scene["version_id"])!=(r["candidate_id"],version,version):
                raise ValueError(f"Native pixels differ from the wider registration for {key}")
            manifests[version] = scene["manifest_path"]
        cutoff = datetime.fromisoformat(r["inputs_available_before_utc"])
        available = max(datetime.fromisoformat(r[f"{s}_archive_available_at_utc"]) for s in SIDES)
        if available>=cutoff:
            raise ValueError(f"Wider registration has a late original product for {key}")
        volume = {k:r[k] for k in ("week_ending","release_date","candidate_id","terminal_id")}
        volume.update(study_phase="holdout",cutoff_utc=cutoff.isoformat(),archive_available_at_utc=available.isoformat(),
                      circular_kbbl_per_m=float(r["circular_bbl_per_m"])/1000,
                      capture_change_kbbl=float(r["registered_change_m"])*float(r["circular_bbl_per_m"])/1000,
                      baseline_red_resolved=r["registered_both_shadow_diagnostics_pass"]=="True",alternate_capture_pair=alternative)
        for side in SIDES:
            volume.update({f"{side}_sentinel2_version":r[f"{side}_version"],f"{side}_capture_at_utc":r[f"{side}_capture_at_utc"],
                           f"{side}_gap_m":r[f"{side}_registered_gap_m"]})
        east,north = (float(r[f"later_minus_earlier_{axis}_m"])/2 for axis in ("east","north"))
        yield r,(volume,endpoints,east,north),manifests


def forward_inputs(issue):
    stem = f"cushing_forward_native_optical_{issue}"
    packet = f"{stem}.sqlite"
    _,observations,scenes,_ = packet_inputs(packet)
    lookup = {(r["candidate_id"],r["version_id"]):n for n,r in observations.items()}
    if len(lookup)!=len(observations):
        raise ValueError("Forward packet repeats a dated roof observation")
    tiles = {r["l1c_product"]:r for r in read(f"cushing_forward_optical_coverage_{issue}_tiles")}
    frozen = json.loads((DERIVED/f"cushing_prerelease_forecast_{issue}.json").read_bytes())
    cutoff = datetime.fromisoformat(frozen["information_cutoff_utc"])
    for r in read(f"{stem}_pairs"):
        if r["registration_status"]!="checked_offset":
            continue
        versions = [r[f"{s}_version"] for s in SIDES]
        endpoints = [(packet,lookup[r["candidate_id"],version]) for version in versions]
        manifests = {version:scenes[observations[n]["scene_id"]]["manifest_path"] for version,(_,n) in zip(versions,endpoints,strict=True)}
        available = max(datetime.fromisoformat(tiles[v]["l1c_with_detector_available_by_utc"]) for v in versions)
        if available>=cutoff:
            continue
        area = float(r["area_m2"])
        volume = {k:r[k] for k in ("candidate_id","terminal_id")}
        volume.update(week_ending=frozen["week_ending"],release_date=issue,study_phase="forward",cutoff_utc=cutoff.isoformat(),
                      archive_available_at_utc=available.isoformat(),circular_kbbl_per_m=area/BARREL_M3/1000,
                      capture_change_kbbl=float(r["registered_change_kbbl"]),alternate_capture_pair=False,
                      baseline_red_resolved=r["registered_both_shadow_diagnostics_pass"]=="True")
        for s,version in zip(SIDES,versions,strict=True):
            volume.update({f"{s}_sentinel2_version":version,f"{s}_capture_at_utc":r[f"{s}_capture"],
                           f"{s}_gap_m":r[f"{s}_registered_gap_m"]})
        east,north = (float(r[f"later_minus_earlier_{axis}_m"])/2 for axis in ("east","north"))
        yield {"circular_area_m2":area},(volume,endpoints,east,north),manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect-masks",action="store_true",help="Fetch and checksum original B03/B08 footprint files")
    parser.add_argument("--forward-release",type=date.fromisoformat,help="Use previously frozen forward B04 packets and cutoff")
    args = parser.parse_args()
    issue = args.forward_release.isoformat() if args.forward_release else None
    stem = f"{STEM}_{issue}" if issue else STEM
    sample = list(forward_inputs(issue) if issue else historical_inputs())
    products = {version:path for _,_,known in sample for version,path in known.items()}
    if args.collect_masks:
        with ThreadPoolExecutor(max_workers=6) as pool:
            export(f"{stem}_masks",list(pool.map(collect,detector_jobs(products))))
    mask_rows = read(f"{stem}_masks")
    masks = {(r["version_id"],r["band"]):r for r in mask_rows}
    if len(masks)!=len(mask_rows) or set(masks)!={(version,band) for version in products for band in NONRED_BANDS}:
        raise ValueError("The spectral footprints differ from the native wider-registration cohort")
    for r in mask_rows:
        data = source_metadata_bytes(ROOT/r["local_file"])
        if len(data)!=int(r["source_bytes"]) or sha256(data).hexdigest()!=r["source_sha256"]:
            raise ValueError(f"Detector footprint changed: {r['version_id']}/{r['band']}")
    with closing(read_only(CALIBRATION)) as db:
        kernels = spectral_kernels(db)
    geometry = {r["candidate_id"]:r for r in read("cushing_modern_lidar_optical_registry")} if issue else legacy_geometry()
    fits = list(measure("forward" if issue else "holdout",geometry,kernels,masks,source_jobs=(job for _,job,_ in sample),
                        native_geometry=bool(issue)))
    groups = defaultdict(list)
    rows = []
    for (r,job,_),fit in zip(sample,fits,strict=True):
        volume = job[0]
        if any(fit[k]!=volume[k] for k in ("candidate_id","release_date","earlier_sentinel2_version","later_sentinel2_version")):
            raise ValueError("The spectral refits did not reproduce the original job order")
        both = fit["nir_resolved"] and fit["joint_resolved"]
        same_direction = both and fit["nir_change_kbbl"]*fit["joint_change_kbbl"] > 0
        row = dict(fit,alternate_capture_pair=volume["alternate_capture_pair"],original_red_resolved=volume["baseline_red_resolved"],
                   nir_and_joint_agree_in_sign=same_direction,circular_area_m2=float(r["circular_area_m2"]))
        rows.append(row)
        groups[row["release_date"]].append(row)
    fields = dict.fromkeys(k for r in rows for k in r)
    export(f"{stem}_pairs",[fields | r for r in rows])
    summary = {}
    for issue,group in sorted(groups.items()):
        summary[issue] = dict(pairs=len(group),original_red_pairs=sum(r["original_red_resolved"] for r in group),
            recovered={name:len({r["candidate_id"] for r in group if not r["original_red_resolved"] and r[name]})
                       for name in ("nir_resolved","joint_resolved","nir_and_joint_agree_in_sign")})
    report = dict(status="physical_diagnostic_on_dated_images",products=len(products),masks=len(masks),releases=summary)
    write(DERIVED/f"{stem}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
