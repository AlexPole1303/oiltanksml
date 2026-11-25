"""Collect exact exterior controls and measure B8/B04 agreement on near-coincident overpasses."""

import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache

from pyproj import Transformer

from audit_landsat_c2_context_registration import (PADDING_M, PATCH_M, SEARCH_RADIUS_M, analysis_grid, estimate,
                                                  sentinel_source)
from audit_landsat_esri_b8_registration import context_crop, pixels
from audit_shadow_precision import describe
from build_landsat_esri_b8_registered_changes import identity, matched_refits, prepare, scenarios
from build_s1_reference_matches import export, read
from build_shadow_registration_inputs import anchor_grid, tank_points
from build_weekly_shadow_panel import DERIVED, RESOLVED
from collect_landsat_c2_registration_context import collect_sentinel
from collect_landsat_esri_b8 import inputs
from collect_public import write

STEM = "landsat_esri_b8_tandem"
OBSERVATION_FIELDS = ("candidate_id","version_id","canonical_l2a_id")
TANDEM_FIELDS = ("candidate_id","sentinel2_version_id","sentinel2_canonical_id")


def pairs():
    pan = {(r["product_id"],r["candidate_id"]):r for r in read("landsat_esri_b8_controls_fits")}
    rows = [r for r in read("landsat_esri_b8_controls_sentinel2")
            if (p:=pan.get((r["landsat_product_id"],r["candidate_id"]))) and p["esri_exterior_overlap_status"]=="shared_pixels_match"]
    needed = {tuple(r[k] for k in TANDEM_FIELDS) for r in rows}
    b04 = {}
    with (DERIVED/"shadow_timing_holdout_observations.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            key = tuple(r[k] for k in OBSERVATION_FIELDS)
            if key in needed and r["geometry_mode"]=="dated_2011_lidar":
                if key in b04:
                    raise ValueError(f"Repeated Sentinel-2 tandem control for {key}")
                b04[key] = r
    if b04.keys()!=needed:
        raise ValueError("Some tandem Sentinel-2 native chip IDs are missing")
    for r in rows:
        a,b = pan[r["landsat_product_id"],r["candidate_id"]],b04[tuple(r[k] for k in TANDEM_FIELDS)]
        first,last = datetime.fromisoformat(a["optical_capture_at_utc"]),datetime.fromisoformat(b["captured_at_utc"])
        minutes = (last-first).total_seconds()/60
        if (not 0<minutes<=45 or abs(minutes-float(r["sentinel2_minus_landsat_minutes"]))>1e-7
            or b["tile_metadata_path"]!=r["sentinel2_metadata_path"] or b["estimated_gap_m"]!=r["sentinel2_gap_m"]
            or b["fit_status"]!=RESOLVED or a["terminal_id"]!=b["terminal_id"]):
            raise ValueError(f"Tandem source provenance differs for {r['candidate_id']}")
        yield r | dict(week_ending=first.date().isoformat(),release_date=r["first_matching_release_date"],
            route="B8_B04",previous_version_id=a["product_id"],
            current_version_id=b["version_id"],current_canonical_id=b["canonical_l2a_id"],current_source_file=b["tile_metadata_path"],
            previous_sensor="B8",previous_observation_source="landsat_c2_b8_native_geometry.sqlite",previous_observation_id=a["observation_id"],
            current_sensor="B04",current_observation_source=b["measurement_source"],current_observation_id=b["observation_id"])


def exterior(rows):
    stages,products,_,_ = inputs()
    rasters = {r["product_id"]:r for r in read("landsat_esri_b8_registration_sources")}
    chips = defaultdict(list)
    for tile in read("landsat_esri_b8_tiles"):
        chips[tile["product_id"]].append(tile)
    sentinel = {r["version_id"]:r for r in read("landsat_c2_registration_context_sentinel2")}
    tanks,exclusion = tank_points()
    anchors = anchor_grid({r["candidate_id"]:tanks[r["candidate_id"]] for r in rows},exclusion,
        search_radius_m=SEARCH_RADIUS_M,patch_m=PATCH_M,padding_m=PADDING_M)
    _,_,bounds = analysis_grid({p for group in anchors.values() for p in group})
    geobounds = Transformer.from_crs(32614,4326,always_xy=True).transform_bounds(*bounds)

    @lru_cache(maxsize=None)
    def landsat(version):
        saved = rasters[version]
        if saved["used_in_weekly_exterior_search"]!="True":
            saved = context_crop(products[version],int(saved["object_id"]),bounds,chips[version])
        if saved["overlap_status"]!="shared_pixels_match":
            raise ValueError(f"The tandem B8 context differs from the roof pixels for {version}")
        return stages[version] | products[version] | saved | dict(band="Esri_B8",source_path=products[version]["level2_metadata_path"],
            archive_modified_at_utc=stages[version]["qa_and_angles_archive_at_utc"])

    @lru_cache(maxsize=None)
    def s2(version,source,canonical,captured):
        saved = sentinel.get(version)
        if saved is None:
            saved = collect_sentinel(version,source,canonical,captured,geobounds)
        if (saved["tile_metadata"],saved["canonical_id"],datetime.fromisoformat(saved["captured_at_utc"])) != (
            source,canonical,datetime.fromisoformat(captured)):
            raise ValueError(f"Tandem Sentinel-2 exterior provenance changed for {version}")
        return sentinel_source(saved)

    sources,pairs = {},defaultdict(list)
    for r in rows:
        a,b = r["previous_version_id"],r["current_version_id"]
        key = ("B8",a,products[a]["level2_metadata_path"]),("B04",b,r["current_source_file"])
        if key[0] not in sources:
            sources[key[0]] = landsat(a)
        if key[1] not in sources:
            sources[key[1]] = s2(b,r["current_source_file"],r["current_canonical_id"],r["sentinel2_capture_at_utc"])
        pairs[key].append(r)
    for source in sources.values():
        source["exterior_sha256"] = source["raster_sha256"]
    return sources,pairs


def refit(row, offset, chip):
    saved = float(row["b8_minus_b04_gap_m"]) if row["b8_minus_b04_gap_m"] else None
    fields = ("candidate_id","terminal_id","landsat_product_id","landsat_capture_at_utc","sentinel2_version_id",
              "sentinel2_capture_at_utc","sentinel2_minus_landsat_minutes","b8_status","sentinel2_fit_status")
    out = {k:row[k] for k in fields} | dict(independent_b8_minus_b04_m=saved,registration_status=offset["registration_status"],
        exterior_fit_residual_m=offset["fit_median_residual_m"],exterior_check_residual_m=offset["check_median_residual_m"],
        b04_minus_b8_east_m=offset["later_minus_earlier_east_m"],b04_minus_b8_north_m=offset["later_minus_earlier_north_m"],
        b8_exterior_raster=offset["earlier_red_raster"],b04_exterior_raster=offset["later_red_raster"],
        zero_offset_joint_b8_minus_b04_m=None,registered_joint_b8_minus_b04_m=None,registered_joint_diagnostics_pass=False,
        offset_stress_all_pass=False,offset_stress_min_b8_minus_b04_m=None,offset_stress_max_b8_minus_b04_m=None)
    if offset["registration_status"]!="checked_offset" or row["b8_status"]=="b8_roof_native_pixels_or_qa_incomplete":
        return out
    _,settings = scenarios(offset)
    zero,central,*stress = matched_refits([chip(identity(row,s)) for s in ("previous","current")],settings)
    return out | dict(zero_offset_joint_b8_minus_b04_m=zero["registered_change_m"],
        registered_joint_b8_minus_b04_m=central["registered_change_m"],
        registered_joint_diagnostics_pass=central["registered_both_shadow_diagnostics_pass"],
        offset_stress_all_pass=all(r["registered_both_shadow_diagnostics_pass"] for r in (central,*stress)),
        offset_stress_min_b8_minus_b04_m=min(r["registered_loss_range_min_m"] for r in (central,*stress)),
        offset_stress_max_b8_minus_b04_m=max(r["registered_loss_range_max_m"] for r in (central,*stress)))


def main():
    rows = list(pairs())
    sources,pair_groups = exterior(rows)
    _,scenes,offsets,_ = estimate(rows,sources_and_pairs=(sources,pair_groups),load_pixels=pixels,context_label="tandem B8/B04")
    by_key = {(r["candidate_id"],r["earlier_version_id"],r["later_version_id"]):r for r in offsets}
    work = [(r,by_key[r["candidate_id"],r["previous_version_id"],r["current_version_id"]]) for r in rows]
    if len(work)!=len(by_key) or len(offsets)!=len(by_key):
        raise ValueError("Tandem exterior offsets differ from the exact control pair inventory")
    chip = prepare([(offset,r,{}) for r,offset in work if offset["registration_status"]=="checked_offset"
                    and r["b8_status"]!="b8_roof_native_pixels_or_qa_incomplete"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        measurements = list(pool.map(lambda job:refit(*job,chip),work))
    export(f"{STEM}_sources",[dict(band=source["band"],version_id=key[1],raster_path=source["raster_path"],
                                  raster_sha256=source["exterior_sha256"],source_file=source["source_path"])
                               for key,source in sorted(sources.items())])
    export(f"{STEM}_pairs",measurements)
    matched = [r for r in measurements if r["registered_joint_diagnostics_pass"] and r["independent_b8_minus_b04_m"] is not None]
    fields = ("independent_b8_minus_b04_m","zero_offset_joint_b8_minus_b04_m","registered_joint_b8_minus_b04_m")
    by_scene = defaultdict(list)
    for r in matched:
        by_scene[r["landsat_product_id"],r["sentinel2_version_id"]].append(r)
    summary = dict(near_coincident_controls=len(measurements),scene_pairs=len(scenes),
        registration_status=Counter(r["registration_status"] for r in measurements),
        both_before_and_after_refit_pass=len(matched),matched_roofs=len({r["candidate_id"] for r in matched}),
        matched_scene_pairs=len(by_scene),same_pair_difference_m={k:describe([r[k] for r in matched]) for k in fields},
        matched_by_scene=[dict(landsat_product_id=a,sentinel2_version_id=b,**{
            k:describe([r[k] for r in group]) for k in fields}) for (a,b),group in sorted(by_scene.items())],
        scope="Identical pairs with both shadow methods passing. Landsat and Sentinel-2 are separated by about 22 to 23 minutes. "
        "Their observed height difference includes physical changes and inter-sensor disagreement. No calibration is learned.")
    payload = json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
