"""Join exact Landsat B8 products to independent lidar and state aerial observations."""
import json
import math
from collections import Counter, defaultdict
from contextlib import closing

from build_landsat_c2_holdout_gap_inventory import ASSETS, parse
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_landsat import catalog
from collect_landsat_c2_gap_geometry import packet, roof_angles
from build_landsat_pan_roof_angles import prepare_tool
from collect_public import ROOT, write

STEM = "landsat_c2_external_controls"
LIDAR_DAY = "2022-12-31"
AERIAL_DAY = "2025-03-01"
MAX_GAP_HOURS = 30
DOWNLOAD_FIELDS = tuple(f"{label.lower().replace('.','_')}_url" for label in ASSETS)


def references():
    for r in read_csv("lidar_roof_elevation_reference"):
        if (r["acquisition_date_utc"]==LIDAR_DAY and r["dataset"]=="OK_Statewide_2_D22"
            and r["outline_fit_status"]=="well_supported_circular_fit" and r["rim_minus_center_plane_native_z"]
            and int(r["center_sectors"])>=6 and int(r["rim_sectors"])>=6):
            yield dict(reference_type="lidar_2022", candidate_id=r["candidate_id"], source_key=r["point_source_id"],
                start_utc=r["first_return_at_utc"], end_utc=r["last_return_at_utc"],
                reference_gap_m=float(r["rim_minus_center_plane_native_z"]), reference_gap_min_m=None,
                reference_gap_max_m=None, reference_file="data/derived/lidar_roof_elevation_reference.csv",
                geometry_basis=r["geometry_basis"], center_east_m=r["center_east_m"], center_north_m=r["center_north_m"],
                radius_m=r["radius_m"])
    for r in read_csv("ok_aerial_2025_roof_shadow_reference"):
        if (r["reference_day"]==AERIAL_DAY and r["exterior_clock_status"]=="exterior_matches_tile_time"
            and r["shadow_reference_status"]=="resolved_aerial_shadow"):
            yield dict(reference_type="state_aerial_2025", candidate_id=r["candidate_id"], source_key=r["tile_group"],
                start_utc=r["start_utc"], end_utc=r["end_exclusive_utc"],
                reference_gap_m=float(r["best_gap_m"]), reference_gap_min_m=float(r["near_gap_min_m"]),
                reference_gap_max_m=float(r["near_gap_max_m"]), reference_file=r["source_raster"],
                geometry_basis=r["geometry_source"], center_east_m=r["east"], center_north_m=r["north"], radius_m=r["radius"])


def catalog_versions():
    refs = {"2022-12-30", "2025-03-01"}
    products = {}
    for year in (2022,2025):
        pc = {r["id"]:r for r in catalog(year,"pc")}
        for item in catalog(year,"usgs"):
            props = item["properties"]
            if props["platform"] not in ("LANDSAT_8","LANDSAT_9") or props["datetime"][:10] not in refs:
                continue
            grid,day = item["id"].split("_")[2:4]
            for name,level2 in pc.items():
                if name[:4]!=item["id"][:4] or name.split("_")[2:4]!=[grid,day]:
                    continue
                metadata = ROOT / f"data/raw/landsat/collection2_context/{year}/{name}/MTL.json"
                if not metadata.is_file():
                    continue
                level1 = json.loads(metadata.read_bytes())["LANDSAT_METADATA_FILE"]["LEVEL1_PROCESSING_RECORD"]
                if level1["LANDSAT_PRODUCT_ID"]==item["id"]:
                    products[item["id"]] = item,level2,metadata,dict(product_id=item["id"],platform=props["platform"],
                        captured_at_utc=props["datetime"],level1_generated_at_utc=level1["DATE_PRODUCT_GENERATED"])
    return products


def main():
    candidates = defaultdict(list)
    for r in references():
        if not math.isfinite(r["reference_gap_m"]) or r["reference_gap_m"]<0 or not (ROOT / r["reference_file"]).is_file():
            continue
        candidates[r["candidate_id"]].append(r)
    with closing(read_only(SOURCE)) as db:
        roofs = dict(db.execute("SELECT candidate_id,documented_roof_type FROM tank_engineering_registry"))
        clear = defaultdict(set)
        for product,tank in db.execute("""SELECT product_id,candidate_id FROM landsat_tank_scene_quality
            WHERE qa_roof_cells>0 AND qa_roof_cells=valid_qa_roof_cells AND qa_roof_cells=clear_qa_roof_cells
              AND signal_cells>0 AND signal_cells=valid_signal_cells
              AND (product_id LIKE '%_20221230_%' OR product_id LIKE '%_20250228_%' OR product_id LIKE '%_20250301_%')"""):
            clear[product].add(tank)
    packets, output = [], []
    tool = prepare_tool()
    for product,(item,level2,metadata,base) in sorted(catalog_versions().items()):
        capture = parse(base["captured_at_utc"])
        by_roof = defaultdict(list)
        for tank in clear[level2["id"]] & candidates.keys():
            for r in candidates[tank]:
                start,end = parse(r["start_utc"]),parse(r["end_utc"])
                max_gap = max(abs(capture-start),abs(capture-end)).total_seconds()/3600
                if max_gap<=MAX_GAP_HOURS:
                    by_roof[tank].append((r,max_gap))
        if not by_roof:
            continue
        p = packet(base,item,level2,metadata)
        packets.append(p | {field:item["assets"][label]["href"] for field,label in zip(DOWNLOAD_FIELDS,ASSETS,strict=True)})
        # Use the geometry of the independent reference when projecting the roof center.
        geometry = {tank: refs[0][0] for tank,refs in by_roof.items()}
        for tank, refs in by_roof.items():
            if any(tuple(r[k] for k in ("center_east_m","center_north_m")) !=
                   tuple(geometry[tank][k] for k in ("center_east_m","center_north_m")) for r,_ in refs):
                raise ValueError(f"Independent roof centers disagree for {product}/{tank}")
        angles = {r["candidate_id"]:r for r in roof_angles(tool,p,by_roof,geometry,"independent_control_geometry")}
        for tank in sorted(by_roof):
            for r,gap in sorted(by_roof[tank],key=lambda item:item[0]["source_key"]):
                a = angles[tank]
                if a["angle_status"]=="resolved":
                    output.append(dict(r,product_id=product,platform=p["platform"],
                        captured_at_utc=p["captured_at_utc"], roof_type=roofs[tank], max_separation_hours=gap,
                        b8_native_line=a["b8_native_line"],b8_native_sample=a["b8_native_sample"],
                        solar_azimuth_deg=a["solar_azimuth_deg"],solar_zenith_deg=a["solar_zenith_deg"],
                        view_azimuth_deg=a["view_azimuth_deg"],view_zenith_deg=a["view_zenith_deg"]))
    export(f"{STEM}_products",packets)
    export(f"{STEM}_roof_pairs",output)
    counts = Counter((r["product_id"],r["reference_type"]) for r in output)
    queue = []
    for p in packets:
        matches = [r for r in output if r["product_id"]==p["product_id"]]
        queue.append(dict(product_id=p["product_id"],platform=p["platform"],captured_at_utc=p["captured_at_utc"],
                          purpose=matches[0]["reference_type"],external_control_observations=len(matches),
                          contributing_gap_weeks=0,contributing_gap_tank_weeks=0,sentinel2_calibration_candidates=0,
                          **{field:p[field] for field in DOWNLOAD_FIELDS}))
    for p in read_csv("landsat_c2_holdout_gap_download_manifest"):
        queue.append(dict(product_id=p["product_id"],platform=p["platform"],captured_at_utc=p["captured_at_utc"],
                          purpose="gap_pair" if int(p["contributing_gap_weeks"]) else "sentinel2_calibration",
                          external_control_observations=0,
                          **{field:int(p[field]) for field in ("contributing_gap_weeks","contributing_gap_tank_weeks",
                                                               "sentinel2_calibration_candidates")},
                          **{field:p[field] for field in DOWNLOAD_FIELDS}))
    if len({r["product_id"] for r in queue})!=len(queue):
        raise ValueError("Duplicate scene in the merged B8 acquisition queue")
    export("landsat_c2_b8_acquisition_queue",[dict(priority=i,**r) for i,r in enumerate(queue,1)])
    summary = dict(lidar_and_aerial_candidate_products=len(packets),max_match_gap_hours=MAX_GAP_HOURS,
                   merged_acquisition_queue_products=len(queue),
                   control_roof_observations=len(output), product_matches=[
                       dict(product_id=product,reference_type=kind,roof_observations=count,
                            distinct_roofs=len({r["candidate_id"] for r in output if r["product_id"]==product}),
                            roof_types=dict(Counter(r["roof_type"] for r in output if r["product_id"]==product)),
                            max_separation_hours=max(r["max_separation_hours"] for r in output if r["product_id"]==product))
                       for (product,kind),count in sorted(counts.items())],
                   interpretation="Physical control candidates separate from the weekly forecast feature book. "
                                  "Roof levels can move during the gap between acquisitions. "
                                  "No panchromatic roof height has been derived for these versions.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
