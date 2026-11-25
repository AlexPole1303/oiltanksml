"""Measure a new Cushing report interval from native Sentinel-2 L1C pixels."""

import argparse
import json
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from itertools import product
from math import fsum

from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from audit_shadow_registration_pixels import check_source, coverage_rows, requested_patches
from build_l1c_quality import QUALITY_FIELDS, candidate_layer, scene_rows
from build_registered_shadow_development import registered_pair, registered_profile
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_calibration import SCHEMA, add_psf, add_scene
from build_shadow_registration_inputs import PATCH_M, PADDING_M, SEARCH_RADIUS_M, anchor_grid, build_windows, describe_source, registration_sources, tank_points
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from collect_s2_l1c import CHANNELS, OBJECTS, download, parse_product, source_metadata_path
from extract_s2_radiometry import BAND_ID, read_xml
from fit_shadow_geometry import inputs, read_only
from fit_shadow_registration import patch_profiles, tank_offset

STEM = "cushing_forward_native_optical"
STORE = ROOT/"data/rasters/cushing_forward_native_sentinel2"


def stage(tiles, roster, release, store, *, catalog_path=None, object_index_store=None):
    if catalog_path is None:
        summary = json.loads((DERIVED/f"cushing_forward_optical_coverage_{release}_summary.json").read_bytes())
        catalog_path = ROOT/summary["source_catalog"]
    catalog = {r["id"]:r for r in json.loads(catalog_path.read_bytes())["features"]}
    geometry = [shape(transform_geom("EPSG:4326","EPSG:32614",mapping(g)))
                for name,g in (*candidate_layer("osm"),*candidate_layer("supplemental")) if name in roster]
    footprint = unary_union(geometry).buffer(SEARCH_RADIUS_M+PATCH_M+2*PADDING_M)
    bounds = shape(transform_geom("EPSG:32614","EPSG:4326",mapping(footprint))).bounds
    store.mkdir(parents=True,exist_ok=True)

    def load(r):
        version = r["l1c_product"]
        key,_,generation = parse_product(version)
        versions = {key:(generation,version,r["l1c_archive_url"].removeprefix(OBJECTS))}
        options = {"object_index_store":object_index_store} if object_index_store is not None else {}
        outcome = download(catalog[r["item_id"]],versions,store,bounds,footprint,verify=True,pack_text=True,**options)
        if outcome["status"] != "ready":
            raise ValueError(f"Original product not staged for {r['item_id']}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(load,tiles))


def build_packet(path, tiles, roster, store):
    temp = path.with_suffix(".part")
    temp.unlink(missing_ok=True)
    candidates = [(name,g) for name,g in (*candidate_layer("osm"),*candidate_layer("supplemental")) if name in roster]
    points = {name:(g.centroid.x,g.centroid.y) for name,g in candidates}
    footprints = dict(candidates)
    log, lookup = [], {}
    with closing(sqlite3.connect(temp)) as db, closing(sqlite3.connect(":memory:")) as catalog, db:
        db.row_factory = sqlite3.Row
        catalog.row_factory = sqlite3.Row
        db.executescript(SCHEMA)
        add_psf(db)
        catalog.executescript("""CREATE TABLE sentinel2_l1c_early_detector_footprint(scene_item_id,l1c_product_name,local_file);
            CREATE TABLE sentinel2_band_radiometry(level,version_id,band,channel,quantification_value,add_offset_dn,
                reflectance_scale,reflectance_offset,nodata_dn,saturated_dn,metadata_status,product_xml);""")
        for number, r in enumerate(tiles,1):
            item = r["item_id"]
            manifest = store/"granules"/r["captured_at_utc"][:4]/item/"item.json"
            record = json.loads(manifest.read_bytes())
            version = r["l1c_product"]
            if (record["product_name"],record["captured_at_utc"]) != (version,r["captured_at_utc"]):
                raise ValueError(f"Wrong original product staged for {item}")
            xml = source_metadata_path(manifest.parent/"MTD_MSIL1C.xml")
            quant,offsets,nodata,saturation,status = read_xml(xml,"L1C",version)
            catalog.executemany("INSERT INTO sentinel2_band_radiometry VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
                ("L1C",version,band,channel,quant,offsets[BAND_ID[band]],1/quant,offsets[BAND_ID[band]]/quant,
                 nodata,saturation,status,str(xml.relative_to(ROOT))) for channel,band in CHANNELS.items()])
            obs = []
            for row in scene_rows(manifest,candidates,points,{},None):
                q = dict(zip(QUALITY_FIELDS,row,strict=True))
                tank = q["candidate_id"]
                g = roster[tank]
                clear = q["cloud_mask_basis"] != "missing" and all(q[k] == 0 for k in ("opaque_roof_cells","cirrus_roof_cells","snow_ice_roof_cells"))
                complete = q["optical_touched_cells_expected"] == q["optical_cells_valid_in_all_bands"]
                detector = q["red_detector_covered_roof_cells"] == q["optical_touched_cells_expected"] and not q["red_detector_ambiguous_roof_cells"] and len(json.loads(q["red_detector_with_view_options_json"]))==1
                log.append(q | dict(segment=r["segment"],clear=clear,complete=complete,detector=detector,qa_pass=clear and complete and detector))
                if not (clear and complete and detector):
                    continue
                lon,lat = points[tank]
                obs.append(dict(candidate_id=tank, scene_item_id=item, optical_captured_at_utc=r["captured_at_utc"],
                    tile_metadata_path=q["tile_metadata_path"], longitude=lon, latitude=lat, documented_roof_type="EFR",
                    first_seen=g["first_seen"], cohort_id=g["cohort_id"], source_document_date=g["roof_evidence_date"],
                    geometry_basis="dated_2022_lidar", outline_fit_status=g["fit_status"],
                    center_east_m=float(g["center_east_m"]), center_north_m=float(g["center_north_m"]), radius_m=float(g["diameter_m"])/2,
                    optical_cells=q["optical_touched_cells_expected"],valid_cells=q["optical_cells_valid_in_all_bands"],
                    cloud_mask_basis=q["cloud_mask_basis"],red_view_options_json=q["red_detector_with_view_options_json"],
                    resolved_b04_detector=True,roof_clear_by_source_mask=True,sun_zenith_deg=q["sun_zenith_deg"],
                    sun_azimuth_grid_deg=q["sun_azimuth_grid_deg"]))
            if obs:
                folder = str(manifest.parent.relative_to(ROOT))
                add_scene(db,catalog,number,("L1C",version,folder),obs,footprints,lookup)
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Forward optical packet integrity check failed")
    temp.replace(path)
    return log


def measure(path, tiles, roster, release, *, segments=("preceding_week","target_week")):
    by_scene = {r["item_id"]:r for r in tiles}
    with closing(read_only(path)) as db:
        db.row_factory = sqlite3.Row
        rows,kernels = inputs(db)
        sources = {r["scene_id"]:describe_source(r["scene_id"],r) for r in registration_sources(db)}
    ends = defaultdict(lambda:defaultdict(list))
    observations = {}
    for r in rows:
        scene,tank = r["scene_id"],r["candidate_id"]
        if tank not in roster or sources[scene]["canonical_l2a_id"] not in by_scene:
            continue
        tile = by_scene[sources[scene]["canonical_l2a_id"]]
        observations[scene,tank] = r
        ends[tank][tile["segment"]].append(scene)
    jobs = []
    for tank, by_segment in sorted(ends.items()):
        for first,last in product(*(by_segment[name] for name in segments)):
            a,b = sources[first],sources[last]
            if a["mgrs_tile"] != b["mgrs_tile"]:
                continue
            at,bt = [by_scene[s["canonical_l2a_id"]]["l1c_with_detector_available_by_utc"] for s in (a,b)]
            jobs.append(("weekly", "forward", release["latest_week_in_issue"], tank, "EFR", roster[tank]["terminal_id"],
                         release["conservative_prerelease_before_utc"], at,bt,a,b))
    pairs,windows,patches,_ = build_windows(jobs,anchor_grid(*tank_points()))
    pairs = {r["scene_pair_id"]:r for r in pairs}
    patches = {r["anchor_id"]:r for r in patches}
    measured = {(r["source_id"],r["anchor_id"]):r for key, ids in requested_patches(windows,pairs,patches).items() if ids
                for r in check_source(sources[key],[patches[anchor] for anchor in sorted(ids)])}
    coverage,_ = coverage_rows(windows,pairs,measured)
    profiles,_ = patch_profiles(coverage,sources,pairs,patches)
    for window in coverage:
        shift = tank_offset(window,profiles)
        pair = pairs[window["scene_pair_id"]]
        tank = window["candidate_id"]
        first,last = [observations[pair[f"{k}_source_id"],tank] for k in ("earlier","later")]
        g = roster[tank]
        area = float(g["circular_area_m2"])
        result = shift | dict(terminal_id=g["terminal_id"],earlier_version=first["version_id"],later_version=last["version_id"],
                              earlier_capture=first["captured_at_utc"],later_capture=last["captured_at_utc"], area_m2=area)
        fit = dict(registered_both_shadow_diagnostics_pass=False, registered_change_m=None,
                   registered_loss_range_min_m=None, registered_loss_range_max_m=None)
        if shift["registration_status"] == "checked_offset":
            shell = float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"])
            x,y = (float(shift[f"later_minus_earlier_{axis}_m"])/2 for axis in ("east","north"))
            fit.update(registered_pair(registered_profile(path,first,None,shell,kernels[first["platform"]],-x,-y),
                                       registered_profile(path,last,None,shell,kernels[last["platform"]],x,y)))
        yield result | fit | {f"{k}_kbbl":fit[f"{k}_m"]*area/BARREL_M3/1000 if fit[f"{k}_m"] is not None else None
                              for k in ("registered_change","registered_loss_range_min","registered_loss_range_max")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",required=True)
    parser.add_argument("--store",type=Path,default=STORE,help="Source clip store inside the workspace")
    args = parser.parse_args()
    store = ROOT/args.store
    if not store.is_relative_to(ROOT):
        parser.error("Keep source clip paths inside the workspace")
    releases = {r["release_date"]:r for r in read("eia_wpsr_forward_releases")}
    if args.release in releases:
        release = releases[args.release]
    else:
        coverage = json.loads((DERIVED/f"cushing_forward_optical_coverage_{args.release}_summary.json").read_bytes())
        if coverage["release_mode"]!="upcoming" or coverage["release_date"]!=args.release:
            raise ValueError("Missing verified optical window for the upcoming report")
        release = dict(latest_week_in_issue=coverage["target_week_end_utc"][:10],
                       previous_week_ending=coverage["preceding_week_end_utc"][:10],
                       conservative_prerelease_before_utc=coverage["forecast_cutoff_utc"])
    cutoff = datetime.fromisoformat(release["conservative_prerelease_before_utc"])
    tiles = [r for r in read(f"cushing_forward_optical_coverage_{args.release}_tiles") if r["segment"] in ("preceding_week","target_week")
             and r["l1c_pre_release"]=="True" and int(r["clear_roofs"])>0 and datetime.fromisoformat(r["l1c_with_detector_available_by_utc"])<cutoff]
    roster = {r["candidate_id"]:r for r in read("cushing_modern_lidar_optical_registry") if r["modern_input_status"]=="available"
              and r["latest_engineering_date"] < cutoff.date().isoformat() and r["first_seen"] <= release["previous_week_ending"]}
    stem = f"{STEM}_{args.release}"
    path = DERIVED/f"{stem}.sqlite"
    stage(tiles,roster,args.release,store)
    quality = build_packet(path,tiles,roster,store)
    rows = list(measure(path,tiles,roster,release))
    unique = {}
    for r in sorted((r for r in rows if r["registered_both_shadow_diagnostics_pass"]), key=lambda r: (
            r["registered_loss_range_max_m"]-r["registered_loss_range_min_m"],r["earlier_version"],r["candidate_id"])):
        unique.setdefault(r["candidate_id"],r)
    pair_path = DERIVED/f"{stem}_pairs.csv"
    if rows:
        fields = dict.fromkeys(k for r in rows for k in r)
        export(f"{stem}_pairs",[fields | r for r in rows])
    else:
        pair_path.unlink(missing_ok=True)
    export(f"{stem}_source_quality",quality)
    report = dict(release_date=args.release, week_ending=release["latest_week_in_issue"],outcome_available=args.release in releases,
                  observation_cutoff=release["conservative_prerelease_before_utc"],
                  native_products=len(tiles), dated_roofs=len(roster), same_tile_pairs=len(rows),
                  registration_statuses=Counter(r["registration_status"] for r in rows), accepted_roofs=len(unique),
                  shadow_endpoint_statuses=Counter(r[k] for r in rows for k in ("earlier_registered_status","later_registered_status") if k in r),
                  accepted_ids=sorted(unique), accepted_terminals=len({r["terminal_id"] for r in unique.values()}),
                  area_m2=fsum(r["area_m2"] for r in unique.values()),
                  dated_roof_area_m2=fsum(float(r["circular_area_m2"]) for r in roster.values()),
                  area_fraction=fsum(r["area_m2"] for r in unique.values())/fsum(float(r["circular_area_m2"]) for r in roster.values()),
                  measured_partial_change_kbbl=fsum(r["registered_change_kbbl"] for r in unique.values()) if unique else None,
                  scope="Partial cylindrical change between exact captures over roofs with dated LiDAR and checked exterior registration.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{stem}_summary.json",payload.encode())
    print(payload,end="")


if __name__=="__main__":
    main()
