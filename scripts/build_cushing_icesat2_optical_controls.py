"""Rebuild optical roof controls around ICESat-2 captures using original pixels."""
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timedelta
from urllib.parse import urlencode

import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Point, shape

from audit_landsat_c2_alternative_access import probe
from audit_s2_sun_clock import xml_time
from audit_shadow_precision import describe
from build_cross_sensor_shadow import PAN_SCHEMA, add_pan_product, fit_pan_samples, pan_psf, published_lsf
from build_landsat_c2_b8_measurements import roof_touched
from build_landsat_c2_holdout_gap_inventory import level1_generation
from build_landsat_pan_roof_angles import angle_values, native_origin, prepare_tool
from build_radar_optical_cushing_controls import build_pixels, optical_inventory
from build_shadow_change_controls import SELECT_QUALITY
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, read_only
from collect_public import EARTH_SEARCH, RAW, ROOT, get, write, write_csv
from collect_landsat import BBOX as LANDSAT_BBOX, USGS
from collect_landsat_c2_gap_geometry import packet, read_azure_asset, roof_angles
from collect_landsat_asset_times import asset_snapshot
from collect_s2_l1c import parse_product, source_metadata_path
from fit_shadow_geometry import estimate, inputs
from stage_landsat_c2_b8_inputs import expected_sha512

STEM = "cushing_icesat2_optical"
STORE = RAW / STEM
S2_DB = DERIVED / f"{STEM}_sentinel2.sqlite"
L8_DB = DERIVED / f"{STEM}_landsat.sqlite"
CDSE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
BOUNDS = (-96.769,35.923,-96.713,36.023)
HOURS = 48
TO_GEO = Transformer.from_crs(32614,4326,always_xy=True)


def profiles():
    groups = defaultdict(dict)
    for r in read_csv("cushing_icesat2_roof_profiles"):
        if r["passes_roof_profile_checks"]!="True":
            continue
        tank = r["candidate_id"]
        group = groups[r["filename"]]
        if tank in group:
            raise ValueError(f"Multiple laser profiles for {tank} in {r['filename']}")
        group[tank] = r
    return groups


def clock(r):
    return tuple(datetime.fromisoformat(r[k]) for k in ("granule_start_utc","granule_end_utc"))


def separation(r,t):
    return max(abs((t-edge).total_seconds()) for edge in clock(r))/3600


def datatake(version):
    (platform,sensing,orbit,_),_,_ = parse_product(version)
    return "_".join((platform,sensing,orbit))


def query_catalog(filename,group,*,store=STORE,bounds=BOUNDS):
    start,end = clock(next(iter(group.values())))
    first = (start-timedelta(hours=HOURS)).isoformat().replace("+00:00","Z")
    last = (end+timedelta(hours=HOURS)).isoformat().replace("+00:00","Z")
    w,s,e,n = bounds
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    filter_text = ("Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') "
                   f"and ContentDate/Start ge {first} and ContentDate/Start le {last} "
                   f"and OData.CSC.Intersects(area=geography'SRID=4326;{poly}')")
    queries = (
        ("copernicus",CDSE,{"$filter":filter_text,"$top":100}),
        ("earth_search",EARTH_SEARCH,dict(collections="sentinel-2-l1c",limit=100,
            bbox=",".join(map(str,bounds)),datetime=f"{first}/{last}")))
    results = {}
    fields = {}
    for provider,endpoint,params in queries:
        url = endpoint+"?"+urlencode(params)
        path = store / provider / f"{filename}.json"
        raw = get(url,path)
        data = json.loads(raw)
        if provider=="copernicus":
            if "@odata.nextLink" in data:
                raise ValueError(f"Incomplete Copernicus search for {filename}")
            versions = {x["Name"] for x in data["value"]}
        else:
            if data["context"]["matched"]!=len(data["features"]):
                raise ValueError(f"Incomplete Earth Search response for {filename}")
            versions = {x["properties"]["s2:product_uri"] for x in data["features"]}
        results[provider] = {datatake(v) for v in versions}
        fields.update({f"{provider}_{k}":v for k,v in dict(url=url,snapshot=str(path.relative_to(ROOT)),
            sha256=hashlib.sha256(raw).hexdigest(),versions_json=json.dumps(sorted(versions)),
            datatakes_json=json.dumps(sorted(results[provider]))).items()})
    return dict(icesat_granule=filename,laser_profiles=len(group),search_start_utc=first,search_end_utc=last,
        archives_agree=results["copernicus"]==results["earth_search"],**fields), results["copernicus"]|results["earth_search"]


def shared_geometry(r):
    return dict(center_east_m=float(r["geometry_center_east_m"]),center_north_m=float(r["geometry_center_north_m"]),
                radius_m=float(r["geometry_radius_m"]))


def shell(r,geometry):
    dataset,height = geometry[r["candidate_id"]]
    if dataset!=r["geometry_dataset"]:
        raise ValueError(f"Different engineering geometry for {r['candidate_id']}")
    return height


def shell_heights():
    from build_weekly_shadow_validation import legacy_geometry

    modern = {r["candidate_id"]:r for r in read_csv("cushing_modern_lidar_optical_registry")
              if r["modern_input_status"]=="available"}
    return {k:(r["dataset"],float(r["prior_rim_z_p95_native"])-float(r["prior_ground_z_p50_native"]))
            for k,r in (modern | legacy_geometry()).items()}


def base_row(p,sensor,version,captured,archived,directory,metadata):
    return dict(icesat_granule=p["filename"],icesat_cmr_id=p["cmr_id"],icesat_beam=p["beam"],
        candidate_id=p["candidate_id"],terminal_id=p["terminal_id"],icesat_granule_start_utc=p["granule_start_utc"],
        icesat_granule_end_utc=p["granule_end_utc"],icesat_wall_to_roof_m=float(p["estimated_wall_to_roof_m"]),
        optical_sensor=sensor,optical_product=version,optical_capture_utc=captured.isoformat(),
        optical_available_at_utc=archived,optical_source_directory=directory,optical_native_metadata=metadata,
        max_separation_hours=separation(p,captured),geometry_dataset=p["geometry_dataset"],
        **shared_geometry(p))


def sentinel_jobs(db,groups,catalogs):
    scene_inventory = optical_inventory(db)
    candidates,inputs_by_scene = [],defaultdict(dict)
    by_datatake = defaultdict(list)
    for version,entry in scene_inventory.items():
        by_datatake[datatake(version)].append((version,entry))
    for filename,group in groups.items():
        for capture in sorted(catalogs[filename]):
            for version,entry in by_datatake[capture]:
                for item in db.execute(SELECT_QUALITY+"AND q.l1c_product_name=? AND q.scene_item_id=? ORDER BY q.candidate_id",
                                       (version,entry["canonical_l2a_id"])):
                    r = dict(item)
                    tank = r["candidate_id"]
                    if tank not in group:
                        continue
                    p = group[tank]
                    meta = str(source_metadata_path(ROOT/r["tile_metadata_path"]).relative_to(ROOT))
                    captured = xml_time(meta)
                    if separation(p,captured)>HOURS:
                        continue
                    source = r["optical_source_directory"]
                    passed = bool(r["roof_clear_by_source_mask"] and r["resolved_b04_detector"] and r["valid_cells"]==r["optical_cells"])
                    if passed:
                        inputs_by_scene["L1C",version,source][tank] = dict(r,optical_captured_at_utc=captured.isoformat())
                    candidates.append(dict(base_row(p,"sentinel2_b04",version,captured,entry["geometry_available_at_utc"],
                        source,meta),optical_datatake=capture,source_qa_pass=passed,valid_roof_cells=r["valid_cells"],
                        total_roof_cells=r["optical_cells"],source_single_detector=bool(r["resolved_b04_detector"])))
    return candidates,inputs_by_scene


def sentinel_fits(candidates,geometry):
    by_key = {(r["candidate_id"],r["optical_product"],r["optical_source_directory"]):r
              for r in candidates if r["source_qa_pass"]}
    with closing(read_only(S2_DB)) as db:
        db.row_factory = sqlite3.Row
        observations,kernels = inputs(db)
        directories = dict(db.execute("SELECT scene_id,source_directory FROM scene"))
    for o in observations:
        key = o["candidate_id"],o["version_id"],directories[o["scene_id"]]
        r = by_key[key]
        same_geometry = dict(o,**{field:r[field] for field in ("center_east_m","center_north_m","radius_m")})
        yield key,estimate(same_geometry,shell(r,geometry),kernels[o["platform"]],calibration_db=S2_DB)


def landsat_jobs(db,groups):
    products = db.execute("""SELECT product_id,captured_at_utc,source_directory FROM landsat_local_product
                            WHERE source_kind='gcs_collection1_level1_pan' ORDER BY captured_at_utc""").fetchall()
    angles = {r["product_id"]:r for r in read_csv("landsat_pan_angle_assets")}
    candidates,plans = [],defaultdict(dict)
    tool = None
    for product,stamp,directory in products:
        t = datetime.fromisoformat(stamp)
        relevant = [p for g in groups.values() for p in g.values() if separation(p,t)<=HOURS]
        if not relevant:
            continue
        a = angles[product]
        angle_file = ROOT/a["local_file"]
        root = ROOT/directory
        if datetime.fromisoformat(a["capture_at_utc"])!=t or angle_file.parent!=root:
            raise ValueError(f"Landsat B8 source and angle identities differ for {product}")
        tool = tool or prepare_tool()
        points = {}
        with rasterio.open(root/"pan.tif") as src:
            x0,y0 = native_origin(root,angle_file)
            transformer = Transformer.from_crs(32614,src.crs,always_xy=True)
            for p in relevant:
                g = shared_geometry(p)
                x,y = transformer.transform(g["center_east_m"],g["center_north_m"])
                row,col = src.index(x,y)
                if 0<=row<src.height and 0<=col<src.width:
                    x,y = src.xy(row,col)
                    points[p["candidate_id"]] = round((y0-y)/15),round((x-x0)/15)
        values = angle_values(tool,angle_file,points.values())
        for p in relevant:
            tank = p["candidate_id"]
            point = points.get(tank)
            native = values.get(point,{})
            valid = native.get("angle_status")=="resolved"
            r = dict(base_row(p,"landsat8_b8",product,t,a["geometry_inputs_available_at_utc"],directory,a["local_file"]),
                     optical_datatake=product,source_qa_pass=valid,valid_roof_cells=None,total_roof_cells=None,
                     source_single_detector=None)
            candidates.append(r)
            if valid:
                plans[product][tank] = dict(r,landsat_source_directory=directory,landsat_angle_file=a["local_file"],
                    landsat_capture_at_utc=stamp,landsat_inputs_available_at_utc=a["geometry_inputs_available_at_utc"],
                    landsat_b8_native_line=point[0],landsat_b8_native_sample=point[1],roof_type="EFR",
                    geometry_basis=p["geometry_dataset"],
                    **{f"landsat_b8_{k}":v for k,v in native.items()})
    return candidates,plans,tool


def landsat_collection2_targets(db,groups):
    products = list(db.execute("""SELECT product_id,captured_at_utc,source_directory
                                  FROM landsat_local_product WHERE source_kind='pc_collection2_level2_context'
                                  ORDER BY captured_at_utc"""))
    for local_id,stamp,directory in products:
        t = datetime.fromisoformat(stamp)
        relevant = [p for group in groups.values() for p in group.values() if separation(p,t)<=HOURS]
        if not relevant:
            continue
        source = ROOT/directory
        metadata_path = str((source/"MTL.json").relative_to(ROOT))
        product,generated = level1_generation(metadata_path)
        day = t.date().isoformat()
        query = urlencode(dict(collections="landsat-c2l1",bbox=LANDSAT_BBOX,limit=100,
            datetime=f"{day}T00:00:00Z/{day}T23:59:59Z"))
        path = STORE / "landsatlook" / f"area_{day}.json"
        raw = get(USGS+query,path)
        listing = json.loads(raw)
        if listing["numberMatched"]!=len(listing["features"]):
            raise ValueError(f"Incomplete LandsatLook results for {day}")
        item, = [i for i in listing["features"] if i["id"]==product]
        pan = item["assets"]["pan"]
        native_time = datetime.fromisoformat(item["properties"]["datetime"])
        if abs((native_time-t).total_seconds())>1 or pan["eo:bands"][0]["gsd"]!=15:
            raise ValueError(f"Landsat Level 1 and Level 2 differ for {local_id}")
        footprint = shape(item["geometry"])
        source_qa = {r[0]:r[1:] for r in db.execute("""SELECT candidate_id,qa_roof_cells,valid_qa_roof_cells,
            clear_qa_roof_cells,signal_cells,valid_signal_cells FROM landsat_tank_scene_quality WHERE product_id=?""",(local_id,))}
        for p in relevant:
            if not footprint.covers(Point(*TO_GEO.transform(float(p["geometry_center_east_m"]),float(p["geometry_center_north_m"])))):
                continue
            tank = p["candidate_id"]
            q = source_qa.get(tank)
            passed = bool(q and 0<q[0]==q[1]==q[2] and 0<q[3]==q[4])
            yield dict(icesat_granule=p["filename"],icesat_cmr_id=p["cmr_id"],candidate_id=tank,
                terminal_id=p["terminal_id"],product_id=product,level2_product_id=local_id,landsat_capture_at_utc=native_time,
                max_separation_hours=separation(p,native_time),native_level2_qa_roof_cells=q[0] if q else None,
                native_level2_clear_roof_cells=q[2] if q else None,native_level2_roof_qa_pass=passed,
                b8_asset=pan["href"],b8_sha512=expected_sha512(pan),
                source_level1_generated_utc=generated,
                **shared_geometry(p),
                level2_qa_pixel_file=str((source/"qa_pixel.tif").relative_to(ROOT)),
                level2_qa_radsat_file=str((source/"qa_radsat.tif").relative_to(ROOT)),
                level2_mtl_path=metadata_path,landsatlook_catalog=str(path.relative_to(ROOT)),
                landsatlook_catalog_sha256=hashlib.sha256(raw).hexdigest())


def collection2_geometry(rows,tool):
    groups = defaultdict(dict)
    for r in rows:
        group = groups[r["product_id"]]
        group[r["candidate_id"]] = r
    geometry,angles = [],[]
    for product,group in sorted(groups.items()):
        row = next(iter(group.values()))
        catalog = json.loads((ROOT/row["landsatlook_catalog"]).read_bytes())
        item, = [i for i in catalog["features"] if i["id"]==product]
        metadata = ROOT/row["level2_mtl_path"]
        l2 = json.loads((metadata.parent/"item.json").read_bytes())
        recorded = dict(product_id=product,platform=item["properties"]["platform"],
            level1_generated_at_utc=row["source_level1_generated_utc"].isoformat(),
            captured_at_utc=row["landsat_capture_at_utc"].isoformat())
        source = packet(recorded,item,l2,metadata,store=STORE/"landsat_c2_geometry")
        directory = (ROOT/source["angle_local_file"]).parent
        evidence = directory/"level2_qa_archive.json"
        original_qa = asset_snapshot(l2,("qa_pixel","qa_radsat"),evidence)["assets"]
        for label in ("qa_pixel","qa_radsat"):
            path = directory/f"{label}.tif"
            payload = read_azure_asset(original_qa[label],path)
            digest = hashlib.sha512(payload).hexdigest()
            if digest!=expected_sha512(item["assets"][label]):
                raise ValueError(f"Uncropped Level-2 QA differs from USGS Level-1 {label} for {product}")
            source.update({f"{label}_file":str(path.relative_to(ROOT)),f"{label}_sha512":digest})
        if hashlib.sha512((ROOT/source["angle_local_file"]).read_bytes()).hexdigest()!=expected_sha512(item["assets"]["ANG.txt"]):
            raise ValueError(f"Angle copy differs from USGS Level-1 for {product}")
        source["qa_archive_snapshot"] = str(evidence.relative_to(ROOT))
        geometry.append(source)
        for result in roof_angles(tool,source,group,group,"matched_icesat2_lidar_circle"):
            angles.append(dict(result,icesat_granule=group[result["candidate_id"]]["icesat_granule"]))
    return geometry,angles


def landsat_fits(source,plans,tool,geometry):
    if not plans:
        return {}
    lsf = np.asarray(published_lsf(),dtype=float)
    tmp = L8_DB.with_suffix(".part")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(tmp)) as db,db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(PAN_SCHEMA.split("CREATE TABLE sensor_match")[0])
        db.executemany("INSERT INTO pan_lsf VALUES (?,?,?)",lsf)
        identities = {}
        for product,group in sorted(plans.items()):
            centers = {}
            for tank,r in group.items():
                lon,lat = TO_GEO.transform(r["center_east_m"],r["center_north_m"])
                centers[tank] = dict(r,longitude=lon,latitude=lat)
            add_pan_product(db,source,tool,product,list(group.values()),centers,
                {k:shell(r,geometry) for k,r in group.items()},identities)
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Landsat B8 control pixel database failed validation")
    tmp.replace(L8_DB)
    result = {}
    with closing(read_only(L8_DB)) as db:
        db.row_factory = sqlite3.Row
        obs = db.execute("""SELECT o.*,p.captured_at_utc,p.along_track_grid_bearing_deg,p.source_directory
                            FROM pan_observation o JOIN pan_product p USING(product_id)""").fetchall()
        for r in obs:
            m = np.asarray(db.execute("""SELECT native_line,native_sample,center_east_m,center_north_m,b8_toa_reflectance,
                all_touching_bqa_valid,touching_bqa_flag_bits,view_azimuth_true_deg,view_zenith_deg,
                solar_azimuth_true_deg,solar_zenith_deg FROM pan_pixel_reflectance
                WHERE pan_observation_id=? ORDER BY native_line,native_sample""",(r["pan_observation_id"],)).fetchall(),dtype=float)
            interior = roof_touched(m[:,2],m[:,3],r["center_east_m"],r["center_north_m"],r["radius_m"])
            roof_clear = bool(interior.any() and np.all(np.isfinite(m[interior,4])&(m[interior,5]==1)&(m[interior,6]==0)))
            key = r["candidate_id"],r["product_id"],r["source_directory"]
            fit = fit_pan_samples(r,m,pan_psf(lsf,r["along_track_grid_bearing_deg"])) if roof_clear else {}
            result[key] = dict(fit,fit_status=fit.get("fit_status","native_bqa_or_b8_roof_failed"),source_qa_pass=roof_clear,
                total_roof_cells=int(interior.sum()),valid_roof_cells=int((interior & np.isfinite(m[:,4])
                    &(m[:,5]==1)&(m[:,6]==0)).sum()))
    return result


def compared(candidates,fits):
    for r in candidates:
        fit = fits.get((r["candidate_id"],r["optical_product"],r["optical_source_directory"]),{})
        gap = fit.get("estimated_gap_m")
        qa = {k:fit.get(k,r[k]) for k in ("source_qa_pass","total_roof_cells","valid_roof_cells")}
        yield dict(r,**qa,
            fit_status=fit.get("fit_status","source_quality_gate_failed"),optical_gap_m=gap,
            optical_minus_laser_m=gap-r["icesat_wall_to_roof_m"] if gap is not None else None,
            **{k:fit.get(k) for k in ("observation_id","fitted_pixels","sun_elevation_deg","view_zenith_deg",
                "near_optimal_gap_min_m","near_optimal_gap_max_m","pixel_rmse","flat_rmse","reversed_sun_rmse")})


def primary(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["fit_status"]==RESOLVED:
            groups[r["icesat_granule"],r["candidate_id"],r["icesat_beam"]].append(r)
    for group in groups.values():
        # Choose the shortest time gap among passing fits without comparing the heights.
        yield min(group,key=lambda r:(r["max_separation_hours"],r["optical_available_at_utc"] or "",
            -(r["valid_roof_cells"] or 0),r["optical_product"],r["optical_source_directory"]))


def export(suffix,rows):
    fields = tuple(dict.fromkeys(k for r in rows for k in r))
    write_csv(DERIVED/f"{STEM}_{suffix}.csv",fields,([r.get(k) for k in fields] for r in rows))


def main():
    groups = profiles()
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = dict(zip(groups,pool.map(lambda g:query_catalog(*g),groups.items()),strict=True))
    inventory = [r for r,_ in results.values()]
    catalogs = {k:passes for k,(_,passes) in results.items()}
    geometry = shell_heights()
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        s2,source_rows = sentinel_jobs(db,groups,catalogs)
        build_pixels(db,source_rows,S2_DB)
        l8,plans,tool = landsat_jobs(db,groups)
        fits = dict(sentinel_fits(s2,geometry)) | landsat_fits(db,plans,tool,geometry)
        collection2 = list(landsat_collection2_targets(db,groups))
    known_captures = {tuple(p.split("_")[i] for i in (0,2,3)) for p in plans}
    missing = [r for r in collection2 if r["native_level2_roof_qa_pass"]
               and tuple(r["product_id"].split("_")[i] for i in (0,2,3)) not in known_captures]
    urls = {r["product_id"]:r["b8_asset"] for r in missing}
    with ThreadPoolExecutor(max_workers=4) as pool:
        probes = [dict(product_id=p,**result) for p,result in zip(urls,pool.map(probe,urls.values()),strict=True)]
    c2_geometry,c2_angles = collection2_geometry(missing,tool or prepare_tool()) if missing else ([],[])
    candidates = list(compared([*s2,*l8],fits))
    selected = list(primary(candidates))
    earlier = read_csv("cushing_icesat2_sentinel2_controls")
    old = {(r["icesat_granule"],r["candidate_id"],r["beam"]):float(r["max_time_separation_hours"]) for r in earlier}
    for r in selected:
        key = r["icesat_granule"],r["candidate_id"],r["icesat_beam"]
        r["prior_timing_match_hours"] = old.get(key)
        r["timing_improvement_hours"] = old[key]-r["max_separation_hours"] if key in old else None
    for kind,rows in (("catalog",inventory),("candidates",candidates),("selected",selected)):
        export(kind,rows)
    if collection2:
        export("landsat_c2_targets",collection2)
    if probes:
        export("landsat_c2_access",probes)
    if c2_geometry:
        export("landsat_c2_geometry",c2_geometry)
        export("landsat_c2_angles",c2_angles)
    report = dict(laser_granules=len(groups),catalog_agreements=sum(r["archives_agree"] for r in inventory),
        no_sentinel2_datatake_within_search_window=sum(not passes for passes in catalogs.values()),
        source_candidates=dict(Counter(r["optical_sensor"] for r in candidates)),
        fit_status_by_sensor={sensor:dict(Counter(r["fit_status"] for r in candidates if r["optical_sensor"]==sensor))
                              for sensor in sorted({r["optical_sensor"] for r in candidates})},
        selected_control_profiles=len(selected),selected_within_24h=sum(r["max_separation_hours"]<=24 for r in selected),
        selected_new_profiles=sum(r["prior_timing_match_hours"] is None for r in selected),
        selected_with_shorter_gap=sum(r["timing_improvement_hours"]>0.01 for r in selected if r["timing_improvement_hours"] is not None),
        selected_sensors=dict(Counter(r["optical_sensor"] for r in selected)),
        landsat_c2_products=len({r["product_id"] for r in collection2}),
        landsat_c2_roof_candidates_with_level2_qa=sum(r["native_level2_roof_qa_pass"] for r in collection2),
        landsat_c2_target_products_without_local_collection1=len(urls),
        landsat_c2_public_tiff_probes=sum(r["tiff_signature"] for r in probes),
        landsat_c2_missing_b8_roofs_with_resolved_angles=sum(r["angle_status"]=="resolved" for r in c2_angles),
        disagreement_within_48h=describe([r["optical_minus_laser_m"] for r in selected]),
        disagreement_within_24h=describe([r["optical_minus_laser_m"] for r in selected if r["max_separation_hours"]<=24]),
        interpretation="Retrospective controls. Disagreements include possible actual roof motion. No EIA targets or market returns used.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
