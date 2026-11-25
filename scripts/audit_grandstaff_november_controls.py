"""Recover retrospective optical and laser controls for three Grandstaff inspections."""

import hashlib
import json
import sqlite3
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode, urlsplit
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq
import rasterio
from affine import Affine
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point

from build_cross_sensor_shadow import PAN_PIXEL, PAN_SHIFTS, PAN_STEP, pan_qa, pan_psf, published_lsf
from build_cushing_inspection_b8 import NATIVE, reflection_window
from build_cushing_nasa_satellite_inventory import CMR, probe
from build_landsat_c2_b8_measurements import roof_touched
from build_s1_reference_matches import DERIVED, export, read
from build_sentinel2_angles import ground_coordinates
from build_weekly_shadow_panel import RESOLVED, read_only
from collect_hls_landsat_gaps import HLS, day_catalog, original_tags
from collect_imagery import PC
from collect_landsat import LANDSAT, catalog
from collect_landsat_esri_b8 import query
from collect_okeis_inspection_remote_controls import CMR_COLLECTIONS
from collect_public import RAW, ROOT, get, write
from fit_shadow_geometry import bearing_vector, geometry_fit

STEM = "cushing_grandstaff_november_controls"
STORE = RAW/"cushing_grandstaff_november"
PAN_STORE = LANDSAT/STEM
LOCAL = ZoneInfo("America/Chicago")
PAN = "LC09_L1TP_027035_20251112_20251112_02_T1"
LATER_PAN = "LC09_L1TP_028035_20251119_20251119_02_T1"
HLS_BANDS = ("Fmask", "SAA", "SZA", "VAA", "VZA")
USGS_ANCILLARY = {"ANG.txt": "ang", "qa_pixel": "qa_pixel", "qa_radsat": "qa_radsat"}


def inputs():
    links = read("cushing_phmsa_controls_ground_links")
    candidates = {r["preferred_candidate_from_incident_point"]: r for r in links}
    if {r["equipment_id"] for r in links} != {"2514", "2522", "2525"} or len(candidates) != 3:
        raise ValueError("Missing the three independently preferred Grandstaff contours")
    geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry") if r["candidate_id"] in candidates}
    if geometry.keys() != candidates.keys() or any(r["modern_input_status"] != "available" for r in geometry.values()):
        raise ValueError("Missing dated LiDAR geometry")
    return candidates, geometry


def roof_geometry(row):
    x, y, diameter, rim, ground = (float(row[k]) for k in ("center_east_m", "center_north_m", "diameter_m", "prior_rim_z_p95_native", "prior_ground_z_p50_native"))
    return x, y, diameter/2, rim-ground


def pan_source(candidates, official):
    url, raw, entries = query(f"Name = '{PAN}'")
    item, = entries
    if item["LANDSAT_PRODUCT_ID"] != PAN or item["LANDSAT_SCENE_ID"] != official["properties"]["landsat:scene_id"]:
        raise ValueError("Esri B8 product differs from exact USGS scene")
    write(PAN_STORE/"esri_catalog.json", raw)
    points = {r["candidate_id"]: (float(r["longitude"]), float(r["latitude"])) for r in read("tank_engineering_registry") if r["candidate_id"] in candidates}
    tile, overlap = reflection_window(PAN, official, item, points.keys(), points, None, [], store=PAN_STORE)
    return tile | overlap | dict(esri_catalog_url=url)


def ancillary_sources(official):
    usgs_path = STORE/"usgs_level2.json"
    earth_search_path = STORE/"landsat_c2_e84.json"
    usgs, = json.loads(usgs_path.read_bytes())["features"]
    earth_search, = json.loads(earth_search_path.read_bytes())["features"]
    if (usgs["id"] != "LC09_L2SP_027035_20251112_20251115_02_T1_SR"
            or earth_search["id"] != "LC09_L2SP_027035_20251112_02_T1"):
        raise ValueError("Expected both Level-2 catalogs to name the November 12 Landsat scene")
    rows = []
    jobs = []
    for key, mirror_key in USGS_ANCILLARY.items():
        l1, l2 = official["assets"][key], usgs["assets"][key]
        checksum = l1["file:checksum"]
        if len(checksum) != 132 or not checksum.startswith("1340") or l2["file:checksum"] != checksum:
            raise ValueError(f"Catalogs disagree on the SHA-512 for {key}")
        s3 = urlsplit(earth_search["assets"][mirror_key]["href"])
        if s3.scheme != "s3" or earth_search["assets"][mirror_key]["href"] != l2["alternate"]["s3"]["href"]:
            raise ValueError(f"Earth Search does not point to the exact USGS Level-2 object for {key}")
        rows.append(dict(asset=key, usgs_level1_id=official["id"], usgs_level2_id=usgs["id"],
                         level1_url=l1["href"], level2_url=l2["href"], sha512=checksum[4:],
                         level2_catalog_file=str(usgs_path.relative_to(ROOT)),
                         earth_search_catalog_file=str(earth_search_path.relative_to(ROOT)),
                         earth_search_asset_url=earth_search["assets"][mirror_key]["href"],
                         exact_bytes_verified=False))
        jobs.extend((key, route, url) for route, url in (
            ("usgs_level1", l1["href"]), ("usgs_level2", l2["href"]),
            ("earth_search_requester_pays", f"https://{s3.netloc}.s3.us-west-2.amazonaws.com{s3.path}")))
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda job: probe("Landsat 9", job[0], job[2]), jobs))
    fields = dict.fromkeys(k for result in results for k in result)
    probes = [dict(route=route) | fields | result for (_, route, _), result in zip(jobs, results, strict=True)]
    if any(row["file_signature"] or (row["http_status"] in (200,206) and row["content_type"] != "text/html")
           for row in probes):
        raise ValueError("An original ancillary object may be downloadable. Inspect and verify its bytes.")
    pc_path = STORE/"pc_level2.json"
    pc = json.loads(pc_path.read_bytes())
    if not any(link["href"].startswith(PC) for link in pc["links"]):
        raise ValueError("Unexpected Planetary Computer catalog")
    return rows, probes, pc["numberReturned"]


def pixel_coordinates(raster, e, n, radius):
    rr, cc = raster.index(e, n)
    rows, cols = np.indices((11, 11)).reshape(2, -1)
    x, y = raster.transform*(cc-5+cols+.5, rr-5+rows+.5)
    roof = roof_touched(x, y, e, n, radius)
    return rr, cc, rows, cols, x, y, roof


def hls_fits(candidates, geometry, tile, official):
    _, _, items = day_catalog(("2025-11-12", "pc"))
    items = [i for i in items if i["id"].split(".")[2] in {"T14SPE", "T14SQE"}]
    if len(items) != 2:
        raise ValueError("Expected two HLS tiles for the same Landsat pass")
    source_path = ROOT/tile["local_path"]
    results, pixels, sources = [], [], []
    lsf = published_lsf()
    # The exact orbit heading requires the original ANG file. Average the published
    # response over evenly spaced bearings for this diagnostic.
    kernel = np.mean([pan_psf(lsf, bearing) for bearing in range(0, 180, 15)], axis=0)
    to_lon_lat = Transformer.from_crs(32614,4326,always_xy=True)
    with rasterio.open(source_path) as pan:
        if pan.crs.to_epsg() != 32614 or pan.res != (15, 15):
            raise ValueError("Unexpected B8 window grid")
        image, mask = pan.read(1), pan.read_masks(1)>0
        for item in items:
            tags = original_tags(item)["tags"]
            lineage = [p.strip() for p in tags["LANDSAT_PRODUCT_ID"].split(";")]
            if PAN not in lineage or datetime.fromisoformat(item["properties"]["datetime"]) != datetime.fromisoformat(official["properties"]["datetime"]):
                raise ValueError("HLS source does not name the exact USGS Landsat acquisition")
            folder = HLS/"granules"/"2025"/item["id"]
            with ExitStack() as stack:
                bands = {}
                for band in HLS_BANDS:
                    path = folder/f"{band}.tif"
                    bands[band] = stack.enter_context(rasterio.open(path))
                    sources.append(dict(hls_item=item["id"], band=band, file=str(path.relative_to(ROOT)),
                                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                        hls_input_products_json=json.dumps(lineage)))
                fmask = bands["Fmask"]
                if any(d.crs!=pan.crs or d.transform!=fmask.transform or d.shape!=fmask.shape or d.res!=(30,30) for d in bands.values()):
                    raise ValueError("HLS mask or angle grid differs")
                f = fmask.read(1)
                valid = (fmask.read_masks(1)>0) & (f!=255)
                flags = f & 31
                for key, ref in candidates.items():
                    e, n, radius, shell = roof_geometry(geometry[key])
                    rr, cc, rows, cols, x, y, roof = pixel_coordinates(pan, e, n, radius)
                    if not (5 <= rr < pan.height-5 and 5 <= cc < pan.width-5):
                        raise ValueError(f"Requested B8 window does not cover {key}")
                    samples = image[rr-5+rows, cc-5+cols]
                    usable = mask[rr-5+rows, cc-5+cols] & (samples>0) & (samples<10000)
                    _, clear_valid, bad = pan_qa(fmask, x, y, f, valid, flags)
                    clear = clear_valid & (bad==0)
                    distance = np.hypot(x-e,y-n)
                    keep = clear & usable & (distance<=radius+16)
                    angles = {band: float(next(d.sample([(e,n)]))[0])*d.scales[0] for band,d in bands.items() if band!="Fmask"}
                    if not 0<angles["SZA"]<90 or not 0<=angles["VZA"]<20:
                        raise ValueError(f"Missing HLS centre angles for {key}")
                    lon, lat = to_lon_lat.transform(e,n)
                    correction = float(ground_coordinates(pan.crs.to_string(),((lon,lat),))[2][0])
                    view = bearing_vector(angles["VZA"], angles["VAA"]+correction)
                    grid = (11,11,(float(x[0]-e),float(y[0]-n)),rows[keep],cols[keep],samples[keep]*.0001,
                            np.exp(-.5*(np.maximum(distance[keep]-radius,0)/7)**2))
                    identity = dict(equipment_id=ref["equipment_id"], candidate_id=key, product_id=PAN, hls_item=item["id"],
                                    hls_input_products_json=json.dumps(lineage), capture_at_utc=official["properties"]["datetime"])
                    for mode, za, az in (("hls_roof_solar_angles", angles["SZA"], angles["SAA"]),
                                        ("usgs_scene_centre_solar_angles", 90-official["properties"]["view:sun_elevation"],
                                         official["properties"]["view:sun_azimuth"])):
                        fit = {}
                        if keep.sum()>=12 and (clear[roof] & usable[roof]).all():
                            sun = bearing_vector(za,az+correction)
                            fit = geometry_fit(grid,radius,shell,"EFR",sun,view,kernel,pixel_m=PAN_PIXEL,step_m=PAN_STEP,shifts=PAN_SHIFTS)
                        results.append(dict(**identity, solar_angle_basis=mode, solar_zenith_deg=za, solar_azimuth_deg=az,
                                            hls_view_zenith_deg=angles["VZA"], hls_view_azimuth_deg=angles["VAA"],
                                            b8_roof_cells=int(roof.sum()), b8_roof_cells_with_reflectance=int((roof&usable).sum()),
                                            b8_roof_cells_with_clear_hls=int((roof&clear).sum()), fitted_pixels=int(keep.sum()),
                                            diameter_m=2*radius, shell_above_ground_m=shell, esri_tiff=tile["local_path"],
                                            esri_sha256=tile["sha256"], hls_fmask_file=str(Path(fmask.name).relative_to(ROOT)),
                                            psf_orientation_basis="mean_of_12_evenly_spaced_bearings",
                                            original_usgs_angles_and_qa_verified=False, admitted_absolute_inventory_label=False, **fit))
                        results[-1].setdefault("fit_status", "proxy_inputs_incomplete")
                    for i in range(len(rows)):
                        pixels.append(dict(**identity, b8_window_row=int(rows[i]), b8_window_col=int(cols[i]), east_m=float(x[i]), north_m=float(y[i]),
                                           roof_intersection=bool(roof[i]), raw_esri_value=int(samples[i]), b8_usable=bool(usable[i]),
                                           hls_overlap_all_clear=bool(clear[i]), hls_overlap_flag_bits=int(bad[i]), used_in_fit=bool(keep[i])))
    fields = dict.fromkeys(key for result in results for key in result)
    return [fields | result for result in results], pixels, sources


def later_qa(key):
    with closing(read_only(NATIVE)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute("SELECT * FROM roof_window WHERE product_id=? AND candidate_id=?",(LATER_PAN,key)).fetchone()
        points = list(db.execute("""SELECT * FROM native_pixel WHERE product_id=? AND native_line BETWEEN ? AND ?
            AND native_sample BETWEEN ? AND ? ORDER BY native_line,native_sample""",(LATER_PAN,*[row[k] for k in (
            "first_native_line","last_native_line","first_native_sample","last_native_sample")])) )
    roof = roof_touched(np.array([p["center_east_m"] for p in points]), np.array([p["center_north_m"] for p in points]),
                        row["center_east_m"],row["center_north_m"],row["radius_m"])
    flags = (1<<1,1<<2,1<<3,1<<4,1<<5,1<<11)
    names = ("dilated_cloud","cirrus","cloud","cloud_shadow","snow","terrain_occlusion")
    return [dict(candidate_id=key, product_id=LATER_PAN, native_line=p["native_line"], native_sample=p["native_sample"],
                 qa_valid=p["touching_qa_valid"], angle_status=p["angle_status"], **{k:bool(p["touching_qa_flag_bits"]&v) for k,v in zip(names,flags,strict=True)})
            for p, selected in zip(points,roof,strict=True) if selected]


def remote(candidates, geometry):
    bbox = "-96.756,35.999,-96.749,36.018"
    queries = []
    for d in sorted({r["inspection_local_date"] for r in candidates.values()}):
        day = date.fromisoformat(d)
        period = f"{day-timedelta(days=3)}T06:00:00Z,{day+timedelta(days=4)}T06:00:00Z"
        for sensor, collection in CMR_COLLECTIONS.items():
            url = CMR+"?"+urlencode(dict(collection_concept_id=collection,bounding_box=bbox,temporal=period,page_size=100))
            path = STORE/f"{d}_{sensor}.json"
            raw = get(url,path)
            data = json.loads(raw)
            if data["hits"] != len(data["items"]) or any(r["meta"]["collection-concept-id"] != collection for r in data["items"]):
                raise ValueError(f"Incomplete NASA CMR response for {d}/{sensor}")
            queries.append(dict(local_inspection_date=d,sensor=sensor,collection_id=collection,source_url=url,
                                source_file=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(raw).hexdigest(),
                                matches=data["hits"],matched_concepts_json=json.dumps([r["meta"]["concept-id"] for r in data["items"]])))
    shots, = [r for r in read("cushing_gedi_coverage_granules") if r["l2a_granule"].startswith("GEDI02_A_2025316002316_")]
    path = ROOT/shots["local_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != shots["local_sha256"]:
        raise ValueError("GEDI original subset checksum differs")
    t = pq.read_table(path)
    e,n = Transformer.from_crs(4326,32614,always_xy=True).transform(t["lon_lowestmode"].to_numpy(),t["lat_lowestmode"].to_numpy())
    nearest = []
    for key, r in candidates.items():
        x,y,radius,_ = roof_geometry(geometry[key])
        d = np.hypot(e-x,n-y)
        i = int(d.argmin())
        nearest.append(dict(equipment_id=r["equipment_id"],candidate_id=key,source_file=shots["local_path"],source_sha256=shots["local_sha256"],
                            granule_id=shots["l2a_granule"],regional_shots=len(d),nearest_shot_number=str(t["shot_number"][i].as_py()),
                            nearest_centre_distance_m=float(d[i]),shot_footprints_touching_roof=int((d<=radius+12.5).sum())))
    scene, = [r for r in read("cushing_nasa_satellite_scenes") if r["cmr_id"]=="G4058345892-ASF"]
    cmr_path = STORE/"2025-11-12_NISAR_GSLC_beta.json"
    cmr_item, = json.loads(cmr_path.read_bytes())["items"]
    if cmr_item["meta"]["concept-id"] != scene["cmr_id"]:
        raise ValueError("The Grandstaff CMR response and the radar scene inventory disagree")
    footprints = wkt.loads(scene["footprint_utm_wkt"])
    access = probe("NISAR","complex_hdf5",scene["hdf5_url"])
    radar = []
    for key, r in candidates.items():
        x,y,radius,_ = roof_geometry(geometry[key])
        stamp = datetime.fromisoformat(scene["capture_start_utc"])
        radar.append(dict(equipment_id=r["equipment_id"],candidate_id=key,cmr_id=scene["cmr_id"],granule_id=scene["granule_id"],
                          full_roof_in_catalog_footprint=footprints.covers(Point(x,y).buffer(radius)),capture_start_utc=stamp.isoformat(),
                          same_local_inspection_day=stamp.astimezone(LOCAL).date().isoformat()==r["inspection_local_date"],
                          capture_start_local=stamp.astimezone(LOCAL).isoformat(),east_grid_m=scene["frequency_a_east_pixel_m"],
                          north_grid_m=scene["frequency_a_north_pixel_m"],hdf5_bytes=scene["hdf5_size_bytes"],
                          catalog_revision_utc=cmr_item["meta"]["revision-date"], cmr_file=str(cmr_path.relative_to(ROOT)),
                          native_roof_pixels_verified=False,**access))
    return queries, nearest, radar


def main():
    candidates, geometry = inputs()
    earlier = {k:v for k,v in candidates.items() if v["inspection_local_date"]=="2025-11-12"}
    official, = [i for i in catalog(2025,"usgs") if i["id"]==PAN]
    tile = pan_source(earlier,official)
    ancillary, probes, pc_matches = ancillary_sources(official)
    rows,pixels,sources = hls_fits(earlier,geometry,tile,official)
    later, = set(candidates)-set(earlier)
    qa = later_qa(later)
    queries,gedi,nisar = remote(candidates,geometry)
    for name, records in (("proxy_fits", rows), ("b8_pixels", pixels), ("later_b8_qa", qa), ("pan_source", [tile]),
                          ("ancillary_sources", ancillary), ("hls_sources", sources),
                          ("ancillary_access", probes),
                          ("cmr_queries",queries),("gedi",gedi),("nisar",nisar)):
        export(f"{STEM}_{name}", records)
    result = dict(ground_events=len(candidates), exact_ancillary_catalog_checksum_matches=len(ancillary),
                  exact_ancillary_bytes_verified=0, ancillary_access_probes=len(probes), pc_level2_catalog_matches=pc_matches,
                  esri_b8_matching_overlap_pixels=tile["shared_valid_pixel_comparisons"], hls_ancillary_files=len(sources),
                  proxy_fits=len(rows), proxy_fit_status=Counter(r["fit_status"] for r in rows),
                  proxy_same_day_diagnostics_pass=sum(r["fit_status"]==RESOLVED for r in rows),
                  later_roof_pixels=len(qa), later_roof_pixels_with_cirrus=sum(r["cirrus"] for r in qa),
                  later_roof_pixels_with_cloud=sum(r["cloud"] for r in qa), later_roof_pixels_with_dilated_cloud=sum(r["dilated_cloud"] for r in qa),
                  cmr_queries=len(queries), gedi_regional_shots=gedi[0]["regional_shots"],
                  gedi_footprints_touching_roofs=sum(r["shot_footprints_touching_roof"] for r in gedi),
                  nisar_same_local_day_roof_catalog_matches=sum(r["same_local_inspection_day"] and r["full_roof_in_catalog_footprint"] for r in nisar),
                  nisar_native_roof_pixels_verified=False, admitted_absolute_inventory_labels=0)
    payload = json.dumps(result,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
