"""Recover native B8 angles and radiometric coefficients from exact Level-2 copies."""
import base64
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.error import URLError
from urllib.request import Request, urlopen

from affine import Affine
from pyproj import Transformer

from build_cross_sensor_shadow import along_track_bearing
from build_landsat_c2_holdout_gap_inventory import STEM as GAP, landsat_catalog, parse
from build_landsat_pan_roof_angles import angle_values, prepare_tool
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from build_weekly_shadow_validation import legacy_geometry
from collect_imagery import sign
from collect_landsat import LANDSAT, catalog
from collect_landsat_asset_times import asset_snapshot
from collect_public import ROOT, TLS, write

STEM = f"{GAP}_geometry"
STORE = LANDSAT / "collection2_b8_geometry"


def read_azure_asset(archive, path):
    if path.is_file():
        payload = path.read_bytes()
    else:
        etag = archive["etag"].strip('"')
        request = Request(sign(archive["url"]), headers={"If-Match": f'"{etag}"'})
        try:
            with urlopen(request, timeout=30, context=TLS) as response:
                if response.headers.get("ETag", "").strip('"') != etag:
                    raise ValueError(f"Landsat archive response has a different ETag for {path}")
                payload = response.read(archive["bytes"] + 1)
        except URLError as error:
            status = getattr(error, "code", None)
            raise RuntimeError(f"Landsat download failed for {path.name} ({status or type(error.reason).__name__})") from None
    md5 = base64.b64encode(hashlib.md5(payload).digest()).decode()
    if len(payload) != archive["bytes"] or md5 != archive["content_md5"]:
        raise ValueError(f"Incorrect source size or MD5 for {path}")
    if not path.is_file():
        write(path, payload)
    return payload


def packet(r, source, item, metadata_path, *, store=STORE):
    product = r["product_id"]
    attributes = source["properties"]
    meta = json.loads(metadata_path.read_bytes())["LANDSAT_METADATA_FILE"]
    record, image, projection, radiometry = (meta[name] for name in (
        "LEVEL1_PROCESSING_RECORD", "IMAGE_ATTRIBUTES", "PROJECTION_ATTRIBUTES", "LEVEL1_RADIOMETRIC_RESCALING"))
    if (record["LANDSAT_PRODUCT_ID"] != product or record["LANDSAT_SCENE_ID"] != attributes["landsat:scene_id"]
            or meta["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"] not in item["assets"]["ang"]["href"]
            or parse(record["DATE_PRODUCT_GENERATED"]) != parse(r["level1_generated_at_utc"])
            or parse(image["DATE_ACQUIRED"] + "T" + image["SCENE_CENTER_TIME"]) != parse(r["captured_at_utc"])):
        raise ValueError(f"Landsat Level-2 metadata does not identify exact input {product}")
    directory = store / product
    evidence = directory / "level2_angle_archive.json"
    snap = asset_snapshot(item, ("ang",), evidence)
    archive = snap["assets"]["ang"]
    path = directory / "ANG.txt"
    payload = read_azure_asset(archive, path)
    angle = payload.decode("ascii")
    values = dict(re.findall(r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*([^\r\n]+)", angle, re.M))
    if (values["LANDSAT_SCENE_ID"].strip().strip('"') != record["LANDSAT_SCENE_ID"]
            or values["SPACECRAFT_ID"].strip().strip('"') != r["platform"]
            or values["MAP_PROJECTION"].strip().strip('"') != "UTM"
            or values["PROJECTION_UNITS"].strip().strip('"') != "METERS"
            or int(values["UTM_ZONE"]) != int(projection["UTM_ZONE"])
            or float(values["BAND08_PIXEL_SIZE"]) != 15
            or float(meta["LEVEL1_PROJECTION_PARAMETERS"]["GRID_CELL_SIZE_PANCHROMATIC"]) != 15):
        raise ValueError(f"Wrong original B8 angle scene identity or projection for {product}")
    ul = tuple(map(float, values["UL_CORNER"].strip().strip("()").split(",")))
    if ul != tuple(float(projection[f"CORNER_UL_PROJECTION_{axis}_PRODUCT"]) for axis in "XY"):
        raise ValueError(f"Level-2 MTL and ANG disagree on native B8 origin for {product}")
    crs = f"EPSG:{32600 + int(projection['UTM_ZONE'])}"
    dimensions = int(values["BAND08_NUM_L1T_LINES"]), int(values["BAND08_NUM_L1T_SAMPS"])
    # The ANG grid includes the centers at both scene edges.
    for axis, index, count in (("X",0,dimensions[1]), ("Y",1,dimensions[0])):
        end = float(projection[f"CORNER_LR_PROJECTION_{axis}_PRODUCT"])
        if not math.isclose(abs(end-ul[index]), (count-1)*15, abs_tol=0.00001):
            raise ValueError(f"Landsat B8 dimensions do not match the scene bounds for {product}")
    if not parse(r["level1_generated_at_utc"]) <= parse(archive["created_at_utc"]) <= parse(archive["last_modified_at_utc"]):
        raise ValueError(f"Landsat angle archive predates underlying Level-1 product {product}")
    return dict(product_id=product, platform=r["platform"], captured_at_utc=r["captured_at_utc"],
                level1_generated_at_utc=r["level1_generated_at_utc"], level1_scene_id=record["LANDSAT_SCENE_ID"],
                level2_item_id=item["id"], level2_metadata_path=str(metadata_path.relative_to(ROOT)),
                angle_url=archive["url"], angle_local_file=str(path.relative_to(ROOT)),
                angle_archive_file=str(evidence.relative_to(ROOT)), angle_etag=archive["etag"], angle_bytes=len(payload),
                angle_sha256=hashlib.sha256(payload).hexdigest(), angle_created_at_utc=archive["created_at_utc"],
                angle_last_modified_at_utc=archive["last_modified_at_utc"], crs=crs, native_grid_m=15,
                native_lines=dimensions[0], native_samples=dimensions[1], ul_b8_center_x_m=ul[0], ul_b8_center_y_m=ul[1],
                along_track_grid_bearing_deg=along_track_bearing(path, crs),
                reflectance_mult_b8=float(radiometry["REFLECTANCE_MULT_BAND_8"]),
                reflectance_add_b8=float(radiometry["REFLECTANCE_ADD_BAND_8"]))


def native_b8_grid(product):
    return Affine(15, 0, product["ul_b8_center_x_m"] - 7.5,
                  0, -15, product["ul_b8_center_y_m"] + 7.5)


def roof_angles(tool, product, tank_ids, geometry, geometry_basis):
    project = Transformer.from_crs(32614, product["crs"], always_xy=True)
    inverse = ~native_b8_grid(product)
    points = {}
    for tank in sorted(tank_ids):
        g = geometry[tank]
        x,y = project.transform(float(g["center_east_m"]),float(g["center_north_m"]))
        sample,line = inverse * (x,y)
        row,col = math.floor(line),math.floor(sample)
        points[tank] = row,col
    angles = angle_values(tool, ROOT / product["angle_local_file"], points.values())
    for tank,(row,col) in points.items():
        yield dict(product_id=product["product_id"], candidate_id=tank, platform=product["platform"],
                   captured_at_utc=product["captured_at_utc"], geometry_basis=geometry_basis,
                   b8_native_line=row, b8_native_sample=col, b8_native_grid_m=15,
                   b8_pixel_center_x_m=product["ul_b8_center_x_m"]+col*15,
                   b8_pixel_center_y_m=product["ul_b8_center_y_m"]-row*15,
                   angle_source_file=product["angle_local_file"],
                   angle_archive_available_at_utc=product["angle_last_modified_at_utc"], **angles[row,col])


def main():
    manifest = {r["product_id"]: r for r in read_csv(f"{GAP}_download_manifest")}
    source = {item["id"]: item for item,_ in landsat_catalog() if item["id"] in manifest}
    contexts, tank_ids = {}, defaultdict(set)
    for r in read_csv(f"{GAP}_products"):
        product = r["product_id"]
        if product not in manifest or r["local_exact_level2_product"] == "":
            continue
        context = r["local_exact_level2_product"], r["local_level2_metadata_path"]
        if contexts.setdefault(product,context) != context:
            raise ValueError(f"Multiple exact Level-2 sources for {product}")
        if r["level1_could_be_generated_by_cutoff"]=="True" and r["local_level2_before_cutoff"]=="True":
            tank_ids[product].update(json.loads(r["clear_efr_candidate_ids_json"]))
    if source.keys() != manifest.keys() or contexts.keys() != manifest.keys() or tank_ids.keys() != manifest.keys():
        raise ValueError("Priority Landsat versions differ from exact local Level-2 contexts")
    ids = {item for item,_ in contexts.values()}
    pc = {item["id"]: item for year in range(2023,2027) for item in catalog(year, "pc") if item["id"] in ids}
    tool, geometry = prepare_tool(), legacy_geometry()
    with ThreadPoolExecutor(max_workers=6) as pool:
        products = list(pool.map(lambda p: packet(manifest[p], source[p], pc[contexts[p][0]], ROOT / contexts[p][1]),
                                 sorted(manifest)))
        rows = [row for result in pool.map(lambda p: list(roof_angles(
                    tool,p,tank_ids[p["product_id"]],geometry,"independent_2011_elevated_circle")),products)
                for row in result]
    angles = {(r["product_id"], r["candidate_id"]):r for r in rows}
    controls = read_csv(f"{GAP}_calibration_controls")
    controls_ready = 0
    for r in controls:
        a = angles[r["landsat_product_id"],r["candidate_id"]]
        controls_ready += (a["angle_status"]=="resolved"
                           and parse(a["angle_archive_available_at_utc"])<parse(r["forecast_cutoff_utc"]))
    endpoints = defaultdict(set)
    planned = {(r["week_ending"],side):set(json.loads(r[f"{side}_b8_pair_candidate_ids_json"]))
               for r in read_csv(f"{GAP}_weeks") for side in ("previous","current")}
    for r in read_csv(f"{GAP}_products"):
        if r["product_id"] not in manifest or r["local_level2_before_cutoff"]!="True":
            continue
        key = r["week_ending"],r["endpoint"]
        for tank in planned[key].intersection(json.loads(r["clear_efr_candidate_ids_json"])):
            a = angles.get((r["product_id"],tank))
            if a and a["angle_status"]=="resolved" and parse(a["angle_archive_available_at_utc"])<parse(r["cutoff_utc"]):
                endpoints[key].add(tank)
    if planned != {key:endpoints[key] for key in planned}:
        raise ValueError("Some proposed B8 endpoints lack independently dated angle coverage")
    export(f"{STEM}_products", products)
    export(f"{STEM}_roof_angles", rows)
    summary = dict(priority_products=len(products), products_by_platform=dict(Counter(r["platform"] for r in products)),
                   total_original_angle_bytes=sum(r["angle_bytes"] for r in products),
                   distinct_roofs=len({r["candidate_id"] for r in rows}), roof_scene_angles=len(rows),
                   angle_status=dict(Counter(r["angle_status"] for r in rows)),
                   calibration_comparisons_with_angles_before_cutoff=controls_ready,
                   reporting_weeks_with_timely_angles_for_all_planned_b8_endpoints=len({week for week,side in endpoints
                                                                                       if endpoints[week,side]}),
                   planned_b8_tank_week_endpoints_with_angles_before_cutoff=sum(map(len,endpoints.values())),
                   interpretation="USGS B8 angles are evaluated on the ANG native grid. The matching Level-2 metadata "
                                  "identifies the exact underlying Level-1 product. The Azure archive timestamps apply "
                                  "to the retrieved Level-2 angle file. B8 raster pixels are not included.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
