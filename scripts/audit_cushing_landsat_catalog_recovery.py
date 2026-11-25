"""Check additional Landsat products reported by the bulk inventory over Cushing roofs."""

import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.parse import urlencode

from affine import Affine
from pyproj import Transformer
from rasterio.io import MemoryFile
from shapely.geometry import Point, Polygon

from audit_cushing_usgs_b8_metadata import STEM as METADATA
from build_shadow_registration_inputs import tank_points
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_landsat import LANDSAT
from collect_landsat_esri_b8 import query, raster_request
from collect_public import ROOT, get, write
from stage_landsat_c2_b8_inputs import expected_sha512

STEM = "cushing_landsat_catalog_recovery"
STORE = LANDSAT/STEM
STAC = "https://landsatlook.usgs.gov/stac-server/collections/landsat-c2l1/items/"
CDSE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
PRODUCT = "Landsat Product Identifier L1"
CORNERS = ("Upper Left","Upper Right","Lower Right","Lower Left")


def missing_intersections():
    tanks,_ = tank_points()
    project = Transformer.from_crs(32614,4326,always_xy=True).transform
    points = {k:Point(*project(*coords)) for k,coords in tanks.items()}
    for row in read_csv(f"{METADATA}_regional_catalog"):
        if row["in_cushing_stac_snapshot"]!="False":
            continue
        polygon = Polygon([(float(row[f"Corner {c} Longitude"]),float(row[f"Corner {c} Latitude"])) for c in CORNERS])
        if not polygon.is_valid:
            raise ValueError(f"Invalid bulk metadata corners for {row[PRODUCT]}")
        covered = sorted(k for k,p in points.items() if polygon.covers(p))
        if covered:
            yield row,covered


def exact_stac(identity):
    url = STAC+identity
    try:
        raw = get(url)
    except HTTPError as error:
        if error.code!=404:
            raise
        return 404,None,None
    item = json.loads(raw)
    if item.get("id")!=identity or item.get("collection")!="landsat-c2l1":
        raise ValueError(f"Unexpected exact STAC item for {identity}")
    path = STORE/"stac"/f"{identity}.json"
    write(path,raw)
    return 200,item,str(path.relative_to(ROOT))


def provider_rows(names):
    esri,cdse = {},{}
    for i in range(0,len(names),10):
        group = names[i:i+10]
        _,raw,items = query("Name IN ("+",".join(f"'{n}'" for n in group)+")")
        write(STORE/f"esri_{i//10:02}.json",raw)
        for r in items:
            name = r["Name"]
            if name not in group or name in esri or r["LANDSAT_PRODUCT_ID"]!=name or r["Category"]!=1:
                raise ValueError(f"Unexpected Esri catalog identity {name}")
            esri[name] = r
        url = CDSE+"?"+urlencode({"$filter":" or ".join(f"Name eq '{n}'" for n in group),"$top":20,"$count":"true"})
        raw = get(url)
        write(STORE/f"cdse_{i//10:02}.json",raw)
        page = json.loads(raw)
        if page.get("@odata.nextLink") or page.get("@odata.count")!=len(page["value"]):
            raise ValueError("Copernicus exact product page is incomplete")
        for r in page["value"]:
            name = r["Name"]
            if name not in group or name in cdse:
                raise ValueError(f"Unexpected Copernicus catalog identity {name}")
            cdse[name] = r
    return esri,cdse


def local_geometry(wanted):
    metadata = {}
    for path in (LANDSAT/"collection2_context").glob("*/*/MTL.json"):
        item = json.loads(path.read_bytes())["LANDSAT_METADATA_FILE"]
        record = item["LEVEL1_PROCESSING_RECORD"]
        name = record["LANDSAT_PRODUCT_ID"]
        if name not in wanted:
            continue
        if name in metadata:
            raise ValueError(f"Multiple Level-2 metadata files identify {name}")
        metadata[name] = path,item
    return metadata


def collect_recovery_crop(row, covered, metadata, source, *, store=STORE, allow_empty=False):
    path,item = metadata
    projection = item["PROJECTION_ATTRIBUTES"]
    b8_projection = item["LEVEL1_PROJECTION_PARAMETERS"]
    crs = f"EPSG:{32600+int(projection['UTM_ZONE'])}"
    if crs!="EPSG:32614" or float(b8_projection["GRID_CELL_SIZE_PANCHROMATIC"])!=15:
        raise ValueError(f"Unexpected recovered B8 projection for {row[PRODUCT]}")
    ul_e,ul_n = (float(projection[f"CORNER_UL_PROJECTION_{k}_PRODUCT"]) for k in "XY")
    grid = Affine(15,0,ul_e-7.5,0,-15,ul_n+7.5)
    points,_ = tank_points()
    x,y = zip(*(points[k] for k in covered))
    c0,r0 = (~grid)*(min(x)-120,max(y)+120)
    c1,r1 = (~grid)*(max(x)+120,min(y)-120)
    window = math.floor(r0),math.ceil(r1),math.floor(c0),math.ceil(c1)
    a,b,c,d = window
    if not (0<=a<b<=int(row["Panchromatic Lines"]) and 0<=c<d<=int(row["Panchromatic Samples"])):
        raise ValueError(f"Recovered B8 footprint crosses grid bounds for {row[PRODUCT]}")
    product = dict(crs=crs,ul_b8_center_x_m=ul_e,ul_b8_center_y_m=ul_n)
    url,transform = raster_request(product,source,window)
    raw = get(url)
    with MemoryFile(raw) as mem,mem.open() as ds:
        if ds.transform!=transform or ds.crs.to_string()!=crs or ds.dtypes!=("int16",) or ds.shape!=(b-a,d-c):
            raise ValueError(f"Recovered Esri pixels disagree with the Level-2 B8 grid for {row[PRODUCT]}")
        v = ds.read(1)
        valid = int(((ds.read_masks(1)>0)&(v>0)&(v<10000)).sum())
        if not valid and not allow_empty:
            raise ValueError(f"Empty Esri raster for {row[PRODUCT]}")
    destination = store/"esri_reflectance"/f"{row[PRODUCT]}.tif"
    write(destination,raw)
    return dict(esri_reflectance_path=str(destination.relative_to(ROOT)),esri_reflectance_url=url,
        esri_reflectance_sha256=hashlib.sha256(raw).hexdigest(),esri_reflectance_valid_pixels=valid,
        esri_reflectance_grid_window_json=json.dumps(window),level2_metadata_path=str(path.relative_to(ROOT)))


def main():
    candidates = sorted(missing_intersections(),key=lambda x:(x[0]["Date Acquired"],x[0][PRODUCT]))
    names = [r[PRODUCT] for r,_ in candidates]
    if len(set(names))!=len(names):
        raise ValueError("Repeated regional recovery product")
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(exact_stac,names))
    esri,cdse = provider_rows(names)
    metadata = local_geometry(set(names))
    downloads = [(r,covered,metadata[r[PRODUCT]],esri[r[PRODUCT]]["OBJECTID"])
                 for r,covered in candidates if r[PRODUCT] in metadata and r[PRODUCT] in esri]
    with ThreadPoolExecutor(max_workers=4) as pool:
        reflectance = dict(zip((r[PRODUCT] for r,*_ in downloads),pool.map(lambda job:collect_recovery_crop(*job),downloads),strict=True))
    rows = []
    for (r,covered),(status,item,path) in zip(candidates,results,strict=True):
        n = r[PRODUCT]
        asset = item["assets"].get("pan") if item else None
        e,c = esri.get(n,{}),cdse.get(n,{})
        if e and e["LANDSAT_SCENE_ID"]!=r["Landsat Scene Identifier"]:
            raise ValueError(f"Esri scene ID differs from USGS bulk record for {n}")
        rows.append(dict(product_id=n,usgs_scene_id=r["Landsat Scene Identifier"],
            acquisition_day=r["Date Acquired"].replace("/","-"),category=r["Collection Category"],
            bulk_csv_record_number=int(r["source_record_number"]),covered_registry_centres=len(covered),
            covered_registry_ids_json=json.dumps(covered,separators=(",",":")),exact_usgs_stac_http_status=status,
            exact_usgs_stac_item_url=STAC+n,local_stac_snapshot=path,
            local_stac_sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest() if path else None,
            b8_expected_usgs_sha512=expected_sha512(asset) if asset else None,
            esri_exact_object_id=e.get("OBJECTID"),esri_sensor_label=e.get("SensorName"),
            cdse_exact_product_id=c.get("Id"),cdse_catalog_publication_at_utc=c.get("PublicationDate"),
            original_b8_acquired=False,exact_b8_availability_before_eia_verified=False,
            **{k:reflectance.get(n,{}).get(k) for k in ("esri_reflectance_path","esri_reflectance_url","esri_reflectance_sha256",
                "esri_reflectance_valid_pixels","esri_reflectance_grid_window_json","level2_metadata_path")}))
    export(STEM,rows)
    report = dict(bulk_products_covering_registry_centres_absent_from_saved_stac=len(rows),
        by_acquisition_year=Counter(r["acquisition_day"][:4] for r in rows),
        direct_usgs_stac_status=Counter(r["exact_usgs_stac_http_status"] for r in rows),
        esri_exact_catalog_matches=sum(r["esri_exact_object_id"] is not None for r in rows),
        cdse_exact_catalog_matches=sum(r["cdse_exact_product_id"] is not None for r in rows),
        esri_reflectance_crops=len(reflectance),esri_reflectance_valid_pixels=sum(r["esri_reflectance_valid_pixels"] for r in reflectance.values()),
        source_coverage_basis="Quadrilateral defined by the four USGS bulk metadata corners and the tank registry centre points.",
        scope="Catalog recovery candidates and processed Esri reflectance on the matching Level-2 grid. The image masks from Esri "
              "are not cloud QA. No exact B8 TIFF, historical B8 publication evidence or roof height is established.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
