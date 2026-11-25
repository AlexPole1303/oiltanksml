"""Recover HLS L30 observations by the source Landsat scene ID in the original raster."""
import argparse
import json
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import rasterio
from shapely.geometry import shape

from collect_imagery import CONFIG, PC, clip, sign
from collect_landsat import BBOX, LANDSAT, catalog, daylight, identity, scene_key, source_coverage
from collect_landsat_asset_times import asset_snapshot
from collect_public import ROOT, get, write, write_csv

HLS = LANDSAT / "hls_l30_gap"
HLS_SPECTRAL = tuple(f"B{n:02}" for n in range(2,8))
HLS_ASSETS = (*HLS_SPECTRAL,"Fmask","SAA","SZA","VAA","VZA")
ROUTES = {
    "pc":(PC+"/api/stac/v1/search?","hls2-l30"),
    "cmr":("https://cmr.earthdata.nasa.gov/stac/LPCLOUD/search?","HLSL30_2.0"),
}


def targets(area):
    found = {}
    for route in ("usgs","pc"):
        found[route] = defaultdict(list)
        for year in range(2017,2027):
            for item in catalog(year,route):
                if daylight(item) and shape(item["geometry"]).intersects(area):
                    props = item["properties"]
                    key = identity(props["platform"],scene_key(item)[0],props["datetime"])
                    found[route][key].append(item)
    reasons = dict.fromkeys(found["usgs"].keys()-found["pc"].keys(),"missing_from_pc_catalog")
    for path in (LANDSAT/"pc_asset_exceptions").glob("*.json"):
        failure = json.loads(path.read_bytes())
        if failure["validation_status"]=="archive_object_is_html" and failure["acquisition_key"] in found["usgs"]:
            reasons[failure["acquisition_key"]] = "invalid_pc_archive_object"
    return {key:(reasons[key],found["usgs"][key]) for key in sorted(reasons)}


def day_catalog(job):
    day,route = job
    base,collection = ROUTES[route]
    url = base+urlencode({"collections":collection,"bbox":BBOX,"limit":100,
                         "datetime":f"{day}T00:00:00Z/{day}T23:59:59.999999Z"})
    path = HLS / "catalog" / f"{day}_{route}.geojson"
    data = json.loads(get(url,path))
    matched = data.get("numberMatched",data.get("context",{}).get("matched"))
    if ((matched is not None and matched!=len(data["features"])) or
            (matched is None and any(link["rel"]=="next" for link in data.get("links",[])))):
        raise ValueError(f"HLS day catalog needs pagination: {day} {route}")
    write(path.with_suffix(".source"),(url+"\n").encode())
    return day,route,data["features"]


def original_tags(item):
    path = HLS / "tags" / f"{item['id']}.json"
    source = item["assets"]["B04"]["href"]
    if path.is_file():
        saved = json.loads(path.read_bytes())
        if saved["source_url"] != source:
            raise ValueError(f"HLS red asset changed for {item['id']}")
        return saved
    with rasterio.Env(**CONFIG),rasterio.open(sign(source)) as dataset:
        if dataset.res!=(30,30):
            raise ValueError(f"Unexpected HLS grid for {item['id']}")
        tags = dataset.tags()
    if not tags.get("LANDSAT_SCENE_ID") or not tags.get("LANDSAT_PRODUCT_ID"):
        raise ValueError(f"Missing HLS Landsat lineage for {item['id']}")
    value = {"hls_id":item["id"],"source_url":source,"tags":tags}
    write(path,json.dumps(value,indent=2).encode())
    return value


def acquire(item, keys, tags, area, footprint):
    capture = item["properties"]["datetime"]
    directory = HLS / "granules" / capture[:4] / item["id"]
    objects = asset_snapshot(item,HLS_ASSETS,directory / "asset_times.json")
    coeffs, grid = {},None
    for label in HLS_ASSETS:
        path = directory / f"{label}.tif"
        href = item["assets"][label]["href"]
        dimensions = clip(sign(href) if not path.is_file() else href,path,area.bounds,footprint,mask_to_footprint=True)
        with rasterio.open(path) as raster:
            signature = raster.crs,raster.transform,raster.shape
            if grid is None:
                grid = signature
            types = {"uint8"} if label=="Fmask" else {"int16"} if label in HLS_SPECTRAL else {"int16","uint16"}
            if (raster.count!=1 or raster.dtypes[0] not in types or raster.res!=(30,30) or grid!=signature
                    or tuple(dimensions)!=(raster.width,raster.height)):
                raise ValueError(f"HLS grid or storage type mismatch for {item['id']} {label}")
            raster.read(1)
            raster.read_masks(1)
            if label in HLS_SPECTRAL:
                coeffs[label] = (raster.scales[0],raster.offsets[0])
    value = {"item":item,"matched_acquisition_keys":sorted(keys),"source_tags_file":str((HLS/"tags"/f"{item['id']}.json").relative_to(ROOT)),
             "source_scene_ids":[s.strip() for s in tags["tags"]["LANDSAT_SCENE_ID"].split(";")],
             "latest_asset_modified_at_utc":objects["latest_asset_modified_at_utc"],"reflectance_coefficients":coeffs}
    write(directory / "item.json",json.dumps(value,indent=2).encode())
    print(item["id"],"ready",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    footprint,area = source_coverage()
    gaps = targets(area)
    days = sorted({key.rsplit(":",1)[1] for key in gaps})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        catalogs = {(day,route):items for day,route,items in pool.map(day_catalog,((d,r) for d in days for r in ROUTES))}
        options = {item["id"]:item for day in days for item in catalogs[day,"pc"] if shape(item["geometry"]).intersects(area)}
        tags = dict(zip(options,pool.map(original_tags,options.values()),strict=True))
        matching, rows = defaultdict(list),[]
        for key,(reason,references) in gaps.items():
            day = key.rsplit(":",1)[1]
            scenes = {x["properties"]["landsat:scene_id"] for x in references}
            matched = [pid for pid in options if options[pid]["properties"]["datetime"].startswith(day)
                       and scenes & {s.strip() for s in tags[pid]["tags"]["LANDSAT_SCENE_ID"].split(";")}]
            for pid in matched:
                matching[pid].append(key)
            rows.append((key,reason,json.dumps(sorted(x["id"] for x in references)),json.dumps(sorted(scenes)),
                         len(catalogs[day,"pc"]),len(catalogs[day,"cmr"]),json.dumps(sorted(matched))))
        write_csv(ROOT/"data"/"derived"/"landsat_hls_gap_evidence.csv",("acquisition_key","gap_reason","usgs_product_ids_json",
                  "usgs_scene_ids_json","pc_hls_day_items","cmr_hls_day_items","exact_hls_matches_json"),rows)
        print(sum(bool(json.loads(row[-1])) for row in rows),"of",len(gaps),"source-scene gaps matched",len(matching),"HLS granules",flush=True)
        granules = sorted(matching)
        for offset in range(0,len(granules),args.workers):
            if shutil.disk_usage(LANDSAT).free < 2.5*1024**3:
                raise RuntimeError("HLS download reached the working disk reserve")
            batch = granules[offset:offset+args.workers]
            list(pool.map(lambda pid:acquire(options[pid],matching[pid],tags[pid],area,footprint),batch))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        detail = re.sub(r"https?://\S+","[archive URL]",str(error))
        raise SystemExit(f"{type(error).__name__}: {detail}") from None
