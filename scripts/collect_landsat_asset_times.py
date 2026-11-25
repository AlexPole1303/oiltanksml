"""Record version evidence from the Azure assets used by Landsat and HLS."""
import argparse
import datetime as dt
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, urlencode, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from shapely.geometry import shape

from collect_imagery import sign
from collect_landsat import CONTEXT_ASSETS, LANDSAT, catalog, daylight, source_coverage
from collect_public import ROOT, TLS, write


def asset_times(urls, alternatives=()):
    addresses = {label: urlsplit(url) for label,url in urls.items()}
    containers = {(p.hostname,p.path.split("/")[1]) for p in addresses.values()}
    host, container = containers.pop()
    if containers or (host,container) not in {("landsateuwest.blob.core.windows.net","landsat-c2"),
                                               ("hls2euwest.blob.core.windows.net","hls2")}:
        raise ValueError("Unexpected Azure archive for Landsat imagery")
    keys = {label:unquote(p.path.split("/",2)[2]) for label,p in addresses.items()}
    prefixes = {key.rsplit("/",1)[0]+"/" for key in keys.values()}
    if len(prefixes) != 1:
        raise ValueError("Azure product assets do not share a directory")
    prefix, = prefixes
    base = f"https://{host}/{container}"
    params = {"restype":"container","comp":"list","prefix":prefix,"maxresults":100}
    objects = {}
    while True:
        url = sign(base)+"&"+urlencode(params)
        request = Request(url,headers={"x-ms-version":"2021-12-02"})
        for attempt in range(3):
            try:
                with urlopen(request,context=TLS,timeout=30) as response:
                    page = ET.fromstring(response.read())
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        for blob in page.findall("./Blobs/Blob"):
            properties = blob.find("Properties")
            objects[blob.findtext("Name")] = {"created_at_utc":parsedate_to_datetime(properties.findtext("Creation-Time")).isoformat(),
                "last_modified_at_utc":parsedate_to_datetime(properties.findtext("Last-Modified")).isoformat(),
                "etag":properties.findtext("Etag"),"bytes":int(properties.findtext("Content-Length")),
                "content_md5":properties.findtext("Content-MD5")}
        marker = page.findtext("NextMarker")
        if not marker:
            break
        if marker == params.get("marker"):
            raise ValueError("Azure returned a repeated listing page")
        params["marker"] = marker
    required = [label for label in urls if label not in alternatives]
    if alternatives:
        required.append(next(label for label in alternatives if keys[label] in objects))
    result = {label:{"url":urls[label],**objects[keys[label]]} for label in required}
    if any(not row["etag"] or row["bytes"]<1 or row["created_at_utc"]>row["last_modified_at_utc"] for row in result.values()):
        raise ValueError("Incomplete Azure object properties")
    return result


def asset_snapshot(item, labels, path, alternatives=()):
    urls = {label:item["assets"][label]["href"] for label in (*labels,*alternatives)}
    if path.is_file():
        value = json.loads(path.read_bytes())
        found = {k:v["url"] for k,v in value["assets"].items()}
        if (value["product_id"] != item["id"] or not set(labels)<=set(found)
                or len(set(found)&set(alternatives))!=int(bool(alternatives))
                or any(urls.get(key)!=url for key,url in found.items())):
            raise ValueError(f"Catalog asset version changed for {item['id']}")
        return value
    objects = asset_times(urls,alternatives)
    latest = max(record["last_modified_at_utc"] for record in objects.values())
    value = {"product_id":item["id"],"captured_at_utc":item["properties"]["datetime"],
             "checked_at_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"latest_asset_modified_at_utc":latest,"assets":objects}
    write(path,json.dumps(value,indent=2).encode())
    return value


def pc_asset_snapshot(item):
    folder = LANDSAT / "pc_asset_times" / item["properties"]["datetime"][:4]
    return asset_snapshot(item,CONTEXT_ASSETS,folder / f"{item['id']}.json",("mtl.json","mtl.xml"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    _,area = source_coverage()
    items = [item for year in range(2017,2027) for item in catalog(year,"pc")
             if daylight(item) and shape(item["geometry"]).intersects(area)]
    print(len(items),"Landsat products for archive-clock verification",flush=True)
    failures = []
    log = ROOT / "data" / "logs" / "landsat_pc_asset_times.jsonl"
    with ThreadPoolExecutor(max_workers=args.workers) as pool,log.open("a") as stream:
        pending = {pool.submit(pc_asset_snapshot,item):item["id"] for item in items}
        for i,future in enumerate(as_completed(pending),1):
            product = pending[future]
            try:
                value = future.result()
                status = {"product_id":product,"status":"ready","latest_asset_modified_at_utc":value["latest_asset_modified_at_utc"]}
            except Exception as error:
                failures.append(product)
                detail = re.sub(r"https?://\S+","[archive URL]",str(error))
                status = {"product_id":product,"status":"error","detail":f"{type(error).__name__}: {detail}"}
                print(status,flush=True)
            print(json.dumps(status),file=stream,flush=True)
            if i%50==0 or i==len(items):
                print(i,"Landsat archive-clock results",flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} Landsat products still lack a complete archive clock")


if __name__ == "__main__":
    main()
