"""Check public high-resolution image catalogs against the Cushing study footprint."""
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode, urljoin

from shapely.geometry import box, shape

from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write, write_csv

STORE = RAW/"cushing_high_resolution"
BOUNDS = (-96.81,35.73,-96.54,36.10)
AREA = box(*BOUNDS)
DATES = "2017-01-01T00:00:00Z/2026-10-31T23:59:59Z"
EVENTS = {"maxar":"https://maxar-opendata.s3.dualstack.us-west-2.amazonaws.com/events/catalog.json",
          "vantor":"https://vantor-opendata.s3.amazonaws.com/events/catalog.json"}
HOT = "https://api.imagery.hotosm.org/stac/search"
HOT_COLLECTIONS = ("maxar-opendata","vantor-opendata","noaa-emergency-response","openaerialmap")
WYVERN = "https://wyvern-odp.com/year"
STEM = "cushing_open_high_resolution"


def source(url,path):
    raw = get(url,path)
    return json.loads(raw),dict(source_url=url,source_snapshot=str(path.relative_to(ROOT)),
        source_sha256=hashlib.sha256(raw).hexdigest())


def links(data,rel,url):
    return sorted({urljoin(url,r["href"]) for r in data["links"] if r["rel"]==rel})


def events():
    requests = []
    root_meta = []
    for provider,url in EVENTS.items():
        data,meta = source(url,STORE/f"{provider}.snapshot")
        urls = links(data,"child",url)
        root_meta.append(dict(provider=provider,linked_records=len(urls),**meta))
        requests.extend((provider,child) for child in urls)

    def visit(pair):
        provider,url = pair
        event = url.rsplit("/",2)[1]
        record,meta = source(url,STORE/provider/f"{event}.json")
        bounds = record["extent"]["spatial"]["bbox"]
        if record["type"]!="Collection" or not bounds or record["id"]!=event:
            raise ValueError(f"Unexpected open data collection: {url}")
        return dict(provider=provider,collection_id=event,license=record.get("license"),bbox_count=len(bounds),
            bounds_intersect=any(box(*b[:4]).intersects(AREA) for b in bounds),
            temporal_extent_json=json.dumps(record["extent"]["temporal"]["interval"]),**meta)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return root_meta,list(pool.map(visit,requests))


def hot():
    url = HOT+"?"+urlencode(dict(bbox=",".join(map(str,BOUNDS)),datetime=DATES,limit=100))
    data,meta = source(url,STORE/"hotosm.snapshot")
    if links(data,"next",url) or len(data["features"])!=data["numberReturned"]:
        raise ValueError("Incomplete HOT STAC search")
    counts = {c:sum(r["collection"]==c and shape(r["geometry"]).intersects(AREA) for r in data["features"])
              for c in HOT_COLLECTIONS}
    return [dict(collection_id=k,intersecting_items=v,**meta) for k,v in counts.items()]


def wyvern():
    urls = set()
    metadata = []
    for year in (2024,2025,2026):
        url = f"{WYVERN}/{year}/collection.json"
        data,meta = source(url,STORE/f"wyvern_{year}.snapshot")
        items = links(data,"item",url)
        metadata.append(dict(provider=f"wyvern_{year}",linked_records=len(items),**meta))
        urls.update(items)

    def visit(url):
        record,meta = source(url,STORE/"wyvern_items"/url.rsplit("/",1)[1])
        name = record["id"]
        if record["type"]!="Feature" or not url.endswith(f"/{name}.json"):
            raise ValueError(f"Unexpected Wyvern identity {url}")
        geometry = shape(record["geometry"])
        if not geometry.is_valid or geometry.is_empty:
            raise ValueError(f"Invalid Wyvern footprint {url}")
        properties = record["properties"]
        return dict(item_id=name,captured_at_utc=properties["datetime"],gsd_m=properties.get("gsd"),
            intersects_area=geometry.intersects(AREA),**meta)

    with ThreadPoolExecutor(max_workers=8) as pool:
        return metadata,list(pool.map(visit,sorted(urls)))


def export(kind,rows):
    fields = tuple(rows[0])
    write_csv(DERIVED/f"{STEM}_{kind}.csv",fields,([r[k] for k in fields] for r in rows))


def main():
    roots,event_rows = events()
    wyvern_roots,scenes = wyvern()
    results = hot()
    for name,rows in (("catalog_sources",roots+wyvern_roots),("event_collections",event_rows),
                      ("hotosm_collections",results),("wyvern_scenes",scenes)):
        export(name,rows)
    summary = dict(search_bbox=BOUNDS,hotosm_datetime=DATES,
        event_collections_checked={p:sum(r["provider"]==p for r in event_rows) for p in EVENTS},
        event_bounding_boxes_intersecting=sum(r["bounds_intersect"] for r in event_rows),
        wyvern_item_footprints_checked=len(scenes),wyvern_item_footprints_intersecting=sum(r["intersects_area"] for r in scenes),
        hotosm_intersecting_imagery_by_collection={r["collection_id"]:r["intersecting_items"] for r in results},
        scope="Only the saved public event and hyperspectral catalogs and the four named HOT collections were searched. "
              "Results do not establish coverage of subscription archives or other datasets.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
