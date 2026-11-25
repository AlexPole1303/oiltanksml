"""Archive narrowly dated public catalog responses for the Rose Rock control."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

from build_cushing_icesat2_optical_controls import CDSE
from build_cushing_nasa_satellite_inventory import ASTER, CMR
from build_s1_reference_matches import DERIVED, export, read
from collect_landsat import USGS
from collect_public import EARTH_SEARCH, RAW, ROOT, TNM, get, write

STEM = "cushing_rose_rock_2536_scene_catalogs"
STORE = RAW/"cushing_rose_rock_2536"/"catalogs"
BOUNDS = (-96.7561, 36.0197, -96.7542, 36.0213)
START, END = "2024-06-21T00:00:00Z", "2024-06-25T00:00:00Z"
NASA_COLLECTIONS = "https://cmr.earthdata.nasa.gov/search/collections.json"
CSDA_KEYWORDS = {"planetscope": "PlanetScope", "vantor": "WorldView", "airbus": "Airbus", "geoeye": "GeoEye",
                 "blacksky": "BlackSky", "satellogic": "Satellogic", "skysat": "SkySat", "pleiades": "Pleiades", "legion": "Legion"}


def snapshot(label, url):
    path = STORE/f"{label}.json"
    raw = get(url, path)
    return json.loads(raw), dict(source_url=url, source_path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest())


def open_queries(bbox):
    w, s, e, n = BOUNDS
    poly = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    where = (f"Collection/Name eq 'SENTINEL-2' and contains(Name,'MSIL1C') and ContentDate/Start ge {START} "
             f"and ContentDate/Start lt {END} and OData.CSC.Intersects(area=geography'SRID=4326;{poly}')")
    daterange = f"{START}/{END}"
    for label, url, mode in (
        ("sentinel2_cdse", CDSE+"?"+urlencode({"$filter": where, "$top": 100}), "odata"),
        ("sentinel2_earth_search", EARTH_SEARCH+"?"+urlencode(dict(collections="sentinel-2-l1c", bbox=bbox, datetime=daterange, limit=100)), "stac"),
        ("landsat_usgs", USGS+urlencode(dict(collections="landsat-c2l1", bbox=bbox, datetime=daterange, limit=100)), "stac"),
        ("spot_usgs_tnm", TNM+"?"+urlencode(dict(datasets="High Resolution Orthoimagery (HRO)", bbox=bbox, max=50)), "tnm"),
    ):
        yield label, url, mode


def cmr_query(label, bbox, **kwargs):
    return label, CMR+"?"+urlencode(dict(kwargs, bounding_box=bbox, temporal=f"{START},{END}", page_size=100)), "cmr"


def granules(job):
    label, url, mode = job
    data, provenance = snapshot(label, url)
    if mode == "cmr":
        ids = [r["umm"]["GranuleUR"] for r in data["items"]]
        expected = data["hits"]
    elif mode == "stac":
        ids = [r["properties"].get("s2:product_uri", r["id"]) for r in data["features"]]
        expected = data.get("numberMatched", data.get("context", {}).get("matched"))
        if expected is None and any(r["rel"] == "next" for r in data["links"]):
            raise ValueError(f"Unretrieved catalog page for {label}")
    elif mode == "odata":
        ids = [r["Name"] for r in data["value"]]
        expected = len(ids) if not data.get("@odata.nextLink") else None
    else:
        ids = [r["title"] for r in data["items"]]
        expected = data["total"] if not data["errors"] else None
    if expected != len(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Catalog coverage is incomplete or duplicated for {label}")
    return dict(catalog_id=label, source_format=mode, matched_items=len(ids), matched_item_ids_json=json.dumps(sorted(ids)), **provenance)


def csda_collections(item):
    key, keyword = item
    size = {"planetscope": 50, "vantor": 80, "airbus": 60}.get(key, 100)
    url = NASA_COLLECTIONS+"?"+urlencode(dict(provider="CSDA", keyword=keyword, page_size=size))
    data, provenance = snapshot(f"{key}_collections", url)
    rows = data["feed"]["entry"]
    if len(rows) >= size or any(link.get("rel", "").endswith("next") for link in data["feed"].get("links", [])):
        raise ValueError(f"Incomplete NASA collection discovery for {keyword}")
    ids = {r["short_name"]: r["id"] for r in rows}
    return ids, dict(keyword=keyword, collections=len(ids), collections_json=json.dumps(ids, sort_keys=True), **provenance)


def main():
    bbox = ",".join(map(str, BOUNDS))
    with ThreadPoolExecutor(max_workers=6) as pool:
        discoveries = list(pool.map(csda_collections, CSDA_KEYWORDS.items()))
        collections = {key: value for names, _ in discoveries for key, value in names.items()}
        jobs = [*open_queries(bbox), cmr_query("aster", bbox, collection_concept_id=ASTER),
                cmr_query("aster_v3", bbox, short_name="AST_L1T", version="003")]
        jobs.extend(cmr_query("planetscope_3b" if key == "PSScene3Band" else key, bbox, collection_concept_id=concept)
                    for key, concept in sorted(collections.items()))
        rows = list(pool.map(granules, jobs))
    by_key = {r["catalog_id"]: set(json.loads(r["matched_item_ids_json"])) for r in rows}
    expected = {r["product_id"] for r in read("cushing_phmsa_controls_rose_rock_optical") if r["same_local_day"] == "True"}
    if expected != by_key["sentinel2_cdse"] or expected != by_key["sentinel2_earth_search"]:
        raise ValueError("The catalog search returned different versions from the original incident-day measurement")
    result = dict(bounds=BOUNDS, capture_search_start_utc=START, capture_search_end_utc=END,
                  tnm_hro_search_uses_geography_only=True, queried_product_catalogs=len(rows), queried_public_csda_collections=len(collections),
                  matched_items={r["catalog_id"]: r["matched_items"] for r in rows if r["matched_items"]},
                  additional_confirmed_scenes=sum(r["matched_items"] for r in rows if not r["catalog_id"].startswith("sentinel2_")),
                  scope="Checks saved public catalog responses. It does not establish the coverage of full commercial archives.")
    export(f"{STEM}_queries", rows)
    export(f"{STEM}_csda_collection_discovery", [row for _, row in discoveries])
    payload = json.dumps(result, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
