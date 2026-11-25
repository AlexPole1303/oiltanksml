"""Search official open ICEYE and Capella archives for independent Cushing observations."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

from shapely.geometry import shape

from build_weekly_shadow_panel import DERIVED
from collect_imagery import candidate_footprint
from collect_public import RAW, ROOT, get, write, write_csv

STEM = "cushing_open_sar"
STORE = RAW / STEM
ICEYE = "https://iceye-open-data-catalog.s3.amazonaws.com/collections/iceye-sar.json"
ICEYE_FOOTPRINTS = "https://iceye-open-data-catalog.s3.amazonaws.com/stac-items/summary/"
CAPELLA = "https://capella-open-data.s3.us-west-2.amazonaws.com/stac/capella-open-data-by-product-type/"
FIELDS = ("provider", "item_id", "captured_at_utc", "mode", "product_type", "bbox", "intersects_tank_footprint",
          "resolution_range_m", "resolution_azimuth_m", "stac_url", "stac_file", "stac_sha256", "asset_urls_json")
START = "2017-01-01"
END = "2026-10-31"


def items(url, provider, collection, refresh):
    path = STORE / provider / f"{collection}.json"
    snapshot = json.loads(get(url, path, refresh=refresh))
    links = {urljoin(url, link["href"]) for link in snapshot["links"] if link["rel"] == "item"}
    if not links or any(link["rel"] == "next" for link in snapshot["links"]):
        raise ValueError(f"Incomplete item link list for {url}")
    return links


def capella_capture(url):
    fields = urlsplit(url).path.rsplit("/", 1)[-1].removesuffix(".json").split("_")
    if len(fields) != 7 or fields[0] != "CAPELLA" or fields[3] not in ("SICD", "SLC"):
        raise ValueError(f"Unexpected Capella product ID at {url}")
    return "_".join(fields[:3] + fields[4:])


def acquisition_day(url):
    filename = urlsplit(url).path.rsplit("/", 1)[-1].removesuffix(".json")
    if filename.startswith("CAPELLA_"):
        stamp = filename.split("_")[5]
    elif filename.startswith("ICEYE_"):
        stamp = filename.split("_")[2]
    else:
        raise ValueError(f"Unrecognized product ID {filename}")
    return datetime.strptime(stamp[:8], "%Y%m%d").date().isoformat()


def scan_item(job, area):
    provider, url = job
    name = urlsplit(url).path.rsplit("/", 1)[-1]
    path = STORE / provider / "items" / name
    raw = get(url, path)
    item = json.loads(raw)
    if item["type"] != "Feature" or item["id"] != path.stem:
        raise ValueError(f"Unexpected STAC identity at {url}")
    footprint = shape(item["geometry"])
    if footprint.is_empty or not footprint.is_valid:
        raise ValueError(f"Invalid STAC footprint at {url}")
    p = item["properties"]
    captured = p["datetime"]
    if not START <= captured[:10] <= END:
        raise ValueError(f"Unexpected capture date {captured} at {url}")
    if not footprint.intersects(area):
        return None
    return dict(provider=provider, item_id=item["id"], captured_at_utc=captured, mode=p.get("sar:instrument_mode"),
                product_type=p.get("sar:product_type"), bbox=json.dumps(item["bbox"]),
                intersects_tank_footprint=True, resolution_range_m=p.get("sar:resolution_range"),
                resolution_azimuth_m=p.get("sar:resolution_azimuth"), stac_url=url, stac_file=str(path.relative_to(ROOT)),
                stac_sha256=hashlib.sha256(raw).hexdigest(),
                asset_urls_json=json.dumps({key: val["href"] for key,val in item["assets"].items()}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-catalogs", action="store_true")
    args = parser.parse_args()
    _, core = candidate_footprint(DERIVED / "osm_tank_candidates.geojson", 120)
    _, supplemental = candidate_footprint(DERIVED / "supplemental_tank_candidates.geojson", 120)
    area = core.union(supplemental)
    iceye = items(ICEYE, "iceye", "collection", args.refresh_catalogs)
    iceye.remove(ICEYE_FOOTPRINTS + "000-iceye-sar-collection-footprints.json")
    footprints = json.loads(get(ICEYE_FOOTPRINTS + "footprint-summary-points.geojson",
                               STORE / "iceye/footprints.geojson", refresh=args.refresh_catalogs))["features"]
    footprint_urls = {r["properties"]["href"] for r in footprints}
    if len(footprint_urls) != len(footprints):
        raise ValueError("Repeated ICEYE footprint item")
    iceye_candidates = iceye - footprint_urls
    iceye_candidates.update(r["properties"]["href"] for r in footprints if shape(r["geometry"]).intersects(area))
    capella = {kind: items(f"{CAPELLA}capella-open-data-{kind}/collection.json", "capella", kind, args.refresh_catalogs)
               for kind in ("sicd", "slc")}
    selected = {"iceye": iceye_candidates, "capella": capella["sicd"] | capella["slc"]}
    jobs = [(provider, url) for provider, urls in selected.items() for url in sorted(urls)
            if START <= acquisition_day(url) <= END]
    print(dict(Counter(provider for provider, _ in jobs)), flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = [row for row in pool.map(lambda job: scan_item(job, area), jobs) if row]
    rows.sort(key=lambda r: (r["provider"], r["captured_at_utc"], r["item_id"]))
    write_csv(DERIVED / f"{STEM}_scenes.csv", FIELDS, ([r[key] for key in FIELDS] for r in rows))
    summary = dict(search_utc=datetime.now(timezone.utc).isoformat(), first_capture_day=START, last_capture_day=END,
                   collection_links=dict(iceye=len(iceye), capella_sicd=len(capella["sicd"]), capella_slc=len(capella["slc"])),
                   iceye_footprint_rows=len(footprints), iceye_collection_items_missing_in_footprints=len(iceye-footprint_urls),
                   iceye_footprints_not_in_collection=len(footprint_urls-iceye),
                   checked_items=dict(Counter(provider for provider, _ in jobs)),
                   matched_items=dict(Counter(r["provider"] for r in rows)),
                   capella_unique_captures=len({capella_capture(r["stac_url"]) for r in rows if r["provider"]=="capella"}),
                   tank_footprint_search_bounds=area.bounds,
                   interpretation="ICEYE uses official per-item footprints, with source STAC for intersecting and unlisted items. "
                                  "Capella uses source STAC polygons. Mapped tanks are buffered by 120 m. "
                                  "Capella source products are retained independently and counted by capture and polarization. "
                                  "A footprint intersection establishes a candidate for source-pixel checks.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
