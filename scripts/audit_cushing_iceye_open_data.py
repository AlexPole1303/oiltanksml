"""Screen the public ICEYE STAC collection for Cushing and reference sites."""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlsplit

from shapely.geometry import box, shape
from shapely import make_valid

from build_s1_reference_matches import export
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write

STEM = "cushing_iceye_open_data"
BUCKET = "iceye-open-data-catalog.s3.amazonaws.com"
COLLECTION = f"https://{BUCKET}/collections/iceye-sar.json"
STORE = RAW / STEM
BOUNDS = (-96.81, 35.73, -96.54, 36.10)
AREA = box(*BOUNDS)


def screen(url):
    address = urlsplit(url)
    if address.scheme != "https" or address.hostname != BUCKET or not address.path.startswith("/stac-items/"):
        raise ValueError("Unexpected ICEYE catalog link")
    payload = get(url)
    item = json.loads(payload)
    if item["type"] != "Feature" or not item["geometry"] or len(item["bbox"]) != 4:
        raise ValueError(f"Missing ICEYE item geometry at {address.path}")
    footprint = make_valid(shape(item["geometry"]))
    if footprint.is_empty:
        raise ValueError(f"Empty ICEYE footprint for {item['id']}")
    bounds = footprint.bounds
    if bounds[2]-bounds[0] > 180:
        # A dateline-crossing ring cannot overlap the central United States.
        if any(abs(x)<100 for x in (bounds[0], bounds[2])):
            raise ValueError(f"Unexpected broad ICEYE footprint for {item['id']}")
        status = "antimeridian"
    else:
        status = "cushing" if footprint.intersects(AREA) else "outside_cushing"
    assets = item["assets"]
    local = ""
    if status == "cushing":
        path = STORE / "matches" / f"{item['id']}.json"
        write(path, payload)
        local = str(path.relative_to(ROOT))
    return dict(iceye_item_id=item["id"], stac_url=url, stac_sha256=hashlib.sha256(payload).hexdigest(),
        captured_utc=item["properties"].get("datetime") or item["properties"].get("start_datetime", ""),
        mode=item["properties"].get("sar:instrument_mode", ""),
        coverage_status=status, bbox_json=json.dumps(bounds), has_complex_tiff="slc-cog" in assets,
        complex_tiff_url=assets.get("slc-cog", {}).get("href", "") if status=="cushing" else "",
        local_stac=local)


def main():
    raw = get(COLLECTION, STORE / "collection.json", refresh=True)
    collection = json.loads(raw)
    if (collection["id"], collection["type"]) != ("iceye-sar", "Collection"):
        raise ValueError("Unexpected ICEYE collection")
    urls = [r["href"] for r in collection["links"] if r["rel"] == "item"]
    if not urls or len(urls) != len(set(urls)) or any(r["rel"] == "next" for r in collection["links"]):
        raise ValueError("Incomplete ICEYE item index")
    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(screen, urls))
    if len({r["iceye_item_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated ICEYE acquisition ID")
    export(f"{STEM}_catalog", rows)
    summary = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), source_collection=COLLECTION,
        source_collection_sha256=hashlib.sha256(raw).hexdigest(), cushing_bbox=BOUNDS, screened_items=len(rows),
        coverage=Counter(r["coverage_status"] for r in rows), items_with_complex_tiff=sum(r["has_complex_tiff"] for r in rows),
        interpretation="Coverage audit of the published static open-data collection. It does not query the commercial archive.")
    payload = json.dumps(summary, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
