"""Cache original Sentinel-2 metadata for geolocation and radiometry."""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit

from collect_imagery import candidate_footprint, canonical_items, earliest_alternate_items
from collect_public import RAW, ROOT, get, write
from extract_s2_radiometry import source_products

KEYS = ("granule_metadata", "product_metadata", "tileinfo_metadata")
PUBLIC_PRODUCT_METADATA = "https://roda.sentinel-hub.com/sentinel-s2-l2a/"


def fetch(item):
    directory = RAW / "sentinel2_metadata" / item["id"]
    result = {"id": item["id"], "files": {}, "errors": {}}
    for name in KEYS:
        url = item["assets"].get(name, {}).get("href", "")
        if not url.startswith("https://"):
            continue
        path = directory / (name + Path(urlsplit(url).path).suffix)
        try:
            content = get(url, path)
            if content.lstrip().startswith((b"<?xml", b"<n1:", b"<Level", b"{")):
                result["files"][name] = len(content)
            else:
                raise ValueError("Response was not recognizable metadata")
        except Exception as error:
            result["errors"][name] = str(error)
    return result


def recover_product_xml(job):
    version, product, path, assets = job
    record = {"id": version, "product_name": product}
    try:
        tile_path = path.parent / "tileinfo_metadata.json"
        if tile_path.is_file():
            tile = json.loads(tile_path.read_bytes())
        else:
            url = assets.get("tileinfo_metadata", {}).get("href", "")
            if not url.startswith("https://"):
                return {**record, "status": "no_tileinfo_route"}
            tile = json.loads(get(url, tile_path))
        product_path = tile.get("productPath", "")
        parts = product_path.split("/")
        if (len(parts) != 5 or parts[0] != "products" or not all(part.isdecimal() for part in parts[1:4])
                or parts[-1] + ".SAFE" != product or tile.get("productName", "") + ".SAFE" != product):
            return {**record, "status": "tileinfo_product_identity_unconfirmed"}
        url = PUBLIC_PRODUCT_METADATA + quote(product_path, safe="/") + "/metadata.xml"
        record["source_url"] = url
        content = get(url)
        root = ET.fromstring(content)
        if root.findtext(".//{*}PRODUCT_URI") != product:
            return {**record, "status": "xml_product_identity_mismatch"}
        write(path, content)
        if path.read_bytes() != content:
            raise ValueError(f"Metadata readback failed for {path}")
        return {**record, "status": "ready", "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
    except HTTPError as error:
        return {**record, "status": "http_error", "http_status": error.code}
    except (OSError, ET.ParseError, ValueError) as error:
        return {**record, "status": "error", "detail": str(error)}


def recover_missing_product_xml(workers):
    jobs = [(version, product, xml, assets)
            for (level, version), (product, _, xml, _, assets) in sorted(source_products().items())
            if level == "L2A" and not xml.is_file()]
    print(f"{len(jobs)} local L2A versions lack an exact product XML", flush=True)
    log = ROOT / "data" / "logs" / "sentinel2_legacy_product_xml.jsonl"
    with log.open("a") as output, ThreadPoolExecutor(max_workers=workers) as pool:
        size = workers * 3
        for index in range(0, len(jobs), size):
            results = list(pool.map(recover_product_xml, jobs[index:index + size]))
            for record in results:
                print(json.dumps(record), file=output, flush=True)
            if any(record.get("http_status") == 429 for record in results):
                print("Public metadata archive requested a wait. Remaining files can resume later.", flush=True)
                return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--historical-versions", action="store_true")
    parser.add_argument("--recover-product-xml", action="store_true",
                        help="Recover missing product XML for exact local L2A versions via the public metadata archive")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.recover_product_xml:
        recover_missing_product_xml(args.workers)
        raise SystemExit(0)
    items = canonical_items()
    if args.extended:
        layer = ROOT / "data" / "derived" / "supplemental_tank_candidates.geojson"
        _, area = candidate_footprint(layer, 120)
        items = canonical_items(RAW / "stac_supplemental", area)
        if args.historical_versions:
            items += earliest_alternate_items(RAW / "stac_supplemental", area)
    elif args.historical_versions:
        items += earliest_alternate_items()
    path = ROOT / "data" / "logs" / "sentinel2_metadata.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Metadata: {len(items)} selected Sentinel-2 items", flush=True)
    with path.open("a") as output, ThreadPoolExecutor(max_workers=args.workers) as workers:
        pending = [workers.submit(fetch, item) for item in items]
        for done in as_completed(pending):
            print(json.dumps(done.result()), file=output, flush=True)
