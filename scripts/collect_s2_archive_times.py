"""Preserve public S3 object clocks for selected Sentinel-2 L2A files."""
import argparse
import csv
import datetime as dt
import io
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote, urlencode, urlsplit

from collect_imagery import BANDS, candidate_footprint, canonical_items, earliest_alternate_items
from collect_public import RAW, ROOT, get, write

ARCHIVE = RAW / "sentinel2_s3_indexes"
NAMESPACE = "{http://s3.amazonaws.com/doc/2006-03-01/}"
KEYS = BANDS + ("granule_metadata", "product_metadata")
HOSTS = {"sentinel-cogs.s3.us-west-2.amazonaws.com", "e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com"}


def prefix(asset):
    location = urlsplit(asset["href"])
    key = unquote(location.path.lstrip("/"))
    if location.hostname not in HOSTS or len(key.split("/")) < 8:
        raise ValueError(f"Unexpected public Sentinel-2 COG asset location: {location.hostname}, {key}")
    return (location.hostname, "/".join(key.split("/")[:6]) + "/"), key


def object_clocks(group):
    host, parent = group
    params = {"list-type": "2", "prefix": parent}
    contents = {}
    for page in range(1, 25):
        url = f"https://{host}/?{urlencode(params)}"
        path = ARCHIVE / host / parent / f"{page:02d}.xml"
        tree = ET.fromstring(get(url, path))
        if tree.findtext(NAMESPACE + "Prefix") != parent:
            raise ValueError(f"S3 archive listing returned a different prefix: {parent}")
        for item in tree.findall(NAMESPACE + "Contents"):
            key = item.findtext(NAMESPACE + "Key")
            contents[key] = (item.findtext(NAMESPACE + "LastModified"), item.findtext(NAMESPACE + "Size"),
                             item.findtext(NAMESPACE + "ETag"))
        if tree.findtext(NAMESPACE + "IsTruncated") == "false":
            return group, contents
        params["continuation-token"] = tree.findtext(NAMESPACE + "NextContinuationToken")
        if not params["continuation-token"]:
            raise ValueError(f"Incomplete S3 archive listing for {parent}")
    raise ValueError(f"Exceeded expected S3 page count for {parent}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    layer = ROOT / "data" / "derived" / "supplemental_tank_candidates.geojson"
    _, area = candidate_footprint(layer, 120)
    items = canonical_items(RAW / "stac_supplemental", area) + earliest_alternate_items(RAW / "stac_supplemental", area)
    groups = set()
    for item in items:
        for key in KEYS:
            if key in item["assets"]:
                groups.add(prefix(item["assets"][key])[0])
    print(f"{len(items)} scene versions and {len(groups)} S3 month prefixes", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        clocks = dict(workers.map(object_clocks, sorted(groups)))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("item_id", "captured_at_utc", "catalog_created_at_utc", "asset_key", "asset_url", "last_modified_utc",
                     "object_size_bytes", "object_etag", "archive_host"))
    found = 0
    for item in items:
        captured = item["properties"]["datetime"]
        for key in KEYS:
            asset = item["assets"].get(key)
            if asset is None:
                continue
            group, object_key = prefix(asset)
            modified, size, etag = clocks[group].get(object_key, (None, None, None))
            if modified and dt.datetime.fromisoformat(modified) < dt.datetime.fromisoformat(captured):
                raise ValueError(f"S3 object modification time predates acquisition: {object_key}")
            writer.writerow((item["id"], captured, item["properties"].get("created"), key, asset["href"], modified, size, etag, group[0]))
            found += modified is not None
    write(ROOT / "data" / "derived" / "sentinel2_archive_asset_times.csv", stream.getvalue().encode())
    print(f"{found} object clocks reconciled to the selected STAC assets")


if __name__ == "__main__":
    main()
