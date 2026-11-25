"""Recover L1C pixels and publication evidence for exact acquisitions missing from GCS."""
import csv
import datetime as dt
import hashlib
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from shapely.geometry import shape

from collect_imagery import candidate_footprint, canonical_items, clip
from collect_public import RAW, ROOT, TLS, get, write, write_csv
from collect_s2_l1c import CHANNELS, OUT, local_granule_files, parse_product

DERIVED = ROOT / "data" / "derived"
AWS = "https://sentinel-s2-l1c.s3.eu-central-1.amazonaws.com/"
CDSE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
CDSE_RAW = RAW / "sentinel2_cdse_l1c_gaps"


def s3_object(url, local_name):
    with urlopen(Request(url, method="HEAD"), context=TLS, timeout=30) as response:
        headers = response.headers
        return {"url": url, "local_name": local_name, "size": int(headers["Content-Length"]),
                "etag": headers["ETag"].strip('"'),
                "last_modified_at_utc": parsedate_to_datetime(headers["Last-Modified"]).isoformat()}


def store_object(asset, directory):
    path = directory / asset["local_name"]
    payload = get(asset["url"], path)
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    if len(payload) != asset["size"] or digest != asset["etag"]:
        raise ValueError(f"AWS object length or ETag mismatch for {path}")


def acquire_aws(l1c, l2a, bounds, footprint, *, out=OUT):
    props = l1c["properties"]
    product, captured = props["s2:product_uri"], l2a["properties"]["datetime"]
    directory = out / "alternate_aws" / "granules" / captured[:4] / l2a["id"]
    asset_path = l1c["assets"]["red"]["href"].removeprefix("s3://sentinel-s2-l1c/").rsplit("/", 1)[0]
    tile_base = AWS + asset_path + "/"
    tile_info = s3_object(tile_base + "tileInfo.json", "tileInfo.json")
    store_object(tile_info, directory)
    info = json.loads((directory / tile_info["local_name"]).read_bytes())
    if info["productName"] + ".SAFE" != product or dt.datetime.fromisoformat(info["timestamp"]) != dt.datetime.fromisoformat(captured):
        raise ValueError(f"AWS acquisition identity mismatch for {l2a['id']}")
    bands = {channel: s3_object(tile_base + band + ".jp2", channel + ".tif") for channel, band in CHANNELS.items()}
    metadata = [s3_object(tile_base + remote, name) for remote, name in (
        ("metadata.xml", "MTD_TL.xml"), ("qi/CLASSI_B00.jp2", "MSK_CLASSI_B00.jp2"),
        ("qi/DETFOO_B04.jp2", "MSK_DETFOO_B04.jp2"))]
    product_meta = s3_object(AWS + info["productPath"] + "/metadata.xml", "MTD_MSIL1C.xml")
    for asset in [*metadata, product_meta]:
        store_object(asset, directory)
    for asset in bands.values():
        asset["clip_dimensions"] = clip(asset["url"], directory / asset["local_name"], bounds, footprint)
    archived = max([props["created"], *(obj["last_modified_at_utc"] for obj in [*bands.values(), *metadata,
                     tile_info, product_meta])], key=dt.datetime.fromisoformat)
    fields = {node.tag.rsplit("}", 1)[-1]: node.text for node in ET.parse(directory / product_meta["local_name"]).iter()}
    if fields["PRODUCT_URI"] != product:
        raise ValueError(f"AWS product XML identity mismatch for {l2a['id']}")
    generated = fields["GENERATION_TIME"]
    record = {"canonical_l2a_id": l2a["id"], "product_name": product, "captured_at_utc": captured,
              "generated_at_utc": generated, "processing_level": "L1C", "assets": bands,
              "granule_metadata": [*metadata, product_meta, tile_info], "archive": AWS,
              "band_and_granule_metadata_objects_created_by_utc": archived}
    local_granule_files(directory, l2a["id"], product, record)
    write(directory / "item.json", json.dumps(record, indent=2).encode())
    return archived, str(directory.relative_to(ROOT)), tile_info["url"]


def cdse_records():
    results = {}
    for path in CDSE_RAW.glob("*.json"):
        for record in json.loads(path.read_bytes())["value"]:
            results[record["Name"]] = record, path
    return results


def main():
    with (DERIVED / "sentinel2_l1c_archive_availability.csv").open() as file:
        missing = [row for row in csv.DictReader(file) if row["status"] == "no_matching_product"]
    with (DERIVED / "eia_satellite_window.csv").open() as file:
        windows = [(dt.datetime.fromisoformat(row["window_starts_after_utc"]),
                    dt.datetime.fromisoformat(row["window_ends_by_utc"]),
                    dt.datetime.fromisoformat(row["inputs_available_before_utc"])) for row in csv.DictReader(file)]
    core_footprint, core_area = candidate_footprint(DERIVED / "osm_tank_candidates.geojson", 120)
    supp_footprint, supp_area = candidate_footprint(DERIVED / "supplemental_tank_candidates.geojson", 120)
    footprint, area = core_footprint.union(supp_footprint), core_area.union(supp_area)
    canonical = {item["id"]: item for item in canonical_items(RAW / "stac_supplemental", area)}
    catalog = RAW / "stac_supplemental" / "sentinel-2-l1c"
    l1c_index = {}
    for year in sorted({row["captured_at_utc"][:4] for row in missing}):
        for item in json.loads((catalog / f"{year}.geojson").read_bytes())["features"]:
            props = item["properties"]
            key, _, _ = parse_product(props["s2:product_uri"])
            l1c_index.setdefault((key[0], key[2], key[3], props["datetime"]), []).append(item)
    cdse = cdse_records()

    def reconcile(row):
        l2a = canonical[row["canonical_l2a_id"]]
        key, _, _ = parse_product(l2a["properties"]["s2:product_uri"])
        l1c = min(l1c_index[(key[0], key[2], key[3], row["captured_at_utc"])],
                  key=lambda item: item["properties"]["created"])
        props = l1c["properties"]
        product = props["s2:product_uri"]
        if product not in cdse:
            url = CDSE + urlencode({"$filter": f"Name eq '{product}'"})
            path = CDSE_RAW / f"{product}.json"
            record, = json.loads(get(url, path))["value"]
        else:
            record, path = cdse[product]
        if record["Name"] != product:
            raise ValueError(f"Copernicus product identity mismatch for {product}")
        captured = dt.datetime.fromisoformat(row["captured_at_utc"])
        relevant_cutoff = max(cutoff for start, end, cutoff in windows if start < captured <= end)
        available, directory, asset_url = acquire_aws(l1c, l2a, area.intersection(shape(l2a["geometry"])).bounds, footprint)
        status = ("complete_aws_asset_clock" if dt.datetime.fromisoformat(available) < relevant_cutoff
                  else "complete_aws_asset_clock_after_study_cutoff")
        print(l2a["id"], status, flush=True)
        source_url = CDSE.split("?")[0] + f"({record['Id']})"
        return (l2a["id"], row["captured_at_utc"], l1c["id"], product, props["created"],
                record["PublicationDate"], status, available, directory, asset_url, source_url,
                str(path.relative_to(ROOT)))

    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(reconcile, missing))
    write_csv(DERIVED / "sentinel2_l1c_alternate_evidence.csv",
              ("canonical_l2a_id", "captured_at_utc", "earth_search_l1c_id", "l1c_product_name",
               "earth_search_created_at_utc", "cdse_published_at_utc", "status", "all_inputs_available_at_utc",
               "local_directory", "aws_tile_info_url", "cdse_product_url", "source_file"), rows)
    timely = sum(row[6] == "complete_aws_asset_clock" for row in rows)
    print(f"{len(rows)} verified local granules, {timely} available before at least one relevant cutoff")


if __name__ == "__main__":
    main()
