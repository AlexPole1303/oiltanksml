"""Reconcile the acquisition catalog, exact archived versions, and local L1C files."""
import argparse
import csv
import datetime as dt
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from collect_imagery import candidate_footprint, canonical_items
from collect_l1c_catalog_gaps import AWS
from collect_public import RAW, ROOT, write_csv
from collect_s2_l1c import OBJECTS, OUT, crc32c_base64, local_granule_files, source_metadata_bytes, source_metadata_path
from collect_s2_l1c_archive_times import CLOUD_MASKS

DERIVED = ROOT / "data" / "derived"
FIELDS = ("canonical_l2a_id", "captured_at_utc", "provider", "version_id", "archive_available_at_utc",
          "local_directory", "status", "detail")


def index_csv(filename):
    with (DERIVED / filename).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {row["canonical_l2a_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate acquisition in {filename}")
    return result


def catalog():
    _, core = candidate_footprint(DERIVED / "osm_tank_candidates.geojson", 120)
    _, supplemental = candidate_footprint(DERIVED / "supplemental_tank_candidates.geojson", 120)
    canonical = {item["id"]: item["properties"]["datetime"]
                 for item in canonical_items(RAW / "stac_supplemental", core.union(supplemental))}
    gcs = index_csv("sentinel2_l1c_archive_availability.csv")
    aws = index_csv("sentinel2_l1c_alternate_evidence.csv")
    if set(gcs) != set(canonical) or set(aws) - set(canonical):
        raise ValueError("Archive evidence and acquisition catalog differ")
    for identity, captured in sorted(canonical.items(), key=lambda item: (item[1], item[0])):
        entry = gcs[identity]
        if dt.datetime.fromisoformat(entry["captured_at_utc"]) != dt.datetime.fromisoformat(captured):
            raise ValueError(f"Catalog acquisition time differs for {identity}")
        if entry["all_required_objects_created_by_utc"]:
            yield identity, captured, "gcs", entry["l1c_product_name"], entry["all_required_objects_created_by_utc"]
        elif identity in aws and aws[identity]["all_inputs_available_at_utc"]:
            entry = aws[identity]
            yield identity, captured, "aws", entry["l1c_product_name"], entry["all_inputs_available_at_utc"]
        else:
            yield identity, captured, None, None, None


def verify_folder(entry, path):
    identity, captured, provider, product, available = entry
    directory = path.parent
    record = local_granule_files(directory, identity, product)
    if dt.datetime.fromisoformat(record["captured_at_utc"]) != dt.datetime.fromisoformat(captured):
        raise ValueError("Manifest acquisition time differs")
    if dt.datetime.fromisoformat(available) < dt.datetime.fromisoformat(captured):
        raise ValueError("Archive availability precedes acquisition")
    if record["archive"] != {"gcs": OBJECTS, "aws": AWS}[provider]:
        raise ValueError("Manifest archive differs")
    xml_bytes = source_metadata_bytes(source_metadata_path(directory / "MTD_MSIL1C.xml"))
    if ET.fromstring(xml_bytes).findtext(".//{*}PRODUCT_URI") != product:
        raise ValueError("Product XML identity differs")
    if provider == "gcs":
        objects = json.loads((OUT / "object_indexes" / f"{product}.json").read_bytes())
        by_name = {obj["name"]: obj for obj in objects}
        for asset in [*record["assets"].values(), *record["granule_metadata"]]:
            reference = by_name[asset["name"]]
            if any(reference[field] != asset[field] for field in ("size", "timeCreated", "crc32c")):
                raise ValueError(f"Object manifest disagrees with GCS index for {asset['name']}")
        product_xml, = [obj for obj in objects if obj["name"].endswith("/MTD_MSIL1C.xml")]
        if len(xml_bytes) != int(product_xml["size"]) or crc32c_base64(xml_bytes) != product_xml["crc32c"]:
            raise ValueError("Product XML checksum differs from GCS index")
        cloud = [asset for asset in record["granule_metadata"] if asset["name"].rsplit("/", 1)[-1] in CLOUD_MASKS]
        tile, = [asset for asset in record["granule_metadata"] if asset["name"].endswith("/MTD_TL.xml")]
        required = [*record["assets"].values(), *cloud, tile, product_xml]
        if not cloud or max(dt.datetime.fromisoformat(asset["timeCreated"]) for asset in required) != dt.datetime.fromisoformat(available):
            raise ValueError("Required GCS objects do not reproduce the archive availability clock")
    elif dt.datetime.fromisoformat(record["band_and_granule_metadata_objects_created_by_utc"]) != dt.datetime.fromisoformat(available):
        raise ValueError("AWS manifest and archive availability clock differ")


def reconcile(entry, manifests):
    if entry[2] is None:
        return [(*entry, None, "unresolved_archive_version", "No exact version with an archive availability clock")]
    if not manifests:
        return [(*entry, None, "missing_local_manifest", "No local granule manifest for this acquisition")]
    rows = []
    for path in sorted(manifests):
        try:
            verify_folder(entry, path)
        except (OSError, KeyError, ValueError, ET.ParseError, EOFError) as error:
            status, detail = "failed_validation", f"{type(error).__name__}: {error}"
        else:
            status, detail = "verified", ""
        rows.append((*entry, str(path.parent.relative_to(ROOT)), status, detail))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true", help="Exit unsuccessfully if any acquisition or folder is unverified")
    args = parser.parse_args()
    expected = list(catalog())
    manifests = defaultdict(list)
    for path in OUT.glob("**/granules/*/*/item.json"):
        manifests[path.parent.name].append(path)
    unexpected = set(manifests) - {entry[0] for entry in expected}
    if unexpected:
        raise ValueError(f"Local acquisitions absent from the acquisition catalog: {sorted(unexpected)}")
    results = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for index, rows in enumerate(pool.map(lambda entry: reconcile(entry, manifests.get(entry[0], ())), expected), 1):
            results.extend(rows)
            if index % 500 == 0:
                print(f"{index}/{len(expected)} acquisitions checked", flush=True)
    write_csv(DERIVED / "sentinel2_l1c_source_readback.csv", FIELDS, results)
    counts = Counter(row[-2] for row in results)
    verified = len({row[0] for row in results if row[-2] == "verified"})
    print(f"{verified}/{len(expected)} acquisitions with verified local files, folder results: {dict(counts)}")
    if args.require_complete and (verified != len(expected) or set(counts) != {"verified"}):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
