"""Preserve the B04 detector masks from the exact original 2017–2018 L1C products."""
import csv
import json
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from collect_public import ROOT, get, write_csv
from collect_s2_l1c import OBJECTS, OUT

DERIVED = ROOT / "data" / "derived"
MASK = "MSK_DETFOO_B04.gml"


def collect(row):
    scene, product = row["canonical_l2a_id"], row["l1c_product_name"]
    archive = json.loads((OUT / "object_indexes" / f"{product}.json").read_bytes())
    assets = [item for item in archive if item["name"].endswith(f"/{MASK}")]
    if len(assets) != 1:
        raise ValueError(f"Expected one B04 detector mask in {product}")
    asset = assets[0]
    url = OBJECTS + quote(asset["name"], safe="/")
    path = OUT / "red_detector_footprints" / row["captured_at_utc"][:4] / scene / MASK
    payload = get(url, path)
    if len(payload) != int(asset["size"]):
        raise ValueError(f"B04 detector mask length disagrees with archive index for {product}")
    root = ET.fromstring(payload)
    polygons = sum(node.tag.rsplit("}", 1)[-1] == "Polygon" for node in root.iter())
    if root.tag != "{http://www.opengis.net/eop/2.0}Mask":
        raise ValueError(f"Unexpected B04 detector mask document type for {product}")
    status = "polygons_present" if polygons else "empty_source_mask"
    return (scene, row["captured_at_utc"], product, polygons, status, asset["size"], asset["crc32c"], asset["timeCreated"],
            str(path.relative_to(ROOT)), url)


def main():
    with (DERIVED / "sentinel2_l1c_archive_availability.csv").open() as file:
        rows = [row for row in csv.DictReader(file) if row["captured_at_utc"][:4] in {"2017", "2018"}]
    with ThreadPoolExecutor(max_workers=8) as workers:
        results = list(workers.map(collect, rows))
    write_csv(DERIVED / "sentinel2_l1c_early_detector_footprints.csv",
              ("scene_item_id", "captured_at_utc", "l1c_product_name", "polygon_count", "mask_status", "bytes",
               "source_crc32c", "source_created_at_utc", "local_file", "source_url"), results)
    print(f"{len(results)} early L1C detector masks verified, {sum(row[3]==0 for row in results)} empty source masks")


if __name__ == "__main__":
    main()
