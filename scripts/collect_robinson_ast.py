"""Collect the published AST labels and every Cushing image in the sample catalogue."""

import csv
import hashlib
import io
import json
import struct
import xml.etree.ElementTree as ET
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from math import hypot
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from pyproj import Transformer

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_ceda_s1_slc_samples import RangeZip
from collect_public import RAW, ROOT, TLS, get, write

STEM = "robinson_ast"
STORE = RAW / STEM
ARTICLES = (22543456, 22665787, 22661005, 22661011)
SMALL = {"xmls.zip", "tank_inventory_shp.zip", "tank_inventory_metadata.csv", "image_characteristics.csv",
         "image_characteristics_metadata.csv", "tile_characteristics.csv", "tile_characteristics_metadata.csv"}


def download(row):
    length = row["source_bytes"]
    path = STORE / row["file_name"]
    if path.is_file():
        data = path.read_bytes()
    else:
        request = Request(row["source_url"], headers={"Range": f"bytes=0-{length-1}", "Accept-Encoding": "identity"})
        with urlopen(request, context=TLS, timeout=50) as response:
            data = response.read(length+1)
    if len(data) != length or hashlib.md5(data).hexdigest() != row["source_md5"]:
        raise ValueError(f"Figshare checksum mismatch for {row['file_name']}")
    if not path.is_file():
        write(path, data)
    row["local_file"] = str(path.relative_to(ROOT))
    row["local_sha256"] = hashlib.sha256(data).hexdigest()


def jpeg(remote, entry):
    path = STORE / entry.filename
    if path.is_file():
        data = path.read_bytes()
    else:
        header = remote.slice(entry.header_offset, 30)
        signature, _, flags, codec, _, _, _, _, _, name_size, extra = struct.unpack("<4s5H3I2H", header)
        if signature != b"PK\x03\x04" or codec != entry.compress_type or flags != entry.flag_bits or flags & 1:
            raise ValueError(f"Unexpected ZIP entry header for {entry.filename}")
        packet = remote.slice(entry.header_offset+30, name_size+extra+entry.compress_size)
        if packet[:name_size].decode("utf-8") != entry.filename:
            raise ValueError(f"ZIP image name disagrees for {entry.filename}")
        compressed = packet[name_size+extra:]
        if codec == ZIP_DEFLATED:
            data = zlib.decompress(compressed, wbits=-15)
        elif codec == ZIP_STORED:
            data = compressed
        else:
            raise ValueError(f"Unexpected ZIP image codec for {entry.filename}")
    if len(data) != entry.file_size or zlib.crc32(data) != entry.CRC:
        raise ValueError(f"ZIP image CRC mismatch for {entry.filename}")
    with Image.open(io.BytesIO(data)) as image:
        if image.size != (512, 512) or image.format != "JPEG":
            raise ValueError(f"Unexpected image format or dimensions for {entry.filename}")
        image.verify()
    if not path.is_file():
        write(path, data)
    return dict(local_jpeg=str(path.relative_to(ROOT)), jpeg_sha256=hashlib.sha256(data).hexdigest(),
                jpeg_zip_crc32=f"{entry.CRC:08x}", jpeg_zip_member_offset=entry.header_offset)


def registry_matches(inventory, labels):
    images = {row["image_name"]: row for row in inventory}
    registry = read("tank_engineering_registry")
    projected = {}
    for crs in {row["utm_projection"] for row in inventory}:
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        projected[crs] = [(r, *transformer.transform(float(r["longitude"]), float(r["latitude"]))) for r in registry]
    for label in labels:
        if label["truncated"] != "0" or label["difficult"] != "0":
            continue
        image = images[label["image_name"]]
        crs = image["utm_projection"]
        origin_x = float(image["nw_x_utm_image_coord"])
        origin_y = float(image["nw_y_utm_image_coord"])
        step_x = (float(image["se_x_utm_image_coord"]) - origin_x) / 511
        step_y = (origin_y - float(image["se_y_utm_image_coord"])) / 511
        west, east = (origin_x + step_x * (label[k] - 1) for k in ("xmin", "xmax"))
        north, south = (origin_y - step_y * (label[k] - 1) for k in ("ymin", "ymax"))
        x, y = (west + east) / 2, (north + south) / 2
        capture = image["capture_date"]
        candidates = [(r, hypot(tx - x, ty - y)) for r, tx, ty in projected[crs] if west <= tx <= east and south <= ty <= north
                      and r["first_seen"] and r["first_seen"] <= capture and r["footprint_diameter_m"]
                      and hypot(tx - x, ty - y) <= max(5, float(r["footprint_diameter_m"]) / 4)]
        row, distance = candidates[0] if len(candidates) == 1 else ({}, None)
        yield dict(label, capture_date=capture, candidate_count=len(candidates), candidate_id=row.get("candidate_id"),
                   registry_roof_type=row.get("documented_roof_type"), box_to_registry_center_m=distance)


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        payloads = list(pool.map(lambda article: get(f"https://api.figshare.com/v2/articles/{article}"), ARTICLES))
    sources = []
    for payload in payloads:
        article = json.loads(payload)
        write(STORE / f"figshare_{article['id']}.json", payload)
        if article["version"] != 1 or article["license"]["name"] != "CC BY 4.0":
            raise ValueError(f"Figshare version or licence changed for {article['id']}")
        for f in article["files"]:
            sources.append(dict(article_doi=article["doi"], article_version=article["version"], licence=article["license"]["url"],
                file_id=f["id"], file_name=f["name"], source_url=f["download_url"]+"?version=1", source_bytes=f["size"],
                source_md5=f["computed_md5"], local_file=None, local_sha256=None))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, [row for row in sources if row["file_name"] in SMALL]))
    export(f"{STEM}_sources", sources)
    with (STORE / "image_characteristics.csv").open(newline="", encoding="utf-8-sig") as source:
        images = {r["image_name"]: r for r in csv.DictReader(source)}
    # Use the existing study extent plus a 1 km buffer for image footprints.
    registry = read("cushing_hrrr_tank_grid")
    west, east = [op(float(r["tank_longitude"]) for r in registry)+pad for op, pad in ((min, -.012), (max, .012))]
    south, north = [op(float(r["tank_latitude"]) for r in registry)+pad for op, pad in ((min, -.009), (max, .009))]
    local = {name for name, r in images.items() if float(r["nw_lat_image_coord"]) >= south
             and float(r["se_lat_image_coord"]) <= north and float(r["nw_lon_image_coord"]) <= east
             and float(r["se_lon_image_coord"]) >= west}
    inventory, labels, catalogue = [], [], []
    national, regional = Counter(), Counter()
    image_zip = next(r for r in sources if r["file_name"] == "images.zip")
    cushing_tiles = {images[name]["tile_name"] for name in local}
    with ZipFile(STORE / "xmls.zip") as xmls, RangeZip(image_zip["source_url"]) as remote, ZipFile(remote) as jpgs:
        if remote.size != image_zip["source_bytes"] or not len(images) == len(jpgs.namelist()) == len(xmls.namelist()):
            raise ValueError("Different Figshare archive and image inventory")
        ordered = sorted(local)
        entries = [jpgs.getinfo(f"images/{name}.jpg") for name in ordered]
        with ThreadPoolExecutor(max_workers=10) as pool:
            local_files = dict(zip(ordered, pool.map(lambda entry: jpeg(remote, entry), entries), strict=True))
        for name, item in sorted(images.items()):
            member = f"xmls/{name}.xml"
            raw = xmls.read(member)
            objects = ET.fromstring(raw).findall("object") if raw else ()
            counts = Counter(o.findtext("name") for o in objects)
            national.update(counts)
            entry = jpgs.getinfo(f"images/{name}.jpg")
            counts_json = json.dumps(counts, sort_keys=True)
            catalogue.append(dict(image_name=name, tile_name=item["tile_name"], image_member=entry.filename,
                image_member_offset=entry.header_offset, image_compressed_bytes=entry.compress_size,
                image_uncompressed_bytes=entry.file_size, image_zip_crc32=f"{entry.CRC:08x}", xml_member=member,
                has_annotations=bool(objects), counts_by_class_json=counts_json,
                cushing_overlap_tile=item["tile_name"] in cushing_tiles, cushing_area_image=name in local))
            if name not in local:
                continue
            regional.update(counts)
            capture = datetime.strptime(item["tile_name"].rsplit("_", 1)[-1], "%Y%m%d").date().isoformat()
            inventory.append(dict(item, **local_files[name], xml_member=member, capture_date=capture,
                annotation_status="objects" if objects else "empty_xml",
                xml_sha256=hashlib.sha256(raw).hexdigest(), annotated_objects=len(objects),
                counts_by_class_json=counts_json))
            for i, obj in enumerate(objects, 1):
                box = {k: int(obj.findtext(f"bndbox/{k}")) for k in ("xmin", "ymin", "xmax", "ymax")}
                if not (0 <= box["xmin"] < box["xmax"] <= 512 and 0 <= box["ymin"] < box["ymax"] <= 512):
                    raise ValueError(f"Invalid PASCAL VOC box in {member}")
                labels.append(dict(image_name=name, tile_name=item["tile_name"], annotation_id=i, category=obj.findtext("name"),
                    truncated=obj.findtext("truncated"), difficult=obj.findtext("difficult"), **box))
        report = dict(licence=image_zip["licence"], source_image_archive_bytes=remote.size, source_image_archive_etag=remote.etag,
            source_image_archive_last_modified=remote.last_modified, all_images=len(images), all_box_annotations=national,
            cushing_images=len(inventory), cushing_naip_tiles=len({r["tile_name"] for r in inventory}),
            cushing_empty_xml=sum(r["annotation_status"] == "empty_xml" for r in inventory), cushing_box_annotations=regional,
            cushing_extent_wgs84=[west,south,east,north], source_image_file_verification="Per-member ZIP CRC32 plus local SHA-256",
            scope="Cushing images overlap the study location and may overlap its source aerial imagery. Keep them out of training "
            "for a geographically held-out Cushing detector evaluation. Bounding boxes and tank types do not measure roof level, "
            "liquid volume or shadow masks. Empty XML is retained as published, not asserted as tank absence.")
    export(f"{STEM}_cushing_images", inventory)
    export(f"{STEM}_cushing_boxes", labels)
    buffer = pa.BufferOutputStream()
    pq.write_table(pa.Table.from_pylist(catalogue), buffer, compression="zstd")
    catalogue_path = DERIVED / f"{STEM}_image_index.parquet"
    write(catalogue_path, buffer.getvalue().to_pybytes())
    report["image_index"] = str(catalogue_path.relative_to(ROOT))
    report["images_in_cushing_overlap_tiles"] = sum(r["cushing_overlap_tile"] for r in catalogue)
    matches = list(registry_matches(inventory, labels))
    export(f"{STEM}_cushing_registry_matches", matches)
    report["cushing_capture_dates"] = sorted({r["capture_date"] for r in inventory})
    report["cushing_clean_boxes"] = len(matches)
    report["cushing_clean_boxes_with_unique_registry_match"] = sum(r["candidate_count"] == 1 for r in matches)
    report["cushing_distinct_registry_candidates_matched"] = len({r["candidate_id"] for r in matches if r["candidate_id"]})
    report["matched_boxes_by_label_and_documented_type"] = dict(Counter(
        f"{r['category']} / {r['registry_roof_type'] or 'unknown'}" for r in matches if r["candidate_id"]))
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
