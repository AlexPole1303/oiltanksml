"""Index original L1C versions and their public GCS object-creation clocks."""
import argparse
import csv
import datetime as dt
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from collect_imagery import candidate_footprint, canonical_items
from collect_public import RAW, ROOT, write
from collect_s2_l1c import CHANNELS, OBJECTS, OUT, earliest_products, parse_product, query_archive

DERIVED = ROOT / "data" / "derived"
CLOUD_MASKS = {"MSK_CLOUDS_B00.gml", "MSK_CLASSI_B00.jp2"}
DETECTOR_MASKS = {"MSK_DETFOO_B04.gml", "MSK_DETFOO_B04.jp2"}


def record(item, versions, *, object_index_store=OUT/"object_indexes"):
    identity, captured = item["id"], item["properties"]["datetime"]
    key, _, _ = parse_product(item["properties"]["s2:product_uri"])
    if key not in versions:
        return (identity, captured, "no_matching_product", *(None,) * 8)
    _, product, prefix = versions[key]
    objects = query_archive(prefix, cache=object_index_store/f"{product}.json")
    bands = [asset for asset in objects if "/IMG_DATA/" in asset["name"]
             and any(asset["name"].endswith(f"_{band}.jp2") for band in CHANNELS.values())]
    by_filename = {asset["name"].rsplit("/", 1)[-1]: asset for asset in objects}
    cloud = [by_filename[mask] for mask in sorted(CLOUD_MASKS & by_filename.keys())]
    detector = [by_filename[mask] for mask in sorted(DETECTOR_MASKS & by_filename.keys())]
    xml = [by_filename[name] for name in ("MTD_TL.xml", "MTD_MSIL1C.xml") if name in by_filename]
    inputs = bands + xml + cloud
    clock, detector_clock, hours = None, None, None
    status = "missing_required_objects"
    if len(bands) == len(CHANNELS) and len(xml) == 2 and cloud:
        clock = max(dt.datetime.fromisoformat(asset["timeCreated"]) for asset in inputs)
        hours = round((clock - dt.datetime.fromisoformat(captured)).total_seconds() / 3600, 6)
        if hours < 0:
            raise ValueError(f"GCS object creation precedes tile sensing: {identity}")
        status = "input_clocks_complete"
        if detector:
            detector_clock = max(clock, *(dt.datetime.fromisoformat(asset["timeCreated"]) for asset in detector))
    return (identity, captured, status, product, OBJECTS + prefix, clock.isoformat() if clock else None,
            detector_clock.isoformat() if detector_clock else None, hours,
            "|".join(sorted(CLOUD_MASKS & by_filename.keys())),
            "|".join(sorted(DETECTOR_MASKS & by_filename.keys())), len(inputs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    _, area = candidate_footprint(DERIVED / "supplemental_tank_candidates.geojson", 120)
    items = canonical_items(RAW / "stac_supplemental", area)
    versions = earliest_products(items, args.workers)
    print(f"Checking GCS objects for {len(items)} acquisitions", flush=True)
    result = io.StringIO(newline="")
    writer = csv.writer(result)
    writer.writerow(("canonical_l2a_id", "captured_at_utc", "status", "l1c_product_name", "gcs_product_url",
                     "all_required_objects_created_by_utc", "with_red_detector_mask_objects_created_by_utc",
                     "gcs_input_lag_hours", "cloud_mask_filenames", "red_detector_mask_filenames", "required_objects"))
    years = defaultdict(lambda: dict(acquisitions=0, clocks_complete=0, within_24h=0, within_72h=0, beyond_7d=0))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(lambda item: record(item, versions), items), 1):
            writer.writerow(row)
            counts = years[row[1][:4]]
            counts["acquisitions"] += 1
            if (hours := row[7]) is not None:
                counts["clocks_complete"] += 1
                counts["within_24h"] += hours <= 24
                counts["within_72h"] += hours <= 72
                counts["beyond_7d"] += hours > 168
            if index % 250 == 0:
                print(f"{index}/{len(items)} L1C clocks checked", flush=True)
    write(DERIVED / "sentinel2_l1c_archive_availability.csv", result.getvalue().encode())
    write(DERIVED / "sentinel2_l1c_archive_availability.json", json.dumps({
        "source": OBJECTS, "clock_field": "timeCreated",
        "required": "Four native bands, tile and product XML, and the product's cloud mask",
        "by_capture_year": years,
    }, indent=2).encode())


if __name__ == "__main__":
    main()
