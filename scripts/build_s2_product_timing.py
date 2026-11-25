"""Record acquisition, product generation and archive object timestamps separately."""
import csv
import datetime as dt
import io
import json
import xml.etree.ElementTree as ET

from collect_imagery import BANDS, acquisition_key, canonical_items, earliest_alternate_items
from collect_public import RAW, ROOT, write
from collect_s2_l1c import CHANNELS, OUT as L1C, source_metadata_bytes, source_metadata_path


def has_bands(directory, bands):
    return (directory / "item.json").is_file() and all((directory / f"{band}.tif").is_file() for band in bands)


def main():
    earlier = {acquisition_key(item): item for item in earliest_alternate_items()}
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("canonical_id", "captured_at_utc", "canonical_generation_at_utc", "canonical_catalog_created_at_utc",
                     "canonical_generation_lag_hours", "earliest_l2a_id", "earliest_l2a_generation_at_utc",
                     "earliest_l2a_generation_lag_hours", "earliest_l2a_local_directory", "l1c_product_name",
                     "l1c_generation_reported_at_utc", "l1c_generation_before_tile_sensing",
                     "l1c_archive_band_and_granule_metadata_created_by_utc", "l1c_local_directory"))
    count_l2a = count_l1c = 0
    for item in canonical_items():
        identity = item["id"]
        observed = item["properties"]
        sensed = dt.datetime.fromisoformat(observed["datetime"])
        year = observed["datetime"][:4]
        generated = dt.datetime.fromisoformat(observed["s2:generation_time"])
        alt = earlier.get(acquisition_key(item), item)
        alt_generated = dt.datetime.fromisoformat(alt["properties"]["s2:generation_time"])
        root = "imagery_historical" if alt["id"] != identity else "imagery_core"
        alt_directory = RAW / root / "sentinel2" / year / alt["id"]
        alt_ready = has_bands(alt_directory, BANDS)
        count_l2a += int(alt_ready)
        l1c_directory = L1C / "granules" / year / identity
        l1c_columns = (None,) * 5
        if has_bands(l1c_directory, CHANNELS):
            l1c = json.loads((l1c_directory / "item.json").read_bytes())
            meta_path = source_metadata_path(l1c_directory / "MTD_MSIL1C.xml")
            meta = ET.fromstring(source_metadata_bytes(meta_path))
            timestamp = next(node.text for node in meta.iter() if node.tag.rsplit("}", 1)[-1] == "GENERATION_TIME")
            l1c_generated = dt.datetime.fromisoformat(timestamp)
            members = list(l1c["assets"].values()) + l1c["granule_metadata"]
            stored_at = max(dt.datetime.fromisoformat(asset["timeCreated"]) for asset in members)
            if stored_at < sensed:
                raise ValueError(f"L1C archive object timestamps predate acquisition for {identity}")
            l1c_columns = (l1c["product_name"], l1c_generated.isoformat(), int(l1c_generated < sensed),
                           stored_at.isoformat(), str(l1c_directory.relative_to(ROOT)))
            count_l1c += 1
        writer.writerow((identity, sensed.isoformat(), generated.isoformat(), observed.get("created"),
                         round((generated - sensed).total_seconds() / 3600, 6), alt["id"], alt_generated.isoformat(),
                         round((alt_generated - sensed).total_seconds() / 3600, 6),
                         str(alt_directory.relative_to(ROOT)) if alt_ready else None, *l1c_columns))
    write(ROOT / "data" / "derived" / "sentinel2_product_timing.csv", stream.getvalue().encode())
    print(f"{count_l2a} earliest L2A versions and {count_l1c} L1C versions available locally")


if __name__ == "__main__":
    main()
