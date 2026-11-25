"""Combine the observed S3 timestamps for the files used in each L2A version."""
import csv
import datetime as dt
import io
import json
from collections import defaultdict

from collect_imagery import BANDS, acquisition_key, candidate_footprint, canonical_items, earliest_alternate_items
from collect_public import RAW, ROOT, write

DERIVED = ROOT / "data" / "derived"
INPUTS = BANDS + ("granule_metadata",)


def timestamp(value):
    return dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc) if value else None


def available_from(clocks, keys):
    times = [clocks.get(key) for key in keys]
    return max(times) if all(times) else None


def clock_rows():
    assets = defaultdict(dict)
    with (DERIVED / "sentinel2_archive_asset_times.csv").open() as source:
        for row in csv.DictReader(source):
            files = assets[row["item_id"]]
            key = row["asset_key"]
            if key in files:
                raise ValueError(f"Duplicate archive asset key for {row['item_id']}: {key}")
            files[key] = timestamp(row["last_modified_utc"])

    _, area = candidate_footprint(DERIVED / "supplemental_tank_candidates.geojson", 120)
    catalog = RAW / "stac_supplemental"
    canonical = canonical_items(catalog, area)
    canonical_ids = {acquisition_key(item): item["id"] for item in canonical}
    for item in canonical + earliest_alternate_items(catalog, area):
        identity = item["id"]
        properties = item["properties"]
        captured = timestamp(properties["datetime"])
        catalog_created = timestamp(properties.get("created"))
        files = assets[identity]
        raster_at = available_from(files, BANDS)
        geometry_at = available_from(files, INPUTS)
        product_at = available_from(files, INPUTS + ("product_metadata",))
        catalog_route_at = max(geometry_at, catalog_created) if geometry_at and catalog_created else None
        lag_hours = round((geometry_at - captured).total_seconds() / 3600, 6) if geometry_at else None
        canonical_id = canonical_ids[acquisition_key(item)]
        yield (identity, canonical_id, item["collection"], captured.isoformat(),
               catalog_created.isoformat() if catalog_created else None,
               raster_at.isoformat() if raster_at else None,
               geometry_at.isoformat() if geometry_at else None,
               product_at.isoformat() if product_at else None,
               catalog_route_at.isoformat() if catalog_route_at else None,
               lag_hours, sum(files.get(key) is not None for key in INPUTS))


def main():
    fields = ("item_id", "canonical_item_id", "collection", "captured_at_utc", "catalog_created_at_utc",
              "raster_objects_last_modified_by_utc", "raster_and_granule_objects_last_modified_by_utc",
              "raster_and_both_xml_objects_last_modified_by_utc", "catalog_route_conservative_from_utc",
              "raster_and_granule_lag_hours", "required_object_clocks")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(fields)
    summary = defaultdict(lambda: dict(versions=0, complete=0, within_24h=0, within_72h=0, beyond_7d=0))
    for row in clock_rows():
        writer.writerow(row)
        item_id, canonical_id, _, captured, *_, lag, _ = row
        totals = summary[(captured[:4], "canonical" if item_id == canonical_id else "earliest_alternate")]
        totals["versions"] += 1
        if lag is not None:
            totals["complete"] += 1
            totals["within_24h"] += lag <= 24
            totals["within_72h"] += lag <= 72
            totals["beyond_7d"] += lag > 168
    write(DERIVED / "sentinel2_archive_availability.csv", output.getvalue().encode())
    write(DERIVED / "sentinel2_archive_availability.json", json.dumps({
        "source": "Public AWS S3 ListObjectsV2 and Element 84 Earth Search STAC",
        "clock_meaning": "Object clocks describe the current archive version and can be later than first publication. The catalog route additionally waits for the recorded STAC creation time.",
        "required_files": list(INPUTS),
        "by_year_and_version": [{"year": year, "version": version, **counts}
                                for (year, version), counts in sorted(summary.items())],
    }, indent=2).encode())


if __name__ == "__main__":
    main()
