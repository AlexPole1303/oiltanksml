"""Find Cushing observations in the public Umbra bucket independently of a search index."""
import argparse
import hashlib
import json
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlencode, urlsplit

from shapely.geometry import box, shape

from build_s1_reference_matches import read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import RAW, ROOT, get, write, write_csv
from collect_umbra_cushing import BUCKET, INDEX


STORE = RAW / "umbra_public_archive"
DATABASE = DERIVED / "umbra_public_archive_screen.sqlite"
AREA = box(-96.81, 35.73, -96.54, 36.10)
PREFIXES = ("sar-data/task-data/", "sar-data/tasks/", "open-data/", "stac/")


def inventory(prefix, refresh):
    query = {"list-type": 2, "prefix": prefix}
    path = STORE / "listings" / prefix.rstrip("/").replace("/", "_")
    entries = []
    tokens = set()
    while True:
        request = BUCKET + "?" + urlencode(query)
        payload = get(request, path / f"{len(tokens):03d}.xml", refresh=refresh)
        root = ET.fromstring(payload)
        if root.findtext("{*}Prefix") != prefix:
            raise ValueError(f"Unexpected listing prefix for {request}")
        for item in root.findall("{*}Contents"):
            key = item.findtext("{*}Key")
            is_item = (key.endswith((".stac.v2.json", "_STAC.json")) if prefix != "stac/"
                       else key.endswith(".json") and not key.endswith("/catalog.json"))
            if is_item:
                entries.append((prefix, key, item.findtext("{*}ETag").strip('"'),
                                item.findtext("{*}LastModified"), int(item.findtext("{*}Size"))))
        state = root.findtext("{*}IsTruncated")
        if state == "false":
            return entries, len(tokens) + 1
        token = root.findtext("{*}NextContinuationToken")
        if state != "true" or not token or token in tokens:
            raise ValueError(f"Invalid listing continuation for {request}")
        tokens.add(token)
        query["continuation-token"] = token


def footprint_status(geometry):
    if not geometry:
        return "missing_geometry"
    footprint = shape(geometry)
    if footprint.is_empty or footprint.geom_type not in {"Polygon", "MultiPolygon"}:
        return "invalid_geometry"
    # A footprint spanning the date line can cover the rest of the world in Cartesian degrees.
    if footprint.bounds[2] - footprint.bounds[0] > 180:
        return "antimeridian_footprint"
    if not footprint.is_valid:
        return "invalid_geometry"
    return "intersects" if footprint.intersects(AREA) else "outside"


def inspect(entry):
    prefix, key, etag, modified, size = entry
    payload = get(BUCKET + quote(key, safe="/"))
    if len(payload) != size:
        raise ValueError(f"Listed size changed for {key}")
    checksum = hashlib.sha256(payload).hexdigest()
    item = json.loads(payload)
    properties = item.get("properties", {})
    status = footprint_status(item.get("geometry"))
    local = ""
    if status == "intersects":
        path = STORE / "matches" / key
        write(path, payload)
        local = str(path.relative_to(ROOT))
    return (prefix, key, etag, modified, size, checksum, status, item.get("id"),
            properties.get("umbra:collect_id"), properties.get("start_datetime") or properties.get("datetime"), local)


def capture_assets(item):
    assets = {}
    for name, asset in item.get("assets", {}).items():
        if not asset.get("href"):
            continue
        label = (asset.get("title") or name).lower()
        if label in {"gec", "sicd", "cphd", "sidd"}:
            assets[label] = asset["href"]
    return assets


def bucket_asset_key(href):
    path = urlsplit(href)
    if path.netloc in {"umbra-open-data-catalog", "umbra-open-data-catalog.s3.amazonaws.com",
                       urlsplit(BUCKET).netloc}:
        return unquote(path.path.lstrip("/"))
    if path.netloc in {"s3.us-west-2.amazonaws.com", "s3.amazonaws.com"} and path.path.startswith("/umbra-open-data-catalog/"):
        return unquote(path.path.removeprefix("/umbra-open-data-catalog/"))
    return None


def inspect_assets(row):
    item = json.loads((ROOT / row["local_metadata_file"]).read_bytes())
    assets = capture_assets(item)
    keys = {label: bucket_asset_key(url) for label, url in assets.items()}
    if row["object_key"].endswith(".stac.v2.json"):
        stem = row["object_key"].removesuffix(".stac.v2.json")
        for label, suffix in (("gec", "_GEC.tif"), ("sicd", "_SICD.nitf"), ("cphd", "_CPHD.cphd")):
            keys[label] = keys.get(label) or stem + suffix
    folders = {key.rsplit("/", 1)[0] + "/" for key in keys.values() if key}
    objects = {}
    for folder in sorted(folders):
        url = BUCKET + "?" + urlencode({"list-type": 2, "prefix": folder})
        path = STORE / "asset_listings" / (hashlib.sha256(folder.encode()).hexdigest() + ".xml")
        listing = ET.fromstring(get(url, path))
        if listing.findtext("{*}IsTruncated") != "false" or listing.findtext("{*}Prefix") != folder:
            raise ValueError(f"Incomplete or mismatched asset listing for {folder}")
        objects.update((node.findtext("{*}Key"), node) for node in listing.findall("{*}Contents"))
    properties = item["properties"]
    result = dict(metadata_object_key=row["object_key"], metadata_item_id=row["item_id"],
                  metadata_modified_utc=row["modified_utc"], collect_id=row["collect_id"],
                  capture_start_utc=row["capture_start_utc"], capture_end_utc=properties.get("end_datetime"))
    for label in ("gec", "sicd", "cphd", "sidd"):
        key = keys.get(label)
        node = objects.get(key)
        result.update({f"{label}_key": key, f"{label}_url": BUCKET + quote(key) if node is not None else None,
                       f"{label}_bytes": int(node.findtext("{*}Size")) if node is not None else None,
                       f"{label}_etag": node.findtext("{*}ETag").strip('"') if node is not None else None,
                       f"{label}_modified_utc": node.findtext("{*}LastModified") if node is not None else None})
    return result


def original_cutoffs(assets):
    with closing(read_only(SOURCE)) as db:
        windows = [(w, phase, *(datetime.fromisoformat(t) for t in (lo, hi, cutoff)))
                   for w, phase, lo, hi, cutoff in db.execute("""SELECT week_ending,study_phase,
                       window_starts_after_utc,window_ends_by_utc,inputs_available_before_utc
                       FROM eia_satellite_window WHERE endpoint='current' ORDER BY week_ending""")]
    for row in assets:
        row.update(original_week_ending=None, study_phase=None, original_cutoff_utc=None,
                   metadata_and_sicd_version_before_cutoff=False)
        if not row["capture_start_utc"] or not row["capture_end_utc"]:
            continue
        start, end = (datetime.fromisoformat(row[k]) for k in ("capture_start_utc", "capture_end_utc"))
        if end < start:
            raise ValueError(f"Reversed capture interval for {row['metadata_object_key']}")
        for week, phase, lower, upper, cutoff in windows:
            if lower < start <= end <= upper:
                clocks = (row["metadata_modified_utc"], row["sicd_modified_utc"])
                row.update(original_week_ending=week, study_phase=phase, original_cutoff_utc=cutoff.isoformat(),
                           metadata_and_sicd_version_before_cutoff=bool(all(clocks) and max(
                               datetime.fromisoformat(c) for c in clocks) < cutoff))
                break


def output(db, entries, pages, fresh, community):
    expected = {row[1]: row[2:5] for row in entries}
    db.row_factory = sqlite3.Row
    checked = [r for r in db.execute("SELECT * FROM screen ORDER BY object_key")
               if expected.get(r["object_key"]) == (r["etag"], r["modified_utc"], r["size_bytes"])]
    if len(checked) != len(entries):
        raise ValueError(f"Archive screening incomplete: {len(checked)} of {len(entries)}")
    catalog = read("umbra_cushing_catalog")
    known = {r["collect_id"] for r in catalog}
    rows = []
    for r in checked:
        if r["status"] != "intersects":
            continue
        payload = (ROOT / r["local_metadata_file"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != r["sha256"]:
            raise ValueError(f"Preserved provider metadata has changed for {r['object_key']}")
        item = json.loads(payload)
        assets = capture_assets(item)
        rows.append(dict(r, in_existing_catalog=r["collect_id"] in known,
                         metadata_url=BUCKET + quote(r["object_key"], safe="/"),
                         gec_url=assets.get("gec"), sicd_url=assets.get("sicd"),
                         cphd_url=assets.get("cphd")))
    if rows:
        fields = tuple(rows[0])
        write_csv(DERIVED / "umbra_public_archive_cushing_items.csv", fields,
                  ([row[f] for f in fields] for row in rows))
    with ThreadPoolExecutor(max_workers=12) as pool:
        assets = list(pool.map(inspect_assets, rows))
    original_cutoffs(assets)
    if assets:
        fields = tuple(assets[0])
        write_csv(DERIVED / "umbra_public_archive_cushing_assets.csv", fields,
                  ([row[f] for f in fields] for row in assets))
    matched_known = {r["collect_id"] for r in rows if r["in_existing_catalog"]}
    known_sicd = {r["collect_id"]: r for r in catalog}
    older_sicd = [r for r in assets if r["sicd_modified_utc"] and r["collect_id"] in known_sicd
                  and datetime.fromisoformat(r["sicd_modified_utc"]) < datetime.fromisoformat(
                      known_sicd[r["collect_id"]]["sicd_s3_last_modified_utc"])]
    result = dict(bucket=BUCKET, search_area_wgs84=AREA.bounds,
                  observed_utc=datetime.now(timezone.utc).isoformat(),
                  refreshed_object_listings=fresh, listing_pages=pages,
                  community_index_items=len(community["features"]),
                  items_by_prefix=Counter(r["prefix"] for r in checked),
                  screening_status=Counter(r["status"] for r in checked),
                  matched_item_rows=len(rows),
                  existing_collect_ids_confirmed=len(matched_known),
                  existing_collect_ids_missing=sorted(known - matched_known),
                  additional_named_collect_ids=sorted({r["collect_id"] for r in rows
                      if r["collect_id"] and not r["in_existing_catalog"]}),
                  matching_items_without_collect_id=sum(not r["collect_id"] for r in rows),
                  matching_items_with_listed_sicd=sum(bool(r["sicd_url"]) for r in assets),
                  unique_listed_sicd_objects=len({r["sicd_key"] for r in assets if r["sicd_url"]}),
                  older_sicd_objects_for_known_collects=older_sicd,
                  metadata_and_sicd_version_rows_before_original_cutoff=sum(
                      r["metadata_and_sicd_version_before_cutoff"] for r in assets),
                  interpretation="STAC screening by footprint. A metadata object timestamp applies only to that object. "
                  "Separate asset listing and the original acquisition metadata are required to verify image availability.")
    write(DERIVED / "umbra_public_archive_screen_summary.json", (json.dumps(result, indent=2) + "\n").encode())
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    community = json.loads(get(INDEX, STORE / "community_search.geojson", refresh=args.refresh))
    if len(community["features"]) >= 100 or any(l["rel"] == "next" for l in community["links"]):
        raise ValueError("Community Cushing index requires pagination")
    print(f"Community index returned {len(community['features'])} items.", flush=True)
    with ThreadPoolExecutor(max_workers=len(PREFIXES)) as pool:
        listings = list(pool.map(lambda prefix: inventory(prefix, args.refresh), PREFIXES))
    entries = [entry for found, _ in listings for entry in found]
    if len(entries) != len({entry[1] for entry in entries}):
        raise ValueError("Duplicate listed archive object keys")
    pages = {prefix: n for prefix, (_, n) in zip(PREFIXES, listings, strict=True)}
    with closing(sqlite3.connect(DATABASE)) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS screen (
            prefix TEXT NOT NULL, object_key TEXT PRIMARY KEY, etag TEXT NOT NULL, modified_utc TEXT NOT NULL,
            size_bytes INTEGER NOT NULL, sha256 TEXT NOT NULL, status TEXT NOT NULL, item_id TEXT,
            collect_id TEXT, capture_start_utc TEXT, local_metadata_file TEXT NOT NULL)""")
        cached = dict((key, (etag, modified, size)) for key, etag, modified, size in db.execute(
            "SELECT object_key,etag,modified_utc,size_bytes FROM screen"))
        pending = [entry for entry in entries if cached.get(entry[1]) != entry[2:5]]
        print(f"{len(entries)} listed STAC items, {len(pending)} to read.", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(pending), 300):
                batch = list(pool.map(inspect, pending[start:start + 300]))
                with db:
                    db.executemany("INSERT OR REPLACE INTO screen VALUES (?,?,?,?,?,?,?,?,?,?,?)", batch)
                for row in batch:
                    if row[6] == "intersects":
                        print(f"Cushing metadata: {row[9]} {row[8]} {row[1]}", flush=True)
                if start % 1200 == 0 or start + 300 >= len(pending):
                    print(f"Read {start + len(batch)} / {len(pending)} uncached STAC items.", flush=True)
        output(db, entries, pages, args.refresh, community)


if __name__ == "__main__":
    main()
