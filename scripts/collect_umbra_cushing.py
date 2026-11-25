"""Verify public Umbra Cushing scenes and match footprints to dated tank records."""
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from itertools import combinations
from urllib.parse import urlencode, urlsplit

from pyproj import Transformer
from shapely import transform
from shapely.geometry import Point, shape

from build_s1_reference_matches import REFERENCE_SQL, export, nearest
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import RAW, ROOT, get, write

STORE = RAW / "umbra_cushing"
BUCKET = "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/"
INDEX = "https://api.umbra-py.space/search?" + urlencode(dict(
    bbox="-96.81,35.73,-96.54,36.10", collections="umbra-open-data", limit=100))
TO_UTM = Transformer.from_crs(4326, 32614, always_xy=True)
FOOTPRINT_CONTEXT_M = 60


def verify(discovered):
    product = discovered["assets"]["gec"]["href"]
    parsed = urlsplit(product)
    bucket_prefix = "/umbra-open-data-catalog/"
    if (parsed.scheme != "https" or parsed.netloc != "s3.us-west-2.amazonaws.com"
            or not parsed.path.startswith(bucket_prefix) or not parsed.path.endswith("_GEC.tif")):
        raise ValueError(f"Unrecognized Umbra public path {product}")
    key = parsed.path.removeprefix(bucket_prefix)
    prefix, _, filename = key.rpartition("/")
    stem = filename.removesuffix("_GEC.tif")
    directory = STORE / discovered["id"]
    listing_url = BUCKET + "?" + urlencode({"list-type": 2, "prefix": prefix + "/"})
    listing = ET.fromstring(get(listing_url, directory / "source_listing.xml"))
    if listing.findtext("{*}IsTruncated") != "false":
        raise ValueError(f"Truncated public object list for {product}")
    objects = {e.findtext("{*}Key"): e for e in listing.findall("{*}Contents")}
    metadata_key = f"{prefix}/{stem}.stac.v2.json"
    metadata_url = BUCKET + metadata_key
    metadata_path = directory / "provider_stac.json"
    payload = get(metadata_url, metadata_path)
    if len(payload) != int(objects[metadata_key].findtext("{*}Size")):
        raise ValueError(f"Original Umbra metadata size changed for {product}")
    metadata = json.loads(payload)
    official, indexed = metadata["properties"], discovered["properties"]
    collect_id = official["umbra:collect_id"]
    if (metadata["id"] != discovered["id"] or collect_id != indexed["umbra:collect_id"]
            or official["datetime"] != indexed["datetime"]):
        raise ValueError(f"Umbra index disagrees with original identity for {product}")

    row = dict(product_id=metadata["id"], collect_id=collect_id, product_type="Umbra_SICD",
        product_name=stem, capture_start_utc=official["start_datetime"], capture_end_utc=official["end_datetime"],
        platform=official["platform"], direction=official["sat:orbit_state"], relative_orbit="",
        look_side=official["sar:observation_direction"], view_azimuth_deg=official["view:azimuth"],
        incidence_deg=official["view:incidence_angle"], polarization="|".join(official["sar:polarizations"]),
        gec_range_resolution_m=official["sar:resolution_range"], gec_azimuth_resolution_m=official["sar:resolution_azimuth"],
        metadata_created_utc=official["created"], metadata_url=metadata_url,
        metadata_file=str(metadata_path.relative_to(ROOT)))
    clocks = [objects[metadata_key].findtext("{*}LastModified")]
    for label, suffix in (("gec", "_GEC.tif"), ("sicd", "_SICD.nitf"), ("cphd", "_CPHD.cphd")):
        asset_key = f"{prefix}/{stem}{suffix}"
        asset = objects[asset_key]
        last_modified = asset.findtext("{*}LastModified")
        row.update({f"{label}_url": BUCKET + asset_key, f"{label}_bytes": int(asset.findtext("{*}Size")),
                    f"{label}_s3_last_modified_utc": last_modified, f"{label}_etag": asset.findtext("{*}ETag").strip('"')})
        if label == "sicd":
            clocks.append(last_modified)
    row["observed_metadata_and_sicd_version_timestamp_utc"] = max(clocks)
    return row, transform(shape(metadata["geometry"]), TO_UTM.transform, interleaved=False)


def compact_json(values):
    return json.dumps(sorted(values), separators=(",", ":"))


def closely_timed_pairs(products):
    for first, last in combinations(products, 2):
        start = datetime.fromisoformat(first["capture_start_utc"])
        end = datetime.fromisoformat(last["capture_start_utc"])
        seconds = (end - start).total_seconds()
        if not 0 <= seconds <= 3600:
            continue
        common = set(json.loads(first["context_efr_candidate_ids_json"])) & set(json.loads(last["context_efr_candidate_ids_json"]))
        if common:
            angle = abs((last["view_azimuth_deg"] - first["view_azimuth_deg"] + 180) % 360 - 180)
            yield dict(first_product_id=first["product_id"], last_product_id=last["product_id"],
                first_capture_start_utc=first["capture_start_utc"], last_capture_start_utc=last["capture_start_utc"],
                separation_seconds=seconds, azimuth_separation_deg=angle,
                different_look_side=first["look_side"] != last["look_side"],
                common_efr_contours=len(common), common_efr_candidate_ids_json=compact_json(common))


def main():
    query = json.loads(get(INDEX, STORE / "discovery_search.geojson"))
    if any(link["rel"] == "next" for link in query["links"]) or len(query["features"]) >= 100:
        raise ValueError("Umbra discovery results need pagination")
    with ThreadPoolExecutor(max_workers=6) as pool:
        verified = sorted(pool.map(verify, query["features"]), key=lambda item: item[0]["capture_start_utc"])
    identities = [row["collect_id"] for row, _ in verified]
    if len(identities) != len(set(identities)):
        raise ValueError("Multiple public copies of a single acquisition require explicit selection")
    with closing(read_only(SOURCE)) as db:
        references = db.execute(REFERENCE_SQL).fetchall()
        tanks = {key: (Point(*TO_UTM.transform(lon, lat)), diameter / 2, roof, first_seen)
                 for key, lon, lat, diameter, roof, first_seen in db.execute("""
            SELECT candidate_id,longitude,latitude,footprint_diameter_m,documented_roof_type,first_seen
            FROM tank_engineering_registry
        """)}
    contours = []
    for row, footprint in verified:
        ids, efr = set(), set()
        capture_day = row["capture_start_utc"][:10]
        for tank, (centre, radius, roof, first_seen) in sorted(tanks.items()):
            if not footprint.covers(centre) or not math.isfinite(radius):
                continue
            margin = footprint.boundary.distance(centre) - radius
            context = margin >= FOOTPRINT_CONTEXT_M
            present = bool(first_seen and first_seen <= capture_day)
            contours.append(dict(product_id=row["product_id"], candidate_id=tank, documented_roof_type=roof,
                first_seen=first_seen, already_seen_on_capture_day=present, margin_beyond_rim_m=round(margin, 3),
                entire_contour_with_60m_context=context))
            if context and present:
                ids.add(tank)
                if roof == "EFR":
                    efr.add(tank)
        row.update(context_contours=len(ids), context_efr_contours=len(efr),
                   covered_candidate_ids_json=compact_json(ids), context_efr_candidate_ids_json=compact_json(efr))

    products = [r for r, _ in verified]
    pairs = list(closely_timed_pairs(products))
    matches = list(nearest(references, products))
    for name, rows in (("catalog", products), ("tank_coverage", contours), ("reference_matches", matches),
                       ("close_capture_pairs", pairs)):
        export(f"umbra_cushing_{name}", rows)
    best_by_kind = defaultdict(list)
    for row in matches:
        best_by_kind[row["reference_kind"]].append(row)
    report = dict(source_index=INDEX, verified_original_scenes=len(products),
        retrieved_utc=datetime.now(timezone.utc).isoformat(),
        capture_first_utc=products[0]["capture_start_utc"], capture_last_utc=products[-1]["capture_end_utc"],
        range_resolution_m=sorted({round(r["gec_range_resolution_m"], 3) for r in products}),
        azimuth_resolution_m=sorted({round(r["gec_azimuth_resolution_m"], 3) for r in products}),
        date_counts=dict(sorted(Counter(r["capture_start_utc"][:10] for r in products).items())),
        contour_centre_matches=len(contours), efr_contours_with_context=len({r["candidate_id"] for r in contours
            if r["documented_roof_type"] == "EFR" and r["entire_contour_with_60m_context"] and r["already_seen_on_capture_day"]}),
        reference_matches=len(matches), reference_matches_within_24h=sum(r["maximum_separation_hours"] <= 24 for r in matches),
        closest_reference_by_kind={kind: min(rows, key=lambda r: r["maximum_separation_hours"])
                                   for kind, rows in best_by_kind.items()},
        close_capture_pairs=pairs,
        interpretation="Search uses a public discovery index. The original Umbra item and AWS object listing are retained for every scene. "
        "GEC advertised resolution is distinct from SICD pixel spacing and vertical accuracy. Object modification dates apply to these copies. "
        "They do not establish first availability of another version. Footprint and date matches are engineering inputs.")
    write(DERIVED / "umbra_cushing_catalog_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(f"Verified {len(products)} original Umbra scenes, {len(pairs)} close capture pairs, {len(matches)} reference matches.", flush=True)


if __name__ == "__main__":
    main()
