"""Stage GEDI waveform and elevation sources for independent Cushing roof controls."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlencode

from PIL import Image
from shapely.geometry import Point

from build_cushing_icesat2_controls import identities, original_access
from build_cushing_nasa_satellite_inventory import BBOX, CMR, footprint
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write


STEM = "cushing_gedi"
STORE = RAW / STEM
COLLECTIONS = {"L1B": "C3974616058-LPCLOUD", "L2A": "C3974616071-LPCLOUD"}
NAME = re.compile(r"GEDI(?:01_B|02_A)_(\d{13}_O\d{5}_\d{2}_T\d{5})_\d{2}_\d{3}_\d{2}_V003")
HOURS = 24


def asset(meta, filename, link_type):
    record, = (r for r in meta["DataGranule"]["ArchiveAndDistributionInformation"] if r["Name"] == filename)
    urls = {r["URL"] for r in meta["RelatedUrls"] if r["Type"] == link_type and r["URL"].startswith("https://")
            and r["URL"].endswith("/" + filename)}
    url, = urls
    digest = record["Checksum"]
    if (digest["Algorithm"] != "SHA-512" or not re.fullmatch(r"[a-f0-9]{128}", digest["Value"])
            or not int(record["SizeInBytes"]) > 0):
        raise ValueError(f"Invalid GEDI source asset {filename}")
    return dict(url=url, size_bytes=int(record["SizeInBytes"]), sha512=digest["Value"])


def inventory(level, collection):
    url = CMR + "?" + urlencode(dict(collection_concept_id=collection, bounding_box=BBOX, page_size=500))
    path = STORE / f"{level.lower()}_v003_catalog.json"
    payload = get(url, path, refresh=True)
    catalog = json.loads(payload)
    if catalog["hits"] != len(catalog["items"]):
        raise ValueError(f"Incomplete GEDI catalogue for {level}")
    catalog_hash = hashlib.sha256(payload).hexdigest()
    rows = {}
    for item in catalog["items"]:
        meta, u = item["meta"], item["umm"]
        name = u["GranuleUR"]
        match = NAME.fullmatch(name)
        if not match or meta["native-id"] != name or meta["collection-concept-id"] != collection:
            raise ValueError(f"Inconsistent GEDI source identity {name}")
        key = match[1]
        clock = u["TemporalExtent"]["RangeDateTime"]
        start, stop = (datetime.fromisoformat(clock[k]) for k in ("BeginningDateTime", "EndingDateTime"))
        created = datetime.fromisoformat(u["DataGranule"]["ProductionDateTime"])
        revision = datetime.fromisoformat(meta["revision-date"])
        if key in rows or not start < stop <= created <= revision:
            raise ValueError(f"Inconsistent GEDI capture identity or clock {name}")
        browse_name, = [r["Name"] for r in u["DataGranule"]["ArchiveAndDistributionInformation"] if r["Name"].endswith(".png")]
        h5 = asset(u, name + ".h5", "GET DATA")
        png = asset(u, browse_name, "GET RELATED VISUALIZATION")
        rows[key] = dict(level=level, collection=collection, concept_id=meta["concept-id"], granule=name,
            capture_start_utc=start.isoformat(), capture_end_utc=stop.isoformat(), product_created_utc=created.isoformat(),
            cmr_revision_utc=revision.isoformat(), hdf5_url=h5["url"], hdf5_size_bytes=h5["size_bytes"], hdf5_sha512=h5["sha512"],
            browse_url=png["url"], browse_size_bytes=png["size_bytes"], browse_sha512=png["sha512"],
            catalog_file=str(path.relative_to(ROOT)), catalog_sha256=catalog_hash, footprint=footprint(u))
    return rows


def roof_centres(rows, tanks):
    centres = {k: Point(r["east_m"], r["north_m"]) for k, r in tanks.items() if r["radius_m"] >= 12.5}
    return {key: {k for k, p in centres.items() if r["footprint"].covers(p)
                 and tanks[k]["first_seen"] <= r["capture_start_utc"][:10]} for key, r in rows.items()}


def radar_scenes():
    rows = [("Umbra_SICD", r) for r in read("umbra_cushing_catalog")]
    rows.extend(("Sentinel1_SLC", r) for r in read("s1_cushing_catalog") if r["product_type"] == "IW_SLC__1S")
    for kind, row in rows:
        yield kind, row, set(json.loads(row["covered_candidate_ids_json"])), *(datetime.fromisoformat(row[k]) for k in
                                                                              ("capture_start_utc", "capture_end_utc"))


def opportunities(rows, covered):
    radars = list(radar_scenes())
    for key, r in rows.items():
        begin, end = (datetime.fromisoformat(r[k]) for k in ("capture_start_utc", "capture_end_utc"))
        for kind, s, radar_roofs, start, stop in radars:
            hours = max(abs((start - end).total_seconds()), abs((stop - begin).total_seconds())) / 3600
            if hours > HOURS or not (common := radar_roofs & covered[key]):
                continue
            yield dict(gedi_acquisition_key=key, gedi_l2a_concept_id=r["concept_id"], gedi_l2a_granule=r["granule"],
                gedi_catalog_start_utc=r["capture_start_utc"], gedi_catalog_end_utc=r["capture_end_utc"], radar_kind=kind,
                radar_product_id=s["product_id"], radar_capture_start_utc=s["capture_start_utc"],
                radar_capture_end_utc=s["capture_end_utc"], maximum_catalog_separation_hours=hours,
                common_roof_centres_in_catalog_footprints=len(common), common_candidate_ids_json=json.dumps(sorted(common)))


def browse(r):
    path = STORE / "browse" / r["browse_url"].rsplit("/", 1)[-1]
    data = get(r["browse_url"], path)
    if len(data) != r["browse_size_bytes"] or hashlib.sha512(data).hexdigest() != r["browse_sha512"]:
        raise ValueError(f"GEDI browse image differs from catalogue for {r['granule']}")
    with Image.open(path) as img:
        if img.format != "PNG":
            raise ValueError(f"Unexpected browse format for {r['granule']}")
        width, height = img.size
        img.verify()
    return dict(level=r["level"], concept_id=r["concept_id"], granule=r["granule"], path=str(path.relative_to(ROOT)),
                source_url=r["browse_url"], size_bytes=len(data), sha512=r["browse_sha512"], width=width, height=height)


def main():
    with ThreadPoolExecutor(max_workers=2) as pool:
        l1b, l2a = pool.map(lambda pair: inventory(*pair), COLLECTIONS.items())
    if l1b.keys() != l2a.keys() or any((r["capture_start_utc"], r["capture_end_utc"]) != (
            l1b[key]["capture_start_utc"], l1b[key]["capture_end_utc"]) for key, r in l2a.items()):
        raise ValueError("GEDI elevation and waveform catalogues disagree on acquisitions")
    tanks, _, _ = identities()
    covered = roof_centres(l2a, tanks)
    pairs = sorted(opportunities(l2a, covered), key=lambda r: (r["maximum_catalog_separation_hours"], r["gedi_l2a_granule"]))
    by_granule = defaultdict(list)
    for pair in pairs:
        by_granule[pair["gedi_acquisition_key"]].append(pair)
    archive = []
    for key, r in l2a.items():
        archive.append(dict(gedi_acquisition_key=key, catalog_roof_centres=len(covered[key]),
            candidate_ids_json=json.dumps(sorted(covered[key])), nearby_radar_scenes=len(by_granule[key]),
            **{f"{level}_{field}": v for level, item in (("l2a", r), ("l1b", l1b[key]))
               for field, v in item.items() if field not in ("level", "footprint")}))
    ordered_keys = list(dict.fromkeys(r["gedi_acquisition_key"] for r in sorted(pairs, key=lambda r: (
        r["radar_kind"] != "Umbra_SICD", r["maximum_catalog_separation_hours"], r["gedi_l2a_granule"]))))
    queue = [dict(priority=rank, **r) for rank, key in enumerate(ordered_keys, 1)
             for r in archive if r["gedi_acquisition_key"] == key]
    candidates = [sample[key] for key in ordered_keys for sample in (l2a, l1b)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        browse_files = list(pool.map(browse, candidates))
        access = list(pool.map(lambda r: dict(level=r["level"], **original_access(dict(r, cmr_id=r["concept_id"]))), candidates))
    for label, sample in (("catalog", archive), ("radar_opportunities", pairs), ("download_queue", queue),
                          ("browse", browse_files), ("hdf5_access", access)):
        export(f"{STEM}_{label}", sample)
    report = dict(observed_utc=datetime.now(timezone.utc).isoformat(), paired_acquisitions=len(archive),
        catalog_acquisitions_with_any_dated_efr_centre=sum(bool(keys) for keys in covered.values()),
        latest_acquisition_utc=max(r["capture_start_utc"] for r in l2a.values()),
        nearby_radar_pairs_by_kind=Counter(r["radar_kind"] for r in pairs), prioritized_acquisitions=len(queue),
        verified_browse_files=len(browse_files), hdf5_anonymous_http=Counter(str(r["http_status"]) for r in access),
        hdf5_signature_responses=sum(r["hdf5_signature"] for r in access),
        priority_full_l2a_bytes=sum(r["l2a_hdf5_size_bytes"] for r in queue),
        priority_full_l1b_bytes=sum(r["l1b_hdf5_size_bytes"] for r in queue),
        scope="GEDI sub-orbit polygons screen acquisitions only. A covered roof centre does not establish a shot on the roof. "
        "Download spatial subsets of Level 2A first, then Level 1B waveform data for retained shots. No HDF5 roof elevations were retrieved.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
