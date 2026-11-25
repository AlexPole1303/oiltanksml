"""Archive Sentinel-1 ETAD timing corrections and test historical eligibility."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import PurePosixPath
from statistics import median
from urllib.error import HTTPError
from urllib.parse import urlencode
from uuid import UUID

from pyproj import Transformer
from shapely import make_valid, transform
from shapely.geometry import Point, shape

from audit_cushing_slc_archive import CATALOG, DOWNLOAD, members, probe
from build_cushing_icesat2_controls import identities
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write

STEM = "s1_etad_cushing"
STORE = RAW / STEM
PROJECT = Transformer.from_crs(4326, 32614, always_xy=True).transform
PRODUCT = re.compile(
    r"(S1[A-D])_IW_(?:ETA__AX|SLC__1S)[A-Z]{2}_(\d{8}T\d{6})_(\d{8}T\d{6})_"
    r"(\d{6})_([0-9A-F]{6})_[0-9A-F]{4}\.SAFE"
)
SIDES = ("previous", "current")
MAX_BOUNDARY_DIFFERENCE_SECONDS = 0.25


def acquisition(item):
    name = item["Name"]
    match = PRODUCT.fullmatch(name)
    if not match:
        raise ValueError(f"Unrecognized product name {name}")
    mission, _, _, orbit, take = match.groups()
    attributes = {row["Name"]: row["Value"] for row in item["Attributes"]}
    if (int(orbit) != attributes["orbitNumber"] or int(take, 16) != attributes.get("datatakeID", int(take, 16))
            or mission[-1] != attributes["platformSerialIdentifier"]):
        raise ValueError(f"Product name and attributes disagree for {name}")
    key = (mission, int(orbit), int(take, 16), attributes["sliceNumber"], attributes["relativeOrbitNumber"],
           attributes["orbitDirection"], attributes["polarisationChannels"])
    start, stop = (datetime.fromisoformat(item["ContentDate"][field]) for field in ("Start", "End"))
    published = datetime.fromisoformat(item["PublicationDate"])
    modified = datetime.fromisoformat(item["ModificationDate"])
    if not start < stop <= published <= modified or not isinstance(item["Online"], bool):
        raise ValueError(f"Inconsistent capture and archive clocks for {name}")
    return key, start, stop, published, modified, attributes


def etad_catalog(box, contexts):
    w, s, e, n = box
    geometry = f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"
    query = f"Collection/Name eq 'SENTINEL-1' and contains(Name,'_IW_ETA__') and OData.CSC.Intersects(area=geography'SRID=4326;{geometry}')"
    url = f"{CATALOG}/Products?" + urlencode({"$filter": query, "$top": 500, "$count": "true",
        "$expand": "Attributes", "$orderby": "ContentDate/Start asc"})
    by_key = defaultdict(list)
    expected_count = None
    page_urls = set()
    while url:
        if not url.startswith(f"{CATALOG}/Products?") or url in page_urls:
            raise ValueError("Unexpected ETAD catalog pagination")
        page_urls.add(url)
        path = STORE / f"catalog_{len(page_urls):02d}.json"
        payload = get(url, path, refresh=True)
        digest = hashlib.sha256(payload).hexdigest()
        page = json.loads(payload)
        if expected_count is None:
            expected_count = page["@odata.count"]
        for item in page["value"]:
            key, start, stop, published, modified, attrs = acquisition(item)
            item_id = str(UUID(item["Id"]))
            name = item["Name"]
            if attrs["productType"] != "IW_ETA__AX" or PurePosixPath(item["S3Path"]).name != name:
                raise ValueError(f"Unexpected ETAD catalog path for {name}")
            footprint = transform(make_valid(shape(item["GeoFootprint"])), PROJECT, interleaved=False)
            covered = {tank for tank, context in contexts.items() if footprint.covers(context)}
            checksums = {r["Algorithm"]: r["Value"] for r in item["Checksum"]}
            row = dict(etad_id=item_id, etad_name=name, mission=key[0], absolute_orbit=key[1], datatake_id=key[2],
                slice_number=key[3], relative_orbit=key[4], direction=key[5], polarisation=key[6],
                capture_start_utc=start, capture_end_utc=stop, published_utc=published, modified_utc=modified,
                capture_to_publication_days=(published-stop).total_seconds()/86400,
                processor_version=attrs["processorVersion"], online=item["Online"], s3_path=item["S3Path"],
                package_bytes=item["ContentLength"], package_md5=checksums.get("MD5", ""),
                full_context_roofs=len(covered), full_context_roof_ids_json=json.dumps(sorted(covered)),
                catalog_url=url, catalog_source_file=str(path.relative_to(ROOT)), catalog_sha256=digest)
            by_key[key].append(row)
        url = page.get("@odata.nextLink")
    rows = [row for versions in by_key.values() for row in versions]
    if len(rows) != expected_count or len({r["etad_id"] for r in rows}) != expected_count:
        raise ValueError("Incomplete ETAD archive results")
    return by_key, rows


def match_parents(by_key):
    requests = defaultdict(list)
    for row in read("cushing_slc_archive_bursts"):
        requests[row["product_id"]].append(row)
    catalog = {row["product_id"]: row for row in read("s1_cushing_catalog")}
    snapshots = {}
    matches = []
    for parent, bursts in sorted(requests.items()):
        record = catalog[parent]
        path = record["source_file"]
        if path not in snapshots:
            snapshots[path] = {r["Id"]: r for r in json.loads((ROOT / path).read_bytes())["value"]}
        item = snapshots[path][parent]
        key, start, stop, _, _, _ = acquisition(item)
        if {r["product_name"] for r in bursts} != {item["Name"]}:
            raise ValueError(f"SLC archive plan and source identity disagree for {parent}")
        scopes = "|".join(sorted({r["scope"] for r in bursts}))
        candidates = by_key[key]
        if not candidates:
            candidates = [None]
        for etad in candidates:
            ds = (etad["capture_start_utc"]-start).total_seconds() if etad else None
            de = (etad["capture_end_utc"]-stop).total_seconds() if etad else None
            status = "no_same_slice_catalog_entry"
            if etad:
                status = ("matched_catalog_slice" if max(abs(ds), abs(de)) <= MAX_BOUNDARY_DIFFERENCE_SECONDS
                          else "same_slice_time_disagreement")
            matches.append(dict(slc_id=parent, slc_name=item["Name"], scopes=scopes,
                slc_catalog_file=path, slc_capture_start_utc=start, slc_capture_end_utc=stop,
                slc_bursts=len(bursts), match_status=status, etad_id=etad["etad_id"] if etad else "",
                etad_name=etad["etad_name"] if etad else "", etad_published_utc=etad["published_utc"] if etad else None,
                etad_start_minus_slc_seconds=ds, etad_end_minus_slc_seconds=de,
                source_slc_identity_verified_in_etad_annotation=False))
    return matches


def safe_members(row):
    key, name = row["etad_id"], row["etad_name"]
    folder = STORE / key
    base = f"{DOWNLOAD}({key})/Nodes"
    try:
        root = members(base, folder / "root.json")[name]
        children = members(root["Nodes"]["uri"], folder / "body.json", root)
        contents = []
        for name_in_safe in ("annotation", "measurement", "preview", "support"):
            entry = children[name_in_safe]
            path = folder / f"{name_in_safe}.json"
            contents.extend((f"{name_in_safe}/{file}", asset, path)
                            for file, asset in members(entry["Nodes"]["uri"], path, entry).items())
        contents.append(("manifest.safe", children["manifest.safe"], folder / "body.json"))
    except HTTPError as error:
        return key, error.code, []
    files = []
    for relative, entry, path in contents:
        if entry["ChildrenNumber"] != 0 or entry["ContentLength"] <= 0:
            raise ValueError(f"Invalid ETAD SAFE member {name}/{relative}")
        files.append(dict(etad_id=key, etad_name=name, safe_relative_path=relative,
            bytes=entry["ContentLength"], s3_path=f"{row['s3_path']}/{relative}",
            download_url=entry["Nodes"]["uri"].removesuffix("/Nodes") + "/$value",
            nodes_source_file=str(path.relative_to(ROOT))))
    expected_nc = f"measurement/{name.rsplit('_', 1)[0]}.nc"
    if sum(r["safe_relative_path"] == expected_nc for r in files) != 1:
        raise ValueError(f"Missing original correction grid for {name}")
    return key, 200, files


def weekly_eligibility(matches, catalog):
    by_slc = defaultdict(list)
    for r in matches:
        if r["match_status"] == "matched_catalog_slice":
            row = catalog[r["etad_id"]]
            by_slc[r["slc_id"]].append((row, set(json.loads(row["full_context_roof_ids_json"]))))
    for rows in by_slc.values():
        rows.sort(key=lambda row: (row[0]["published_utc"], row[0]["etad_id"]))
    pairs = []
    by_week = defaultdict(list)
    for r in read("s1_slc_orbit_geometry_pairs"):
        tank = r["candidate_id"]
        cutoff = datetime.fromisoformat(r["forecast_cutoff_utc"])
        pair = {k: r[k] for k in ("week_ending", "release_date", "candidate_id", "terminal_id", "swath",
                                 "polarization", "forecast_cutoff_utc", "previous_product_id", "current_product_id")}
        decisions = []
        for side in SIDES:
            zero_doppler = datetime.fromisoformat(r[f"{side}_rim_zero_doppler_at_utc"])
            candidates = [row for row, covered in by_slc[r[f"{side}_product_id"]] if tank in covered
                          and row["capture_start_utc"] <= zero_doppler <= row["capture_end_utc"]]
            selected = candidates[0] if candidates else {}
            published = selected.get("published_utc")
            ready = published is not None and published < cutoff
            pair.update({f"{side}_etad_id": selected.get("etad_id", ""), f"{side}_etad_versions": len(candidates),
                         f"{side}_etad_published_utc": published, f"{side}_published_before_cutoff": ready,
                         f"{side}_publication_minus_cutoff_hours":
                             (published-cutoff).total_seconds()/3600 if published else None})
            decisions.append(ready)
        pair["both_endpoints_have_etad"] = all(pair[f"{side}_etad_id"] for side in SIDES)
        pair["both_endpoints_published_before_cutoff"] = all(decisions)
        pairs.append(pair)
        by_week[r["week_ending"]].append(pair)

    weeks = []
    for r in read("s1_holdout_gap_burst_weeks"):
        candidates = by_week[r["week_ending"]]
        roof_count = len({p["candidate_id"] for p in candidates})
        if roof_count != int(r["roofs_with_full_context"]):
            raise ValueError(f"ETAD and native SLC coverage plans differ on {r['week_ending']}")
        weeks.append(dict(week_ending=r["week_ending"], release_date=r["release_date"],
            planned_slc_tank_weeks=roof_count,
            tank_weeks_with_both_etad_catalog_matches=len({p["candidate_id"] for p in candidates
                                                         if p["both_endpoints_have_etad"]}),
            tank_weeks_with_both_etad_published_before_cutoff=len({p["candidate_id"] for p in candidates
                                                                  if p["both_endpoints_published_before_cutoff"]})))
    return pairs, weeks


def main():
    tanks, _, box = identities()
    contexts = {tank: Point(r["east_m"], r["north_m"]).buffer(r["radius_m"]+120) for tank, r in tanks.items()}
    by_key, catalog = etad_catalog(box, contexts)
    by_id = {r["etad_id"]: r for r in catalog}
    matches = match_parents(by_key)
    matched_ids = sorted({r["etad_id"] for r in matches if r["match_status"] == "matched_catalog_slice"})
    pairs, weeks = weekly_eligibility(matches, by_id)
    for label, rows in (("catalog", catalog), ("slc_matches", matches), ("pairs", pairs), ("weeks", weeks)):
        export(f"{STEM}_{label}", rows)
    print(f"ETAD catalog {len(catalog)}, SLC slices matched {len(matched_ids)}, auditing SAFE files", flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        listings = list(pool.map(safe_members, (by_id[key] for key in matched_ids)))
    files = [row for _, _, group in listings for row in group]
    jobs = [("etad_package", key, f"{DOWNLOAD}({key})/$value") for key in matched_ids]
    jobs.extend(("etad_grid" if r["safe_relative_path"].endswith(".nc") else "etad_manifest",
                 r["etad_id"], r["download_url"]) for r in files
                if r["safe_relative_path"].endswith(".nc") or r["safe_relative_path"] == "manifest.safe")
    with ThreadPoolExecutor(max_workers=6) as pool:
        access = list(pool.map(probe, jobs))
    export(f"{STEM}_safe_members", files)
    export(f"{STEM}_access", access)
    latency = [by_id[key]["capture_to_publication_days"] for key in matched_ids]
    matches_by_scope = defaultdict(set)
    for row in matches:
        if row["match_status"] == "matched_catalog_slice":
            for scope in row["scopes"].split("|"):
                matches_by_scope[scope].add(row["slc_id"])
    summary = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), geographic_query_bbox=box,
        roof_context_buffer_m=120, roof_contexts=len(tanks), etad_catalog_versions=len(catalog),
        planned_slc_parent_scenes=len({r["slc_id"] for r in matches}),
        slc_parents_with_etad_match_by_scope={k: len(v) for k, v in matches_by_scope.items()},
        matched_etad_versions=len(matched_ids),
        max_allowed_catalog_capture_boundary_difference_seconds=MAX_BOUNDARY_DIFFERENCE_SECONDS,
        max_observed_capture_boundary_difference_seconds=max(abs(r[f"etad_{side}_minus_slc_seconds"])
            for r in matches if r["match_status"] == "matched_catalog_slice" for side in ("start", "end")),
        capture_to_etad_publication_days=dict(minimum=min(latency), median=median(latency), maximum=max(latency)),
        safe_listing_http_status=Counter(str(code) for _, code, _ in listings),
        safe_members=len(files), listed_correction_grids=sum(r["safe_relative_path"].endswith(".nc") for r in files),
        anonymous_http_status_by_role={role: Counter(str(r["status"]) for r in access if r["route"] == role)
            for role in sorted({r["route"] for r in access})},
        planned_slc_weeks=sum(r["planned_slc_tank_weeks"] > 0 for r in weeks),
        planned_slc_tank_weeks=sum(r["planned_slc_tank_weeks"] for r in weeks),
        weeks_with_both_etad_catalog_matches=sum(r["tank_weeks_with_both_etad_catalog_matches"] > 0 for r in weeks),
        tank_weeks_with_both_etad_catalog_matches=sum(r["tank_weeks_with_both_etad_catalog_matches"] for r in weeks),
        weeks_with_both_etad_published_before_cutoff=sum(r["tank_weeks_with_both_etad_published_before_cutoff"] > 0 for r in weeks),
        tank_weeks_with_both_etad_published_before_cutoff=sum(r["tank_weeks_with_both_etad_published_before_cutoff"] for r in weeks),
        interpretation="Catalog identity uses spacecraft, orbit, data take, slice, polarization and both sensing "
        "boundaries. Product version inside ETAD annotation requires the archive file. Historical eligibility uses "
        "strictly earlier CDSE PublicationDate, 120 m of roof context and each endpoint's zero-Doppler time. "
        "This audit inventories correction grids. It does not download or apply correction arrays.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
