"""Verify Copernicus B8 file listings and compare the mirror clock with EIA cutoffs."""
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from urllib.parse import quote, urlsplit
from uuid import UUID

from audit_landsat_c2_alternative_access import copernicus, probe
from build_landsat_c2_holdout_gap_inventory import SIDES, parse
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_landsat import LANDSAT
from collect_imagery import sign
from collect_public import ROOT, get, write

STEM = "cushing_b8_archive"
STORE = LANDSAT / "collection2_b8_archive"
CDSE_DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"


def azure_exact_access(requests):
    def check(r):
        original = urlsplit(r["b8_download_url"])
        if original.hostname != "landsatlook.usgs.gov" or not original.path.startswith("/data/collection02/level-1/"):
            raise ValueError(f"Unexpected USGS B8 URL for {r['product_id']}")
        url = "https://landsateuwest.blob.core.windows.net/landsat-c2/" + original.path.removeprefix("/data/")
        result = probe(sign(url))
        return dict(product_id=r["product_id"], platform=r["platform"], url=url,
                    http_status=result["http_status"], tiff_signature=result["tiff_signature"])

    with ThreadPoolExecutor(max_workers=5) as pool:
        return list(pool.map(check, requests))


def listing(url, path):
    data = json.loads(get(url,path,refresh=True))
    if data.get("@odata.nextLink") or not isinstance(data.get("result"),list):
        raise ValueError(f"Incomplete Copernicus listing at {path.name}")
    return data["result"]


def scene_file(request, product, *, store=STORE):
    identity = request["product_id"]
    provider_id = str(UUID(product["Id"]))
    basename = f"{identity}_B8.TIF"
    capture = parse(request["captured_at_utc"])
    published,modified = (parse(product[k]) for k in ("PublicationDate","ModificationDate"))
    if (product["Name"] != identity or PurePosixPath(product["S3Path"]).name != identity
        or abs((parse(product["ContentDate"]["Start"])-capture).total_seconds()) >= 1
        or not capture <= published <= modified or not isinstance(product["Online"],bool)):
        raise ValueError(f"Inconsistent Copernicus source identity or dates for {identity}")
    base = f"{CDSE_DOWNLOAD}({provider_id})"
    folder = store/identity
    root, = listing(f"{base}/Nodes",folder/"root.json")
    files = f"{base}/Nodes({quote(identity)})/Nodes"
    if root["Name"] != identity or root["Nodes"]["uri"] != files:
        raise ValueError(f"Unexpected product folder for {identity}")
    contents = listing(files,folder/"files.json")
    if len(contents)!=root["ChildrenNumber"] or len({r["Name"] for r in contents})!=len(contents):
        raise ValueError(f"Unexpected product content count for {identity}")
    band, = [r for r in contents if r["Name"]==basename]
    size = band["ContentLength"]
    if band["ChildrenNumber"]!=0 or not isinstance(size,int) or size<=0 or basename!=request["original_b8_filename"]:
        raise ValueError(f"Invalid B8 file entry for {identity}")
    sums = {c["Algorithm"]:c for c in product["Checksum"]}
    bundle_md5 = sums["MD5"]["Value"]
    if not re.fullmatch(r"[a-fA-F0-9]{32}",bundle_md5):
        raise ValueError(f"Invalid bundle checksum for {identity}")
    # The bundle checksum is not a checksum of its B8 member.
    checksum_times = [parse(c["ChecksumDate"]) for c in sums.values()]
    return dict(product_id=identity, platform=request["platform"],purpose=request["purpose"],
        captured_at_utc=capture.isoformat(),original_b8_filename=basename,cdse_product_id=provider_id,
        cdse_bundle_bytes=product["ContentLength"],cdse_bundle_md5=bundle_md5,
        cdse_bundle_checksums_latest_utc=max(checksum_times).isoformat(),cdse_catalog_publication_at_utc=published.isoformat(),
        cdse_catalog_modification_at_utc=modified.isoformat(),cdse_online=product["Online"],
        cdse_b8_bytes=size,cdse_b8_download_url=f"{base}/Nodes({quote(identity)})/Nodes({quote(basename)})/$value",
        cdse_b8_s3_path=f"{product['S3Path']}/{basename}",cdse_nodes_snapshot=str((folder/"files.json").relative_to(ROOT)),
        usgs_expected_b8_sha512=request["expected_original_b8_sha512"],
        b8_original_verified_locally=request["b8_status"]=="verified_native_crop")


def clock_rows(pairs, scenes):
    seen = set()
    for r in pairs:
        key = tuple(r[k] for k in ("week_ending","candidate_id","previous_version_id","current_version_id",
                                    "previous_source_file","current_source_file"))
        if key in seen:
            raise ValueError(f"Duplicate B8 candidate {key}")
        seen.add(key)
        cutoff = parse(r["forecast_cutoff_utc"])
        required = [scenes[r[f"{side}_version_id"]] for side in SIDES if r[f"{side}_sensor"]=="B8"]
        if len(required)!=r["route"].split("_").count("B8") or not required:
            raise ValueError(f"B8 candidate routes disagree for {key}")
        published = max(parse(s["cdse_catalog_publication_at_utc"]) for s in required)
        modified = max(parse(s["cdse_catalog_modification_at_utc"]) for s in required)
        bundle = max(parse(s["cdse_bundle_checksums_latest_utc"]) for s in required)
        support = max(parse(r[f"{side}_support_available_at_utc"]) for side in SIDES)
        conservative = max(published,modified,bundle,support)
        yield dict(week_ending=r["week_ending"],release_date=r["release_date"],forecast_cutoff_utc=r["forecast_cutoff_utc"],
            candidate_id=r["candidate_id"],terminal_id=r["terminal_id"],route=r["route"],
            **{f"{side}_{k}":r[f"{side}_{k}"] for side in SIDES for k in ("version_id","source_file")},
            b8_files_json=json.dumps([s["original_b8_filename"] for s in required]),
            cdse_latest_required_product_publication_at_utc=published.isoformat(),
            cdse_latest_required_product_modification_at_utc=modified.isoformat(),
            cdse_latest_required_bundle_checksum_at_utc=bundle.isoformat(),
            other_support_latest_at_utc=support.isoformat(),cdse_catalog_publication_before_cutoff=published<cutoff,
            cdse_catalog_all_clocks_before_cutoff=conservative<cutoff,
            hours_from_latest_evidence_to_cutoff=(cutoff-conservative).total_seconds()/3600,
            historical_b8_file_availability_confirmed=False)


def main():
    queue = read_csv("cushing_b8_download_manifest")
    azure = azure_exact_access(queue)
    export("cushing_b8_azure_exact_access",azure)
    mirrors = copernicus(queue,store=STORE)
    if len(queue)!=len(mirrors) or mirrors.keys()!={r["product_id"] for r in queue}:
        raise ValueError("Copernicus products differ from the B8 queue")
    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(lambda r:scene_file(r,mirrors[r["product_id"]]),queue))
        replies = list(pool.map(lambda r:probe(r["cdse_b8_download_url"]),files))
    for row,response in zip(files,replies,strict=True):
        row.update(cdse_anonymous_http_status=response["http_status"],
                   cdse_anonymous_response_host=response["response_host"],
                   cdse_anonymous_tiff_signature=response["tiff_signature"])
    by_product = {r["product_id"]:r for r in files}
    pairs = list(clock_rows(read_csv("landsat_c2_b8_weekly_pairs"),by_product))
    by_week = defaultdict(list)
    for r in pairs:
        by_week[r["week_ending"]].append(r)
    weeks = []
    for row in read_csv("landsat_c2_b8_weekly_weeks"):
        group = by_week[row["week_ending"]]
        passed = [r for r in group if r["cdse_catalog_all_clocks_before_cutoff"]]
        if len(group)!=int(row["exact_pair_candidates"]) or any(
            (r["release_date"],r["forecast_cutoff_utc"])!=(row["release_date"],row["forecast_cutoff_utc"]) for r in group
        ):
            raise ValueError(f"Copernicus timing candidates differ on {row['week_ending']}")
        weeks.append(dict(week_ending=row["week_ending"],release_date=row["release_date"],
            forecast_cutoff_utc=row["forecast_cutoff_utc"],candidate_pairs=len(group),
            cdse_catalog_pairs_before_cutoff=len(passed),cdse_catalog_tanks_before_cutoff=len({r["candidate_id"] for r in passed}),
            confirmed_historical_b8_file_pairs=0))
    for label,rows in (("files",files),("pairs",pairs),("weeks",weeks)):
        export(f"{STEM}_{label}",rows)
    passed = [r for r in pairs if r["cdse_catalog_all_clocks_before_cutoff"]]
    available = [r for r in files if r["cdse_anonymous_tiff_signature"]]
    report = dict(exact_b8_files=len(files),exact_b8_bytes_from_mirror=sum(r["cdse_b8_bytes"] for r in files),
        exact_b8_file_size_range_bytes=[min(r["cdse_b8_bytes"] for r in files),max(r["cdse_b8_bytes"] for r in files)],
        first_four_control_bytes_from_mirror=sum(r["cdse_b8_bytes"] for r in files[:4]),
        weekly_pair_original_bytes_from_mirror=sum(r["cdse_b8_bytes"] for r in files if r["purpose"]=="gap_pair"),
        anonymous_file_probe_status=dict(Counter(r["cdse_anonymous_http_status"] for r in files)),
        azure_exact_file_probe_status=dict(Counter(r["http_status"] for r in azure)),
        azure_exact_tiff_responses=sum(r["tiff_signature"] for r in azure),
        anonymous_tiff_responses=len(available),mirror_online_products=sum(r["cdse_online"] for r in files),
        all_candidate_pairs=len(pairs),pairs_with_all_cdse_catalog_clocks_before_cutoff=len(passed),
        tank_weeks_with_all_cdse_catalog_clocks_before_cutoff=sum(r["cdse_catalog_tanks_before_cutoff"] for r in weeks),
        weeks_with_all_cdse_catalog_clocks_before_cutoff=sum(r["cdse_catalog_pairs_before_cutoff"]>0 for r in weeks),
        checked_historical_file_pairs=0,scope="Copernicus publishes bundle-level timestamps and lists B8 as a file today. "
        "The audit found no historic individual-file checksum or access timestamp and retrieved no B8 original.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
