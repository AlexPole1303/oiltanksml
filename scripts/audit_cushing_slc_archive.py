"""Inventory exact Sentinel-1 SLC burst inputs and verify the public download boundary."""
import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import PurePosixPath
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from uuid import UUID

from audit_cushing_b8_archive import listing
from build_s1_reference_matches import export, read
from build_s1_slc_orbit_geometry import burst_catalog
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, TLS, get, write


STEM = "cushing_slc_archive"
STORE = RAW / STEM
CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD = "https://download.dataspace.copernicus.eu/odata/v1/Products"
BURST_DATA = "https://bursts.dataspace.copernicus.eu/v1/product"
MEASUREMENT = re.compile(r"s1[a-d]-(iw[123])-slc-(vv|vh)-[a-z0-9-]+\.tiff")


def request_rows():
    parents = {r["product_id"]: r for r in read("s1_holdout_gap_burst_parents")}
    holdout = [dict(row, scope="holdout", cdse_catalog_file=parents[row["product_id"]]["cdse_catalog_file"],
                    asf_catalog_file=parents[row["product_id"]]["asf_catalog_file"])
               for row in read("s1_holdout_gap_burst_download_manifest")]
    controls = {}
    for row in read("s1_slc_2019_reference_burst_plan"):
        for pol in ("vv", "vh"):
            key = row[f"cdse_{pol}_burst_uuid"]
            request = dict(scope="aerial_2019", product_id=row["original_slc_parent_id"],
                product_name=row["original_slc_parent_product"], cdse_uuid=key, swath=row["selected_swath"],
                polarization=pol.upper(), asf_data_url=row[f"asf_{pol}_data_url"],
                asf_annotation_url=row[f"asf_{pol}_annotation_url"], cdse_catalog_file=row["cdse_source_file"],
                asf_catalog_file=row["asf_source_file"], weeks="")
            if controls.setdefault(key, request) != request:
                raise ValueError(f"Conflicting control identities for {key}")
    for row in read("cushing_slc_laser_controls"):
        for pol in ("vv", "vh"):
            key = row[f"{pol}_cdse_uuid"]
            request = dict(scope="icesat2_control", product_id=row["product_id"], product_name=row["product_name"],
                cdse_uuid=key, swath=row["swath"], polarization=pol.upper(), asf_data_url=row[f"{pol}_asf_data_url"],
                asf_annotation_url=row[f"{pol}_asf_annotation_url"], cdse_catalog_file=row[f"{pol}_cdse_catalog_file"],
                asf_catalog_file=row[f"{pol}_asf_catalog_file"], weeks="")
            if controls.setdefault(key, request) != request:
                raise ValueError(f"Conflicting laser control identities for {key}")
    rows = [*controls.values(), *holdout]
    if len({r["cdse_uuid"] for r in rows}) != len(rows):
        raise ValueError("Duplicate Copernicus burst IDs in the requested inventory")
    return rows


def members(url, path, parent=None):
    rows = listing(url, path)
    if len(rows) != len({r["Name"] for r in rows}):
        raise ValueError(f"Repeated members in {url}")
    if parent is not None and len(rows) != parent["ChildrenNumber"]:
        raise ValueError(f"Incomplete folder listing {url}")
    for r in rows:
        if r["Name"] != r["Id"] or r["Nodes"]["uri"] != f"{url}({quote(r['Name'])})/Nodes":
            raise ValueError(f"Unexpected source path for {r['Name']}")
    return {r["Name"]: r for r in rows}


def asf_catalog_record(request):
    if not request["asf_data_url"]:
        return None
    data = json.loads((ROOT / request["asf_catalog_file"]).read_bytes())
    record, = (f["properties"] for f in data["features"] if f["properties"]["url"] == request["asf_data_url"])
    if record["additionalUrls"] != [request["asf_annotation_url"]]:
        raise ValueError(f"Different ASF XML route for {request['cdse_uuid']}")
    return record


def parent_files(job, *, store=STORE):
    parent_id, requests = job
    name, = {r["product_name"] for r in requests}
    folder = store / parent_id
    source_url = f"{CATALOG}/Products({UUID(parent_id)})"
    meta = json.loads(get(source_url, folder / "product.json", refresh=True))
    captured = datetime.fromisoformat(meta["ContentDate"]["Start"])
    published = datetime.fromisoformat(meta["PublicationDate"])
    modified = datetime.fromisoformat(meta["ModificationDate"])
    if (meta["Id"] != parent_id or meta["Name"] != name or PurePosixPath(meta["S3Path"]).name != name
            or not captured < published <= modified or not isinstance(meta["Online"], bool)):
        raise ValueError(f"Unexpected parent product identity for {name}")
    base = f"{DOWNLOAD}({parent_id})/Nodes"
    root = members(base, folder / "root.json")[name]
    body = members(root["Nodes"]["uri"], folder / "body.json", root)
    measurement = members(body["measurement"]["Nodes"]["uri"], folder / "measurement.json", body["measurement"])
    annotation = members(body["annotation"]["Nodes"]["uri"], folder / "annotation.json", body["annotation"])
    calibration = members(annotation["calibration"]["Nodes"]["uri"], folder / "calibration.json", annotation["calibration"])
    assets = {}
    bursts = []
    for r in requests:
        burst = burst_catalog(r["cdse_catalog_file"])[r["cdse_uuid"]]
        swath, pol = r["swath"].lower(), r["polarization"].lower()
        path = PurePosixPath(burst["S3Path"])
        filename = path.name
        name_match = MEASUREMENT.fullmatch(filename)
        if (burst["ParentProductId"] != parent_id or burst["ParentProductName"] != name
                or (burst["SwathIdentifier"].lower(), burst["PolarisationChannels"].lower()) != (swath, pol)
                or path.parent.parent.as_posix() != meta["S3Path"] or not name_match
                or name_match.groups() != (swath, pol)):
            raise ValueError(f"Unexpected burst and source TIFF identity for {r['cdse_uuid']}")
        xml = filename.removesuffix(".tiff") + ".xml"
        files = {
            "measurement": (measurement[filename], path.as_posix()),
            "annotation": (annotation[xml], f"{meta['S3Path']}/annotation/{xml}"),
            "calibration": (calibration[f"calibration-{xml}"], f"{meta['S3Path']}/annotation/calibration/calibration-{xml}"),
            "noise": (calibration[f"noise-{xml}"], f"{meta['S3Path']}/annotation/calibration/noise-{xml}"),
            "manifest": (body["manifest.safe"], f"{meta['S3Path']}/manifest.safe"),
        }
        for role, (entry, remote) in files.items():
            if entry["ChildrenNumber"] or entry["ContentLength"] <= 0:
                raise ValueError(f"Not a downloadable file: {remote}")
            url = entry["Nodes"]["uri"].removesuffix("/Nodes") + "/$value"
            assets[remote] = dict(product_id=parent_id, product_name=name, role=role, filename=entry["Name"],
                bytes=entry["ContentLength"], s3_path=remote, download_url=url)
        line_count = int(burst["LinesPerBurst"])
        samples = int(burst["SamplesPerBurst"])
        expected_complex_bytes = line_count * samples * 4
        offset = int(burst["ByteOffset"])
        if not (0 < offset < offset + expected_complex_bytes <= measurement[filename]["ContentLength"]):
            raise ValueError(f"Catalog burst span exceeds parent TIFF for {r['cdse_uuid']}")
        asf = asf_catalog_record(r)
        if asf and (int(asf["bytes"]) != expected_complex_bytes
                    or int(asf["burst"]["samplesPerBurst"]) != samples
                    or int(asf["burst"]["relativeBurstID"]) != burst["BurstId"]):
            raise ValueError(f"ASF burst geometry differs for {r['cdse_uuid']}")
        bursts.append(dict(r, relative_burst_id=burst["BurstId"], relative_orbit=burst["RelativeOrbitNumber"],
            cdse_absolute_burst_id=burst["AbsoluteBurstId"], azimuth_time_utc=burst["AzimuthTime"],
            catalogue_start_utc=burst["BeginningDateTime"], catalogue_end_utc=burst["EndingDateTime"],
            cdse_parent_publication_utc=meta["PublicationDate"], cdse_parent_modified_utc=meta["ModificationDate"],
            cdse_parent_online=meta["Online"], cdse_parent_package_bytes=meta["ContentLength"],
            byte_offset=offset, lines_per_burst=line_count, samples_per_burst=samples,
            expected_complex_sample_bytes=expected_complex_bytes,
            asf_catalog_complex_sample_bytes=int(asf["bytes"]) if asf else None,
            asf_burst_index=int(asf["burst"]["burstIndex"]) if asf else None,
            cdse_burst_download_url=f"{CATALOG}/Bursts({r['cdse_uuid']})/$value",
            cdse_direct_burst_url=f"{BURST_DATA}/{r['cdse_uuid']}",
            **{f"{role}_s3_path": remote for role, (_, remote) in files.items()},
            cdse_parent_metadata_file=str((folder / "product.json").relative_to(ROOT))))
    return bursts, list(assets.values())


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def probe(job):
    role, identity, url = job
    opener = build_opener(HTTPSHandler(context=TLS), NoRedirect())
    for attempt in range(3):
        try:
            with opener.open(Request(url, headers={"Range": "bytes=0-63"}), timeout=25) as r:
                status, content, location, body = r.status, r.headers.get_content_type(), "", r.read(64)
        except HTTPError as e:
            with e:
                status, content = e.code, e.headers.get_content_type()
                location = urlsplit(e.headers.get("Location", "")).hostname or ""
                body = e.read(64)
            if status == 429 and attempt < 2:
                time.sleep(2 ** (attempt + 1))
                continue
        break
    signature = ""
    if 200 <= status <= 299:
        if body[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
            signature = "TIFF"
        elif body.startswith(b"PK\x03\x04"):
            signature = "ZIP"
        elif content in {"application/xml", "text/xml"} and body.lstrip().startswith(b"<"):
            signature = "XML"
    return dict(route=role, archive_id=identity, url=url, status=status,
                response_content_type=content, redirect_host=location, source_signature=signature)


def main():
    groups = defaultdict(list)
    for row in request_rows():
        groups[row["product_id"]].append(row)
    with ThreadPoolExecutor(max_workers=10) as pool:
        parents = list(pool.map(parent_files, groups.items()))
    bursts = [row for rows, _ in parents for row in rows]
    assets = [row for _, rows in parents for row in rows]
    asf = [r for r in bursts if r["asf_data_url"]]
    jobs = [("cdse_burst", r["cdse_uuid"], r["cdse_burst_download_url"]) for r in bursts]
    jobs.extend((route, r["cdse_uuid"], r[column]) for r in asf for route, column in
                (("asf_tiff", "asf_data_url"), ("asf_xml", "asf_annotation_url")))
    jobs.extend(("cdse_" + r["role"], r["s3_path"], r["download_url"]) for r in assets)
    with ThreadPoolExecutor(max_workers=12) as pool:
        probes = list(pool.map(probe, jobs))
    with ThreadPoolExecutor(max_workers=4) as pool:
        probes.extend(pool.map(probe, (("cdse_direct_burst", r["cdse_uuid"], r["cdse_direct_burst_url"]) for r in bursts)))
    files = {r["s3_path"]: r for r in assets}
    if len(files) != len(assets):
        raise ValueError("Repeated exact SAFE asset keys")
    for r in bursts:
        r["cdse_parent_swath_bytes"] = files[r["measurement_s3_path"]]["bytes"]
    export(STEM + "_bursts", bursts)
    export(STEM + "_safe_files", assets)
    export(STEM + "_anonymous_probes", probes)
    distinct_weeks = {w for r in bursts for w in r["weeks"].split("|") if w}
    report = dict(observed_utc=datetime.now(timezone.utc).isoformat(), requested_bursts=len(bursts),
        parent_products=len(parents), scopes=Counter(r["scope"] for r in bursts), holdout_weeks=len(distinct_weeks),
        asf_burst_routes=len(asf), parent_products_online=sum(rows[0]["cdse_parent_online"] for rows, _ in parents),
        safe_files_by_role=Counter(r["role"] for r in assets),
        parent_swath_measurement_bytes=sum(r["bytes"] for r in assets if r["role"] == "measurement"),
        burst_complex_sample_bytes=sum(r["expected_complex_sample_bytes"] for r in bursts),
        burst_complex_sample_bytes_by_scope={scope: sum(r["expected_complex_sample_bytes"] for r in bursts if r["scope"] == scope)
                                             for scope in sorted({r["scope"] for r in bursts})},
        asf_cdse_complex_sample_sizes_agree=len(asf),
        anonymous_responses_by_route={route: Counter(str(r["status"]) for r in probes if r["route"] == route)
                                      for route in sorted({r["route"] for r in probes})},
        anonymous_source_signatures=Counter(r["source_signature"] for r in probes if r["source_signature"]),
        interpretation="SAFE file listings are observations of the current archive. Burst sample bytes are computed "
        "from the catalogue grid and 16-bit I/Q, exclusive of container overhead. Listing and probe do not establish "
        "historic availability of the listed member or validity of its pixel rows.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / (STEM + "_summary.json"), payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
