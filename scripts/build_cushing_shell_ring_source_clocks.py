"""Archive source publication and object clocks for the aerial-supported shell geometry."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from audit_cushing_aerial_shell_rings import STEM as AERIAL, digest
from build_s1_reference_matches import export, read
from collect_public import RAW, ROOT, TLS, get, write

STEM = "cushing_shell_ring_source_clock"
STORE = RAW / "ok_aerial_2025" / "source_metadata" / "shell_ring_clock"
CATALOG = "https://okmaps.org/OGI/xmlmetadatadownload.aspx?LayerUUID=156f5313-8d11-4774-8cc7-4529fd159dff"


def publication_date():
    path = STORE / "ok_aerial_2025_rgb_iso.xml"
    data = get(CATALOG, path)
    ns = {"gmd": "http://www.isotc211.org/2005/gmd", "gco": "http://www.isotc211.org/2005/gco"}
    root = ET.fromstring(data)
    citation = root.find(".//gmd:identificationInfo//gmd:CI_Citation", ns)
    if citation.findtext("gmd:title/gco:CharacterString", namespaces=ns) != "OK Aerial 2025 3-Band RGB Download":
        raise ValueError("Oklahoma source catalog is not the RGB imagery product")
    stamp, = [r.findtext("gmd:date/gco:Date", namespaces=ns) for r in citation.findall("gmd:date/gmd:CI_Date", ns)
              if r.find("gmd:dateType/gmd:CI_DateTypeCode", ns).get("codeListValue") == "publication"]
    return stamp, str(path.relative_to(ROOT)), hashlib.sha256(data).hexdigest()


def source_receipt(job):
    path, campaign, expected, publication = job
    file = ROOT/path
    if digest(file) != expected:
        raise ValueError(f"Source raster changed since the geometry comparison: {path}")
    item_file = file.parent / "item.json"
    item = json.loads(item_file.read_bytes())
    rgb = campaign == "ok_aerial_2025_rgb"
    url = item["properties"]["ftp_tif"] if rgb else item["assets"]["image"]["href"]
    receipt = STORE / (hashlib.sha256(url.encode()).hexdigest()[:20]+".json")
    if not receipt.is_file():
        with urlopen(Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}), timeout=25, context=TLS) as response:
            info = dict(url=url, content_length=int(response.headers["Content-Length"]), etag=response.headers["ETag"],
                        last_modified_utc=parsedate_to_datetime(response.headers["Last-Modified"]).astimezone(timezone.utc).isoformat(),
                        retrieved_at_utc=datetime.now(timezone.utc).isoformat())
        write(receipt, (json.dumps(info, indent=2)+"\n").encode())
    info = json.loads(receipt.read_bytes())
    if info["url"] != url or not info["etag"] or (rgb and info["content_length"] != file.stat().st_size):
        raise ValueError(f"Source object differs from the original tile receipt: {path}")
    modified_day = info["last_modified_utc"][:10]
    day = max(modified_day, publication[0]) if rgb else modified_day
    return dict(source_path=path, campaign=campaign, source_url=url, source_catalog_path=str(item_file.relative_to(ROOT)),
                catalog_publication_date=publication[0] if rgb else None,
                publication_metadata_path=publication[1] if rgb else None, publication_metadata_sha256=publication[2] if rgb else None,
                object_last_modified_at_utc=info["last_modified_utc"], object_etag=info["etag"], object_bytes=info["content_length"],
                head_receipt=str(receipt.relative_to(ROOT)), source_raster_sha256=expected,
                local_full_size_matches_source=rgb, date_evidence_for_reconstruction=day)


def main():
    accepted = [r for r in read(f"{AERIAL}_two_campaigns") if r["same_lidar_candidate_both_campaigns"] == "True"]
    candidates = {r["candidate_id"] for r in accepted}
    inventory, associations = {}, []
    publication = publication_date()
    for campaign, stem in (("ok_aerial_2025_rgb", AERIAL), ("naip_2023", f"{AERIAL}_naip_2023")):
        for r in read(f"{stem}_comparisons"):
            if r["candidate_id"] not in candidates:
                continue
            for path, expected in json.loads(r["source_paths_sha256_json"]).items():
                args = path, campaign, expected, publication
                if inventory.setdefault(path, args) != args:
                    raise ValueError(f"Conflicting imagery identity for {path}")
                associations.append((r["candidate_id"], path))
    with ThreadPoolExecutor(max_workers=5) as pool:
        sources = {r["source_path"]: r for r in pool.map(source_receipt, inventory.values())}
    dependencies = [dict(candidate_id=candidate, **sources[path]) for candidate, path in sorted(set(associations))]
    export(f"{STEM}_sources", list(sources.values()))
    export(f"{STEM}_roof_dependencies", dependencies)
    print(f"Checked {len(sources)} aerial sources and {len(dependencies)} dependencies across {len(candidates)} contours")


if __name__ == "__main__":
    main()
