"""Archive public OKEIS records and identify the Cushing terminals by permit number."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlencode

from pypdf import PdfReader

from build_s1_reference_matches import DERIVED, export
from collect_public import RAW, ROOT, write, write_csv

STEM = "cushing_okeis"
STORE = RAW / "documents" / "okeis_cushing"
BASE = "https://okeis.deq.ok.gov"
MAP = BASE + "/ext/nsite/DEFAULT/map"
SERVICE = BASE + "/ext/nsite/ss/"
REGION = "-96.92,35.72,-96.52,36.07"
ROUTES = {
    "profile": "1-profile/0-site-details",
    "permits": "3-permit/1-permits",
    "applications": "3-permit/2-active-applications",
    "evaluations": "4-compliance/1-evaluations",
    "submissions": "4-compliance/2-submissions",
    "violations": "5-enforcement/2-violations",
    "actions": "5-enforcement/3-compliance-actions",
    "documents": "6-documents/1-documents",
}
RELEVANT = re.compile(r"\b(?:CUSHING|TERMINAL|TANK|WILDHORSE|DEEPROCK|BUXTON|ALLIED|LITTLE FARM|MARKETLINK|STROUD)\b", re.I)
OFF_SCOPE = re.compile(r"\b(?:DRY CLEANERS|CLNR|TOWER|REPEATER|QUARRY|INCINERATOR|HOSP|TELECOMMUNICATION|WALMART|RELO)\b", re.I)
ID = re.compile(r"-?\d+")
HREF = re.compile(r'<a href="(https://okeis\.deq\.ok\.gov/ext/ncore/downloadpdf/(-?\d+))">Download</a>')


def retrieve(url, path, refresh, *, body=None, accept="application/json"):
    if path.is_file() and not refresh:
        return path.read_bytes()
    command = ["curl", "-fsS", "--compressed", "--http1.1", "--max-time", "35", "--retry", "2", "--max-filesize",
               str(64 * 1024**2), "-H", "User-Agent: Mozilla/5.0", "-H", f"Accept: {accept}", "-H", f"Referer: {MAP}/help"]
    if body is not None:
        command += ["-H", "Content-Type: application/json", "-H", f"Origin: {BASE}", "--data-binary", "@-"]
    payload = subprocess.run([*command, "--", url], input=body, capture_output=True, check=True).stdout
    if accept == "application/pdf":
        if not payload.startswith(b"%PDF-"):
            raise ValueError(f"Response lacks PDF signature at {url}")
    else:
        json.loads(payload)
    write(path, payload)
    return payload


def site_search(refresh):
    west, south, east, north = REGION.split(",")
    query = dict(displayHeight=1400, displayWidth=1800, isIncludeUnmappable="true", filterValuesJson=None,
                 modeId="DEFAULT", searchTerm="", responseContentType="application/json", filterString="",
                 latitudeMin=south, latitudeMax=north, longitudeMin=west, longitudeMax=east)
    payload = json.dumps(dict(insertUpdateList=[query])).encode()
    path = STORE / "map_search.json"
    result = json.loads(retrieve(SERVICE + "explorersites", path, refresh, body=payload))
    candidates = {}
    for group in result["insertUpdateList"]:
        for r in json.loads(group["clusterSites"]) if group.get("clusterSites") else [group]:
            ident, name = r["siteId"], r["siteName"]
            if not ID.fullmatch(ident) or (ident in candidates and candidates[ident] != name):
                raise ValueError(f"Inconsistent OKEIS map identity {ident}")
            candidates[ident] = name
    return result["totalSiteCount"], candidates


def source(site, kind, refresh):
    where = dict(filter=[dict(id=site)])
    params = urlencode(dict(responseContentType="application/json", includeMetadataInResponse="true", loadChildren="true",
                            queryParams=json.dumps(where, separators=(",", ":")), filterString="", top="undefined"))
    url = SERVICE + f"api/nsite-explorer/default-mode/profiles/{ROUTES[kind]}?{params}"
    path = STORE / "sites" / site / f"{kind}.json"
    payload = retrieve(url, path, refresh)
    decoded = json.loads(payload)
    if decoded["hasResultsRemaining"] or decoded["totalCount"] not in (None, len(decoded["queryResults"])):
        raise ValueError(f"Incomplete OKEIS {kind} for {site}")
    return decoded["queryResults"], dict(okeis_site_id=site, kind=kind, rows=len(decoded["queryResults"]),
                                         source_url=url, local_file=str(path.relative_to(ROOT)),
                                         sha256=hashlib.sha256(payload).hexdigest())


def key(permit):
    return re.sub(r"[^0-9A-Z]", "", (permit or "").upper())


def archive_pdf(r, refresh):
    doc = r["document_id"]
    if not doc:
        return dict(r, local_pdf=None, bytes=None, sha256=None, pages=None)
    existing = STORE.parent / "okeis_grandstaff" / f"{doc}.pdf"
    path = existing if existing.is_file() else STORE / "pdfs" / f"{doc}.pdf"
    data = retrieve(r["document_url"], path, refresh, accept="application/pdf")
    if not data.startswith(b"%PDF-"):
        raise ValueError(f"Invalid PDF stored for {doc}")
    return dict(r, local_pdf=str(path.relative_to(ROOT)), bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                pages=len(PdfReader(path).pages))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    displayed_count, found = site_search(args.refresh)
    with (ROOT / "data/reference/permit_documents.csv").open(newline="") as handle:
        references = list(csv.DictReader(handle))
    by_number = defaultdict(list)
    for r in references:
        by_number[key(r["permit_number"])].append(r)
    sites = sorted(i for i, name in found.items() if RELEVANT.search(name) and not OFF_SCOPE.search(name))
    work = [(i, kind) for i in sites for kind in ("profile", "permits")]
    with ThreadPoolExecutor(max_workers=5) as pool:
        data = dict(zip(work, pool.map(lambda job: source(*job, args.refresh), work), strict=True))
    facilities, permit_matches = [], []
    for i in sites:
        profile, _ = data[i, "profile"]
        if len(profile) != 1 or profile[0]["siteName"] != found[i]:
            raise ValueError(f"Site identity differs between OKEIS map and profile: {i}")
        p = profile[0]
        permits, _ = data[i, "permits"]
        matches = [dict(okeis_site_id=i, state_facility_id=ref["facility_id"], facility_key=ref["facility_key"],
                        terminal_id=ref["terminal_id"], permit_number=row["prmtPrmtNumAlt"],
                        reference_permit_number=ref["permit_number"], reference_url=ref["source_url"])
                   for row in permits for ref in by_number[key(row["prmtPrmtNumAlt"])]]
        permit_matches.extend(matches)
        ids = sorted({r["state_facility_id"] for r in matches})
        facilities.append(dict(okeis_site_id=i, name=p["siteName"], operator=p["siteRltdSiteSiteName"], county=p["siteCnty"],
                               address=p["siteAddr"], latitude=p["siteLatitude"], longitude=p["siteLongitude"],
                               operating_status=p["operatingStatusStatus"], permit_rows=len(permits),
                               matched_state_facility_ids_json=json.dumps(ids), site_profile_url=f"{MAP}/results/detail/{i}/Profile"))
    # Inspect public compliance and document tabs for all screened sites, including possible aliases.
    work = [(i, kind) for i in sites for kind in ROUTES if kind not in {"profile", "permits"}]
    with ThreadPoolExecutor(max_workers=5) as pool:
        data.update(zip(work, pool.map(lambda job: source(*job, args.refresh), work), strict=True))
    records = []
    for f in facilities:
        i = f["okeis_site_id"]
        for kind in ROUTES:
            f[f"{kind}_rows"] = len(data[i, kind][0])
        for r in data[i, "documents"][0]:
            link = r["docMgmtDocurl"]
            match = HREF.fullmatch(link) if link else None
            if link and not match:
                raise ValueError(f"Unexpected public download link at {i}")
            records.append(dict(okeis_site_id=i, site_name=f["name"], operator=f["operator"],
                                matched_state_facility_ids_json=f["matched_state_facility_ids_json"],
                                title=r["docMgmtDocName"], description=r["docMgmtDocDescr"], category=r["docMgmtCategory"],
                                created_or_received_at=r["docMgmtDocRvcdCreatedDate"], submission=r["docMgmtSourcenumber"],
                                submission_type=r["docMgmtSourcetype"], submission_status=r["docMgmtSourcestatus"],
                                document_id=match[2] if match else None, document_url=match[1] if match else None,
                                site_documents_url=f"{MAP}/results/detail/{i}/Documents", first_public_availability_verified=False))
    # Multiple site cards can link the same attachment. Download each file once.
    unique = {r["document_id"]: r for r in records if r["document_id"]}
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = {r["document_id"]: r for r in pool.map(lambda r: archive_pdf(r, False), unique.values())}
    documents = [dict(r, **{k: files.get(r["document_id"], {}).get(k) for k in ("local_pdf", "bytes", "sha256", "pages")}) for r in records]
    for suffix, rows in (("sites", facilities), ("permit_matches", permit_matches),
                         ("source_packets", [packet for _, packet in data.values()]), ("documents", documents)):
        export(f"{STEM}_{suffix}", rows)
    write_csv(DERIVED / f"{STEM}_search_hits.csv", ("okeis_site_id", "name", "screened"),
              ((i, name, i in sites) for i, name in sorted(found.items())))
    summary = dict(run_utc=started, refreshed_catalogue=args.refresh, search_bounds=REGION,
                   map_total_site_count=displayed_count, distinct_site_ids_from_returned_pins=len(found),
                   screened_facility_profiles=len(sites), profiles_matched_by_permit=sum(f["matched_state_facility_ids_json"] != "[]" for f in facilities),
                   distinct_state_facility_ids=sorted({r["state_facility_id"] for r in permit_matches}),
                   tab_rows={k: sum(f[f"{k}_rows"] for f in facilities) for k in ROUTES},
                   document_rows=len(documents), documents_without_public_link=sum(not r["document_id"] for r in documents),
                   distinct_pdfs=len(files), pdf_bytes=sum(r["bytes"] for r in files.values()),
                   pdf_pages=sum(r["pages"] for r in files.values()), document_types=Counter(r["submission_type"] for r in documents))
    payload = (json.dumps(summary, indent=2) + "\n").encode()
    write(DERIVED / f"{STEM}_summary.json", payload)
    print(payload.decode(), end="")


if __name__ == "__main__":
    main()
