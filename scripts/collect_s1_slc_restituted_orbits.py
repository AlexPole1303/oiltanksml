"""Recover dated restituted orbits from the ASF historical public directory."""
import gzip
import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener, urlopen

import numpy as np

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import ROOT, TLS, get, write, write_csv
from collect_s1_slc_orbits import ORBIT_NAME, STORE, compact, parse_orbit, utc

STEM = "s1_slc_restituted_orbits"
INDEX = "https://s1qc.asf.alaska.edu/aux_resorb/"
AWS = "https://s1-orbits.s3.us-west-2.amazonaws.com/AUX_RESORB/"
LIST_ENTRY = re.compile(r'<a href="([^"/]+\.EOF)">[^\n]*?</a>\s+(\d\d-[A-Za-z]{3}-\d{4} \d\d:\d\d)')


def listed_orbits():
    source = STORE/"asf_resorb_index.html.gz"
    page = get(INDEX)
    write(source,gzip.compress(page,mtime=0))
    rows = defaultdict(list)
    for filename, stamp in LIST_ENTRY.findall(page.decode("ascii")):
        parts = ORBIT_NAME.fullmatch(filename)
        if parts is None or parts[2]!="RESORB":
            continue
        modified = datetime.strptime(stamp,"%d-%b-%Y %H:%M").replace(tzinfo=timezone.utc)
        creation,start,stop = map(compact,parts.groups()[2:])
        rows[parts[1]].append((filename,modified,creation,start,stop))
    if not rows:
        raise ValueError("ASF directory contains no restituted orbit files")
    return rows,str(source.relative_to(ROOT)),hashlib.sha256(page).hexdigest()


def select_orbit(product,rows):
    begin,end = (utc(product[k]) for k in ("capture_start_utc","capture_end_utc"))
    found = [r for r in rows.get(product["satellite"],()) if r[3]<=begin<end<=r[4]]
    if not found:
        return None
    # Earliest listed archive version covering the entire parent interval.
    return min(found,key=lambda r:(r[1],r[2],r[0]))


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def verify_file(job):
    product,selected = job
    if selected is None:
        return dict(product_id=product["product_id"],product_name=product["product_name"],
                    status="no_covering_file_in_asf_index")
    filename,listing_modified,_,_,_ = selected
    url = INDEX+filename
    path = STORE/f"{filename}.gz"
    fields = dict(product_id=product["product_id"],product_name=product["product_name"],
        orbit_filename=filename,asf_directory_orbit_url=url,asf_directory_modified_at_utc=listing_modified.isoformat())
    try:
        with build_opener(NoRedirect(),HTTPSHandler(context=TLS)).open(Request(url,method="HEAD"),timeout=20) as response:
            fields["asf_http_status"] = response.status
    except HTTPError as e:
        fields.update(asf_http_status=e.code,asf_redirect_host=urlsplit(e.headers.get("Location","")).hostname)
    candidates = ([url] if fields["asf_http_status"]==200 else [])+[AWS+filename]
    for download in candidates:
        try:
            with urlopen(Request(download),context=TLS,timeout=25) as response:
                if response.url!=download or response.headers.get_content_type()=="text/html":
                    raise ValueError(f"Wrong restituted orbit response for {filename}")
                header_modified = parsedate_to_datetime(response.headers["Last-Modified"])
                size = int(response.headers["Content-Length"])
                if not 0<size<4_000_000:
                    raise ValueError(f"Unexpected restituted orbit file size for {filename}")
                payload = response.read(size+1)
                etag = response.headers.get("ETag","").strip('"')
        except HTTPError as e:
            if e.code not in (401,403,404):
                raise
            fields["mirror_http_status"] = e.code
        else:
            break
    else:
        return dict(fields,status="asf_listing_only")
    orbit = parse_orbit(payload,filename)
    begin,end = (utc(product[k]) for k in ("capture_start_utc","capture_end_utc"))
    if (len(payload)!=size or not orbit["start"]<=begin<end<=orbit["stop"]
        or not orbit["seconds"][0]<=(begin-orbit["start"]).total_seconds()<(end-orbit["start"]).total_seconds()<=orbit["seconds"][-1]
        or (download==url and header_modified.replace(second=0,microsecond=0)!=listing_modified)
        or (download!=url and hashlib.md5(payload).hexdigest()!=etag)
        or orbit["created"]>header_modified or orbit["satellite"]!=product["satellite"]):
        raise ValueError(f"ASF restituted orbit does not match source metadata for {filename}")
    around = np.abs(orbit["seconds"]-(begin-orbit["start"]).total_seconds())<30
    labels = set(map(int,orbit["orbit_numbers"][around]))
    if int(product["absolute_orbit"]) not in labels:
        raise ValueError(f"Restituted state vectors disagree with parent orbit for {filename}")
    write(path,gzip.compress(payload,mtime=0))
    return dict(fields,status="verified_resorb",orbit_source_url=download,archive_last_modified_at_utc=header_modified.isoformat(),
        orbit_created_at_utc=orbit["created"].isoformat(),validity_start_utc=orbit["start"].isoformat(),
        validity_end_utc=orbit["stop"].isoformat(),source_bytes=size,orbit_sha256=hashlib.sha256(payload).hexdigest(),
        local_orbit_gzip=str(path.relative_to(ROOT)))


def main():
    records,source,digest = listed_orbits()
    products = [r for r in read_csv("s1_slc_orbits") if r["scope"]=="holdout_gap"]
    jobs = [(product,select_orbit(product,records)) for product in products]
    with ThreadPoolExecutor(max_workers=6) as pool:
        values = list(pool.map(verify_file,jobs))
    ready = {r["product_id"]:r for r in values if r["status"]=="verified_resorb"}
    listed = {r["product_id"]:r for r in values if "asf_directory_modified_at_utc" in r}
    weeks, listed_weeks = defaultdict(set),defaultdict(set)
    clocks = {r["week_ending"]:utc(r["cutoff_utc"]) for r in read_csv("holdout_optical_weeks")}
    for r in read_csv("s1_holdout_gap_burst_tank_pairs"):
        week = r["week_ending"]
        parents = [ready.get(r[f"{side}_product_id"]) for side in ("previous","current")]
        if r["full_context_in_both_bursts"]!="True":
            continue
        if all(p and utc(p["archive_last_modified_at_utc"])<clocks[week] for p in parents):
            weeks[week].add(r["candidate_id"])
        listings = [listed.get(r[f"{side}_product_id"]) for side in ("previous","current")]
        if all(p and utc(p["asf_directory_modified_at_utc"])<clocks[week] for p in listings):
            listed_weeks[week].add(r["candidate_id"])
    export(f"{STEM}_weeks",[dict(week_ending=r["week_ending"],release_date=r["release_date"],
        tank_weeks_with_both_verified_resorb_mtimes_before_cutoff=len(weeks[r["week_ending"]]),
        tank_weeks_with_both_asf_resorb_listed_before_cutoff=len(listed_weeks[r["week_ending"]]))
        for r in read_csv("s1_holdout_gap_burst_weeks")])
    fields = tuple(dict.fromkeys(k for r in values for k in r))
    write_csv(DERIVED/f"{STEM}.csv",fields,([r.get(k) for k in fields] for r in values))
    report = dict(planned_parent_scenes=len(products),parent_scenes_with_verified_resorb=len(ready),
        parent_scenes_with_asf_resorb_listing=len(listed),
        weeks_with_both_verified_resorb_mtimes_before_cutoff=sum(bool(r) for r in weeks.values()),
        tank_weeks_with_both_verified_resorb_mtimes_before_cutoff=sum(map(len,weeks.values())),
        weeks_with_both_asf_resorb_listed_before_cutoff=sum(bool(r) for r in listed_weeks.values()),
        tank_weeks_with_both_asf_resorb_listed_before_cutoff=sum(map(len,listed_weeks.values())),
        source_directory_snapshot=source,source_directory_sha256=digest,
        scope="ASF directory and HTTP Last-Modified clocks are archive evidence for the specific restituted orbit objects. "
              "They do not establish availability of the SLC signal or valid radar sample lines.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
