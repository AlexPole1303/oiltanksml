"""Download and validate precise public orbit ephemerides for planned Cushing SLC inputs."""
import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import PurePosixPath
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import RAW, ROOT, TLS, write

STEM = "s1_slc_orbits"
STORE = RAW / STEM
ASF = "https://s1-orbits.asf.alaska.edu/scene"
ORBIT_NAME = re.compile(r"(S1[A-D])_OPER_AUX_(POEORB|RESORB)_OPOD_(\d{8}T\d{6})_V(\d{8}T\d{6})_(\d{8}T\d{6})\.EOF")


def utc(stamp):
    return datetime.fromisoformat(stamp.removeprefix("UTC=")).replace(tzinfo=timezone.utc)


def compact(stamp):
    return datetime.strptime(stamp,"%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


def target_products():
    result = {r["product_name"]:r for r in read_csv("s1_holdout_gap_burst_parents")}
    catalog = {r["product_name"]:r for r in read_csv("s1_cushing_catalog") if r["product_type"]=="IW_SLC__1S"}
    control, = {(r["original_slc_parent_product"],r["original_slc_parent_id"])
                for r in read_csv("s1_slc_2019_reference_burst_plan") if r["qualified_aerial_shadow_reference"]=="True"}
    name,identity = control
    if name in result or catalog[name]["product_id"]!=identity:
        raise ValueError("The 2019 SLC control differs from the source parent catalogue")
    result[name] = catalog[name]
    return [dict(catalog[name],scope="2019_aerial_control" if name==control[0] else "holdout_gap") for name in sorted(result)]


def parse_orbit(payload,filename):
    match = ORBIT_NAME.fullmatch(filename)
    if not match:
        raise ValueError(f"Unexpected orbit filename {filename}")
    satellite,kind,*stamps = match.groups()
    created,start,stop = map(compact,stamps)
    tree = ET.fromstring(payload)
    header = tree.find("Earth_Explorer_Header/Fixed_Header")
    vector_header = tree.find("Earth_Explorer_Header/Variable_Header")
    if (header.findtext("File_Name")!=filename.removesuffix(".EOF") or header.findtext("File_Type")!=f"AUX_{kind}"
        or header.findtext("Mission")!=f"Sentinel-1{satellite[-1]}"
        or (utc(header.findtext("Source/Creation_Date")),utc(header.findtext("Validity_Period/Validity_Start")),
            utc(header.findtext("Validity_Period/Validity_Stop")))!=(created,start,stop)
        or (vector_header.findtext("Ref_Frame"),vector_header.findtext("Time_Reference"))!=("EARTH_FIXED","UTC")):
        raise ValueError(f"Wrong orbit header for {filename}")
    vectors = tree.find("Data_Block/List_of_OSVs")
    stamps = [utc(row.findtext("UTC")) for row in vectors]
    seconds = np.array([(stamp-start).total_seconds() for stamp in stamps])
    state = np.array([[float(row.findtext(axis)) for axis in ("X","Y","Z","VX","VY","VZ")] for row in vectors])
    if (len(stamps)!=int(vectors.attrib["count"]) or len(stamps)<3 or not np.isfinite(state).all()
        or not np.allclose(np.diff(seconds),10,rtol=0,atol=1e-7)
        or not start<stop or abs((stamps[0]-start).total_seconds())>10 or abs((stamps[-1]-stop).total_seconds())>10):
        raise ValueError(f"Incomplete or nonphysical state vectors for {filename}")
    if any(row.find(axis).attrib.get("unit") != ("m" if len(axis)==1 else "m/s")
           for row in vectors for axis in ("X","Y","Z","VX","VY","VZ")):
        raise ValueError(f"Unexpected orbit state-vector units for {filename}")
    return dict(satellite=satellite,orbit_kind=kind,created=created,start=start,stop=stop,
                seconds=seconds,state=state,orbit_numbers=np.array([int(r.findtext("Absolute_Orbit")) for r in vectors]))


def orbit_for(product, *, accept_relative_osv_labels=False):
    name = product["product_name"]
    request_url = f"{ASF}/{name.removesuffix('.SAFE')}"
    pointer = STORE / f"{name.removesuffix('.SAFE')}.json"
    if pointer.is_file():
        entry = json.loads(pointer.read_bytes())
        payload = gzip.decompress((ROOT/entry["local_orbit_gzip"]).read_bytes())
    else:
        with urlopen(Request(request_url),context=TLS,timeout=40) as response:
            final = urlsplit(response.url)
            filename = PurePosixPath(final.path).name
            if (final.scheme!="https" or final.hostname not in ("s1-orbits.s3.amazonaws.com","s1-orbits.s3.us-west-2.amazonaws.com")
                or not ORBIT_NAME.fullmatch(filename) or final.path.split("/")[1]!="AUX_POEORB"):
                raise ValueError(f"ASF returned an unexpected orbit source for {name}")
            etag = response.headers["ETag"].strip('"')
            size = int(response.headers["Content-Length"])
            if not 0<size<16_000_000:
                raise ValueError(f"Unexpected orbit file size for {name}")
            payload = response.read(size+1)
            last = parsedate_to_datetime(response.headers["Last-Modified"])
        local = STORE / f"{filename}.gz"
        entry = dict(orbit_filename=filename, orbit_source_url=final.geturl(), public_asf_request_url=request_url,
            source_etag=etag, source_bytes=size, archive_last_modified_at_utc=last.isoformat(),
            local_orbit_gzip=str(local.relative_to(ROOT)),orbit_sha256=hashlib.sha256(payload).hexdigest())
    if (entry["public_asf_request_url"]!=request_url or len(payload)!=entry["source_bytes"]
        or hashlib.sha256(payload).hexdigest()!=entry["orbit_sha256"]
        or hashlib.md5(payload).hexdigest()!=entry["source_etag"]):
        raise ValueError(f"Orbit payload differs from the archive for {name}")
    orbit = parse_orbit(payload,entry["orbit_filename"])
    begin,end = (utc(product[k]) for k in ("capture_start_utc","capture_end_utc"))
    if (orbit["satellite"]!=product["satellite"] or orbit["orbit_kind"]!="POEORB"
        or not orbit["start"]<=begin<end<=orbit["stop"]
        or not orbit["seconds"][0]<=(begin-orbit["start"]).total_seconds()<(end-orbit["start"]).total_seconds()<=orbit["seconds"][-1]
        or orbit["created"]>utc(entry["archive_last_modified_at_utc"])):
        raise ValueError(f"Precise orbit does not cover the parent acquisition {name}")
    near = np.abs(orbit["seconds"]-(begin-orbit["start"]).total_seconds())<30
    near_orbits = set(map(int,orbit["orbit_numbers"][near]))
    matches_orbit = int(product["absolute_orbit"]) in near_orbits
    # Some control EOFs use small orbit labels. Preserve the mismatch for review.
    relative_labels = max(near_orbits, default=100) < 30
    known_2019_control = product["scope"] == "2019_aerial_control" and relative_labels
    verify_against_grd = accept_relative_osv_labels and relative_labels and max(orbit["orbit_numbers"]) < 30
    if not (matches_orbit or known_2019_control or verify_against_grd):
        raise ValueError(f"Precise orbit disagrees with absolute orbit of {name}")
    if not pointer.exists():
        write(ROOT/entry["local_orbit_gzip"],gzip.compress(payload,mtime=0))
        write(pointer,json.dumps(entry,indent=2).encode())
    return dict(product_name=name,product_id=product["product_id"],scope=product["scope"],satellite=product["satellite"],
        absolute_orbit=product["absolute_orbit"],capture_start_utc=begin.isoformat(),capture_end_utc=end.isoformat(),
        osvs_match_parent_absolute_orbit=matches_orbit,osv_orbit_numbers_near_acquisition_json=json.dumps(sorted(near_orbits)),
        precise_orbit_created_at_utc=orbit["created"].isoformat(),validity_start_utc=orbit["start"].isoformat(),
        validity_end_utc=orbit["stop"].isoformat(),state_vectors=len(orbit["seconds"]),**entry)


def main():
    targets = target_products()
    with ThreadPoolExecutor(max_workers=5) as pool:
        records = []
        for i,row in enumerate(pool.map(orbit_for,targets),1):
            records.append(row)
            if i%10==0 or i==len(targets):
                print(f"Verified precise orbit data for {i}/{len(targets)} Sentinel-1 parent scenes",flush=True)
    export(STEM,records)
    paths = {ROOT/r["local_orbit_gzip"] for r in records}
    report = dict(parent_products=len(records),precise_orbit_files=len(paths),local_compressed_bytes=sum(p.stat().st_size for p in paths),
        scope="Verified public ASF precise orbit files. These provide spacecraft states and are typically retrospective. "
              "They do not supply SLC complex samples or an as-of orbit for an EIA cutoff.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
