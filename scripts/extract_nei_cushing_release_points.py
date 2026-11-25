"""Archive EPA FF10 rows for the EIA-803 counties and audit equipment coordinates."""

import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

from pyproj import Geod

from build_nei_roof_evidence import equipment_source
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from extract_nei_cushing_processes import COUNTIES, OUTPUT

URL = "https://gaftp.epa.gov/air/nei/2020/doc/flat_files/SmokeFlatFile_POINT_20230330.zip"
MEMBER = "2020NEI_point_full_20230330.csv"
CRC = 0x4547C2E4
ZIP_SIZE = 155_476_343
MODIFIED = "Thu, 30 Mar 2023 21:20:39 GMT"
SOURCE = OUTPUT / "nei_2020_cushing_county_release_points.csv.gz"
STEM = "cushing_nei_2020_release_point"
PREFIX = re.compile(rb'^"?US"?,"?(' + "|".join(COUNTIES).encode() + rb')"?,')


def extract(path):
    with ZipFile(path) as archive:
        info = archive.getinfo(MEMBER)
    if path.stat().st_size != ZIP_SIZE or info.CRC != CRC:
        raise ValueError("Unexpected EPA 2020 FF10 archive version")
    selected = bytearray()
    counts = Counter()
    with subprocess.Popen(["unzip", "-p", str(path), MEMBER], stdout=subprocess.PIPE) as process:
        for line in process.stdout:
            if not line.startswith(b"#"):
                fields = next(csv.reader([line.decode("ascii").strip()]))
                if fields[:4] != ["COUNTRY_CD", "REGION_CD", "TRIBAL_CODE", "FACILITY_ID"]:
                    raise ValueError("Unrecognized EPA FF10 header")
                selected.extend(line)
                break
        for line in process.stdout:
            if match := PREFIX.match(line):
                counts[match[1].decode()] += 1
                selected.extend(line)
        if process.wait():
            raise ValueError("Could not extract and verify EPA FF10 ZIP member")
    if counts.keys() != COUNTIES.keys():
        raise ValueError("EPA FF10 lacks a study county")
    data = gzip.compress(selected, mtime=0)
    with path.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = dict(inventory_year=2020, archive_url=URL, archive_member=MEMBER, archive_bytes=ZIP_SIZE,
                    archive_sha256=source_hash, member_crc32=f"{CRC:08x}", source_http_last_modified_utc=MODIFIED,
                    rows_by_county={COUNTIES[k]: counts[k] for k in COUNTIES}, extract_file=SOURCE.name,
                    extract_sha256=hashlib.sha256(data).hexdigest())
    write(SOURCE, data)
    write(SOURCE.with_suffix(".json"), (json.dumps(manifest, indent=2)+"\n").encode())


def coordinates():
    manifest = json.loads(SOURCE.with_suffix(".json").read_bytes())
    source = SOURCE.read_bytes()
    if hashlib.sha256(source).hexdigest() != manifest["extract_sha256"]:
        raise ValueError("EPA FF10 county extract does not match its manifest")
    types = ("AGY_FACILITY_ID", "AGY_UNIT_ID", "AGY_REL_POINT_ID", "ERPTYPE", "LONGITUDE", "LATITUDE", "LL_DATUM",
             "HORIZ_COLL_MTHD", "STKHGT", "STKDIAM", "FUG_HEIGHT", "FUG_WIDTH_XDIM", "FUG_LENGTH_YDIM", "FUG_ANGLE")
    with gzip.open(SOURCE, "rt", encoding="latin-1", newline="") as stream:
        groups = defaultdict(list)
        county_counts = Counter()
        for row in csv.DictReader(stream):
            if None in row or any(row[k] is None for k in types):
                raise ValueError("Malformed row in the saved EPA FF10 extract")
            county_counts[row["REGION_CD"]] += 1
            if row["FACILITY_ID"] and row["UNIT_ID"] and row["REL_POINT_ID"]:
                key = row["FACILITY_ID"], row["UNIT_ID"], row["REL_POINT_ID"], *(row[k].strip() for k in types)
                groups[key].append(row["SCC"])
        if {COUNTIES[k]: county_counts[k] for k in county_counts} != manifest["rows_by_county"]:
            raise ValueError("County row counts differ from archive extraction")
    points = []
    for key, codes in sorted(groups.items()):
        facility, unit, release, *details = key
        points.append(dict(eis_facility_id=facility, eis_unit_id=unit, eis_release_point_id=release,
                           **{k.lower(): v for k, v in zip(types, details, strict=True)},
                           scc_codes="|".join(sorted(set(codes))), ff10_pollutant_rows=len(codes)))
    return manifest, points


def equipment_audit(points):
    geod = Geod(ellps="WGS84")
    evidence = {(r["eis_facility_id"], r["eis_unit_id"]): r for r in read("nei_roof_equipment_evidence")
                if r["inventory_year"] == "2020"}
    candidates = defaultdict(list)
    for r in read("tank_engineering_registry"):
        for equipment in json.loads(r["possible_equipment_ids_json"] or "[]"):
            candidates[r["documented_terminal_id"], equipment].append(r)
    site_positions = defaultdict(set)
    path, _, _ = equipment_source(2020)
    with gzip.open(path, "rt", encoding="latin-1", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["site longitude"] and row["site latitude"]:
                site_positions[row["eis facility id"]].add((float(row["site longitude"]), float(row["site latitude"])))
    for p in points:
        key = p["eis_facility_id"], p["eis_unit_id"]
        if key not in evidence:
            continue
        r = evidence[key]
        terminal, equipment = r["terminal_id"], r["equipment_id"]
        sites = site_positions[key[0]]
        if p["longitude"] and p["latitude"]:
            lon, lat = (float(p[k]) for k in ("longitude", "latitude"))
            facility_distance = min((geod.inv(lon, lat, x, y)[2] for x, y in sites), default=None)
            distances = {c["candidate_id"]: geod.inv(lon, lat, float(c["longitude"]), float(c["latitude"]))[2]
                         for c in candidates[terminal, equipment]}
            coincides = any(abs(lon-x) <= .000005 and abs(lat-y) <= .000005 for x, y in sites) if sites else None
        else:
            facility_distance, distances, coincides = None, {}, None
        yield p | dict(terminal_id=terminal, equipment_id=equipment, possible_contours=len(candidates[terminal, equipment]),
                       facility_reference_positions_lonlat_json=json.dumps(sorted(sites)),
                       distance_to_facility_reference_m=facility_distance, candidate_centre_distances_m_json=json.dumps(distances),
                       coincides_with_facility_reference=coincides,
                       individual_contour_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, nargs="?", help="Download the archived EPA ZIP before passing this path")
    args = parser.parse_args()
    if args.archive is not None:
        extract(args.archive)
    manifest, points = coordinates()
    checked = list(equipment_audit(points))
    cvr = [r for r in checked if r["terminal_id"] == "cushing_connect"]
    if {r["equipment_id"] for r in cvr} != {f"25C{i}" for i in range(1, 7)}:
        raise ValueError("The FF10 extract does not contain the full Cushing Connect roster")
    export(f"{STEM}_coordinates", points)
    export(f"{STEM}_equipment_audit", checked)
    export(f"{STEM}_cushing_connect", cvr)
    report = dict(ff10_county_rows=manifest["rows_by_county"], release_point_coordinate_rows=len(points),
                  matched_equipment_rows=len(checked), matched_rows_coincident_with_facility=sum(
                      r["coincides_with_facility_reference"] is True for r in checked),
                  cushing_connect_equipment_rows=len(cvr),
                  cushing_connect_rows_coincident_with_facility=sum(r["coincides_with_facility_reference"] for r in cvr),
                  cushing_connect_distinct_coordinate_pairs=len({(r["longitude"], r["latitude"]) for r in cvr}))
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
