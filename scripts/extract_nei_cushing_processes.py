"""Archive the published EPA NEI process records for the three Cushing study counties."""
import argparse
import csv
import gzip
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from collect_eia_803_area import FIPS
from collect_public import RAW, write

EPA = "https://gaftp.epa.gov/air/nei/"
OUTPUT = RAW / "documents" / "epa_nei"
ARCHIVES = {
    2014: (
        EPA + "2014/data_summaries/2014v2/2014neiv2_facility_process_byregions.zip",
        "2014neiv2_facility_process_byregions/process_678910.csv", "eb976230",
        "Thu, 19 Nov 2020 15:05:15 GMT", "state_and_county_fips_code",
    ),
    2017: (
        EPA + "2017/data_summaries/2017v1/2017neiJan_facility_process_byregions.zip",
        "point_678910.csv", "2c1669ab", "Mon, 08 Feb 2021 11:51:44 GMT", "fips",
    ),
    2020: (
        EPA + "2020/data_summaries/2020nei_facility_process_byregions.zip",
        "point_6.csv", "4ef7e616", "Fri, 14 Jul 2023 12:44:13 GMT", "fips code",
    ),
}
COUNTIES = dict(zip(FIPS, ("Creek", "Lincoln", "Payne"), strict=True))
COUNTY_PATTERN = re.compile(rb',"?(' + "|".join(FIPS).encode() + rb')"?,')


def extract(year, archive):
    url, member, expected_crc, last_modified, fips_field = ARCHIVES[year]
    with ZipFile(archive) as zipped:
        info = zipped.getinfo(member)
    member_crc = f"{info.CRC:08x}"
    if member_crc != expected_crc:
        raise ValueError(f"{year} NEI member changed from the archived EPA version")
    counts = Counter()
    selected = bytearray()
    # Info-ZIP supports Deflate64, used for the 2017 and 2020 process members.
    with subprocess.Popen(["unzip", "-p", str(archive), member], stdout=subprocess.PIPE) as process:
        header = process.stdout.readline()
        fields = next(csv.reader([header.decode("utf-8-sig")]))
        index = fields.index(fips_field)
        selected.extend(header)
        for line in process.stdout:
            if not COUNTY_PATTERN.search(line):
                continue
            row = next(csv.reader([line.decode("latin-1")]))
            if len(row) != len(fields):
                raise ValueError(f"{year} NEI process row has an unexpected number of fields")
            if row[index] in COUNTIES:
                counts[row[index]] += 1
                selected.extend(line)
        if process.wait():
            raise ValueError(f"Failed to extract or verify the CRC of {year} NEI member {member}")
    if set(counts) != COUNTIES.keys():
        raise ValueError(f"{year} NEI extract does not cover all three counties")
    archive_file = OUTPUT / f"nei_{year}_cushing_county_processes.csv.gz"
    data = gzip.compress(selected, mtime=0)
    write(archive_file, data)
    manifest = dict(inventory_year=year, archive_url=url, archive_member=member, member_crc32=member_crc,
                    member_uncompressed_bytes=info.file_size, source_http_last_modified_utc=last_modified,
                    rows_by_county={COUNTIES[key]: counts[key] for key in COUNTIES},
                    extract_file=archive_file.name, extract_sha256=hashlib.sha256(data).hexdigest())
    write(archive_file.with_name(f"nei_{year}_extract_manifest.json"), json.dumps(manifest, indent=2).encode() + b"\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("year", type=int, choices=ARCHIVES)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    extract(args.year, args.archive)
