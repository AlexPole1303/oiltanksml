"""Get historical surface weather reports around Cushing."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from datetime import date, datetime, timezone
from urllib.parse import urlencode

from collect_public import RAW, get, run_jobs, write

END = date(2026, 11, 12)
STATIONS = ("CUH", "SWO", "TUL", "GOK", "PNC")
ADDITIONAL_STATIONS = ("CQB", "OWP", "SNL", "SRE", "RVS", "OKM")
FIELDS = ("tmpf", "dwpf", "relh", "drct", "sknt", "p01i", "vsby", "gust", "skyc1", "skyc2",
          "skyc3", "skyl1", "skyl2", "skyl3", "wxcodes", "metar")
ADDITIONAL_FIELDS = ("alti", "mslp", "skyc4", "skyl4")


def download(year, expanded=False):
    end = min(END, date(year + 1, 1, 1))
    stations = STATIONS + ADDITIONAL_STATIONS if expanded else STATIONS
    fields = FIELDS + ADDITIONAL_FIELDS if expanded else FIELDS
    parameters = ([("station", station) for station in stations] +
                  [("data", field) for field in fields] +
                  [("year1", year), ("month1", 1), ("day1", 1),
                   ("year2", end.year), ("month2", end.month), ("day2", end.day),
                   ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "yes"),
                   ("missing", "empty"), ("trace", "T" if expanded else "empty"), ("report_type", "3"),
                   ("report_type", "4")])
    url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" + urlencode(parameters)
    directory = RAW / "weather"
    if expanded:
        path = directory / f"asos_expanded_{year}.csv.gz"
        cached = path.is_file()
        data = gzip.decompress(path.read_bytes()) if cached else get(url)
    else:
        data = get(url, directory / f"asos_{year}.csv")
    if not data.startswith(b"station,valid,"):
        raise ValueError(f"Unexpected weather response for {year}")
    if expanded:
        reader = csv.DictReader(io.StringIO(data.decode()))
        if not set(fields) <= set(reader.fieldnames or ()):
            raise ValueError(f"Expanded response does not contain requested weather fields for {year}")
        counts = Counter()
        for row in reader:
            if None in row or row["station"] not in stations or not row["valid"].startswith(f"{year}-"):
                raise ValueError(f"Unexpected expanded weather row for {year}")
            counts[row["station"]] += 1
        if not counts:
            raise ValueError(f"Empty weather response for {year}")
        if not cached:
            write(path, gzip.compress(data, compresslevel=6, mtime=0))
        checksum = hashlib.sha256(data).hexdigest()
        if hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest() != checksum:
            raise ValueError(f"Weather payload failed readback verification for {year}")
        provenance = path.with_suffix(".source.json")
        if not cached or not provenance.is_file():
            write(provenance, json.dumps({"request_url": url, "year": year, "uncompressed_sha256": checksum,
                  "uncompressed_bytes": len(data), "reports_by_station": dict(sorted(counts.items())),
                  "downloaded_at_utc": datetime.now(timezone.utc).isoformat() if not cached else None}, indent=2).encode())
        return sum(counts.values())
    return data.count(b"\n") - 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", action="store_true", help="Collect 11 stations, pressure and trace precipitation")
    parser.add_argument("--year", type=int, choices=range(2017, 2027), action="append")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    years = sorted(set(args.year)) if args.year else range(2017, 2027)
    name = "weather_expanded.json" if args.expanded else "weather.json"
    if args.year:
        name = name.removesuffix(".json") + "_" + "_".join(map(str, years)) + ".json"
    run_jobs([(f"asos:{year}", download, (year, args.expanded)) for year in years], name, workers=args.workers)
