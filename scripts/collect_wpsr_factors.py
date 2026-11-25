"""Preserve the actual releases of the petroleum controls used in the backtest."""
import argparse
import csv
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from urllib.error import HTTPError
from urllib.parse import urljoin

from collect_public import RAW, ROOT, WPSR, get, write

TABLES = (1, 2, 4, 7, 9)
RELEASES = RAW / "eia" / "wpsr"


def fetch_table(releases, slug, table, paused):
    if paused.is_set():
        return None
    path = releases / f"{slug}_table{table}.csv"
    if path.is_file() and path.with_suffix(".source").is_file():
        return None
    source_table4 = (releases / f"{slug}_table4.source").read_text().strip()
    if not source_table4.startswith(f"{WPSR}{slug[:4]}/{slug}/"):
        raise ValueError(f"Unexpected source directory for archived issue {slug}")
    url = urljoin(source_table4, f"table{table}.csv")
    result = {"release_date": slug.replace("_", "-"), "table": table}
    try:
        body = get(url)
        try:
            decoded = body.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = body.decode("cp1252")
        header = next(csv.reader(io.StringIO(decoded)))
        if len(header) < 3 or not header[0].startswith("STUB"):
            raise ValueError("Response does not resemble an archived WPSR table")
        result["bytes"] = len(body)
        write(path, body)
        write(path.with_suffix(".source"), (url + "\n").encode())
    except (HTTPError, OSError, UnicodeError, ValueError, StopIteration) as error:
        result["error"] = str(error)
        if isinstance(error, HTTPError) and error.code == 429:
            paused.set()
    time.sleep(.6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-11-11")
    parser.add_argument("--directory", type=Path, default=RELEASES)
    parser.add_argument("--tables", type=int, nargs="+", choices=TABLES, default=TABLES)
    parser.add_argument("--workers", type=int, choices=(1,2,3,4), default=1)
    args = parser.parse_args()
    releases = args.directory
    names = sorted(slug for path in releases.glob("*_table4.csv")
                   if args.start.replace("-", "_") <= (slug := path.name[:10]) <= args.end.replace("-", "_"))
    stem = "wpsr_factors" if releases.resolve() == RELEASES else f"wpsr_factors_{releases.name}"
    log = ROOT / "data" / "logs" / f"{stem}.jsonl"
    paused = Event()
    print(f"{len(names)} releases, tables {args.tables}", flush=True)
    jobs = ((slug,table) for slug in names for table in sorted(set(args.tables)))
    with log.open("a") as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(lambda pair:fetch_table(releases,*pair,paused),jobs):
            if result:
                print(json.dumps(result), file=stream, flush=True)
    if paused.is_set():
        raise SystemExit("EIA rate limit reached. Saved verified responses for a later retry.")


if __name__ == "__main__":
    main()
