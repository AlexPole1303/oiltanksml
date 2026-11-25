"""Mirror the USGS point-cloud products intersecting the Cushing research area."""
import argparse
import concurrent.futures
import json
import os
import re
import subprocess
from urllib.parse import urlsplit

from collect_public import RAW, ROOT

TARGET = Path(os.environ.get("OILTANKS_RASTER_STORE", ROOT / "data" / "rasters")) / "3dep"


def download(item):
    candidates = [item.get(key) for key in ("downloadLazURL", "downloadURL")]
    url = next((candidate for candidate in candidates
                if candidate and urlsplit(candidate).path.lower().endswith((".las", ".laz"))), None)
    if not url:
        raise ValueError("Missing LAS/LAZ download URL")
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", item["sourceId"])
    destination = TARGET / name / urlsplit(url).path.rsplit("/", 1)[1]
    if destination.is_file():
        with destination.open("rb") as file:
            if file.read(4) == b"LASF":
                return destination.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    process = subprocess.run([
        "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
        "--max-time", "480", "--continue-at", "-",
        "--output", str(temporary), url,
    ], capture_output=True, text=True)
    if process.returncode != 0:
        raise OSError(process.stderr.strip())
    with temporary.open("rb") as file:
        if file.read(4) != b"LASF":
            raise ValueError("Response did not have a LAS file signature")
    temporary.replace(destination)
    return destination.stat().st_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not TARGET.parent.is_dir():
        raise RuntimeError("Public raster storage is not mounted")
    path = RAW / "usgs" / "lidar_products.json"
    items = json.loads(path.read_bytes())["items"]
    if args.project:
        items = [item for item in items if args.project in item.get("downloadURL", "")]
    if args.limit is not None:
        items = items[:args.limit]
    log = ROOT / "data" / "logs" / "lidar_download.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"USGS 3DEP: {len(items)} items", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor, log.open("a") as stream:
        for i in range(0, len(items), args.workers):
            pending = {executor.submit(download, item): item["sourceId"]
                       for item in items[i:i + args.workers]}
            rate_limited = False
            for future in concurrent.futures.as_completed(pending):
                row = {"id": pending[future]}
                try:
                    row["bytes"] = future.result()
                except Exception as error:
                    row["error"] = str(error)
                    rate_limited |= "error: 429" in row["error"]
                print(json.dumps(row), file=stream, flush=True)
                if "error" in row:
                    print(json.dumps(row), flush=True)
            if rate_limited:
                print("USGS returned HTTP 429. Stopping the queue.", flush=True)
                break



if __name__ == "__main__":
    main()
