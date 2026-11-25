"""Keep the original WIKI daily rows for the study's historical equity symbols."""
import csv
import hashlib
import json
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

from collect_public import RAW, ROOT, TLS, write

DATASET = "https://www.kaggle.com/datasets/marketneutral/quandl-wiki-prices-us-equites"
DOWNLOAD = "https://www.kaggle.com/api/v1/datasets/download/marketneutral/quandl-wiki-prices-us-equites"
STORE = RAW / "equity_reference" / "quandl_wiki"
MEMBER = "WIKI_PRICES.csv"
FIRST = b"2016-01-04"
HEADER = b"ticker,date,open,high,low,close,volume,ex-dividend,split_ratio,adj_open,adj_high,adj_low,adj_close,adj_volume"


def collect(symbols):
    extracts = {symbol.encode(): [] for symbol in symbols}
    digest = hashlib.sha256()
    request = Request(DOWNLOAD, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/zip"})
    with tempfile.TemporaryFile() as archive:
        with urlopen(request, timeout=90, context=TLS) as source:
            if source.status != 200 or source.headers.get_content_type() != "application/zip":
                raise ValueError("WIKI download returned an unexpected response")
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                archive.write(chunk)
                digest.update(chunk)
        size = archive.tell()
        with ZipFile(archive) as dataset, dataset.open(MEMBER) as prices:
            header = prices.readline()
            if header.strip().lower() != HEADER:
                raise ValueError(f"Unexpected WIKI column definitions: {header!r}")
            for line in prices:
                symbol, day, _ = line.split(b",", 2)
                if symbol in extracts and day >= FIRST:
                    extracts[symbol].append(line)
            member = dataset.getinfo(MEMBER)
    results = {}
    for symbol, lines in extracts.items():
        name = symbol.decode()
        path = STORE / f"{name}.csv"
        ordered = sorted(lines)
        payload = header + b"".join(ordered)
        dates = [line.split(b",", 2)[1].decode() for line in ordered]
        if dates != sorted(set(dates)):
            raise ValueError(f"WIKI supplied duplicate dates for {name}")
        write(path, payload)
        results[name] = dict(rows=len(dates), first=min(dates, default=None), last=max(dates, default=None),
                             file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(payload).hexdigest())
    manifest = dict(dataset_url=DATASET, download_url=DOWNLOAD, archive_size_bytes=size,
                    archive_sha256=digest.hexdigest(), member=MEMBER, member_crc32=f"{member.CRC:08x}",
                    member_size_bytes=member.file_size, first_requested=FIRST.decode(), symbols=results)
    write(STORE / "manifest.json", json.dumps(manifest, indent=2).encode() + b"\n")
    return manifest


if __name__ == "__main__":
    with (ROOT / "data/reference/historical_equity_listing_windows.csv").open() as source:
        symbols = {row["symbol"] for row in csv.DictReader(source)} | {"TLP"}
    print(json.dumps(collect(sorted(symbols)), indent=2))
