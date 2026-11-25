"""Download Cushing source records and preserve provider responses."""
import argparse
import concurrent.futures
import csv
import datetime as dt
import io
import json
import os
import re
import ssl
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
BBOX = "-96.92,35.86,-96.63,36.03"
EARTH_SEARCH = "https://earth-search.aws.element84.com/v1/search"
TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
WPSR = "https://www.eia.gov/petroleum/supply/weekly/archive/"
EIA_XLS = "https://www.eia.gov/dnav/pet/hist_xls/"
TLS = ssl.create_default_context()
if Path("/etc/ssl/cert.pem").is_file():
    TLS.load_verify_locations("/etc/ssl/cert.pem")


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_csv(path, fields, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(fields)
    writer.writerows(rows)
    write(path, output.getvalue().encode())


def get(url, path=None, refresh=False):
    if path is not None and path.exists() and not refresh:
        return path.read_bytes()
    request = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=35, context=TLS) as response:
                content = response.read()
            if path is not None:
                write(path, content)
            return content
        except (OSError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code in {400, 401, 403, 404, 406}:
                raise
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def stac(collection, start, end, bbox=BBOX):
    url = EARTH_SEARCH + "?" + urlencode({
        "collections": collection, "bbox": bbox, "limit": 100,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
    })
    pages = []
    features = {}
    while url:
        if url in pages:
            raise ValueError(f"Repeated STAC page for {collection} {start}")
        pages.append(url)
        payload = json.loads(get(url))
        for feature in payload["features"]:
            features[feature["id"]] = feature
        url = next((link["href"] for link in payload["links"] if link["rel"] == "next"), None)
    return {"type": "FeatureCollection", "features": list(features.values()), "page_urls": pages}


def save_stac(collection, start, end, refresh, bbox=BBOX, root=RAW / "stac"):
    path = root / collection / f"{start[:4]}.geojson"
    if path.exists() and not refresh:
        return len(json.loads(path.read_bytes())["features"])
    payload = stac(collection, start, end, bbox)
    write(path, json.dumps(payload, separators=(",", ":")).encode())
    return len(payload["features"])


def wpsr_tables(issue_path, output):
    slug = issue_path.split("/")[-2]
    directory = urljoin(WPSR, "/".join(issue_path.split("/")[:-1]) + "/")
    path = output / f"{slug}_table4.csv"
    expected = urljoin(directory, "csv/table4.csv")
    try:
        content = get(expected)
    except HTTPError as error:
        if error.code != 404:
            raise
        issue_url = urljoin(WPSR, issue_path)
        issue = get(issue_url, output / f"{slug}_issue.html").decode(errors="replace")
        match = re.search(r'href=["\']([^"\']*table4[^"\']*\.csv(?:\?[^"\']*)?)', issue, re.I)
        if not match:
            raise ValueError("No archived Table 4 CSV link")
        expected = urljoin(issue_url, match.group(1))
        if not expected.startswith(directory):
            raise ValueError("CSV link points outside the archived issue")
        content = get(expected)
    if b"Cushing" not in content or b"<html" in content[:500].lower():
        raise ValueError("Invalid archived Table 4 response")
    write(path, content)
    write(path.with_suffix(".source"), (expected + "\n").encode())
    return len(content)


def tnm_lidar():
    url = TNM + "?" + urlencode({"bbox": BBOX, "datasets": "Lidar Point Cloud (LPC)", "max": 1000})
    result = json.loads(get(url))
    if result.get("total") != len(result["items"]):
        raise ValueError("USGS response is incomplete")
    write(RAW / "usgs" / "lidar_products.json", json.dumps(result).encode())
    return len(result["items"])


def wpsr_jobs(as_of, start="2016-01-01", output=RAW / "eia" / "wpsr"):
    index = get(WPSR, RAW / "eia" / "wpsr_index.html", refresh=True).decode(errors="replace")
    issues = sorted(set(re.findall(r'(/petroleum/supply/weekly/archive/20\d\d/20\d\d_\d\d_\d\d/wpsr_20\d\d_\d\d_\d\d\.php)', index)))
    for issue in issues:
        slug = issue.split("/")[-2]
        if start.replace("-", "_") <= slug <= as_of.replace("-", "_"):
            path = output / f"{slug}_table4.csv"
            if not path.is_file() or not path.with_suffix(".source").is_file():
                yield (f"wpsr:{slug}", wpsr_tables, (issue, output))


def fetch_file(name, url):
    path = RAW / name
    data = get(url, path)
    return len(data)


SERIES = {
    "cushing_weekly": "W_EPC0_SAX_YCUOK_MBBLw.xls",
    "us_commercial_crude_weekly": "WCESTUS1w.xls",
    "padd2_commercial_crude_weekly": "W_EPC0_SAX_R20_MBBLw.xls",
    "padd3_commercial_crude_weekly": "W_EPC0_SAX_R30_MBBLw.xls",
    "us_spr_crude_weekly": "WCSSTUS1w.xls",
    "us_gasoline_stocks_weekly": "WGTSTUS1w.xls",
    "us_distillate_stocks_weekly": "WDISTUS1w.xls",
    "us_gasoline_supplied_weekly": "WGFUPUS2w.xls",
    "us_distillate_supplied_weekly": "WDIUPUS2w.xls",
    "us_production_weekly": "WCRFPUS2w.xls",
    "us_crude_imports_weekly": "WCRIMUS2w.xls",
    "us_crude_exports_weekly": "WCREXUS2w.xls",
    "us_refinery_inputs_weekly": "WCRRIUS2w.xls",
    "padd2_refinery_utilization_weekly": "W_NA_YUP_R20_PERw.xls",
    "wti_spot_daily": "RWTCd.xls",
    "brent_spot_daily": "RBRTEd.xls",
    **{f"nymex_contract_{i}_daily": f"RCLC{i}d.xls" for i in range(1, 5)},
}
EXTRA = {
    "eia/release_schedule.html": "https://www.eia.gov/petroleum/supply/weekly/schedule.php",
    **{f"eia/press{number}.html": f"https://www.eia.gov/pressroom/releases/press{number}.php"
       for number in (506, 507, 545)},
    "eia/storage_capacity.html": "https://www.eia.gov/petroleum/storagecapacity/",
    "eia/storagecapacity.xlsx": "https://www.eia.gov/petroleum/storagecapacity/storagecapacity.xlsx",
    "eia/crudeoilstorage.xlsx": "https://www.eia.gov/petroleum/storagecapacity/crudeoilstorage.xlsx",
    **{f"factors/cftc_disaggregated_{year}.zip": f"https://www.cftc.gov/files/dea/history/fut_disagg_txt_{year}.zip"
       for year in range(2016, 2027)},
    "factors/cftc_catchup_2019.html": "https://www.cftc.gov/PressRoom/PressReleases/7864-19",
    "factors/cftc_catchup_2025.html": "https://www.cftc.gov/PressRoom/PressReleases/9147-25",
    "factors/cftc_release_schedule.html": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule/index.htm",
    "factors/cftc_special_announcements.html": "https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm",
    **{f"factors/{series}.csv": f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
       for series in ("DEXCAUS", "DTB3", "DGS10", "VIXCLS", "OVXCLS", "BAMLH0A0HYM2")},
    "factors/OVX_cboe.csv": "https://cdn.cboe.com/api/global/us_indices/daily_prices/OVX_History.csv",
    "factors/VIX_cboe.csv": "https://cdn-api.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "factors/fama_french_daily.zip": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "documents/enbridge_cushing.html": "https://www.enbridge.com/about-us/liquids-pipelines/crude-and-liquids-contract-terminals",
    "documents/energy_transfer_assets.html": "https://www.energytransfer.com/crude-oil/",
    "documents/energy_transfer_history.html": "https://www.energytransfer.com/history/",
    "documents/oneok_assets.html": "https://www.oneok.com/customers/rpco/crude-oil/crude-oil",
    "documents/oneok_magellan_faq.html": "https://ir.oneok.com/recent-acquisitions/mmp-acquisition-faq",
    "documents/keyera_wildhorse_sale.html": "https://www.keyera.com/news-and-stories/news-releases/keyera-announces-2025-fourth-quarter-and-year-end-results/",
    "documents/ok_deq_permits.html": "https://applications.deq.ok.gov/PermitsPublicReview/viewpermits.aspx",
    "documents/south_bow_marketlink.html": "https://www.southbow.com/marketlink-pipeline-system-customer-information/",
    "documents/cp_energy_tanks.html": "https://www.cpenergy.com/facilities/tanks-terminals",
    "documents/transmontaigne_assets.html": "https://www.transmontaignepartners.com/operations/",
}


def run_jobs(jobs, filename, workers):
    entries = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(function, *args): name for name, function, args in jobs}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                entry = {"dataset": name, "size_or_count": future.result()}
            except Exception as error:
                entry = {"dataset": name, "error": str(error)}
            entries.append(entry)
            if not name.startswith("wpsr:") or "error" in entry:
                print(json.dumps(entry), flush=True)
    write(ROOT / "data" / "logs" / filename, json.dumps(sorted(entries, key=lambda e: e["dataset"]), indent=2).encode())
    print(f"{filename}: {len(entries)} jobs, {sum('error' in entry for entry in entries)} errors", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["catalogs", "sentinel2-supplemental", "series", "wpsr"])
    parser.add_argument("--as-of", default="2026-11-11")
    parser.add_argument("--wpsr-start", default="2016-01-01")
    parser.add_argument("--wpsr-output", type=Path, default=RAW / "eia" / "wpsr")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    as_of = dt.date.fromisoformat(args.as_of)
    jobs = []
    if args.group == "catalogs":
        for collection in ("sentinel-2-c1-l2a", "sentinel-2-l2a", "sentinel-2-l1c", "sentinel-1-grd"):
            for year in range(2017, as_of.year + 1):
                end = min(as_of, dt.date(year, 12, 31)).isoformat()
                jobs.append((f"{collection}:{year}", save_stac, (collection, f"{year}-01-01", end, args.refresh)))
        jobs += [("naip:all", save_stac, ("naip", "2003-01-01", args.as_of, args.refresh)),
                 ("usgs:lidar", tnm_lidar, ())]
    elif args.group == "sentinel2-supplemental":
        from shapely.geometry import shape
        from shapely.ops import unary_union

        layer = ROOT / "data" / "derived" / "supplemental_tank_candidates.geojson"
        features = json.loads(layer.read_bytes())["features"]
        west, south, east, north = unary_union([shape(item["geometry"]) for item in features]).bounds
        bbox = ",".join(f"{value:.6f}" for value in (west - .002, south - .002, east + .002, north + .002))
        for collection in ("sentinel-2-c1-l2a", "sentinel-2-l2a", "sentinel-2-l1c"):
            for year in range(2017, as_of.year + 1):
                end = min(as_of, dt.date(year, 12, 31)).isoformat()
                jobs.append((f"{collection}:{year}", save_stac,
                             (collection, f"{year}-01-01", end, args.refresh, bbox, RAW / "stac_supplemental")))
    elif args.group == "series":
        sources = {f"eia/{key}.xls": EIA_XLS + value for key, value in SERIES.items()} | EXTRA
        write(RAW / "source_urls.json", json.dumps(sources, indent=2).encode())
        jobs = [(name, fetch_file, (name, url)) for name, url in sources.items()]
    else:
        jobs = list(wpsr_jobs(args.as_of, dt.date.fromisoformat(args.wpsr_start).isoformat(), args.wpsr_output))
    log_name = args.group
    if args.group == "wpsr" and args.wpsr_output != RAW / "eia" / "wpsr":
        log_name = f"wpsr_{args.wpsr_start}_{args.as_of}"
    run_jobs(jobs, log_name + ".json", workers=4 if args.group == "wpsr" else 6)


if __name__ == "__main__":
    main()
