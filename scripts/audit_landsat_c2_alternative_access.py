"""Record alternative archive coverage for exact priority Landsat B8 versions."""
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from urllib.error import HTTPError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from build_landsat_c2_holdout_gap_inventory import STEM as GAP
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_imagery import PC
from collect_landsat import LANDSAT
from collect_public import TLS, get, write

STEM = f"{GAP}_alternative_access"
STORE = LANDSAT / "collection2_alternative_access"
CDSE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


def copernicus(rows, *, store=STORE):
    files = {}
    for i in range(0,len(rows),8):
        group = rows[i:i+8]
        filter_value = " or ".join(f"Name eq '{r['product_id']}'" for r in group)
        url = CDSE + "?" + urlencode({"$filter":filter_value,"$top":20,"$count":"true"})
        path = store / f"cdse_{i//8:02d}.json"
        snapshot = json.loads(get(url,path,refresh=True))
        if snapshot.get("@odata.nextLink") or len(snapshot["value"])!=snapshot["@odata.count"]:
            raise ValueError("Incomplete Copernicus product results")
        for item in snapshot["value"]:
            if item["Name"] not in {r["product_id"] for r in group} or item["Name"] in files:
                raise ValueError(f"Unexpected or duplicated Landsat identity in Copernicus: {item['Name']}")
            files[item["Name"]] = item
    return files


def probe(url):
    try:
        request = Request(url,headers={"Range":"bytes=0-31","User-Agent":"Mozilla/5.0"})
        with urlopen(request,context=TLS,timeout=20) as response:
            status,host,body = response.status,urlsplit(response.url).hostname,response.read(32)
    except HTTPError as error:
        status,host,body = error.code,urlsplit(error.url).hostname,error.read(32)
    tiff = body[:4] in (b"II*\x00",b"MM\x00*",b"II+\x00",b"MM\x00+")
    return dict(url=url,http_status=status,response_host=host,tiff_signature=tiff)


def main():
    rows = read_csv(f"{GAP}_download_manifest")
    pc = PC+"/api/stac/v1/collections/landsat-c2-l1"
    with ThreadPoolExecutor(max_workers=2) as pool:
        cdse_job = pool.submit(copernicus, rows)
        catalog = json.loads(get(pc, STORE/"pc_level1_collection.json", refresh=True))
        cdse = cdse_job.result()
    collection_end = catalog["extent"]["temporal"]["interval"][0][1]
    if catalog["id"]!="landsat-c2-l1" or not collection_end:
        raise ValueError("The public Planetary Computer Level-1 catalog has no verified temporal end")
    output = []
    for r in rows:
        product = r["product_id"]
        data = cdse.get(product)
        published = data["PublicationDate"] if data else None
        cutoffs = [datetime.fromisoformat(e["forecast_cutoff_utc"]) for e in json.loads(r["week_endpoints_json"])]
        output.append(dict(product_id=product,platform=r["platform"],captured_at_utc=r["captured_at_utc"],
                           pc_level1_catalog_end_utc=collection_end,
                           pc_level1_catalog_end_before_capture=datetime.fromisoformat(collection_end)<datetime.fromisoformat(r["captured_at_utc"]),
                           cdse_product_id=data["Id"] if data else None,cdse_s3_path=data["S3Path"] if data else None,
                           cdse_publication_at_utc=published,cdse_archive_product_bytes=data["ContentLength"] if data else None,
                           cdse_present_before_any_assigned_gap_cutoff=(
                               any(datetime.fromisoformat(published)<t for t in cutoffs) if published and cutoffs else None)))
    examples = [next(r for r in rows if r["platform"]==p) for p in ("LANDSAT_8","LANDSAT_9")]
    checks = []
    for r in examples:
        item = cdse.get(r["product_id"])
        if item:
            url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({item['Id']})/$value"
            checks.append(dict(product_id=r["product_id"],route="cdse_direct_download",**probe(url)))
        path = urlsplit(r["pan_url"]).path.removeprefix("/data/")
        url = f"https://storage.googleapis.com/gcp-public-data-landsat/{path}"
        checks.append(dict(product_id=r["product_id"],route="legacy_public_gcs_exact_path",**probe(url)))
        url = f"https://storage.googleapis.com/gee-public-data-landsat/{path}"
        checks.append(dict(product_id=r["product_id"],route="collection2_requester_pays_gcs",**probe(url)))
    export(STEM,output)
    export(f"{STEM}_probes",checks)
    summary = dict(priority_products=len(rows),pc_level1_catalog_end_utc=collection_end,cdse_exact_matches=len(cdse),
                   cdse_matches_before_any_assigned_gap_cutoff=sum(
                       r["cdse_present_before_any_assigned_gap_cutoff"] is True for r in output),
                   public_download_probes=checks,
                   interpretation="Copernicus publication dates refer to the mirrored catalog. "
                                  "They do not determine first availability in the USGS archive.")
    payload=json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
