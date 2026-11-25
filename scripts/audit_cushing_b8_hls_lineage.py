"""Find independently dated NASA HLS products naming the exact Landsat Level-1 inputs."""
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from audit_cushing_b8_archive import STEM as ARCHIVE
from build_landsat_c2_holdout_gap_inventory import parse
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_hls_landsat_gaps import day_catalog
from collect_landsat import LANDSAT
from collect_public import ROOT, get, write

STEM = "cushing_b8_hls_lineage"
STORE = LANDSAT / STEM
COLLECTION_REFS = ({"ShortName":"HLSL30","Version":"2.0"},
                   {"EntryTitle":"HLS Landsat Operational Land Imager Surface Reflectance and TOA Brightness Daily Global 30m v2.0"})


def granule(item):
    url, = [link["href"] for link in item["links"] if link["rel"]=="via" and link["href"].endswith(".umm_json")]
    match = re.fullmatch(r"https://cmr\.earthdata\.nasa\.gov/search/concepts/(G\d+-LPCLOUD)\.umm_json",url)
    if not match:
        raise ValueError(f"Unexpected CMR metadata route for {item['id']}")
    path = STORE / f"{match[1]}.json"
    data = json.loads(get(url,path))
    if data["GranuleUR"]!=item["id"] or data["CollectionReference"] not in COLLECTION_REFS:
        raise ValueError(f"CMR granule metadata differs for {item['id']}")
    lineage, = [a["Values"] for a in data["AdditionalAttributes"] if a["Name"]=="LANDSAT_PRODUCT_ID"]
    if not lineage or len(lineage)!=len(set(lineage)) or not all(re.fullmatch(
        r"LC0[89]_L1TP_\d{6}_\d{8}_\d{8}_02_(?:T1|T2|RT)",p) for p in lineage
    ):
        raise ValueError(f"Invalid Landsat lineage for {item['id']}")
    dates = data["ProviderDates"]
    insert, = [parse(d["Date"]) for d in dates if d["Type"]=="Insert"]
    update, = [parse(d["Date"]) for d in dates if d["Type"]=="Update"]
    production = parse(data["DataGranule"]["ProductionDateTime"])
    if not production<=insert<=update:
        raise ValueError(f"CMR granule dates are out of order for {item['id']}")
    return lineage, dict(hls_granule_id=item["id"],cmr_concept_id=match[1],hls_production_at_utc=production.isoformat(),
        cmr_insert_at_utc=insert.isoformat(),cmr_update_at_utc=update.isoformat(),lineage_latest_clock_utc=update.isoformat(),
        source_landsat_products_json=json.dumps(lineage),cmr_metadata_url=url,cmr_metadata_path=str(path.relative_to(ROOT)))


def main():
    queue = {r["product_id"]:r for r in read_csv("cushing_b8_download_manifest")}
    dates = sorted({r["captured_at_utc"][:10] for r in queue.values()})
    options = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for day,_,items in pool.map(day_catalog,((d,"cmr") for d in dates)):
            for item in items:
                if item["properties"]["datetime"][:10]!=day or item["id"] in options:
                    raise ValueError(f"Unexpected or duplicated CMR HLS granule for {day}")
                options[item["id"]] = item
        lineage_rows = []
        for i,(products,record) in enumerate(pool.map(granule,options.values()),1):
            for product in sorted(set(products)&queue.keys()):
                capture = parse(queue[product]["captured_at_utc"])
                if capture > parse(record["hls_production_at_utc"]):
                    raise ValueError(f"HLS production precedes its Landsat input capture for {product}")
                lineage_rows.append(dict(product_id=product,landsat_capture_at_utc=capture.isoformat(),**record))
            if i%40==0 or i==len(options):
                print(f"Checked exact source metadata in {i}/{len(options)} HLS granules",flush=True)
    by_product = defaultdict(list)
    for r in lineage_rows:
        by_product[r["product_id"]].append(r)
    earliest = {p:min(group,key=lambda r:(parse(r["lineage_latest_clock_utc"]),r["hls_granule_id"]))
                for p,group in by_product.items()}
    context = {}
    for r in read_csv("landsat_c2_context_registration_pairs"):
        key = tuple(r[k] for k in ("week_or_control_day","candidate_id","earlier_version_id","later_version_id",
                                    "earlier_source","later_source"))
        if key in context:
            raise ValueError(f"Duplicate exterior context for {key}")
        context[key] = r["registration_status"]=="checked_offset" and r["red_context_before_cutoff"]=="True"
    pairs = []
    for r in read_csv(f"{ARCHIVE}_pairs"):
        cutoff = parse(r["forecast_cutoff_utc"])
        files = json.loads(r["b8_files_json"])
        required = [earliest.get(f.removesuffix("_B8.TIF")) for f in files]
        stamps = [parse(a["lineage_latest_clock_utc"]) for a in required if a]
        all_inputs = len(stamps)==len(required)
        latest = max(stamps) if all_inputs else None
        hls = latest is not None and latest<cutoff
        key = tuple(r[k] for k in ("week_ending","candidate_id","previous_version_id","current_version_id",
                                    "previous_source_file","current_source_file"))
        checked = context.pop(key)
        pairs.append(dict(r, all_b8_products_named_in_hls=all_inputs,
            hls_lineage_cmr_concepts_json=json.dumps([a["cmr_concept_id"] if a else None for a in required]),
            latest_hls_lineage_at_utc=latest.isoformat() if latest else None,
            all_hls_lineage_before_cutoff=hls,red_context_checked_before_cutoff=checked,
            either_catalog_clock_and_red_context_before_cutoff=checked and (hls or r["cdse_catalog_all_clocks_before_cutoff"]=="True")))
    if context:
        raise ValueError("Exterior context and B8 provenance pairs do not match")
    passed = [r for r in pairs if r["all_hls_lineage_before_cutoff"]]
    cdse = [r for r in pairs if r["cdse_catalog_all_clocks_before_cutoff"]=="True"]
    either = [r for r in pairs if r["all_hls_lineage_before_cutoff"] or r["cdse_catalog_all_clocks_before_cutoff"]=="True"]
    by_week = defaultdict(list)
    for r in pairs:
        by_week[r["week_ending"]].append(r)
    weeks = []
    for row in read_csv(f"{ARCHIVE}_weeks"):
        group = by_week[row["week_ending"]]
        ready = [r for r in group if r["all_hls_lineage_before_cutoff"]]
        evidence = [r for r in group if r["all_hls_lineage_before_cutoff"] or r["cdse_catalog_all_clocks_before_cutoff"]=="True"]
        both = [r for r in evidence if r["either_catalog_clock_and_red_context_before_cutoff"]]
        if len(group)!=int(row["candidate_pairs"]):
            raise ValueError(f"HLS pair counts differ from archive audit for {row['week_ending']}")
        weeks.append(dict(row,hls_pairs_before_cutoff=len(ready),
            hls_tanks_before_cutoff=len({r["candidate_id"] for r in ready}),
            either_catalog_pairs_before_cutoff=len(evidence),
            either_catalog_tanks_before_cutoff=len({r["candidate_id"] for r in evidence}),
            either_catalog_and_red_context_pairs_before_cutoff=len(both),
            either_catalog_and_red_context_tanks_before_cutoff=len({r["candidate_id"] for r in both})))
    for label,rows in (("sources",lineage_rows),("pairs",pairs),("weeks",weeks)):
        export(f"{STEM}_{label}",rows)
    report = dict(checked_days=len(dates),hls_catalog_granules=len(options),landsat_originals_with_exact_hls_lineage=len(earliest),
        matching_landsat_hls_records=len(lineage_rows),all_candidate_pairs=len(pairs),
        pairs_with_all_hls_lineage_before_cutoff=len(passed),
        tank_weeks_with_all_hls_lineage_before_cutoff=sum(r["hls_tanks_before_cutoff"] for r in weeks),
        weeks_with_all_hls_lineage_before_cutoff=sum(r["hls_pairs_before_cutoff"]>0 for r in weeks),
        pairs_with_all_cdse_bundle_clocks_before_cutoff=len(cdse),pairs_with_either_proxy_before_cutoff=len(either),
        tank_weeks_with_either_proxy_before_cutoff=sum(r["either_catalog_tanks_before_cutoff"] for r in weeks),
        weeks_with_either_proxy_before_cutoff=sum(r["either_catalog_pairs_before_cutoff"]>0 for r in weeks),
        pairs_with_either_proxy_and_checked_red_context=sum(r["either_catalog_and_red_context_pairs_before_cutoff"] for r in weeks),
        tank_weeks_with_either_proxy_and_checked_red_context=sum(r["either_catalog_and_red_context_tanks_before_cutoff"] for r in weeks),
        weeks_with_either_proxy_and_checked_red_context=sum(r["either_catalog_and_red_context_pairs_before_cutoff"]>0 for r in weeks),
        verified_historical_b8_file_pairs=0,scope="HLS metadata shows that NASA used the exact Level-1 source in a dated "
            "derivative. HLS does not establish when the individual B8 file could be downloaded.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
