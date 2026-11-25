"""Extract Landsat 8/9 Level-1 rows and verify the exact Cushing B8 download queue."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from build_landsat_c2_holdout_gap_inventory import SIDES, parse
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_landsat import LANDSAT
from collect_public import ROOT, write, write_csv
from stage_landsat_c2_b8_inputs import expected_sha512

STEM = "cushing_usgs_b8_metadata"
BULK_URL = "https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_OT_C2_L1.csv.gz"
PRODUCT = "Landsat Product Identifier L1"
TRACKS = {(26,35),(27,35),(28,35)}
TRACK_KEYS = {(str(path),str(row)) for path,row in TRACKS}
FIRST_YEAR = 2017


def archive_receipt(path, headers):
    accepted = {"etag","last-modified","content-length","content-type"}
    facts = {}
    for line in headers.read_text().splitlines():
        if line.startswith("HTTP/"):
            facts = {"http_status":int(line.split()[1])}
        field,sep,value = line.partition(":")
        if sep and field.lower() in accepted:
            facts[field.lower()] = value.strip()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle,"sha256").hexdigest()
    if facts["http_status"]!=200 or path.stat().st_size!=int(facts["content-length"]) or "gzip" not in facts["content-type"]:
        raise ValueError("USGS bulk metadata file does not match the download response")
    return dict(source_url=BULK_URL,local_archive=str(path.resolve()),bytes=path.stat().st_size,sha256=digest,
                etag=facts["etag"],http_last_modified_utc=parsedate_to_datetime(facts["last-modified"]).isoformat())


def regional_records(path, originals):
    records, exact, count = [],{},0
    with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as stream:
        reader = csv.reader(stream)
        fields = next(reader)
        name,path_id,row_id,acquired,day = [fields.index(k) for k in (PRODUCT,"WRS Path","WRS Row","Date Acquired","Day/Night Indicator")]
        for number,values in enumerate(reader,2):
            count += 1
            region = (values[path_id].lstrip("0"),values[row_id].lstrip("0")) in TRACK_KEYS
            if values[name] not in originals and not (region and values[acquired][:4]>=str(FIRST_YEAR) and values[day]=="DAY"):
                continue
            if len(values)!=len(fields):
                raise ValueError(f"Malformed source row {number}")
            row = dict(zip(fields,values,strict=True))
            records.append((number,row))
            if row[PRODUCT] in originals:
                if row[PRODUCT] in exact:
                    raise ValueError(f"Repeated original in the USGS index at row {number}")
                exact[row[PRODUCT]] = number,row
    return count,fields,records,exact


def stac_records():
    items = {}
    for file in sorted((LANDSAT/"usgs_catalog").glob("*.geojson")):
        if not file.stem.isdecimal() or int(file.stem)<FIRST_YEAR:
            continue
        for item in json.loads(file.read_bytes())["features"]:
            p = item["properties"]
            track = int(p["landsat:wrs_path"]),int(p["landsat:wrs_row"])
            if p["view:sun_elevation"]<=0 or track not in TRACKS:
                continue
            if item["id"] in items:
                raise ValueError(f"Repeated STAC item {item['id']}")
            items[item["id"]] = item,str(file.relative_to(ROOT))
    return items


def metadata_rows(queue, bulk, items):
    for q in queue:
        name = q["product_id"]
        item,source = items[name]
        p,pan = item["properties"],item["assets"]["pan"]
        stamp = parse(q["captured_at_utc"])
        num,r = bulk[name]
        day = date.fromisoformat(r["Date Acquired"].replace("/","-"))
        generated = date.fromisoformat(r["Date Product Generated L1"].replace("/","-"))
        start,stop = (parse(r[k]).replace(tzinfo=timezone.utc) for k in ("Start Time","Stop Time"))
        if (r[PRODUCT]!=name or r["Display ID"]!=name or r["Ordering ID"]!=r["Landsat Scene Identifier"]
            or r["Landsat Scene Identifier"]!=p["landsat:scene_id"] or stamp!=parse(p["datetime"])
            or not start<=stamp<=stop or stamp.date()!=day or day.strftime("%Y%m%d")!=name.split("_")[3]
            or generated.strftime("%Y%m%d")!=name.split("_")[4]
            or (int(r["Satellite"]),int(r["WRS Path"]),int(r["WRS Row"]))!=(int(name[2:4]),int(p["landsat:wrs_path"]),int(p["landsat:wrs_row"]))
            or r["Collection Category"]!=name.split("_")[-1] or float(r["Grid Cell Size Panchromatic"])!=15
            or [int(r[k]) for k in ("Panchromatic Lines","Panchromatic Samples")]!=pan["proj:shape"]
            or pan["href"]!=q["b8_download_url"] or expected_sha512(pan)!=q["expected_original_b8_sha512"]):
            raise ValueError(f"Bulk metadata, STAC and queue differ for {name}")
        yield dict(product_id=name,scene_id=r["Landsat Scene Identifier"],earthexplorer_ordering_id=r["Ordering ID"],
            download_priority=int(q["download_priority"]),purpose=q["purpose"],
            captured_at_utc=stamp.isoformat(),bulk_start_at_utc=start.isoformat(),bulk_stop_at_utc=stop.isoformat(),
            bulk_level1_generation_day=generated.isoformat(),bulk_csv_record_number=num,
            stac_metadata_created_at_utc=p.get("created"),stac_metadata_updated_at_utc=p.get("updated"),
            stac_asset_created_at_utc=pan.get("created"),stac_asset_updated_at_utc=pan.get("updated"),
            stac_asset_published_at_utc=pan.get("published"),stac_source=source,stac_item_url=next(
                link["href"] for link in item["links"] if link["rel"]=="self"),
            landsatlook_b8_url=pan["href"],usgs_s3_b8_url=pan["alternate"]["s3"]["href"],
            cdse_b8_url=q["cdse_b8_download_url"],b8_filename=q["original_b8_filename"],
            b8_expected_sha512=q["expected_original_b8_sha512"],b8_mirror_bytes=int(q["cdse_b8_bytes"]),
            original_b8_file_clock_verified=False)


def pair_clocks(scenes):
    for r in read_csv("landsat_c2_b8_weekly_pairs"):
        products = [scenes[r[f"{s}_version_id"]] for s in SIDES if r[f"{s}_sensor"]=="B8"]
        cutoff = parse(r["forecast_cutoff_utc"])
        support = max(parse(r[f"{s}_support_available_at_utc"]) for s in SIDES)
        current_stac = max(parse(p["stac_metadata_created_at_utc"]) for p in products)
        stac_last_updated = max(parse(p["stac_metadata_updated_at_utc"]) for p in products)
        fields = ("week_ending","release_date","forecast_cutoff_utc","candidate_id","route","previous_version_id","current_version_id",
                  "previous_source_file","current_source_file")
        yield {k:r[k] for k in fields} | dict(latest_required_stac_metadata_created_at_utc=current_stac.isoformat(),
            latest_required_stac_metadata_updated_at_utc=stac_last_updated.isoformat(),
            metadata_created_and_support_before_cutoff=max(support,current_stac)<cutoff,
            metadata_last_updated_and_support_before_cutoff=max(support,stac_last_updated)<cutoff,
            original_b8_file_clock_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--headers",type=Path,required=True)
    args = parser.parse_args()
    receipt = archive_receipt(args.archive,args.headers)
    queue = sorted(read_csv("landsat_esri_b8_original_priority"),key=lambda r:int(r["download_priority"]))
    names = {r["product_id"] for r in queue}
    count,fields,regional,exact = regional_records(args.archive,names)
    items = stac_records()
    if len(queue)!=len(names) or exact.keys()!=names or not names<=items.keys():
        raise ValueError(f"Exact queue and USGS indexes differ. Missing in bulk {sorted(names-exact.keys())}")
    scenes = list(metadata_rows(queue,exact,items))
    pairs = list(pair_clocks({r["product_id"]:r for r in scenes}))
    grouped = defaultdict(list)
    for r in pairs:
        grouped[r["week_ending"]].append(r)
    weeks = []
    for r in read_csv("landsat_c2_b8_weekly_weeks"):
        week = r["week_ending"]
        group = grouped.pop(week,[])
        if len(group)!=int(r["exact_pair_candidates"]) or any(x["forecast_cutoff_utc"]!=r["forecast_cutoff_utc"] for x in group):
            raise ValueError(f"USGS metadata clock calendar differs for {week}")
        created = [x for x in group if x["metadata_created_and_support_before_cutoff"]]
        updated = [x for x in group if x["metadata_last_updated_and_support_before_cutoff"]]
        weeks.append({k:r[k] for k in ("week_ending","release_date","forecast_cutoff_utc")} | dict(
            exact_pair_candidates=len(group),pairs_with_stac_created_before_cutoff=len(created),
            tanks_with_stac_created_before_cutoff=len({x["candidate_id"] for x in created}),
            pairs_with_stac_last_updated_before_cutoff=len(updated),
            tanks_with_stac_last_updated_before_cutoff=len({x["candidate_id"] for x in updated}),original_b8_file_clocks_verified=0))
    if grouped:
        raise ValueError("Unexpected weekly candidate rows")
    export(f"{STEM}_scenes",scenes)
    export(f"{STEM}_pairs",pairs)
    export(f"{STEM}_weeks",weeks)
    write_csv(DERIVED/f"{STEM}_regional_catalog.csv",("source_record_number","in_cushing_stac_snapshot",*fields),
              ([i,r[PRODUCT] in items,*(r[k] for k in fields)] for i,r in regional))
    write(DERIVED/f"{STEM}_earthexplorer_product_ids.txt",("\n".join(r["product_id"] for r in scenes)+"\n").encode())
    report = dict(receipt,bulk_rows_examined=count,regional_daytime_rows=len(regional),
        regional_tracks=sorted(TRACKS),first_regional_year=FIRST_YEAR,exact_queue_metadata_matches=len(scenes),
        exact_queue_with_ordering_ids=len({r["earthexplorer_ordering_id"] for r in scenes}),
        stac_metadata_creation_by_month=Counter(r["stac_metadata_created_at_utc"][:7] for r in scenes),
        stac_assets_with_direct_clock=sum(bool(r["stac_asset_created_at_utc"] or r["stac_asset_updated_at_utc"]
                                              or r["stac_asset_published_at_utc"]) for r in scenes),
        exact_weekly_pair_candidates=len(pairs),pairs_with_stac_metadata_created_before_cutoff=sum(r["pairs_with_stac_created_before_cutoff"] for r in weeks),
        weeks_with_stac_metadata_created_before_cutoff=sum(r["pairs_with_stac_created_before_cutoff"]>0 for r in weeks),
        pairs_with_stac_metadata_last_updated_before_cutoff=sum(r["pairs_with_stac_last_updated_before_cutoff"] for r in weeks),
        weeks_with_stac_metadata_last_updated_before_cutoff=sum(r["pairs_with_stac_last_updated_before_cutoff"]>0 for r in weeks),
        original_b8_file_clocks_verified=0,
        scope="The daily bulk CSV records the archive inventory and the production date. STAC item timestamps describe catalog metadata. "
              "Neither supplies the publication time of the B8 GeoTIFF. The regional inventory is track-based and includes scenes outside Cushing.")
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
