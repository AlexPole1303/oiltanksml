"""Check exact B8 download routes and order the full queue by marginal coverage."""

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from audit_landsat_c2_alternative_access import probe
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import write, write_csv
from verify_cushing_b8_originals import check, stac_records

STEM = "cushing_b8_acquisition_plan"
ORIGINALS = Path("/Volumes/Untitled/oiltanksml-public-data/landsat-b8-originals")
PAIR_FIELDS = ("week_ending", "candidate_id", "route", "previous_version_id", "current_version_id",
               "previous_source_file", "current_source_file")
EXTERNAL = ("lidar_2022", "state_aerial_2025", "icesat2_laser_reference")


def index(rows, fields):
    result = {tuple(r[k] for k in fields): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Repeated keys for {fields}")
    return result


def routes(meta):
    usgs = urlsplit(meta["landsatlook_b8_url"])
    aws = urlsplit(meta["usgs_s3_b8_url"])
    if usgs.hostname != "landsatlook.usgs.gov" or aws.netloc != "usgs-landsat" or usgs.path != "/data"+aws.path:
        raise ValueError(f"Conflicting original archive paths for {meta['product_id']}")
    return {
        "landsatlook": meta["landsatlook_b8_url"],
        "copernicus_exact_file": meta["cdse_b8_url"],
        "usgs_aws": "https://usgs-landsat.s3.us-west-2.amazonaws.com"+aws.path,
        "google_requester_pays": "https://storage.googleapis.com/gee-public-data-landsat"+aws.path,
    }


def access(meta):
    result = []
    checked_at = datetime.now(timezone.utc).isoformat()
    for route, url in routes(meta).items():
        try:
            response = probe(url)
        except OSError as error:
            response = dict(url=url, http_status=None, response_host=None, tiff_signature=False)
            exception = type(error).__name__
        else:
            exception = None
        result.append(dict(product_id=meta["product_id"], route=route, checked_at_utc=checked_at, **response,
                           request_error=exception))
    return result


def pairs(originals):
    hls = index(read("cushing_b8_hls_lineage_pairs"), PAIR_FIELDS)
    candidates = index(read("landsat_c2_b8_weekly_pairs"), PAIR_FIELDS)
    registered = index(read("landsat_esri_b8_registered_changes_pairs"), PAIR_FIELDS)
    if candidates.keys() != hls.keys() or not registered.keys() <= candidates.keys():
        raise ValueError("The exact B8 option, timestamp and registration inventories disagree")
    for key, r in candidates.items():
        clock, fit = hls[key], registered.get(key, {})
        if r["forecast_cutoff_utc"] != clock["forecast_cutoff_utc"]:
            raise ValueError(f"Inconsistent EIA cutoff for {key}")
        required = frozenset(r[f"{side}_version_id"] for side in ("previous", "current") if r[f"{side}_sensor"]=="B8")
        if not required or not required <= originals or len(required) != r["route"].split("_").count("B8"):
            raise ValueError(f"Wrong original files for {key}")
        time_proxy = clock["all_hls_lineage_before_cutoff"] == "True" or clock["cdse_catalog_all_clocks_before_cutoff"] == "True"
        resolved = fit.get("registered_both_shadow_diagnostics_pass") == "True"
        if fit and time_proxy != (fit["cdse_source_bundle_clocks_before_cutoff"] == "True" or fit["hls_exact_scene_lineage_before_cutoff"] == "True"):
            raise ValueError(f"Timestamp attribution differs for {key}")
        yield dict({k: r[k] for k in PAIR_FIELDS}, terminal_id=r["terminal_id"], release_date=r["release_date"],
                   forecast_cutoff_utc=r["forecast_cutoff_utc"], required=required, time_proxy=time_proxy,
                   registered_esri_fit=resolved, registered_esri_fit_and_time_proxy=resolved and time_proxy,
                   original_file_availability_before_cutoff=False)


def order_queue(records, options, verified):
    groups = ("geometry", "time_proxy", "registered_esri_fit", "registered_esri_fit_and_time_proxy")
    required_by_file = defaultdict(list)
    covered = {k: set() for k in groups}
    covered_options = set()
    for i, row in enumerate(options):
        for name in row["required"]:
            required_by_file[name].append(i)

    def unlock(indices):
        unlocked = {k: set() for k in groups}
        for i in indices:
            r = options[i]
            target = r["week_ending"], r["candidate_id"]
            for group in groups:
                if group == "geometry" or r[group]:
                    unlocked[group].add(target)
        return unlocked

    def add(indices):
        gains = unlock(indices)
        for k in groups:
            gains[k].difference_update(covered[k])
            covered[k].update(gains[k])
        covered_options.update(indices)
        return gains

    staged = set(verified)
    add({i for i, row in enumerate(options) if row["required"] <= staged})
    pending = records.keys() - staged
    step = 0
    ordered = []
    while pending:
        choices = {}
        for name in pending:
            indices = [i for i in required_by_file[name] if i not in covered_options and options[i]["required"] <= staged|{name}]
            gains = unlock(indices)
            row = records[name]
            purpose = row["purpose"]
            score = (EXTERNAL.index(purpose) if purpose in EXTERNAL else len(EXTERNAL),
                     *(-len(gains[k]-covered[k]) for k in reversed(groups)),
                     -len(required_by_file[name]), int(row["download_priority"]), name)
            choices[name] = score, indices
        name = min(choices, key=lambda n: choices[n][0])
        pending.remove(name)
        staged.add(name)
        step += 1
        _, indices = choices[name]
        gains = add(indices)
        ordered.append(dict(download_step=step, **records[name], exact_pair_options_newly_ready=len(indices),
                            weeks_with_new_geometry_tank_weeks=len({a for a,_ in gains["geometry"]}),
                            **{f"new_tank_weeks_{k}": len(gains[k]) for k in groups},
                            **{f"cumulative_tank_weeks_{k}": len(covered[k]) for k in groups}))
    return ordered, covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--originals", type=Path, default=ORIGINALS)
    parser.add_argument("--reuse-access", action="store_true")
    args = parser.parse_args()
    directory = args.originals.resolve(strict=True)
    if not directory.is_dir():
        parser.error("--originals must name a directory")
    meta = read("cushing_usgs_b8_metadata_scenes")
    indexed = index(meta, ("product_id",))
    items = stac_records()
    files = [check(directory, row, items) for row in meta]
    verified = {r["product_id"] for r in files if r["status"] == "verified_original_b8"}
    if args.reuse_access:
        results = read(f"{STEM}_access")
    else:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = [r for batch in pool.map(access, meta) for r in batch]
        export(f"{STEM}_access", results)
    expected = {(row["product_id"], route): url for row in meta for route,url in routes(row).items()}
    if len(results) != len(expected) or {(r["product_id"],r["route"]):r["url"] for r in results} != expected:
        raise ValueError("Incomplete download route check")
    accessible = defaultdict(list)
    for r in results:
        if r["tiff_signature"] in (True, "True"):
            accessible[r["product_id"]].append(r["route"])
    records = {}
    for row in files:
        name = row["product_id"]
        original = indexed[name,]
        records[name] = dict(row, capture_utc=original["captured_at_utc"], exact_bytes=int(original["b8_mirror_bytes"]),
                             earthexplorer_ordering_id=original["earthexplorer_ordering_id"],
                             landsatlook_url=original["landsatlook_b8_url"], cdse_url=original["cdse_b8_url"],
                             usgs_s3_url=original["usgs_s3_b8_url"], anonymous_tiff_routes_json=json.dumps(accessible[name]))
    options = list(pairs(records.keys()))
    queue, covered = order_queue(records, options, verified)
    if queue:
        export(f"{STEM}_queue", queue)
    else:
        fields = ("download_step", *next(iter(records.values())), "exact_pair_options_newly_ready", "weeks_with_new_geometry_tank_weeks",
                  *(f"{prefix}_tank_weeks_{group}" for prefix in ("new", "cumulative") for group in covered))
        write_csv(DERIVED/f"{STEM}_queue.csv", fields, [])
    write(DERIVED/f"{STEM}_earthexplorer_product_ids.txt", "".join(f"{r['product_id']}\n" for r in queue).encode())
    steps = {r["product_id"]: r["download_step"] for r in queue}
    export(f"{STEM}_pairs", [dict({k:v for k,v in r.items() if k!="required"},
                                   original_products_json=json.dumps(sorted(r["required"])),
                                   waiting_original_products_json=json.dumps(sorted(r["required"]-verified)),
                                   earliest_planned_step=max((steps[n] for n in r["required"]-verified),default=0)) for r in options])
    summary = dict(local_directory=str(directory), originals=len(meta), verified_originals=len(verified),
                   local_status=Counter(r["status"] for r in files), pending_bytes=sum(r["exact_bytes"] for r in queue),
                   download_route_checks=len(results), anonymous_tiff_products=sum(bool(v) for v in accessible.values()),
                   landsatlook_signin_responses=sum(r["route"]=="landsatlook" and r["response_host"]=="ers.cr.usgs.gov" for r in results),
                   route_status={route:Counter(str(r["http_status"]) for r in results if r["route"]==route)
                                 for route in sorted({r["route"] for r in results})},
                   pair_options=len(options), potential_tank_weeks={k:len(v) for k,v in covered.items()},
                   potential_geometry_weeks=len({a for a,_ in covered["geometry"]}),
                   top_weekly_after_external_controls=[{k:r[k] for k in ("product_id", "download_step", "new_tank_weeks_geometry",
                       "new_tank_weeks_time_proxy", "new_tank_weeks_registered_esri_fit_and_time_proxy")} for r in queue
                       if r["purpose"] not in EXTERNAL][:5],
                   scope="Marginal acquisition order deduplicates roof-week combinations across exact scene options. "
                         "Counts refer to available candidate geometry, historical proxies or retrospective Esri diagnostics. "
                         "File receipt and proxy dates do not establish historical B8 availability or a resolved USGS measurement.")
    payload = json.dumps(summary, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
