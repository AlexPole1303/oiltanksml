"""Check local cloud coverage and original L1C archive clocks for new releases."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from build_s1_reference_matches import DERIVED, export, read
from collect_imagery import CONFIG, clip
from collect_public import RAW, ROOT, write
from collect_s2_l1c import parse_product, query_archive
from collect_s2_l1c_archive_times import record
from publication_clocks import utc

STEM = "cushing_forward_optical_coverage"
RAW_OPTICAL = RAW/"cushing_forward_sentinel2"
# SCL classes 4, 5 and 6 are vegetation, unvegetated surface and water.
CLEAR = (4, 5, 6)
EASTERN = ZoneInfo("America/New_York")


def archive_record(item, *, index_store=RAW_OPTICAL/"l1c_indexes", object_index_store=None):
    key, _, _ = parse_product(item["properties"]["s2:product_uri"])
    sensor, sensing, _, tile = key
    prefix = f"tiles/{tile[:2]}/{tile[2]}/{tile[3:]}/{sensor}_MSIL1C_{sensing}_"
    directories = query_archive(prefix, directories=True, cache=index_store/f"{sensor}_{sensing}_{tile}.json")
    matches = []
    for directory in directories:
        product = directory.rstrip("/").rsplit("/", 1)[-1]
        found, _, generation = parse_product(product)
        if found == key:
            matches.append((generation, product, directory))
    versions = {key:min(matches)} if matches else {}
    if object_index_store is None:
        return record(item, versions)
    return record(item, versions, object_index_store=object_index_store)


def scene_coverage(item, tanks, output, *, index_store=RAW_OPTICAL/"l1c_indexes", object_index_store=None):
    properties = item["properties"]
    tile = shape(item["geometry"])
    candidates = {name: roof for name, roof in tanks.items() if tile.intersects(roof)}
    bounds = unary_union(list(candidates.values())).bounds
    url = item["assets"]["scl"]["href"]
    raster_path = output/f"{item['id']}_scl.tif"
    clip(url, raster_path, bounds)
    with rasterio.Env(**CONFIG), rasterio.open(raster_path) as source:
        pixels = source.read(1)
        to_crs = source.crs
        clear, present = set(), set()
        for name, roof in candidates.items():
            geometry = rasterio.warp.transform_geom("EPSG:4326", to_crs, mapping(roof))
            mask = geometry_mask([geometry], out_shape=pixels.shape, transform=source.transform, invert=True, all_touched=True)
            values = pixels[mask]
            if not values.size:
                raise ValueError(f"Missing footprint pixels for {name} on {item['id']}")
            if np.all(values != 0) and tile.covers(roof):
                present.add(name)
                if np.isin(values, CLEAR).all():
                    clear.add(name)
    _, captured, status, product, archive, available, detector_ready, lag, *_ = archive_record(
        item,index_store=index_store,object_index_store=object_index_store)
    return dict(item_id=item["id"], captured_at_utc=captured, grid=properties["grid:code"],
                tile_cloud_pct=properties["eo:cloud_cover"], scl_source_url=url,
                scl_clip_sha256=hashlib.sha256(raster_path.read_bytes()).hexdigest(),
                fully_observed_roofs=len(present), clear_roofs=len(clear), l1c_status=status,
                l1c_product=product, l1c_archive_url=archive, l1c_inputs_available_by_utc=available,
                l1c_with_detector_available_by_utc=detector_ready, archive_lag_hours=lag), clear


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--week-ending", type=date.fromisoformat, help="Survey week ending for an upcoming release")
    args = parser.parse_args()
    issue = date.fromisoformat(args.release)
    if args.week_ending is not None:
        week = args.week_ending
        if week.weekday()!=4 or not 0<(issue-week).days<=7:
            parser.error("Choose the Friday covered by the upcoming report")
        as_of = datetime.now(EASTERN)
        cutoff = min(datetime.fromisoformat(utc(issue)),as_of)
        latest = datetime.fromisoformat(utc(week,time(7)))
        prior_day = week-timedelta(days=7)
    else:
        release, = [r for r in read("eia_wpsr_forward_releases") if r["release_date"] == args.release]
        cutoff, latest = map(datetime.fromisoformat, (release["conservative_prerelease_before_utc"], release["measured_at_utc"]))
        prior_day = date.fromisoformat(release["previous_week_ending"])
    prior = datetime.fromisoformat(utc(prior_day,time(7)))
    lower = datetime.fromisoformat(utc(prior_day-timedelta(days=7),time(7)))
    known = {r["candidate_id"] for r in read("tank_engineering_registry") if r["documented_roof_type"] == "EFR"
             and r["first_seen"] and r["first_seen"] <= lower.date().isoformat()}
    tanks = {f["properties"]["candidate_id"]: shape(f["geometry"])
             for stem in ("osm_tank_candidates", "supplemental_tank_candidates")
             for f in json.loads((DERIVED/f"{stem}.geojson").read_bytes())["features"] if f["properties"]["candidate_id"] in known}
    if tanks.keys() != known:
        raise ValueError("The EFR registry and roof footprints disagree")
    footprint = unary_union(list(tanks.values()))
    items = sorted((item for item in json.loads(args.catalog.read_bytes())["features"] if shape(item["geometry"]).intersects(footprint)),
                   key=lambda item: (item["properties"]["datetime"], item["id"]))
    output = RAW_OPTICAL/"scl_clips"
    with ThreadPoolExecutor(max_workers=6) as executor:
        records = list(executor.map(lambda item: scene_coverage(item, tanks, output), items))
    daily, rows = {}, []
    for r, clear in records:
        captured = datetime.fromisoformat(r["captured_at_utc"])
        segment = ("before_reference_window" if captured <= lower else "preceding_week" if captured <= prior
                   else "target_week" if captured <= latest else "after_target")
        ready = r["l1c_with_detector_available_by_utc"]
        on_time = ready is not None and captured < datetime.fromisoformat(ready) < cutoff
        daily.setdefault((captured.date().isoformat(), segment), set()).update(clear if on_time else ())
        rows.append(r | dict(release_date=args.release, segment=segment, l1c_pre_release=on_time))
    export(f"{STEM}_{args.release}_tiles", rows)
    passes = [dict(date=day, segment=segment, distinct_clear_roofs=len(clear)) for (day, segment), clear in sorted(daily.items())]
    preceding = set().union(*(clear for (_, segment), clear in daily.items() if segment == "preceding_week"))
    target = set().union(*(clear for (_, segment), clear in daily.items() if segment == "target_week"))
    report = dict(release_date=args.release, release_mode="upcoming" if args.week_ending else "archived",
                  preceding_week_end_utc=prior.isoformat(), target_week_end_utc=latest.isoformat(),
                  forecast_cutoff_utc=cutoff.isoformat(), source_catalog=str(args.catalog.resolve().relative_to(ROOT)),
                  sample_roofs=len(tanks), clear_on_both_sides=len(preceding & target), passes=passes,
                  scope="Local L2A scene classification paired with original L1C archive clocks. No shadow or height admission.")
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED/f"{STEM}_{args.release}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
