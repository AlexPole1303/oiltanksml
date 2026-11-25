"""Index Stroud point returns and retain the unresolved GPS week."""
import datetime as dt
import hashlib
import json
import xml.etree.ElementTree as ET

import laspy
import numpy as np
from pyproj import Transformer
from shapely import intersects_xy, transform
from shapely.geometry import shape

from collect_public import RAW, ROOT, get, write_csv

DATASET = "USGS_LPC_OK_NRCS_Area1C_2012_LAS_2018"
TILE = "35096F6_NEA"
FGDC_FILE = f"USGS_LPC_OK_NRCS_Area1C_2012_{TILE}_LAS_2018_meta.xml"
ORIGINAL_FILE = f"USGS_LPC_NRCS_Dam_Rehab_Assessment_{TILE}.laz"
FGDC_URL = ("https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/lidar_point_cloud/las/"
            "OK_NRCS-Area1C_2012/" + FGDC_FILE)
STAC_URL = f"https://s3-us-west-2.amazonaws.com/usgs-lidar-stac/ept/{DATASET}.json"
GPS_UTC_LEAP_SECONDS = 15
WEEK = dt.timedelta(days=7)


def feasible_utc_times(gps_week_seconds, first_day, last_day):
    sunday = first_day - dt.timedelta(days=(first_day.weekday() + 1) % 7)
    possible = []
    while sunday <= last_day + dt.timedelta(days=1):
        time = dt.datetime.combine(sunday, dt.time(), dt.timezone.utc)
        time += dt.timedelta(seconds=float(gps_week_seconds) - GPS_UTC_LEAP_SECONDS)
        if first_day <= time.date() <= last_day:
            possible.append(time.isoformat())
        sunday += WEEK
    return possible


def original_point_times(path, features):
    result = {identity: [] for identity in features}
    with laspy.open(path) as reader:
        crs = reader.header.parse_crs()
        if crs is None or int(reader.header.global_encoding.gps_time_type) != 0:
            raise ValueError(f"Expected a georeferenced original LAS with GPS week time: {path}")
        convert = Transformer.from_crs(4326, crs, always_xy=True)
        polygons = {identity: transform(poly, convert.transform, interleaved=False) for identity, poly in features.items()}
        for points in reader.chunk_iterator(500_000):
            x, y = np.asarray(points.x), np.asarray(points.y)
            eligible = ~np.asarray(points.withheld, dtype=bool) & ~np.asarray(points.synthetic, dtype=bool)
            eligible &= (np.asarray(points.return_number) == 1) & ~np.isin(np.asarray(points.classification), [7, 18])
            for identity, polygon in polygons.items():
                keep = eligible & intersects_xy(polygon, x, y)
                if keep.any():
                    result[identity].append(np.asarray(points.gps_time)[keep])
    return {identity: np.concatenate(parts) for identity, parts in result.items()}


def main():
    derived = ROOT / "data" / "derived"
    root = RAW / "usgs"
    ept = root / "ept_supplemental" / DATASET
    fgdc_file = root / "legacy_metadata" / FGDC_FILE
    dates = ET.fromstring(get(FGDC_URL, fgdc_file)).find(".//idinfo/timeperd/timeinfo/rngdates")
    earliest, latest = (dt.datetime.strptime(dates.findtext(tag), "%Y%m%d").date() for tag in ("begdate", "enddate"))
    project = json.loads(get(STAC_URL, root / "stac" / f"{DATASET}.json"))
    ept_url = project["assets"]["ept.json"]["href"]
    origins_url = ept_url.removesuffix("/ept.json") + "/ept-sources/list.json"
    origins = json.loads(get(origins_url, ept / "ept-sources-list.json"))
    features = json.loads((derived / "supplemental_tank_candidates.geojson").read_bytes())["features"]
    footprints = {f["properties"]["candidate_id"]: shape(f["geometry"])
                  for f in features if f["properties"]["candidate_id"].startswith("ortho-stroud-")}
    to_ept = Transformer.from_crs(4326, 3857, always_xy=True)
    polygons = {identity: transform(poly, to_ept.transform, interleaved=False) for identity, poly in footprints.items()}
    original = root / "original_legacy" / ORIGINAL_FILE
    original_times = original_point_times(original, footprints) if original.is_file() else {}
    if original_times:
        with original.open("rb") as source:
            original_hash = hashlib.file_digest(source, "sha256").hexdigest()
    else:
        original_hash = None
    selected = {identity: [] for identity in polygons}
    for path in sorted((ept / "nodes").glob("*.laz")):
        data = laspy.read(path)
        if data.header.parse_crs().to_epsg() != 3857:
            raise ValueError(f"Unexpected horizontal CRS in {path}")
        gps = np.asarray(data.gps_time)
        if int(data.header.global_encoding.gps_time_type) != 0 or not np.all((0 <= gps) & (gps < WEEK.total_seconds())):
            raise ValueError(f"Expected LAS GPS week time in {path}")
        x, y = np.asarray(data.x), np.asarray(data.y)
        eligible = ~np.asarray(data.withheld, dtype=bool) & ~np.asarray(data.synthetic, dtype=bool)
        eligible &= (np.asarray(data.return_number) == 1) & ~np.isin(np.asarray(data.classification), [7, 18])
        for identity, polygon in polygons.items():
            keep = eligible & intersects_xy(polygon, x, y)
            if keep.any():
                selected[identity].append((path, gps[keep], np.asarray(data.z)[keep],
                                           np.asarray(data.OriginId)[keep]))
    rows = []
    for identity, records in sorted(selected.items()):
        if not records:
            raise ValueError(f"No usable point return for {identity}")
        gps = np.concatenate([entry[1] for entry in records])
        z = np.concatenate([entry[2] for entry in records])
        origin_ids = np.unique(np.concatenate([entry[3] for entry in records])).astype(int).tolist()
        source_tiles = [origins[index]["path"] for index in origin_ids]
        if any(TILE not in tile for tile in source_tiles):
            raise ValueError(f"Review tile metadata for additional sources of {identity}: {source_tiles}")
        first_options, last_options = (feasible_utc_times(value, earliest, latest) for value in (gps.min(), gps.max()))
        if len(first_options) < 2 or len(last_options) < 2:
            raise ValueError(f"Recheck date assignment for {identity}")
        from_original = original_times.get(identity)
        matches = int(np.count_nonzero(np.isin(gps, from_original))) if from_original is not None else None
        rows.append((identity, DATASET, TILE, earliest, latest, len(z), *np.round(np.percentile(z, (10, 50, 90)), 3),
                     round(float(gps.min()), 6), round(float(gps.max()), 6), json.dumps(first_options), json.dumps(last_options),
                     "gps_week_number_unresolved",
                     json.dumps(source_tiles), json.dumps([str(entry[0].relative_to(ROOT)) for entry in records]),
                     FGDC_URL, ept_url,
                     len(from_original) if from_original is not None else None, matches,
                     str(original.relative_to(ROOT)) if original_times else None, original_hash))
    fields = ("candidate_id", "dataset", "source_tile", "fgdc_flight_start", "fgdc_flight_end", "interior_first_returns",
              "interior_z_p10_native", "interior_z_p50_native", "interior_z_p90_native", "gps_week_time_min_s", "gps_week_time_max_s",
              "utc_options_for_min_week_time_json", "utc_options_for_max_week_time_json", "date_status", "ept_origin_paths_json", "local_laz_files_json",
              "fgdc_source_url", "ept_source_url", "original_interior_first_returns", "ept_returns_matching_original_gps",
              "local_original_laz", "original_laz_sha256")
    write_csv(derived / "lidar_stroud_legacy_observations.csv", fields, rows)
    print(f"{len(rows)} Stroud contours with point returns and independently retained time records")


if __name__ == "__main__":
    main()
