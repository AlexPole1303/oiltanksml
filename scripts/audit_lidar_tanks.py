"""Count LiDAR returns inside mapped roofs and around them."""
import argparse
import csv
import datetime as dt
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import laspy
import numpy as np
from pyproj import Transformer
from shapely import intersects_xy, transform
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from collect_public import RAW, ROOT, write

DERIVED = ROOT / "data" / "derived"
DATASET = "OK_Statewide_2_D22"
LEGACY_DATASET = "USGS_LPC_OK_Area1_1A_2011_LAS_2016"
GPS_EPOCH = dt.datetime(1980, 1, 6, tzinfo=dt.timezone.utc)
ADJUSTED_GPS_TO_UTC_SECONDS = 1_000_000_000 - 18
FIRST_FLIGHT = dt.date(2022, 12, 17)
LAST_FLIGHT = dt.date(2023, 1, 26)
REPORT = "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/metadata/OK_Statewide_D22/USGS_OK_Statewide_D22_Project_Report.pdf"
TIME_SPEC = "https://www.usgs.gov/ngp-standards-and-specifications/lidar-base-specification-data-processing-and-handling-requirements"
LEGACY_TILE_METADATA = ("https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/elevation/lidar_point_cloud/las/"
                        "OK_Area1_1A_2011/USGS_LPC_OK_Area1_1A_2011_35096H6_NWC_LAS_2016_meta.xml")


@dataclass(frozen=True)
class Survey:
    dataset: str
    first_flight: dt.date
    last_flight: dt.date
    gps_to_utc_seconds: int
    date_source: str
    vertical_method: str


SURVEYS = {
    DATASET: Survey(DATASET, FIRST_FLIGHT, LAST_FLIGHT, ADJUSTED_GPS_TO_UTC_SECONDS, REPORT,
                    "Native EPT Z. The report declares NAVD88/GEOID18 in US survey feet before EPT conversion. "
                    "The EPT manifest declares only horizontal CRS."),
    LEGACY_DATASET: Survey(LEGACY_DATASET, dt.date(2011, 12, 7), dt.date(2011, 12, 8), 1_000_000_000 - 15,
                           LEGACY_TILE_METADATA, "Native EPT Z. The EPT manifest declares only horizontal CRS. "
                           "The separate ground comparison checks scale against USGS EPQS in metres."),
}


def as_utc(adjusted_gps_time, offset=ADJUSTED_GPS_TO_UTC_SECONDS):
    return GPS_EPOCH + dt.timedelta(seconds=float(adjusted_gps_time) + offset)


def write_observations(observations, flags, file_prefix, survey):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("candidate_id", "dataset", "acquisition_date_utc", "point_source_id",
                     "first_interior_return_at_utc", "last_interior_return_at_utc",
                     "interior_points", "rim_points", "ground_points", "interior_z_p10", "interior_z_p50",
                     "interior_z_p90", "rim_z_p50", "rim_z_p95", "ground_z_p10", "ground_z_p50", "ground_z_p90"))
    n_candidates, dates = set(), set()
    for (identity, day, source), row in sorted(observations.items()):
        if not row["interior"]:
            continue
        acquisition = (GPS_EPOCH.date() + dt.timedelta(days=day)).isoformat()
        region_points = [np.concatenate(row[region]) if row[region] else np.empty(0)
                         for region in ("interior", "rim", "ground")]
        percentiles = []
        for values, quantiles in zip(region_points, [(10, 50, 90), (50, 95), (10, 50, 90)], strict=True):
            percentiles.extend(np.round(np.percentile(values, quantiles), 3) if values.size else [None] * len(quantiles))
        writer.writerow((identity, survey.dataset, acquisition, source,
                         as_utc(row["first"], survey.gps_to_utc_seconds).isoformat(timespec="seconds"),
                         as_utc(row["last"], survey.gps_to_utc_seconds).isoformat(timespec="seconds"),
                         *[len(values) for values in region_points], *percentiles))
        n_candidates.add(identity)
        dates.add(acquisition)
    metadata = {"dataset": survey.dataset, "project_report": survey.date_source, "time_specification": TIME_SPEC,
                "gps_leap_second_source": "https://www.nist.gov/pml/time-and-frequency-division/time-realization/leap-seconds",
                "project_collection_window": [str(survey.first_flight), str(survey.last_flight)],
                "adjusted_gps_time_to_utc_seconds": survey.gps_to_utc_seconds,
                "ept_las_gps_time_type_values": dict(flags),
                "time_method": "EPT headers declare week time. Numerical timestamps exceed one week and match the official collection window as adjusted GPS time. Raw points are preserved.",
                "vertical_method": survey.vertical_method,
                "interior": "Candidate polygon inset by 15% of equivalent radius (0.8 to 5 m); first returns",
                "rim": "Band 1.5 m on both sides of candidate boundary; first returns",
                "ground": "Classification 2 ground returns 10 to 35 m outside candidate boundary",
                "invalid_returns": "Withheld, synthetic, low noise and high noise returns excluded",
                "observations_scope": "Descriptive elevation distributions by candidate, day and flight line. Roof type, wall elevation and oil volume require independent interpretation.",
                "candidates_with_interior": len(n_candidates), "acquisition_dates_utc": sorted(dates)}
    write(DERIVED / f"{file_prefix}_observations.csv", stream.getvalue().encode())
    write(DERIVED / f"{file_prefix}_observations.json", json.dumps(metadata, indent=2).encode())
    print(f"{len(n_candidates)} candidates with dated elevation observations, UTC dates {sorted(dates)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplemental", action="store_true")
    parser.add_argument("--dataset", choices=SURVEYS, default=DATASET)
    args = parser.parse_args()
    survey = SURVEYS[args.dataset]
    source_name = "supplemental_tank_candidates.geojson" if args.supplemental else "osm_tank_candidates.geojson"
    scope = "ept_supplemental" if args.supplemental else "ept"
    file_prefix = "lidar_supplemental_tank" if args.supplemental else "lidar_tank"
    if survey.dataset == LEGACY_DATASET:
        file_prefix = file_prefix.replace("lidar_", "lidar_2011_", 1)
    candidates = json.loads((DERIVED / source_name).read_bytes())["features"]
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    to_lidar = Transformer.from_crs("EPSG:32614", "EPSG:3857", always_xy=True)
    identifiers, roofs, interiors, rims, surrounds = [], [], [], [], []
    for item in candidates:
        polygon = transform(shape(item["geometry"]), to_utm.transform, interleaved=False)
        identifiers.append(item["properties"]["candidate_id"])
        roofs.append(transform(polygon, to_lidar.transform, interleaved=False))
        inset = min(5.0, max(0.8, math.sqrt(polygon.area / math.pi) * 0.15))
        interiors.append(transform(polygon.buffer(-inset), to_lidar.transform, interleaved=False))
        rim = polygon.buffer(1.5).difference(polygon.buffer(-1.5))
        rims.append(transform(rim, to_lidar.transform, interleaved=False))
        ring = polygon.buffer(35).difference(polygon.buffer(10))
        surrounds.append(transform(ring, to_lidar.transform, interleaved=False))
    extents = [box(*polygon.bounds) for polygon in surrounds]
    tree = STRtree(extents)
    coverage = Counter()
    flags = Counter()
    observations = defaultdict(lambda: {"interior": [], "rim": [], "ground": [],
                                       "first": float("inf"), "last": float("-inf")})
    source = RAW / "usgs" / scope / survey.dataset
    nodes = sorted((source / "nodes").glob("*.laz"))
    if not nodes:
        raise ValueError(f"No local LAZ files in {source}")
    for path in nodes:
        with laspy.open(path) as reader:
            header = reader.header
            if header.parse_crs().to_epsg() != 3857:
                raise ValueError(f"Unexpected horizontal CRS in {path}")
            flags[int(header.global_encoding.gps_time_type)] += 1
            xmin, ymin, _ = header.mins
            xmax, ymax, _ = header.maxs
            selected = tree.query(box(xmin, ymin, xmax, ymax), predicate="intersects")
            if not len(selected):
                continue
            data = reader.read()
        x, y, z = np.asarray(data.x), np.asarray(data.y), np.asarray(data.z)
        classes = np.asarray(data.classification)
        ground = classes == 2
        adjusted_gps = np.asarray(data.gps_time)
        if not np.isfinite(adjusted_gps).all() or not (
            survey.first_flight <= as_utc(adjusted_gps.min(), survey.gps_to_utc_seconds).date()
            <= as_utc(adjusted_gps.max(), survey.gps_to_utc_seconds).date() <= survey.last_flight
        ):
            raise ValueError(f"GPS times outside USGS project dates in {path}")
        days = np.floor((adjusted_gps + survey.gps_to_utc_seconds) / 86_400).astype(int)
        source_ids = np.asarray(data.point_source_id)
        eligible = ~np.asarray(data.withheld, dtype=bool) & ~np.asarray(data.synthetic, dtype=bool)
        eligible &= ~np.isin(classes, [7, 18])
        first = (np.asarray(data.return_number) == 1) & eligible
        flight_groups = [(int(day), int(source_id), (days == day) & (source_ids == source_id))
                         for day, source_id in np.unique(np.column_stack((days, source_ids)), axis=0)]
        for i in selected:
            roof = intersects_xy(roofs[i], x, y)
            ring = intersects_xy(surrounds[i], x, y)
            coverage[i, "interior"] += int(roof.sum())
            coverage[i, "ring"] += int(ring.sum())
            coverage[i, "ring_ground"] += int(np.count_nonzero(ring & ground))
            interior = first & intersects_xy(interiors[i], x, y)
            rim = first & intersects_xy(rims[i], x, y)
            ground_ring = eligible & ground & ring
            for day, source_id, group in flight_groups:
                if not np.any(group & (interior | rim | ground_ring)):
                    continue
                obs = observations[identifiers[i], day, source_id]
                for region, mask in (("interior", interior), ("rim", rim), ("ground", ground_ring)):
                    selected_points = mask & group
                    if selected_points.any():
                        obs[region].append(z[selected_points])
                        if region == "interior":
                            timestamps = adjusted_gps[selected_points]
                            obs["first"] = min(obs["first"], timestamps.min())
                            obs["last"] = max(obs["last"], timestamps.max())
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("candidate_id", "returns_inside_polygon", "returns_10_to_35m_from_polygon",
                     "ground_class_2_returns_10_to_35m"))
    for i, identity in enumerate(identifiers):
        writer.writerow([identity, coverage[i, "interior"], coverage[i, "ring"], coverage[i, "ring_ground"]])
    write(DERIVED / f"{file_prefix}_coverage.csv", stream.getvalue().encode())
    print(f"{sum(bool(coverage[i, 'interior']) for i in range(len(identifiers)))} of {len(identifiers)} candidates have returns inside their mapped geometry")
    write_observations(observations, flags, file_prefix, survey)


if __name__ == "__main__":
    main()
