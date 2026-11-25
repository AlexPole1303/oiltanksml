"""Describe native LiDAR elevation by contour, flight, radial band and grid bearing."""
import csv
import datetime as dt
import json
from collections import defaultdict
from dataclasses import dataclass

import laspy
import numpy as np
from pyproj import Transformer
from shapely.geometry import box
from shapely.strtree import STRtree

from audit_lidar_tanks import DATASET, GPS_EPOCH, SURVEYS, as_utc
from collect_public import RAW, ROOT, write, write_csv

DERIVED = ROOT / "data" / "derived"
RADII = (0.0, 0.60, 0.85, 0.97, 1.03, 1.15)
ZONES = ("center", "inner", "inner_edge", "boundary", "outside")
SECTORS = 8
SCHEMA = """
CREATE TABLE lidar_radial_profile (
    candidate_id TEXT,dataset TEXT,geometry_basis TEXT,outline_fit_status TEXT,center_east_m REAL,center_north_m REAL,
    radius_m REAL,acquisition_date_utc TEXT,point_source_id INTEGER,radial_band TEXT,radius_fraction_lower REAL,
    radius_fraction_upper REAL,sector INTEGER,azimuth_grid_lower_deg REAL,azimuth_grid_upper_deg REAL,
    first_return_at_utc TEXT,last_return_at_utc TEXT,first_returns INTEGER,ground_class_2_returns INTEGER,
    z_min_native REAL,z_p10_native REAL,z_p50_native REAL,z_p90_native REAL,z_p95_native REAL,z_max_native REAL,
    nonground_z_p50_native REAL,nonground_z_p95_native REAL,source_directory TEXT,source_node_ids_json TEXT,
    PRIMARY KEY(candidate_id,dataset,geometry_basis,acquisition_date_utc,point_source_id,radial_band,sector));
CREATE VIEW lidar_radial_boundary_comparison AS
    SELECT c.candidate_id,c.dataset,c.geometry_basis,c.outline_fit_status,c.acquisition_date_utc,c.point_source_id,
        c.center_east_m,c.center_north_m,c.radius_m,c.first_returns AS center_first_returns,
        b.first_returns AS boundary_first_returns,c.ground_class_2_returns AS center_ground_returns,
        b.ground_class_2_returns AS boundary_ground_returns,c.z_p50_native AS center_z_p50_native,
        b.z_p95_native AS boundary_z_p95_native,round(b.z_p95_native-c.z_p50_native,3) AS boundary_minus_center_native_z,
        c.nonground_z_p50_native AS center_nonground_z_p50_native,b.nonground_z_p95_native AS boundary_nonground_z_p95_native,
        round(b.nonground_z_p95_native-c.nonground_z_p50_native,3) AS nonground_boundary_minus_center_native_z,
        l.rim_z_p95 AS mapped_polygon_rim_z_p95,l.interior_z_p50 AS mapped_polygon_interior_z_p50,
        round(l.rim_z_p95-l.interior_z_p50,3) AS mapped_polygon_rim_minus_interior_native_z,
        t.latest_aerial_date,t.latest_aerial_top_observation,t.documented_roof_type,
        t.source_document_date AS roof_type_document_date,c.source_directory
    FROM lidar_radial_profile c JOIN lidar_radial_profile b
        USING (candidate_id,dataset,geometry_basis,acquisition_date_utc,point_source_id)
    LEFT JOIN lidar_tank_observation l USING (candidate_id,dataset,acquisition_date_utc,point_source_id)
    LEFT JOIN tank_engineering_registry t ON t.candidate_id=c.candidate_id
    WHERE c.radial_band='center' AND c.sector=-1 AND b.radial_band='boundary' AND b.sector=-1;
"""
FIELDS = (
    "candidate_id", "dataset", "geometry_basis", "outline_fit_status", "center_east_m", "center_north_m", "radius_m",
    "acquisition_date_utc", "point_source_id", "radial_band", "radius_fraction_lower", "radius_fraction_upper",
    "sector", "azimuth_grid_lower_deg", "azimuth_grid_upper_deg", "first_return_at_utc", "last_return_at_utc",
    "first_returns", "ground_class_2_returns", "z_min_native", "z_p10_native", "z_p50_native", "z_p90_native",
    "z_p95_native", "z_max_native", "nonground_z_p50_native", "nonground_z_p95_native", "source_directory",
    "source_node_ids_json",
)


@dataclass(frozen=True)
class Circle:
    candidate: str
    basis: str
    fit_status: str | None
    east: float
    north: float
    radius: float


def reference_circles():
    with (DERIVED / "tank_lidar_outlines.csv").open() as stream:
        outlines = {row["candidate_id"]: row for row in csv.DictReader(stream)}
    by_scope = defaultdict(list)
    with (DERIVED / "tank_footprint_geometry.csv").open() as stream:
        for row in csv.DictReader(stream):
            candidate = row["candidate_id"]
            scope = "ept_supplemental" if row["source_file"].endswith("supplemental_tank_candidates.geojson") else "ept"
            by_scope[scope].append(Circle(candidate, "mapped_footprint_circle", None, float(row["center_east_m"]),
                                         float(row["center_north_m"]), float(row["equivalent_diameter_m"]) / 2))
            fitted = outlines[candidate]
            if fitted["diameter_m"]:
                by_scope[scope].append(Circle(candidate, "lidar_elevated_circle", fitted["fit_status"],
                                             float(fitted["center_east_m"]), float(fitted["center_north_m"]),
                                             float(fitted["diameter_m"]) / 2))
    return by_scope


def select_returns(circles, survey, node_paths, *, include_xy=False, include_origin=False):
    to_ept = Transformer.from_crs(32614, 3857, always_xy=True)
    to_utm = Transformer.from_crs(3857, 32614, always_xy=True)
    boxes = []
    for circle in circles:
        limit = circle.radius * RADII[-1]
        boxes.append(box(*to_ept.transform_bounds(circle.east - limit, circle.north - limit,
                                                   circle.east + limit, circle.north + limit)))
    tree = STRtree(boxes)
    selected = defaultdict(list)
    for node, path in enumerate(node_paths):
        with laspy.open(path) as reader:
            header = reader.header
            if header.parse_crs().to_epsg() != 3857:
                raise ValueError(f"Unexpected horizontal CRS in {path}")
            x0, y0, _ = header.mins
            x1, y1, _ = header.maxs
            nearby = tree.query(box(x0, y0, x1, y1), predicate="intersects")
            if not len(nearby):
                continue
            points = reader.read()
        classes = np.asarray(points.classification)
        keep = ~np.asarray(points.withheld, dtype=bool) & ~np.asarray(points.synthetic, dtype=bool)
        keep &= (np.asarray(points.return_number) == 1) & ~np.isin(classes, (7, 18))
        if not np.any(keep):
            continue
        x, y = to_utm.transform(np.asarray(points.x)[keep], np.asarray(points.y)[keep])
        gps = np.asarray(points.gps_time)[keep]
        if not np.isfinite(gps).all() or not (
            survey.first_flight <= as_utc(gps.min(), survey.gps_to_utc_seconds).date()
            <= as_utc(gps.max(), survey.gps_to_utc_seconds).date() <= survey.last_flight
        ):
            raise ValueError(f"GPS time outside survey dates in {path}")
        days = np.floor((gps + survey.gps_to_utc_seconds) / 86_400)
        z = np.asarray(points.z)[keep]
        sources = np.asarray(points.point_source_id)[keep]
        ground = classes[keep] == 2
        for index in nearby:
            circle = circles[index]
            dx, dy = x - circle.east, y - circle.north
            relative = np.hypot(dx, dy) / circle.radius
            inside = relative < RADII[-1]
            if not inside.any():
                continue
            bands = np.searchsorted(RADII, relative[inside], side="right") - 1
            bearings = np.degrees(np.arctan2(dx[inside], dy[inside])) % 360
            sectors = np.floor(bearings * SECTORS / 360).astype(int) % SECTORS
            columns = (days[inside], sources[inside], bands, sectors, gps[inside],
                       z[inside], ground[inside], np.full(int(inside.sum()), node))
            if include_xy:
                columns += (dx[inside], dy[inside])
            if include_origin:
                columns += (np.asarray(points.OriginId)[keep][inside],)
            selected[index].append(np.column_stack(columns))
    return selected


def summaries(points):
    order = np.lexsort(tuple(points[:, column] for column in (3, 2, 1, 0)))
    ordered = points[order]
    changes = np.flatnonzero(np.any(ordered[1:, :3] != ordered[:-1, :3], axis=1)) + 1
    for band in np.split(ordered, changes):
        yield -1, band
        splits = np.flatnonzero(band[1:, 3] != band[:-1, 3]) + 1
        for sector in np.split(band, splits):
            yield int(sector[0, 3]), sector


def profile_rows(circles, survey, directory):
    node_paths = sorted(directory.glob("*.laz"))
    if not node_paths:
        raise ValueError(f"No local point files in {directory}")
    selected = select_returns(circles, survey, node_paths)
    for index, groups in sorted(selected.items()):
        circle = circles[index]
        prefix = (circle.candidate, survey.dataset, circle.basis, circle.fit_status, circle.east, circle.north, circle.radius)
        for sector, points in summaries(np.concatenate(groups)):
            day, line, zone = map(int, points[0, :3])
            gps, z, ground, nodes = points[:, 4:].T
            day = (GPS_EPOCH.date() + dt.timedelta(days=day)).isoformat()
            start, end = (as_utc(value, survey.gps_to_utc_seconds).isoformat() for value in (gps.min(), gps.max()))
            lower, upper = (None, None) if sector == -1 else (sector * 360 / SECTORS, (sector + 1) * 360 / SECTORS)
            quantiles = np.round(np.percentile(z, (0, 10, 50, 90, 95, 100)), 3)
            other = z[ground == 0]
            nonground = np.round(np.percentile(other, (50, 95)), 3) if len(other) else (None, None)
            identifiers = [node_paths[int(node)].stem for node in np.unique(nodes)]
            yield (*prefix, day, line, ZONES[zone], RADII[zone], RADII[zone + 1], sector, lower, upper, start, end,
                   len(z), int(ground.sum()), *quantiles, *nonground, str(directory.relative_to(ROOT)), json.dumps(identifiers))


def main():
    circles = reference_circles()
    rows = []
    for survey in SURVEYS.values():
        for scope, references in circles.items():
            if survey.dataset != DATASET:
                references = [circle for circle in references if circle.basis == "mapped_footprint_circle"]
            directory = RAW / "usgs" / scope / survey.dataset / "nodes"
            rows.extend(profile_rows(references, survey, directory))
    rows.sort(key=lambda row: (row[1], row[0], row[2], row[7], row[8], RADII.index(row[10]), row[12]))
    path = DERIVED / "lidar_radial_profile.csv"
    write_csv(path, FIELDS, rows)
    write(path.with_suffix(".json"), json.dumps({
        "dataset_references": {survey.dataset: survey.date_source for survey in SURVEYS.values()},
        "horizontal_crs": "EPSG:32614", "vertical_values": "Native EPT Z without a datum transformation",
        "radial_bands": dict(zip(ZONES, zip(RADII[:-1], RADII[1:], strict=True), strict=True)),
        "radial_band_endpoint_convention": "lower inclusive, upper exclusive",
        "bearing": "Clockwise from grid north, eight 45-degree sectors; -1 aggregates every sector directly from points",
        "eligibility": "First returns, excluding synthetic, withheld and classification 7/18",
        "geometry": "Mapped equivalent-area circles for both surveys. Fitted elevated circles are available for the 2022 survey.",
        "scope": "Height distributions for measurement review. Geometry bases can include different sets of returns.",
        "profile_rows": len(rows),
    }, indent=2).encode())
    print(f"{len(rows):,} LiDAR elevation profiles across {len({row[0] for row in rows})} contours")


if __name__ == "__main__":
    main()
