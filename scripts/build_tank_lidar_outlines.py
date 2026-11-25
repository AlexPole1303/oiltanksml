"""Fit circular upper-elevation outlines as independent geometry observations."""
import csv
import json
import math
from collections import defaultdict

import laspy
import numpy as np
from pyproj import Transformer
from shapely import transform
from shapely.geometry import box, shape
from shapely.strtree import STRtree

from audit_lidar_tanks import DATASET, REPORT, SURVEYS, as_utc
from collect_public import RAW, ROOT, write, write_csv

DERIVED = ROOT / "data" / "derived"
BBL_PER_CUBIC_METRE = 1 / 0.158987294928
SECTORS = 72
FOOTPRINT_FIELDS = ("candidate_id", "center_east_m", "center_north_m", "area_m2", "perimeter_m",
                    "equivalent_diameter_m", "minimum_rectangle_short_m", "minimum_rectangle_long_m",
                    "circularity", "volume_gradient_bbl_per_m", "source_file")
OUTLINE_FIELDS = ("candidate_id", "dataset", "first_upper_return_at_utc", "last_upper_return_at_utc",
                  "prior_rim_z_p95_native", "prior_ground_z_p50_native", "upper_points", "sectors_present",
                  "center_east_m", "center_north_m", "diameter_m", "center_shift_m", "sectors_within_055m",
                  "radial_residual_median_m", "radial_residual_p90_m", "fit_status", "source_directory")


def footprint_data(layer):
    path = DERIVED / f"{layer}_tank_candidates.geojson"
    features = json.loads(path.read_bytes())["features"]
    to_utm = Transformer.from_crs(4326, 32614, always_xy=True)
    rows = []
    for f in features:
        identity = f["properties"]["candidate_id"]
        poly = transform(shape(f["geometry"]), to_utm.transform, interleaved=False)
        if poly.is_empty or not poly.is_valid or poly.area <= 0:
            raise ValueError(f"Invalid footprint for {identity}")
        x, y = poly.centroid.coords[0]
        diameter = 2 * math.sqrt(poly.area / math.pi)
        rectangle = np.asarray(poly.minimum_rotated_rectangle.exterior.coords)
        sides = np.linalg.norm(np.diff(rectangle, axis=0), axis=1)
        rows.append((identity, x, y, poly.area, poly.length, diameter, float(sides.min()), float(sides.max()),
                     4 * math.pi * poly.area / poly.length**2, poly.area * BBL_PER_CUBIC_METRE,
                     str(path.relative_to(ROOT))))
    return rows


def elevation_priors(path):
    samples = defaultdict(list)
    with path.open() as file:
        for row in csv.DictReader(file):
            if row["rim_z_p95"] and row["ground_z_p50"]:
                samples[row["candidate_id"]].append((float(row["rim_z_p95"]), float(row["ground_z_p50"])))
    return {identity: np.median(values, axis=0) for identity, values in samples.items()}


def upper_points(footprints, priors, folder):
    utm = np.array([row[1:3] for row in footprints])
    to_native = Transformer.from_crs(32614, 3857, always_xy=True)
    to_utm = Transformer.from_crs(3857, 32614, always_xy=True)
    center_native = np.column_stack(to_native.transform(*utm.T))
    search = np.array([row[5] / 2 + 12 for row in footprints])
    extents = [box(x - radius * 1.3, y - radius * 1.3, x + radius * 1.3, y + radius * 1.3)
               for (x, y), radius in zip(center_native, search, strict=True)]
    tree = STRtree(extents)
    selected = defaultdict(list)
    for path in sorted(folder.glob("*.laz")):
        with laspy.open(path) as reader:
            header = reader.header
            if header.parse_crs().to_epsg() != 3857:
                raise ValueError(f"Unexpected LiDAR CRS in {path}")
            x0, y0, _ = header.mins
            x1, y1, _ = header.maxs
            indices = tree.query(box(x0, y0, x1, y1), predicate="intersects")
            if not len(indices):
                continue
            raw = reader.read()
        eligible = ~np.asarray(raw.synthetic, dtype=bool) & ~np.asarray(raw.withheld, dtype=bool)
        eligible &= ~np.isin(np.asarray(raw.classification), [2, 7, 18])
        eligible &= np.asarray(raw.return_number) == 1
        x, y = to_utm.transform(np.asarray(raw.x)[eligible], np.asarray(raw.y)[eligible])
        z, gps = np.asarray(raw.z)[eligible], np.asarray(raw.gps_time)[eligible]
        for i in indices:
            identity, _, _, _, _, diameter, *_ = footprints[i]
            if identity not in priors:
                continue
            rim, ground = priors[identity]
            dx, dy = x - utm[i, 0], y - utm[i, 1]
            radial = np.hypot(dx, dy)
            mask = (radial > diameter * 0.29) & (radial < search[i])
            mask &= (z > max(rim - 1.0, ground + 1.0)) & (z < rim + 2.0)
            if mask.any():
                selected[i].append(np.column_stack((dx[mask], dy[mask], gps[mask])))
    return selected


def circle(points):
    matrix = np.column_stack((2 * points[:, 0], 2 * points[:, 1], np.ones(len(points))))
    x, y, c = np.linalg.lstsq(matrix, (points[:, :2]**2).sum(axis=1), rcond=None)[0]
    squared = c + x*x + y*y
    if squared <= 0:
        raise ValueError("Nonpositive fitted circle radius")
    return np.array((x, y)), math.sqrt(squared)


def fit_upper_outline(points):
    radial = np.linalg.norm(points[:, :2], axis=1)
    points = points[np.argsort(radial)]
    azimuth = np.mod(np.arctan2(points[:, 1], points[:, 0]), 2 * math.pi)
    sectors = (azimuth * SECTORS / (2 * math.pi)).astype(int)
    envelope = points[[np.flatnonzero(sectors == index)[-1] for index in np.unique(sectors)]]
    if len(envelope) < 24:
        return len(envelope), None
    center, radius = circle(envelope)
    for _ in range(5):
        errors = abs(np.linalg.norm(envelope[:, :2] - center, axis=1) - radius)
        keep = errors <= np.clip(np.median(errors) * 3, 0.55, 2.0)
        if np.count_nonzero(keep) < 24:
            break
        center, radius = circle(envelope[keep])
    errors = abs(np.linalg.norm(envelope[:, :2] - center, axis=1) - radius)
    median, p90 = np.percentile(errors, (50, 90))
    return len(envelope), (center, 2 * radius, int(np.count_nonzero(errors <= 0.55)), float(median), float(p90))


def outline_rows(footprints, priors, folder, survey=SURVEYS[DATASET]):
    points = upper_points(footprints, priors, folder)
    local = str(folder.relative_to(ROOT))
    for i, (identity, east, north, *_) in enumerate(footprints):
        rim, ground = priors.get(identity, (None, None))
        if i not in points:
            yield (identity, survey.dataset, None, None, rim, ground, 0, 0, *([None] * 7), "no_upper_returns", local)
            continue
        raw = np.concatenate(points[i])
        dates = tuple(as_utc(value, survey.gps_to_utc_seconds).isoformat()
                      for value in (raw[:, 2].min(),raw[:, 2].max()))
        sectors, fit = fit_upper_outline(raw)
        if fit is None:
            yield (identity, survey.dataset, *dates, rim, ground, len(raw), sectors, *([None] * 7), "limited_angular_support", local)
            continue
        center, diameter, close, median, p90 = fit
        status = "well_supported_circular_fit" if sectors >= 60 and close >= 48 else "circular_fit_with_residuals"
        yield (identity, survey.dataset, *dates, rim, ground, len(raw), sectors, east + center[0], north + center[1],
               diameter, float(np.linalg.norm(center)), close, median, p90, status, local)


def main():
    footprints, outlines = [], []
    for layer, source, priors_name in (("osm", "ept", "lidar_tank_observations"),
                                       ("supplemental", "ept_supplemental", "lidar_supplemental_tank_observations")):
        polygons = footprint_data(layer)
        priors = elevation_priors(DERIVED / f"{priors_name}.csv")
        folder = RAW / "usgs" / source / DATASET / "nodes"
        footprints.extend(polygons)
        outlines.extend(outline_rows(polygons, priors, folder))
    write_csv(DERIVED / "tank_footprint_geometry.csv", FOOTPRINT_FIELDS, sorted(footprints))
    write_csv(DERIVED / "tank_lidar_outlines.csv", OUTLINE_FIELDS, sorted(outlines))
    write(DERIVED / "tank_lidar_outlines.json", json.dumps({
        "dataset": DATASET, "source_project_report": REPORT, "horizontal_crs": "EPSG:32614",
        "method": "First nonground returns near the previous rim P95, from the clipped USGS EPT point cloud. Fit a circle to the outermost point per 5-degree sector, with iterative radial outlier rejection.",
        "interpretation": "Elevated circular envelope, which can include exterior appurtenances or a dome. Fitted diameter is an independent observation with residuals, and a permit shell or liquid diameter has its own definition.",
        "footprint_scale": "Projected footprint area times 6.289810770 cubic-metre-to-barrel factor. This is a dimensional scenario per metre of height change.",
        "candidate_count": len(footprints), "circles_fit": sum(row[10] is not None for row in outlines),
    }, indent=2).encode())
    print(f"{len(footprints)} footprints, {sum(row[10] is not None for row in outlines)} fitted elevated outlines")


if __name__ == "__main__":
    main()
