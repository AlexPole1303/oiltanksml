"""Interpolate official Sentinel-2 solar and red-band viewing grids at tank centroids."""
import argparse
import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from pyproj import Geod, Transformer
from shapely.geometry import shape

from collect_public import RAW, ROOT, write

DERIVED = ROOT / "data" / "derived"
GEOD = Geod(ellps="WGS84")
SPEC = "https://sentiwiki.copernicus.eu/web/s2-processing"


def array(grid, angle):
    part = grid.find(angle)
    values = np.array([[float(value) for value in row.text.split()] for row in part.findall("Values_List/VALUES")])
    if values.shape != (23, 23) or (part.findtext("COL_STEP"), part.findtext("ROW_STEP")) != ("5000", "5000"):
        raise ValueError("Unexpected Sentinel-2 angle grid size or spacing")
    return values


def stencil(column, row):
    c, r = np.floor(column).astype(int), np.floor(row).astype(int)
    if ((c < 0) | (c >= 22) | (r < 0) | (r >= 22)).any():
        raise ValueError("A tank centroid lies outside the Sentinel-2 angle mesh")
    x, y = column - c, row - r
    return (np.array([r, r, r + 1, r + 1]), np.array([c, c + 1, c, c + 1]),
            np.array([(1 - x) * (1 - y), x * (1 - y), (1 - x) * y, x * y]))


def interpolate(grid, angle, weights, circular=False):
    rows, columns, fractions = weights
    values = array(grid, angle)[rows, columns]
    if circular:
        radians = np.radians(values)
        return np.degrees(np.arctan2((fractions * np.sin(radians)).sum(axis=0),
                                     (fractions * np.cos(radians)).sum(axis=0))) % 360
    return (fractions * values).sum(axis=0)


@lru_cache(maxsize=None)
def ground_coordinates(crs, centers):
    transform = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    lon, lat = np.asarray(centers).T
    east, north = transform.transform(lon, lat)
    true_north_lon, true_north_lat, _ = GEOD.fwd(lon, lat, np.zeros_like(lon), np.full_like(lon, 100))
    north_east, north_north = transform.transform(true_north_lon, true_north_lat)
    true_north_bearing = np.degrees(np.arctan2(north_east - east, north_north - north))
    return east, north, true_north_bearing


def observation_rows(scene, ids, coordinates, path, *, root=None):
    if root is None:
        root = ET.parse(path).getroot()
    position = root.find(".//Tile_Geocoding/Geoposition[@resolution='10']")
    crs = root.findtext(".//Tile_Geocoding/HORIZONTAL_CS_CODE")
    left, top = float(position.findtext("ULX")), float(position.findtext("ULY"))
    east, north, true_north = ground_coordinates(crs, tuple(coordinates[identity] for identity in ids))
    weights = stencil((east - left) / 5000, (top - north) / 5000)
    solar = root.find(".//Sun_Angles_Grid")
    sun_zenith, sun_azimuth = interpolate(solar, "Zenith", weights), interpolate(solar, "Azimuth", weights, True)
    if not ((sun_zenith > 0) & (sun_zenith < 90) & np.isfinite(sun_azimuth)).all():
        raise ValueError(f"Invalid daytime solar geometry in {scene}")
    mean_solar = root.find(".//Mean_Sun_Angle")
    mean_zenith = float(mean_solar.findtext("ZENITH_ANGLE"))
    view = root.find(".//Mean_Viewing_Incidence_Angle[@bandId='3']")
    view_mean = (float(view.findtext("ZENITH_ANGLE")), float(view.findtext("AZIMUTH_ANGLE"))) if view is not None else (None, None)
    looks = defaultdict(list)
    for grid in root.findall(".//Viewing_Incidence_Angles_Grids[@bandId='3']"):
        detector = int(grid.attrib["detectorId"])
        zenith, azimuth = interpolate(grid, "Zenith", weights), interpolate(grid, "Azimuth", weights, True)
        for i in np.flatnonzero(np.isfinite(zenith) & np.isfinite(azimuth)):
            looks[i].append((detector, round(float(zenith[i]), 5), round(float(azimuth[i]), 5)))
    relative = path.relative_to(ROOT)
    for i, identity in enumerate(ids):
        options = sorted(looks[i])
        yield (scene, identity, crs, round(float(east[i]), 3), round(float(north[i]), 3),
               round(float(sun_zenith[i]), 5), round(float(sun_azimuth[i]), 5),
               round(float((sun_azimuth[i] + true_north[i]) % 360), 5), mean_zenith,
               round(float(true_north[i]), 5), len(options), json.dumps(options, separators=(",", ":")),
               *view_mean, str(relative))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplemental", action="store_true")
    args = parser.parse_args()
    layer = "supplemental_tank_candidates" if args.supplemental else "osm_tank_candidates"
    quality = "sentinel2_supplemental_tank_scene_quality" if args.supplemental else "sentinel2_tank_scene_quality"
    output = DERIVED / ("sentinel2_supplemental_angles.csv" if args.supplemental else "sentinel2_angles.csv")
    candidates = json.loads((DERIVED / f"{layer}.geojson").read_bytes())["features"]
    points = {item["properties"]["candidate_id"]: shape(item["geometry"]).centroid for item in candidates}
    coordinates = {key: (point.x, point.y) for key, point in points.items()}
    scenes = defaultdict(list)
    with (DERIVED / f"{quality}.csv").open() as stream:
        for row in csv.DictReader(stream):
            scenes[row["scene_item_id"]].append(row["candidate_id"])
    temporary = output.with_suffix(".part")
    fields = ("scene_item_id", "candidate_id", "crs", "center_east_m", "center_north_m", "sun_zenith_deg",
              "sun_azimuth_true_deg", "sun_azimuth_grid_deg", "tile_mean_sun_zenith_deg", "true_north_grid_bearing_deg",
              "red_view_options", "red_view_detector_zenith_azimuth_true_json", "tile_mean_red_view_zenith_deg",
              "tile_mean_red_view_azimuth_true_deg", "source_xml")
    counts = Counter()
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        for scene, ids in sorted(scenes.items()):
            path = RAW / "sentinel2_metadata" / scene / "granule_metadata.xml"
            for row in observation_rows(scene, sorted(ids), coordinates, path):
                counts[row[10]] += 1
                writer.writerow(row)
    temporary.replace(output)
    write(output.with_suffix(".json"), json.dumps({"granules": len(scenes), "candidate_granule_rows": sum(counts.values()),
        "red_detector_option_counts": dict(sorted(counts.items())), "source_specification": SPEC,
        "method": "Bilinear zenith and circular azimuth interpolation at contour centroids on native 5 km metadata grids. "
        "Red band uses bandId 3. Complete detector stencils are retained separately. Multiple options remain ambiguous "
        "until the detector-footprint mask resolves them. Tile-wide means are provided as separate metadata. "
        "Grid bearings account for local meridian convergence."}, indent=2).encode())
    print(f"{sum(counts.values())} angle rows from {len(scenes)} granules; B04 view option counts {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
