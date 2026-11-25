"""Measure tank-level coverage on the exact local Sentinel-2 L1C products."""
import argparse
import csv
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from contextlib import ExitStack

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.errors import RasterioIOError
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, from_bounds
from shapely import make_valid
from shapely.geometry import Polygon, box, shape

from build_scene_quality import grid, grid_candidates
from build_sentinel2_angles import observation_rows
from collect_public import RAW, ROOT, write
from collect_s2_l1c import CHANNELS, OUT as L1C, source_metadata_bytes, source_metadata_path

DERIVED = ROOT / "data" / "derived"
GML = "{http://www.opengis.net/gml/3.2}"
CLOUD_TYPES = ("OPAQUE", "CIRRUS", "SNOW_ICE")
SOURCE_SPEC = "https://sentinels.copernicus.eu/documents/d/sentinel/sentinel-2-products-specification-document-15_1"
CHECK_FIELDS = ("scene_item_id", "capture_year", "source_directory", "status", "tank_rows", "detail")
QUALITY_FIELDS = (
    "candidate_id", "scene_item_id", "captured_at_utc", "l1c_product_name",
    "optical_touched_cells_expected", "optical_cells_valid_in_all_bands",
    "cloud_mask_basis", "opaque_roof_cells", "cirrus_roof_cells", "snow_ice_roof_cells",
    "context_valid_cells", "context_opaque_cells", "context_cirrus_cells", "context_snow_ice_cells",
    "red_detector_mask_basis", "red_detector_covered_roof_cells", "red_detector_ambiguous_roof_cells",
    "red_detector_center_ids_json", "red_detector_with_view_options_json",
    "sun_zenith_deg", "sun_azimuth_grid_deg", "source_directory", "tile_metadata_path",
)


def gml_groups(path, reference, is_detector, detector_band="B04"):
    root = ET.fromstring(source_metadata_bytes(path))
    groups = defaultdict(list)
    extent = box(*reference.bounds)
    envelope = root.find(f".//{GML}Envelope")
    if envelope is not None:
        epsg = envelope.get("srsName", "").rsplit(":", 1)[-1]
        if epsg != str(reference.crs.to_epsg()):
            raise ValueError(f"Mask and optical raster use different CRS: {path}")

    def ring(element):
        pos = element.find(f".//{GML}posList")
        width = int(pos.get("srsDimension", "2"))
        return np.fromstring(pos.text, sep=" ").reshape(-1, width)[:, :2]

    for feature in root.findall(".//{*}MaskFeature"):
        label = feature.findtext("{*}maskType")
        if is_detector:
            identity = re.fullmatch(rf"detector_footprint-{detector_band}-(\d{{2}})-\d+", feature.get(f"{GML}id", ""))
            if label != "DETECTOR_FOOTPRINT" or identity is None or not 1 <= int(identity[1]) <= 12:
                raise ValueError(f"Unexpected {detector_band} detector mask feature in {path}")
            label = int(identity[1])
        elif label not in CLOUD_TYPES:
            raise ValueError(f"Unknown L1C cloud mask label {label} in {path}")
        for poly in feature.findall(f".//{GML}Polygon"):
            geom = Polygon(ring(poly.find(f"{GML}exterior")), [ring(hole) for hole in poly.findall(f"{GML}interior")])
            if extent.intersects(geom.envelope):
                groups[label].append(geom if geom.is_valid else make_valid(geom))
    return groups, bool(root.findall(".//{*}MaskFeature"))


def paint(shapes, reference):
    if not shapes:
        return np.zeros(reference.shape, dtype="uint8")
    return rasterize(shapes, out_shape=reference.shape, transform=reference.transform, dtype="uint8")


def native_mask(path, reference, metadata, bands):
    with rasterio.open(path) as source:
        if source.count != bands:
            raise ValueError(f"Expected {bands} mask bands in {path}, found {source.count}")
        geocoding = metadata.find(".//Tile_Geocoding")
        crs = geocoding.findtext("HORIZONTAL_CS_CODE")
        size = next(node for node in geocoding.findall("Size")
                    if (int(node.findtext("NROWS")), int(node.findtext("NCOLS"))) == source.shape)
        position = geocoding.find(f"Geoposition[@resolution='{size.get('resolution')}']")
        source_grid = Affine(float(position.findtext("XDIM")), 0, float(position.findtext("ULX")),
                             0, float(position.findtext("YDIM")), float(position.findtext("ULY")))
        if source.crs is not None and (source.crs != crs or not source.transform.almost_equals(source_grid)):
            raise ValueError(f"Mask grid disagrees with the tile XML in {path}")
        if reference.crs != crs:
            raise ValueError(f"Mask CRS disagrees with the optical CRS in {path}")
        bounds = from_bounds(*reference.bounds, source_grid)
        left, top = max(0, math.floor(bounds.col_off)), max(0, math.floor(bounds.row_off))
        right = min(source.width, math.ceil(bounds.col_off + bounds.width))
        bottom = min(source.height, math.ceil(bounds.row_off + bounds.height))
        window = Window(left, top, right - left, bottom - top)
        values = source.read(window=window)
    target = np.zeros((bands, *reference.shape), dtype=values.dtype)
    reproject(values, target, src_transform=source_grid * Affine.translation(left, top), src_crs=crs,
              dst_transform=reference.transform, dst_crs=reference.crs, resampling=Resampling.nearest)
    return target


def provider_masks(directory, external_detector, reference, metadata):
    clouds = [None, None, None]
    if (path := source_metadata_path(directory / "MSK_CLOUDS_B00.gml")).is_file():
        groups, _ = gml_groups(path, reference, False)
        clouds[:2] = [paint(groups.get(label, []), reference).ravel() for label in CLOUD_TYPES[:2]]
        cloud_basis = "gml_opaque_cirrus"
    elif (path := directory / "MSK_CLASSI_B00.jp2").is_file():
        values = native_mask(path, reference, metadata, 3)
        if values.max() > 1:
            raise ValueError(f"Unexpected non-binary L1C classification mask in {path}")
        clouds = [band.ravel() for band in values]
        cloud_basis = "jp2_opaque_cirrus_snow_ice"
    else:
        cloud_basis = "missing"

    bits = np.zeros(reference.shape, dtype="uint16")
    path = source_metadata_path(directory / "MSK_DETFOO_B04.gml")
    if not path.is_file() and external_detector is not None:
        path = external_detector
    if path.is_file():
        groups, has_features = gml_groups(path, reference, True)
        for detector, polygons in groups.items():
            bits |= paint(polygons, reference).astype("uint16") << (detector - 1)
        detector_basis = "gml" if has_features else "empty_source_mask"
    elif (path := directory / "MSK_DETFOO_B04.jp2").is_file():
        codes = native_mask(path, reference, metadata, 1)[0]
        if codes.max() > 12:
            raise ValueError(f"Unexpected B04 detector ID in {path}")
        valid = codes > 0
        bits[valid] = np.left_shift(np.uint16(1), codes[valid].astype("uint16") - 1)
        detector_basis = "jp2"
    else:
        detector_basis = "missing"
    return clouds, cloud_basis, bits.ravel(), detector_basis


def scene_rows(manifest, candidates, points, cache, detector):
    item = json.loads(manifest.read_bytes())
    scene, captured = item["canonical_l2a_id"], item["captured_at_utc"]
    directory = manifest.parent
    xml = source_metadata_path(directory / "MTD_TL.xml")
    metadata = ET.fromstring(source_metadata_bytes(xml))
    with ExitStack() as stack:
        bands = [stack.enter_context(rasterio.open(directory / f"{band}.tif")) for band in CHANNELS]
        reference = bands[0]
        raster_grid = grid(reference)
        if any(grid(band) != raster_grid for band in bands[1:]):
            raise ValueError(f"L1C band grids differ in {directory}")
        if raster_grid not in cache:
            cache[raster_grid] = grid_candidates(reference, candidates)
        roofs = cache[raster_grid]
        if not roofs:
            return
        optical = np.ones(reference.shape, dtype=bool)
        for band in bands:
            optical &= (band.read_masks(1) != 0) & (band.read(1) != 0)
        optical = optical.ravel()
        clouds, cloud_basis, bits, detector_basis = provider_masks(directory, detector, reference, metadata)
        angles = observation_rows(scene, tuple(roofs), points, xml, root=metadata)
        for angle in angles:
            identity = angle[1]
            expected, roof, context = roofs[identity]
            roof, context = roof[optical[roof]], context[optical[context]]
            row, col = reference.index(angle[3], angle[4])
            flags = int(bits[row * reference.width + col]) if 0 <= row < reference.height and 0 <= col < reference.width else 0
            detectors = [detector_id for detector_id in range(1, 13) if flags & (1 << (detector_id - 1))]
            views = [view for view in json.loads(angle[11]) if view[0] in detectors]
            covered = bits[roof]
            roof_counts = [int(np.count_nonzero(band[roof])) if band is not None else None for band in clouds]
            context_counts = [int(np.count_nonzero(band[context])) if band is not None else None for band in clouds]
            yield (identity, scene, captured, item["product_name"], expected, len(roof), cloud_basis, *roof_counts,
                   len(context), *context_counts, detector_basis, int(np.count_nonzero(covered)),
                   int(np.count_nonzero(covered & (covered - 1))), json.dumps(detectors, separators=(",", ":")),
                   json.dumps(views, separators=(",", ":")), angle[5], angle[7], str(directory.relative_to(ROOT)), angle[14])


def candidate_layer(label):
    features = json.loads((DERIVED / f"{label}_tank_candidates.geojson").read_bytes())["features"]
    return [(item["properties"]["candidate_id"], shape(item["geometry"])) for item in features]


def retain_other_years(output, checks_path, writer, checker, years, counts, granules, statuses):
    kept_counts = Counter()
    with output.open(newline="") as source:
        rows = csv.reader(source)
        if tuple(next(rows)) != QUALITY_FIELDS:
            raise ValueError("Existing quality table has different columns. Rebuild all years.")
        for row in rows:
            year = row[2][:4]
            if year not in years:
                writer.writerow(row)
                kept_counts[year] += 1
    with checks_path.open(newline="") as source:
        rows = csv.reader(source)
        if tuple(next(rows)) != CHECK_FIELDS:
            raise ValueError("Existing file checks have different columns. Rebuild all years.")
        for row in rows:
            _, year, _, status, n_rows, _ = row
            if year not in years:
                checker.writerow(row)
                counts[year] += int(n_rows)
                granules[year] += 1
                statuses[status] += 1
    if kept_counts != counts:
        raise ValueError("Existing quality rows disagree with file checks. Rebuild all years.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", action="append", type=int, help="Rescreen this year and keep the existing results for other years")
    args = parser.parse_args()
    years = {str(year) for year in args.year or ()}
    os.environ.setdefault("OPJ_NUM_THREADS", "1")
    core = candidate_layer("osm")
    supplemental = candidate_layer("supplemental")
    points = {identity: (geom.centroid.x, geom.centroid.y) for identity, geom in core + supplemental}
    roots = (
        (L1C / "granules", core),
        (L1C / "supplemental" / "granules", supplemental),
        (L1C / "combined_2019_2023" / "granules", core + supplemental),
        (L1C / "combined_2024_2026" / "granules", core + supplemental),
        (L1C / "alternate_aws" / "granules", core + supplemental),
    )
    early_detectors = {}
    with (DERIVED / "sentinel2_l1c_early_detector_footprints.csv").open() as stream:
        for row in csv.DictReader(stream):
            early_detectors[row["scene_item_id"]] = ROOT / row["local_file"]
    output = DERIVED / "sentinel2_l1c_tank_quality.csv"
    temporary = output.with_suffix(".part")
    counts = Counter()
    granules = Counter()
    checks_path = DERIVED / "sentinel2_l1c_local_file_checks.csv"
    checks_temp = checks_path.with_suffix(".part")
    statuses = Counter()
    if years and (not output.is_file() or not checks_path.is_file()):
        parser.error("A complete first build is required before rescreening individual years")
    processed = 0
    with temporary.open("w", newline="") as stream, checks_temp.open("w", newline="") as checks:
        writer = csv.writer(stream)
        writer.writerow(QUALITY_FIELDS)
        checker = csv.writer(checks)
        checker.writerow(CHECK_FIELDS)
        if years:
            retain_other_years(output, checks_path, writer, checker, years, counts, granules, statuses)
        for root, candidates in roots:
            cache = {}
            for manifest in sorted(root.glob("*/*/item.json")):
                scene, year = manifest.parent.name, manifest.parent.parent.name
                if years and year not in years:
                    continue
                detector = early_detectors.get(scene)
                try:
                    rows = list(scene_rows(manifest, candidates, points, cache, detector))
                except (RasterioIOError, OSError, ET.ParseError, ValueError, StopIteration) as error:
                    rows = []
                    status, detail = "failed_local_read_or_validation", f"{type(error).__name__}: {error}"
                else:
                    status, detail = ("screened" if rows else "no_candidate_cells"), ""
                statuses[status] += 1
                writer.writerows(rows)
                checker.writerow((scene, year, str(manifest.parent.relative_to(ROOT)), status, len(rows), detail))
                counts[year] += len(rows)
                granules[year] += 1
                processed += 1
                if processed % 100 == 0:
                    print(f"{processed} L1C local granule folders indexed in this run", flush=True)
        if years - set(granules):
            raise ValueError(f"No local manifests for selected years: {sorted(years - set(granules))}")
    temporary.replace(output)
    checks_temp.replace(checks_path)
    write(output.with_suffix(".json"), json.dumps({"rows_by_capture_year": counts, "folders_by_capture_year": granules,
        "local_folder_validation": statuses,
        "cloud_specification": SOURCE_SPEC,
        "roof_counts": "Optical roof cells touched by the footprint on the native 10 m grid. Counts use the subset with all four valid nonzero bands. GML sampled at pixel centers; JP2 reprojected nearest neighbor. Null means that mask category was unavailable.",
        "detectors": "B04 footprint mask at the tank centroid, plus complete B04 angle stencils from the same local L1C granule XML. GML overlapping detector polygons are retained.",
        "scope": "Local downloaded granules indexed in this run. Cloud categories are provider masks, and the counts do not establish that a roof height can be retrieved."}, indent=2).encode())
    print(f"{sum(counts.values())} L1C tank-granule rows from {sum(granules.values())} local folders")


if __name__ == "__main__":
    main()
