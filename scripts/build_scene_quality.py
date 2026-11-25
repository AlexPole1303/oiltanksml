"""Index local Sentinel-2 optical and SCL coverage by candidate and granule."""
import argparse
import csv
import json
from contextlib import ExitStack

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask, geometry_window
from shapely import transform
from shapely.geometry import box, mapping, shape

from collect_imagery import BANDS, canonical_items
from collect_public import RAW, ROOT, write

DERIVED = ROOT / "data" / "derived"
OPTICAL = BANDS[:-1]
CONTEXT_EXCLUSION_CLASSES = (0, 1, 3, 8, 9, 10, 11)


def pixel_indices(dataset, region):
    geometry = [mapping(region)]
    window = geometry_window(dataset, geometry, boundless=True)
    included = geometry_mask(geometry, (int(window.height), int(window.width)),
                             dataset.window_transform(window), all_touched=True, invert=True)
    rows, columns = np.nonzero(included)
    rows += int(window.row_off)
    columns += int(window.col_off)
    present = ((rows >= 0) & (rows < dataset.height) & (columns >= 0) & (columns < dataset.width))
    return len(rows), rows[present] * dataset.width + columns[present]


def grid(dataset):
    return dataset.crs.to_wkt(), dataset.transform, dataset.shape


def grid_candidates(dataset, candidates):
    to_native = Transformer.from_crs("EPSG:4326", dataset.crs.to_string(), always_xy=True)
    coverage = box(*dataset.bounds)
    masks = {}
    for identity, geometry in candidates:
        roof = transform(geometry, to_native.transform, interleaved=False)
        if not coverage.intersects(roof):
            continue
        expected, indices = pixel_indices(dataset, roof)
        if not len(indices):
            continue
        context = roof.buffer(30).difference(roof)
        _, context_indices = pixel_indices(dataset, context)
        masks[identity] = (expected, indices, np.setdiff1d(context_indices, indices, assume_unique=True))
    return masks


def write_scene(writer, item, candidates, index, raster_store):
    captured = item["properties"]["datetime"]
    directory = raster_store / captured[:4] / item["id"]
    with ExitStack() as stack:
        bands = [stack.enter_context(rasterio.open(directory / f"{name}.tif")) for name in OPTICAL]
        scl = stack.enter_context(rasterio.open(directory / "scl.tif"))
        optical_grid, scl_grid = grid(bands[0]), grid(scl)
        if any(grid(dataset) != optical_grid for dataset in bands[1:]):
            raise ValueError(f"Optical bands use different grids in {item['id']}")
        for key, dataset in ((optical_grid, bands[0]), (scl_grid, scl)):
            if key not in index:
                index[key] = grid_candidates(dataset, candidates)
        valid_optical = bands[0].read_masks(1) != 0
        for band in bands[1:]:
            valid_optical &= band.read_masks(1) != 0
        valid_optical = valid_optical.ravel()
        classes = scl.read(1).ravel()
    if classes.min() < 0 or classes.max() > 11:
        raise ValueError(f"Unexpected Sentinel-2 scene classification value in {item['id']}")
    total = 0
    for identity, (expected, optical, _) in index[optical_grid].items():
        expected_scl, roof_scl, context_scl = index[scl_grid].get(identity, (0, [], []))
        labels = np.bincount(classes[roof_scl], minlength=12)
        context_labels = classes[context_scl]
        writer.writerow((identity, item["id"], captured, expected, np.count_nonzero(valid_optical[optical]),
                         expected_scl, *labels, len(context_scl),
                         np.count_nonzero(np.isin(context_labels, CONTEXT_EXCLUSION_CLASSES))))
        total += 1
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplemental", action="store_true")
    args = parser.parse_args()
    raster_store = RAW / ("sentinel2_supplemental" if args.supplemental else "imagery_core") / "sentinel2"
    catalog = (sorted((json.loads(path.read_bytes()) for path in raster_store.glob("*/*/item.json")),
                      key=lambda item: (item["properties"]["datetime"], item["id"])) if args.supplemental else canonical_items())
    if not catalog:
        raise ValueError(f"No complete granules in {raster_store}")
    source_name = "supplemental_tank_candidates.geojson" if args.supplemental else "osm_tank_candidates.geojson"
    tanks = json.loads((DERIVED / source_name).read_bytes())["features"]
    candidates = [(item["properties"]["candidate_id"], shape(item["geometry"])) for item in tanks]
    output = DERIVED / ("sentinel2_supplemental_tank_scene_quality.csv" if args.supplemental else
                        "sentinel2_tank_scene_quality.csv")
    temporary = output.with_suffix(".part")
    index = {}
    rows = 0
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("candidate_id", "scene_item_id", "captured_at_utc", "optical_touched_cells_expected",
                         "optical_cells_valid_in_all_bands", "scl_touched_cells_expected",
                         *[f"scl_class_{i}" for i in range(12)], "context_scl_cells",
                         "context_nodata_defect_cloud_shadow_snow_cells"))
        for number, item in enumerate(catalog, 1):
            rows += write_scene(writer, item, candidates, index, raster_store)
            if number % 250 == 0:
                print(f"{number}/{len(catalog)} Sentinel-2 granules indexed", flush=True)
    temporary.replace(output)
    metadata = {"source": "Sentinel-2 L2A / Collection 1 from Element 84 Earth Search", "granules": len(catalog),
                "contours": len(candidates), "rows": rows, "context_width_m": 30,
                "pixel_method": "All touched cells on the native raster grids. Invalid and outside-tile cells are excluded from the optical count. Outside-tile SCL cells do not contribute to any class count.",
                "context_exclusion_classes": CONTEXT_EXCLUSION_CLASSES,
                "caveat": "This table records local raster coverage and mask counts. Multiple granules can belong to one acquisition. SCL class 2 is kept separately and can include topographic shadow or dark features. Quality gates are applied by the physical analysis."}
    write(output.with_suffix(".json"), json.dumps(metadata, indent=2).encode())
    print(f"{rows} candidate-granule rows over {len(catalog)} granules")


if __name__ == "__main__":
    main()
