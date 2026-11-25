"""Index orthophoto pixels available at the mapped tank centroids."""
import argparse
import csv
import io
import json
from collections import Counter

import numpy as np
import rasterio
from rasterio.warp import transform
from shapely.geometry import shape

from collect_public import RAW, ROOT, write

DERIVED = ROOT / "data" / "derived"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplemental", action="store_true")
    args = parser.parse_args()
    source = DERIVED / ("supplemental_tank_candidates.geojson" if args.supplemental else "osm_tank_candidates.geojson")
    raster_store = "naip_supplemental" if args.supplemental else "naip_tanks"
    coverage_file = "naip_supplemental_tank_image_coverage.csv" if args.supplemental else "naip_tank_image_coverage.csv"
    candidates = json.loads(source.read_bytes())["features"]
    tanks = [(item["properties"]["candidate_id"], shape(item["geometry"]).centroid)
             for item in candidates]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("candidate_id", "acquisition_date", "item_id", "asset_path", "pixel_width_m", "coverage"))
    observations = set()
    for path in sorted((RAW / raster_store / "naip").rglob("item.json")):
        meta = json.loads(path.read_bytes())
        date = meta["properties"]["datetime"][:10]
        image = path.parent / "image.tif"
        area = shape(meta["geometry"])
        selected = [(identity, point) for identity, point in tanks if area.covers(point)]
        if not selected:
            continue
        with rasterio.open(image) as dataset:
            longitude, latitude = zip(*[(point.x, point.y) for _, point in selected])
            east, north = transform("EPSG:4326", dataset.crs, longitude, latitude)
            samples = dataset.sample(zip(east, north), indexes=1, masked=True)
            for (identity, _), value in zip(selected, samples, strict=True):
                if not np.ma.is_masked(value):
                    writer.writerow([identity, date, meta["id"], str(image.relative_to(ROOT)),
                                     abs(dataset.res[0]), "pixel_valid"])
                    observations.add((identity, date[:4]))
    write(DERIVED / coverage_file, output.getvalue().encode())
    print(dict(sorted(Counter(year for _, year in observations).items())))


if __name__ == "__main__":
    main()
