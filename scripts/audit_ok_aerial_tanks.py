"""Check how much of each mapped contour is covered by the 2025 aerial imagery."""
import csv
import io
import json

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import geometry_mask, geometry_window
from shapely import transform
from shapely.geometry import mapping, shape

from collect_naip_seamlines import candidates as mapped_tanks
from collect_ok_aerial import TARGET
from collect_public import ROOT, write

DERIVED = ROOT / "data" / "derived"


def merge_valid_pixels(target, target_transform, valid, transform):
    column = (transform.c - target_transform.c) / target_transform.a
    row = (transform.f - target_transform.f) / target_transform.e
    x, y = round(column), round(row)
    if not (abs(column - x) < 1e-5 and abs(row - y) < 1e-5):
        raise ValueError("State imagery tiles do not use the same 15 cm pixel grid")
    x0, x1 = max(0, x), min(target.shape[1], x + valid.shape[1])
    y0, y1 = max(0, y), min(target.shape[0], y + valid.shape[0])
    source = valid[y0 - y:y1 - y, x0 - x:x1 - x]
    if np.count_nonzero(source) != np.count_nonzero(valid):
        raise ValueError("A candidate's valid pixels fall outside its reference grid window")
    target[y0:y1, x0:x1] |= source


def main():
    projection = Transformer.from_crs("EPSG:4326", "EPSG:6350", always_xy=True)
    tanks = []
    for identity, polygon in mapped_tanks():
        native = transform(polygon, projection.transform, interleaved=False)
        tanks.append((identity, polygon, mapping(native)))
    tiles = {path.parent.name: path for path in (TARGET / "tank_clips").glob("*/item.json")}
    tiles.update({path.parent.name: path for path in (TARGET / "tiles").glob("*/item.json")})
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("candidate_id", "tile_id", "tile_majority_date", "tile_time_raw", "asset_path",
                     "candidate_pixels", "valid_pixels", "valid_fraction"))
    covered, fully_covered = set(), set()
    union = {}
    for tile_id, path in sorted(tiles.items()):
        item = json.loads(path.read_bytes())
        tile = shape(item["geometry"])
        candidates = [(identity, geometry) for identity, tank, geometry in tanks if tile.intersects(tank)]
        if not candidates:
            continue
        image = path.parent / "image.tif"
        with rasterio.open(image) as dataset:
            if dataset.crs.to_epsg() != 6350:
                raise ValueError(f"Unexpected CRS for Oklahoma aerial raster {image}")
            for identity, geometry in candidates:
                window = geometry_window(dataset, [geometry], boundless=True)
                image_transform = dataset.window_transform(window)
                region = geometry_mask([geometry], out_shape=(int(window.height), int(window.width)),
                                       transform=image_transform, invert=True)
                possible = int(np.count_nonzero(region))
                if possible == 0:
                    continue
                mask = dataset.read_masks(1, window=window, boundless=True) != 0
                valid = mask & region
                count = int(np.count_nonzero(valid))
                if count:
                    date = item["properties"]["majorityda"][:10]
                    writer.writerow((identity, tile_id, date,
                                     item["properties"].get("time"), str(image.relative_to(ROOT)),
                                     possible, count, round(count / possible, 6)))
                    covered.add(identity)
                    if count == possible:
                        fully_covered.add(identity)
                    if identity not in union:
                        union[identity] = (valid, image_transform, possible, {date})
                    else:
                        all_valid, target_transform, expected, dates = union[identity]
                        if expected != possible:
                            raise ValueError(f"Inconsistent native pixel count for {identity}")
                        merge_valid_pixels(all_valid, target_transform, valid, image_transform)
                        dates.add(date)
    write(DERIVED / "ok_aerial_tank_image_coverage.csv", stream.getvalue().encode())
    summary = io.StringIO(newline="")
    writer = csv.writer(summary)
    writer.writerow(("candidate_id", "candidate_pixels", "union_valid_pixels", "tile_majority_dates"))
    full_union = 0
    for identity, (valid, _, possible, dates) in sorted(union.items()):
        count = int(np.count_nonzero(valid))
        writer.writerow((identity, possible, count, "|".join(sorted(dates))))
        full_union += int(count == possible)
    write(DERIVED / "ok_aerial_tank_union_coverage.csv", summary.getvalue().encode())
    print(f"2025 state imagery: {len(covered)} candidates with pixels, {len(fully_covered)} in one tile, {full_union} fully covered across tiles")


if __name__ == "__main__":
    main()
