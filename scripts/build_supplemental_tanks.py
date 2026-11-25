"""Georeference reviewed round structures that lack a matching core OSM outline."""
import csv
import io
import json

import rasterio
from pyproj import Transformer
from shapely import transform
from shapely.geometry import Point, mapping, shape
from shapely.strtree import STRtree

from build_core_aerial_sweep import SIZE as ATLAS_SIZE
from build_eia_803_aerial_review import HALF_SPAN, SIZE as WMS_SIZE
from collect_public import RAW, ROOT, write

DERIVED = ROOT / "data" / "derived"
REFERENCE = ROOT / "data" / "reference"
STATE = RAW / "ok_aerial_2025"


def indexed(path, key):
    with path.open() as stream:
        return {row[key]: row for row in csv.DictReader(stream)}


def state_position(mark, tile):
    x, y, radius = (float(mark[field]) for field in ("pixel_x", "pixel_y", "radius_px"))
    column, row = int(tile["sheet_column"]) - 1, int(tile["sheet_row"]) - 1
    x = x - (6 + column * (ATLAS_SIZE + 6))
    y = y - (6 + row * (ATLAS_SIZE + 25))
    if not (0 <= x < ATLAS_SIZE and 0 <= y < ATLAS_SIZE):
        raise ValueError(f"Mark outside its review tile: {mark['candidate_id']}")
    with rasterio.open(ROOT / tile["source_path"]) as source:
        east, north = source.transform * (x * source.width / ATLAS_SIZE, y * source.height / ATLAS_SIZE)
        return Point(east, north), radius * source.width * source.res[0] / ATLAS_SIZE


def wms_position(mark, review):
    x, y, radius = (float(mark[field]) for field in ("pixel_x", "pixel_y", "radius_px"))
    if not (0 <= x < WMS_SIZE and 0 <= y < WMS_SIZE):
        raise ValueError(f"Mark outside its WMS preview: {mark['candidate_id']}")
    east, north = (float(review[field]) for field in ("center_easting_6350", "center_northing_6350"))
    pixel_size = 2 * HALF_SPAN / WMS_SIZE
    return Point(east - HALF_SPAN + x * pixel_size, north + HALF_SPAN - y * pixel_size), radius * pixel_size


def main():
    to_geo = Transformer.from_crs(6350, 4326, always_xy=True)
    to_state = Transformer.from_crs(4326, 6350, always_xy=True)
    core = json.loads((DERIVED / "osm_tank_candidates.geojson").read_bytes())["features"]
    core_tanks = [transform(shape(row["geometry"]), to_state.transform, interleaved=False) for row in core]
    sweep = indexed(DERIVED / "aerial_review" / "sweep_2025" / "index.csv", "tile_id")
    previews = indexed(DERIVED / "eia_803_aerial_review" / "index.csv", "registry_id")
    native = [(path.parent, json.loads(path.read_bytes())) for path in sorted((STATE / "tiles").glob("*/item.json"))]
    bounds = [shape(item["geometry"]) for _, item in native]
    raster_index = STRtree(bounds)
    marks = indexed(REFERENCE / "supplemental_tank_image_marks.csv", "candidate_id")
    features = []
    for identity, mark in marks.items():
        if mark["source_kind"] == "state_atlas":
            tile = sweep[mark["source_id"]]
            center, radius = state_position(mark, tile)
            image = tile["source_path"]
            review_path = tile["sheet"]
            source_url = json.loads((ROOT / image).with_name("item.json").read_bytes())["properties"]["ftp_tif"]
        else:
            review = previews[mark["source_id"]]
            center, radius = wms_position(mark, review)
            image = review["source_path"]
            review_path = str((DERIVED / "eia_803_aerial_review" /
                               f"{(int(review['review_number']) - 1) // 6 + 1:02d}.png").relative_to(ROOT))
            source_url = review["source_url"]
        circle = center.buffer(radius)
        overlap = max(tank.intersection(circle).area / circle.area for tank in core_tanks)
        if overlap > 0.15:
            raise ValueError(f"Candidate {identity} substantially overlaps the existing core ({overlap:.1%})")
        location = transform(center, to_geo.transform, interleaved=False)
        geographic = transform(circle, to_geo.transform, interleaved=False)
        matches = [native[i] for i in raster_index.query(location, predicate="intersects")]
        props = {
            "candidate_id": identity, "source_site": mark["source_site"], "appearance": mark["appearance"],
            "longitude": location.x, "latitude": location.y, "approx_diameter_m": round(2 * radius, 1),
            "core_overlap_fraction": round(overlap, 5), "image_path": image, "review_path": review_path,
            "native_tile_ids": "|".join(item["properties"]["strtile"] for _, item in matches),
            "native_tile_majority_dates": "|".join(item["properties"]["majorityda"][:10] for _, item in matches),
            "source_url": source_url, "validation_status": "visual_screen_only", "notes": mark["notes"],
        }
        features.append({"type": "Feature", "properties": props, "geometry": mapping(geographic)})
    write(DERIVED / "supplemental_tank_candidates.geojson",
          json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")).encode())
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=list(features[0]["properties"]))
    writer.writeheader()
    writer.writerows(item["properties"] for item in features)
    write(DERIVED / "supplemental_tank_candidates.csv", csv_buffer.getvalue().encode())
    print(len(features), "additional candidates", sum(bool(item["properties"]["native_tile_ids"]) for item in features),
          "with local original tile")


if __name__ == "__main__":
    main()
