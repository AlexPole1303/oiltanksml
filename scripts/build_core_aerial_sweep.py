"""Render 2025 orthophoto tiles with the mapped tank outlines for a visual sweep."""
import csv
import io
import json

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.enums import Resampling
from shapely import transform
from shapely.geometry import box, shape

from collect_public import RAW, ROOT, write

OUT = ROOT / "data" / "derived" / "aerial_review" / "sweep_2025"
SOURCE = RAW / "ok_aerial_2025"
SIZE = 590


def main():
    source_index = json.loads((SOURCE / "tile_index.geojson").read_bytes())["features"]
    tank_index = json.loads((OUT.parents[1] / "osm_tank_candidates.geojson").read_bytes())["features"]
    project = Transformer.from_crs(4326, 6350, always_xy=True)
    tanks = [transform(shape(row["geometry"]), project.transform, interleaved=False) for row in tank_index]
    tiles = sorted(row["properties"]["strtile"] for row in source_index)
    buffer = io.StringIO(newline="")
    rows = csv.writer(buffer)
    rows.writerow(("tile_id", "outlined_contours", "sheet", "sheet_row", "sheet_column", "source_path"))
    for offset in range(0, len(tiles), 4):
        margin, gap = 6, 25
        canvas = Image.new("RGB", (2 * (SIZE + margin) + margin, 2 * (SIZE + gap) + margin), "#1c2128")
        draw = ImageDraw.Draw(canvas)
        sheet = OUT / f"{offset // 4 + 1:02d}.png"
        for i, tile in enumerate(tiles[offset:offset + 4]):
            source = SOURCE / "tiles" / tile / "image.tif"
            with rasterio.open(source) as ds:
                pixels = ds.read((1, 2, 3), out_shape=(3, SIZE, SIZE), resampling=Resampling.average)
                transform_to_pixels = ~(ds.transform * rasterio.Affine.scale(ds.width / SIZE, ds.height / SIZE))
                candidates = [tank for tank in tanks if tank.intersects(box(*ds.bounds))]
            photo = Image.fromarray(np.moveaxis(pixels, 0, -1))
            overlay = ImageDraw.Draw(photo)
            for tank in candidates:
                overlay.line([transform_to_pixels * point for point in tank.exterior.coords],
                             fill="#00f4f0", width=1)
            r, c = divmod(i, 2)
            x, y = margin + c * (SIZE + margin), margin + r * (SIZE + gap)
            canvas.paste(photo, (x, y))
            draw.text((x, y + SIZE + 3), f"{tile}   |   {len(candidates)} mapped contours",
                      font=ImageFont.load_default(size=13), fill="white")
            rows.writerow((tile, len(candidates), sheet.relative_to(ROOT), r + 1, c + 1, source.relative_to(ROOT)))
        png = io.BytesIO()
        canvas.save(png, format="PNG")
        write(sheet, png.getvalue())
    write(OUT / "index.csv", buffer.getvalue().encode())
    print(len(tiles), "tiles in the visual sweep")


if __name__ == "__main__":
    main()
