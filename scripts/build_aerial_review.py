"""Render a dated aerial review atlas on common per-tank image windows."""
import argparse
import csv
import io
import json
import math
from collections import defaultdict
from contextlib import ExitStack

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely import transform
from shapely.geometry import shape

from collect_public import ROOT, write

DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "aerial_review"
YEARS = ("2010", "2013", "2015", "2017", "2019", "2021", "2023", "2025")
CELL = 144
SIZE = 136
TITLE_FONT = ImageFont.load_default(size=15)
LABEL_FONT = ImageFont.load_default(size=12)
UTM = "EPSG:32614"
NODATA = 256


def sources_by_tank(candidates, supplemental):
    sources = defaultdict(list)
    tables = [("naip_supplemental_tank_image_coverage.csv" if supplemental else "naip_tank_image_coverage.csv",
               "acquisition_date", "item_id")]
    if supplemental:
        for item in candidates:
            props = item["properties"]
            for date, tile in zip(props["native_tile_majority_dates"].split("|"),
                                  props["native_tile_ids"].split("|"), strict=True):
                if tile:
                    relative = f"data/raw/ok_aerial_2025/tiles/{tile}/image.tif"
                    sources[props["candidate_id"], "2025"].append((date, tile, relative))
    else:
        tables.append(("ok_aerial_tank_image_coverage.csv", "tile_majority_date", "tile_id"))
    for table, date_field, item_field in tables:
        with (DERIVED / table).open() as stream:
            for row in csv.DictReader(stream):
                date = row[date_field]
                sources[row["candidate_id"], date[:4]].append((date, row[item_field], row["asset_path"]))
    return sources


def draw_crop(dataset, bounds, target):
    native = transform_bounds(UTM, dataset.crs, *bounds)
    requested = from_bounds(*native, dataset.transform)
    left, top = max(0, math.floor(requested.col_off)), max(0, math.floor(requested.row_off))
    right = min(dataset.width, math.ceil(requested.col_off + requested.width))
    bottom = min(dataset.height, math.ceil(requested.row_off + requested.height))
    if right <= left or bottom <= top:
        return
    window = Window(left, top, right - left, bottom - top)
    raster = dataset.read((1, 2, 3), window=window).astype(np.uint16)
    raster[:, dataset.read_masks(1, window=window) == 0] = NODATA
    result = np.full_like(target, NODATA)
    reproject(raster, result, src_transform=dataset.window_transform(window), src_crs=dataset.crs,
              dst_transform=rasterio.transform.from_bounds(*bounds, SIZE, SIZE), dst_crs=UTM,
              src_nodata=NODATA, dst_nodata=NODATA, resampling=Resampling.average)
    valid = (result < NODATA).all(axis=0) & (target == NODATA).all(axis=0)
    target[:, valid] = result[:, valid]


def save_sheet(image, path):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    write(path, buffer.getvalue())


def overview(year, output):
    rows = [row for row in csv.DictReader((output / "index.csv").open()) if row["year"] == year]
    if not rows:
        raise ValueError(f"No indexed imagery for {year}")
    dense = year != "2017"
    columns, row_count, cell, height, size = (10, 12, 98, 103, 89) if dense else (7, 6, 144, 161, 136)
    page_size = columns * row_count
    year_index = YEARS.index(year)
    for offset in range(0, len(rows), page_size):
        page_number = offset // page_size + 1
        canvas = Image.new("RGB", (columns * cell, 29 + row_count * height), "#16191d")
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 6), f"CUSHING   |   {year}   |   {page_number:02d}", font=TITLE_FONT, fill="white")
        for index, row in enumerate(rows[offset:offset + page_size]):
            y, x = divmod(index, columns)
            source_x = 142 + year_index * CELL
            source_y = 29 + (int(row["history_row"]) - 1) * 153
            with Image.open(ROOT / row["history_sheet"]) as sheet:
                crop = sheet.crop((source_x, source_y, source_x + SIZE, source_y + SIZE))
            if dense:
                crop = crop.resize((size, size), Image.Resampling.LANCZOS)
            x, y = 4 + x * cell, 29 + y * height
            canvas.paste(crop, (x, y))
            label = f'{int(row["review_number"]):03d}   {row["acquisition_dates"][5:10]}'
            draw.text((x, y + size), label, font=LABEL_FONT, fill="white")
        save_sheet(canvas, output / f"naip_{year}" / f"{page_number:02d}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overview", choices=YEARS[:-1], action="append")
    parser.add_argument("--supplemental", action="store_true")
    args = parser.parse_args()
    output = OUT / "supplemental" if args.supplemental else OUT
    if args.overview:
        for year in args.overview:
            overview(year, output)
        return
    source_name = "supplemental_tank_candidates.geojson" if args.supplemental else "osm_tank_candidates.geojson"
    candidates = json.loads((DERIVED / source_name).read_bytes())["features"]
    candidates.sort(key=lambda item: item["properties"]["candidate_id"])
    sources = sources_by_tank(candidates, args.supplemental)
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    index = io.StringIO(newline="")
    writer = csv.writer(index)
    writer.writerow(("review_number", "candidate_id", "year", "acquisition_dates", "source_items", "source_paths",
                     "history_sheet", "history_row", "latest_sheet", "latest_row", "latest_column",
                     "equivalent_diameter_m", "crop_width_m", "valid_display_pixels"))
    history_rows = 8
    latest_rows, latest_cols = 6, 7
    history_row_height, latest_row_height = 153, 161
    history_page = latest_page = None
    cache = {}
    with ExitStack() as stack:
        for offset, item in enumerate(candidates):
            properties = item["properties"]
            identity = properties["candidate_id"]
            polygon = transform(shape(item["geometry"]), to_utm.transform, interleaved=False)
            xmin, ymin, xmax, ymax = polygon.bounds
            half = max(xmax - xmin, ymax - ymin) * 0.62 + 7
            center = polygon.centroid
            bounds = center.x - half, center.y - half, center.x + half, center.y + half
            number = offset + 1
            page, row = divmod(offset, history_rows)
            hist_path = output / "history" / f"{page + 1:02d}.png"
            latest_index, position = divmod(offset, latest_rows * latest_cols)
            latest_row, latest_col = divmod(position, latest_cols)
            latest_path = output / "state_2025" / f"{latest_index + 1:02d}.png"
            if row == 0:
                history_page = Image.new("RGB", (140 + len(YEARS) * CELL, 29 + history_rows * history_row_height), "#16191d")
                history_draw = ImageDraw.Draw(history_page)
                history_draw.text((9, 6), f"CUSHING {page + 1:02d}", font=TITLE_FONT, fill="white")
                for col, year in enumerate(YEARS):
                    history_draw.text((145 + col * CELL, 6), year, font=TITLE_FONT, fill="white")
            if position == 0:
                latest_page = Image.new("RGB", (latest_cols * CELL, 29 + latest_rows * latest_row_height), "#16191d")
                latest_draw = ImageDraw.Draw(latest_page)
                latest_draw.text((6, 6), f"OKLAHOMA RGB 15 CM   |   CUSHING   |   {latest_index + 1:02d}",
                                 font=TITLE_FONT, fill="white")
            y = 29 + row * history_row_height
            diameter = properties.get("equivalent_diameter_m", properties.get("approx_diameter_m"))
            history_draw.text((9, y + 10), f"{number:03d}  /  {diameter:.1f} m", font=TITLE_FONT, fill="white")
            review_id = str(properties.get("osm_way_id", identity.removeprefix("ortho-")))
            history_draw.text((9, y + 32), review_id, font=LABEL_FONT, fill="#b8c3cd")
            for col, year in enumerate(YEARS):
                tile = np.full((3, SIZE, SIZE), NODATA, dtype=np.uint16)
                used = []
                for date, tile_id, relative in sorted(sources[identity, year]):
                    if relative not in cache:
                        cache[relative] = stack.enter_context(rasterio.open(ROOT / relative))
                    before = np.count_nonzero(tile[0] < NODATA)
                    draw_crop(cache[relative], bounds, tile)
                    if np.count_nonzero(tile[0] < NODATA) > before:
                        used.append((date, tile_id, relative))
                dates = "|".join(sorted({values[0] for values in used}))
                valid = int(np.count_nonzero(tile[0] < NODATA))
                tile[tile == NODATA] = 30
                crop = Image.fromarray(np.moveaxis(tile.astype(np.uint8), 0, -1))
                x = 142 + col * CELL
                history_page.paste(crop, (x, y))
                label = dates[:10] + (" +" if len(dates) > 10 else "") if dates else "no pixels"
                history_draw.text((x, y + SIZE), label, font=LABEL_FONT, fill="#b8c3cd")
                if year == "2025":
                    x_latest, y_latest = 4 + latest_col * CELL, 29 + latest_row * latest_row_height
                    latest_page.paste(crop, (x_latest, y_latest))
                    latest_draw.text((x_latest, y_latest + SIZE), f"{number:03d}   {diameter:.1f} m   {label[5:]}",
                                     font=LABEL_FONT, fill="white")
                writer.writerow((number, identity, year, dates, "|".join(value[1] for value in used),
                                 "|".join(value[2] for value in used), str(hist_path.relative_to(ROOT)), row + 1,
                                 str(latest_path.relative_to(ROOT)) if year == "2025" else None,
                                 latest_row + 1 if year == "2025" else None, latest_col + 1 if year == "2025" else None,
                                 diameter, round(2 * half, 2), valid))
            if row + 1 == history_rows or number == len(candidates):
                save_sheet(history_page, hist_path)
            if position + 1 == latest_rows * latest_cols or number == len(candidates):
                save_sheet(latest_page, latest_path)
                print(f"{number}/{len(candidates)} contours rendered", flush=True)
    write(output / "index.csv", index.getvalue().encode())


if __name__ == "__main__":
    main()
