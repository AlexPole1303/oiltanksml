"""Build dated state-ortho previews for petroleum facilities outside the core cluster."""
import csv
import io
import json
import math
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyproj import Transformer
from shapely.geometry import shape

from collect_eia_803_area import screening_reason
from collect_public import RAW, ROOT, get, write

DERIVED = ROOT / "data" / "derived"
OUT = DERIVED / "eia_803_aerial_review"
WMS = "https://okmaps.org/geoserver/wms"
LAYER = "ogi:OK2025_RGB"
SIZE = 700
HALF_SPAN = 420
TO_STATE = Transformer.from_crs(4326, 6350, always_xy=True)


def render(params):
    number, properties, east, north, distance = params
    bbox = (east - HALF_SPAN, north - HALF_SPAN, east + HALF_SPAN, north + HALF_SPAN)
    url = WMS + "?" + urlencode({"SERVICE": "WMS", "REQUEST": "GetMap", "VERSION": "1.1.1",
                                  "LAYERS": LAYER, "STYLES": "", "SRS": "EPSG:6350",
                                  "BBOX": ",".join(f"{value:.3f}" for value in bbox),
                                  "WIDTH": SIZE, "HEIGHT": SIZE, "FORMAT": "image/png", "TRANSPARENT": "true"})
    path = RAW / "ok_aerial_2025" / "epa_facility_previews" / f"{properties['REGISTRY_ID']}.png"
    raw = get(url)
    with Image.open(io.BytesIO(raw)) as source:
        alpha = source.convert("RGBA").getchannel("A")
        coverage = 1 - alpha.histogram()[0] / SIZE ** 2
    write(path, raw)
    return (number, properties["REGISTRY_ID"], properties["FAC_NAME"], properties["FAC_FIPS_CODE"],
            properties["FAC_LAT"], properties["FAC_LONG"], east, north, round(distance, 1),
            round(coverage, 4), str(path.relative_to(ROOT)), url)


def main():
    tanks = json.loads((DERIVED / "osm_tank_candidates.geojson").read_bytes())["features"]
    centers = [shape(tank["geometry"]).centroid for tank in tanks]
    x, y = TO_STATE.transform([point.x for point in centers], [point.y for point in centers])
    features = json.loads((RAW / "documents" / "epa_echo_eia_803_counties.geojson").read_bytes())["features"]
    jobs = []
    for item in sorted(features, key=lambda feature: (feature["properties"]["FAC_FIPS_CODE"],
                                                     feature["properties"]["REGISTRY_ID"])):
        properties = item["properties"]
        lon, lat = properties["FAC_LONG"], properties["FAC_LAT"]
        if not screening_reason(properties) or lon is None or lat is None:
            continue
        east, north = TO_STATE.transform(lon, lat)
        distance = np.hypot(np.asarray(x) - east, np.asarray(y) - north).min()
        if math.isfinite(distance) and distance > 750:
            jobs.append((len(jobs) + 1, properties, east, north, distance))
    with ThreadPoolExecutor(max_workers=5) as workers:
        rows = list(workers.map(render, jobs))
    sheet_size, width, height = 6, 365, 390
    font = ImageFont.load_default(size=13)
    for offset in range(0, len(rows), sheet_size):
        canvas = Image.new("RGB", (width * 3, height * 2), "#181c21")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(rows[offset:offset + sheet_size]):
            top, left = divmod(i, 3)
            left, top = left * width + 5, top * height + 5
            with Image.open(ROOT / row[-2]) as raster:
                thumbnail = raster.convert("RGBA").resize((350, 350), Image.Resampling.LANCZOS)
                canvas.paste(thumbnail, (left, top), thumbnail)
            draw.ellipse((left + 171, top + 171, left + 179, top + 179), outline="#ff633e", width=2)
            draw.text((left, top + 351), f"{row[0]:02d}  |  {row[1]}  |  {row[2][:32]}", font=font, fill="white")
            draw.text((left, top + 368), row[2][32:85], font=font, fill="white")
        png = io.BytesIO()
        canvas.save(png, format="PNG")
        write(OUT / f"{offset // sheet_size + 1:02d}.png", png.getvalue())
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer)
    writer.writerow(("review_number", "registry_id", "facility_name", "reported_fips", "latitude", "longitude",
                     "center_easting_6350", "center_northing_6350", "nearest_core_tank_m", "valid_image_fraction",
                     "source_path", "source_url"))
    writer.writerows(rows)
    write(OUT / "index.csv", csv_buffer.getvalue().encode())
    print(len(rows), "facility previews", sum(row[-3] > 0 for row in rows), "with image coverage")


if __name__ == "__main__":
    main()
