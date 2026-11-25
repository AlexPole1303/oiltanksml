"""Extract mapped storage tanks into a QGIS-ready candidate layer."""
import argparse
import json
import math
from collections import Counter

import osmium
from rasterio.warp import transform
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union

from collect_imagery import BOUNDS, IMAGERY
from collect_public import RAW, ROOT, write


class Tanks(osmium.SimpleHandler):
    def __init__(self, area):
        super().__init__()
        self.features = []
        self.aoi = area

    def way(self, way):
        if way.tags.get("man_made") != "storage_tank" or not way.is_closed():
            return
        if not all(node.location.valid() for node in way.nodes):
            return
        coordinates = [(node.lon, node.lat) for node in way.nodes]
        if len(coordinates) < 4:
            return
        polygon = Polygon(coordinates)
        if not polygon.is_valid or not self.aoi.intersects(polygon):
            return
        x, y = transform("EPSG:4326", "EPSG:32614", *zip(*coordinates))
        area = Polygon(zip(x, y)).area
        self.features.append({
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "candidate_id": f"osm-way-{way.id}",
                "osm_way_id": way.id,
                "osm_edit_timestamp": str(way.timestamp),
                "area_m2": round(area, 2),
                "equivalent_diameter_m": round(2 * math.sqrt(area / math.pi), 2),
                "product_tag": way.tags.get("content"),
                "operator_tag": way.tags.get("operator"),
                "tank_type_tag": way.tags.get("storage_tank"),
                "source": f"https://www.openstreetmap.org/way/{way.id}",
                "validation_status": "unverified",
            },
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eia-counties", action="store_true")
    args = parser.parse_args()
    pbf = IMAGERY / "osm" / "oklahoma-latest.osm.pbf"
    derived = ROOT / "data" / "derived"
    core = derived / "osm_tank_candidates.geojson"
    counties = []
    if args.eia_counties:
        county_file = RAW / "documents" / "eia_803_counties.geojson"
        counties = [(item["properties"], shape(item["geometry"]))
                    for item in json.loads(county_file.read_bytes())["features"]]
        core_ids = {item["properties"]["candidate_id"] for item in json.loads(core.read_bytes())["features"]}
    handler = Tanks(unary_union([geometry for _, geometry in counties]) if counties else box(*BOUNDS))
    handler.apply_file(str(pbf), locations=True, idx="flex_mem")
    if not handler.features:
        raise ValueError("No tank candidates found in the AOI")
    if counties:
        for feature in handler.features:
            center = shape(feature["geometry"]).centroid
            matches = [props for props, area in counties if area.covers(center)]
            if len(matches) != 1:
                raise ValueError(f"Tank centroid county is ambiguous: {feature['properties']['candidate_id']}")
            props = feature["properties"]
            props["county"] = matches[0]["NAME"]
            props["county_fips"] = matches[0]["GEOID"]
            props["inside_existing_core_inventory"] = props["candidate_id"] in core_ids
    payload = {
        "type": "FeatureCollection",
        "attribution": "© OpenStreetMap contributors, ODbL",
        "license": "https://www.openstreetmap.org/copyright",
        "extract": "https://download.geofabrik.de/north-america/us/oklahoma.html",
        "features": handler.features,
    }
    output = derived / "eia_803_osm_tank_candidates.geojson" if counties else core
    write(output, json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
    print(len(handler.features), "candidates")
    print(Counter(feature["properties"]["product_tag"] for feature in handler.features))
    if counties:
        print(Counter((feature["properties"]["county"], feature["properties"]["inside_existing_core_inventory"])
                      for feature in handler.features))


if __name__ == "__main__":
    main()
