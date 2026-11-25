"""Match published USGS and NOAA lidar project footprints to the Cushing registry."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlencode

from pyproj import Transformer
from shapely import make_valid, transform
from shapely.geometry import Point, box, shape

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, TNM, get, write

STEM = "cushing_lidar_archive_coverage"
STORE = RAW / "usgs" / STEM
USGS = "https://index.nationalmap.gov/arcgis/rest/services/3DEPElevationIndex/MapServer"
NOAA = "https://maps.coast.noaa.gov/arcgis/rest/services/USInteragencyElevationInventory/USIEIv2/MapServer"
LAYERS = {"usgs_lidar_point_cloud": (USGS, 8), "noaa_topographic": (NOAA, 2),
          "noaa_topobathymetric": (NOAA, 0), "noaa_ifsar_insar": (NOAA, 3)}
PROJECT = Transformer.from_crs(4326, 32614, always_xy=True).transform
INVERSE = Transformer.from_crs(32614, 4326, always_xy=True).transform


def dated_ms(value):
    return datetime.fromtimestamp(value/1000, timezone.utc).date().isoformat() if value is not None else None


def catalog(name, url, refresh):
    path = STORE / f"{name}.json"
    raw = get(url, path, refresh=refresh)
    receipt = dict(catalog=name, source_url=url, local_file=str(path.relative_to(ROOT)),
                   sha256=hashlib.sha256(raw).hexdigest(), source_bytes=len(raw))
    return json.loads(raw), receipt


def catalog_layer(name, layer, bounds, refresh):
    base, number = layer
    query = dict(where="1=1", geometry=",".join(f"{x:.9f}" for x in bounds), geometryType="esriGeometryEnvelope",
                 inSR=4326, outSR=4326, spatialRel="esriSpatialRelIntersects")
    url = f"{base}/{number}/query?"
    result, receipt = catalog(name, url+urlencode(query | dict(f="geojson", outFields="*", returnGeometry="true")), refresh)
    count, count_receipt = catalog(f"{name}_count", url+urlencode(query | dict(f="json", returnCountOnly="true")), refresh)
    if result.get("type") != "FeatureCollection" or len(result["features"]) != count.get("count"):
        raise ValueError(f"Incomplete inventory for {name}")
    return result["features"], receipt, count_receipt


def project_rows(source, features, roofs):
    rows, overlaps = [], []
    for f in features:
        p = f["properties"]
        polygon = transform(make_valid(shape(f["geometry"])), PROJECT, interleaved=False)
        if polygon.is_empty or polygon.area <= 0:
            raise ValueError(f"Missing polygon for {source}/{p['OBJECTID']}")
        selected = [(r, circ, polygon.intersection(circ).area/circ.area) for r,circ in roofs if polygon.intersects(circ)]
        usgs = source == "usgs_lidar_point_cloud"
        name = p["workunit"] if usgs else p["Title"]
        first, last = (dated_ms(p[k]) for k in ("collect_start", "collect_end")) if usgs else (None,None)
        pub = dated_ms(p["lpc_pub_date"]) if usgs else None
        rows.append(dict(source=source, object_id=p["OBJECTID"], name=name, project=p.get("project"),
                         capture_first_day=first, capture_last_day=last, published_lpc_day=pub,
                         noaa_display_capture=p.get("collectiondate"), quality=p.get("ql",p.get("qualitylevel")),
                         horizontal_crs=p.get("horiz_crs",p.get("horizontaldatum")), vertical_crs=p.get("vert_crs",p.get("verticaldatum")),
                         source_point_cloud_link=p.get("lpc_link"), provider_status=p.get("Status"),
                         intersected_roofs=sum(frac>0 for _,_,frac in selected),
                         fully_covered_roofs=sum(frac>=.999999 for _,_,frac in selected),
                         fully_covered_efr_roofs=sum(frac>=.999999 and r["documented_roof_type"]=="EFR" for r,_,frac in selected)))
        for r, circ, fraction in selected:
            if fraction <= 0:
                continue
            overlaps.append(dict(source=source, object_id=p["OBJECTID"], name=name, candidate_id=r["candidate_id"],
                terminal_id=r["documented_terminal_id"], documented_roof_type=r["documented_roof_type"],
                centroid_inside=polygon.covers(circ.centroid), footprint_fraction=min(fraction, 1.),
                full_footprint_inside=fraction>=.999999, capture_first_day=first, capture_last_day=last, published_lpc_day=pub))
    return rows, overlaps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    roofs = [(r,Point(float(r["footprint_east_m"]),float(r["footprint_north_m"])).buffer(float(r["footprint_diameter_m"])/2))
             for r in read("tank_engineering_registry")]
    extent = box(min(c.bounds[0] for _,c in roofs),min(c.bounds[1] for _,c in roofs),
                 max(c.bounds[2] for _,c in roofs),max(c.bounds[3] for _,c in roofs)).buffer(100)
    bounds = transform(extent, INVERSE, interleaved=False).bounds
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda name:catalog_layer(name, LAYERS[name], bounds, args.refresh), LAYERS))
    rows, overlap, receipts = [], [], []
    for name, (features, receipt, count_receipt) in zip(LAYERS, responses, strict=True):
        inventory, matches = project_rows(name, features, roofs)
        rows.extend(inventory)
        overlap.extend(matches)
        receipts.extend((receipt,count_receipt))
    tnm_url = TNM+"?"+urlencode(dict(bbox=",".join(map(str,bounds)),datasets="Lidar Point Cloud (LPC)",max=1000))
    tiles, receipt = catalog("tnm_lpc_tiles", tnm_url, args.refresh)
    if tiles.get("total") != len(tiles["items"]) or tiles.get("errors"):
        raise ValueError("Incomplete National Map tile result")
    receipts.append(receipt)
    tile_rows = []
    for r in tiles["items"]:
        b = r["boundingBox"]
        envelope = transform(box(b["minX"],b["minY"],b["maxX"],b["maxY"]),PROJECT,interleaved=False)
        hit = [roof["candidate_id"] for roof, circ in roofs if envelope.intersects(circ)]
        url = r["downloadURL"]
        tile_rows.append(dict(source_id=r["sourceId"], workunit=url.split("/")[-3], title=r["title"],
                              tile_bbox_roof_count=len(hit),tile_bbox_candidate_ids=json.dumps(hit),
                              publication_date=r["publicationDate"], file_url=url))
    export(f"{STEM}_projects", rows)
    export(f"{STEM}_roofs", overlap)
    export(f"{STEM}_sources", receipts)
    export(f"{STEM}_tiles", tile_rows)
    published = [r for r in rows if r["source"]=="usgs_lidar_point_cloud" and r["intersected_roofs"]]
    report = dict(registry_roofs=len(roofs),search_bounds=bounds,projects_by_source=Counter(r["source"] for r in rows),
                  usgs_work_units_with_roof_intersection=[r["name"] for r in published],
                  most_recent_catalog_capture_end=max((r["capture_last_day"] for r in published if r["capture_last_day"]),default=None),
                  national_map_tile_records=len(tile_rows), national_map_tile_bbox_hits=Counter(r["workunit"] for r in tile_rows if r["tile_bbox_roof_count"]),
                  scope="Published project polygons and per-tile bounds. Project flight dates and coverage do not prove "
                        "roof returns or exact per-tank acquisition time. NOAA may list the same projects as USGS.")
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
