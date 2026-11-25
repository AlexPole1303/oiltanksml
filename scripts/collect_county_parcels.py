"""Intersect mapped tank contours with the public assessor parcel services."""
import argparse
import csv
import datetime as dt
import io
import json
import subprocess
from urllib.parse import urlencode

from pyproj import Transformer
from shapely import make_valid, transform
from shapely.geometry import shape
from shapely.strtree import STRtree

from collect_public import RAW, ROOT, write

BASE = "https://pro.paynecountyassessor.org/arcgis/rest/services"
LINCOLN = ("https://www.efsedge.com/arcgisportalb/sharing/servers/88944d5727df49d683f9a4cd1c772265/"
           "rest/services/Lincoln_OK/Lincoln_OK_Vector/MapServer/11")
LINCOLN_PARCEL = "Lincoln_Oklahoma.DBO.Parcels."
LINCOLN_ASSESSOR = "Lincoln_Oklahoma.DBO.GIS_Out."
YEARS = range(2021, 2026)
FIELDS = ("OBJECTID", "ParcelID", "PropertyID", "YearID", "OwnerNameFormatted", "BusinessName",
          "LandUseCode", "SaleDate", "LegalReference")


def request(url, **params):
    location = url + "?" + urlencode(params)
    payload = subprocess.check_output(["/usr/bin/curl", "-fsS", "--retry", "2", "--max-time", "40", location])
    data = json.loads(payload)
    if "error" in data:
        raise ValueError(f"Assessor query failed at {url}: {data['error']}")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--county", choices=("payne", "lincoln"), default="payne")
    args = parser.parse_args()
    derived = ROOT / "data" / "derived"
    items = json.loads((derived / "osm_tank_candidates.geojson").read_bytes())["features"]
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    tanks = [transform(make_valid(shape(item["geometry"])), to_utm.transform, interleaved=False) for item in items]
    tree = STRtree(tanks)
    bounds = [shape(item["geometry"]).bounds for item in items]
    envelope = (min(b[0] for b in bounds), min(b[1] for b in bounds), max(b[2] for b in bounds), max(b[3] for b in bounds))
    query_bounds = ",".join(map(str, envelope))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("candidate_id", "snapshot_year", "object_id", "parcel_id", "property_id", "source_year_id",
                     "owner_name_recorded", "business_name_recorded", "land_use_code", "sale_date_raw",
                     "legal_reference", "tank_area_fraction", "source_url"))
    if args.county == "payne":
        snapshots = [(year, f"{BASE}/PayneParcels_{year}/MapServer/0") for year in YEARS]
        fields = FIELDS
        output = "tank_parcel_intersections.csv"
    else:
        snapshots = [(dt.datetime.now(dt.timezone.utc).year, LINCOLN)]
        fields = (LINCOLN_PARCEL + "OBJECTID", LINCOLN_PARCEL + "Parcel_ID", LINCOLN_ASSESSOR + "OWNERNAME",
                  LINCOLN_ASSESSOR + "PARCELTYPE", LINCOLN_ASSESSOR + "PARCELLGL")
        output = "tank_lincoln_parcel_intersections.csv"
    for year, layer_url in snapshots:
        query = layer_url + "/query"
        available = request(query, geometry=query_bounds, geometryType="esriGeometryEnvelope", inSR="4326",
                            spatialRel="esriSpatialRelIntersects", returnIdsOnly="true", f="json")
        ids = available["objectIds"]
        identity_field = available["objectIdFieldName"]
        if not ids:
            raise ValueError(f"No parcels returned in the study area for {year}")
        subset, matches, returned = [], set(), set()
        for offset in range(0, len(ids), 75):
            chunk = ids[offset:offset + 75]
            data = request(query, objectIds=",".join(map(str, chunk)), outFields=",".join(fields),
                           returnGeometry="true", outSR="4326", f="geojson")
            for item in data["features"]:
                properties = item["properties"]
                identity = properties[identity_field]
                if args.county == "payne":
                    attributes = [properties[name] for name in FIELDS[1:]]
                else:
                    _, parcel_id, owner, land_use, legal = [properties[name] for name in fields]
                    attributes = [parcel_id, None, None, owner, None, land_use, None, legal]
                returned.add(identity)
                geometry = transform(make_valid(shape(item["geometry"])), to_utm.transform, interleaved=False)
                intersections = []
                for index in tree.query(geometry, predicate="intersects"):
                    fraction = tanks[index].intersection(geometry).area / tanks[index].area
                    if fraction <= 0.0001:
                        continue
                    candidate = items[index]["properties"]["candidate_id"]
                    intersections.append(candidate)
                    matches.add(candidate)
                    writer.writerow((candidate, year, identity, *attributes, round(fraction, 6), layer_url))
                if intersections:
                    subset.append(item)
        if returned != set(ids):
            raise ValueError(f"Incomplete public parcel response for {year}: {len(returned)}/{len(ids)}")
        write(RAW / "documents" / f"{args.county}_parcels" / f"{year}.geojson", json.dumps({
            "type": "FeatureCollection", "features": subset, "source_layer": layer_url,
            "source_query_count": len(ids), "snapshot_year": year,
            "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }, ensure_ascii=False).encode())
        print(f"{args.county} {year}: {len(subset)} intersecting parcels, {len(matches)} candidate contours", flush=True)
    write(derived / output, stream.getvalue().encode())


if __name__ == "__main__":
    main()
