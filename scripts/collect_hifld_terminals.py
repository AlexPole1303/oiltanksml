"""Archive the 2025 HIFLD POL layer and screen points against the EIA-803 area."""
import datetime as dt
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

from pyproj import Transformer
from shapely import transform
from shapely.geometry import shape

from collect_public import RAW, ROOT, write, write_csv

LAYER = "https://mapservices.pasda.psu.edu/server/rest/services/pasda/HIFLD_FEMA/MapServer/126"
DERIVED = ROOT / "data" / "derived"
ARCHIVE = RAW / "documents" / "hifld_pol"
TO_METRES = Transformer.from_crs(4326, 32614, always_xy=True)


def request(path, url):
    # Use the system HTTPS trust store for the PASDA endpoint.
    response = path.read_bytes() if path.is_file() else subprocess.check_output(
        ["curl", "-fsSL", "--connect-timeout", "10", "--max-time", "45", "--retry", "2", url])
    document = json.loads(response)
    if "error" in document or document.get("exceededTransferLimit"):
        raise ValueError(f"Incomplete HIFLD response for {path.name}")
    if not path.is_file():
        write(path, response)
    return document


def batch(identities):
    params = {"f": "geojson", "objectIds": ",".join(map(str, identities)), "outFields": "*", "outSR": 4326}
    return request(ARCHIVE / f"objects_{identities[0]}_{identities[-1]}.geojson",
                   f"{LAYER}/query?{urlencode(params)}")["features"]


def day(value):
    return dt.datetime.fromtimestamp(value / 1000, dt.timezone.utc).date().isoformat() if value and value > 0 else None


def main():
    request(ARCHIVE / "layer.json", f"{LAYER}?f=json")
    index = request(ARCHIVE / "object_ids.json", f"{LAYER}/query?f=json&where=1%3D1&returnIdsOnly=true")
    field, ids = index["objectIdFieldName"], sorted(index["objectIds"])
    batches = [ids[i:i+200] for i in range(0, len(ids), 200)]
    with ThreadPoolExecutor(max_workers=4) as workers:
        features = [feature for page in workers.map(batch, batches) for feature in page]
    if sorted(f["properties"][field] for f in features) != ids:
        raise ValueError("HIFLD responses do not match the original object list")
    write(ARCHIVE / "national.geojson", json.dumps({"type": "FeatureCollection", "features": features},
                                                  separators=(",", ":")).encode())

    counties = json.loads((RAW / "documents" / "eia_803_counties.geojson").read_bytes())["features"]
    areas = [(f["properties"]["GEOID"], shape(f["geometry"])) for f in counties]
    candidates = [(f["properties"]["candidate_id"], transform(shape(f["geometry"]), TO_METRES.transform,
                                                            interleaved=False))
                  for layer in ("osm_tank_candidates", "supplemental_tank_candidates")
                  for f in json.loads((DERIVED / f"{layer}.geojson").read_bytes())["features"]]
    rows, regional = [], []
    for feature in features:
        p = feature["properties"]
        if feature["geometry"] is None:
            continue
        point = shape(feature["geometry"])
        fips = next((code for code, area in areas if area.covers(point)), None)
        if fips is None and p["COUNTYFIPS"] not in {code for code, _ in areas}:
            continue
        regional.append(feature)
        projected = transform(point, TO_METRES.transform, interleaved=False)
        distance, candidate = min((projected.distance(geometry), identity) for identity, geometry in candidates)
        rows.append((p[field], p["TERM_ID"], p["NAME"], p["CITY"], p["COUNTYFIPS"], fips,
                     point.x, point.y, p["TYPE"], p["COMMODITY"], p["CRUDE_OIL"], p["CAPACITY"],
                     p["STATUS"], p["OWNER"], p["OPERATOR"], day(p["SOURCEDATE"]), day(p["VAL_DATE"]),
                     candidate, round(distance, 2), LAYER))
    write(DERIVED / "hifld_pol_eia_counties.geojson", json.dumps(
        {"type": "FeatureCollection", "features": regional, "source_url": LAYER}, separators=(",", ":")).encode())
    write_csv(DERIVED / "hifld_pol_eia_counties.csv", ("object_id", "terminal_id", "name", "city", "reported_county_fips",
               "coordinate_county_fips", "longitude", "latitude", "reported_type", "commodity", "crude_oil",
               "reported_capacity", "status", "reported_owner", "reported_operator", "source_date",
               "validation_date", "nearest_tank_candidate", "nearest_tank_distance_m", "source_url"), sorted(rows))
    print(f"{len(features)} archived terminal records, {len(regional)} records in the EIA study counties")


if __name__ == "__main__":
    main()
