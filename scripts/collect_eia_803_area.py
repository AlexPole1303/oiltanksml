"""Archive county boundaries and an EPA facility snapshot for the EIA-803 search area."""
import argparse
import datetime as dt
import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

from collect_public import RAW, get, write

DOCUMENTS = RAW / "documents"
CENSUS = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query"
EPA = "https://echogeo.epa.gov/arcgis/rest/services/ECHO/Facilities/MapServer/0/query"
FIPS = ("40037", "40081", "40119")
FIELDS = ("OBJECTID", "REGISTRY_ID", "FAC_NAME", "FAC_CITY", "FAC_COUNTY", "FAC_FIPS_CODE",
          "FAC_ACTIVE_FLAG", "AIR_IDS", "CAA_NAICS", "FAC_NAICS_CODES", "DFR_URL", "FAC_LAT", "FAC_LONG")
NAICS = re.compile(r"\b(?:48611[0]?|42471[0]?|424720)\b")
NAME = re.compile(r"\b(?:TANK\s+FAR\w*|TANK\s+BATT\w*|TANK\s+STATION|CRUDE|TERML?|TERMINAL|PIPELINE)\b", re.I)


def query(base, **params):
    url = f"{base}?{urlencode(params)}"
    payload = json.loads(get(url))
    if "error" in payload or payload.get("exceededTransferLimit"):
        raise ValueError(f"Incomplete response from {base}: {payload.get('error')}")
    return url, payload


def save(name, payload):
    write(DOCUMENTS / name, json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())


def epa_batch(ids):
    return query(EPA, f="geojson", objectIds=",".join(map(str, ids)), outFields=",".join(FIELDS),
                 returnGeometry="true", outSR=4326)


def screening_reason(properties):
    reasons = []
    if NAICS.search(" ".join(properties[key] or "" for key in ("CAA_NAICS", "FAC_NAICS_CODES"))):
        reasons.append("naics")
    if NAME.search(properties["FAC_NAME"] or ""):
        reasons.append("facility_name")
    return "|".join(reasons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    fips_filter = "(" + ",".join(f"'{code}'" for code in FIPS) + ")"
    if args.refresh or not (DOCUMENTS / "eia_803_counties.geojson").is_file():
        url, boundaries = query(CENSUS, f="geojson", where=f"GEOID IN {fips_filter}",
                                outFields="GEOID,NAME,STATE,COUNTY", outSR=4326, returnGeometry="true")
        if {feature["properties"]["GEOID"] for feature in boundaries["features"]} != set(FIPS):
            raise ValueError("Census response does not match EIA county FIPS")
        save("eia_803_counties.geojson", boundaries)
        write(DOCUMENTS / "eia_803_counties.source", (url + "\n").encode())
    if args.refresh or not (DOCUMENTS / "epa_echo_eia_803_counties.geojson").is_file():
        ids_url, index = query(EPA, f="json", where=f"FAC_STATE='OK' AND FAC_FIPS_CODE IN {fips_filter}",
                               returnIdsOnly="true")
        ids = sorted(index["objectIds"])
        batches = [ids[i:i + 75] for i in range(0, len(ids), 75)]
        with ThreadPoolExecutor(max_workers=5) as workers:
            pages = list(workers.map(epa_batch, batches))
        facilities = [feature for _, page in pages for feature in page["features"]]
        if sorted(feature["properties"]["OBJECTID"] for feature in facilities) != ids:
            raise ValueError("EPA responses do not match the initial object ID list")
        save("epa_echo_eia_803_counties.geojson", {"type": "FeatureCollection", "features": facilities,
                                                   "downloaded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                                   "source_urls": [ids_url] + [url for url, _ in pages]})
        print(len(facilities), "EPA facility records")


if __name__ == "__main__":
    main()
