"""Archive Sentinel-1 products and screen their acquisition and catalog clocks."""
import json
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime
from urllib.parse import urlencode

from shapely import make_valid
from shapely.geometry import Point, shape

from collect_public import BBOX, RAW, ROOT, get, write, write_csv

CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOCS = "https://documentation.dataspace.copernicus.eu/APIs/OData.html"
STORE = RAW / "sentinel1_cushing_catalog"
DERIVED = ROOT / "data/derived"
SLC = "IW_SLC__1S"
GRD = "IW_GRDH_1S"
COG = "IW_GRDH_1S-COG"
CENTRE = Point(-96.74, 35.96)


def annual_catalog(year):
    west, south, east, north = BBOX.split(",")
    polygon = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    start = "2016-12-01" if year == 2016 else f"{year}-01-01"
    end = "2026-11-01" if year == 2026 else f"{year + 1}-01-01"
    query = (f"Collection/Name eq 'SENTINEL-1' and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') "
             f"and ContentDate/Start ge {start}T00:00:00.000Z and ContentDate/Start lt {end}T00:00:00.000Z "
             "and (contains(Name,'_IW_SLC__1S') or contains(Name,'_IW_GRDH_1S'))")
    url = CATALOG + "?" + urlencode({"$filter": query, "$expand": "Attributes", "$count": "true",
                                        "$orderby": "ContentDate/Start asc", "$top": 500})
    rows = []
    count = None
    page = 1
    while url:
        path = STORE / f"{year}_{page:02d}.json"
        raw = get(url, path=path)
        result = json.loads(raw)
        if count is None:
            count = result["@odata.count"]
        rows.extend((r, str(path.relative_to(ROOT))) for r in result["value"])
        url = result.get("@odata.nextLink")
        if url and not url.startswith(CATALOG + "?"):
            raise ValueError("Unexpected host in the Copernicus pagination link")
        page += 1
    if len(rows) != count or len({row["Id"] for row, _ in rows}) != count:
        raise ValueError(f"Incomplete Copernicus catalog for {year}")
    print(f"Sentinel-1 {year}: {count} product versions", flush=True)
    return rows


def product_rows(products):
    with closing(sqlite3.connect(f"{(DERIVED / 'cushing.sqlite').as_uri()}?mode=ro", uri=True)) as db:
        centres = [(cid, Point(lon, lat)) for cid, lon, lat in db.execute(
            "SELECT candidate_id,longitude,latitude FROM tank_engineering_registry ORDER BY candidate_id")]
    for item, path in products:
        attributes = {row["Name"]: row["Value"] for row in item["Attributes"]}
        kind = attributes["productType"]
        if kind not in {SLC, GRD, COG}:
            raise ValueError(f"Unexpected Sentinel-1 product type {kind}")
        footprint = make_valid(shape(item["GeoFootprint"]))
        present = [cid for cid, point in centres if footprint.covers(point)]
        yield dict(product_id=item["Id"], product_name=item["Name"], product_type=kind,
            satellite=item["Name"].split("_")[0], relative_orbit=int(attributes["relativeOrbitNumber"]),
            absolute_orbit=int(attributes["orbitNumber"]), direction=attributes["orbitDirection"],
            polarization=attributes.get("polarisationChannels"), capture_start_utc=item["ContentDate"]["Start"],
            capture_end_utc=item["ContentDate"]["End"], catalog_publication_utc=item["PublicationDate"],
            catalog_modification_utc=item["ModificationDate"], online_at_query=item["Online"],
            bytes=item["ContentLength"], covers_cushing_reference_point=footprint.covers(CENTRE),
            covered_registry_centroids=len(present), covered_candidate_ids_json=json.dumps(present, separators=(",", ":")),
            source_file=path, catalog_url=f"{CATALOG}({item['Id']})")


def overlap_weeks(products):
    candidates = [(p, *(datetime.fromisoformat(p[f]) for f in ("capture_start_utc", "capture_end_utc", "catalog_publication_utc")))
                  for p in products if p["covers_cushing_reference_point"] and p["product_type"] in {SLC, GRD}]
    with closing(sqlite3.connect(f"{(DERIVED / 'cushing.sqlite').as_uri()}?mode=ro", uri=True)) as db:
        windows = db.execute("""SELECT week_ending,study_phase,endpoint,window_starts_after_utc,
            window_ends_by_utc,inputs_available_before_utc FROM eia_satellite_window ORDER BY week_ending,endpoint""").fetchall()
    matched = defaultdict(dict)
    for week, phase, endpoint, lower, upper, cutoff in windows:
        start, finish, ready = map(datetime.fromisoformat, (lower, upper, cutoff))
        matched[week, phase][endpoint] = [p for p, begin, end, published in candidates
                                         if begin > start and end <= finish and published < ready]
    for (week, phase), pair in matched.items():
        row = dict(week_ending=week, study_phase=phase)
        for kind, label in ((SLC, "slc"), (GRD, "grd")):
            previous, current = ([r for r in pair[side] if r["product_type"] == kind] for side in ("previous", "current"))
            tracks = [{(r["relative_orbit"], r["direction"]) for r in side} for side in (previous, current)]
            row.update({f"{label}_previous_products": len(previous), f"{label}_current_products": len(current),
                        f"{label}_both_windows": bool(previous and current),
                        f"{label}_common_relative_track": bool(tracks[0] & tracks[1]),
                        f"{label}_previous_product_ids_json": json.dumps([r["product_id"] for r in previous]),
                        f"{label}_current_product_ids_json": json.dumps([r["product_id"] for r in current])})
        yield row


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        source = [item for group in pool.map(annual_catalog, range(2016, 2027)) for item in group]
    products = sorted(product_rows(source), key=lambda r: (r["capture_start_utc"], r["product_id"]))
    weeks = list(overlap_weeks(products))
    for name, rows in (("s1_cushing_catalog", products), ("s1_cushing_weekly_catalog_coverage", weeks)):
        fields = tuple(rows[0])
        write_csv(DERIVED / f"{name}.csv", fields, ([r[f] for f in fields] for r in rows))
    coverage = {}
    for phase in sorted({r["study_phase"] for r in weeks}):
        period = [r for r in weeks if r["study_phase"] == phase]
        coverage[phase] = dict(weeks=len(period), **{key: sum(r[key] for r in period) for key in (
            "slc_both_windows", "slc_common_relative_track", "grd_both_windows", "grd_common_relative_track")})
    report = dict(products=len(products), by_product_type=Counter(r["product_type"] for r in products),
        product_versions_covering_cushing_reference_point=sum(r["covers_cushing_reference_point"] for r in products),
        phase_coverage=coverage, source=DOCS,
        interpretation="Metadata screening over the core footprint. Weekly screening uses a single Cushing reference "
        "point, original SLC/GRD product types, the recorded catalog PublicationDate and conservative report clock. "
        "The footprint fields concern scene geometry. No radar pixels, roof-height measurements, EIA outcomes or "
        "market outcomes are used. Catalog coverage alone does not establish availability for a usable roof fit.")
    write(DERIVED / "s1_cushing_catalog_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
