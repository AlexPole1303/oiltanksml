"""Check whether extra acquisitions cover the roofs in a frozen EIA forecast."""

import argparse
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import numpy as np
from rasterio.io import MemoryFile
from shapely import make_valid
from shapely.geometry import Point, Polygon, shape

from build_cushing_icesat2_controls import identities
from build_s1_reference_matches import DERIVED, export
from collect_cushing_s1_rtc import BANDS, read_headers
from collect_imagery import sign
from collect_public import RAW, ROOT, get, write
from fit_cushing_overpass_covariance import pass_id
from register_cushing_leading_models import checked_hashes

STEM = "cushing_forward_independent_passes"
CATALOGUES = {
    "sentinel2": ("https://stac.dataspace.copernicus.eu/v1/search","sentinel-2-l1c"),
    "landsat": ("https://landsatlook.usgs.gov/stac-server/search","landsat-c2l1"),
    "radar_rtc": ("https://planetarycomputer.microsoft.com/api/stac/v1/search","sentinel-1-rtc"),
    "radar_slc": ("https://stac.dataspace.copernicus.eu/v1/search","sentinel-1-slc"),
}
BBOX = "-97.0,35.55,-96.40,36.27"
ESRI = "https://landsat.arcgis.com/arcgis/rest/services/Landsat/Pan/ImageServer"
PLAN_PREFIX = "https://sentinels.copernicus.eu/documents/d/sentinel/"
PLANS = {
    "S2A":"s2a_mp_acq__kml_20261029t150000_20261116t180000",
    "S2B":"s2b_mp_acq__kml_20261105t120000_20261123t150000",
    "S2C":"s2c_mp_acq__kml_20261029t120000_20261116t150000",
}


def capture(item):
    return datetime.fromisoformat(item["properties"].get("start_datetime") or item["properties"]["datetime"])


def catalogue(name, start, stop, store):
    endpoint, collection = CATALOGUES[name]
    initial = endpoint+"?"+urlencode(dict(collections=collection,bbox=BBOX,
                                          datetime=f"{start.isoformat()}/{stop.isoformat()}",limit=100))
    url = initial
    items, sources, seen = {}, [], set()
    while url:
        if url in seen or not url.startswith(endpoint):
            raise ValueError(f"Unexpected pagination in {name}")
        seen.add(url)
        path = store/f"{name}_{len(seen):02d}.json"
        raw = get(url,path)
        q = json.loads(raw)
        features = q["features"]
        if q["type"] != "FeatureCollection" or q.get("numberReturned",len(features)) != len(features):
            raise ValueError(f"Incomplete page in {name}")
        for item in features:
            if item["id"] in items or not start <= capture(item) <= stop:
                raise ValueError(f"Unexpected scene in {name}/{item['id']}")
            items[item["id"]] = item
        sources.append(dict(url=url,file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest()))
        url = next((link["href"] for link in q["links"] if link["rel"]=="next"),None)
    return name, sorted(items.values(),key=capture), sources


def footprint(geometry, roofs, *, projected=False):
    fields = ("east_m","north_m") if projected else ("longitude","latitude")
    return {i for i,q in roofs.items() if geometry.covers(Point(*(q[k] for k in fields)))}


def landsat_check(item, roofs, original_roofs, store):
    name = item["id"]
    url = ESRI+"/query?"+urlencode(dict(f="json",where=f"Name = '{name}'",outFields="OBJECTID,Name,LANDSAT_PRODUCT_ID,Category",
                                      returnGeometry="true",outSR=32614))
    path = store/f"{name}_esri.json"
    raw = get(url,path)
    j = json.loads(raw)
    if j.get("error") or j.get("exceededTransferLimit"):
        raise ValueError(f"The Esri scene search is incomplete for {name}")
    hits = j["features"]
    if not hits:
        return dict(esri_scene_found=False,esri_query_url=url,esri_query_file=str(path.relative_to(ROOT)))
    entry, = hits
    attrs = entry["attributes"]
    if attrs["LANDSAT_PRODUCT_ID"] != name or attrs["Name"] != name or attrs["Category"] != 1:
        raise ValueError(f"The Esri response differs from {name}")
    geom = entry["geometry"]
    if geom["spatialReference"]["wkid"] != 32614:
        raise ValueError("Unexpected projection for the Landsat valid-data footprint")
    ring, = geom["rings"]
    covered = footprint(Polygon(ring),roofs,projected=True)
    scene = attrs["OBJECTID"]
    # Read pixels on an aligned 15 m diagnostic grid over the frozen roofs.
    # The source is not promoted to native USGS radiometry or per-pixel QA.
    xs = [roofs[q]["east_m"] for q in original_roofs]
    ys = [roofs[q]["north_m"] for q in original_roofs]
    x0,y0 = (15*math.floor((min(t)-150)/15) for t in (xs,ys))
    x1,y1 = (15*math.ceil((max(t)+150)/15) for t in (xs,ys))
    width,height = (x1-x0)//15,(y1-y0)//15
    url_tiff = ESRI+"/exportImage?"+urlencode(dict(f="image",bbox=f"{x0},{y0},{x1},{y1}",bboxSR=32614,imageSR=32614,
        size=f"{width},{height}",format="tiff",interpolation="RSP_NearestNeighbor",
        mosaicRule=json.dumps(dict(mosaicMethod="esriMosaicLockRaster",lockRasterIds=[scene],mosaicOperation="MT_FIRST")),
        renderingRule=json.dumps(dict(rasterFunction="None"))))
    tiff = store/f"{name}_esri_b8.tif"
    raster = get(url_tiff,tiff)
    with MemoryFile(raster) as mem, mem.open() as src:
        if (src.shape != (height,width) or src.count != 1 or src.crs.to_epsg() != 32614
                or (src.transform.a,src.transform.e,src.transform.c,src.transform.f) != (15,-15,x0,y1)):
            raise ValueError(f"Unexpected grid in Landsat check for {name}")
        values = src.read(1,masked=True)
        useful = ~np.ma.getmaskarray(values) & (values.data > 0) & (values.data < 10000)
        centers = {i:src.index(roofs[i]["east_m"],roofs[i]["north_m"]) for i in original_roofs}
        valid = sorted(i for i,(r,c) in centers.items() if useful[r,c])
    return dict(esri_scene_found=True, esri_scene_id=scene, esri_footprint_registry_roofs=len(covered),
        esri_footprint_frozen_roofs=len(covered & original_roofs), esri_positive_valid_pixels=int(useful.sum()),
        esri_valid_frozen_roof_centers=valid, esri_query_url=url, esri_query_file=str(path.relative_to(ROOT)),
        esri_b8_url=url_tiff, esri_b8_file=str(tiff.relative_to(ROOT)), esri_b8_sha256=hashlib.sha256(raster).hexdigest())


def planned_observations(stop, deadline, roofs, store):
    plans, upcoming = [], []
    for sensor, suffix in PLANS.items():
        url = PLAN_PREFIX+suffix
        path = store/f"{sensor}_plan.kml"
        data = get(url,path)
        plans.append(dict(platform=sensor,url=url,file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest()))
        for scene in ET.fromstring(data).findall(".//{*}Placemark"):
            fields = {q.attrib["name"]:q.findtext("{*}value") for q in scene.findall(".//{*}Data")}
            if fields.get("Mode") != "NOBS":
                continue
            span = tuple(datetime.fromisoformat(fields[f"ObservationTime{edge}"]+"+00:00") for edge in ("Start","Stop"))
            if not stop < span[0] < span[1] < deadline:
                continue
            overlaps = set()
            for polygon in scene.findall(".//{*}Polygon"):
                ring = polygon.findtext("./{*}outerBoundaryIs/{*}LinearRing/{*}coordinates")
                vertices = [tuple(map(float,v.split(",")[:2])) for v in ring.split()]
                overlaps |= footprint(make_valid(Polygon(vertices)),roofs)
            if overlaps:
                upcoming.append(dict(platform=sensor, datatake=fields["ID"], relative_orbit=int(fields["OrbitRelative"]),
                    planned_start_utc=span[0].isoformat(), planned_end_utc=span[1].isoformat(),
                    planned_frozen_roof_centers=len(overlaps)))
    return plans, sorted(upcoming,key=lambda q:q["planned_start_utc"])


def main(issue, stop):
    reg = DERIVED/f"cushing_pass_reliability_{issue}_registration.json"
    original = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    recovery = original.with_name(f"{original.stem}_recovered_optical.json")
    registration = json.loads(reg.read_bytes())
    checked_hashes(registration["sources_sha256"])
    cutoff = datetime.fromisoformat(registration["information_cutoff_utc"])
    deadline = datetime.fromisoformat(registration["registration_deadline_utc"])
    if not cutoff <= stop <= min(deadline, datetime.now(timezone.utc)):
        raise ValueError("Choose an observation time within the open forecast window")
    week = date.fromisoformat(registration["week_ending"])
    end = datetime.combine(week,datetime.min.time(),timezone.utc)+timedelta(hours=12)
    previous = end-timedelta(days=7)
    start = previous-timedelta(days=14)
    baseline_scenes = {q["pass_id"] for q in registration["deleted_passes"]}
    used = {q["candidate_id"] for path,k in ((original,"measured_pairs"),(recovery,"recovered_roofs"))
            for q in json.loads(path.read_bytes())[k]}
    roofs, _, _ = identities()
    if used-roofs.keys():
        raise ValueError("Frozen optical roofs are missing from the independent geometry")
    store = RAW / STEM / issue / stop.strftime("%Y%m%dT%H%M%SZ")
    with ThreadPoolExecutor(max_workers=4) as pool:
        catalogs = list(pool.map(lambda name:catalogue(name,start,stop,store),CATALOGUES))
    scenes, landsat, radar = [], [], []
    for name, items, _ in catalogs:
        for item in items:
            stamp = capture(item)
            props = item["properties"]
            covered = footprint(shape(item["geometry"]),roofs)
            if stamp <= previous:
                segment = "preceding_week" if stamp > previous-timedelta(days=7) else "older"
            else:
                segment = "target_week" if stamp <= end else "after_week"
            pass_key = pass_id(item["id"]+".SAFE") if name == "sentinel2" else None
            row = dict(source=name, product_id=item["id"], acquisition_utc=stamp.isoformat(), segment=segment,
                catalog_published_utc=props.get("published"), catalog_created_utc=props.get("created"),
                scene_cloud_pct=props.get("eo:cloud_cover"), registry_centers=len(covered),
                frozen_roof_centers=len(covered & used), relative_orbit=props.get("sat:relative_orbit"),
                orbit_direction=props.get("sat:orbit_state"), original_optical_pass=pass_key in baseline_scenes,
                frozen_roof_ids_json=json.dumps(sorted(covered & used)))
            if name=="radar_rtc" and covered:
                headers = {band:read_headers(sign(item["assets"][band]["href"])) for band in BANDS}
                ready = max(q["object_modified_at_utc"] for q in headers.values())
                radar.append(dict(row, archive_ready_utc=ready.isoformat(), archive_ready_by_original_cutoff=ready<cutoff))
            if name=="landsat" and covered and segment not in ("older","after_week"):
                landsat.append(row | landsat_check(item,roofs,used,store))
            scenes.append(row)
    pairs = []
    for first in radar:
        if first["segment"]!="preceding_week":
            continue
        for last in radar:
            same_track = (first["relative_orbit"],first["orbit_direction"]) == (last["relative_orbit"],last["orbit_direction"])
            if last["segment"]!="target_week" or not same_track:
                continue
            shared = set(json.loads(first["frozen_roof_ids_json"])) & set(json.loads(last["frozen_roof_ids_json"]))
            if shared:
                pairs.append(dict(first=first["product_id"],last=last["product_id"],frozen_roof_centers=len(shared),
                                  archive_ready_by_original_cutoff=all(q["archive_ready_by_original_cutoff"] for q in (first,last))))
    sources = {name:receipts for name,_,receipts in catalogs}
    plans, upcoming = planned_observations(stop,deadline,{q:roofs[q] for q in used},store)
    stem = f"{STEM}_{issue}_{stop.strftime('%Y%m%dT%H%M%SZ')}"
    export(f"{stem}_scenes",scenes)
    report = dict(release_date=issue, observation_stop_utc=stop.isoformat(), original_cutoff_utc=cutoff.isoformat(),
        preceding_week_end_utc=previous.isoformat(), end_of_target_week_utc=end.isoformat(), frozen_roofs=len(used),
        catalogues=sources, landsat_checks=landsat, radar_over_roofs=radar, matching_radar_pairs=pairs,
        published_acquisition_plans=plans, planned_sentinel2_before_registration_deadline=upcoming,
        original_passes=sorted(baseline_scenes), additional_sentinel2_passes_in_reference_weeks=sorted(
            {r["acquisition_utc"] for r in scenes if r["source"]=="sentinel2" and r["frozen_roof_centers"]
             and r["segment"] in ("preceding_week","target_week") and not r["original_optical_pass"]}),
        scope="Acquisition, archive timing and spatial coverage. A roof intersecting a scene is not a qualified height measurement.")
    payload = (json.dumps(report,indent=2,allow_nan=False)+"\n").encode()
    write(DERIVED/f"{stem}_summary.json",payload)
    print(stem, len(scenes),[(q["product_id"],q["frozen_roof_centers"],q["esri_footprint_frozen_roofs"],
                              len(q["esri_valid_frozen_roof_centers"])) for q in landsat if q["esri_scene_found"]],pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat, required=True)
    parser.add_argument("--observed-through", type=datetime.fromisoformat, required=True)
    args = parser.parse_args()
    main(args.release.isoformat(),args.observed_through.astimezone(timezone.utc))
