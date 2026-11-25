"""Collect independent radar and laser controls near Plains and South Bow inspections."""

import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from urllib.parse import urlencode

import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import Point

from audit_cushing_2020_radar_week import PASS, fit_scene, local_day
from audit_cushing_slc_archive import probe
from audit_s1_grd_native_inputs import saved_assets, saved_windows, source_sample
from build_cushing_icesat2_controls import COLLECTION, TO_GEO, TO_UTM, identities
from build_cushing_okeis_plains_control import STEM as PLAINS
from build_cushing_okeis_southbow_controls import STEM as SOUTHBOW
from build_cushing_gedi_inventory import COLLECTIONS
from build_cushing_inspection_controls import LOCAL_TIME
from build_cushing_nasa_satellite_inventory import CMR, NISAR
from build_opera_cslc_holdout_inventory import access_probe, day_records, static_layers
from build_s1_holdout_gap_burst_plan import CONTEXT_M, bursts
from build_s1_reference_matches import DERIVED, export, read
from calibrate_s1_grd_native_windows import read_band, window
from collect_ceda_s1_slc_annotations import archive_urls
from collect_cushing_2020_radar_bursts import liveeo_inventory
from collect_public import RAW, ROOT, get, write
from collect_s1_grd_roof_inputs import collect

STEM = "cushing_okeis_inspection_remote"
STORE = RAW / STEM
DAYS = 4
CMR_COLLECTIONS = {"ATL03_v007": COLLECTION, "GEDI_L2A_v003": COLLECTIONS["L2A"],
                   **{f"NISAR_GSLC_{k}": v for k, v in NISAR.items()}}


def local_start(day):
    return datetime.combine(day, time(), LOCAL_TIME).astimezone(timezone.utc)


def sites():
    a, = read(f"{PLAINS}_ground_record")
    b = read(f"{SOUTHBOW}_ground_levels")
    return {"plains_25c6": (date.fromisoformat(a["source_local_date"]), set(json.loads(a["possible_candidate_ids_json"])),
                            "-96.724,35.936,-96.719,35.942"),
            "southbow": (date.fromisoformat(b[0]["source_local_date"]), {k for r in b if r["optically_visible_roof"] == "True"
                           for k in json.loads(r["possible_candidate_ids_json"])}, "-96.769,35.919,-96.752,35.930")}


def catalogue(job):
    site, (day, _, bbox), label, collection = job
    start, end = local_start(day-timedelta(days=DAYS)), local_start(day+timedelta(days=DAYS+1))
    period = ",".join(t.isoformat().replace("+00:00", "Z") for t in (start, end))
    url = CMR + "?" + urlencode(dict(collection_concept_id=collection, bounding_box=bbox, temporal=period, page_size=100))
    path = STORE / "cmr" / f"{site}_{label}.json"
    payload = get(url, path)
    result = json.loads(payload)
    if result["hits"] != len(result["items"]) or any(x["meta"]["collection-concept-id"] != collection for x in result["items"]):
        raise ValueError(f"Invalid NASA catalogue result for {site}/{label}")
    return dict(site=site, sensor=label, collection=collection, bounding_box=bbox, searched_at_utc=period,
                source_url=url, catalog_file=str(path.relative_to(ROOT)), catalog_sha256=hashlib.sha256(payload).hexdigest(),
                matching_granules=result["hits"], ids_json=json.dumps([x["meta"]["concept-id"] for x in result["items"]])), result["items"]


def laser_sources(cmr, target, geometry):
    atl = {r["cmr_id"]: r for r in read("cushing_icesat2_native_extracts")}
    gedi = {r["l2a_granule"]: r for r in read("cushing_gedi_coverage_granules")}
    rows = []
    for query, matches in cmr:
        label = query["sensor"]
        for m in matches:
            ident = m["meta"]["concept-id"]
            name = m["umm"]["GranuleUR"]
            candidates = sorted(target[query["site"]][1])
            if label == "ATL03_v007":
                source = atl[ident]
                path = source["packet_path"]
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != source["response_sha256"]:
                    raise ValueError("ATL03 subset checksum differs")
                count = int(source["used_photons"])
                if count != 0:
                    raise ValueError("ATL03 packet contains candidate measurements and needs footprint analysis")
                nearest = {roof: (None, None, 0) for roof in candidates}
            elif label == "GEDI_L2A_v003":
                source = gedi[name]
                path = source["local_path"]
                if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != source["local_sha256"]:
                    raise ValueError("GEDI subset checksum differs")
                table = pq.read_table(ROOT / path)
                count = table.num_rows
                x, y = TO_UTM.transform(table["lon_lowestmode"].to_numpy(), table["lat_lowestmode"].to_numpy())
                nearest = {}
                for roof in candidates:
                    g = geometry[roof]
                    d = np.hypot(x-g["east_m"], y-g["north_m"])
                    idx = int(d.argmin())
                    nearest[roof] = float(d[idx]), str(table["shot_number"][idx].as_py()), int((d <= g["radius_m"]+12.5).sum())
            else:
                continue
            rows.extend(dict(site=query["site"], sensor=label, granule=name, cmr_id=ident, candidate_id=roof,
                             archived_regional_rows=count, nearest_sample_distance_m=d, nearest_sample_id=sample,
                             samples_intersecting_roof=hits, local_extract=path, catalog_file=query["catalog_file"])
                        for roof, (d, sample, hits) in nearest.items())
    return rows


def radar_catalog(target):
    products = read("s1_cushing_catalog")
    assets = []
    for site, (day, roofs, _) in target.items():
        lo, hi = local_start(day), local_start(day+timedelta(days=1))
        for r in products:
            t = datetime.fromisoformat(r["capture_start_utc"])
            covered = roofs & set(json.loads(r["covered_candidate_ids_json"]))
            if abs((local_day(r["capture_start_utc"])-day).days) > DAYS or not covered:
                continue
            assets.append(dict(r, site=site, event_day=day.isoformat(), candidate_ids_json=json.dumps(sorted(covered)),
                               minimum_possible_inspection_separation_hours=max(0., (lo-t).total_seconds(), (t-hi).total_seconds())/3600,
                               maximum_possible_inspection_separation_hours=max(abs((t-lo).total_seconds()), abs((t-hi).total_seconds()))/3600))
    return assets


def inspection_geometry(target):
    geometry, _, _ = identities()
    required = set().union(*(roofs for _, roofs, _ in target.values()))
    roster = {r["candidate_id"]: r for r in read("tank_engineering_registry") if r["candidate_id"] in required}
    for r in read("cushing_modern_lidar_optical_registry"):
        k = r["candidate_id"]
        if k not in required or r["modern_input_status"] not in {"available", "insufficient_lidar_outline_quality"}:
            continue
        x, y, diameter = (float(r[key]) for key in ("center_east_m", "center_north_m", "diameter_m"))
        lon, lat = TO_GEO.transform(x, y)
        geometry[k] = dict(candidate_id=k, east_m=x, north_m=y, longitude=lon, latitude=lat, radius_m=diameter/2,
                           first_seen=roster[k]["first_seen"], terminal_id=roster[k]["documented_terminal_id"],
                           geometry_dataset=r["dataset"], geometry_status=r["modern_input_status"],
                           lidar_top_navd88_m=float(r["prior_rim_z_p95_native"]), lidar_ground_navd88_m=float(r["prior_ground_z_p50_native"]))
    if not required <= geometry.keys() or any(not 0 < geometry[k]["lidar_top_navd88_m"]-geometry[k]["lidar_ground_navd88_m"] < 35
                                               for k in required):
        raise ValueError("Inspection candidates lack usable geometric inputs")
    return geometry


def grd(p, roofs, geometry):
    locations = {k: (g["longitude"], g["latitude"], 2*g["radius_m"]) for k, g in geometry.items()
                 if g["first_seen"] <= p["event_day"] and any(np.hypot(g["east_m"]-geometry[r]["east_m"],
                    g["north_m"]-geometry[r]["north_m"]) < 400 for r in roofs)}
    manifest, rows = collect(dict(p, selection_reasons="okeis_inspection"), locations, store=STORE / "grd")
    _, metadata = saved_assets([manifest])
    unique = saved_windows([dict(r, centre_pixel_valid=str(r["centre_pixel_valid"])) for r in rows])
    verified = source_sample(unique, per_product=True)
    bands = {pol: read_band(manifest, pol) for pol in manifest["bands"]}
    calibrated = {k: window(r, bands[r["polarization"]], output_root=STORE / "sigma0") for k, r in unique.items()}
    fits = [dict(r, site=p["site"], geometry_dataset=geometry[r["candidate_id"]]["geometry_dataset"],
                 geometry_status=geometry[r["candidate_id"]].get("geometry_status"))
            for r in fit_scene(manifest, rows, calibrated, geometry) if r["candidate_id"] in roofs]
    if {r["candidate_id"] for r in fits} != roofs:
        raise ValueError(f"Missing radar fits for {p['site']}")
    return rows, list(calibrated.values()), fits, metadata, verified


def slc(p, roofs, geometry):
    contexts = {k: Point(geometry[k]["east_m"], geometry[k]["north_m"]).buffer(geometry[k]["radius_m"]+CONTEXT_M) for k in roofs}
    result = []
    requests = [("ceda_parent_zip", p["product_id"], base+".zip?download=1") for base in archive_urls(p["product_name"])]
    requests.extend(("ceda_metadata_zip", p["product_id"], base+".metadata_only.zip?download=1") for base in archive_urls(p["product_name"]))
    for r in bursts(p)["bursts"]:
        covered = sorted(k for k, polygon in contexts.items() if r["footprint"].covers(polygon))
        if covered and r["cdse_uuid"]:
            ident = r["cdse_uuid"]
            result.append(dict(site=p["site"], parent_product_id=p["product_id"], burst_id=ident, relative_burst_id=r["relative_burst_id"],
                               swath=r["swath"], polarization=r["polarization"], full_context_roofs_json=json.dumps(covered),
                               asf_data_url=r["asf_data_url"], cdse_catalog_file=r["cdse_catalog_file"], asf_catalog_file=r["asf_catalog_file"]))
            requests.append(("cdse_direct_burst", ident, f"https://bursts.dataspace.copernicus.eu/{ident}"))
            if r["asf_data_url"]:
                requests.append(("asf_burst_tiff", ident, r["asf_data_url"]))
    if set().union(*(set(json.loads(r["full_context_roofs_json"])) for r in result)) != roofs:
        raise ValueError(f"Missing full-context burst coverage for {p['site']}")
    return result, [dict(probe(r), site=p["site"]) for r in requests], liveeo_inventory(p)


def opera(products, slcs, geometry):
    parents = {p["product_id"]: p for p in products if p["product_type"] == "IW_SLC__1S"}
    matched, sources = {}, {}
    for group, _, _ in slcs:
        for b in group:
            if b["polarization"] != "VV":
                continue
            p = parents[b["parent_product_id"]]
            start, end = (datetime.fromisoformat(p[k]) for k in ("capture_start_utc", "capture_end_utc"))
            key = (int(p["relative_orbit"]), b["relative_burst_id"], b["swath"], int(p["absolute_orbit"]), p["satellite"])
            matches = [g for g in day_records(start.date().isoformat()) if g["polarization"] == "VV" and g["direction"].upper() == p["direction"]
                       and (g["orbit"], g["relative_burst_id"], g["swath"], g["absolute_orbit"], g["satellite"]) == key
                       and start <= g["capture_start"] < g["capture_end"] <= end]
            matched[b["burst_id"]] = matches
            sources[b["burst_id"]] = b
    static = static_layers(matched)
    rows = []
    for ident, group in matched.items():
        for r in group:
            b = sources[ident]
            layer = static[r["orbit"], r["relative_burst_id"], r["swath"]]
            footprint = r["footprint"].intersection(layer["footprint"])
            covered = [k for k in json.loads(b["full_context_roofs_json"])
                       if footprint.covers(Point(geometry[k]["east_m"], geometry[k]["north_m"]).buffer(geometry[k]["radius_m"]+CONTEXT_M))]
            rows.append(dict(r, kind="CSLC_VV", site=b["site"], burst_id=ident, static_concept_id=layer["concept_id"],
                             full_context_roofs_json=json.dumps(covered)))
    rows.extend(dict(r, kind="static_geometry") for r in static.values())
    fields = ("kind", "site", "burst_id", "granule", "concept_id", "static_concept_id", "full_context_roofs_json", "capture_start", "capture_end",
              "inserted_at", "size_bytes", "md5", "download_url", "catalog_file")
    with ThreadPoolExecutor(max_workers=3) as pool:
        probes = list(pool.map(access_probe, (r["download_url"] for r in rows)))
    return [{k: r.get(k) for k in fields} | dict(catalog_sha256=hashlib.sha256((ROOT / r["catalog_file"]).read_bytes()).hexdigest())
            | {k: a.get(k) for k in ("http_status", "host", "hdf5_signature", "error")} for r, a in zip(rows, probes, strict=True)]


def main():
    target = sites()
    geometry = inspection_geometry(target)
    products = radar_catalog(target)
    tasks = [(site, spec, label, key) for site, spec in target.items() for label, key in CMR_COLLECTIONS.items()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        cmr = list(pool.map(catalogue, tasks))
        grd_jobs = [pool.submit(grd, p, set(json.loads(p["candidate_ids_json"])), geometry)
                    for p in products if p["product_type"] == "IW_GRDH_1S"]
        slc_jobs = [pool.submit(slc, p, set(json.loads(p["candidate_ids_json"])), geometry)
                    for p in products if p["product_type"] == "IW_SLC__1S"]
        grds, slcs = [f.result() for f in grd_jobs], [f.result() for f in slc_jobs]
    lasers = laser_sources(cmr, target, geometry)
    fits = [r for packet in grds for r in packet[2]]
    probes = [r for _, group, _ in slcs for r in group]
    products_opera = opera(products, slcs, geometry)
    for suffix, rows in (("cmr_searches", [q for q, _ in cmr]), ("laser_samples", lasers), ("sentinel1_catalog", products),
                         ("slc_bursts", [r for group, _, _ in slcs for r in group]), ("slc_probes", probes),
                         ("slc_liveeo", [row for _, _, row in slcs]), ("opera_assets", products_opera),
                         ("grd_native", [r for packet in grds for r in packet[0]]),
                         ("grd_sigma0", [r for packet in grds for r in packet[1]]), ("grd_fits", fits)):
        export(f"{STEM}_{suffix}", rows)
    report = dict(nasa_queries=len(cmr), nasa_granules=sum(q["matching_granules"] for q, _ in cmr),
                  laser_roof_intersections=sum(r["samples_intersecting_roof"] for r in lasers),
                  original_grd_products=len(grds), grd_metadata_files_verified=sum(r[3] for r in grds),
                  grd_source_windows_read_back=sum(r[4] for r in grds),
                  grd_fits_by_status=Counter(r["status"] for r in fits), grd_resolved_heights=sum(r["status"] == PASS for r in fits),
                  slc_parents=len(slcs), anonymous_complex_signatures=sum(r["source_signature"] in {"TIFF", "ZIP"} for r in probes),
                  opera_cslc_products=sum(r["kind"] == "CSLC_VV" for r in products_opera),
                  opera_hdf5_signatures=sum(r["hdf5_signature"] for r in products_opera))
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
