"""Preserve burst identifiers, footprints, and access evidence for December 2020."""

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import cos, radians
from urllib.parse import urlencode

import numpy as np
from shapely.geometry import Point

from audit_cushing_2020_radar_week import STEM, local_day, radar_sources
from audit_cushing_slc_archive import DOWNLOAD, parent_files, probe
from audit_shadow_precision import describe
from build_cushing_icesat2_controls import identities
from build_opera_cslc_holdout_inventory import access_probe, day_records, match_granules, static_layers
from build_s1_holdout_gap_burst_plan import asf_records
from build_s1_slc_2019_reference_plan import WORST_NOMINAL_SLC_SLANT_RESOLUTION_M
from build_s1_slc_orbit_geometry import burst_catalog, orbit_grid, tank_geometry
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from build_weekly_shadow_validation import legacy_geometry
from collect_ceda_s1_slc_annotations import archive_urls
from collect_ceda_s1_slc_regional import selection
from collect_public import RAW, ROOT, get, write
from collect_s1_slc_orbits import orbit_for, utc


def bursts_for(product, contexts):
    requests, roofs = selection(product, contexts)
    source, members = parent_files((product["product_id"], requests))
    # The CEDA selector does not need ASF links. Join the extractor's public search separately.
    cdse = [dict(RelativeOrbitNumber=int(r["relative_orbit"]), BurstId=int(r["relative_burst_id"]), SwathIdentifier=r["swath"])
            for r in source]
    records, archive = asf_records(product, cdse)
    urls = {}
    for feature in records:
        p = feature["properties"]
        b = p["burst"]
        key = str(b["relativeBurstID"]), b["subswath"].upper(), p["polarization"].upper()
        urls[key] = p
    results = []
    for r in source:
        key = str(r["relative_burst_id"]), r["swath"], r["polarization"]
        asf = urls.get(key)
        if asf and (int(asf["bytes"]) != r["expected_complex_sample_bytes"] or int(asf["burst"]["samplesPerBurst"]) != r["samples_per_burst"]):
            raise ValueError(f"ASF burst grid differs from Copernicus for {key}")
        results.append(dict(r, scope="december_2020_aerial_control", asf_catalog_file=archive,
                            asf_data_url=asf["url"] if asf else None,
                            asf_annotation_url=asf["additionalUrls"][0] if asf else None,
                            full_context_roof_count=len(roofs[r["cdse_uuid"]]),
                            full_context_roofs_json=json.dumps(roofs[r["cdse_uuid"]])))
    return results, members


def orbit_check(orbit, grd_sources):
    pair = (orbit["satellite"], orbit["absolute_orbit"])
    product = grd_sources[pair]
    origin, position, velocity = orbit_grid(orbit["local_orbit_gzip"], orbit["orbit_filename"], orbit["orbit_sha256"])
    annotation = product["objects"]["annotation/iw-vv.xml"]["local_file"]
    root = ET.parse(ROOT / annotation)
    embedded = root.findall("./generalAnnotation/orbitList/orbit")
    if len(embedded) < 4 or any(r.findtext("frame") != "Earth Fixed" for r in embedded):
        raise ValueError(f"Missing GRD embedded state vectors for {orbit['product_name']}")
    t = [(utc(n.findtext("time"))-origin).total_seconds() for n in embedded]
    vectors = [np.array([[float(r.findtext(f"{key}/{axis}")) for axis in "xyz"] for r in embedded])
               for key in ("position", "velocity")]
    dp = np.linalg.norm(position(t)-vectors[0], axis=1)
    dv = np.linalg.norm(velocity(t)-vectors[1], axis=1)
    if not np.isfinite(dp).all() or not np.isfinite(dv).all():
        raise ValueError(f"Precise orbit does not span embedded vectors for {orbit['product_name']}")
    return dict(orbit, grd_product_name=product["product"]["product_name"], embedded_grd_annotation=annotation,
                matched_embedded_state_vectors=len(t), median_position_difference_m=float(np.median(dp)),
                max_position_difference_m=float(dp.max()), max_velocity_difference_mps=float(dv.max()),
                verified_state_vector_agreement=bool(dp.max()<=.10 and dv.max()<=.001))


def opera_assets(products, bursts, contexts):
    parents = {r["product_id"]: r for r in products}
    vv = {r["product_name"]: r for r in bursts if r["polarization"] == "VV"}
    requests = {}
    for name, r in vv.items():
        parent = parents[r["product_id"]]
        requests[name] = dict(parent_product_name=name, relative_burst_id=int(r["relative_burst_id"]),
                              swath=r["swath"], orbit=int(r["relative_orbit"]), direction=parent["direction"],
                              satellite=parent["satellite"], absolute_orbit=int(parent["absolute_orbit"]),
                              start=utc(parent["capture_start_utc"]))
    days = sorted({r["start"].date().isoformat() for r in requests.values()})
    with ThreadPoolExecutor(max_workers=3) as pool:
        by_day = dict(zip(days, pool.map(day_records, days), strict=True))
    matched = match_granules(requests, by_day)
    static = static_layers(matched)
    assets = []
    for name, granules in matched.items():
        granule, = granules
        r = vv[name]
        start_offset = (granule["capture_start"] - utc(r["azimuth_time_utc"])).total_seconds()
        if abs(start_offset) > 1:
            raise ValueError(f"OPERA and Copernicus burst start times disagree for {name}")
        layer = static[granule["orbit"], granule["relative_burst_id"], granule["swath"]]
        coverage = granule["footprint"].intersection(layer["footprint"])
        roofs = [k for k in json.loads(r["full_context_roofs_json"]) if coverage.covers(contexts[k])]
        assets.append(dict(product_kind="cslc_vv", slc_product_name=name, slc_cdse_uuid=r["cdse_uuid"],
                           burst_start_difference_seconds=start_offset, static_concept_id=layer["concept_id"],
                           full_context_roof_count=len(roofs), full_context_roofs_json=json.dumps(roofs), **granule))
    for layer in static.values():
        roofs = sorted(k for k, context in contexts.items() if layer["footprint"].covers(context))
        assets.append(dict(product_kind="static", static_concept_id=layer["concept_id"],
                           full_context_roof_count=len(roofs), full_context_roofs_json=json.dumps(roofs), **layer))
    keys = ("granule", "concept_id", "revision", "polarization", "capture_start", "capture_end", "production_at", "inserted_at",
            "revision_at", "size_bytes", "md5", "download_url", "catalog_file", "product_kind", "slc_product_name", "slc_cdse_uuid",
            "burst_start_difference_seconds", "static_concept_id", "full_context_roof_count", "full_context_roofs_json")
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = list(pool.map(access_probe, (r["download_url"] for r in assets)))
    return [dict({k: r[k].isoformat() if k in r and hasattr(r[k], "isoformat") else r.get(k) for k in keys},
                 catalog_sha256=hashlib.sha256((ROOT / r["catalog_file"]).read_bytes()).hexdigest(),
                 **{k: p.get(k) for k in ("http_status", "host", "hdf5_signature", "error")})
            for r, p in zip(assets, checks, strict=True)]


def liveeo_inventory(product):
    name = product["product_name"]
    day = utc(product["capture_start_utc"])
    prefix = f"{day:%Y/%m/%d}/{name}/"
    url = "https://sentinel1-slc.s3.eu-west-1.amazonaws.com/?"+urlencode({"list-type": "2", "prefix": prefix})
    path = RAW / "cushing_slc_archive" / product["product_id"] / "liveeo_inventory.xml"
    payload = get(url, path)
    root = ET.fromstring(payload)
    count = int(root.findtext("{*}KeyCount"))
    keys = [item.findtext("{*}Key") for item in root.findall("{*}Contents")]
    if (root.findtext("{*}Name") != "sentinel1-slc" or root.findtext("{*}Prefix") != prefix
            or root.findtext("{*}IsTruncated") != "false" or len(keys) != count or any(not k.startswith(prefix) for k in keys)):
        raise ValueError(f"Unexpected LiveEO archive listing for {name}")
    return dict(product_name=name, archive_url=url, object_count=count, inventory_file=str(path.relative_to(ROOT)),
                inventory_sha256=hashlib.sha256(payload).hexdigest())


def same_day_slant_controls(geometry):
    rows = {(r["candidate_id"], local_day(r["rim_zero_doppler_at_utc"]).isoformat()): r for r in geometry
            if r["rim_zero_doppler_inside_parent"]}
    controls = []
    for a in read("aerialok_native_optical_aerial"):
        r = rows.get((a["candidate_id"], a["flight_date"]))
        if not r or a["aerial_fit_status"] != "visible_shadow_fit" or a["clock_status"] != "shadow_inferred_clock":
            continue
        view_factor = cos(radians(r["rim_incidence_deg"]))
        gap = float(a["aerial_gap_m"])
        low = float(a["aerial_gap_min_over_clock_profile_m"])
        high = float(a["aerial_gap_max_over_clock_profile_m"])
        shift, shift_low, shift_high = (v * view_factor for v in (gap, low, high))
        radar_time = utc(r["rim_zero_doppler_at_utc"])
        clock = [utc(a[k]) for k in ("exterior_trial_first_utc", "exterior_trial_last_utc")]
        offsets = [(radar_time-t).total_seconds()/3600 for t in clock]
        controls.append(dict(candidate_id=a["candidate_id"], terminal_id=a["terminal_id"], aerial_date=a["flight_date"],
                             aerial_source=a["source_path"], aerial_sha256=a["source_sha256"], product_name=r["product_name"],
                             cdse_uuid=r["cdse_uuid"], aerial_clock_earliest_utc=clock[0], aerial_clock_latest_utc=clock[1],
                             zero_doppler_utc=radar_time, radar_minus_aerial_earliest_h=offsets[0],
                             radar_minus_aerial_latest_h=offsets[1], rim_incidence_deg=r["rim_incidence_deg"],
                             aerial_gap_m=gap, aerial_profile_low_m=low, aerial_profile_high_m=high,
                             all_near_clock_fits_visible=a["all_near_clock_fits_visible"], projected_slant_shift_m=shift,
                             projected_slant_shift_low_m=shift_low, projected_slant_shift_high_m=shift_high,
                             shift_low_at_least_native_slc_resolution=shift_low >= WORST_NOMINAL_SLC_SLANT_RESOLUTION_M))
    return controls


def main():
    products = [r for r in radar_sources() if r["product_type"] == "IW_SLC__1S"]
    tanks, _, _ = identities()
    contexts = {k: Point(tanks[k]["east_m"], tanks[k]["north_m"]).buffer(tanks[k]["radius_m"]+120) for k in legacy_geometry()}
    with ThreadPoolExecutor(max_workers=3) as pool:
        retrieved = list(pool.map(lambda p: bursts_for(p, contexts), products))
    bursts = [r for rows, _ in retrieved for r in rows]
    members = [r for _, rows in retrieved for r in rows]
    checks = [("ceda_full_zip", r["product_id"], u+".zip?download=1") for r in products for u in archive_urls(r["product_name"])]
    checks.extend(("cdse_safe", r["product_id"], f"{DOWNLOAD}({r['product_id']})/$value") for r in products)
    for r in bursts:
        for key in ("cdse_direct_burst_url", "cdse_burst_download_url", "asf_data_url", "asf_annotation_url"):
            if r[key]:
                checks.append((key, r["cdse_uuid"], r[key]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        access = list(pool.map(probe, checks))
    with ThreadPoolExecutor(max_workers=3) as pool:
        orbits = list(pool.map(lambda p: orbit_for(dict(p, scope="december_2020_aerial_control"),
                                                   accept_relative_osv_labels=True), products))
        liveeo = list(pool.map(liveeo_inventory, products))
    manifests = [json.loads((ROOT / r["source_manifest"]).read_bytes()) for r in read(f"{STEM}_grd_sources")]
    grd_sources = {(m["product"]["satellite"], m["product"]["absolute_orbit"]): m for m in manifests}
    orbits = [orbit_check(r, grd_sources) for r in orbits]
    export(f"{STEM}_slc_bursts", bursts)
    export(f"{STEM}_slc_members", members)
    export(f"{STEM}_slc_access", access)
    export(f"{STEM}_slc_liveeo_inventory", liveeo)
    export(f"{STEM}_precise_orbits", orbits)
    exact_orbits = {r["product_id"]: r for r in orbits if r["verified_state_vector_agreement"]}
    geometry = []
    for r in bursts:
        if r["product_id"] not in exact_orbits or r["polarization"] != "VV":
            continue
        orbit = exact_orbits[r["product_id"]]
        catalog = burst_catalog(r["cdse_catalog_file"])[r["cdse_uuid"]]
        for roof in json.loads(r["full_context_roofs_json"]):
            values = tank_geometry(tanks[roof], orbit, catalog)
            geometry.append(dict(candidate_id=roof, product_name=r["product_name"], cdse_uuid=r["cdse_uuid"], **values))
    export(f"{STEM}_slc_verified_orbit_geometry", geometry)
    opera = opera_assets(products, bursts, contexts)
    export(f"{STEM}_opera_cslc", opera)
    controls = same_day_slant_controls(geometry)
    export(f"{STEM}_slc_aerial_projection", controls)
    groups = defaultdict(lambda: defaultdict(set))
    for r in bursts:
        groups[r["product_name"]][r["relative_burst_id"], r["swath"], r["polarization"]].update(json.loads(r["full_context_roofs_json"]))
    product_names = sorted(groups)
    keys = set.intersection(*(set(groups[k]) for k in product_names))
    common = {f"{burst}/{swath}/{pol}": len(set.intersection(*(groups[p][burst, swath, pol] for p in product_names)))
              for burst, swath, pol in sorted(keys)}
    report = dict(slc_products=len(products), requested_burst_polarizations=len(bursts), common_roofs_by_relative_burst=common,
                  retrieved_precise_orbits=len(orbits), verified_absolute_orbit_matches=sum(r["osvs_match_parent_absolute_orbit"] for r in orbits),
                  verified_against_grd_embedded_vectors=len(exact_orbits),
                  roof_positions_with_verified_orbit=len(geometry),
                  roof_positions_inside_parent=sum(r["rim_zero_doppler_inside_parent"] for r in geometry),
                  roof_positions_inside_cdse_catalog_time=sum(r["rim_zero_doppler_inside_cdse_catalog_time"] for r in geometry),
                  access_status=Counter(f"{r['route']}:{r['status']}" for r in access),
                  anonymously_readable_complex_bursts=sum(r["source_signature"]=="TIFF" and r["route"]!="asf_annotation_url" for r in access),
                  liveeo_products_with_objects=sum(r["object_count"] > 0 for r in liveeo),
                  opera_vv_matches=sum(r["product_kind"]=="cslc_vv" for r in opera),
                  opera_static_layers=sum(r["product_kind"]=="static" for r in opera),
                  anonymously_readable_opera_hdf5=sum(r["hdf5_signature"] is True for r in opera),
                  same_day_aerial_projections=len(controls),
                  same_day_time_offset_hours=dict(minimum=min(r["radar_minus_aerial_latest_h"] for r in controls),
                                                  maximum=max(r["radar_minus_aerial_earliest_h"] for r in controls)),
                  projected_slant_displacement_m=describe([r["projected_slant_shift_m"] for r in controls]),
                  projected_lower_bound_at_least_native_slc_resolution=sum(r["shift_low_at_least_native_slc_resolution"] for r in controls),
                  projected_lower_bound_at_least_native_slc_resolution_and_all_clock_trials_visible=sum(
                      r["shift_low_at_least_native_slc_resolution"] and r["all_near_clock_fits_visible"]=="True" for r in controls),
                  scope="Exact Cushing burst geometries with 120 m context and precise orbit clocks. Archive checks are anonymous. No I/Q files are asserted from a catalog match.")
    write(DERIVED/f"{STEM}_slc_summary.json", (json.dumps(report, indent=2)+"\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
