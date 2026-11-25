"""Archive native radar data and test roof-height resolution around NRC 1402673."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from shapely.geometry import Point

from audit_cushing_2020_radar_week import PASS, fit_scene, local_day
from audit_cushing_slc_archive import parent_files, probe
from audit_rose_rock_registered_changes import INPUT, STEM as OPTICAL
from audit_s1_grd_native_inputs import saved_assets, saved_windows, source_sample
from build_ceda_s1_slc_windows import NativeGeometry
from build_cushing_icesat2_controls import TO_GEO, identities
from build_opera_cslc_holdout_inventory import access_probe, day_records, match_granules, static_layers
from build_s1_holdout_gap_burst_plan import CONTEXT_M, bursts
from build_s1_reference_matches import DERIVED, export, read
from build_s1_slc_orbit_geometry import burst_catalog, tank_geometry
from calibrate_s1_grd_native_windows import read_band, window
from collect_ceda_s1_slc_annotations import archive_urls
from collect_ceda_s1_slc_annotations import collect as collect_annotations
from collect_cushing_2020_radar_bursts import liveeo_inventory, orbit_check
from collect_public import RAW, ROOT, write
from collect_s1_grd_roof_inputs import collect
from collect_s1_slc_orbits import orbit_for, utc
from collect_umbra_sicd_windows import with_ellipsoid_heights

STEM = "cushing_nrc_rose_rock_sentinel1"
STORE = RAW / STEM


def inputs():
    targets = {r["candidate_id"] for r in read(f"{INPUT}_inventory_candidates")
               if r["capacity_at_least_reported_inventory"] == "True"}
    catalogue = {r["product_id"]: r for r in read("s1_cushing_catalog")}
    event_products = read(f"{OPTICAL}_sentinel1_catalog")
    products = sorted((catalogue[r["product_id"]] | {"hours_from_reported_event": r["hours_from_reported_event"]}
                       for r in event_products), key=lambda p: p["capture_start_utc"])
    slc = [p for p in products if p["product_type"] == "IW_SLC__1S"]
    grd = [p for p in products if p["product_type"] == "IW_GRDH_1S"]
    if (len(slc) != 2 or len(grd) != 2 or any(p["product_name"].endswith("_COG.SAFE") for p in grd)
            or [p["absolute_orbit"] for p in slc] != [p["absolute_orbit"] for p in grd]):
        raise ValueError("Expected two original Sentinel-1 GRD and SLC products from the same passes")
    geometry, _, _ = identities()
    modern = set()
    for r in read("cushing_modern_lidar_optical_registry"):
        roof = r["candidate_id"]
        if (roof not in geometry or r["modern_input_status"] != "available"
                or utc(r["last_upper_return_at_utc"]) >= utc(slc[0]["capture_start_utc"])):
            continue
        modern.add(roof)
        x, y, diameter = (float(r[k]) for k in ("center_east_m", "center_north_m", "diameter_m"))
        lon, lat = TO_GEO.transform(x, y)
        geometry[roof].update(east_m=x, north_m=y, longitude=lon, latitude=lat, radius_m=diameter/2,
                              geometry_dataset=r["dataset"], geometry_source_directory=r["source_directory"],
                              lidar_top_navd88_m=float(r["prior_rim_z_p95_native"]),
                              lidar_ground_navd88_m=float(r["prior_ground_z_p50_native"]))
    neighbours = {roof: g for roof, g in geometry.items()
                  if datetime.fromisoformat(g["first_seen"]).date() <= local_day(slc[0]["capture_start_utc"])
                  and any((g["east_m"]-geometry[t]["east_m"])**2 + (g["north_m"]-geometry[t]["north_m"])**2 < 450**2
                          for t in targets)}
    if not targets <= neighbours.keys() & modern:
        raise ValueError("A target roof is missing dated geometry")
    return targets, neighbours, slc, grd


def collect_opera(products, found, contexts):
    parents = {p["product_id"]: p for p in products}
    vv = {r["cdse_uuid"]: r for r in found if r["polarization"] == "VV"}
    requests = {}
    for ident, r in vv.items():
        p = parents[r["product_id"]]
        requests[ident] = dict(parent_product_name=r["product_name"], relative_burst_id=int(r["relative_burst_id"]),
                               swath=r["swath"], orbit=int(r["relative_orbit"]), direction=p["direction"], satellite=p["satellite"],
                               absolute_orbit=int(p["absolute_orbit"]), start=utc(r["azimuth_time_utc"]).replace(microsecond=0))
    days = sorted({r["start"].date().isoformat() for r in requests.values()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        matched = match_granules(requests, dict(zip(days, pool.map(day_records, days), strict=True)))
    static = static_layers(matched)
    assets = []
    for ident, granules in matched.items():
        for g in granules:
            r = vv[ident]
            timing = (g["capture_start"] - utc(r["azimuth_time_utc"])).total_seconds()
            if abs(timing) >= 1:
                raise ValueError(f"OPERA capture clock differs from Sentinel-1 burst {ident}")
            layer = static[g["orbit"], g["relative_burst_id"], g["swath"]]
            footprint = g["footprint"].intersection(layer["footprint"])
            roofs = [k for k in json.loads(r["full_context_roofs_json"]) if footprint.covers(contexts[k])]
            assets.append(dict(g, kind="CSLC_VV", cdse_uuid=ident, slc_product=r["product_name"],
                               full_context_roofs_json=json.dumps(roofs), static_concept_id=layer["concept_id"],
                               burst_start_difference_s=timing))
    for layer in static.values():
        roofs = [k for k, ctx in contexts.items() if layer["footprint"].covers(ctx)]
        assets.append(dict(layer, kind="static_geometry", full_context_roofs_json=json.dumps(roofs)))
    fields = ("kind", "granule", "concept_id", "cdse_uuid", "slc_product", "static_concept_id", "capture_start", "capture_end",
              "production_at", "inserted_at", "revision_at", "burst_start_difference_s", "full_context_roofs_json",
              "catalog_file", "size_bytes", "md5", "download_url")
    with ThreadPoolExecutor(max_workers=3) as pool:
        checks = list(pool.map(access_probe, (r["download_url"] for r in assets)))
    rows = [{k: r.get(k) for k in fields} | dict(catalog_sha256=hashlib.sha256((ROOT / r["catalog_file"]).read_bytes()).hexdigest())
            | {k: check.get(k) for k in ("http_status", "host", "hdf5_signature", "error")}
            for r, check in zip(assets, checks, strict=True)]
    export(f"{STEM}_opera_assets", rows)
    return dict(opera_matching_cslc=sum(r["kind"] == "CSLC_VV" for r in rows), opera_static_layers=len(static),
                opera_hdf5_signatures=sum(r["hdf5_signature"] for r in rows),
                opera_http_status=dict(Counter(r["http_status"] for r in rows)))


def collect_slc(products, contexts):
    keys = ("cdse_uuid", "swath", "polarization", "asf_data_url", "asf_annotation_url", "cdse_catalog_file", "asf_catalog_file")

    def collect_parent(p):
        source = bursts(p)
        coverage, requests = {}, []
        for b in source["bursts"]:
            covered = sorted(roof for roof, context in contexts.items() if b["footprint"].covers(context))
            if not covered or not b["cdse_uuid"]:
                continue
            coverage[b["cdse_uuid"]] = covered
            requests.append({k: b[k] for k in keys} | dict(scope="nrc_1402673", product_id=p["product_id"],
                            product_name=p["product_name"], weeks=""))
        if set().union(*(set(v) for v in coverage.values())) != contexts.keys():
            raise ValueError(f"Incomplete full-context burst coverage for {p['product_name']}")
        found, assets = parent_files((p["product_id"], requests), store=STORE / "slc")
        return [dict(r, full_context_roofs_json=json.dumps(coverage[r["cdse_uuid"]]),
                     hours_from_reported_event=p["hours_from_reported_event"]) for r in found], assets, liveeo_inventory(p)

    with ThreadPoolExecutor(max_workers=2) as pool:
        retrieved = list(pool.map(collect_parent, products))
    found = [r for group, _, _ in retrieved for r in group]
    assets = [r for _, group, _ in retrieved for r in group]
    inventory = [r for _, _, r in retrieved]
    jobs = [(label, r["cdse_uuid"], r[key]) for r in found for label, key in (
        ("cdse_burst", "cdse_burst_download_url"), ("cdse_direct_burst", "cdse_direct_burst_url"),
        ("asf_burst_tiff", "asf_data_url"), ("asf_burst_xml", "asf_annotation_url")) if r[key]]
    jobs.extend(("cdse_parent_tiff", r["s3_path"], r["download_url"]) for r in assets if r["role"] == "measurement")
    jobs.extend(("ceda_parent_zip", r["product_id"], base + ".zip?download=1")
                for r in products for base in archive_urls(r["product_name"]))
    with ThreadPoolExecutor(max_workers=6) as pool:
        probes = list(pool.map(probe, jobs))
    for suffix, rows in (("slc_bursts", found), ("slc_assets", assets), ("slc_probes", probes), ("slc_liveeo", inventory)):
        export(f"{STEM}_{suffix}", rows)
    return dict(slc_bursts=len(found), slc_parent_members=len(assets), liveeo_objects=sum(r["object_count"] for r in inventory),
                anonymous_slc_access={k: dict(Counter(r["status"] for r in probes if r["route"] == k))
                                      for k in sorted({r["route"] for r in probes})},
                source_tiff_signatures=sum(r["source_signature"] == "TIFF" for r in probes),
                source_zip_signatures=sum(r["source_signature"] == "ZIP" for r in probes),
                **collect_opera(products, found, contexts))


def slc_geometry(orbits, targets):
    found = read(f"{STEM}_slc_bursts")
    measurements = {r["s3_path"]: r["bytes"] for r in read(f"{STEM}_slc_assets") if r["role"] == "measurement"}
    orbit_by_id = {r["product_id"]: r for r in orbits}
    groups = [(ident, [dict(r, cdse_parent_swath_bytes=measurements[r["measurement_s3_path"]])
                       for r in found if r["product_id"] == ident]) for ident in orbit_by_id]
    with ThreadPoolExecutor(max_workers=2) as pool:
        packets = list(pool.map(collect_annotations, groups))
    export(f"{STEM}_slc_annotation_access", [parent for parent, _, _ in packets])
    annotations = [r for _, rows, _ in packets for r in rows]
    export(f"{STEM}_slc_annotations", annotations)
    export(f"{STEM}_slc_annotation_files", [r for _, _, files in packets for r in files])
    native = {r["cdse_uuid"]: NativeGeometry(r) for r in annotations}
    if any(max(model.grid_errors) > .05 for model in native.values()):
        raise ValueError("Native SLC mapping differs from SAFE geolocation grid")
    rows = []
    for r in found:
        ident = r["cdse_uuid"]
        model = native.get(ident)
        burst = burst_catalog(r["cdse_catalog_file"])[ident]
        for key in json.loads(r["full_context_roofs_json"]):
            roof = targets[key]
            g = tank_geometry(roof, orbit_by_id[r["product_id"]], burst)
            roi = model.window(roof, g["rim_zero_doppler_at_utc"]) if model else {}
            rows.append(dict(candidate_id=key, cdse_uuid=ident, product_id=r["product_id"], polarization=r["polarization"],
                             native_annotation_verified=model is not None, **g,
                             **{k: roi.get(k) for k in ("first_line", "end_line_exclusive", "first_sample", "end_sample_exclusive",
                                                        "full_window_valid_by_native_mask")}))
    export(f"{STEM}_slc_geometry", rows)
    shifts = [r["rim_to_ground_slant_range_m"] for r in rows]
    return dict(slc_native_annotations_verified=len(native), slc_orbit_geometry_rows=len(rows),
                slc_native_windows_planned=sum(r["full_window_valid_by_native_mask"] is True for r in rows),
                slc_windows_inside_cdse_catalog_clock=sum(r["rim_zero_doppler_inside_cdse_catalog_time"] for r in rows),
                slc_shell_slant_range_shift_min_m=min(shifts), slc_shell_slant_range_shift_max_m=max(shifts),
                slc_parent_metadata_http_status=dict(Counter(p["metadata_http_status"] for p, _, _ in packets)))


def grd_diagnostic(products, geometry, targets, reuse):
    locations = {k: (g["longitude"], g["latitude"], 2*g["radius_m"]) for k, g in geometry.items()}
    paths = [STORE / "grd" / p["product_name"].removesuffix(".SAFE") / "native_manifest.json" for p in products]
    if reuse:
        manifests = [json.loads(path.read_bytes()) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            packets = list(pool.map(lambda p: collect(dict(p, selection_reasons="nrc_1402673"), locations,
                                                      store=STORE / "grd"), products))
        manifests = [m for m, _ in packets]
        rows = [r for _, group in packets for r in group]
        export(f"{STEM}_grd_native_windows", rows)
    rows = read(f"{STEM}_grd_native_windows")
    clocks, metadata = saved_assets(manifests)
    unique = saved_windows(rows)
    verified = source_sample(unique, per_product=True)
    fits, sigma = [], {}
    for m, p in zip(manifests, products, strict=True):
        ident = p["product_id"]
        if m["product"]["product_id"] != ident:
            raise ValueError(f"Unexpected manifest for {ident}")
        bands = {pol: read_band(m, pol) for pol in m["bands"]}
        sigma.update({k: window(r, bands[r["polarization"]], output_root=STORE / "sigma0")
                      for k, r in unique.items() if r["product_id"] == ident})
        group = [dict(r, hours_from_reported_event=p["hours_from_reported_event"],
                      geometry_dataset=geometry[r["candidate_id"]]["geometry_dataset"])
                 for r in fit_scene(m, rows, sigma, geometry) if r["candidate_id"] in targets]
        if {r["candidate_id"] for r in group} != targets:
            raise ValueError(f"Missing target roof fits for {ident}")
        fits.append({r["candidate_id"]: r for r in group})
    earlier, later = fits
    changes = []
    for roof in sorted(targets):
        a, b = earlier[roof], later[roof]
        valid = a["status"] == b["status"] == PASS
        changes.append(dict(candidate_id=roof, earlier_product=a["product_id"], later_product=b["product_id"],
                            earlier_local_date=a["radar_local_date"], later_local_date=b["radar_local_date"],
                            earlier_status=a["status"], later_status=b["status"],
                            roof_height_change_m=a["geometry_template_gap_m"]-b["geometry_template_gap_m"] if valid else None))
    rows = [r for group in fits for r in group.values()]
    sources = [dict(product_id=p["product_id"], product_name=p["product_name"],
                    source_manifest=str(path.relative_to(ROOT)), archive_inputs_ready_utc=clocks[p["product_id"]].isoformat())
               for p, path in zip(products, paths, strict=True)]
    for suffix, records in (("grd_sigma0", list(sigma.values())), ("grd_fits", rows), ("grd_changes", changes), ("grd_sources", sources)):
        export(f"{STEM}_{suffix}", records)
    return dict(grd_metadata_verified=metadata, grd_native_windows=len(unique), grd_source_windows_read_back=verified,
                grd_target_fits=len(rows), grd_fit_status=dict(Counter(r["status"] for r in rows)),
                grd_two_pass_changes=sum(r["roof_height_change_m"] is not None for r in changes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-grd", action="store_true")
    args = parser.parse_args()
    targets, geometry, slc, grd = inputs()
    contexts = {k: Point(geometry[k]["east_m"], geometry[k]["north_m"]).buffer(geometry[k]["radius_m"] + CONTEXT_M)
                for k in sorted(targets)}
    with ThreadPoolExecutor(max_workers=2) as pool:
        amplitude = pool.submit(grd_diagnostic, grd, geometry, targets, args.reuse_grd)
        complex_data = pool.submit(collect_slc, slc, contexts)
        report = dict(capacity_compatible_roofs=len(targets), neighbour_geometry_count=len(geometry)-len(targets),
                      slc_products=len(slc), grd_products=len(grd), **amplitude.result(), **complex_data.result())
    manifests = [json.loads((ROOT / r["source_manifest"]).read_bytes()) for r in read(f"{STEM}_grd_sources")]
    grd_by_orbit = {(r["product"]["satellite"], r["product"]["absolute_orbit"]): r for r in manifests}
    orbits = [orbit_check(orbit_for(dict(p, scope="nrc_1402673"), accept_relative_osv_labels=True), grd_by_orbit) for p in slc]
    export(f"{STEM}_orbit_checks", orbits)
    targets_hae = with_ellipsoid_heights({k: geometry[k] for k in targets})
    report.update(precise_orbit_files=len(orbits), precise_orbits_verified_against_grd=sum(r["verified_state_vector_agreement"] for r in orbits),
                  **slc_geometry(orbits, targets_hae))
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
