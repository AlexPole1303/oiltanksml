"""Recover exact-grid Landsat windows around the operator's inspection dates."""

import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from xml.etree import ElementTree as ET

from affine import Affine
from pyproj import Transformer
from rasterio.io import MemoryFile

from audit_cushing_b8_archive import scene_file
from audit_landsat_c2_alternative_access import copernicus, probe
from audit_landsat_esri_b8_registration import context_crop, padded_chip_bounds
from build_cross_sensor_shadow import published_lsf
from build_cushing_icesat2_b8_inputs import b8_window
from build_landsat_c2_b8_measurements import ANGLES, RESOLVED, engineering_inputs, prepare_native
from build_landsat_esri_b8_controls import fit_tile
from build_s1_reference_matches import DERIVED, export, read
from collect_landsat import LANDSAT, catalog
from collect_landsat_asset_times import asset_snapshot
from collect_landsat_c2_gap_geometry import native_b8_grid, packet, read_azure_asset, roof_angles
from collect_landsat_esri_b8 import query, raster_request
from collect_public import ROOT, get, write
from stage_landsat_c2_b8_inputs import QA_LABELS, expected_sha512

STEM = "cushing_operator_inspection_b8"
STORE = LANDSAT / STEM
NATIVE = DERIVED / f"{STEM}_native.sqlite"
SOURCE = "cushing_operator_inspection_catalog_candidates"


def instant(text):
    return datetime.fromisoformat(text)


def scene_inputs(name, official, level2, tank_ids, geometry, tool, *, store, purpose, geometry_basis):
    folder = store / name
    support = asset_snapshot(level2, QA_LABELS, folder / "level2_objects.json", ("mtl.json", "mtl.xml"))
    files = support["assets"]
    metadata_file = folder / "MTL.json"
    if "mtl.json" in files:
        document = json.loads(read_azure_asset(files["mtl.json"], metadata_file))
    else:
        xml = ET.fromstring(read_azure_asset(files["mtl.xml"], folder / "MTL.xml"))
        document = {xml.tag: {group.tag: {field.tag: field.text for field in group} for group in xml}}
        write(metadata_file, json.dumps(document, indent=2).encode())
    metadata = document["LANDSAT_METADATA_FILE"]
    processing = metadata["LEVEL1_PROCESSING_RECORD"]
    if processing["LANDSAT_PRODUCT_ID"] != name:
        raise ValueError(f"Level-2 metadata describes a different Level-1 version for {name}")
    product = packet(
        dict(product_id=name, platform=official["properties"]["platform"].upper().replace("-", "_").replace(" ", "_"),
             captured_at_utc=official["properties"]["datetime"], level1_generated_at_utc=processing["DATE_PRODUCT_GENERATED"]),
        official, level2, metadata_file, store=store,
    )
    asset = official["assets"]["pan"]
    if tuple(asset["proj:shape"]) != (product["native_lines"], product["native_samples"]) or (
        Affine(*asset["proj:transform"][:6]) != native_b8_grid(product)
    ):
        raise ValueError(f"LandsatLook B8 grid differs from ANG and MTL for {name}")
    angle_hash = expected_sha512(official["assets"]["ANG.txt"])
    if hashlib.sha512((ROOT / product["angle_local_file"]).read_bytes()).hexdigest() != angle_hash:
        raise ValueError(f"Level-2 ANG differs from the exact Level-1 checksum for {name}")
    stage = dict(
        purpose=purpose, local_angle_file=product["angle_local_file"],
        source_angle_sha512=angle_hash, expected_original_b8_sha512=expected_sha512(asset),
        b8_saturation_flag_in_scene_mtl=metadata["IMAGE_ATTRIBUTES"]["SATURATION_BAND_8"],
        level2_support_snapshot=str((folder / "level2_objects.json").relative_to(ROOT)),
        source_metadata_format="json" if "mtl.json" in files else "xml",
    )
    for label in QA_LABELS:
        path = folder / f"{label}.tif"
        checksum = expected_sha512(official["assets"][label])
        if hashlib.sha512(read_azure_asset(files[label], path)).hexdigest() != checksum:
            raise ValueError(f"{label} differs from the exact Level-1 checksum for {name}")
        stage.update({f"{label}_path": str(path.relative_to(ROOT)), f"{label}_sha512": checksum})
    windows, angles = [], {}
    for row in roof_angles(tool, product, tank_ids & geometry.keys(), geometry, geometry_basis):
        tank = row["candidate_id"]
        if row["angle_status"] != "resolved":
            continue
        g = geometry[tank]
        window = b8_window(g | dict(product_id=name, candidate_id=tank, radius_m=float(g["diameter_m"]) / 2),
                           product, row, g)
        windows.append(window)
        angles[name, tank] = (window["center_native_line"], window["center_native_sample"], *(row[k] for k in ANGLES))
    return product, stage, windows, angles


def reflection_window(name, official, esri, tank_ids, points, product, windows, *, store):
    pan = official["assets"]["pan"]
    grid = Affine(*pan["proj:transform"][:6])
    ny, nx = pan["proj:shape"]
    crs = f"EPSG:{official['properties']['proj:epsg']}"
    reference = product or dict(
        product_id=name, crs=crs, native_lines=ny, native_samples=nx,
        ul_b8_center_x_m=grid.c + 7.5, ul_b8_center_y_m=grid.f - 7.5,
    )
    if windows:
        bounds = tuple(fn(int(w[key]) for w in windows) + offset for key, fn, offset in (
            ("b8_first_line", min, 0), ("b8_last_line", max, 1),
            ("b8_first_sample", min, 0), ("b8_last_sample", max, 1),
        ))
    else:
        transformer = Transformer.from_crs(4326, crs, always_xy=True)
        centres = [(~grid) * transformer.transform(*points[k]) for k in tank_ids]
        bounds = (int(min(y for x, y in centres)) - 12, int(max(y for x, y in centres)) + 13,
                  int(min(x for x, y in centres)) - 12, int(max(x for x, y in centres)) + 13)
    r0, r1, c0, c1 = bounds
    if not (0 <= r0 < r1 <= ny and 0 <= c0 < c1 <= nx):
        raise ValueError(f"Requested window exceeds B8 grid for {name}")
    url, expected = raster_request(reference, esri["OBJECTID"], bounds)
    path = store / name / "reflectance.tif"
    data = get(url, path)
    with MemoryFile(data) as mem, mem.open() as ds:
        if ds.transform != expected or ds.crs.to_string() != crs or ds.dtypes != ("int16",):
            raise ValueError(f"Esri B8 export has an unexpected grid for {name}")
        pixels = ds.read(1)
        useful = int(((ds.read_masks(1) > 0) & (pixels > 0) & (pixels < 10000)).sum())
        clipped = ds.shape != (r1 - r0, c1 - c0)
        if clipped and (ds.height > r1-r0 or ds.width > c1-c0):
            raise ValueError(f"Esri B8 export exceeds the requested window for {name}")
        height, width = ds.shape
    tile = dict(product_id=name, object_id=esri["OBJECTID"], first_line=r0, end_line_exclusive=r1,
                first_sample=c0, end_sample_exclusive=c1, reflectance_scale=.0001, useful_pixels=useful,
                local_path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(data).hexdigest(), source_url=url,
                exported_lines=height, exported_samples=width)
    if clipped:
        overlap = dict(overlap_status="clipped_export_requires_complete_window", shared_valid_pixel_comparisons=0)
    else:
        overlap = context_crop(reference, esri["OBJECTID"], padded_chip_bounds(reference, [tile]), [tile])
    return tile, overlap


def download_files(scenes, official, candidates):
    requests = [dict(product_id=r["product_id"], platform=official[r["product_id"]]["properties"]["platform"],
                     purpose="operator_inspection", captured_at_utc=r["captured_at_utc"],
                     original_b8_filename=f"{r['product_id']}_B8.TIF", b8_status="awaiting_original_b8",
                     expected_original_b8_sha512=r["expected_original_b8_sha512"]) for r in scenes]
    mirrors = copernicus(requests, store=STORE)
    main_queue = {r["product_id"] for r in read("cushing_b8_download_manifest")}
    on_date = defaultdict(set)
    for row in candidates:
        if int(row["scene_minus_inspection_local_days"]) == 0:
            on_date[row["source_catalog_id"]].add((row["terminal_id"], row["equipment_id"], row["inspection_local_date"]))

    def lookup(request):
        name = request["product_id"]
        item = official[name]
        mirror = mirrors.get(name)
        info = scene_file(request, mirror, store=STORE / "cdse") if mirror else {}
        url = info.get("cdse_b8_download_url")
        access = probe(url) if url else {}
        support = {label: item["assets"][asset] for label, asset in (
            ("mtl", "MTL.json"), ("angles", "ANG.txt"), ("qa_pixel", "qa_pixel"), ("qa_radsat", "qa_radsat"))}
        return request | dict(
            scene_id=item["properties"]["landsat:scene_id"], same_day_inspections=len(on_date[name]),
            in_main_weekly_or_control_queue=name in main_queue, usgs_b8_url=item["assets"]["pan"]["href"],
            **{f"usgs_{label}_url": asset["href"] for label, asset in support.items()},
            **{f"expected_{label}_sha512": expected_sha512(asset) for label, asset in support.items()},
            cdse_b8_url=url, cdse_b8_bytes=info.get("cdse_b8_bytes"), cdse_nodes_snapshot=info.get("cdse_nodes_snapshot"),
            cdse_anonymous_http_status=access.get("http_status"), cdse_anonymous_tiff_signature=access.get("tiff_signature"),
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(lookup, requests))
    return sorted(rows, key=lambda r: (-r["same_day_inspections"], r["captured_at_utc"], r["product_id"]))


def recover(tanks, official, *, store, native, purpose, dated_geometry=None, geometry_basis="dated_2011_lidar"):
    from build_landsat_pan_roof_angles import prepare_tool

    points = {r["candidate_id"]: (float(r["longitude"]), float(r["latitude"])) for r in read("tank_engineering_registry")}
    if official.keys() != tanks.keys():
        raise ValueError("A requested scene is absent from the saved exact USGS catalog")
    _, payload, hits = query("Name IN (" + ",".join(f"'{n}'" for n in sorted(tanks)) + ")")
    write(store / "esri_catalog.json", payload)
    esri = {r["Name"]: r for r in hits}
    if len(esri) != len(hits) or set(esri) - tanks.keys():
        raise ValueError("Unexpected duplicate or extra Esri product")
    years = {int(i["properties"]["datetime"][:4]) for i in official.values()}
    level2 = {r["id"]: r for year in sorted(years) for r in catalog(year, "pc")}
    _, legacy = engineering_inputs()
    geometry = legacy if dated_geometry is None else dated_geometry
    tool = prepare_tool()

    def fetch(name):
        primary = official[name]
        source = esri.get(name)
        _, _, wrs, day, _, _, tier = name.split("_")
        l2 = level2.get(f"{name[:4]}_L2SP_{wrs}_{day}_02_{tier}")
        fields = dict(product_id=name, roof_candidates=len(tanks[name]),
                      captured_at_utc=primary["properties"]["datetime"], esri_exact_product_available=source is not None,
                      usgs_platform=primary["properties"]["platform"], esri_sensor_label=source["SensorName"] if source else None,
                      level2_catalog_item=l2["id"] if l2 else None,
                      original_b8_url=primary["assets"]["pan"]["href"],
                      expected_original_b8_sha512=expected_sha512(primary["assets"]["pan"]),
                      original_b8_verified=False)
        if source is None:
            return fields | dict(status="exact_esri_scene_missing"), None
        stamp = datetime.fromtimestamp(source["AcquisitionDate"] / 1000, timezone.utc)
        if (source["LANDSAT_PRODUCT_ID"] != name or source["Category"] != 1
                or source["LANDSAT_SCENE_ID"] != primary["properties"]["landsat:scene_id"]
                or abs((stamp - instant(fields["captured_at_utc"])).total_seconds()) >= 1):
            raise ValueError(f"Esri version or capture time differs for {name}")
        product, stage, windows, angles = scene_inputs(
            name, primary, l2, tanks[name], geometry, tool, store=store, purpose=purpose, geometry_basis=geometry_basis,
        ) if l2 else (None, None, [], {})
        tile, overlap = reflection_window(name, primary, source, tanks[name], points, product, windows, store=store)
        if overlap["overlap_status"] != "shared_pixels_match":
            status = "reflectance_overlap_unverified"
        elif windows:
            status = "native_qa_and_angles_ready"
        elif stage:
            status = "reflectance_only_no_dated_geometry"
        else:
            status = "reflectance_only_exact_level2_support_missing"
        record = fields | dict(
            status=status,
            reflectance_path=tile["local_path"], reflectance_sha256=tile["sha256"], reflectance_source_url=tile["source_url"],
            reflectance_exported_lines=tile["exported_lines"], reflectance_exported_samples=tile["exported_samples"],
            useful_pixels=tile["useful_pixels"], overlap_status=overlap["overlap_status"],
            overlap_shared_pixels=overlap["shared_valid_pixel_comparisons"], roofs_with_angles=len(windows),
            angle_sha512=stage["source_angle_sha512"] if stage else None,
            source_metadata_format=stage["source_metadata_format"] if stage else None,
            qa_pixel_sha512=stage["qa_pixel_sha512"] if stage else None,
            qa_radsat_sha512=stage["qa_radsat_sha512"] if stage else None,
        )
        return record, (product, stage, windows, angles, tile, overlap) if status == "native_qa_and_angles_ready" else None

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, sorted(tanks)))
    scenes = [row for row, _ in results]
    prepared = [p for _, p in results if p]
    stats = prepare_native({p["product_id"]: s for p, s, *_ in prepared},
                           {p["product_id"]: p for p, *_ in prepared},
                           {p["product_id"]: w for p, _, w, *_ in prepared},
                           {k: v for _, _, _, a, *_ in prepared for k, v in a.items()}, database=native,
                           dated_geometry=dated_geometry, geometry_basis=geometry_basis)
    lsf = published_lsf()
    fits = [fit | dict(overlap_status=o["overlap_status"]) for p, s, w, _, t, o in prepared
            for fit in fit_tile(t, w, p, s, lsf, native)]
    normalized = []
    for rows in (scenes, fits):
        fields = tuple(dict.fromkeys(k for row in rows for k in row))
        normalized.append([{k: row.get(k) for k in fields} for row in rows])
    return normalized[0], normalized[1], stats


def main():
    candidates = [r for r in read(SOURCE) if r["sensor"] == "Landsat 8/9 OLI"]
    tanks = defaultdict(set)
    for row in candidates:
        tanks[row["source_catalog_id"]].update(json.loads(row["covered_possible_candidate_ids_json"]))
    official = {i["id"]: i for year in (2025, 2026) for i in catalog(year, "usgs") if i["id"] in tanks}
    scenes, fits, stats = recover(tanks, official, store=STORE, native=NATIVE, purpose="retrospective_operator_inspection")
    by_roof = {(r["product_id"], r["candidate_id"]): r for r in fits}
    matches = []
    for row in candidates:
        name = row["source_catalog_id"]
        for tank in json.loads(row["covered_possible_candidate_ids_json"]):
            fit = by_roof.get((name, tank), {})
            matches.append(dict(
                terminal_id=row["terminal_id"], equipment_id=row["equipment_id"], candidate_id=tank, product_id=name,
                inspection_local_date=row["inspection_local_date"], reported_state=row["reported_state"],
                scene_minus_inspection_local_days=int(row["scene_minus_inspection_local_days"]),
                fit_status=fit.get("fit_status", "measurement_missing"), gap_m=fit.get("estimated_gap_m"),
                admissible_as_paired_gauge=False,
            ))
    export(f"{STEM}_scenes", scenes)
    export(f"{STEM}_fits", fits)
    export(f"{STEM}_matches", matches)
    downloads = download_files(scenes, official, candidates)
    export(f"{STEM}_download_files", downloads)
    resolved = [r for r in matches if r["fit_status"] == RESOLVED]
    summary = dict(
        stats, candidate_products=len(scenes), esri_exact_products=sum(r["esri_exact_product_available"] for r in scenes),
        scene_status=dict(Counter(r["status"] for r in scenes)),
        fit_status=dict(Counter(r["fit_status"] for r in fits)),
        same_day_resolved_inspection_records=len({(r["terminal_id"], r["equipment_id"], r["inspection_local_date"])
                                                 for r in resolved if not r["scene_minus_inspection_local_days"]}),
        cdse_listed_originals=sum(bool(r["cdse_b8_url"]) for r in downloads),
        cdse_original_bytes=sum(r["cdse_b8_bytes"] or 0 for r in downloads),
        cdse_download_response=dict(Counter(r["cdse_anonymous_http_status"] for r in downloads if r["cdse_b8_url"])),
        original_b8_files_verified=0, verified_ground_pairs=0,
        scope="Exact-scene Esri reflectance, retrospective validation only. Ground measurements remain tied to equipment cohorts.",
    )
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload)


if __name__ == "__main__":
    main()
