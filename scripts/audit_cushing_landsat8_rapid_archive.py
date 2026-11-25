"""Inventory the Landsat 8 real-time bundles that can fill development weeks."""

import hashlib
import io
import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from shapely.geometry import Point, shape

from build_cross_sensor_shadow import pan_qa
from build_landsat_coverage import native_masks
from build_landsat_pan_roof_angles import angle_values, prepare_tool
from collect_cushing_landsat7 import DERIVED, LANDSAT, ROOT, check_grid, crop, export, legacy_geometry, parse_mtl, rows, save_json, source_bytes
from collect_landsat import GCS_OBJECT, PAN_PRODUCT, catalog, daylight, gcs_index, scene_key
from collect_landsat_c2_registration_context import gcs_version
from collect_public import write

STEM = "cushing_landsat8_rapid_archive"
STORE = LANDSAT / STEM
SUFFIXES = ("B8.TIF", "BQA.TIF", "MTL.txt", "ANG.txt")


def archive_products():
    paths = sorted((LANDSAT / "legacy_gcs_index").glob("*_TP.json"))
    prefixes = {p for path in paths for p in json.loads(path.read_bytes())["records"]
                if p.rstrip("/").endswith("_RT")}
    if not prefixes:
        raise ValueError("The Landsat 8 real-time prefix inventory is empty")
    geometry = legacy_geometry()
    to_geographic = Transformer.from_crs(32614, 4326, always_xy=True).transform
    centres = {k: Point(*to_geographic(float(r["center_east_m"]), float(r["center_north_m"]))) for k,r in geometry.items()}
    years = sorted({p.split("/")[-2].split("_")[3][:4] for p in prefixes})
    references = {scene_key(i):i for year in years for i in catalog(int(year), "usgs")
                  if i["properties"]["platform"] == "LANDSAT_8" and daylight(i)}
    for prefix in sorted(prefixes):
        name = prefix.rstrip("/").rsplit("/", 1)[-1]
        match = PAN_PRODUCT.fullmatch(name)
        if match is None:
            raise ValueError(f"Wrong real-time Landsat 8 product: {name}")
        track, acquired, _, _ = match.groups()
        reference = references.get((track, acquired))
        covered = []
        if reference:
            footprint = shape(reference["geometry"])
            covered = [tank for tank, point in centres.items() if footprint.covers(point)]
        yield name, prefix, reference, covered


def bundle(job):
    name, prefix, reference, covered = job
    directory = STORE / name
    objects = {Path(r["name"]).name:r for r in gcs_index(prefix, directory / "objects.json")}
    assets = {suffix:objects.get(f"{name}_{suffix}") for suffix in SUFFIXES}
    missing = [suffix for suffix, item in assets.items() if item is None]
    result = dict(product_id=name, collection2_reference_id=reference["id"], nominal_dated_roof_centers=len(covered),
        source_index=str((directory / "objects.json").relative_to(ROOT)), missing_assets="|".join(missing),
        captured_at_utc=None, archive_bundle_available_utc=None, b8_url=None, b8_archive_crc32c=None,
        b8_archive_generation=None, b8_archive_bytes=None)
    if missing:
        return result
    mtl = parse_mtl(source_bytes(assets["MTL.txt"], directory / "MTL.txt"))
    captured = datetime.fromisoformat(f"{mtl['DATE_ACQUIRED']}T{mtl['SCENE_CENTER_TIME']}")
    reference_capture = datetime.fromisoformat(reference["properties"]["datetime"])
    if (mtl["LANDSAT_PRODUCT_ID"] != name or mtl["SPACECRAFT_ID"] != "LANDSAT_8"
        or abs((captured-reference_capture).total_seconds()) > 1):
        raise ValueError(f"Landsat 8 Collection 1 and 2 acquisition clocks differ for {name}")
    known = max(datetime.fromisoformat(r["timeCreated"]) for r in assets.values())
    if known < captured:
        raise ValueError(f"Landsat 8 archive files predate acquisition for {name}")
    pan = assets["B8.TIF"]
    url = GCS_OBJECT + pan["name"]
    return dict(result, captured_at_utc=captured.isoformat(), archive_bundle_available_utc=known.isoformat(),
                b8_url=url, b8_archive_crc32c=pan["crc32c"], b8_archive_generation=gcs_version(url, pan),
                b8_archive_bytes=int(pan["size"]))


def pan_pixels(pan, bqa, bounds):
    west, south, east, north = bounds
    dn, grid, window = crop(pan, bounds)
    q, qa_grid, _ = crop(bqa, (west-30, south-30, east+30, north+30))
    rr, cc = np.indices(dn.shape)
    xx, yy = grid * (cc+.5, rr+.5)
    view = SimpleNamespace(transform=qa_grid, width=q.shape[1], height=q.shape[0])
    flags = np.logical_or.reduce(native_masks(q, None, "pan")).astype("uint16")
    _, ok, bad = pan_qa(view, xx, yy, q, (q & 1)==0, flags)
    return dn, grid, window, q, qa_grid, xx, yy, ok & (bad==0) & (dn>0) & (dn<65535)


def stage(r, tool, geometry, *, store=STORE):
    product = r["product_id"]
    directory = store / product
    packet, receipt = directory / "native.npz", directory / "native.json"
    coefficients = directory / "ANG.txt"
    if receipt.is_file() and packet.is_file() and coefficients.is_file():
        item = json.loads(receipt.read_bytes())
        if (item["product_id"] == product and item["b8_archive_generation"] == r["b8_archive_generation"]
            and item["native_sha256"] == hashlib.sha256(packet.read_bytes()).hexdigest()
            and item["angle_sha256"] == hashlib.sha256(coefficients.read_bytes()).hexdigest()
            and sorted(x["candidate_id"] for x in item["roofs"]) == r["dated_candidate_ids"].split("|")):
            return item
    objects = {Path(a["name"]).name:a for a in json.loads((ROOT/r["source_index"]).read_bytes())["records"]}
    assets = {suffix:objects[f"{product}_{suffix}"] for suffix in SUFFIXES}
    qa = source_bytes(assets["BQA.TIF"])
    raw_angles = source_bytes(assets["ANG.txt"], coefficients)
    mtl = parse_mtl(source_bytes(assets["MTL.txt"], directory / "MTL.txt"))
    if gcs_version(r["b8_url"], assets["B8.TIF"]) != r["b8_archive_generation"]:
        raise ValueError(f"Landsat 8 object generation changed for {product}")
    with (rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="2"),
          rasterio.open(r["b8_url"]) as pan, MemoryFile(qa) as qa_file, qa_file.open() as bqa):
        check_grid(pan, mtl, 15, "uint16", "PANCHROMATIC")
        check_grid(bqa, mtl, 30, "uint16", "REFLECTIVE")
        if pan.crs.to_epsg() != 32614:
            raise ValueError(f"Unexpected Landsat 8 CRS for {product}")
        sites = []
        for tank in r["dated_candidate_ids"].split("|"):
            g = geometry[tank]
            x, y = (float(g[f"center_{axis}_m"]) for axis in ("east", "north"))
            sites.append((tank, x, y, float(g["diameter_m"])/2, *pan.index(x, y)))
        margin = 35
        west = min(x-radius-margin for _, x, _, radius, _, _ in sites)
        south = min(y-radius-margin for _, _, y, radius, _, _ in sites)
        east = max(x+radius+margin for _, x, _, radius, _, _ in sites)
        north = max(y+radius+margin for _, _, y, radius, _, _ in sites)
        dn, grid, window, q, qa_grid, xx, yy, usable = pan_pixels(pan,bqa,(west,south,east,north))
        angles = angle_values(tool, coefficients, [(line, sample) for _, _, _, _, line, sample in sites])
        roofs = []
        for tank, x, y, radius, line, sample in sites:
            in_roof = (xx-x)**2+(yy-y)**2 <= radius**2
            angle = angles[line, sample]
            roofs.append(dict(product_id=product, candidate_id=tank, center_east_m=x, center_north_m=y, radius_m=radius,
                b8_native_line=line, b8_native_sample=sample, roof_pixels=int(in_roof.sum()),
                roof_pixels_passing_qa=int((in_roof&usable).sum()), **angle))
        with io.BytesIO() as out:
            np.savez_compressed(out, pan_dn=dn, pan_usable=usable, bqa=q, pan_transform=tuple(grid)[:6],
                qa_transform=tuple(qa_grid)[:6], crs=pan.crs.to_string())
            payload = out.getvalue()
    if gcs_version(r["b8_url"], assets["B8.TIF"]) != r["b8_archive_generation"]:
        raise ValueError(f"Landsat 8 object generation changed while reading {product}")
    write(packet, payload)
    item = dict(product_id=product, b8_archive_generation=r["b8_archive_generation"], native_packet=str(packet.relative_to(ROOT)),
        native_sha256=hashlib.sha256(payload).hexdigest(), angle_file=str(coefficients.relative_to(ROOT)),
        angle_sha256=hashlib.sha256(raw_angles).hexdigest(), pan_crop_first_row=int(window.row_off),
        pan_crop_first_col=int(window.col_off), reflectance_mult_b8=float(mtl["REFLECTANCE_MULT_BAND_8"]),
        reflectance_add_b8=float(mtl["REFLECTANCE_ADD_BAND_8"]), roofs=roofs)
    save_json(receipt, item)
    return item


def weekly_coverage(calendar, scenes):
    landed = defaultdict(list)
    for r in scenes:
        landed[r["week_ending"]].append(r)
    landsat7 = defaultdict(set)
    path = DERIVED / "cushing_landsat7_rapid_archive_native_scenes.csv"
    for r in rows(path):
        if int(r["roofs_also_documented_before_cutoff"]) > 0:
            landsat7[r["week_ending"]].add(r["product_id"])
    for r in calendar:
        week = r["week_ending"]
        if not "2017" <= week[:4] <= "2021":
            continue
        clear = [s for s in landed[week] if s["roofs_with_clear_pixels_and_angles"] > 0]
        yield dict(week_ending=week, release_date=r["release_date"], cutoff_utc=r["cutoff_utc"],
            baseline_b04_tanks=int(r["baseline_b04_tanks"]), landsat7_rt_scenes_with_usable_dated_roofs=len(landsat7[week]),
            landsat8_rt_scenes_with_usable_dated_roofs=len(clear),
            landsat8_rt_scene_ids="|".join(sorted(s["product_id"] for s in clear)))


def main():
    inventory = list(archive_products())
    jobs = [job for job in inventory if job[-1]]
    with ThreadPoolExecutor(max_workers=8) as pool:
        bundles = list(pool.map(bundle, jobs))
    calendar = [r for r in rows(DERIVED / "sentinel2_space_time_coverage_eia_comparison.csv") if r["study_phase"] == "development"]
    ends = [datetime.fromisoformat(r["current_measurement_at_utc"]) for r in calendar]
    if ends != sorted(set(ends)):
        raise ValueError("The EIA development calendar is not strictly sorted")
    registry = {r["candidate_id"]: r for r in rows(DERIVED / "sentinel2_space_time_coverage_roof_registry.csv")}
    output = []
    for r, (_, _, _, covered) in zip(bundles, jobs, strict=True):
        fields = dict(week_ending=None, release_date=None, cutoff_utc=None, bundle_known_before_cutoff=False,
                      dated_roof_centers_before_cutoff=0, baseline_b04_tanks=None, dated_candidate_ids="")
        if r["captured_at_utc"]:
            capture = datetime.fromisoformat(r["captured_at_utc"])
            index = bisect_left(ends, capture)
            if index >= len(calendar):
                raise ValueError(f"Landsat 8 scene is outside the development calendar: {r['product_id']}")
            week = calendar[index]
            prior, cutoff = (datetime.fromisoformat(week[k]) for k in ("previous_measurement_at_utc", "cutoff_utc"))
            if not prior < capture <= ends[index]:
                raise ValueError(f"Wrong Landsat 8 reporting interval for {r['product_id']}")
            dated = [tank for tank in covered if registry[tank]["has_both_dated_sources"] == "True"
                     and registry[tank]["first_seen"] <= capture.date().isoformat()
                     and max(registry[tank]["roof_evidence_date"], registry[tank]["lidar_source_available_date"]) < cutoff.date().isoformat()]
            fields.update(week_ending=week["week_ending"], release_date=week["release_date"], cutoff_utc=cutoff.isoformat(),
                bundle_known_before_cutoff=datetime.fromisoformat(r["archive_bundle_available_utc"]) < cutoff,
                dated_roof_centers_before_cutoff=len(dated), baseline_b04_tanks=int(week["baseline_b04_tanks"]),
                dated_candidate_ids="|".join(sorted(dated)))
        output.append(r | fields)
    export(f"{STEM}_bundles", output)
    timely = [r for r in output if r["bundle_known_before_cutoff"]]
    actionable = [r for r in timely if r["dated_roof_centers_before_cutoff"]]
    tool, geometry = prepare_tool(), legacy_geometry()
    with ThreadPoolExecutor(max_workers=6) as pool:
        items = list(pool.map(lambda r:stage(r, tool, geometry), actionable))
    native = [dict(r, native_packet=item["native_packet"], native_sha256=item["native_sha256"],
        angle_file=item["angle_file"], angle_sha256=item["angle_sha256"],
        roofs_with_clear_pixels_and_angles=sum(x["roof_pixels_passing_qa"]>0 and x["angle_status"]=="resolved" for x in item["roofs"]))
        for r,item in zip(actionable, items, strict=True)]
    if native:
        export(f"{STEM}_native_scenes", native)
        export(f"{STEM}_roof_windows", [r for item in items for r in item["roofs"]])
    weeks = list(weekly_coverage(calendar, native))
    export(f"{STEM}_week_coverage", weeks)
    report = dict(gcs_l1tp_rt_prefixes=len(inventory), matched_collection2_references=sum(i[2] is not None for i in inventory),
        collection2_footprints_intersecting_dated_roofs=len(output), complete_bundles=sum(not r["missing_assets"] for r in output),
        bundles_before_eia_cutoff=len(timely), bundles_before_cutoff_by_year=dict(Counter(r["captured_at_utc"][:4] for r in timely)),
        weeks_with_timely_bundles_and_dated_geometry=len({r["week_ending"] for r in actionable}),
        weeks_with_timely_bundles_and_dated_geometry_without_b04=len({r["week_ending"] for r in actionable if not r["baseline_b04_tanks"]}),
        native_scenes=len(native), native_roof_windows=sum(len(item["roofs"]) for item in items),
        native_roof_windows_with_clear_pixels_and_angles=sum(r["roofs_with_clear_pixels_and_angles"] for r in native),
        weeks_with_clear_pixels_and_angles=len({r["week_ending"] for r in native if r["roofs_with_clear_pixels_and_angles"]}),
        weeks_with_clear_pixels_and_angles_without_b04=len({r["week_ending"] for r in native
            if r["roofs_with_clear_pixels_and_angles"] and not r["baseline_b04_tanks"]}),
        weeks_with_landsat8_coverage_and_no_landsat7_rt_or_b04=sum(r["landsat8_rt_scenes_with_usable_dated_roofs"]>0
            and r["landsat7_rt_scenes_with_usable_dated_roofs"]==0 and r["baseline_b04_tanks"]==0 for r in weeks),
        scope="Collection 1 Landsat 8 RT inventory and historical source clocks. Usable pixel windows and B8 angles are staged. Height and relative registration require separate measurements.")
    save_json(DERIVED / f"{STEM}_summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
