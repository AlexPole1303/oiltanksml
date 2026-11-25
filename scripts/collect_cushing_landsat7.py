"""Preserve Landsat 7 native pixels and quality masks for historical roof checks."""

import argparse
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import re
import subprocess
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.windows import Window, from_bounds
from shapely.geometry import Point, Polygon

from build_cross_sensor_shadow import pan_qa
from build_shadow_registration_inputs import tank_points
from build_weekly_shadow_panel import DERIVED, export
from build_weekly_shadow_validation import legacy_geometry
from collect_landsat import GCS_OBJECT, LANDSAT, gcs_index
from collect_public import ROOT, TLS, get, write, write_csv
from collect_s2_l1c import crc32c_base64

STEM = "cushing_landsat7"
STORE = LANDSAT / "landsat7_index"
PIXELS = LANDSAT / "landsat7_native"
TOOL_WORK = ROOT / "tmp/landsat7_angle_tool"
TOOL_NAME = "LANDSAT_ANGLES_15_3_0.tgz"
TOOL_URL = f"https://landsat.usgs.gov/sites/default/files/documents/{TOOL_NAME}"
TOOL_SHA256 = "1b1c146b3305ba91570fdd05ab7059d382f67ba8f19952d3ea21d60efb5c6da7"
BULK_URL = "https://landsat.usgs.gov/landsat/metadata_service/bulk_metadata_files/LANDSAT_ETM_C2_L1.csv.gz"
TRACKS = ("026035", "027035", "028035")
CORNERS = ("Upper Left", "Upper Right", "Lower Right", "Lower Left")
TIERS = {"T1": 0, "T2": 1, "RT": 2}
PRODUCT = re.compile(r"LE07_(L1TP|L1GT|L1GS)_(\d{6})_(\d{8})_(\d{8})_01_(T1|T2|RT)")
ASSETS = {"pan": "B8.TIF", "quality": "BQA.TIF", "gap": "GM_B8.TIF.gz", "mtl": "MTL.txt", "angles": "ANG.txt"}
MARGIN_M = 30
UTC = dt.timezone.utc


def save_json(path, value):
    write(path, (json.dumps(value, indent=2) + "\n").encode())


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


class CatalogStream:
    def __init__(self, response):
        self.response = response
        self.sha256 = hashlib.sha256()
        self.size = 0

    def read(self, size=-1):
        payload = self.response.read(size)
        self.sha256.update(payload)
        self.size += len(payload)
        return payload


def refresh_catalog():
    request = Request(BULK_URL, headers={"User-Agent": "Mozilla/5.0", "Accept-Encoding": "identity"})
    regional, count = [], 0
    with urlopen(request, timeout=90, context=TLS) as response:
        if response.status != 200 or "gzip" not in response.headers.get("Content-Type", ""):
            raise ValueError("USGS did not return the complete gzip catalog")
        source = CatalogStream(response)
        with gzip.GzipFile(fileobj=source) as raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            fields = next(reader)
            p, r, t, product = (fields.index(k) for k in ("WRS Path", "WRS Row", "Date Acquired", "Display ID"))
            for entry in reader:
                count += 1
                if entry[t][:4] >= "2017" and entry[p].zfill(3) + entry[r].zfill(3) in TRACKS:
                    if len(entry) != len(fields):
                        raise ValueError(f"Invalid USGS catalog row {count + 1}")
                    regional.append(entry)
        if source.size != int(response.headers["Content-Length"]):
            raise ValueError("USGS bulk catalog size differs from the HTTP response")
        receipt = dict(source_url=BULK_URL, bytes=source.size, sha256=source.sha256.hexdigest(),
            http_etag=response.headers["ETag"],
            http_last_modified_utc=parsedate_to_datetime(response.headers["Last-Modified"]).isoformat(),
            world_rows=count, regional_rows=len(regional))
    path = STORE / "usgs_c2_regional.csv"
    regional.sort(key=lambda row: (row[t], row[p], row[r], row[product]))
    write_csv(path, fields, regional)
    receipt["regional_csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(STORE / "usgs_c2_regional_receipt.json", receipt)


def prepare_tool():
    archive = STORE / TOOL_NAME
    payload = get(TOOL_URL, archive)
    if hashlib.sha256(payload).hexdigest() != TOOL_SHA256:
        raise ValueError("The USGS ETM+ angle tool differs from the verified release")
    library = TOOL_WORK / "landsat_angles/ias_lib"
    if not (library / "Makefile").is_file():
        with tarfile.open(archive) as source:
            source.extractall(TOOL_WORK, filter="data")
    # The vendor code's malloc header and wait macros need these portable forms on macOS.
    for name, expression in (("xxx_GetDirFiles.c", "Command"), ("xxx_RecursiveDeletion.c", "DeleteCommand")):
        path = library / name
        original = path.read_text()
        patched = original.replace(f"if ((WEXITSTATUS(status = system({expression}))) != 0)",
                                   f"status = system({expression});\n    if (WEXITSTATUS(status) != 0)")
        if patched != original:
            path.write_text(patched)
    path = library / "xxx_LogError.c"
    original = path.read_text()
    if "#include <malloc.h>" in original:
        path.write_text(original.replace("#include <malloc.h>", "#include <stdlib.h>"))
    flags = ["-O2", "-D_DARWIN_C_SOURCE", "-DIAS_NO_SENSOR_META_SUPPORT", "-DIAS_NOVAS_NOT_AVAILABLE"]
    subprocess.run(["make", "-C", str(library), "OPTS=" + " ".join(flags)], check=True, capture_output=True)
    tool = TOOL_WORK / "landsat_etm_b8_angle_points"
    subprocess.run(["cc", *flags, "-I", str(library), str(ROOT / "scripts/landsat_etm_b8_angle_points.c"),
                    str(library / "liblandsatang.a"), "-lm", "-o", str(tool)], check=True, capture_output=True)
    return tool


def collection1():
    variants = defaultdict(list)
    for track in TRACKS:
        for prefix in gcs_index(f"LE07/01/{track[:3]}/{track[3:]}/", STORE / f"{track}.json", True):
            name = prefix.rstrip("/").rsplit("/", 1)[-1]
            match = PRODUCT.fullmatch(name)
            if match:
                kind, grid, date, processed, tier = match.groups()
                variants[grid, date].append((kind, TIERS[tier], processed, name, prefix))
    return variants


def catalog(legacy):
    candidates, _ = tank_points()
    transform = Transformer.from_crs(32614, 4326, always_xy=True).transform
    footprints = {key: Point(*transform(*xy)) for key, xy in candidates.items()}
    dated = {key: Point(*transform(float(r["center_east_m"]), float(r["center_north_m"]))) for key, r in legacy.items()}
    variants = collection1()
    jobs, seen, result = [], set(), []
    regional_path = STORE / "usgs_c2_regional.csv"
    receipt = json.loads((STORE / "usgs_c2_regional_receipt.json").read_bytes())
    expected_hash = receipt.get("regional_csv_sha256")
    if expected_hash and hashlib.sha256(regional_path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("The regional Landsat 7 catalog differs from its receipt")
    regional = rows(regional_path)
    if len(regional) != receipt["regional_rows"]:
        raise ValueError("The regional Landsat 7 catalog has a different row count from its receipt")
    for r in regional:
        date = r["Date Acquired"].replace("/", "-")
        track = f"{int(r['WRS Path']):03d}{int(r['WRS Row']):03d}"
        identity = r["Landsat Product Identifier L1"]
        if identity in seen or track not in TRACKS:
            raise ValueError(f"Unexpected regional inventory row: {identity}")
        seen.add(identity)
        footprint = Polygon([(float(r[f"Corner {c} Longitude"]), float(r[f"Corner {c} Latitude"])) for c in CORNERS])
        if not footprint.is_valid:
            raise ValueError(f"Invalid Collection 2 corners: {identity}")
        candidate_count = sum(footprint.covers(p) for p in footprints.values())
        roofs = [key for key, p in dated.items() if footprint.covers(p)]
        versions = sorted(variants[track, date.replace("-", "")])
        terrain_corrected = [v for v in versions if v[0] == "L1TP"]
        selected = min(terrain_corrected, default=None, key=lambda v: (v[1], -int(v[2]), v[3]))
        result.append(dict(collection2_product_id=identity, collection2_scene_id=r["Landsat Scene Identifier"],
            acquired_date=date, track=track, c2_start_utc=r["Start Time"], c2_stop_utc=r["Stop Time"],
            c2_cloud_cover_pct=r["Scene Cloud Cover L1"], c2_tier=r["Collection Category"],
            c2_day_night=r["Day/Night Indicator"], c2_nominal_candidate_centers=candidate_count,
            c2_nominal_dated_roof_centers=len(roofs), c1_archive_versions="|".join(v[3] for v in versions),
            selected_c1_l1tp_product_id=selected[3] if selected else ""))
        if selected and roofs and r["Day/Night Indicator"] == "DAY":
            jobs.append((selected[3], selected[4], result[-1], roofs))
    result.sort(key=lambda r: (r["acquired_date"], r["track"], r["collection2_product_id"]))
    if len({j[0] for j in jobs}) != len(jobs):
        raise ValueError("A Collection 1 acquisition matched multiple Collection 2 rows")
    jobs.sort(key=lambda j: (j[2]["c2_tier"] != "T1", -j[2]["c2_nominal_dated_roof_centers"],
                             float(j[2]["c2_cloud_cover_pct"]), j[0]))
    return result, jobs


def source_bytes(asset, path=None):
    payload = get(GCS_OBJECT + asset["name"], path)
    if len(payload) != int(asset["size"]) or crc32c_base64(payload) != asset["crc32c"]:
        raise ValueError(f"Archive CRC32C or size mismatch for {asset['name']}")
    return payload


def crop(raster, bounds):
    inside = from_bounds(*bounds, transform=raster.transform)
    c0 = max(0, math.floor(inside.col_off))
    r0 = max(0, math.floor(inside.row_off))
    c1 = min(raster.width, math.ceil(inside.col_off + inside.width))
    r1 = min(raster.height, math.ceil(inside.row_off + inside.height))
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f"No roof pixels inside {raster.name}")
    window = Window(c0, r0, c1 - c0, r1 - r0)
    return raster.read(1, window=window), raster.window_transform(window), window


def usable_b8(dn, gaps, grid, q, qa_grid):
    if (gaps > 1).any():
        raise ValueError("Unexpected SLC gap mask values")
    rr, cc = np.indices(dn.shape)
    xx, yy = grid * (cc + .5, rr + .5)
    valid_qa = (q & 3) == 0
    bad_qa = ((q & 28) != 0) | (((q >> 5) & 3) == 3) | (((q >> 7) & 3) == 3) | (((q >> 9) & 3) == 3)
    qa_view = SimpleNamespace(transform=qa_grid, width=q.shape[1], height=q.shape[0])
    _, qa_ok, qa_bad = pan_qa(qa_view, xx, yy, q, valid_qa, bad_qa.astype("uint16"))
    real = (gaps == 1) & (dn > 0)
    return xx, yy, real, real & (dn < 255) & qa_ok & (qa_bad == 0)


def check_grid(raster, mtl, resolution, dtype, count_type):
    crs = f"EPSG:{32600 + int(mtl['UTM_ZONE'])}"
    a = raster.transform
    x, y = a * (0.5, 0.5)
    expected = tuple(int(mtl[f"{count_type}_{name}"]) for name in ("LINES", "SAMPLES"))
    center = tuple(float(mtl[f"CORNER_UL_PROJECTION_{axis}_PRODUCT"]) for axis in "XY")
    if (raster.count != 1 or raster.crs is None or raster.crs.to_string() != crs or raster.shape != expected
        or raster.dtypes != (dtype,) or (a.a, a.b, a.d, a.e) != (resolution, 0, 0, -resolution) or (x, y) != center):
        raise ValueError(f"Raster grid differs from exact Collection 1 MTL: {raster.name}")


def local_angles(binary, coefficients, raster, pixels):
    coordinates = "".join(f"{line} {sample}\n" for line, sample in pixels)
    output = subprocess.run([binary, coefficients], input=coordinates, text=True, check=True, capture_output=True)
    records = list(csv.reader(io.StringIO(output.stdout)))
    if len(records) != 1 + len(pixels) or records[0][0] != "grid":
        raise ValueError(f"Unexpected ETM+ angle output for {coefficients}")
    header = tuple(map(float, records[0][1:]))
    if header != (*raster.shape, 15, *(raster.transform * (0.5, 0.5))):
        raise ValueError(f"ETM+ angle grid differs from B8: {coefficients}")
    output = []
    for a in records[1:]:
        outside_sun, outside_view, sun_z, sun_a, view_z, view_a = map(float, a)
        valid = (outside_sun == outside_view == 0 and 0 < sun_z < 90 and 0 <= view_z < 20
                 and abs(sun_a) <= 180 and abs(view_a) <= 180)
        output.append((valid, sun_z, sun_a % 360, view_z, view_a % 360))
    return output


def parse_mtl(payload):
    fields = re.findall(r"^[ \t]*([A-Z0-9_]+)[ \t]*=[ \t]*(.*?)[ \t]*$", payload.decode(), re.MULTILINE)
    return {key: value.strip().strip('"') for key, value in fields}


def stage(job, legacy, roof_evidence, binary):
    product, prefix, reference, roof_ids = job
    directory = PIXELS / product
    receipt = directory / "item.json"
    packet = directory / "native.npz"
    if receipt.is_file():
        item = json.loads(receipt.read_bytes())
        for relative, digest in item["local_sha256"].items():
            if hashlib.sha256((directory / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Saved Landsat 7 data failed readback: {product}/{relative}")
        return item

    objects = {Path(obj["name"]).name: obj for obj in gcs_index(prefix, directory / "objects.json")}
    assets = {key: objects[f"{product}_{suffix}"] for key, suffix in ASSETS.items()}
    mtl_file, ang_file = directory / "MTL.txt", directory / "ANG.txt"
    mtl_raw = source_bytes(assets["mtl"], mtl_file)
    ang_raw = source_bytes(assets["angles"], ang_file)
    mtl = parse_mtl(mtl_raw)
    if (mtl["LANDSAT_PRODUCT_ID"] != product or mtl["SPACECRAFT_ID"] != "LANDSAT_7" or mtl["SENSOR_ID"] != "ETM"
        or mtl["DATE_ACQUIRED"] != reference["acquired_date"] or float(mtl["SUN_ELEVATION"]) <= 0):
        raise ValueError(f"Wrong Landsat 7 MTL identity for {product}")
    captured = dt.datetime.fromisoformat(f"{mtl['DATE_ACQUIRED']}T{mtl['SCENE_CENTER_TIME']}")
    start, stop = (dt.datetime.fromisoformat(reference[f"c2_{key}_utc"]).replace(tzinfo=UTC) for key in ("start", "stop"))
    if not start - dt.timedelta(seconds=1) <= captured <= stop + dt.timedelta(seconds=1):
        raise ValueError(f"Collection 1 time is outside the Collection 2 interval: {product}")

    qa_raw = source_bytes(assets["quality"])
    gap_raw = source_bytes(assets["gap"])
    with (rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="2"),
          rasterio.open(GCS_OBJECT + assets["pan"]["name"]) as pan,
          MemoryFile(qa_raw) as qa_file, qa_file.open() as qa,
          MemoryFile(gzip.decompress(gap_raw)) as gap_file, gap_file.open() as gap):
        check_grid(pan, mtl, 15, "uint8", "PANCHROMATIC")
        check_grid(gap, mtl, 15, "uint8", "PANCHROMATIC")
        check_grid(qa, mtl, 30, "uint16", "REFLECTIVE")
        project = Transformer.from_crs(32614, pan.crs, always_xy=True).transform
        roofs = []
        for key in roof_ids:
            g = legacy[key]
            x, y = project(float(g["center_east_m"]), float(g["center_north_m"]))
            line, sample = pan.index(x, y)
            if 0 <= line < pan.height and 0 <= sample < pan.width:
                roofs.append((key, x, y, float(g["diameter_m"]) / 2, line, sample))
        if not roofs:
            raise ValueError(f"No dated roofs inside the Collection 1 raster: {product}")
        bounds = (min(x-r for _, x, _, r, _, _ in roofs)-MARGIN_M, min(y-r for _, _, y, r, _, _ in roofs)-MARGIN_M,
                  max(x+r for _, x, _, r, _, _ in roofs)+MARGIN_M, max(y+r for _, _, y, r, _, _ in roofs)+MARGIN_M)
        dn, grid, window = crop(pan, bounds)
        gaps = gap.read(1, window=window)
        q, qa_grid, _ = crop(qa, (bounds[0]-30, bounds[1]-30, bounds[2]+30, bounds[3]+30))
        angles = local_angles(binary, ang_file, pan, [(line, sample) for _, _, _, _, line, sample in roofs])
        xx, yy, real, usable = usable_b8(dn, gaps, grid, q, qa_grid)
        records = []
        for (key, x, y, radius, line, sample), angle in zip(roofs, angles, strict=True):
            distance = np.hypot(xx-x, yy-y)
            inside, around = distance <= radius, (distance > radius) & (distance <= radius + MARGIN_M)
            angle_ok, sun_z, sun_a, view_z, view_a = angle
            evidence = roof_evidence[key]
            records.append(dict(candidate_id=key, terminal_id=evidence["terminal_id"],
                roof_evidence_date=evidence["roof_evidence_date"], lidar_available_date=evidence["lidar_source_available_date"],
                center_east_m=x, center_north_m=y, radius_m=radius, b8_native_line=line, b8_native_sample=sample,
                roof_pixels=int(inside.sum()), roof_pixels_in_original=int((inside & real).sum()),
                roof_pixels_passing_qa=int((inside & usable).sum()), context_pixels=int(around.sum()),
                context_pixels_passing_qa=int((around & usable).sum()), angle_valid=angle_ok,
                sun_zenith_deg=sun_z, sun_azimuth_deg=sun_a, view_zenith_deg=view_z, view_azimuth_deg=view_a))
        with io.BytesIO() as out:
            np.savez_compressed(out, pan_dn=dn, b8_slc_gap_mask=gaps, pan_usable=usable, bqa=q,
                                pan_transform=tuple(grid)[:6], qa_transform=tuple(qa_grid)[:6], crs=pan.crs.to_string())
            packed = out.getvalue()

    write(packet, packed)
    local = {p.name: hashlib.sha256(raw).hexdigest() for p, raw in ((mtl_file, mtl_raw), (ang_file, ang_raw), (packet, packed))}
    item = dict(product_id=product, collection2_reference_id=reference["collection2_product_id"],
        captured_at_utc=captured.isoformat(), archive_assets_created_by_utc=max(a["timeCreated"] for a in assets.values()),
        crs=mtl["UTM_ZONE"], native_pan_shape=[int(mtl["PANCHROMATIC_LINES"]), int(mtl["PANCHROMATIC_SAMPLES"])],
        native_window=[int(window.row_off), int(window.col_off), int(window.height), int(window.width)],
        scene_sun_elevation_deg=float(mtl["SUN_ELEVATION"]), scene_sun_azimuth_deg=float(mtl["SUN_AZIMUTH"]),
        b8_reflectance_mult=float(mtl["REFLECTANCE_MULT_BAND_8"]), b8_reflectance_add=float(mtl["REFLECTANCE_ADD_BAND_8"]),
        b8_scene_saturation_flag=mtl["SATURATION_BAND_8"], native_packet=str(packet.relative_to(ROOT)),
        b8_verification="native_windows_by_https_range_full_object_crc32c_recorded_not_recomputed",
        local_sha256=local, source_objects=assets, roofs=records)
    save_json(receipt, item)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle-tool", type=Path, help="Use an already compiled USGS ETM+ angle wrapper.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Only stage the first N entries in deterministic priority order.")
    parser.add_argument("--catalog-only", action="store_true")
    parser.add_argument("--refresh-usgs-catalog", action="store_true", help="Verify the full gzip stream and save the three regional tracks.")
    args = parser.parse_args()
    if args.refresh_usgs_catalog:
        refresh_catalog()
    legacy = legacy_geometry()
    evidence = {r["candidate_id"]: r for r in rows(DERIVED / "sentinel2_space_time_coverage_roof_registry.csv")}
    if set(legacy) != set(evidence):
        raise ValueError("The dated geometry and roof-evidence inventories differ")
    inventory, jobs = catalog(legacy)
    export(f"{STEM}_archive_catalog", inventory)
    selected = [] if args.catalog_only else jobs[:args.limit]
    failures = {}
    tool = (args.angle_tool or prepare_tool()) if selected else None
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(stage, j, legacy, evidence, tool): j[0] for j in selected}
        for future in as_completed(futures):
            try:
                future.result()
            except (KeyError, OSError, ValueError, subprocess.CalledProcessError) as error:
                failures[futures[future]] = str(error)
                print(f"{futures[future]}: {error}", flush=True)
    scenes, roofs = [], []
    for product, _, reference, _ in jobs:
        receipt = PIXELS / product / "item.json"
        if not receipt.is_file() or product in failures:
            continue
        item = json.loads(receipt.read_bytes())
        records = item["roofs"]
        scenes.append({k: v for k, v in item.items() if k not in ("roofs", "source_objects", "local_sha256")}
                      | {"roof_windows": len(records), "roofs_with_some_usable_b8": sum(r["roof_pixels_passing_qa"] > 0 for r in records),
                         "receipt": str(receipt.relative_to(ROOT))})
        for r in records:
            roofs.append(dict(product_id=product, collection2_reference_id=reference["collection2_product_id"],
                captured_at_utc=item["captured_at_utc"], archive_assets_created_by_utc=item["archive_assets_created_by_utc"]) | r)
    if scenes:
        export(f"{STEM}_native_scenes", scenes)
        export(f"{STEM}_roof_windows", roofs)
    archive_lags = [(dt.datetime.fromisoformat(r["archive_assets_created_by_utc"])
                     - dt.datetime.fromisoformat(r["captured_at_utc"])).total_seconds() / 86400 for r in scenes]
    if archive_lags and min(archive_lags) < 0:
        raise ValueError("Archive object predates the Landsat 7 acquisition")
    usable = [r for r in roofs if r["roof_pixels_passing_qa"] > 0 and r["angle_valid"]]
    summary = dict(collection2_regional_products=len(inventory),
        collection2_nominally_intersects_any_candidate=sum(r["c2_nominal_candidate_centers"] > 0 for r in inventory),
        collection2_nominally_intersects_dated_roofs=sum(r["c2_nominal_dated_roof_centers"] > 0 for r in inventory),
        collection2_dated_roof_products_by_year=dict(sorted(Counter(r["acquired_date"][:4] for r in inventory if r["c2_nominal_dated_roof_centers"]).items())),
        public_l1tp_c1_acquisitions_eligible_for_staging=len(jobs), staged_native_c1_products=len(scenes),
        staged_native_c1_products_by_year=dict(sorted(Counter(r["captured_at_utc"][:4] for r in scenes).items())),
        staged_native_c1_products_by_tier=dict(sorted(Counter(r["product_id"].rsplit("_", 1)[-1] for r in scenes).items())),
        staged_roof_windows=len(roofs), windows_with_some_usable_roof_pixels=sum(r["roof_pixels_passing_qa"] > 0 for r in roofs),
        windows_with_usable_roof_pixels_and_valid_angles=len(usable),
        windows_all_roof_and_context_pixel_centers_usable=sum(r["roof_pixels_passing_qa"] == r["roof_pixels"]
            and r["context_pixels_passing_qa"] == r["context_pixels"] > 0 for r in usable),
        distinct_roofs_with_usable_pixels_and_angles=len({r["candidate_id"] for r in usable}),
        scenes_with_any_usable_roof_pixels_and_angles=len({r["product_id"] for r in usable}),
        archive_lag_days=(dict(minimum=min(archive_lags), median=median(archive_lags), maximum=max(archive_lags))
                          if archive_lags else None),
        errors=failures)
    save_json(DERIVED / f"{STEM}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
