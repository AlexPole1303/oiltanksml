"""Save unmasked, native Landsat red-band context for exterior registration."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit
from urllib.error import URLError
from urllib.request import Request, urlopen

import rasterio
from affine import Affine
from pyproj import Transformer

from build_landsat_c2_holdout_gap_inventory import SIDES, parse
from build_shadow_registration_inputs import tank_points
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_imagery import clip, sign
from collect_landsat import LANDSAT
from collect_landsat_c2_gap_geometry import native_b8_grid
from collect_public import ROOT, TLS, write
from collect_s2_l1c import OBJECTS
from stage_landsat_c2_b8_inputs import normalized, raster_grid, sha256_file

STEM = "landsat_c2_registration_context"
STORE = LANDSAT / STEM
CONTEXT_MARGIN_M = 1_300


def archive_headers(url, archive):
    request = Request(url, method="HEAD", headers={"If-Match": f'"{archive["etag"]}"'})
    with urlopen(request, timeout=25, context=TLS) as response:
        identity = response.headers["ETag"].strip('"'), int(response.headers["Content-Length"])
        if identity != (archive["etag"], archive["bytes"]):
            raise ValueError("Red-band archive object differs from the saved ETag and length")


def source_bounds(pairs):
    tanks, _ = tank_points()
    east, north = zip(*(tanks[r["candidate_id"]] for r in pairs), strict=True)
    bounds = (min(east)-CONTEXT_MARGIN_M, min(north)-CONTEXT_MARGIN_M,
              max(east)+CONTEXT_MARGIN_M, max(north)+CONTEXT_MARGIN_M)
    return Transformer.from_crs(32614, 4326, always_xy=True).transform_bounds(*bounds)


def collect_scene(product, stage, bounds):
    identity = product["product_id"]
    mtl = ROOT / product["level2_metadata_path"]
    item = json.loads(mtl.with_name("item.json").read_bytes())
    data = json.loads(mtl.read_bytes())["LANDSAT_METADATA_FILE"]
    if (data["LEVEL1_PROCESSING_RECORD"]["LANDSAT_PRODUCT_ID"] != identity
        or item["id"] != product["level2_item_id"]
        or parse(item["properties"]["datetime"]) != parse(product["captured_at_utc"])):
        raise ValueError(f"Red context has a different exact source for {identity}")
    full_id = data["PRODUCT_CONTENTS"]["LANDSAT_PRODUCT_ID"]
    clock = ROOT / stage["qa_clock_source"]
    archive = json.loads(clock.read_bytes())["assets"]["red"]
    if archive["url"] != item["assets"]["red"]["href"] or not archive["url"].endswith(f"/{full_id}_SR_B4.TIF"):
        raise ValueError(f"Red context URL differs from exact Level-2 MTL for {identity}")
    raster = STORE / identity / "red.tif"
    metadata = raster.with_name("provenance.json")
    record = dict(product_id=identity, level2_product_id=full_id, captured_at_utc=product["captured_at_utc"],
        archive_url=archive["url"], archive_etag=archive["etag"], archive_bytes=archive["bytes"],
        archive_modified_at_utc=archive["last_modified_at_utc"], archive_evidence=str(clock.relative_to(ROOT)),
        raster_path=str(raster.relative_to(ROOT)), qa_pixel_path=stage["qa_pixel_path"],
        qa_pixel_sha512=stage["qa_pixel_sha512"], qa_radsat_path=stage["qa_radsat_path"],
        qa_radsat_sha512=stage["qa_radsat_sha512"], crop_bounds_wgs84=list(bounds))
    reused = raster.is_file() and metadata.is_file()
    if reused:
        prior = json.loads(metadata.read_bytes())
        reused = all(prior.get(k) == v for k, v in record.items()) and prior.get("raster_sha256") == sha256_file(raster)
    if not reused:
        url = sign(archive["url"])
        archive_headers(url, archive)
        width, height = clip(url, raster, bounds, replace=True)
        archive_headers(url, archive)
        if not width or not height:
            raise ValueError(f"Red context does not cover the project bounds for {identity}")
    native = native_b8_grid(product)
    # Level-2 30 m cells share the upper-left center with the Level-1 B8 grid.
    red_grid = Affine(30, 0, native.c-7.5, 0, -30, native.f+7.5)
    with rasterio.open(raster) as source:
        row, col = raster_grid(source, product, 30, red_grid)
        if row < 0 or col < 0 or row+source.height > (product["native_lines"]+1)//2 or (
            col+source.width > (product["native_samples"]+1)//2
        ):
            raise ValueError(f"Red context is outside the native grid for {identity}")
        values = source.read(1, masked=True)
        if not ((values > 0).filled(False)).any():
            raise ValueError(f"Red context has no valid original pixels for {identity}")
        record.update(first_native_row=row, first_native_col=col, rows=source.height, cols=source.width,
                      valid_red_pixels=int((values > 0).filled(False).sum()), crs=source.crs.to_string(),
                      reflectance_scale=source.scales[0], reflectance_offset=source.offsets[0])
    record["raster_sha256"] = sha256_file(raster)
    record["verified_at_utc"] = prior["verified_at_utc"] if reused else datetime.now(timezone.utc).isoformat()
    if not reused:
        write(metadata, json.dumps(record, indent=2).encode())
    return record


def gcs_version(url, asset):
    with urlopen(Request(url, method="HEAD"), timeout=25, context=TLS) as response:
        hashes = {value.strip() for value in response.headers.get("x-goog-hash", "").split(",")}
        generation = response.headers.get("x-goog-generation", "")
        if (int(response.headers["Content-Length"]) != int(asset["size"])
            or f"crc32c={asset['crc32c']}" not in hashes or not generation.isdecimal()):
            raise ValueError("Sentinel-2 archive object differs from the saved length or CRC32C")
        return generation


def collect_sentinel(version, source, canonical, capture, bounds, *, store=STORE):
    tile = ROOT/source
    item = json.loads(tile.with_name("item.json").read_bytes())
    if ((item["product_name"], item["canonical_l2a_id"], parse(item["captured_at_utc"])) != (
            version, canonical, parse(capture)) or item["processing_level"] != "L1C"):
        raise ValueError(f"Red context does not identify the exact Sentinel-2 tile {version}")
    asset = item["assets"]["red"]
    url = asset["url"]
    if url != OBJECTS + quote(asset["name"], safe="/") or urlsplit(url).query:
        raise ValueError(f"Unexpected Sentinel-2 archive URL for {version}")
    raster = store / "sentinel2" / version / "red.tif"
    metadata = raster.with_name("provenance.json")
    record = dict(version_id=version, canonical_id=canonical, captured_at_utc=capture, tile_metadata=source,
        archive_url=url, archive_crc32c=asset["crc32c"], archive_bytes=int(asset["size"]),
        archive_created_at_utc=asset["timeCreated"], archive_modified_at_utc=asset["updated"],
        raster_path=str(raster.relative_to(ROOT)), crop_bounds_wgs84=list(bounds))
    reused = raster.is_file() and metadata.is_file()
    if reused:
        prior = json.loads(metadata.read_bytes())
        reused = all(prior.get(k) == v for k,v in record.items()) and prior.get("raster_sha256") == sha256_file(raster)
    if not reused:
        generation = gcs_version(url, asset)
        width,height = clip(url, raster, bounds, replace=True)
        if not width or not height or gcs_version(url, asset) != generation:
            raise ValueError(f"Sentinel-2 source did not retain its version or coverage for {version}")
    with rasterio.open(raster) as image:
        if (image.crs != "EPSG:32614" or image.res != (10,10) or image.dtypes != ("uint16",)
            or (image.transform.c % 10, image.transform.f % 10) != (0,0)):
            raise ValueError(f"Sentinel-2 source differs from the 10 m tile grid for {version}")
        count = int((image.read(1) > 0).sum())
        if not count:
            raise ValueError(f"Sentinel-2 archive context is empty for {version}")
        record.update(rows=image.height,cols=image.width,valid_red_pixels=count,crs=image.crs.to_string())
    record["raster_sha256"] = sha256_file(raster)
    record["archive_generation"] = prior["archive_generation"] if reused else generation
    record["verified_at_utc"] = prior["verified_at_utc"] if reused else datetime.now(timezone.utc).isoformat()
    if not reused:
        write(metadata,json.dumps(record,indent=2).encode())
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sentinel2", action="store_true", help="Collect native Sentinel-2 B04 exterior context")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    pairs = read_csv("landsat_c2_b8_weekly_pairs")
    bounds = source_bounds(pairs)
    if args.sentinel2:
        jobs = sorted({tuple(r[f"{side}_{k}"] for k in ("version_id","source_file","canonical_id","capture_at_utc"))
                       for r in pairs for side in SIDES if r[f"{side}_sensor"] == "B04"})
        run = lambda job: collect_sentinel(*job, bounds)
        stem = f"{STEM}_sentinel2"
    else:
        jobs = sorted({r[f"{side}_version_id"] for r in pairs for side in SIDES if r[f"{side}_sensor"] == "B8"})
        products = {r["product_id"]:normalized(r) for r in read_csv("landsat_c2_holdout_gap_geometry_products")}
        staged = {r["product_id"]:r for r in read_csv("landsat_c2_b8_stage_products")}
        run = lambda job: collect_scene(products[job], staged[job], bounds)
        stem = STEM
    records, failures = [], []

    def download(job):
        try:
            return run(job)
        except (OSError, ValueError, KeyError, URLError, rasterio.errors.RasterioError) as error:
            identity = job[0] if args.sentinel2 else job
            return identity,type(error).__name__

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(download, jobs):
            if isinstance(result, dict):
                records.append(result)
                identity = result.get("product_id",result.get("version_id"))
                print(f"Verified red context {len(records)}/{len(jobs)}: {identity}", flush=True)
            else:
                # A provider error can contain a signed URL. Report identity and error type only.
                failures.append(result[0])
                print(f"{result[0]}: {result[1]}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} red contexts are pending: {', '.join(sorted(failures))}")
    export(stem, records)
    summary = dict(scenes=len(records), valid_native_red_pixels=sum(r["valid_red_pixels"] for r in records),
                   local_raster_bytes=sum((ROOT/r["raster_path"]).stat().st_size for r in records),
                   scope=("Native Sentinel-2 B04 context, with original cloud masks held in the source archive packet."
                          if args.sentinel2 else "Exact Level-2 red context and linked QA. Original B8 is separate."))
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
