"""Compare exterior Landsat 7 and Sentinel-2 features on the development scenes."""

import gzip
import hashlib
import io
import json
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from affine import Affine
from rasterio.io import MemoryFile
from pyproj import Transformer

from audit_landsat_c2_context_registration import analysis_pixels, estimate, resample_pixels, sentinel_source
from build_shadow_registration_inputs import CRS
from collect_cushing_landsat7 import DERIVED, LANDSAT, PIXELS, ROOT, STEM, check_grid, crop, export, parse_mtl, rows, save_json, source_bytes, usable_b8
from collect_landsat import GCS_OBJECT
from collect_landsat_c2_registration_context import collect_sentinel, gcs_version, source_bounds
from collect_public import write

STORE = LANDSAT / "landsat7_registration"


def pan_context(product, bounds, *, store=STORE):
    source = PIXELS / product
    item = json.loads((source / "item.json").read_bytes())
    if item["product_id"] != product:
        raise ValueError(f"Archive receipt names a different product for {product}")
    directory = store / product
    data, receipt = directory / "pan.npz", directory / "item.json"
    assets = item["source_objects"]
    url = GCS_OBJECT + assets["pan"]["name"]
    record = dict(product_id=product, bounds_wgs84=list(bounds), archive_url=url,
                  archive_crc32c=assets["pan"]["crc32c"], archive_bytes=int(assets["pan"]["size"]),
                  archive_modified_at_utc=item["archive_assets_created_by_utc"],
                  raster_path=str(data.relative_to(ROOT)))
    if receipt.is_file() and data.is_file():
        prior = json.loads(receipt.read_bytes())
        if (all(prior.get(k) == v for k, v in record.items())
            and prior["raster_sha256"] == hashlib.sha256(data.read_bytes()).hexdigest()):
            return prior
    mtl = parse_mtl(source_bytes(assets["mtl"], source / "MTL.txt"))
    qa_raw = source_bytes(assets["quality"])
    gap_raw = source_bytes(assets["gap"])
    generation = gcs_version(url, assets["pan"])
    with (rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="2"),
          rasterio.open(url) as pan, MemoryFile(qa_raw) as qa_file, qa_file.open() as qa,
          MemoryFile(gzip.decompress(gap_raw)) as gap_file, gap_file.open() as gap):
        check_grid(pan, mtl, 15, "uint8", "PANCHROMATIC")
        check_grid(gap, mtl, 15, "uint8", "PANCHROMATIC")
        check_grid(qa, mtl, 30, "uint16", "REFLECTIVE")
        if pan.crs != CRS:
            raise ValueError(f"Registration context does not share the project CRS for {product}")
        native_bounds = Transformer.from_crs(4326, pan.crs, always_xy=True).transform_bounds(*bounds)
        dn, grid, window = crop(pan, native_bounds)
        gaps = gap.read(1, window=window)
        west, south, east, north = native_bounds
        q, qa_grid, _ = crop(qa, (west-30, south-30, east+30, north+30))
        *_, usable = usable_b8(dn, gaps, grid, q, qa_grid)
        with io.BytesIO() as out:
            np.savez_compressed(out, pan_dn=dn, pan_usable=usable, b8_slc_gap_mask=gaps, bqa=q,
                                pan_transform=tuple(grid)[:6], qa_transform=tuple(qa_grid)[:6], crs=pan.crs.to_string())
            payload = out.getvalue()
    if generation != gcs_version(url, assets["pan"]):
        raise ValueError(f"Archive generation changed during the Landsat 7 read: {product}")
    old_path = ROOT / item["native_packet"]
    if hashlib.sha256(old_path.read_bytes()).hexdigest() != item["local_sha256"]["native.npz"]:
        raise ValueError(f"Roof pixel packet checksum differs for {product}")
    with np.load(old_path) as roof:
        transform = Affine(*roof["pan_transform"])
        col, row = (~grid) * (transform.c, transform.f)
        if max(abs(col-round(col)), abs(row-round(row))) > 1e-8:
            raise ValueError(f"Context and roof rasters have different native grids for {product}")
        col, row = round(col), round(row)
        old = roof["pan_dn"]
        r0, r1 = max(0, row), min(dn.shape[0], row+old.shape[0])
        c0, c1 = max(0, col), min(dn.shape[1], col+old.shape[1])
        if r1 <= r0 or c1 <= c0 or not np.array_equal(dn[r0:r1, c0:c1], old[r0-row:r1-row, c0-col:c1-col]):
            raise ValueError(f"Overlapping native B8 samples differ for {product}")
    write(data, payload)
    record.update(archive_generation=generation, raster_sha256=hashlib.sha256(payload).hexdigest(),
                  native_first_line=int(window.row_off), native_first_sample=int(window.col_off),
                  original_pixels_checked_against_roof_crop=(r1-r0)*(c1-c0))
    save_json(receipt, record)
    return record


def analysis_raster(source, grid):
    if source["band"] not in ("Landsat7_B8", "Landsat8_B8"):
        return analysis_pixels(source, grid)
    path = ROOT / source["raster_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["raster_sha256"]:
        raise ValueError(f"Native B8 context checksum mismatch: {path}")
    with np.load(path) as stored:
        return resample_pixels((stored["pan_dn"], stored["pan_usable"], Affine(*stored["pan_transform"])), grid)


def masked_gradients(image, sampling):
    values, valid = image
    dy, dx = np.gradient(values)
    grad = np.stack((dx, dy), axis=-1)
    # Central differences need the source cell and all four immediate neighbors.
    support = np.zeros_like(valid)
    support[1:-1, 1:-1] = (valid[1:-1, 1:-1] & valid[:-2, 1:-1] & valid[2:, 1:-1]
                            & valid[1:-1, :-2] & valid[1:-1, 2:])
    x, y, wx, wy = sampling
    sw, se = (1-wx)*(1-wy), wx*(1-wy)
    nw, ne = (1-wx)*wy, wx*wy
    weights = np.stack((sw, se, nw, ne), axis=0)
    locations = ((y, x), (y, x+1), (y+1, x), (y+1, x+1))
    pixels = np.stack([grad[rr, cc] for rr, cc in locations])
    neighbors = np.stack([support[rr, cc] for rr, cc in locations])
    gradients = (pixels*weights).sum(axis=0).reshape(len(x), -1, 2)
    valid = (neighbors | (weights[..., 0] == 0)).all(axis=0).reshape(len(x), -1)
    return gradients, valid


def masked_scores(first, second, sampling):
    a, a_valid = masked_gradients(first, sampling)
    b, b_valid = masked_gradients(second, sampling)
    a, a_valid = a[::-1], a_valid[::-1]
    shared = a_valid & b_valid
    counts = shared.sum(axis=1)
    # Every tested displacement is evaluated with at least half of the same-sized patch.
    enough = counts >= shared.shape[1]/2
    if not enough.any():
        return None
    for gradient in (a, b):
        valid = np.where(shared[..., None], gradient, 0.)
        center = valid.sum(axis=1) / np.maximum(1, counts)[:, None]
        gradient[:] = np.where(shared[..., None], gradient - center[:, None, :], 0.)
        # A single strong edge cannot locate the patch along both axes.
        sv = np.linalg.svd(gradient, compute_uv=False)
        enough &= (sv[:, 1] > .05*sv[:, 0])
    if not enough.any():
        return None
    norm = np.linalg.norm(a, axis=(1, 2)) * np.linalg.norm(b, axis=(1, 2))
    correlation = np.full(len(norm), -1.)
    np.divide(np.einsum("ijk,ijk->i", a, b), norm, out=correlation, where=enough)
    # Unavailable alternative displacements cannot be used as evidence of a unique peak.
    if (correlation == -1).mean() > .10:
        return None
    return np.clip(correlation, -1, 1)


def register(controls, *, stem, store=STORE, masked=False, dated=False):
    bounds = source_bounds(controls)
    products = sorted({r["landsat_product_id"] for r in controls})
    sentinels = sorted({tuple(r[k] for k in ("sentinel_version_id", "sentinel_tile_metadata",
                                           "sentinel_canonical_id", "sentinel_captured_at_utc")) for r in controls})
    with ThreadPoolExecutor(max_workers=5) as pool:
        pan_futures = [pool.submit(pan_context, product, bounds, store=store) for product in products]
        sentinel_futures = [pool.submit(collect_sentinel, *job, bounds, store=store) for job in sentinels]
        pan = {r["product_id"]: r for r in (f.result() for f in pan_futures)}
        sentinel = {r["version_id"]: r for r in (f.result() for f in sentinel_futures)}
    if len(sentinel) != len(sentinels):
        raise ValueError("The Sentinel-2 version has conflicting source identities")
    sources = {("ETM", k, v["raster_path"]): dict(v, band="Landsat7_B8") for k, v in pan.items()}
    sources.update({("MSI", k, v["raster_path"]): sentinel_source(v) for k, v in sentinel.items()})
    pairs = defaultdict(list)
    for r in controls:
        b8, b04 = r["landsat_product_id"], r["sentinel_version_id"]
        key = (("ETM", b8, pan[b8]["raster_path"]), ("MSI", b04, sentinel[b04]["raster_path"]))
        pairs[key].append(dict(candidate_id=r["candidate_id"], terminal_id=r["terminal_id"], route="landsat7_b8_to_b04",
            week_ending=r["week_ending"] if dated else r["landsat_captured_at_utc"][:10],
            release_date=r["release_date"] if dated else "", forecast_cutoff_utc=r["forecast_cutoff_utc"] if dated else None))
    _, scenes, offsets, patches = estimate(controls, sources_and_pairs=(sources, pairs),
        load_pixels=analysis_raster, context_label="Landsat 7 B8 and Sentinel-2 B04", study_phase="engineering",
        partial_patch_scores=masked_scores if masked else None)
    for name, data in (("pan_sources", list(pan.values())), ("sentinel_sources", list(sentinel.values())),
                       ("pairs", scenes), ("offsets", offsets), ("patches", patches)):
        if data:
            export(f"{stem}_{name}", data)
    accepted = [r for r in offsets if r["registration_status"] == "checked_offset"]
    def quantiles(values):
        return dict(median=float(np.median(values)), p90=float(np.quantile(values, .9))) if len(values) else None
    report = dict(scene_pairs=len(scenes), offsets=len(offsets), exterior_patch_profiles=len(patches),
        offsets_by_status=dict(Counter(r["registration_status"] for r in offsets)),
        checked_and_context_available_before_cutoff=sum(r["red_context_before_cutoff"] is True for r in accepted),
        checked_scene_pairs=len({r["scene_pair_id"] for r in accepted}),
        checked_roof_offsets_m=quantiles(np.hypot([r["later_minus_earlier_east_m"] for r in accepted],
                                                 [r["later_minus_earlier_north_m"] for r in accepted])),
        median_control_patch_residual_m=quantiles([r["check_median_residual_m"] for r in accepted]),
        offset_sign="Sentinel-2 minus Landsat 7 in EPSG:32614; source order denotes sensors, not time")
    save_json(DERIVED / f"{stem}_summary.json", report)
    print(json.dumps(report, indent=2))


def register_mixed():
    stem = f"{STEM}_mixed"
    controls = [r for r in rows(DERIVED / f"{stem}_candidate_pairs.csv")
                if r["landsat_native_height_diagnostic_pass"] == "True"]
    register(controls, stem=f"{stem}_registration", store=LANDSAT / "landsat7_mixed_registration", masked=True, dated=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masked", action="store_true", help="Allow partially covered control patches with a two-axis texture check.")
    parser.add_argument("--mixed", action="store_true", help="Register dated RT / B04 reporting-window candidates.")
    args = parser.parse_args()
    if args.mixed:
        register_mixed()
        return
    controls = rows(DERIVED / f"{STEM}_sentinel2_controls.csv")
    register(controls, stem=f"{STEM}_registration" + ("_masked" if args.masked else ""), masked=args.masked)


if __name__ == "__main__":
    main()
