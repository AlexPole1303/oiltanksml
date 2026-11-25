"""Estimate independent exterior offsets for exact B8/B04 candidate scene pairs."""
import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds

from audit_shadow_registration_pixels import cloud_arrays
from build_landsat_c2_b8_measurements import verified_file
from build_landsat_c2_holdout_gap_inventory import SIDES, parse
from build_shadow_registration_inputs import CRS, anchor_grid, tank_points
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import ROOT, write
from collect_s2_l1c import crc32c_base64, source_metadata_bytes, source_metadata_path
from fit_shadow_registration import peak, registration_sampling, shifted_gradients, tank_offset
from stage_landsat_c2_b8_inputs import sha256_file

STEM = "landsat_c2_context_registration"
PIXEL_M = 30
PATCH_M = 300
PADDING_M = 60
SEARCH_RADIUS_M = 1_200
PATCH_PIXELS = (PATCH_M + 2*PADDING_M)//PIXEL_M
OFFSETS_M = np.arange(-24, 25, 4)
SHIFTS = np.stack(np.meshgrid(OFFSETS_M, OFFSETS_M), axis=-1).reshape(-1, 2)
ZERO = int(np.flatnonzero(np.all(SHIFTS == 0, axis=1))[0])
SAMPLING = registration_sampling(SHIFTS, patch_m=PATCH_M, padding_m=PADDING_M, pixel_m=PIXEL_M)
MAX_DISAGREEMENT_M = 8
PEAK_SEPARATION_M = 12


def sentinel_source(native):
    source = native["tile_metadata"]
    tile = ROOT/source
    version = native["version_id"]
    item = json.loads(tile.with_name("item.json").read_bytes())
    manifest = {r["name"].rsplit("/",1)[-1]:r for r in item["granule_metadata"]}
    mask_name = "MSK_CLASSI_B00.jp2" if "MSK_CLASSI_B00.jp2" in manifest else "MSK_CLOUDS_B00.gml"
    labels = "MTD_TL.xml", mask_name
    paths = [source_metadata_path(tile.with_name(k)) for k in labels]
    for path, label in zip(paths, labels, strict=True):
        asset = manifest[label]
        payload = source_metadata_bytes(path)
        if len(payload) != int(asset["size"]) or crc32c_base64(payload) != asset["crc32c"]:
            raise ValueError(f"Sentinel-2 tile metadata or mask differs from the archive for {version}")
    basis = "jp2_opaque_cirrus_snow_ice" if mask_name.endswith(".jp2") else "gml_opaque_cirrus"
    return dict(native, source_path=source, band="Sentinel2_B04", cloud_mask_basis=basis,
        cloud_mask=str(paths[1].relative_to(ROOT)), tile_metadata=str(paths[0].relative_to(ROOT)),
        context_available_at_utc=max(parse(native["archive_modified_at_utc"]),
            *(parse(manifest[k]["updated"]) for k in labels)).isoformat())


def source_inputs(rows):
    landsat = {r["product_id"]: r for r in read_csv("landsat_c2_registration_context")}
    sentinel = {(r["version_id"],r["tile_metadata"]):r for r in read_csv("landsat_c2_registration_context_sentinel2")}
    sources, pairs = {}, defaultdict(list)
    for r in rows:
        ids = []
        for side in SIDES:
            sensor, version, canonical, source, capture = (r[f"{side}_{k}"] for k in (
                "sensor", "version_id", "canonical_id", "source_file", "capture_at_utc"))
            key = sensor, version, source
            ids.append(key)
            if key in sources:
                if (sources[key]["canonical_id"], parse(sources[key]["captured_at_utc"])) != (canonical,parse(capture)):
                    raise ValueError(f"Conflicting source identity for {version}")
                continue
            if sensor == "B8":
                l = landsat[version]
                if canonical != version or parse(capture) != parse(l["captured_at_utc"]):
                    raise ValueError(f"Red context differs from the B8 scene for {version}")
                sources[key] = dict(l, source_path=source, band="Landsat_SR_B4", canonical_id=canonical)
            else:
                native = sentinel[version,source]
                if (native["canonical_id"], parse(native["captured_at_utc"])) != (canonical,parse(capture)):
                    raise ValueError(f"Red context differs from the Sentinel-2 source for {version}")
                sources[key] = sentinel_source(native)
        pairs[tuple(ids)].append(r)
    return sources, pairs


def analysis_grid(anchors):
    west = min(x-PADDING_M for x, _, _ in anchors)
    south = min(y-PADDING_M for _, y, _ in anchors)
    east = max(x+PATCH_M+PADDING_M for x, _, _ in anchors)
    north = max(y+PATCH_M+PADDING_M for _, y, _ in anchors)
    transform = from_origin(west, north, PIXEL_M, PIXEL_M)
    return transform, ((north-south)//PIXEL_M, (east-west)//PIXEL_M), (west,south,east,north)


def red_inputs(source, bounds):
    landsat = source["band"] == "Landsat_SR_B4"
    path = ROOT/source["raster_path"]
    if sha256_file(path) != source["raster_sha256"]:
        raise ValueError(f"Red context checksum mismatch for {path.parent.name}")
    with ExitStack() as stack:
        image = stack.enter_context(rasterio.open(path))
        size = PIXEL_M if landsat else 10
        if (image.crs != CRS or image.res != (size,size) or image.count != 1 or image.dtypes != ("uint16",)
            or image.transform.b or image.transform.d):
            raise ValueError(f"Unexpected original grid in {path}")
        a = from_bounds(*bounds, image.transform)
        start_x, start_y = max(0, math.floor(a.col_off)), max(0, math.floor(a.row_off))
        stop_x, stop_y = min(image.width, math.ceil(a.col_off+a.width)), min(image.height, math.ceil(a.row_off+a.height))
        if stop_x <= start_x or stop_y <= start_y:
            return None
        window = rasterio.windows.Window(start_x,start_y,stop_x-start_x,stop_y-start_y)
        raw = image.read(1, window=window)
        valid = (image.read_masks(1, window=window) > 0) & (raw > 0) & (raw < 65535)
        transform = image.window_transform(window)
        reference = SimpleNamespace(bounds=image.window_bounds(window), shape=raw.shape, crs=image.crs,
                                    transform=transform)
        if landsat:
            quality = []
            for label in ("qa_pixel", "qa_radsat"):
                path = verified_file(ROOT/source[f"{label}_path"], source[f"{label}_sha512"])
                mask = stack.enter_context(rasterio.open(path))
                u, v = (~mask.transform)*(transform.c, transform.f)
                if mask.crs != image.crs or mask.res != image.res or (u,v) != (round(u),round(v)):
                    raise ValueError(f"QA and red-band cells disagree for {source['product_id']}")
                patch = rasterio.windows.Window(round(u),round(v),raw.shape[1],raw.shape[0])
                data = mask.read(1, window=patch)
                if data.shape != raw.shape or mask.window_transform(patch) != transform:
                    raise ValueError(f"QA window does not cover red context for {source['product_id']}")
                valid &= mask.read_masks(1, window=patch) > 0
                quality.append(data)
            q, saturation = quality
            # Pixel QA bits 0–5. RADSAT bit 3 is B4 saturation, bit 11 terrain occlusion.
            valid &= ((q & 63) == 0) & ((saturation & ((1<<3)|(1<<11))) == 0)
        else:
            valid &= ~np.logical_or.reduce(cloud_arrays(source, reference))
        return raw, valid, transform


def analysis_pixels(source, grid):
    return resample_pixels(red_inputs(source,grid[2]),grid)


def resample_pixels(native, grid):
    transform, shape, bounds = grid
    signal, coverage = np.zeros(shape, dtype="float32"), np.zeros(shape, dtype="float32")
    if native is None:
        return signal, coverage.astype(bool)
    raw, valid, src_transform = native
    # Explicit zeros outside the native window ensure edge cells need complete coverage too.
    src_transform *= Affine.translation(-3, -3)
    for field, output in ((raw, signal), (valid, coverage)):
        padded = np.pad(field.astype("float32"), 3)
        reproject(padded, output, src_transform=src_transform, src_crs=CRS,
                  dst_transform=transform, dst_crs=CRS, resampling=Resampling.average)
    return signal, coverage >= 1-1e-6


def estimate(rows, *, sources_and_pairs=None, load_pixels=analysis_pixels, context_label="red", study_phase="holdout",
             partial_patch_scores=None):
    sources, pairs = source_inputs(rows) if sources_and_pairs is None else sources_and_pairs
    centers, exclusion = tank_points()
    wanted = {r["candidate_id"] for r in rows}
    anchors = anchor_grid({k: centers[k] for k in wanted}, exclusion, search_radius_m=SEARCH_RADIUS_M,
                          patch_m=PATCH_M, padding_m=PADDING_M)
    grid = analysis_grid({patch for patches in anchors.values() for patch in patches})
    inverse = ~grid[0]
    rasters = {}
    for i, (key, source) in enumerate(sorted(sources.items()), 1):
        rasters[key] = load_pixels(source, grid)
        if i % 30 == 0 or i == len(sources):
            print(f"Checked {context_label} pixels and masks for {i}/{len(sources)} sources", flush=True)

    @lru_cache(maxsize=480)
    def gradients(key, west, south):
        signal, valid = rasters[key]
        x, y = inverse*(west-PADDING_M,south+PATCH_M+PADDING_M)
        x, y = round(x), round(y)
        cells = np.s_[y:y+PATCH_PIXELS,x:x+PATCH_PIXELS]
        if valid[cells].shape != (PATCH_PIXELS,PATCH_PIXELS):
            return None
        values = signal[cells]
        if partial_patch_scores is not None:
            return values, valid[cells]
        if not valid[cells].all():
            return None
        dy, dx = np.gradient(values)
        if np.linalg.matrix_rank(np.column_stack((dx.ravel(),dy.ravel()))) < 2:
            return None
        return shifted_gradients(values, SAMPLING)

    results, controls, scene_pairs = [], [], []
    for number, (key, group) in enumerate(sorted(pairs.items()), 1):
        scene_pairs.append(dict(scene_pair_id=number,route=group[0]["route"],earlier_version_id=key[0][1],
            later_version_id=key[1][1],earlier_source=key[0][2],later_source=key[1][2],candidate_pairs=len(group)))
        profiles = {}
        for west,south,role in sorted({p for r in group for p in anchors[r["candidate_id"]]}):
            a, b = (gradients(k,west,south) for k in key)
            if a is None or b is None:
                continue
            scores = (np.clip(np.einsum("ij,ij->i",a[::-1],b),-1,1) if partial_patch_scores is None
                      else partial_patch_scores(a,b,SAMPLING))
            if scores is None:
                continue
            best, separation, status = peak(scores, shifts=SHIFTS, separation_distance_m=PEAK_SEPARATION_M)
            name = f"{west}_{south}"
            profiles[number,name] = scores,best,status
            controls.append(dict(scene_pair_id=number, anchor_id=name, west_m=west, south_m=south,
                role=("fit","spatial_check")[role], patch_status=status,
                east_m=int(SHIFTS[best,0]),north_m=int(SHIFTS[best,1]),
                peak_correlation=float(scores[best]),zero_correlation=float(scores[ZERO]),peak_separation=separation))
        for r in group:
            tank = r["candidate_id"]
            ids = [[],[]]
            for west,south,role in anchors[tank]:
                name = f"{west}_{south}"
                if (number,name) in profiles:
                    ids[role].append(name)
            job = dict(comparison=r["route"],study_phase=study_phase,week_or_control_day=r["week_ending"],
                       candidate_id=tank,scene_pair_id=number,fit_screened_patch_ids="|".join(ids[0]),
                       spatial_check_screened_patch_ids="|".join(ids[1]))
            offset = tank_offset(job,profiles,shifts=SHIFTS,zero=ZERO,disagreement_m=MAX_DISAGREEMENT_M)
            cutoff = parse(r["forecast_cutoff_utc"]) if r["forecast_cutoff_utc"] else None
            availability = max(parse(sources[k].get("context_available_at_utc", sources[k]["archive_modified_at_utc"]))
                               for k in key)
            results.append(dict(offset, release_date=r["release_date"],terminal_id=r["terminal_id"],
                earlier_version_id=key[0][1],later_version_id=key[1][1],earlier_source=key[0][2],later_source=key[1][2],
                earlier_red_raster=sources[key[0]]["raster_path"],later_red_raster=sources[key[1]]["raster_path"],
                red_context_available_at_utc=availability.isoformat(),red_context_before_cutoff=availability < cutoff if cutoff else None,
                fit_screened_patches=len(ids[0]),spatial_check_screened_patches=len(ids[1])))
        if number % 30 == 0 or number == len(pairs):
            print(f"Matched exterior {context_label} pixels for {number}/{len(pairs)} scene pairs", flush=True)
    return sources, scene_pairs, results, controls


def main():
    rows = read_csv("landsat_c2_b8_weekly_pairs")
    sources,pairs,results,patches = estimate(rows)
    checked = [r for r in results if r["registration_status"] == "checked_offset"]
    timed = [r for r in checked if r["red_context_before_cutoff"]]
    if len(results) != len(rows):
        raise ValueError("Exterior checks are missing exact weekly candidate pairs")
    export(f"{STEM}_scenes",pairs)
    export(f"{STEM}_pairs",results)
    export(f"{STEM}_patches",patches)
    by_week = defaultdict(list)
    for r in results:
        by_week[r["week_or_control_day"]].append(r)
    weeks = []
    for r in read_csv("landsat_c2_b8_weekly_weeks"):
        day = r["week_ending"]
        group = by_week[day]
        if (len(group) != int(r["exact_pair_candidates"]) or len({p["candidate_id"] for p in group}) != int(r["candidate_tanks"])
            or any(p["release_date"] != r["release_date"] for p in group)):
            raise ValueError(f"Exterior context differs from the exact candidate ledger on {day}")
        passed = [p for p in group if p["registration_status"] == "checked_offset" and p["red_context_before_cutoff"]]
        weeks.append(dict(week_ending=day, release_date=r["release_date"], forecast_cutoff_utc=r["forecast_cutoff_utc"],
            exact_pair_candidates=len(group), checked_context_pairs=len(passed),
            checked_context_candidate_tanks=len({p["candidate_id"] for p in passed})))
    export(f"{STEM}_weeks",weeks)
    stats = dict(red_sources=len(sources),scene_pairs=len(pairs),candidate_pairs=len(results),
        registration_patch_matches=len(patches),status=dict(Counter(r["registration_status"] for r in results)),
        candidate_pairs_with_two_screened_patches_per_group=sum(min(r["fit_screened_patches"],
            r["spatial_check_screened_patches"]) >= 2 for r in results),
        checked_pairs_by_route=dict(Counter(r["comparison"] for r in checked)),
        checked_pairs_with_context_before_cutoff=len(timed),
        checked_tank_weeks=len({(r["week_or_control_day"],r["candidate_id"]) for r in timed}),
        checked_weeks=len({r["week_or_control_day"] for r in timed}),
        median_checked_offset_m=float(np.median([math.hypot(r["later_minus_earlier_east_m"],
            r["later_minus_earlier_north_m"]) for r in checked])) if checked else None,
        median_check_patch_disagreement_m=float(np.median([r["check_median_residual_m"] for r in checked])) if checked else None,
        step_m=4, common_grid_m=PIXEL_M, patch_m=PATCH_M, padding_m=PADDING_M, search_radius_m=SEARCH_RADIUS_M,
        scope="Relative red-context offsets only. Checked means separate exterior patches passed on the common grid. "
              "No B8 roof measurement, B8-specific registration accuracy or historical B8 availability is established.")
    payload = json.dumps(stats,indent=2) + "\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
