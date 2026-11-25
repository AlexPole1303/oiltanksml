"""Archive hourly NOAA HRRR fields around the Cushing registry from a pinned snapshot."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import icechunk
import numpy as np
import zarr
from pyproj import CRS, Transformer

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write

STEM = "cushing_hrrr"
STORE = RAW / STEM
COLLECTION = "https://stac.dynamical.org/noaa-hrrr-analysis/collection.json"
HOST = "https://dynamical-noaa-hrrr.s3.us-west-2.amazonaws.com/noaa-hrrr-analysis/"
FIRST_HOUR = "2016-12-22T00:00:00+00:00"
LAST_HOUR = "2026-11-05T00:00:00+00:00"
SCENE_FIELDS = (
    "total_cloud_cover_atmosphere",
    "geopotential_height_cloud_ceiling",
    "downward_short_wave_radiation_flux_surface",
    "visible_beam_downward_solar_flux_surface",
    "precipitation_surface",
    "composite_reflectivity",
    "snow_area_fraction_surface",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_gust_surface",
)


def time(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError(f"Missing timezone for {value}")
    return int(stamp.timestamp())


def utc(seconds):
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).isoformat()


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def grid(group):
    tanks = read("tank_engineering_registry")
    coordinates = np.array([[float(r["longitude"]), float(r["latitude"])] for r in tanks])
    crs = CRS.from_wkt(group["spatial_ref"].attrs["crs_wkt"])
    forward = Transformer.from_crs(4326, crs, always_xy=True)
    inverse = Transformer.from_crs(crs, 4326, always_xy=True)
    xp, yp = forward.transform(*coordinates.T)
    x, y = group["x"][:], group["y"][:]
    if not (np.allclose(np.diff(x), 3000) and np.allclose(np.diff(y), -3000)):
        raise ValueError("HRRR projection axes differ from the documented 3 km grid")
    ix = np.abs(xp[:, None]-x).argmin(axis=1)
    iy = np.abs(yp[:, None]-y).argmin(axis=1)
    x_right = np.searchsorted(x, xp)
    y_top = len(y)-np.searchsorted(y[::-1], yp)-1
    distance = np.hypot(x[ix]-xp, y[iy]-yp)
    if np.max(distance) > 2122:
        raise ValueError("HRRR grid does not contain a candidate tank")
    neighbors = [((b,a-1), (b,a), (b+1,a-1), (b+1,a)) for a,b in zip(x_right, y_top, strict=True)]
    locations = sorted({k for block in neighbors for k in block})
    indices = {k: i for i, k in enumerate(locations)}
    gy, gx = np.array(locations).T
    col = slice(int(gx.min()), int(gx.max())+1)
    row = slice(int(gy.min()), int(gy.max())+1)
    xx, yy = x[gx], y[gy]
    lon, lat = inverse.transform(xx, yy)
    if not (np.allclose(lon, group["longitude"].vindex[gy, gx], atol=0.0001, rtol=0)
            and np.allclose(lat, group["latitude"].vindex[gy, gx], atol=0.0001, rtol=0)):
        raise ValueError("HRRR projected grid differs from its longitude and latitude arrays")
    mapping = [dict(candidate_id=r["candidate_id"], tank_longitude=coord[0], tank_latitude=coord[1],
        hrrr_cell_index=indices[b,a], source_grid_col=int(a), source_grid_row=int(b),
        grid_longitude=lon[indices[b,a]], grid_latitude=lat[indices[b,a]],
        distance_to_nearest_cell_m=float(d), neighboring_cell_indices_json=json.dumps([indices[k] for k in adjacent]))
        for r, coord, a, b, d, adjacent in zip(tanks, coordinates, ix, iy, distance, neighbors, strict=True)]
    return row, col, gx, gy, xx, yy, lon, lat, mapping, crs.to_wkt()


def variable(group, name, span, row, col, cells, output, hours):
    array = group[name]
    if array.ndim != 3 or array.shape != (group["time"].shape[0], group["y"].shape[0], group["x"].shape[0]):
        raise ValueError(f"Unexpected HRRR dimensions for {name}")
    path = output / f"{name}.npy"
    temporary = path.with_suffix(".part")
    shape = (len(hours), len(cells[0]))
    # Write in model-year blocks to bound memory. No value is resampled or interpolated.
    payload = np.lib.format.open_memmap(temporary, mode="w+", dtype=array.dtype, shape=shape)
    valid_count = 0
    try:
        for offset in range(0, len(hours), 24*365):
            count = min(24*365, len(hours)-offset)
            values = array[slice(span.start+offset, span.start+offset+count), row, col][:, cells[0], cells[1]]
            if values.shape != (count, shape[1]) or np.isinf(values).any():
                raise ValueError(f"Invalid HRRR field {name} at {utc(hours[offset])}")
            payload[offset:offset+count] = values
            valid_count += int(np.isfinite(values).sum())
        payload.flush()
        del payload
        saved = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if saved.shape != shape or saved.dtype != array.dtype or valid_count == 0:
            raise ValueError(f"Local HRRR array failed verification for {name}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    attributes = dict(array.attrs)
    print(f"HRRR {name}: {valid_count}/{int(np.prod(shape))} finite values", flush=True)
    return dict(variable=name, source_units=attributes["units"], long_name=attributes["long_name"],
        step_type=attributes["step_type"], source_dtype=str(array.dtype), local_file=str(path.relative_to(ROOT)),
        bytes=path.stat().st_size, sha256=file_hash(path), finite_values=valid_count, total_values=int(np.prod(shape)),
        variable_attributes_json=json.dumps(attributes))


def acquisition_clocks():
    sources = (
        ("satellite_weather_clock", "collection", "item_id", "captured_at_utc"),
        ("ceda_s1_slc_regional_assets", "sentinel-1-slc", "product_id", "capture_start_utc"),
        ("cushing_s1_rtc_scenes", "sentinel-1-rtc", "rtc_id", "capture_start_utc"),
        ("landsat_c2_holdout_gap_geometry_products", "landsat-c2-l1", "product_id", "captured_at_utc"),
        ("umbra_cushing_catalog", "umbra-sicd", "product_id", "capture_start_utc"),
    )
    seen = {}
    for table, collection, key, when in sources:
        for row in read(table):
            identity = (row.get(collection, collection), row[key])
            stamp = row[when]
            if seen.setdefault(identity, stamp) != stamp:
                raise ValueError(f"Conflicting acquisition timestamps for {identity}")
    return seen


def scene_weather(clocks, hours, fields, grid_map):
    # Use a fixed Cushing reference point for the regional summary. Per-tank cells
    # remain available through the grid map and native hourly arrays.
    reference = min(grid_map, key=lambda r: (r["grid_latitude"]-35.96)**2+
                    ((r["grid_longitude"]+96.74)*0.81)**2)
    cell = reference["hrrr_cell_index"]
    files = {r["variable"]: np.load(ROOT/r["local_file"], mmap_mode="r", allow_pickle=False)
             for r in fields if r["variable"] in SCENE_FIELDS}
    out = []
    for (collection, item_id), stamp in sorted(clocks.items()):
        epoch = time(stamp)
        index = (epoch-int(hours[0]))//3600
        in_archive = 0 <= index < len(hours)
        model_hour = int(hours[index]) if in_archive else None
        values = {name: float(array[index, cell]) if in_archive else float("nan") for name, array in files.items()}
        out.append(dict(collection=collection, item_id=item_id, capture_utc=stamp,
            model_hour_at_or_before_capture_utc=utc(model_hour) if in_archive else "",
            capture_minus_model_hour_seconds=epoch-model_hour if in_archive else None,
            array_time_index=index if in_archive else None, hrrr_cell_index=cell,
            model_hour_in_archive=in_archive, **{k: v if np.isfinite(v) else None for k, v in values.items()}))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id")
    args = parser.parse_args()
    raw = get(COLLECTION, STORE / "collection.json", refresh=True)
    stac = json.loads(raw)
    asset = stac["assets"]["icechunk-https"]["href"]
    if stac["id"] != "noaa-hrrr-analysis" or not asset.startswith(HOST):
        raise ValueError("Unexpected NOAA HRRR source")
    repo = icechunk.Repository.open(icechunk.http_storage(asset))
    session = repo.readonly_session(snapshot_id=args.snapshot_id) if args.snapshot_id else repo.readonly_session("main")
    group = zarr.open_group(session.store, mode="r", use_consolidated=False)
    epoch = group["time"][:]
    if group["time"].attrs["units"] != "seconds since 1970-01-01" or np.any(np.diff(epoch) != 3600):
        raise ValueError("HRRR timestamps are not a continuous UTC hourly grid")
    lo, hi = map(time, (FIRST_HOUR, LAST_HOUR))
    a, b = map(int, np.searchsorted(epoch, [lo, hi]))
    hours = epoch[a:b]
    if hours[0] != lo or hours[-1] != hi-3600 or len(hours) != (hi-lo)//3600:
        raise ValueError("The selected HRRR snapshot does not cover the research period")
    row, col, gx, gy, xx, yy, lon, lat, mapping, wkt = grid(group)
    output = STORE / str(session.snapshot_id)
    output.mkdir(parents=True, exist_ok=True)
    coord = output / "coordinates.npz"
    np.savez_compressed(coord, time_epoch_utc=hours, x_m=xx, y_m=yy, longitude=lon, latitude=lat,
                        source_grid_row=gy, source_grid_col=gx)
    coordinate_hash = file_hash(coord)
    variables = sorted(set(group.array_keys())-{"time", "x", "y", "longitude", "latitude", "spatial_ref"})
    if not set(SCENE_FIELDS) <= set(variables):
        raise ValueError("HRRR snapshot is missing required fields")
    print(f"HRRR snapshot {session.snapshot_id}, {len(hours)} hours, {len(variables)} variables, "
          f"{len(xx)} native cells around the registry", flush=True)
    fields = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(variable, group, name, slice(a, b), row, col, (gy-row.start, gx-col.start), output, hours)
                for name in variables]
        for future in as_completed(jobs):
            fields.append(future.result())
    fields.sort(key=lambda r: r["variable"])
    links = scene_weather(acquisition_clocks(), hours, fields, mapping)
    for label, rows in (("fields", fields), ("tank_grid", mapping), ("scene_weather", links)):
        export(f"{STEM}_{label}", rows)
    summary = dict(observed_utc=datetime.now(timezone.utc).isoformat(), source_collection=COLLECTION,
        source_store=asset, source_collection_sha256=hashlib.sha256(raw).hexdigest(), snapshot_id=str(session.snapshot_id),
        coordinates_file=str(coord.relative_to(ROOT)), coordinates_sha256=coordinate_hash,
        first_hour_utc=utc(hours[0]), last_hour_utc=utc(hours[-1]), hours=len(hours), weather_fields=len(fields),
        native_grid_spacing_m=3000, crs_wkt=wkt, source_row_bounds=[row.start, row.stop], source_col_bounds=[col.start, col.stop],
        cells_in_region=int(xx.size), cells_nearest_tanks=len({(r["source_grid_row"],r["source_grid_col"]) for r in mapping}),
        candidate_tanks=len(mapping), scene_clock_rows=len(links),
        scene_clocks_inside_archive=sum(r["model_hour_in_archive"] for r in links),
        scene_clocks_with_cloud_and_shortwave=sum(r[SCENE_FIELDS[0]] is not None and r[SCENE_FIELDS[2]] is not None for r in links),
        array_bytes=sum(r["bytes"] for r in fields), missing_values_by_variable={
            r["variable"]: r["total_values"]-r["finite_values"] for r in fields},
        interpretation="Published HRRR model analysis (dynamical.org snapshot). Scene links use the hour at or before "
        "acquisition. This archive does not establish first availability of the operational model file. "
        "Fields are inputs to engineering diagnostics. No roof height or trade is inferred.")
    body = json.dumps(summary, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", body.encode())
    print({k: summary[k] for k in ("hours", "weather_fields", "cells_in_region", "candidate_tanks",
                                  "scene_clock_rows", "scene_clocks_with_cloud_and_shortwave", "array_bytes")}, flush=True)


if __name__ == "__main__":
    main()
