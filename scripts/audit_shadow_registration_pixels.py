"""Read the original padded Sentinel-2 windows and measure their registration inputs."""
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import sqrt
from types import SimpleNamespace

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from build_l1c_quality import CLOUD_TYPES, gml_groups, native_mask, paint
from build_shadow_registration_inputs import PATCH_M, PADDING_M, PIXEL_M
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_public import ROOT, write
from collect_s2_l1c import source_metadata_bytes

STEM = "shadow_registration"


def index(stem, field):
    rows = read_csv(f"{STEM}_{stem}")
    result = {r[field]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {field} in {stem}")
    return result


def patch_bounds(row):
    return tuple(int(row[k]) + sign*PADDING_M for k, sign in zip(
        ("west", "south", "east", "north"), (-1, -1, 1, 1), strict=True))


def requested_patches(windows, pairs, patches):
    wanted = defaultdict(set)
    for row in windows:
        pair = pairs[row["scene_pair_id"]]
        ids = set(filter(None, (row["fit_patch_ids"] + "|" + row["spatial_check_patch_ids"]).split("|")))
        if not ids <= patches.keys():
            raise ValueError(f"Unknown context patch for {row['scene_pair_id']}")
        for side in ("earlier", "later"):
            wanted[pair[f"{side}_source_id"]].update(ids)
    return wanted


def cloud_arrays(source, reference):
    path = ROOT / source["cloud_mask"]
    if source["cloud_mask_basis"] == "gml_opaque_cirrus":
        groups, _ = gml_groups(path, reference, False)
        return tuple(paint(groups.get(label, []), reference) for label in CLOUD_TYPES)
    metadata = ET.fromstring(source_metadata_bytes(ROOT / source["tile_metadata"]))
    return native_mask(path, reference, metadata, 3)


def texture(red, usable):
    """Record contrast along both axes in the central, unpadded patch."""
    pad = PADDING_M // PIXEL_M
    size = PATCH_M // PIXEL_M
    core, before, after = slice(pad, pad+size), slice(pad-1, pad+size-1), slice(pad+1, pad+size+1)
    good = usable[core, core] & usable[core, before] & usable[core, after] & usable[before, core] & usable[after, core]
    if not good.any():
        return 0, None, None
    dx = ((red[core, after] - red[core, before])/2)[good]
    dy = ((red[before, core] - red[after, core])/2)[good]
    xx, yy, xy = float(np.mean(dx*dx)), float(np.mean(dy*dy)), float(np.mean(dx*dy))
    smallest = max(0., ((xx+yy) - sqrt((xx-yy)**2 + 4*xy**2))/2)
    return int(good.sum()), sqrt(xx+yy), sqrt(smallest)


def check_source(source, patches):
    boxes = {p["anchor_id"]: patch_bounds(p) for p in patches}
    bounds = tuple(reduce(b[index] for b in boxes.values()) for index, reduce in enumerate((min, min, max, max)))
    with rasterio.open(ROOT / source["red_raster"]) as dataset:
        window = from_bounds(*bounds, dataset.transform)
        raw = dataset.read(1, window=window)
        valid = (dataset.read_masks(1, window=window) != 0) & ~np.isin(
            raw, [int(source["red_nodata_dn"]), int(source["red_saturated_dn"])])
        reference = SimpleNamespace(bounds=bounds, shape=raw.shape, crs=dataset.crs,
                                    transform=dataset.window_transform(window))
        red = raw.astype("float32")*float(source["red_reflectance_scale"]) + float(source["red_reflectance_offset"])
    clouds = cloud_arrays(source, reference)
    if any(layer.shape != raw.shape or int(layer.max()) > 1 for layer in clouds):
        raise ValueError(f"Invalid native cloud-mask window for source {source['source_id']}")
    expected = ((PATCH_M + 2*PADDING_M)//PIXEL_M)**2
    result = []
    for patch in patches:
        anchor = patch["anchor_id"]
        window = from_bounds(*boxes[anchor], reference.transform)
        if any(abs(v-round(v)) > 1e-8 for v in (window.row_off, window.col_off, window.height, window.width)):
            raise ValueError(f"Registration patch is off the original grid: {anchor}")
        cells = window.toslices()
        valid_cells = valid[cells]
        opaque, cirrus, snow = (mask[cells] != 0 for mask in clouds)
        if valid_cells.size != expected:
            raise ValueError(f"Registration patch is clipped in source {source['source_id']}: {anchor}")
        clear = valid_cells & ~(opaque | cirrus | snow)
        count, magnitude, minor_axis = texture(red[cells], clear)
        result.append(dict(source_id=source["source_id"], anchor_id=anchor, role=patch["role"],
            padded_pixels=expected, valid_red_pixels=int(valid_cells.sum()), opaque_pixels=int(opaque.sum()),
            cirrus_pixels=int(cirrus.sum()), snow_ice_pixels=int(snow.sum()) if source["cloud_mask_basis"].startswith("jp2") else None,
            usable_by_available_source_masks_pixels=int(clear.sum()), usable_gradient_pixels=count,
            rms_gradient_reflectance=magnitude, minor_axis_gradient_reflectance=minor_axis))
    return result


def coverage_rows(windows, pairs, measured):
    rows, coverage = [], defaultdict(Counter)
    for row in windows:
        pair = pairs[row["scene_pair_id"]]
        sources = [pair[f"{side}_source_id"] for side in ("earlier", "later")]
        clear = {}
        for role in ("fit", "spatial_check"):
            ids = filter(None, row[f"{role}_patch_ids"].split("|"))
            clear[role] = [anchor for anchor in ids if all(
                (item := measured[source, anchor])["usable_by_available_source_masks_pixels"] == item["padded_pixels"]
                and item["minor_axis_gradient_reflectance"] > 0 for source in sources)]
        status = ("at_least_two_patches_in_each_group" if min(map(len, clear.values())) >= 2 else
                  "at_least_one_patch_in_each_group" if all(clear.values()) else "one_or_both_groups_empty")
        coverage[f"{row['comparison']}/{row['study_phase']}"][status] += 1
        rows.append(dict(comparison=row["comparison"], study_phase=row["study_phase"],
                         week_or_control_day=row["week_or_control_day"], candidate_id=row["candidate_id"],
                         scene_pair_id=row["scene_pair_id"], mask_and_texture_status=status,
                         **{f"{role}_screened_patch_count": len(ids) for role, ids in clear.items()},
                         **{f"{role}_screened_patch_ids": "|".join(ids) for role, ids in clear.items()}))
    return rows, dict(coverage)


def main():
    sources, pairs, patches = (index(table, field) for table, field in (
        ("sources", "source_id"), ("scene_pairs", "scene_pair_id"), ("patches", "anchor_id")))
    windows = read_csv(f"{STEM}_tank_windows")
    wanted = requested_patches(windows, pairs, patches)
    arguments = [(sources[key], [patches[anchor] for anchor in sorted(ids)])
                 for key, ids in sorted(wanted.items(), key=lambda item: int(item[0])) if ids]
    measured = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = pool.map(check_source, (r[0] for r in arguments), (r[1] for r in arguments))
        for number, rows in enumerate(results, 1):
            for row in rows:
                measured[row["source_id"], row["anchor_id"]] = row
            if number % 50 == 0 or number == len(arguments):
                print(f"Checked pixels in {number}/{len(arguments)} exact scenes", flush=True)
    if len(measured) != sum(map(len, wanted.values())):
        raise ValueError("The original raster audit did not cover each requested context window")
    coverage, counts = coverage_rows(windows, pairs, measured)
    export(f"{STEM}_patch_quality", list(measured.values()))
    export(f"{STEM}_pixel_coverage", coverage)
    report = dict(scenes=len(arguments), source_patch_windows=len(measured), tank_windows=len(coverage), coverage=counts,
        interpretation="Both scenes must have valid B04 across the padded patch, clear pixels according to available "
        "source masks, and positive gradient energy in both dimensions. Landmark stability and registration accuracy "
        "require pairwise checks. Early GML products lack a separate snow classification layer.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_pixel_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
