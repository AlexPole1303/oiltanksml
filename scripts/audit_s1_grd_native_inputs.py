"""Verify radar windows, source clocks and matched weekly coverage."""
import argparse
import hashlib
import json
from collections import defaultdict
from contextlib import closing
from datetime import datetime

import numpy as np
import rasterio
from rasterio.windows import Window

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import ROOT, write
from collect_s1_grd_roof_inputs import STORE


def saved_assets(manifests):
    clocks = {}
    count = 0
    for manifest in manifests:
        identity = manifest["product"]["product_id"]
        objects = manifest["objects"]
        inputs = ["manifest.safe", "productInfo.json"]
        for pol in manifest["bands"]:
            label = f"iw-{pol.lower()}"
            inputs.extend((f"measurement/{label}.tiff", f"annotation/{label}.xml",
                           f"annotation/calibration/calibration-{label}.xml", f"annotation/calibration/noise-{label}.xml"))
        for name in inputs:
            if name.startswith("measurement/"):
                continue
            evidence = objects[name]
            content = (ROOT / evidence["local_file"]).read_bytes()
            if len(content) != evidence["bytes"] or hashlib.sha256(content).hexdigest() != evidence["sha256"]:
                raise ValueError(f"Saved source annotation differs for {identity}/{name}")
            count += 1
        clocks[identity] = max(datetime.fromisoformat(objects[name]["last_modified_utc"]) for name in inputs)
    return clocks, count


def saved_windows(rows):
    by_file = defaultdict(list)
    for row in rows:
        by_file[row["window_file"]].append(row)
    for path, same_file in by_file.items():
        row = same_file[0]
        common = ("product_id", "polarization", "window_sha256", "window_rows", "window_cols",
                  "window_row_offset", "window_col_offset", "source_url")
        if any(any(other[k] != row[k] for k in common) for other in same_file):
            raise ValueError(f"Multiple definitions for native radar window {path}")
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != row["window_sha256"]:
            raise ValueError(f"Saved window hash differs for {path}")
        with np.load(ROOT / path, allow_pickle=False) as packet:
            dn, valid = packet["dn"], packet["valid"]
            shape = tuple(int(row[key]) for key in ("window_rows", "window_cols"))
            if dn.dtype != np.uint16 or valid.dtype != bool or dn.shape != shape or valid.shape != shape:
                raise ValueError(f"Saved window schema differs for {path}")
            for candidate in same_file:
                i = int(float(candidate["native_line"])) - int(row["window_row_offset"])
                j = int(float(candidate["native_sample"])) - int(row["window_col_offset"])
                if not (0 <= i < shape[0] and 0 <= j < shape[1]) or bool(valid[i, j]) != (candidate["centre_pixel_valid"] == "True"):
                    raise ValueError(f"Saved candidate does not match its native radar pixel in {path}")
    return {path: group[0] for path, group in by_file.items()}


def source_sample(windows, *, per_product=False):
    checked = set()
    for row in sorted(windows.values(), key=lambda r: r["product_name"]):
        period = row["product_id"] if per_product else row["product_name"].split("_")[4][:4]
        key = period, row["polarization"]
        if key in checked:
            continue
        path = ROOT / row["window_file"]
        window = Window(*(int(row[k]) for k in ("window_col_offset", "window_row_offset", "window_cols", "window_rows")))
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="25"):
            with rasterio.open(row["source_url"]) as source, np.load(path, allow_pickle=False) as local:
                if (not np.array_equal(source.read(1, window=window), local["dn"])
                        or not np.array_equal(source.read_masks(1, window=window) > 0, local["valid"])):
                    raise ValueError(f"Raw download differs from the source window for {path.name}")
        checked.add(key)
    return len(checked)


def weekly_coverage(manifests, rows, clocks):
    products = {m["product"]["product_id"]: m["product"] for m in manifests}
    captured = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if row["centre_pixel_valid"] == "True":
            captured[row["product_id"]][row["polarization"]].add(row["candidate_id"])
    with closing(read_only(SOURCE)) as db:
        ends = {(day, side): (datetime.fromisoformat(start), datetime.fromisoformat(stop), datetime.fromisoformat(cutoff))
                for day, side, start, stop, cutoff in db.execute("""SELECT week_ending,endpoint,window_starts_after_utc,
                    window_ends_by_utc,inputs_available_before_utc FROM eia_satellite_window WHERE study_phase='validation'""")}
        known_efr = dict(db.execute("""SELECT candidate_id,first_seen FROM tank_engineering_registry
            WHERE documented_roof_type='EFR' AND first_seen IS NOT NULL"""))
    weeks = defaultdict(lambda: defaultdict(list))
    for identity, product in products.items():
        for reason in product["selection_reasons"].split("|"):
            if reason.startswith("optical_gap_"):
                _, _, day, side = reason.split("_")
                weeks[day][side].append(identity)
    for day, sides in sorted(weeks.items()):
        for earlier in sides["previous"]:
            for later in sides["current"]:
                old, new = products[earlier], products[later]
                if (old["relative_orbit"], old["direction"]) != (new["relative_orbit"], new["direction"]):
                    continue
                start, finish, cutoff = ends[day, "current"]
                checks = []
                for side, identity in (("previous", earlier), ("current", later)):
                    lower, upper, available = ends[day, side]
                    p = products[identity]
                    first, last = (datetime.fromisoformat(p[field]) for field in ("capture_start_utc", "capture_end_utc"))
                    checks.append(first > lower and last <= upper and clocks[identity] < available)
                a, b = (datetime.fromisoformat(p["capture_start_utc"]) for p in (old, new))
                complete = [set.intersection(*captured[identity].values()) for identity in (earlier, later)]
                paired = complete[0] & complete[1]
                yield dict(week_ending=day, previous_product_id=earlier, current_product_id=later,
                    previous_capture_start_utc=a.isoformat(), current_capture_start_utc=b.isoformat(),
                    previous_archive_ready_utc=clocks[earlier].isoformat(), current_archive_ready_utc=clocks[later].isoformat(),
                    conservative_eia_cutoff_utc=cutoff.isoformat(), both_products_pass_archive_and_capture_clocks=all(checks),
                    same_relative_orbit=old["relative_orbit"], direction=old["direction"],
                    paired_valid_native_centres=len(paired),
                    paired_current_registry_efr_seen_before_capture=sum(tank in known_efr and known_efr[tank] < a.date().isoformat()
                                                                         for tank in paired),
                    capture_interval_hours=round((b-a).total_seconds()/3600, 6),
                    hours_between_earlier_capture_and_measurement_start=round((start-a).total_seconds()/3600, 6),
                    hours_between_later_capture_and_measurement_end=round((finish-b).total_seconds()/3600, 6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recheck-source", action="store_true")
    args = parser.parse_args()
    rows = read("s1_grd_native_window_index")
    manifests = [json.loads(path.read_bytes()) for path in sorted(STORE.glob("*/native_manifest.json"))]
    expected = {r["product_id"] for r in read("s1_grd_native_pilot_manifest")}
    if {m["product"]["product_id"] for m in manifests} != expected or {r["product_id"] for r in rows} != expected:
        raise ValueError("Native radar products differ from the collection manifest")
    keys = {(r["product_id"], r["polarization"], r["candidate_id"]) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("Repeated radar product/polarization/contour rows")
    clocks, annotation_count = saved_assets(manifests)
    windows = saved_windows(rows)
    source_count = source_sample(windows) if args.recheck_source else None
    weeks = list(weekly_coverage(manifests, rows, clocks))
    export("s1_grd_validation_native_coverage", weeks)
    report = dict(products=len(expected), source_metadata_files_verified=annotation_count,
        native_windows_verified=len(windows), repeated_source_byte_range_reads_verified=source_count,
        matched_2023_weeks=len({r["week_ending"] for r in weeks}),
        matched_2023_weeks_passing_capture_and_archive_clocks=len({r["week_ending"] for r in weeks
            if r["both_products_pass_archive_and_capture_clocks"]}),
        interpretation="The latest listed AWS LastModified across measurement, annotation, noise, calibration and "
        "product metadata is a conservative clock for that source copy. Candidate counts refer to valid native "
        "pixel samples near mapped centres. Clock and pixel checks do not establish a radar roof-height measurement.")
    write(DERIVED / "s1_grd_native_verification.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
