"""Test repeatability of September Grandstaff roof heights on overlapping Sentinel-2 tiles."""

import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from statistics import median

from build_grandstaff_okeis_records import STEM as OPERATOR, TERMINAL
from build_grandstaff_september_landsat import MONTH
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_timing_measurements import fit_additional
from build_weekly_shadow_panel import RESOLVED, SOURCE, read_only
from build_weekly_shadow_validation import DATED, legacy_geometry
from collect_public import write

STEM = "cushing_grandstaff_september_sentinel_tiles"
TILES = ("T14SPE", "T14SQE")


def endpoints(intervals):
    wanted = {(r["candidate_id"], r[f"{side}_version"]) for r in intervals for side in ("earlier", "later")}
    observations = {}
    pairs = {}
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        for roof, version in sorted(wanted):
            fields = version.split("_")
            if fields[5] not in TILES:
                raise ValueError(f"Unexpected Sentinel-2 tile {version}")
            rows = []
            for tile in TILES:
                exact = "_".join((*fields[:5], tile, fields[6]))
                row, = db.execute("""SELECT candidate_id,scene_item_id AS canonical_l2a_id,l1c_product_name AS version_id,
                    captured_at_utc,source_directory,tile_metadata_path,cloud_mask_basis,opaque_roof_cells,cirrus_roof_cells,
                    optical_touched_cells_expected,optical_cells_valid_in_all_bands,red_detector_covered_roof_cells
                    FROM sentinel2_l1c_tank_quality WHERE candidate_id=? AND l1c_product_name=?""", (roof, exact))
                key = roof, exact, row["canonical_l2a_id"]
                observations[key] = dict(row, study_phase="holdout", tile=tile)
                rows.append(key)
            pairs[roof, version] = tuple(rows)
    return observations, pairs


def comparisons(intervals):
    observations, pairs = endpoints(intervals)
    fits, _ = fit_additional("holdout", STEM + "_native", {key: {DATED} for key in observations}, observations,
                             legacy_geometry(), reuse_pixels=False)
    archived = {(r["candidate_id"], r["version_id"]): r for r in read("weekly_shadow_holdout_observations")
                if r["terminal_id"] == TERMINAL and r["geometry_mode"] == DATED}
    results = {}
    for (roof, original), keys in pairs.items():
        record = dict(candidate_id=roof, primary_version_id=original)
        captured = []
        for key in keys:
            source, fit = observations[key], fits[DATED, *key]
            tile = source["tile"].lower()
            captured.append(datetime.fromisoformat(source["captured_at_utc"]))
            if key[1] == original:
                saved = archived[roof, original]
                if fit["fit_status"] != saved["fit_status"] or fit["estimated_gap_m"] != float(saved["estimated_gap_m"]):
                    raise ValueError(f"Baseline could not be reproduced {roof}/{original}")
            record.update({f"{tile}_{field}": source[field] for field in (
                "version_id", "captured_at_utc", "source_directory", "tile_metadata_path", "cirrus_roof_cells")})
            record.update({f"{tile}_{field}": fit.get(field) for field in (
                "fit_status", "estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m")})
        elapsed = abs((captured[1] - captured[0]).total_seconds())
        if elapsed > 10:
            raise ValueError(f"Different acquisition times for overlapping Sentinel-2 tiles {roof}/{original}")
        valid = all(record[f"{tile.lower()}_fit_status"] == RESOLVED for tile in TILES)
        record.update(capture_difference_seconds=elapsed, both_fits_resolved=valid,
                      qe_minus_pe_gap_m=record["t14sqe_estimated_gap_m"] - record["t14spe_estimated_gap_m"] if valid else None)
        results[roof, original] = record
    return list(results.values()), results


def main():
    intervals = [r for r in read(f"{OPERATOR}_optical_throughput_intervals") if r["report_month"] == MONTH]
    endpoints_rows, fits = comparisons(intervals)
    rows = []
    for r in intervals:
        roof = r["candidate_id"]
        first, last = [fits[roof, r[f"{side}_version"]] for side in ("earlier", "later")]
        baseline_tile = r["earlier_version"].split("_")[5].lower()
        if baseline_tile != r["later_version"].split("_")[5].lower():
            raise ValueError("Mixed baseline tiles require comparisons using both physical grid locations")
        other, = [tile.lower() for tile in TILES if tile.lower() != baseline_tile]
        both = first["both_fits_resolved"] and last["both_fits_resolved"]
        primary_change = first[f"{baseline_tile}_estimated_gap_m"] - last[f"{baseline_tile}_estimated_gap_m"]
        alternate = first[f"{other}_estimated_gap_m"] - last[f"{other}_estimated_gap_m"] if both else None
        rows.append(dict(candidate_id=roof, earlier_version=r["earlier_version"], later_version=r["later_version"],
                         baseline_roof_height_change_m=primary_change, alternate_roof_height_change_m=alternate,
                         alternate_minus_baseline_change_m=alternate-primary_change if both else None,
                         original_net_change_equivalent_bbl=float(r["satellite_net_change_equivalent_bbl"])))
    export(f"{STEM}_endpoints", endpoints_rows)
    export(f"{STEM}_intervals", rows)
    valid = [r for r in endpoints_rows if r["both_fits_resolved"]]
    complete = [r for r in rows if r["alternate_roof_height_change_m"] is not None]
    summary = dict(same_pass_roof_height_candidates=len(endpoints_rows), both_tile_heights_resolved=len(valid),
                   same_pass_median_absolute_difference_m=median(abs(r["qe_minus_pe_gap_m"]) for r in valid) if valid else None,
                   same_pass_maximum_absolute_difference_m=max((abs(r["qe_minus_pe_gap_m"]) for r in valid), default=None),
                   both_tile_changes_resolved=len(complete),
                   alternate_tile_median_absolute_change_difference_m=median(abs(r["alternate_minus_baseline_change_m"])
                       for r in complete) if complete else None,
                   overlap_is_independent_sensor=False)
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
