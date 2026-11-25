"""Compare 15 cm state roof measurements with native Sentinel-2 pixels."""
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime

from audit_s2_sun_clock import xml_time
from audit_shadow_precision import describe
from build_radar_optical_cushing_controls import build_pixels, fit_optical, optical_inventory
from build_s1_reference_matches import read
from build_shadow_change_controls import SELECT_QUALITY
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, export, read_only
from collect_public import ROOT, write
from collect_s2_l1c import source_metadata_path

DB = DERIVED / "ok_aerial_2025_sentinel2_controls.sqlite"
PHOTO_RESOLVED = "resolved_aerial_shadow"
MAX_HOURS = 96


def matches(source):
    photos = {(r["candidate_id"], r["nominal_tile_id"], r["reference_day"]): r
              for r in read("ok_aerial_2025_roof_shadow_reference")}
    manifest = [r for r in read("aerial_optical_pair") if r["aerial_source"] == "ok_aerial_2025_tile_majority"
                and r["level"] == "L1C" and float(r["maximum_separation_seconds"]) <= MAX_HOURS*3600+60
                and (r["candidate_id"], r["aerial_id"], r["aerial_date"]) in photos]
    catalog = optical_inventory(source)
    by_version = defaultdict(list)
    for m in manifest:
        if m["version_id"] in catalog:
            by_version[m["version_id"]].append(m)
    groups = defaultdict(dict)
    results = []
    for version, entries in sorted(by_version.items()):
        c = catalog[version]
        native = {r["candidate_id"]: dict(r) for r in source.execute(SELECT_QUALITY+" AND q.l1c_product_name=?", (version,))}
        for m in entries:
            tank, tile, day = (m[k] for k in ("candidate_id", "aerial_id", "aerial_date"))
            photo, q = photos[tank, tile, day], native[tank]
            start, stop = (datetime.fromisoformat(photo[k]) for k in ("start_utc", "end_exclusive_utc"))
            if (start, stop) != tuple(datetime.fromisoformat(m[k]) for k in ("aerial_start_utc", "aerial_end_exclusive_utc")):
                raise ValueError(f"The state aerial clock disagrees for {tank} in {tile}")
            if q["optical_source_directory"] != m["satellite_source_directory"] or q["scene_item_id"] != c["canonical_l2a_id"]:
                raise ValueError(f"The Sentinel-2 source identity disagrees for {tank} in {version}")
            path = str(source_metadata_path(ROOT / q["tile_metadata_path"]).relative_to(ROOT))
            time = xml_time(path)
            lag = max(abs((time-start).total_seconds()), abs((time-stop).total_seconds()))
            if lag > MAX_HOURS*3600:
                continue
            passed = q["optical_cells"] == q["valid_cells"] and q["roof_clear_by_source_mask"] and q["resolved_b04_detector"]
            if passed:
                groups[q["level"], version, q["optical_source_directory"]][tank] = dict(q, optical_captured_at_utc=time.isoformat())
            results.append(dict(candidate_id=tank, terminal_id=photo["documented_terminal_id"], roof_type=photo["documented_roof_type"],
                aerial_tile_id=tile, aerial_nominal_start_utc=start.isoformat(), aerial_nominal_end_utc=stop.isoformat(),
                aerial_timing_basis=photo["timing_basis"], exterior_clock_status=photo["exterior_clock_status"],
                aerial_fit_status=photo["shadow_reference_status"], aerial_gap_m=photo["best_gap_m"],
                shared_lidar_rim_to_ground_m=float(photo["shell_m"]), optical_version=version,
                optical_source_directory=q["optical_source_directory"], optical_native_start_utc=time.isoformat(),
                optical_native_metadata=path, optical_archive_available_at_utc=c["archive_available_at_utc"],
                optical_geometry_available_at_utc=c["geometry_available_at_utc"], optical_source_quality_pass=bool(passed),
                optical_valid_cells=q["valid_cells"], optical_side="before" if time < start else "after" if time >= stop else "overlap",
                maximum_nominal_separation_hours=lag/3600, selected_for_comparison=False))
    return results, groups


def compare(pairs, fits):
    by_reference = defaultdict(list)
    for row in pairs:
        key = row["candidate_id"], row["optical_version"], row["optical_source_directory"]
        fit = fits.get(key, {})
        gap = fit.get("estimated_gap_m")
        aerial = float(row["aerial_gap_m"]) if row["aerial_gap_m"] else None
        row.update(optical_fit_status=fit.get("fit_status", "source_quality_gate_failed"), optical_gap_m=gap,
            optical_observation_id=fit.get("observation_id"),
            both_diagnostics_pass=row["aerial_fit_status"] == PHOTO_RESOLVED and fit.get("fit_status") == RESOLVED,
            optical_minus_aerial_height_m=aerial-gap if None not in (aerial, gap) else None)
        by_reference[row["candidate_id"], row["aerial_tile_id"], row["optical_side"]].append(row)
    for rows in by_reference.values():
        best = min(rows, key=lambda r: (not r["optical_source_quality_pass"], r["maximum_nominal_separation_hours"],
                                       -r["optical_valid_cells"], r["optical_version"]))
        best["selected_for_comparison"] = True


def main():
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        pairs, groups = matches(source)
        build_pixels(source, groups, DB)
    fits = fit_optical(pairs, 4, DB)
    compare(pairs, fits)
    export("ok_aerial_2025_sentinel2_roof_comparisons", pairs)
    selected = [r for r in pairs if r["selected_for_comparison"]]
    accepted = [r for r in selected if r["roof_type"] == "EFR" and r["both_diagnostics_pass"]]
    by_side = {side: describe([r["optical_minus_aerial_height_m"] for r in accepted if r["optical_side"] == side])
               for side in ("before", "after")}
    report = dict(pairs=len(pairs), selected_pairs=len(selected), archived_optical_observations=len(fits),
        selected_efr_both_diagnostics_pass=len(accepted), distinct_efr_tanks_with_comparison=len({r["candidate_id"] for r in accepted}),
        closest_accepted_nominal_lag_hours=min((r["maximum_nominal_separation_hours"] for r in accepted), default=None),
        disagreement_by_side=by_side,
        disagreement_exterior_clock_corroborated=describe([r["optical_minus_aerial_height_m"] for r in accepted
            if r["exterior_clock_status"] == "exterior_matches_tile_time"]),
        interpretation="The aerial tile timestamp is nominal. The exact native Sentinel-2 granule time is preserved. "
        "Aerial and satellite measurements can differ from each other due to scene timing, geometry and physical inventory movement.")
    write(DERIVED / "ok_aerial_2025_sentinel2_roof_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
