"""Date the HCS roof evidence and measure its historical Sentinel-2 input pairs."""
import argparse
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import lru_cache

from audit_sentinel2_alternate_registration import packet_inputs, pair_key
from audit_sentinel2_space_time_coverage import roof_registry
from audit_sentinel2_weekly_volume import PHASES, observations
from build_registered_shadow_development import DATED, RETROSPECTIVE, baseline_pairs, registered_pair, registered_profile, study_weeks
from build_shadow_calibration import CONTEXT_M
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write
from terminal_aerial_roof_evidence import HCS_SCOPE, evidence_rows

TERMINAL = "amid_cushing"
STEM = "hcs_dated_shadow"


def provenance(registry):
    fields = ("scope_id", "candidate_id", "observed_date", "top_observation", "detail", "naip_item_id",
              "image_file", "image_url", "review_sheet", "review_row", "review_column", "publication_date",
              "publication_basis", "publication_source_file", "publication_source_url")
    rows = [dict(zip(fields, row, strict=True)) for row in evidence_rows() if row[0] == HCS_SCOPE]
    if len(rows) != len(registry) or {r["candidate_id"]: r["publication_date"] for r in rows} != {
        candidate: g["roof_evidence_date"] for candidate, g in registry.items()
    }:
        raise ValueError("HCS geometry and verified roof evidence do not describe the same tanks")
    return rows


def fit_phase(phase, weeks, registry, geometry):
    offsets = {pair_key(r): r for r in read_csv(f"weekly_shadow_registered_{phase}_pairs")
               if r["geometry_mode"] == RETROSPECTIVE and r["terminal_id"] == TERMINAL}
    candidates = [(packet, row) for mode, packet, _, row in baseline_pairs(phase)
                  if mode == RETROSPECTIVE and row["terminal_id"] == TERMINAL]
    if len(candidates) != len(offsets) or {pair_key(r) for _, r in candidates} != offsets.keys():
        raise ValueError(f"HCS input and registration pairs differ in {phase}")

    @lru_cache(maxsize=600)
    def profile(packet, number, east, north):
        path, native, _, kernels = packet_inputs(packet)
        row = native[number]
        old = geometry[row["candidate_id"]]
        x, y, diameter = (float(old[field]) for field in ("center_east_m", "center_north_m", "diameter_m"))
        distance = math.hypot(x-row["center_east_m"], y-row["center_north_m"])
        if distance+max(0., diameter/2-row["radius_m"]) > CONTEXT_M-16:
            return None
        observation = dict(row, center_east_m=x, center_north_m=y, radius_m=diameter/2)
        shell = registry[row["candidate_id"]]["shell_above_ground_m"]
        return registered_profile(path, observation, None, shell, kernels[row["platform"]], east, north)

    def solve(item):
        packet, row = item
        packet += ".sqlite"
        week, tank = pair_key(row)
        offset = offsets[week, tank]
        _, native, _, _ = packet_inputs(packet)
        roof = registry[tank]
        start, end, release, cutoff = weeks[week]
        cutoff_day = cutoff[:10]
        fields = ("week_ending", "release_date", "candidate_id", "terminal_id", "earlier_version", "later_version",
                  "earlier_capture_at_utc", "later_capture_at_utc", "inputs_available_before_utc",
                  "earlier_archive_available_at_utc", "later_archive_available_at_utc")
        result = dict(study_phase=phase, geometry_mode=DATED, **{field: offset[field] for field in fields},
            native_packet=packet, earlier_observation_id=int(row["earlier_observation_id"]),
            later_observation_id=int(row["later_observation_id"]), roof_evidence_date=roof["roof_evidence_date"],
            lidar_source_available_date=roof["lidar_source_available_date"], circular_area_m2=roof["circular_area_m2"],
            circular_bbl_per_m=roof["circular_area_m2"]/BARREL_M3, offset_status=offset["offset_status"],
            offset_east_m=offset["offset_east_m"], offset_north_m=offset["offset_north_m"],
            fit_status=None, registered_both_shadow_diagnostics_pass=False,
            earlier_registered_gap_m=None, later_registered_gap_m=None, earlier_registered_status=None,
            later_registered_status=None, registered_change_m=None, registered_centre_east_m=None,
            registered_centre_north_m=None, registered_loss_range_min_m=None, registered_loss_range_max_m=None)
        before, after = (datetime.fromisoformat(result[f"{side}_capture_at_utc"]) for side in ("earlier", "later"))
        archive = max(datetime.fromisoformat(result[f"{side}_archive_available_at_utc"]) for side in ("earlier", "later"))
        result.update(fraction_of_capture_inside_eia_interval=(after-start)/(after-before),
                      fraction_of_eia_interval_observed=(after-start)/(end-start))
        if ((release, cutoff) != (result["release_date"], result["inputs_available_before_utc"])
            or not before <= start < after <= end or not after <= archive < datetime.fromisoformat(cutoff)):
            raise ValueError(f"HCS capture clocks differ for {week}/{tank}")
        for side in ("earlier", "later"):
            o = native[result[f"{side}_observation_id"]]
            if (o["version_id"], o["candidate_id"]) != (result[f"{side}_version"], tank):
                raise ValueError(f"HCS optical source differs for {week}/{tank}/{side}")
        if max(result["roof_evidence_date"], result["lidar_source_available_date"]) >= cutoff_day:
            return dict(result, fit_status="engineering_evidence_after_cutoff")
        if offset["offset_status"] != "checked_offset" or not row["optical_level_change_m"]:
            return dict(result, fit_status="unconfirmed_exterior_alignment_or_native_measurement")
        east, north = (float(offset[f"offset_{axis}_m"])/2 for axis in ("east", "north"))
        both = [profile(packet, result[f"{side}_observation_id"], sign*east, sign*north)
                for side, sign in (("earlier", -1), ("later", 1))]
        if any(item is None for item in both):
            return dict(result, fit_status="dated_geometry_outside_chip")
        fit = registered_pair(*both)
        result.update(fit)
        result["fit_status"] = "resolved" if fit["registered_both_shadow_diagnostics_pass"] else "shadow_diagnostics_failed"
        return result

    with ThreadPoolExecutor(max_workers=5) as pool:
        return list(pool.map(solve, candidates))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("holdout",), help="Write the independent holdout optical inputs")
    phase = parser.parse_args().phase
    registry = {r["candidate_id"]: r for r in roof_registry() if r["terminal_id"] == TERMINAL}
    geometry = legacy_geometry()
    sources = provenance(registry)
    if phase == "holdout":
        weeks = study_weeks(phase)
        fitted = fit_phase(phase, weeks, registry, geometry)
        volumes = list(observations(phase, weeks, geometry, fitted))
        export(f"{STEM}_holdout_pairs", fitted)
        export(f"{STEM}_holdout_volumes", volumes)
        print(f"HCS holdout: {len(fitted)} native pairs, {len(volumes)} resolved pairs", flush=True)
        return
    all_fits, all_volumes, report = [], [], {}
    for phase in PHASES:
        weeks = study_weeks(phase)
        fitted = fit_phase(phase, weeks, registry, geometry)
        volumes = list(observations(phase, weeks, geometry, fitted))
        report[phase] = dict(native_pairs=len(fitted), fits_by_status=dict(Counter(r["fit_status"] for r in fitted)),
                            resolved_pairs=len(volumes), resolved_weeks=len({r["week_ending"] for r in volumes}))
        all_fits.extend(fitted)
        all_volumes.extend(volumes)
    holdout = []
    for row in read_csv("weekly_shadow_holdout_input_manifest"):
        if row["terminal_id"] != TERMINAL:
            continue
        roof = registry[row["candidate_id"]]
        dated = max(roof["roof_evidence_date"], roof["lidar_source_available_date"]) < row["inputs_available_before_utc"][:10]
        holdout.append(dict(row, hcs_verified_roof_evidence_date=roof["roof_evidence_date"],
                            hcs_verified_lidar_source_date=roof["lidar_source_available_date"],
                            hcs_dated_engineering_evidence_before_cutoff=dated))
    for suffix, rows in (("roof_evidence", sources), ("pairs", all_fits), ("volumes", all_volumes),
                         ("holdout_inputs", holdout)):
        export(f"{STEM}_{suffix}", rows)
    report["holdout"] = dict(input_pairs=len(holdout), dated_input_pairs=sum(
        r["hcs_dated_engineering_evidence_before_cutoff"] for r in holdout))
    payload = json.dumps(dict(phases=report, interpretation="HCS roof dates use the NAIP source publication, "
        "with exact paired inputs and separately documented LiDAR availability. This package contains no EIA or price outcomes."),
        indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
