"""Match verified local Sentinel-1 SLC profiles to observations of the same roof."""

import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import pairwise

from audit_shadow_precision import describe
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from check_ceda_s1_slc_roof_profiles import PASS
from collect_public import ROOT, write

STEM = "cushing_slc_independent_controls"
RESOLVED_OPTICAL = "resolved_in_geometry_diagnostic"
HOURS = (12, 24, 48, 72)


def timestamp(value):
    return datetime.fromisoformat(value)


def separation_hours(observed, start, stop):
    return max(abs((observed - timestamp(value)).total_seconds()) for value in (start, stop)) / 3600


def reference(kind, source, identity, candidate, start, stop, gap, status, usable, *, low=None, high=None, shell=None,
              asset=None, sha=None, registration=None):
    return dict(reference_kind=kind, reference_table=f"data/derived/{source}.csv", reference_id=identity,
                candidate_id=candidate, reference_start_utc=start, reference_end_utc=stop,
                reference_gap_m=float(gap) if gap not in (None, "") else None,
                reference_low_m=float(low) if low not in (None, "") else None,
                reference_high_m=float(high) if high not in (None, "") else None,
                reference_status=status, reference_usable=usable, reference_shell_height_m=shell,
                reference_asset=asset, reference_asset_sha256=sha, reference_registration=registration)


def direct_controls(profiles):
    for source, kind in (("ceda_s1_concurrent_optical_48h_controls", "sentinel2_shadow"),
                         ("ceda_s1_slc_reference_pairs", "aerial_shadow")):
        for r in read(source):
            tank = r["candidate_id"]
            aerial = kind == "aerial_shadow"
            if aerial and r["within_72_hours"] != "True":
                continue
            if aerial:
                start, stop = r["reference_start_utc"], r["reference_end_exclusive_utc"]
                asset = r["reference_asset"]
                ref = reference(kind, source, asset, tank, start, stop, r["reference_gap_to_rim_m"],
                                r["reference_exterior_clock_status"], True, low=r["reference_near_gap_low_m"],
                                high=r["reference_near_gap_high_m"], asset=asset)
            else:
                start, stop = r["optical_capture_start_utc"], r["optical_capture_end_utc"]
                ref = reference(kind, source, r["optical_versions_json"], tank, start, stop, r["optical_gap_to_rim_m"],
                                RESOLVED_OPTICAL, True, low=r["low_over_all_optical_fits_m"],
                                high=r["high_over_all_optical_fits_m"])
            for pol in ("vv", "vh"):
                uuid = r[f"radar_{pol}_cdse_uuid" if aerial else f"radar_{pol}_burst_id"]
                profile = profiles[uuid, tank]
                if not aerial and profile["product_id"] != r["radar_product_id"]:
                    raise ValueError(f"Sentinel-2 reference has the wrong Sentinel-1 parent for {uuid}/{tank}")
                yield profile, ref


def landsat_references():
    for source in ("landsat_esri_b8_controls_fits", "cushing_recovered_landsat_b8_fits", "cushing_slc_march_b8_control_fits"):
        for r in read(source):
            if r["roof_type"] != "EFR":
                continue
            esri = r["source_type"] == "esri_apparent_reflectance"
            consistent = r.get("esri_exterior_overlap_status") == "shared_pixels_match" if esri else (
                r["usgs_original_verified"] == "True")
            yield reference("landsat_esri_b8" if esri else "landsat_original_b8", source, r["product_id"],
                            r["candidate_id"], r["optical_capture_at_utc"], r["optical_capture_at_utc"],
                            r.get("estimated_gap_m"), r["fit_status"], consistent and r["fit_status"] == RESOLVED_OPTICAL,
                            low=r.get("near_optimal_gap_min_m"), high=r.get("near_optimal_gap_max_m"),
                            shell=float(r["shell_above_ground_m"]), asset=r["local_path"], sha=r["raster_sha256"])


def commercial_references():
    source = "cushing_radar_arc_profiles"
    for r in read(source):
        if r["roof_type"] != "EFR":
            continue
        shell = float(r["lidar_shell_height_m"])
        yield reference(r["sensor"].lower(), source, r["product_id"], r["candidate_id"],
                        r["capture_start_utc"], r["capture_start_utc"], shell - float(r["height_m"]),
                        r["fit_status"], r["fit_status"] == "resolved_arc_diagnostic",
                        low=shell - float(r["omitted_block_max_m"]), high=shell - float(r["omitted_block_min_m"]),
                        shell=shell, asset=r["source_window_file"], sha=r["source_window_sha256"], registration=r["registration"])


def laser_references():
    source = "cushing_gedi_ground_controls"
    for r in read(source):
        gap = r["wall_to_mode_ground_corrected_m"] or r["native_wall_to_mode_m"]
        usable = (r["native_l2a_quality_flag_rel3"] == "1" and int(r["clean_same_beam_ground_shots"]) >= 3
                  and r["native_peaks"] == "1" and r["within_physical_wall_limits"] == "True"
                  and bool(r["wall_to_mode_ground_corrected_m"]))
        yield reference("gedi", source, r["shot_number"], r["candidate_id"], r["observed_at_utc"],
                        r["observed_at_utc"], gap, "qualified" if usable else "laser_quality_gate_failed", usable,
                        asset=r["native_shot_subset"])
    for source in ("cushing_icesat2_native_profiles", "cushing_icesat2_photon_recovery_profiles"):
        for r in read(source):
            first, last = r.get("first_roof_signal_photon_utc"), r.get("last_roof_signal_photon_utc")
            usable = r["passes_roof_profile_checks"] == "True" and bool(first and last)
            if source.endswith("_recovery_profiles") and r["new_qualified_reference"] != "True":
                continue
            yield reference("icesat2", source, f"{r['filename']}/{r['beam']}", r["candidate_id"],
                            first or r["granule_start_utc"], last or r["granule_end_utc"],
                            r["estimated_wall_to_roof_m"], "qualified" if usable else "laser_quality_gate_failed",
                            usable, asset=r["photon_packet"])


def extra_controls(profiles):
    refs = defaultdict(list)
    for group in (landsat_references(), commercial_references(), laser_references()):
        for ref in group:
            refs[ref["candidate_id"]].append(ref)
    for profile in profiles.values():
        for ref in refs[profile["candidate_id"]]:
            yield profile, ref


def comparisons(profiles):
    seen = set()
    for group in (direct_controls(profiles), extra_controls(profiles)):
        for p, r in group:
            when = timestamp(p["rim_zero_doppler_utc"])
            separation = separation_hours(when, r["reference_start_utc"], r["reference_end_utc"])
            if separation > max(HOURS):
                continue
            key = (p["cdse_uuid"], r["candidate_id"], r["reference_kind"], r["reference_id"],
                   r["reference_registration"], r["reference_start_utc"])
            if key in seen:
                raise ValueError(f"Duplicate SLC comparison {key}")
            seen.add(key)
            estimate = float(p["gap_m"])
            passed = p["status"] == PASS and r["reference_usable"] and r["reference_gap_m"] is not None
            yield dict(
                radar_product_id=p["product_id"], radar_burst_uuid=p["cdse_uuid"], radar_polarization=p["polarization"],
                radar_capture_utc=p["rim_zero_doppler_utc"], radar_scene_start_utc=p["acquisition_start_utc"],
                terminal_id=p["terminal_id"], radar_radius_m=float(p["radius_m"]),
                radar_shell_height_m=float(p["lidar_wall_height_m"]), radar_fit_status=p["status"],
                radar_gap_m=estimate, radar_profile_packet_sha256=p["calibrated_packet_sha256"],
                radar_profile_offset=int(p["profile_offset"]), radar_profile_height_steps=int(p["profile_height_steps"]),
                maximum_separation_hours=separation, both_pass=passed,
                radar_minus_reference_gap_m=estimate - r["reference_gap_m"] if r["reference_gap_m"] is not None else None, **r,
            )


def paired_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["reference_kind"], row["reference_registration"]].append(row)
    summary = {}
    for (kind, mode), group in groups.items():
        windows = {}
        summary[kind if mode is None else f"{kind}_{mode}"] = windows
        for window in HOURS:
            candidates = [r for r in group if r["maximum_separation_hours"] <= window]
            passed = [r for r in candidates if r["both_pass"]]
            windows[f"within_{window}h"] = dict(
                pairs=len(candidates), usable_reference_pairs=sum(r["reference_usable"] for r in candidates),
                passing_radar_pairs=sum(r["radar_fit_status"] == PASS for r in candidates),
                joint_pairs=len(passed), joint_roofs=len({r["candidate_id"] for r in passed}),
                joint_radar_dates=len({r["radar_scene_start_utc"][:10] for r in passed}),
                joint_reference_dates=len({r["reference_start_utc"][:10] for r in passed}),
                gap_error_m=describe([r["radar_minus_reference_gap_m"] for r in passed]),
            )
    return summary


def best_references(rows):
    nearest = {}
    for row in rows:
        if not row["reference_usable"] or row["reference_gap_m"] is None:
            continue
        key = (row["radar_burst_uuid"], row["candidate_id"], row["reference_kind"], row["reference_registration"])
        order = (row["maximum_separation_hours"], row["reference_id"], row["reference_start_utc"])
        if key not in nearest or order < nearest[key][0]:
            nearest[key] = (order, row)
    return {key: pair[1] for key, pair in nearest.items()}


def changes(profiles, references):
    scenes = defaultdict(dict)
    for p in profiles.values():
        scenes[p["acquisition_start_utc"], p["product_id"]][p["candidate_id"], p["polarization"]] = p
    for first, last in pairwise(sorted(scenes)):
        first_rows, last_rows = scenes[first], scenes[last]
        for key in sorted(first_rows.keys() & last_rows.keys()):
            a, b = first_rows[key], last_rows[key]
            if a["relative_burst_id"] != b["relative_burst_id"]:
                continue
            tank, pol = key
            fitted = a["status"] == b["status"] == PASS
            radar = float(a["gap_m"]) - float(b["gap_m"])
            for kind, mode in (("sentinel2_shadow", None), ("aerial_shadow", None), ("landsat_esri_b8", None),
                               ("landsat_original_b8", None), ("umbra", "local"), ("umbra", "scene"),
                               ("capella", "local"), ("capella", "scene"), ("gedi", None), ("icesat2", None)):
                x, y = (references.get((p["cdse_uuid"], tank, kind, mode)) for p in (a, b))
                if not x or not y or timestamp(x["reference_end_utc"]) >= timestamp(y["reference_start_utc"]):
                    continue
                ref = x["reference_gap_m"] - y["reference_gap_m"]
                low = x["reference_low_m"] - y["reference_high_m"] if (
                    x["reference_low_m"] is not None and y["reference_high_m"] is not None) else None
                high = x["reference_high_m"] - y["reference_low_m"] if (
                    x["reference_high_m"] is not None and y["reference_low_m"] is not None) else None
                yield dict(
                    candidate_id=tank, terminal_id=a["terminal_id"], polarization=pol, reference_kind=kind,
                    reference_registration=mode, earlier_radar_burst=a["cdse_uuid"], later_radar_burst=b["cdse_uuid"],
                    earlier_radar_utc=a["rim_zero_doppler_utc"], later_radar_utc=b["rim_zero_doppler_utc"],
                    earlier_radar_fit_status=a["status"], later_radar_fit_status=b["status"],
                    both_radar_fits_pass=fitted, radar_candidate_rise_m=radar if fitted else None,
                    earlier_reference_id=x["reference_id"], later_reference_id=y["reference_id"],
                    earlier_reference_utc=x["reference_start_utc"], later_reference_utc=y["reference_start_utc"],
                    earlier_separation_hours=x["maximum_separation_hours"], later_separation_hours=y["maximum_separation_hours"],
                    reference_rise_m=ref, reference_rise_low_m=low, reference_rise_high_m=high,
                    radar_minus_reference_rise_m=radar - ref if fitted else None,
                )


def weekly_passes(profiles, all_rows):
    ledger = defaultdict(list)
    for row in all_rows:
        ledger[row["radar_burst_uuid"], row["candidate_id"]].append(row)
    for week in read("ceda_s1_slc_roof_profiles_weekly"):
        if week["both_pass"] != "True":
            continue
        tank = week["candidate_id"]
        for side in ("previous", "current"):
            uuid = week[f"{side}_{week['polarization'].lower()}"]
            profile = profiles[uuid, tank]
            rows = ledger[uuid, tank]
            resolved = [r for r in rows if r["reference_usable"]]
            within = [r for r in resolved if r["maximum_separation_hours"] <= 48]
            names = {name for r in within for name in (
                json.loads(r["reference_id"]) if r["reference_kind"] == "sentinel2_shadow" else [r["reference_id"]])}
            yield dict(
                week_ending=week["week_ending"], candidate_id=tank, polarization=week["polarization"], side=side,
                radar_burst_uuid=uuid, radar_utc=profile["rim_zero_doppler_utc"], radar_gap_m=float(profile["gap_m"]),
                radar_pair_candidate_rise_m=float(week["candidate_rise_m"]), reference_candidates_72h=len(rows),
                usable_reference_candidates_48h=len(within),
                usable_reference_ids_48h=json.dumps(sorted(names)),
                closest_usable_reference_hours=min((r["maximum_separation_hours"] for r in resolved), default=None),
                both_source_versions_before_cutoff=week["all_versions_before_cutoff"] == "True",
            )


def main():
    source = read("ceda_s1_slc_roof_profiles")
    profiles = {(r["cdse_uuid"], r["candidate_id"]): r for r in source}
    if len(profiles) != len(source):
        raise ValueError("Repeated SLC profile identity")
    rows = list(comparisons(profiles))
    for row in rows:
        asset = row["reference_asset"]
        if asset and not (ROOT / asset).exists():
            raise FileNotFoundError(asset)
    nearest = best_references(rows)
    repeated = list(changes(profiles, nearest))
    weekly = list(weekly_passes(profiles, rows))
    joint_changes = [r for r in repeated if r["both_radar_fits_pass"]
                     and max(r["earlier_separation_hours"], r["later_separation_hours"]) <= 48]
    for name, records in (("pairs", rows), ("changes", repeated), ("weekly_passes", weekly)):
        export(f"{STEM}_{name}", records)
    summary = dict(
        radar_scenes=len({r["product_id"] for r in source}), profiles=len(profiles),
        comparison_rows=len(rows), comparisons_by_source=paired_summary(rows),
        repeated_reference_rows=len(repeated), repeated_reference_by_source=dict(Counter(r["reference_kind"] for r in repeated)),
        repeated_both_radar_fits_pass=sum(r["both_radar_fits_pass"] for r in repeated),
        repeated_48h_both_radar_fits_pass=len(joint_changes),
        repeated_48h_gap_change_error_m=describe([r["radar_minus_reference_rise_m"] for r in joint_changes]),
        weekly_pass_endpoints_with_reference_48h=sum(r["usable_reference_candidates_48h"] > 0 for r in weekly),
        weekly_pass_endpoints=len(weekly),
        scope="Retrospective comparisons on the same roof, retaining both measurement clocks. Reference fits may "
              "share geometry and the roof can move between acquisitions. No model selection, EIA or trading labels used.",
    )
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
