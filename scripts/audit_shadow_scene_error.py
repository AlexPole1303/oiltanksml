"""Measure correlated volume errors and test a fixed-roof adjustment on matched controls."""
import json
import math
from collections import defaultdict
from contextlib import closing
from statistics import fmean, median

from audit_shadow_precision import describe
from build_operator_development_inputs import one_per_key
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

FIXED_TYPES = {"IFR", "Domed EFR"}
RAW_CHANGE = "shared_registered_change_m"
ADJUSTED = "adjusted_change_m"
M3_PER_BBL = 0.158987294928


def corrections(pairs, sites):
    grouped = defaultdict(list)
    for row in pairs:
        if row["comparison"] == "fixed_roof" and row["study_phase"] != "development":
            continue
        grouped[row["comparison"], row["scene_pair_id"]].append(row)
    for (kind, pair_id), rows in sorted(grouped.items()):
        fixed = defaultdict(list)
        for row in rows:
            site = sites[row["candidate_id"]]
            if row["roof_type"] in FIXED_TYPES and site:
                fixed[site].append(float(row[RAW_CHANGE]))
        for row in rows:
            if (kind == "fixed_roof" and row["roof_type"] not in FIXED_TYPES
                    or kind == "tandem" and row["roof_type"] != "EFR"):
                continue
            site = sites[row["candidate_id"]]
            if not site:
                continue
            donors = {k: v for k, v in fixed.items() if k != site}
            donor_count = sum(map(len, donors.values()))
            if donor_count < 3 or len(donors) < 2:
                continue
            bias = median(median(values) for values in donors.values())
            value = float(row[RAW_CHANGE])
            yield dict(row, terminal_id=site, other_control_tanks=donor_count,
                other_control_terminals=len(donors), estimated_common_change_m=bias, **{RAW_CHANGE: value, ADJUSTED: value - bias})


def tandem_totals(rows, estimates):
    groups = defaultdict(list)
    for row in rows:
        if row["comparison"] != "tandem" or row["roof_type"] != "EFR" or row["baseline_both_resolved"] != "True":
            continue
        before, after = (estimates[row["candidate_id"], row[f"{side}_version"]] for side in ("earlier", "later"))
        if before["diameter_m"] != after["diameter_m"]:
            raise ValueError(f"Optical geometry differs across tandem captures for {row['candidate_id']}")
        area = math.pi * float(before["diameter_m"]) ** 2 / 4
        volume_gradient = area / M3_PER_BBL
        groups[row["scene_pair_id"]].append((row, volume_gradient))
    for pair, sample in sorted(groups.items()):
        rows, gradients = zip(*sample, strict=True)
        versions = {(r["earlier_version"], r["later_version"]) for r in rows}
        if len(versions) != 1 or len(rows) != len({r["candidate_id"] for r in rows}):
            raise ValueError(f"Expected one pair of captures and unique tanks for {pair}")
        volume = [float(r[RAW_CHANGE]) * g for r, g in zip(rows, gradients, strict=True)]
        raw = sum(volume)
        adjusted = sum(r[ADJUSTED] * g for r, g in zip(rows, gradients, strict=True))
        separate = sum(float(r["separate_change_m"]) * g for r, g in zip(rows, gradients, strict=True))
        unregistered = sum(float(r["shared_unregistered_change_m"]) * g for r, g in zip(rows, gradients, strict=True))
        independent_signs = math.sqrt(sum(v * v for v in volume))
        before, after = next(iter(versions))
        yield dict(scene_pair_id=pair, capture_day=rows[0]["week_or_control_day"], earlier_version=before,
            later_version=after, tanks=len(rows), nominal_volume_gradient_bbl_per_m=sum(gradients),
            separate_fit_sum_difference_bbl=separate, shared_unregistered_sum_difference_bbl=unregistered,
            observed_sum_difference_bbl=raw, adjusted_sum_difference_bbl=adjusted,
            random_sign_reference_rms_bbl=independent_signs,
            absolute_observed_to_random_sign_rms=abs(raw) / independent_signs if independent_signs else None)


def main():
    pairs = read_csv("shadow_registered_control_fit")
    estimates = one_per_key(read_csv("shadow_change_control_estimates"), "candidate_id", "version_id")
    with closing(read_only(SOURCE)) as db:
        sites = dict(db.execute("SELECT candidate_id,documented_terminal_id FROM tank_engineering_registry"))
    rows = list(corrections(pairs, sites))
    volume = list(tandem_totals(rows, estimates))
    effects = {}
    for kind in ("fixed_roof", "tandem"):
        sample = [r for r in rows if r["comparison"] == kind]
        effects[kind] = dict(tank_pairs=len(sample), scene_pairs=len({r["scene_pair_id"] for r in sample}),
            methods={f: dict(describe([r[f] for r in sample]), mean_absolute_m=fmean(abs(r[f]) for r in sample))
                     for f in (RAW_CHANGE, ADJUSTED)})
    export("shadow_scene_error_controls", rows)
    export("shadow_scene_error_volume", volume)
    volume_methods = {field: fmean(abs(row[field]) for row in volume) for field in (
        "separate_fit_sum_difference_bbl", "shared_unregistered_sum_difference_bbl",
        "observed_sum_difference_bbl", "adjusted_sum_difference_bbl")}
    report = dict(corrections=effects, tandem_volume=volume, tandem_mean_absolute_sum_difference_bbl=volume_methods,
        interpretation="The fixed-roof correction subtracts the median of other terminal medians, excluding "
        "all tanks at the target terminal. At least three reference roofs in at least two other terminals are required. "
        "The volume sums are geometric equivalents on matched open roofs in nearly simultaneous captures. "
        "The random-sign RMS holds individual error magnitudes fixed as a scale for comparison. It is not a confidence "
        "interval. The four tandem pairs are engineering controls. No inventory values or equity outcomes are read.")
    write(DERIVED / "shadow_scene_error_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
