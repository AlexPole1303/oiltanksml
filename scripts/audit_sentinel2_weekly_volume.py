"""Reconcile Sentinel-2 roof volumes and fixed weekly estimates with first-release EIA stocks."""
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime
from statistics import fmean, median

import numpy as np

from build_registered_shadow_development import DATED, study_weeks
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write

PHASES = ("development", "validation")
METHODS = ("previous_stock", "previous_change", "satellite_change", "satellite_overlap_change",
           "satellite_minimum_change", "previous_change_plus_satellite", "previous_change_plus_minimum_satellite")


def observations(phase, dates, geometry, source_rows=None, *, geometry_mode=DATED):
    seen = set()
    if source_rows is None:
        source_rows = read_csv(f"weekly_shadow_registered_{phase}_pairs")
    for row in source_rows:
        if row["geometry_mode"] != geometry_mode or row["registered_both_shadow_diagnostics_pass"] not in (True, "True"):
            continue
        week, candidate = row["week_ending"], row["candidate_id"]
        if (week, candidate) in seen:
            raise ValueError(f"Duplicate usable physical measurement for {phase}/{week}/{candidate}")
        seen.add((week, candidate))
        first, last, release, cutoff = dates[week]
        before, after = (datetime.fromisoformat(row[f"{side}_capture_at_utc"]) for side in ("earlier", "later"))
        available = max(datetime.fromisoformat(row[f"{side}_archive_available_at_utc"]) for side in ("earlier", "later"))
        if (release != row["release_date"] or cutoff != row["inputs_available_before_utc"]
                or not before <= first < after <= last or not after <= available < datetime.fromisoformat(cutoff)):
            raise ValueError(f"Ineligible capture time or archive availability for {week}/{candidate}")
        g = geometry[candidate]
        diameter = float(g["diameter_m"])
        shell = float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"])
        area = float(row["circular_area_m2"])
        kbbl_per_m = float(row["circular_bbl_per_m"])/1000
        if not math.isclose(area, math.pi*diameter**2/4, rel_tol=1e-9):
            raise ValueError(f"The registered pair uses a different dated diameter for {candidate}")
        earlier, later = (float(row[f"{side}_registered_gap_m"]) for side in ("earlier", "later"))
        lower, upper = (float(row[f"registered_loss_range_{bound}_m"]) for bound in ("min", "max"))
        change = earlier-later
        if not 0 <= earlier <= shell or not 0 <= later <= shell or not lower <= change <= upper:
            raise ValueError(f"Inconsistent physical roof gaps or loss range for {week}/{candidate}")
        minimum_change = max(lower, min(upper, 0.))
        overlap = (after-first)/(after-before)
        if not math.isclose(overlap, float(row["fraction_of_capture_inside_eia_interval"]), abs_tol=1e-9):
            raise ValueError(f"Overlap clock differs for {week}/{candidate}")
        yield dict(study_phase=phase, week_ending=week, release_date=release, candidate_id=candidate,
            terminal_id=row["terminal_id"], previous_measurement_at_utc=first.isoformat(), current_measurement_at_utc=last.isoformat(),
            earlier_capture_at_utc=before.isoformat(), later_capture_at_utc=after.isoformat(), archive_available_at_utc=available.isoformat(),
            cutoff_utc=cutoff, earlier_sentinel2_version=row["earlier_version"], later_sentinel2_version=row["later_version"],
            geometry_source=g["dataset"], geometry_file=g["source_directory"], diameter_m=diameter, shell_above_ground_m=shell,
            circular_area_m2=area, circular_kbbl_per_m=kbbl_per_m, earlier_gap_m=earlier, later_gap_m=later,
            earlier_cylinder_equivalent_kbbl=(shell-earlier)*kbbl_per_m, later_cylinder_equivalent_kbbl=(shell-later)*kbbl_per_m,
            capture_change_kbbl=change*kbbl_per_m, overlap_change_kbbl=change*overlap*kbbl_per_m,
            joint_loss_min_change_kbbl=lower*kbbl_per_m, joint_loss_max_change_kbbl=upper*kbbl_per_m,
            minimum_magnitude_change_kbbl=minimum_change*kbbl_per_m, overlap_fraction=overlap,
            fraction_of_eia_interval_observed=(after-first)/(last-first))


def releases(phase):
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        return [dict(r) for r in source.execute("""
            SELECT week_ending,release_date,cushing_kbbl,change_kbbl_as_known_at_release,previous_week_ending,
                latest_week_known_before_release,latest_kbbl_known_before_release,latest_vintage_known_before_release,
                latest_weekly_change_known_before_release,prior_vintage_for_latest_known_change,source_url
            FROM eia_inventory_release_panel WHERE study_phase=? ORDER BY week_ending
        """, (phase,))]


def comparison(phase, pairs, targets):
    grouped = defaultdict(list)
    for r in pairs:
        grouped[r["week_ending"]].append(r)
    for r in targets:
        week, release = r["week_ending"], r["release_date"]
        sample = grouped[week]
        if sample and any(s["release_date"] != release for s in sample):
            raise ValueError(f"EIA release and optical inputs disagree for {week}")
        baseline_valid = (r["latest_week_known_before_release"] == r["previous_week_ending"]
                          and r["latest_vintage_known_before_release"] and r["latest_vintage_known_before_release"] < release
                          and r["prior_vintage_for_latest_known_change"] and r["prior_vintage_for_latest_known_change"] < release)
        anchor = r["latest_kbbl_known_before_release"] if baseline_valid else None
        prior = r["latest_weekly_change_known_before_release"] if baseline_valid else None
        raw, overlap, minimum = (sum(s[field] for s in sample) if sample else None for field in
                                  ("capture_change_kbbl", "overlap_change_kbbl", "minimum_magnitude_change_kbbl"))
        changes = (0., prior, raw, overlap, minimum, None if None in (prior, raw) else prior+raw,
                   None if None in (prior, minimum) else prior+minimum)
        predictions = {name+"_predicted_stock_kbbl": anchor+delta if None not in (anchor, delta) else None
                       for name, delta in zip(METHODS, changes, strict=True)}
        errors = {name+"_stock_error_kbbl": predictions[name+"_predicted_stock_kbbl"]-r["cushing_kbbl"]
                  if predictions[name+"_predicted_stock_kbbl"] is not None else None for name in METHODS}
        yield dict(study_phase=phase, week_ending=week, release_date=release,
            first_release_cushing_kbbl=r["cushing_kbbl"], first_release_change_kbbl=r["change_kbbl_as_known_at_release"],
            previous_stock_known_before_issue_kbbl=anchor, previous_change_known_before_issue_kbbl=prior,
            previous_vintage_date=r["latest_vintage_known_before_release"], eia_source_url=r["source_url"],
            matched_tanks=len(sample), matched_circle_area_m2=sum(s["circular_area_m2"] for s in sample),
            median_week_interval_covered=median(s["fraction_of_eia_interval_observed"] for s in sample) if sample else None,
            earlier_visible_cylinder_kbbl=sum(s["earlier_cylinder_equivalent_kbbl"] for s in sample) if sample else None,
            later_visible_cylinder_kbbl=sum(s["later_cylinder_equivalent_kbbl"] for s in sample) if sample else None,
            satellite_change_kbbl=raw, satellite_overlap_change_kbbl=overlap, satellite_minimum_change_kbbl=minimum,
            **predictions, **errors)


def error_metrics(signed):
    absolute = list(map(abs, signed))
    return dict(weeks=len(signed), mean_absolute_kbbl=fmean(absolute), median_absolute_kbbl=median(absolute),
                p90_absolute_kbbl=float(np.percentile(absolute, 90)), mean_signed_kbbl=fmean(signed))


def main():
    geometry = legacy_geometry()
    all_pairs, all_weeks, results = [], [], {}
    for phase in PHASES:
        dates = study_weeks(phase)
        pairs = list(observations(phase, dates, geometry))
        panel = list(comparison(phase, pairs, releases(phase)))
        if {r["week_ending"] for r in panel} != dates.keys():
            raise ValueError(f"EIA and satellite reporting calendar differ for {phase}")
        common = [r for r in panel if all(r[m+"_stock_error_kbbl"] is not None for m in METHODS)]
        baseline = [r for r in panel if r["previous_change_stock_error_kbbl"] is not None]
        optical = [r for r in panel if r["matched_tanks"]]
        results[phase] = dict(reporting_weeks=len(panel), weeks_with_optical_measurements=sum(bool(r["matched_tanks"]) for r in panel),
            matched_tank_weeks=len(pairs), common_eia_comparison_weeks=len(common),
            typical_actual_change_mean_abs_kbbl=fmean(abs(r["first_release_change_kbbl"]) for r in common),
            median_tanks_per_matched_week=median(r["matched_tanks"] for r in common),
            median_fraction_of_reporting_week_observed=median(r["median_week_interval_covered"] for r in common),
            satellite_change_vs_first_issue_eia={f: error_metrics([r[f]-r["first_release_change_kbbl"] for r in optical])
                for f in ("satellite_change_kbbl", "satellite_overlap_change_kbbl", "satellite_minimum_change_kbbl")},
            all_anchored_week_scores={f: error_metrics([r[f+"_stock_error_kbbl"] for r in baseline]) for f in METHODS[:2]},
            common_week_scores={f: error_metrics([r[f+"_stock_error_kbbl"] for r in common]) for f in METHODS})
        all_pairs.extend(pairs)
        all_weeks.extend(panel)
    export("sentinel2_weekly_tank_volumes", all_pairs)
    export("sentinel2_weekly_volume_eia_comparison", all_weeks)
    report = dict(phases=results, interpretation="The model uses Sentinel-2 with dated static tank geometry. "
        "Optical volumes are geometric cylinder equivalents above the LiDAR ground reference on observed tanks. "
        "Full-hub estimates add changes to the previous stock available before the current issue. "
        "The minimum-magnitude change uses the closest point to zero in the shared-centre descriptive loss range. "
        "All compared rules use the same weeks and no fitted EIA coefficients. The 2023 check is retrospective. "
        "The 2024–2026 holdout EIA outcomes are outside this report.")
    write(DERIVED / "sentinel2_weekly_volume_eia_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
