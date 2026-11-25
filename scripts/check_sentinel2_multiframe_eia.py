"""Compare a fixed, robust multi-capture roof trend with paired optical estimates."""
import json
import math
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from statistics import median

from audit_sentinel2_weekly_volume import releases
from build_registered_shadow_development import DATED, study_weeks
from build_shadow_timing_measurements import candidates, observation_key
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_development_check import ACTUAL, read_csv, scores
from build_weekly_shadow_panel import DERIVED, RESOLVED, export
from collect_public import write

PHASES = ("development", "validation")
PHYSICAL = ("previous_capture_pair_kbbl", "linear_boundary_pair_kbbl", "median_multiframe_kbbl")
METHODS = ("zero_change_kbbl", "last_published_change_kbbl", *PHYSICAL, *("last_plus_"+m for m in PHYSICAL))


def measurement_rows(phase):
    weeks = study_weeks(phase)
    _, groups, dated, wanted, _ = candidates(phase, weeks)
    fits = {observation_key(r): r for r in read_csv(f"shadow_timing_{phase}_observations")
            if r["geometry_mode"] == DATED and r["fit_status"] == RESOLVED}
    for row in read_csv(f"shadow_timing_{phase}_pairs"):
        if row["geometry_mode"] != DATED or row["timing_status"] != "both_boundaries_interpolated":
            continue
        week, tank = row["week_ending"], row["candidate_id"]
        if not dated[week, tank]:
            raise ValueError(f"A dated pair lacks engineering provenance for {week}/{tank}")
        start, stop, release, cutoff = weeks[week]
        before, after = (datetime.fromisoformat(row[f"{side}_captured_at_utc"]) for side in ("previous_before", "current_after"))
        by_pass = defaultdict(list)
        for options in groups[week, tank].values():
            for key, context, orbit in options:
                if key not in fits:
                    continue
                source = wanted[key]
                when = datetime.fromisoformat(source["captured_at_utc"])
                if before <= when <= after:
                    if max(datetime.fromisoformat(source[f]) for f in ("captured_at_utc", "archive_available_at_utc",
                           "geometry_available_at_utc")) >= datetime.fromisoformat(cutoff):
                        raise ValueError(f"A multi-frame input was unavailable for {week}/{tank}")
                    by_pass[orbit].append((-context, key, when))
        selected = sorted(min(v) for v in by_pass.values())
        points = sorted((when, float(fits[key]["estimated_gap_m"]), key) for _, key, when in selected)
        if len(points) < 3:
            continue
        if not points[0][0] <= start < stop < points[-1][0]:
            continue
        area = float(row["circular_area_m2"])
        if any(not math.isclose(area, math.pi*float(fits[key]["diameter_m"])**2/4) for _, _, key in points):
            raise ValueError(f"A multi-frame input uses another tank diameter for {week}/{tank}")
        slopes = [(old_gap-new_gap)/(later-earlier).total_seconds()
                  for (earlier, old_gap, _), (later, new_gap, _) in combinations(points, 2) if earlier < later]
        volume = median(slopes)*(stop-start).total_seconds()*area/BARREL_M3/1000
        yield dict(study_phase=phase, week_ending=week, release_date=release, candidate_id=tank,
            terminal_id=row["terminal_id"], cutoff_utc=cutoff, earliest_capture_utc=points[0][0].isoformat(),
            latest_capture_utc=points[-1][0].isoformat(), optical_overpasses=len(points),
            source_versions_json=json.dumps([k[1] for _, _, k in points]),
            capture_times_utc_json=json.dumps([t.isoformat() for t, _, _ in points]),
            observed_gaps_m_json=json.dumps([gap for _, gap, _ in points]), area_m2=area,
            previous_capture_pair_kbbl=float(row["pre_measurement_pair_visible_kbbl"]),
            linear_boundary_pair_kbbl=float(row["interpolated_visible_kbbl"]), median_multiframe_kbbl=volume)


def evaluation(phase, rows):
    by_week = defaultdict(list)
    for row in rows:
        by_week[row["week_ending"]].append(row)
    for target in releases(phase):
        week, release = target["week_ending"], target["release_date"]
        data = by_week[week]
        if not data or target["latest_week_known_before_release"] != target["previous_week_ending"]:
            continue
        prior = target["latest_weekly_change_known_before_release"]
        if (any(r["release_date"] != release for r in data) or prior is None
                or any(not target[k] or target[k] >= release for k in (
                    "latest_vintage_known_before_release", "prior_vintage_for_latest_known_change"))):
            raise ValueError(f"EIA and optics use different publication clocks for {week}")
        signals = {m: sum(r[m] for r in data) for m in PHYSICAL}
        yield dict(study_phase=phase, week_ending=week, release_date=release, matched_tanks=len(data),
            median_optical_overpasses=median(r["optical_overpasses"] for r in data),
            **{ACTUAL: target[ACTUAL]}, zero_change_kbbl=0., last_published_change_kbbl=prior, **signals,
            **{"last_plus_"+m: prior+delta for m, delta in signals.items()})


def main():
    report, observations, comparisons = {}, [], []
    for phase in PHASES:
        measured = list(measurement_rows(phase))
        panel = list(evaluation(phase, measured))
        observations.extend(measured)
        comparisons.extend(panel)
        report[phase] = dict(candidate_tank_weeks=len(measured), scored_tank_weeks=sum(r["matched_tanks"] for r in panel),
            comparable_weeks=len(panel), results={method: scores(panel, method) for method in METHODS})
    export("sentinel2_multiframe_tank_changes", observations)
    export("sentinel2_multiframe_eia_comparison", comparisons)
    summary = dict(phases=report, interpretation="Use the median rate across pairs of three or more resolved Sentinel-2 overpasses. "
        "Both reporting endpoints lie within the span of observed captures available by the original cutoff. "
        "All variants use identical tanks and weeks. The method has no fitted EIA coefficient and assumes a locally constant median rate. "
        "The 2023 check is retrospective. Holdout targets are excluded.")
    write(DERIVED / "sentinel2_multiframe_eia_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    for phase, result in report.items():
        print(phase, result["comparable_weeks"], {method: round(value["mean_abs_error_kbbl"], 1)
              for method, value in result["results"].items() if value["weeks"]}, flush=True)


if __name__ == "__main__":
    main()
