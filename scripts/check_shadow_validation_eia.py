"""Score fixed optical measurements against the last published EIA change in 2023."""
import json
from datetime import date
from statistics import fmean, median

from build_operator_development_inputs import one_per_key
from build_weekly_shadow_development_check import ACTUAL, numeric, read_csv, scores
from build_weekly_shadow_panel import DERIVED
from collect_public import write

PRIOR = "last_published_change_kbbl"
COMPARISONS = {
    "standard_capture_pair": {
        "observed_change_kbbl": "dated_2011_resolved_visible_kbbl",
    },
    "same_tanks_before_and_after_registration": {
        "original_change_kbbl": "dated_2011_registration_both_baseline_visible_kbbl",
        "registered_change_kbbl": "dated_2011_registration_both_registered_visible_kbbl",
    },
    "same_tanks_before_and_after_time_alignment": {
        "capture_to_capture_change_kbbl": "dated_2011_timing_matched_pre_measurement_visible_kbbl",
        "interpolated_change_kbbl": "dated_2011_timing_interpolated_visible_kbbl",
    },
}


def direction(value):
    return (value > 0) - (value < 0)


def actual_movement(sample):
    magnitudes = [abs(row[ACTUAL]) for row in sample]
    return dict(mean_abs_change_kbbl=fmean(magnitudes) if magnitudes else None,
                median_abs_change_kbbl=median(magnitudes) if magnitudes else None)


def comparison(sample, field):
    hits = [direction(r[field]) == direction(r[ACTUAL]) for r in sample]
    prior_hits = [direction(r[PRIOR]) == direction(r[ACTUAL]) for r in sample]
    return dict(scores(sample, field), correct_direction=sum(hits),
        zero_predictions=sum(r[field] == 0 for r in sample),
        only_this_rule_correct=sum(hit and not prior for hit, prior in zip(hits, prior_hits, strict=True)),
        only_last_eia_correct=sum(prior and not hit for hit, prior in zip(hits, prior_hits, strict=True)))


def main():
    inputs = one_per_key(read_csv("study_validation_eia_weeks"), "week_ending")
    targets = one_per_key(read_csv("study_validation_eia_targets"), "week_ending")
    if inputs.keys() != targets.keys() or len(inputs) != 52:
        raise ValueError("Expected matching inputs and first-release targets for 52 reporting weeks")

    panel = []
    for (week,), raw in sorted(inputs.items()):
        target = targets[week,]
        issue = raw["release_date"]
        if (date.fromisoformat(week).year != 2023 or raw["study_phase"] != "validation"
                or issue != target["release_date"] or raw["source_url"] != target["source_url"]
                or not raw["latest_vintage_known_before_release"]
                or raw["latest_vintage_known_before_release"] >= issue
                or not raw["prior_vintage_for_latest_known_change"]
                or raw["prior_vintage_for_latest_known_change"] >= issue):
            raise ValueError(f"Invalid reporting phase or publication clock for {week}")
        prior = numeric(raw["latest_weekly_change_known_before_release"])
        actual = numeric(target[ACTUAL])
        if prior is None or actual is None:
            raise ValueError(f"Missing EIA change for {week}")
        row = {"week_ending": week, "release_date": issue, PRIOR: prior, ACTUAL: actual}
        for fields in COMPARISONS.values():
            for field, source in fields.items():
                value = numeric(raw[source])
                row[field] = value
                row["last_eia_plus_" + field] = prior + value if value is not None else None
        panel.append(row)

    results = {}
    for name, fields in COMPARISONS.items():
        sample = [r for r in panel if r[next(iter(fields))] is not None]
        if any(any(r[f] is None for f in fields) != all(r[f] is None for f in fields) for r in panel):
            raise ValueError(f"Missing one of the matched optical sums for {name}")
        rules = (PRIOR, *fields, *("last_eia_plus_" + f for f in fields))
        results[name] = dict(reporting_weeks=len(sample),
            unique_release_dates=len({r["release_date"] for r in sample}),
            week_endings=[r["week_ending"] for r in sample],
            eia_actual_movement=actual_movement(sample),
            results={f: comparison(sample, f) for f in rules})

    report = dict(study_phase="validation", reporting_year=2023, geometry="dated_2011_lidar",
        reporting_weeks=len(panel), unique_release_dates=len({r["release_date"] for r in panel}),
        eia_actual_movement_all_weeks=actual_movement(panel),
        last_eia_all_weeks=comparison(panel, PRIOR), comparisons=results,
        interpretation="First-issue EIA changes and optical sums prepared under the historical input cutoff. "
            "Each comparison uses identical reporting weeks and the paired optical variants also use identical tanks. "
            "Additive rules use fixed unit scale with no fit to the targets. Missing optical weeks are excluded from "
            "conditional scores. These are descriptive validation scores. The sealed test and market returns are excluded.")
    write(DERIVED / "study_validation_eia_direction_check.json", (json.dumps(report, indent=2) + "\n").encode())
    for name, result in results.items():
        print(name, result["reporting_weeks"],
              {field: (m["correct_direction"], round(m["mean_abs_error_kbbl"], 1))
               for field, m in result["results"].items()})


if __name__ == "__main__":
    main()
