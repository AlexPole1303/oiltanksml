"""Compare original and registered optical sums on identical development weeks and tanks."""
import json
from collections import defaultdict

from build_operator_development_inputs import one_per_key
from build_registered_shadow_development import STEM
from build_weekly_shadow_development_check import ACTUAL, BASELINES, numeric, read_csv, scores
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write

MEASUREMENTS = ("both_baseline_visible_kbbl", "both_registered_visible_kbbl",
                "both_baseline_overlap_kbbl", "both_registered_overlap_kbbl")
ADDITIONS = tuple("last_plus_" + field for field in MEASUREMENTS)


def comparison_rows():
    public = one_per_key(read_csv("weekly_shadow_development_check"), "week_ending")
    for raw in read_csv(f"{STEM}_weeks"):
        week, issue = raw["week_ending"], raw["release_date"]
        baseline = public[week,]
        if issue != baseline["release_date"] or any(
            release >= issue for release in baseline["seasonal_source_releases"].split("|") if release
        ):
            raise ValueError(f"The public baselines have a different or later release for {week}")
        row = {field: numeric(baseline[field]) for field in (*BASELINES, ACTUAL)}
        row.update({field: numeric(raw[field]) for field in MEASUREMENTS})
        latest = row["last_published_change_kbbl"]
        row.update({"last_plus_"+field: latest+row[field] if latest is not None and row[field] is not None else None
                    for field in MEASUREMENTS})
        yield dict(week_ending=week, release_date=issue, geometry_mode=raw["geometry_mode"],
                   resolved_identical_tank_pairs=int(raw["both_resolved_pairs"]),
                   common_resolved_circle_area_m2=float(raw["both_resolved_area_m2"]),
                   common_median_eia_interval_fraction=numeric(raw["both_median_eia_interval_fraction"]), **row)


def main():
    rows = list(comparison_rows())
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["geometry_mode"]].append(row)
    summaries = {}
    for mode, panel in sorted(grouped.items()):
        public = [r for r in panel if all(r[k] is not None for k in (*BASELINES, ACTUAL))]
        matched = [r for r in public if all(r[k] is not None for k in MEASUREMENTS)]
        if any((r["resolved_identical_tank_pairs"] > 0) != all(r[k] is not None for k in MEASUREMENTS) for r in panel):
            raise ValueError(f"The optical sums must be available together for {mode}")
        summaries[mode] = dict(all_week_scores={field: scores(public, field) for field in BASELINES},
            matched_week_scores={field: scores(matched, field) for field in (*BASELINES, *MEASUREMENTS, *ADDITIONS)})
    export(f"{STEM}_eia_comparison", rows)
    report = dict(study_phase="development", geometry=summaries, interpretation="First-release EIA values. "
        "The optical comparisons use the same weeks and the same tanks under both methods. Visible cylinder-equivalent "
        "sums and the additions to the last published change use fixed unit scale. Overlap weighting assumes uniform "
        "movement between captures. The analysis is descriptive development evidence.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_eia_summary.json", payload.encode())
    for mode, report in summaries.items():
        fields = report["matched_week_scores"]
        print(mode, {name: (value["weeks"], value.get("mean_abs_error_kbbl")) for name, value in fields.items()}, flush=True)


if __name__ == "__main__":
    main()
