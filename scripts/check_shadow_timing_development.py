"""Compare interpolated roof changes and capture-to-capture changes on identical tanks."""
import json
from collections import defaultdict
from statistics import median

from build_operator_development_inputs import one_per_key
from build_weekly_shadow_development_check import ACTUAL, BASELINES, numeric, read_csv, scores
from build_weekly_shadow_panel import DERIVED, export
from collect_public import write

STEM = "shadow_timing_development"
VISIBLE = ("matched_pre_measurement_visible_kbbl", "interpolated_visible_kbbl")
ADDITIONS = tuple("last_plus_"+field for field in VISIBLE)


def rows():
    public = one_per_key(read_csv("weekly_shadow_development_check"), "week_ending")
    for row in read_csv(f"{STEM}_weeks"):
        week, release, mode = (row[field] for field in ("week_ending", "release_date", "geometry_mode"))
        baseline = public[week,]
        if baseline["release_date"] != release or any(d >= release for d in baseline["seasonal_source_releases"].split("|") if d):
            raise ValueError(f"Published baseline clock differs for {week}")
        count = int(row["bracketed_pairs"])
        predictors = {key: numeric(baseline[key]) for key in (*BASELINES, ACTUAL)}
        for signal in VISIBLE:
            value = numeric(row[signal])
            last = predictors["last_published_change_kbbl"]
            if (value is None) != (count == 0):
                raise ValueError(f"Mismatched timing comparison sample for {week}/{mode}")
            predictors[signal] = value
            predictors["last_plus_"+signal] = value+last if value is not None and last is not None else None
        yield dict(week_ending=week, release_date=release, geometry_mode=mode, bracketed_pairs=count,
            bracketed_circle_area_m2=numeric(row["bracketed_circle_area_m2"]),
            bracketed_median_previous_hours=numeric(row["bracketed_median_previous_hours"]),
            bracketed_median_current_hours=numeric(row["bracketed_median_current_hours"]), **predictors)


def main():
    panel = list(rows())
    groups = defaultdict(list)
    for row in panel:
        groups[row["geometry_mode"]].append(row)
    results = {}
    for mode, series in groups.items():
        sample = [r for r in series if r["bracketed_pairs"] and all(r[f] is not None for f in (*BASELINES, ACTUAL))]
        results[mode] = dict(matched_weeks=len(sample),
            matched_tank_weeks=sum(r["bracketed_pairs"] for r in sample),
            median_bracketed_tanks_per_week=median(r["bracketed_pairs"] for r in sample) if sample else None,
            metrics={field: scores(sample, field) for field in (*BASELINES, *VISIBLE, *ADDITIONS)})
    export(f"{STEM}_eia_comparison", panel)
    report = dict(study_phase="development", geometry=results, interpretation="Both optical sums are computed "
        "on the same resolved EFR tanks in the same reporting weeks. Linear interpolation uses actual bracketed "
        "captures available before the report. The additive baselines use fixed unit scale. First-issue EIA data are "
        "used only for development comparison.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_eia_summary.json", payload.encode())
    for mode, output in results.items():
        print(mode, {key: (val["weeks"], val.get("mean_abs_error_kbbl")) for key, val in output["metrics"].items()}, flush=True)


if __name__ == "__main__":
    main()
