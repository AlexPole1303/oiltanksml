"""Summarize time gaps and height agreement at equal date-specific selection sizes."""

import argparse
from pathlib import Path

import pandas as pd
from fit_cushing_inventory_network import DERIVED, MODEL_DIR
from train_shadow_quality import best_at_count


def summarize(data):
    result = []
    for day, cohort in data.groupby("optical_day", sort=True):
        count = int(cohort.original_pass.sum())
        for name, score in (("ML", "ml_score"), ("physics", "physics_rank")):
            selected = pd.DataFrame(
                best_at_count(cohort.to_dict("records"), score, count)
            )
            errors = selected.absolute_difference_m
            hours = selected.sensor_gap_hours
            result.append(
                {
                    "optical_day": day,
                    "method": name,
                    "eligible": len(cohort),
                    "selected": len(selected),
                    "within_2_m": int((errors <= 2).sum()),
                    "median_abs_m": errors.median(),
                    "p90_abs_m": errors.quantile(0.9),
                    "min_gap_h": hours.min(),
                    "median_gap_h": hours.median(),
                    "max_gap_h": hours.max(),
                }
            )
    return pd.DataFrame(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=MODEL_DIR / "lidar_pairs.csv")
    parser.add_argument("--export-inputs", action="store_true")
    args = parser.parse_args()
    fields = [
        "optical_day",
        "candidate_id",
        "version_id",
        "sensor_gap_hours",
        "absolute_difference_m",
        "original_pass",
        "physics_rank",
        "ml_score",
    ]
    if args.export_inputs:
        pd.read_csv(
            DERIVED / "shadow_ml_repeatability_lidar.csv",
            dtype={"candidate_id": str, "version_id": str},
        )[fields].to_csv(args.input, index=False)
    data = pd.read_csv(args.input, dtype={"candidate_id": str, "version_id": str})
    if (
        data[fields].isna().any().any()
        or data.duplicated(["optical_day", "candidate_id", "version_id"]).any()
    ):
        raise ValueError("Incomplete or duplicate optical/lidar observations")
    summarize(data).to_csv(MODEL_DIR / "lidar_control.csv", index=False)
