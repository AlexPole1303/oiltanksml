"""Extract dated WTI curve marks, archived regional refinery reports and local NOAA weather for the API calibration."""

import json

import numpy as np
import pandas as pd
from collect_public import ROOT, write
from fit_cushing_inventory_network import MODEL_DIR

DERIVED = ROOT / "data/derived"
OUTPUT = MODEL_DIR / "balance_inputs.csv"
CURVE = {
    "slope_usd_bbl_month": "slope",
    "slope_change_5_sessions_usd_bbl_month": "slope_change",
    "bend_usd_bbl": "bend",
    "bend_change_5_sessions_usd_bbl": "bend_change",
}
REGION = (
    "midwest_runs",
    "midwest_runs_change",
    "gulf_runs_change",
    "ex_cushing",
    "national_prior",
    "us_exports_change",
)
GROUPS = {
    "curve_time": tuple(CURVE.values()),
    "weather_time": ("temp_change",),
    "regional_time": REGION,
    "optical_time": ("optical_move", "optical_coverage", "optical_measured"),
}


def stamped_values(row, when, week):
    result = {}
    if row is not None and str(row["week_ending"]) != week:
        raise ValueError("Balance measurements refer to a different week")
    for time, cols in GROUPS.items():
        available = False
        if row is not None and pd.notna(row[time]) and row[time]:
            ts = pd.Timestamp(row[time])
            if ts.tzinfo is None:
                raise ValueError("Balance measurement lacks a timezone")
            available = ts < when
        for col in cols:
            result[col] = float(row[col]) if available and pd.notna(row[col]) else np.nan
    return result


def regional_changes():
    raw = pd.read_csv(DERIVED / "eia_control_vintage.csv")
    metrics = [
        "padd2_crude_refiner_inputs",
        "padd3_crude_refiner_inputs",
        "padd2_commercial_crude",
        "cushing_crude",
        "us_commercial_crude",
        "us_crude_exports",
    ]
    selected = raw[raw.metric.isin(metrics)]
    if selected.duplicated(["release_date", "report_date", "metric"]).any():
        raise ValueError("Duplicate EIA vintage")
    rows = []
    for day, group in selected.groupby("release_date", sort=True):
        pivot = group.pivot(index="report_date", columns="metric", values="value").sort_index()
        if len(pivot) < 2 or set(metrics) - set(pivot.columns):
            continue
        week = pivot.index[-1]
        if pd.Timestamp(week) - pd.Timestamp(pivot.index[-2]) != pd.Timedelta(days=7):
            continue
        current = pivot.iloc[-1]
        delta = (current - pivot.iloc[-2]) / 1000.0
        # Conservative project-wide label clock, the next midnight in New York.
        when = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
        rows.append(
            {
                "regional_week": week,
                "regional_time": when,
                "midwest_runs": current["padd2_crude_refiner_inputs"] / 1000.0,
                "midwest_runs_change": delta["padd2_crude_refiner_inputs"],
                "gulf_runs_change": delta["padd3_crude_refiner_inputs"],
                "ex_cushing": delta["padd2_commercial_crude"] - delta["cushing_crude"],
                "national_prior": delta["us_commercial_crude"],
                "us_exports_change": delta["us_crude_exports"],
            }
        )
    return pd.DataFrame(rows)


def prepare():
    panel = pd.read_csv(
        MODEL_DIR / "weekly_inputs.csv",
        usecols=["release_date", "week_ending", "cutoff", "raw", "coverage", "measured"],
    )
    example = json.loads((MODEL_DIR / "nov18_inputs.json").read_text())
    panel.loc[len(panel)] = {
        "release_date": example["release_date"],
        "week_ending": example["week_ending"],
        "cutoff": example["deadline_utc"],
    }
    curve = pd.read_csv(DERIVED / "cushing_cl_curve_shape_forecasts.csv")
    curve = curve[curve.channel == "market"].set_index("release_date")
    weather = pd.concat(
        [pd.read_csv(DERIVED / f"cushing_hrrr_{split}_features.csv") for split in ("training", "holdout")]
    ).set_index(["release_date", "week_ending"], drop=False)
    if not curve.index.is_unique or not weather.index.is_unique:
        raise ValueError("Duplicate source issue")
    region = regional_changes()
    result = []
    for r in panel.itertuples():
        when = pd.Timestamp(r.cutoff)
        row = {"release_date": r.release_date, "week_ending": r.week_ending}
        if pd.notna(r.raw) and r.measured:
            row.update(
                optical_time=(when - pd.Timedelta(nanoseconds=1)).isoformat(),
                optical_move=r.raw,
                optical_coverage=r.coverage,
                optical_measured=r.measured,
            )
        if r.release_date in curve.index:
            v = curve.loc[r.release_date]
            ts = pd.Timestamp(v.latest_cl_mark_utc)
            if v.week_ending != r.week_ending or not ts < when:
                raise ValueError("Late curve or wrong reporting week")
            row.update(curve_time=ts.isoformat(), **{b: v[a] for a, b in CURVE.items()})
        if (r.release_date, r.week_ending) in weather.index:
            v = weather.loc[(r.release_date, r.week_ending)]
            if v.week_ending != r.week_ending or pd.Timestamp(v.forecast_cutoff_utc) != when:
                raise ValueError("Wrong weather cutoff")
            if v.both_endpoints_all_four_before_cutoff:
                if (
                    not pd.Timestamp(v.previous_model_valid_hour_utc)
                    < pd.Timestamp(v.current_model_valid_hour_utc)
                    < when
                ):
                    raise ValueError("Weather endpoint out of order")
                # Only a strict upper bound on public availability is available for this derived pair.
                row.update(
                    weather_time=(when - pd.Timedelta(nanoseconds=1)).isoformat(),
                    temp_change=v.delta_temperature_2m_c_median,
                )
        history = region[(region.regional_time < when) & (region.regional_week < r.week_ending)]
        if len(history):
            v = history.iloc[-1]
            if (pd.Timestamp(r.week_ending) - pd.Timestamp(v.regional_week)).days <= 14:
                row.update(v.to_dict())
                row["regional_time"] = v.regional_time.isoformat()
        result.append(row)
    write(OUTPUT, pd.DataFrame(result).to_csv(index=False).encode())


if __name__ == "__main__":
    prepare()
