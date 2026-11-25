"""Build clock-gated weather inputs for the frozen Cushing release calendar."""

import argparse
import json
from collections import Counter
from contextlib import closing
from datetime import datetime

import numpy as np

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_cushing_hrrr_operational import STEM, WANTED
from collect_public import ROOT, write

OUTPUT = "cushing_hrrr_holdout_features"
ROLES = ("previous", "current", "prior_cutoff_hour")
VARIABLES = tuple(name for fields in WANTED.values() for name, _ in fields.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="store_true", help="build development and validation features")
    training = parser.parse_args().training
    stem = f"{STEM}_training" if training else STEM
    output = "cushing_hrrr_training_features" if training else OUTPUT
    source = json.loads((DERIVED / f"{stem}_summary.json").read_bytes())
    with np.load(ROOT / source["native_cell_values_file"], allow_pickle=False) as saved:
        values = saved["values"]
    cells = sorted({int(r["hrrr_cell_index"]) for r in read("cushing_hrrr_tank_grid")})
    metadata = read(f"{stem}_messages")
    if len(metadata) != len(values) or len(cells) != 17:
        raise ValueError("Different NOAA message inventory or nearest-tank cells")
    by_week = {}
    for r in read(f"{stem}_links"):
        kind = r["reference_kind"]
        if kind not in ("eia_endpoint", "eia_cutoff_prior_hour"):
            continue
        role = r["endpoint"] if kind == "eia_endpoint" else "prior_cutoff_hour"
        key = (r["week_ending"], role)
        if key in by_week:
            raise ValueError(f"Duplicate NOAA weather endpoint {key}")
        by_week[key] = r
    rows = []
    coverage = {}
    if training:
        with closing(read_only(SOURCE)) as db:
            releases = [dict(study_phase=phase, week_ending=week, release_date=date, forecast_cutoff_utc=cutoff)
                        for phase, week, date, cutoff in db.execute("""
                SELECT study_phase,week_ending,release_date,inputs_available_before_utc
                FROM eia_satellite_window WHERE endpoint='current' AND study_phase IN ('development','validation')
            """)]
    else:
        releases = read("cushing_holdout_release_index")
    for release in sorted(releases, key=lambda r: r["week_ending"]):
        week = release["week_ending"]
        cutoff = datetime.fromisoformat(release["forecast_cutoff_utc"])
        item = {k: release[k] for k in ("study_phase", "week_ending", "release_date", "forecast_cutoff_utc")}
        phase_coverage = coverage.setdefault(item["study_phase"], {role: Counter() for role in ROLES})
        item["nearest_tank_cells"] = len(cells)
        for role in ROLES:
            link = by_week.pop((week, role))
            if datetime.fromisoformat(link["forecast_cutoff_utc"]) != cutoff:
                raise ValueError(f"Different forecast cutoff for {week}/{role}")
            item[f"{role}_model_valid_hour_utc"] = link["model_valid_hour_utc"]
            available = []
            for field in VARIABLES:
                index = link[f"{field}_array_row"]
                known = False
                if index:
                    meta = metadata[int(index)]
                    if (meta["variable"], meta["model_valid_hour_utc"]) != (field, link["model_valid_hour_utc"]):
                        raise ValueError(f"Different native NOAA message for {week}/{role}/{field}")
                    known = datetime.fromisoformat(meta["archive_last_modified_utc"]) < cutoff
                if link[f"{field}_precutoff"] != str(known):
                    raise ValueError(f"Different archive cutoff status for {week}/{role}/{field}")
                item[f"{role}_{field}_median"] = float(np.median(values[int(index), cells])) if known else None
                available.append(known)
                phase_coverage[role][field] += known
            item[f"{role}_all_four_before_cutoff"] = all(available)
        item["both_endpoints_all_four_before_cutoff"] = all(item[f"{role}_all_four_before_cutoff"] for role in ROLES[:2])
        for field in VARIABLES:
            current, previous = (item[f"{role}_{field}_median"] for role in ("current", "previous"))
            item[f"delta_{field}_median"] = current - previous if current is not None and previous is not None else None
        rows.append(item)
    if by_week or len(rows) != source["endpoint_weeks" if training else "holdout_endpoint_weeks"]:
        raise ValueError("NOAA weather inventory differs from the release calendar")
    export(output, rows)
    report = {
        "release_weeks": len(rows),
        "nearest_tank_cells": len(cells),
        "weeks_both_endpoints_complete": sum(r["both_endpoints_all_four_before_cutoff"] for r in rows),
        "weeks_prior_cutoff_hour_complete": sum(r["prior_cutoff_hour_all_four_before_cutoff"] for r in rows),
        "prior_cutoff_hour_incomplete_weeks": [r["week_ending"] for r in rows if not r["prior_cutoff_hour_all_four_before_cutoff"]],
        "coverage_by_phase_role_and_field": coverage,
        "endpoint_weeks_complete_by_phase": {phase: sum(r["study_phase"] == phase and
            r["both_endpoints_all_four_before_cutoff"] for r in rows) for phase in coverage},
        "aggregation": "Unweighted median of the 17 distinct native NOAA grid cells nearest to the 454 catalogued tank candidates. Deltas are between endpoint medians.",
        "timing": "Values are emitted only if the inspected NOAA archive object's Last-Modified is strictly before the release-specific cutoff. Missing or late values are empty. No EIA outcome or market price is used.",
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    write(DERIVED / f"{output}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
