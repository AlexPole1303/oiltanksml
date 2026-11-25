"""Join retrospective per-tank weather to existing optical engineering controls."""

import json
from collections import Counter
from datetime import datetime

import numpy as np

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_cushing_hrrr import SCENE_FIELDS, time, utc
from collect_public import ROOT, write

STEM = "shadow_weather_controls"
TABLES = ("shadow_change_control_estimates", "shadow_geometry_estimates")


def main():
    hrrr = json.loads((DERIVED / "cushing_hrrr_summary.json").read_bytes())
    with np.load(ROOT / hrrr["coordinates_file"], allow_pickle=False) as coord:
        hours = coord["time_epoch_utc"]
    by_tank = {r["candidate_id"]: int(r["hrrr_cell_index"]) for r in read("cushing_hrrr_tank_grid")}
    fields = {r["variable"]: np.load(ROOT / r["local_file"], mmap_mode="r", allow_pickle=False)
              for r in read("cushing_hrrr_fields") if r["variable"] in SCENE_FIELDS}
    if set(fields) != set(SCENE_FIELDS):
        raise ValueError("Missing weather fields in the pinned NOAA analysis")
    rows = []
    for table in TABLES:
        identities = set()
        for r in read(table):
            identity = r["observation_id"], r["candidate_id"], r["version_id"]
            if identity in identities:
                raise ValueError(f"Duplicate optical weather control {table}/{identity}")
            identities.add(identity)
            stamp = datetime.fromisoformat(r["optical_capture_at_utc"])
            t = time(r["optical_capture_at_utc"])
            index = (t - int(hours[0])) // 3600
            inside = 0 <= index < len(hours)
            tank = by_tank[r["candidate_id"]]
            conditions = {}
            for name, array in fields.items():
                value = float(array[index, tank]) if inside else None
                conditions[name] = value if value is not None and np.isfinite(value) else None
            clock = int(hours[index]) if inside else None
            rows.append({k: r[k] for k in ("observation_id", "candidate_id", "cohort_id", "roof_type", "version_id",
                         "optical_capture_at_utc", "fit_status")} |
                        dict(control_table=table, capture_day=stamp.date().isoformat(), hrrr_cell_index=tank,
                             model_valid_hour_utc=utc(clock) if inside else None,
                             model_age_seconds=stamp.timestamp()-clock if inside else None,
                             model_archive_snapshot_id=hrrr["snapshot_id"], **conditions))
    export(STEM, rows)
    report = dict(rows=len(rows), optical_capture_days=len({r["capture_day"] for r in rows}),
        model_hours=len({r["model_valid_hour_utc"] for r in rows if r["model_valid_hour_utc"]}),
        controls_by_table=Counter(r["control_table"] for r in rows),
        rows_with_cloud_and_shortwave=sum(r["total_cloud_cover_atmosphere"] is not None and
            r["downward_short_wave_radiation_flux_surface"] is not None for r in rows),
        observations_by_field=Counter({name: sum(r[name] is not None for r in rows) for name in fields}),
        interpretation="Retrospective physics and ML quality diagnostics. Every row joins the existing fit to the nearest "
        "native HRRR cell and the hour at or before acquisition. The row count includes repeated tanks and shared scene "
        "conditions. Group evaluation by acquisition date and terminal. This table does not establish archive availability "
        "before the image or EIA cutoff and contains no EIA target or market price.")
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
