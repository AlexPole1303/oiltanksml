"""Inventory timestamped optical options around the two EIA measurement boundaries."""
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from statistics import median

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, WEEKLY_ELIGIBLE_INPUTS, endpoint_ctes, export, read_only
from collect_public import write
from collect_s2_l1c import parse_product

STEM = "shadow_timing_candidates"
PHASES = ("development", "validation", "holdout")
POST_MEASUREMENT = "post_measurement"
ROLES = ("previous", "current", POST_MEASUREMENT)
ELIGIBILITY = f"""
    SELECT * FROM ({WEEKLY_ELIGIBLE_INPUTS}) WHERE study_phase IN ('development','validation','holdout')
    UNION ALL
    SELECT w.week_ending,'{POST_MEASUREMENT}',w.study_phase,w.release_date,w.inputs_available_before_utc,
        w.window_ends_by_utc,w.inputs_available_before_utc,i.canonical_l2a_id,i.version_id,i.captured_at_utc,
        i.archive_available_at_utc,i.geometry_available_at_utc,
        CASE i.archive WHEN 'gcs' THEN 'gcs_exact_object_creation_times'
            WHEN 'aws' THEN 'aws_exact_objects_and_earth_search_catalog' END
    FROM eia_satellite_window w JOIN sentinel2_l1c_input_inventory i
      ON julianday(i.captured_at_utc)>julianday(w.window_ends_by_utc)
     AND julianday(i.captured_at_utc)<julianday(w.inputs_available_before_utc)
    WHERE w.endpoint='current' AND w.study_phase IN ('development','validation','holdout')
      AND i.input_status='local_exact_version_with_radiometry'
      AND julianday(i.archive_available_at_utc)<julianday(w.inputs_available_before_utc)
      AND julianday(i.geometry_available_at_utc)<julianday(w.inputs_available_before_utc)
      AND (SELECT count(*) FROM sentinel2_band_radiometry r WHERE r.level='L1C' AND r.version_id=i.version_id
           AND r.band IN ('B02','B03','B04','B08') AND r.reflectance_scale IS NOT NULL AND r.reflectance_offset IS NOT NULL
           AND r.nodata_dn IS NOT NULL AND r.saturated_dn IS NOT NULL)=4
"""


def clock_and_candidates():
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        clocks = {r["week_ending"]: dict(r) for r in db.execute("""
            SELECT c.week_ending,c.study_phase,c.release_date,c.inputs_available_before_utc,
                p.window_ends_by_utc AS previous_measurement_at_utc,c.window_ends_by_utc AS current_measurement_at_utc
            FROM eia_satellite_window p JOIN eia_satellite_window c USING(week_ending)
            WHERE p.endpoint='previous' AND c.endpoint='current'
              AND c.study_phase IN ('development','validation','holdout') ORDER BY c.week_ending
        """)}
        query = endpoint_ctes("AND t.documented_roof_type='EFR'", eligible_inputs=ELIGIBILITY)
        rows = [dict(r) for r in db.execute(query + "SELECT * FROM ranked ORDER BY week_ending,candidate_id,endpoint,rank")]
    return clocks, rows


def verify_and_group(rows, clocks):
    groups = defaultdict(lambda: defaultdict(list))
    keys = set()
    for row in rows:
        week, tank, endpoint = (row[field] for field in ("week_ending", "candidate_id", "endpoint"))
        key = week, tank, endpoint, row["canonical_l2a_id"], row["version_id"]
        if key in keys or endpoint not in ROLES:
            raise ValueError(f"Duplicate or unknown timing observation: {key}")
        keys.add(key)
        clock = clocks[week]
        if any(row[field] != clock[field] for field in ("study_phase", "release_date", "inputs_available_before_utc")):
            raise ValueError(f"Timing candidate uses a different release: {key}")
        cutoff, capture, archive, geometry = (datetime.fromisoformat(row[field]) for field in
            ("inputs_available_before_utc", "captured_at_utc", "archive_available_at_utc", "geometry_available_at_utc"))
        start, end = (datetime.fromisoformat(row[f"window_{side}_utc"]) for side in ("starts_after", "ends_by"))
        if not (start < capture <= end and max(capture, archive, geometry) < cutoff):
            raise ValueError(f"Observation breaches its timing gate: {key}")
        (sensor, day, orbit, _), _, _ = parse_product(row["version_id"])
        row["overpass_key"] = f"{sensor}_{day[:8]}_{orbit}"
        groups[week, tank][endpoint].append(row)
    for stem in ("weekly_shadow_endpoint_manifest", "weekly_shadow_terminal_endpoint_manifest"):
        for row in read_csv(stem):
            if row["roof_type"] == "EFR" and row["study_phase"] in PHASES and tuple(row[field] for field in (
                "week_ending", "candidate_id", "endpoint", "canonical_l2a_id", "version_id")) not in keys:
                raise ValueError(f"Timing catalog lost a frozen input for {row['week_ending']}/{row['candidate_id']}")
    return groups


def closest(options, target):
    return min(options, key=lambda r: (abs(datetime.fromisoformat(r["captured_at_utc"])-target),
                                       -r["context_valid_cells"], r["version_id"]), default=None)


def tank_windows(clocks, groups):
    for (week, tank), roles in sorted(groups.items()):
        clock = clocks[week]
        previous, current = (datetime.fromisoformat(clock[f"{boundary}_measurement_at_utc"])
                             for boundary in ("previous", "current"))
        before, inside, after = (roles[role] for role in ROLES)
        selections = {"previous_before": closest(before, previous), "previous_after": closest(inside, previous),
                      "current_before": closest(inside, current), "current_after": closest(after, current)}
        site = next(rows[0]["terminal_id"] for rows in roles.values() if rows)
        result = dict(clock, candidate_id=tank, terminal_id=site,
            **{f"{role}_overpasses": len({r["overpass_key"] for r in roles[role]}) for role in ROLES},
            frozen_style_pair_available=bool(before and inside),
            pair_available_with_post_measurement=bool(before and (inside or after)),
            new_pair_only_with_post_measurement=bool(before and not inside and after),
            both_boundaries_bracketed=bool(before and inside and after))
        for name, row in selections.items():
            target = previous if name.startswith("previous") else current
            gap = (datetime.fromisoformat(row["captured_at_utc"])-target).total_seconds()/3600 if row else None
            result.update({f"{name}_hours_from_measurement": gap})
            result.update({f"{name}_{field}": row[field] if row else None for field in (
                "version_id", "canonical_l2a_id", "captured_at_utc", "archive_available_at_utc", "geometry_available_at_utc")})
        yield result


def aggregate(rows):
    paired = [r for r in rows if r["frozen_style_pair_available"]]
    post = [r for r in rows if r["current_after_version_id"]]
    return dict(tanks_with_any_capture=len(rows), tanks_with_frozen_style_pairs=len(paired),
        tanks_with_pairs_allowing_post_measurement=sum(r["pair_available_with_post_measurement"] for r in rows),
        tanks_newly_paired_by_post_measurement=sum(r["new_pair_only_with_post_measurement"] for r in rows),
        tanks_with_both_boundaries_bracketed=sum(r["both_boundaries_bracketed"] for r in rows),
        tanks_with_post_measurement_capture=len(post),
        median_hours_last_capture_before_current_measurement=median(-r["current_before_hours_from_measurement"] for r in paired) if paired else None,
        median_hours_first_capture_after_current_measurement=median(r["current_after_hours_from_measurement"] for r in post) if post else None)


def main():
    clocks, candidates = clock_and_candidates()
    tanks = list(tank_windows(clocks, verify_and_group(candidates, clocks)))
    by_week = defaultdict(list)
    for row in tanks:
        by_week[row["week_ending"]].append(row)
    weekly = [dict(clock, **aggregate(by_week[week])) for week, clock in clocks.items()]
    for name, rows in (("observations", candidates), ("tanks", tanks), ("weeks", weekly)):
        export(f"{STEM}_{name}", rows)
    coverage = {}
    for phase in PHASES:
        weeks = [r for r in weekly if r["study_phase"] == phase]
        phase_tanks = [r for r in tanks if r["study_phase"] == phase]
        counts = Counter(r["endpoint"] for r in candidates if r["study_phase"] == phase)
        coverage[phase] = dict(reporting_weeks=len(weeks), candidate_observations_by_window=counts,
            weeks_with_frozen_style_pairs=sum(r["tanks_with_frozen_style_pairs"] > 0 for r in weeks),
            weeks_with_pairs_allowing_post_measurement=sum(r["tanks_with_pairs_allowing_post_measurement"] > 0 for r in weeks),
            weeks_with_both_boundaries_bracketed=sum(r["tanks_with_both_boundaries_bracketed"] > 0 for r in weeks),
            **aggregate(phase_tanks))
    report = dict(weeks=len(weekly), tank_weeks=len(tanks), source_observations=len(candidates), coverage=coverage,
        interpretation="The post-measurement interval ends at the conservative prerelease cutoff. Exact source imagery and "
        "geometry must both be available before that cutoff. The tables measure eligible capture coverage for documented "
        "EFR candidates. They contain no inferred heights, inventory outcomes or price data.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
