"""Compare development-only optical changes and historical EIA reference rules."""
import csv
import json
import math
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime
from statistics import fmean, median

from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only, write

KNOWN_GEOMETRY = """
WITH geometry AS MATERIALIZED (
    SELECT t.candidate_id,coalesce(l.diameter_m,c.diameter_m,t.footprint_diameter_m) AS diameter_m
    FROM tank_engineering_registry t LEFT JOIN tank_lidar_outline l USING(candidate_id)
    LEFT JOIN enbridge_main_covered_roof_outline c USING(candidate_id)
    WHERE t.documented_roof_type='EFR' AND coalesce(l.diameter_m,c.diameter_m,t.footprint_diameter_m)>0
), evidence AS MATERIALIZED (
    SELECT w.week_ending,g.*,(SELECT max(a.observed_date) FROM tank_aerial_visual_observation a
        WHERE a.candidate_id=g.candidate_id AND a.observed_date<substr(w.window_starts_after_utc,1,10)) AS aerial_date
    FROM eia_satellite_window w CROSS JOIN geometry g
    WHERE w.endpoint='previous' AND w.study_phase='development'
)
SELECT e.week_ending,e.candidate_id,e.diameter_m,e.aerial_date FROM evidence e
WHERE e.aerial_date IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM tank_aerial_visual_observation a WHERE a.candidate_id=e.candidate_id
        AND a.observed_date=e.aerial_date AND a.structure_observation IS NOT 'aboveground_tank_seen'
)
ORDER BY e.week_ending,e.candidate_id
"""
WARMUP_HISTORY = """
WITH first_issues AS (
    SELECT *,row_number() OVER (PARTITION BY week_ending ORDER BY release_date,reported_as) AS issue_number
    FROM eia_cushing_as_published WHERE week_ending<'2017-01-01'
)
SELECT i.week_ending,i.release_date,i.cushing_kbbl-(
    SELECT p.cushing_kbbl FROM eia_cushing_as_published p
    WHERE p.week_ending=date(i.week_ending,'-7 days') AND p.release_date<=i.release_date
    ORDER BY p.release_date DESC,p.reported_as DESC LIMIT 1
) AS change_kbbl
FROM first_issues i WHERE i.issue_number=1 ORDER BY i.week_ending
"""
SEASONAL_FIELD = "seasonal_median_kbbl"
BASELINES = ("zero_change_kbbl", "last_published_change_kbbl", SEASONAL_FIELD)
OPTICAL = ("visible_fitted_change_kbbl", "visible_resolved_change_kbbl")
ACTUAL = "change_kbbl_as_known_at_release"


def read_csv(stem):
    with (DERIVED/f"{stem}.csv").open(newline="",encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value):
    return float(value) if value not in (None, "") else None


def coverage(db, pairs):
    areas = defaultdict(dict)
    for week,candidate,diameter,_ in db.execute(KNOWN_GEOMETRY):
        areas[week][candidate] = math.pi*diameter**2/4
    paired = defaultdict(list)
    for pair in pairs:
        if pair["roof_type"]=="EFR":
            paired[pair["week_ending"]].append(pair)
    windows = defaultdict(dict)
    for week,endpoint,ends in db.execute("""SELECT week_ending,endpoint,window_ends_by_utc
        FROM eia_satellite_window WHERE study_phase='development'"""):
        windows[week][endpoint] = datetime.fromisoformat(ends)
    result = {}
    for week,reference in areas.items():
        known = sum(reference.values())
        intersection = [p for p in paired[week] if p["candidate_id"] in reference]
        in_pairs = sum(reference[p["candidate_id"]] for p in intersection)
        in_resolved = sum(reference[p["candidate_id"]] for p in intersection
                          if p["both_shadow_diagnostics_pass"]=="True" and p["optical_level_change_m"]!="")
        earlier = [datetime.fromisoformat(p["earlier_capture_at_utc"]) for p in paired[week]]
        later = [datetime.fromisoformat(p["later_capture_at_utc"]) for p in paired[week]]
        gap_hours = [(b-a).total_seconds()/3600 for a,b in zip(earlier,later,strict=True)]
        result[week] = dict(reference_efr_tanks=len(reference),reference_efr_area_m2=known,
            paired_in_reference_tanks=len(intersection),paired_outside_reference_tanks=len(paired[week])-len(intersection),
            paired_reference_area_fraction=in_pairs/known,resolved_reference_area_fraction=in_resolved/known,
            paired_capture_gap_hours_min=min(gap_hours,default=None),
            paired_capture_gap_hours_median=median(gap_hours) if gap_hours else None,
            paired_capture_gap_hours_max=max(gap_hours,default=None),
            earlier_capture_to_eia_hours_max=max(((windows[week]["previous"]-a).total_seconds()/3600 for a in earlier),default=None),
            later_capture_to_eia_hours_max=max(((windows[week]["current"]-b).total_seconds()/3600 for b in later),default=None))
    return result


def seasonal_history(db):
    history = defaultdict(list)
    development = db.execute(f"SELECT week_ending,release_date,{ACTUAL} FROM eia_inventory_release_panel WHERE study_phase='development'")
    for rows in (db.execute(WARMUP_HISTORY), development):
        for week,release,value in rows:
            if value is not None:
                iso = date.fromisoformat(week).isocalendar()
                history[iso.week].append((iso.year,week,release,value))
    return history


def build_rows(db, weeks, pairs):
    areas = coverage(db,pairs)
    history = seasonal_history(db)
    for week in weeks:
        day,issue = week["week_ending"],week["release_date"]
        iso = date.fromisoformat(day).isocalendar()
        prior_seasons = sorted((w,r,v) for year,w,r,v in history[iso.week]
                               if iso.year-3<=year<iso.year and w<day and r<issue)
        fitted,resolved = (numeric(week[f"{kind}_visible_change_equivalent_bbl"]) for kind in ("fitted", "resolved"))
        yield dict(week_ending=day,release_date=issue,**{ACTUAL:numeric(week[ACTUAL])},
            input_efr_pairs=int(week["input_efr_pairs"]),fitted_efr_pairs=int(week["fitted_efr_pairs"]),
            resolved_efr_pairs=int(week["resolved_efr_pairs"]),**areas[day],
            visible_fitted_change_kbbl=fitted/1000 if fitted is not None else None,
            visible_resolved_change_kbbl=resolved/1000 if resolved is not None else None,
            zero_change_kbbl=0,last_published_change_kbbl=numeric(week["latest_weekly_change_known_before_release"]),
            seasonal_median_kbbl=median(v for _,_,v in prior_seasons) if prior_seasons else None,
            seasonal_observations=len(prior_seasons),seasonal_source_weeks="|".join(w for w,_,_ in prior_seasons),
            seasonal_source_releases="|".join(r for _,r,_ in prior_seasons),eia_source_url=week["source_url"])


def scores(rows, field):
    values = [(row[ACTUAL],row[field]) for row in rows if row[ACTUAL] is not None and row[field] is not None]
    if not values:
        return dict(weeks=0)
    actual,measure = zip(*values)
    errors = [m-a for a,m in values]
    signs = [((a>0)-(a<0))==((m>0)-(m<0)) for a,m in values]
    actual_mean,measured_mean = fmean(actual),fmean(measure)
    sxx = sum((a-actual_mean)**2 for a in actual)
    syy = sum((m-measured_mean)**2 for m in measure)
    correlation = sum((a-actual_mean)*(m-measured_mean) for a,m in values)/math.sqrt(sxx*syy) if sxx and syy else None
    return dict(weeks=len(values),mean_abs_error_kbbl=fmean(map(abs,errors)),median_abs_error_kbbl=median(map(abs,errors)),
        mean_error_kbbl=fmean(errors),sign_agreement=fmean(signs),correlation=correlation)


def summary(rows, pairs):
    common_baselines = [r for r in rows if all(r[field] is not None for field in (*BASELINES,ACTUAL))]
    matched = {}
    for optical in OPTICAL:
        overlap = [r for r in common_baselines if r[optical] is not None]
        matched[optical] = {field:scores(overlap,field) for field in (*BASELINES,optical)}
    paired = [r for r in rows if r["input_efr_pairs"]]
    resolved = [r for r in rows if r["resolved_efr_pairs"]]
    efr_intervals = [(datetime.fromisoformat(p["later_capture_at_utc"])-datetime.fromisoformat(p["earlier_capture_at_utc"])).total_seconds()/3600
                     for p in pairs if p["roof_type"]=="EFR"]
    fixed_changes = [abs(float(p["optical_level_change_m"])) for p in pairs
                     if p["roof_type"] in ("IFR", "Domed EFR") and p["optical_level_change_m"]!=""]
    return dict(development_weeks=len(rows),weeks_with_paired_efr=len(paired),weeks_with_resolved_efr=len(resolved),
        reference_efr_tanks_min=min(r["reference_efr_tanks"] for r in rows),
        reference_efr_tanks_max=max(r["reference_efr_tanks"] for r in rows),
        median_paired_reference_area_fraction=median(r["paired_reference_area_fraction"] for r in paired) if paired else None,
        median_resolved_reference_area_fraction=median(r["resolved_reference_area_fraction"] for r in resolved) if resolved else None,
        paired_outside_reference_tank_weeks=sum(r["paired_outside_reference_tanks"] for r in rows),
        efr_pair_median_capture_interval_hours=median(efr_intervals) if efr_intervals else None,
        fixed_roof_pairs_with_numeric_fits=len(fixed_changes),
        fixed_roof_median_absolute_apparent_change_m=median(fixed_changes) if fixed_changes else None,
        available_sample_scores={field:scores(rows,field) for field in (*BASELINES,*OPTICAL)},
        matched_sample_scores=matched,
        scope="Development only. The raw visible cylinder-equivalent volume sum is a descriptive measurement, with no extrapolation or adjustment to EIA. Reference areas use retrospectively matched engineering geometry and last aerial observation before the earlier window. Reference area is not EIA capacity.")


def main():
    weeks = read_csv("weekly_shadow_development_weeks")
    pairs = read_csv("weekly_shadow_development_pairs")
    with closing(read_only(SOURCE)) as db:
        rows = list(build_rows(db,weeks,pairs))
        expected_weeks = {week for week, in db.execute("SELECT week_ending FROM eia_inventory_release_panel WHERE study_phase='development'")}
    if len(rows)!=len(expected_weeks) or {row["week_ending"] for row in rows}!=expected_weeks:
        raise ValueError("Expected one row per development week")
    export("weekly_shadow_development_check",rows)
    result = summary(rows,pairs)
    write(DERIVED/"weekly_shadow_development_check.json",json.dumps(result,indent=2).encode())
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
