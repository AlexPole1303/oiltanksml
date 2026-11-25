"""Match public Aerial Oklahoma calendar dates to exact local Sentinel-2 products."""
import csv
import datetime as dt
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from itertools import combinations
from zoneinfo import ZoneInfo

from collect_public import ROOT, write, write_csv

REFERENCE = ROOT / "data" / "reference" / "aerialok_cushing_flights.csv"
DERIVED = ROOT / "data" / "derived"
CENTRAL = ZoneInfo("America/Chicago")
SELECT_OBSERVATIONS = """
    SELECT q.candidate_id,t.documented_roof_type,t.cohort_id,q.l1c_product_name,q.captured_at_utc,q.source_directory,
        q.tile_metadata_path,i.archive,i.archive_available_at_utc,i.geometry_available_at_utc,
        q.optical_touched_cells_expected,q.optical_cells_valid_in_all_bands,
        q.opaque_roof_cells=0 AND q.cirrus_roof_cells=0 AND coalesce(q.snow_ice_roof_cells,0)=0 AS roof_clear,
        q.red_detector_covered_roof_cells=q.optical_touched_cells_expected AND q.red_detector_ambiguous_roof_cells=0
            AND json_array_length(q.red_detector_with_view_options_json)=1 AS resolved_b04_detector
    FROM sentinel2_l1c_tank_quality q JOIN tank_engineering_registry t USING(candidate_id)
    JOIN sentinel2_l1c_input_inventory i ON i.canonical_l2a_id=q.scene_item_id AND i.version_id=q.l1c_product_name
    WHERE t.documented_roof_type IS NOT NULL AND substr(q.captured_at_utc,1,10)>=t.first_seen
        AND q.captured_at_utc>=? AND q.captured_at_utc<?
    ORDER BY q.captured_at_utc,q.candidate_id,q.l1c_product_name
"""


def matches(db, dates):
    start = min(dates)-dt.timedelta(days=1)
    end = max(dates)+dt.timedelta(days=2)
    for row in db.execute(SELECT_OBSERVATIONS,(start.isoformat(),end.isoformat())):
        captured = dt.datetime.fromisoformat(row["captured_at_utc"])
        satellite_day = captured.astimezone(CENTRAL).date()
        for day_offset in (-1,0,1):
            flight = satellite_day-dt.timedelta(days=day_offset)
            if flight not in dates:
                continue
            local_start = dt.datetime.combine(flight,dt.time(),CENTRAL).astimezone(dt.timezone.utc)
            local_end = dt.datetime.combine(flight+dt.timedelta(days=1),dt.time(),CENTRAL).astimezone(dt.timezone.utc)
            since = (captured-local_start).total_seconds()/3600
            until = (captured-local_end).total_seconds()/3600
            usable = (row["roof_clear"] and row["resolved_b04_detector"]
                      and row["optical_touched_cells_expected"]==row["optical_cells_valid_in_all_bands"])
            yield dict(flight_date=flight.isoformat(),source_url=dates[flight]["source_url"],
                       aerial_time_precision=dates[flight]["time_precision"],satellite_calendar_day_offset=day_offset,
                       minimum_possible_separation_hours=max(until,-since,0),
                       maximum_possible_separation_hours=max(abs(since),abs(until)),
                       complete_clear_b04_geometry=bool(usable),**dict(row))


def flight_pairs(dates, rows):
    same_day = defaultdict(set)
    near_day = defaultdict(lambda: defaultdict(set))
    for row in rows:
        if not row["complete_clear_b04_geometry"] or row["documented_roof_type"]!="EFR":
            continue
        day = dt.date.fromisoformat(row["flight_date"])
        near_day[day][row["candidate_id"]].add(day+dt.timedelta(days=row["satellite_calendar_day_offset"]))
        if row["satellite_calendar_day_offset"]==0:
            same_day[day].add(row["candidate_id"])
    pairs = []
    for earlier,later in combinations(sorted(dates),2):
        elapsed = (later-earlier).days
        if elapsed>14:
            continue
        common_same = same_day[earlier]&same_day[later]
        common_near = {candidate for candidate in near_day[earlier].keys()&near_day[later].keys()
                       if min(near_day[earlier][candidate])<max(near_day[later][candidate])}
        pairs.append(dict(earlier_flight_date=earlier.isoformat(),later_flight_date=later.isoformat(),elapsed_days=elapsed,
            same_day_common_efr_contours=len(common_same),near_day_common_efr_contours=len(common_near),
            same_day_common_efr_candidate_ids_json=json.dumps(sorted(common_same),separators=(",",":"))))
    return same_day,near_day,pairs


def export(name, rows):
    if rows:
        fields = tuple(rows[0])
        write_csv(DERIVED/f"{name}.csv",fields,([row[field] for field in fields] for row in rows))


def main():
    with REFERENCE.open(newline="") as stream:
        dates = {dt.date.fromisoformat(row["flight_date"]):row for row in csv.DictReader(stream)}
    database = (DERIVED/"cushing.sqlite").as_uri()+"?mode=ro"
    with closing(sqlite3.connect(database,uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = list(matches(db,dates))
    same_day,near_day,pairs = flight_pairs(dates,rows)
    export("aerialok_sentinel2_flight_matches",rows)
    export("aerialok_candidate_change_intervals",pairs)
    viable = [r for r in pairs if r["same_day_common_efr_contours"]>0]
    summary = dict(public_flights=len(dates),satellite_match_rows=len(rows),
        dates_with_same_day_usable_efr=sum(bool(v) for v in same_day.values()),
        dates_with_near_day_usable_efr=sum(bool(v) for v in near_day.values()),
        intervals_up_to_14_days=len(pairs),intervals_with_both_same_day_optical=len(viable),
        same_day_optical_intervals_5_to_9_days=[r for r in viable if 5<=r["elapsed_days"]<=9],
        per_date=[dict(date=day.isoformat(),same_day_efr=len(same_day[day]),near_day_efr=len(near_day[day])) for day in sorted(dates)],
        scope="Counts concern Sentinel-2 geometry and quality. Airborne per-tank coverage, timestamps and roof elevation measurements have not been verified.")
    write(DERIVED/"aerialok_flight_match_summary.json",json.dumps(summary,indent=2).encode())
    print({key:value for key,value in summary.items() if not isinstance(value,list)},flush=True)


if __name__=="__main__":
    main()
