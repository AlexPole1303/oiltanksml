"""Measure optical repeatability with tandem Sentinel-2 and fixed exterior roofs."""
import argparse
import datetime as dt
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from itertools import combinations

import numpy as np

from build_l1c_quality import candidate_layer
from build_shadow_calibration import SCHEMA, add_psf, add_scene, insert
from collect_public import ROOT, write, write_csv
from collect_s2_l1c import parse_product
from fit_shadow_geometry import FIELDS, estimate, inputs, read_only

DERIVED = ROOT / "data" / "derived"
CONTROL_DB = DERIVED / "shadow_change_controls.sqlite"
BBL_PER_M3 = 1 / 0.158987294928
FIXED = ("IFR", "Domed EFR")
RESOLVED = "resolved_in_geometry_diagnostic"
CAMPAIGNS = (("S2C", "2024-10-29", "2024-12-19"), ("S2B", "2026-02-02", "2026-02-22"))
CONTROL_SCHEMA = """
CREATE TABLE tandem_granule (
    capture_day TEXT,orbit TEXT,tile TEXT,other_sensor TEXT,a_version TEXT,other_version TEXT,
    a_capture_at_utc TEXT,other_capture_at_utc TEXT,other_minus_a_seconds REAL,
    a_archive TEXT,other_archive TEXT,a_archive_available_at_utc TEXT,other_archive_available_at_utc TEXT,
    a_geometry_available_at_utc TEXT,other_geometry_available_at_utc TEXT,
    PRIMARY KEY(a_version,other_version));
"""
SELECT_QUALITY = """
    SELECT q.candidate_id,'L1C' AS level,q.scene_item_id,q.l1c_product_name AS version_id,
        q.source_directory AS optical_source_directory,q.captured_at_utc AS optical_captured_at_utc,q.tile_metadata_path,
        t.documented_roof_type,t.cohort_id,t.source_document_date,t.first_seen,t.longitude,t.latitude,
        CASE WHEN l.diameter_m IS NOT NULL THEN 'lidar_elevated_circle'
             WHEN c.diameter_m IS NOT NULL THEN 'lidar_covered_surface_circle' ELSE 'mapped_area_circle' END AS geometry_basis,
        coalesce(l.fit_status,c.fit_status) AS outline_fit_status,
        CASE WHEN l.diameter_m IS NOT NULL THEN l.center_east_m ELSE coalesce(c.center_east_m,t.footprint_east_m) END AS center_east_m,
        CASE WHEN l.diameter_m IS NOT NULL THEN l.center_north_m ELSE coalesce(c.center_north_m,t.footprint_north_m) END AS center_north_m,
        coalesce(l.diameter_m,c.diameter_m,t.footprint_diameter_m)/2 AS radius_m,
        q.sun_zenith_deg,q.sun_azimuth_grid_deg,q.red_detector_with_view_options_json AS red_view_options_json,
        q.optical_touched_cells_expected AS optical_cells,q.optical_cells_valid_in_all_bands AS valid_cells,
        q.opaque_roof_cells=0 AND q.cirrus_roof_cells=0 AND coalesce(q.snow_ice_roof_cells,0)=0 AS roof_clear_by_source_mask,
        q.red_detector_covered_roof_cells=q.optical_touched_cells_expected AND q.red_detector_ambiguous_roof_cells=0
            AND json_array_length(q.red_detector_with_view_options_json)=1 AS resolved_b04_detector,q.cloud_mask_basis
    FROM sentinel2_l1c_tank_quality q JOIN tank_engineering_registry t USING(candidate_id)
    LEFT JOIN tank_lidar_outline l USING(candidate_id) LEFT JOIN enbridge_main_covered_roof_outline c USING(candidate_id)
    WHERE t.documented_roof_type IS NOT NULL AND substr(q.captured_at_utc,1,10)>=t.first_seen
"""


def timestamp(text):
    return dt.datetime.fromisoformat(text)


def tandem_granules(source):
    rows = source.execute("""SELECT DISTINCT version_id,captured_at_utc,archive,archive_available_at_utc,geometry_available_at_utc
        FROM sentinel2_l1c_input_inventory WHERE version_id IS NOT NULL AND local_folders_screened>0""")
    groups = defaultdict(dict)
    for version,captured,archive,available,geometry in rows:
        (sensor,_,orbit,tile),_,_ = parse_product(version)
        day = captured[:10]
        for other,start,end in CAMPAIGNS:
            if start<=day<=end and sensor in ("S2A",other):
                key = day,orbit,tile,other
                if sensor in groups[key]:
                    raise ValueError(f"More than one local exact product for {sensor} in {key}")
                groups[key][sensor] = (version,captured,archive,available,geometry)
    for (day,orbit,tile,other),sensors in sorted(groups.items()):
        if len(sensors)!=2:
            continue
        a,b = sensors["S2A"],sensors[other]
        seconds = (timestamp(b[1])-timestamp(a[1])).total_seconds()
        if abs(seconds)<=60:
            yield (day,orbit,tile,other,a[0],b[0],a[1],b[1],seconds,a[2],b[2],a[3],b[3],a[4],b[4])


def scene_inputs(source, tandems):
    groups = defaultdict(dict)
    fixed = source.execute(SELECT_QUALITY+"""AND t.documented_roof_type IN ('IFR','Domed EFR')
        AND optical_cells=valid_cells AND roof_clear_by_source_mask AND resolved_b04_detector""")
    versions = sorted({version for pair in tandems for version in pair[4:6]})
    tandem = source.execute(SELECT_QUALITY+f"AND q.l1c_product_name IN ({','.join('?' for _ in versions)})",versions)
    for cursor in (fixed,tandem):
        for row in cursor:
            key = row["level"],row["version_id"],row["optical_source_directory"]
            groups[key][row["candidate_id"]] = row
    return groups


def build_input_database(source):
    temporary = CONTROL_DB.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    tandems = list(tandem_granules(source))
    if not tandems:
        raise ValueError("No local Sentinel-2 tandem granules were verified")
    groups = scene_inputs(source,tandems)
    candidates = dict((*candidate_layer("osm"),*candidate_layer("supplemental")))
    identities = {}
    with closing(sqlite3.connect(temporary)) as db, db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA+CONTROL_SCHEMA)
        add_psf(db)
        insert(db,"tandem_granule",tandems)
        for number,(key,values) in enumerate(sorted(groups.items()),1):
            add_scene(db,source,number,key,[values[candidate] for candidate in sorted(values)],candidates,identities)
            if number%50==0 or number==len(groups):
                print(f"Prepared {number}/{len(groups)} exact source granules",flush=True)
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Control pixel database integrity check failed")
    temporary.replace(CONTROL_DB)


def fit_archive(db, source):
    observations,kernels = inputs(db)
    shells = dict(source.execute("SELECT candidate_id,lidar_rim_to_ground_native_z FROM tank_engineering_registry"))

    def fit(row):
        return estimate(row,shells[row["candidate_id"]],kernels[row["platform"]],calibration_db=CONTROL_DB)

    rows = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for row in pool.map(fit,observations):
            rows.append(row)
            if len(rows)%1000==0 or len(rows)==len(observations):
                print(f"Estimated {len(rows)}/{len(observations)} L1C observations",flush=True)
    write_csv(DERIVED/"shadow_change_control_estimates.csv",FIELDS,([r.get(f) for f in FIELDS] for r in rows))
    return rows


def scale(row):
    return math.pi * (row["diameter_m"]/2)**2 * BBL_PER_M3


def tandem_rows(db, estimates):
    lookup = defaultdict(dict)
    for row in estimates:
        lookup[row["version_id"]][row["candidate_id"]] = row
    pairs = []
    for item in db.execute("SELECT * FROM tandem_granule ORDER BY capture_day,tile"):
        a,b = lookup[item["a_version"]],lookup[item["other_version"]]
        for candidate in sorted(a.keys()&b.keys()):
            a_fit,other_fit = a[candidate],b[candidate]
            ag,bg = a_fit.get("estimated_gap_m"),other_fit.get("estimated_gap_m")
            difference = bg-ag if ag is not None and bg is not None else None
            pairs.append(dict(item)|dict(candidate_id=candidate,roof_type=a_fit["roof_type"],
                a_observation_id=a_fit["observation_id"],other_observation_id=other_fit["observation_id"],
                a_status=a_fit["fit_status"],other_status=other_fit["fit_status"],a_gap_m=ag,other_gap_m=bg,
                other_minus_a_gap_m=difference,other_minus_a_gap_equivalent_bbl=difference*scale(a_fit) if difference is not None else None,
                common_usable_pixels=min(a_fit.get("fitted_pixels",0),other_fit.get("fitted_pixels",0)),
                both_resolved=a_fit["fit_status"]==other_fit["fit_status"]==RESOLVED,selected_for_summary=False))
    distinct = defaultdict(list)
    for row in pairs:
        if row["other_minus_a_gap_m"] is not None:
            distinct[row["capture_day"],row["other_sensor"],row["candidate_id"]].append(row)
    for alternatives in distinct.values():
        best = min(alternatives,key=lambda r:(-r["common_usable_pixels"],r["tile"]))
        best["selected_for_summary"] = True
    return pairs


def tandem_change_rows(pairs):
    groups = defaultdict(list)
    for row in pairs:
        if row["selected_for_summary"]:
            groups[row["other_sensor"],row["candidate_id"]].append(row)
    for (sensor,candidate),values in sorted(groups.items()):
        for first,last in combinations(sorted(values,key=lambda r:r["capture_day"]),2):
            a_change = first["a_gap_m"]-last["a_gap_m"]
            other_change = first["other_gap_m"]-last["other_gap_m"]
            volume_change_difference = first["other_minus_a_gap_equivalent_bbl"]-last["other_minus_a_gap_equivalent_bbl"]
            yield dict(other_sensor=sensor,candidate_id=candidate,roof_type=first["roof_type"],
                       earlier_day=first["capture_day"],later_day=last["capture_day"],
                       elapsed_days=(dt.date.fromisoformat(last["capture_day"])-dt.date.fromisoformat(first["capture_day"])).days,
                       a_earlier_version=first["a_version"],a_later_version=last["a_version"],
                       other_earlier_version=first["other_version"],other_later_version=last["other_version"],
                       a_change_m=a_change,other_change_m=other_change,other_minus_a_change_m=other_change-a_change,
                       other_minus_a_change_equivalent_bbl=volume_change_difference,
                       all_four_resolved=first["both_resolved"] and last["both_resolved"])


def fixed_weekly_pairs(source, estimates):
    versions = defaultdict(list)
    for row in estimates:
        if row["roof_type"] in FIXED and row.get("estimated_gap_m") is not None:
            versions[row["version_id"]].append(row)
    endpoints = {}
    for window in source.execute("""SELECT i.week_ending,i.endpoint,i.version_id,i.captured_at_utc,i.archive_available_at_utc,
        i.geometry_available_at_utc,w.release_date,w.study_phase,w.inputs_available_before_utc
        FROM eia_satellite_window_input i JOIN eia_satellite_window w USING(week_ending,endpoint)
        WHERE i.level='L1C' AND i.local_exact_version_screened=1 AND i.geometry_available_at_utc IS NOT NULL
          AND julianday(i.geometry_available_at_utc)<julianday(w.inputs_available_before_utc)
        ORDER BY i.week_ending,i.endpoint,i.captured_at_utc DESC,i.version_id"""):
        for row in versions[window["version_id"]]:
            endpoints.setdefault((window["week_ending"],row["candidate_id"],window["endpoint"]),(window,row))
    for (week,candidate,endpoint),(last_clock,last) in sorted(endpoints.items()):
        if endpoint!="current" or (week,candidate,"previous") not in endpoints:
            continue
        first_clock,first = endpoints[week,candidate,"previous"]
        change = first["estimated_gap_m"]-last["estimated_gap_m"]
        yield dict(week_ending=week,release_date=last_clock["release_date"],study_phase=last_clock["study_phase"],
                   inputs_available_before_utc=last_clock["inputs_available_before_utc"],candidate_id=candidate,roof_type=last["roof_type"],
                   earlier_version=first["version_id"],later_version=last["version_id"],
                   earlier_capture_at_utc=first_clock["captured_at_utc"],later_capture_at_utc=last_clock["captured_at_utc"],
                   earlier_archive_available_at_utc=first_clock["archive_available_at_utc"],later_archive_available_at_utc=last_clock["archive_available_at_utc"],
                   earlier_geometry_available_at_utc=first_clock["geometry_available_at_utc"],later_geometry_available_at_utc=last_clock["geometry_available_at_utc"],
                   elapsed_days=(timestamp(last_clock["captured_at_utc"])-timestamp(first_clock["captured_at_utc"])).total_seconds()/86400,
                   earlier_sun_elevation_deg=first["sun_elevation_deg"],later_sun_elevation_deg=last["sun_elevation_deg"],
                   earlier_status=first["fit_status"],later_status=last["fit_status"],
                   earlier_gap_m=first["estimated_gap_m"],later_gap_m=last["estimated_gap_m"],apparent_level_change_m=change,
                   apparent_change_equivalent_bbl=change*scale(last),both_resolved=first["fit_status"]==last["fit_status"]==RESOLVED)


def differences(values):
    if not len(values):
        return dict(observations=0)
    values = np.asarray(values,dtype=float)
    return dict(observations=len(values),median_abs_m=float(np.median(abs(values))),p90_abs_m=float(np.quantile(abs(values),0.9)),
                maximum_abs_m=float(max(abs(values))),median_signed_m=float(np.median(values)),exact_matches=int(np.count_nonzero(values==0)))


def reports(db, estimates, tandem, changes, weekly):
    summary = dict(native_pixels=db.execute("SELECT count(*) FROM pixel").fetchone()[0],
                   source_scenes=db.execute("SELECT count(*) FROM scene").fetchone()[0],
                   observation_rows=len(estimates),tandem_granules=db.execute("SELECT count(*) FROM tandem_granule").fetchone()[0],
                   status_counts=dict(Counter(r["fit_status"] for r in estimates)))
    same_pass = defaultdict(list)
    for row in tandem:
        same_pass[row["capture_day"],row["other_sensor"],row["roof_type"]].append(row)
    summary["tandem_same_pass"] = []
    for (day,sensor,roof),rows in sorted(same_pass.items()):
        selected = [r for r in rows if r["selected_for_summary"]]
        summary["tandem_same_pass"].append(dict(day=day,other_sensor=sensor,roof_type=roof,
            candidates=len({r["candidate_id"] for r in rows}),both_resolved=sum(r["both_resolved"] for r in selected),
            **differences([r["other_minus_a_gap_m"] for r in selected])))
    between = defaultdict(list)
    for row in changes:
        between[row["other_sensor"],row["earlier_day"],row["later_day"],row["roof_type"]].append(row)
    summary["tandem_level_change_agreement"] = [dict(other_sensor=sensor,earlier_day=first,later_day=last,roof_type=roof,
        all_four_resolved=sum(r["all_four_resolved"] for r in rows),**differences([r["other_minus_a_change_m"] for r in rows]))
        for (sensor,first,last,roof),rows in sorted(between.items())]
    summary["fixed_weekly"] = []
    for roof in FIXED:
        rows = [r for r in weekly if r["roof_type"]==roof]
        summary["fixed_weekly"].append(dict(roof_type=roof,candidates=len({r["candidate_id"] for r in rows}),
            weeks=len({r["week_ending"] for r in rows}),both_resolved=sum(r["both_resolved"] for r in rows),
            minimum_sun_elevation_deg=min((r["later_sun_elevation_deg"] for r in rows),default=None),
            maximum_sun_elevation_deg=max((r["later_sun_elevation_deg"] for r in rows),default=None),
            **differences([r["apparent_level_change_m"] for r in rows])))
    write(DERIVED/"shadow_change_control_summary.json",json.dumps(summary,indent=2).encode())


def export(name, rows):
    if not rows:
        return
    fields = tuple(rows[0])
    write_csv(DERIVED/f"{name}.csv",fields,([row[f] for f in fields] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-inputs",action="store_true")
    args = parser.parse_args()
    with closing(read_only(DERIVED/"cushing.sqlite")) as source:
        source.row_factory = sqlite3.Row
        if not args.reuse_inputs:
            build_input_database(source)
        with closing(read_only(CONTROL_DB)) as db:
            db.row_factory = sqlite3.Row
            estimates = fit_archive(db,source)
            tandem = tandem_rows(db,estimates)
            changes = list(tandem_change_rows(tandem))
            weekly = list(fixed_weekly_pairs(source,estimates))
            export("shadow_tandem_pass_pairs",tandem)
            export("shadow_tandem_change_pairs",changes)
            export("shadow_fixed_weekly_pairs",weekly)
            reports(db,estimates,tandem,changes,weekly)


if __name__=="__main__":
    main()
