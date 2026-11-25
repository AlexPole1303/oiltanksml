"""Package the dated LiDAR references and nearby native Sentinel-2 data for the new EFR cohorts."""
import csv
import io
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing

import numpy as np

from audit_lidar_tanks import DATASET
from build_shadow_calibration import DERIVED, SCHEMA, add_psf, add_scene, geospatial_inputs, insert
from build_shadow_change_controls import SELECT_QUALITY
from build_tank_geometry_registry import TERMINAL_GEOMETRY_STUDY as TERMINALS
from collect_public import write

OUTPUT = DERIVED / "shadow_terminal_calibration.sqlite"
START, END = "2022-12-17", "2023-01-14"
B04_SCHEMA = """
CREATE TABLE b04_view_evidence (
    observation_id INTEGER PRIMARY KEY REFERENCES observation, archive_single_detector_angle INTEGER,
    method TEXT, detector_id INTEGER, zenith_deg REAL, azimuth_true_deg REAL, azimuth_grid_deg REAL,
    source_angle_nodes_json TEXT, interpolation_weights_json TEXT);
"""


def three_node_weights(nodes):
    xy = np.array([(r["node_east_m"]-r["center_east_m"], r["node_north_m"]-r["center_north_m"])
                   for r in nodes]) / 5000
    weights = np.linalg.solve(np.vstack((xy.T, np.ones(3))), [0, 0, 1])
    return weights if weights.min() >= -1e-9 else None


def angular_mean(nodes, weights):
    zenith = float(weights @ np.array([r["source_zenith_deg"] for r in nodes]))
    directions = np.radians([r["source_azimuth_deg"] for r in nodes])
    azimuth = math.degrees(math.atan2(weights @ np.sin(directions), weights @ np.cos(directions))) % 360
    return zenith, azimuth


def archive_nodes(source, candidates):
    nodes = defaultdict(list)
    for row in source.execute("""SELECT candidate_id,version_id,detector_id,grid_row,grid_col,
        stencil_weight,source_zenith_deg,source_azimuth_deg,node_east_m,node_north_m,center_east_m,center_north_m
        FROM lidar_calibration_b04_angle_node WHERE node_role='bilinear_stencil'
        ORDER BY candidate_id,version_id,detector_id,grid_row,grid_col"""):
        if row["candidate_id"] in candidates:
            nodes[row["candidate_id"],row["version_id"],row["detector_id"]].append(row)
    return nodes


def mask_detector(db, observation):
    masks = [mask for mask, in db.execute("""SELECT DISTINCT red_detector_bits FROM pixel
        WHERE observation_id=? AND mapped_footprint_cell""", (observation,))]
    if len(masks)==1 and masks[0] and masks[0].bit_count()==1:
        return masks[0].bit_length()
    return None


def b04_rows(db, nodes):
    for row in db.execute("""SELECT o.observation_id,o.candidate_id,s.version_id,
        o.resolved_b04_detector,o.red_detector_id,o.red_view_zenith_deg,o.red_view_azimuth_grid_deg,
        o.true_north_grid_bearing_deg FROM observation o JOIN scene s USING(scene_id)
        WHERE s.level='L1C' ORDER BY o.observation_id"""):
        identity, candidate, version, native, detector, zenith, grid_azimuth, north = row
        if native:
            if detector!=mask_detector(db, identity):
                raise ValueError(f"Native B04 angle and detector mask disagree for {candidate} in {version}")
            yield (identity, 1, "archive_bilinear", detector, zenith, (grid_azimuth-north)%360, grid_azimuth, None, None)
            continue
        detector = mask_detector(db, identity)
        stencil = nodes.get((candidate, version, detector), ())
        finite = [r for r in stencil if r["source_zenith_deg"] is not None and r["source_azimuth_deg"] is not None]
        weights = three_node_weights(finite) if len(stencil)==4 and len(finite)==3 else None
        if weights is None:
            yield (identity, 0, "unresolved_archive_angle", detector, None, None, None, None, None)
            continue
        zenith, azimuth = angular_mean(finite, weights)
        yield (identity, 0, "three_native_nodes_inside_triangle", detector, zenith, azimuth, (azimuth+north)%360,
               json.dumps([dict(r) for r in stencil]), json.dumps(weights.tolist()))


def validate_triangles(db, nodes):
    errors = []
    for candidate, version, detector in db.execute("""SELECT o.candidate_id,s.version_id,b.detector_id
        FROM b04_view_evidence b JOIN observation o USING(observation_id) JOIN scene s USING(scene_id)
        WHERE b.method='archive_bilinear'"""):
        rows = nodes.get((candidate,version,detector), ())
        if len(rows)!=4 or any(r["source_zenith_deg"] is None or r["source_azimuth_deg"] is None for r in rows):
            continue
        z0,a0 = angular_mean(rows, np.array([r["stencil_weight"] for r in rows]))
        for index in range(4):
            subset = rows[:index]+rows[index+1:]
            weights = three_node_weights(subset)
            if weights is None:
                continue
            zenith, azimuth = angular_mean(subset, weights)
            errors.append((abs(z0-zenith), abs((a0-azimuth+180)%360-180)))
    if not errors:
        raise ValueError("No fully observed B04 stencils for an independent interpolation check")
    return dict(withheld_corner_comparisons=len(errors),maximum_zenith_difference_deg=max(e[0] for e in errors),
                maximum_circular_azimuth_difference_deg=max(e[1] for e in errors))


def load_references(db, source, candidates):
    fields = ",".join(row[1] for row in db.execute("PRAGMA table_info(lidar_reference)"))
    markers = ",".join("?" for _ in candidates)
    insert(db,"lidar_reference",source.execute(
        f"SELECT {fields} FROM lidar_roof_elevation_reference WHERE candidate_id IN ({markers})", tuple(candidates)))
    db.execute("""INSERT INTO lidar_pair
        SELECT o.observation_id,o.candidate_id,l.dataset,l.acquisition_date_utc,l.point_source_id,
            CASE WHEN julianday(s.captured_at_utc)<julianday(l.first_return_at_utc) THEN 'before_lidar'
                 WHEN julianday(s.captured_at_utc)>julianday(l.last_return_at_utc) THEN 'after_lidar'
                 ELSE 'within_lidar_return_interval' END,
            l.first_return_at_utc,l.last_return_at_utc,
            round(24*(julianday(s.captured_at_utc)-julianday(l.first_return_at_utc)),3),
            round(24*(julianday(s.captured_at_utc)-julianday(l.last_return_at_utc)),3)
        FROM observation o JOIN scene s USING(scene_id) JOIN lidar_reference l USING(candidate_id)
        WHERE l.dataset=?""", (DATASET,))


def export(db, sql, suffix):
    rows = db.execute(sql)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(column[0] for column in rows.description)
    writer.writerows(rows)
    write(OUTPUT.with_name(OUTPUT.stem+suffix),buffer.getvalue().encode())


def main():
    temporary = OUTPUT.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    uri = (DERIVED/"cushing.sqlite").as_uri()+"?mode=ro"
    with closing(sqlite3.connect(uri,uri=True)) as source, closing(sqlite3.connect(temporary)) as db, db:
        source.row_factory = db.row_factory = sqlite3.Row
        markers = ",".join("?" for _ in TERMINALS)
        candidates = {r[0]:r[1] for r in source.execute(f"""SELECT candidate_id,documented_terminal_id
            FROM tank_engineering_registry WHERE documented_terminal_id IN ({markers}) AND documented_roof_type='EFR'""", TERMINALS)}
        if not candidates:
            raise ValueError("The registry has no documented EFR for the selected terminals")
        shapes, calibration = geospatial_inputs(source)
        groups = defaultdict(list)
        for key,rows in calibration.items():
            if key[0]=="L2A":
                selected = [r for r in rows if r["candidate_id"] in candidates]
                if selected:
                    groups[key] = selected
        sql = SELECT_QUALITY+f"""AND t.documented_terminal_id IN ({markers})
            AND substr(q.captured_at_utc,1,10) BETWEEN ? AND ?
            ORDER BY version_id,optical_source_directory,q.candidate_id"""
        for row in source.execute(sql, (*TERMINALS,START,END)):
            if row["candidate_id"] in candidates:
                groups["L1C",row["version_id"],row["optical_source_directory"]].append(row)
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA+B04_SCHEMA)
        add_psf(db)
        identities = {}
        for i,(key,rows) in enumerate(sorted(groups.items()),1):
            add_scene(db,source,i,key,rows,shapes,identities)
        load_references(db,source,candidates)
        nodes = archive_nodes(source,candidates)
        insert(db,"b04_view_evidence",b04_rows(db,nodes))
        checks = validate_triangles(db,nodes)
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("The terminal calibration database did not pass integrity checks")
        if {r[0] for r in db.execute("SELECT DISTINCT candidate_id FROM observation_lidar")}!=candidates.keys():
            raise ValueError("A documented roof is missing from the native-pixel and LiDAR package")
        coverage = [dict(r) for r in db.execute("""SELECT level,substr(captured_at_utc,1,10) AS capture_day,
            count(*) AS tank_products,count(DISTINCT candidate_id) AS tanks,
            count(DISTINCT CASE WHEN roof_clear_by_source_mask AND resolved_b04_detector THEN candidate_id END) AS clear_native_angle_tanks,
            min(abs(optical_minus_lidar_end_hours)) AS closest_lidar_end_hours
            FROM observation_lidar GROUP BY 1,2 ORDER BY 1,2""")]
        counts = {table:db.execute(f"SELECT count(*) FROM {table}").fetchone()[0] for table in
                  ("scene","observation","pixel","lidar_reference","lidar_pair")}
        summary = dict(tank_counts={terminal:list(candidates.values()).count(terminal) for terminal in TERMINALS},
                       source_window=[START,END],**counts,coverage=coverage,triangle_validation=checks,
                       angle_methods=dict(db.execute("SELECT method,count(*) FROM b04_view_evidence GROUP BY method")),
                       notes="Triangle angles are optional research inputs. Archive eligibility flags remain in the observation table.")
        export(db,"SELECT * FROM observation_lidar ORDER BY observation_id,point_source_id","_observations.csv")
        export(db,"""SELECT o.candidate_id,s.version_id,s.captured_at_utc,b.* FROM b04_view_evidence b
            JOIN observation o USING(observation_id) JOIN scene s USING(scene_id) ORDER BY o.candidate_id,s.captured_at_utc""",
               "_b04_geometry.csv")
    temporary.replace(OUTPUT)
    write(OUTPUT.with_name(OUTPUT.stem+"_summary.json"),json.dumps(summary,indent=2).encode()+b"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
