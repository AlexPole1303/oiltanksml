"""Compare independently fitted 2011 and 2022 upper outlines for the newly documented EFR cohorts."""
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing

from audit_lidar_tanks import LEGACY_DATASET, SURVEYS
from build_lidar_radial_profiles import Circle
from build_lidar_roof_reference import reference_rows
from build_tank_geometry_registry import TERMINAL_GEOMETRY_STUDY
from build_tank_lidar_outlines import DERIVED, OUTLINE_FIELDS, footprint_data, outline_rows
from collect_public import ROOT, write_csv

def write_geometry(terminals, output):
    uri = (DERIVED/"cushing.sqlite").as_uri()+"?mode=ro"
    with closing(sqlite3.connect(uri,uri=True)) as db:
        db.row_factory = sqlite3.Row
        markers = ",".join("?" for _ in terminals)
        eligible = {r["candidate_id"]:r for r in db.execute(f"""WITH flights AS (
            SELECT e.candidate_id,e.documented_terminal_id,e.cohort_id,r.source_directory,
            r.point_source_id,r.rim_minus_center_plane_native_z AS mapped_2011_gap_m,
            r.rim_sector_p95_range_native AS mapped_2011_rim_sector_range_m,
            l.center_east_m AS east_2022,l.center_north_m AS north_2022,l.diameter_m AS diameter_2022,
            l.fit_status AS fit_status_2022,h.rim_z_p95,h.ground_z_p50,
            row_number() OVER (PARTITION BY e.candidate_id ORDER BY r.rim_minus_center_plane_native_z IS NULL,
                r.rim_sectors DESC,r.center_sectors DESC,r.rim_nonground_returns DESC,r.point_source_id) AS rank
            FROM tank_engineering_registry e JOIN lidar_roof_elevation_reference r USING(candidate_id)
            JOIN lidar_tank_observation h USING(candidate_id,dataset,acquisition_date_utc,point_source_id)
            JOIN tank_lidar_outline l USING(candidate_id)
            WHERE e.documented_terminal_id IN ({markers}) AND r.dataset=? AND e.documented_roof_type='EFR'
            AND r.geometry_basis='mapped_footprint_circle'
        ) SELECT * FROM flights WHERE rank=1""", (*terminals, LEGACY_DATASET))}
        fields = [r[1] for r in db.execute("PRAGMA table_info(lidar_roof_elevation_reference)")]
    selected = [row for layer in ("osm", "supplemental") for row in footprint_data(layer) if row[0] in eligible]
    if not selected or len(selected)!=len(eligible):
        raise ValueError("The documented legacy roof set does not match the candidate outlines")
    priors = {candidate:(r["rim_z_p95"],r["ground_z_p50"]) for candidate,r in eligible.items()}
    folders = defaultdict(list)
    for row in selected:
        folders[eligible[row[0]]["source_directory"]].append(row)
    fits = []
    profiles = defaultdict(dict)
    for directory, candidates in folders.items():
        folder = ROOT / directory
        group = [dict(zip(OUTLINE_FIELDS, row, strict=True))
                 for row in outline_rows(candidates, priors, folder, SURVEYS[LEGACY_DATASET])]
        fits.extend(group)
        circles = [Circle(row["candidate_id"], "independent_2011_elevated_circle", row["fit_status"],
                          row["center_east_m"], row["center_north_m"], row["diameter_m"] / 2)
                   for row in group if row["diameter_m"] is not None]
        for raw in reference_rows(circles, SURVEYS[LEGACY_DATASET], folder):
            row = dict(zip(fields, raw, strict=True))
            profiles[row["candidate_id"]][row["point_source_id"]] = row
    result = []
    for row in fits:
        r = eligible[row["candidate_id"]]
        profile = profiles[row["candidate_id"]].get(r["point_source_id"],{})
        comparable = row["diameter_m"] is not None and r["diameter_2022"] is not None
        row.update(terminal_id=r["documented_terminal_id"],cohort_id=r["cohort_id"],diameter_2022_m=r["diameter_2022"],
                   fit_status_2022=r["fit_status_2022"],center_distance_between_surveys_m=math.hypot(
                       row["center_east_m"]-r["east_2022"],row["center_north_m"]-r["north_2022"]) if comparable else None,
                   diameter_2011_minus_2022_m=row["diameter_m"]-r["diameter_2022"] if comparable else None,
                   mapped_2011_gap_m=r["mapped_2011_gap_m"],mapped_2011_rim_sector_range_m=r["mapped_2011_rim_sector_range_m"],
                   fitted_2011_gap_m=profile.get("rim_minus_center_plane_native_z"),
                   fitted_2011_rim_sector_range_m=profile.get("rim_sector_p95_range_native"),
                   fitted_2011_rim_sectors=profile.get("rim_sectors"),
                   fitted_2011_source_nodes_json=profile.get("source_node_ids_json"),
                   point_source_id=r["point_source_id"])
        result.append(row)
    fields = tuple(result[0])
    write_csv(output,fields,([r[k] for k in fields] for r in result))
    print(json.dumps({"legacy_contours":len(result),"fitted_circles":sum(r["diameter_m"] is not None for r in result),
                      "numeric_fitted_profiles":sum(r["fitted_2011_gap_m"] is not None for r in result)},indent=2))


if __name__=="__main__":
    write_geometry(TERMINAL_GEOMETRY_STUDY, DERIVED/"terminal_legacy_lidar_geometry.csv")
