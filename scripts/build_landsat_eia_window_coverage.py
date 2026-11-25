"""Index Landsat alternatives for each archived EIA optical endpoint."""
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime

from build_weekly_shadow_panel import DERIVED, ENDPOINT_CTES, SOURCE, export, read_only, write

L1C = "sentinel2_l1c"
PAN = "gcs_collection1_level1_pan"
CONTEXT = "pc_collection2_level2_context"
HLS = "azure_hls_l30_gap_context"
ROUTES = (L1C,PAN,CONTEXT,HLS)
SENTINEL = ENDPOINT_CTES + """
SELECT week_ending,endpoint,study_phase,release_date,candidate_id,roof_type,terminal_id,diameter_m,
    version_id AS product_id,captured_at_utc,archive_available_at_utc,geometry_available_at_utc,
    inputs_available_before_utc,window_ends_by_utc,prior_aerial_date,availability_basis,source_directory,
    tile_metadata_path AS metadata_path,rank
FROM ranked
"""
LANDSAT = """
WITH scene_windows AS MATERIALIZED (
    SELECT w.week_ending,w.endpoint,w.study_phase,w.release_date,w.inputs_available_before_utc,w.window_ends_by_utc,l.*
    FROM landsat_local_product l JOIN eia_satellite_window w
      ON julianday(l.captured_at_utc)>julianday(w.window_starts_after_utc)
     AND julianday(l.captured_at_utc)<=julianday(w.window_ends_by_utc)
), quality AS MATERIALIZED (
    SELECT w.*,q.candidate_id,q.signal_cells,q.valid_signal_cells,q.qa_roof_cells,q.clear_qa_roof_cells,q.clear_context_qa_cells,
        t.documented_roof_type AS roof_type,t.documented_terminal_id AS terminal_id,
        coalesce(l.diameter_m,c.diameter_m,t.footprint_diameter_m) AS diameter_m,
        (SELECT max(a.observed_date) FROM tank_aerial_visual_observation a
         WHERE a.candidate_id=q.candidate_id AND a.observed_date<substr(w.captured_at_utc,1,10)) AS prior_aerial_date
    FROM scene_windows w CROSS JOIN landsat_tank_scene_quality q USING(product_id)
    JOIN tank_engineering_registry t USING(candidate_id)
    LEFT JOIN tank_lidar_outline l USING(candidate_id) LEFT JOIN enbridge_main_covered_roof_outline c USING(candidate_id)
    WHERE t.documented_roof_type IS NOT NULL AND q.signal_cells>0 AND q.signal_cells=q.valid_signal_cells
      AND q.qa_roof_cells>0 AND q.qa_roof_cells=q.valid_qa_roof_cells AND q.qa_roof_cells=q.clear_qa_roof_cells
)
SELECT q.*,a.assets,a.latest_archive_object_time_utc
FROM quality q LEFT JOIN landsat_product_archive_clock a ON a.product_id=q.product_id
WHERE q.prior_aerial_date IS NOT NULL AND q.diameter_m>0 AND NOT EXISTS (
    SELECT 1 FROM tank_aerial_visual_observation v WHERE v.candidate_id=q.candidate_id
        AND v.observed_date=q.prior_aerial_date AND v.structure_observation IS NOT 'aboveground_tank_seen')
ORDER BY q.week_ending,q.endpoint,q.candidate_id,q.source_kind,q.captured_at_utc,q.product_id
"""


def timestamp(text):
    return datetime.fromisoformat(text)


def slot(row):
    return row["week_ending"],row["candidate_id"],row["endpoint"]


def make_endpoint(r,route,*,archive,geometry,angle_file=None,archive_status="before_cutoff"):
    capture = timestamp(r["captured_at_utc"])
    return dict(week_ending=r["week_ending"],endpoint=r["endpoint"],study_phase=r["study_phase"],release_date=r["release_date"],
        candidate_id=r["candidate_id"],roof_type=r["roof_type"],terminal_id=r["terminal_id"],diameter_m=r["diameter_m"],
        route=route,product_id=r["product_id"],acquisition_key=r.get("acquisition_key"),captured_at_utc=r["captured_at_utc"],
        archive_available_at_utc=archive,geometry_available_at_utc=geometry,
        inputs_available_before_utc=r["inputs_available_before_utc"],window_ends_by_utc=r["window_ends_by_utc"],
        hours_before_eia_measurement=(timestamp(r["window_ends_by_utc"])-capture).total_seconds()/3600,
        prior_aerial_date=r["prior_aerial_date"],source_directory=r["source_directory"],metadata_path=r["metadata_path"],
        angle_file=angle_file,clock_basis=r.get("availability_basis",r.get("clock_basis")),archive_status=archive_status,
        signal_grid_m=r.get("native_signal_gsd_m",10),signal_cells=r.get("signal_cells"),qa_roof_cells=r.get("qa_roof_cells"),
        clear_context_qa_cells=r.get("clear_context_qa_cells"))


def landsat_rows(db,angles):
    expected_assets = {PAN:3,CONTEXT:9,HLS:11}
    expected_basis = {PAN:"latest_gcs_object_creation",CONTEXT:"latest_azure_asset_modified",HLS:"latest_azure_asset_modified"}
    for raw in db.execute(LANDSAT):
        row = dict(raw)
        route,product = row["source_kind"],row["product_id"]
        archived = row["latest_archive_object_time_utc"]
        angle = angles.get(product) if route==PAN else None
        if angle and (timestamp(angle["capture_at_utc"])!=timestamp(row["captured_at_utc"])
                      or timestamp(angle["pan_assets_available_at_utc"])!=timestamp(archived)):
            raise ValueError(f"Angle manifest disagrees with the original Landsat product clock for {product}")
        geometry = angle["geometry_inputs_available_at_utc"] if angle else None
        exact = (row["assets"]==expected_assets[route] and row["clock_basis"]==expected_basis[route]
                 and archived and row["source_created_utc"] and timestamp(archived)==timestamp(row["source_created_utc"]))
        cutoff = timestamp(row["inputs_available_before_utc"])
        if not exact or (route==PAN and not angle):
            status = "missing_exact_archive_clock_or_angles"
        elif timestamp(archived)>=cutoff or (geometry and timestamp(geometry)>=cutoff):
            status = "exact_inputs_after_cutoff"
        else:
            status = "before_cutoff"
        yield make_endpoint(row,route,archive=archived if exact else None,geometry=geometry,
                            angle_file=angle["local_file"] if angle else None,archive_status=status)


def index_landsat(rows):
    selected = defaultdict(dict)
    pan_days = defaultdict(list)
    def order(row):
        return timestamp(row["captured_at_utc"]),row["clear_context_qa_cells"],row["product_id"]
    for row in rows:
        key,route = slot(row),row["route"]
        status = row["archive_status"]
        if route==PAN and status in ("before_cutoff","exact_inputs_after_cutoff"):
            pan_days[key+(row["captured_at_utc"][:10],)].append(row)
        if status!="before_cutoff":
            continue
        current = selected[key].get(route)
        if current is None or order(row)>order(current):
            selected[key][route] = row
    return selected,pan_days


def add_sentinel(db,selected,pan_days):
    matches = {}
    for raw in db.execute(SENTINEL):
        r = dict(raw)
        key = slot(r)
        pans = pan_days.get(key+(r["captured_at_utc"][:10],),())
        if r["rank"]!=1 and not pans:
            continue
        sentinel = make_endpoint(r,L1C,archive=r["archive_available_at_utc"],geometry=r["geometry_available_at_utc"])
        if r["rank"]==1:
            selected[key][L1C] = sentinel
        for pan in pans:
            lag = (timestamp(pan["captured_at_utc"])-timestamp(sentinel["captured_at_utc"])).total_seconds()/60
            if abs(lag)>45:
                continue
            match = pan["candidate_id"],pan["product_id"]
            order = pan["archive_status"]!="before_cutoff",abs(lag),sentinel["product_id"],pan["week_ending"],pan["endpoint"]
            previous = matches.get(match)
            if previous is None or order<previous[0]:
                matches[match] = order,pan,sentinel,lag
    return [dict(candidate_id=p["candidate_id"],roof_type=p["roof_type"],terminal_id=p["terminal_id"],diameter_m=p["diameter_m"],
                landsat_product_id=p["product_id"],sentinel2_product_id=s["product_id"],landsat_capture_at_utc=p["captured_at_utc"],
                sentinel2_capture_at_utc=s["captured_at_utc"],landsat_minus_sentinel2_minutes=lag,
                landsat_source_directory=p["source_directory"],sentinel2_source_directory=s["source_directory"],
                landsat_angle_file=p["angle_file"],sentinel2_tile_metadata_path=s["metadata_path"],
                landsat_inputs_available_at_utc=p["geometry_available_at_utc"],
                sentinel2_archive_available_at_utc=s["archive_available_at_utc"],
                sentinel2_geometry_available_at_utc=s["geometry_available_at_utc"],
                both_inputs_before_matching_cutoff=p["archive_status"]=="before_cutoff",
                matching_week_ending=p["week_ending"],matching_endpoint=p["endpoint"],
                matching_release_cutoff_at_utc=p["inputs_available_before_utc"],study_phase=p["study_phase"])
            for _,p,s,lag in (matches[k] for k in sorted(matches))]


def attach_roof_angles(matches):
    fields = ("native_line","native_sample","angle_status","solar_azimuth_deg","solar_zenith_deg",
              "view_azimuth_deg","view_zenith_deg","center_pixel_b8_dn")
    with (DERIVED/"landsat_pan_roof_angles.csv").open(newline="",encoding="utf-8") as stream:
        lookup = {(r["candidate_id"],r["product_id"]):{f"landsat_b8_{k}":r[k] for k in fields} for r in csv.DictReader(stream)}
    return [dict(r,**lookup[r["candidate_id"],r["landsat_product_id"]]) for r in matches]


def later(choices,routes):
    return max((choices[route] for route in routes if route in choices),
               key=lambda r:(timestamp(r["captured_at_utc"]),r["route"]==L1C),default=None)


def pair_rows(selected):
    choices = defaultdict(dict)
    for (week,candidate,endpoint),routes in selected.items():
        choices[week,candidate][endpoint] = routes
    for (week,candidate),windows in sorted(choices.items()):
        if len(windows)!=2:
            continue
        earlier,later_window = (windows[k] for k in ("previous","current"))
        previous_10_15,current_10_15 = (later(c,(L1C,PAN)) for c in (earlier,later_window))
        previous_any,current_any = (later(c,ROUTES) for c in (earlier,later_window))
        reference = current_any
        result = dict(week_ending=week,candidate_id=candidate,study_phase=reference["study_phase"],
                      release_date=reference["release_date"],roof_type=reference["roof_type"],
                      terminal_id=reference["terminal_id"],diameter_m=reference["diameter_m"],
                      paired_l1c=L1C in earlier and L1C in later_window,paired_landsat_pan=PAN in earlier and PAN in later_window,
                      paired_10_15m_inputs=bool(previous_10_15 and current_10_15),paired_any_context_inputs=True)
        for name,a,b in (("l1c",earlier.get(L1C),later_window.get(L1C)),("landsat_pan",earlier.get(PAN),later_window.get(PAN)),
                         ("combined_10_15m",previous_10_15,current_10_15),("including_30m_context",previous_any,current_any)):
            for endpoint,row in (("previous",a),("current",b)):
                for field in ("route","product_id","captured_at_utc","hours_before_eia_measurement"):
                    result[f"{name}_{endpoint}_{field}"] = row[field] if row else None
        yield result


def weekly_coverage(db,pairs):
    grouped = defaultdict(list)
    for row in pairs:
        if row["roof_type"]=="EFR":
            grouped[row["week_ending"]].append(row)
    for week,release,phase in db.execute("""SELECT week_ending,release_date,study_phase FROM eia_satellite_window
        WHERE endpoint='current' ORDER BY week_ending"""):
        rows = grouped[week]
        old = [r for r in rows if r["paired_l1c"]]
        combined = [r for r in rows if r["paired_10_15m_inputs"]]
        gains = [r["l1c_current_hours_before_eia_measurement"]-r["combined_10_15m_current_hours_before_eia_measurement"] for r in old]
        gains_previous = [r["l1c_previous_hours_before_eia_measurement"]-r["combined_10_15m_previous_hours_before_eia_measurement"] for r in old]
        yield dict(week_ending=week,release_date=release,study_phase=phase,efr_paired_sentinel2=len(old),
            efr_paired_landsat_pan=sum(r["paired_landsat_pan"] for r in rows),efr_paired_10_15m_inputs=len(combined),
            efr_added_by_15m_pan=len(combined)-len(old),efr_improved_current_endpoint=sum(g>0 for g in gains),
            efr_improved_current_endpoint_at_least_24h=sum(g>=24 for g in gains),
            efr_improved_previous_endpoint_at_least_24h=sum(g>=24 for g in gains_previous),
            max_current_endpoint_improvement_hours=max(gains,default=None),
            efr_paired_including_30m_context=len(rows),efr_added_only_by_context=len(rows)-len(combined))


def summarize(weeks,landsat,matches):
    by_phase = defaultdict(list)
    for row in weeks:
        by_phase[row["study_phase"]].append(row)
    phase_counts = {}
    for phase,rows in by_phase.items():
        keys = ("efr_paired_sentinel2","efr_paired_landsat_pan","efr_paired_10_15m_inputs","efr_paired_including_30m_context")
        phase_counts[phase] = dict(weeks_total=len(rows),weeks_with_pairs={k:sum(r[k]>0 for r in rows) for k in keys},
            tank_week_pairs={k:sum(r[k] for r in rows) for k in keys},
            tank_week_pairs_current_endpoint_improved=sum(r["efr_improved_current_endpoint"] for r in rows),
            tank_week_pairs_current_endpoint_improved_24h=sum(r["efr_improved_current_endpoint_at_least_24h"] for r in rows),
            tank_week_pairs_previous_endpoint_improved_24h=sum(r["efr_improved_previous_endpoint_at_least_24h"] for r in rows),
            max_current_endpoint_improvement_hours=max((r["max_current_endpoint_improvement_hours"] for r in rows
                                                         if r["max_current_endpoint_improvement_hours"] is not None),default=None))
    statuses = defaultdict(set)
    for row in landsat:
        statuses[row["route"],row["archive_status"]].add(row["product_id"])
    anchors = [r for r in matches if r["roof_type"]=="EFR"]
    before_cutoff = [r for r in anchors if r["both_inputs_before_matching_cutoff"]]
    return dict(phase_coverage=phase_counts,
        clear_landsat_endpoint_options_by_route_and_clock=dict(Counter(f"{r['route']}:{r['archive_status']}" for r in landsat)),
        distinct_landsat_products_by_route_and_clock={f"{k}:{s}":len(v) for (k,s),v in sorted(statuses.items())},
        matched_efr_landsat_sentinel2_calibration_rows=len(anchors),
        matched_efr_landsat_products=len({r["landsat_product_id"] for r in anchors}),
        matched_efr_landsat_rows_before_cutoff=len(before_cutoff),
        matched_efr_landsat_products_before_cutoff=len({r["landsat_product_id"] for r in before_cutoff}),
        scope="Observation coverage only. The 15 m B8 archive has an exact source clock and angle coefficients. "
              "Mixed coverage requires cross-sensor validation of shadow measurements. 30 m context routes do not enter 10–15 m coverage. "
              "Retrospective roof attribution is retained separately from satellite archive timestamps. No EIA outcome values are selected.")


def main():
    with (DERIVED/"landsat_pan_angle_assets.csv").open(newline="",encoding="utf-8") as stream:
        angles = {row["product_id"]:row for row in csv.DictReader(stream)}
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        landsat = list(landsat_rows(db,angles))
        print(f"Screened {len(landsat)} Landsat endpoint options",flush=True)
        selected,pan_days = index_landsat(landsat)
        matches = attach_roof_angles(add_sentinel(db,selected,pan_days))
        print(f"Found {len(matches)} same-day cross-sensor tank matches",flush=True)
        pairs = list(pair_rows(selected))
        weeks = list(weekly_coverage(db,pairs))
    export("landsat_eia_endpoint_options",landsat)
    export("landsat_eia_endpoint_selection",[r for key in sorted(selected) for _,r in sorted(selected[key].items())])
    export("landsat_eia_tank_week_coverage",pairs)
    export("landsat_eia_weekly_coverage",weeks)
    export("landsat_sentinel2_pan_calibration_matches",matches)
    report = summarize(weeks,landsat,matches)
    write(DERIVED/"landsat_eia_coverage_summary.json",json.dumps(report,indent=2).encode())
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
