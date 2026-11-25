"""Freeze weekly optical inputs and build the development-period physical measurements."""
import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

from build_l1c_quality import candidate_layer
from build_shadow_calibration import SCHEMA, add_psf, add_scene
from build_shadow_change_controls import SELECT_QUALITY, scale
from build_tank_geometry_registry import TERMINAL_GEOMETRY_STUDY
from collect_public import ROOT, write, write_csv
from fit_shadow_geometry import FIELDS, estimate, inputs, read_only

DERIVED = ROOT/"data/derived"
SOURCE = DERIVED/"cushing.sqlite"
PIXELS = DERIVED/"weekly_shadow_development.sqlite"
RESOLVED = "resolved_in_geometry_diagnostic"
WEEKLY_ELIGIBLE_INPUTS = """
    SELECT w.week_ending,w.endpoint,w.study_phase,w.release_date,w.inputs_available_before_utc,
        w.window_starts_after_utc,w.window_ends_by_utc,i.canonical_l2a_id,i.version_id,i.captured_at_utc,
        i.archive_available_at_utc,i.geometry_available_at_utc,i.availability_basis
    FROM eia_satellite_window_input i JOIN eia_satellite_window w USING(week_ending,endpoint)
    WHERE i.level='L1C' AND i.local_exact_version_screened
        AND julianday(i.captured_at_utc)>julianday(w.window_starts_after_utc)
        AND julianday(i.captured_at_utc)<=julianday(w.window_ends_by_utc)
        AND julianday(i.archive_available_at_utc)<julianday(w.inputs_available_before_utc)
        AND julianday(i.geometry_available_at_utc)<julianday(w.inputs_available_before_utc)
        AND (SELECT count(*) FROM sentinel2_band_radiometry r WHERE r.level='L1C' AND r.version_id=i.version_id
             AND r.band IN ('B02','B03','B04','B08') AND r.reflectance_scale IS NOT NULL AND r.reflectance_offset IS NOT NULL
             AND r.nodata_dn IS NOT NULL AND r.saturated_dn IS NOT NULL)=4
"""


def endpoint_ctes(geometry_filter="", *, eligible_inputs=WEEKLY_ELIGIBLE_INPUTS):
    return f"""
WITH eligible_inputs AS MATERIALIZED (
    {eligible_inputs}
), geometry AS MATERIALIZED (
    SELECT t.candidate_id,t.documented_roof_type,t.cohort_id,t.documented_terminal_id,t.source_document_date,t.engineering_source_url,
        coalesce(l.diameter_m,c.diameter_m,t.footprint_diameter_m) AS diameter_m,
        CASE WHEN l.diameter_m IS NOT NULL THEN 'lidar_elevated_circle'
             WHEN c.diameter_m IS NOT NULL THEN 'lidar_covered_surface_circle' ELSE 'mapped_area_circle' END AS geometry_basis
    FROM tank_engineering_registry t LEFT JOIN tank_lidar_outline l USING(candidate_id)
    LEFT JOIN enbridge_main_covered_roof_outline c USING(candidate_id)
    WHERE t.documented_roof_type IS NOT NULL AND coalesce(l.diameter_m,c.diameter_m,t.footprint_diameter_m)>0
        {geometry_filter}
), qualified AS MATERIALIZED (
    SELECT i.week_ending,i.endpoint,i.study_phase,i.release_date,i.inputs_available_before_utc,
        i.window_starts_after_utc,i.window_ends_by_utc,i.canonical_l2a_id,i.version_id,i.captured_at_utc,
        i.archive_available_at_utc,i.geometry_available_at_utc,i.availability_basis,q.candidate_id,
        q.source_directory,q.tile_metadata_path,q.context_valid_cells,t.documented_roof_type AS roof_type,
        t.cohort_id,t.documented_terminal_id AS terminal_id,t.source_document_date,t.engineering_source_url,
        t.diameter_m,t.geometry_basis,
        (SELECT max(a.observed_date) FROM tank_aerial_visual_observation a
         WHERE a.candidate_id=q.candidate_id AND a.observed_date<substr(i.captured_at_utc,1,10)) AS prior_aerial_date
    FROM eligible_inputs i CROSS JOIN sentinel2_l1c_tank_quality q ON q.scene_item_id=i.canonical_l2a_id AND q.l1c_product_name=i.version_id
    CROSS JOIN geometry t ON t.candidate_id=q.candidate_id
    WHERE q.optical_cells_valid_in_all_bands=q.optical_touched_cells_expected
        AND q.opaque_roof_cells=0 AND q.cirrus_roof_cells=0 AND coalesce(q.snow_ice_roof_cells,0)=0
        AND q.red_detector_covered_roof_cells=q.optical_touched_cells_expected AND q.red_detector_ambiguous_roof_cells=0
        AND json_array_length(q.red_detector_with_view_options_json)=1
), ranked AS (
    SELECT q.*,count(*) OVER (PARTITION BY q.week_ending,q.endpoint,q.candidate_id) AS eligible_products,
        row_number() OVER (PARTITION BY q.week_ending,q.endpoint,q.candidate_id
             ORDER BY substr(q.captured_at_utc,1,10) DESC,q.context_valid_cells DESC,q.captured_at_utc DESC,q.version_id) AS rank
    FROM qualified q
    WHERE (SELECT count(*) FROM tank_aerial_visual_observation a
           WHERE a.candidate_id=q.candidate_id AND a.observed_date=q.prior_aerial_date)>0
      AND NOT EXISTS (SELECT 1 FROM tank_aerial_visual_observation a WHERE a.candidate_id=q.candidate_id
           AND a.observed_date=q.prior_aerial_date AND a.structure_observation IS NOT 'aboveground_tank_seen')
)
"""

ENDPOINT_CTES = endpoint_ctes()
PAIRED_ENDPOINTS = """
, paired AS (
    SELECT week_ending,candidate_id FROM ranked WHERE rank=1 GROUP BY week_ending,candidate_id HAVING count(*)=2
)
SELECT r.* FROM ranked r JOIN paired p USING(week_ending,candidate_id)
WHERE r.rank=1 ORDER BY r.week_ending,r.candidate_id,r.endpoint
"""


def export(name, rows):
    fields = tuple(rows[0])
    write_csv(DERIVED/f"{name}.csv",fields,([r.get(k) for k in fields] for r in rows))


def catalog(source):
    markers = ",".join("?" for _ in TERMINAL_GEOMETRY_STUDY)
    sql = endpoint_ctes(f"AND (t.documented_terminal_id IS NULL OR t.documented_terminal_id NOT IN ({markers}))")
    endpoints = [dict(row) for row in source.execute(sql + PAIRED_ENDPOINTS, TERMINAL_GEOMETRY_STUDY)]
    pairs = defaultdict(dict)
    for row in endpoints:
        row.pop("rank")
        pairs[row["week_ending"],row["candidate_id"]][row["endpoint"]] = row
    export("weekly_shadow_endpoint_manifest",endpoints)
    return list(pairs.values())


def phase_observations(source, pairs, phase):
    keys = {(r["candidate_id"],r["version_id"],r["canonical_l2a_id"]) for p in pairs for r in p.values()
            if r["study_phase"]==phase}
    wanted = defaultdict(set)
    for candidate,version,scene in keys:
        wanted[version,scene].add(candidate)
    groups = defaultdict(list)
    for (version,scene),candidates in sorted(wanted.items()):
        rows = source.execute(SELECT_QUALITY+"AND q.l1c_product_name=? AND q.scene_item_id=? ORDER BY q.candidate_id",(version,scene))
        for row in rows:
            if row["candidate_id"] in candidates:
                groups[row["level"],row["version_id"],row["optical_source_directory"]].append(row)
    if not keys or sum(map(len,groups.values()))!=len(keys):
        raise ValueError(f"{phase} raster selection differs from the frozen endpoint manifest")
    return groups


def build_pixels(source,pairs,output=PIXELS,*,phase="development"):
    groups = phase_observations(source,pairs,phase)
    temporary = output.with_suffix(".part")
    temporary.unlink(missing_ok=True)
    candidates = dict((*candidate_layer("osm"),*candidate_layer("supplemental")))
    identities = {}
    with closing(sqlite3.connect(temporary)) as db,db:
        db.execute("PRAGMA foreign_keys=ON")
        db.executescript(SCHEMA)
        add_psf(db)
        for number,(key,rows) in enumerate(sorted(groups.items()),1):
            add_scene(db,source,number,key,rows,candidates,identities)
            if number%50==0 or number==len(groups):
                print(f"Stored {number}/{len(groups)} {phase} granules",flush=True)
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok" or db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError(f"Weekly {phase} pixel database integrity check failed")
    temporary.replace(output)


def fit_development(source):
    shells = dict(source.execute("SELECT candidate_id,lidar_rim_to_ground_native_z FROM tank_engineering_registry"))
    with closing(read_only(PIXELS)) as db:
        db.row_factory = sqlite3.Row
        observations,kernels = inputs(db)
        scene_ids = dict(db.execute("SELECT scene_id,scene_item_id FROM scene"))
    def fit(o):
        return estimate(o,shells[o["candidate_id"]],kernels[o["platform"]],calibration_db=PIXELS)
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(observation,result) in enumerate(zip(observations,pool.map(fit,observations),strict=True),1):
            rows[result["candidate_id"],result["version_id"],scene_ids[observation["scene_id"]]] = result
            if i%2000==0 or i==len(observations):
                print(f"Fit {i}/{len(observations)} development observations",flush=True)
    write_csv(DERIVED/"weekly_shadow_development_observations.csv",FIELDS,([r.get(k) for k in FIELDS] for r in rows.values()))
    return rows


def pair_rows(pairs,estimates,*,phase="development"):
    for pair in pairs:
        earlier,later = pair["previous"],pair["current"]
        if later["study_phase"]!=phase:
            continue
        a,b = (estimates[r["candidate_id"],r["version_id"],r["canonical_l2a_id"]] for r in (earlier,later))
        previous,current = a.get("estimated_gap_m"),b.get("estimated_gap_m")
        change = previous-current if previous is not None and current is not None else None
        resolved = a["fit_status"]==b["fit_status"]==RESOLVED
        diameter = later["diameter_m"]
        area_m2 = math.pi*diameter**2/4
        yield dict(week_ending=later["week_ending"],release_date=later["release_date"],candidate_id=later["candidate_id"],
            terminal_id=later["terminal_id"],cohort_id=later["cohort_id"],roof_type=later["roof_type"],
            earlier_observation_id=a["observation_id"],later_observation_id=b["observation_id"],
            earlier_version=earlier["version_id"],later_version=later["version_id"],
            earlier_capture_at_utc=earlier["captured_at_utc"],later_capture_at_utc=later["captured_at_utc"],
            inputs_available_before_utc=later["inputs_available_before_utc"],geometry_basis=later["geometry_basis"],
            earlier_status=a["fit_status"],later_status=b["fit_status"],both_shadow_diagnostics_pass=resolved,
            earlier_gap_m=previous,later_gap_m=current,optical_level_change_m=change,
            circular_area_m2=area_m2,circular_bbl_per_m=scale(b),optical_change_equivalent_bbl=change*scale(b) if change is not None else None,
            source_document_date=later["source_document_date"],source_document_dated_before_cutoff=(
                later["source_document_date"]<later["inputs_available_before_utc"][:10] if later["source_document_date"] else None))


def aggregate(rows):
    filled = [r for r in rows if r["optical_level_change_m"] is not None]
    resolved = [r for r in filled if r["both_shadow_diagnostics_pass"]]
    return dict(input_efr_pairs=len(rows),fitted_efr_pairs=len(filled),resolved_efr_pairs=len(resolved),
        paired_circle_area_m2=sum(r["circular_area_m2"] for r in rows),resolved_circle_area_m2=sum(r["circular_area_m2"] for r in resolved),
        fitted_visible_change_equivalent_bbl=sum(r["optical_change_equivalent_bbl"] for r in filled) if filled else None,
        resolved_visible_change_equivalent_bbl=sum(r["optical_change_equivalent_bbl"] for r in resolved) if resolved else None)


def development_tables(source,rows):
    by_week,by_terminal = defaultdict(list),defaultdict(list)
    for r in rows:
        if r["roof_type"]=="EFR":
            by_week[r["week_ending"]].append(r)
            by_terminal[r["week_ending"],r["terminal_id"]].append(r)
    weeks = []
    for target in source.execute("""SELECT week_ending,release_date,cushing_kbbl,change_kbbl_as_known_at_release,
        latest_weekly_change_known_before_release,cushing_printed_precision_kbbl,source_url
        FROM eia_inventory_release_panel WHERE study_phase='development' ORDER BY week_ending"""):
        weeks.append(dict(target)|aggregate(by_week[target["week_ending"]]))
    terminals = [dict(week_ending=w,terminal_id=t,**aggregate(v)) for (w,t),v in sorted(by_terminal.items())]
    export("weekly_shadow_development_weeks",weeks)
    export("weekly_shadow_development_terminals",terminals)
    return weeks


def report(source,pairs,rows,weeks):
    phase_counts = defaultdict(Counter)
    unique_weeks = defaultdict(set)
    for pair in pairs:
        r = pair["current"]
        phase = r["study_phase"]
        phase_counts[phase][r["roof_type"]] += 1
        if r["roof_type"]=="EFR":
            unique_weeks[phase].add(r["week_ending"])
    summary = dict(manifest_pairs_by_phase_and_roof=dict(phase_counts),
        weeks_with_efr_inputs_by_phase={k:len(v) for k,v in unique_weeks.items()},
        all_weeks_by_phase=dict(source.execute("SELECT study_phase,count(*) FROM eia_satellite_window WHERE endpoint='current' GROUP BY 1")),
        development_pair_rows=len(rows),development_efr_fitted_pairs=sum(r["fitted_efr_pairs"] for r in weeks),
        development_efr_resolved_pairs=sum(r["resolved_efr_pairs"] for r in weeks),
        development_weeks_with_resolved_efr=sum(r["resolved_efr_pairs"]>0 for r in weeks),
        development_weeks_with_fitted_efr=sum(r["fitted_efr_pairs"]>0 for r in weeks),
        roof_fit_code_sha256=hashlib.sha256((ROOT/"scripts/fit_shadow_geometry.py").read_bytes()).hexdigest(),
        scope="The full manifest contains satellite availability and prior aerial observation dates. Physical estimates and EIA outcomes are joined only for development. Engineering matching uses retrospective geometry and document dates, which do not establish historical publication of a specification.")
    write(DERIVED/"weekly_shadow_panel_summary.json",json.dumps(summary,indent=2).encode())
    print(summary,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only",action="store_true")
    args = parser.parse_args()
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        pairs = catalog(source)
        print(f"Selected {len(pairs)} exact endpoint pairs across the archive",flush=True)
        if args.manifest_only:
            return
        build_pixels(source,pairs)
        estimates = fit_development(source)
        rows = list(pair_rows(pairs,estimates))
        export("weekly_shadow_development_pairs",rows)
        weeks = development_tables(source,rows)
        report(source,pairs,rows,weeks)


if __name__=="__main__":
    main()
