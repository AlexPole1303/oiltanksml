"""Verify public geometry evidence and isolate its effect on cylindrical volume changes."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pdfplumber

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write

STEM = "cushing_tank_volume_geometry"
FEET_TO_M = .3048
BARREL_M3 = .158987294928
DOCUMENTS = {
    "2017-03-09": ("4182", "deq_wildhorse_2017.pdf", 6, "proposed_project_landing_emissions"),
    "2025-01-03": ("8910", "8910.pdf", 4, "draft_construction_memo_as_built_roster"),
    "2025-03-18": ("8978", "deq_wildhorse_2025_titlev.pdf", 4, "draft_operating_memo_as_built_roster"),
}
WEEKLY_SOURCES = {
    "development_and_validation": "sentinel2_weekly_tank_volumes",
    "holdout": "sentinel2_holdout_weekly_tank_volumes",
}
ALTERNATES = ("mapped_polygon", "lidar_2022", "permit_or_emissions_diameter")


def source_pdfs():
    evidence, equipment, landing = [], {}, []
    for day,(number,name,page,role) in DOCUMENTS.items():
        url = f"https://applications.deq.ok.gov/permitspublic/storedpermits/{number}.pdf"
        path = RAW/"documents"/"source_pdfs"/name
        content = path.read_bytes() if path.exists() else get(url)
        if not content.startswith(b"%PDF-"):
            raise ValueError(f"Oklahoma DEQ did not return PDF {number}")
        if not path.exists():
            write(path, content)
        provenance = dict(document_id=number, memo_date=day, source_role=role, source_pdf_page=page,
                          source_url=url, source_file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(content).hexdigest())
        evidence.append(provenance)
        with pdfplumber.open(path) as pdf:
            sheet = pdf.pages[page-1]
            if number == "4182":
                text = sheet.extract_text() or ""
                if "Table 3. Roof Landing Emissions" not in text:
                    raise ValueError("Missing Wildhorse project-design landing table")
                for tank, area, volume in re.findall(r"^(T-\d+) ([\d,]+\.\d+) ([\d,]+) \d",text,re.M):
                    area, volume = (float(v.replace(",","")) for v in (area,volume))
                    landing.append(dict(tank_id=tank, modeled_bottom_area_ft2=area, modeled_vapor_volume_ft3=volume,
                                        equivalent_vapor_space_height_ft=volume/area, **provenance))
                if len(landing)!=19:
                    raise ValueError("Incomplete Wildhorse project-design landing table")
                continue
            parsed = {}
            for table in sheet.extract_tables():
                header = [re.sub(r"\s+"," ",v or "").strip() for v in table[0]]
                if header!=["EU ID#","Contents","Roof Type","Bottom Design","Capacity (Nominal) (bbl)",
                            "Working Height (ft)","Diameter (ft)","Construction Date"]:
                    continue
                for tank,product,roof,bottom,capacity,height,diameter,built in table[1:]:
                    if tank in parsed or product!="Crude Oil" or roof not in ("EFR","IFR"):
                        raise ValueError(f"Unexpected Wildhorse equipment row {tank} in {number}")
                    parsed[tank] = dict(tank_id=tank, stored_product=product, roof_type=roof,
                                        bottom_design=re.sub(r"\s+","",bottom).lower(), nominal_capacity_bbl=int(capacity.replace(",","")),
                                        reported_working_height_ft=float(height), permit_diameter_ft=float(diameter),
                                        reported_construction_date=datetime.strptime(built,"%m/%d/%Y").date().isoformat())
            if Counter(r["roof_type"] for r in parsed.values())!={"EFR":12,"IFR":1}:
                raise ValueError(f"Missing Wildhorse as-built rows for {number}")
            equipment[number] = parsed,provenance
    a,b = equipment["8910"],equipment["8978"]
    if a[0] != b[0]:
        raise ValueError("Wildhorse construction and operating memo dimensions differ")
    return evidence,[r | p for rows,p in equipment.values() for r in rows.values()],landing,a[0]


def circular_scale(diameter):
    return math.pi*diameter**2/(4*BARREL_M3)


def estimate_fields(r, legacy, modern, emissions):
    key = r["candidate_id"]
    li = legacy.get(key,{})
    md = modern.get(key,{})
    em = emissions.get(key,{})
    nominal = float(r["permit_diameter_m"]) if r["permit_diameter_m"] else None
    document_kind = "engineering_permit_dimension" if nominal else None
    if em.get("emissions_diameter_m"):
        if nominal is not None:
            raise ValueError(f"Both nominal and emissions diameters for {key}; review their definitions")
        nominal = float(em["emissions_diameter_m"])
        document_kind = "emissions_model_parameter"
    d2011 = float(li["diameter_m"]) if li else None
    d2022 = float(r["lidar_upper_outline_diameter_m"]) if r["lidar_circular_area_bbl_per_m"] else None
    fields = dict(candidate_id=key, terminal_id=r["documented_terminal_id"], cohort_id=r["cohort_id"],
                  mapping_basis=r["evidence_basis"], possible_equipment_ids_json=r["possible_equipment_ids_json"],
                  mapped_polygon_bbl_per_m=float(r["footprint_bbl_per_m"]), mapped_polygon_source_file=r["footprint_source_file"],
                  lidar_2011_diameter_m=d2011,lidar_2011_bbl_per_m=circular_scale(d2011) if d2011 else None,
                  lidar_2011_source_directory=li.get("source_directory"), lidar_2011_fit_status=li.get("fit_status"),
                  lidar_2011_outline_observed_utc=li.get("first_upper_return_at_utc"),
                  lidar_2022_diameter_m=d2022,lidar_2022_bbl_per_m=circular_scale(d2022) if d2022 else None,
                  lidar_2022_outline_observed_utc=r["first_upper_return_at_utc"],
                  lidar_2022_fit_status=r["lidar_fit_status"],lidar_2022_geometry_availability_date=md.get("lidar_source_available_date"),
                  lidar_2022_roof_geometry_input_status=md.get("modern_input_status"),
                  permit_or_emissions_diameter_m=nominal,permit_or_emissions_diameter_bbl_per_m=circular_scale(nominal) if nominal else None,
                  diameter_document_kind=document_kind,
                  diameter_document_date=(em.get("application_received_date") or r["source_document_date"]) if nominal else None,
                  diameter_document_file=(em.get("source_file") or r["engineering_source_file"]) if nominal else None,
                  diameter_document_url=(em.get("source_url") or r["engineering_source_url"]) if nominal else None,
                  reported_height_min_m=r["reported_height_min_m"],reported_height_max_m=r["reported_height_max_m"],
                  reported_height_basis=r["reported_height_basis"],reported_capacity_min_bbl=r["reported_capacity_min_bbl"],
                  reported_capacity_max_bbl=r["reported_capacity_max_bbl"],reported_capacity_basis=r["reported_capacity_basis"],
                  working_height_documented=r["reported_height_basis"]=="permit_working_height",
                  observed_roof_floating_state_verified=False, liquid_height_datum_and_strapping_verified=False)
    if d2022 and not math.isclose(fields["lidar_2022_bbl_per_m"],float(r["lidar_circular_area_bbl_per_m"]),rel_tol=1e-6):
        raise ValueError(f"LiDAR cylinder arithmetic does not reconcile for {key}")
    return fields


def wildhorse_match(geometry, equipment):
    for r in geometry.values():
        if r["terminal_id"]!="wildhorse":
            continue
        candidates = json.loads(r["possible_equipment_ids_json"])
        d = r["permit_or_emissions_diameter_m"]
        group = [tank for tank in equipment.values() if tank["roof_type"]=="EFR" and math.isclose(tank["permit_diameter_ft"]*FEET_TO_M,d)]
        if not group or {tank["tank_id"] for tank in group}!={"T-"+str(value).removeprefix("T-") for value in candidates}:
            raise ValueError(f"Wildhorse permit equipment group differs for {r['candidate_id']}")
        heights = [t["reported_working_height_ft"]*FEET_TO_M for t in group]
        caps = {t["nominal_capacity_bbl"] for t in group}
        cap, = caps
        vol = [circular_scale(d)*h for h in heights]
        yield dict(candidate_id=r["candidate_id"], cohort_id=r["cohort_id"], possible_equipment_ids_json=json.dumps(sorted(t["tank_id"] for t in group)),
                   unique_equipment_mapping=len(group)==1, permit_diameter_m=d,lidar_upper_diameter_m=r["lidar_2022_diameter_m"],
                   diameter_based_bbl_per_m=circular_scale(d), lidar_based_bbl_per_m=r["lidar_2022_bbl_per_m"],
                   lidar_area_above_permit_circle_pct=100 * (r["lidar_2022_bbl_per_m"] / circular_scale(d) - 1),
                   working_height_min_m=min(heights),working_height_max_m=max(heights),reported_nominal_capacity_bbl=cap,
                   cylindrical_volume_at_reported_height_min_bbl=min(vol),cylindrical_volume_at_reported_height_max_bbl=max(vol),
                   cylinder_minus_nominal_min_pct=100*(min(vol)/cap-1),cylinder_minus_nominal_max_pct=100*(max(vol)/cap-1),
                   actual_roof_landing_height_verified=False, strapping_table_verified=False)


def weeks(geometry):
    for study, name in WEEKLY_SOURCES.items():
        grouped = defaultdict(list)
        for r in read(name):
            grouped[r["study_phase"],r["week_ending"],r["release_date"]].append(r)
        for (phase,week,release),rows in sorted(grouped.items()):
            if len({r["candidate_id"] for r in rows})!=len(rows):
                raise ValueError(f"Repeated weekly geometry row for {week}")
            matches = {alt: [] for alt in ALTERNATES}
            for r in rows:
                tank = geometry[r["candidate_id"]]
                scale = circular_scale(float(r["diameter_m"]))
                height = float(r["earlier_gap_m"]) - float(r["later_gap_m"])
                baseline = float(r["capture_change_kbbl"]) * 1000
                if not math.isclose(height * scale, baseline, rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError(f"Existing cylindrical volume does not reconcile for {week}/{r['candidate_id']}")
                for alt, subset in matches.items():
                    value = tank[f"{alt}_bbl_per_m"]
                    if value is not None:
                        subset.append((baseline, height * value))
            for alt, subset in matches.items():
                base, replacement = (sum(x[i] for x in subset) for i in (0,1)) if subset else (None,None)
                yield dict(study_source=study, study_phase=phase, week_ending=week, release_date=release, alternative_area_basis=alt,
                           all_measured_roofs=len(rows), matched_roofs=len(subset), matched_baseline_bbl=base,matched_alternative_bbl=replacement,
                           alternative_minus_baseline_bbl=replacement-base if subset else None,
                           signs_differ_on_same_roofs=bool(subset and base*replacement<0), historical_replay_input=False)


def main():
    evidence, equipment, land, official = source_pdfs()
    legacy = {r["candidate_id"]:r for name in ("core_terminal_legacy_lidar_geometry","terminal_legacy_lidar_geometry") for r in read(name)}
    modern = {r["candidate_id"]:r for r in read("cushing_modern_lidar_optical_registry")}
    emissions = {r["candidate_id"]:r for r in read("tank_emissions_geometry_comparison")}
    geometry = {r["candidate_id"]:estimate_fields(r,legacy,modern,emissions) for r in read("tank_engineering_registry")
                if r["documented_roof_type"]=="EFR"}
    wildhorse = list(wildhorse_match(geometry,official))
    if len(wildhorse)!=12:
        raise ValueError("Incomplete Wildhorse outline group mapping")
    scenarios = list(weeks(geometry))
    for name, rows in (("sources",evidence),("wildhorse_equipment",equipment),("wildhorse_project_landing",land),
                      ("wildhorse_groups",wildhorse),("registry",list(geometry.values())),("weekly_sensitivity",scenarios)):
        export(f"{STEM}_{name}",rows)
    sample = [100*(r["lidar_2022_bbl_per_m"]/r["mapped_polygon_bbl_per_m"]-1) for r in geometry.values() if r["lidar_2022_bbl_per_m"]]
    sensitivity = {}
    for group in sorted({r["study_phase"] for r in scenarios}):
        sensitivity[group] = {}
        for alt in ALTERNATES:
            rows = [r for r in scenarios if r["study_phase"]==group and r["alternative_area_basis"]==alt and r["matched_roofs"]]
            delta = [abs(r["alternative_minus_baseline_bbl"]) for r in rows]
            sensitivity[group][alt] = dict(matched_weeks=len(rows), matched_roof_weeks=sum(r["matched_roofs"] for r in rows),
                full_measured_sample_weeks=sum(r["matched_roofs"]==r["all_measured_roofs"] for r in rows),
                median_abs_difference_bbl=float(np.median(delta)) if delta else None,
                p90_abs_difference_bbl=float(np.percentile(delta,90)) if delta else None,
                max_abs_difference_bbl=max(delta) if delta else None,
                sign_disagreements=sum(r["signs_differ_on_same_roofs"] for r in rows))
    summary = dict(external_floating_roof_registry=len(geometry), lidar_2011_geometry=sum(r["lidar_2011_bbl_per_m"] is not None for r in geometry.values()),
                   lidar_2022_geometry=len(sample), document_diameter=sum(r["permit_or_emissions_diameter_m"] is not None for r in geometry.values()),
                   diameter_document_counts=dict(Counter(r["diameter_document_kind"] for r in geometry.values() if r["diameter_document_kind"])),
                   working_height_documented=sum(r["working_height_documented"] for r in geometry.values()),
                   verified_absolute_capacity_tables=sum(r["liquid_height_datum_and_strapping_verified"] for r in geometry.values()),
                   lidar2022_vs_mapped_polygon_median_absolute_area_difference_pct=float(np.median(np.abs(sample))),
                   lidar2022_vs_mapped_polygon_difference_over_5pct=sum(abs(x)>5 for x in sample),
                   wildhorse_confirmed_equipment_2025=len(official), wildhorse_confirmed_efr_equipment=sum(r["roof_type"]=="EFR" for r in official.values()),
                   wildhorse_efr_nominal_capacity_bbl=sum(r["nominal_capacity_bbl"] for r in official.values() if r["roof_type"]=="EFR"),
                   wildhorse_lidar_area_above_permit_circle_pct=[min(r["lidar_area_above_permit_circle_pct"] for r in wildhorse),
                                                                max(r["lidar_area_above_permit_circle_pct"] for r in wildhorse)],
                   wildhorse_old_project_landing_rows=len(land),
                   wildhorse_old_project_equivalent_vapor_height_ft=[min(r["equivalent_vapor_space_height_ft"] for r in land),max(r["equivalent_vapor_space_height_ft"] for r in land)],
                   sensitivity_by_phase=sensitivity,
                   scope="Sensitivity replaces area only on the same measured roofs with fixed fitted changes in height. "
                         "It describes alternative cylindrical equivalents. It does not infer absolute stock, operating limits, or historical input availability.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
