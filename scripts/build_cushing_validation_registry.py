"""Link reported levels, source scenes and independent height controls without treating them as inventory truth."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_public import ROOT, write

STEM = "cushing_validation_registry"
CHICAGO = ZoneInfo("America/Chicago")
PLAIN = "cushing_okeis_plains_25c6"
SOUTH = "cushing_okeis_southbow"
WEBFIRE = "cushing_operator_inspection"


def make_event(source, terminal, equipment, day, clock, state, level, candidates, doc, pages, url, path, sha,
               *, roof=None, volume=None, location_confirmed=False, notes=None):
    local = date.fromisoformat(day)
    start = datetime.combine(local, time.fromisoformat(clock) if clock else time(), CHICAGO)
    # Minute precision is retained when the field notes provide HH:MM.
    end = start+timedelta(minutes=1) if clock else datetime.combine(local+timedelta(days=1), time(), CHICAGO)
    event_id = f"{source}:{terminal}:{equipment or doc}:{day}"
    gaps = []
    if not location_confirmed:
        gaps.append("individual_equipment_to_contour")
    if volume is None:
        if not clock:
            gaps.append("local_measurement_clock")
        if level is not None:
            gaps.extend(("gauge_zero_datum", "roof_to_liquid_offset", "tank_calibration_table"))
        else:
            gaps.append("numeric_gauge_or_stock_measurement")
    else:
        gaps.append("stock_measurement_clock_and_basis")
    return dict(event_id=event_id, source_family=source, terminal_id=terminal, equipment_id=equipment, local_date=day,
                reported_time_local=clock, clock_basis=("incident_date_only_stock_clock_unknown" if volume is not None else
                    "reported_operator_local_minute" if clock else "local_date_only"),
                possible_measurement_start_utc=start.astimezone(timezone.utc).isoformat() if volume is None else None,
                possible_measurement_end_exclusive_utc=end.astimezone(timezone.utc).isoformat() if volume is None else None,
                state=state, reported_liquid_level_m=level, reported_stock_bbl=volume, roof_type=roof,
                possible_candidate_ids_json=json.dumps(sorted(candidates)), possible_candidates=len(candidates),
                individual_contour_verified=location_confirmed, gauge_datum_verified=False, calibration_table_verified=False,
                source_document_id=doc, source_pages=pages, source_url=url, source_path=path, source_sha256=sha,
                original_report_note=notes, missing_for_absolute_validation_json=json.dumps(gaps), admitted_absolute_inventory_label=False)


def events():
    for r in read(f"{WEBFIRE}_levels"):
        yield make_event("epa_webfire", r["terminal_id"], r["equipment_id"], r["inspection_local_date"], r["inspection_time_local"],
                         r["reported_state"], float(r["reported_liquid_level_m"]) if r["reported_liquid_level_m"] else None,
                         json.loads(r["possible_candidate_ids_json"]), r["source_document_id"], r["source_pages"], r["source_url"],
                         r["source_file"], r["source_sha256"], roof="EFR", location_confirmed=r["individual_contour_verified"] == "True")
    r, = read(f"{PLAIN}_ground_record")
    yield make_event("oklahoma_deq", r["terminal_id"], r["equipment_id"], r["source_local_date"], r["source_local_time"],
                     "measured_level", float(r["liquid_level_m"]), json.loads(r["possible_candidate_ids_json"]),
                     r["source_document_id"], f"{r['source_form_pdf_page']}|{r['source_field_notes_pdf_page']}",
                     r["source_url"], r["source_pdf"], r["source_pdf_sha256"], roof="EFR", notes=r["level_raw"])
    for r in read(f"{SOUTH}_ground_levels"):
        yield make_event("oklahoma_deq", r["terminal_id"], r["equipment_id"], r["source_local_date"], r["source_local_time"],
                         "measured_level", float(r["operator_liquid_level_m"]), json.loads(r["possible_candidate_ids_json"]),
                         r["source_document_id"], r["source_pdf_page"], r["source_url"], r["source_pdf"], r["source_pdf_sha256"], roof=r["roof_type"])
    for r in read("cushing_okeis_operator_events"):
        if r["event_kind"] != "observed_roof_landing_then_refill":
            continue
        yield make_event("oklahoma_deq", r["terminal_id"], r["equipment_id"], r["observed_event_local_date"], r["actual_event_local_time"],
                         r["event_kind"], None, json.loads(r["possible_candidate_ids_json"]), r["source_document_id"], r["source_page"],
                         r["source_url"], r["source_pdf"], r["source_sha256"], roof="EFR", notes=f"Reported duration {r['reported_duration_hours']} h")
    reports = [r for r in read("cushing_nrc_incidents_reports") if r["reported_amount_in_tank"]]
    original, = [r for r in reports if not r["follow_up_to_report_id"]]
    if any(float(r["reported_amount_in_tank"]) != float(original["reported_amount_in_tank"]) or r["amount_in_tank_unit"] != "BARREL(S)" for r in reports):
        raise ValueError("NRC inventories or reported units differ within the event")
    source, = [r for r in read("cushing_nrc_incidents_sources") if r["workbook_year"] == original["workbook_year"]]
    stamp = datetime.strptime(original["incident_local_time_as_reported"], "%m/%d/%Y %H:%M")
    roofs = [r["candidate_id"] for r in read("cushing_nrc_rose_rock_landsat_lidar_2022_inventory_candidates")
             if r["capacity_at_least_reported_inventory"] == "True"]
    yield make_event("uscg_nrc", "semgroup_cushing", None, stamp.date().isoformat(), None, "reported_stock_in_tank", None,
                     roofs, original["report_id"], f"INCIDENTS row {original['incidents_excel_row']}",
                     source["source_url"], source["source_file"], source["source_sha256"], roof="EFR candidate", volume=float(original["reported_amount_in_tank"]),
                     notes=f"Incident time {original['incident_local_time_as_reported']}; related reports {','.join(r['report_id'] for r in reports)}")


def optical_inputs():
    for r in read(f"{WEBFIRE}_optical_fit_candidates"):
        yield ("epa_webfire", r["terminal_id"], r["equipment_id"], r["inspection_local_date"]), r["candidate_id"], r["version_id"], r["capture_utc"], "sentinel2_b04", r["geometry_mode"], r["fit_status"], r["estimated_gap_m"], f"{WEBFIRE}_optical_fit_candidates"
    sources = {r["product_id"]: r["captured_at_utc"] for r in read(f"{WEBFIRE}_b8_scenes")}
    for r in read(f"{WEBFIRE}_b8_matches"):
        if r["product_id"] in sources:
            yield ("epa_webfire", r["terminal_id"], r["equipment_id"], r["inspection_local_date"]), r["candidate_id"], r["product_id"], sources[r["product_id"]], "landsat_b8", "dated_2011_lidar", r["fit_status"], r["gap_m"], f"{WEBFIRE}_b8_matches"
    r, = read(f"{PLAIN}_ground_record")
    key = "oklahoma_deq", r["terminal_id"], r["equipment_id"], r["source_local_date"]
    for s in read(f"{PLAIN}_comparisons"):
        yield key, s["candidate_id"], s["product_id"], s["capture_at_utc"], s["sensor"], "initial_geometry", s["fit_status"], s["fitted_gap_m"], f"{PLAIN}_comparisons"
    for s in read(f"{PLAIN}_detector_recovery_single_scene_fits"):
        yield key, s["candidate_id"], s["version_id"], s["capture_at_utc"], f"sentinel2_{s['method']}", "recovered_detector_angles", s["fit_status"], s["gap_m"], f"{PLAIN}_detector_recovery_single_scene_fits"
    for s in read(f"{PLAIN}_shell_ring_optical"):
        if s["reference_kind"] == "conditional_operator_liquid_level" and s["test"] == "fine_grid_upper_ring_xy_z":
            yield key, s["candidate_id"], s["version_id"], s["capture_at_utc"], s["band"], s["test"], s["fit_status"], s["gap_m"], f"{PLAIN}_shell_ring_optical"
    by_contour = defaultdict(list)
    for r in read(f"{SOUTH}_ground_levels"):
        if r["optically_visible_roof"] == "True":
            for k in json.loads(r["possible_candidate_ids_json"]):
                by_contour[k].append(("oklahoma_deq", r["terminal_id"], r["equipment_id"], r["source_local_date"]))
    for r in read("cushing_lidar_shell_ring_optical_south_bow_fits"):
        for key in by_contour[r["candidate_id"]]:
            yield key, r["candidate_id"], r["product_id"], r["captured_at_utc"], "sentinel2_b04", r["variant"], r["fit_status"], r["estimated_gap_m"], "cushing_lidar_shell_ring_optical_south_bow_fits"
    nrc = "cushing_nrc_rose_rock_landsat_lidar_2022"
    candidates = {r["candidate_id"] for r in read(f"{nrc}_inventory_candidates") if r["capacity_at_least_reported_inventory"] == "True"}
    nrc_key = "uscg_nrc", "semgroup_cushing", None, "2024-06-22"
    for r in read(f"{nrc}_spectral_fits"):
        if r["candidate_id"] in candidates:
            for method in ("red", "nir", "joint"):
                yield nrc_key, r["candidate_id"], r["version_id"], r["captured_at_utc"], f"sentinel2_{method}", "dated_2022_lidar", r[f"{method}_fit_status"], r[f"{method}_gap_m"], f"{nrc}_spectral_fits"
    for r in read(f"{nrc}_fits"):
        if r["candidate_id"] in candidates:
            yield nrc_key, r["candidate_id"], r["product_id"], r["optical_capture_at_utc"], "landsat_b8", "dated_2022_lidar", r["fit_status"], r["estimated_gap_m"], f"{nrc}_fits"


def optical_links(records):
    index = {(r["source_family"], r["terminal_id"], r["equipment_id"], r["local_date"]): r for r in records}
    candidates = {r["event_id"]: set(json.loads(r["possible_candidate_ids_json"])) for r in records}
    for key, candidate, product, stamp, sensor, variant, status, gap, table in optical_inputs():
        r = index[key]
        if candidate not in candidates[r["event_id"]]:
            raise ValueError(f"Optical observation does not match an equipment candidate for {r['event_id']}")
        capture = datetime.fromisoformat(stamp)
        day_only_stock = r["source_family"] == "uscg_nrc"
        if not day_only_stock:
            start, end = (datetime.fromisoformat(r[k]) for k in ("possible_measurement_start_utc", "possible_measurement_end_exclusive_utc"))
        yield dict(event_id=r["event_id"], candidate_id=candidate, sensor=sensor, product_id=product,
                   capture_at_utc=capture.isoformat(), local_date_offset=(capture.astimezone(CHICAGO).date()-date.fromisoformat(r["local_date"])).days,
                   minimum_time_separation_hours=max(0., (start-capture).total_seconds(), (capture-end).total_seconds())/3600 if not day_only_stock else None,
                   maximum_time_separation_hours=max(abs((capture-start).total_seconds()), abs((capture-end).total_seconds()))/3600 if not day_only_stock else None,
                   fit_status=status, roof_gap_m=float(gap) if gap else None, model_variant=variant,
                   source_table=f"data/derived/{table}.csv", same_instant_verified=False, admitted_absolute_inventory_label=False)


def controls():
    sources = (
        ("shadow_geometry_estimates", "USGS airborne lidar", "Sentinel-2 B04", "version_id", "optical_capture_at_utc", "lidar_gap_m", "estimated_gap_m", "maximum_sensor_gap_hours", "fit_status", lambda r: r["selected_for_pass_metrics"] == "True" and r["roof_type"] == "EFR"),
        ("cushing_icesat2_native_optical_controls", "ICESat-2 ATL03", "Sentinel-2 B04", "sentinel2_version_id", "sentinel2_captured_at_utc", "icesat_wall_to_roof_m", "sentinel2_estimated_gap_m", "max_time_separation_hours", None, lambda r: True),
        ("cushing_icesat2_esri_b8_comparisons", "ICESat-2 ATL03", "Landsat B8 Esri reflectance", "product_id", "landsat_capture_at_utc", "icesat_gap_m", "b8_gap_m", "maximum_separation_hours", "fit_status", lambda r: True),
        ("cushing_gedi_native_optical_fits", "GEDI native waveform", "Sentinel-2 B04", "sentinel2_version_id", "sentinel2_capture_utc", "gedi_corrected_wall_to_roof_m", "b04_wall_to_roof_m", "sentinel2_minus_gedi_time_hours", "fit_status", lambda r: True),
        ("cushing_naip_satellite_controls_pairs", "USDA NAIP aerial shadow", "per source row", "satellite_version", "satellite_captured_utc", "naip_gap_m", "satellite_gap_m", "maximum_separation_hours", "satellite_fit_status", lambda r: r["nearest_before_quality_filter"] == "True"),
        (f"{PLAIN}_lidar_reference_comparisons", "USGS airborne lidar", "per source row", "version_id", "optical_capture_utc", "lidar_gap_m", "optical_gap_m", "maximum_possible_separation_hours", "fit_status", lambda r: True),
    )
    for stem, reference, sensor, product, capture, reference_gap, optical_gap, hours, status, selected in sources:
        for i, r in enumerate(read(stem), 2):
            if not selected(r) or not r[reference_gap]:
                continue
            fit = r[status] if status else RESOLVED
            reference_pass = not r.get("naip_fit_status") or r["naip_fit_status"] == "resolved_aerial_shadow"
            a, b = float(r[reference_gap]), float(r[optical_gap]) if r[optical_gap] else None
            yield dict(candidate_id=r["candidate_id"], reference_type=reference, sensor=r.get("satellite_sensor", r.get("method", sensor)),
                       product_id=r[product], capture_at_utc=r[capture], reference_gap_m=a, optical_gap_m=b, fit_status=fit,
                       reference_diagnostic_pass=reference_pass,
                       max_time_separation_hours=abs(float(r[hours])) if r[hours] else None,
                       paired_height_difference_m=b-a if b is not None and fit == RESOLVED and reference_pass else None,
                       source_table=f"data/derived/{stem}.csv", source_csv_row=i, admitted_absolute_inventory_label=False)


def main():
    ground = list(events())
    if len({r["event_id"] for r in ground}) != len(ground):
        raise ValueError("Duplicate ground observation")
    checked = {}
    for r in ground:
        path = ROOT/Path(r["source_path"])
        if path not in checked:
            with path.open("rb") as stream:
                checked[path] = hashlib.file_digest(stream, "sha256").hexdigest()
        if checked[path] != r["source_sha256"]:
            raise ValueError(f"Original ground evidence changed: {path}")
    links, heights = list(optical_links(ground)), list(controls())
    for suffix, records in (("operator_events", ground), ("candidate_optical_links", links), ("height_comparisons", heights)):
        export(f"{STEM}_{suffix}", records)
    resolved = {r["event_id"] for r in links if r["fit_status"] == RESOLVED and r["local_date_offset"] == 0}
    usable_heights = [r for r in heights if r["paired_height_difference_m"] is not None]
    report = dict(ground_events=len(ground), verified_original_ground_documents=len(checked), ground_state=Counter(r["state"] for r in ground),
                  numeric_liquid_levels=sum(r["reported_liquid_level_m"] is not None for r in ground),
                  numeric_levels_with_a_reported_minute=sum(r["reported_liquid_level_m"] is not None and bool(r["reported_time_local"]) for r in ground),
                  possible_optical_links=len(links), ground_events_with_same_local_day_resolved_optical_fit=len(resolved),
                  height_comparisons=len(heights), height_comparisons_by_reference=Counter(r["reference_type"] for r in heights),
                  height_comparisons_passing_both_diagnostics_by_reference=Counter(r["reference_type"] for r in usable_heights),
                  qualified_height_comparisons_within_24h=sum(r["max_time_separation_hours"] is not None and r["max_time_separation_hours"] <= 24 for r in usable_heights),
                  admitted_absolute_inventory_labels=sum(r["admitted_absolute_inventory_label"] for r in ground))
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
