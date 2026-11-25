"""Record South Bow inspection levels and test comparisons invariant to tank numbering."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta, timezone
from statistics import mean

from shapely.geometry import Point

from build_cushing_inspection_controls import LOCAL_TIME
from build_cushing_okeis_plains_control import scenes, sentinel
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_public import ROOT, write

STEM = "cushing_okeis_southbow"
DAY = date(2026, 4, 16)
DOCUMENT = "-2293699492712251186"
SHA256 = "32d819fa703693180d0cf55100fe5297f8f8cc093b9a33167ed9847d43c58f37"
# Attachment I forms, verified against the page images. Values are in decimal feet.
INSPECTIONS = (
    (1, 35, 10.11, None), (2, 41, 16.89, None), (3, 47, 8.10, None), (4, 52, 7.98, "08:00"),
    (5, 55, 6.36, "09:00"), (6, 59, 23.78, None), (7, 65, 23.66, None), (8, 71, 18.34, None),
    (9, 77, 16.54, None), (10, 83, 23.16, None),
)
# Two versions of Compliance Table 6, transcribed independently from PDF pages 105 and 23.
FLOW_2025 = (8873218, 6810316, 6175897, 6300654, 8329251, 8913658, 7809484, 8102806, 7237005, 10299798, 8678736, 9739594)
ROLL_2025 = (95405142, 94597389, 93445732, 92367006, 92641092, 93427764, 92915136, 91632281, 91020458, 94596365, 96722274, 97270419)
FLOW_LATER = (7809484, 8102806, 7237005, 10299798, 8678736, 9739594, 10088254, 8624422, 9140504, 9659641, 11023484, 10002901)
ROLL_LATER = (92915136, 91632281, 91020458, 94596365, 96722274, 97270419, 98485455, 100299560, 103264167, 106623153, 109317386, 110406628)


def throughput(source_url):
    if FLOW_2025[-6:] != FLOW_LATER[:6] or ROLL_2025[-6:] != ROLL_LATER[:6]:
        raise ValueError("The six overlapping South Bow throughput months disagree")
    volumes = (*FLOW_2025, *FLOW_LATER[6:])
    rolling = (*ROLL_2025, *ROLL_LATER[6:])
    residuals = [volumes[k] - volumes[k-12] - (rolling[k] - rolling[k-1]) for k in range(12, 18)]
    if max(abs(x) for x in residuals) > 1:
        raise ValueError("Reported South Bow 12-month totals fail monthly recurrence")
    return [dict(terminal_id="southbow_cushing", month=f"{2025+k//12}-{k%12+1:02d}", floating_roof_tank_throughput_bbl=value,
                 rolling_12month_bbl=rolling[k], recurrence_residual_bbl=residuals[k-12] if k >= 12 else None,
                 source_pdf_pages="105" if k < 6 else "23,105" if k < 12 else "23", source_document_id=DOCUMENT,
                 source_url=source_url, observed_inventory_bbl=None, first_public_availability_verified=False)
            for k, value in enumerate(volumes)]


def operator_records():
    source, = [r for r in read("cushing_okeis_documents") if r["document_id"] == DOCUMENT]
    if hashlib.sha256((ROOT / source["local_pdf"]).read_bytes()).hexdigest() != SHA256:
        raise ValueError("The visually verified South Bow report has changed")
    permits = {r["tank_id"]: r for r in read("permit_tanks") if r["document_id"] == "6816" and r["tank_id"].startswith("P-")}
    groups = defaultdict(list)
    for r in read("tank_engineering_registry"):
        if r["documented_terminal_id"] == "southbow_cushing":
            groups[tuple(sorted(json.loads(r["possible_equipment_ids_json"])))].append(r)
    records, points = [], {}
    for n, page, ft, clock in INSPECTIONS:
        ident = f"P-{n}"
        permit = permits[ident]
        group, = [g for g in groups if ident in g]
        candidates = sorted(r["candidate_id"] for r in groups[group])
        if len(group) != len(candidates) or any(r["documented_roof_type"] != permit["roof_type"] for r in groups[group]):
            raise ValueError(f"South Bow engineering group differs from its permit for {ident}")
        roof_type = permit["roof_type"]
        if roof_type == "EFR":
            points.update({r["candidate_id"]: Point(float(r["longitude"]), float(r["latitude"])) for r in groups[group]})
        shell = float(permit["height_ft"]) * .3048
        records.append(dict(terminal_id="southbow_cushing", state_facility_id="9633", equipment_id=ident,
                            source_document_id=DOCUMENT, source_pdf_page=page, source_pdf=source["local_pdf"],
                            source_pdf_sha256=SHA256, source_url=source["document_url"], source_local_date=DAY.isoformat(),
                            source_local_time=clock, operator_liquid_level_ft=ft, operator_liquid_level_m=ft*.3048,
                            level_datum_reported=False, roof_type=roof_type, diameter_ft=float(permit["diameter_ft"]),
                            shell_ft=float(permit["height_ft"]), conditional_design_shell_minus_liquid_m=shell-ft*.3048,
                            cohort_id=groups[group][0]["cohort_id"], possible_candidate_ids_json=json.dumps(candidates),
                            individual_contour_verified=False, permit_url=permit["source_url"],
                            permit_pdf_page=permit["pdf_page"], optically_visible_roof=roof_type == "EFR"))
    return records, points


def compare(records, fits):
    ground, by_product = defaultdict(list), defaultdict(dict)
    for r in records:
        if r["optically_visible_roof"]:
            ground[r["cohort_id"]].append(r)
    for r in fits:
        by_product[r["product_id"]][r["candidate_id"]] = r
    first = datetime.combine(DAY, time(), LOCAL_TIME).astimezone(timezone.utc)
    last = datetime.combine(DAY + timedelta(days=1), time(), LOCAL_TIME).astimezone(timezone.utc)
    result = []
    for product, by_roof in sorted(by_product.items()):
        t = datetime.fromisoformat(next(iter(by_roof.values()))["captured_at_utc"])
        for cohort, rows in sorted(ground.items()):
            keys = json.loads(rows[0]["possible_candidate_ids_json"])
            supported = [by_roof[k] for k in keys if k in by_roof and by_roof[k]["fit_status"] == RESOLVED]
            values = [r["conditional_design_shell_minus_liquid_m"] for r in rows]
            complete = len(supported) == len(keys) == len(values)
            estimate = mean(r["estimated_gap_m"] for r in supported) if complete else None
            result.append(dict(cohort_id=cohort, product_id=product, capture_at_utc=t.isoformat(),
                               min_possible_hours_from_inspection=max(0., (first-t).total_seconds(), (t-last).total_seconds())/3600,
                               max_possible_hours_from_inspection=max(abs((first-t).total_seconds()), abs((last-t).total_seconds()))/3600,
                               operator_equipment_count=len(rows), candidate_roofs=len(keys), resolved_roofs=len(supported),
                               covered_candidate_ids_json=json.dumps(keys),
                               resolved_candidate_ids_json=json.dumps(sorted(r["candidate_id"] for r in supported)),
                               conditional_operator_cohort_mean_gap_m=mean(values),
                               optical_cohort_mean_gap_m=estimate,
                               conditional_mean_gap_difference_m=estimate-mean(values) if complete else None,
                               synchronized_roof_measurement=False))
    return result


def main():
    records, points = operator_records()
    flows = throughput(records[0]["source_url"])
    geometry = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry")
                if r["candidate_id"] in points and r["modern_input_status"] in {"available", "insufficient_lidar_outline_quality"}
                and r["latest_engineering_date"] < DAY.isoformat()}
    if geometry.keys() != points.keys():
        raise ValueError("South Bow geometry is missing for an externally visible inspection candidate")
    sources = scenes(points, DAY)
    fits = [dict(r, geometry_status=geometry[r["candidate_id"]]["modern_input_status"]) for r in sentinel(sources, geometry, stem=STEM)]
    strict = [r for r in fits if r["geometry_status"] == "available"]
    comparisons = [dict(r, geometry_scope=scope) for scope, rows in (("qualified_lidar", strict), ("all_dated_lidar_exploratory", fits))
                   for r in compare(records, rows)]
    keys = dict.fromkeys(k for row in fits for k in row)
    for suffix, rows in (("ground_levels", records), ("monthly_throughput", flows), ("scene_catalog", sources),
                         ("sentinel_fits", [{k: row.get(k) for k in keys} for row in fits]), ("cohort_comparisons", comparisons)):
        export(f"{STEM}_{suffix}", rows)
    summary = dict(inspection_date=DAY.isoformat(), levels=len(records), externally_visible_roofs=len(points),
                   roofs_with_qualified_lidar=sum(g["modern_input_status"] == "available" for g in geometry.values()),
                   measured_open_roof_level_range_m=[min(r["operator_liquid_level_m"] for r in records if r["optically_visible_roof"]),
                                                    max(r["operator_liquid_level_m"] for r in records if r["optically_visible_roof"])],
                   same_day_catalog_products=sum(r["scene_minus_inspection_days"] == 0 for r in sources),
                   candidate_scene_products=len(sources), sentinel_fits=len(fits), fit_status=Counter(r["fit_status"] for r in fits),
                   complete_cohort_comparisons=Counter(r["geometry_scope"] for r in comparisons if r["optical_cohort_mean_gap_m"] is not None),
                   operator_monthly_throughput_rows=len(flows), overlapping_throughput_months_verified=6,
                   admitted_synchronized_height_errors=0)
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
