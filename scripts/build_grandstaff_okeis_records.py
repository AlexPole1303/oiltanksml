"""Reconcile the Grandstaff operator report with the existing equipment registry."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

from build_s1_reference_matches import DERIVED, export, read
from collect_public import RAW, ROOT, write

STEM = "cushing_grandstaff_okeis"
STORE = RAW / "documents" / "okeis_grandstaff"
PROFILE = "https://okeis.deq.ok.gov/ext/nsite/DEFAULT/map/results/detail/312394640967090259/Documents"
DOWNLOAD = "https://okeis.deq.ok.gov/ext/ncore/downloadpdf/"
REPORT = "-2797061616394687541"
NOTICE = "6263408250108625246"
TERMINAL = "enbridge_blueknight"
PERIOD = dict(report_period_start="2026-04-19", report_period_end="2026-10-18")
MONTHS = tuple(f"2026-{month:02d}" for month in range(4, 11))
MONTHLY_TOTALS_BBL = (1423207, 597434, 409522, 94720, 201682, 92372, 124722)
ROLLING_TOTALS_BBL = (15002933, 14249070, 13426054, 12395888, 11564959, 11018649, 10148807)
ROLLING_LANDING_VOC_TONS = (3.41, 2.69, 2.69, 1.98, 1.98, 1.16, 1.16)
NON_NUMERIC = {"-": "dash_in_report", "Out of Service": "out_of_service", "Demolished": "demolished"}
LOCAL_TIME = ZoneInfo("America/Chicago")
DOCUMENTS = (
    (REPORT, "2026-10-28", "SAR-0000375", "2021-0303-TVR4", TERMINAL, 23,
     "ad528ec82d9311c5ec605139f6911cb1c19a77565cb9e25513c93fa85c0cfcc3"),
    (NOTICE, "2026-08-27", "NSPS-0000173", "2021-0303-TVR4", TERMINAL, 1,
     "1d6d2444627cd3710d53d7e2bc907de9bb13ad3b1e5865b88a745500d40666da"),
    ("8142668143451501197", "2026-10-29", "ACC-0008100", "2021-0303-TVR4", TERMINAL, 5,
     "a47f48b84b37729d4ccc794c7ed472c51769a248963311a025facd5847fb2f81"),
    ("-2205531015528168648", "2026-10-29", "ACC-0008100", "2021-0377-TVR3", "enbridge_cushing", 19,
     "e4a9794b30494a9aa0c55503d945370aaf94c1c66e53227e44077b9fe93c81e5"),
)


def sources():
    for key, created, submission, permit, terminal, pages, expected in DOCUMENTS:
        path = STORE / f"{key}.pdf"
        data = path.read_bytes()
        if not data.startswith(b"%PDF-") or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Unexpected source PDF bytes for {key}")
        yield dict(document_id=key, source_terminal_listing=TERMINAL, document_terminal=terminal,
                   permit=permit, submission=submission, listing_created_local_date=created,
                   first_public_availability_verified=False, pages=pages, bytes=len(data), sha256=expected,
                   source_listing_url=PROFILE, source_url=DOWNLOAD + key, source_file=str(path.relative_to(ROOT)))


def cohorts():
    equipment = defaultdict(list)
    for row in read("north_farm_contour_reconciliation"):
        if row["terminal_id"] == TERMINAL and row["possible_equipment_ids_json"]:
            for number in json.loads(row["possible_equipment_ids_json"]):
                equipment[number].append((row["candidate_id"], row["cohort_id"]))
    return equipment


def tank_rows(equipment):
    path = ROOT / "data/reference/grandstaff_okeis_throughput_transcription.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["equipment_id", *MONTHS]:
            raise ValueError("Unexpected Grandstaff throughput months")
        for row in reader:
            operator_id = row["equipment_id"]
            number = operator_id.removeprefix("CG-")
            if operator_id == number or not number.isdecimal():
                raise ValueError(f"Invalid Grandstaff equipment ID {operator_id}")
            possible = equipment[number]
            groups = {group for _, group in possible}
            if len(groups) > 1:
                raise ValueError(f"Equipment appears in multiple Grandstaff cohorts {operator_id}")
            for month in MONTHS:
                raw = row[month]
                value = int(raw) if raw.isdecimal() else None
                yield dict(terminal_id=TERMINAL, operator_equipment_id=operator_id, equipment_id=number,
                           report_month=month, **PERIOD, source_entry=raw, withdrawal_throughput_bbl=value,
                           source_status="numeric_withdrawal" if value is not None else NON_NUMERIC[raw],
                           cohort_id=next(iter(groups), None),
                           possible_candidate_ids_json=json.dumps(sorted(tank for tank, _ in possible)),
                           individual_contour_verified=False, liquid_level_observed=False,
                           source_pdf_page=6, source_document_id=REPORT, source_url=DOWNLOAD + REPORT)


def monthly_rows(rows):
    for month, total, rolling, voc in zip(MONTHS, MONTHLY_TOTALS_BBL, ROLLING_TOTALS_BBL,
                                          ROLLING_LANDING_VOC_TONS, strict=True):
        group = [r for r in rows if r["report_month"] == month]
        known = [r["withdrawal_throughput_bbl"] for r in group if r["withdrawal_throughput_bbl"] is not None]
        if sum(known) != total:
            raise ValueError(f"Grandstaff individual amounts do not sum to published total for {month}")
        yield dict(terminal_id=TERMINAL, report_month=month, **PERIOD, reported_total_withdrawal_bbl=total,
                   summed_numeric_withdrawal_bbl=sum(known), rolling_12_month_withdrawal_bbl=rolling,
                   rolling_12_month_landing_and_refilling_voc_tons=voc, monthly_landing_voc_entry="-",
                   numeric_tank_rows=len(known), out_of_service_tank_rows=sum(r["source_status"] == "out_of_service" for r in group),
                   reported_total_pdf_page=5, tank_rows_pdf_page=6, landing_emissions_pdf_page=17,
                   source_document_id=REPORT, source_url=DOWNLOAD + REPORT)


def notice_rows(equipment):
    # The public letter lists all 31 EFRs and states that timing depends on tank levels.
    numbers = {str(n) for n in (*range(1001, 1004), *range(2001, 2004), *range(2501, 2526))}
    if not numbers <= equipment.keys() or any(not equipment[number] for number in numbers):
        raise ValueError("Grandstaff notice equipment does not match the 31-roof registry")
    for number in sorted(numbers, key=int):
        yield dict(terminal_id=TERMINAL, equipment_id=number, notice_local_date="2026-08-27",
                   planned_inspection_on_or_after_local_date="2026-10-27", primary_gap_planned=number == "2513",
                   secondary_gap_planned=True, actual_inspection_at_local=None, actual_liquid_level_m=None,
                   source_document_id=NOTICE, source_pdf_page=1, source_url=DOWNLOAD + NOTICE)


def optical_controls(monthly, tanks):
    # For a consistent volume balance, gross withdrawals must cover each net draw on disjoint intervals.
    # Select by timing alone and compare at site and cohort level without assigning individual equipment IDs.
    complete_months = {r["report_month"]: r for r in monthly if r["report_month"] not in (MONTHS[0], MONTHS[-1])}
    fits = {r["observation_id"]: r for r in read("weekly_shadow_holdout_observations")
            if r["terminal_id"] == TERMINAL and r["geometry_mode"] == "dated_2011_lidar"}
    pair_key = ("candidate_id", "week_ending", "earlier_version", "later_version")
    registered = {tuple(r[field] for field in pair_key): r for r in read("weekly_shadow_registered_holdout_pairs")
                  if r["terminal_id"] == TERMINAL and r["geometry_mode"] == "dated_2011_lidar"}
    groups = defaultdict(list)
    for r in read("weekly_shadow_holdout_pairs"):
        if (r["terminal_id"] != TERMINAL or r["geometry_mode"] != "dated_2011_lidar"
                or r["both_shadow_diagnostics_pass"] != "True"):
            continue
        start, end = (datetime.fromisoformat(r[f"{side}_capture_at_utc"]).astimezone(LOCAL_TIME)
                      for side in ("earlier", "later"))
        month = start.strftime("%Y-%m")
        if month in complete_months and month == end.strftime("%Y-%m"):
            if end <= start:
                raise ValueError(f"Reversed optical interval for {r['candidate_id']}")
            groups[month, r["candidate_id"]].append((start, end, r))
    intervals = []
    for (month, tank), choices in sorted(groups.items()):
        last = None
        for start, end, r in sorted(choices, key=lambda x: (x[1], x[0], x[2]["week_ending"])):
            if last is not None and start < last:
                continue
            last = end
            change = float(r["optical_change_equivalent_bbl"])
            first, second = (fits[r[f"{side}_observation_id"]] for side in ("earlier", "later"))
            for fit, side in ((first, "earlier"), (second, "later")):
                if (fit["candidate_id"], fit["version_id"]) != (tank, r[f"{side}_version"]):
                    raise ValueError(f"Optical interval and its native fit differ for {tank}")
            min_gap_increase = float(second["near_optimal_gap_min_m"]) - float(first["near_optimal_gap_max_m"])
            scale = float(r["circular_bbl_per_m"])
            reg = registered[tuple(r[field] for field in pair_key)]
            reg_change = (float(reg["registered_change_equivalent_bbl"])
                          if reg["registered_both_shadow_diagnostics_pass"] == "True" else None)
            intervals.append(dict(report_month=month, candidate_id=tank, cohort_id=r["cohort_id"],
                                  earlier_capture_local=start.isoformat(), later_capture_local=end.isoformat(),
                                  week_ending=r["week_ending"], earlier_version=r["earlier_version"],
                                  later_version=r["later_version"], satellite_net_change_equivalent_bbl=change,
                                  satellite_decrease_equivalent_bbl=max(0, -change),
                                  near_optimal_min_decrease_bbl=max(0, min_gap_increase) * scale,
                                  min_decrease_with_extra_one_meter_per_interval_bbl=max(0, min_gap_increase - 1) * scale,
                                  registered_fit_status=reg["fit_status"], registered_change_equivalent_bbl=reg_change))
    by_cohort = defaultdict(list)
    for r in tanks:
        if r["cohort_id"] and r["report_month"] in complete_months:
            by_cohort[r["report_month"], r["cohort_id"]].append(r)
    scopes = {(month, "terminal"): value["reported_total_withdrawal_bbl"] for month, value in complete_months.items()}
    scopes.update({key: sum(r["withdrawal_throughput_bbl"] or 0 for r in group) for key, group in by_cohort.items()})
    selected = defaultdict(list)
    for r in intervals:
        for scope in ("terminal", r["cohort_id"]):
            key = r["report_month"], scope
            if key not in scopes:
                raise ValueError(f"Optical candidate is outside the report's mapped cohorts {key}")
            selected[key].append(r)
    controls = []
    for (month, scope), reported in sorted(scopes.items()):
        group = selected[month, scope]
        draw = sum(r["satellite_decrease_equivalent_bbl"] for r in group)
        near = sum(r["near_optimal_min_decrease_bbl"] for r in group)
        extra = sum(r["min_decrease_with_extra_one_meter_per_interval_bbl"] for r in group)
        registered_group = [r for r in group if r["registered_change_equivalent_bbl"] is not None]
        registered_draw = sum(max(0, -r["registered_change_equivalent_bbl"]) for r in registered_group)
        controls.append(dict(report_month=month, scope=scope, reported_numeric_withdrawal_bbl=reported,
                             optical_intervals=len(group), optical_tanks=len({r["candidate_id"] for r in group}),
                             satellite_decrease_equivalent_bbl=draw if group else None,
                             amount_above_reported_bbl=max(0, draw - reported) if group else None,
                             excess_satellite_draw=draw > reported if group else None,
                             near_optimal_min_decrease_bbl=near if group else None,
                             near_optimal_min_still_above_reported=near > reported if group else None,
                             min_decrease_with_extra_one_meter_per_interval_bbl=extra if group else None,
                             registered_comparable_intervals=len(registered_group),
                             baseline_decrease_on_registered_subset_bbl=(
                                 sum(r["satellite_decrease_equivalent_bbl"] for r in registered_group)
                                 if registered_group else None),
                             registered_decrease_equivalent_bbl=registered_draw if registered_group else None,
                             operator_report_used_in_forecast=False, source_document_id=REPORT))
    return intervals, controls


def main():
    inventory = list(sources())
    equipment = cohorts()
    rows = list(tank_rows(equipment))
    keys = {(r["equipment_id"], r["report_month"]) for r in rows}
    if len(keys) != len(rows) or len(rows) != 40 * len(MONTHS):
        raise ValueError("The transcription does not contain 40 complete equipment records")
    monthly = list(monthly_rows(rows))
    notice = list(notice_rows(equipment))
    intervals, controls = optical_controls(monthly, rows)
    for suffix, table in (("sources", inventory), ("tank_throughput", rows), ("terminal_throughput", monthly),
                          ("planned_inspections", notice), ("optical_throughput_intervals", intervals),
                          ("optical_throughput_checks", controls)):
        export(f"{STEM}_{suffix}", table)
    states = Counter(r["source_status"] for r in rows)
    summary = dict(archived_public_documents=len(inventory), tanks=40, tank_month_rows=len(rows),
                   tank_month_status=dict(states), seven_month_columns_reconciled=len(monthly),
                   summed_reported_withdrawal_bbl=sum(r["reported_total_withdrawal_bbl"] for r in monthly),
                   equipment_with_mapped_cohort=len({r["equipment_id"] for r in rows if r["cohort_id"]}),
                   numeric_tanks=len({r["equipment_id"] for r in rows if r["withdrawal_throughput_bbl"] is not None}),
                   nsps_kb_inspections_during_report_period=0, nsps_kb_inspection_summary_pdf_page=3,
                   tanks_with_planned_seal_gap_measurements=len(notice), confirmed_liquid_levels=0,
                   full_calendar_months_in_optical_check=5, nonoverlapping_optical_intervals=len(intervals),
                   terminal_months_with_satellite_decrease_above_reported=[r["report_month"] for r in controls
                       if r["scope"] == "terminal" and r["excess_satellite_draw"]],
                   cohort_months_with_satellite_decrease_above_reported=sum(bool(r["excess_satellite_draw"])
                       for r in controls if r["scope"] != "terminal"))
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
