"""Compare terminal stock changes on the same tanks across independent passes."""
import json
import math
from collections import Counter, defaultdict
from contextlib import closing
from datetime import date
from statistics import median

import numpy as np

from audit_sentinel2_space_time_coverage import STEM as SPACE_TIME
from build_operator_development_inputs import one_per_key
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write

STEM = "terminal_change_repeatability"
HISTORICAL = ("development", "validation")
VOLUME_FILES = ("sentinel2_weekly_tank_volumes", "sentinel2_alternate_registration_added_volumes", "hcs_dated_shadow_volumes")
RADAR_MAX_SEPARATION_HOURS = 24
TANDEM_MAX_SEPARATION_SECONDS = 60


def direction(value):
    return "increase" if value > 0 else "decrease" if value < 0 else "flat"


def limits(values):
    values = list(values)
    if not values:
        return dict(n=0)
    return dict(n=len(values), median=median(values), p90=float(np.percentile(values, 90)), maximum=max(values))


def registry():
    return {r["candidate_id"]: dict(r, kbbl_per_m=float(r["circular_area_m2"])/BARREL_M3/1000)
            for r in read_csv(f"{SPACE_TIME}_roof_registry")}


def tandem_observations(roof):
    chosen = one_per_key((r for r in read_csv("shadow_tandem_pass_pairs") if r["selected_for_summary"] == "True"),
                         "other_sensor", "capture_day", "candidate_id")
    for row in read_csv("shadow_tandem_change_pairs"):
        tank = row["candidate_id"]
        if row["roof_type"] != "EFR" or tank not in roof:
            continue
        sensor, start, finish = (row[k] for k in ("other_sensor", "earlier_day", "later_day"))
        first, last = (chosen[sensor, day, tank] for day in (start, finish))
        if any(abs(float(endpoint["other_minus_a_seconds"])) > TANDEM_MAX_SEPARATION_SECONDS for endpoint in (first, last)):
            raise ValueError(f"Non-simultaneous tandem control for {start}/{finish}/{tank}")
        duration = (date.fromisoformat(finish)-date.fromisoformat(start)).days
        if duration != int(row["elapsed_days"]):
            raise ValueError(f"Inconsistent elapsed days for {start}/{finish}/{tank}")
        factor = roof[tank]["kbbl_per_m"]
        changes = []
        for series in ("a", "other"):
            if any(r[f"{series}_version"] != row[f"{series}_{side}_version"] for r, side in ((first, "earlier"), (last, "later"))):
                raise ValueError(f"Different exact scenes for {start}/{finish}/{tank}")
            change = float(first[f"{series}_gap_m"])-float(last[f"{series}_gap_m"])
            if change != float(row[f"{series}_change_m"]):
                raise ValueError(f"Cannot reproduce {series} height change for {start}/{finish}/{tank}")
            changes.append(change*factor)
        all_four = first["both_resolved"] == last["both_resolved"] == "True"
        if all_four != (row["all_four_resolved"] == "True"):
            raise ValueError("Four-endpoint fit flags disagree")
        yield dict(candidate_id=tank, terminal_id=roof[tank]["terminal_id"], earlier_day=start, later_day=finish,
            earlier_scene_pair=f"{first['a_version']}|{first['other_version']}", later_scene_pair=f"{last['a_version']}|{last['other_version']}",
            elapsed_days=duration, other_sensor=sensor, all_four_resolved=all_four,
            maximum_tandem_endpoint_gap_seconds=max(abs(float(r["other_minus_a_seconds"])) for r in (first, last)),
            kbbl_per_m=factor, first_platform_change_kbbl=changes[0], second_platform_change_kbbl=changes[1],
            second_minus_first_kbbl=changes[1]-changes[0])


def summed_control(members):
    first = math.fsum(r["first_platform_change_kbbl"] for r in members)
    second = math.fsum(r["second_platform_change_kbbl"] for r in members)
    combined_disagreement = math.fsum(r["second_minus_first_kbbl"] for r in members)
    area = math.fsum(r["kbbl_per_m"] for r in members)
    if not math.isclose(second-first, combined_disagreement, abs_tol=1e-9):
        raise ValueError("Terminal disagreement does not reconcile")
    return dict(tanks=len(members), total_kbbl_per_m=area, first_platform_change_kbbl=first,
        second_platform_change_kbbl=second, first_direction=direction(first), second_direction=direction(second),
        opposite_nonzero_directions=first*second < 0, same_nonzero_direction=first*second > 0,
        second_minus_first_kbbl=combined_disagreement, absolute_disagreement_kbbl=abs(combined_disagreement),
        sum_absolute_tank_disagreement_kbbl=math.fsum(abs(r["second_minus_first_kbbl"]) for r in members),
        mean_area_weighted_disagreement_m=abs(combined_disagreement)/area)


def terminal_controls(rows):
    groups = defaultdict(list)
    for row in rows:
        for cohort in ("all_numeric", "all_four_resolved") if row["all_four_resolved"] else ("all_numeric",):
            groups[row["earlier_day"], row["later_day"], row["other_sensor"], row["terminal_id"], cohort].append(row)
    for (first, last, sensor, terminal, cohort), members in sorted(groups.items()):
        yield dict(earlier_day=first, later_day=last, other_sensor=sensor, terminal_id=terminal, cohort=cohort,
            elapsed_days=members[0]["elapsed_days"], **summed_control(members))


def common_difference_checks(controls):
    with closing(read_only(SOURCE)) as db:
        terminal_for = dict(db.execute("SELECT candidate_id,documented_terminal_id FROM tank_engineering_registry"))
    fixed = defaultdict(dict)
    for r in read_csv("shadow_tandem_pass_pairs"):
        if (r["selected_for_summary"] == "True" and r["roof_type"] in ("IFR", "Domed EFR")
                and abs(float(r["other_minus_a_seconds"])) <= TANDEM_MAX_SEPARATION_SECONDS):
            fixed[r["capture_day"], r["other_sensor"]][r["candidate_id"]] = float(r["other_minus_a_gap_m"])
    intervals = defaultdict(list)
    for r in controls:
        intervals[r["earlier_day"], r["later_day"], r["other_sensor"]].append(r)
    for (first, last, sensor), members in sorted(intervals.items()):
        left, right = fixed[first, sensor], fixed[last, sensor]
        for r in members:
            terminal, factor = r["terminal_id"], r["total_kbbl_per_m"]
            donors = [k for k in sorted(left.keys() & right.keys()) if terminal_for.get(k) and terminal_for[k] != terminal]
            if len(donors) < 3 or len(members) < 3:
                raise ValueError(f"Insufficient independent control sites for {first}/{last}/{terminal}")
            fixed_shift = median(left[k]-right[k] for k in donors)
            other_efr = [p for p in members if p["terminal_id"] != terminal]
            peer_shift = median(p["second_minus_first_kbbl"]/p["total_kbbl_per_m"] for p in other_efr)
            raw = r["second_minus_first_kbbl"]
            yield dict(earlier_day=first, later_day=last, other_sensor=sensor, terminal_id=terminal,
                measured_tanks=r["tanks"], kbbl_per_m=factor, raw_difference_kbbl=raw,
                fixed_roof_donor_ids="|".join(donors), fixed_roof_donor_count=len(donors),
                fixed_roof_median_difference_m=fixed_shift, fixed_roof_adjusted_difference_kbbl=raw-fixed_shift*factor,
                peer_efr_donor_terminals="|".join(sorted(p["terminal_id"] for p in other_efr)),
                peer_efr_median_difference_m=peer_shift, peer_efr_adjusted_difference_kbbl=raw-peer_shift*factor)


def historical_terminal_measurements(roof, volume_files=VOLUME_FILES, phases=HISTORICAL):
    results = defaultdict(dict)
    for file in volume_files:
        for r in read_csv(file):
            phase, week, terminal, tank = (r[k] for k in ("study_phase", "week_ending", "terminal_id", "candidate_id"))
            members = results[phase, week, terminal]
            if (phase not in phases or tank in members or roof[tank]["terminal_id"] != terminal
                    or not math.isclose(roof[tank]["kbbl_per_m"], float(r["circular_kbbl_per_m"]), rel_tol=1e-9)):
                raise ValueError(f"Historical physical input identity differs for {phase}/{week}/{tank}")
            amount, low, high = (float(r[k]) for k in ("capture_change_kbbl", "joint_loss_min_change_kbbl", "joint_loss_max_change_kbbl"))
            if not low-1e-8 <= amount <= high+1e-8:
                raise ValueError(f"Physical solution outside its loss envelope for {week}/{tank}")
            members[tank] = dict(change=amount, low=low, high=high, kbbl_per_m=roof[tank]["kbbl_per_m"])
    return results


def fitted_change_envelope(members):
    estimate, low, high = (math.fsum(r[k] for r in members) for k in ("change", "low", "high"))
    return dict(tanks=len(members), capture_change_kbbl=estimate, loss_min_kbbl=low, loss_max_kbbl=high,
                loss_envelope_excludes_zero=low > 1e-8 or high < -1e-8)


def terminal_envelopes(history, physical=SPACE_TIME):
    area = one_per_key(read_csv(f"{physical}_terminal_weeks"), "study_phase", "week_ending", "terminal_id")
    for key, members in sorted(history.items()):
        row = area[key]
        envelope = fitted_change_envelope(list(members.values()))
        if envelope["tanks"] != int(row["baseline_b04_tanks"]) or not math.isclose(
                envelope["capture_change_kbbl"], float(row["capture_change_kbbl"]), abs_tol=1e-8):
            raise ValueError(f"Terminal measurement does not reconcile for {key}")
        yield dict(study_phase=key[0], week_ending=key[1], terminal_id=key[2], release_date=row["release_date"],
            candidate_ids="|".join(sorted(members)), dated_area_fraction=float(row["baseline_b04_area_fraction"]),
            observed_space_time_fraction=float(row["baseline_b04_space_time_fraction"]), **envelope)


def operator_envelopes(history, rows=None, phases=HISTORICAL):
    if rows is None:
        rows = read_csv("study_operator_signal_inputs")
    for row in rows:
        if row["study_phase"] not in phases or row["asset_capture_change_kbbl"] in (None, ""):
            continue
        phase, week, issue, ticker = (row[k] for k in ("study_phase", "week_ending", "release_date", "instrument"))
        terminal_ids = row["asset_measured_terminal_ids"].split("|")
        if len(set(terminal_ids)) != len(terminal_ids):
            raise ValueError(f"Duplicate measured terminal in {issue}/{ticker}")
        members = [r for terminal in terminal_ids for r in history[phase, week, terminal].values()]
        envelope = fitted_change_envelope(members)
        if envelope["tanks"] != float(row["asset_baseline_b04_tanks"]) or not math.isclose(
                envelope["capture_change_kbbl"], float(row["asset_capture_change_kbbl"]), abs_tol=1e-8):
            raise ValueError(f"Issuer measurement does not reconcile for {issue}/{ticker}")
        yield dict(study_phase=phase, week_ending=week, release_date=issue, instrument=ticker, study_role=row["study_role"],
            asset_measured_terminal_ids=row["asset_measured_terminal_ids"], candidate_raw_direction=row["candidate_raw_direction"],
            phase_window_eligible=row["phase_window_eligible"] in (True,"True"), matched_usd_quote_window=row["matched_usd_quote_window"] == "True",
            candidate_direction_cost_inputs_observed=row["candidate_direction_cost_inputs_observed"] == "True", **envelope)


def historical_comparisons(history, tandem):
    by_terminal = defaultdict(lambda: defaultdict(dict))
    for r in tandem:
        if r["all_four_resolved"] and r["elapsed_days"] == 10:
            key = r["earlier_day"], r["later_day"], r["other_sensor"]
            by_terminal[r["terminal_id"]][key][r["candidate_id"]] = r
    for (phase, week, terminal), members in sorted(history.items()):
        for (first, last, sensor), controls in sorted(by_terminal[terminal].items()):
            intersection = sorted(members.keys() & controls.keys())
            if not intersection:
                continue
            comparable = [members[k] for k in intersection]
            control = summed_control([controls[k] for k in intersection])
            signal = math.fsum(r["change"] for r in comparable)
            low, high = (math.fsum(r[k] for r in comparable) for k in ("low", "high"))
            error = control["absolute_disagreement_kbbl"]
            yield dict(study_phase=phase, week_ending=week, terminal_id=terminal, control_earlier_day=first,
                control_later_day=last, control_other_sensor=sensor, candidate_ids="|".join(intersection),
                historical_total_measured_tanks=len(members), historical_matched_tanks=len(intersection),
                matched_fraction_of_historical_area=control["total_kbbl_per_m"]/math.fsum(r["kbbl_per_m"] for r in members.values()),
                historical_matched_change_kbbl=signal, historical_loss_min_kbbl=low, historical_loss_max_kbbl=high,
                historical_loss_sign_resolved=low > 0 or high < 0,
                control_change_first_platform_kbbl=control["first_platform_change_kbbl"],
                control_change_second_platform_kbbl=control["second_platform_change_kbbl"],
                control_absolute_disagreement_kbbl=error,
                control_sum_absolute_tank_disagreement_kbbl=control["sum_absolute_tank_disagreement_kbbl"],
                historical_magnitude_exceeds_control_disagreement=abs(signal) > error,
                historical_magnitude_exceeds_twice_control_disagreement=abs(signal) > 2*error)


def radar_checks(roof):
    for r in read_csv("radar_optical_cushing_control_changes"):
        tank = r["candidate_id"]
        hours = float(r["both_endpoints_maximum_separation_seconds"])/3600
        if r["all_scene_optical_fits_resolved"] != "True" or hours > RADAR_MAX_SEPARATION_HOURS or tank not in roof:
            continue
        optical = float(r["optical_height_change_m"])
        radar = float(r["radar_scene_height_change_m"])
        factor = roof[tank]["kbbl_per_m"]
        yield dict(candidate_id=tank, terminal_id=roof[tank]["terminal_id"], earlier_radar=r["earlier_radar_start_utc"],
            later_radar=r["later_radar_start_utc"], earlier_radar_product_id=r["earlier_radar_product_id"],
            later_radar_product_id=r["later_radar_product_id"], earlier_optical_version=r["earlier_optical_version"],
            later_optical_version=r["later_optical_version"], maximum_end_time_difference_hours=hours,
            radar_elapsed_days=float(r["radar_elapsed_days"]), optical_elapsed_days=float(r["optical_elapsed_days"]),
            optical_change_kbbl=optical*factor, radar_change_kbbl=radar*factor, radar_minus_optical_kbbl=(radar-optical)*factor,
            same_nonzero_direction=optical*radar > 0)


def main():
    roof = registry()
    tandem = list(tandem_observations(roof))
    terminals = list(terminal_controls(tandem))
    history = historical_terminal_measurements(roof)
    envelopes = list(terminal_envelopes(history))
    operators = list(operator_envelopes(history))
    comparisons = list(historical_comparisons(history, tandem))
    radar = list(radar_checks(roof))
    ten_day = [r for r in terminals if r["elapsed_days"] == 10 and r["cohort"] == "all_four_resolved"]
    common = list(common_difference_checks(ten_day))
    hist, economic_coverage = {}, {}
    for phase in HISTORICAL:
        observed = [r for r in envelopes if r["study_phase"] == phase]
        priced = [r for r in operators if r["study_phase"] == phase and r["study_role"] == "core_operator"
                  and r["phase_window_eligible"] and r["matched_usd_quote_window"]]
        priced_with_costs = [r for r in priced if r["candidate_direction_cost_inputs_observed"]]
        economic_coverage[phase] = dict(priced_core_measured_windows=len(priced), priced_with_observed_costs=len(priced_with_costs),
            priced_with_loss_envelope_excluding_zero=sum(r["loss_envelope_excludes_zero"] for r in priced),
            priced_with_costs_and_loss_envelope_excluding_zero=sum(r["loss_envelope_excludes_zero"] for r in priced_with_costs),
            distinct_issues_with_costs_and_loss_envelope_excluding_zero=len({r["release_date"] for r in priced_with_costs
                if r["loss_envelope_excludes_zero"]}))
        matched = [r for r in comparisons if r["study_phase"] == phase]
        both = Counter((r["week_ending"], r["terminal_id"]) for r in matched if r["historical_magnitude_exceeds_control_disagreement"])
        repeat = Counter((r["week_ending"], r["terminal_id"]) for r in matched)
        hist[phase] = dict(measured_terminal_weeks=len(observed), summed_loss_envelope_excludes_zero=sum(
            r["loss_envelope_excludes_zero"] for r in observed),
            terminal_weeks_with_any_matched_tandem_subset=len(repeat), comparisons=len(matched),
            matched_area_fraction=limits(r["matched_fraction_of_historical_area"] for r in matched),
            historical_abs_change_same_tanks_kbbl=limits(abs(r["historical_matched_change_kbbl"]) for r in matched),
            tandem_abs_disagreement_same_tanks_kbbl=limits(r["control_absolute_disagreement_kbbl"] for r in matched),
            matched_comparisons_larger_than_observed_disagreement=sum(r["historical_magnitude_exceeds_control_disagreement"] for r in matched),
            matched_comparisons_larger_than_twice_disagreement=sum(r["historical_magnitude_exceeds_twice_control_disagreement"] for r in matched),
            terminal_weeks_beating_both_tandem_disagreements=sum(n == 2 and both[key] == n for key, n in repeat.items()),
            terminal_weeks_with_both_tandem_comparisons=sum(n == 2 for n in repeat.values()))
    report = dict(tandem_control_tank_intervals=len(tandem), control_date_pairs=sorted({(r["earlier_day"], r["later_day"]) for r in tandem}),
        ten_day_terminal_control=dict(terminal_intervals=len(ten_day), terminals=len({r["terminal_id"] for r in ten_day}),
            same_nonzero_direction=sum(r["same_nonzero_direction"] for r in ten_day),
            opposite_nonzero_directions=sum(r["opposite_nonzero_directions"] for r in ten_day),
            absolute_disagreement_kbbl=limits(r["absolute_disagreement_kbbl"] for r in ten_day)),
        common_disagreement=dict(
            directions_by_interval={f"{a}/{b}": sorted({direction(r["raw_difference_kbbl"]) for r in common
                if (r["earlier_day"], r["later_day"]) == (a,b)}) for a,b in sorted({(r["earlier_day"],r["later_day"]) for r in common})},
            fixed_roof_adjusted_abs_disagreement_kbbl=limits(abs(r["fixed_roof_adjusted_difference_kbbl"]) for r in common),
            peer_efr_adjusted_abs_disagreement_kbbl=limits(abs(r["peer_efr_adjusted_difference_kbbl"]) for r in common),
            fixed_roof_improved_intervals=sum(abs(r["fixed_roof_adjusted_difference_kbbl"]) < abs(r["raw_difference_kbbl"]) for r in common),
            peer_efr_improved_intervals=sum(abs(r["peer_efr_adjusted_difference_kbbl"]) < abs(r["raw_difference_kbbl"]) for r in common)),
        historical=hist, operator_windows=economic_coverage,
        radar_optical=dict(max_endpoint_separation_hours=RADAR_MAX_SEPARATION_HOURS, matched_efr_changes=len(radar),
            matched_within_one_hour=sum(r["maximum_end_time_difference_hours"] <= 1 for r in radar),
            same_nonzero_direction=sum(r["same_nonzero_direction"] for r in radar),
            absolute_disagreement_kbbl=limits(abs(r["radar_minus_optical_kbbl"]) for r in radar)),
        interpretation="Tandem disagreement is the difference of two ten- or twenty-day physical changes on the same tanks. "
        "Two satellites see each endpoint less than a minute apart. The two adjacent ten-day intervals share a middle date. "
        "The matched historical comparison uses exactly the intersecting tanks and the same conversion factors. It compares scale, "
        "not a same-week estimate of error. The summed historical loss envelope describes fit ambiguity and has no calibrated "
        "coverage probability. The common-difference adjustments use fixed exterior roofs or the other terminals only. "
        "They estimate inter-platform differences on these tandem scenes and do not identify the true volume or support a "
        "single-pass correction. Radar comparisons may include real movement between sensors. No closed EIA values or market "
        "outcomes are read.")
    for suffix, rows in (("tank_controls", tandem), ("terminal_controls", terminals), ("scene_bias_checks", common), ("historical_matches", comparisons),
                         ("terminal_envelopes", envelopes), ("operator_envelopes", operators), ("radar_checks", radar)):
        export(f"{STEM}_{suffix}", rows)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
