"""Inventory dated roof area, image-window coverage and historical EIA comparisons."""
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from statistics import median

from audit_sentinel2_weekly_volume import PHASES, comparison, error_metrics, releases
from build_registered_shadow_development import study_weeks
from build_shadow_timing_measurements import date_evidence, engineering_dates
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write
from collect_s2_spectral_detector_masks import VOLUMES

STEM = "sentinel2_space_time_coverage"
INPUT_PHASES = (*PHASES, "holdout")


def roof_registry():
    geometry, evidence = legacy_geometry(), date_evidence()
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(r) for r in db.execute("""SELECT candidate_id,documented_terminal_id AS terminal_id,cohort_id,
                first_seen,engineering_source_url,source_document_date FROM tank_engineering_registry
                WHERE documented_roof_type='EFR' ORDER BY candidate_id""") if r["candidate_id"] in geometry]
    if {r["candidate_id"] for r in rows} != geometry.keys():
        raise ValueError("Some dated geometry is absent from the current EFR registry")
    result = []
    for row in rows:
        g = geometry[row["candidate_id"]]
        if not row["first_seen"] or not row["terminal_id"]:
            raise ValueError(f"Missing construction or terminal evidence for {row['candidate_id']}")
        dates = engineering_dates(row, evidence)
        diameter = float(g["diameter_m"])
        shell = float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"])
        if not (diameter > 0 and shell > 0):
            raise ValueError(f"Invalid cylinder geometry for {row['candidate_id']}")
        area = math.pi*diameter**2/4
        result.append(dict(row, lidar_source_available_date=dates[0], roof_evidence_date=dates[1],
            has_both_dated_sources=all(dates),
            geometry_dataset=g["dataset"], geometry_file=g["source_directory"], diameter_m=diameter,
            shell_above_ground_m=shell, circular_area_m2=area, cylindrical_shell_equivalent_kbbl=area*shell/BARREL_M3/1000))
    return result


def week_key(row):
    return row["week_ending"], row["candidate_id"]


def tank_rows(registry, *, phases=INPUT_PHASES, measurements=None, alternative_checked=None):
    measured_phases = PHASES if measurements is None else phases
    if measurements is None:
        measurements = (dict(row, measurement_source=f"{stem}.csv")
            for stem in (*VOLUMES, "hcs_dated_shadow_volumes") for row in read_csv(stem)
            if row["study_phase"] in phases)
    optical = {}
    for row in measurements:
        key = week_key(row)
        if row["study_phase"] not in measured_phases or key in optical:
            raise ValueError(f"Unexpected or duplicate B04 measurement {key}")
        optical[key] = row
    candidates = {week_key(r): r for r in read_csv("shadow_timing_candidates_tanks")}
    if alternative_checked is None:
        alternative_checked = {week_key(r) for r in read_csv("sentinel2_alternate_registration_pairs")
                               if r["registration_status"] == "checked_offset"}
    counted = set()
    for phase in phases:
        checked = {(r["week_or_control_day"], r["candidate_id"])
                   for r in read_csv(f"shadow_registration_{phase}_offsets") if r["registration_status"] == "checked_offset"}
        checked.update(alternative_checked)
        for week, (start, end, release, cutoff) in study_weeks(phase).items():
            for roof in registry:
                if not roof["has_both_dated_sources"] or max(roof["lidar_source_available_date"], roof["roof_evidence_date"]) >= cutoff[:10] or (
                    roof["first_seen"] > start.date().isoformat()
                ):
                    continue
                key = week, roof["candidate_id"]
                c, m = candidates.get(key), optical.get(key)
                if c and (c["release_date"], c["terminal_id"], c["inputs_available_before_utc"]) != (
                    release, roof["terminal_id"], cutoff
                ):
                    raise ValueError(f"Timing catalog disagrees for {key}")
                has_candidate = c is not None and c["frozen_style_pair_available"] == "True"
                has_offset = key in checked
                if has_offset and not has_candidate:
                    raise ValueError(f"Registered pair lacks an eligible pair of captures {key}")
                days = (end-start).total_seconds()/3600
                result = dict(study_phase=phase, week_ending=week, release_date=release, cutoff_utc=cutoff,
                    previous_measurement_at_utc=start.isoformat(), current_measurement_at_utc=end.isoformat(),
                    reporting_interval_hours=days, candidate_id=roof["candidate_id"], terminal_id=roof["terminal_id"],
                    first_seen=roof["first_seen"], lidar_source_available_date=roof["lidar_source_available_date"],
                    roof_evidence_date=roof["roof_evidence_date"], circular_area_m2=roof["circular_area_m2"],
                    cylindrical_shell_equivalent_kbbl=roof["cylindrical_shell_equivalent_kbbl"],
                    valid_input_pair=has_candidate, checked_exterior_alignment=has_offset,
                    both_boundaries_have_bracketing_inputs=c is not None and c["both_boundaries_bracketed"] == "True",
                    baseline_b04_resolved=bool(m) if phase in measured_phases else None, earlier_capture_at_utc=None,
                    later_capture_at_utc=None, earlier_sentinel2_version=None, later_sentinel2_version=None,
                    archive_available_at_utc=None, measurement_source=None, capture_interval_hours=None, preceding_hours=None,
                    reporting_overlap_hours=None, following_hours=None, capture_change_kbbl=None,
                    overlap_change_kbbl=None, extrapolated_week_rate_change_kbbl=None)
                if m:
                    before, after = (datetime.fromisoformat(m[f"{s}_capture_at_utc"]) for s in ("earlier", "later"))
                    archived = datetime.fromisoformat(m["archive_available_at_utc"])
                    if (not has_offset or m["study_phase"] != phase or m["release_date"] != release
                        or m["cutoff_utc"] != cutoff or m["terminal_id"] != roof["terminal_id"]
                        or not before <= start < after <= end or not after <= archived < datetime.fromisoformat(cutoff)
                        or not math.isclose(float(m["circular_area_m2"]), roof["circular_area_m2"], rel_tol=1e-9)
                    ):
                        raise ValueError(f"Baseline physical measurement disagrees for {key}")
                    counted.add(key)
                    duration = (after-before).total_seconds()/3600
                    overlap = (after-start).total_seconds()/3600
                    change = float(m["capture_change_kbbl"])
                    adjusted = change*overlap/duration
                    if not math.isclose(adjusted, float(m["overlap_change_kbbl"]), abs_tol=1e-10):
                        raise ValueError(f"Overlap conversion disagrees for {key}")
                    result.update({field: m[field] for field in ("earlier_capture_at_utc", "later_capture_at_utc",
                        "earlier_sentinel2_version", "later_sentinel2_version", "archive_available_at_utc", "measurement_source")})
                    result.update(capture_interval_hours=duration, preceding_hours=(start-before).total_seconds()/3600,
                        reporting_overlap_hours=overlap, following_hours=(end-after).total_seconds()/3600,
                        capture_change_kbbl=change, overlap_change_kbbl=adjusted,
                        extrapolated_week_rate_change_kbbl=change*days/duration)
                yield result
    if counted != optical.keys():
        raise ValueError(f"Baseline measurements missing from the dated area ledger: {sorted(optical.keys()-counted)[:5]}")


def summarize_area(clock, rows, *, measured_phase=None):
    by_status = {}
    for name, condition in (
        ("dated_reference", lambda r: True), ("eligible_input", lambda r: r["valid_input_pair"]),
        ("exterior_checked", lambda r: r["checked_exterior_alignment"]), ("baseline_b04", lambda r: r["baseline_b04_resolved"]),
    ):
        members = [r for r in rows if condition(r)]
        by_status[name] = members
    base = dict(clock)
    if measured_phase is None:
        measured_phase = clock["study_phase"] in PHASES
    base.update({f"{stage}_{field}": value if stage != "baseline_b04" or measured_phase else None
        for stage, members in by_status.items()
        for field, value in (("tanks", len(members)), ("area_m2", sum(r["circular_area_m2"] for r in members)),
            ("shell_equivalent_kbbl", sum(r["cylindrical_shell_equivalent_kbbl"] for r in members)))})
    measured = by_status["baseline_b04"]
    area = base["dated_reference_area_m2"]
    time = base["reporting_interval_hours"]
    weights = base["baseline_b04_area_m2"]
    base.update(eligible_input_area_fraction=base["eligible_input_area_m2"]/area if area else None,
                exterior_checked_area_fraction=base["exterior_checked_area_m2"]/area if area else None,
                baseline_b04_area_fraction=weights/area if weights is not None and area else None,
                dated_bracket_candidate_tanks=sum(r["both_boundaries_have_bracketing_inputs"] for r in rows))
    for field in ("capture_interval_hours", "preceding_hours", "reporting_overlap_hours", "following_hours"):
        base[f"area_weighted_{field}"] = sum(r[field]*r["circular_area_m2"] for r in measured)/weights if measured else None
    base["baseline_b04_space_time_fraction"] = (sum(r["reporting_overlap_hours"]*r["circular_area_m2"] for r in measured)/(area*time)
                                               if weights is not None and area else None)
    for field in ("capture_change_kbbl", "overlap_change_kbbl", "extrapolated_week_rate_change_kbbl"):
        base[field] = sum(r[field] for r in measured) if measured else None
    return base


RULES = ("prior_kbbl", "raw_kbbl", "overlap_kbbl", "weekly_rate_kbbl", "area_scaled_raw_kbbl",
         "prior_plus_raw_kbbl", "prior_plus_overlap_kbbl", "prior_for_unobserved_plus_overlap_kbbl")


def diagnostic_panel(weeks):
    targets = {r["week_ending"]: r for phase in PHASES for r in comparison(phase, [], releases(phase))}
    for week in weeks:
        if week["study_phase"] not in PHASES:
            continue
        eia = targets[week["week_ending"]]
        if (week["release_date"], week["study_phase"]) != (eia["release_date"], eia["study_phase"]):
            raise ValueError(f"EIA issue differs for {week['week_ending']}")
        prior, raw, overlap = (eia["previous_change_known_before_issue_kbbl"],
                               week["capture_change_kbbl"], week["overlap_change_kbbl"])
        area, space_time = week["baseline_b04_area_fraction"], week["baseline_b04_space_time_fraction"]
        yield dict(week, eia_first_release_change_kbbl=eia["first_release_change_kbbl"],
            eia_source_url=eia["eia_source_url"], prior_kbbl=prior, raw_kbbl=raw, overlap_kbbl=overlap,
            weekly_rate_kbbl=week["extrapolated_week_rate_change_kbbl"],
            area_scaled_raw_kbbl=raw/area if raw is not None else None,
            prior_plus_raw_kbbl=prior+raw if None not in (prior, raw) else None,
            prior_plus_overlap_kbbl=prior+overlap if None not in (prior, overlap) else None,
            prior_for_unobserved_plus_overlap_kbbl=prior*(1-space_time)+overlap if None not in (prior, overlap) else None)


def summaries(panel):
    target = "eia_first_release_change_kbbl"
    common = [r for r in panel if r[target] is not None and all(r[m] is not None for m in RULES)]
    dev = [r for r in common if r["study_phase"] == "development"]
    # Choose descriptive coverage thresholds using the observation geometry alone.
    area = median(r["baseline_b04_area_fraction"] for r in dev)
    lag = median(r["area_weighted_following_hours"] for r in dev)
    report = {}
    for phase in PHASES:
        sample = [r for r in common if r["study_phase"] == phase]
        by_scope = {"all": sample,
                    "at_least_development_median_area": [r for r in sample if r["baseline_b04_area_fraction"] >= area],
                    "below_development_median_area": [r for r in sample if r["baseline_b04_area_fraction"] < area],
                    "at_most_development_median_end_gap": [r for r in sample if r["area_weighted_following_hours"] <= lag],
                    "above_development_median_end_gap": [r for r in sample if r["area_weighted_following_hours"] > lag],
                    "both_higher_area_and_shorter_gap": [r for r in sample if r["baseline_b04_area_fraction"] >= area
                                                           and r["area_weighted_following_hours"] <= lag]}
        report[phase] = {}
        for label, rows in by_scope.items():
            report[phase][label] = dict(weeks=len(rows), errors={rule: error_metrics([r[rule]-r[target] for r in rows])
                                                               for rule in RULES} if rows else {})
    return dict(development_median_area_fraction=area, development_median_following_hours=lag, phases=report)


def market_links(terminals):
    relationships = defaultdict(list)
    for row in read_csv("study_terminal_input_coverage"):
        relationships[row["week_ending"], row["terminal_id"]].append(row)
    execution = {(r["study_phase"], r["release_date"], r["instrument"]): r for r in read_csv("study_execution_inputs")}
    fields = ("decision_at_utc", "latest_week_for_market_issue", "issuer", "ticker", "role", "service_agreement_only",
              "valid_from", "valid_to_exclusive", "date_precision", "exposure_date_status", "exposure_source_url",
              "parent_equity_exposure_known_at_cutoff", "deeprock_asset_period_known_at_cutoff",
              "deeprock_merged_scope_confirmed_for_pair_window", "legal_transfer_not_public_at_cutoff",
              "market_window_status", "matched_usd_quote_window", "matched_cad_reference_window", "market_calendar_status")
    cost_fields = ("phase_window_eligible", "entry_session", "fifth_session", "both_legs_have_two_spreads",
                   "observed_cost_inputs_short_instrument", "observed_cost_inputs_short_amlp")
    for row in terminals:
        identity = row["week_ending"], row["terminal_id"]
        for rel in relationships[identity]:
            if ((row["study_phase"], row["release_date"]) != (rel["study_phase"], rel["release_date"])
                or datetime.fromisoformat(row["cutoff_utc"]) != datetime.fromisoformat(rel["inputs_available_before_utc"])):
                raise ValueError(f"Ownership or market input clock differs for {identity}")
            costs = execution.get((row["study_phase"], row["release_date"], rel["ticker"]), {}) if (
                rel["latest_week_for_market_issue"] == "1") else {}
            yield dict(row, **{k: rel[k] for k in fields}, **{k: costs.get(k) for k in cost_fields})
        if not relationships[identity]:
            raise ValueError(f"Missing terminal/issue row in market coverage {identity}")


def main():
    registry = roof_registry()
    tanks = list(tank_rows(registry))
    by_week, by_terminal = defaultdict(list), defaultdict(list)
    for row in tanks:
        by_week[row["week_ending"]].append(row)
        by_terminal[row["week_ending"], row["terminal_id"]].append(row)
    weeks, terminals = [], []
    sites = sorted({r["terminal_id"] for r in registry})
    for phase in INPUT_PHASES:
        for week, (start, end, release, cutoff) in study_weeks(phase).items():
            clock = dict(study_phase=phase, week_ending=week, release_date=release, cutoff_utc=cutoff,
                previous_measurement_at_utc=start.isoformat(), current_measurement_at_utc=end.isoformat(),
                reporting_interval_hours=(end-start).total_seconds()/3600)
            weeks.append(summarize_area(clock, by_week[week]))
            terminals.extend(dict(summarize_area(clock, by_terminal[week, site]), terminal_id=site) for site in sites)
    panel = list(diagnostic_panel(weeks))
    linked = list(market_links(terminals))
    coverage = {}
    for phase in INPUT_PHASES:
        sample = [r for r in weeks if r["study_phase"] == phase]
        observed = [r for r in sample if r["baseline_b04_tanks"]]
        coverage[phase] = dict(weeks=len(sample), weeks_with_dated_reference=sum(bool(r["dated_reference_tanks"]) for r in sample),
            weeks_with_eligible_pairs=sum(bool(r["eligible_input_tanks"]) for r in sample),
            weeks_with_checked_exterior=sum(bool(r["exterior_checked_tanks"]) for r in sample),
            weeks_with_baseline_measurements=len(observed) if phase in PHASES else None,
            baseline_measurements=sum(r["baseline_b04_tanks"] for r in observed) if phase in PHASES else None,
            median_on_measured_weeks={field: median(r[field] for r in observed) for field in (
                "baseline_b04_area_fraction", "baseline_b04_space_time_fraction", "area_weighted_preceding_hours",
                "area_weighted_following_hours", "area_weighted_reporting_overlap_hours")} if observed else None)
    counts = Counter(r["study_phase"] for r in weeks)
    if counts != {phase: len(study_weeks(phase)) for phase in INPUT_PHASES}:
        raise ValueError("Dated roof registry must cover every EIA calendar week")
    for suffix, rows in (("roof_registry", registry), ("tank_weeks", tanks), ("weeks", weeks),
                         ("terminal_weeks", terminals), ("terminal_market_links", linked), ("eia_comparison", panel)):
        export(f"{STEM}_{suffix}", rows)
    report = dict(registered_roofs=len(registry), roofs_with_both_dated_sources=sum(r["has_both_dated_sources"] for r in registry),
        tank_weeks=len(tanks), terminal_weeks=len(terminals), terminal_market_link_rows=len(linked),
        coverage=coverage, matched_eia=summaries(panel),
        interpretation="The area denominator is the dated EFR registry, selected before the issue cutoff and known "
        "to exist by the prior measurement date. Shell equivalent is geometric volume, independent of working capacity. "
        "Space-time coverage is the area-weighted portion of the EIA reporting interval directly observed, relative to "
        "the dated EFR registry. Observed B04 pairs include the newly dated HCS group. "
        "Scaling and prior mixtures are fixed sensitivity tests for the whole-hub target. "
        "Both thresholds use development coverage only. Terminal market links retain each relationship and its "
        "original status, without summing duplicate roles. Holdout rows contain engineering inputs without measured stocks.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
