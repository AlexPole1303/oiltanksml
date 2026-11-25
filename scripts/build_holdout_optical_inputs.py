"""Publish independently registered holdout optics and WTI coverage without outcome labels."""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

from audit_sentinel2_space_time_coverage import roof_registry, summarize_area, tank_rows, week_key
from audit_sentinel2_weekly_volume import observations
from build_registered_shadow_development import study_weeks
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write

PHASE = "holdout"
STEM = "holdout_optical"
SOURCE_FIELDS = ("official", "archive", "ibkr")
REPLAY_SOURCE_ORDER = ("archive", "ibkr")


def indexed(stem):
    rows = [r for r in read_csv(stem) if r["study_phase"] == PHASE]
    result = {r["week_ending"]:r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate holdout report weeks in {stem}")
    return result


def measurements(weeks, with_alternates):
    primary = list(observations(PHASE, weeks, legacy_geometry()))
    export("sentinel2_holdout_weekly_tank_volumes", primary)
    accepted = {week_key(r):dict(r, measurement_source="sentinel2_holdout_weekly_tank_volumes.csv") for r in primary}
    hcs = read_csv("hcs_dated_shadow_holdout_volumes")
    for r in hcs:
        key = week_key(r)
        if key in accepted:
            raise ValueError(f"Duplicate HCS baseline volume for {key}")
        accepted[key] = dict(r, measurement_source="hcs_dated_shadow_holdout_volumes.csv")
    counts = Counter(key[0] for key in accepted)
    checked = set()
    if with_alternates:
        stem = "sentinel2_alternate_registration_holdout"
        checked = {week_key(r) for r in read_csv(f"{stem}_pairs") if r["registration_status"] == "checked_offset"}
        for r in read_csv(f"{stem}_added_volumes"):
            key = week_key(r)
            if key not in accepted:
                accepted[key] = dict(r, measurement_source=f"{stem}_added_volumes.csv")
    return list(accepted.values()), counts, checked


def gap_status(area, availability):
    if area["baseline_b04_tanks"]:
        return "measured"
    if not all(int(availability[f"{side}_l1c_archive_products"]) for side in ("previous", "current")):
        return "no_l1c_in_one_measurement_window"
    if not area["eligible_input_tanks"]:
        return "no_clear_pair_with_dated_geometry"
    if not area["exterior_checked_tanks"]:
        return "no_independently_checked_registration"
    return "no_pair_passed_b04_shadow_diagnostics"


def week_panels(weeks, tanks, registry, primary_counts):
    indexed_tanks, indexed_terminals = defaultdict(list), defaultdict(list)
    for r in tanks:
        indexed_tanks[r["week_ending"]].append(r)
        indexed_terminals[r["week_ending"],r["terminal_id"]].append(r)
    readiness = indexed("eia_weekly_satellite_readiness")
    timing = indexed("shadow_timing_candidates_weeks")
    landsat = indexed("landsat_eia_weekly_coverage")
    market = indexed("wti_cushing_event_windows")
    radar = indexed("s1_cushing_weekly_catalog_coverage")
    sites = sorted({r["terminal_id"] for r in registry})
    by_week, by_terminal = [], []
    if not all(table.keys() == weeks.keys() for table in (readiness, timing, landsat, market, radar)):
        raise ValueError("Holdout coverage files use different report calendars")
    for week, (start,end,release,cutoff) in weeks.items():
        m, a, t, l, s = (table[week] for table in (market,readiness,timing,landsat,radar))
        if any(r["release_date"] != release for r in (m,a,t,l)) or m["open_numeric"] != "False":
            raise ValueError(f"Holdout release or WTI boundary differs on {week}")
        if any(datetime.fromisoformat(r["inputs_available_before_utc"]) != datetime.fromisoformat(cutoff) for r in (a,t)):
            raise ValueError(f"Optical decision cutoff differs on {week}")
        clock = dict(study_phase=PHASE, week_ending=week, release_date=release, cutoff_utc=cutoff,
            previous_measurement_at_utc=start.isoformat(), current_measurement_at_utc=end.isoformat(),
            reporting_interval_hours=(end-start).total_seconds()/3600)
        area = summarize_area(clock,indexed_tanks[week],measured_phase=True)
        prices = {f"wti_{source}_complete_window":int(m[f"{source}_available_leg_sessions"])==10 for source in SOURCE_FIELDS}
        by_week.append(dict(area, b04_status=gap_status(area,a), primary_b04_tanks=primary_counts[week],
            alternate_added_b04_tanks=area["baseline_b04_tanks"]-primary_counts[week],
            **{k:int(a[k]) for k in ("previous_l1c_archive_products","current_l1c_archive_products",
                                    "previous_l1c_screened_products","current_l1c_screened_products")},
            **{k:int(t[k]) for k in ("tanks_with_pairs_allowing_post_measurement","tanks_with_both_boundaries_bracketed")},
            landsat_15m_paired_tanks=int(l["efr_paired_landsat_pan"]),
            landsat_30m_context_only_tanks=int(l["efr_added_only_by_context"]),
            sentinel1_slc_common_relative_track=s["slc_common_relative_track"]=="True",
            sentinel1_grd_common_relative_track=s["grd_common_relative_track"]=="True",
            wti_near_month=m["near_month"],wti_far_month=m["far_month"],**prices,
            wti_replay_source=next((source for source in REPLAY_SOURCE_ORDER if prices[f"wti_{source}_complete_window"]),None),
            wti_any_complete_window=any(prices.values())))
        by_terminal.extend(dict(summarize_area(clock,indexed_terminals[week,site],measured_phase=True),terminal_id=site)
                           for site in sites)
    return by_week, by_terminal


def view_features(weekly, terminal, volumes, *, with_alternates):
    for stem in ("sentinel2_weekly_multiband_holdout_recovery", "sentinel2_weekly_refined_alignment_holdout"):
        summary = json.loads((DERIVED/f"{stem}_summary.json").read_text())
        if summary.get("included_alternate_pairs",False)!=with_alternates:
            raise ValueError(f"The {stem} cohort differs from the requested B04 cohort")
    spectral = indexed("sentinel2_weekly_multiband_holdout_recovery_weeks")
    refined = {r["week_ending"]:r for r in read_csv("sentinel2_weekly_refined_alignment_holdout_weeks")
               if r["comparison_scope"]=="all_modes_resolved"}
    if spectral.keys()!=refined.keys() or spectral.keys()!={r["week_ending"] for r in weekly}:
        raise ValueError("The holdout optical methods use different report calendars")
    registered, amounts = defaultdict(list), defaultdict(list)
    for r in volumes:
        if r["terminal_id"]!="amid_cushing":
            registered[r["week_ending"]].append(float(r["capture_change_kbbl"]))
    for r in terminal:
        if r["baseline_b04_tanks"]:
            amounts[r["week_ending"]].append(r["capture_change_kbbl"])
    for r in weekly:
        week, issue = r["week_ending"],r["release_date"]
        s, a = spectral[week],refined[week]
        if any((q["release_date"],q["cutoff_utc"])!=(issue,r["cutoff_utc"]) for q in (s,a)):
            raise ValueError(f"The holdout methods disagree on the report cutoff for {week}")
        raw = r["capture_change_kbbl"]
        register = sum(registered[week]) if registered[week] else None
        bands = median(float(s[f"{m}_expanded_visible_change_kbbl"]) for m in ("nir","joint")) if int(s["original_tanks"]) else None
        alignment = median(float(a[f"{m}_visible_change_kbbl"]) for m in ("coarse","local","scene")) if int(a["coarse_matched_tanks"]) else None
        variants = [x for x in (register,bands,alignment) if x is not None]
        gross = sum(abs(value) for value in amounts[week])
        yield dict(study_phase=PHASE,week_ending=week,release_date=issue,cutoff_utc=r["cutoff_utc"],
            included_alternate_pairs=with_alternates,
            measured_b04=raw is not None,b04_tanks=r["baseline_b04_tanks"],b04_capture_change_kbbl=raw,
            b04_overlap_change_kbbl=r["overlap_change_kbbl"],b04_area_fraction=r["baseline_b04_area_fraction"],
            b04_space_time_fraction=r["baseline_b04_space_time_fraction"],
            b04_following_hours=r["area_weighted_following_hours"],measured_terminals=len(amounts[week]),
            terminal_coherence=(min(1.,abs(raw)/gross) if gross else 0.) if raw is not None else None,
            registered_variant_kbbl=register,spectral_variant_kbbl=bands,refined_alignment_variant_kbbl=alignment,
            multiview_capture_change_kbbl=median([raw,*variants]) if raw is not None else None,
            nir_additional_tanks=int(s["nir_additional_tanks"]),joint_additional_tanks=int(s["joint_additional_tanks"]),
            alignment_comparable_tanks=int(a["coarse_matched_tanks"]),wti_replay_source=r["wti_replay_source"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-alternates", action="store_true", help="Include additional registered pre-measurement pairs")
    parser.add_argument("--with-views", action="store_true", help="Join computed spectral and geometric inputs")
    args = parser.parse_args()
    weeks = study_weeks(PHASE)
    volumes, primary_counts, checked = measurements(weeks,args.with_alternates)
    registry = roof_registry()
    tanks = list(tank_rows(registry,phases=(PHASE,),measurements=volumes,alternative_checked=checked))
    weekly, terminal = week_panels(weeks,tanks,registry,primary_counts)
    measured = [r for r in weekly if r["baseline_b04_tanks"]]
    report = dict(reporting_weeks=len(weeks), weeks_with_primary_measurements=len(primary_counts),
        included_alternate_pairs=args.with_alternates, weeks_with_b04_measurements=len(measured),
        weeks_with_b04_and_complete_wti=sum(r["wti_any_complete_window"] for r in measured),
        wti_replay_source_order=REPLAY_SOURCE_ORDER,
        wti_replay_windows_by_source=dict(Counter(r["wti_replay_source"] or "missing" for r in weekly)),
        tank_weeks_with_b04_measurements=len(volumes),
        weeks_by_b04_status=dict(Counter(r["b04_status"] for r in weekly)),
        median_area_fraction_on_measured_weeks=median(r["baseline_b04_area_fraction"] for r in measured) if measured else None,
        median_space_time_fraction_on_measured_weeks=median(r["baseline_b04_space_time_fraction"] for r in measured) if measured else None,
        interpretation="Coverage uses the as-of dated EFR registry. A B04 result requires two resolved shadows and "
        "independent exterior alignment. Landsat and Sentinel-1 fields describe candidate availability only. "
        "No inventory values or market closes from the holdout are included.")
    for suffix,rows in (("tank_weeks",tanks),("weeks",weekly),("terminal_weeks",terminal)):
        export(f"{STEM}_{suffix}",rows)
    if args.with_views:
        views = list(view_features(weekly,terminal,volumes,with_alternates=args.with_alternates))
        view_file = f"{STEM}_{'expanded' if args.with_alternates else 'primary'}_view_features"
        export(view_file,views)
        report["view_features"] = dict(file=f"{view_file}.csv",weeks=len(views),weeks_with_b04_and_all_three_views=sum(
            r["measured_b04"] and all(r[k] is not None for k in ("registered_variant_kbbl", "spectral_variant_kbbl",
                                                               "refined_alignment_variant_kbbl")) for r in views))
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
