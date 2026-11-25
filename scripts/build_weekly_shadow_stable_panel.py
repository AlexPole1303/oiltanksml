"""Measure temporal overlap and fixed-cohort coverage on the development period."""
import json
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from statistics import StatisticsError, correlation, fmean, median

from build_weekly_shadow_development_check import ACTUAL, numeric, read_csv, scores
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only, write

SELECTION_CUTOFF = "2019-01-01"
TANKS_PER_TERMINAL = 3
BASE = "last_published_change_kbbl"
RESIDUAL = "actual_minus_last_published_kbbl"
SIGNALS = (
    "resolved_visible_kbbl",
    "resolved_overlap_kbbl",
    "stable_visible_kbbl",
    "stable_overlap_kbbl",
)


def select_cohort(pairs):
    by_terminal = defaultdict(dict)
    for pair in pairs:
        if pair["release_date"] >= SELECTION_CUTOFF or pair["roof_type"] != "EFR":
            continue
        candidate = pair["candidate_id"]
        row = by_terminal[pair["terminal_id"]].setdefault(candidate,dict(
            terminal_id=pair["terminal_id"],candidate_id=candidate,cohort_id=pair["cohort_id"],
            circular_area_m2=float(pair["circular_area_m2"]),selection_pairs=0,selection_resolved_pairs=0,
            selection_last_release=None,selection_last_resolved_release=None))
        row["selection_pairs"] += 1
        row["selection_last_release"] = max(row["selection_last_release"] or "",pair["release_date"])
        if pair["both_shadow_diagnostics_pass"] == "True":
            row["selection_resolved_pairs"] += 1
            row["selection_last_resolved_release"] = max(row["selection_last_resolved_release"] or "",pair["release_date"])
    for terminal,candidates in sorted(by_terminal.items()):
        ranked = sorted(candidates.values(),key=lambda r:(-r["selection_resolved_pairs"],-r["selection_pairs"],
                                                          -r["circular_area_m2"],r["candidate_id"]))
        for rank,row in enumerate(ranked,1):
            yield dict(row,selection_release_before=SELECTION_CUTOFF,terminal_rank=rank,
                       in_fixed_cohort=rank<=TANKS_PER_TERMINAL and row["selection_resolved_pairs"]>0)


def window_clock(db):
    rows = db.execute("""SELECT p.week_ending,p.window_ends_by_utc,c.window_ends_by_utc,c.release_date,c.inputs_available_before_utc
        FROM eia_satellite_window p JOIN eia_satellite_window c USING(week_ending)
        WHERE p.endpoint='previous' AND c.endpoint='current' AND c.study_phase='development' ORDER BY p.week_ending""")
    return {week:(datetime.fromisoformat(start),datetime.fromisoformat(end),release,cutoff) for week,start,end,release,cutoff in rows}


def pair_rows(pairs, clocks, cohort):
    ranks = {(r["terminal_id"],r["candidate_id"]):r["terminal_rank"] for r in cohort if r["in_fixed_cohort"]}
    for p in pairs:
        week = p["week_ending"]
        start,end,release,cutoff = clocks[week]
        first,last = (datetime.fromisoformat(p[f"{side}_capture_at_utc"]) for side in ("earlier","later"))
        if not first<=start<last<=end or release!=p["release_date"] or cutoff!=p["inputs_available_before_utc"]:
            raise ValueError(f"Pair disagrees with EIA window for {week} {p['candidate_id']}")
        capture = (last-first).total_seconds()/3600
        overlap = (last-start).total_seconds()/3600
        eia_interval = (end-start).total_seconds()/3600
        raw = numeric(p["optical_change_equivalent_bbl"])
        rank = ranks.get((p["terminal_id"],p["candidate_id"]))
        yield dict(p,previous_eia_measurement_at_utc=start.isoformat(),current_eia_measurement_at_utc=end.isoformat(),
            capture_interval_hours=capture,eia_interval_hours=eia_interval,overlap_hours=overlap,
            capture_before_eia_interval_hours=capture-overlap,eia_after_last_capture_hours=eia_interval-overlap,
            fraction_of_capture_inside_eia_interval=overlap/capture,fraction_of_eia_interval_observed=overlap/eia_interval,
            overlap_equivalent_bbl=raw*overlap/capture if raw is not None else None,
            fixed_cohort_terminal_rank=rank,in_fixed_cohort=rank is not None)


def optics(rows, members):
    resolved = [p for p in rows if p["both_shadow_diagnostics_pass"]=="True" and p["overlap_equivalent_bbl"] is not None]
    stable_pairs = [p for p in rows if p["in_fixed_cohort"]]
    stable = [p for p in resolved if p["in_fixed_cohort"]]
    complete = len(stable)==len(members)>0
    area = sum(p["circular_area_m2"] for p in members)
    return dict(all_efr_pairs=len(rows),all_resolved_pairs=len(resolved),fixed_cohort_size=len(members),
        fixed_cohort_eligible_pairs=len(stable_pairs),fixed_cohort_resolved_pairs=len(stable),fixed_cohort_complete=complete,
        fixed_cohort_candidate_ids="|".join(sorted(p["candidate_id"] for p in members)),
        fixed_cohort_resolved_candidate_ids="|".join(sorted(p["candidate_id"] for p in stable)),
        fixed_cohort_area_m2=area,fixed_cohort_resolved_area_m2=sum(float(p["circular_area_m2"]) for p in stable),
        median_resolved_capture_interval_hours=median(p["capture_interval_hours"] for p in resolved) if resolved else None,
        median_resolved_eia_interval_fraction=median(p["fraction_of_eia_interval_observed"] for p in resolved) if resolved else None,
        resolved_visible_kbbl=sum(float(p["optical_change_equivalent_bbl"]) for p in resolved)/1000 if resolved else None,
        resolved_overlap_kbbl=sum(p["overlap_equivalent_bbl"] for p in resolved)/1000 if resolved else None,
        stable_visible_kbbl=sum(float(p["optical_change_equivalent_bbl"]) for p in stable)/1000 if stable else None,
        stable_overlap_kbbl=sum(p["overlap_equivalent_bbl"] for p in stable)/1000 if stable else None)


def exposure_rows(db):
    out = defaultdict(list)
    rows = db.execute("""SELECT release_date,terminal_id,ticker,issuer,role,interest_pct,valid_from,valid_to_exclusive,
        date_precision,exposure_date_status,source_url FROM eia_event_operator_exposure x
        WHERE EXISTS (SELECT 1 FROM eia_satellite_window w WHERE w.endpoint='current'
            AND w.study_phase='development' AND w.release_date=x.release_date)
        ORDER BY release_date,terminal_id,ticker,role,valid_from""")
    fields = [d[0] for d in rows.description[2:]]
    for release,terminal,*values in rows:
        out[release,terminal].append(dict(zip(fields,values,strict=True)))
    return out


def panels(db, pairs, cohort, clocks):
    by_week = defaultdict(list)
    by_terminal = defaultdict(list)
    members = defaultdict(list)
    terminals = {p["terminal_id"] for p in pairs if p["roof_type"]=="EFR"}
    for row in cohort:
        if row["in_fixed_cohort"]:
            members[row["terminal_id"]].append(row)
    for pair in pairs:
        if pair["roof_type"]=="EFR":
            by_week[pair["week_ending"]].append(pair)
            by_terminal[pair["week_ending"],pair["terminal_id"]].append(pair)
    exposures = exposure_rows(db)
    selected = [p for p in cohort if p["in_fixed_cohort"]]
    weekly,terminal_weekly = [],[]
    for week,release,actual,base,last_issue,previous_issue,source in db.execute(f"""SELECT week_ending,release_date,{ACTUAL},
        latest_weekly_change_known_before_release,latest_vintage_known_before_release,prior_vintage_for_latest_known_change,source_url
        FROM eia_inventory_release_panel WHERE study_phase='development' AND week_ending>=? ORDER BY week_ending""",(SELECTION_CUTOFF,)):
        if not (last_issue and previous_issue and last_issue<release and previous_issue<release):
            raise ValueError(f"EIA reference timing missing or invalid for {week}")
        start,end,_,cutoff = clocks[week]
        clock = dict(week_ending=week,release_date=release,previous_eia_measurement_at_utc=start.isoformat(),
                     current_eia_measurement_at_utc=end.isoformat(),inputs_available_before_utc=cutoff)
        week_optics = optics(by_week[week],selected)
        combined = {f"last_plus_{name}":base+week_optics[name] if week_optics[name] is not None else None for name in SIGNALS}
        weekly.append(dict(clock,**week_optics,**{ACTUAL:actual,BASE:base,RESIDUAL:actual-base},**combined,
                           last_eia_vintage=last_issue,prior_eia_vintage=previous_issue,eia_source_url=source))
        for terminal in sorted(terminals):
            terminal_weekly.append(dict(clock,terminal_id=terminal,**optics(by_terminal[week,terminal],members[terminal]),
                release_day_issuer_relations_json=json.dumps(exposures[release,terminal],ensure_ascii=False)))
    return weekly,terminal_weekly


def residual_comparison(rows, name):
    sample = [r for r in rows if r[name] is not None]
    signal = [r[name] for r in sample]
    actual = [r[RESIDUAL] for r in sample]
    try:
        corr = correlation(signal,actual)
    except StatisticsError:
        corr = None
    with_signal = scores(sample,f"last_plus_{name}")
    base = scores(sample,BASE)
    sign_matches = [((x>0)-(x<0))==((y>0)-(y<0)) for x,y in zip(signal,actual,strict=True)]
    return dict(matched_weeks=len(sample),distinct_issue_dates=len({r["release_date"] for r in sample}),
        residual_correlation=corr,residual_sign_agreement=fmean(sign_matches) if sample else None,
        last_published=base,last_plus_optical_unit_weight=with_signal,
        delta_mean_abs_error_kbbl=(with_signal["mean_abs_error_kbbl"]-base["mean_abs_error_kbbl"]) if sample else None)


def summarize(weeks, terminals, pairs, cohort):
    fixed_counts = Counter(r["terminal_id"] for r in cohort if r["in_fixed_cohort"])
    complete = Counter(r["terminal_id"] for r in terminals if r["fixed_cohort_complete"])
    resolved = [p for p in pairs if p["week_ending"]>=SELECTION_CUTOFF and p["roof_type"]=="EFR"
                and p["both_shadow_diagnostics_pass"]=="True"]
    by_year = defaultdict(list)
    for row in weeks:
        by_year[row["week_ending"][:4]].append(row)
    common_weeks = [r for r in weeks if all(r[s] is not None for s in SIGNALS)]
    return dict(selection_release_before=SELECTION_CUTOFF,selection_tanks_per_terminal_max=TANKS_PER_TERMINAL,
        assessment_weeks=len(weeks),fixed_cohort_members_by_terminal=dict(fixed_counts),
        fully_resolved_terminal_weeks={k:complete[k] for k in fixed_counts},
        fixed_cohort_complete_all_terminal_weeks=sum(r["fixed_cohort_complete"] for r in weeks),
        weeks_with_any_fixed_cohort_measurement=sum(r["fixed_cohort_resolved_pairs"]>0 for r in weeks),
        median_resolved_efr_pair_fraction_of_eia_interval=median(p["fraction_of_eia_interval_observed"] for p in resolved) if resolved else None,
        median_resolved_efr_pair_fraction_of_capture_interval=median(p["fraction_of_capture_inside_eia_interval"] for p in resolved) if resolved else None,
        median_resolved_efr_gap_hours_from_last_capture_to_eia=median(p["eia_after_last_capture_hours"] for p in resolved) if resolved else None,
        coverage_by_week_year={year:dict(weeks=len(rows),any_resolved=sum(r["all_resolved_pairs"]>0 for r in rows),
            any_fixed=sum(r["fixed_cohort_resolved_pairs"]>0 for r in rows)) for year,rows in by_year.items()},
        residual_comparisons={name:residual_comparison(weeks,name) for name in SIGNALS},
        common_weeks_residual_comparisons={name:residual_comparison(common_weeks,name) for name in SIGNALS},
        scope="Exploratory development assessment only. The overlap term assumes constant movement during the observed interval. "
              "A partially observed cohort remains partial. No extrapolation, outcome-based selection, or weight fitting is used. "
              "Release-day issuer relations use retrospective registry evidence and do not establish first public availability.")


def main():
    source_pairs = read_csv("weekly_shadow_development_pairs")
    cohort = list(select_cohort(source_pairs))
    with closing(read_only(SOURCE)) as db:
        clocks = window_clock(db)
        pairs = list(pair_rows(source_pairs,clocks,cohort))
        weeks,terminals = panels(db,pairs,cohort,clocks)
    export("weekly_shadow_stable_cohort",cohort)
    export("weekly_shadow_eia_overlap_pairs",pairs)
    export("weekly_shadow_stable_weeks",weeks)
    export("weekly_shadow_stable_terminals",terminals)
    report = summarize(weeks,terminals,pairs,cohort)
    write(DERIVED/"weekly_shadow_stable_summary.json",json.dumps(report,indent=2).encode())
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
