"""Index pre-release inputs, sealed target provenance, and dated CL coverage by issue."""
import json
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime

from backtest_wti_cushing_spread import NEW_YORK, SOURCES
from build_holdout_optical_inputs import PHASE, indexed
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import write
from fit_cushing_nowcast_ensemble import MAX_RELEASE, METRICS
from fit_cushing_sensor_market_nowcast import MARKET_HORIZONS

STEM = "cushing_holdout_release"
SIGNAL_RANKS = (1, 2, 4)
PRICE_FIELDS = {
    "official": ("wti_eia_dated_close", "delivery_month", "source_url", "official_close", "official_close IS NOT NULL"),
    "archive": ("wti_tradingcharts_bar", "delivery_month", "source_file", "reported_close",
                "reported_close IS NOT NULL AND volume>0 AND session_date<capture_date_chicago"),
    "ibkr": ("tws_futures_closed_sessions", "contract_month", "source_file", "close",
             "instrument='CL' AND close IS NOT NULL AND volume>0"),
}


def price_rows(db, source, *, numeric_through=None):
    table, month, reference, close, eligibility = PRICE_FIELDS[source]
    query = f"SELECT session_date,{month},{reference} FROM {table} WHERE {eligibility}"
    if numeric_through is None:
        return db.execute(query)
    return db.execute(f"SELECT session_date,{month},{reference},{close} FROM {table} "
                      f"WHERE {eligibility} AND session_date<=?",(numeric_through,))


def price_coverage(db):
    books = {source:{(d,m):ref for d,m,ref in price_rows(db,source)} for source in PRICE_FIELDS}
    curves = defaultdict(dict)
    for d,rank,month in db.execute("SELECT session_date,curve_rank,delivery_month FROM wti_nearby_curve"):
        curves[d][rank] = month
    official = [d for d, in db.execute("""SELECT session_date FROM wti_eia_dated_close WHERE official_close IS NOT NULL
        GROUP BY session_date HAVING count(*)=4 ORDER BY session_date""")]
    days = official + [d for d, in db.execute("""SELECT session_date FROM wti_curve_daily_coverage
        WHERE coverage_status='four_prices' AND session_date>? ORDER BY session_date""",(official[-1],))]
    return books, curves, days


def predictor_coverage(issue, boundary_name, boundary, books, curves, days):
    date_local = boundary.astimezone(NEW_YORK).date()
    end = bisect_left(days,date_local.isoformat())-1
    if end<max(MARKET_HORIZONS):
        raise ValueError(f"Incomplete session calendar before {issue}/{boundary_name}")
    mark = days[end]
    rows, found, ranks = [], {}, curves[mark]
    for lag in (0,*MARKET_HORIZONS):
        day = days[end-lag]
        for rank in (1,2,3,4) if lag==0 else SIGNAL_RANKS:
            month = ranks[rank]
            ranked = month in curves[day].values()
            refs = {source:book.get((day,month)) for source,book in books.items()}
            found[lag,rank] = {source:ranked and bool(ref) for source,ref in refs.items()}
            rows.append(dict(release_date=issue,boundary=boundary_name,cutoff_utc=boundary.isoformat(),
                reference_session=mark,lag_sessions=lag,observed_session=day,rank_on_reference_session=rank,
                delivery_month=month,within_first_four_on_observed_session=ranked,
                **{f"{source}_reference":ref for source,ref in refs.items()}))
    covered = {}
    for source in SOURCES:
        coverage = 0
        for lag in (0,*MARKET_HORIZONS):
            for required in ((1,),(1,2),(1,4)):
                coverage += all(found[0,rank][source] and found[lag,rank][source] for rank in required)
        covered[f"{source}_features"] = coverage
        covered[f"{source}_four_leg_snapshot"] = all(found[0,rank][source] for rank in (1,2,3,4))
    return rows, dict(reference_session=mark,calendar_days_before_boundary=(date_local-datetime.fromisoformat(mark).date()).days,
                      **covered)


def release_evidence(db):
    data = {}
    for row in db.execute("""SELECT p.week_ending,p.release_date,p.reported_as,p.source_url,
          p.change_kbbl_as_known_at_release IS NOT NULL,p.latest_weekly_change_known_before_release IS NOT NULL,
          p.latest_vintage_known_before_release,p.prior_vintage_for_latest_known_change,
          c.last_wpsr_issue,c.latest_wpsr_observation_week,c.cutoff_utc,c.last_complete_wti_curve_session,
          c.last_cftc_report_date,t.conservative_research_from_utc,
          coalesce(z.open_interest>0 AND z.money_manager_long IS NOT NULL AND z.money_manager_short IS NOT NULL,0),
          e.conservative_postrelease_from_utc,v.conservative_postrelease_from_utc,
          u.conservative_postrelease_from_utc
        FROM eia_inventory_release_panel p
        JOIN market_context_clock c ON c.release_date=p.release_date AND c.phase='pre_release'
        JOIN eia_release_clock e ON e.release_date=p.release_date
        LEFT JOIN eia_release_clock v ON v.release_date=p.latest_vintage_known_before_release
        LEFT JOIN eia_release_clock u ON u.release_date=p.prior_vintage_for_latest_known_change
        LEFT JOIN cftc_release_clock t ON t.report_date=c.last_cftc_report_date
        LEFT JOIN cftc_wti z ON z.report_date=c.last_cftc_report_date
        WHERE p.study_phase=? AND e.latest_week_in_issue=p.week_ending""",(PHASE,)):
        fields = ("week_ending","release_date","eia_target_vintage_kind","eia_target_reference","eia_target_present",
            "eia_prior_change_present","prior_stock_vintage","prior_stock_base_vintage","control_vintage",
            "control_observation_week","forecast_cutoff_utc","curve_session","cftc_report_date","cftc_available_at_utc",
            "cftc_position_inputs_present","target_available_at_utc","prior_stock_available_at_utc",
            "prior_stock_base_available_at_utc")
        item = dict(zip(fields,row,strict=True))
        data[item["week_ending"]] = item
    controls = defaultdict(dict)
    markers = ",".join("?" for _ in METRICS)
    for issue,metric,url in db.execute(f"""SELECT release_date,metric,source_url FROM eia_market_context
          WHERE phase='pre_release' AND release_date>? AND value IS NOT NULL AND metric IN ({markers})""",
          (MAX_RELEASE,*METRICS)):
        controls[issue][metric] = url
    availability = dict(db.execute("SELECT release_date,conservative_postrelease_from_utc FROM eia_release_clock"))
    return data, controls, availability


def validate_trade(issue, m, choice, bars):
    key = [(int(r["session_number"]),r["leg"]) for r in bars]
    if (len(key)!=10 or set(key)!={(i,leg) for i in range(1,6) for leg in ("near","far")}
        or choice!=("archive" if int(m["archive_available_leg_sessions"])==10 else "ibkr")
        or m["phase_ok"]!="True" or m["open_numeric"]!="False"):
        raise ValueError(f"Invalid execution calendar for {issue}")
    days = defaultdict(set)
    for row in bars:
        leg = row["leg"]
        days[int(row["session_number"])].add(row["session_date"])
        if (row["delivery_month"]!=m[f"{leg}_month"] or row[f"{choice}_present"]!="True"
            or not row[f"{choice}_reference"] or any(row[f"{s}_close_ticks"] for s in SOURCES)
            or not m["entry_session"]<=row["session_date"]<=m["exit_session"]
            or datetime.fromisoformat(row["settlement_reference_time"])<=datetime.fromisoformat(m["decision_at_utc"])):
            raise ValueError(f"Ineligible execution bar for {issue}/{leg}")
    paired = [min(days[i]) for i in range(1,6)]
    if (any(len(d)!=1 for d in days.values()) or paired!=sorted(set(paired))
        or paired[0]!=m["entry_session"] or paired[-1]!=m["exit_session"]):
        raise ValueError(f"Misaligned settlement sessions for {issue}")


def assemble(db):
    weekly, views, market = (indexed(stem) for stem in ("holdout_optical_weeks", "holdout_optical_expanded_view_features",
                                                      "wti_cushing_event_windows"))
    eia, controls, availability = release_evidence(db)
    if not weekly or not all(weekly.keys()==x.keys() for x in (views,market,eia)):
        raise ValueError("The holdout inputs disagree on issue coverage")
    bars = defaultdict(list)
    for row in read_csv("wti_cushing_event_bars"):
        if row["study_phase"]==PHASE:
            bars[row["release_date"]].append(row)
    book, curves, days = price_coverage(db)
    marks, output, last_exit = [], [], {}
    for week,w in sorted(weekly.items()):
        issue, v, m, e = w["release_date"],views[week],market[week],eia[week]
        cutoff = datetime.fromisoformat(w["cutoff_utc"])
        target_time = datetime.fromisoformat(e["target_available_at_utc"])
        decision = datetime.fromisoformat(m["decision_at_utc"])
        previous = (e["prior_stock_available_at_utc"],e["prior_stock_base_available_at_utc"],
                    availability[e["control_vintage"]],e["cftc_available_at_utc"])
        if (any(r["release_date"]!=issue for r in (v,m,e)) or v["included_alternate_pairs"]!="True"
            or v["wti_replay_source"]!=w["wti_replay_source"] or not cutoff<target_time<=decision
            or any(datetime.fromisoformat(r)!=cutoff for r in (e["forecast_cutoff_utc"],v["cutoff_utc"]))
            or any(not stamp or datetime.fromisoformat(stamp)>cutoff for stamp in previous)):
            raise ValueError(f"Unaligned holdout input clocks on {issue}")
        validate_trade(issue,m,w["wti_replay_source"],bars[issue])
        boundaries = {"report_week_end":datetime.fromisoformat(w["current_measurement_at_utc"]),"pre_release":cutoff}
        refs = {}
        for name,boundary in boundaries.items():
            rows, covered = predictor_coverage(issue,name,boundary,book,curves,days)
            marks.extend(rows)
            refs.update({f"{name}_{k}":value for k,value in covered.items()})
        if refs["pre_release_reference_session"]!=e["curve_session"]:
            raise ValueError(f"Different previous-session WTI convention on {issue}")
        c = controls[issue]
        windows = {}
        for scope,eligible in (("all",True),("b04",v["measured_b04"]=="True")):
            available = eligible and m["entry_session"]>last_exit.get(scope,"")
            if available:
                last_exit[scope] = m["exit_session"]
            windows[f"nonoverlapping_window_{scope}"] = available
        output.append(dict(study_phase=PHASE, **{k:v for k,v in e.items() if k!="curve_session"},
            control_metrics_present=len(c),control_metrics_missing="|".join(k for k in METRICS if k not in c),
            control_source_references="|".join(sorted(set(c.values()))), b04_status=w["b04_status"],
            **{k:v[k] for k in ("b04_tanks","b04_area_fraction","b04_space_time_fraction","nir_additional_tanks",
                "joint_additional_tanks","alignment_comparable_tanks","wti_replay_source")},
            optical_views_available=sum(bool(v[f"{k}_variant_kbbl"]) for k in ("registered","spectral","refined_alignment")),
            **{k:m[k] for k in ("near_month","far_month","decision_at_utc","entry_session","exit_session","entry_mark","exit_mark")},
            **refs, **windows))
    return output, marks


def main():
    with closing(read_only(SOURCE)) as db:
        output,marks = assemble(db)
        parity = {kind:dict(compared_bars=n,different_bars=d) for kind,n,d in db.execute("""SELECT comparison,count(*),
            sum(difference<>0) FROM wti_close_crosscheck WHERE session_date<=? GROUP BY comparison""",(MAX_RELEASE,))}
    report = dict(releases=len(output),predictor_price_references=len(marks),
        target_references_present=sum(r["eia_target_present"] for r in output),
        prior_change_references_present=sum(r["eia_prior_change_present"] for r in output),
        complete_control_metrics=sum(r["control_metrics_present"]==len(METRICS) for r in output),
        cftc_inputs_present=sum(r["cftc_position_inputs_present"] for r in output),
        execution_sources=dict(Counter(r["wti_replay_source"] for r in output)),
        b04_with_all_optical_views=sum(int(r["b04_tanks"])>0 and r["optical_views_available"]==3 for r in output),
        nonoverlapping_windows={scope:sum(r[f"nonoverlapping_window_{scope}"] for r in output) for scope in ("all","b04")},
        both_predictor_anchors_with_four_legs={source:sum(r[f"pre_release_{source}_four_leg_snapshot"] and
            r[f"report_week_end_{source}_four_leg_snapshot"] for r in output) for source in SOURCES},
        both_predictor_anchors_with_same_provider=sum(any(r[f"pre_release_{source}_four_leg_snapshot"] and
            r[f"report_week_end_{source}_four_leg_snapshot"] for source in SOURCES) for r in output),
        predictor_features_available={source:dict(sorted(Counter(r[f"pre_release_{source}_features"]+
            r[f"report_week_end_{source}_features"] for r in output).items())) for source in SOURCES},
        open_period_price_crosscheck=parity,
        interpretation="Coverage and provenance only. Neither target values nor WTI closes are exported. "
        "Predictor lags stay within the first four contracts on each observed session, as in the original EIA feature book. "
        "Longer maturities are retained as references for source audits, and do not count toward the feature vector.")
    export(f"{STEM}_index",output)
    export(f"{STEM}_predictor_marks",marks)
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
