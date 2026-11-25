"""Measure WTI spread reactions on synchronized minute bars of exact dated contracts."""

import csv
import io
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from statistics import median
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from backtest_wti_cushing_spread import tick_price
from build_s1_reference_matches import DERIVED, export, read
from build_wti_curve import MONTH_CODES
from collect_public import RAW, get, write

STEM = "cushing_cl_intraday_sample"
REVISION = "29efdfa21c5a5b2d7aa306397385cf116e011559"
BASE = f"https://huggingface.co/datasets/lynx1231/historical-futures-data-sample/resolve/{REVISION}/"
STORE = RAW/"wti_reference"/"intraday_sample"
CHICAGO = ZoneInfo("America/Chicago")
HORIZONS = (1, 5, 15, 30, 60)
OFFSETS = (-2, *HORIZONS)
POLICIES = ("curve", "public_ml", "adaptive_scene_measured_only", "recovered_scene_measured_only")
ACTIVE_MARK = "both_legs_within_minute"


def symbol(month):
    return "CL"+MONTH_CODES[int(month[4:])-1]+month[2:4]


def catalog():
    data = get(BASE+"files.csv",STORE/"files.csv")
    result = {}
    for r in csv.DictReader(io.StringIO(data.decode())):
        if (r["root"],r["exchange"],r["frequency"])!=("CL","NYMEX","minute"):
            continue
        month = f"{int(r['delivery_year']):04d}{int(r['delivery_month']):02d}"
        if symbol(month)!=r["contract_symbol"] or month in result:
            raise ValueError(f"Incorrect provider instrument identity for {month}")
        result[month] = r
    return result


def read_marks(month, metadata, wanted):
    path = STORE/f"{symbol(month)}.parquet"
    url = BASE+metadata["path"]
    raw = get(url,path)
    digest = sha256(raw).hexdigest()
    if digest!=metadata["sha256"]:
        raise ValueError(f"Provider checksum differs for {month}")
    columns = ["exchange","contract_symbol","timestamp_ms","timestamp_chicago_wall","open","high","low","close","volume"]
    table = pq.read_table(pa.BufferReader(raw),columns=columns)
    if len(table)!=int(metadata["row_count"]):
        raise ValueError(f"Provider row count differs for {month}")
    local = [t.astimezone(CHICAGO).replace(tzinfo=None) for t in wanted]
    selected = table.filter(pc.is_in(table["timestamp_chicago_wall"],value_set=pa.array(local,type=pa.timestamp("ns"))))
    marks, seen = {}, set()
    for r in selected.to_pylist():
        wall = r["timestamp_chicago_wall"]
        if (r["exchange"],r["contract_symbol"])!=("NYMEX",symbol(month)) or int(wall.replace(tzinfo=timezone.utc).timestamp()*1000)!=r["timestamp_ms"]:
            raise ValueError(f"Wrong contract or inconsistent provider clock for {month}")
        if wall.replace(tzinfo=CHICAGO,fold=0).utcoffset()!=wall.replace(tzinfo=CHICAGO,fold=1).utcoffset():
            raise ValueError(f"Ambiguous Chicago clock for {month}")
        instant = wall.replace(tzinfo=CHICAGO).astimezone(timezone.utc)
        if instant in seen:
            raise ValueError(f"Duplicate minute for {month} at {instant}")
        seen.add(instant)
        if r["volume"] is None or r["volume"]<=0:
            continue
        o,h,l,c = (tick_price(r[k]) for k in ("open","high","low","close"))
        if None in (o,h,l,c) or not l<=min(o,c)<=max(o,c)<=h:
            raise ValueError(f"Invalid OHLC for {month} at {instant}")
        marks[instant] = (c,r["volume"],digest)
    return month,marks


def experiment(rows, sources, quotes, *, lookback=0):
    for r in rows:
        issue = r["release_date"]
        near,far = r["near_month"],r["far_month"]
        report,cutoff,entry,exit_at = [datetime.fromisoformat(r[k]).astimezone(timezone.utc) for k in (
            "scheduled_report_at_utc","cutoff_utc","entry_at_utc","exit_at_utc")]
        if not cutoff<entry<report+timedelta(minutes=HORIZONS[-1])<exit_at or not entry<report-timedelta(minutes=2)<report:
            raise ValueError(f"Invalid reporting or market clock on {issue}")
        covered = near in sources and far in sources
        for horizon in HORIZONS:
            target = (report-timedelta(minutes=2),report+timedelta(minutes=horizon))
            times = []
            for i,t in enumerate(target):
                candidates = (t-timedelta(minutes=lag) for lag in range(lookback+1)
                              if i==0 or t-timedelta(minutes=lag)>=report)
                times.append(next((q for q in candidates if all(q in quotes.get(m,{}) for m in (near,far))),t))
            marks = [quotes.get(month,{}).get(t) for t in times for month in (near,far)]
            ready = all(mark is not None for mark in marks)
            move = 10*(marks[2][0]-marks[3][0]-marks[0][0]+marks[1][0]) if ready else None
            yield dict(release_date=issue,study_phase=r["study_phase"],near_month=near,far_month=far,
                       measured_roofs=int(r["measured_roofs"]),report_utc=report.isoformat(),horizon_minutes=horizon,
                       prior_bar_max_age_minutes=lookback,
                       pre_target_label_utc=target[0].isoformat(),post_target_label_utc=target[1].isoformat(),
                       pre_bar_label_utc=times[0].isoformat(),post_bar_label_utc=times[1].isoformat(),
                       mark_type=ACTIVE_MARK,
                       contracts_in_public_sample=covered,positive_volume_both_legs_both_marks=ready,
                       long_spread_move_usd=move,
                       **{f"{leg}_{phase}_{field}":mark[idx] if mark else None for (phase,leg),mark in zip(
                           ((p,l) for p in ("pre","post") for l in ("near","far")), marks,strict=True)
                           for idx,field in enumerate(("close_ticks","volume","sha256"))})


def evaluate(rows, policies):
    output = {}
    for h in HORIZONS:
        sample = [r for r in rows if r["horizon_minutes"]==h and r["positive_volume_both_legs_both_marks"]]
        if not sample:
            output[h] = dict(paired_releases=0)
            continue
        results = {}
        for method in POLICIES:
            # These directions were calculated in the prior study with the
            # previous-day settlement entry. This step measures their sign at
            # different marks without fitting or using the intraday outcomes.
            sides = [policies[r["release_date"],method] for r in sample]
            gross = sum(s*r["long_spread_move_usd"] for s,r in zip(sides,sample,strict=True))
            trades = sum(s!=0 for s in sides)
            results[method] = dict(trades=trades,gross_usd=gross,net_170_usd=gross-170*trades,net_340_usd=gross-340*trades)
        moves = [abs(r["long_spread_move_usd"]) for r in sample]
        output[h] = dict(paired_releases=len(sample),measured_releases=sum(r["measured_roofs"]>0 for r in sample),
                         median_absolute_long_spread_move_usd=median(moves),max_absolute_long_spread_move_usd=max(moves),
                         moves_above_170_usd=sum(v>170 for v in moves),
                         release_dates=[r["release_date"] for r in sample],policies=results)
    return output


def clock_crosscheck(rows, quotes):
    """Compare the documented clock with shifts of one hour in each direction."""
    checks = {}
    for offset in (-60,0,60):
        ready = 0
        totals = []
        for r in rows:
            near,far = (r[k] for k in ("near_month","far_month"))
            if near not in quotes or far not in quotes:
                continue
            target = datetime.fromisoformat(r["scheduled_report_at_utc"]).astimezone(timezone.utc)+timedelta(minutes=offset)
            nearby = (target+timedelta(minutes=i) for i in range(-2,3))
            groups = [[quotes[m][t][1] for m in (near,far)] for t in nearby if all(t in quotes[m] for m in (near,far))]
            if groups:
                ready += 1
                totals.append(sum(sum(g) for g in groups))
        checks[offset] = dict(releases_with_common_traded_minute=ready, total_paired_trade_volume=sum(totals))
    return checks


def main():
    source = catalog()
    archived = read("cushing_prerelease_sensor_trades")
    events = [r for r in archived if r["method"]=="curve" and r["study_phase"]=="holdout"]
    policies = {(r["release_date"],r["method"]):int(r["near_side"]) for r in archived if r["method"] in POLICIES}
    wanted = defaultdict(set)
    for r in events:
        near,far = r["near_month"],r["far_month"]
        if near not in source or far not in source:
            continue
        report = datetime.fromisoformat(r["scheduled_report_at_utc"]).astimezone(timezone.utc)
        for m in (near,far):
            wanted[m].update(report+timedelta(minutes=offset-lag) for offset in OFFSETS for lag in range(3))
            wanted[m].update(report+timedelta(minutes=offset+lag) for offset in (-60,0,60) for lag in range(-2,3))
    with ThreadPoolExecutor(max_workers=4) as pool:
        marks = dict(pool.map(lambda m:read_marks(m,source[m],wanted[m]),sorted(wanted)))
    outcomes = list(experiment(events,source,marks))
    bounded = list(experiment(events,source,marks,lookback=2))
    results = evaluate(outcomes,policies)
    report = dict(status="limited_public_trade_bar_sample",provider_revision=REVISION,reviewed_releases=len(events),
                  releases_with_both_expiries_in_catalog=sum(r["contracts_in_public_sample"] and r["horizon_minutes"]==1 for r in outcomes),
                  source_files=[dict(month=m,url=BASE+source[m]["path"],sha256=source[m]["sha256"]) for m in sorted(wanted)],
                  interpretation="Trade-bar closes at the same minute label are reference marks. Bid/ask and fill simultaneity are unobserved.",
                  clock_alignment_diagnostic=clock_crosscheck(events,marks), horizons=results,
                  two_minute_backwards_sensitivity=evaluate(bounded,policies))
    export(f"{STEM}_windows",outcomes+bounded)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
