"""Verify event-window trade bars from the read-only IBKR market-data connector."""

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from backtest_wti_cushing_spread import tick_price
from build_s1_reference_matches import DERIVED, export, read
from collect_public import RAW, ROOT, write

STEM = "cushing_ibkr_intraday"
STORE = RAW/"wti_reference"/"ibkr_intraday"
SIZES = {"ONE_MIN":60,"FIVE_MINS":300,"FIFTEEN_MINS":900,"THIRTY_MINS":1800}


def read_packets():
    identifiers = {r["contract_month"]:r["contract_id"] for r in json.loads((RAW/"ibkr/cl_contracts.json").read_bytes())["contracts"]}
    book, sources = {}, []
    for path in sorted(STORE.glob("*.json")):
        raw = path.read_bytes()
        packet = json.loads(raw)
        response, request = packet["response"],packet["request"]
        month = packet["contract_month"]
        step = SIZES[request["step"]]
        if (request["contract_id"]!=identifiers[month] or request["exchange"]!="NYMEX" or request["security_type"]!="FUT"
            or response["source"]!="Last" or response["chart_step"]!=step or not request["outside_rth"]):
            raise ValueError(f"Incorrect contract identity or bar definition in {path.name}")
        stamps = [datetime.fromisoformat(t) for t in response["time"]]
        fetched = datetime.fromisoformat(packet["retrieved_utc"])
        start,end = (datetime.fromisoformat(response[k]) for k in ("chart_start","chart_end"))
        if not stamps or stamps!=sorted(set(stamps)) or not start<=stamps[0]<=stamps[-1]<end<=fetched:
            raise ValueError(f"Inconsistent response clock in {path.name}")
        digest = sha256(raw).hexdigest()
        positive, carried = 0, 0
        for t,o,h,l,c,v in zip(stamps,*(response[k] for k in ("open","high","low","close","volume")),strict=True):
            if t.timestamp()%step or t+timedelta(seconds=step)>fetched or v is None or v<0:
                raise ValueError(f"Invalid bar clock or volume in {path.name}")
            if v==0:
                carried += 1
                continue
            o,h,l,c = (tick_price(x) for x in (o,h,l,c))
            if None in (o,h,l,c) or not l<=min(o,c)<=max(o,c)<=h:
                raise ValueError(f"Invalid OHLC in {path.name} at {t}")
            key = month,step,t
            mark = c,v,digest
            if key in book and book[key][:2]!=mark[:2]:
                raise ValueError(f"Different published IBKR observations for {key}")
            book[key] = mark
            positive += 1
        sources.append(dict(contract_month=month,conid=request["contract_id"],bar_seconds=step,first_utc=start.isoformat(),
            end_utc=end.isoformat(),downloaded_utc=fetched.isoformat(),reported_delay_seconds=response["delayed"],
            bars_received=len(stamps),positive_volume_bars=positive,zero_volume_bars=carried,
            source_file=str(path.relative_to(ROOT)),sha256=digest))
    return book,sources


def events():
    for r in read("cushing_prerelease_sensor_trades"):
        if r["method"]=="curve" and r["study_phase"]=="holdout":
            yield r["release_date"],r["scheduled_report_at_utc"],r["near_month"],r["far_month"]
    releases = {r["release_date"]:r for r in read("eia_wpsr_forward_releases")}
    for window in read("cushing_forward_wti_coverage_windows"):
        issue = window["release_date"]
        if releases[issue]["scheduled_release_at_utc"]:
            yield issue,releases[issue]["scheduled_release_at_utc"],window["near_month"],window["far_month"]


def study(book,sources, *, common_minutes=None):
    coverage = defaultdict(list)
    for r in sources:
        coverage[r["contract_month"],r["bar_seconds"]].append(
            (datetime.fromisoformat(r["first_utc"]),datetime.fromisoformat(r["end_utc"])))
    for issue,stamp,near,far in events():
        report = datetime.fromisoformat(stamp).astimezone(timezone.utc)
        for step in SIZES.values():
            if not all(any(first<=report<last for first,last in coverage[m,step]) for m in (near,far)):
                continue
            step_delta = timedelta(seconds=step)
            # In the common-clock diagnostic all resolutions have the same
            # pre-release close boundary and the same post-release close times.
            horizons = (15,30,60) if common_minutes else (1,5,15,30,60)
            for minutes in (i for i in horizons if i*60>=step):
                desired_end = report+timedelta(minutes=minutes)
                pre_end = report-timedelta(minutes=common_minutes) if common_minutes else report-step_delta
                # A 30-minute NYMEX bar cannot close 15 minutes before a
                # regularly scheduled 10:30 ET release. Do not round its label.
                if common_minutes and (pre_end.timestamp()%step or desired_end.timestamp()%step):
                    continue
                pre = pre_end-step_delta
                post = desired_end-(step_delta if common_minutes else timedelta(0))
                marks = [book.get((m,step,t)) for t in (pre,post) for m in (near,far)]
                complete = all(mark is not None for mark in marks)
                move = 10*(marks[2][0]-marks[3][0]-marks[0][0]+marks[1][0]) if complete else None
                yield dict(release_date=issue,report_utc=report.isoformat(),near_month=near,far_month=far,bar_seconds=step,
                    clock=(f"pre_close_minus_{common_minutes}m" if common_minutes else "resolution_specific_bar_starts"),
                    horizon_minutes=minutes,pre_bar_start_utc=pre.isoformat(),post_bar_start_utc=post.isoformat(),
                    pre_bar_end_utc=(pre+step_delta).isoformat(),post_bar_end_utc=(post+step_delta).isoformat(),
                    positive_volume_both_legs_both_marks=complete,long_spread_move_usd=move,
                    **{f"{leg}_{edge}_{field}":mark[i] if mark else None for (edge,leg),mark in zip(
                        ((e,l) for e in ("pre","post") for l in ("near","far")),marks,strict=True)
                        for i,field in enumerate(("ticks","volume","source_sha256"))})


def main():
    book,sources = read_packets()
    windows = list(study(book,sources))
    valid = [r for r in windows if r["positive_volume_both_legs_both_marks"]]
    common = list(study(book,sources,common_minutes=15))
    half_hour = list(study(book,sources,common_minutes=30))
    same_boundaries = defaultdict(list)
    for r in common:
        if not r["positive_volume_both_legs_both_marks"]:
            continue
        key = r["release_date"],r["horizon_minutes"]
        same_boundaries[key].append(r)
    agreement = []
    for (day,horizon),group in sorted(same_boundaries.items()):
        if len(group)>1:
            values = {r["bar_seconds"]:r["long_spread_move_usd"] for r in group}
            agreement.append(dict(release_date=day,horizon_minutes=horizon,move_usd_by_bar_seconds=values,
                                  spread_across_bar_sizes_usd=max(values.values())-min(values.values())))
    forecasts = {r["issue"]:r for r in read("cushing_prerelease_sensor_forecasts")}
    satellite = []
    for (day,horizon),group in sorted(same_boundaries.items()):
        r = forecasts.get(day)
        if r is None or not int(r["measured_roofs"]):
            continue
        quote = min(group,key=lambda q:q["bar_seconds"])
        estimate, baseline, actual = [float(r[k]) for k in ("early_adaptive_scene","early_public_ml","inventory")]
        if not datetime.fromisoformat(r["cutoff_utc"])<datetime.fromisoformat(quote["pre_bar_start_utc"]):
            raise ValueError(f"Satellite forecast timestamp differs from IBKR event on {day}")
        delta = estimate-baseline
        side = int(delta<0)-int(delta>0)
        move = quote["long_spread_move_usd"]
        satellite.append(dict(release_date=day,horizon_minutes=horizon,bar_seconds=quote["bar_seconds"],
            measured_roofs=int(r["measured_roofs"]),optical_revision_kbbl=1000*delta,realized_surprise_kbbl=1000*(actual-baseline),
            long_spread_move_usd=move,direction_only_gross_usd=side*move,direction_only_net_170_usd=side*move-170*abs(side)))
    broad = defaultdict(list)
    for r in half_hour:
        if r["positive_volume_both_legs_both_marks"]:
            broad[r["release_date"],r["horizon_minutes"]].append(r)
    half_hour_results = {}
    for h in (30,60):
        groups = [g for (_,horizon),g in sorted(broad.items()) if horizon==h]
        sample = [min(g,key=lambda r:r["bar_seconds"]) for g in groups]
        moves = [abs(r["long_spread_move_usd"]) for r in sample]
        optical = []
        for r in sample:
            f = forecasts.get(r["release_date"])
            if f is None or not int(f["measured_roofs"]):
                continue
            if not datetime.fromisoformat(f["cutoff_utc"])<datetime.fromisoformat(r["pre_bar_start_utc"]):
                raise ValueError(f"Optical clock differs on {r['release_date']}")
            update = float(f["early_adaptive_scene"])-float(f["early_public_ml"])
            side = int(update<0)-int(update>0)
            optical.append(dict(release_date=r["release_date"],horizon_minutes=h,bar_seconds=r["bar_seconds"],
                optical_revision_kbbl=1000*update, move_usd=r["long_spread_move_usd"],near_side=side,
                net_170_usd=side*r["long_spread_move_usd"]-170*abs(side)))
        half_hour_results[h] = dict(releases=len(sample),max_absolute_move_usd=max(moves,default=None),
            moved_more_than_170_usd=sum(v>170 for v in moves),matched_optical_releases=len(optical),
            optical_direction_net_170_usd=sum(r["net_170_usd"] for r in optical),optical_release_rows=optical)
    report = dict(status="delayed_trade_bars_reference_marks",sources=sources,release_dates=sorted({r["release_date"] for r in windows}),
        releases_with_complete_marks=sorted({r["release_date"] for r in valid}),valid_windows=len(valid),requested_windows=len(windows),
        largest_abs_move_usd=max((abs(r["long_spread_move_usd"]) for r in valid),default=None),
        moves_above_170_usd=sum(abs(r["long_spread_move_usd"])>170 for r in valid), common_clock_paired_bar_sizes=agreement,
        common_half_hour_event_coverage=sorted({r["release_date"] for r in half_hour if r["positive_volume_both_legs_both_marks"]}),
        half_hour_pre_close_diagnostic=half_hour_results,
        common_clock_optical_direction_diagnostic={h:dict(releases=sum(r["horizon_minutes"]==h for r in satellite),
                net_170_usd=sum(r["direction_only_net_170_usd"] for r in satellite if r["horizon_minutes"]==h)) for h in (15,30,60)})
    if windows:
        export(f"{STEM}_windows",windows+common+half_hour)
    if satellite:
        export(f"{STEM}_optical_diagnostic",satellite)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
