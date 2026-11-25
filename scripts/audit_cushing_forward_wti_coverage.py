"""Stage dated WTI closes for releases beyond the frozen backtest."""

import json
from bisect import bisect_right
from contextlib import closing
from datetime import datetime
from zoneinfo import ZoneInfo

from backtest_wti_cushing_spread import full_sessions, settlement, tick_price
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from build_wti_curve import archive_file
from collect_public import RAW, ROOT, write

STEM = "cushing_forward_wti_coverage"
LOCAL = ZoneInfo("America/Chicago")
ARCHIVE = RAW/"wti_reference/tradingcharts_forward"


def archives(*, as_of=None):
    marks = {}
    files = []
    for folder in sorted(ARCHIVE.iterdir()):
        for path in sorted(folder.glob("CL*.json")):
            month, url, checksum, rows = archive_file(path)
            captured = datetime.fromtimestamp(path.stat().st_mtime,LOCAL)
            if as_of is not None and captured>as_of:
                continue
            relative = str(path.relative_to(ROOT))
            dates = [r["tradingDay"] for r in rows]
            if dates != sorted(set(dates)):
                raise ValueError(f"Duplicate or unordered source sessions in {path}")
            files.append(dict(month=month,source_file=relative,source_url=url,source_sha256=checksum,
                              captured_at=captured.isoformat(),bars=len(rows),first_session=min(dates),last_session=max(dates)))
            for row in rows:
                day = row["tradingDay"]
                if row["volume"]<=0 or datetime.fromisoformat(row["timestamp"]).astimezone(LOCAL).date().isoformat()!=day:
                    continue
                if settlement(day)>=captured:
                    continue
                ticks = tick_price(row["close"])
                current = marks.get((day,month))
                if current is None or captured>datetime.fromisoformat(current[3]):
                    marks[day,month] = (ticks,relative,checksum,captured.isoformat())
    return files, marks


def main():
    releases = read("eia_wpsr_forward_releases")
    files, marks = archives()
    with closing(read_only(SOURCE)) as db:
        sessions = full_sessions(db,releases[-1]["release_date"])
        contracts = list(db.execute("SELECT delivery_month,last_trading_date FROM wti_delivery_calendar ORDER BY delivery_month"))
    expiries = [expiry for _,expiry in contracts]
    legs, windows = [], []
    for r in releases:
        issue = r["release_date"]
        decision = datetime.fromisoformat(r["conservative_postrelease_from_utc"])
        start = bisect_right(sessions,issue)
        while settlement(sessions[start])<=decision:
            start += 1
        dates = sessions[start:start+5]
        front = bisect_right(expiries,sessions[start+6])
        near, far = contracts[front][0], contracts[front+3][0]
        count = 0
        for index, day in enumerate(dates,1):
            for leg, month in (("near",near),("far",far)):
                values = marks.get((day,month))
                count += values is not None
                legs.append(dict(release_date=issue,session_index=index,session_date=day,leg=leg,contract_month=month,
                                 settlement_at=settlement(day).isoformat(),archive_close_ticks=values[0] if values else None,
                                 source_file=values[1] if values else None,source_sha256=values[2] if values else None,
                                 captured_at=values[3] if values else None))
        windows.append(dict(release_date=issue,decision_at=decision.isoformat(),entry_session=dates[0],exit_session=dates[-1],
                            near_month=near,far_month=far,observed_leg_sessions=count,expected_leg_sessions=10,
                            complete_window=count==10))
    export(f"{STEM}_sources",files)
    export(f"{STEM}_windows",windows)
    export(f"{STEM}_bars",legs)
    summary = dict(archived_contract_files=len(files),completed_windows=sum(r["complete_window"] for r in windows),windows=windows,
                   status="historical_data_collection_for_post_holdout_releases")
    payload = json.dumps(summary,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
