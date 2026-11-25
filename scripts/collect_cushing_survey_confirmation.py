"""Recover survey releases by matching dated reporting against the FX168 Cushing calendar."""

import argparse
import json
import re
import time
from datetime import datetime
from datetime import time as day_time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
from collect_cushing_weekly_survey import CACHE, article, candidates, extract_changes
from collect_public import ROOT, TLS, write
from cushing_survey_dispatches import dispatches, match_dispatch
from publication_clocks import EASTERN

FOLDER = ROOT / "models/cushing_inventory"
EXTRA = FOLDER / "confirmed_observations.csv"
ARCHIVE = CACHE / "fx168"
ENDPOINT = "https://centerapi.fx168api.com/cms/api/economic/calendar/getCalendarDetailList"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def calendar_rows(through, calendar_id=215654, cached_only=False):
    directory = ARCHIVE if calendar_id == 215654 else ARCHIVE / str(calendar_id)
    rows = []
    page = 1
    while True:
        cache = directory / f"{through}-{page}.json"
        if not cache.exists():
            if cached_only:
                raise FileNotFoundError(cache)
            query = urlencode(
                {
                    "calendarId": calendar_id,
                    "siteCode": "001",
                    "pageNo": page,
                    "pageSize": 20,
                    "dataTimeStart": "2018-01-01",
                    "dataTimeEnd": through,
                }
            )
            with urlopen(Request(ENDPOINT + "?" + query), context=TLS, timeout=25) as response:
                body = response.read()
            if json.loads(body).get("code") != 0:
                raise ValueError("Calendar returned an error")
            write(cache, body)
        result = json.loads(cache.read_bytes())["data"]
        rows.extend(result["items"])
        if page >= result["totalPage"]:
            return rows
        page += 1
        if not (directory / f"{through}-{page}.json").exists():
            time.sleep(10)


def current_values(mentions):
    values = set(extract_changes(mentions))
    # Additional explicit current-week clauses. The independent, week-specific exact match
    # is still required. A missing magnitude unit is accepted only as a candidate.
    for mention in mentions:
        for clause in re.findall(r"\bCushing\b(?:(?!\.(?:\s|$)).){0,260}", mention, re.IGNORECASE):
            clause = re.sub(r"—[^—]*—", " ", clause)
            clause = re.split(r"\bafter\b|\bcompared\b|\bprior\b|\bprevious\b", clause, flags=re.IGNORECASE)[0]
            for m in re.finditer(
                r"\b(gained|fell|rose|lost|declined|increased|dropped|draw at cushing to be)"
                r"\s+(?:by\s+)?([\d,]+(?:\.\d+)?)\s+(million|thousand)\b",
                clause,
                re.IGNORECASE,
            ):
                value = float(m[2].replace(",", "")) * (1000 if m[3].lower() == "million" else 1)
                if m[1].lower() in {"fell", "lost", "declined", "dropped", "draw at cushing to be"}:
                    value = -value
                values.add(round(value, 4))
    return values


def corroborate(record, event, week, deadline):
    if (
        event.get("indicatorsId") != 9831
        or event.get("dataTime") != week
        or not event.get("pubValue")
        or not record.get("modified")
        or "american petroleum institute" not in record.get("head", "").lower()
    ):
        return None
    stamp = pd.Timestamp(
        datetime.fromisoformat(event["pubDate"] + "T" + event["pubTime"]).replace(tzinfo=SHANGHAI)
    )
    published, modified = (pd.Timestamp(record[k]) for k in ("published", "modified"))
    if published.tzinfo is None or modified.tzinfo is None:
        return None
    day = published.tz_convert(EASTERN)
    if (
        not stamp <= published <= modified < deadline
        or published - stamp > pd.Timedelta(hours=7)
        or day.time() < day_time(16, 30)
        or day.date() != modified.tz_convert(EASTERN).date()
        or not 1 <= (day.date() - datetime.fromisoformat(week).date()).days <= 7
    ):
        return None
    value = round(float(event["pubValue"]) * 10, 4)
    if current_values(record.get("mentions", [])) != {value}:
        return None
    return {
        "week_ending": week,
        "api_change_kbbl": value,
        "published_utc": published.tz_convert("UTC").isoformat(),
        "available_utc": modified.tz_convert("UTC").isoformat(),
        "source_url": record["url"],
        "sha256": record["sha256"],
        "confirmation_url": f"https://www.fx168news.com/calendar/{event['calendarId']}",
        "calendar_time_utc": stamp.tz_convert("UTC").isoformat(),
    }


def main(through, expand=False):
    events = calendar_rows(through)
    panel = pd.read_csv(FOLDER / "weekly_inputs.csv", usecols=["release_date", "week_ending", "cutoff"])
    old = pd.read_csv(FOLDER / "api_observations.csv").set_index("release_date")
    panel = panel[panel.release_date <= through]
    missing_weeks = set(panel.loc[~panel.release_date.isin(old.index), "week_ending"])
    # Limit downloads to publication windows for this API series, never condition on an EIA outcome.
    weeks = set(panel.week_ending)
    eligible = [e for e in events if e["dataTime"] in weeks and e.get("pubValue")]
    if expand:
        days = {
            pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz=SHANGHAI).tz_convert(EASTERN).date()
            for e in eligible
            if e["dataTime"] in missing_weeks
        }
        for url in candidates(days):
            try:
                article(url)
            except (OSError, ValueError) as exc:
                print(f"Unavailable {url}: {exc}", flush=True)
    news = {}
    for file in sorted(CACHE.glob("*.json")):
        record = json.loads(file.read_bytes())
        if isinstance(record, dict) and record.get("published") and record.get("mentions"):
            news.setdefault(pd.Timestamp(record["published"]).tz_convert(EASTERN).date(), []).append(record)
    wires = {}
    original_by_week = old.set_index("week_ending")["api_change_kbbl"].to_dict()
    unresolved = [
        e
        for e in eligible
        if e["dataTime"] not in original_by_week
        or abs(float(e["pubValue"]) * 10 - original_by_week[e["dataTime"]]) > 50
    ]
    for item in dispatches(unresolved):
        day = pd.Timestamp(item["published_utc"]).tz_convert(EASTERN).date()
        wires.setdefault(day, []).append(item)
    accepted = []
    for r in panel.itertuples():
        matches = []
        conflicting_dispatches = False
        for e in (ev for ev in eligible if ev["dataTime"] == r.week_ending):
            day = pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz=SHANGHAI).tz_convert(EASTERN).date()
            event_clock = pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz=SHANGHAI)
            issue_wires = [
                m
                for m in wires.get(day, [])
                if event_clock <= pd.Timestamp(m["published_utc"]) < pd.Timestamp(r.cutoff)
            ]
            if (
                issue_wires
                and max(m["api_change_kbbl"] for m in issue_wires)
                - min(m["api_change_kbbl"] for m in issue_wires)
                > 50
            ):
                conflicting_dispatches = True
            for item in issue_wires:
                result = match_dispatch(item, e, r.week_ending, pd.Timestamp(r.cutoff))
                if result is not None:
                    matches.append(result)
            if r.release_date in old.index:
                continue
            for item in news.get(day, []):
                result = corroborate(item, e, r.week_ending, pd.Timestamp(r.cutoff))
                if result is not None:
                    matches.append(result)
        if not matches or conflicting_dispatches:
            continue
        if max(m["api_change_kbbl"] for m in matches) - min(m["api_change_kbbl"] for m in matches) > 50:
            continue
        first = min(matches, key=lambda m: pd.Timestamp(m["available_utc"]))
        action = "add"
        if r.release_date in old.index:
            if abs(old.loc[r.release_date, "api_change_kbbl"] - first["api_change_kbbl"]) <= 50:
                continue
            action = "replace"
        accepted.append(dict(release_date=r.release_date, action=action, **first))
    if not accepted:
        raise RuntimeError("No corroborated observations")
    frame = pd.DataFrame(accepted)
    write(EXTRA, frame.to_csv(index=False).encode())
    print(
        f"{len(eligible)} calendar records checked, {len(frame)} dated additions or replacements.", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True)
    parser.add_argument(
        "--expand-articles",
        action="store_true",
        help="Also fetch the publisher's missing holiday-date articles",
    )
    args = parser.parse_args()
    main(args.through, args.expand_articles)
