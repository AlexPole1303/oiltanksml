"""Recover pre-release Cushing survey observations from Sina Finance bulletins and Hicend morning reports."""

import argparse
import hashlib
import html
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from collect_cushing_survey_confirmation import SHANGHAI, calendar_rows
from collect_public import RAW, TLS, write
from cushing_morning_reports import morning_reports
from fit_cushing_survey_residuals import MODEL_DIR, sources

CACHE = RAW / "cushing_survey_experts/sina"
FEED = "https://zhibo.sina.com.cn/api/zhibo/feed"
OUTPUT = MODEL_DIR / "wire_observations.csv"
AMOUNT = r"([+\-−]?\d+(?:\.\d+)?)\s*(万桶|百万桶|千桶|桶)"


def get_value(text, location):
    if location == "cushing":
        label = r"(?:API\s*)?(?:俄克拉荷马州)?库欣(?:地区)?(?:原油|石油)?库存(?:变动|变化)?"
    else:
        label = r"API\s*(?:商业)?原油库存(?:变动|变化)?"
    hit = re.search(
        label + r"[\s：:为录得达]*(增加|减少|下降|上升|下降了|增加了|减少了|增长|下滑)?[\s至为达了]*" + AMOUNT,
        text,
        re.IGNORECASE,
    )
    if not hit:
        return None
    direction, number, unit = hit.groups()
    value = float(number.replace("−", "-")) * {"万桶": 10, "百万桶": 1000, "千桶": 1, "桶": 0.001}[unit]
    if direction in {"减少", "下降", "下降了", "减少了", "下滑"}:
        value = -abs(value)
    return round(value, 4)


def bulletin(item):
    raw = item.get("rich_text", "")
    text = html.unescape(re.sub(r"<[^>]+>", " ", raw))
    if "API" not in text.upper() or "库欣" not in text or re.search(r"EIA|能源信息署", text, re.IGNORECASE):
        return None
    value = get_value(text, "cushing")
    if value is None:
        return None
    pub = pd.Timestamp(item["create_time"], tz=SHANGHAI).tz_convert("UTC")
    if not item.get("update_time"):
        return None
    mod = pd.Timestamp(item["update_time"], tz=SHANGHAI).tz_convert("UTC")
    if mod < pub:
        return None
    return {
        "id": item["id"],
        "published_utc": pub.isoformat(),
        "available_utc": mod.isoformat(),
        "api_change_kbbl": value,
        "national_api_kbbl": get_value(text, "national"),
        "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "docurl": item.get("docurl") or "",
    }


LOCKS = [Lock() for _ in range(31)]


def page(before=None):
    with LOCKS[(before or 0) % len(LOCKS)]:
        return load_page(before)


def load_page(before=None):
    cache = CACHE / f"{before or 'latest'}.json"
    if before and cache.exists():
        return json.loads(cache.read_bytes())
    params = {"zhibo_id": 152, "page_size": 60, "type": 1 if before else 0}
    if before:
        params["id"] = before
    url = FEED + "?" + urlencode(params)
    for attempt in range(3):
        try:
            with urlopen(url, context=TLS, timeout=12) as res:
                payload = json.load(res)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    items = payload["result"]["data"]["feed"]["list"]
    if not items:
        data = {"first_id": 0, "last_id": 0, "first": 0, "last": 0, "reports": []}
    else:

        def sec(i):
            return int(pd.Timestamp(i["create_time"], tz=SHANGHAI).timestamp())

        data = {
            "first_id": int(items[-1]["id"]),
            "last_id": int(items[0]["id"]),
            "first": min(map(sec, items)),
            "last": max(map(sec, items)),
            "reports": [v for i in items if (v := bulletin(i))],
        }
    write(cache, json.dumps(data).encode())
    time.sleep(0.22)
    return data


def collect(through, cached_only=False):
    old = sources().set_index("week_ending")
    panel = pd.read_csv(
        MODEL_DIR / "weekly_inputs.csv", usecols=["release_date", "week_ending", "cutoff"]
    ).set_index("week_ending")
    events = [
        e
        for e in calendar_rows(through, cached_only=cached_only)
        if e.get("pubValue")
        and e["dataTime"] in panel.index
        and (
            e["dataTime"] not in old.index
            or abs(old.at[e["dataTime"], "api_change_kbbl"] - float(e["pubValue"]) * 10) > 50
        )
    ]
    pages = {int(f.stem): json.loads(f.read_bytes()) for f in CACHE.glob("*.json") if f.stem.isdigit()}
    if cached_only:
        reports = {r["id"]: r for part in pages.values() for r in part["reports"]}
        rows = []
        for e in events:
            info = panel.loc[e["dataTime"]]
            start = event_time(e)
            found = [
                r
                for r in reports.values()
                if start <= pd.Timestamp(r["published_utc"]) <= start + pd.Timedelta(hours=2)
                and pd.Timestamp(r["available_utc"]) < pd.Timestamp(info.cutoff)
            ]
            if match := corroborate(e, info, found, old):
                rows.append(match)
    else:
        latest = page()
        pages[latest["last_id"] + 1] = latest
        years = sorted({e["pubDate"][:4] for e in events})
        groups = [[e for e in events if e["pubDate"].startswith(year)] for year in years]
        with ThreadPoolExecutor(max_workers=3) as workers:
            parts = workers.map(lambda group: scan(group, pages, latest, panel, old), groups)
            rows = [r for part in parts for r in part]
    for row in morning_reports(events, panel, cached_only=cached_only):
        row["action"] = "replace" if row["week_ending"] in old.index else "add"
        rows.append(row)
    merged = []
    for _, group in pd.DataFrame(rows).groupby("release_date") if rows else []:
        if group.api_change_kbbl.nunique() == 1:
            merged.append(group.sort_values("available_utc").iloc[0].to_dict())
    rows = merged
    fields = [
        "release_date",
        "week_ending",
        "published_utc",
        "available_utc",
        "api_change_kbbl",
        "national_api_kbbl",
        "sha256",
        "source_url",
        "confirmation_url",
        "action",
    ]
    write(OUTPUT, pd.DataFrame(rows, columns=fields).to_csv(index=False).encode())
    print(f"Verified {len(rows)} additional or corrected reports, of {len(events)} source gaps.", flush=True)


def event_time(e):
    return pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz=SHANGHAI).tz_convert("UTC")


def corroborate(e, info, matches, old):
    values = {r["api_change_kbbl"] for r in matches}
    if len(values) != 1 or abs(next(iter(values)) - float(e["pubValue"]) * 10) >= 0.01:
        return None
    result = min(matches, key=lambda r: r["available_utc"]).copy()
    post_id = result.pop("id")
    doc = result.pop("docurl")
    if not doc.startswith("https://"):
        # The feed cursor is exclusive. This link returns the exact public record.
        doc = FEED + "?" + urlencode({"zhibo_id": 152, "page_size": 1, "type": 1, "id": int(post_id) + 1})
    return dict(
        release_date=info.release_date,
        week_ending=e["dataTime"],
        **result,
        source_url=doc,
        confirmation_url=f"https://www.fx168news.com/calendar/{e['calendarId']}",
        action="replace" if e["dataTime"] in old.index else "add",
    )


def scan(events, initial, latest, panel, old):
    pages = dict(initial)

    def get(k):
        if k not in pages:
            pages[k] = page(k)
        return pages[k]

    rows = []
    for e in sorted(events, key=lambda e: e["pubDate"]):
        start = event_time(e)
        end = int((start + pd.Timedelta(hours=2)).timestamp())
        if end > latest["last"]:
            continue
        high = min(k for k, p in pages.items() if p["last"] >= end)
        low = max((k for k, p in pages.items() if k < high and p["last"] < end), default=1)
        while high - low > 1 and pages[high]["first"] > end:
            ltime = pages.get(low, {}).get("last", 0)
            htime = pages[high]["last"]
            fraction = (end - ltime) / (htime - ltime) if ltime and htime > ltime else 0.5
            mid = min(high - 1, max(low + 1, round(low + max(0.04, min(0.96, fraction)) * (high - low))))
            if get(mid)["last"] < end:
                low = mid
            else:
                high = mid
        data = pages[high]
        matches = []
        info = panel.loc[e["dataTime"]]
        while data["last"] >= start.timestamp():
            for r in data["reports"]:
                if start <= pd.Timestamp(r["published_utc"]) <= pd.Timestamp(
                    end, unit="s", tz="UTC"
                ) and pd.Timestamp(r["available_utc"]) < pd.Timestamp(info.cutoff):
                    matches.append(r)
            if data["first"] <= start.timestamp() or data["first_id"] <= 1:
                break
            prior = get(data["first_id"])
            if prior["last_id"] >= data["first_id"]:
                raise ValueError("Nondecreasing Sina pagination")
            data = prior
        if match := corroborate(e, info, matches, old):
            rows.append(match)
    print(f"Finished {len(events)} windows, {len(rows)} matched", flush=True)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", required=True)
    parser.add_argument(
        "--cached-only", action="store_true", help="Use previously downloaded bulletins and morning editions"
    )
    args = parser.parse_args()
    collect(args.through, args.cached_only)
