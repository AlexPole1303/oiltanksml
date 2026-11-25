"""Read timestamped copies of FinancialJuice dispatches from a public channel for the Cushing API survey."""

import hashlib
import json
import re
import time
from urllib.request import urlopen

import pandas as pd
from collect_cushing_weekly_survey import CACHE
from collect_public import TLS, write
from lxml import html

FIRST = "https://t.me/s/FinancialJuice"
PATTERN = re.compile(r"\bUS API CUSHING STOCK CHANGE ACTUAL\s+([+−-]?\d+(?:\.\d+)?)\s*([MK])\b", re.IGNORECASE)


def extract(text):
    hits = list(PATTERN.finditer(" ".join(text.split())))
    m = hits[0] if len(hits) == 1 else None
    if not m:
        return None
    number, unit = m.groups()
    scale = 1000 if unit.upper() == "M" else 1
    decimals = len(number.partition(".")[2])
    return round(float(number.replace("−", "-")) * scale, 4), 0.5 * scale * 10**-decimals


def read_page(before=None):
    cache = CACHE / "financialjuice_by_date" / f"{before or 'latest'}.json"
    if before is not None and cache.exists():
        return json.loads(cache.read_bytes())
    url = FIRST + (f"?before={before}" if before is not None else "")
    with urlopen(url, context=TLS, timeout=20) as response:
        tree = html.fromstring(response.read().decode("utf8"))
    clocks, messages = [], []
    for node in tree.xpath("//*[@data-post]"):
        post = node.get("data-post", "")
        stamps = node.xpath('.//a[contains(@class,"message_date")]/time/@datetime')
        if not re.fullmatch(r"FinancialJuice/\d+", post) or len(stamps) != 1:
            continue
        clocks.append((int(post.split("/")[1]), stamps[0]))
        body = node.xpath('.//div[contains(@class,"js-message_text")]')
        meta = node.xpath('.//*[contains(@class,"message_meta")]')
        if not body or any("edited" in m.text_content().lower() for m in meta):
            continue
        content = " ".join(body[0].itertext())
        result = extract(content)
        if result is not None:
            value, rounding = result
            messages.append(
                {
                    "source_url": "https://t.me/" + post,
                    "published_utc": stamps[0],
                    "sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "api_change_kbbl": value,
                    "rounding_kbbl": rounding,
                }
            )
    if not clocks:
        raise ValueError("Public channel archive returned no dated posts")
    page = {
        "first_id": clocks[0][0],
        "last_id": clocks[-1][0],
        "first": clocks[0][1],
        "last": clocks[-1][1],
        "messages": messages,
    }
    write(cache, json.dumps(page).encode())
    time.sleep(0.5)
    return page


def dispatches(events):
    # The public channel links previous pages by message ID. Locate each issue in time,
    # even when the release falls in a long gap in channel coverage.
    latest = read_page()
    upper = latest["last_id"] + 1
    cache_dir = CACHE / "financialjuice_by_date"
    pages = {int(f.stem): json.loads(f.read_bytes()) for f in cache_dir.glob("*.json") if f.stem.isdigit()}
    pages[upper] = latest

    def fetch(before):
        if before not in pages:
            pages[before] = read_page(before)
        return pages[before]

    found = {}
    for e in sorted(events, key=lambda x: x["pubDate"]):
        stamp = pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz="Asia/Shanghai").tz_convert("UTC")
        end = (stamp + pd.Timedelta(hours=2)).isoformat()
        if end > latest["last"]:
            continue
        high = min(k for k, p in pages.items() if p["last"] >= end)
        low = max((k for k, p in pages.items() if p["last"] < end and k < high), default=1)
        while high - low > 1 and pages[high]["first"] > end:
            mid = (low + high) // 2
            if fetch(mid)["last"] < end:
                low = mid
            else:
                high = mid
        page = pages[high]
        while page["last"] >= stamp.isoformat():
            for m in page["messages"]:
                if stamp <= pd.Timestamp(m["published_utc"]) <= pd.Timestamp(end):
                    found[m["source_url"]] = m
            if page["first"] <= stamp.isoformat() or page["first_id"] <= 1:
                break
            prior = fetch(page["first_id"])
            if prior["last_id"] >= page["first_id"]:
                raise ValueError("Channel pagination did not move backwards")
            page = prior
    return list(found.values())


def match_dispatch(item, event, week, deadline):
    if event["indicatorsId"] != 9831 or event["dataTime"] != week or not event.get("pubValue"):
        return None
    stamp = pd.Timestamp(event["pubDate"] + "T" + event["pubTime"], tz="Asia/Shanghai")
    clock = pd.Timestamp(item["published_utc"])
    if (
        clock.tzinfo is None
        or not stamp <= clock < deadline
        or clock - stamp > pd.Timedelta(hours=7)
        or not 1 <= (clock.tz_convert("America/New_York").date() - pd.Timestamp(week).date()).days <= 7
    ):
        return None
    value = item["api_change_kbbl"]
    if abs(value - float(event["pubValue"]) * 10) > min(50.0, item["rounding_kbbl"]) + 1e-7:
        return None
    return {
        "week_ending": week,
        "api_change_kbbl": value,
        "published_utc": item["published_utc"],
        "available_utc": item["published_utc"],
        "source_url": item["source_url"],
        "sha256": item["sha256"],
        "confirmation_url": f"https://www.fx168news.com/calendar/{event['calendarId']}",
        "calendar_time_utc": stamp.tz_convert("UTC").isoformat(),
    }
