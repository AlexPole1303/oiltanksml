"""Verify more precise Cushing quotes in dated, unedited copies of Jin10 releases."""

import argparse
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
from collect_cushing_survey_confirmation import calendar_rows
from collect_public import RAW, TLS, write
from fit_cushing_measurement_filters import MODEL_DIR, v4, wire_records
from lxml import html

CACHE = RAW / "cushing_survey_precision" / "pages_v2"
LEDGER = MODEL_DIR / "precision_observations.csv"
CHANNELS = ("Jin10Data", "FinanceNewsDaily", "Jin10News", "cailianshe", "jin10light", "feixun")
PATTERN = re.compile(
    r"美国(?:至|截至)(\d{1,2})月(\d{1,2})日当周API库欣(?:地区)?原油库存(?:变动|变化)?"
    r"\s*([+-]?\d+(?:\.\d+)?)\s*万桶"
)
# This issue is missing from the FX168 local series. The original dated research note
# reports API +952 kbbl for this Friday. It corroborates the magnitude only.
EXTRA_CONFIRMATION = {"2025-05-30": (952.0, "https://www.dfhjqh.com/news/1169.html")}
FIELDS = [
    "release_date",
    "week_ending",
    "published_utc",
    "available_utc",
    "api_change_kbbl",
    "source_url",
    "confirmation_url",
    "sha256",
    "action",
]


def parse(text, when):
    text = re.sub(r"[\s#]+", "", text).replace("−", "-")
    if len(text) > 1300:
        return None
    hits = list(PATTERN.finditer(text))
    if len(hits) == 1:
        m = hits[0]
        if re.search(r"(?:前值|预计|预测|预期为)", text[: m.end()]):
            return None
        month, day, value = m[1], m[2], float(m[3]) * 10
    else:
        if not re.search(r"美国石油协会|美国API|API[）)]?(?:报告|数据)", text) or re.search(
            r"EIA|能源信息署", text, re.IGNORECASE
        ):
            return None
        obs = re.findall(
            r"(?:上周API|上周|当周API)库欣(?:地区)?原油库存(?:变动|变化)?[：:]?(增加|减少|下降|上升)?([+-]?\d+(?:\.\d+)?)万桶",
            text,
        )
        if len(obs) != 1:
            return None
        direction, n = obs[0]
        value = float(n) * 10
        if direction in ("减少", "下降"):
            value = -abs(value)
        stated = re.findall(r"(\d{1,2})月(\d{1,2})日当周", text)
        if len(set(stated)) > 1:
            return None
        month, day = stated[0] if stated else (None, None)
    local = pd.Timestamp(when).tz_convert("Asia/Shanghai").tz_localize(None).normalize()
    if month is None:
        week = local - pd.Timedelta(days=(local.day_of_week - 4) % 7)
        if 1 <= (local - week).days <= 7:
            return week.date().isoformat(), round(value, 4)
    else:
        for year in (local.year, local.year - 1):
            try:
                week = pd.Timestamp(year=year, month=int(month), day=int(day))
            except ValueError:
                continue
            if week.day_of_week == 4 and 1 <= (local - week).days <= 8:
                return week.date().isoformat(), round(value, 4)
    return None


def read_page(channel, cursor):
    path = CACHE / channel / f"{cursor}.json"
    if cursor and path.is_file():
        return json.loads(path.read_bytes())
    url = "https://t.me/s/" + channel + ("?" + urlencode({"before": cursor}) if cursor else "")
    for attempt in range(3):
        try:
            with urlopen(url, context=TLS, timeout=12) as res:
                doc = html.fromstring(res.read())
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    posts = []
    quotes = []
    for el in doc.xpath("//*[@data-post]"):
        stamp = el.xpath(".//time/@datetime")
        if not stamp:
            continue
        post = el.get("data-post")
        posts.append((int(post.split("/")[-1]), int(pd.Timestamp(stamp[0]).timestamp())))
        body = el.xpath('.//div[contains(@class,"tgme_widget_message_text")]')
        footer = el.xpath('.//*[contains(@class,"tgme_widget_message_info")]')
        if not body or not footer or "edited" in html.tostring(footer[0], encoding="unicode").lower():
            continue
        text = body[0].text_content()
        quote = parse(text, stamp[0])
        if quote is not None:
            quotes.append(
                {
                    "week_ending": quote[0],
                    "api_change_kbbl": quote[1],
                    "published_utc": stamp[0],
                    "available_utc": stamp[0],
                    "source_url": "https://t.me/" + post,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            )
    result = (
        {
            "first": posts[0][1],
            "last": posts[-1][1],
            "first_id": posts[0][0],
            "last_id": posts[-1][0],
            "quotes": quotes,
        }
        if posts
        else {"first": 0, "last": 0, "first_id": 0, "last_id": 0, "quotes": []}
    )
    if not posts and not doc.xpath('//*[contains(@class,"tgme_channel_info")]'):
        raise ValueError("Unexpected Telegram response")
    write(path, json.dumps(result).encode())
    time.sleep(0.12)
    return result


def scan(channel, events, span=2700):
    latest = read_page(channel, 0)
    if not latest["last"]:
        return
    pages = {int(f.stem): json.loads(f.read_bytes()) for f in (CACHE / channel).glob("*.json") if f.stem != "0"}
    pages[latest["last_id"] + 1] = latest

    def page(k):
        if k not in pages:
            pages[k] = read_page(channel, k)
        return pages[k]

    for e in events:
        start = e["stamp"]
        end = min(start + span, e["cutoff"] - 1)
        if end > latest["last"]:
            continue
        hi = min(k for k, p in pages.items() if p["last"] >= end)
        lo = max((k for k, p in pages.items() if k < hi and p["last"] < end), default=0)
        while hi - lo > 1 and pages[hi]["first"] > end:
            lowtime = pages.get(lo, {}).get("last", 0)
            hightime = pages[hi]["last"]
            frac = (end - lowtime) / (hightime - lowtime) if lowtime and hightime > lowtime else 0.5
            k = max(lo + 1, min(hi - 1, round(lo + (hi - lo) * max(0.002, min(0.998, frac)))))
            if page(k)["last"] >= end:
                hi = k
            else:
                lo = k
        current = pages[hi]
        while current["last"] >= start and current["first"] > start and current["first_id"] > 1:
            prev = page(current["first_id"])
            if prev["last_id"] >= current["first_id"]:
                raise ValueError("Non-monotone archive")
            current = prev
    print(channel, "checked", len(events), "release windows", flush=True)


def dispatch_rounding(text):
    normalized = re.sub(r"[\s#]+", "", text)
    # Whole tenths of a million across crude, Cushing, gasoline and distillate
    # identify a rounded bulletin. Report only what the dispatch actually printed.
    patterns = (r"美国API原油库存", r"上周API库欣原油库存", r"(?:API成品油)?汽油库存", r"馏分油库存")
    values = []
    for pat in patterns:
        hits = re.findall(pat + r"([+-]?\d+(?:\.\d+)?)万桶", normalized)
        if not hits:
            return 0.0
        values.append(float(hits[0]) * 10)
    return 50.0 if all(abs(v / 100 - round(v / 100)) < 1e-9 for v in values) else 0.0


def rounded_record(row, cached_only):
    path = CACHE.parent / "precision" / (hashlib.sha256(row["source_url"].encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_bytes())["radius"]
    if cached_only:
        return 0.0
    with urlopen(row["source_url"] + "?embed=1", context=TLS, timeout=12) as response:
        el = html.fromstring(response.read())
    text = el.xpath('.//div[contains(@class,"tgme_widget_message_text")]')
    stamp = el.xpath(".//time/@datetime")
    radius = 0.0
    if (
        len(text) == 1
        and len(stamp) == 1
        and pd.Timestamp(stamp[0]) == pd.Timestamp(row["published_utc"])
        and hashlib.sha256(text[0].text_content().encode()).hexdigest() == row["sha256"]
    ):
        radius = dispatch_rounding(text[0].text_content())
    write(path, json.dumps({"radius": radius}).encode())
    return radius


def collect(through, cached_only):
    orig = wire_records(v4.sources()).set_index("week_ending")
    panel = pd.read_csv(MODEL_DIR / "weekly_inputs.csv", usecols=["release_date", "week_ending", "cutoff"])
    events = {
        e["dataTime"]: e
        for e in calendar_rows(through, cached_only=cached_only)
        if e.get("indicatorsId") == 9831 and e.get("pubValue")
    }
    country = {
        e["dataTime"]: e for e in calendar_rows(through, 215651, cached_only=cached_only) if e.get("pubValue")
    }
    windows = []
    for r in panel.itertuples():
        if r.release_date > through:
            continue
        wk = r.week_ending
        e = events.get(wk)
        if e is not None:
            value = round(10 * float(e["pubValue"]), 4)
            link = f"https://www.fx168news.com/calendar/{e['calendarId']}"
        elif wk in EXTRA_CONFIRMATION and wk in country:
            e = country[wk]
            value, link = EXTRA_CONFIRMATION[wk]
        else:
            continue
        if wk in orig.index and abs(orig.at[wk, "api_change_kbbl"] - value) < 0.01:
            continue
        stamp = pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz="Asia/Shanghai")
        win = {
            "week_ending": wk,
            "release_date": r.release_date,
            "value": value,
            "link": link,
            "stamp": int(stamp.timestamp()),
            "cutoff": int(pd.Timestamp(r.cutoff).timestamp()),
        }
        if 0 < win["cutoff"] - win["stamp"] < 4 * 86400:
            windows.append(win)
    if not cached_only:
        # Some mirrors started later. Search newest weeks first to establish timestamp bounds quickly.
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(
                pool.map(lambda ch: scan(ch, sorted(windows, key=lambda x: x["stamp"], reverse=True)), CHANNELS)
            )
        observed = {
            q["week_ending"]
            for channel in CHANNELS
            for f in (CACHE / channel).glob("*.json")
            for q in json.loads(f.read_bytes())["quotes"]
        }
        remaining = [
            w
            for w in reversed(windows)
            if w["week_ending"] not in orig.index and w["week_ending"] not in observed
        ]
        if remaining:
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(
                    pool.map(
                        lambda ch: scan(ch, remaining, 4 * 3600),
                        ("FinanceNewsDaily", "Jin10News", "cailianshe"),
                    )
                )
    quotes = {}
    for f in sorted(f for channel in CHANNELS for f in (CACHE / channel).glob("*.json")):
        for q in json.loads(f.read_bytes())["quotes"]:
            quotes.setdefault(q["week_ending"], {})[q["source_url"]] = q
    rows = []
    for e in windows:
        wk = e["week_ending"]
        q = [
            v
            for v in quotes.get(wk, {}).values()
            if e["stamp"] <= pd.Timestamp(v["available_utc"]).timestamp() < e["cutoff"]
        ]
        if not q:
            continue
        matches = []
        for item in q:
            error = abs(item["api_change_kbbl"] - e["value"])
            radius = 0.0 if error < 0.001 else rounded_record(item, cached_only)
            if error > radius + 0.001:
                break
            matches.append((radius, item["available_utc"], item["source_url"], item))
        else:
            if not matches:
                continue
            radius, _, _, row = min(matches)
            if wk in orig.index and (
                abs(orig.at[wk, "api_change_kbbl"] - e["value"]) <= radius
                or abs(orig.at[wk, "api_change_kbbl"] - row["api_change_kbbl"]) < 0.001
            ):
                continue
            rows.append(
                dict(
                    row,
                    release_date=e["release_date"],
                    confirmation_url=e["link"],
                    action="add" if wk not in orig.index else "replace",
                )
            )

    write(LEDGER, pd.DataFrame(rows, columns=FIELDS).to_csv(index=False).encode())
    print(
        "Verified observations", len(rows), "of", len(windows), "missing or differing local quotes", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", default="2026-11-18")
    parser.add_argument("--cached-only", action="store_true")
    args = parser.parse_args()
    collect(args.through, args.cached_only)
