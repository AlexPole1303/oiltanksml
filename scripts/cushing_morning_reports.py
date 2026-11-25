"""Verify report-week API changes against Hicend Futures' dated public morning editions."""

import hashlib
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode, urljoin
from urllib.request import urlopen

import pandas as pd
from collect_public import RAW, TLS, write
from lxml import html
from pypdf import PdfReader

ORIGIN = "https://www.hicend.com"
CACHE = RAW / "cushing_survey_experts/hicend"


def issue_links(earliest, latest):
    found = {}
    for number in range(1, 100):
        cache = CACHE / f"list_{number}.html"
        if not cache.exists():
            params = urlencode({"page": number, "topicCode": "Hedgearbitrage", "row": 80, "stick": 0})
            with urlopen(ORIGIN + "/Pc/articleList?" + params, context=TLS, timeout=20) as response:
                write(cache, response.read())
        content = cache.read_bytes()
        if not content.strip():
            break
        tree = html.fromstring(content.decode("utf8"))
        dates = []
        for link in tree.xpath("//a[@href]"):
            title = "".join(link.text_content().split())
            match = re.search(r"(20\d{6})(?:海证(?:期货)?)?研究所[早晨]评", title)
            href = link.get("href")
            if not match or not href.startswith("/ftpfile/") or not href.lower().endswith(".pdf"):
                continue
            date = pd.to_datetime(match[1], format="%Y%m%d").date().isoformat()
            dates.append(date)
            if earliest <= date <= latest:
                found.setdefault(date, []).append(urljoin(ORIGIN, href))
        if not dates or min(dates) < earliest:
            break
    return found


def parse_changes(text, week):
    compact = "".join(text.split()).replace("－", "-").replace("−", "-")
    want = pd.Timestamp(week)
    values = set()
    for m in re.finditer(
        r"库欣(?:地区)?(?:原油|石油)?库存(?:变化|变动)?[：:为]*(增加|减少|下降|增长|上升)?([+\-]?\d+(?:\.\d+)?)(万桶|百万桶|千桶|桶)",
        compact,
    ):
        before = compact[max(0, m.start() - 200) : m.start()]
        api = max(before.upper().rfind("API"), before.rfind("美国石油协会"))
        other = max(before.upper().rfind("EIA"), before.rfind("能源信息署"))
        if api < 0 or other > api:
            continue
        section = before[api:]
        stated = re.findall(r"(\d{1,2})月(\d{1,2})日(?:的)?当周", before)
        if stated and (int(stated[-1][0]), int(stated[-1][1])) != (want.month, want.day):
            continue
        if re.search(r"(?:此前|前值|之前一周|前一周|上上周)[，,：:\s]*$", section):
            continue
        value = float(m[2]) * {"万桶": 10, "百万桶": 1000, "千桶": 1, "桶": 0.001}[m[3]]
        if m[1] in {"减少", "下降"}:
            value = -abs(value)
        values.add(round(value, 4))
    return values


def extract_issue(url, e, panel_row):
    cache = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if cache.exists():
        stored = json.loads(cache.read_bytes())
        return stored.get("row")
    stamp = pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz="Asia/Shanghai").tz_convert("UTC")
    cutoff = pd.Timestamp(panel_row.cutoff)
    with urlopen(url, context=TLS, timeout=20) as response:
        origin_time = response.headers.get("Last-Modified")
        if not origin_time:
            return None
        upload = pd.Timestamp(parsedate_to_datetime(origin_time)).tz_convert("UTC")
        if not stamp <= upload < cutoff:
            write(cache, b'{"status":"outside_file_window"}')
            return None
        raw = response.read()
    doc = PdfReader(io.BytesIO(raw))
    dates = [doc.metadata.creation_date, doc.metadata.modification_date]
    if any(d is None or d.tzinfo is None for d in dates):
        return None
    clocks = [pd.Timestamp(d).tz_convert("UTC") for d in dates]
    if min(clocks) < stamp or max(clocks + [upload]) >= cutoff:
        return None
    possible = set()
    for page in doc.pages:
        text = page.extract_text() or ""
        if "库欣" in "".join(text.split()):
            possible.update(parse_changes(text, e["dataTime"]))
    expected = round(float(e["pubValue"]) * 10, 4)
    row = None
    if possible == {expected}:
        row = {
            "release_date": panel_row.release_date,
            "week_ending": e["dataTime"],
            "api_change_kbbl": expected,
            "published_utc": upload.isoformat(),
            "available_utc": max(clocks + [upload]).isoformat(),
            "national_api_kbbl": None,
            "source_url": url,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "confirmation_url": f"https://www.fx168news.com/calendar/{e['calendarId']}",
        }
    write(cache, json.dumps({"row": row}).encode())
    return row


def morning_reports(events, panel, cached_only=False):
    if cached_only:
        wanted = {e["dataTime"]: e for e in events}
        rows = []
        for path in sorted(CACHE.glob("*.json")):
            row = json.loads(path.read_bytes()).get("row")
            if row and row["week_ending"] in wanted:
                event = wanted[row["week_ending"]]
                info = panel.loc[row["week_ending"]]
                if (
                    row["release_date"] == info.release_date
                    and abs(row["api_change_kbbl"] - float(event["pubValue"]) * 10) < 0.01
                    and pd.Timestamp(row["available_utc"]) < pd.Timestamp(info.cutoff)
                ):
                    rows.append(row)
        return rows
    if not events:
        return []
    windows = [
        (
            e,
            panel.loc[e["dataTime"]],
            pd.Timestamp(panel.loc[e["dataTime"]].cutoff).tz_convert("Asia/Shanghai").date(),
        )
        for e in events
    ]
    links = issue_links(min(e["pubDate"] for e in events), max(end for _, _, end in windows).isoformat())
    jobs = [
        (url, e, info)
        for e, info, end in windows
        for day in pd.date_range(e["pubDate"], end)
        for url in sorted(set(links.get(day.date().isoformat(), [])))
    ]

    def get(args):
        for attempt in range(3):
            try:
                return extract_issue(*args)
            except (OSError, ValueError) as ex:
                if attempt == 2:
                    print("Unavailable morning edition", args[0], type(ex).__name__, flush=True)
        return None

    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [row for row in pool.map(get, jobs) if row]
    print(f"{len(jobs)} relevant Hicend editions located, {len(rows)} corroborated before cutoff", flush=True)
    return rows
