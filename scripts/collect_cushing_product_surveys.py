"""Corroborate early weekly US gasoline and distillate survey estimates in dated OilPrice reporting."""

import argparse
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from collect_cushing_survey_confirmation import calendar_rows
from collect_cushing_weekly_survey import AMOUNT, DOWN, SIGN, download
from collect_cushing_weekly_survey import CACHE as ARTICLES
from collect_public import RAW, write
from fit_cushing_inventory_network import MODEL_DIR

CACHE = RAW / "cushing_survey_precision/products"
OUTPUT = MODEL_DIR / "product_observations.csv"
SERIES = {"gasoline": (215653, 9423), "distillate": (215652, 9422)}
LABEL = re.compile(r"\b(gasoline|distillates?)\b", re.IGNORECASE)


def extract(body):
    out = {k: set() for k in SERIES}
    for sentence in re.split(r"(?<=[.!?])\s+|\s{2,}", body):
        # Separate adjacent product clauses; references to an earlier release terminate the current clause.
        for clause in re.split(
            r"[,;]?\s*\b(?:while|whereas)\b|;|,\s*and\s+(?=(?:gasoline|distillate))",
            sentence,
            flags=re.IGNORECASE,
        ):
            cut = re.split(
                r",?\s*(?:compared (?:to|with)|after (?:last|a |an |the previous)|versus|vs\.?\s|previously|following (?:last|the previous))",
                clause,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            hits = {m[1].lower().rstrip("s") for m in LABEL.finditer(cut)}
            if len(hits) != 1 or re.search(
                r"\bEIA\b|Energy Information Administration|expected|analysts|forecast|week prior|previous week|five.year|refinery|demand|prices|crude oil inventories",
                cut,
                re.IGNORECASE,
            ):
                continue
            amounts = list(AMOUNT.finditer(cut))
            if len(amounts) != 1:
                continue
            m = amounts[0]
            signs = SIGN.findall(cut[: m.start()])
            suffix = SIGN.match(cut[m.end() :].lstrip(" -"))
            if len({s.lower() for s in signs}) > 1:
                continue
            sign = suffix[0] if suffix else signs[-1] if signs else None
            if not sign:
                continue
            value = (
                float(m[1].replace(",", "")) * {"million": 1000, "thousand": 1, "": 0.001}[(m[2] or "").lower()]
            )
            out[hits.pop()].add(round(-value if re.fullmatch(DOWN, sign, re.IGNORECASE) else value, 4))
    return {k: sorted(s) for k, s in out.items()}


def retrieve(item):
    path = CACHE / (hashlib.sha256(item["url"].encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_bytes())
    try:
        raw = download(item["url"])
        markup = raw.decode("utf8", errors="replace")
        for tag in re.findall(
            r'<script\b[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            markup,
            re.IGNORECASE | re.DOTALL,
        ):
            try:
                ld = json.loads(tag)
            except ValueError:
                continue
            if (
                not isinstance(ld, dict)
                or not ld.get("articleBody")
                or not ld.get("dateModified")
                or not ld.get("datePublished")
            ):
                continue
            data = {
                "url": item["url"],
                "published": ld["datePublished"],
                "modified": ld["dateModified"],
                "sha256": hashlib.sha256(raw).hexdigest(),
                "values": extract(html.unescape(ld["articleBody"])),
            }
            write(path, json.dumps(data).encode())
            return data
    except (OSError, ValueError) as e:
        print("Article unavailable", item["url"], type(e).__name__, flush=True)
    return None


def collect(through, cache_only=False):
    docs = []
    for f in ARTICLES.glob("*.json"):
        a = json.loads(f.read_bytes())
        if (
            isinstance(a, dict)
            and a.get("modified")
            and "american petroleum institute" in a.get("head", "").lower()
        ):
            docs.append(a)
    if not cache_only:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(retrieve, docs))
    only = {a["url"] for a in docs}
    records = {}
    for f in sorted(CACHE.glob("*.json")):
        d = json.loads(f.read_bytes())
        if d["url"] not in only:
            continue
        t = pd.Timestamp(d["published"])
        m = pd.Timestamp(d["modified"])
        if t.tzinfo and m.tzinfo and t <= m:
            records.setdefault(t.tz_convert("America/New_York").date(), []).append(d)
    panel = pd.read_csv(MODEL_DIR / "weekly_inputs.csv", usecols=["release_date", "week_ending", "cutoff"])
    ex = json.loads((MODEL_DIR / "nov18_inputs.json").read_bytes())
    panel.loc[len(panel)] = [ex["release_date"], ex["week_ending"], ex["deadline_utc"]]
    panel = panel.set_index("week_ending")
    rows = []
    for prod, (key, indicator) in SERIES.items():
        for e in calendar_rows(through, key, cached_only=cache_only):
            wk = e["dataTime"]
            if e["indicatorsId"] != indicator or not e.get("pubValue") or wk not in panel.index:
                continue
            r = panel.loc[wk]
            stamp = pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz="Asia/Shanghai")
            target = round(10 * float(e["pubValue"]), 4)
            eligible = [
                d
                for d in records.get(stamp.tz_convert("America/New_York").date(), [])
                if d["values"][prod] == [target]
                and stamp
                <= pd.Timestamp(d["published"])
                <= pd.Timestamp(d["modified"])
                < pd.Timestamp(r.cutoff)
                and pd.Timestamp(d["published"]) - stamp < pd.Timedelta(hours=8)
            ]
            if not eligible:
                continue
            d = min(eligible, key=lambda a: pd.Timestamp(a["modified"]))
            rows.append(
                {
                    "release_date": r.release_date,
                    "week_ending": wk,
                    "product": prod,
                    "value_kbbl": target,
                    "published_utc": pd.Timestamp(d["published"]).tz_convert("UTC").isoformat(),
                    "available_utc": pd.Timestamp(d["modified"]).tz_convert("UTC").isoformat(),
                    "source_url": d["url"],
                    "sha256": d["sha256"],
                    "confirmation_url": f"https://www.fx168news.com/calendar/{e['calendarId']}",
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "release_date",
            "week_ending",
            "product",
            "value_kbbl",
            "published_utc",
            "available_utc",
            "source_url",
            "sha256",
            "confirmation_url",
        ],
    ).sort_values(["release_date", "product"])
    if frame.duplicated(["week_ending", "product"]).any():
        raise ValueError("Duplicate verified product observation")
    write(OUTPUT, frame.to_csv(index=False).encode())
    print("Verified product numbers", frame["product"].value_counts().to_dict(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--through", default="2026-11-18")
    p.add_argument("--cached-only", action="store_true")
    args = p.parse_args()
    collect(args.through, args.cached_only)
