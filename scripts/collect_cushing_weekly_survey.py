"""Index dated public reporting of the API Cushing survey, independently of the EIA outcome."""
import argparse
import csv
import io
import hashlib
import html
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time as daytime, timedelta, timezone
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from collect_public import RAW, ROOT, TLS, write
from publication_clocks import EASTERN

CACHE = RAW / "cushing_survey_reporting"
ORIGIN = "https://oilprice.com"
COLLECTION = ROOT / "models/cushing_inventory/api_observations.csv"


def download(url):
    if urlsplit(url).hostname != "oilprice.com":
        raise ValueError("Unexpected publisher")
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Cushing-inventory-research/1.0"}), context=TLS, timeout=12) as response:
                return response.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def article(url):
    cache = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".json")
    if cache.exists():
        return json.loads(cache.read_bytes())
    raw = download(url)
    text = raw.decode("utf8", errors="replace")
    for tag in re.findall(r'<script\b[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', text, re.I | re.S):
        try:
            ld = json.loads(tag)
        except ValueError:
            continue
        if isinstance(ld, dict) and ld.get("articleBody") and ld.get("datePublished"):
            body = html.unescape(ld["articleBody"])
            result = dict(url=url, published=ld["datePublished"], modified=ld.get("dateModified"),
                          head=body[:750], mentions=[body[max(0,m.start()-130):m.start()+360] for m in re.finditer(r"\bCushing\b",body,re.I)],
                          sha256=hashlib.sha256(raw).hexdigest())
            break
    else:
        result = dict(url=url, mentions=[], status="no_dated_article")
    write(cache, json.dumps(result, ensure_ascii=False).encode())
    return result


def candidates(dates):
    result = []
    for year in sorted({d.year for d in dates}):
        path = CACHE / f"sitemap_news_{year}.xml"
        if not path.exists():
            write(path, download(f"{ORIGIN}/sitemap_news_{year}.xml"))
        for entry in ET.fromstring(path.read_bytes()):
            url = entry.findtext("{*}loc")
            stamp = entry.findtext("{*}lastmod")
            if not stamp or not url:
                continue
            dt = datetime.fromisoformat(stamp)
            if dt.date() not in dates or dt.hour < 13:
                continue
            title = url.rsplit("/", 1)[-1]
            if re.search(r"inventor|crude|stockpile|barrel|API-|oil-stock|oil-build|oil-draw|oil-suppl", title, re.I):
                result.append(url)
    return sorted(set(result))


UP = r"rose|risen|rising|rise|increas(?:e|ed|ing)|gain(?:ed|s|ing)?|add(?:ed|ing)?|build(?:s)?|built|up|climbed|grew|jumped|growth|addition|surged"
DOWN = r"fell|fallen|fall(?:ing)?|declin(?:e|ed|ing)|decreas(?:e|ed|ing)|draw(?:down|s)?|drew|drop(?:ped)?|down|lost|losing|loss(?:es)?|shed|slid|slipped|shrank|dip(?:ped)?|sank|shaved|reduction"
AMOUNT = re.compile(r"(?<![0-9.,])([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)[\s-]*(million|thousand)?[\s-]*barrels?\b", re.I)
SIGN = re.compile(r"\b(" + UP + "|" + DOWN + r")\b", re.I)


def extract_changes(mentions):
    values = {}
    for mention in mentions:
        for hit in re.finditer(r"\bCushing\b", mention, re.I):
            prefix = re.split(r"[.!?]\s+|\s{2,}|,\s*(?:while|whereas)\s+", mention[:hit.start()], flags=re.I)[-1]
            if re.search(r"previous|prior|last week|earlier week|official EIA|analysts|forecast", prefix, re.I):
                continue
            phrase = mention[hit.start():].replace("U.S.", "US").replace("U. S.", "US")
            phrase = re.sub(r"—[^—]{1,150}—", " ", phrase)
            phrase = re.split(r"(?<![0-9])\.(?:\s|$)|(?<=[0-9])\.\s|\bafter\b|\bcompared (?:with|to)\b", phrase, maxsplit=1, flags=re.I)[0]
            if re.match(r"Cushing (?:inventor(?:ies|y) )?(?:saw no change|(?:was|were|remained) unchanged)", phrase, re.I):
                values[0.] = phrase
                continue
            m = AMOUNT.search(phrase)
            if not m or m.start() > 185:
                continue
            before, after = phrase[:m.start()], phrase[m.end():]
            if re.search(r"prior|previous|last week.s|week before|EIA|gasoline|distillate|production|strategic|forecast|expected|to (?:a total of )?$", before, re.I):
                continue
            if re.match(r"^[ ,]*(?:(?:in|during|for|from|of) )?(?:the )?(?:previous|prior|preceding) week|^ (?:in|during|for) the week (?:before|prior)", after, re.I):
                continue
            post = re.match(r"^[ -]*(" + UP + "|" + DOWN + r")\b", after, re.I)
            signs = SIGN.findall(before)
            direction = post.group(1) if post else signs[-1] if signs else None
            if direction is None:
                continue
            value = float(m.group(1).replace(",", ""))
            unit = (m.group(2) or "").lower()
            value *= {"million": 1000, "thousand": 1, "": .001}[unit]
            if re.fullmatch(DOWN, direction, re.I):
                value = -value
            if .5 <= abs(value) < 10000:
                values[round(value, 4)] = phrase[:m.end()+min(55,len(after))]
    return values


def observations(urls, dates):
    accepted, audit = {}, []
    for url in urls:
        cache = CACHE / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if not cache.is_file():
            continue
        row = json.loads(cache.read_bytes())
        if not row.get("mentions") or "american petroleum institute" not in row.get("head","").lower():
            continue
        if not row.get("modified"):
            continue
        published, modified = (datetime.fromisoformat(row[k]).astimezone(EASTERN) for k in ("published", "modified"))
        day = published.date()
        if day not in dates or published.time() < daytime(16, 30) or modified < published or modified.date() != day:
            continue
        issue = day + timedelta(days=1)
        week = day - timedelta(days=(day.weekday()-4) % 7)
        stated = re.search(r"week end(?:ing|ed) ([A-Za-z]+) (\d{1,2})", row["head"].replace("U.S.", "US").split(". ")[0], re.I)
        if stated:
            try:
                parsed = datetime.strptime(stated.group(1)[:3] + " " + stated.group(2) + " " + str(week.year), "%b %d %Y")
            except ValueError:
                continue
            if (parsed.month, parsed.day) != (week.month, week.day):
                continue
        vals = extract_changes(row["mentions"])
        if len(vals) != 1:
            continue
        value, snippet = next(iter(vals.items()))
        obs = dict(release_date=issue.isoformat(), week_ending=week.isoformat(), api_change_kbbl=value,
                   published_utc=published.astimezone(timezone.utc).isoformat(), available_utc=modified.astimezone(timezone.utc).isoformat(),
                   source_url=url, sha256=row["sha256"])
        accepted.setdefault(issue, []).append(obs)
        audit.append(dict(obs, matched=snippet))
    safe = []
    for issue, group in sorted(accepted.items()):
        if len({r["api_change_kbbl"] for r in group}) == 1:
            safe.append(min(group,key=lambda r:r["available_utc"]))
    if not safe:
        raise ValueError("No unambiguous dated survey observations")
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, safe[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(safe)
    write(COLLECTION, out.getvalue().encode())
    write(CACHE / "extraction_audit.json", json.dumps(audit,indent=2).encode())
    print(f"{len(safe)} unambiguous timestamped estimates", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--through", type=datetime.fromisoformat, required=True, help="Last EIA release date to index")
    args = parser.parse_args()
    with (ROOT / "models/cushing_inventory/weekly_inputs.csv").open() as f:
        dates = {datetime.fromisoformat(r["release_date"]).date() - timedelta(days=1) for r in csv.DictReader(f)
                 if "2018-01-01" <= r["release_date"] <= args.through.date().isoformat()}
    if COLLECTION.exists():
        with COLLECTION.open() as f:
            dates.update(datetime.fromisoformat(r["release_date"]).date() - timedelta(days=1) for r in csv.DictReader(f)
                         if r["release_date"] <= args.through.date().isoformat())
    dates.add(args.through.date() - timedelta(days=1))
    urls = candidates(dates)
    print(f"{len(urls)} publisher URLs in the dated pre-release windows", flush=True)
    def retrieve(url):
        try:
            article(url)
        except (OSError, ValueError) as exc:
            print(f"Unavailable {url}: {exc}", flush=True)
    if not args.cached_only:
        with ThreadPoolExecutor(max_workers=3) as workers:
            list(workers.map(retrieve, urls))
    observations(urls, dates)
    print(f"Indexed {len(urls)} candidate URLs", flush=True)


if __name__ == "__main__":
    main()
