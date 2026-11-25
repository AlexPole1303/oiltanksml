"""Prepare cutoff-checked US API releases and a separate retrospective radar benchmark."""

import argparse
import hashlib
import json
import re
from urllib.request import urlopen

import numpy as np
import pandas as pd
from collect_cushing_survey_confirmation import calendar_rows
from collect_cushing_weekly_survey import AMOUNT, CACHE, DOWN, SIGN
from collect_public import RAW, TLS, write
from fit_cushing_inventory_network import MODEL_DIR

INDEX_URL = "https://research-api.pelyr.com/v1/oil-storage/downloads/cushing-index.csv"
OUTPUT = MODEL_DIR / "measurement_inputs.csv"


def national_estimate(head):
    if "american petroleum institute" not in head.lower():
        return None
    clause = re.split(
        r"\bafter\b|\bcompared\b|\banalysts\b|\bprevious\b|\bweek prior\b|\bexpected\b",
        head,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    clause = re.split(
        r"(?<!\d)\.\s|(?<=\d)\.\s", clause.replace("U.S.", "US").replace("U. S.", "US"), maxsplit=1
    )[0]
    if not re.search(r"\bcrude\b", clause, re.IGNORECASE) or re.search(
        r"Cushing|gasoline|distillate", clause, re.IGNORECASE
    ):
        return None
    numbers = list(AMOUNT.finditer(clause))
    if len(numbers) != 1:
        return None
    m = numbers[0]
    signs = SIGN.findall(clause[: m.start()])
    post = SIGN.match(clause[m.end() :].lstrip(" -"))
    sign = post.group() if post else signs[-1] if signs else None
    if sign is None:
        return None
    unit = (m[2] or "").lower()
    result = float(m[1].replace(",", "")) * {"million": 1000, "thousand": 1, "": 0.001}[unit]
    return round(-result if re.fullmatch(DOWN, sign, re.IGNORECASE) else result, 4)


def radar_features(scenes, week, cutoff):
    end_week = pd.Timestamp(week, tz="UTC") + pd.Timedelta(days=1)
    eligible = (
        scenes[(scenes.available_at < cutoff) & (scenes.acq_time < end_week)].sort_values("acq_time").tail(2)
    )
    if len(eligible) != 2:
        return (np.nan, np.nan, "", "")
    first, last = eligible.itertuples(index=False)
    age = (end_week - last.acq_time) / pd.Timedelta(days=1)
    days = (last.acq_time - first.acq_time) / pd.Timedelta(days=1)
    if not (0 <= age <= 12 and 4 <= days <= 30):
        return (np.nan, np.nan, "", "")
    return (
        round(float(np.clip((last.kbbl - first.kbbl) / 1000 * 7 / days, -5, 5)), 8),
        round(float(age), 4),
        last.acq_time.isoformat(),
        max(first.available_at, last.available_at).isoformat(),
    )


def prepare(through):
    panel = pd.read_csv(MODEL_DIR / "weekly_inputs.csv", usecols=["release_date", "week_ending", "cutoff"])
    # Include the documented unlabeled example for parity with command-line inference.
    example = json.loads((MODEL_DIR / "nov18_inputs.json").read_text())
    panel = pd.concat(
        [
            panel,
            pd.DataFrame(
                [
                    {
                        "release_date": example["release_date"],
                        "week_ending": example["week_ending"],
                        "cutoff": example["deadline_utc"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    panel = panel[panel.release_date <= through]
    reports = {}
    for path in CACHE.glob("*.json"):
        item = json.loads(path.read_bytes())
        if not isinstance(item, dict) or not item.get("published") or not item.get("modified"):
            continue
        value = national_estimate(item.get("head", ""))
        if value is not None:
            pub = pd.Timestamp(item["published"])
            mod = pd.Timestamp(item["modified"])
            if pub.tzinfo and mod.tzinfo and pub <= mod:
                reports.setdefault(pub.tz_convert("America/New_York").date(), []).append(
                    (value, pub, mod, item)
                )
    if not reports:
        raise ValueError("Build the OilPrice survey article cache before preparing national inputs")
    national = {
        e["dataTime"]: e
        for e in calendar_rows(through, 215651)
        if e["indicatorsId"] == 9424 and e.get("pubValue")
    }
    f = RAW / "cushing_survey_experts/pelyr_index.csv"
    if not f.exists():
        with urlopen(INDEX_URL, context=TLS, timeout=20) as res:
            write(f, res.read())
    scenes = pd.read_csv(f)
    for k in ("acq_time", "available_at"):
        scenes[k] = pd.to_datetime(scenes[k], utc=True, format="mixed")
    if (scenes.available_at < scenes.acq_time).any() or not np.isfinite(scenes.kbbl).all():
        raise ValueError("Invalid radar chronology")
    wire_path = MODEL_DIR / "wire_observations.csv"
    wires = pd.read_csv(wire_path).set_index("release_date") if wire_path.is_file() else pd.DataFrame()
    if not wires.index.is_unique:
        raise ValueError("Duplicate bulletin date")
    rows = []
    for r in panel.itertuples():
        cutoff = pd.Timestamp(r.cutoff)
        radar = radar_features(scenes, r.week_ending, cutoff)
        row = {
            "release_date": r.release_date,
            "week_ending": r.week_ending,
            "national_kbbl": np.nan,
            "national_time": "",
            "national_url": "",
            "national_confirmation": "",
            "radar_weekly_million": radar[0],
            "radar_age_days": radar[1],
            "radar_acquired": radar[2],
            "radar_available": radar[3],
        }
        e = national.get(r.week_ending)
        if e:
            stamp = pd.Timestamp(e["pubDate"] + "T" + e["pubTime"], tz="Asia/Shanghai")
            day = stamp.tz_convert("America/New_York").date()
            eligible = [
                a
                for a in reports.get(day, [])
                if stamp <= a[1] <= a[2] < cutoff
                and a[1] - stamp <= pd.Timedelta(hours=7)
                and abs(a[0] - float(e["pubValue"]) * 10) < 0.001
            ]
            if eligible:
                value, pub, mod, item = min(eligible, key=lambda a: a[2])
                row.update(
                    national_kbbl=value,
                    national_time=mod.tz_convert("UTC").isoformat(),
                    national_url=item["url"],
                    national_confirmation=f"https://www.fx168news.com/calendar/{e['calendarId']}",
                )
            if r.release_date in wires.index:
                wire = wires.loc[r.release_date]
                if (
                    pd.notna(wire.national_api_kbbl)
                    and wire.week_ending == r.week_ending
                    and stamp <= pd.Timestamp(wire.published_utc) <= pd.Timestamp(wire.available_utc) < cutoff
                    and abs(wire.national_api_kbbl - float(e["pubValue"]) * 10) < 0.001
                    and (
                        not row["national_time"]
                        or pd.Timestamp(wire.available_utc) < pd.Timestamp(row["national_time"])
                    )
                ):
                    row.update(
                        national_kbbl=wire.national_api_kbbl,
                        national_time=wire.available_utc,
                        national_url=wire.source_url,
                        national_confirmation=f"https://www.fx168news.com/calendar/{e['calendarId']}",
                    )
        rows.append(row)
    out = pd.DataFrame(rows)
    write(OUTPUT, out.to_csv(index=False).encode())
    print(
        "US API:",
        out.national_kbbl.count(),
        "Radar pairs:",
        out.radar_weekly_million.count(),
        "Index SHA256:",
        hashlib.sha256(f.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--through", default="2026-11-18")
    prepare(parser.parse_args().through)
