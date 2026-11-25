"""Link the earlier aerial roof review to the exact source tile and publication evidence."""
import csv
import json
import re
from datetime import datetime

from collect_public import RAW, ROOT

REVIEW = ROOT / "data/reference/terminal_early_aerial_roof_review.csv"
ATLAS = ROOT / "data/derived/aerial_review"
ARCHIVE = RAW / "documents/naip_roof_source_metadata"
STATE_METADATA = "https://okmaps.org/ogi/RestDataAccessItem.aspx?UUID=d7150e32-d624-4d2a-abe2-ac52ea89566a"
TABLE = "terminal_early_aerial_roof_evidence"
HCS_SCOPE = "hcs_ground_lease"
SCHEMA = f"""CREATE TABLE IF NOT EXISTS {TABLE} (
    scope_id TEXT NOT NULL,candidate_id TEXT NOT NULL,observed_date TEXT NOT NULL,top_observation TEXT NOT NULL,
    detail TEXT NOT NULL,naip_item_id TEXT NOT NULL,image_file TEXT NOT NULL,image_url TEXT NOT NULL,
    review_sheet TEXT NOT NULL,review_row INTEGER NOT NULL,review_column INTEGER NOT NULL,
    publication_date TEXT NOT NULL,publication_basis TEXT NOT NULL,publication_source_file TEXT NOT NULL,
    publication_source_url TEXT NOT NULL,PRIMARY KEY(scope_id,candidate_id,observed_date));"""


def publication(item):
    identifier = item["id"]
    props = item["properties"]
    if props["naip:state"] != "ok":
        raise ValueError(f"The NAIP item is outside the Oklahoma study archive: {identifier}")
    if props["naip:year"] in (2017, 2019) and "metadata" in item["assets"]:
        path = ARCHIVE / f"{identifier}.txt"
        metadata = path.read_text()
        match = re.search(r"Publication_Date:\s*(\d{8})", metadata)
        capture = props["datetime"][:10].replace("-", "")
        if not match or not re.search(rf"Calendar_Date:\s*{capture}\b", metadata) or "USDA-FSA-APFO" not in metadata:
            raise ValueError(f"Cannot verify USDA publication metadata for {identifier}")
        date = datetime.strptime(match[1], "%Y%m%d").date().isoformat()
        return date, "usda_fgdc_quarter_quad_publication", path, item["assets"]["metadata"]["href"]
    if props["naip:year"] != 2021:
        raise ValueError(f"No independently dated NAIP source for {identifier}")
    path = ARCHIVE / "ok_naip_2021_statewide.html"
    metadata = path.read_text()
    match = re.search(r">Date</td><td>(\d{4}-\d{2}-\d{2})</td></tr><tr><td[^>]*>Date Type</td><td>publication", metadata)
    if not match or "NAIP 2021 NC Statewide" not in metadata:
        raise ValueError("Cannot verify the Oklahoma 2021 statewide NAIP publication")
    return match[1], "oklahoma_2021_naip_statewide_publication", path, STATE_METADATA


def evidence_rows():
    atlas = {}
    for path in (ATLAS / "index.csv", ATLAS / "supplemental/index.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                atlas[row["candidate_id"], row["acquisition_dates"]] = row
    with REVIEW.open(newline="") as stream:
        for row in csv.DictReader(stream):
            scope, candidate, observed, top, detail = row.values()
            source = atlas[candidate, observed]
            image_file = ROOT / source["source_paths"]
            item = json.loads(image_file.with_name("item.json").read_bytes())
            if (item["id"] != source["source_items"] or item["properties"]["datetime"][:10] != observed
                    or not image_file.is_file() or not (ROOT / source["history_sheet"]).is_file()):
                raise ValueError(f"The reviewed NAIP source changed for {candidate} on {observed}")
            available, basis, path, url = publication(item)
            if available < observed:
                raise ValueError(f"Publication precedes acquisition for {candidate}")
            column = ("2010", "2013", "2015", "2017", "2019", "2021", "2023", "2025").index(observed[:4]) + 1
            yield (scope, candidate, observed, top, detail, item["id"], source["source_paths"],
                   item["assets"]["image"]["href"], source["history_sheet"], int(source["history_row"]), column,
                   available, basis, str(path.relative_to(ROOT)), url)


def install_early_roof_review(db):
    db.execute(SCHEMA)
    db.execute(f"DELETE FROM {TABLE}")
    rows = list(evidence_rows())
    db.executemany(f"INSERT INTO {TABLE} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    for _, candidate, observed, top, detail, *source in rows:
        update = db.execute("""UPDATE tank_aerial_visual_observation SET top_observation=?,detail=?
            WHERE candidate_id=? AND observed_date=? AND image_source='USDA NAIP'
            AND structure_observation='aboveground_tank_seen'""", (top, detail, candidate, observed))
        if update.rowcount != 1:
            raise ValueError(f"The earlier roof review did not match a unique existing aerial row for {candidate}")


def hcs_roof_dates():
    return {row[1]: row[11] for row in evidence_rows()
            if row[0] == HCS_SCOPE and row[3] == "inset_wall_shadow_visible"}
