"""Index the EPA review milestones for permit numbers in the local research registry."""
import csv
import datetime as dt
import html
import re

from collect_public import RAW, ROOT, get, write_csv

URL = "https://www.epa.gov/caa-permitting/operating-permit-timeline-oklahoma"
SOURCE = RAW / "documents" / "epa_ok_operating_permit_timeline.html"
OUTPUT = ROOT / "data" / "derived" / "permit_epa_review_timeline.csv"
FIELDS = ("document_id", "facility_key", "permit_number", "epa_permit_number_raw", "epa_facility_label",
          "deq_memo_date", "epa_review_end_date", "epa_petition_deadline", "epa_comments", "source_url", "source_file")
SCHEMA = """
CREATE TABLE permit_epa_review_timeline (
    document_id TEXT,facility_key TEXT,permit_number TEXT,epa_permit_number_raw TEXT,epa_facility_label TEXT,
    deq_memo_date TEXT,epa_review_end_date TEXT,epa_petition_deadline TEXT,epa_comments TEXT,
    source_url TEXT,source_file TEXT,PRIMARY KEY(document_id,epa_review_end_date,epa_petition_deadline));
"""


def permit_key(value):
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def cells(row):
    return [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", cell)).split())
            for cell in re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", row, re.IGNORECASE | re.DOTALL)]


def timeline_rows(payload):
    with (ROOT / "data" / "reference" / "permit_documents.csv").open(newline="") as stream:
        permits = {permit_key(row["permit_number"]): row for row in csv.DictReader(stream)}
    header_seen = False
    for raw in re.findall(r"<tr\b[^>]*>(.*?)</tr>", payload.decode(), re.IGNORECASE | re.DOTALL):
        row = cells(raw)
        if row[:4] == ["Company", "Permit Number", "End of EPA Review", "Petition Deadline"]:
            header_seen = True
            continue
        if not header_seen or len(row) < 5:
            continue
        company, permit_number, review_end, petition_deadline, comments = row[:5]
        permit = permits.get(permit_key(permit_number))
        if permit is None:
            continue
        end = dt.datetime.strptime(review_end, "%m/%d/%Y").date()
        deadline = dt.datetime.strptime(petition_deadline, "%m/%d/%Y").date()
        if deadline < end:
            raise ValueError(f"EPA petition deadline precedes review end for {permit_number}")
        yield (permit["document_id"], permit["facility_key"], permit["permit_number"], permit_number, company,
               permit["memo_date"], end.isoformat(), deadline.isoformat(), comments, URL, str(SOURCE.relative_to(ROOT)))
    if not header_seen:
        raise ValueError("EPA permit review table header changed")


def main():
    rows = sorted(timeline_rows(get(URL, SOURCE)), key=lambda row: (row[6], row[0]))
    if not rows:
        raise ValueError("No local permit numbers matched the EPA review timeline")
    write_csv(OUTPUT, FIELDS, rows)
    print(f"{len(rows)} EPA review windows across {len({row[0] for row in rows})} local permit documents")


if __name__ == "__main__":
    main()
