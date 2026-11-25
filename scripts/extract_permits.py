"""Index Cushing-related Oklahoma DEQ permit records and preserve source pages."""
import re
from html.parser import HTMLParser
from urllib.parse import urljoin

from collect_public import RAW, ROOT, get, run_jobs, write

URL = "https://applications.deq.ok.gov/PermitsPublicReview/viewpermits.aspx"
INDEX = RAW / "documents" / "ok_deq_permits.html"


class PermitTable(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = {"cells": [], "links": []}
        elif tag == "td" and self.row is not None:
            self.cell = []
        elif tag == "a" and self.row is not None:
            href = dict(attrs).get("href", "")
            if "permitinformation.aspx?id=" in href.lower():
                self.row["links"].append(urljoin(URL, href))

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None and self.row is not None:
            self.row["cells"].append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)


def details(url, identifier):
    path = RAW / "documents" / "deq_permits" / f"{identifier}.html"
    content = get(url, path)
    if b"permit" not in content.lower() or len(content) < 1000:
        raise ValueError("Response did not resemble a DEQ permit detail page")
    return len(content)


def main():
    import json
    html = get(URL, INDEX).decode(errors="replace")
    parser = PermitTable()
    parser.feed(html)
    rows = []
    terms = re.compile(r"\bcushing\b|\bwildhorse\b|\bdeeprock\b", re.I)
    for row in parser.rows:
        cells = row["cells"]
        if len(cells) < 9 or cells[0] != "AQD" or not terms.search(" ".join(cells[2:4])):
            continue
        links = list(dict.fromkeys(row["links"]))
        if not links:
            continue
        url = links[0]
        identifier = url.split("id=")[-1]
        rows.append({"id": identifier, "county": cells[1], "entity": cells[2], "facility": cells[3],
                     "permit_number": cells[4], "tier": cells[5], "type": cells[6],
                     "review_stage": cells[7], "status": cells[9] if len(cells) > 9 else None,
                     "detail_url": url})
    path = ROOT / "data" / "derived" / "deq_cushing_permits.json"
    write(path, json.dumps(rows, ensure_ascii=False, indent=2).encode())
    run_jobs([(f"deq:{row['id']}", details, (row["detail_url"], row["id"])) for row in rows],
             "deq.json", workers=3)


if __name__ == "__main__":
    main()
