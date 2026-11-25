"""Archive dated NYMEX pipeline snapshots and reconcile its EIA stock tables."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean

import pdfplumber

from audit_cushing_holdout_eia_vintages import stocks, weekly_snapshot
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, get, write

STEM = "cushing_cme_supply"
STORE = RAW / "documents" / "cme_cushing_deliverable_supply"
CME = "https://www.cmegroup.com"
FILINGS = {
    "17-266": ("https://www.cftc.gov/sites/default/files/filings/ptc/17/07/ptc072417nymexdcm006.pdf",
               range(43, 45), range(45, 46)),
    "19-011": ("https://www.cftc.gov/sites/default/files/filings/ptc/19/02/ptc020419nymexdcm004.pdf",
               range(41, 43), range(43, 44)),
    "20-106": ("https://www.cftc.gov/sites/default/files/filings/ptc/20/02/ptc020620nymexdcm002.pdf",
               range(24, 26), range(25, 27)),
    "21-109": (f"{CME}/market-regulation/rule-filings/2021/2/21-109.pdf", range(11, 14), range(13, 15)),
    "21-154": (f"{CME}/market-regulation/rule-filings/2021/6/21-154.pdf", range(16, 18), range(18, 19)),
    "21-435": (f"{CME}/market-regulation/rule-filings/2021/10/21-435.pdf", range(17, 20), range(20, 21)),
    "23-064": (f"{CME}/content/dam/cmegroup/market-regulation/rule-filings/2023/3/23-064_2.pdf",
               range(24, 27), range(27, 28)),
    "23-446": (f"{CME}/content/dam/cmegroup/market-regulation/rule-filings/2023/10/23-446.pdf",
               range(13, 16), range(16, 17)),
    "24-162": (f"{CME}/content/dam/cmegroup/market-regulation/rule-filings/2024/5/24-162_1.pdf",
               range(19, 23), range(23, 24)),
    "24-216": (f"{CME}/content/dam/cmegroup/market-regulation/rule-filings/2024/6/24-216.pdf",
               range(14, 18), range(18, 19)),
}
EXPECTED_COUNTS = {"2013-02": (11, 0), "2015-03": (14, 16), "2018-07": (14, 14),
                   "2020-12": (16, 12), "2023-12": (16, 13)}
STOCK_WINDOWS = {
    "17-266": ("2013-01", 36), "19-011": ("2015-01", 43), "20-106": ("2016-01", 44),
    "21-109": ("2018-01", 36), "21-154": ("2018-01", 36), "21-435": ("2018-01", 36),
    "23-064": ("2020-02", 36), "23-446": ("2020-10", 36), "24-162": ("2021-04", 36),
    "24-216": ("2021-04", 36),
}
# Rounded hub-wide estimates in the narrative sections. The appendix lists
# separate route-level ranges whose endpoints need not sum to these values.
NARRATIVE_INFLOW = {
    "2015-03": ("17-266", 42, 1_600_000, 1_800_000),
    "2018-07": ("21-109", 9, 2_200_000, 2_500_000),
    "2020-12": ("21-109", 9, 2_300_000, 2_600_000),
    "2023-12": ("24-162", 16, 2_200_000, 2_800_000),
}
# The 23-064 stock table is a raster on page 27. Transcribed in its printed order.
RASTER_STOCKS_KBBL = (
    33228, 24765, 25476, 26735, 25212, 25415, 22754, 22340, 25973, 26931, 23926, 25035,
    32648, 33767, 27227, 30810, 34887, 34084, 36952, 42810, 45491, 46148, 47594, 47298,
    53570, 58353, 60373, 59464, 54744, 52722, 50002, 46927, 57986, 56831, 39614, 38237,
)
NUM = r"\d{1,3}(?:,\d{3})+"
CAPACITY = re.compile(rf"^(.+?) ({NUM}) (.+)$")
FLOW = re.compile(rf"^(.+?) ({NUM})(?:\s*(?:-|–|to)\s*({NUM}))?(?: BD)?\s*\((.+)\)$")
SNAPSHOT = re.compile(r"^Crude Oil Flows to Cushing \(as of ([A-Za-z]+ \d{4})\)$")


def number(raw):
    return int(raw.replace(",", "").replace(" ", ""))


def parse_pipeline(line, direction):
    name, capacity, tail = CAPACITY.fullmatch(line).groups()
    details = dict(route_name_as_printed=name, capacity_bpd=number(capacity), owner_as_printed=tail,
                   flow_low_bpd=None, flow_high_bpd=None, crude_mix_as_printed=None, wti_pct_as_printed=None)
    if direction == "inbound":
        owner, lo, hi, mix = FLOW.fullmatch(tail).groups()
        wti = re.search(r"(\d+)% WTI\b", mix)
        details.update(owner_as_printed=owner, flow_low_bpd=number(lo), flow_high_bpd=number(hi or lo),
                       crude_mix_as_printed=mix, wti_pct_as_printed=int(wti[1]) if wti else None)
        if details["flow_low_bpd"] > details["flow_high_bpd"]:
            raise ValueError(f"Reversed flow bounds in {line}")
    return details


def pipeline_tables(pdf, pages, source):
    rows, totals = [], []
    month = direction = None
    for n in pages:
        lines = (pdf.pages[n - 1].extract_text(x_tolerance=2) or "").splitlines()
        for line in lines:
            if match := SNAPSHOT.fullmatch(line):
                month = datetime.strptime(match[1], "%B %Y").strftime("%Y-%m")
                direction = None
            elif line.startswith("Incoming Pipelines") and month:
                direction = "inbound"
            elif line.startswith("Outgoing Pipelines") and month:
                direction = "outbound"
            elif line.startswith("TOTAL") and direction:
                capacity, = re.findall(r"([\d.]+) Million", line)
                flow = re.search(rf"WTI Flow: ({NUM})\s*[-–]\s*({NUM})", line)
                if (direction == "inbound") != bool(flow):
                    raise ValueError(f"Missing inbound WTI totals: {line}")
                totals.append(dict(snapshot_month=month, direction=direction, pdf_page=n,
                                   printed_capacity_bpd=round(float(capacity) * 1e6),
                                   printed_wti_low_bpd=number(flow[1]) if flow else None,
                                   printed_wti_high_bpd=number(flow[2]) if flow else None,
                                   **source))
                direction = None
            elif direction and CAPACITY.fullmatch(line):
                details = parse_pipeline(line, direction)
                mode = "rail" if details["route_name_as_printed"] == "Bakken Crude via Rail" else "pipeline"
                rows.append(dict(snapshot_month=month, direction=direction, transport_mode=mode,
                                 pdf_page=n, **details, **source))
    return rows, totals


def stock_rows(pdf, pages, source):
    records = []
    filing = source["filing_id"]
    first_month, month_count = STOCK_WINDOWS[filing]
    first = datetime.strptime(first_month, "%Y-%m")
    year = first.year
    for page in pages:
        if filing == "23-064":
            image, = pdf.pages[page - 1].images
            image_sha = hashlib.sha256(image["stream"].get_data()).hexdigest()
            for i, kbbl in enumerate(RASTER_STOCKS_KBBL):
                y, m = divmod(2023 * 12 - i, 12)
                records.append((f"{y}-{m+1:02d}", kbbl, page, "manual_image_transcription", image_sha))
            continue
        text = pdf.pages[page - 1].extract_text(x_tolerance=2) or ""
        if "Cushing Storage" in text or "Cushing Stocks" in text:
            text = re.split(r"Cushing (?:Storage|Stocks)\d+", text, maxsplit=1)[1]
        for line in text.splitlines():
            if match := re.fullmatch(r"(\d{4})", line):
                year = int(match[1])
            elif match := re.fullmatch(r"([A-Z][a-z]{2}-\d{2}) ([\d, ]+)", line):
                month = datetime.strptime(match[1], "%b-%y").strftime("%Y-%m")
                records.append((month, number(match[2]), page, "pdf_text", None))
            elif match := re.fullmatch(r"(?:(\d{4}) )?([A-Z][a-z]{2}|July) ([\d,]+)", line):
                if match[1]:
                    year = int(match[1])
                if match[2] == "Jan" and records and records[-1][0] == f"{year}-12":
                    year += 1
                month = datetime.strptime(f"{year} {match[2][:3]}", "%Y %b").strftime("%Y-%m")
                records.append((month, number(match[3]), page, "pdf_text", None))
    months = [divmod(first.year * 12 + first.month - 1 + i, 12) for i in range(month_count)]
    if sorted(r[0] for r in records) != [f"{year}-{month+1:02d}" for year, month in months]:
        raise ValueError(f"Expected {month_count} successive monthly stock rows in {filing}, found {records}")
    return [dict(stock_month=month, cme_mean_weekly_stocks_kbbl=kbbl, pdf_page=page,
                 extraction=method, embedded_image_sha256=sha, **source)
            for month, kbbl, page, method, sha in sorted(records)]


def load_filings():
    sources, routes, totals, monthly = [], [], [], []
    for filing, (url, pipeline_pages, stock_pages) in FILINGS.items():
        path = STORE / f"{filing}.pdf"
        raw = path.read_bytes() if path.exists() else get(url)
        if not raw.startswith(b"%PDF-"):
            raise ValueError(f"{url} did not return a PDF")
        if not path.exists():
            write(path, raw)
        with pdfplumber.open(path) as pdf:
            cover = pdf.pages[0].extract_text() or ""
            match = re.search(r"Filing Date \(mm/dd/yy\): (\d+/\d+/\d+)", cover)
            if not match or f"(optional): {filing}" not in cover:
                raise ValueError(f"Unexpected cover sheet for {filing}")
            day, fmt = match[1], "%m/%d/%Y" if len(match[1].split("/")[-1]) == 4 else "%m/%d/%y"
            source = dict(filing_id=filing, filing_date=datetime.strptime(day, fmt).date().isoformat(),
                          publication_timestamp_verified=False,
                          source_url=url, source_path=str(path.relative_to(ROOT)),
                          source_sha256=hashlib.sha256(raw).hexdigest())
            sources.append(source)
            rows, reported_totals = pipeline_tables(pdf, pipeline_pages, source)
            routes.extend(rows)
            totals.extend(reported_totals)
            monthly.extend(stock_rows(pdf, stock_pages, source))
    return sources, routes, totals, monthly


def reconcile_routes(routes, totals):
    editions = defaultdict(list)
    for r in routes:
        editions[r["snapshot_month"], r["direction"], r["filing_id"]].append(r)
    total_index = {(r["snapshot_month"], r["direction"], r["filing_id"]): r for r in totals}
    if total_index.keys() != editions.keys() or len(total_index) != len(totals):
        raise ValueError("Pipeline rows and reported totals have different coverage")
    for key, rows in editions.items():
        month, direction, _ = key
        expected = EXPECTED_COUNTS[month][direction == "outbound"]
        if len(rows) != expected or len({r["route_name_as_printed"] for r in rows}) != expected:
            raise ValueError(f"Incomplete or duplicated route rows in {key}: {len(rows)} of {expected}")
        total = total_index[key]
        total["route_count"] = len(rows)
        total["sum_route_capacity_bpd"] = sum(r["capacity_bpd"] for r in rows)
        for bound in ("low", "high"):
            total[f"sum_route_crude_{bound}_bpd"] = (
                sum(r[f"flow_{bound}_bpd"] for r in rows) if direction == "inbound" else None)
            total[f"sum_route_explicit_wti_{bound}_bpd"] = (
                sum(r[f"flow_{bound}_bpd"] * r["wti_pct_as_printed"] / 100
                    for r in rows if r["wti_pct_as_printed"] is not None) if direction == "inbound" else None)
    versions = defaultdict(list)
    for key, rows in editions.items():
        versions[key[:2]].append(rows)
    earliest = []
    for groups in versions.values():
        first = min(groups, key=lambda rows: rows[0]["filing_date"])
        fields = ("route_name_as_printed", "capacity_bpd", "owner_as_printed", "flow_low_bpd", "flow_high_bpd",
                  "crude_mix_as_printed")
        fingerprint = {tuple(r[f] for f in fields) for r in first}
        for rows in groups:
            total = total_index[(rows[0]["snapshot_month"], rows[0]["direction"], rows[0]["filing_id"])]
            total["matches_earliest_edition_routes"] = fingerprint == {tuple(r[f] for f in fields) for r in rows}
            total["earliest_filing_id_in_archive"] = first[0]["filing_id"]
        earliest.extend(first)
    return sorted(earliest, key=lambda r: (r["snapshot_month"], r["direction"], r["route_name_as_printed"]))


def hub_snapshots(totals, sources):
    source_index = {r["filing_id"]: r for r in sources}
    first = {(r["snapshot_month"], r["direction"]): r for r in totals
             if r["filing_id"] == r["earliest_filing_id_in_archive"]}
    for month in sorted(EXPECTED_COUNTS):
        inbound = first[month, "inbound"]
        outbound = first.get((month, "outbound"), {})
        narrative = NARRATIVE_INFLOW.get(month)
        if narrative:
            filing, page, low, high = narrative
            source = source_index[filing]
            with pdfplumber.open(ROOT / source["source_path"]) as pdf:
                passage = " ".join((pdf.pages[page - 1].extract_text() or "").split())
            marker = rf"ranged from {low/1e6:g}(?: million)? to {high/1e6:g} million barrels per day"
            if not re.search(marker, passage):
                raise ValueError(f"Missing narrative inflow range in {filing}, page {page}: {marker}")
        else:
            source, page, low, high = {}, None, None, None
        yield dict(
            snapshot_month=month, printed_inbound_capacity_bpd=inbound["printed_capacity_bpd"],
            printed_outbound_capacity_bpd=outbound.get("printed_capacity_bpd"),
            printed_wti_inbound_low_bpd=inbound["printed_wti_low_bpd"],
            printed_wti_inbound_high_bpd=inbound["printed_wti_high_bpd"],
            narrative_crude_inbound_low_bpd=low, narrative_crude_inbound_high_bpd=high,
            narrative_inflow_filing_id=source.get("filing_id"), narrative_inflow_pdf_page=page,
            narrative_inflow_filing_date=source.get("filing_date"),
            narrative_inflow_source_url=source.get("source_url"),
            inbound_table_filing_id=inbound["filing_id"], inbound_table_filing_date=inbound["filing_date"],
            inbound_table_total_pdf_page=inbound["pdf_page"], inbound_table_source_url=inbound["source_url"],
            outbound_table_filing_id=outbound.get("filing_id"), outbound_table_total_pdf_page=outbound.get("pdf_page"),
            outbound_table_source_url=outbound.get("source_url"),
        )


def first_stock_values():
    values = {r["week_ending"]: int(r["cushing_kbbl"]) for r in read("eia_inventory_release_panel")
              if r["reported_as"] == "current_week"}
    training = RAW / "eia" / "wpsr_training_history"
    paths = sorted(training.glob("201[3-5]*_table4.source")) + sorted((RAW / "eia" / "wpsr").glob("2016*_table4.source"))
    for source in paths:
        issue = source.name[:10].replace("_", "-")
        week, _, current, *_ = stocks(issue, directory=source.parent)
        values.setdefault(week, int(current))
    return values


def reconcile_stocks(monthly):
    original = first_stock_values()
    latest = weekly_snapshot()
    for row in monthly:
        month = row["stock_month"]
        # The 2024 filings define their inventory period through March 22.
        end = "2024-03-22" if month == "2024-03" else None
        dates = [day for day in sorted(latest) if day.startswith(month) and (end is None or day <= end)]
        known = [original[day] for day in dates if day in original]
        live_mean = float(mean(latest[day] for day in dates))
        initial_mean = mean(known) if len(known) == len(dates) else None
        row.update(week_count=len(dates), first_release_week_count=len(known),
                   week_endings_json=json.dumps(dates), missing_first_release_weeks_json=json.dumps(
                       [day for day in dates if day not in original]),
                   last_week_used=dates[-1],
                   first_release_mean_kbbl=initial_mean, eia_snapshot_mean_kbbl=live_mean,
                   cme_minus_first_release_mean_kbbl=row["cme_mean_weekly_stocks_kbbl"] - initial_mean
                   if initial_mean is not None else None,
                   cme_minus_eia_snapshot_mean_kbbl=row["cme_mean_weekly_stocks_kbbl"] - live_mean)
    return monthly


def main():
    sources, routes, totals, monthly = load_filings()
    first = reconcile_routes(routes, totals)
    monthly = reconcile_stocks(monthly)
    hub = list(hub_snapshots(totals, sources))
    for name, rows in (("sources", sources), ("route_editions", routes), ("routes", first),
                       ("reported_totals", totals), ("hub_snapshots", hub), ("stock_crosscheck", monthly)):
        export(f"{STEM}_{name}", rows)
    matched = [r for r in monthly if r["first_release_mean_kbbl"] is not None]
    mismatches = [r for r in monthly if abs(r["cme_minus_eia_snapshot_mean_kbbl"]) > .5]
    summary = dict(
        filings=len(sources), route_edition_rows=len(routes), earliest_edition_route_rows=len(first),
        route_counts_by_snapshot_month=dict(sorted(Counter(r["snapshot_month"] for r in first).items())),
        earliest_filing_by_snapshot_month={r["snapshot_month"]: r["filing_date"] for r in first},
        route_tables_differing_from_earliest_edition=sum(not r["matches_earliest_edition_routes"] for r in totals),
        stock_rows=len(monthly), unique_stock_months=len({r["stock_month"] for r in monthly}),
        stock_rows_with_complete_first_release_weeks=len(matched),
        stock_rows_matching_first_release_to_printed_precision=sum(
            abs(r["cme_minus_first_release_mean_kbbl"]) <= .5 for r in matched),
        stock_rows_matching_eia_snapshot_to_printed_precision=len(monthly) - len(mismatches),
        stock_snapshot_disagreements=[
            {k: r[k] for k in ("filing_id", "stock_month", "cme_mean_weekly_stocks_kbbl",
                               "eia_snapshot_mean_kbbl", "cme_minus_eia_snapshot_mean_kbbl")}
            for r in mismatches
        ],
        scope="Occasional NYMEX estimates with filing dates. Outbound tables list capacity. "
              "Monthly stock values are averages of weekly EIA observations. "
              "Publication timestamps are unverified and these tables are research context only.",
    )
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
