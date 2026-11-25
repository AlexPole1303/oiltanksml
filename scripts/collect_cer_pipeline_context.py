"""Preserve Canadian pipeline source snapshots and normalize the monthly readings."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

from build_s1_reference_matches import export
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, write


STEM = "cer_pipeline_context"
STORE = RAW / "cer_pipeline"
CATALOG = "https://open.canada.ca/data/api/3/action/package_show?id=dc343c43-a592-4a27-8ee7-c77df56afb34"
FILES = ("keystone-throughput-and-capacity", "enbridge-mainline-throughput-and-capacity", "trans-mountain-throughput-and-capacity",
         "archived-trans-mountain-throughput-and-capacity", "apportionment", "oil-pipeline-apportionment-data-dictionary",
         "oil-pipeline-data-dictionary")
POINTS = {"keystone-throughput-and-capacity": ("International boundary at or near Haskett, Manitoba",),
          "enbridge-mainline-throughput-and-capacity": ("ex-Gretna",),
          "trans-mountain-throughput-and-capacity": ("Sumas", "Westridge")}
KBPD_PER_THOUSAND_M3_PER_DAY = Decimal(1) / Decimal("0.158987294928")
METRICS = {"throughput": "Throughput", "committed": "Committed Volumes", "uncommitted": "Uncommitted Volumes",
           "nameplate_capacity": "Nameplate Capacity", "available_capacity": "Available Capacity",
           "original_nominations": "Original Nominations", "accepted_nominations": "Accepted Nominations"}


def fetch(url, path):
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use the operating system's certificate trust store for the CER endpoints.
        with TemporaryDirectory(dir=path.parent) as folder:
            target = Path(folder) / path.name
            run(["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "2", "--max-time", "30",
                 "--output", str(target), url], check=True)
            write(path, target.read_bytes())
    return path.read_bytes()


def source_rows(payload):
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    for number, raw in enumerate(reader, 2):
        if None in raw or any(v is None for v in raw.values()):
            raise ValueError(f"Invalid CSV record {number}")
        yield number, {k: " ".join(v.split()) for k, v in raw.items()}


def decimal(value):
    if value:
        number = Decimal(value)
        if not number.is_finite():
            raise ValueError(f"Non-finite CER value {value}")
        return number
    return None


def normalize(file, payload):
    for number, row in source_rows(payload):
        month = date.fromisoformat(row["Date"])
        if month.day != 1 or (month.year, month.month) != (int(row["Year"]), int(row["Month"])):
            raise ValueError(f"CER row month differs from source fields {file}/{number}")
        measurements = {k: decimal(row.get(f"{column} (1000 m3/d)")) for k, column in METRICS.items()}
        yield dict(source_file_id=file, source_record=number, data_month=month.isoformat(),
            reported_company=row["Company"], pipeline=row["Pipeline"], key_point=row["Key Point"],
            flow_direction=row.get("Direction Of Flow"), trade_type=row.get("Trade Type"), product=row.get("Product"),
            **{f"{k}_thousand_m3_per_day": v for k, v in measurements.items()},
            **{f"{k}_kbbl_per_day": v * KBPD_PER_THOUSAND_M3_PER_DAY if v is not None else None for k, v in measurements.items()},
            apportionment_source_fraction=decimal(row.get("Apportionment Percentage")),
            reason_for_capacity_variance=row.get("Reason For Variance"),
            historical_publication_at_utc=None)


def point_summary(rows):
    groups = defaultdict(list)
    for r in rows:
        if r["key_point"] in POINTS.get(r["source_file_id"], ()):
            groups[r["source_file_id"], r["key_point"], r["data_month"]].append(r)
    for (file, point, month), sample in sorted(groups.items()):
        def sum_flow(product_names):
            values = [r["throughput_kbbl_per_day"] for r in sample if r["product"] in product_names
                      and r["throughput_kbbl_per_day"] is not None]
            return sum(values) if values else None

        def one_capacity(kind):
            values = {r[f"{kind}_capacity_kbbl_per_day"] for r in sample if r[f"{kind}_capacity_kbbl_per_day"] is not None}
            return next(iter(values)) if len(values) == 1 else None

        yield dict(source_file_id=file, key_point=point, data_month=month,
            heavy_kbbl_per_day=sum_flow({"domestic heavy"}), light_or_light_ngl_kbbl_per_day=sum_flow({"domestic light", "domestic light / ngl"}),
            foreign_light_kbbl_per_day=sum_flow({"foreign light"}), refined_products_kbbl_per_day=sum_flow({"refined petroleum products"}),
            available_capacity_kbbl_per_day=one_capacity("available"), nameplate_capacity_kbbl_per_day=one_capacity("nameplate"),
            contributing_source_records_json=json.dumps([r["source_record"] for r in sample]),
            historical_publication_at_utc=None)


def main():
    catalog = fetch(CATALOG, STORE / "catalog.json")
    package = json.loads(catalog)["result"]
    resources = {Path(urlsplit(r["url"]).path).stem: r for r in package["resources"]
                 if "/open/energy/throughput-capacity/" in r["url"]}
    sources, flows, apportioned = [], [], []
    for file in FILES:
        source = resources[file]
        url = source["url"]
        path = STORE / f"{file}.csv"
        payload = fetch(url, path)
        rows = list(source_rows(payload))
        sources.append(dict(source_file_id=file, catalog_resource_id=source["id"], title=source["name"].strip(),
            path=str(path.relative_to(ROOT)), url=url, sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
            rows=len(rows), publisher_last_modified=source.get("last_modified"), license=package["license_title"]))
        if "Date" in rows[0][1]:
            (apportioned if file == "apportionment" else flows).extend(normalize(file, payload))
    monthly = list(point_summary(flows))
    for suffix, rows in (("sources", sources), ("throughput", flows), ("apportionment", apportioned), ("key_points", monthly)):
        export(f"{STEM}_{suffix}", rows)
    summary = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), catalog_url=CATALOG,
        catalog_sha256=hashlib.sha256(catalog).hexdigest(), source_files=len(sources),
        throughput_rows=len(flows), apportionment_rows=len(apportioned), key_point_months=len(monthly),
        key_point_months_by_file=Counter(r["source_file_id"] for r in monthly),
        latest_data_month=max(r["data_month"] for r in flows),
        scope="Current publisher snapshots. Observation month and download time do not establish historical publication or revision timing. "
              "The Enbridge light category includes NGL. Trans Mountain points are kept separately.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
