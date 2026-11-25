"""Normalize the EIA historical crude-oil stocks and capacity workbook."""
import csv
import datetime as dt
import io
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from collect_public import RAW, ROOT, write

XLSX = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
LINK = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
SOURCE = "https://www.eia.gov/petroleum/storagecapacity/crudeoilstorage.xlsx"
DATE_BASE = dt.date(1899, 12, 30)


def cells(row, strings):
    result = {}
    for node in row.findall("s:c", XLSX):
        value = node.findtext("s:v", namespaces=XLSX)
        if value is not None:
            result[node.get("r")] = strings[int(value)] if node.get("t") == "s" else value
    return result


def main():
    workbook = RAW / "eia" / "crudeoilstorage.xlsx"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("area", "period_month", "workbook_header_date", "metric_label", "unit",
                     "raw_value", "numeric_value", "source_cell", "source_url"))
    with ZipFile(workbook) as archive:
        sheets = ET.fromstring(archive.read("xl/workbook.xml")).find("s:sheets", XLSX)
        links = {item.get("Id"): item.get("Target") for item in ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))}
        strings = ["".join(node.text or "" for node in item.findall(".//s:t", XLSX))
                   for item in ET.fromstring(archive.read("xl/sharedStrings.xml"))]
        for sheet in sheets:
            area = sheet.get("name")
            contents = ET.fromstring(archive.read("xl/" + links[sheet.get(LINK)]))
            data = {int(row.get("r")): cells(row, strings) for row in contents.findall(".//s:row", XLSX)}
            dates = {re.sub(r"\d+$", "", cell): (DATE_BASE + dt.timedelta(days=int(value))).isoformat()
                     for cell, value in data[4].items() if value.isdigit()}
            if len(dates) < 20:
                raise ValueError(f"Unexpected date header in {area}")
            for number, row in data.items():
                if number <= 4:
                    continue
                label = row.get(f"A{number}", "").strip()
                if not label:
                    continue
                unit = "fraction" if "utilization rate" in label.lower() else "thousand_barrels"
                for column, observed in dates.items():
                    cell = f"{column}{number}"
                    raw = row.get(cell)
                    if raw is None:
                        continue
                    try:
                        value = float(raw)
                    except ValueError:
                        value = None
                    writer.writerow((area, observed[:7], observed, label, unit, raw, value, cell, SOURCE))
            print(area, len(dates), min(dates.values()), max(dates.values()))
    write(ROOT / "data" / "derived" / "eia_crude_storage_capacity.csv", buffer.getvalue().encode())


if __name__ == "__main__":
    main()
