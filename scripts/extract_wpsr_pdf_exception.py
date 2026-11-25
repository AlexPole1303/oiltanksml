"""Recover the July 3, 2019 WPSR values printed in the correct archived PDFs."""
import csv
import io
import re

import pdfplumber

from collect_public import RAW, ROOT, WPSR, write

ISSUE = "2019_07_03"
CURRENT_WEEK = "2019-06-28"
PREVIOUS_WEEK = "2019-06-21"
EXPECTED_ROWS = {1: 50, 2: 65, 4: 26, 7: 35, 9: 538}
NUMBER = re.compile(r"[+-]?\d[\d,]*(?:\.\d+)?\Z")


def main():
    directory = RAW / "eia" / "wpsr"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("release_date", "table_number", "pdf_page", "source_line", "column_role", "report_date",
                     "raw_label", "raw_value", "numeric_value", "source_url"))
    for table, expected in EXPECTED_ROWS.items():
        path = directory / f"{ISSUE}_table{table}.pdf"
        url = f"{WPSR}2019/{ISSUE}/pdf/table{table}.pdf"
        with pdfplumber.open(path) as document:
            count = 0
            for page in document.pages:
                lines = (page.extract_text() or "").splitlines()
                for line_number, line in enumerate(lines, 1):
                    row = re.fullmatch(r"(.+?)\s*\.{4,}\s*(.+)", line)
                    if not row:
                        continue
                    label = row[1].strip()
                    values = row[2].split()[:2]
                    if len(values) != 2 or any(not NUMBER.fullmatch(value) and value not in {"–", "—", "-", "W", "NA"}
                                                for value in values):
                        raise ValueError(f"Unrecognized values in {path.name}, page {page.page_number}, line {line_number}")
                    for role, date, raw in zip(("current_week", "previous_week"),
                                               (CURRENT_WEEK, PREVIOUS_WEEK), values, strict=True):
                        number = raw.replace(",", "") if NUMBER.fullmatch(raw) else None
                        writer.writerow(("2019-07-03", table, page.page_number, line_number, role,
                                         date, label, raw, number, url))
                    count += 1
            if count != expected:
                raise ValueError(f"Unexpected row count in {path.name}: {count}, expected {expected}")
            print(f"WPSR table {table}: {count} printed rows")
    write(ROOT / "data" / "derived" / "wpsr_2019_07_03_pdf_values.csv", stream.getvalue().encode())


if __name__ == "__main__":
    main()
