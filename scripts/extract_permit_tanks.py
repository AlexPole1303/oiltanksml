"""Extract tank equipment tables from identified public Oklahoma DEQ memoranda."""
import csv
import io
import re
from dataclasses import asdict, dataclass

import pdfplumber

from collect_public import RAW, ROOT, get, write, write_csv

BASE = "https://applications.deq.ok.gov/permitspublic/storedpermits/"
STROUD_APPLICATION = RAW / "documents" / "source_pdfs" / "deq_plains_south_6184830.pdf"
STROUD_APPLICATION_URL = "https://applications.deq.ok.gov/PermitsPublicReview/permitinformation.aspx?id=1072185"
FILES = {
    "3329": ("enbridge_cushing", "1514", "2015-1810-TVR2", "2016-11-30", range(4, 9)),
    "8499": ("enbridge_cushing", "1514", "2021-0377-TVR3", "2023-06-28", range(2, 8)),
    "8122": ("enbridge_grandstaff", "1105", "2021-0303-TVR4", "2022-11-02", range(3, 5)),
    "8303": ("enbridge_grandstaff", "1105", "2021-0303-C (M-1)", "2022-09-19", (2, 3)),
    "6844": ("rose_rock_cushing_north", "7238", "2014-2368-C (M-2)", "2020-06-15", (2, 3)),
    "8118": ("magellan_cushing_south", "7205", "2020-0296-TVR", "2021-12-06", (2,)),
    "8699": ("magellan_cushing_central", "25", "2015-1147-C (M-1) PSD", "2024-02-27", (3,)),
    "8766": ("magellan_cushing_central", "25", "2020-0266-TVR", "2024-06-06", (2, 3)),
    "6816": ("southbow_cushing", "9633", "2016-0267-TV", "2020-04-23", (2, 4)),
    "8795": ("plains_cushing", "2103", "2020-0308-TVR2", "2024-08-28", (6, 7, 8, 9)),
    "9030": ("plains_cushing_south_outside_aoi", "6854", "2024-1235-C", "2025-06-05", (2,)),
    "5618": ("plains_cushing", "2103", "2015-1352-C (M-3) PSD", "2019-06-12", range(3, 8)),
    "8910": ("wildhorse", "16676", "2016-1066-C (M-2) PSD", "2025-01-03", (4,)),
    "6854": ("tidal_cushing", "7643", "2019-0873-TVR", "2020-07-13", (2,)),
    "9147": ("tidal_cushing", "7643", "2025-0311-TVR2", "2025-09-09", (3,)),
}


@dataclass
class Tank:
    facility_key: str
    facility_id: str
    permit_number: str
    memo_date: str
    document_id: str
    pdf_page: int
    group: str
    tank_id: str
    emissions_unit_id: str
    roof_type: str | None
    status_in_draft: str
    capacity_value: float | None
    capacity_unit: str | None
    capacity_bbl: float | None
    construction_or_modification_raw: str | None
    height_ft: float | None
    diameter_ft: float | None
    source_url: str
    stored_product_raw: str | None
    capacity_basis: str | None
    height_basis: str | None
    bottom_design_raw: str | None


def clean(value):
    return " ".join((value or "").split())


def numeric(value):
    stripped = value.replace(",", "")
    return float(stripped) if re.fullmatch(r"\d+(?:\.\d+)?", stripped) else None


def record(document, page, group, tank_id, eu, roof, status, value, unit, date=None, height=None, diameter=None,
           product=None, capacity_basis=None, height_basis=None, bottom=None):
    facility, fid, permit, memo, _ = FILES[document]
    multiplier = {"bbl": 1.0, "gal": 1 / 42, "million_gal": 1_000_000 / 42}
    capacity_bbl = round(value * multiplier[unit], 3) if value is not None else None
    return Tank(facility, fid, permit, memo, document, page, group, tank_id, eu, roof, status,
                value, unit, capacity_bbl, date or None, height, diameter, BASE + document + ".pdf", product,
                capacity_basis or ("permit_capacity_unspecified" if value is not None else None),
                height_basis or ("permit_reported_tank_height" if height is not None else None), bottom)


def permit_tables(page, fill_merged=False, strip_tank_id_footnotes=False):
    extracted = []
    for table in page.find_tables():
        rows = table.extract()
        if strip_tank_id_footnotes and clean(rows[0][1]) == "EU ID#":
            for structure, row in zip(table.rows[1:], rows[1:], strict=True):
                if not clean(row[1]).isdigit():
                    continue
                digits = [glyph for glyph in page.crop(structure.cells[1]).chars if glyph["text"].isdigit()]
                largest = max(glyph["size"] for glyph in digits)
                row[1] = "".join(glyph["text"] for glyph in digits if glyph["size"] >= largest - 0.5)
        if fill_merged and len(table.columns) == 7:
            for col in (3, 6):
                cells = [(structure.cells[col], row[col])
                         for structure, row in zip(table.rows, rows, strict=True)
                         if structure.cells[col] is not None and clean(row[col])]
                for structure, row in zip(table.rows, rows, strict=True):
                    if structure.cells[col] is not None or structure.cells[4] is None:
                        continue
                    _, top, _, bottom = structure.cells[4]
                    midpoint = (top + bottom) / 2
                    covering = [value for (_, cell_top, _, cell_bottom), value in cells
                                if cell_top < midpoint < cell_bottom]
                    if len(covering) == 1:
                        row[col] = covering[0]
        extracted.append(rows)
    return extracted


def enbridge(document, page, tables):
    for table in tables:
        for row in table:
            if len(row) != 7:
                continue
            group, eu, _, label, tank, capacity, date = map(clean, row)
            if not re.fullmatch(r"CG-\d+[A-Z]?", tank):
                continue
            status = "removed" if "REMOVED" in label or "DEMOLISHED" in label else "listed"
            roof = re.search(r"\b(EFR|IFR)\b", label)
            value = numeric(capacity)
            if not roof and value is None and status != "removed":
                status = "uncertain_merged_cell"
            yield record(document, page, group, tank, eu, roof.group(1) if roof else None,
                         status, value, "million_gal" if value is not None else None, date)


def equipment(document, page, tables):
    for table in tables:
        group = next((clean(row[0]) for row in table if clean(row[0]).startswith("EUG")), "")
        for row in table:
            if len(row) != 6:
                if document == "8699" and len(row) == 5 and clean(row[1]) == "Fixed":
                    tank, _, capacity, _, date = map(clean, row)
                    yield record(document, page, "EUG 2", tank, "2", "FR", "listed", numeric(capacity), "gal", date)
                continue
            eu, tank, roof, capacity, _, date = map(clean, row)
            if document == "8766" and roof == "Fixed":
                roof, group = "FR", "EUG 2"
            if document == "6844" and roof == "Cone":
                roof = "FR"
            if roof not in {"EFR", "IFR", "FR"} or not tank.isdigit():
                continue
            value = numeric(capacity)
            if value is None:
                continue
            yield record(document, page, group, tank, eu, roof, "listed", value,
                         "bbl" if document in {"8122", "8303", "6844"} else "gal", date)


def magellan_central_fixed_capacity_evidence():
    for document, conditions_page in (("8699", 29), ("8766", 18)):
        facility, _, _, memo, _ = FILES[document]
        path = RAW / "documents" / "source_pdfs" / f"{document}.pdf"
        with pdfplumber.open(path) as pdf:
            for page, section in ((3, "equipment_memorandum"), (conditions_page, "draft_specific_conditions")):
                fixed = [row for table in pdf.pages[page - 1].extract_tables() for row in table
                         if "0011" in row and "Fixed" in row]
                fixed_row, = fixed
                capacity = numeric(clean(fixed_row[fixed_row.index("Fixed") + 1]))
                if capacity is None:
                    raise ValueError(f"Missing Cushing Central tank 0011 capacity in {document} page {page}")
                yield (facility, "0011", document, memo, page, section, capacity, round(capacity / 42, 3),
                       BASE + document + ".pdf", str(path.relative_to(ROOT)))


def stroud_application_tanks():
    if not STROUD_APPLICATION.is_file():
        return
    with pdfplumber.open(STROUD_APPLICATION) as pdf:
        cover = pdf.pages[2].extract_text() or ""
        equipment = pdf.pages[10].extract_text() or ""
        dimensions = pdf.pages[38].extract_text() or ""
    if "November 26, 2025" not in cover or "Tank Identification TANK1 - TANK2" not in dimensions:
        raise ValueError("Unexpected date or equipment group in the Stroud operating permit application")
    installs = dict(re.findall(r"^(TANK[12])\s+(\d{1,2}/\d{4})\s+", equipment, re.MULTILINE))
    diameter = re.search(r"^Diameter, ft\s+([\d.]+)$", dimensions, re.MULTILINE)
    height = re.search(r"^Shell Height or Length, ft\s+([\d.]+)$", dimensions, re.MULTILINE)
    capacity = re.search(r"^Nominal Capacity, gal\s+([\d,]+)$", dimensions, re.MULTILINE)
    if set(installs) != {"TANK1", "TANK2"} or not diameter or not height or not capacity:
        raise ValueError("Missing Stroud tank dimensions or installation dates")
    value = numeric(capacity[1])
    for tank, installation in sorted(installs.items()):
        yield Tank("plains_cushing_south_outside_aoi", "6854", "2024-1235-O", "2025-11-26", "6184830", 39,
                   "TANK1 - TANK2", tank, tank, "IFR", "listed", value, "gal", round(value / 42, 3),
                   installation, float(height[1]), float(diameter[1]), STROUD_APPLICATION_URL, "Crude Oil",
                   "application_reported_nominal_capacity", "application_emissions_input_shell_height", None)


def transcanada(pages):
    dimension_groups = {}
    for table in pages[4]:
        for row in table:
            if len(row) != 5:
                continue
            label, capacity, height, diameter, roof = map(clean, row)
            if roof not in {"EFR", "Domed EFR"}:
                continue
            numbers = [int(value) for value in re.findall(r"EU-(\d+)", label)]
            if "thru" in label and len(numbers) == 2:
                numbers = list(range(numbers[0], numbers[1] + 1))
            for number in numbers:
                dimension_groups[number] = (numeric(capacity), numeric(height), numeric(diameter), roof)
    for table in pages[2]:
        for row in table:
            if len(row) != 4:
                continue
            eu, tank, label, date = map(clean, row)
            match = re.fullmatch(r"EU-(\d+)", eu)
            if not match or "Crude Oil Storage Tank" not in label:
                continue
            number = int(match.group(1))
            capacity, height, diameter, roof = dimension_groups[number]
            yield record("6816", 2, "EUG 2", tank, eu, roof,
                         "proposed_not_constructed" if date == "TBD" else "listed",
                         capacity, "gal", date, height, diameter, "Crude Oil")


def plains(document, pages):
    group = None
    for page, tables in pages.items():
        for table in tables:
            if len(table[0]) != 6 or clean(table[0][2]) != "Contents":
                continue
            for row in table[1:]:
                eug, tank, product, roof, capacity, date = map(clean, row)
                if eug:
                    group = int(eug)
                if roof not in {"EFR", "IFR"}:
                    continue
                value = numeric(capacity)
                if group is None or value is None or not tank:
                    raise ValueError(f"Invalid Plains tank table row on page {page}: {row}")
                observation = record(document, page, str(group), tank, tank, roof, "listed", value, "bbl", date,
                                     product=product)
                if group in (11, 12):
                    observation.facility_key = "former_cvr_cushing"
                    observation.facility_id = "7955"
                yield observation


def plains_south(page, tables):
    for table in tables:
        for row in table:
            if len(row) != 7:
                continue
            tank, label, capacity, _, _, date, _ = map(clean, row)
            if not re.fullmatch(r"TANK[12]", tank) or label != "Internal Floating Roof Crude Oil Storage Tank":
                continue
            value = numeric(capacity.removesuffix("-bbl"))
            if value is None:
                raise ValueError(f"Missing Plains Cushing South tank capacity: {row}")
            yield record("9030", page, "equipment", tank, tank, "IFR", "listed", value, "bbl", date,
                         product="Crude Oil")


def wildhorse(page, tables):
    for table in tables:
        if len(table[0]) != 8 or clean(table[0][5]) != "Working Height (ft)":
            continue
        for row in table[1:]:
            tank, product, roof, bottom, capacity, height, diameter, date = map(clean, row)
            if roof not in {"EFR", "IFR"} or not re.fullmatch(r"T-\d{3}", tank):
                raise ValueError(f"Unexpected Wildhorse tank row {row}")
            yield record("8910", page, "as-built tank table", tank, tank, roof, "as_built_reported",
                         numeric(capacity), "bbl", date, numeric(height), numeric(diameter), product,
                         "nominal_reported", "permit_working_height", bottom)


def tidal(document, page, tables):
    for table in tables:
        for row in table:
            if len(row) != 5 or not re.fullmatch(r"TANK1\s*–\s*TANK\s*8 \(8 Tanks\)", clean(row[0])):
                continue
            capacity = numeric(clean(row[2]))
            if capacity != 16800:
                raise ValueError(f"Unexpected Tidal tank capacity in {document}")
            for number in range(1, 9):
                tank = f"TANK{number}"
                yield record(document, page, "8 fixed-roof condensate tanks", tank, tank, "FR", "listed",
                             capacity, "gal", clean(row[4]), product="Condensate")


def main():
    raw = RAW / "documents" / "source_pdfs"
    rows = []
    for document, (_, _, _, _, selection) in FILES.items():
        path = raw / (document + ".pdf")
        if not path.is_file():
            payload = get(BASE + document + ".pdf")
            if not payload.startswith(b"%PDF-"):
                raise ValueError(f"DEQ response is not a PDF for permit {document}")
            write(path, payload)
        with pdfplumber.open(path) as pdf:
            pages = {n: permit_tables(pdf.pages[n - 1], document in {"3329", "8499"}, document == "5618")
                     for n in selection}
        if document == "6816":
            found = list(transcanada(pages))
        elif document in {"5618", "8795"}:
            found = list(plains(document, pages))
        elif document == "8910":
            found = list(wildhorse(4, pages[4]))
        elif document in {"6854", "9147"}:
            found = [row for page, tables in pages.items() for row in tidal(document, page, tables)]
        else:
            if document == "9030":
                found = list(plains_south(2, pages[2]))
            else:
                handler = enbridge if document in {"3329", "8499"} else equipment
                found = [row for page, tables in pages.items() for row in handler(document, page, tables)]
        if not found or len(found) != len({(row.facility_key, row.tank_id) for row in found}):
            raise ValueError(f"Missing or repeated tank IDs in the selected tables for {document}")
        print(document, len(found))
        rows.extend(found)
    rows.extend(stroud_application_tanks())
    stream = io.StringIO(newline="")
    fields = list(asdict(rows[0]))
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    write(ROOT / "data" / "derived" / "permit_tanks.csv", stream.getvalue().encode())
    write_csv(ROOT / "data" / "derived" / "permit_tank_capacity_crosscheck.csv",
              ("facility_key", "tank_id", "document_id", "memo_date", "pdf_page", "document_section", "capacity_gal",
               "capacity_bbl", "source_url", "source_file"), magellan_central_fixed_capacity_evidence())


if __name__ == "__main__":
    main()
