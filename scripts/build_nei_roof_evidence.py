"""Match EPA equipment identifiers across NEI releases and documented terminal cohorts."""
import csv
import gzip
import hashlib
import json
import re
from collections import defaultdict
from email.utils import parsedate_to_datetime

from collect_public import RAW, ROOT
from extract_nei_cushing_processes import ARCHIVES, COUNTIES, OUTPUT

NEI_2020_FILE = OUTPUT / "nei_2020_cushing_county_processes.csv.gz"
NEI_2020_URL = ARCHIVES[2020][0]
CVR_FACILITY = "17115511"
PHILLIPS_FACILITY = "16492611"
ENTERPRISE_EAST_FACILITY = "15496211"
ENTERPRISE_WEST_FACILITY = "16497111"
SUNOCO_LITTLE_FARM_FACILITY = "16494311"
COHORTS = {
    (CVR_FACILITY, "former_cvr_efr_2011"): ("cushing_connect", "EFR", {
        "113457313": "25C1", "113457413": "25C2", "113457513": "25C3", "113457613": "25C4",
    }),
    (CVR_FACILITY, "former_cvr_efr_2015"): ("cushing_connect", "EFR", {
        "128893213": "25C5", "128893313": "25C6",
    }),
    (PHILLIPS_FACILITY, "phillips_buxton_efr"): ("phillips66_cushing", "EFR", {
        "105456913": "T1501", "105457013": "T2501",
    }),
    (PHILLIPS_FACILITY, "phillips_cushing_efr"): ("phillips66_cushing", "EFR", {
        "105456513": "T824", "105456613": "T1151",
    }),
    (PHILLIPS_FACILITY, "phillips_cushing_unmapped_ifr"): ("phillips66_cushing", "IFR", {
        "105456713": "T1252",
    }),
}
ROOF_CODES = {
    "40301132": "EFR", "40301152": "IFR",
    "40400130": "EFR", "40400140": "EFR", "40400149": "EFR", "40400304": "EFR", "40400306": "EFR",
    "40400160": "IFR", "40400179": "IFR", "40400303": "EFR", "40400305": "IFR", "40400307": "IFR",
}
ADDITIONAL_COHORTS = {
    (ENTERPRISE_EAST_FACILITY, "enterprise_east_efr"): ("enterprise_cushing", "EFR", {
        "97012713": "1011", "97012813": "1004", "97012913": "1010", "97013013": "1007",
        "97013113": "1005", "97013213": "1009", "97013413": "1003", "97013513": "1002",
        "97013613": "1001", "97013713": "1008",
    }),
    (ENTERPRISE_WEST_FACILITY, "enterprise_west_owned_efr"): ("enterprise_cushing", "EFR", {
        "105460013": "41076", "105460913": "41124", "112174013": "41130",
    }),
    (ENTERPRISE_WEST_FACILITY, "enterprise_west_owned_ifr"): ("enterprise_cushing", "IFR", {
        "105460113": "41119", "105460213": "41120", "105460313": "41121", "105460413": "41122",
        "105460513": "41123", "105460613": "41111", "105460713": "41112",
    }),
    (ENTERPRISE_WEST_FACILITY, "enterprise_west_large_efr_operator_roster"): ("enterprise_cushing", "EFR", {
        "105461013": "41150", "105461113": "41151", "105461213": "41152",
        "105461313": "41153", "105461413": "41154",
    }),
    (SUNOCO_LITTLE_FARM_FACILITY, "sunoco_little_farm_efr"): ("energy_transfer_cushing", "EFR", {
        "105465313": "74",
    }),
    (SUNOCO_LITTLE_FARM_FACILITY, "sunoco_little_farm_ifr"): ("energy_transfer_cushing", "IFR", {
        "105464813": "70", "105464913": "71", "105465213": "73",
    }),
}
ID_FIELDS = {
    2014: ("eis_facility_site_id", "eis_emissions_unit_id", "alt_agency_id", "agency_unit_id", None,
           "design_capacity", "dc_unit_of_measure"),
    2017: ("eis_facility_id", "eis_unit_id", "agency_facility_id", "agency_unit_id", "unit_description",
           "design_capacity", "design_capacity_uom"),
    2020: ("eis facility id", "eis unit id", "agency facility id", "agency unit id", "unit description",
           "design capacity", "design capacity uom"),
}
NEI_ROOF_SCHEMA = """
CREATE TABLE IF NOT EXISTS nei_roof_equipment_evidence (
    inventory_year INTEGER, archive_version_date TEXT, terminal_id TEXT, cohort_id TEXT,
    eis_facility_id TEXT, state_facility_id TEXT, eis_unit_id TEXT, state_unit_id TEXT,
    equipment_id TEXT, unit_description_as_published TEXT, documented_roof_type TEXT, scc_codes TEXT,
    design_capacity_bbl REAL, group_mapping_basis TEXT, source_url TEXT, source_file TEXT,
    PRIMARY KEY(inventory_year,eis_facility_id,eis_unit_id));
"""


def equipment_source(year):
    manifest_path = OUTPUT / f"nei_{year}_extract_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    path = OUTPUT / manifest["extract_file"]
    url, member, crc, modified, _ = ARCHIVES[year]
    if set(manifest["rows_by_county"]) != set(COUNTIES.values()):
        raise ValueError(f"{year} NEI extract is outside the EIA-803 county scope")
    if (manifest["inventory_year"], manifest["archive_url"], manifest["archive_member"], manifest["member_crc32"],
        manifest["source_http_last_modified_utc"], manifest["extract_sha256"]) != (
            year, url, member, crc, modified, hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError(f"{year} NEI provenance or extract checksum mismatch")
    return path, url, parsedate_to_datetime(modified).date().isoformat()


def evidence_rows():
    code_roof = {}
    for code, roof in ROOF_CODES.items():
        source = json.loads((RAW / "documents" / f"epa_scc_{code}.json").read_text())
        description = source["attributes"]["scc level four"]["text"].lower()
        roof_word = "ext" if roof == "EFR" else "int"
        named_float = "float" in description or description.startswith("crude oil: breathing loss:")
        if source["code"] != code or roof_word not in description or not named_float:
            raise ValueError(f"EPA changed the meaning of roof code {code}")
        code_roof[code] = roof
    for year, fields in ID_FIELDS.items():
        path, url, version = equipment_source(year)
        facility_key, unit_key, state_facility_key, state_unit_key, description_key, capacity_key, uom_key = fields
        units = defaultdict(list)
        with gzip.open(path, "rt", encoding="latin-1", newline="") as stream:
            for row in csv.DictReader(stream):
                units[row[facility_key], row[unit_key]].append(row)
        for (facility, cohort), (terminal, roof, group) in COHORTS.items():
            observed = [unit for unit in group if (facility, unit) in units]
            if observed and len(observed) != len(group):
                raise ValueError(f"Only part of {cohort} is present in the {year} NEI")
            if year == 2020 and not observed:
                raise ValueError(f"Expected cohort {cohort} is missing from the 2020 NEI")
            for unit in observed:
                rows = units[facility, unit]
                codes = sorted({r["scc"] for r in rows if r["scc"] in code_roof})
                state_ids = {(r[state_facility_key], r[state_unit_key]) for r in rows}
                capacities = {float(r[capacity_key]) for r in rows if r[capacity_key] and r[uom_key] == "BBL"}
                if {code_roof[code] for code in codes} != {roof} or len(state_ids) != 1 or len(capacities) > 1:
                    raise ValueError(f"Contradictory NEI equipment evidence for {unit}, {year}")
                state_facility, state_unit = state_ids.pop()
                description, = {r.get(description_key) for r in rows}
                label = group[unit]
                if description:
                    text = description.upper().replace("-", "").replace(" ", "")
                    if label not in text:
                        raise ValueError(f"NEI equipment label for {unit} no longer matches {label}")
                    basis = "equipment_label_and_roof_codes_in_archive"
                elif cohort == "former_cvr_efr_2011":
                    basis = "four_pre_expansion_efr_units_on_single_cvr_site"
                else:
                    basis = "equipment_label_recovered_from_later_nei_by_eis_unit_id"
                yield (year, version, terminal, cohort, facility, state_facility, unit, state_unit, label, description,
                       roof, "|".join(codes), next(iter(capacities), None), basis, url, str(path.relative_to(ROOT)))
        for (facility, cohort), (terminal, expected_roof, group) in ADDITIONAL_COHORTS.items():
            for unit, label in group.items():
                rows = units[facility, unit]
                codes = sorted({row["scc"] for row in rows if row["scc"] in code_roof})
                roof, = {code_roof[code] for code in codes}
                (state_facility, state_unit), = {(row[state_facility_key], row[state_unit_key]) for row in rows}
                description, = {row.get(description_key) for row in rows}
                capacities = {float(row[capacity_key]) for row in rows
                              if row[capacity_key] and row[uom_key] == "BBL"}
                if description:
                    normalized = description.lower().replace("floaring", "floating")
                    roof_word = "external" if roof == "EFR" else "internal"
                    if label not in re.findall(r"\d+", normalized) or f"{roof_word} floating roof" not in normalized:
                        raise ValueError(f"NEI label and roof code disagree for {facility}/{unit}, {year}")
                    if roof != expected_roof:
                        raise ValueError(f"Unexpected labelled roof for {facility}/{unit}, {year}")
                    basis = "equipment_label_and_roof_codes_in_archive"
                elif cohort == "enterprise_east_efr":
                    basis = "ten_efr_units_on_enterprise_east_site"
                elif roof != expected_roof:
                    basis = "roof_code_differs_from_later_labelled_nei"
                else:
                    basis = "equipment_label_recovered_from_later_nei_by_eis_unit_id"
                if len(capacities) > 1:
                    raise ValueError(f"Conflicting NEI capacities for {facility}/{unit}, {year}")
                yield (year, version, terminal, cohort, facility, state_facility, unit, state_unit, label, description,
                       roof, "|".join(codes), next(iter(capacities), None), basis, url, str(path.relative_to(ROOT)))
        unit = "105465113"
        little_farm_fixed = units[SUNOCO_LITTLE_FARM_FACILITY, unit]
        if year == 2014:
            codes = {row["scc"] for row in little_farm_fixed}
            (state_facility, state_unit), = {(row[state_facility_key], row[state_unit_key]) for row in little_farm_fixed}
            if codes != {"40400301", "40400302"}:
                raise ValueError("The older Sunoco fixed-roof process record changed")
            for code in codes:
                source = json.loads((RAW / "documents" / f"epa_scc_{code}.json").read_text())
                if source["code"] != code or "Fixed Roof Tank" not in source["attributes"]["scc level four"]["text"]:
                    raise ValueError(f"EPA changed the fixed-roof SCC {code}")
            yield (year, version, "energy_transfer_cushing", "sunoco_little_farm_unlabelled_fixed",
                   SUNOCO_LITTLE_FARM_FACILITY, state_facility, unit, state_unit, None, None, "Fixed",
                   "|".join(sorted(codes)), None, "fixed_roof_codes_in_2014_only_equipment_label_unavailable", url,
                   str(path.relative_to(ROOT)))
        elif little_farm_fixed:
            raise ValueError(f"The old Little Farm fixed-roof unit reappeared in the {year} archive")


def install_nei_roof_evidence(db):
    db.execute(NEI_ROOF_SCHEMA)
    db.execute("DELETE FROM nei_roof_equipment_evidence")
    db.executemany("INSERT INTO nei_roof_equipment_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", evidence_rows())
