"""Publish exact optical endpoints for the newly documented terminal cohorts."""
import csv
import json
import sqlite3
from collections import Counter
from contextlib import closing

from audit_lidar_tanks import DATASET, LEGACY_DATASET
from build_lidar_roof_reference import NEI_ROOF_DATES
from build_terminal_lidar_source_timing import SOURCE_FILE as LIDAR_TIMING
from build_tank_geometry_registry import TERMINAL_GEOMETRY_STUDY as TERMINALS
from build_weekly_shadow_panel import DERIVED, PAIRED_ENDPOINTS, SOURCE, endpoint_ctes, export, read_only
from collect_public import ROOT, write

STEM = "weekly_shadow_terminal_endpoint_manifest"
VISUAL_DATE_REQUIRED = {"enterprise_west_owned_efr", "sunoco_little_farm_efr"}


def main():
    dates, diameters = {}, {}
    with LIDAR_TIMING.open(newline="") as stream:
        for r in csv.DictReader(stream):
            key = r["candidate_id"],r["dataset"]
            dates[key] = max(dates.get(key,""),r["source_catalog_date"])
            if key in diameters and diameters[key]!=float(r["diameter_m"]):
                raise ValueError(f"More than one diameter for dated LiDAR source {key}")
            diameters[key] = float(r["diameter_m"])
    with closing(read_only(SOURCE)) as source:
        source.row_factory = sqlite3.Row
        markers = ",".join("?" for _ in TERMINALS)
        sql = endpoint_ctes(f"AND t.documented_roof_type='EFR' AND t.documented_terminal_id IN ({markers})")
        rows = [dict(row) for row in source.execute(sql + PAIRED_ENDPOINTS, TERMINALS)]
        roof_dates = {(terminal, cohort): day for terminal, cohort, _, day in source.execute(
            NEI_ROOF_DATES + " HAVING documented_roof_type='EFR'")}
    if not rows:
        raise ValueError("No exact weekly endpoints for the new cohorts")
    pairs = {}
    for row in rows:
        row.pop("rank")
        earlier = row["candidate_id"],LEGACY_DATASET
        modern = row["candidate_id"],DATASET
        cutoff = row["inputs_available_before_utc"][:10]
        old_known = earlier in dates and dates[earlier]<cutoff
        modern_known = dates[modern]<cutoff
        roof_date = roof_dates.get((row["terminal_id"], row["cohort_id"]))
        nei_known = int(roof_date < cutoff) if roof_date else None
        document_known = int(row["source_document_date"] < cutoff) if row["source_document_date"] else None
        roof_known = bool(nei_known) and (row["cohort_id"] not in VISUAL_DATE_REQUIRED or bool(document_known))
        row.update(lidar_2011_source_catalog_date=dates.get(earlier),lidar_2011_diameter_m=diameters.get(earlier),
                   lidar_2022_source_catalog_date=dates[modern],lidar_2022_diameter_m=diameters[modern],
                   lidar_2011_source_dated_before_cutoff=old_known,lidar_2022_source_dated_before_cutoff=modern_known,
                   lidar_diameter_m_from_dated_source=diameters[modern] if modern_known else diameters.get(earlier) if old_known else None,
                   lidar_dataset_from_dated_source=DATASET if modern_known else LEGACY_DATASET if old_known else None,
                   earliest_nei_roof_archive_date=roof_date,nei_roof_archive_dated_before_cutoff=nei_known,
                   source_document_dated_before_cutoff=document_known,roof_evidence_dated_before_cutoff=roof_known)
        key = row["week_ending"],row["candidate_id"]
        endpoints = pairs.setdefault(key,{})
        endpoint = row["endpoint"]
        if endpoint in endpoints:
            raise ValueError(f"Duplicate selected endpoint for {key}")
        endpoints[endpoint] = row
    counts = Counter()
    roof_and_lidar = Counter()
    for key, pair in pairs.items():
        if set(pair)!={"current","previous"}:
            raise ValueError(f"Incomplete exact optical pair for {key}")
        current, previous = pair["current"],pair["previous"]
        if previous["captured_at_utc"]>=current["captured_at_utc"]:
            raise ValueError(f"Nonchronological optical pair for {key}")
        for row in (current,previous):
            if not all((ROOT/row["source_directory"]/f"{band}.tif").is_file() for band in ("blue","green","red","nir")):
                raise ValueError(f"Missing four-band L1C product for {key}")
            metadata = ROOT/row["tile_metadata_path"]
            if not metadata.is_file() and not metadata.with_suffix(metadata.suffix+".gz").is_file():
                raise ValueError(f"Missing L1C angle XML for {key}")
        counts[current["terminal_id"],current["study_phase"],bool(current["nei_roof_archive_dated_before_cutoff"])]+=1
        if current["roof_evidence_dated_before_cutoff"] and current["lidar_dataset_from_dated_source"]:
            roof_and_lidar[current["terminal_id"],current["study_phase"]]+=1
    export(STEM,rows)
    report = dict(pairs=len(pairs),endpoints=len(rows),
                  counts=[dict(terminal_id=terminal,study_phase=phase,nei_roof_archive_dated_before_cutoff=dated,pairs=n)
                          for (terminal,phase,dated),n in sorted(counts.items())],
                  pairs_with_dated_roof_and_lidar_sources=[
                      dict(terminal_id=terminal,study_phase=phase,pairs=n) for (terminal,phase),n in sorted(roof_and_lidar.items())],
                  scope="Exact L1C pairs for documented external roofs. The dated roof flag includes aerial roof reconciliation for the Enterprise West and Sunoco Little Farm groups. The original diameter is retrospective. A separate diameter is recorded if the catalog dates its source before the cutoff.")
    write(DERIVED/f"{STEM}_summary.json",json.dumps(report,indent=2).encode()+b"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
