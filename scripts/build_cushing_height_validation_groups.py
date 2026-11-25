"""Attach reference clocks and shared-acquisition groups to the height registry."""

import json
import math
import re
from collections import defaultdict
from contextlib import closing
from datetime import datetime

from build_cushing_validation_registry import PLAIN, STEM as REGISTRY
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import read_only
from collect_public import write
from collect_s2_l1c import parse_product
from fit_shadow_geometry import CALIBRATION

STEM = "cushing_height_validation_groups"
LANDSAT = re.compile(r"(L[A-Z]\d{2})_L\w+_(\d{6})_(\d{8})_\d{8}_\d{2}_\w+")


def optical_pass(product):
    if product.startswith("S2"):
        (platform, sensing, orbit, _), _, _ = parse_product(product)
        return ":".join((platform, sensing, orbit))
    return ":".join(LANDSAT.fullmatch(product).groups())


def lidar_sources():
    flights = defaultdict(list)
    with closing(read_only(CALIBRATION)) as db:
        for i, dataset, day, line, start, end in db.execute("""SELECT observation_id,dataset,acquisition_date_utc,
            point_source_id,first_return_at_utc,last_return_at_utc FROM observation_lidar
            WHERE level='L1C' AND rim_minus_center_plane_native_z IS NOT NULL"""):
            flights[str(i)].append((f"{dataset}:{day}", line, datetime.fromisoformat(start), datetime.fromisoformat(end)))
    return flights


def timestamps(row, first, last):
    return datetime.fromisoformat(row[first]), datetime.fromisoformat(row[last])


def reference_clock(source, row, lidar):
    if source == "shadow_geometry_estimates":
        flights = lidar[row["observation_id"]]
        campaign, = {r[0] for r in flights}
        if len(flights) != int(row["lidar_flight_lines"]):
            raise ValueError(f"LiDAR flight lines differ for observation {row['observation_id']}")
        lines = ",".join(map(str, sorted(r[1] for r in flights)))
        return campaign, f"{campaign}:{lines}", min(r[2] for r in flights), max(r[3] for r in flights)
    if source == f"{PLAIN}_lidar_reference_comparisons":
        return ("OK_Statewide_2_D22:2022-12-31", row["candidate_id"],
                *timestamps(row, "lidar_first_return_utc", "lidar_last_return_utc"))
    if source == "cushing_icesat2_native_optical_controls":
        return (row["icesat_granule"], row["beam"], *timestamps(row, "icesat_granule_start_utc", "icesat_granule_end_utc"))
    if source == "cushing_icesat2_esri_b8_comparisons":
        return (row["icesat_granule"], row["icesat_cmr_id"],
                *timestamps(row, "icesat_first_roof_photon_utc", "icesat_last_roof_photon_utc"))
    if source == "cushing_gedi_native_optical_fits":
        stamp = datetime.fromisoformat(row["gedi_capture_utc"])
        return row["gedi_l1b_granule"], row["gedi_shot_number"], stamp, stamp
    if source == "cushing_naip_satellite_controls_pairs":
        return f"NAIP:{row['naip_date']}", row["naip_polygon_id"], *timestamps(row, "naip_start_utc", "naip_end_exclusive_utc")
    raise ValueError(f"Unknown height control source {source}")


def enrich(heights):
    lidar = lidar_sources()
    sources = {path: read(path.rsplit("/", 1)[-1].removesuffix(".csv")) for path in {r["source_table"] for r in heights}}
    for r in heights:
        path = r["source_table"]
        source = path.rsplit("/", 1)[-1].removesuffix(".csv")
        row = sources[path][int(r["source_csv_row"])-2]
        if row["candidate_id"] != r["candidate_id"]:
            raise ValueError(f"Height source row points to a different roof in {path}")
        campaign, reference, start, end = reference_clock(source, row, lidar)
        capture = datetime.fromisoformat(r["capture_at_utc"])
        separation = max(abs((capture-start).total_seconds()), abs((capture-end).total_seconds()))/3600
        if end < start or not math.isclose(separation, float(r["max_time_separation_hours"]), abs_tol=.000501):
            raise ValueError(f"Reference clock differs for {path} row {r['source_csv_row']}")
        optical = optical_pass(r["product_id"])
        campaign_pair = f"{campaign}|{optical}"
        yield dict(r, reference_campaign_id=campaign, reference_source_id=reference, reference_start_utc=start.isoformat(),
                   reference_end_utc=end.isoformat(), optical_acquisition_id=optical, campaign_pair_id=campaign_pair,
                   roof_acquisition_pair_id=f"{campaign_pair}|{r['candidate_id']}",
                   exact_maximum_time_separation_hours=separation)


def counts(rows):
    return dict(rows=len(rows), roofs=len({r["candidate_id"] for r in rows}),
                roof_acquisition_pairs=len({r["roof_acquisition_pair_id"] for r in rows}),
                campaign_pairs=len({r["campaign_pair_id"] for r in rows}))


def main():
    rows = list(enrich(read(f"{REGISTRY}_height_comparisons")))
    export(f"{STEM}_rows", rows)
    cohorts = {"all": rows} | {ref: [r for r in rows if r["reference_type"] == ref]
                               for ref in sorted({r["reference_type"] for r in rows})}
    summary = {name: {"all_rows": counts(group)} | {
        f"qualified_within_{hours}h": counts([r for r in group if r["paired_height_difference_m"]
                                              and r["exact_maximum_time_separation_hours"] <= hours])
        for hours in (6, 12, 24, 48)} for name, group in cohorts.items()}
    payload = json.dumps(summary, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
