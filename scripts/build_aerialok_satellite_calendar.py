"""Fit native optical heights over the complete public Aerial Oklahoma flight calendar."""

import json
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from csv import DictReader
from datetime import date, datetime, time, timedelta, timezone
from functools import partial
from itertools import combinations
from zoneinfo import ZoneInfo

from pyproj import Transformer

from audit_cushing_naip_satellite_controls import fit_sentinel2_sources
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, RESOLVED
from build_weekly_shadow_validation import legacy_geometry
from collect_cushing_gedi_optical_windows import kernels
from collect_cushing_landsat7 import PIXELS
from collect_public import ROOT, write
from measure_cushing_landsat7 import CENTRAL_FWHM, response_kernel, scene


STEM = "aerialok_satellite_calendar"
FLIGHTS = ROOT / "data/reference/aerialok_cushing_flights.csv"
ZONE = ZoneInfo("America/Chicago")
GAP_FIELDS = ("estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m")


def dates_and_roofs(geometry):
    with FLIGHTS.open(newline="") as stream:
        dates = {date.fromisoformat(r["flight_date"]) for r in DictReader(stream)}
    metadata = {r["candidate_id"]: r for r in read("tank_engineering_registry")}
    first_seen = {k: date.fromisoformat(metadata[k]["first_seen"]) for k in geometry}
    return dates, first_seen


def sentinel_jobs(geometry, first_seen):
    grouped = defaultdict(dict)
    for r in read("aerialok_sentinel2_flight_matches"):
        roof = r["candidate_id"]
        if roof not in geometry or r["documented_roof_type"] != "EFR" or first_seen[roof] > date.fromisoformat(r["flight_date"]):
            continue
        g = geometry[roof]
        east, north, diameter = (float(g[k]) for k in ("center_east_m", "center_north_m", "diameter_m"))
        key = (r["source_directory"], r["l1c_product_name"], r["captured_at_utc"])
        grouped[key][roof] = dict(east=east, north=north, radius=diameter/2)
    return grouped


def day_offset(stamp, days):
    local = datetime.fromisoformat(stamp).astimezone(ZONE).date()
    for offset in (-1, 0, 1):
        flight = local - timedelta(days=offset)
        if flight in days:
            yield flight, offset


def sensor_record(sensor, orbit, version, stamp, source, r, passed, fields):
    values = {key: r.get(field) for key, field in zip(GAP_FIELDS, fields, strict=True)}
    return dict(satellite_sensor=sensor, satellite_pass=orbit, satellite_version=version, satellite_captured_at_utc=stamp,
                source_path=source, native_sha256=r.get("satellite_source_sha256"), candidate_id=r["candidate_id"],
                diameter_m=r["diameter_m"], satellite_fit_status=r["fit_status"], satellite_passed=passed, **values)


def sentinel_records(geometry, first_seen):
    grouped = sentinel_jobs(geometry, first_seen)
    geographic = Transformer.from_crs(32614, 4326, always_xy=True)
    psf = kernels({version[:3] for _, version, _ in grouped})
    records, sources = [], []
    for roofs in grouped.values():
        for r in roofs.values():
            r["longitude"], r["latitude"] = geographic.transform(r["east"], r["north"])
    work = partial(fit_sentinel2_sources, legacy=geometry, psf=psf)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for number, (fits, source) in enumerate(pool.map(work, sorted(grouped.items())), 1):
            records.extend(sensor_record("Sentinel-2 B04", *fit[1:]) for fit in fits)
            sources.append(source)
            if number % 5 == 0 or number == len(grouped):
                print(f"Sentinel-2 source tiles fitted: {number}/{len(grouped)}", flush=True)
    return records, sources


def landsat7_records(dates, geometry, first_seen):
    grouped = defaultdict(list)
    rejected = []
    for r in read("cushing_landsat7_roof_windows"):
        product, roof, captured = (r[k] for k in ("product_id", "candidate_id", "captured_at_utc"))
        if not product.endswith("_T1") or roof not in geometry:
            continue
        if not any(first_seen[roof] <= flight for flight, _ in day_offset(captured, dates)):
            continue
        if r["angle_valid"] == "True" and int(r["roof_pixels_passing_qa"]) > 0:
            grouped[product].append((r, None))
        else:
            rejected.append(dict(r, fit_status="missing_usable_b8_pixels_or_angles", diameter_m=2*float(r["radius_m"])))
    kernel = {CENTRAL_FWHM: response_kernel(CENTRAL_FWHM)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        measured = [r for fits in pool.map(lambda job: scene(job, geometry, kernel), sorted(grouped.items())) for r in fits]
    for r in (*measured, *rejected):
        product = r["product_id"]
        path = (PIXELS / product / "item.json").relative_to(ROOT)
        yield sensor_record("Landsat 7 T1 B8", product, product, r["captured_at_utc"], str(path), r,
                            r["fit_status"] == RESOLVED and not r.get("center_search_on_boundary"), GAP_FIELDS)


def flight_records(records, dates, first_seen):
    for r in records:
        captured = datetime.fromisoformat(r["satellite_captured_at_utc"])
        roof = r["candidate_id"]
        for flight, offset in day_offset(r["satellite_captured_at_utc"], dates):
            if first_seen[roof] > flight:
                continue
            midnight = [datetime.combine(flight + timedelta(days=i), time(), ZONE).astimezone(timezone.utc) for i in (0, 1)]
            hours = max(abs((captured - end).total_seconds()) for end in midnight)/3600
            yield dict(flight_date=flight.isoformat(), satellite_local_date=captured.astimezone(ZONE).date().isoformat(),
                       satellite_calendar_day_offset=offset, maximum_calendar_day_separation_hours=hours, **r)


def interval_opportunities(rows, dates, geometry):
    flights = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        flight = date.fromisoformat(r["flight_date"])
        key = r["satellite_sensor"], r["satellite_pass"], r["satellite_local_date"]
        roofs = flights[flight][key]
        k = r["candidate_id"]
        previous = roofs.get(k)
        roofs[k] = r["satellite_passed"] or previous is True
    area = {k: float(r["diameter_m"])**2 for k, r in geometry.items()}
    total = sum(area.values())
    for first, last in combinations(sorted(dates), 2):
        days = (last-first).days
        if not 5 <= days <= 9:
            continue
        for (sensor_a, pass_a, date_a), a in sorted(flights[first].items()):
            for (sensor_b, pass_b, date_b), b in sorted(flights[last].items()):
                if date_a >= date_b:
                    continue
                common = a.keys() & b.keys()
                passed = sorted(k for k in common if a[k] and b[k])
                yield dict(first_flight_date=first.isoformat(), last_flight_date=last.isoformat(), days=days,
                           first_satellite_sensor=sensor_a, last_satellite_sensor=sensor_b, first_satellite_pass=pass_a,
                           last_satellite_pass=pass_b, first_satellite_local_date=date_a, last_satellite_local_date=date_b,
                           same_satellite_sensor=sensor_a.split()[:2] == sensor_b.split()[:2],
                           same_processing_route=sensor_a == sensor_b,
                           both_same_calendar_day=date_a == first.isoformat() and date_b == last.isoformat(),
                           roof_candidates=len(common), roofs_passing_both_satellite_fits=len(passed),
                           share_of_dated_registry_roof_area=sum(area[k] for k in passed)/total,
                           passing_candidate_ids_json=json.dumps(passed, separators=(",", ":")))


def summarize(dates, matched, intervals, sources):
    stats = Counter(r["satellite_sensor"] for r in matched)
    return dict(public_flights=len(dates), source_sentinel2_tiles=len(sources), matches_by_sensor=stats,
                passing_rows_by_sensor=Counter(r["satellite_sensor"] for r in matched if r["satellite_passed"]),
                distinct_flights_with_any_passing_satellite=len({r["flight_date"] for r in matched if r["satellite_passed"]}),
                interval_options=len(intervals), with_passing_roofs=sum(r["roofs_passing_both_satellite_fits"] > 0 for r in intervals),
                with_passing_same_day_same_sensor=sum(r["roofs_passing_both_satellite_fits"] > 0 and r["both_same_calendar_day"] and
                                                     r["same_satellite_sensor"] for r in intervals),
                scope="Candidate satellite heights within one calendar day of each aerial flight. Only the dated LiDAR circles are used. "
                      "Passing a satellite fit does not certify airborne coverage or an aerial height. This is a retrospective calendar audit.")


def landsat8_records(stem, sensor):
    for r in read(stem):
        yield sensor_record(sensor, r["product_id"], r["product_id"], r["captured_at_utc"], r["native_packet"],
                           dict(r, satellite_source_sha256=r["native_sha256"]), r["fit_status"] == RESOLVED and r["center_at_boundary"] == "False",
                           ("gap_m", "gap_min_m", "gap_max_m"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-intervals", action="store_true", help="Use already calculated fits and add the available Landsat 8 T1 fits")
    args = parser.parse_args()
    geometry = legacy_geometry()
    dates, first_seen = dates_and_roofs(geometry)
    if args.refresh_intervals:
        sources = read(f"{STEM}_sources")
        matched = [dict(r, satellite_passed=r["satellite_passed"] == "True") for r in read(f"{STEM}_fits")
                   if r["satellite_sensor"] != "Landsat 8 T1 B8"]
    else:
        sentinel, sources = sentinel_records(geometry, first_seen)
        landsat7 = list(landsat7_records(dates, geometry, first_seen))
        landsat8 = list(landsat8_records("cushing_landsat8_rapid_archive_height_fits", "Landsat 8 RT B8"))
        matched = list(flight_records((*sentinel, *landsat7, *landsat8), dates, first_seen))
    if (DERIVED / "aerialok_landsat8_t1_fits.csv").is_file():
        t1 = landsat8_records("aerialok_landsat8_t1_fits", "Landsat 8 T1 B8")
        matched.extend(flight_records(t1, dates, first_seen))
    intervals = sorted(interval_opportunities(matched, dates, geometry), key=lambda r: (-r["roofs_passing_both_satellite_fits"],
                       r["first_flight_date"], r["last_flight_date"], r["first_satellite_pass"], r["last_satellite_pass"]))
    report = summarize(dates, matched, intervals, sources)
    export(f"{STEM}_fits", matched)
    export(f"{STEM}_sources", sources)
    export(f"{STEM}_intervals", intervals)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(report, flush=True)


if __name__ == "__main__":
    main()
