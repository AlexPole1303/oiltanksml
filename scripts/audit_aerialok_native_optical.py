"""Compare dated Aerial Oklahoma plans with same-roof native satellite estimates."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain, combinations

import numpy as np
from pyproj import Transformer

from audit_cushing_landsat8_rapid_height_controls import passed as landsat_passed
from audit_shadow_precision import describe
from build_aerialok_concurrent_controls import CENTRAL, NORTH, SITES as CONCURRENT, daylight_trials, exterior_profile
from build_aerialok_interval_controls import DECEMBER_SITES
from build_aerialok_october_controls import GEOD, interior_model, output, read_plate, register
from build_naip_2025_roof_reference import solar_module
from build_s1_reference_matches import read
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_panel import DERIVED, RESOLVED
from build_weekly_shadow_validation import DATED, legacy_geometry
from collect_public import ROOT, write
from measure_cushing_landsat7 import laser_references


STEM = "aerialok_native_optical"
NEW_SITES = {
    "north": (-96.75485, 36.011, 2630, 660, 3150, 3546, 4050),
    "west": (-96.756, 35.947, 2800, 1320, 3150, 3546, 4050),
}
SITES = DECEMBER_SITES | CONCURRENT | {f"{prefix}_{k}": v for prefix in ("oct23", "nov20", "dec01", "dec08")
                                    for k, v in NEW_SITES.items()}
for prefix in ("sep07", "sep15", "may28", "jun04"):
    SITES.update({f"{prefix}_north": DECEMBER_SITES["dec_north"], f"{prefix}_west": NEW_SITES["west"]})
PLATES = (("2020-09-07", "sep07", ("north", "west")), ("2020-09-15", "sep15", ("north", "west")),
          ("2020-10-23", "oct23", ("west",)), ("2020-11-20", "nov20", ("north", "west")),
          ("2020-12-01", "dec01", ("north", "west")), ("2020-12-04", "dec", ("north", "west", "east")),
          ("2020-12-08", "dec08", ("north", "west")), ("2020-12-10", "dec", ("north", "west", "east")),
          ("2021-05-28", "may28", ("north", "west")), ("2021-06-04", "jun04", ("north", "west")),
          ("2021-06-18", "concurrent", ("north", "west", "east")))


def engineering():
    project = Transformer.from_crs(32614, 4326, always_xy=True)
    by_id = {r["candidate_id"]: r for r in read("tank_engineering_registry")}
    for roof, g in sorted(legacy_geometry().items()):
        record = by_id[roof]
        terminal = record["documented_terminal_id"]
        side = "north" if terminal in NORTH else "east" if terminal == "ngl_cushing" else "west"
        lon, lat = project.transform(float(g["center_east_m"]), float(g["center_north_m"]))
        boundaries = {"enbridge_cushing": 35.948, "enbridge_blueknight": 36.015, "magellan_cushing": 35.936}
        block = terminal
        if terminal in boundaries:
            block += "_north" if lat >= boundaries[terminal] else "_south"
        yield dict(candidate_id=roof, terminal_id=terminal, side=side, shadow_block=block,
                   first_seen=record["first_seen"], roof_type=record["documented_roof_type"], longitude=lon, latitude=lat,
                   diameter_m=float(g["diameter_m"]), geometry_source_directory=g["source_directory"],
                   shell_m=float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"]))


def in_nominal_plate(t, image, lon, lat):
    bearing, _, distance = GEOD.inv(lon, lat, t["longitude"], t["latitude"])
    east, north = distance * math.sin(math.radians(bearing)), distance * math.cos(math.radians(bearing))
    return abs(east) <= image.shape[1]/2+60 and abs(north) <= image.shape[0]/2+60


def plate(job, geometry, spa, *, centre_shift_m=None):
    day, site, side = job
    image, manifest = read_plate(site, day, sites=SITES)
    lon, lat = SITES[site][:2]
    visible = [t for t in geometry if t["side"] == side and t["first_seen"] <= day and in_nominal_plate(t, image, lon, lat)]
    registered = list(register(site, image, visible, sites=SITES, coarse_search_m=200,
                               centre_shift_m=centre_shift_m)) if visible else []
    times, bearings, ratios = daylight_trials(spa, day, lon, lat)
    donors = defaultdict(dict)
    for t in registered:
        x, y = t["plate_x_at_1m"], t["plate_y_at_1m"]
        if (min(x, y, image.shape[1]-x, image.shape[0]-y) < t["diameter_m"]/2+50 or t["rim_edge_contrast"] < 12):
            continue
        v = exterior_profile(image, t, registered, bearings, ratios)
        if v is not None:
            donors[t["shadow_block"]][t["candidate_id"]] = v
    results = []
    for t in registered:
        roof = t["candidate_id"]
        x, y = t["plate_x_at_1m"], t["plate_y_at_1m"]
        full = min(x, y, image.shape[1]-x, image.shape[0]-y) >= t["diameter_m"]/2+1
        if not full or t["rim_edge_contrast"] < 12:
            status = "incomplete_or_weak_rim"
        else:
            status = "insufficient_exterior_donors"
        values = [v for k, v in donors[t["shadow_block"]].items() if k != roof]
        row = dict(t, flight_date=day, site=site, source_path=manifest["source_path"], source_sha256=manifest["sha256"],
                   exterior_other_tanks=len(values), clock_status=status, exterior_best_utc=None,
                   exterior_trial_first_utc=None, exterior_trial_last_utc=None, exterior_near_span_minutes=None,
                   exterior_gain=None, exterior_distinct_peak_margin=None, aerial_fit_status="no_checked_exterior_clock",
                   aerial_gap_m=None, aerial_near_gap_min_m=None, aerial_near_gap_max_m=None,
                   aerial_gap_min_over_clock_profile_m=None, aerial_gap_max_over_clock_profile_m=None,
                   all_near_clock_fits_visible=False, full_roof_inside_plate=full)
        if len(values) >= 3 and status != "incomplete_or_weak_rim":
            gain = np.median(values, axis=0)
            best = int(np.argmax(gain))
            near = np.flatnonzero(gain >= gain[best]-.025)
            margin = gain[best]-max(gain[np.abs(np.arange(len(times))-best) >= 10], default=0)
            ok = gain[best] >= .08 and margin >= .025
            row.update(clock_status="shadow_inferred_clock" if ok else "ambiguous_exterior_clock",
                       exterior_best_utc=times[best].isoformat(), exterior_trial_first_utc=times[near[0]].isoformat(),
                       exterior_trial_last_utc=times[near[-1]].isoformat(),
                       exterior_near_span_minutes=(times[near[-1]]-times[near[0]]).total_seconds()/60,
                       exterior_gain=float(gain[best]), exterior_distinct_peak_margin=float(margin))
            if ok:
                row.update(interior_model(image, t, bearings[best], ratios[best]))
                alternatives = [interior_model(image, t, bearings[i], ratios[i]) for i in near]
                row.update(aerial_gap_min_over_clock_profile_m=min(r["aerial_near_gap_min_m"] for r in alternatives),
                           aerial_gap_max_over_clock_profile_m=max(r["aerial_near_gap_max_m"] for r in alternatives),
                           all_near_clock_fits_visible=all(r["aerial_fit_status"] == "visible_shadow_fit" for r in alternatives))
        results.append(row)
    print(day, site, len(results), Counter(r["aerial_fit_status"] for r in results), flush=True)
    return manifest, results


def local_window(day):
    d = date.fromisoformat(day)
    return tuple(datetime.combine(d+timedelta(days=offset), time(), CENTRAL).astimezone(timezone.utc) for offset in (0, 1))


def max_hours(capture, window):
    return max(abs((capture-endpoint).total_seconds()) for endpoint in window)/3600


def satellite():
    for r in read("shadow_timing_development_observations"):
        if r["geometry_mode"] == DATED:
            yield ("Sentinel-2 B04", r["version_id"], r["captured_at_utc"], r["measurement_source"], r["observation_id"],
                   r["fit_status"] == RESOLVED, r, ("estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m"))
    for source, sensor in (("cushing_landsat8_rapid_archive_height_fits", "Landsat 8 RT B8"),
                           ("aerialok_landsat8_t1_fits", "Landsat 8 T1 B8")):
        for r in read(source):
            yield (sensor, r["product_id"], r["captured_at_utc"], r["native_packet"], None,
                   landsat_passed(r), r, ("gap_m", "gap_min_m", "gap_max_m"))


def calendar_satellite():
    for r in read("aerialok_satellite_calendar_fits"):
        if r["satellite_sensor"] in {"Landsat 8 RT B8", "Landsat 8 T1 B8"}:
            continue
        fields = ("estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m")
        source = r["source_path"]
        sensor = "Sentinel-2 B04 direct" if r["satellite_sensor"] == "Sentinel-2 B04" else r["satellite_sensor"]
        yield (r["flight_date"], sensor, r["satellite_version"], r["satellite_captured_at_utc"], source, None,
               r["satellite_passed"] == "True", dict(r, source_directory=source, fit_status=r["satellite_fit_status"]), fields)


def pair_observations(aerial):
    matches = defaultdict(list)
    for a in aerial:
        day = a["flight_date"]
        matches[a["candidate_id"]].append((a, local_window(day)))
    previous = ((None, *record) for record in satellite())
    for selected_day, sensor, version, stamp, source, obs_id, passed, r, fields in chain(previous, calendar_satellite()):
        roof = r["candidate_id"]
        if roof not in matches:
            continue
        capture = datetime.fromisoformat(stamp)
        for a, window in matches[roof]:
            if selected_day and selected_day != a["flight_date"]:
                continue
            hours = max_hours(capture, window)
            if hours > 72:
                continue
            diameter = a["diameter_m"]
            if not math.isclose(diameter, float(r["diameter_m"]), rel_tol=0, abs_tol=1e-8):
                raise ValueError(f"Satellite and aerial dated diameters differ for {roof}/{version}")
            sat, low, high = (float(r[k]) if r[k] else None for k in fields)
            shadow = a["aerial_gap_m"]
            good = passed and a["aerial_fit_status"] == "visible_shadow_fit"
            inferred = (a["exterior_trial_first_utc"], a["exterior_trial_last_utc"])
            span = max_hours(capture, tuple(map(datetime.fromisoformat, inferred))) if a["clock_status"] == "shadow_inferred_clock" else None
            yield dict(candidate_id=roof, terminal_id=a["terminal_id"], diameter_m=diameter,
                       flight_date=a["flight_date"], site=a["site"],
                       aerial_source=a["source_path"], aerial_sha256=a["source_sha256"], clock_status=a["clock_status"],
                       aerial_clock_min_utc=inferred[0], aerial_clock_max_utc=inferred[1],
                       maximum_calendar_day_separation_hours=hours, maximum_inferred_separation_hours=span,
                       same_local_day=a["flight_date"]==capture.astimezone(CENTRAL).date().isoformat(),
                       satellite_sensor=sensor, satellite_version=version, satellite_captured_at_utc=stamp,
                       satellite_source=source, satellite_observation_id=obs_id, satellite_fit_status=r["fit_status"],
                       satellite_local_source_directory=r.get("source_directory"), satellite_native_sha256=r.get("native_sha256"),
                       satellite_tile_metadata=r.get("tile_metadata_path"),
                       satellite_gap_m=sat, satellite_gap_min_m=low, satellite_gap_max_m=high,
                       satellite_diagnostics_pass=passed, aerial_fit_status=a["aerial_fit_status"], aerial_gap_m=shadow,
                       aerial_gap_min_m=a["aerial_gap_min_over_clock_profile_m"], aerial_gap_max_m=a["aerial_gap_max_over_clock_profile_m"],
                       all_near_clock_fits_visible=a["all_near_clock_fits_visible"], both_pass_at_inferred_best=good,
                       satellite_minus_aerial_gap_m=sat-shadow if good else None)


def change_comparisons(rows):
    grouped = defaultdict(list)
    for r in rows:
        if r["both_pass_at_inferred_best"] and r["maximum_inferred_separation_hours"] <= 30:
            grouped[r["candidate_id"]].append(r)
    for _, group in sorted(grouped.items()):
        for a, b in combinations(sorted(group, key=lambda r: r["flight_date"]), 2):
            days = (date.fromisoformat(b["flight_date"])-date.fromisoformat(a["flight_date"])).days
            if not 0 < days <= 14 or datetime.fromisoformat(a["satellite_captured_at_utc"]) >= datetime.fromisoformat(b["satellite_captured_at_utc"]):
                continue
            satellite = a["satellite_gap_m"]-b["satellite_gap_m"]
            aerial = a["aerial_gap_m"]-b["aerial_gap_m"]
            low = a["satellite_gap_min_m"]-b["satellite_gap_max_m"]-(a["aerial_gap_max_m"]-b["aerial_gap_min_m"])
            high = a["satellite_gap_max_m"]-b["satellite_gap_min_m"]-(a["aerial_gap_min_m"]-b["aerial_gap_max_m"])
            area = math.pi*(a["diameter_m"]/2)**2
            yield dict(candidate_id=a["candidate_id"], terminal_id=a["terminal_id"], first_flight=a["flight_date"],
                       last_flight=b["flight_date"], aerial_interval_days=days, first_satellite_version=a["satellite_version"],
                       last_satellite_version=b["satellite_version"], first_satellite_sensor=a["satellite_sensor"],
                       last_satellite_sensor=b["satellite_sensor"], first_satellite_capture_utc=a["satellite_captured_at_utc"],
                       last_satellite_capture_utc=b["satellite_captured_at_utc"], first_max_hours=a["maximum_inferred_separation_hours"],
                       last_max_hours=b["maximum_inferred_separation_hours"],
                       both_aerial_stable=a["all_near_clock_fits_visible"] and b["all_near_clock_fits_visible"],
                       satellite_rise_m=satellite, aerial_rise_m=aerial, satellite_minus_aerial_rise_m=satellite-aerial,
                       difference_min_m=low, difference_max_m=high, roof_area_m2=area,
                       satellite_cylindrical_change_bbl=satellite*area/BARREL_M3,
                       aerial_cylindrical_change_bbl=aerial*area/BARREL_M3,
                       satellite_minus_aerial_cylindrical_change_bbl=(satellite-aerial)*area/BARREL_M3,
                       both_inferred_separations_within_24h=max(a["maximum_inferred_separation_hours"], b["maximum_inferred_separation_hours"]) <= 24,
                       both_calendar_separations_within_48h=max(a["maximum_calendar_day_separation_hours"], b["maximum_calendar_day_separation_hours"]) <= 48)


def laser_comparisons(aerial):
    geometry = legacy_geometry()
    by_roof = defaultdict(list)
    for a in aerial:
        by_roof[a["candidate_id"]].append(a)
    for roof, sensor, identity, first, last, gap in laser_references(geometry, by_roof):
        stamps = tuple(map(datetime.fromisoformat, (first, last)))
        for a in by_roof[roof]:
            hours = max(max_hours(stamp, local_window(a["flight_date"])) for stamp in stamps)
            if hours > 48:
                continue
            start, end = a["exterior_trial_first_utc"], a["exterior_trial_last_utc"]
            window = tuple(map(datetime.fromisoformat, (start, end))) if a["clock_status"] == "shadow_inferred_clock" else None
            inferred_hours = max(max_hours(stamp, window) for stamp in stamps) if window else None
            accepted = a["aerial_fit_status"] == "visible_shadow_fit"
            yield dict(candidate_id=roof, terminal_id=a["terminal_id"], flight_date=a["flight_date"], site=a["site"],
                       aerial_source=a["source_path"], aerial_sha256=a["source_sha256"], laser_sensor=sensor,
                       laser_source=identity, laser_first_utc=first, laser_last_utc=last, laser_gap_m=gap,
                       maximum_calendar_day_separation_hours=hours, maximum_inferred_separation_hours=inferred_hours,
                       clock_status=a["clock_status"], aerial_fit_status=a["aerial_fit_status"], aerial_gap_m=a["aerial_gap_m"],
                       all_near_clock_fits_visible=a["all_near_clock_fits_visible"],
                       aerial_minus_laser_m=a["aerial_gap_m"]-gap if accepted else None)


def change_summary(changes, geometry):
    groups = defaultdict(list)
    for r in changes:
        key = tuple(r[k] for k in ("first_flight", "last_flight", "first_satellite_sensor", "last_satellite_sensor"))
        groups[key].append(r)
    registry_area = sum(math.pi*(r["diameter_m"]/2)**2 for r in geometry)
    out = []
    for (first, last, a, b), group in sorted(groups.items()):
        for stable in (False, True):
            rows = [r for r in group if not stable or r["both_aerial_stable"]]
            if not rows:
                continue
            out.append(dict(first_flight=first, last_flight=last, first_satellite_sensor=a, last_satellite_sensor=b,
                            only_stable_aerial_fits=stable, roofs=len(rows),
                            roof_area_share_of_legacy_registry=sum(r["roof_area_m2"] for r in rows)/registry_area,
                            both_nonzero_same_sign=sum(r["aerial_rise_m"]*r["satellite_rise_m"] > 0 for r in rows),
                            opposite_sign=sum(r["aerial_rise_m"]*r["satellite_rise_m"] < 0 for r in rows),
                            at_least_one_zero=sum(r["aerial_rise_m"]*r["satellite_rise_m"] == 0 for r in rows),
                            difference_range_excludes_zero=sum(r["difference_min_m"] > 0 or r["difference_max_m"] < 0 for r in rows),
                            summed_satellite_cylindrical_change_bbl=sum(r["satellite_cylindrical_change_bbl"] for r in rows),
                            summed_aerial_cylindrical_change_bbl=sum(r["aerial_cylindrical_change_bbl"] for r in rows),
                            **describe([r["satellite_minus_aerial_rise_m"] for r in rows])))
    return out


def same_week_endpoint_sensitivity(rows):
    first_day, last_day = "2020-12-01", "2020-12-08"
    b04, b8 = "Sentinel-2 B04", "Landsat 8 RT B8"
    available = {(r["candidate_id"], r["flight_date"], r["satellite_sensor"]): r for r in rows if r["both_pass_at_inferred_best"]}
    for roof, day, sensor in sorted(available):
        if (day, sensor) != (first_day, b04):
            continue
        a = available[roof, day, sensor]
        b, alternative = (available.get((roof, last_day, kind)) for kind in (b8, b04))
        if b is None or alternative is None or max(a["maximum_inferred_separation_hours"], b["maximum_inferred_separation_hours"]) > 24:
            continue
        if alternative["aerial_sha256"] != b["aerial_sha256"] or alternative["aerial_gap_m"] != b["aerial_gap_m"]:
            raise ValueError(f"Alternate week endpoint uses a different aerial measurement for {roof}")
        aerial = a["aerial_gap_m"]-b["aerial_gap_m"]
        mixed = a["satellite_gap_m"]-b["satellite_gap_m"]
        same_sensor = a["satellite_gap_m"]-alternative["satellite_gap_m"]
        yield dict(candidate_id=roof, terminal_id=a["terminal_id"], first_aerial_flight=first_day, last_aerial_flight=last_day,
                   first_satellite_version=a["satellite_version"], mixed_last_satellite_version=b["satellite_version"],
                   same_sensor_last_satellite_version=alternative["satellite_version"],
                   first_capture_utc=a["satellite_captured_at_utc"], mixed_last_capture_utc=b["satellite_captured_at_utc"],
                   same_sensor_last_capture_utc=alternative["satellite_captured_at_utc"],
                   mixed_last_maximum_inferred_separation_hours=b["maximum_inferred_separation_hours"],
                   same_sensor_last_maximum_inferred_separation_hours=alternative["maximum_inferred_separation_hours"],
                   aerial_rise_m=aerial, mixed_satellite_rise_m=mixed, same_sensor_rise_m=same_sensor,
                   mixed_minus_aerial_rise_m=mixed-aerial, same_sensor_minus_aerial_rise_m=same_sensor-aerial)


def same_week_first_endpoint(changes):
    flights = ("2020-12-01", "2020-12-08")
    anchor = "Landsat 8 T1 B8"
    alternatives = ("Landsat 7 T1 B8", "Sentinel-2 B04")
    index = {(r["candidate_id"], r["first_satellite_sensor"]): r for r in changes
             if (r["first_flight"], r["last_flight"]) == flights and r["last_satellite_sensor"] == anchor}
    shared = {k for k, sensor in index if sensor == alternatives[0]} & {k for k, sensor in index if sensor == alternatives[1]}
    for roof in sorted(shared):
        a, b = (index[roof, sensor] for sensor in alternatives)
        if any(a[k] != b[k] for k in ("aerial_rise_m", "last_satellite_version", "last_satellite_capture_utc")):
            raise ValueError(f"Inconsistent common week comparison for {roof}")
        yield dict(candidate_id=roof, terminal_id=a["terminal_id"], first_flight=flights[0], last_flight=flights[1],
                   first_landsat7_version=a["first_satellite_version"], first_sentinel2_version=b["first_satellite_version"],
                   last_landsat8_version=a["last_satellite_version"], first_landsat7_capture_utc=a["first_satellite_capture_utc"],
                   first_sentinel2_capture_utc=b["first_satellite_capture_utc"], last_landsat8_capture_utc=a["last_satellite_capture_utc"],
                   first_capture_separation_hours=abs((datetime.fromisoformat(a["first_satellite_capture_utc"])-
                                                       datetime.fromisoformat(b["first_satellite_capture_utc"])).total_seconds())/3600,
                   aerial_rise_m=a["aerial_rise_m"], landsat7_landsat8_rise_m=a["satellite_rise_m"],
                   sentinel2_landsat8_rise_m=b["satellite_rise_m"],
                   landsat7_landsat8_minus_aerial_m=a["satellite_minus_aerial_rise_m"],
                   sentinel2_landsat8_minus_aerial_m=b["satellite_minus_aerial_rise_m"])


def measure_plates():
    geometry, spa = list(engineering()), solar_module()
    jobs = [(day, f"{prefix}_{side}", side) for day, prefix, sides in PLATES for side in sides]
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(lambda job: plate(job, geometry, spa), jobs))
    sources = [m for m, _ in result]
    aerial = [row for _, group in result for row in group]
    output(f"{STEM}_sheets", sources)
    fields = dict.fromkeys(k for r in aerial for k in r)
    output(f"{STEM}_aerial", [{k: r.get(k) for k in fields} for r in aerial])
    return sources, aerial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-only", action="store_true")
    parser.add_argument("--reuse-aerial", action="store_true")
    args = parser.parse_args()
    if args.measure_only and args.reuse_aerial:
        parser.error("Choose either --measure-only or --reuse-aerial")
    if args.reuse_aerial:
        sources = read(f"{STEM}_sheets")
        for row in sources:
            with (ROOT / row["source_path"]).open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                    raise ValueError(f"Aerial source was replaced: {row['source_path']}")
        aerial = [dict(r, diameter_m=float(r["diameter_m"]), all_near_clock_fits_visible=r["all_near_clock_fits_visible"] == "True",
                       **{k: float(r[k]) if r[k] else None for k in (
                           "aerial_gap_m", "aerial_gap_min_over_clock_profile_m", "aerial_gap_max_over_clock_profile_m")})
                  for r in read(f"{STEM}_aerial")]
    else:
        sources, aerial = measure_plates()
    if args.measure_only:
        return
    geometry = list(engineering())
    pairs = sorted(pair_observations(aerial), key=lambda r: (
        r["maximum_calendar_day_separation_hours"], r["satellite_version"], r["satellite_source"]))
    nearest = {}
    for r in pairs:
        key = tuple(r[k] for k in ("candidate_id", "flight_date", "satellite_sensor"))
        r["nearest_by_calendar_before_quality"] = key not in nearest
        nearest.setdefault(key, r)
    selected = list(nearest.values())
    changes = list(change_comparisons(selected))
    change_groups = change_summary(changes, geometry)
    alternatives = list(same_week_endpoint_sensitivity(selected))
    first_alternatives = list(same_week_first_endpoint(changes))
    laser = list(laser_comparisons(aerial))
    accepted = [r for r in selected if r["both_pass_at_inferred_best"]]
    by_group = defaultdict(list)
    for r in selected:
        by_group[r["flight_date"], r["satellite_sensor"]].append(r)
    report = dict(sheets=len(sources), aerial_rows=len(aerial), satellite_pairs=len(pairs), nearest_pairs=len(selected),
                  comparison_by_flight_and_sensor={f"{day}/{sensor}": dict(
                      candidates=len(g), same_day=sum(r["same_local_day"] for r in g),
                      satellite_pass=sum(r["satellite_diagnostics_pass"] for r in g),
                      both_pass=describe([r["satellite_minus_aerial_gap_m"] for r in g if r["both_pass_at_inferred_best"]]),
                      both_pass_same_day=sum(r["both_pass_at_inferred_best"] and r["same_local_day"] for r in g),
                      both_pass_within_3h=sum(r["both_pass_at_inferred_best"] and r["maximum_inferred_separation_hours"] <= 3 for r in g),
                      both_pass_stable_clock=sum(r["both_pass_at_inferred_best"] and r["all_near_clock_fits_visible"] for r in g))
                      for (day, sensor), g in sorted(by_group.items())},
                  joint_fit_pairs=len(accepted), joint_fit_stable_clock=sum(r["all_near_clock_fits_visible"] for r in accepted),
                  laser_comparisons=len(laser), laser_both_pass=sum(r["aerial_minus_laser_m"] is not None for r in laser),
                  change_pairs=len(changes), change_pairs_with_stable_aerial_at_both_ends=sum(r["both_aerial_stable"] for r in changes),
                  changes_with_both_inferred_separations_within_24h=sum(r["both_inferred_separations_within_24h"] for r in changes),
                  change_comparison_by_flights=change_groups,
                  week_alternative_endpoint={k: describe([r[k] for r in alternatives])
                      for k in ("mixed_minus_aerial_rise_m", "same_sensor_minus_aerial_rise_m")},
                  week_alternative_first_endpoint={k: describe([r[k] for r in first_alternatives])
                      for k in ("landsat7_landsat8_minus_aerial_m", "sentinel2_landsat8_minus_aerial_m")},
                  scope="Retrospective checks using the 2011 LiDAR circles, separate from forecast and trading tests. "
                  "Aerial clocks are inferred from other tanks in the same block. The calendar-day bound is the actual metadata bound. "
                  "Both sensors use shadows and may share geometry errors. Difference ranges are sensitivity ranges.")
    output(f"{STEM}_pairs", pairs)
    if changes:
        output(f"{STEM}_changes", changes)
    if alternatives:
        output(f"{STEM}_week_alternate_endpoint", alternatives)
    if first_alternatives:
        output(f"{STEM}_week_first_sensor", first_alternatives)
    if laser:
        output(f"{STEM}_laser", laser)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(len(pairs), "satellite comparisons;", len(accepted), "accepted;", len(changes), "changes", flush=True)


if __name__ == "__main__":
    main()
