"""Build same-calendar-day aerial and satellite controls from the public Cushing archive."""
import datetime as dt
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from itertools import combinations
from zoneinfo import ZoneInfo

import numpy as np

from audit_shadow_precision import describe
from build_aerialok_interval_controls import DECEMBER_SITES
from build_aerialok_october_controls import (DERIVED, direction, interior_model, output, read_plate, register, sample,
                                             satellite_measurements, scores)
from build_naip_2025_roof_reference import solar_module
from build_shadow_change_controls import RESOLVED, scale
from build_weekly_shadow_development_check import read_csv
from collect_public import write
from fit_shadow_geometry import read_only

STEM = "aerialok_concurrent"
DAYS = ("2021-05-14", "2021-06-18", "2021-07-23")
SITES = {f"concurrent_{site.removeprefix('dec_')}": location for site, location in DECEMBER_SITES.items()}
SITES["concurrent_north"] = (*SITES["concurrent_north"][:3], 1320, 3150, 3546, 4050)
NORTH = {"enbridge_blueknight", "semgroup_cushing"}
CENTRAL = ZoneInfo("America/Chicago")


def matched_products():
    products = {}
    for row in read_csv("aerialok_sentinel2_flight_matches"):
        if (row["flight_date"] not in DAYS or row["satellite_calendar_day_offset"] != "0"
                or row["complete_clear_b04_geometry"] != "True"):
            continue
        key = row["flight_date"], row["candidate_id"], row["l1c_product_name"]
        if key in products:
            raise ValueError(f"Duplicate candidate or exact L1C product for {key}")
        products[key] = row
    return products


def tank_registry(source, ids):
    rows = source.execute(f"""SELECT candidate_id,documented_terminal_id AS terminal_id,cohort_id,longitude,latitude,
        lidar_upper_outline_diameter_m AS diameter_m,lidar_rim_to_ground_native_z AS shell_m,documented_roof_type,
        first_seen FROM tank_engineering_registry
        WHERE candidate_id IN ({','.join('?' for _ in ids)}) ORDER BY candidate_id""", sorted(ids))
    tanks = {}
    for row in rows:
        terminal = row["terminal_id"]
        site = "north" if terminal in NORTH else "east" if terminal == "ngl_cushing" else "west"
        if not row["diameter_m"] or not row["shell_m"]:
            continue
        block = terminal
        if terminal in ("enbridge_cushing", "enbridge_blueknight", "magellan_cushing"):
            boundary = {"enbridge_cushing": 35.948, "enbridge_blueknight": 36.015, "magellan_cushing": 35.936}
            block += "_north" if row["latitude"] >= boundary[terminal] else "_south"
        tanks[row["candidate_id"]] = dict(row, site=f"concurrent_{site}", shadow_block=block)
    return tanks


def daylight_trials(spa, day, lon, lat):
    date = dt.date.fromisoformat(day)
    midnight = dt.datetime.combine(date, dt.time(), CENTRAL)
    times = np.array([(midnight + dt.timedelta(minutes=m)).timestamp() for m in range(0, 1440, 3)])
    delta_t = spa.calculate_deltat(date.year, date.month)
    _, zenith, _, elevation, sun_azimuth, _ = spa.solar_position(times, lat, lon, 300, 0, 15, delta_t, 0.5667)
    keep = elevation >= 20
    times = [dt.datetime.fromtimestamp(t, dt.timezone.utc) for t in times[keep]]
    return times, (sun_azimuth[keep] + 180) % 360, np.tan(np.deg2rad(zenith[keep]))


def exterior_profile(image, tank, group, bearings, ratios):
    radius, shell = tank["diameter_m"] / 2, tank["shell_m"]
    east, south = tank["plate_x_at_1m"], tank["plate_y_at_1m"]
    reach = min(58, math.ceil(shell * float(max(ratios)) + 4))
    axis = np.arange(-radius-reach, radius+reach+1, 2)
    x, y = (a.ravel() for a in np.meshgrid(axis, axis))
    radial = np.hypot(x, y)
    keep = (radial > radius+3) & (radial < radius+reach)
    for other in group:
        if other["candidate_id"] != tank["candidate_id"]:
            keep &= np.hypot(east+x-other["plate_x_at_1m"], south+y-other["plate_y_at_1m"]) > other["diameter_m"]/2+3
    gray = sample(image, east+x, south+y)
    keep &= np.isfinite(gray)
    if keep.sum() < 160:
        return None
    x, y, gray = x[keep], y[keep], gray[keep]
    dx, dy = (a[:,None] for a in direction(bearings))
    travel = np.clip(dx*x + dy*y, 0, shell*ratios[:,None])
    shadow = (x-dx*travel)**2 + (y-dy*travel)**2 <= radius**2
    return scores(gray, shadow)[0]


def roof_measurements(tanks, products):
    spa = solar_module()
    manifests, rows = [], []
    available = {(d,t) for d,t,_ in products}
    for site, location in SITES.items():
        for day in DAYS:
            image, manifest = read_plate(site, day, sites=SITES)
            manifests.append(manifest)
            selected = [t for t in tanks.values() if t["site"] == site and t["first_seen"] <= day]
            if not selected:
                continue
            fitted = list(register(site, image, selected, sites=SITES, coarse_search_m=560))
            times, bearings, ratios = daylight_trials(spa, day, *location[:2])
            donors = defaultdict(dict)
            for t in fitted:
                x, y = t["plate_x_at_1m"], t["plate_y_at_1m"]
                if min(x, y, image.shape[1]-x, image.shape[0]-y) < t["diameter_m"]/2+3:
                    continue
                if t["rim_edge_contrast"] >= 12 and (profile := exterior_profile(image, t, fitted, bearings, ratios)) is not None:
                    donors[t["shadow_block"]][t["candidate_id"]] = profile
            for t in fitted:
                tank = t["candidate_id"]
                if (day,tank) not in available:
                    continue
                series = [value for identity,value in donors[t["shadow_block"]].items() if identity != tank]
                row = dict(t, flight_date=day, source_path=manifest["source_path"], source_sha256=manifest["sha256"],
                           exterior_other_tanks=len(series), clock_status="insufficient_exterior_donors",
                           exterior_best_utc=None, exterior_trial_first_utc=None, exterior_trial_last_utc=None,
                           exterior_near_span_minutes=None, exterior_gain=None, exterior_distinct_peak_margin=None,
                           exterior_shadow_bearing_deg=None, exterior_horizontal_per_vertical=None,
                           aerial_gap_m=None, aerial_fit_status="no_checked_exterior_clock",
                           aerial_shadow_explained_fraction=None, aerial_lit_minus_shadow_gray=None,
                           aerial_near_gap_min_m=None, aerial_near_gap_max_m=None, full_roof_inside_plate=None)
                if len(series) >= 3:
                    gain = np.median(series, axis=0)
                    best = int(np.argmax(gain))
                    near = np.flatnonzero(gain >= gain[best] - .025)
                    margin = gain[best] - max(gain[abs(np.arange(len(times))-best) >= 10], default=0)
                    good = gain[best] >= .08 and margin >= .025
                    row.update(clock_status="shadow_inferred_clock" if good else "ambiguous_exterior_clock",
                               exterior_best_utc=times[best].isoformat(),
                               exterior_trial_first_utc=times[near[0]].isoformat(),
                               exterior_trial_last_utc=times[near[-1]].isoformat(),
                               exterior_near_span_minutes=(times[near[-1]]-times[near[0]]).total_seconds()/60,
                               exterior_gain=float(gain[best]), exterior_distinct_peak_margin=float(margin),
                               exterior_shadow_bearing_deg=float(bearings[best]), exterior_horizontal_per_vertical=float(ratios[best]))
                    if good:
                        row.update(interior_model(image, t, bearings[best], ratios[best]))
                rows.append(row)
            print(site, day, len(fitted), "registered, donor blocks", len(donors), flush=True)
    return manifests, rows


def satellite_comparisons(aerial, optical, products):
    for a in aerial:
        day, tank = a["flight_date"], a["candidate_id"]
        s = optical.get((tank, day))
        if s is None:
            continue
        version = s["version_id"]
        p = products[day,tank,version]
        capture = dt.datetime.fromisoformat(s["optical_capture_at_utc"])
        if (capture != dt.datetime.fromisoformat(p["captured_at_utc"])
                or capture.astimezone(CENTRAL).date().isoformat() != day
                or s["roof_type"] != a["documented_roof_type"]
                or not math.isclose(s["diameter_m"], a["diameter_m"], rel_tol=0, abs_tol=1e-8)):
            raise ValueError(f"Optical capture time disagrees with the archived product for {day}/{tank}")
        checked = a["clock_status"] == "shadow_inferred_clock"
        both = a["aerial_fit_status"] == "visible_shadow_fit" and s["fit_status"] == RESOLVED
        gap = (s["estimated_gap_m"] - a["aerial_gap_m"]) if both else None
        trial = (capture-dt.datetime.fromisoformat(a["exterior_best_utc"])).total_seconds()/3600 if checked else None
        yield dict(flight_date=day, candidate_id=tank, site=a["site"], terminal_id=a["terminal_id"],
                   roof_type=a["documented_roof_type"], source_path=a["source_path"],
                   exact_aerial_timestamp_known=False, aerial_inferred_capture_utc=a["exterior_best_utc"] if checked else None,
                   aerial_inferred_near_first_utc=a["exterior_trial_first_utc"] if checked else None,
                   aerial_inferred_near_last_utc=a["exterior_trial_last_utc"] if checked else None,
                   aerial_clock_status=a["clock_status"], satellite_version=version, satellite_capture_utc=capture.isoformat(),
                   maximum_separation_from_calendar_day_hours=float(p["maximum_possible_separation_hours"]),
                   satellite_minus_inferred_aerial_hours=trial, aerial_status=a["aerial_fit_status"], optical_status=s["fit_status"],
                   aerial_gap_m=a["aerial_gap_m"], satellite_gap_m=s["estimated_gap_m"], both_geometry_diagnostics=both,
                   satellite_minus_aerial_gap_m=gap, satellite_minus_aerial_gap_equivalent_bbl=gap*scale(s) if both else None)


def level_changes(comparisons):
    by_tank = defaultdict(list)
    for r in comparisons:
        if r["roof_type"] == "EFR" and r["both_geometry_diagnostics"]:
            by_tank[r["candidate_id"]].append(r)
    for tank, group in sorted(by_tank.items()):
        for a,b in combinations(sorted(group,key=lambda r:r["flight_date"]),2):
            satellite = a["satellite_gap_m"]-b["satellite_gap_m"]
            aerial = a["aerial_gap_m"]-b["aerial_gap_m"]
            yield dict(candidate_id=tank, terminal_id=a["terminal_id"], first_day=a["flight_date"], last_day=b["flight_date"],
                       first_satellite_version=a["satellite_version"], last_satellite_version=b["satellite_version"],
                       first_inferred_lag_hours=a["satellite_minus_inferred_aerial_hours"],
                       last_inferred_lag_hours=b["satellite_minus_inferred_aerial_hours"],
                       aerial_level_change_m=aerial, satellite_level_change_m=satellite, satellite_minus_aerial_change_m=satellite-aerial)


def main():
    products = matched_products()
    with closing(read_only(DERIVED/"cushing.sqlite")) as source:
        source.row_factory = sqlite3.Row
        tanks = tank_registry(source, {t for _,t,_ in products})
        manifests, aerial = roof_measurements(tanks, products)
        optical = satellite_measurements(source, list(tanks.values()), days=DAYS,
                                         database=DERIVED/f"{STEM}_satellite.sqlite", workers=6)
    pairs = list(satellite_comparisons(aerial, optical, products))
    changes = list(level_changes(pairs))
    accepted = [r for r in pairs if r["both_geometry_diagnostics"] and r["roof_type"] == "EFR"]
    near = [r for r in accepted if max(abs((dt.datetime.fromisoformat(r["satellite_capture_utc"])
            - dt.datetime.fromisoformat(r[field])).total_seconds()) for field in (
                "aerial_inferred_near_first_utc", "aerial_inferred_near_last_utc")) <= 7200]
    fixed = [r for r in pairs if r["roof_type"] != "EFR"]
    by_day = []
    for day in DAYS:
        p = [r for r in pairs if r["flight_date"] == day]
        by_day.append(dict(flight_date=day, aerial_satellite_pairs=len(p),
                           aerial_clock_status_counts=dict(Counter(r["aerial_clock_status"] for r in p)),
                           visible_aerial_shadow=sum(r["aerial_status"] == "visible_shadow_fit" for r in p),
                           resolved_satellite_shadow=sum(r["optical_status"] == RESOLVED for r in p),
                           both_efr_diagnostics=sum(r["both_geometry_diagnostics"] and r["roof_type"] == "EFR" for r in p)))
    summary = dict(source_sheets=len(manifests), flight_dates=list(DAYS), distinct_candidate_tanks=len(tanks),
                   aerial_observations=len(aerial), exact_same_day_optical_pairs=len(pairs),
                   efr_pairs_with_both_diagnostics=len(accepted), efr_tanks_with_both_diagnostics=len({r["candidate_id"] for r in accepted}),
                   diagnostics_by_day=by_day,
                   entire_inferred_time_range_within_two_hours=len(near), covered_roof_controls=len(fixed),
                   covered_roofs_with_visible_aerial_shadow_fit=sum(r["aerial_status"]=="visible_shadow_fit" for r in fixed),
                   covered_roofs_with_checked_exterior_clock=sum(r["aerial_clock_status"] == "shadow_inferred_clock" for r in fixed),
                   same_day_gap_disagreement=describe([r["satellite_minus_aerial_gap_m"] for r in accepted]),
                   shortest_absolute_nominal_lag_hours=min(
                       (abs(r["satellite_minus_inferred_aerial_hours"]) for r in accepted), default=None),
                   accepted_between_day_change_pairs=len(changes),
                   between_day_change_disagreement=describe([r["satellite_minus_aerial_change_m"] for r in changes]),
                   interpretation="Aerial view time inferred independently of the target roof using exterior shadows of other tanks. "
                                  "Solar trials and agreement diagnostics do not provide an exposure clock or ground-truth roof height.")
    for suffix, rows in (("export_manifest", manifests), ("aerial_observations", aerial),
                         ("satellite_comparisons", pairs), ("level_changes", changes)):
        if rows:
            output(f"{STEM}_{suffix}", rows)
    write(DERIVED/f"{STEM}_summary.json", json.dumps(summary, indent=2).encode()+b"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
