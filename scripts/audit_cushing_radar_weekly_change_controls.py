"""Compare short-interval roof changes across radar and optical acquisitions."""

import json
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from math import isclose, pi

from audit_shadow_precision import describe
from build_cushing_icesat2_controls import identities
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import write

STEM = "cushing_radar_weekly_change_controls"
INPUT = "radar_optical_cushing_control_48h_comparisons"
MODES = ("local", "scene")
BARREL_M3 = 0.158987294928
MIN_DAYS, MAX_DAYS = 5, 9


def capture(row, field):
    return datetime.fromisoformat(row[field])


def interval_options(rows):
    by_tank = defaultdict(list)
    for r in rows:
        if (r["roof_type"], r["selected_granule_for_metrics"], r["radar_sensor"]) == ("EFR", "True", "Umbra"):
            by_tank[r["candidate_id"]].append(r)
    for tank, series in sorted(by_tank.items()):
        ordered = sorted(series,key=lambda r:(capture(r,"radar_start_utc"),r["radar_product_id"],r["optical_datatake"]))
        for a,b in combinations(ordered,2):
            start, stop = (capture(r,"radar_start_utc") for r in (a,b))
            days = (stop-start).total_seconds()/86400
            if not MIN_DAYS <= days <= MAX_DAYS or a["optical_datatake"]==b["optical_datatake"]:
                continue
            first,last = (capture(r,"optical_native_granule_start_utc") for r in (a,b))
            if first >= last:
                continue
            if a["terminal_id"]!=b["terminal_id"] or not isclose(float(a["shared_lidar_rim_to_ground_m"]),
                                                                 float(b["shared_lidar_rim_to_ground_m"]), abs_tol=1e-8):
                raise ValueError(f"Inconsistent identity or LiDAR geometry for {tank}")
            yield (tank,start.date().isoformat(),stop.date().isoformat()), a, b, days, (last-first).total_seconds()/86400


def selection_rank(option):
    _, a, b, _, _ = option
    separations = [float(r["maximum_separation_seconds"]) for r in (a,b)]
    return (sum(r["optical_source_quality_pass"]!="True" for r in (a,b)), max(separations), sum(separations),
            a["radar_product_id"],b["radar_product_id"],a["optical_datatake"],b["optical_datatake"])


def measurement(option, tank, chosen, chosen_matching_interval):
    (key,first,last), a, b, radar_days, optical_days = option
    radius = tank["radius_m"]
    factor = pi*radius**2/BARREL_M3
    optical = [float(r["optical_height_m"]) if r["optical_height_m"] else None for r in (a,b)]
    result = dict(candidate_id=key, terminal_id=a["terminal_id"], radar_first_day=first, radar_last_day=last,
                  radar_interval_days=radar_days, optical_interval_days=optical_days,
                  maximum_separation_hours=max(float(r["maximum_separation_seconds"]) for r in (a,b))/3600,
                  optical_both_source_quality_pass=all(r["optical_source_quality_pass"]=="True" for r in (a,b)),
                  chosen_by_source_quality_and_time=chosen, chosen_with_matching_sensor_interval=chosen_matching_interval,
                  radius_m=radius, cylindrical_barrels_per_m=factor,
                  optical_change_m=optical[1]-optical[0] if None not in optical else None,
                  within_original_cutoffs=all(r["both_copies_available_before_original_cutoff"]=="True" for r in (a,b)))
    fields = ("radar_product_id","radar_start_utc","radar_end_utc","radar_native_window","radar_sicd_copy_time_utc",
              "radar_capture_reporting_week","original_release_cutoff_at_utc","optical_archive_available_at_utc",
              "optical_version","optical_datatake","optical_native_granule_start_utc","optical_observation_id",
              "optical_native_metadata","optical_source_directory","maximum_separation_seconds","optical_fit_status")
    for side,r in zip(("first","last"),(a,b),strict=True):
        result.update({f"{side}_{field}":r[field] for field in fields})
    for mode in MODES:
        passed = all(r[f"both_{mode}_optical_resolved"]=="True" for r in (a,b))
        difference = float(b[f"radar_{mode}_height_m"])-float(a[f"radar_{mode}_height_m"]) if passed else None
        opt = result["optical_change_m"]
        result.update({f"{mode}_first_radar_fit_status":a[f"radar_{mode}_fit_status"],
                       f"{mode}_last_radar_fit_status":b[f"radar_{mode}_fit_status"],
                       f"{mode}_four_fits_pass":passed,f"{mode}_radar_change_m":difference,
                       f"{mode}_radar_change_cylindrical_bbl":difference*factor if passed else None,
                       f"{mode}_radar_minus_optical_change_m":difference-opt if passed else None,
                       f"{mode}_radar_minus_optical_cylindrical_bbl":(difference-opt)*factor if passed else None})
    return result


def pair_metrics(rows, mode, *, aggregate=False):
    passed = [r for r in rows if r[f"{mode}_four_fits_pass"]]
    errors = [r[f"{mode}_radar_minus_optical_change_m"] for r in passed]
    bbl = [r[f"{mode}_radar_minus_optical_cylindrical_bbl"] for r in passed]
    result = dict(candidates=len(rows), shared_qualified_roofs=len(passed),
                qualified_under_24h=sum(r["maximum_separation_hours"]<=24 for r in passed),
                qualified_under_12h=sum(r["maximum_separation_hours"]<=12 for r in passed),
                qualified_under_3h=sum(r["maximum_separation_hours"]<=3 for r in passed),
                qualified_with_optical_interval_5_to_9_days=sum(MIN_DAYS<=r["optical_interval_days"]<=MAX_DAYS for r in passed),
                maximum_time_gap_hours=max((r["maximum_separation_hours"] for r in passed),default=None),
                difference=describe(errors),
                distinct_optical_acquisition_pairs=len({(r["first_optical_datatake"],r["last_optical_datatake"]) for r in passed}),
                distinct_radar_product_pairs=len({(r["first_radar_product_id"],r["last_radar_product_id"]) for r in passed}))
    if aggregate and len({(r["first_optical_datatake"],r["last_optical_datatake"],r["first_radar_product_id"],r["last_radar_product_id"])
                          for r in passed})<=1:
        result.update(optical_cylindrical_change_bbl=sum(r["optical_change_m"]*r["cylindrical_barrels_per_m"] for r in passed) if bbl else None,
                      radar_cylindrical_change_bbl=sum(r[f"{mode}_radar_change_cylindrical_bbl"] for r in passed) if bbl else None,
                      aggregate_cylindrical_difference_bbl=sum(bbl) if bbl else None)
    return result


def main():
    tanks,_,_ = identities()
    groups = defaultdict(list)
    for option in interval_options(read(INPUT)):
        groups[option[0]].append(option)
    rows = []
    for key,options in sorted(groups.items()):
        ranked = sorted(options,key=selection_rank)
        matched = next((o for o in ranked if MIN_DAYS<=o[4]<=MAX_DAYS),None)
        rows.extend(measurement(o,tanks[key[0]],i==0,o is matched) for i,o in enumerate(ranked))
    export(f"{STEM}_options",rows)
    selected = [r for r in rows if r["chosen_by_source_quality_and_time"]]
    matched = [r for r in rows if r["chosen_with_matching_sensor_interval"]]
    export(STEM,selected)
    export(f"{STEM}_matched_intervals",matched)
    by_date = defaultdict(list)
    for r in selected:
        by_date[r["radar_first_day"],r["radar_last_day"]].append(r)
    summary = dict(radar_day_interval=[MIN_DAYS,MAX_DAYS],exact_options=len(rows),selected_roof_date_pairs=len(selected),
        distinct_roofs=len({r["candidate_id"] for r in selected}),
        joint_scene_and_local_qualified=sum(all(r[f"{m}_four_fits_pass"] for m in MODES) for r in selected),
        qualified_with_both_versions_before_historical_cutoffs=sum(r["within_original_cutoffs"] and any(
            r[f"{mode}_four_fits_pass"] for mode in MODES) for r in selected),
        results={mode:pair_metrics(selected,mode) for mode in MODES},
        results_with_both_sensor_intervals_5_to_9_days={mode:pair_metrics(
            [r for r in selected if MIN_DAYS<=r["optical_interval_days"]<=MAX_DAYS],mode) for mode in MODES},
        results_selecting_only_matching_sensor_intervals={mode:pair_metrics(matched,mode) for mode in MODES},
        by_radar_date_pair={" / ".join(key):{mode:pair_metrics(group,mode,aggregate=True) for mode in MODES}
                            for key,group in sorted(by_date.items())},
        scope="Selection uses native optical input quality and capture proximity without fitted heights. Both sensor series use "
              "the same static LiDAR roof geometry. Differences include timing and both measurement errors. Adjacent date pairs "
              "can reuse radar or optical dates. No EIA label, oil volume ground truth or trading result is used.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(json.dumps({k:v for k,v in summary.items() if k!="by_radar_date_pair"},indent=2),flush=True)


if __name__ == "__main__":
    main()
