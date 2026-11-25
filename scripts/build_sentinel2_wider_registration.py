"""Recover archived B04 pairs using more exterior landmarks, with physical controls."""
import json
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

import numpy as np

from audit_registered_shadow_fit import refit as control_refit
from audit_sentinel2_alternate_registration import alternatives, attach_sources, refit as alternate_refit, registration_products
from audit_sentinel2_weekly_volume import observations
from audit_shadow_registration_pixels import index
from audit_shadow_precision import describe
from build_registered_shadow_development import DATED, observation_sources, prepare_jobs, refit, study_weeks
from build_shadow_registration_inputs import SEARCH_RADIUS_M, anchor_grid, tank_points
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED
from collect_public import write, write_csv
from fit_shadow_registration import tank_offset

STEM = "sentinel2_wider_registration"
RADIUS_M = 1600
SIDES = ("earlier", "later")
OFFSETS = ("later_minus_earlier_east_m", "later_minus_earlier_north_m")
CHECKED = "checked_offset"
GAP_REASON = "no_independently_checked_registration"


def identity(row):
    return row["comparison"], row["week_or_control_day"], row["candidate_id"]


def week_tank(row):
    return row["week_ending"], row["candidate_id"]


def source_links(rows):
    sources, pairs = index("sources", "source_id"), index("scene_pairs", "scene_pair_id")
    for r in sources.values():
        for field in ("west","south","east","north"):
            r[field] = float(r[field])
    links = []
    for row in rows:
        pair = pairs[row["scene_pair_id"]]
        first,last = (sources[pair[f"{side}_source_id"]] for side in SIDES)
        links.append((row["comparison"], row["study_phase"], row["week_or_control_day"], row["candidate_id"],
                      row["roof_type"], row["terminal_id"],
                      row["inputs_available_before_utc"], *(row[f"{side}_archive_available_at_utc"] for side in SIDES), first,last))
    return links


def measure(links, *, radius=RADIUS_M):
    tanks, exclusion = tank_points()
    anchors = anchor_grid({link[3]:tanks[link[3]] for link in links}, exclusion, search_radius_m=radius)
    sources = {r["source_id"]:r for link in links for r in link[-2:]}
    screened, pairs, patches, profiles, patch_rows = registration_products(links,sources,anchors)
    offsets, by_pair = {}, {}
    for w,link in zip(screened,links,strict=True):
        p = pairs[w["scene_pair_id"]]
        if (identity(w)!=(link[0],link[2],link[3])
                or tuple(p[f"{s}_source_id"] for s in SIDES) != tuple(r["source_id"] for r in link[-2:])):
            raise ValueError("Registration screening differs from the exact input manifest")
        versions = tuple(r["version_id"] for r in link[-2:])
        by_pair[w["scene_pair_id"]] = versions
        if identity(w) in offsets:
            raise ValueError(f"Duplicate registration job {identity(w)}")
        offsets[identity(w)] = dict(tank_offset(w,profiles),
            **{f"{s}_version":v for s,v in zip(SIDES,versions,strict=True)},
            fit_screened_patches=w["fit_screened_patch_count"],check_screened_patches=w["spatial_check_screened_patch_count"])
    patch_evidence = [dict(r, **{f"{s}_version":v for s,v in zip(SIDES,by_pair[r["scene_pair_id"]],strict=True)},
                           **{k:patches[r["anchor_id"]][k] for k in ("west","south","east","north")})
                      for r in patch_rows]
    return offsets,patch_evidence


def offset_value(row):
    return tuple(float(row[k]) for k in OFFSETS) if row["registration_status"] == CHECKED else None


def physical_controls(windows):
    control_rows = [r for r in windows if r["comparison"]=="tandem" or
                    (r["comparison"],r["study_phase"])==("fixed_roof","development")]
    old_offsets = {identity(r):r for r in read_csv("shadow_registration_control_offsets")}
    old = {identity(r):r for r in read_csv("shadow_registered_control_fit")}
    links = source_links(control_rows)
    if old_offsets.keys()!={identity(r) for r in control_rows}:
        raise ValueError("The exterior control offsets and windows use different samples")
    wide, patches = measure(links)
    refits = [(r, tuple(r[f"{s}_version"] for s in SIDES)) for key,r in wide.items()
              if offset_value(r) is not None and offset_value(r)!=offset_value(old_offsets[key])]
    changed = {identity(r):r for r in control_refit(refits)} if refits else {}
    if len(changed)!=len(refits):
        raise ValueError("Control refits did not preserve exact tank pairs")
    output = []
    for window in control_rows:
        key = identity(window)
        r,previous = wide[key],old.get(key)
        new = changed.get(key,previous) if r["registration_status"]==CHECKED else None
        if previous and any(r[f"{s}_version"]!=previous[f"{s}_version"] for s in SIDES):
            raise ValueError(f"Control source changed for {key}")
        output.append(dict(r,roof_type=window["roof_type"],original_registration_status=old_offsets[key]["registration_status"],
                           original_registered_change_m=float(previous["shared_registered_change_m"]) if previous else None,
                           wider_registered_change_m=float(new["shared_registered_change_m"]) if new else None,
                           repeated_roof_baseline_passed=new["baseline_both_resolved"] if new else None))
    return output,patches


def regular_recovery(weeks, selected, offsets):
    native,saved,geometry = observation_sources("holdout")
    jobs = []
    for mode, base, record, _, keys, *times in prepare_jobs(weeks,native,saved,"holdout"):
        key = week_tank(base)
        if mode!=DATED or key not in selected:
            continue
        offset = offsets["weekly",*key]
        if any(offset[f"{s}_version"]!=base[f"{s}_version"] for s in SIDES):
            raise ValueError(f"Different native pair in wider registration for {key}")
        jobs.append((mode,base,record,offset,keys,*times))
    if {week_tank(base) for _,base,*_ in jobs}!=selected:
        raise ValueError("Wider registration omitted dated B04 pairs")
    return list(refit(jobs,native,saved,geometry)), geometry


def recover(weeks, gap_weeks, windows):
    baseline = {week_tank(r):r for r in read_csv("weekly_shadow_registered_holdout_pairs")
                if r["geometry_mode"]==DATED and r["week_ending"] in gap_weeks}
    links = source_links([r for r in windows if (r["comparison"],r["study_phase"])==("weekly","holdout")
                         and (r["week_or_control_day"],r["candidate_id"]) in baseline])
    offsets,patches = measure(links)
    primary, geometry = regular_recovery(weeks,baseline.keys(),offsets)
    alt, _ = alternatives("holdout",weeks,excluded_terminals=("amid_cushing",))
    alt = {k:v for k,v in alt.items() if k[0] in gap_weeks}
    old_alt = {week_tank(r):r for r in read_csv("sentinel2_alternate_registration_holdout_pairs")
               if r["week_ending"] in gap_weeks}
    if alt.keys()!=old_alt.keys():
        raise ValueError("Wider alternate registration uses different original candidates")
    if alt:
        _, alt_links = attach_sources(alt,"holdout")
        alt_offsets, alt_patches = measure(alt_links)
        patches.extend(dict(r,scene_pair_id=f"alternate:{r['scene_pair_id']}") for r in alt_patches)
        alt_fit = alternate_refit(alt,{key:alt_offsets["alternate_weekly",*key] for key in alt},geometry,weeks)
    else:
        alt_offsets,alt_fit = {},[]
    result = []
    for method,rows,old,offset_book,prefix in (
            ("primary",primary,baseline,offsets,"weekly"),("alternate",alt_fit,old_alt,alt_offsets,"alternate_weekly")):
        for r in rows:
            k = week_tank(r)
            before = old[k]
            if any(r[f"{s}_version"]!=before[f"{s}_version"] or datetime.fromisoformat(r[f"{s}_capture_at_utc"])
                   !=datetime.fromisoformat(before[f"{s}_capture_at_utc"]) for s in SIDES):
                raise ValueError(f"Wider refit changed the physical observation for {k}")
            offset = offset_book[prefix,*k]
            physical = next(observations("holdout",weeks,geometry,(r,)),None)
            result.append(dict(week_ending=k[0],release_date=r["release_date"],candidate_id=k[1],terminal_id=r["terminal_id"],
                pair_source=method,inputs_available_before_utc=r["inputs_available_before_utc"],
                **{f"{s}_{f}":r[f"{s}_{f}"] for s in SIDES for f in ("version","capture_at_utc","archive_available_at_utc")},
                original_registration_status=before.get("offset_status",before.get("registration_status")),
                wider_registration_status=offset["registration_status"],
                **{f:offset[f] for f in (*OFFSETS,"fit_inliers","spatial_check_inliers","check_median_residual_m",
                                       "check_median_correlation_at_zero","check_median_correlation_at_offset")},
                **{f:r[f] for f in ("circular_area_m2","circular_bbl_per_m","fraction_of_capture_inside_eia_interval",
                    "fraction_of_eia_interval_observed","registered_both_shadow_diagnostics_pass","registered_change_m",
                    "registered_loss_range_min_m","registered_loss_range_max_m","earlier_registered_status","later_registered_status",
                    "earlier_registered_gap_m","later_registered_gap_m")},
                capture_change_kbbl=physical["capture_change_kbbl"] if physical else None))
    return result, patches


def summarize_controls(rows):
    result = {}
    for label in ("tandem_efr","fixed_roof"):
        sample = [r for r in rows if
                  (r["comparison"],r["roof_type"])==("tandem","EFR")] if label=="tandem_efr" else [
                      r for r in rows if r["comparison"]=="fixed_roof"]
        common = [r for r in sample if r["original_registered_change_m"] is not None and r["wider_registered_change_m"] is not None]
        new = [r for r in sample if r["original_registered_change_m"] is None and r["wider_registered_change_m"] is not None]
        result[label] = dict(input_pairs=len(sample),original_checked=sum(r["original_registered_change_m"] is not None for r in sample),
            wider_checked=sum(r["wider_registered_change_m"] is not None for r in sample),
            common_checked=len(common), wider_only_checked=len(new),
            original_common=describe([r["original_registered_change_m"] for r in common]),
            wider_common=describe([r["wider_registered_change_m"] for r in common]),
            wider_only=describe([r["wider_registered_change_m"] for r in new]))
        if common:
            before = np.array([abs(r["original_registered_change_m"]) for r in common])
            after = np.array([abs(r["wider_registered_change_m"]) for r in common])
            result[label].update(common_improved=int((after<before).sum()),common_worsened=int((after>before).sum()))
    return result


def weekly_rows(pairs, gap_weeks):
    for week,row in sorted(gap_weeks.items()):
        related = [r for r in pairs if r["week_ending"]==week]
        selected = {}
        for r in related:
            if r["registered_both_shadow_diagnostics_pass"]:
                selected.setdefault(r["candidate_id"],r)
        values = list(selected.values())
        area = sum(r["circular_area_m2"] for r in values)
        yield dict(week_ending=week,release_date=row["release_date"],cutoff_utc=row["cutoff_utc"],
                   baseline_status=row["b04_status"],checked_candidates=sum(r["wider_registration_status"]==CHECKED for r in related),
                   input_pairs=len(related),resolved_tanks=len(values),resolved_area_m2=area,
                   resolved_area_fraction=area/float(row["dated_reference_area_m2"]),
                   resolved_tanks_with_range_excluding_zero=sum(r["registered_loss_range_max_m"]<0 or
                                                                  r["registered_loss_range_min_m"]>0 for r in values),
                   resolved_tanks_median_interval_fraction=median(r["fraction_of_eia_interval_observed"] for r in values) if values else None,
                   capture_change_kbbl=sum(r["capture_change_kbbl"] for r in values) if values else None)


def publish(suffix, rows):
    fields = tuple(rows[0])
    write_csv(DERIVED/f"{STEM}_{suffix}.csv",fields,([r.get(k) for k in fields] for r in rows))


def main():
    weeks = study_weeks("holdout")
    gap_weeks = {r["week_ending"]:r for r in read_csv("holdout_optical_weeks") if r["b04_status"]==GAP_REASON}
    windows = read_csv("shadow_registration_tank_windows")
    pairs,gap_patches = recover(weeks,gap_weeks,windows)
    weekly = list(weekly_rows(pairs,gap_weeks))
    controls,control_patches = physical_controls(windows)
    for suffix,rows in (("gap_pairs",pairs),("gap_weeks",weekly),("gap_patches",gap_patches),
                        ("controls",controls),("control_patches",control_patches)):
        publish(suffix,rows)
    summary = dict(original_radius_m=SEARCH_RADIUS_M,wider_radius_m=RADIUS_M,gap_weeks=len(weekly),gap_candidate_pairs=len(pairs),
        gap_offset_statuses=dict(Counter(r["wider_registration_status"] for r in pairs)),
        weeks_with_checked_offset=[r["week_ending"] for r in weekly if r["checked_candidates"]],
        weeks_with_resolved_roof_measurement=[r["week_ending"] for r in weekly if r["resolved_tanks"]],
        additional_resolved_tank_weeks=sum(r["resolved_tanks"] for r in weekly),
        exterior_patches_in_gap_pairs=len(gap_patches), exterior_patches_in_controls=len(control_patches),
        controls=summarize_controls(controls),
        interpretation="Exploratory wider exterior registration. Candidate radius was tried on coverage. "
        "Patch screening and independent spatial checks use the same settings as the frozen B04 panel. "
        "Physical controls and gap recovery do not use EIA stock levels or market outcomes.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
