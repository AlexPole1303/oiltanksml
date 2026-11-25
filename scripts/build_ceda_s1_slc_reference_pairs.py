"""Link recovered radar windows to independent dated aerial height controls."""

import json
from collections import defaultdict
from datetime import datetime
from statistics import median

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import write

STEM = "ceda_s1_slc_reference_pairs"
SOURCES = ("ok_aerial_2025_roof_shadow_reference", "naip_2025_roof_shadow_reference")
MAX_BRACKET_HOURS = 8*24
S2_MAX_HOURS = 48


def hours(delta):
    return delta.total_seconds()/3600


def sentinel2_links(panels):
    for r in read("weekly_shadow_holdout_observations"):
        if (r["geometry_mode"], r["fit_status"], r["roof_type"]) != (
                "dated_2011_lidar", "resolved_in_geometry_diagnostic", "EFR"):
            continue
        captured = datetime.fromisoformat(r["optical_capture_at_utc"])
        tank = r["candidate_id"]
        for pols in panels[tank].values():
            if set(pols) != {"VV", "VH"}:
                continue
            begin, end = (datetime.fromisoformat(pols["VV"][k]) for k in ("capture_start_utc", "capture_end_utc"))
            distance = max(abs(hours(captured-begin)), abs(hours(captured-end)))
            if distance > S2_MAX_HOURS:
                continue
            yield dict(candidate_id=tank, terminal_id=r["terminal_id"], sentinel2_version_id=r["version_id"],
                sentinel2_observation_id=r["observation_id"], sentinel2_capture_utc=captured,
                sentinel2_gap_to_rim_m=float(r["estimated_gap_m"]), sentinel2_near_gap_low_m=float(r["near_optimal_gap_min_m"]),
                sentinel2_near_gap_high_m=float(r["near_optimal_gap_max_m"]), radar_capture_start_utc=begin,
                radar_capture_end_utc=end, radar_vv_cdse_uuid=pols["VV"]["cdse_uuid"], radar_vh_cdse_uuid=pols["VH"]["cdse_uuid"],
                maximum_capture_separation_hours=distance, within_24_hours=distance<=24)


def controls():
    for source in SOURCES:
        for r in read(source):
            if r["documented_roof_type"] != "EFR" or r["shadow_reference_status"] != "resolved_aerial_shadow":
                continue
            begin, end = map(datetime.fromisoformat, (r["start_utc"], r["end_exclusive_utc"]))
            low, value, high = map(float, (r["near_gap_min_m"], r["best_gap_m"], r["near_gap_max_m"]))
            if not (begin < end and 0 <= low <= value <= high):
                raise ValueError(f"Invalid aerial height reference for {r['candidate_id']}")
            yield r["candidate_id"], dict(reference_table=f"data/derived/{source}.csv", reference_start_utc=begin,
                reference_end_exclusive_utc=end, reference_gap_to_rim_m=value, reference_near_gap_low_m=low,
                reference_near_gap_high_m=high, reference_pixel_spacing_m=float(r["native_pixel_m"]),
                reference_exterior_clock_status=r.get("exterior_clock_status", "source_polygon_clock"),
                reference_asset=r["source_raster"])


def main():
    assets = {r["cdse_uuid"]: r for r in read("ceda_s1_slc_regional_assets")}
    panels = defaultdict(dict)
    for r in read("ceda_s1_slc_regional_windows"):
        source = assets[r["cdse_uuid"]]
        if (source["product_id"], source["polarization"]) != (r["product_id"], r["polarization"]):
            raise ValueError("A radar window points to a different acquisition or polarization")
        group = panels[r["candidate_id"]].setdefault(r["product_id"], {})
        group[r["polarization"]] = source
    rows = []
    for tank, r in controls():
        start, stop = (r[k] for k in ("reference_start_utc", "reference_end_exclusive_utc"))
        nearby = {}
        for pols in panels[tank].values():
            if set(pols) != {"VV", "VH"}:
                continue
            scene = pols["VV"]
            a, b = map(datetime.fromisoformat, (scene["capture_start_utc"], scene["capture_end_utc"]))
            delta = max(abs(hours(start-b)), abs(hours(stop-a)))
            if delta > MAX_BRACKET_HOURS:
                continue
            side = "before" if b < start else "after" if a >= stop else "overlap"
            if side not in nearby or delta < nearby[side][0]:
                nearby[side] = delta, a, b, pols
        for side, (distance, a, b, bands) in sorted(nearby.items()):
            rows.append(dict(candidate_id=tank, **r, radar_side=side, radar_capture_start_utc=a, radar_capture_end_utc=b,
                maximum_capture_separation_hours=distance, radar_vv_cdse_uuid=bands["VV"]["cdse_uuid"],
                radar_vh_cdse_uuid=bands["VH"]["cdse_uuid"], both_sides_available="before" in nearby and "after" in nearby,
                within_24_hours=distance<=24, within_72_hours=distance<=72))
    export(STEM, rows)
    sentinel2 = list(sentinel2_links(panels))
    export("ceda_s1_slc_sentinel2_pairs", sentinel2)
    summary = {}
    for source in SOURCES:
        group = [r for r in rows if r["reference_table"].endswith(f"/{source}.csv")]
        summary[source] = dict(distinct_roofs=len({r["candidate_id"] for r in group}),
            rows=len(group), roofs_with_both_radar_sides=len({r["candidate_id"] for r in group if r["both_sides_available"]}),
            roofs_within_24_hours=len({r["candidate_id"] for r in group if r["within_24_hours"]}),
            roofs_within_72_hours=len({r["candidate_id"] for r in group if r["within_72_hours"]}),
            closest_before_hours=min((r["maximum_capture_separation_hours"] for r in group if r["radar_side"]=="before"), default=None),
            closest_after_hours=min((r["maximum_capture_separation_hours"] for r in group if r["radar_side"]=="after"), default=None))
    report = dict(aerial_sources=summary, maximum_radar_to_aerial_separation_hours=MAX_BRACKET_HOURS,
        sentinel2_diagnostic_pairs=len(sentinel2), sentinel2_roofs=len({r["candidate_id"] for r in sentinel2}),
        sentinel2_passes=len({r["radar_vv_cdse_uuid"] for r in sentinel2}),
        sentinel2_pairs_within_24_hours=sum(r["within_24_hours"] for r in sentinel2),
        interpretation="Aerial shadow fits provide independent geometric observations at the aerial capture time. "
        "Time gaps allow oil levels to change. Brackets are temporal matching evidence, without contemporaneous truth labels.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
