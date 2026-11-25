"""Compare OLI RT heights to timestamped laser and Sentinel-2 controls."""

import json
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta

from audit_shadow_precision import describe
from build_cushing_landsat7_mixed_inputs import sentinel_endpoints
from measure_cushing_landsat7 import RESOLVED, laser_references
from measure_cushing_landsat8_rapid import DERIVED, STEM, export, legacy_geometry, rows, save_json


def timed_landsat(fits):
    grouped = defaultdict(list)
    for r in fits:
        if r["angle_status"] == "resolved" and int(r["roof_pixels_passing_qa"]) > 0:
            grouped[r["candidate_id"]].append((datetime.fromisoformat(r["captured_at_utc"]),r))
    return grouped


def passed(r):
    return r["fit_status"]==RESOLVED and r["center_at_boundary"]=="False"


def difference(row, reference):
    return float(row["gap_m"])-reference if passed(row) else None


def laser_matches(by_roof, geometry):
    for tank, sensor, source, a, b, gap in laser_references(geometry, by_roof):
        first,last = map(datetime.fromisoformat,(a,b))
        middle = first+(last-first)/2
        stamp, fit = min(by_roof[tank],key=lambda x:(abs(x[0]-middle),x[1]["product_id"]))
        span = max(abs(stamp-first),abs(stamp-last)).total_seconds()/3600
        if span <= 168:
            yield dict(candidate_id=tank, reference_sensor=sensor, reference_source=source, reference_start_utc=a,
                reference_end_utc=b, reference_gap_m=gap, landsat_product_id=fit["product_id"],
                landsat_captured_at_utc=stamp.isoformat(), capture_separation_hours=span, landsat_fit_status=fit["fit_status"],
                landsat_center_at_boundary=fit["center_at_boundary"], landsat_gap_m=fit["gap_m"],
                signed_gap_difference_m=difference(fit,gap))


def sentinel_matches(by_roof, controls):
    for tank, items in by_roof.items():
        if tank not in controls:
            continue
        times, observations = controls[tank]
        for stamp,r in items:
            i = bisect_left(times,stamp)
            closest = min(observations[max(0,i-1):i+1],key=lambda x:(abs(x.captured-stamp),x.record["version_id"]))
            span = abs(closest.captured-stamp)
            if span <= timedelta(hours=48):
                s = closest.record
                reference = float(s["estimated_gap_m"])
                yield dict(candidate_id=tank, landsat_product_id=r["product_id"], landsat_captured_at_utc=stamp.isoformat(),
                    sentinel_version_id=s["version_id"], sentinel_captured_at_utc=closest.captured.isoformat(),
                    sentinel_observation_source=s["measurement_source"], sentinel_observation_id=s["observation_id"],
                    capture_separation_hours=span.total_seconds()/3600, reference_gap_m=reference, landsat_gap_m=r["gap_m"],
                    landsat_fit_status=r["fit_status"], landsat_center_at_boundary=r["center_at_boundary"],
                    signed_gap_difference_m=difference(r,reference))


def cohorts(records, thresholds):
    return {f"within_{n}h":dict(candidate_controls=sum(r["capture_separation_hours"]<=n for r in records),
        **describe([r["signed_gap_difference_m"] for r in records if r["capture_separation_hours"]<=n
                    and r["signed_gap_difference_m"] is not None])) for n in thresholds}


def main():
    geometry = legacy_geometry()
    fits = timed_landsat(rows(DERIVED/f"{STEM}_height_fits.csv"))
    laser = list(laser_matches(fits,geometry))
    sentinel = list(sentinel_matches(fits,sentinel_endpoints(geometry)))
    for suffix,records in (("laser_controls",laser),("sentinel_controls",sentinel)):
        if records:
            export(f"{STEM}_height_{suffix}",records)
    report = dict(laser=cohorts(laser,(3,24,48,168)),sentinel2=cohorts(sentinel,(3,24,48)),
        selection="Nearest usable-pixel control chosen before Landsat height diagnostics. Differences include capture-time mismatch and both sensors' errors.")
    save_json(DERIVED/f"{STEM}_height_controls_summary.json",report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    main()
