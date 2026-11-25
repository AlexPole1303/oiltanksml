"""Measure the geometric and native-grid sensitivity of a ratio of shadow areas."""
import json
import math
from contextlib import ExitStack, closing
from functools import lru_cache
from statistics import median

import numpy as np

from audit_sentinel2_weekly_volume import PHASES
from build_registered_shadow_development import DATED, baseline_pairs, observation_sources
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export, read_only
from collect_public import write
from fit_shadow_geometry import PIXEL_M, bearing_vector

STEM = "sentinel2_two_shadow_area_geometry"
SIDES = ("earlier", "later")


def crescent_area(radius, displacement):
    """Area of one circle outside a translated circle of equal radius."""
    fraction = min(displacement/(2*radius), 1)
    return 2*radius**2 * (math.asin(fraction) + fraction*math.sqrt(1-fraction**2))


def area_fill_fraction(radius, shell, gap, sun_elevation):
    tangent = math.tan(math.radians(sun_elevation))
    return 1-crescent_area(radius, gap/tangent)/crescent_area(radius, shell/tangent)


def shadow_pixel_counts(radius, exterior_shift, interior_shift, origin, direction):
    """Label perfect point samples on an extended copy of the source raster grid."""
    span = math.ceil((radius + exterior_shift) / PIXEL_M) + 1
    offsets = np.arange(-span, span + 1) * PIXEL_M
    east = (offsets + origin[0] % PIXEL_M)[None, :]
    north = (offsets + origin[1] % PIXEL_M)[:, None]
    rim = east**2 + north**2 <= radius**2
    sunward = east*direction[0] + north*direction[1]
    reach = np.clip(-sunward, 0, exterior_shift)
    ground = ((east+reach*direction[0])**2 + (north+reach*direction[1])**2 <= radius**2) & ~rim
    inner = ((east+interior_shift*direction[0])**2 + (north+interior_shift*direction[1])**2 > radius**2) & rim
    return int(inner.sum()), int(ground.sum())


def audit(phase, volumes, stack):
    packets, _, _ = observation_sources(phase)
    databases = {name: stack.enter_context(closing(read_only(packet[0]))) for name, packet in packets.items()}
    original = {(row["week_ending"], row["candidate_id"]): (packet, row)
                for mode, packet, _, row in baseline_pairs(phase) if mode == DATED}

    @lru_cache(maxsize=2048)
    def pixel_origin(packet, observation_id):
        return databases[packet].execute("""SELECT center_east_m,center_north_m FROM pixel_red_reflectance
            WHERE observation_id=? ORDER BY tile_row,tile_col LIMIT 1""", (observation_id,)).fetchone()

    for row in volumes:
        if row["study_phase"] != phase:
            continue
        packet, prior = original[row["week_ending"], row["candidate_id"]]
        _, native, _, _ = packets[packet]
        diameter, shell, kbbl_per_m = (float(row[field]) for field in (
            "diameter_m", "shell_above_ground_m", "circular_kbbl_per_m"))
        results = dict(study_phase=phase, week_ending=row["week_ending"], candidate_id=row["candidate_id"],
                       diameter_m=diameter, shell_m=shell)
        for side in SIDES:
            observation_id = int(prior[f"{side}_observation_id"])
            chip = native[observation_id]
            if (chip["version_id"] != row[f"{side}_sentinel2_version"]
                    or chip["candidate_id"] != row["candidate_id"]):
                raise ValueError(f"Solar geometry version differs for {row['candidate_id']}/{side}")
            gap = float(row[f"{side}_gap_m"])
            elevation = 90-chip["sun_zenith_deg"]
            physical = 1-gap/shell
            radius = diameter/2
            ratio = area_fill_fraction(radius, shell, gap, elevation)
            if not (0 <= ratio <= physical + 1e-12 and 0 <= physical <= 1):
                raise ValueError(f"Invalid ideal shadow geometry for {row['candidate_id']}/{side}")
            sun = bearing_vector(chip["sun_zenith_deg"], chip["sun_azimuth_grid_deg"])
            slope = float(np.linalg.norm(sun))
            centre = chip["center_east_m"], chip["center_north_m"]
            origin = np.subtract(pixel_origin(packet, observation_id), centre)
            inner, outer = shadow_pixel_counts(radius, shell*slope, gap*slope, origin, sun/slope)
            inner_area = crescent_area(radius, gap*slope)
            outer_area = diameter*shell*slope
            full_exterior_ratio = 1-inner_area/outer_area
            native_ratio = 1-inner/outer if outer else None
            sorted_ratio = 1-min(inner, outer)/max(inner, outer) if inner and outer else None
            values = dict(version_id=chip["version_id"], sun_elevation_deg=elevation,
                fill_fraction_from_gap=physical, fill_fraction_from_area_ratio=ratio,
                area_ratio_bias_percentage_points=100*(ratio-physical),
                area_ratio_bias_kbbl=(ratio-physical)*shell*kbbl_per_m,
                inner_shadow_area_m2=inner_area, full_exterior_shadow_area_m2=outer_area,
                full_exterior_fill_fraction=full_exterior_ratio,
                full_exterior_bias_pp=100*(full_exterior_ratio-physical),
                inner_shadow_pixel_centres=inner, full_exterior_shadow_pixel_centres=outer,
                positive_gap_with_no_inner_pixel=gap > 0 and inner == 0,
                native_fill_fraction=native_ratio,
                native_grid_effect_pp=100*(native_ratio-full_exterior_ratio) if outer else None,
                native_bias_pp=100*(native_ratio-physical) if outer else None,
                native_bias_kbbl=(native_ratio-physical)*shell*kbbl_per_m if outer else None,
                sorted_native_fill_fraction=sorted_ratio,
                sorted_native_bias_pp=100*(sorted_ratio-physical) if sorted_ratio is not None else None,
                sorted_native_bias_kbbl=(sorted_ratio-physical)*shell*kbbl_per_m if sorted_ratio is not None else None)
            results.update({f"{side}_{field}": value for field, value in values.items()})
        results["area_ratio_bias_in_change_kbbl"] = results["later_area_ratio_bias_kbbl"]-results["earlier_area_ratio_bias_kbbl"]
        for prefix in ("native", "sorted_native"):
            earlier, later = (results[f"{side}_{prefix}_bias_kbbl"] for side in SIDES)
            results[f"{prefix}_bias_in_change_kbbl"] = later-earlier if earlier is not None and later is not None else None
        yield results


def distribution(values):
    values = [abs(value) for value in values if value is not None]
    if not values:
        return dict(count=0)
    return dict(count=len(values), median=median(values), p90=float(np.percentile(values, 90)), maximum=max(values))


def main():
    volumes = read_csv("sentinel2_weekly_tank_volumes")
    with ExitStack() as stack:
        rows = [row for phase in PHASES for row in audit(phase, volumes, stack)]
    if len(rows) != len(volumes):
        raise ValueError("The ideal check does not match the original accepted volume sample")
    metrics = {}
    for phase in PHASES:
        sample = [row for row in rows if row["study_phase"] == phase]
        level = [abs(r[f"{side}_area_ratio_bias_percentage_points"]) for r in sample for side in SIDES]
        change = [abs(r["area_ratio_bias_in_change_kbbl"]) for r in sample]
        sides = [(r, side) for r in sample for side in SIDES]
        metrics[phase] = dict(pairs=len(sample), median_abs_fill_bias_pp=median(level), max_abs_fill_bias_pp=max(level),
            median_abs_change_bias_kbbl=median(change), max_abs_change_bias_kbbl=max(change),
            full_exterior_continuous_abs_bias_pp=distribution(r[f"{s}_full_exterior_bias_pp"] for r,s in sides),
            native_grid_abs_effect_pp=distribution(r[f"{s}_native_grid_effect_pp"] for r,s in sides),
            native_grid_abs_bias_pp=distribution(r[f"{s}_native_bias_pp"] for r,s in sides),
            native_grid_abs_weekly_pair_bias_kbbl=distribution(r["native_bias_in_change_kbbl"] for r in sample),
            sorted_native_abs_bias_pp=distribution(r[f"{s}_sorted_native_bias_pp"] for r,s in sides),
            sorted_native_abs_weekly_pair_bias_kbbl=distribution(r["sorted_native_bias_in_change_kbbl"] for r in sample),
            positive_gap_endpoints=sum(r[f"{s}_fill_fraction_from_gap"] < 1 for r,s in sides),
            positive_gap_without_inner_pixel=sum(r[f"{s}_positive_gap_with_no_inner_pixel"] for r,s in sides),
            endpoints_without_exterior_pixel=sum(r[f"{s}_full_exterior_shadow_pixel_centres"] == 0 for r,s in sides),
            native_endpoints_with_inner_larger_than_outer=sum(r[f"{s}_inner_shadow_pixel_centres"] >
                r[f"{s}_full_exterior_shadow_pixel_centres"] for r,s in sides))
    example = area_fill_fraction(40, 20, 10, 20)
    report = dict(phases=metrics, example=dict(diameter_m=80, shell_m=20, gap_m=10, solar_elevation_deg=20,
        fill_fraction_from_gap=0.5, fill_fraction_from_area_ratio=example), interpretation="The model assumes a vertical "
        "circular cylinder and roof of the same radius, parallel light, level ground and nadir viewing. "
        "The original results use two exact circular crescents. The extended check uses the entire exterior ground "
        "shadow from the swept cylinder and the interior crescent. Simulated 10 m point samples match the source "
        "raster grid phase at the mapped centre and use the per-granule solar direction. The grid extends past the "
        "archived crop to include all ideal exterior shadow. Point sampling isolates quantization and omits the "
        "sensor response, view angle, and imagery-based segmentation. The size-sorted calculation requires both "
        "shadows to contain a pixel centre and applies one minus the smaller-to-larger count ratio. The known simulation gaps are inputs from "
        "accepted Sentinel-2 fits. They are model values. The check uses no EIA or metered volumes.")
    export(STEM, rows)
    payload = (json.dumps(report, indent=2) + "\n").encode()
    write(DERIVED / f"{STEM}_summary.json", payload)
    print(payload.decode(), end="", flush=True)


if __name__ == "__main__":
    main()
