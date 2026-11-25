"""Test a fixed geometric scattering model on the 2019 NAIP/GRD comparison."""
import json
from collections import Counter
from contextlib import closing
from functools import lru_cache
from itertools import combinations
from statistics import fmean, median

import numpy as np

from build_s1_naip_calibration_inputs import native_grid
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_public import ROOT, write

RANGE_RESOLUTION_M = 20.0
AZIMUTH_RESOLUTION_M = 22.0
SAMPLE_SPACING_M = 10.0
PROFILE_STEP_M = 0.5
MIN_GAIN = 0.10
MIN_GAIN_OVER_PLACEBO = 0.05
MAX_NEAR_SPAN_M = 3.0
MAX_SHIFT_SPAN_M = 2.0
SHIFTS_M = ((0, 0), (-5, 0), (5, 0), (0, -5), (0, 5))
TWO_SQRT_2_LOG_2 = 2 * np.sqrt(2 * np.log(2))
RANGE_SIGMA = RANGE_RESOLUTION_M / TWO_SQRT_2_LOG_2
AZIMUTH_SIGMA = AZIMUTH_RESOLUTION_M / TWO_SQRT_2_LOG_2
PAPER = "https://elib.dlr.de/142408/1/IEEE%20JSTARS%20-%20Oil%20storage.pdf"
SPEC = "https://sentiwiki.copernicus.eu/web/s1-products"


@lru_cache(maxsize=10)
def packet(path):
    with np.load(ROOT / path, allow_pickle=False) as source:
        return source["sigma0"], source["valid"]


def native_chip(source, x0, y0, radius, maximum_layover, others):
    raster, valid = packet(source["calibrated_window_file"])
    shape = int(source["window_rows"]), int(source["window_cols"])
    if raster.shape != shape or valid.shape != shape:
        raise ValueError("Radar window shape changed after calibration")
    row0, col0 = int(source["window_row_offset"]), int(source["window_col_offset"])
    row, col = np.indices(shape)
    x = (col0 + col + 0.5 - x0) * SAMPLE_SPACING_M
    y = (row0 + row + 0.5 - y0) * SAMPLE_SPACING_M
    in_window = ((x >= -radius - maximum_layover - 25) & (x <= radius + 25) & (np.abs(y) <= radius + 25))
    x, y, target = x[in_window], y[in_window], raster[in_window]
    keep = valid[in_window] & np.isfinite(target)
    total = len(target)
    for neighbor_x, neighbor_y, neighbor_radius, neighbor_layover in others:
        # A raised return from a neighboring cylinder can extend toward the sensor.
        near_end = neighbor_x - neighbor_layover
        distance_x = x - np.clip(x, near_end, neighbor_x)
        keep &= np.hypot(distance_x, y - neighbor_y) > neighbor_radius + RANGE_RESOLUTION_M / 2
    return x[keep], y[keep], target[keep].astype(float), total


def arc(x, y, radius, offsets, far):
    phase = np.linspace(-np.pi / 2, np.pi / 2, 64)
    cross_range = radius * np.cos(phase) * (1 if far else -1)
    azimuth = radius * np.sin(phase)
    weights = np.exp(-0.5 * ((y[:, None] - azimuth) / AZIMUTH_SIGMA)**2)
    spread = np.exp(-0.5 * ((x[:, None, None] - offsets[None, :, None] - cross_range) / RANGE_SIGMA)**2)
    return np.einsum("ijk,ik->ij", spread, weights) / len(phase)


def positive_reduction(columns, signal):
    """Exact least-squares gain with two or three constrained arc amplitudes."""
    gram, cross = columns.T @ columns, columns.T @ signal
    best = 0.0
    for count in range(1, len(cross) + 1):
        for active in combinations(range(len(cross)), count):
            chosen = np.asarray(active)
            sub = gram[np.ix_(chosen, chosen)]
            if not np.all(np.diag(sub) > 1e-12):
                continue
            try:
                amplitudes = np.linalg.solve(sub, cross[chosen])
            except np.linalg.LinAlgError:
                continue
            if (amplitudes >= 0).all():
                best = max(best, float(amplitudes @ cross[chosen]))
    return best


def fit_shift(x, y, residual, basis, radius, shell, cotangent, gaps):
    def centered(templates):
        return templates - basis @ (basis.T @ templates)

    anchors = centered(arc(x, y, radius, np.array([0, -shell * cotangent]), far=False))
    baseline = max(float(residual @ residual) - positive_reduction(anchors, residual), 1e-12)
    candidates = -(shell - gaps) * cotangent

    def gain(far):
        shapes = centered(arc(x, y, radius, candidates, far))
        values = [positive_reduction(np.column_stack((anchors, shape)), residual) for shape in shapes.T]
        anchor_gain = positive_reduction(anchors, residual)
        return (np.asarray(values) - anchor_gain) / baseline

    correct, placebo = gain(True), gain(False)
    optimum = int(np.argmax(correct))
    span = gaps[correct >= correct[optimum] - 0.025]
    return float(gaps[optimum]), float(correct[optimum]), float(placebo.max()), float(span.min()), float(span.max())


def tank_fit(source, row, geometry, positions):
    tank = row["candidate_id"]
    x0, y0 = (float(row[k]) for k in ("approximate_native_sample", "approximate_native_line"))
    radius = float(row["nominal_rim_diameter_m"]) / 2
    theta = np.deg2rad(float(row["incidence_angle_deg"]))
    cotangent = float(1 / np.tan(theta))
    shell = geometry[tank][1]
    others = []
    for other, (px, py) in positions.items():
        if tank == other:
            continue
        nx, ny = (px - x0) * SAMPLE_SPACING_M, (py - y0) * SAMPLE_SPACING_M
        other_radius, other_shell = geometry[other]
        if abs(nx) <= radius + other_radius + 100 and abs(ny) <= radius + other_radius + 50:
            others.append((nx, ny, other_radius, other_shell * cotangent))
    x, y, signal, total = native_chip(source, x0, y0, radius, shell * cotangent, others)
    result = dict(native_vv_calibrated_window=source["calibrated_window_file"], native_cells_before_mask=total,
        retained_native_cells=len(signal), nominal_grd_ground_range_resolution_m=RANGE_RESOLUTION_M,
        reference_layover_m=None, reference_layover_low_m=None, reference_layover_high_m=None,
        geometry_template_gap_m=None, geometry_template_gain=None, wrong_side_gain=None,
        geometry_template_near_gap_low_m=None, geometry_template_near_gap_high_m=None,
        gap_span_under_half_pixel_shifts_m=None, status="insufficient_uncontaminated_samples")
    if row["qualified_aerial_shadow_reference"] == "True":
        result.update(reference_layover_m=float(row["aerial_gap_m"]) * cotangent,
            reference_layover_low_m=float(row["aerial_gap_low_m"]) * cotangent,
            reference_layover_high_m=float(row["aerial_gap_high_m"]) * cotangent)
    if len(signal) < 35 or len(signal) < 0.30 * total:
        return result
    basis, _ = np.linalg.qr(np.column_stack((np.ones(len(x)), x / radius, y / radius)), mode="reduced")
    residual = signal - basis @ (basis.T @ signal)
    gaps = np.arange(0, min(shell, 24.0) + 1e-8, PROFILE_STEP_M)
    trials = [fit_shift(x - dx, y - dy, residual, basis, radius, shell, cotangent, gaps) for dx, dy in SHIFTS_M]
    gap, gain, placebo, low, high = trials[0]
    span = max(t[0] for t in trials) - min(t[0] for t in trials)
    result.update(geometry_template_gap_m=gap, geometry_template_gain=gain, wrong_side_gain=placebo,
        geometry_template_near_gap_low_m=low, geometry_template_near_gap_high_m=high,
        gap_span_under_half_pixel_shifts_m=span)
    if gap in (gaps[0], gaps[-1]) or gain < MIN_GAIN:
        status = "weak_or_boundary_template"
    elif gain - placebo < MIN_GAIN_OVER_PLACEBO:
        status = "wrong_side_not_rejected"
    elif high - low > MAX_NEAR_SPAN_M or span > MAX_SHIFT_SPAN_M:
        status = "gap_not_stable"
    else:
        status = "passes_single_scene_template_diagnostic"
    return dict(result, status=status)


def main():
    calibration = read("s1_naip_2019_calibration_inputs")
    product, = {r["radar_product_id"] for r in calibration}
    source_grid = native_grid(ROOT / calibration[0]["source_annotation"])[2]
    if not (np.diff(source_grid[:, :, 1], axis=1) > 0).all():
        raise ValueError("Annotation does not show increasing slant range with sample index")
    windows = {r["native_window_file"]: r for r in read("s1_grd_sigma0_window_index") if r["product_id"] == product}
    positions = {r["candidate_id"]: (float(r["native_sample"]), float(r["native_line"]))
                 for r in read("s1_grd_native_window_index") if r["product_id"] == product and r["polarization"] == "VV"}
    with closing(read_only(SOURCE)) as db:
        geometry = {tank: (diameter / 2, shell if shell else 25.0) for tank, diameter, shell in db.execute("""SELECT
            candidate_id,coalesce(lidar_upper_outline_diameter_m,footprint_diameter_m),lidar_rim_to_ground_native_z
            FROM tank_engineering_registry""")}
    for row in calibration:
        tank = row["candidate_id"]
        _, shell = geometry[tank]
        geometry[tank] = float(row["nominal_rim_diameter_m"]) / 2, shell
    rows = [dict(row, **tank_fit(windows[row["vv_window_file"]], row, geometry, positions)) for row in calibration]
    references = [r for r in rows if r["qualified_aerial_shadow_reference"] == "True"]
    negatives = [r for r in rows if r["roof_type"] != "EFR"]
    checked = [r for r in references if r["geometry_template_gap_m"] is not None]
    qualified = [r for r in checked if r["status"] == "passes_single_scene_template_diagnostic"]
    errors = [abs(r["geometry_template_gap_m"] - float(r["aerial_gap_m"])) for r in qualified]
    report = dict(radar_product_id=product, aerial_roof_references=len(references),
        median_nominal_layover_m=median(r["reference_layover_m"] for r in references),
        reference_above_native_sample_spacing=sum(r["reference_layover_m"] >= SAMPLE_SPACING_M for r in references),
        references_with_upper_layover_at_least_nominal_range_resolution=sum(
            r["reference_layover_high_m"] >= RANGE_RESOLUTION_M for r in references),
        nominal_range_resolution_m=RANGE_RESOLUTION_M, native_pixel_spacing_m=SAMPLE_SPACING_M,
        status_counts_aerial_references=dict(Counter(r["status"] for r in references)),
        status_counts_covered_controls=dict(Counter(r["status"] for r in negatives)),
        qualified_template_vs_aerial_count=len(errors), qualified_template_vs_aerial_mean_absolute_difference_m=fmean(errors) if errors else None,
        physical_model_source=PAPER, product_specification=SPEC,
        interpretation="This is one retrospective scene. Nominal layover is the apparent displacement in ground "
        "range for a vertical separation at the same plan location. The arc model is a diagnostic using the local "
        "incidence angle, a Gaussian resolution approximation, known circular dimensions and a mask for nearby tanks. "
        "The aerial comparison has a roughly eight-hour observation gap. It is not an independently validated radar estimator.")
    export("s1_grd_2019_roof_resolvability", rows)
    write(DERIVED / "s1_grd_2019_roof_resolvability_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
