"""Radiometrically calibrate the native IW GRD pilot windows using source XML LUTs."""
import argparse
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import ROOT, write
from collect_s1_grd_roof_inputs import STORE

OUTPUT = DERIVED / "s1_grd_sigma0"
CALIBRATION_SOURCE = "https://sentinels.copernicus.eu/documents/247904/1877131/Sentinel-1-Product-Specification-18052021.pdf"
NOISE_SOURCE = ("https://sentiwiki.copernicus.eu/__attachments/1673968/DI-MPC-TN-MPC-0392%20-%20Thermal%20"
                "Denoising%20of%20Products%20Generated%20by%20Sentinel-1%20IPF%202024%20-%202.2.pdf"
                "?inst-v=b0801e75-09ce-4554-81b3-1df420f76802")


@dataclass(frozen=True)
class RangeVector:
    line: float
    pixels: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class NoiseBlock:
    swath: str
    first_line: int
    last_line: int
    first_pixel: int
    last_pixel: int
    azimuth_lines: np.ndarray
    azimuth_values: np.ndarray


def series(node, tag):
    element = node.find(tag)
    if element is None:
        raise ValueError(f"Missing {tag} in {node.tag}")
    values = np.fromstring(element.text or "", sep=" ")
    if len(values) != int(element.attrib["count"]) or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid {tag} LUT in {node.tag}")
    return values


def records(root, tag):
    group = root.find(tag)
    if group is None or len(group) != int(group.attrib["count"]) or not len(group):
        raise ValueError(f"Invalid {tag} in {root.tag}")
    return list(group)


def increasing(values):
    return bool((np.diff(values) > 0).all())


def vector(node, tag, line):
    pixels, values = (series(node, k) for k in ("pixel", tag))
    if len(pixels) != len(values) or not increasing(pixels):
        raise ValueError(f"Invalid range samples for {tag}")
    return RangeVector(line, pixels, values)


def read_band(manifest, pol):
    objects = manifest["objects"]
    label = f"iw-{pol.lower()}"
    roots = []
    paths = [f"annotation/{label}.xml", f"annotation/calibration/calibration-{label}.xml",
             f"annotation/calibration/noise-{label}.xml"]
    for name in paths:
        asset = objects[name]
        content = (ROOT / asset["local_file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != asset["sha256"]:
            raise ValueError(f"Source annotation hash changed for {asset['local_file']}")
        roots.append(ET.fromstring(content))
    annotation, calibration, noise = roots
    fields = ("missionId", "productType", "polarisation", "mode", "startTime", "stopTime", "absoluteOrbitNumber")
    headers = {tuple(x.findtext(f"./adsHeader/{field}") for field in fields) for x in roots}
    if len(headers) != 1 or annotation.findtext("./adsHeader/polarisation") != pol:
        raise ValueError(f"Band identities differ for {manifest['product']['product_name']}/{pol}")
    thermal = annotation.findtext("./imageAnnotation/processingInformation/thermalNoiseCorrectionPerformed")
    if thermal not in {"true", "false"}:
        raise ValueError("Missing source thermal noise correction state")
    info = annotation.find("./imageAnnotation/imageInformation")
    first = datetime.fromisoformat(info.findtext("productFirstLineUtcTime"))
    step = float(info.findtext("azimuthTimeInterval"))
    cal = [vector(v, "sigmaNought", int(v.findtext("line"))) for v in records(calibration, "calibrationVectorList")]
    if not increasing([v.line for v in cal]) or any((v.values <= 0).any() for v in cal):
        raise ValueError("Invalid calibration factor table")
    ranges, clock_differences = [], []
    for v in records(noise, "noiseRangeVectorList"):
        row = (datetime.fromisoformat(v.findtext("azimuthTime")) - first).total_seconds() / step
        clock_differences.append(abs(row - int(v.findtext("line"))))
        ranges.append(vector(v, "noiseRangeLut", row))
    if not increasing([v.line for v in ranges]) or any((v.values < 0).any() for v in ranges):
        raise ValueError("Invalid noise range table")
    blocks = []
    for node in records(noise, "noiseAzimuthVectorList"):
        lines, factors = (series(node, k) for k in ("line", "noiseAzimuthLut"))
        bounds = [int(node.findtext(k)) for k in ("firstAzimuthLine", "lastAzimuthLine", "firstRangeSample", "lastRangeSample")]
        if len(lines) != len(factors) or not increasing(lines) or (factors <= 0).any():
            raise ValueError("Invalid azimuth noise table")
        if not (bounds[0] <= lines[0] <= lines[-1] <= bounds[1] and bounds[2] <= bounds[3]):
            raise ValueError("Azimuth noise LUT lies outside its block")
        blocks.append(NoiseBlock(node.findtext("swath"), *bounds, lines, factors))
    return cal, ranges, blocks, thermal == "true", max(clock_differences)


def interpolate_axis(at, coordinates, values):
    """Linearly interpolate a scalar or vector along the first axis, extending its end segments."""
    if len(coordinates) == 1:
        return np.broadcast_to(values[0], (len(at), *values.shape[1:]))
    bracket = np.searchsorted(coordinates, at, side="right").clip(1, len(coordinates) - 1)
    before, after = bracket - 1, bracket
    weight = (at - coordinates[before]) / (coordinates[after] - coordinates[before])
    if values.ndim > 1:
        weight = weight[:, None]
    return (1 - weight) * values[before] + weight * values[after]


def calibration_grid(vectors, lines, pixels):
    offsets = np.array([v.line for v in vectors])
    if lines[0] < offsets[0] or lines[-1] > offsets[-1] or any(
            pixels[0] < v.pixels[0] or pixels[-1] > v.pixels[-1] for v in vectors):
        raise ValueError("Calibration grid does not bracket the native window")
    samples = np.array([np.interp(pixels, v.pixels, v.values) for v in vectors])
    return interpolate_axis(lines, offsets, samples)


def block_ranges(block, blocks, vectors):
    # At mission timestamp precision, a vector recorded on an edge can differ by a fraction of a line.
    epsilon = 0.005

    def within(v, candidate):
        return candidate.first_line - epsilon <= v.line <= candidate.last_line + epsilon

    def has_support(v):
        return np.count_nonzero((v.pixels > block.first_pixel) & (v.pixels < block.last_pixel) & (v.values > 0)) >= 2

    inside = [v for v in vectors if within(v, block) and has_support(v)]
    if inside:
        return inside
    center = (block.first_line + block.last_line) / 2
    same_swath = [v for v in vectors if has_support(v) and any(
        candidate.swath == block.swath and within(v, candidate) for candidate in blocks)]
    if not same_swath:
        raise ValueError(f"No range noise vector for block {block.swath}/{block.first_line}")
    return [min(same_swath, key=lambda v: abs(v.line - center))]


def noise_grid(vectors, blocks, lines, pixels):
    output = np.full((len(lines), len(pixels)), np.nan)
    for block in blocks:
        row = np.flatnonzero((lines >= block.first_line) & (lines <= block.last_line))
        col = np.flatnonzero((pixels >= block.first_pixel) & (pixels <= block.last_pixel))
        if not len(row) or not len(col):
            continue
        area = np.ix_(row, col)
        if np.isfinite(output[area]).any():
            raise ValueError("Overlapping source azimuth noise blocks")
        chosen = block_ranges(block, blocks, vectors)
        samples = []
        for v in chosen:
            inside = (v.pixels > block.first_pixel) & (v.pixels < block.last_pixel) & (v.values > 0)
            samples.append(interpolate_axis(pixels[col], v.pixels[inside], v.values[inside]))
        range_factor = interpolate_axis(lines[row], np.array([v.line for v in chosen]), np.array(samples))
        azimuth_factor = interpolate_axis(lines[row], block.azimuth_lines, block.azimuth_values)
        output[area] = range_factor * azimuth_factor[:, None]
    return output


def window(row, parameters, *, output_root=OUTPUT):
    cal, vectors, blocks, thermal_applied, _ = parameters
    raw = ROOT / row["window_file"]
    payload = raw.read_bytes()
    if hashlib.sha256(payload).hexdigest() != row["window_sha256"]:
        raise ValueError(f"Native input hash differs for {raw}")
    with np.load(io.BytesIO(payload), allow_pickle=False) as packet:
        dn, valid = packet["dn"], packet["valid"]
    r0, c0, nr, nc = (int(row[k]) for k in ("window_row_offset", "window_col_offset", "window_rows", "window_cols"))
    if dn.shape != (nr, nc) or dn.dtype != np.uint16 or valid.shape != dn.shape or valid.dtype != bool:
        raise ValueError(f"Native measurement schema differs for {raw}")
    lines, pixels = np.arange(r0, r0 + nr), np.arange(c0, c0 + nc)
    gains = calibration_grid(cal, lines, pixels)
    noise = noise_grid(vectors, blocks, lines, pixels)
    if ((~np.isfinite(noise) | (noise <= 0)) & valid).any():
        raise ValueError(f"Unusable noise table at a valid measurement in {raw}")
    sigma_noise = noise / gains**2
    clean = (dn.astype(float)**2 - (0 if thermal_applied else noise)) / gains**2
    output = io.BytesIO()
    np.savez_compressed(output, sigma0=np.where(valid, clean, np.nan).astype("float32"),
                        noise_equivalent_sigma0=np.where(valid, sigma_noise, np.nan).astype("float32"), valid=valid)
    path = output_root / row["product_name"].removesuffix(".SAFE") / row["polarization"] / raw.name
    packet = output.getvalue()
    write(path, packet)
    return dict(product_id=row["product_id"], product_name=row["product_name"], polarization=row["polarization"],
        native_window_file=row["window_file"], native_window_sha256=row["window_sha256"],
        window_row_offset=r0, window_col_offset=c0, window_rows=nr, window_cols=nc,
        calibrated_window_file=str(path.relative_to(ROOT)), calibrated_window_sha256=hashlib.sha256(packet).hexdigest(),
        source_thermal_correction_already_applied=thermal_applied, valid_pixels=int(valid.sum()),
        nonpositive_sigma0_after_noise=int(((clean <= 0) & valid).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-gaps", action="store_true")
    args = parser.parse_args()
    stem = "s1_grd_holdout_gap" if args.holdout_gaps else "s1_grd"
    products = {m["product"]["product_id"]: m for p in sorted(STORE.glob("*/native_manifest.json"))
                if (m := json.loads(p.read_bytes()))}
    grouped = defaultdict(dict)
    for row in read(f"{stem}_native_window_index"):
        grouped[row["product_id"], row["polarization"]][row["window_file"]] = row
    rows = []
    max_line_difference = 0.0
    for (identity, pol), windows in sorted(grouped.items()):
        parameters = read_band(products[identity], pol)
        max_line_difference = max(max_line_difference, parameters[-1])
        rows.extend(window(row, parameters) for row in windows.values())
    export(f"{stem}_sigma0_window_index", rows)
    report = dict(products=len({r["product_id"] for r in rows}), calibrated_windows=len(rows),
        windows_by_polarization=dict(Counter(r["polarization"] for r in rows)),
        source_already_denoised_windows=sum(r["source_thermal_correction_already_applied"] for r in rows),
        valid_window_pixels=sum(r["valid_pixels"] for r in rows),
        nonpositive_calibrated_window_pixels=sum(r["nonpositive_sigma0_after_noise"] for r in rows),
        max_annotated_range_noise_line_vs_azimuth_time_difference_pixels=max_line_difference,
        calibration_source=CALIBRATION_SOURCE, thermal_noise_source=NOISE_SOURCE,
        interpretation="Linear sigma0 and noise equivalent sigma0 use the source calibration factors. Noise range "
        "vectors are timed by azimuthTime and combined with azimuth factors within the annotated swath blocks. "
        "The native sample positions and masks are preserved. Signed values after subtraction remain available; "
        "dB conversion requires positive power. No terrain correction, speckle filtering, or roof-height estimate is applied.")
    write(DERIVED / f"{stem}_sigma0_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
