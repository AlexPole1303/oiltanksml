"""Calibrate detected power on the original CEDA Sentinel-1 TOPS burst grid."""

import gzip
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import numpy as np

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from calibrate_s1_grd_native_windows import calibration_grid, increasing, records, series, vector
from collect_public import ROOT, write

STEM = "ceda_s1_slc_regional_calibration"
INPUT = "ceda_s1_slc_regional"
OUTPUT = DERIVED / STEM
ROLES = ("annotation", "calibration", "noise")


def checked_xml(asset, files):
    roots = {}
    for role in ROLES:
        path = asset[f"local_{role}"]
        raw = gzip.decompress((ROOT / path).read_bytes())
        source = files[asset["product_id"], path]
        if (len(raw) != int(source["native_bytes"]) or hashlib.sha256(raw).hexdigest() != source["sha256"]
                or hashlib.md5(raw).hexdigest() != source["manifest_md5"] or source["role"] != role):
            raise ValueError(f"Changed SAFE metadata for {path}")
        roots[role] = ET.fromstring(raw)
    fields = ("missionId", "productType", "polarisation", "mode", "swath", "startTime", "stopTime", "absoluteOrbitNumber")
    identities = {tuple(root.findtext(f"adsHeader/{field}") for field in fields) for root in roots.values()}
    (header,) = identities
    if header[:5] != (asset["product_name"][:3], "SLC", asset["polarization"], "IW", asset["swath"]):
        raise ValueError(f"Wrong metadata identity for {asset['cdse_uuid']}")
    return (roots[k] for k in ROLES)


def source_parameters(asset, rows, files):
    annotation, cal, noise = checked_xml(asset, files)
    line_count = int(asset["lines_per_burst"])
    burst_id, = {r["relative_burst_id"] for r in rows}
    bursts = records(annotation.find("swathTiming"), "burstList")
    (index, burst), = [(i, r) for i, r in enumerate(bursts) if r.findtext("burstId") == burst_id]
    if annotation.findtext("imageAnnotation/processingInformation/thermalNoiseCorrectionPerformed") != "false":
        raise ValueError(f"Unexpected SLC denoising state for {asset['cdse_uuid']}")
    first, last = (series(burst, k).astype(np.int32) for k in ("firstValidSample", "lastValidSample"))
    if len(first) != line_count or len(last) != line_count:
        raise ValueError(f"Wrong SLC valid mask length for {asset['cdse_uuid']}")
    cal = [vector(v, "sigmaNought", int(v.findtext("line"))) for v in records(cal, "calibrationVectorList")]
    if not increasing([v.line for v in cal]) or any((v.values <= 0).any() for v in cal):
        raise ValueError(f"Invalid calibration LUT for {asset['cdse_uuid']}")
    start = index * line_count
    (range_node,) = [v for v in records(noise, "noiseRangeVectorList")
                     if v.findtext("azimuthTime") == burst.findtext("azimuthTime") and int(v.findtext("line")) == start]
    range_noise = vector(range_node, "noiseRangeLut", start)
    (az_node,) = [v for v in records(noise, "noiseAzimuthVectorList") if v.findtext("swath") == asset["swath"]
                  and int(v.findtext("firstAzimuthLine")) <= start
                  and int(v.findtext("lastAzimuthLine")) >= start+line_count-1]
    az_lines, az_noise = (series(az_node, tag) for tag in ("line", "noiseAzimuthLut"))
    az_sample_limits = tuple(int(az_node.findtext(k)) for k in ("firstRangeSample", "lastRangeSample"))
    if (not increasing(az_lines) or len(az_lines) != len(az_noise) or (az_noise <= 0).any()
            or (range_noise.values < 0).any()):
        raise ValueError(f"Invalid noise LUT for {asset['cdse_uuid']}")
    return start, first, last, cal, range_noise, az_lines, az_noise, az_sample_limits


def calibrate_window(row, iq, parameters):
    start, first, last, cal, range_noise, az_lines, az_noise, az_sample_limits = parameters
    a, b, c, d = (int(row[k]) for k in ("first_line", "end_line_exclusive", "first_sample", "end_sample_exclusive"))
    offset, count = (int(row[k]) for k in ("offset_complex_samples", "complex_samples"))
    if (not 0 <= a < b <= len(first) or not 0 <= c < d or (b-a, d-c) != (int(row["height"]), int(row["width"]))
            or count != (b-a)*(d-c) or not 0 <= offset < offset+count <= len(iq)
            or row["full_window_valid_by_native_mask"] != "True" or np.any(first[a:b] < 0)
            or np.any(first[a:b] > c) or np.any(last[a:b] < d-1)):
        raise ValueError(f"Invalid original SLC window for {row['candidate_id']}")
    lines, pixels = np.arange(start+a, start+b), np.arange(c, d)
    if (lines[0] < az_lines[0] or lines[-1] > az_lines[-1] or pixels[0] < max(az_sample_limits[0], range_noise.pixels[0])
            or pixels[-1] > min(az_sample_limits[1], range_noise.pixels[-1])):
        raise ValueError(f"Noise LUT fails to bracket the SLC window for {row['candidate_id']}")
    gain_sq = calibration_grid(cal, lines, pixels)**2
    noise = np.interp(lines, az_lines, az_noise)[:, None] * np.interp(pixels, range_noise.pixels, range_noise.values)
    raw = np.square(iq[offset:offset+count].astype(np.float64)).sum(axis=-1).reshape(gain_sq.shape)
    return raw/gain_sq, noise/gain_sq, (raw-noise)/gain_sq


def process(asset, rows, files):
    uuid = asset["cdse_uuid"]
    source = (ROOT / asset["local_packet"]).read_bytes()
    if len(source) != int(asset["packet_bytes"]) or hashlib.sha256(source).hexdigest() != asset["packet_sha256"]:
        raise ValueError(f"Changed native radar packet for {uuid}")
    with np.load(io.BytesIO(source), allow_pickle=False) as data:
        iq = data["iq_int16"]
        if str(data["cdse_uuid"]) != uuid or iq.dtype != np.dtype("<i2") or iq.ndim != 2 or iq.shape[1] != 2:
            raise ValueError(f"Invalid native radar packet for {uuid}")
    rows = sorted(rows, key=lambda r: int(r["offset_complex_samples"]))
    if not np.array_equal(np.cumsum([0]+[int(r["complex_samples"]) for r in rows]),
                          [int(r["offset_complex_samples"]) for r in rows]+[len(iq)]):
        raise ValueError(f"Invalid window offsets for {uuid}")
    parameters = source_parameters(asset, rows, files)
    arrays = np.empty((3, len(iq)), dtype=np.float32)
    results = []
    for row in rows:
        start = int(row["offset_complex_samples"])
        end = start+int(row["complex_samples"])
        power, noise, corrected = calibrate_window(row, iq, parameters)
        arrays[:, start:end] = [power.ravel(), noise.ravel(), corrected.ravel()]
        results.append(dict(row, source_packet_sha256=asset["packet_sha256"], burst_first_line_in_swath=parameters[0],
                            median_detected_sigma0=float(np.median(power)), median_noise_sigma0=float(np.median(noise)),
                            median_corrected_sigma0=float(np.median(corrected)), below_noise_pixels=int((corrected < 0).sum())))
    out = io.BytesIO()
    np.savez_compressed(out, cdse_uuid=np.array(uuid), detected_sigma0=arrays[0], noise_sigma0=arrays[1],
                        signed_corrected_sigma0=arrays[2])
    path = OUTPUT / f"{uuid}.npz"
    payload = out.getvalue()
    write(path, payload)
    digest = hashlib.sha256(payload).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise OSError(f"Failed readback for {path}")
    return (dict(asset, calibrated_packet=str(path.relative_to(ROOT)), calibrated_packet_bytes=len(payload),
                 calibrated_packet_sha256=digest, calibrated_complex_pixels=len(iq), calibrated_windows=len(rows)), results)


def main():
    assets = read(f"{INPUT}_assets")
    files = {(r["product_id"], r["local_gzip"]): r for r in read(f"{INPUT}_annotation_files")}
    windows = defaultdict(list)
    for row in read(f"{INPUT}_windows"):
        windows[row["cdse_uuid"]].append(row)
    if len(windows) != len(assets) or set(windows) != {r["cdse_uuid"] for r in assets}:
        raise ValueError("Mismatch between CEDA SLC windows and measurement assets")
    products, rows = [], []
    for asset in assets:
        result, calibrated = process(asset, windows[asset["cdse_uuid"]], files)
        products.append(result)
        rows.extend(calibrated)
    export(f"{STEM}_assets", products)
    export(f"{STEM}_windows", rows)
    summary = dict(scenes=len({r["product_id"] for r in products}), burst_polarizations=len(products),
                   windows_by_polarization=Counter(r["polarization"] for r in rows),
                   detected_power_samples=sum(r["calibrated_complex_pixels"] for r in products),
                   below_noise_pixels_by_polarization={pol: sum(r["below_noise_pixels"] for r in rows if r["polarization"] == pol)
                                                       for pol in sorted({r["polarization"] for r in rows})},
                   scope="Detected power on the native burst grid. Signed results retain values below the estimated noise. "
                         "The original I/Q packets remain the source for phase analysis.")
    payload = json.dumps(summary, indent=2)+"\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
