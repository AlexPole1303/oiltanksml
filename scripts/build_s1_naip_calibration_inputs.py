"""Link the 2019 aerial shadow controls to native GRD measurements and viewing geometry."""
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from statistics import median

import numpy as np
from pyproj import Geod

from audit_s1_grd_native_inputs import saved_assets
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import ROOT, write
from collect_s1_grd_roof_inputs import STORE

GEO = Geod(ellps="WGS84")
FIELDS = ("incidenceAngle", "slantRangeTime", "longitude", "latitude", "height")


def native_grid(path):
    root = ET.parse(path).getroot()
    grouped = defaultdict(list)
    for node in root.findall("./geolocationGrid/geolocationGridPointList/geolocationGridPoint"):
        grouped[int(node.findtext("line"))].append((int(node.findtext("pixel")), *(float(node.findtext(k)) for k in FIELDS)))
    lines = sorted(grouped)
    records = np.array([sorted(grouped[line]) for line in lines], dtype=float)
    if len(lines) < 2 or records.shape[1] < 2 or not np.all(records[:, :, 0] == records[0, :, 0]):
        raise ValueError(f"Annotation geolocation grid is missing or nonrectangular for {path}")
    return np.asarray(lines), records[0, :, 0], records[:, :, 1:]


def interpolate(grid, line, pixel):
    lines, pixels, values = grid
    if not lines[0] <= line <= lines[-1] or not pixels[0] <= pixel <= pixels[-1]:
        raise ValueError(f"Native radar location outside annotation grid {line}, {pixel}")
    row = min(int(np.searchsorted(lines, line, side="right")) - 1, len(lines) - 2)
    col = min(int(np.searchsorted(pixels, pixel, side="right")) - 1, len(pixels) - 2)
    dy = (line - lines[row]) / (lines[row + 1] - lines[row])
    dx = (pixel - pixels[col]) / (pixels[col + 1] - pixels[col])
    left = values[row, col] * (1 - dy) + values[row + 1, col] * dy
    right = values[row, col + 1] * (1 - dy) + values[row + 1, col + 1] * dy
    return left * (1 - dx) + right * dx


def main():
    aerial = read("naip_2019_roof_shadow_reference")
    product_ids = {r["radar_product_id"] for r in aerial if r["radar_product_id"]}
    if len(product_ids) != 1:
        raise ValueError("Expected one verified radar acquisition for the aerial calibration set")
    product_id, = product_ids
    candidate = next(r for r in aerial if r["radar_product_id"])
    product = next(r for r in read("s1_grd_native_pilot_manifest") if r["product_id"] == candidate["radar_product_id"])
    manifest = json.loads((STORE / product["product_name"].removesuffix(".SAFE") / "native_manifest.json").read_bytes())
    clocks, _ = saved_assets([manifest])
    native = {(r["candidate_id"], r["polarization"]): r for r in read("s1_grd_native_window_index") if r["product_id"] == product_id}
    vv_annotation = manifest["objects"]["annotation/iw-vv.xml"]["local_file"]
    grid = native_grid(ROOT / vv_annotation)
    rows = []
    for photo in aerial:
        tank = photo["candidate_id"]
        vv, vh = (native[tank, pol] for pol in ("VV", "VH"))
        if any(vv[k] != vh[k] for k in ("native_line", "native_sample")):
            raise ValueError(f"The radar polarization grids differ for {tank}")
        line, pixel = float(vv["native_line"]), float(vv["native_sample"])
        incidence, slant, lon, lat, terrain = interpolate(grid, line, pixel)
        _, _, distance = GEO.inv(lon, lat, float(photo["longitude"]), float(photo["latitude"]))
        begin, end = (datetime.fromisoformat(photo[k]) for k in ("start_utc", "end_exclusive_utc"))
        sar_start, sar_end = (datetime.fromisoformat(product[k]) for k in ("capture_start_utc", "capture_end_utc"))
        if sar_start < end:
            raise ValueError("Expected the radar capture after the aerial photography")
        valid = vv["centre_pixel_valid"] == vh["centre_pixel_valid"] == "True"
        qualified = (photo["documented_roof_type"] == "EFR" and valid
                     and photo["shadow_reference_status"] == "resolved_aerial_shadow")
        rows.append(dict(candidate_id=tank, terminal_id=photo["documented_terminal_id"],
            roof_type=photo["documented_roof_type"], aerial_reference_status=photo["shadow_reference_status"],
            qualified_aerial_shadow_reference=qualified, aerial_gap_m=photo["best_gap_m"] if qualified else None,
            aerial_gap_low_m=photo["near_gap_min_m"] if qualified else None,
            aerial_gap_high_m=photo["near_gap_max_m"] if qualified else None,
            aerial_raster=photo["source_raster"], aerial_interval_start_utc=photo["start_utc"],
            aerial_interval_end_exclusive_utc=photo["end_exclusive_utc"],
            minimum_aerial_to_radar_hours=(sar_start - end).total_seconds() / 3600,
            maximum_aerial_to_radar_hours=(sar_end - begin).total_seconds() / 3600,
            radar_product_id=product_id, radar_product_name=product["product_name"],
            radar_archive_inputs_ready_utc=clocks[product_id].isoformat(), radar_start_utc=sar_start.isoformat(),
            radar_end_utc=sar_end.isoformat(), approximate_native_line=line, approximate_native_sample=pixel,
            nominal_rim_diameter_m=2 * float(photo["radius"]),
            incidence_angle_deg=float(incidence), two_way_slant_range_time_s=float(slant), terrain_height_m=float(terrain),
            annotation_geolocation_difference_m=distance, valid_native_centre_in_both_polarizations=valid,
            source_annotation=vv_annotation, vv_window_file=vv["window_file"], vh_window_file=vh["window_file"],
            window_row_offset=vv["window_row_offset"], window_col_offset=vv["window_col_offset"]))
    accepted = [r for r in rows if r["qualified_aerial_shadow_reference"]]
    report = dict(radar_product_id=product_id, linked_contours=len(rows),
        efr_contours=sum(r["roof_type"] == "EFR" for r in rows),
        covered_roof_controls=sum(r["roof_type"] != "EFR" for r in rows),
        qualified_efr_shadow_references=len(accepted),
        median_accepted_incidence_angle_deg=median(r["incidence_angle_deg"] for r in accepted),
        median_accepted_rim_diameter_m=median(r["nominal_rim_diameter_m"] for r in accepted),
        median_annotation_geolocation_difference_m=median(r["annotation_geolocation_difference_m"] for r in rows),
        radar_archive_inputs_ready_utc=clocks[product_id].isoformat(),
        interpretation="The package links two native radar polarizations with provisional aerial shadow references. "
        "Viewing geometry is interpolated on the product's native annotation grid. Valid samples do not imply a radar "
        "roof estimate. The exact local AWS copy is retrospective, and the acquisition gap admits physical roof movement.")
    export("s1_naip_2019_calibration_inputs", rows)
    write(DERIVED / "s1_naip_2019_calibration_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
