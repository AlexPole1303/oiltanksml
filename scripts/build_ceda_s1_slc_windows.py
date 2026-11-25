"""Map dated Cushing roofs to native Sentinel-1 SLC rows and columns."""

import csv
import gzip
import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from email.utils import parsedate_to_datetime
from functools import cache

import numpy as np
from pyproj import Transformer
from scipy.constants import c
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq

from build_cushing_icesat2_controls import identities
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_ceda_s1_slc_annotations import utc
from collect_public import ROOT, write

STEM = "ceda_s1_slc_windows"
UTM_TO_ECEF = Transformer.from_crs(32614, 4978, always_xy=True).transform
GEO_TO_ECEF = Transformer.from_crs(4979, 4978, always_xy=True).transform
CONTEXT_M = 120
SIDES = ("previous", "current")


class NativeGeometry:
    def __init__(self, row):
        root = ET.fromstring(gzip.decompress((ROOT / row["local_annotation_gzip"]).read_bytes()))
        nodes = root.findall("./generalAnnotation/orbitList/orbit")
        if len(nodes) < 4 or any(r.findtext("frame") != "Earth Fixed" for r in nodes):
            raise ValueError(f"Invalid embedded orbit for {row['cdse_uuid']}")
        self.origin = utc(nodes[0].findtext("time"))
        self.seconds = np.array([self.seconds_from_origin(r.findtext("time")) for r in nodes])
        vectors = [np.array([[float(r.findtext(f"{key}/{axis}")) for axis in "xyz"] for r in nodes])
                   for key in ("position", "velocity")]
        self.position = CubicHermiteSpline(self.seconds, *vectors, extrapolate=False)
        self.velocity = self.position.derivative()
        self.start = self.seconds_from_origin(row["slc_burst_azimuth_time_utc"])
        self.dt = float(row["native_azimuth_interval_seconds"])
        self.dr = float(row["native_range_pixel_spacing_m"])
        self.first_range = c/2 * float(row["native_slant_range_first_sample_seconds"])
        grid = root.findall("./geolocationGrid/geolocationGridPointList/geolocationGridPoint")
        errors = []
        for node in grid:
            when = self.seconds_from_origin(node.findtext("azimuthTime"))
            lon, lat, height = (float(node.findtext(k)) for k in ("longitude", "latitude", "height"))
            t, sample = self.locate(GEO_TO_ECEF(lon, lat, height), when)
            errors.append([abs(t-when)/self.dt, abs(sample-float(node.findtext("pixel")))])
        if not errors:
            raise ValueError(f"Missing native geolocation grid for {row['cdse_uuid']}")
        self.grid_errors = np.max(errors, axis=0)
        self.grid_points = len(grid)
        with gzip.open(ROOT / row["valid_mask_gzip"], "rt") as handle:
            records = list(csv.DictReader(handle))
        self.first_valid = np.array([int(r["first_valid_sample"]) for r in records])
        self.last_valid = np.array([int(r["last_valid_sample"]) for r in records])

    def seconds_from_origin(self, value):
        return (utc(value)-self.origin).total_seconds()

    def locate(self, coordinates, near):
        target = np.asarray(coordinates)
        root = brentq(lambda t: np.dot(self.position(t)-target, self.velocity(t)),
                      max(self.seconds[0], near-2), min(self.seconds[-1], near+2), xtol=1e-9)
        distance = np.linalg.norm(self.position(root)-target)
        return root, (distance-self.first_range)/self.dr

    def window(self, tank, near):
        near = self.seconds_from_origin(near)
        east, north = (tank[k] for k in ("east_m", "north_m"))
        ground, rim = (tank[k] for k in ("ground_hae_approx_m", "top_hae_approx_m"))
        when, center_sample = self.locate(UTM_TO_ECEF(east, north, rim), near)
        _, ground_sample = self.locate(UTM_TO_ECEF(east, north, ground), near)
        radius = tank["radius_m"]+CONTEXT_M
        envelope = []
        # The square encloses the circular context and both ends of the physical height range.
        for x in (east-radius, east+radius):
            for y in (north-radius, north+radius):
                for height in (ground, rim):
                    t, sample = self.locate(UTM_TO_ECEF(x, y, height), near)
                    envelope.append(((t-self.start)/self.dt, sample))
        bounds = np.array(envelope)
        row0, col0 = (math.floor(x)-2 for x in bounds.min(axis=0))
        row1, col1 = (math.ceil(x)+3 for x in bounds.max(axis=0))
        center_line = (when-self.start)/self.dt
        if 0 <= row0 < row1 <= len(self.first_valid) and col0 >= 0:
            valid = bool(np.all((self.first_valid[row0:row1] >= 0) & (self.first_valid[row0:row1] <= col0)
                                & (self.last_valid[row0:row1] >= col1-1)))
        else:
            valid = False
        return dict(rim_center_burst_line=center_line, rim_center_sample=center_sample, ground_center_sample=ground_sample,
                    first_line=row0, end_line_exclusive=row1, first_sample=col0, end_sample_exclusive=col1,
                    full_window_valid_by_native_mask=valid)


def main():
    tanks, _, _ = identities()
    annotations = {r["cdse_uuid"]: r for r in read("ceda_s1_slc_annotations_bursts")}
    raw = {r["cdse_uuid"]: r for r in read("ceda_s1_slc_samples_bursts")}
    if not raw.keys() <= annotations.keys():
        raise ValueError("A local radar burst is missing its matching native annotation")
    orbits = {(r["week_ending"], r["candidate_id"], r["previous_product_id"], r["current_product_id"],
              r["swath"], r["polarization"], r["common_relative_burst_id"]): r
              for r in read("s1_slc_orbit_geometry_pairs")}

    @cache
    def geometry(burst):
        return NativeGeometry(annotations[burst])

    windows = {}
    for r in read("s1_holdout_gap_burst_tank_pairs"):
        if r["full_context_in_both_bursts"] != "True":
            continue
        key = tuple(r[k] for k in ("week_ending", "candidate_id", "previous_product_id", "current_product_id",
                                  "swath", "polarization", "common_relative_burst_id"))
        precise = orbits[key]
        tank = r["candidate_id"]
        for side in SIDES:
            burst = r[f"{side}_cdse_uuid"]
            if burst not in annotations or (burst, tank) in windows:
                continue
            model = geometry(burst)
            source = raw.get(burst, {})
            values = model.window(tanks[tank], precise[f"{side}_rim_zero_doppler_at_utc"])
            windows[burst, tank] = dict(cdse_uuid=burst, candidate_id=tank, terminal_id=r["terminal_id"],
                product_id=r[f"{side}_product_id"], swath=r["swath"], polarization=r["polarization"],
                source_radar_bytes_file=source.get("source_radar_bytes_file", ""),
                local_radar_samples=bool(source), **values)

    used_bursts = {burst for burst, _ in windows}
    checks = [dict(cdse_uuid=burst, product_id=r["product_id"], product_name=r["product_name"],
        geolocation_grid_points=geometry(burst).grid_points, max_abs_native_grid_line_error=geometry(burst).grid_errors[0],
        max_abs_native_grid_sample_error=geometry(burst).grid_errors[1]) for burst, r in annotations.items()
        if burst in used_bursts]
    # The grid is generated from the same product model and verifies indexing conventions only.
    if any(max(r["max_abs_native_grid_line_error"], r["max_abs_native_grid_sample_error"]) > .05 for r in checks):
        raise ValueError("Native line/sample geometry disagrees with embedded product grid")
    pairs = []
    by_week = defaultdict(set)
    for row in read("s1_holdout_gap_burst_tank_pairs"):
        if row["full_context_in_both_bursts"] != "True":
            continue
        uuids = [row[f"{side}_cdse_uuid"] for side in SIDES]
        match = [windows.get((burst, row["candidate_id"]), {}) for burst in uuids]
        if all(v.get("full_window_valid_by_native_mask") and v["local_radar_samples"] for v in match):
            key = tuple(row[k] for k in ("week_ending", "candidate_id", "previous_product_id", "current_product_id",
                                       "swath", "polarization", "common_relative_burst_id"))
            cutoff = utc(orbits[key]["forecast_cutoff_utc"])
            modified = [parsedate_to_datetime(raw[burst]["source_zip_last_modified_http"]) for burst in uuids]
            published = [utc(row[f"{side}_catalog_publication_utc"]) for side in SIDES]
            pairs.append(dict(week_ending=row["week_ending"], candidate_id=row["candidate_id"],
                terminal_id=row["terminal_id"], forecast_cutoff_utc=cutoff, previous_cdse_uuid=uuids[0],
                current_cdse_uuid=uuids[1], previous_ceda_zip_last_modified_utc=modified[0],
                current_ceda_zip_last_modified_utc=modified[1], previous_cdse_parent_publication_utc=published[0],
                current_cdse_parent_publication_utc=published[1], both_archive_object_clocks_before_cutoff=max(modified)<cutoff,
                both_parent_catalog_clocks_before_cutoff=max(published)<cutoff))
            by_week[row["week_ending"]].add(row["candidate_id"])
    export(f"{STEM}_grid_checks", checks)
    export(f"{STEM}_inventory", list(windows.values()))
    export(f"{STEM}_pairs", pairs)
    summary = dict(verified_native_annotation_grids=len(checks), native_context_beyond_roof_m=CONTEXT_M,
        highest_grid_line_error=max(r["max_abs_native_grid_line_error"] for r in checks),
        highest_grid_sample_error=max(r["max_abs_native_grid_sample_error"] for r in checks),
        unique_burst_tank_windows=len(windows),
        full_windows_by_mask=sum(r["full_window_valid_by_native_mask"] for r in windows.values()),
        full_windows_with_local_iq=sum(r["local_radar_samples"] and r["full_window_valid_by_native_mask"] for r in windows.values()),
        weeks_with_full_native_radar_pairs=len(by_week),
        tank_weeks_with_full_native_radar_pairs=sum(map(len, by_week.values())),
        tank_weeks_with_both_archive_and_catalog_clocks_before_cutoff=len({(r["week_ending"], r["candidate_id"])
            for r in pairs if r["both_archive_object_clocks_before_cutoff"] and r["both_parent_catalog_clocks_before_cutoff"]}),
        full_native_radar_pairs_by_week={w: len(ids) for w, ids in sorted(by_week.items())},
        interpretation="The native SAFE orbit maps each roof to radar coordinates. The internal geolocation grid "
        "checks indexing, not independent position or roof height accuracy. A window covers the entire projected "
        "context and height range under the annotation model and passed the valid sample mask on every line. "
        "HTTP Last-Modified and catalog publication are archive evidence, without a historical delivery guarantee. "
        "No oil level or stock estimate is inferred.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
