"""Archive source-versioned GEDI v2 shots and screen their footprints over Cushing."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from csv import DictReader
from datetime import datetime, timedelta, timezone
from io import RawIOBase
from urllib.request import Request, urlopen

import numpy as np
import pyarrow.compute as pc
from pyarrow.parquet import ParquetFile
from scipy.spatial import cKDTree
from shapely.geometry import box, shape

from build_cushing_icesat2_controls import TO_UTM, ground_heights, identities, original_access, other_structure_context
from build_cushing_icesat2_optical_controls import query_catalog
from build_cushing_nasa_satellite_inventory import BBOX
from build_s1_reference_matches import read
from build_weekly_shadow_panel import DERIVED
from collect_public import RAW, ROOT, TLS, get, write, write_csv


STEM = "cushing_gedi_olm"
STORE = RAW / STEM
CATALOG = "https://s3.eu-central-1.wasabisys.com/stac/openlandmap/GEDI02/stac_items_geojson"
OBJECT_ROOT = "https://s3.opengeohub.org/global/glidar/gedi-ard/level2/"
GEDI_EPOCH = datetime(2018, 1, 1, tzinfo=timezone.utc)
FOOTPRINT_RADIUS_M = 12.5
EDGE_GUARD_M = 10
GROUND_CONTEXT_M = 300
SOURCE_FIELDS = ("delta_time", "beamname", "shotnumber", "latitude", "longitude", "elev_lowestmode",
                 "rh100", "rh99", "rh95", "sensitivity", "night_flag", "num_detectedmodes",
                 "selected_l2a_algorithm", "surface_flag", "l2b_quality_flag")


class RemoteParquet(RawIOBase):
    def __init__(self, url):
        self.url = url
        with urlopen(Request(url, method="HEAD"), context=TLS, timeout=25) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers["ETag"]
            self.modified = response.headers["Last-Modified"]
        self.position = 0
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (0, self.position, self.size)[whence] + offset
        if position < 0:
            raise ValueError("Cannot seek before start of Parquet object")
        self.position = position
        return position

    def read(self, size=-1):
        available = self.size - self.position
        size = min(size, available) if size >= 0 else available
        if size <= 0:
            return b""
        end = self.position + size - 1
        expected = f"bytes {self.position}-{end}/{self.size}"
        request = Request(self.url, headers={"Range": f"bytes={self.position}-{end}", "If-Match": self.etag,
                                             "Accept-Encoding": "identity"})
        with urlopen(request, context=TLS, timeout=45) as response:
            if (response.status != 206 or response.headers["Content-Range"] != expected
                    or response.headers["ETag"] != self.etag):
                raise ValueError(f"GEDI source version or byte range differs for {self.url}")
            data = response.read()
        if len(data) != size:
            raise ValueError(f"Incomplete GEDI source byte range for {self.url}")
        self.position += size
        self.bytes_read += size
        return data


def export(path, rows, fields=()):
    fields = tuple(dict.fromkeys(k for row in rows for k in row)) or fields
    write_csv(path, fields, ([r.get(k) for k in fields] for r in rows))


def archive_tile(item, bounds):
    url = item["asset_file"]
    if not url.startswith(OBJECT_ROOT):
        raise ValueError(f"Unexpected source for {item['item_id']}")
    year = int(item["year"])
    west, south, east, north = bounds
    shots = []
    with RemoteParquet(url) as remote:
        data = ParquetFile(remote, pre_buffer=True)
        if (data.metadata.num_rows != int(item["point_counts"]) or remote.size != round(item["size_in_mb"] * 2**20)
                or not set(SOURCE_FIELDS) <= set(data.schema.names)):
            raise ValueError(f"GEDI schema, size or row count differs for {item['item_id']}")
        for batch in data.iter_batches(columns=SOURCE_FIELDS):
            lon, lat = (batch.column(batch.schema.get_field_index(name)) for name in ("longitude", "latitude"))
            mask = pc.and_(pc.and_(pc.greater_equal(lon, west), pc.less_equal(lon, east)),
                           pc.and_(pc.greater_equal(lat, south), pc.less_equal(lat, north)))
            for row in batch.filter(mask).to_pylist():
                capture = GEDI_EPOCH + timedelta(seconds=row["delta_time"])
                if capture.year != year:
                    raise ValueError(f"Unexpected capture time for shot {row['shotnumber']}")
                shots.append(dict(row, approximate_capture_utc=capture.isoformat(), source_tile_id=item["item_id"]))
        version = dict(source_tile_id=item["item_id"], year=year, source_url=url, source_bytes=remote.size,
            source_rows=data.metadata.num_rows, source_etag=remote.etag, source_last_modified=remote.modified,
            bytes_read_by_range=remote.bytes_read, local_shot_count=len(shots),
            bounding_box_wgs84_json=json.dumps(bounds), source_first_shot_utc=item["start_date"],
            source_last_shot_utc=item["end_date"])
    shots.sort(key=lambda r: (r["delta_time"], r["shotnumber"]))
    for i, row in enumerate(shots, 1):
        row["extract_row"] = i
    path = STORE / "shots" / f"{item['item_id']}.csv"
    export(path, shots, (*SOURCE_FIELDS, "approximate_capture_utc", "source_tile_id", "extract_row"))
    return dict(version, shot_extract_path=str(path.relative_to(ROOT)),
                shot_extract_sha256=hashlib.sha256(path.read_bytes()).hexdigest()), shots


def roof_candidates(shots, tanks):
    longitude, latitude = ([r[k] for r in shots] for k in ("longitude", "latitude"))
    x, y = TO_UTM.transform(longitude, latitude)
    xy = np.column_stack((x, y))
    tree = cKDTree(xy)
    dems = read("cushing_icesat2_ground_assets")
    centres, radii = other_structure_context()
    structures = cKDTree(centres)
    for t in tanks.values():
        centre = np.array([t["east_m"], t["north_m"]])
        radius = t["radius_m"]
        for i in tree.query_ball_point(centre, radius + FOOTPRINT_RADIUS_M):
            shot = shots[i]
            if t["first_seen"] > shot["approximate_capture_utc"][:10]:
                continue
            edge = radius - float(np.linalg.norm(xy[i] - centre))
            above_ground = (shot["elev_lowestmode"] / 100 - t["ground_hae_approx_m"]
                            if shot["elev_lowestmode"] != -2147483648 else None)
            plausible = above_ground is not None and 0 <= above_ground <= t["top_hae_approx_m"] - t["ground_hae_approx_m"]
            guard = edge >= FOOTPRINT_RADIUS_M + EDGE_GUARD_M
            ground_count, bias, ground_spread = local_ground(shot, shots, centre, xy, tree, centres, radii, structures,
                                                             dems, t["geoid18_interpolated_m"]) if guard else (0, None, None)
            depth = t["top_hae_approx_m"] - shot["elev_lowestmode"] / 100 if above_ground is not None else None
            yield dict(shot, candidate_id=t["candidate_id"], terminal_id=t["terminal_id"],
                first_seen=t["first_seen"], lidar_source=t["geometry_source_directory"],
                centre_to_roof_edge_m=edge, roof_radius_m=radius, nominal_footprint_radius_m=FOOTPRINT_RADIUS_M,
                nominal_footprint_inside_roof=edge >= FOOTPRINT_RADIUS_M,
                footprint_inside_with_10m_margin=guard,
                roof_above_lidar_ground_approx_m=above_ground, plausible_above_ground_and_below_rim=plausible,
                lidar_wall_ellipsoid_approx_m=t["top_hae_approx_m"], wall_to_lowest_mode_approx_m=depth,
                same_beam_ground_candidates=ground_count, local_ground_bias_m=bias, local_ground_p90_p10_m=ground_spread,
                corrected_wall_to_lowest_mode_approx_m=depth + bias if depth is not None and bias is not None else None)


def local_ground(shot, shots, centre, xy, tree, centres, radii, structures, dems, geoid):
    exterior = np.array([i for i in tree.query_ball_point(centre, GROUND_CONTEXT_M)
        if shots[i]["beamname"] == shot["beamname"] and abs(shots[i]["delta_time"] - shot["delta_time"]) <= 2
        and shots[i]["elev_lowestmode"] != -2147483648], dtype=int)
    for i in structures.query_ball_point(centre, GROUND_CONTEXT_M + max(radii) + FOOTPRINT_RADIUS_M + EDGE_GUARD_M):
        exterior = exterior[np.linalg.norm(xy[exterior] - centres[i], axis=1) > radii[i] + FOOTPRINT_RADIUS_M + EDGE_GUARD_M]
    if not len(exterior):
        return 0, None, None
    ground = ground_heights(xy[exterior], dems)
    offsets = np.array([shots[i]["elev_lowestmode"] / 100 for i in exterior]) - ground - geoid
    offsets = offsets[np.isfinite(offsets) & (abs(offsets) <= 2)]
    if len(offsets) < 3:
        return len(offsets), None, None
    p10, median, p90 = np.percentile(offsets, [10, 50, 90])
    return len(offsets), float(median), float(p90 - p10)


def optical_comparisons(shots):
    ids = {r["candidate_id"] for r in shots}
    estimates = []
    for phase in ("development", "validation", "holdout"):
        with (DERIVED / f"shadow_timing_{phase}_observations.csv").open(newline="") as source:
            estimates.extend((phase, r) for r in DictReader(source) if r["candidate_id"] in ids
                and r["geometry_mode"] == "dated_2011_lidar" and r["fit_status"] == "resolved_in_geometry_diagnostic")
    for shot in shots:
        capture = datetime.fromisoformat(shot["approximate_capture_utc"])
        for phase, r in estimates:
            if shot["candidate_id"] != r["candidate_id"]:
                continue
            hours = (abs((datetime.fromisoformat(r["captured_at_utc"]) - capture).total_seconds()) + 1) / 3600
            if hours > 48:
                continue
            yield dict(candidate_id=r["candidate_id"], source_tile_id=shot["source_tile_id"], extract_row=shot["extract_row"],
                gedi_approximate_capture_utc=shot["approximate_capture_utc"], max_separation_hours=hours, phase=phase,
                sentinel2_version_id=r["version_id"], sentinel2_observation_id=r["observation_id"],
                sentinel2_captured_at_utc=r["captured_at_utc"], sentinel2_gap_m=float(r["estimated_gap_m"]),
                gedi_wall_to_lowest_mode_approx_m=shot["wall_to_lowest_mode_approx_m"],
                gedi_local_ground_bias_m=shot["local_ground_bias_m"],
                sentinel2_minus_gedi_uncorrected_m=float(r["estimated_gap_m"]) - shot["wall_to_lowest_mode_approx_m"],
                sentinel2_tile_metadata_path=r["tile_metadata_path"])


def archive_matches(shots):
    catalogue = read("cushing_gedi_catalog")
    for shot in shots:
        capture = datetime.fromisoformat(shot["approximate_capture_utc"])
        orbit = f"O{shot['shotnumber'] // 10**13:05d}"
        for row in catalogue:
            if orbit not in row["gedi_acquisition_key"].split("_"):
                continue
            start, end = (datetime.fromisoformat(row[k]) for k in ("l2a_capture_start_utc", "l2a_capture_end_utc"))
            if capture + timedelta(seconds=1) < start or capture > end:
                continue
            yield dict(candidate_id=shot["candidate_id"], source_tile_id=shot["source_tile_id"], extract_row=shot["extract_row"],
                gedi_approximate_capture_utc=shot["approximate_capture_utc"], latitude=shot["latitude"], longitude=shot["longitude"],
                source_shotnumber_rounded=shot["shotnumber"], beamname=shot["beamname"], **row)


def check_sources(matches):
    granules = {r["gedi_acquisition_key"]: r for r in matches}
    assets = [dict(gedi_acquisition_key=key, level=level, cmr_id=r[f"{level}_concept_id"],
                   hdf5_url=r[f"{level}_hdf5_url"]) for key, r in granules.items() for level in ("l2a", "l1b")]
    optical = []
    for key, r in granules.items():
        shot_times = [datetime.fromisoformat(s["gedi_approximate_capture_utc"]) for s in matches if s["gedi_acquisition_key"] == key]
        optical.append((key, {key: dict(granule_start_utc=min(shot_times).isoformat(),
                                       granule_end_utc=(max(shot_times) + timedelta(seconds=1)).isoformat())}))
    with ThreadPoolExecutor(max_workers=5) as pool:
        checks = list(pool.map(lambda r: r | original_access(r), assets))
        scenes = list(pool.map(lambda job: query_catalog(*job, store=STORE / "sentinel2_catalog"), optical))
    return checks, [dict(gedi_acquisition_key=r["icesat_granule"], search_start_utc=r["search_start_utc"],
                        search_end_utc=r["search_end_utc"], archives_agree=r["archives_agree"],
                        unique_sentinel2_datatakes=len(versions), **{k: v for k, v in r.items()
                            if k.startswith(("copernicus_", "earth_search_"))}) for r, versions in scenes]


def main():
    bounds = tuple(map(float, BBOX.split(",")))
    source = STORE / "openlandmap_tile_index.geojson"
    payload = get(CATALOG, source)
    area = box(*bounds)
    items = sorted((r["properties"] for r in json.loads(payload)["features"] if shape(r["geometry"]).intersects(area)),
                   key=lambda r: (r["year"], r["item_id"]))
    if not items:
        raise ValueError("OpenLandMap GEDI has no indexed tiles for Cushing")
    tanks, _, _ = identities()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda item: archive_tile(item, bounds), items))
    versions, shots = zip(*results)
    shots = [r for group in shots for r in group]
    position_keys = {(r["delta_time"], r["beamname"], r["latitude"], r["longitude"]) for r in shots}
    if len(position_keys) != len(shots):
        raise ValueError("OpenLandMap tile extracts contain duplicate GEDI time and coordinates")
    candidates = sorted(roof_candidates(shots, tanks), key=lambda r: (r["delta_time"], r["candidate_id"], r["shotnumber"]))
    interior = [r for r in candidates if r["nominal_footprint_inside_roof"]]
    guarded = [r for r in interior if r["footprint_inside_with_10m_margin"]]
    plausible = [r for r in guarded if r["plausible_above_ground_and_below_rim"]]
    optical = sorted(optical_comparisons(plausible), key=lambda r: r["max_separation_hours"])
    original_matches = list(archive_matches(guarded))
    original_checks, optical_catalog = check_sources(original_matches)
    export(DERIVED / f"{STEM}_sources.csv", versions)
    export(DERIVED / f"{STEM}_roof_candidates.csv", candidates, ("shotnumber", "candidate_id"))
    export(DERIVED / f"{STEM}_optical_comparisons.csv", optical, ("candidate_id", "source_tile_id", "extract_row"))
    export(DERIVED / f"{STEM}_v003_original_candidates.csv", original_matches, ("candidate_id", "source_tile_id", "extract_row"))
    export(DERIVED / f"{STEM}_v003_original_access.csv", original_checks, ("gedi_acquisition_key", "level"))
    export(DERIVED / f"{STEM}_sentinel2_catalog.csv", optical_catalog, ("gedi_acquisition_key",))
    summary = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), source_index_url=CATALOG,
        source_index_file=str(source.relative_to(ROOT)), source_index_sha256=hashlib.sha256(payload).hexdigest(),
        bounding_box_wgs84=list(bounds), tiles=len(versions), source_rows=sum(r["source_rows"] for r in versions),
        bytes_read_by_range=sum(r["bytes_read_by_range"] for r in versions), local_shots=len(shots),
        local_shots_by_year=Counter(r["approximate_capture_utc"][:4] for r in shots),
        rows_beyond_unique_rounded_source_shotnumbers=len(shots)-len({r["shotnumber"] for r in shots}),
        roof_footprint_intersections=len(candidates), nominal_footprints_inside_roof=len(interior),
        footprints_inside_roof_with_10m_margin=len(guarded), distinct_guarded_roofs=len({r["candidate_id"] for r in guarded}),
        guarded_with_plausible_ellipsoid_elevation=len(plausible),
        guarded_with_local_ground_bias=sum(r["local_ground_bias_m"] is not None for r in guarded),
        sentinel2_comparisons_within_48h=len(optical), sentinel2_comparisons_within_24h=sum(r["max_separation_hours"] <= 24 for r in optical),
        guarded_v003_granule_candidates=len(original_matches), guarded_distinct_v003_granules=len(optical_catalog),
        guarded_v003_files_requiring_earthdata=sum(r["earthdata_signin"] for r in original_checks),
        guarded_granules_with_sentinel2_catalog_candidates=sum(r["unique_sentinel2_datatakes"] > 0 for r in optical_catalog),
        interpretation="Pre-filtered and scaled OpenLandMap GEDI02 v002. Capture time is quantized to one second. "
        "The public pipeline converts the shot number through a floating-point array. Use source tile and row for identity, "
        "and orbit, beam, position and time for candidate matches to originals. "
        "Nominal geometric containment and plausible L2A elevations are screening results. No L1B waveforms or validated roof heights.")
    report = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", report.encode())
    print(report, end="", flush=True)


if __name__ == "__main__":
    main()
