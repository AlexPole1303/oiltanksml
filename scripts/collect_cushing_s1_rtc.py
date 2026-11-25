"""Collect native-grid Sentinel-1 RTC intensity and audit weekly archive timing."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from tempfile import TemporaryDirectory
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import numpy as np
import rasterio
from rasterio.windows import Window

from build_cushing_icesat2_controls import identities
from build_s1_holdout_gap_burst_plan import roofs
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED, SOURCE, read_only
from collect_imagery import CONFIG, PC, sign
from collect_public import RAW, ROOT, TLS, get, write

STEM = "cushing_s1_rtc"
STORE = RAW / STEM
STAC = PC + "/api/stac/v1/collections/sentinel-1-rtc/items"
SOURCE_PREFIX = "https://sentinel1euwestrtc.blob.core.windows.net/sentinel1-grd-rtc/"
SAMPLE_M = 10
CONTEXT_M = 120
TILE = 64
BANDS = ("vv", "vh")
SIDES = ("previous", "current")
SCENE = re.compile(
    r"(S1[A-D])_IW_GRDH_1SDV_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{6})(?:_([A-F0-9]{6}))?_rtc"
)


def query_catalog(geographic_box, start, stop):
    url = STAC + "?" + urlencode(dict(bbox=",".join(map(str, geographic_box)), datetime=f"{start}/{stop}", limit=100))
    products, visited = {}, set()
    while url:
        if not url.startswith(STAC + "?") or url in visited:
            raise ValueError("Unexpected Planetary Computer pagination")
        visited.add(url)
        path = STORE / f"catalog_{len(visited):02d}.geojson"
        body = get(url, path, refresh=True)
        data = json.loads(body)
        if data["type"] != "FeatureCollection" or data["numberReturned"] != len(data["features"]):
            raise ValueError("Incomplete RTC catalog page")
        digest = hashlib.sha256(body).hexdigest()
        for f in data["features"]:
            key = f["id"]
            if not SCENE.fullmatch(key) or key in products:
                raise ValueError(f"Unexpected or repeated RTC item {key}")
            products[key] = (f, str(path.relative_to(ROOT)), digest)
        url = next((link["href"] for link in data["links"] if link["rel"] == "next"), None)
    return products


def read_headers(signed_url):
    request = Request(signed_url, method="HEAD", headers={"x-ms-version": "2021-12-02"})
    with urlopen(request, context=TLS, timeout=30) as response:
        headers = response.headers
        record = dict(source_etag=headers["ETag"], source_bytes=int(headers["Content-Length"]),
            object_created_at_utc=parsedate_to_datetime(headers["x-ms-creation-time"]),
            object_modified_at_utc=parsedate_to_datetime(headers["Last-Modified"]),
            source_version_id=headers.get("x-ms-version-id", ""), source_md5_base64=headers.get("Content-MD5", ""))
    if (not record["source_etag"] or record["source_bytes"] <= 0
            or record["object_modified_at_utc"] < record["object_created_at_utc"]):
        raise ValueError("Invalid Azure archive object metadata")
    return record


def site_tiles(tanks):
    span = SAMPLE_M*TILE
    cells = set()
    roof_bounds = {}
    for tank, r in tanks.items():
        radius = r["radius_m"]+CONTEXT_M
        x, y = r["east_m"], r["north_m"]
        bounds = (x-radius, y-radius, x+radius, y+radius)
        roof_bounds[tank] = bounds
        a, b, c, d = (math.floor(v/span) for v in bounds)
        cells.update((col, row) for col in range(a, c+1) for row in range(b, d+1))
    return np.array(sorted(cells), dtype=np.int32), roof_bounds


def read_band(item, band, cells, bounds):
    asset = item["assets"][band]
    unsigned = asset["href"]
    if not unsigned.startswith(SOURCE_PREFIX) or urlsplit(unsigned).query:
        raise ValueError(f"Unexpected RTC band host for {item['id']}/{band}")
    source_product = urlsplit(unsigned).path.split("/")[-3]
    signed = sign(unsigned)
    before = read_headers(signed)
    properties = item["properties"]
    with rasterio.Env(**CONFIG), rasterio.open(signed) as source:
        tr = source.transform
        if (source.count != 1 or source.dtypes != ("float32",) or source.nodata != -32768
                or source.crs.to_epsg() != 32614 or (tr.a, tr.b, tr.d, tr.e) != (10, 0, 0, -10)
                or any(v % SAMPLE_M for v in (tr.c, tr.f))):
            raise ValueError(f"Invalid RTC TIFF storage grid for {item['id']}/{band}")
        window = source.window(*bounds)
        if any(v != int(v) for v in window.flatten()):
            raise ValueError(f"Regional window does not align to source grid for {item['id']}/{band}")
        if window.col_off < 0 or window.row_off < 0 or window.col_off+window.width > source.width or window.row_off+window.height > source.height:
            raise ValueError(f"RTC image does not enclose regional tile grid: {item['id']}/{band}")
        data = source.read(1, window=window, masked=True)
        clipped = np.asarray(data.filled(source.nodata))
        if not np.all(np.isfinite(clipped)) or not np.any(clipped > 0):
            raise ValueError(f"Empty or nonfinite RTC data for {item['id']}/{band}")
        native_tags = source.tags()
        acquisition_tag = native_tags.get("Acquisition_DateTime", "")
        source_transform = tuple(tr)
        source_shape = tuple(source.shape)
        if abs((datetime.fromisoformat(acquisition_tag)-datetime.fromisoformat(properties["datetime"])).total_seconds()) > 1:
            raise ValueError(f"RTC TIFF acquisition timestamp differs from catalog for {item['id']}/{band}")
        parts = np.empty((len(cells), TILE, TILE), dtype=np.float32)
        for i, (x, y) in enumerate(cells):
            left = int(x)*TILE-int(bounds[0]/SAMPLE_M)
            top = int(bounds[3]/SAMPLE_M)-(int(y)+1)*TILE
            parts[i] = clipped[top:top+TILE, left:left+TILE]
    if before != read_headers(signed):
        raise ValueError(f"RTC object changed during collection: {item['id']}/{band}")
    return parts, dict(rtc_id=item["id"], band=band.upper(), unsigned_asset_url=unsigned,
        source_grd_product=source_product, **before, source_raster_shape_json=json.dumps(source_shape),
        catalog_shape_json=json.dumps(properties["proj:shape"]),
        catalog_shape_matches_raster=tuple(properties["proj:shape"]) == source_shape,
        source_raster_transform_json=json.dumps(source_transform), catalog_transform_json=json.dumps(properties["proj:transform"]),
        catalog_transform_matches_raster=tuple(properties["proj:transform"]) == source_transform,
        tiff_acquisition_timestamp=acquisition_tag)


def roof_data(cells, images, roof_bounds):
    span = SAMPLE_M*TILE
    tile_index = {tuple(cell): i for i, cell in enumerate(cells)}
    valid = np.all(images != -32768, axis=0)
    stats = []
    for tank, (left, bottom, right, top) in roof_bounds.items():
        x0, y0, x1, y1 = (math.floor(left/span), math.floor(bottom/span),
                          math.ceil(right/span), math.ceil(top/span))
        count, total = 0, 0
        for y in range(y0, y1):
            for x in range(x0, x1):
                i = tile_index[x, y]
                a = math.floor((max(left, x*span)-x*span)/SAMPLE_M)
                b = math.ceil((min(right, (x+1)*span)-x*span)/SAMPLE_M)
                c = math.floor(((y+1)*span-min(top, (y+1)*span))/SAMPLE_M)
                d = math.ceil(((y+1)*span-max(bottom, y*span))/SAMPLE_M)
                count += int(np.count_nonzero(valid[i, c:d, a:b]))
                total += (b-a)*(d-c)
        if not 0 <= count <= total or total == 0:
            raise ValueError(f"Invalid RTC context accounting for {tank}")
        stats.append(dict(candidate_id=tank, valid_dual_pol_context_pixels=count, context_pixels=total,
                          complete_context_in_both_bands=(count == total)))
    return stats


def collect(product, cells, bounds, roof_bounds):
    item, catalog_path, catalog_hash = product
    scene = item["id"]
    props = item["properties"]
    mission, first, last, orbit, take = SCENE.fullmatch(scene).groups()
    start, stop = map(datetime.fromisoformat, (props["start_datetime"], props["end_datetime"]))
    midpoint = datetime.fromisoformat(props["datetime"])
    if not (start <= midpoint <= stop and first == start.strftime("%Y%m%dT%H%M%S")
            and last == stop.strftime("%Y%m%dT%H%M%S") and int(orbit) == int(props["sat:absolute_orbit"])
            and props["platform"].upper() == f"SENTINEL-1{mission[-1]}" and props["sar:instrument_mode"] == "IW"
            and props["sar:product_type"] == "GRD" and props["proj:epsg"] == 32614
            and set(props["sar:polarizations"]) == {"VV", "VH"}):
        raise ValueError(f"Invalid RTC scene identity for {scene}")
    if take and int(take, 16) != int(props["s1:datatake_id"]):
        raise ValueError(f"RTC data take differs for {scene}")
    layers, assets = zip(*(read_band(item, pol, cells, bounds) for pol in BANDS), strict=True)
    if any(assets[0][k] != assets[1][k] for k in
           ("source_grd_product", "source_raster_transform_json", "source_raster_shape_json")):
        raise ValueError(f"RTC polarizations differ in source acquisition or grid for {scene}")
    if not all(a["object_created_at_utc"] >= stop for a in assets):
        raise ValueError(f"RTC archive creation is earlier than acquisition for {scene}")
    images = np.stack(layers)
    local = STORE / "tiles" / f"{scene}.npz"
    local.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=local.parent) as staging:
        target = local.parent / staging / local.name
        np.savez_compressed(target, power=images, tiles_xy=cells, epsg=np.int32(32614),
                            pixel_spacing_m=np.int32(SAMPLE_M), nodata=np.float32(-32768), bands=np.array(["VV", "VH"]))
        with np.load(target, allow_pickle=False) as packet:
            if not np.array_equal(images, packet["power"]) or not np.array_equal(cells, packet["tiles_xy"]):
                raise ValueError(f"RTC data failed local readback for {scene}")
        target.replace(local)
    windows = [dict(rtc_id=scene, **r) for r in roof_data(cells, images, roof_bounds)]
    return dict(rtc_id=scene, platform=mission, absolute_orbit=int(orbit),
        relative_orbit=int(props["sat:relative_orbit"]), direction=props["sat:orbit_state"],
        capture_start_utc=start, capture_end_utc=stop, datatake_id=props["s1:datatake_id"],
        rtc_ready_at_utc=max(a["object_modified_at_utc"] for a in assets), nominal_range_resolution_m=props["sar:resolution_range"],
        nominal_azimuth_resolution_m=props["sar:resolution_azimuth"], native_rtc_grid_spacing_m=SAMPLE_M,
        source_grd_product=assets[0]["source_grd_product"], used_raster_shape_over_catalog=not all(
            a["catalog_shape_matches_raster"] for a in assets), used_raster_transform_over_catalog=not all(
            a["catalog_transform_matches_raster"] for a in assets), tile_count=len(cells),
        local_tile_packet=str(local.relative_to(ROOT)), local_packet_bytes=local.stat().st_size,
        local_packet_sha256=hashlib.sha256(local.read_bytes()).hexdigest(), catalog_file=catalog_path,
        catalog_sha256=catalog_hash), assets, windows


def weekly(scenes, tank_windows, clock, optics):
    eligible = roofs()
    windows = defaultdict(set)
    for r in tank_windows:
        if r["complete_context_in_both_bands"]:
            windows[r["rtc_id"]].add(r["candidate_id"])
    pairs, weeks = [], []
    native = {r["week_ending"] for r in read("s1_grd_holdout_gap_coverage_weeks")
              if int(r["dated_roofs_with_exact_versions_pre_cutoff"]) > 0}
    for week, optical in sorted(optics.items()):
        issue = optical["release_date"]
        cutoff = datetime.fromisoformat(optical["cutoff_utc"])
        candidates = []
        for side in SIDES:
            start, stop, bound = clock[week, side]
            if cutoff != bound:
                raise ValueError(f"RTC and optical timing differ on {week}")
            candidates.append([r for r in scenes if start < r["capture_start_utc"] <= r["capture_end_utc"] <= stop])
        tanks = {r["candidate_id"]: r["terminal_id"] for r, *_ in eligible[week]}
        same_track, covered, causal = set(), set(), set()
        for before in candidates[0]:
            for after in candidates[1]:
                if (before["relative_orbit"], before["direction"]) != (after["relative_orbit"], after["direction"]):
                    continue
                ids = before["rtc_id"], after["rtc_id"]
                same_track.add(ids)
                ready = max(before["rtc_ready_at_utc"], after["rtc_ready_at_utc"])
                paired_tanks = tanks.keys() & windows[ids[0]] & windows[ids[1]]
                covered |= paired_tanks
                if ready < cutoff:
                    causal |= paired_tanks
                pairs.extend(dict(week_ending=week, release_date=issue, candidate_id=tank, terminal_id=tanks[tank],
                    previous_rtc_id=ids[0], current_rtc_id=ids[1], relative_orbit=after["relative_orbit"],
                    forecast_cutoff_utc=cutoff, both_asset_versions_modified_before_cutoff=ready<cutoff,
                    latest_required_archive_modification_utc=ready) for tank in sorted(paired_tanks))
        weeks.append(dict(week_ending=week, release_date=issue, forecast_cutoff_utc=cutoff, b04_status=optical["b04_status"],
            previous_rtc_products=len(candidates[0]), current_rtc_products=len(candidates[1]), same_track_rtc_pairs=len(same_track),
            tank_weeks_with_complete_context=len(covered), tank_weeks_with_both_versions_before_cutoff=len(causal),
            native_grd_gap_weeks_with_verified_pair=week in native))
    return pairs, weeks


def main():
    tanks, _, bbox = identities()
    with closing(read_only(SOURCE)) as db:
        clock = {(week, side): tuple(map(datetime.fromisoformat, (a, b, c))) for week, side, a, b, c in db.execute(
            "SELECT week_ending,endpoint,window_starts_after_utc,window_ends_by_utc,inputs_available_before_utc "
            "FROM eia_satellite_window WHERE study_phase='holdout'")}
    optics = {r["week_ending"]: r for r in read("holdout_optical_weeks")}
    start = min(a for a, _, _ in clock.values())
    stop = max(b for _, b, _ in clock.values())
    products = query_catalog(bbox, start.isoformat(), stop.isoformat())
    cells, roof_bounds = site_tiles(tanks)
    span = SAMPLE_M*TILE
    first = cells.min(axis=0)*span
    end = (cells.max(axis=0)+1)*span
    bounds = tuple(map(int, (*first, *end)))
    scenes, assets, windows = [], [], []
    print(f"Found {len(products)} RTC scenes and {len(cells)} regional tiles", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        pending = {pool.submit(collect, value, cells, bounds, roof_bounds): key for key, value in products.items()}
        for n, future in enumerate(as_completed(pending), 1):
            try:
                scene, bands, valid = future.result()
            except Exception as error:
                # Signed URLs from transport errors should not be persisted.
                detail = re.sub(r"https?://\S+", "[archive URL]", str(error))
                failures.append(dict(rtc_id=pending[future], type=type(error).__name__, detail=detail))
                print(f"{pending[future]}: {type(error).__name__}: {detail}", flush=True)
            else:
                scenes.append(scene)
                assets.extend(bands)
                windows.extend(valid)
            if n % 10 == 0 or n == len(pending):
                print(f"RTC scene results {n}/{len(pending)}", flush=True)
    scenes.sort(key=lambda r: (r["capture_start_utc"], r["rtc_id"]))
    pairs, weeks = weekly(scenes, windows, clock, optics)
    for name, rows in (("scenes", scenes), ("assets", sorted(assets, key=lambda r: (r["rtc_id"], r["band"]))),
                       ("windows", sorted(windows, key=lambda r: (r["rtc_id"], r["candidate_id"]))),
                       ("weekly_pairs", pairs), ("weeks", weeks)):
        export(f"{STEM}_{name}", rows)
    before = [r for r in weeks if r["tank_weeks_with_both_versions_before_cutoff"]]
    before_gaps = [r for r in before if r["b04_status"] != "measured"]
    report = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(), catalog_scenes=len(products),
        staged_scenes=len(scenes), staged_bands=len(assets), failures=failures, registered_roof_contexts=len(tanks),
        staged_scene_counts_by_platform=Counter(r["platform"] for r in scenes), native_tiles_per_scene=len(cells),
        local_packets_bytes=sum(r["local_packet_bytes"] for r in scenes),
        scenes_with_different_catalog_geotransform=sum(r["used_raster_transform_over_catalog"] for r in scenes),
        scenes_with_different_catalog_shape=sum(r["used_raster_shape_over_catalog"] for r in scenes),
        source_objects_modified_after_creation=sum(r["object_modified_at_utc"] > r["object_created_at_utc"] for r in assets),
        all_report_weeks=len(weeks), all_weeks_with_catalog_and_valid_same_track_pairs=sum(bool(r["tank_weeks_with_complete_context"]) for r in weeks),
        weeks_with_both_rtc_versions_before_cutoff=len(before), optical_gap_weeks_with_both_versions_before_cutoff=len(before_gaps),
        optical_gap_tank_weeks_with_both_versions_before_cutoff=sum(r["tank_weeks_with_both_versions_before_cutoff"] for r in before_gaps),
        optical_gap_weeks_with_rtc_and_without_verified_native_grd=[r["week_ending"] for r in before_gaps
                                                                  if not r["native_grd_gap_weeks_with_verified_pair"]],
        interpretation="RTC contains terrain-corrected GRD backscatter intensity. The grid is 10 m, with nominal "
        "source resolution 20 m by 22 m. The archived object Last-Modified is checked for both polarizations "
        "at both measurement endpoints. This is a radar covariate, without an inferred roof height or stock volume.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} RTC products still need collection")


if __name__ == "__main__":
    main()
