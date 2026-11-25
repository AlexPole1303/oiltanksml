"""Build dated NAIP roof estimates and compare them with nearby satellite measurements."""

import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from datetime import datetime, timezone
from functools import lru_cache
from statistics import median

import rasterio
from pyproj import Transformer
from shapely import make_valid, transform
from shapely.geometry import Point, Polygon, shape

from audit_cushing_landsat_catalog_recovery import CORNERS, PRODUCT, collect_recovery_crop, local_geometry
from audit_cushing_landsat8_rapid_height_controls import passed as rapid_passed
from audit_cushing_usgs_b8_metadata import STEM as BULK
from audit_shadow_precision import describe
from audit_s2_sun_clock import xml_time
from build_l1c_quality import QUALITY_FIELDS, scene_rows
from build_cross_sensor_shadow import published_lsf
from build_cushing_recovered_landsat_b8 import scene as recovered_scene
from build_landsat_c2_b8_measurements import prepare_native
from build_landsat_esri_b8_controls import fit_tile
from build_landsat_pan_roof_angles import prepare_tool
from build_naip_2019_roof_reference import SOURCE_SQL, find_asset, fit
from build_naip_2025_roof_reference import RESOLVED as AERIAL_RESOLVED, SPA_SHA256, solar_module
from build_s1_reference_matches import export, read
from build_terminal_development_measurements import DATED
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, read_only
from build_weekly_shadow_validation import legacy_geometry
from collect_cushing_gedi_optical_windows import fit_geometry, kernels
from collect_landsat_esri_b8 import query
from collect_public import RAW, ROOT, write
from collect_s2_l1c import source_metadata_path
from measure_cushing_landsat7 import CENTRAL_FWHM, laser_references, response_kernel, scene


STEM = "cushing_naip_satellite_controls"
STORE = RAW / "landsat" / STEM
LIMIT_HOURS = 168
DIRECT_HOURS = 48
GAP_FIELDS = ("estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m")
TIMING_PHASES = ("development", "validation", "holdout")


@lru_cache(maxsize=None)
def checksum(path):
    with (ROOT / path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@lru_cache(maxsize=None)
def source_polygons(path):
    project = Transformer.from_crs(4326, 32614, always_xy=True).transform
    features = json.loads((ROOT / path).read_bytes())["features"]
    return {str(f["id"]): transform(make_valid(shape(f["geometry"])), project, interleaved=False)
            for f in features}


def flight_sources(legacy):
    clock = defaultdict(list)
    for r in read("naip_tank_source_clock"):
        if float(r["contour_fraction"]) >= .99 and r["dated_asset_paths_json"] != "[]":
            clock[r["candidate_id"], r["source_acquisition_date"]].append(r)
    registry = {r["candidate_id"]: r for r in read("tank_engineering_registry")}
    dates = sorted({day for _, day in clock})
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        fixed = {(r["candidate_id"], day): dict(r) for day in dates for r in db.execute(SOURCE_SQL, (day, day))
                 if r["documented_roof_type"] in {"IFR", "Domed EFR"}}
    geographic = Transformer.from_crs(32614, 4326, always_xy=True)
    for (tank, day), windows in sorted(clock.items()):
        info = registry[tank]
        if not info["first_seen"] or info["first_seen"] > day or (tank not in legacy and (tank, day) not in fixed):
            continue
        if tank in legacy and info["documented_roof_type"] == "EFR":
            g = legacy[tank]
            east, north, diameter = (float(g[k]) for k in ("center_east_m", "center_north_m", "diameter_m"))
            lon, lat = geographic.transform(east, north)
            shell = float(g["prior_rim_z_p95_native"]) - float(g["prior_ground_z_p50_native"])
            geometry = dict(east=east, north=north, radius=diameter / 2, shell_m=shell, longitude=lon,
                            latitude=lat, geometry_source=g["dataset"], geometry_basis=DATED)
        elif (tank, day) in fixed and fixed[tank, day]["shell_m"]:
            f = fixed[tank, day]
            geometry = {k: f[k] for k in ("east", "north", "radius", "shell_m", "longitude", "latitude", "geometry_source")}
            geometry["geometry_basis"] = "retrospective_fixed_roof_control"
        else:
            continue
        circle = Point(geometry["east"], geometry["north"]).buffer(geometry["radius"], quad_segs=64)
        fractions = [(circle.intersection(source_polygons(r["source_file"])[r["source_polygon_id"]]).area / circle.area, r)
                     for r in windows]
        coverage, window = min(fractions, key=lambda item: (-item[0], item[1]["source_polygon_id"]))
        times = [(r["start_utc"], r["end_exclusive_utc"]) for fraction, r in fractions if fraction >= .99]
        begin = min((start for start, _ in times), default=window["start_utc"])
        end = max((stop for _, stop in times), default=window["end_exclusive_utc"])
        yield window | dict(geometry, terminal_id=info["documented_terminal_id"],
                            roof_type=info["documented_roof_type"], first_seen=info["first_seen"],
                            circle_fraction_in_selected_polygon=coverage, matching_dated_source_polygons=len(times),
                            start_utc=begin, end_exclusive_utc=end)


def fit_flights(tanks, spa):
    paths = sorted({p for tank in tanks for p in json.loads(tank["dated_asset_paths_json"])})
    with ExitStack() as stack:
        rasters = {p: stack.enter_context(rasterio.open(ROOT / p)) for p in paths if (ROOT / p).is_file()}
        results = []
        for tank in tanks:
            available = [p for p in json.loads(tank["dated_asset_paths_json"]) if p in rasters]
            selection = find_asset(dict(tank, dated_asset_paths_json=json.dumps(available)), rasters)
            if tank["circle_fraction_in_selected_polygon"] < .99:
                result = {"shadow_reference_status": "dated_circle_crosses_source_boundary"}
            elif selection is None:
                result = {"shadow_reference_status": "missing_local_dated_naip_pixels"}
            else:
                result = fit(tank, selection, spa, rasters)
                result["source_raster_sha256"] = checksum(result["source_raster"])
                result["native_pixel_north_south_m"] = abs(selection[-1][4].e)
            results.append(dict(tank, **result, seamline_sha256=checksum(tank["source_file"])))
    return results


def span(start, end, stamp):
    a, b = ((stamp - limit).total_seconds() / 3600 for limit in (start, end))
    return max(-a, b, 0), max(abs(a), abs(b))


def flight_index(flights):
    indexed = defaultdict(list)
    for row in flights:
        if row["roof_type"] == "EFR":
            limits = tuple(datetime.fromisoformat(row[k]) for k in ("start_utc", "end_exclusive_utc"))
            indexed[row["candidate_id"]].append((row, *limits))
    return indexed


def overlaps(index, roof, capture):
    if roof in index:
        stamp = datetime.fromisoformat(capture)
        for row, start, end in index[roof]:
            nearest, furthest = span(start, end, stamp)
            if furthest <= LIMIT_HOURS:
                yield row, nearest, furthest


def etmp_fits(index, legacy):
    cached = {(r["product_id"], r["candidate_id"]): r for r in read("cushing_landsat7_height_fits")
              if r["basis"] == "native_location" and int(r["effective_fwhm_m"]) == CENTRAL_FWHM}
    pending = defaultdict(list)
    for r in read("cushing_landsat7_roof_windows"):
        product, roof, capture = (r[k] for k in ("product_id", "candidate_id", "captured_at_utc"))
        if product.endswith("_T1") and list(overlaps(index, roof, capture)):
            if r["angle_valid"] != "True" or int(r["roof_pixels_passing_qa"]) == 0:
                yield dict(r, fit_status="missing_usable_b8_pixels_or_angles")
            elif (product, roof) in cached:
                yield cached[product, roof]
            else:
                pending[product].append((r, None))
    kernel = {CENTRAL_FWHM: response_kernel(CENTRAL_FWHM)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for records in pool.map(lambda job: scene(job, legacy, kernel), sorted(pending.items())):
            yield from records


def sentinel_pass(version):
    platform, _, datatake, _, orbit, *_ = version.split("_")
    return f"{platform}:{orbit}:{datatake[:8]}"


def nearby_sentinel2_sources(flights):
    source_tanks = defaultdict(dict)
    by_key = {(r["source_acquisition_date"], r["candidate_id"]): r for r in flights if r["roof_type"] == "EFR"}
    with closing(read_only(SOURCE)) as db:
        db.row_factory = sqlite3.Row
        for date in sorted({day for day, _ in by_key}):
            rows = db.execute("""SELECT DISTINCT candidate_id,satellite_source_directory,satellite_captured_at_utc,version_id
                FROM aerial_optical_pair WHERE aerial_source='usda_naip_source_polygon' AND level='L1C'
                AND aerial_date=? AND aerial_contour_fraction>=.99 AND maximum_separation_seconds<=?""",
                (date, DIRECT_HOURS * 3600)).fetchall()
            for row in rows:
                tank = by_key.get((date, row["candidate_id"]))
                if tank is not None:
                    key = tuple(row[k] for k in ("satellite_source_directory", "version_id", "satellite_captured_at_utc"))
                    source_tanks[key][row["candidate_id"]] = tank
    return source_tanks


def fit_sentinel2_sources(job, legacy, psf):
    (directory, version, catalogue_time), tanks = job
    manifest = ROOT / directory / "item.json"
    if json.loads(manifest.read_bytes())["product_name"] != version:
        raise ValueError(f"Aerial comparison points to a different Sentinel-2 version: {version}")
    project = Transformer.from_crs(32614, 4326, always_xy=True).transform
    polygons = [(roof, transform(Point(r["east"], r["north"]).buffer(r["radius"], quad_segs=64),
                                 project, interleaved=False)) for roof, r in tanks.items()]
    points = {roof: (r["longitude"], r["latitude"]) for roof, r in tanks.items()}
    quality = {r[0]: dict(zip(QUALITY_FIELDS, r, strict=True)) for r in scene_rows(manifest, polygons, points, {}, None)}
    metadata = str(source_metadata_path(ROOT / directory / "MTD_TL.xml").relative_to(ROOT))
    captured = xml_time(metadata).isoformat()
    # The catalog and original tile metadata record the same pass with millisecond rounding differences.
    if abs((datetime.fromisoformat(captured) - datetime.fromisoformat(catalogue_time)).total_seconds()) > .01:
        raise ValueError(f"Sentinel-2 tile time differs from inventory: {version}")
    dependencies = [str(p.relative_to(ROOT)) for p in sorted(manifest.parent.iterdir()) if p.is_file()]
    digests = {p: checksum(p) for p in dependencies}
    output = []
    for roof, tank in tanks.items():
        row = quality.get(roof)
        r = dict(candidate_id=roof, diameter_m=2 * tank["radius"], fit_status="roof_outside_raster")
        if row is not None:
            expected = row["optical_touched_cells_expected"]
            full = expected > 0 and row["optical_cells_valid_in_all_bands"] == expected
            clear = row["cloud_mask_basis"] != "missing" and not any(row[k] for k in (
                "opaque_roof_cells", "cirrus_roof_cells", "snow_ice_roof_cells"))
            detector = (row["red_detector_covered_roof_cells"] == expected and not row["red_detector_ambiguous_roof_cells"]
                        and len(json.loads(row["red_detector_with_view_options_json"])) == 1)
            status = next((label for ok, label in ((full, "incomplete_valid_pixels"),
                          (clear, "source_cloud_mask_or_obscured_roof"), (detector, "ambiguous_b04_detector")) if not ok), None)
            if status is None:
                g = legacy[roof]
                inputs = dict(tank, radius_m=tank["radius"], lidar_top_navd88_m=float(g["prior_rim_z_p95_native"]),
                              lidar_ground_navd88_m=float(g["prior_ground_z_p50_native"]))
                r.update(fit_geometry(row, inputs, psf[version[:3]]))
            else:
                r["fit_status"] = status
        r.update(satellite_source_sha256=checksum(f"{directory}/red.tif"),
                 satellite_manifest_sha256=checksum(str(manifest.relative_to(ROOT))))
        output.append(("Sentinel-2 B04 direct", sentinel_pass(version), version, captured, directory, r,
                       r["fit_status"] == RESOLVED, GAP_FIELDS))
    return output, dict(version_id=version, satellite_source=directory, catalogue_capture_at_utc=catalogue_time,
                        tile_capture_at_utc=captured, source_tank_candidates=len(tanks),
                        source_file_sha256_json=json.dumps(digests, sort_keys=True))


def optical_observations(index, legacy):
    for phase in TIMING_PHASES:
        for r in read(f"shadow_timing_{phase}_observations"):
            if r["geometry_mode"] == DATED:
                version = r["version_id"]
                yield ("Sentinel-2 B04", sentinel_pass(version), version,
                       r["captured_at_utc"], r["measurement_source"], r, r["fit_status"] == RESOLVED, GAP_FIELDS)
    for r in etmp_fits(index, legacy):
        product, capture = r["product_id"], r["captured_at_utc"]
        good = r["fit_status"] == RESOLVED and str(r.get("center_search_on_boundary")) == "False"
        yield ("Landsat 7 T1 B8", product, product, capture, "landsat7_native", r, good, GAP_FIELDS)
    for r in read("cushing_landsat8_rapid_archive_height_fits"):
        product = r["product_id"]
        yield ("Landsat 8 RT B8", product, product, r["captured_at_utc"], r["native_packet"], r,
               rapid_passed(r), ("gap_m", "gap_min_m", "gap_max_m"))
    for r in read("cushing_recovered_landsat_b8_fits"):
        if r["geometry_basis"] == DATED:
            platform = r["platform"].replace("LANDSAT_", "Landsat ")
            product = r["product_id"]
            yield (f"{platform} recovered B8", product, product, r["optical_capture_at_utc"], r["local_path"], r,
                   r["fit_status"] == RESOLVED, GAP_FIELDS)


def nearby_landsat_sources(flights):
    by_day = defaultdict(list)
    for a in flights:
        if a["roof_type"] == "EFR":
            by_day[datetime.fromisoformat(a["source_acquisition_date"]).date()].append(a)
    for r in read(f"{BULK}_regional_catalog"):
        day = datetime.fromisoformat(r["Date Acquired"].replace("/", "-")).date()
        relevant = [a for d, group in by_day.items() if abs((day - d).days) <= 2 for a in group]
        if not relevant or r["Collection Category"] != "T1":
            continue
        first, last = [datetime.fromisoformat(r[k]).replace(tzinfo=timezone.utc) for k in ("Start Time", "Stop Time")]
        bounds = Polygon([(float(r[f"Corner {p} Longitude"]), float(r[f"Corner {p} Latitude"])) for p in CORNERS])
        if not bounds.is_valid:
            raise ValueError(f"Invalid Landsat scene footprint for {r[PRODUCT]}")
        covered = []
        for a in relevant:
            start, end = [datetime.fromisoformat(a[k]) for k in ("start_utc", "end_exclusive_utc")]
            max_hours = max(span(start, end, first)[1], span(start, end, last)[1])
            if max_hours <= DIRECT_HOURS and bounds.covers(Point(a["longitude"], a["latitude"])):
                covered.append(a["candidate_id"])
        if covered:
            yield r, sorted(set(covered))


def direct_landsat(flights, legacy):
    candidates = sorted(nearby_landsat_sources(flights), key=lambda item: item[0][PRODUCT])
    names = {r[PRODUCT] for r, _ in candidates}
    if not names:
        return [], []
    url, raw, items = query("Name IN (" + ",".join(f"'{n}'" for n in sorted(names)) + ")")
    receipt = STORE / "esri_catalog.json"
    write(receipt, raw)
    selected = {r["Name"]: r for r in items}
    if len(selected) != len(items) or not selected.keys() <= names:
        raise ValueError("Esri catalog returned unexpected Landsat identities")
    metadata = local_geometry(names)
    tool = prepare_tool()
    prepared, sources = [], []
    for row, covered in candidates:
        product = row[PRODUCT]
        source = selected.get(product)
        if source and (source["LANDSAT_PRODUCT_ID"] != product or source["LANDSAT_SCENE_ID"] != row["Landsat Scene Identifier"]
                       or source["Category"] != 1):
            raise ValueError(f"Esri returned a different Landsat acquisition for {product}")
        state = dict(product_id=product, usgs_scene_id=row["Landsat Scene Identifier"],
                     acquisition_day=row["Date Acquired"].replace("/", "-"),
                     covered_registry_ids_json=json.dumps(covered), b8_expected_usgs_sha512=None,
                     esri_catalog_url=url, esri_catalog_receipt=str(receipt.relative_to(ROOT)),
                     esri_catalog_sha256=hashlib.sha256(raw).hexdigest(), esri_exact_object_id=source["OBJECTID"] if source else None,
                     status="missing_esri_match" if source is None else "missing_local_level2_geometry")
        if source and product in metadata:
            state.update(collect_recovery_crop(row, covered, metadata[product], source["OBJECTID"], store=STORE, allow_empty=True))
            if state["esri_reflectance_valid_pixels"]:
                product_data, stage, windows, angles, tile, unresolved = recovered_scene(
                    state, row, legacy, tool, store=STORE, purpose="naip_shadow_validation")
                prepared.append((product_data, stage, windows, angles, tile))
                state.update(product_data | stage, status="ready", roofs_with_angles=len(windows), roofs_without_angles=len(unresolved))
            else:
                state["status"] = "empty_esri_reflectance_crop"
        sources.append(state)
    if not prepared:
        return [], sources
    products = {p["product_id"]: p for p, *_ in prepared}
    staged = {p["product_id"]: s for p, s, *_ in prepared}
    windows = {p["product_id"]: w for p, _, w, *_ in prepared}
    angles = {k: value for _, _, _, a, _ in prepared for k, value in a.items()}
    database = DERIVED / f"{STEM}_landsat_native.sqlite"
    prepare_native(staged, products, windows, angles, database=database)
    lsf = published_lsf()
    results = []
    for p, stage, w, _, tile in prepared:
        product = p["product_id"]
        platform = p["platform"].replace("LANDSAT_", "Landsat ")
        for r in fit_tile(tile, w, p, stage, lsf, database):
            if r["geometry_basis"] != DATED:
                raise ValueError(f"Landsat NAIP control uses a different roof geometry: {product}")
            results.append((f"{platform} B8 direct Esri", product, product, r["optical_capture_at_utc"], r["local_path"],
                            dict(r, satellite_source_sha256=r["raster_sha256"]), r["fit_status"] == RESOLVED, GAP_FIELDS))
    return results, sources


def laser_observations(index, legacy):
    for roof, sensor, source, first, last, gap in laser_references(legacy, index):
        start, end = map(datetime.fromisoformat, (first, last))
        middle = (start + (end - start) / 2).isoformat()
        r = dict(candidate_id=roof, fit_status="passed_laser_profile", gap_m=gap,
                 laser_first_utc=first, laser_last_utc=last)
        yield sensor, source, source, middle, "qualified_laser_profiles", r, True, ("gap_m", None, None)


def comparison_rows(index, observations):
    for sensor, orbit, version, stamp, source, r, passed, fields in observations:
        roof = r["candidate_id"]
        for a, nearest, furthest in overlaps(index, roof, stamp):
            if "laser_first_utc" in r:
                start, end = (datetime.fromisoformat(a[k]) for k in ("start_utc", "end_exclusive_utc"))
                first, last = (datetime.fromisoformat(r[k]) for k in ("laser_first_utc", "laser_last_utc"))
                nearest = max((first-end).total_seconds(), (start-last).total_seconds(), 0) / 3600
                furthest = max(abs((first-end).total_seconds()), abs((last-start).total_seconds())) / 3600
                if furthest > LIMIT_HOURS:
                    continue
            diameter = 2 * a["radius"]
            if r.get("diameter_m") and not math.isclose(diameter, float(r["diameter_m"]), abs_tol=1e-8):
                raise ValueError(f"Unmatched dated geometry for {roof} on {version}")
            aerial_passed = a["shadow_reference_status"] == AERIAL_RESOLVED
            gap, low, high = (float(r[k]) if k and r.get(k) not in (None, "") else None for k in fields)
            signed = gap - a["best_gap_m"] if passed and aerial_passed else None
            lower = low - a["near_gap_max_m"] if signed is not None and low is not None else None
            upper = high - a["near_gap_min_m"] if signed is not None and high is not None else None
            yield dict(candidate_id=roof, terminal_id=a["terminal_id"], diameter_m=diameter,
                       naip_date=a["source_acquisition_date"], naip_polygon_id=a["source_polygon_id"],
                       naip_start_utc=a["start_utc"], naip_end_exclusive_utc=a["end_exclusive_utc"],
                       naip_raster=a.get("source_raster"), naip_raster_sha256=a.get("source_raster_sha256"),
                       satellite_sensor=sensor, satellite_pass=orbit, satellite_version=version,
                       satellite_captured_utc=stamp, satellite_source=source, satellite_observation_id=r.get("observation_id"),
                       satellite_source_sha256=r.get("satellite_source_sha256"),
                       satellite_manifest_sha256=r.get("satellite_manifest_sha256"),
                       satellite_laser_first_utc=r.get("laser_first_utc"), satellite_laser_last_utc=r.get("laser_last_utc"),
                       minimum_separation_hours=nearest, maximum_separation_hours=furthest,
                       satellite_fit_status=r["fit_status"], satellite_passed=passed, satellite_gap_m=gap,
                       satellite_gap_min_m=low, satellite_gap_max_m=high, naip_fit_status=a["shadow_reference_status"],
                       naip_gap_m=a.get("best_gap_m"), naip_gap_min_m=a.get("near_gap_min_m"),
                       naip_gap_max_m=a.get("near_gap_max_m"), satellite_minus_naip_m=signed,
                       difference_range_min_m=lower, difference_range_max_m=upper)


def summary(aerial, comparisons):
    flights = defaultdict(list)
    for r in aerial:
        flights[r["source_acquisition_date"], r["roof_type"]].append(r)
    groups = defaultdict(list)
    for r in comparisons:
        groups[r["satellite_sensor"], r["naip_date"][:4]].append(r)
    details = {}
    for (sensor, year), group in sorted(groups.items()):
        by_limit = {}
        for cap in (3, 24, 48, LIMIT_HOURS):
            selected = [r for r in group if r["maximum_separation_hours"] <= cap]
            paired = [r for r in selected if r["satellite_minus_naip_m"] is not None]
            by_limit[f"within_{cap}h"] = dict(candidate_rows=len(selected), distinct_joint_roofs=len({r["candidate_id"] for r in paired}),
                joint_flights=sorted({r["naip_date"] for r in paired}), joint_satellite_passes=len({r["satellite_pass"] for r in paired}),
                satellite_passed=sum(r["satellite_passed"] for r in selected),
                naip_passed=sum(r["naip_fit_status"] == AERIAL_RESOLVED for r in selected),
                difference_range_excludes_zero=sum(r["difference_range_min_m"] is not None and
                    (r["difference_range_min_m"] > 0 or r["difference_range_max_m"] < 0) for r in paired),
                **describe([r["satellite_minus_naip_m"] for r in paired]))
        details.setdefault(sensor, {})[year] = by_limit
    accepted = [r for r in aerial if r["roof_type"] == "EFR" and r["shadow_reference_status"] == AERIAL_RESOLVED]
    return dict(aerial_fits=len(aerial), distinct_aerial_rasters=len({r.get("source_raster") for r in aerial if r.get("source_raster")}),
                aerial_efr_resolved=len(accepted),
                aerial_median_time_only_span_m=median(r["time_only_gap_max_m"] - r["time_only_gap_min_m"] for r in accepted)
                if accepted else None,
                aerial_fits_by_flight={day: {kind: dict(Counter(r["shadow_reference_status"] for r in flights[day, kind]))
                                           for date, kind in sorted(flights) if date == day} for day in sorted({d for d, _ in flights})},
                comparisons=details, spa_sha256=SPA_SHA256,
                interpretation="The nearest candidate per roof, sensor and NAIP date is selected by timing before fit quality. "
                "The 48-hour limit is applied to the maximum possible separation under the source clocks. Longer separations "
                "are exploration only. The high-resolution NAIP estimate uses a separate sensor with the same circular geometry "
                "and a related shadow model. Camera look and physical movement between acquisitions are unmeasured. "
                "No EIA, prices or prediction features are used.")


def main():
    legacy = legacy_geometry()
    by_year = defaultdict(list)
    for r in flight_sources(legacy):
        by_year[r["acquisition_year"]].append(r)
    spa = solar_module()
    aerial = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for year, results in zip(sorted(by_year), pool.map(lambda rows: fit_flights(rows, spa),
                                                           [by_year[y] for y in sorted(by_year)]), strict=True):
            aerial.extend(results)
            print(year, len(results), "dated NAIP roof fits", flush=True)
    index = flight_index(aerial)
    pairs = list(comparison_rows(index, optical_observations(index, legacy)))
    pairs.extend(comparison_rows(index, laser_observations(index, legacy)))
    nearby = nearby_sentinel2_sources(aerial)
    psf = kernels({key[1][:3] for key in nearby})
    direct_sources = []
    for job in sorted(nearby.items()):
        results, source = fit_sentinel2_sources(job, legacy, psf)
        pairs.extend(comparison_rows(index, results))
        direct_sources.append(source)
    landsat, landsat_sources = direct_landsat(aerial, legacy)
    pairs.extend(comparison_rows(index, landsat))
    pairs.sort(key=lambda r: (r["maximum_separation_hours"], r["satellite_version"], r["satellite_source"]))
    nearest = set()
    for r in pairs:
        key = tuple(r[k] for k in ("candidate_id", "satellite_sensor", "naip_date"))
        r["nearest_before_quality_filter"] = key not in nearest
        nearest.add(key)
    fields = dict.fromkeys(k for row in aerial for k in row)
    export(f"{STEM}_aerial", [{k: r.get(k) for k in fields} for r in aerial])
    export(f"{STEM}_pairs", pairs)
    export(f"{STEM}_sentinel2_sources", direct_sources)
    if landsat_sources:
        fields = dict.fromkeys(k for row in landsat_sources for k in row)
        export(f"{STEM}_landsat_sources", [{k: r.get(k) for k in fields} for r in landsat_sources])
    report = summary(aerial, [r for r in pairs if r["nearest_before_quality_filter"]])
    report.update(candidate_pairs_including_non_nearest=len(pairs), nearest_pairs=len(nearest),
                  input_naip_clock_sha256=checksum("data/derived/naip_tank_source_clock.csv"))
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(len(aerial), "aerial fits;", len(pairs), "candidate comparisons;", len(nearest), "nearest comparisons", flush=True)


if __name__ == "__main__":
    main()
