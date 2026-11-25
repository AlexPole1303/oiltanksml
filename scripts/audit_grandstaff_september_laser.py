"""Audit ATL03 roof-edge returns against the fixed Grandstaff optical observations."""

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pyarrow.parquet as pq
from shapely import transform, union_all
from shapely.geometry import Point

from build_cushing_icesat2_controls import (CMR, COLLECTION, CONTEXT_M, INSET_M, TO_GEO, TO_UTM, ground_heights,
                                          identities, other_structure_context, query)
from build_grandstaff_okeis_records import LOCAL_TIME, STEM as OPERATOR
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_cushing_icesat2_native import BEAM_NAMES
from collect_public import RAW, ROOT, get, write

STEM = "cushing_grandstaff_september_laser"
STORE = RAW / STEM
MONTH = "2026-09"
SEARCH_RANGE = "2026-08-30T00:00:00Z,2026-10-02T23:59:59Z"
INSETS = (0, 3, 5, 7, INSET_M)


def granules(tanks):
    footprint = union_all([Point(t["east_m"], t["north_m"]).buffer(t["radius_m"] + CONTEXT_M) for t in tanks.values()])
    raw = transform(footprint, TO_GEO.transform, interleaved=False).bounds
    bounds = tuple((math.floor if i < 2 else math.ceil)(v * 1e5) / 1e5 for i, v in enumerate(raw))
    url = query(CMR, dict(collection_concept_id=COLLECTION, bounding_box=",".join(map(str, bounds)),
                          temporal=SEARCH_RANGE, page_size=200))
    path = STORE / "cmr_atl03_v7.json"
    payload = get(url, path, refresh=True)
    catalog = json.loads(payload)
    if catalog["hits"] != len(catalog["items"]):
        raise ValueError("Incomplete ATL03 catalog response")
    subsets = {r["cmr_id"]: r for r in read("cushing_icesat2_native_extracts")}
    results = []
    for item in catalog["items"]:
        key = item["meta"]["concept-id"]
        if item["meta"]["collection-concept-id"] != COLLECTION:
            raise ValueError(f"Unexpected ATL03 collection for {key}")
        source = subsets.get(key)
        times = item["umm"]["TemporalExtent"]["RangeDateTime"]
        if source and (not item["umm"]["GranuleUR"].startswith(source["filename"])
                       or datetime.fromisoformat(times["BeginningDateTime"]) != datetime.fromisoformat(source["granule_start_utc"])):
            raise ValueError(f"The local extract differs from the catalog granule {key}")
        results.append(dict(cmr_id=key, granule_ur=item["umm"]["GranuleUR"], captured_start_utc=times["BeginningDateTime"],
                            captured_end_utc=times["EndingDateTime"], catalog_url=url, catalog_path=str(path.relative_to(ROOT)),
                            catalog_sha256=hashlib.sha256(payload).hexdigest(), extract_available=bool(source),
                            photon_parquet=source["csv_path"] if source else None,
                            photon_parquet_sha256=source["parquet_sha256"] if source else None))
    return bounds, results


def geometry_options(tanks):
    modern = {r["candidate_id"]: r for r in read("cushing_modern_lidar_optical_registry") if r["modern_input_status"] == "available"}
    for tank, g in sorted(tanks.items()):
        yield tank, "2011_lidar", g
        if tank in modern:
            row = modern[tank]
            yield tank, "2022_lidar", g | dict(east_m=float(row["center_east_m"]), north_m=float(row["center_north_m"]),
                                                radius_m=float(row["diameter_m"]) / 2,
                                                lidar_top_navd88_m=float(row["prior_rim_z_p95_native"]))


def ground_bias(xy, height, indices, datum, dems):
    ref = xy[indices]
    offsets = height[indices] - ground_heights(ref, dems) - datum
    good = abs(offsets) < 2
    _, groups = np.unique(np.floor(ref[good] / 2), axis=0, return_inverse=True)
    cell_medians = [np.median(offsets[good][groups == k]) for k in np.unique(groups)]
    return len(cell_medians), float(np.median(cell_medians)) if len(cell_medians) >= 10 else None


def profile_rows(source, tanks, dems, structures):
    path = ROOT / source["photon_parquet"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != source["photon_parquet_sha256"]:
        raise ValueError(f"Unexpected ATL03 subset bytes {path.name}")
    fields = ("longitude", "latitude", "height", "gt", "quality_ph", "atl03_cnf", "time_ns")
    table = pq.read_table(path, columns=list(fields))
    lon, lat, height, beam, quality, conf, stamp = (table[k].to_numpy() for k in fields)
    xy = np.column_stack(TO_UTM.transform(lon, lat))
    normal = (quality == 0) & (conf >= 3)
    centers, radii = structures
    for tank, basis, g in geometry_options(tanks):
        center = np.array([g["east_m"], g["north_m"]])
        edge = np.linalg.norm(xy - center, axis=1) - g["radius_m"]
        near = edge <= CONTEXT_M
        exterior = near & (edge > 0)
        for other, radius in zip(centers, radii, strict=True):
            if np.linalg.norm(other - center) <= CONTEXT_M + g["radius_m"] + radius:
                exterior &= np.linalg.norm(xy - other, axis=1) > radius
        for b in np.unique(beam[near & normal]):
            selection = (beam == b) & normal
            datum = g["geoid18_interpolated_m"]
            cells, bias = ground_bias(xy, height, np.flatnonzero(selection & exterior), datum, dems)
            deepest = float(-min(edge[selection]))
            for inset in INSETS:
                valid = selection & (edge <= -inset)
                h, clock = height[valid], stamp[valid]
                if h.size:
                    low, median, high = (float(v) for v in np.quantile(h, [.1, .5, .9]))
                    extent = float(np.linalg.norm(np.ptp(xy[valid], axis=0)))
                    first, last = (datetime.fromtimestamp(float(v.astype("int64")) / 1e9, timezone.utc).isoformat()
                                   for v in (min(clock), max(clock)))
                else:
                    low = median = high = extent = first = last = None
                gap = g["lidar_top_navd88_m"] + datum + bias - median if bias is not None and median is not None else None
                yield dict(candidate_id=tank, geometry_basis=basis, cmr_id=source["cmr_id"], beam=BEAM_NAMES[b],
                           roof_inset_m=inset, deepest_photon_inside_rim_m=deepest, photon_count=int(h.size),
                           distinct_photon_timestamps=len(np.unique(clock)), first_photon_utc=first, last_photon_utc=last,
                           roof_hae_p10_m=low, roof_hae_median_m=median, roof_hae_p90_m=high, sampled_extent_m=extent,
                           local_ground_2m_cells=cells, local_ground_bias_m=bias, wall_navd88_m=g["lidar_top_navd88_m"],
                           apparent_gap_m=gap, meets_standard_inset=inset == INSET_M and h.size > 0,
                           used_as_validation_label=False, photon_parquet=source["photon_parquet"])


def optical_diagnostic(profiles):
    edge = defaultdict(list)
    for r in profiles:
        if r["roof_inset_m"] == 5 and r["photon_count"] >= 10 and r["apparent_gap_m"] is not None:
            edge[r["candidate_id"]].append(r)
    comparisons = []
    for r in read("cushing_grandstaff_september_sentinel_tiles_endpoints"):
        for p in edge[r["candidate_id"]]:
            for tile in ("t14spe", "t14sqe"):
                if r[f"{tile}_fit_status"] != RESOLVED:
                    continue
                t0, t1 = (datetime.fromisoformat(p[f"{side}_photon_utc"]) for side in ("first", "last"))
                optical = datetime.fromisoformat(r[f"{tile}_captured_at_utc"])
                hours = (optical - (t0 + (t1-t0)/2)).total_seconds() / 3600
                if abs(hours) > 72:
                    continue
                comparisons.append(dict(candidate_id=p["candidate_id"], geometry_basis=p["geometry_basis"],
                                        laser_beam=p["beam"], laser_date_local=t0.astimezone(LOCAL_TIME).date().isoformat(),
                                        laser_utc=t0.isoformat(), optical_version_id=r[f"{tile}_version_id"],
                                        optical_utc=optical.isoformat(), optical_minus_laser_hours=hours,
                                        laser_edge_apparent_gap_m=p["apparent_gap_m"],
                                        optical_gap_m=float(r[f"{tile}_estimated_gap_m"]),
                                        optical_minus_laser_gap_m=float(r[f"{tile}_estimated_gap_m"]) - p["apparent_gap_m"],
                                        used_as_validation_label=False))
    return comparisons


def main():
    intervals = [r for r in read(f"{OPERATOR}_optical_throughput_intervals") if r["report_month"] == MONTH]
    wanted = {r["candidate_id"] for r in intervals}
    tanks = {roof: t for roof, t in identities()[0].items() if roof in wanted}
    if tanks.keys() != wanted:
        raise ValueError("Some Grandstaff control roofs lack dated LiDAR geometry")
    bounds, sources = granules(tanks)
    dems, structures = read("cushing_icesat2_ground_assets"), other_structure_context()
    profiles = [r for source in sources if source["extract_available"] for r in profile_rows(source, tanks, dems, structures)]
    comparisons = optical_diagnostic(profiles)
    for name, rows in (("sources", sources), ("profiles", profiles), ("optical_diagnostic", comparisons)):
        export(f"{STEM}_{name}", rows)
    report = dict(target_roofs=len(tanks), catalog_granules=len(sources), granules_with_saved_photons=sum(r["extract_available"] for r in sources),
                  catalog_bbox_wgs84=bounds, roofs_with_detected_photons_inside={str(i): len({r["candidate_id"] for r in profiles
                      if r["roof_inset_m"] == i and r["photon_count"] > 0}) for i in INSETS},
                  sensitivity_comparisons=len(comparisons), unique_laser_roof_crossings=len({
                      (r["cmr_id"], r["candidate_id"], r["beam"]) for r in profiles if r["photon_count"] > 0}),
                  admitted_laser_validation_labels=0)
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
