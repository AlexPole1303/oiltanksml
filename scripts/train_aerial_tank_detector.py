"""Locate cylindrical tanks on Cushing orthophoto and validate on separate facilities."""
import json
import math
from collections import Counter, defaultdict
from contextlib import ExitStack, closing

import cv2
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.merge import merge
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, SOURCE, export, read_only
from collect_public import ROOT, write
from train_shadow_quality import fit_logistic, score

STEM = "aerial_tank_detector"
GSD = 0.9
HALO_M = 60
MAX_RADIUS_M = 49
RADIUS_SEARCH = ((6, 16, 14), (15, 30, 18), (29, MAX_RADIUS_M, 22))
MATCH_IOU = 0.5
NORTH_BOUNDARY = 3986500
EAST_BOUNDARY = 705000
BUFFER_M = 150
L2 = 0.012
TO_UTM = Transformer.from_crs(6350, 32614, always_xy=True)
TO_RASTER = Transformer.from_crs(32614, 6350, always_xy=True)
TO_GPS = Transformer.from_crs(32614, 4326, always_xy=True)
FEATURES = ("visible_ring_sector_p25", "visible_ring_median", "visible_ring_outer_contrast_ratio",
    "nir_ring_sector_p25", "nir_ring_median", "nir_ring_outer_contrast_ratio", "visible_roof_spread",
    "nir_roof_spread", "visible_ring_sector_spread", "visible_rim_balance", "radius_m")
ANGLES = np.linspace(0, 2*np.pi, 96, endpoint=False)


def partition(x, y):
    if y >= NORTH_BOUNDARY+BUFFER_M:
        return "test_north"
    if y > NORTH_BOUNDARY-BUFFER_M or abs(x-EAST_BOUNDARY) < BUFFER_M:
        return "spatial_buffer"
    return "validation_east" if x > EAST_BOUNDARY else "train"


def registry():
    with closing(read_only(SOURCE)) as db:
        rows = db.execute("""SELECT t.candidate_id,t.documented_terminal_id,t.documented_roof_type,
            coalesce(l.center_east_m,t.footprint_east_m),coalesce(l.center_north_m,t.footprint_north_m),
            coalesce(l.diameter_m,t.footprint_diameter_m)/2,a.observed_date,a.structure_observation,
            CASE WHEN l.center_east_m IS NOT NULL THEN l.dataset ELSE t.outline_source END
            FROM tank_engineering_registry t JOIN tank_aerial_visual_observation a USING(candidate_id)
            LEFT JOIN tank_lidar_outline l ON l.candidate_id=t.candidate_id AND l.fit_status='well_supported_circular_fit'
            WHERE a.observed_date BETWEEN '2025-03-01' AND '2025-03-07'
            ORDER BY a.observed_date,t.candidate_id""").fetchall()
    result = {}
    for identity, terminal, roof, x, y, radius, day, seen, geometry in rows:
        if identity in result or not radius or not (0 < radius <= MAX_RADIUS_M):
            raise ValueError(f"Ambiguous or unsupported aerial annotation for {identity}")
        result[identity] = dict(candidate_id=identity, terminal_id=terminal, roof_type=roof, easting=x, northing=y,
            radius_m=radius, observed_day=day, structure_observation=seen, geometry_source=geometry,
            spatial_partition=partition(x, y), eligible_source_tiles=[], best_tile_context_m=None)
    return result


def in_bounds(bounds, x, y):
    return bounds.left <= x < bounds.right and bounds.bottom < y <= bounds.top


def ring_samples(band, x, y, radii):
    xy = np.array([y + np.sin(ANGLES)*radii, x + np.cos(ANGLES)*radii])
    return map_coordinates(band, xy.reshape(2, -1), order=1, mode="nearest", prefilter=False).reshape(radii.size, -1)


def ring_diagnostics(band, x, y, radius):
    steps = radius+np.arange(-3, 4, dtype=float)[:, None]
    differences = np.abs(ring_samples(band, x, y, steps+1)-ring_samples(band, x, y, steps-1))
    # Refine Hough's radius by requiring contrast in several independent angular sectors.
    sector = np.median(differences.reshape(-1, 12, 8), axis=2)
    support = np.quantile(sector, 0.25, axis=1)
    best = int(np.argmax(support))
    refined = steps[best, 0]
    outer = np.abs(ring_samples(band, x, y, np.array([[refined+6]]))-ring_samples(band, x, y, np.array([[refined+4]])))
    interior = ring_samples(band, x, y, refined*np.array([[0.3], [0.55], [0.78]]))
    roof_spread = float(np.quantile(interior, 0.9)-np.quantile(interior, 0.1))
    return refined, float(support[best]), float(np.median(differences[best])), float(np.median(outer)), roof_spread, sector[best]


def extract(source, neighbours, truth):
    bounds = source[1].bounds
    extent = bounds.left-HALO_M, bounds.bottom-HALO_M, bounds.right+HALO_M, bounds.top+HALO_M
    touching = [ds for _, ds in neighbours if ds.bounds.left < extent[2] and ds.bounds.right > extent[0]
                and ds.bounds.bottom < extent[3] and ds.bounds.top > extent[1]]
    pixels, affine = merge(touching, bounds=extent, res=GSD, indexes=(1, 2, 3, 4), masked=True, resampling=Resampling.average)
    valid = ~np.ma.getmaskarray(pixels).any(axis=0)
    distance = distance_transform_edt(np.pad(valid, 1))[1:-1, 1:-1]*GSD
    gray = np.einsum("ijk,i->jk", pixels.data[:3], [0.2126, 0.7152, 0.0722]).astype("float32")
    nir = pixels.data[3].astype("float32")
    # Distance to valid pixels, including outside the merged crop, prevents partial circles at scene edges.
    for row in truth.values():
        if row["observed_day"] != source[0]["tile_majority_date"] or row["structure_observation"] != "aboveground_tank_seen":
            continue
        x, y = TO_RASTER.transform(row["easting"], row["northing"])
        if not in_bounds(bounds, x, y):
            continue
        col, line = (~affine) * (x, y)
        context = distance[int(line), int(col)]
        if context >= row["radius_m"]+8:
            row["eligible_source_tiles"].append(source[0]["tile_id"])
            row["best_tile_context_m"] = max(context, row["best_tile_context_m"] or 0)
    smooth = cv2.GaussianBlur(np.rint(gray).astype("uint8"), (5, 5), 0.9)
    for minimum, maximum, votes in RADIUS_SEARCH:
        proposals = cv2.HoughCircles(smooth, cv2.HOUGH_GRADIENT, dp=1.2, minDist=9/GSD, param1=80, param2=votes,
                                     minRadius=math.floor(minimum/GSD), maxRadius=math.ceil(maximum/GSD))
        if proposals is None:
            continue
        for col, line, radius in proposals[0]:
            x, y = affine * (float(col)+0.5, float(line)+0.5)
            if not in_bounds(bounds, x, y) or distance[round(line), round(col)] < radius*GSD+11:
                continue
            east, north = TO_UTM.transform(x, y)
            split = partition(east, north)
            if split == "spatial_buffer":
                continue
            r, visible_support, visible_edge, visible_outer, visible_spread, sectors = ring_diagnostics(gray, col, line, radius)
            _, nir_support, nir_edge, nir_outer, nir_spread, _ = ring_diagnostics(nir, col, line, r)
            radius_m = float(r*GSD)
            vector = (visible_support, visible_edge, visible_edge/(visible_outer+3), nir_support, nir_edge,
                nir_edge/(nir_outer+3), visible_spread, nir_spread, float(np.std(sectors)),
                visible_support/(visible_edge+3), radius_m)
            yield dict(spatial_partition=split, source_tile=source[0]["tile_id"], source_day=source[0]["tile_majority_date"],
                source_crop=source[0]["crop_path"], source_nominal_time_utc=source[0]["tile_nominal_time_utc"],
                easting=east, northing=north, radius_m=radius_m, hough_radius_m=float(radius*GSD),
                valid_context_m=float(distance[round(line), round(col)]), ring_score=visible_support, features=vector)


def circle_iou(a, b):
    r, s = a["radius_m"], b["radius_m"]
    d = math.hypot(a["easting"]-b["easting"], a["northing"]-b["northing"])
    if d >= r+s:
        return 0.0
    if d <= abs(r-s):
        return (min(r, s)/max(r, s))**2
    overlap = (r*r*math.acos(max(-1, min(1, (d*d+r*r-s*s)/(2*d*r))))
        + s*s*math.acos(max(-1, min(1, (d*d+s*s-r*r)/(2*d*s))))
        - 0.5*math.sqrt(max(0, (-d+r+s)*(d+r-s)*(d-r+s)*(d+r+s))))
    return overlap/(math.pi*(r*r+s*s)-overlap)


def tree(rows):
    return cKDTree([(r["easting"], r["northing"]) for r in rows])


def nearest_truth(proposals, annotations):
    known = tree(annotations)
    for row in proposals:
        indexes = known.query_ball_point((row["easting"], row["northing"]), row["radius_m"]+MAX_RADIUS_M)
        options = [(circle_iou(row, annotations[i]), annotations[i]["candidate_id"]) for i in indexes
                   if row["source_day"] == annotations[i]["observed_day"]]
        overlap, identity = max(options, default=(0., ""))
        row.update(best_reference_iou=overlap, nearest_reference_id=identity if overlap > 0 else "")


def rank(proposals, annotations, field):
    ordered = sorted(proposals, key=lambda row: (-row[field], row["proposal_id"]))
    positions = tree(ordered)
    valid_ids = {r["candidate_id"] for r in annotations}
    seen, suppressed = set(), set()
    for index, row in enumerate(ordered):
        if index in suppressed:
            continue
        candidates = positions.query_ball_point((row["easting"], row["northing"]), row["radius_m"]+MAX_RADIUS_M+3)
        suppressed.update(j for j in candidates if j > index and circle_iou(row, ordered[j]) >= MATCH_IOU)
        identity = row["nearest_reference_id"]
        matched = row["best_reference_iou"] >= MATCH_IOU and identity in valid_ids and identity not in seen
        if matched:
            seen.add(identity)
        yield row, matched


def operating_point(ranking, field, denominator):
    best = None
    hits = 0
    for index, (row, matched) in enumerate(ranking, 1):
        hits += matched
        if index < len(ranking) and ranking[index][0][field] == row[field]:
            continue
        f1 = 2*hits/(denominator+index)
        if best is None or f1 > best[0]:
            best = f1, row[field]
    if not best or best[0] == 0:
        raise ValueError(f"No usable validation threshold for {field}")
    return best[1]


def metrics(ranking, field, threshold, denominator):
    selected = [(row, matched) for row, matched in ranking if row[field] >= threshold]
    hits = sum(m for _, m in selected)
    return dict(labels=denominator, detections=len(selected), matches=hits, missed=denominator-hits,
        unmatched=len(selected)-hits, recall=hits/denominator, precision_vs_registry=hits/len(selected) if selected else None,
        f1=2*hits/(denominator+len(selected)))


def detection_geojson(rows):
    features = []
    for row in rows:
        coords = [TO_GPS.transform(row["easting"]+row["radius_m"]*math.cos(a), row["northing"]+row["radius_m"]*math.sin(a))
                  for a in np.linspace(0, 2*np.pi, 49, endpoint=False)]
        coords.append(coords[0])
        features.append(dict(type="Feature", geometry=dict(type="Polygon", coordinates=[coords]), properties={
            k: row[k] for k in ("proposal_id", "spatial_partition", "source_tile", "source_day", "radius_m", "ml_score",
                                "ring_score", "detector_method", "detector_score", "nearest_reference_id", "best_reference_iou",
                                "matched_registry")}))
    return dict(type="FeatureCollection", features=features)


def main():
    cv2.setNumThreads(2)
    truth = registry()
    proposals = []
    with ExitStack() as stack:
        datasets = [(row, stack.enter_context(rasterio.open(ROOT / row["crop_path"])))
                    for row in read_csv("ok_aerial_2025_hr4_window_index")]
        by_day = defaultdict(list)
        for item in datasets:
            if item[1].crs.to_epsg() != 6350:
                raise ValueError(f"Unexpected aerial raster CRS for {item[0]['tile_id']}")
            by_day[item[0]["tile_majority_date"]].append(item)
        for source in datasets:
            proposals.extend(extract(source, by_day[source[0]["tile_majority_date"]], truth))
            print(f"Scanned {source[0]['tile_id']}, cumulative proposals {len(proposals)}", flush=True)
    for index, row in enumerate(proposals, 1):
        row["proposal_id"] = f"circle_{index:06d}"
    annotated = [r for r in truth.values() if r["structure_observation"] == "aboveground_tank_seen"]
    eligible = [r for r in annotated if r["eligible_source_tiles"] and r["spatial_partition"] != "spatial_buffer"]
    # Use all known rings for candidate labelling, including rings outside the full-context evaluation area.
    nearest_truth(proposals, annotated)
    labelled = [r for r in proposals if r["spatial_partition"] == "train" and
                (r["best_reference_iou"] < 0.3 or r["best_reference_iou"] >= MATCH_IOU)]
    model = fit_logistic([r["features"] for r in labelled], [r["best_reference_iou"] >= MATCH_IOU for r in labelled], FEATURES, L2)
    for row, probability in zip(proposals, score(model, [r["features"] for r in proposals]), strict=True):
        row["ml_score"] = float(probability)
    results, accepted, threshold = {}, defaultdict(list), {}
    partitions = ("validation_east", "test_north", "train")
    for split in partitions:
        targets = [r for r in eligible if r["spatial_partition"] == split]
        candidates = [r for r in proposals if r["spatial_partition"] == split]
        if not targets or not candidates:
            raise ValueError(f"Empty geographical split {split}")
        ranking = {field: list(rank(candidates, targets, field)) for field in ("ring_score", "ml_score")}
        if split == "validation_east":
            threshold = {field: operating_point(items, field, len(targets)) for field, items in ranking.items()}
        results[split] = {field: metrics(items, field, threshold[field], len(targets)) for field, items in ranking.items()}
        results[split]["raw_proposals"] = len(candidates)
        results[split]["registry_rings_with_at_least_one_candidate"] = len({r["nearest_reference_id"] for r in candidates
            if r["best_reference_iou"] >= MATCH_IOU} & {r["candidate_id"] for r in targets})
        for field, items in ranking.items():
            accepted[field].extend(dict(r, matched_registry=matched, detector_method=field, detector_score=r[field])
                                   for r, matched in items if r[field] >= threshold[field])
    method = max(threshold, key=lambda field: results["validation_east"][field]["f1"])
    selected = {field: {r["proposal_id"]: r["matched_registry"] for r in items} for field, items in accepted.items()}
    detections = {r["nearest_reference_id"]: r for r in accepted[method] if r["matched_registry"]}
    truth_rows = [dict(r, eligible_source_tiles="|".join(sorted(r["eligible_source_tiles"])),
        eligible_for_evaluation=r in eligible, matched_proposal_id=detections.get(r["candidate_id"], {}).get("proposal_id"))
        for r in truth.values()]
    proposal_rows = [{k: v for k, v in r.items() if k != "features"} | {
        name: value for field, members in selected.items() for name, value in (
            (f"{field}_accepted", r["proposal_id"] in members), (f"{field}_matched_registry", members.get(r["proposal_id"])))
        } | dict(zip(FEATURES, r["features"], strict=True)) for r in proposals]
    model.update(ml_threshold=threshold["ml_score"], selected_method=method, selected_threshold=threshold[method],
                 gsd_m=GSD, hough_radius_bands_m=RADIUS_SEARCH, l2=L2)
    summary = dict(source_crops=len(datasets), annotated_candidates=len(truth), visible_annotated_tanks=len(annotated),
        evaluable_visible_tanks=len(eligible), visible_tanks_by_partition=dict(Counter(r["spatial_partition"] for r in annotated)),
        features=list(FEATURES), raw_proposals=len(proposals), training_proposals=len(labelled),
        positive_training_proposals=sum(r["best_reference_iou"] >= MATCH_IOU for r in labelled),
        minimum_circle_iou=MATCH_IOU, north_boundary_m=NORTH_BOUNDARY, east_boundary_m=EAST_BOUNDARY,
        spatial_exclusion_buffer_m=BUFFER_M, validation_selected_threshold=threshold,
        validation_selected_method=method, partitions=results,
        interpretation="Proposals are computed from aerial pixels across entire crops with no reference centres. The registry supplies "
        "labels. Thresholds are selected on the eastern spatial validation region and evaluated on separate northern facilities. "
        "Multiple acquisitions and unrelated new geographies are not tested. Unmatched accepted circles are unverified candidates, "
        "including possible registry omissions. Detection describes visible circular tank footprints, not roof type, height or oil volume.")
    export(f"{STEM}_proposals", proposal_rows)
    export(f"{STEM}_registry", truth_rows)
    for file, content in (("model.json", model), ("summary.json", summary), ("detections.geojson", detection_geojson(accepted[method]))):
        write(DERIVED / f"{STEM}_{file}", (json.dumps(content, indent=2, allow_nan=False)+"\n").encode())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
