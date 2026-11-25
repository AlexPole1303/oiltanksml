"""Recover ATL03 photons from spatial subsets of previously capped requests."""
import json
import math
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
from shapely import transform
from shapely.geometry import Point, box
from shapely.ops import unary_union
from scipy.spatial.distance import cdist

from build_cushing_icesat2_controls import (CONTEXT_M, STORE, TO_GEO, TO_UTM, airborne_matches, collect_photons,
    holdout_calendar, identities, measurements, optical_matches, other_structure_context, rows_from_csv, unique_profiles)
from build_s1_reference_matches import read
from build_weekly_shadow_panel import DERIVED
from collect_public import ROOT, write, write_csv


STEM = "cushing_icesat2_photon_recovery"
TILE_M = 1250
SOURCE_FIELDS = ("filename", "cmr_id", "granule_start_utc", "granule_end_utc", "granule_catalog_inserted_utc", "utc_granule_day")
WEST_DIAGNOSTIC = (-96.765, 35.936, -96.741, 35.961)


def groups(tanks):
    by_tile = defaultdict(dict)
    for key, r in sorted(tanks.items()):
        cell = tuple(math.floor(r[field] / TILE_M) for field in ("east_m", "north_m"))
        by_tile[cell][key] = r
    for number, (cell, members) in enumerate(sorted(by_tile.items()), 1):
        contexts = [Point(t["east_m"], t["north_m"]).buffer(t["radius_m"] + CONTEXT_M) for t in members.values()]
        utm_envelope = box(*unary_union(contexts).bounds)
        raw = transform(utm_envelope, TO_GEO.transform, interleaved=False).bounds
        bounds = tuple((math.floor if i < 2 else math.ceil)(v * 1e6) / 1e6 for i, v in enumerate(raw))
        if not all(box(*bounds).covers(transform(poly, TO_GEO.transform, interleaved=False)) for poly in contexts):
            raise ValueError(f"Geographic subset does not cover tank context in tile {cell}")
        yield f"tile_{number:02d}", bounds, members


def recover(job):
    old, tile, bounds, tanks = job
    day, track = old["requested_date"], int(old["track_id"])
    asset, photons = collect_photons((day, track), bounds, store=STORE / "recovery" / tile)
    listed = json.loads(asset["packet_granules_json"])
    if (listed and listed != [old["filename"]]) or (photons and not listed):
        raise ValueError(f"Source version differs for ATL03 {day}/{track}/{tile}")
    if any(not (bounds[0] <= float(r["longitude"]) <= bounds[2] and bounds[1] <= float(r["latitude"]) <= bounds[3])
           for r in photons):
        raise ValueError(f"Photons outside the requested bounds {day}/{tile}")
    result = dict(asset, **{k: old[k] for k in SOURCE_FIELDS}, tile=tile,
                  bounds_wgs84_json=json.dumps(bounds), tank_ids_json=json.dumps(sorted(tanks)),
                  broad_csv_path=old["csv_path"], broad_packet_path=old["packet_path"],
                  broad_photon_count=int(old["photon_count"]), broad_maximum_stride=int(old["maximum_subsample_stride"]))
    print(f"{day} {tile}: {len(photons):,} photon rows, max stride {asset['maximum_subsample_stride']}.", flush=True)
    return result, photons


def profile_key(r):
    return tuple(r[k] for k in ("filename", "candidate_id", "beam"))


def export(suffix, rows, *, empty_fields=()):
    fields = tuple(dict.fromkeys(k for row in rows for k in row)) or empty_fields
    write_csv(DERIVED / f"{STEM}_{suffix}.csv", fields, ([r.get(k) for k in fields] for r in rows))


def west_diagnostic(tanks):
    source, photons = collect_photons(("2021-03-17", 1270), WEST_DIAGNOSTIC, store=STORE / "recovery" / "west_diagnostic")
    lon, lat = ([float(p[field]) for p in photons] for field in ("longitude", "latitude"))
    xy = np.column_stack(TO_UTM.transform(lon, lat))
    centres = [(t["east_m"], t["north_m"]) for t in tanks.values()]
    radii = np.array([t["radius_m"] for t in tanks.values()])
    distance = (cdist(xy, centres) - radii).min(axis=1)
    return dict(source, bounds_wgs84=list(WEST_DIAGNOSTIC),
        nearest_roof_edge_m=float(distance.min()) if len(distance) else None,
        photons_in_tank_context=int((distance <= CONTEXT_M).sum()))


def main():
    broad = [r for r in read("cushing_icesat2_extracts") if int(r["maximum_subsample_stride"]) > 1]
    if not broad:
        raise ValueError("No capped photon extracts to recover")
    tanks, _, _ = identities()
    structures = other_structure_context()
    dems = read("cushing_icesat2_ground_assets")
    areas = list(groups(tanks))
    if Counter(k for _, _, group in areas for k in group) != Counter(tanks.keys()):
        raise ValueError("Each roof must have exactly one full-context subset per acquisition")
    jobs = [(old, *area) for old in broad for area in areas]
    extracts, candidates = [], []
    by_date = defaultdict(Counter)
    with ThreadPoolExecutor(max_workers=5) as pool:
        for (_, _, _, sample), (asset, photons) in zip(jobs, pool.map(recover, jobs), strict=True):
            extracts.append(asset)
            candidates.extend(measurements(asset, photons, sample, dems, structures))
            by_date[asset["filename"]] |= Counter(tuple(p.values()) for p in photons)
    if len({profile_key(p) for p in candidates}) != len(candidates):
        raise ValueError("A recovered profile was produced more than once")
    original = {profile_key(r): r for r in read("cushing_icesat2_roof_profiles")}
    profiles = []
    for r in unique_profiles(candidates):
        prior = original.get(profile_key(r), {})
        profiles.append(dict(r, broad_profile_exists=bool(prior),
            broad_profile_pass=prior.get("passes_roof_profile_checks") == "True",
            broad_interior_medium_or_high_photons=prior.get("interior_medium_or_high_photons"),
            broad_estimated_wall_to_roof_m=prior.get("estimated_wall_to_roof_m"),
            new_qualified_reference=r["passes_roof_profile_checks"] and prior.get("passes_roof_profile_checks") != "True"))
    new = [r for r in profiles if r["new_qualified_reference"]]
    optical, airborne = list(optical_matches(new)), list(airborne_matches(new))
    weeks = list(holdout_calendar(new))
    overlap = {r["filename"]: Counter(tuple(p.values()) for p in rows_from_csv((ROOT / r["csv_path"]).read_bytes())) for r in broad}
    export("extracts", extracts)
    export("profiles", profiles, empty_fields=("filename", "candidate_id", "beam", "passes_roof_profile_checks"))
    export("new_optical_controls", optical, empty_fields=("candidate_id", "icesat_granule", "sentinel2_version_id"))
    export("new_airborne_controls", airborne, empty_fields=("candidate_id", "icesat_granule", "airborne_dataset"))
    export("new_holdout_weeks", weeks)
    summary = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), broad_requests=len(broad),
        local_subset_requests=len(extracts), tanks_per_acquisition=len(tanks), distinct_full_context_subsets=len(areas),
        extracts_with_photons=sum(r["photon_count"] > 0 for r in extracts),
        verified_source_packets=sum(bool(r["packet_path"]) for r in extracts),
        profiles=len(profiles), passing_profiles=sum(r["passes_roof_profile_checks"] for r in profiles),
        new_passing_profiles=len(new), new_distinct_roofs=len({r["candidate_id"] for r in new}),
        new_sentinel2_controls_within_48h=len(optical), new_sentinel2_controls_within_24h=sum(r["max_time_separation_hours"] <= 24 for r in optical),
        new_airborne_controls_within_14d=len(airborne), west_diagnostic=west_diagnostic(tanks),
        same_profile_height_change_m=[r["estimated_wall_to_roof_m"] - float(r["broad_estimated_wall_to_roof_m"]) for r in profiles
                                      if r["broad_profile_pass"] and r["passes_roof_profile_checks"]],
        per_granule=[dict(filename=name, broad_photon_rows=sum(overlap[name].values()),
                          distinct_subset_photon_rows=sum(sample.values()),
                          subset_photon_rows_absent_in_broad=sum((sample - overlap[name]).values()))
                     for name, sample in sorted(by_date.items())],
        interpretation="Local subsets preserve the full roof and ground context and the same ATL03 granule version. "
        "Source extracts may still be subsampled. A recovered row is a photon, and only separately qualified profiles are roof controls.")
    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    write(DERIVED / (STEM + "_summary.json"), payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
