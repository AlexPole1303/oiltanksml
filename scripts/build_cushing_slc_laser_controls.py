"""Match qualified ICESat-2 roof profiles to native Sentinel-1 SLC burst geometry."""
import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from shapely.geometry import Point

from build_cushing_icesat2_controls import identities
from build_s1_holdout_gap_burst_plan import CONTEXT_M, bursts
from build_s1_reference_matches import export, read
from build_s1_slc_orbit_geometry import look
from build_weekly_shadow_panel import DERIVED
from collect_public import write
from collect_s1_roof_catalog import SLC
from collect_s1_slc_orbits import orbit_for


STEM = "cushing_slc_laser_controls"
HOURS = 24


def time_matches(native=False):
    radar = [(r, set(json.loads(r["covered_candidate_ids_json"])), datetime.fromisoformat(r["capture_start_utc"]),
              datetime.fromisoformat(r["capture_end_utc"])) for r in read("s1_cushing_catalog") if r["product_type"] == SLC]
    source = "cushing_icesat2_native_profiles" if native else "cushing_icesat2_roof_profiles"
    for laser in read(source):
        if laser["passes_roof_profile_checks"] != "True":
            continue
        if native:
            laser = dict(laser, granule_start_utc=laser["first_roof_signal_photon_utc"],
                         granule_end_utc=laser["last_roof_signal_photon_utc"])
        begin, end = (datetime.fromisoformat(laser[k]) for k in ("granule_start_utc", "granule_end_utc"))
        if not begin <= end:
            raise ValueError(f"Reversed source interval for {laser['cmr_id']}")
        for product, tanks, start, stop in radar:
            if laser["candidate_id"] not in tanks or laser["first_seen"] > start.date().isoformat():
                continue
            separation = max(abs((stop - begin).total_seconds()), abs((start - end).total_seconds())) / 3600
            if separation <= HOURS:
                yield product, laser, separation


def matched_controls(matches, products, orbits, geometry):
    for original, laser, separation in matches:
        product = products[original["product_id"]]
        orbit = orbits[product["product_id"]]
        candidate = laser["candidate_id"]
        reference = geometry[candidate]
        east, north, radius = (float(laser[k]) for k in
                               ("geometry_center_east_m", "geometry_center_north_m", "geometry_radius_m"))
        if any(abs(reference[k] - v) > 1e-5 for k, v in (("east_m", east), ("north_m", north), ("radius_m", radius))):
            raise ValueError(f"Laser and orbit geometry use different tank outlines for {candidate}")
        area = Point(east, north).buffer(radius + CONTEXT_M)
        supported = defaultdict(dict)
        for b in product["bursts"]:
            if b["identity_evidence"] == "cdse_and_asf" and b["footprint"].covers(area):
                supported[b["relative_burst_id"], b["swath"]][b["polarization"]] = b
        choices = {key: pair for key, pair in supported.items() if {"VV", "VH"} <= set(pair)}
        if not choices:
            continue
        key = max(choices, key=lambda k: min(choices[k][pol]["footprint"].boundary.distance(area) for pol in ("VV", "VH")))
        match = choices[key]
        roof = float(laser["roof_hae_p50_m"])
        wall = reference["top_hae_approx_m"] + float(laser["local_ground_bias_m"])
        if abs(wall - roof - float(laser["estimated_wall_to_roof_m"])) > .06:
            raise ValueError(f"Laser wall-to-roof height differs from the common vertical frame for {candidate}")
        rim = look(reference, wall, orbit)
        roof_geometry = look(reference, roof, orbit)
        if not (rim[3] and roof_geometry[3] and roof_geometry[1] > rim[1]):
            raise ValueError(f"Unexpected slant range for qualified laser roof {candidate}")
        yield dict(candidate_id=candidate, terminal_id=laser["terminal_id"], product_id=product["product_id"],
            product_name=product["product_name"], radar_capture_start_utc=product["capture_start_utc"],
            radar_capture_end_utc=product["capture_end_utc"], laser_granule_id=laser["cmr_id"],
            laser_granule_start_utc=laser["granule_start_utc"], laser_granule_end_utc=laser["granule_end_utc"],
            laser_beam=laser["beam"], laser_track=laser["track_id"], maximum_separation_hours=separation,
            roof_hae_m=roof, wall_hae_m=wall, laser_wall_to_roof_m=wall - roof, expected_rim_to_roof_slant_shift_m=roof_geometry[1] - rim[1],
            rim_zero_doppler_at_utc=rim[0], roof_zero_doppler_at_utc=roof_geometry[0], incidence_deg=rim[2],
            relative_burst_id=key[0], swath=key[1], full_context_m=CONTEXT_M,
            laser_photon_packet=laser["photon_packet"], precise_orbit_file=orbit["local_orbit_gzip"],
            **{f"{pol.lower()}_{k}": match[pol][k] for pol in ("VV", "VH")
               for k in ("cdse_uuid", "cdse_catalog_file", "asf_catalog_file", "asf_data_url", "asf_annotation_url")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="match with the exact photon-time ATL03 profiles")
    args = parser.parse_args()
    stem = STEM + "_native" if args.native else STEM
    matches = list(time_matches(args.native))
    parents = {r["product_id"]: r for r, _, _ in matches}
    with ThreadPoolExecutor(max_workers=4) as pool:
        products = dict(zip(parents, pool.map(bursts, parents.values()), strict=True))
        orbits = {r["product_id"]: r for r in pool.map(orbit_for,
                  (dict(parent, scope="icesat2_control") for parent in parents.values()))}
    geometry, _, _ = identities()
    rows = list(matched_controls(matches, products, orbits, geometry))
    if not rows:
        raise ValueError("No qualifying radar and laser burst matches")
    rows.sort(key=lambda r: (r["maximum_separation_hours"], r["candidate_id"], r["laser_granule_id"]))
    export(stem, rows)
    export(stem + "_orbits", list(orbits.values()))
    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(), parent_candidates=len(parents),
        parent_candidate_roof_matches=len(matches), qualified_dual_polarization_roof_matches=len(rows),
        distinct_roofs=len({r["candidate_id"] for r in rows}),
        verified_orbit_files=len(orbits),
        matches_within_12h=sum(r["maximum_separation_hours"] <= 12 for r in rows),
        min_maximum_separation_hours=min(r["maximum_separation_hours"] for r in rows),
        max_maximum_separation_hours=max(r["maximum_separation_hours"] for r in rows),
        distinct_burst_assets=len({r[f"{pol}_cdse_uuid"] for r in rows for pol in ("vv", "vh")}),
        expected_slant_shift_at_least_3_5m=sum(r["expected_rim_to_roof_slant_shift_m"] >= 3.5 for r in rows),
        interpretation="ICESat-2 directly constrains the roof surface at the laser observation. "
        "The range shift is predicted from the two positions and a precise orbit. Between the two acquisitions "
        "the roof may move. Actual complex SLC samples are required to measure a radar height.")
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / (stem + "_summary.json"), payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
