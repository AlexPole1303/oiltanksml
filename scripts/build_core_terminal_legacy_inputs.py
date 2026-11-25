"""Reconstruct dated engineering inputs and development estimates for the main terminal panel."""
import csv
import datetime as dt
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing

from pyproj import Transformer

from build_shadow_calibration import CONTEXT_M
from build_terminal_development_measurements import DATED, run_fits
from build_terminal_legacy_lidar import write_geometry
from build_terminal_lidar_source_timing import old_origins
from build_weekly_shadow_panel import DERIVED, PIXELS, SOURCE, aggregate, export, pair_rows, read_only
from collect_permit_review_timeline import SOURCE as REVIEW_SOURCE, timeline_rows
from collect_public import ROOT, write, write_csv
from fit_shadow_geometry import FIELDS, inputs
from terminal_aerial_roof_evidence import TABLE as EARLY_ROOF_TABLE

CORE = ("plains_cushing", "enbridge_cushing", "enbridge_blueknight", "magellan_cushing", "semgroup_cushing",
        "ngl_cushing", "southbow_cushing", "wildhorse")
STEM = "core_terminal_legacy"
GEOMETRY = DERIVED / f"{STEM}_lidar_geometry.csv"
ENBRIDGE_RENEWAL_SOURCE = "https://applications.deq.ok.gov/permitspublic/storedpermits/8499.pdf#page=3"


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def lidar_clocks(geometry):
    catalog = {r["tile_id"]: r for r in read_rows(DERIVED / "lidar_2011_source_tiles.csv")}
    rows = []
    for candidate, tile, origin, path in old_origins(geometry):
        metadata = catalog[tile]
        fgdc = dt.datetime.strptime(metadata["fgdc_publication_date_raw"], "%Y%m%d").date().isoformat()
        rows.append(dict(candidate_id=candidate, terminal_id=geometry[candidate]["terminal_id"], tile_id=tile,
                         origin_id=origin, origin_path=path, tnm_publication_date=metadata["tnm_publication_date"],
                         fgdc_publication_date=fgdc, available_by_date=max(fgdc, metadata["tnm_publication_date"]),
                         fgdc_url=metadata["fgdc_url"], laz_url=metadata["original_laz_url"]))
    dates = {}
    for row in rows:
        candidate, day = row["candidate_id"], row["available_by_date"]
        dates[candidate] = max(day, dates.get(candidate, day))
    if dates.keys() != geometry.keys():
        raise ValueError("The core legacy geometries and dated USGS origins differ")
    export(f"{STEM}_lidar_source_timing", sorted(rows, key=lambda r: (r["candidate_id"], r["origin_id"])))
    return dates


def roof_clocks():
    permits = {doc: (max(memo, review), "matched_epa_review_end", url)
               for doc, _, _, _, _, memo, review, _, _, url, _ in timeline_rows(REVIEW_SOURCE.read_bytes())}
    # The subsequent renewal records the exact issuance date of the 2016 draft's permit number.
    permits["3329"] = ("2017-02-09", "issuance_of_same_permit_number_recorded_in_later_renewal", ENBRIDGE_RENEWAL_SOURCE)
    with closing(read_only(SOURCE)) as db:
        ngl = {candidate: (day, "dated_naip_inset_wall_shadow", url) for candidate, day, url in db.execute(f"""
            SELECT candidate_id,publication_date,publication_source_url FROM {EARLY_ROOF_TABLE}
            WHERE scope_id='ngl_older_parcel' AND top_observation='inset_wall_shadow_visible'""")}
    if len(ngl) != 14:
        raise ValueError("The NGL legacy roof cohort and dated aerial review differ")
    return permits, ngl


def pairs_and_manifest(geometry, dates, permits, ngl):
    pairs = defaultdict(dict)
    with (DERIVED / "weekly_shadow_endpoint_manifest.csv").open(newline="") as stream:
        for r in csv.DictReader(stream):
            if r["terminal_id"] in CORE and r["roof_type"] == "EFR":
                key = r["week_ending"], r["candidate_id"]
                if r["endpoint"] in pairs[key]:
                    raise ValueError(f"Duplicate core optical endpoint {key}")
                pairs[key][r["endpoint"]] = r
    ledger, selected = [], []
    for (week, candidate), pair in sorted(pairs.items()):
        before, after = (pair[end] for end in ("previous", "current"))
        cutoff = after["inputs_available_before_utc"]
        if (before["captured_at_utc"] >= after["captured_at_utc"] or any(
                dt.datetime.fromisoformat(row[field]) >= dt.datetime.fromisoformat(cutoff)
                for row in pair.values() for field in ("archive_available_at_utc", "geometry_available_at_utc"))):
            raise ValueError(f"Core optical pair {week}/{candidate} violates its cutoff")
        doc = after["engineering_source_url"].rsplit("/", 1)[-1].removesuffix(".pdf")
        roof_date, basis, url = ngl.get(candidate, permits.get(doc, (None, None, None)))
        lidar_date = dates.get(candidate)
        if lidar_date and geometry[candidate]["cohort_id"] != after["cohort_id"]:
            raise ValueError(f"Legacy geometry is assigned to a different cohort for {candidate}")
        if not lidar_date:
            status = "no_independent_2011_geometry"
        elif not roof_date:
            status = "no_dated_regulatory_roof_evidence"
        elif max(lidar_date, roof_date) >= cutoff[:10]:
            status = "engineering_evidence_after_cutoff"
        else:
            status = "dated_engineering_inputs"
            if after["study_phase"] == "development":
                selected.append({end: dict(row, diameter_m=float(geometry[candidate]["diameter_m"]),
                                           geometry_basis="independent_2011_elevated_circle") for end, row in pair.items()})
        ledger.append(dict(week_ending=week, release_date=after["release_date"], study_phase=after["study_phase"],
                           candidate_id=candidate, terminal_id=after["terminal_id"], cohort_id=after["cohort_id"],
                           previous_scene_item_id=before["canonical_l2a_id"], previous_version=before["version_id"],
                           current_scene_item_id=after["canonical_l2a_id"], current_version=after["version_id"],
                           inputs_available_before_utc=cutoff, lidar_source_date=lidar_date, roof_evidence_date=roof_date,
                           roof_evidence_basis=basis, engineering_source_url=after["engineering_source_url"],
                           roof_evidence_url=url, eligibility=status))
    export(f"{STEM}_pair_manifest", ledger)
    return selected, ledger


def fit_native_pairs(pairs, geometry):
    wanted = {(r["candidate_id"], r["version_id"], r["canonical_l2a_id"]): r for p in pairs for r in p.values()}
    with closing(read_only(PIXELS)) as db:
        db.row_factory = sqlite3.Row
        observations, kernels = inputs(db)
        scenes = {r[0]: r[1:] for r in db.execute("SELECT scene_id,scene_item_id,crs FROM scene")}
    transforms = {crs: Transformer.from_crs(32614, crs, always_xy=True) for _, crs in set(scenes.values())}
    jobs, identities = [], {}
    for r in observations:
        item, crs = scenes[r["scene_id"]]
        key = r["candidate_id"], r["version_id"], item
        if key not in wanted:
            continue
        old = geometry[key[0]]
        east, north = transforms[crs].transform(float(old["center_east_m"]), float(old["center_north_m"]))
        radius = float(old["diameter_m"]) / 2
        shift = math.hypot(east - r["center_east_m"], north - r["center_north_m"])
        if shift + max(0, radius - r["radius_m"]) > CONTEXT_M - 16:
            raise ValueError(f"The dated geometry needs a larger native chip for {key}")
        if dt.datetime.fromisoformat(r["captured_at_utc"]) != dt.datetime.fromisoformat(wanted[key]["captured_at_utc"]):
            raise ValueError(f"Native observation time mismatch for {key}")
        shell = float(old["prior_rim_z_p95_native"]) - float(old["prior_ground_z_p50_native"])
        jobs.append((DATED, dict(r, center_east_m=east, center_north_m=north, radius_m=radius,
                                 outline_fit_status=old["fit_status"]), shell))
        identities[r["observation_id"]] = key
    if len(identities) != len(wanted) or set(identities.values()) != wanted.keys():
        raise ValueError("The dated core manifest differs from the development native pixel inventory")
    estimates = {identities[row["observation_id"]]: row for row in run_fits(jobs, kernels, PIXELS)}
    write_csv(DERIVED / f"{STEM}_observations.csv", FIELDS, ([r.get(k) for k in FIELDS] for r in estimates.values()))
    return list(pair_rows(pairs, estimates))


def main():
    write_geometry(CORE, GEOMETRY)
    geometry = {r["candidate_id"]: r for r in read_rows(GEOMETRY) if r["diameter_m"] and r["fitted_2011_gap_m"]}
    pairs, ledger = pairs_and_manifest(geometry, lidar_clocks(geometry), *roof_clocks())
    measurements = fit_native_pairs(pairs, geometry)
    export(f"{STEM}_development_pairs", measurements)
    grouped = defaultdict(list)
    for row in measurements:
        grouped[row["week_ending"], row["release_date"], row["terminal_id"], row["inputs_available_before_utc"]].append(row)
    export(f"{STEM}_development_terminals", [dict(week_ending=w, release_date=r, terminal_id=t,
            inputs_available_before_utc=c, geometry_mode=DATED, **aggregate(rows))
            for (w, r, t, c), rows in sorted(grouped.items())])
    counts = Counter((r["study_phase"], r["terminal_id"], r["eligibility"]) for r in ledger)
    report = dict(legacy_geometry_candidates=len(geometry), manifest_pairs=len(ledger),
                  manifest_counts=[dict(phase=p, terminal_id=t, eligibility=s, pairs=n) for (p, t, s), n in sorted(counts.items())],
                  development_pairs=len(measurements), resolved_pairs=sum(r["both_shadow_diagnostics_pass"] for r in measurements),
                  scope="Roof evidence dates use regulatory milestones or dated NAIP shadow observations. Present-day contour reconciliation and the same native-chip selection are retained. Physical estimates cover development only.")
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps({k: v for k, v in report.items() if k != "manifest_counts"}), flush=True)


if __name__ == "__main__":
    main()
