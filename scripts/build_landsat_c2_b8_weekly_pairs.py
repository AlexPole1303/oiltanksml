"""Materialize exact, unregistered B8 and B04 pair candidates for each gap week."""
import json
import math
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing

from build_landsat_c2_b8_measurements import NATIVE, STEM as MEASUREMENTS
from build_landsat_c2_holdout_gap_inventory import SIDES, STEM as GAP, parse, sentinel_candidates
from build_operator_development_inputs import one_per_key
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, RESOLVED, SOURCE, export, read_only
from collect_public import ROOT, write
from stage_landsat_c2_b8_inputs import planned_b8_endpoints

STEM = "landsat_c2_b8_weekly"
PAIR_ROUTES = (("B8", "B8"), ("B8", "B04"), ("B04", "B8"))


def week_inputs(plan):
    tanks = defaultdict(dict)
    for r in read_csv("holdout_optical_tank_weeks"):
        week, tank = r["week_ending"], r["candidate_id"]
        if week not in plan:
            continue
        if tank in tanks[week] or (r["release_date"], parse(r["cutoff_utc"])) != (
                plan[week]["release_date"], parse(plan[week]["forecast_cutoff_utc"])):
            raise ValueError(f"Dated tank identity or release clock differs for {week}/{tank}")
        tanks[week][tank] = r
    windows = defaultdict(list)
    with closing(read_only(SOURCE)) as db:
        for week, side, issue, start, end, cutoff in db.execute("""SELECT week_ending, endpoint, release_date,
                window_starts_after_utc, window_ends_by_utc, inputs_available_before_utc
                FROM eia_satellite_window WHERE study_phase='holdout' ORDER BY week_ending, endpoint"""):
            if week not in plan:
                continue
            if (issue, parse(cutoff)) != (plan[week]["release_date"], parse(plan[week]["forecast_cutoff_utc"])):
                raise ValueError(f"Source release clock differs for {week}/{side}")
            windows[week].append((side, parse(start), parse(end), parse(cutoff)))
    if windows.keys() != plan.keys() or any({w[0] for w in spans} != set(SIDES) or len(spans) != 2
                                          for spans in windows.values()):
        raise ValueError("Incomplete reporting windows for B8 gap weeks")
    return windows, tanks


def b8_endpoints(plan, windows, tanks):
    evidence = read_csv("landsat_c2_b8_stage_products")
    staged = one_per_key(evidence, "product_id")
    fits = one_per_key(read_csv(f"{MEASUREMENTS}_roofs"), "product_id", "candidate_id")
    with closing(read_only(NATIVE)) as db:
        db.row_factory = sqlite3.Row
        native = [dict(row) for row in db.execute("""SELECT w.product_id, w.candidate_id, w.terminal_id,
                    w.radius_m*2 AS diameter_m, w.pan_observation_id AS observation_id,
                    w.roof_touching_b8_cells AS b8_roof_center_cells,
                    w.roof_touching_b8_cells_passing_qa AS b8_roof_center_cells_pass_qa,
                    s.captured_at_utc, s.original_b8_sha512
                    FROM roof_window w JOIN scene s USING (product_id)""")]
    footprints = one_per_key(native, "product_id", "candidate_id")
    clocks = {(week, side): (start, end, cutoff) for week, spans in windows.items()
              for side, start, end, cutoff in spans}
    result = defaultdict(list)
    seen = set()
    for r, tank in planned_b8_endpoints(evidence, native, plan):
        week, side, product = r["week_ending"], r["endpoint"], r["product_id"]
        key = product, tank
        foot, stage, fit = footprints[key], staged[product,], fits.get(key)
        start, end, cutoff = clocks[week, side]
        capture = parse(r["captured_at_utc"])
        support = max(parse(stage["qa_and_angles_archive_at_utc"]), parse(r["local_level2_assets_at_utc"]),
                      parse(r["level1_generated_at_utc"]))
        if ((week, side, *key) in seen or not start < capture <= end or not capture <= support < cutoff
            or parse(r["cutoff_utc"]) != cutoff or stage["b8_download_url"] != r["pan_url"]
            or parse(foot["captured_at_utc"]) != capture
            or foot["original_b8_sha512"] != stage["expected_original_b8_sha512"]
            or stage["b8_status"] not in ("awaiting_original_b8", "verified_native_crop")
            or foot["terminal_id"] != tanks[week][tank]["terminal_id"]):
            raise ValueError(f"B8 source identity or timing differs for {week}/{side}/{key}")
        seen.add((week, side, *key))
        if fit and (stage["b8_status"] != "verified_native_crop" or fit["roof_type"] != "EFR"
            or parse(fit["optical_capture_at_utc"]) != capture
            or int(fit["observation_id"]) != foot["observation_id"]
            or fit["terminal_id"] != foot["terminal_id"] or float(fit["diameter_m"]) != foot["diameter_m"]
            or fit["b8_crop_sha256"] != stage["b8_crop_sha256"]):
            raise ValueError(f"B8 fit differs from its verified footprint for {key}")
        fit_status = fit["fit_status"] if fit else ("fit_not_run" if stage["b8_status"] == "verified_native_crop"
                                                   else stage["b8_status"])
        result[week, side, tank].append(dict(sensor="B8", version_id=product, canonical_id=product,
            capture_at_utc=capture.isoformat(), support_available_at_utc=support.isoformat(), diameter_m=foot["diameter_m"],
            observation_id=foot["observation_id"], observation_source=NATIVE.name,
            source_file=r["local_level2_metadata_path"], original_b8_filename=stage["original_b8_filename"],
            b8_input_status=stage["b8_status"], b8_historical_availability_verified=False, fit_status=fit_status,
            gap_m=fit["estimated_gap_m"] if fit and fit_status == RESOLVED else None))
    return result


def b04_endpoint(item):
    capture, available, r = item
    source = ROOT / r["tile_metadata_path"]
    if not source.is_file():
        raise ValueError(f"Sentinel-2 metadata missing for {r['version_id']}")
    return dict(sensor="B04", version_id=r["version_id"], canonical_id=r["canonical_l2a_id"],
        capture_at_utc=capture.isoformat(), support_available_at_utc=available.isoformat(), diameter_m=float(r["diameter_m"]),
        observation_id=r["observation_id"], observation_source=r["measurement_source"],
        source_file=r["tile_metadata_path"], original_b8_filename=None, b8_input_status=None,
        b8_historical_availability_verified=None,
        fit_status=r["fit_status"], gap_m=r["estimated_gap_m"])


def candidate_pairs(plan, windows, tanks, b8, b04):
    other_side = {"previous": "current", "current": "previous"}
    endpoints = {"B8": b8, "B04": {(week, side, tank): [b04_endpoint(item) for item in items]
        for (week, side, tank), items in b04.items() if (week, other_side[side], tank) in b8}}
    clocks = {(week, side): end for week, spans in windows.items() for side, _, end, _ in spans}
    for week, r in sorted(plan.items()):
        for tank, meta in sorted(tanks[week].items()):
            for a, b in PAIR_ROUTES:
                for previous in endpoints[a].get((week, "previous", tank), ()):
                    for current in endpoints[b].get((week, "current", tank), ()):
                        before, after = (parse(endpoint["capture_at_utc"]) for endpoint in (previous, current))
                        p, c = (clocks[week, side] for side in SIDES)
                        if (not before <= p < after <= c or not math.isclose(previous["diameter_m"],
                                current["diameter_m"], abs_tol=1e-8, rel_tol=0)):
                            raise ValueError(f"Incompatible observation geometry or chronology for {week}/{tank}")
                        yield dict(week_ending=week, release_date=r["release_date"],
                            forecast_cutoff_utc=r["forecast_cutoff_utc"], candidate_id=tank,
                            terminal_id=meta["terminal_id"], area_m2=float(meta["circular_area_m2"]),
                            route=f"{a}_{b}", capture_interval_hours=(after-before).total_seconds()/3600,
                            current_hours_before_eia_measurement=(c-after).total_seconds()/3600,
                            all_required_b8_staged=all(endpoint["b8_input_status"] in (None, "verified_native_crop")
                                                      for endpoint in (previous, current)),
                            pair_registration_status="pending_independent_registration",
                            **{f"{side}_{key}": value for side, endpoint in zip(SIDES, (previous, current), strict=True)
                               for key, value in endpoint.items()})


def validate_and_summarize(pairs, plan):
    reference = one_per_key(read_csv(f"{MEASUREMENTS}_footprint_qa_weeks"), "week_ending")
    grouped = defaultdict(list)
    seen = set()
    for r in pairs:
        key = tuple(r[k] for k in ("week_ending", "candidate_id", "route", "previous_version_id",
                                    "current_version_id"))
        if key in seen:
            raise ValueError(f"Repeated exact pair {key}")
        seen.add(key)
        grouped[r["week_ending"]].append(r)
    weeks = []
    for week, r in sorted(plan.items()):
        sources = grouped[week]
        two = {p["candidate_id"] for p in sources if p["route"] == "B8_B8"}
        mixed = {p["candidate_id"] for p in sources if p["route"] != "B8_B8"}
        qa = reference[week,]
        if (two | mixed != set(json.loads(qa["clear_candidate_ids_json"]))
            or (len(two), len(mixed)) != (int(qa["clear_two_b8_pair_candidates"]),
                                        int(qa["clear_b8_sentinel2_pair_candidates"]))):
            raise ValueError(f"Exact pairs disagree with full-roof QA on {week}")
        weeks.append(dict(week_ending=week, release_date=r["release_date"],
            forecast_cutoff_utc=r["forecast_cutoff_utc"], b04_status=r["b04_status"],
            exact_pair_candidates=len(sources), candidate_tanks=len(two | mixed),
            b8_only_route_tanks=len(two-mixed), mixed_only_route_tanks=len(mixed-two), both_route_tanks=len(two & mixed),
            b8_originals_json=json.dumps(sorted({p[f"{side}_original_b8_filename"] for p in sources
                                                for side in SIDES if p[f"{side}_sensor"] == "B8"})),
            tanks_with_all_required_b8_staged=len({p["candidate_id"] for p in sources if p["all_required_b8_staged"]}),
            tanks_with_both_fits_resolved=len({p["candidate_id"] for p in sources if
                                               all(p[f"{side}_fit_status"] == RESOLVED for side in SIDES)})))
    return weeks


def main():
    plan = {r["week_ending"]: r for r in read_csv(f"{GAP}_weeks")}
    windows, tanks = week_inputs(plan)
    b8 = b8_endpoints(plan, windows, tanks)
    b04, _ = sentinel_candidates(windows, tanks)
    pairs = list(candidate_pairs(plan, windows, tanks, b8, b04))
    weeks = validate_and_summarize(pairs, plan)
    export(f"{STEM}_pairs", pairs)
    export(f"{STEM}_weeks", weeks)
    summary = dict(gap_weeks=len(weeks), weeks_with_candidates=sum(r["candidate_tanks"] > 0 for r in weeks),
        candidate_tank_weeks=sum(r["candidate_tanks"] for r in weeks), exact_pair_candidates=len(pairs),
        exact_pairs_by_route=dict(Counter(r["route"] for r in pairs)),
        exact_b8_originals=len({r[f"{side}_original_b8_filename"] for r in pairs
                               for side in SIDES if r[f"{side}_sensor"] == "B8"}),
        tank_weeks_with_all_required_b8_staged=sum(r["tanks_with_all_required_b8_staged"] for r in weeks),
        tank_weeks_with_both_fits_resolved=sum(r["tanks_with_both_fits_resolved"] for r in weeks),
        scope="Candidate coverage with original B8 identities and available B04 fits. B8 support timestamps "
              "do not establish the historical availability of the original B8. Pair registration is pending.")
    payload = json.dumps(summary, indent=2) + "\n"
    write(DERIVED / f"{STEM}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
