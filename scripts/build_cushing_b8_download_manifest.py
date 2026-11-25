"""Build a single download list for exact B8 originals from both Cushing studies."""
import re
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from urllib.parse import urlsplit

from build_landsat_c2_holdout_gap_inventory import SIDES
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, read_only
from collect_landsat import LANDSAT
from collect_public import ROOT, write_csv

OUTPUT = DERIVED / "cushing_b8_download_manifest.csv"
ORIGINALS = (LANDSAT / "collection2_b8_original").relative_to(ROOT)


def native_scenes(database):
    with closing(read_only(DERIVED / database)) as db:
        rows = db.execute("""SELECT s.product_id, s.platform, s.captured_at_utc, s.original_b8_sha512,
            COUNT(w.candidate_id), SUM(w.roof_touching_b8_cells > 0 AND
                w.roof_touching_b8_cells = w.roof_touching_b8_cells_passing_qa)
            FROM scene s JOIN roof_window w USING (product_id) GROUP BY s.product_id""").fetchall()
    return {p: (platform, datetime.fromisoformat(capture), digest, windows, full_roofs)
            for p, platform, capture, digest, windows, full_roofs in rows}


def manifest_row(request, stage, native, priority):
    identity = request["product_id"]
    platform, capture, digest, windows, full_roofs = native
    url = urlsplit(request["pan_url"])
    filename = f"{identity}_B8.TIF"
    if (
        (request["platform"], datetime.fromisoformat(request["captured_at_utc"])) != (platform, capture)
        or (stage["platform"], stage["purpose"]) != (platform, request["purpose"])
        or request["pan_url"] != stage["b8_download_url"]
        or url.scheme != "https" or url.netloc != "landsatlook.usgs.gov"
        or url.path.split("/")[-2:] != [identity, filename]
        or stage["original_b8_filename"] != filename
        or not re.fullmatch(r"[0-9a-f]{128}", digest)
        or digest != stage["expected_original_b8_sha512"]
        or ("roof_windows" in stage and int(stage["roof_windows"]) != windows)
    ):
        raise ValueError(f"Inconsistent B8 source records for {identity}")
    status = stage["b8_status"]
    verified = status == "verified_native_crop"
    if status not in ("verified_native_crop", "awaiting_original_b8") or (
        verified and (stage["original_b8_sha512"] != digest or not stage["b8_crop_path"])
    ) or (not verified and (stage["original_b8_sha512"] or stage["b8_crop_path"])):
        raise ValueError(f"Inconsistent B8 staging status for {identity}")
    return dict(priority=priority, product_id=identity, platform=platform, captured_at_utc=capture.isoformat(),
        purpose=request["purpose"], external_control_observations=int(request["external_control_observations"]),
        planned_gap_weeks=int(request.get("contributing_gap_weeks", 0)),
        planned_gap_tank_weeks=int(request.get("contributing_gap_tank_weeks", 0)),
        sentinel2_calibration_candidates=int(request.get("sentinel2_calibration_candidates", 0)),
        roof_windows=windows, full_roof_qa_and_angles=full_roofs, b8_status=status,
        original_b8_filename=filename, original_b8_relative_path=str(ORIGINALS / filename),
        expected_original_b8_sha512=digest, b8_download_url=request["pan_url"])


def download_coverage(rows):
    by_file = {r["original_b8_filename"]: r for r in rows}
    staged = {name for name, r in by_file.items() if r["b8_status"] == "verified_native_crop"}
    context = {}
    for r in read_csv("landsat_c2_context_registration_pairs"):
        key = (r["week_or_control_day"],r["candidate_id"],r["earlier_version_id"],r["later_version_id"],
               r["earlier_source"],r["later_source"])
        if key in context:
            raise ValueError(f"Duplicate B8 registration context for {key}")
        context[key] = r["registration_status"] == "checked_offset" and r["red_context_before_cutoff"] == "True"
    requirements = defaultdict(set)
    context_requirements = defaultdict(set)
    for r in read_csv("landsat_c2_b8_weekly_pairs"):
        names = frozenset(r[f"{side}_original_b8_filename"] for side in SIDES if r[f"{side}_sensor"] == "B8")
        if len(names) != r["route"].split("_").count("B8") or not names or any(
            name not in by_file or by_file[name]["purpose"] != "gap_pair" for name in names
        ) or str(names <= staged) != r["all_required_b8_staged"]:
            raise ValueError(f"B8 pair file requirements differ for {r['week_ending']}/{r['candidate_id']}")
        key = r["week_ending"],r["candidate_id"]
        requirements[key].add(names)
        if context.pop((*key,r["previous_version_id"],r["current_version_id"],
                        r["previous_source_file"],r["current_source_file"])):
            context_requirements[key].add(names)
    if context:
        raise ValueError("Unused B8 registration context pairs")
    if {name for options in requirements.values() for names in options for name in names} != {
        name for name, r in by_file.items() if r["purpose"] == "gap_pair"
    }:
        raise ValueError("Exact pair files differ from the weekly B8 download set")
    single_file, next_file, involved = defaultdict(set), defaultdict(set), defaultdict(set)
    next_checked = defaultdict(set)
    for key, options in requirements.items():
        already_staged = any(names <= staged for names in options)
        for names in options:
            for name in names:
                involved[name].add(key)
            if len(names) == 1:
                name, = names
                single_file[name].add(key)
            missing = names - staged
            if not already_staged and len(missing) == 1:
                name, = missing
                next_file[name].add(key)
    for key, options in context_requirements.items():
        if any(names <= staged for names in options):
            continue
        for names in options:
            missing = names - staged
            if len(missing) == 1:
                name, = missing
                next_checked[name].add(key)
    for row in rows:
        name = row["original_b8_filename"]
        row.update(full_qa_candidate_tank_weeks=len(involved[name]),
            mixed_candidate_tank_weeks_with_this_b8_alone=len(single_file[name]),
            additional_candidate_tank_weeks_if_staged_next=len(next_file[name]),
            additional_candidate_weeks_if_staged_next=len({week for week, _ in next_file[name]}),
            additional_red_context_checked_tank_weeks_if_staged_next=len(next_checked[name]))


def main():
    requests = sorted(read_csv("landsat_c2_b8_acquisition_queue"), key=lambda r: int(r["priority"]))
    staged = read_csv("landsat_c2_b8_stage_products")
    laser = sorted(read_csv("cushing_icesat2_b8_products"), key=lambda r: r["captured_at_utc"])
    weekly_native = native_scenes("landsat_c2_b8_native_geometry.sqlite")
    laser_native = native_scenes("cushing_icesat2_b8_native.sqlite")
    laser_controls = Counter(r["product_id"] for r in read_csv("cushing_icesat2_b8_laser_comparisons"))
    if (
        len(requests) != len(weekly_native) or {r["product_id"] for r in requests} != weekly_native.keys()
        or len(staged) != len(weekly_native) or {r["product_id"] for r in staged} != weekly_native.keys()
        or len(laser) != len(laser_native) or {r["product_id"] for r in laser} != laser_native.keys()
        or laser_controls.keys() != laser_native.keys() or weekly_native.keys() & laser_native.keys()
        or [r["purpose"] for r in requests[:2]] != ["lidar_2022", "state_aerial_2025"]
    ):
        raise ValueError("B8 download lists and native input databases disagree")
    laser_requests = [dict(r, pan_url=r["b8_download_url"],
                           external_control_observations=laser_controls[r["product_id"]]) for r in laser]
    native = weekly_native | laser_native
    stages = {r["product_id"]: r for r in staged + laser}
    ordered = requests[:2] + laser_requests + requests[2:]
    rows = [manifest_row(r, stages[r["product_id"]], native[r["product_id"]], i)
            for i, r in enumerate(ordered, 1)]
    download_coverage(rows)
    fields = tuple(rows[0])
    write_csv(OUTPUT, fields, ([row[k] for k in fields] for row in rows))
    print(f"{len(rows)} B8 originals in {OUTPUT.relative_to(ROOT)}", flush=True)
    for status, total in sorted(Counter(r["b8_status"] for r in rows).items()):
        print(f"{status}: {total}", flush=True)


if __name__ == "__main__":
    main()
