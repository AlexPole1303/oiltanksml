"""Compare permit cohorts with historical aerial structure observations without using oil levels."""

import csv
import json
from collections import Counter, defaultdict

import rasterio
from PIL import Image, ImageDraw
from rasterio.windows import Window

from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write

STEM = "cushing_grandstaff_construction"
REVIEW = ROOT / "data" / "reference" / "grandstaff_historic_aerial_review.csv"
CENTRAL = "grandstaff_central_2004_2006"
STATES = {"no_shell", "circular_pad", "raised_shell_without_clear_roof", "roof_and_shell"}


def observations(tanks, reviewed):
    rows = []
    for scene in read("cushing_grandstaff_historic_aerial_dates"):
        candidate, day = scene["candidate_id"], scene["source_acquisition_date"]
        state = reviewed[candidate][day]
        if state not in STATES or tanks[candidate]["cohort_id"] != scene["cohort_id"]:
            raise ValueError(f"Unexpected observation for {candidate} on {day}")
        rows.append(dict(**scene, structure_observation=state, source_review_file=str(REVIEW.relative_to(ROOT)),
                         review_method="manual_comparison_at_native_resolution",
                         review_atlas=f"data/derived/aerial_review/{scene['cohort_id']}_historic_naip.png"))
    if len(rows) != len(tanks) * 4 or len({(r["candidate_id"], r["source_acquisition_date"]) for r in rows}) != len(rows):
        raise ValueError("Incomplete aerial review grid")
    return rows


def build_atlases(rows):
    by_cohort = defaultdict(list)
    for row in rows:
        by_cohort[row["cohort_id"]].append(row)
    size = 150
    for group in by_cohort.values():
        candidates = sorted({row["candidate_id"] for row in group})
        days = sorted({row["source_acquisition_date"] for row in group})
        canvas = Image.new("RGB", (size * len(days), size * len(candidates)), "#bbb")
        draw = ImageDraw.Draw(canvas)
        for column, day in enumerate(days):
            selected = {row["candidate_id"]: row for row in group if row["source_acquisition_date"] == day}
            image, = {row["crop_file"] for row in selected.values()}
            with rasterio.open(ROOT / image) as dataset:
                margin = round(43 / dataset.res[0])
                for line, candidate in enumerate(candidates):
                    row = selected[candidate]
                    px, py = ~dataset.transform * (float(row["easting_m"]), float(row["northing_m"]))
                    if not (margin <= px < dataset.width - margin and margin <= py < dataset.height - margin):
                        raise ValueError(f"Atlas window extends outside source crop for {candidate}")
                    window = Window(round(px) - margin, round(py) - margin, margin * 2, margin * 2)
                    pixels = dataset.read(window=window).transpose(1, 2, 0)
                    x, y = size * column, size * line
                    canvas.paste(Image.fromarray(pixels).resize((size, size), Image.Resampling.NEAREST), (x, y))
                    draw.rectangle((x, y, x + 106, y + 17), fill="black")
                    draw.text((x + 3, y + 3), f"{day[:4]} {candidate[-6:]}", fill="white")
        path = ROOT / group[0]["review_atlas"]
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)


def permit_years():
    years = defaultdict(dict)
    for row in read("permit_tanks"):
        if row["document_id"] in {"8122", "8303"}:
            years[row["tank_id"]][row["document_id"]] = row["construction_or_modification_raw"]
    return years


def candidate_sets(tanks, reviewed, years):
    central = {k for k, v in tanks.items() if v["cohort_id"] == CENTRAL}
    roofs = {k for k in central if reviewed[k]["2005-06-06"] == "roof_and_shell"}
    equipment = defaultdict(list)
    for candidate, tank in tanks.items():
        for ident in json.loads(tank["possible_equipment_ids_json"]):
            equipment[ident].append(candidate)
    year = {}
    for ident in equipment:
        a, b = (int(years[ident][document]) for document in ("8122", "8303"))
        if a != b:
            raise ValueError(f"Permit dates disagree for Grandstaff tank {ident}")
        year[ident] = a
    if len(roofs) != 4 or sum(year[ident] == 2004 for ident, ids in equipment.items() if set(ids) == central) != len(roofs):
        raise ValueError("Central 2004 cohort requires review")
    for ident, ids in sorted(equipment.items()):
        ids = set(ids)
        if ids == central:
            reduced = roofs if year[ident] == 2004 else central - roofs
            status = "conditional_construction_cohort"
            assumption = ("The four complete shells seen by 2005-06-06 are the four structures listed for 2004. "
                          "This assumes no replacement or renumbering and that the dates in the permit describe "
                          "construction of the current shell. Earlier construction within 2005 is possible.")
        else:
            reduced, status, assumption = ids, "original_cohort", ""
        yield dict(
            equipment_id=ident, terminal_id="enbridge_blueknight", reported_construction_year=year[ident],
            status=status, assumption=assumption, original_count=len(ids), candidate_count=len(reduced),
            original_candidate_ids_json=json.dumps(sorted(ids)), candidate_ids_json=json.dumps(sorted(reduced)),
            individual_contour_verified=False, admissible_as_paired_gauge=False,
            image_source="data/derived/cushing_grandstaff_historic_aerial_dates.csv",
            permit_source="https://applications.deq.ok.gov/permitspublic/storedpermits/8122.pdf#page=4",
        )


def main():
    tanks = {r["candidate_id"]: r for r in read("tank_engineering_registry")
             if r["documented_terminal_id"] == "enbridge_blueknight"}
    with REVIEW.open(newline="") as stream:
        reviewed = {r["candidate_id"]: r for r in csv.DictReader(stream)}
    if set(tanks) != set(reviewed):
        raise ValueError("Grandstaff review and roster have different footprints")
    aerial = observations(tanks, reviewed)
    build_atlases(aerial)
    candidates = list(candidate_sets(tanks, reviewed, permit_years()))
    gauge_rows = []
    index = {row["equipment_id"]: row for row in candidates}
    for gauge in read("cushing_operator_inspection_levels"):
        if gauge["terminal_id"] != "enbridge_blueknight":
            continue
        mapping = index[gauge["equipment_id"]]
        gauge_rows.append(dict(
            **{key: gauge[key] for key in ("equipment_id", "inspection_local_date", "reported_state", "reported_liquid_level",
                                           "source_document_id", "source_pages")},
            **{key: mapping[key] for key in ("status", "original_count", "candidate_count", "candidate_ids_json",
                                            "individual_contour_verified", "admissible_as_paired_gauge")},
        ))
    for suffix, rows in (("observations", aerial), ("candidate_sets", candidates), ("inspection_candidates", gauge_rows)):
        export(f"{STEM}_{suffix}", rows)
    stats = dict(
        contours=len(tanks), observation_rows=len(aerial), observed_source_dates=sorted({r["source_acquisition_date"] for r in aerial}),
        central_roof_and_shell_counts_by_date=dict(Counter(
            r["source_acquisition_date"] for r in aerial if r["cohort_id"] == CENTRAL and r["structure_observation"] == "roof_and_shell"
        )),
        equipment_with_narrower_conditional_set=sum(r["candidate_count"] < r["original_count"] for r in candidates),
        inspection_rows_with_narrower_conditional_set=sum(r["candidate_count"] < r["original_count"] for r in gauge_rows),
        verified_individual_mappings=sum(r["individual_contour_verified"] for r in candidates),
        northern_100000bbl_contours_seen_in_2003_with_no_shell_in_2006=[
            k for k, r in reviewed.items()
            if tanks[k]["cohort_id"] == "grandstaff_north_100000bbl"
            and r["2003-07-11"] == "roof_and_shell" and r["2006-04-18"] == "no_shell"
        ],
    )
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(stats, indent=2) + "\n").encode())
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
