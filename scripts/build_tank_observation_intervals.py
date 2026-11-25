"""Derive structural change windows bounded by dated aerial observations."""
import csv
import io
from collections import defaultdict

from build_supplemental_tanks import indexed
from collect_public import ROOT, write

ABSENT = "no_aboveground_tank_seen"
TANK = "aboveground_tank_seen"
SHELL = "shell_or_construction_seen"
OTHER = "earlier_structure_different_footprint"
EVENT = {
    (ABSENT, TANK): "tank_visible_after_absence",
    (ABSENT, SHELL): "construction_visible_after_absence",
    (SHELL, TANK): "tank_visible_after_shell_or_construction",
    (OTHER, TANK): "later_tank_footprint_after_smaller_structure",
    (OTHER, SHELL): "construction_after_smaller_structure",
    (TANK, ABSENT): "tank_no_longer_visible",
    (SHELL, ABSENT): "construction_no_longer_visible",
}


def main():
    reference = ROOT / "data" / "reference"
    derived = ROOT / "data" / "derived"
    candidates = defaultdict(list)
    with (reference / "tank_aerial_visual_observations.csv").open() as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames
        for row in reader:
            candidates[row["candidate_id"]].append(row)
    supplementary = derived / "aerial_review" / "supplemental" / "index.csv"
    if supplementary.is_file():
        assessment = indexed(reference / "supplemental_tank_aerial_assessment.csv", "candidate_id")
        supplemental_rows = io.StringIO(newline="")
        observations_writer = csv.DictWriter(supplemental_rows, fieldnames=fields)
        observations_writer.writeheader()
        with supplementary.open() as source:
            for row in csv.DictReader(source):
                identity, year = row["candidate_id"], row["year"]
                observed = assessment[identity]
                structure = (TANK if year >= observed["first_aboveground_year"] else
                             SHELL if year == observed["earlier_shell_year"] else ABSENT)
                dates = row["acquisition_dates"].split("|")
                if len(dates) != 1 or not dates[0]:
                    raise ValueError(f"Expected a single observation date for {identity} in {year}")
                item = dict(candidate_id=identity, review_number=row["review_number"], observed_date=dates[0],
                            image_source="Oklahoma Aerial 2025" if year == "2025" else "USDA NAIP",
                            structure_observation=structure,
                            top_observation=observed["top_2025"] if year == "2025" else "not_assessed",
                            detail=observed["detail"], review_sheet=row["history_sheet"], review_row=row["history_row"],
                            review_column=("2010", "2013", "2015", "2017", "2019", "2021", "2023", "2025").index(year) + 1,
                            source_asset_paths=row["source_paths"])
                observations_writer.writerow(item)
                candidates[identity].append(item)
        write(derived / "tank_aerial_supplemental_observations.csv", supplemental_rows.getvalue().encode())
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(("candidate_id", "review_number", "event", "earlier_observed_date", "later_observed_date",
                     "earlier_observation", "later_observation", "earlier_review_sheet", "later_review_sheet"))
    count = 0
    for identity, observations in sorted(candidates.items()):
        observations.sort(key=lambda item: item["observed_date"])
        for first, last in zip(observations, observations[1:]):
            pair = first["structure_observation"], last["structure_observation"]
            if pair[0] == pair[1]:
                continue
            event = EVENT.get(pair, "change_in_observed_structure")
            writer.writerow((identity, first["review_number"], event, first["observed_date"], last["observed_date"],
                             *pair, first["review_sheet"], last["review_sheet"]))
            count += 1
    write(derived / "tank_aerial_change_intervals.csv", output.getvalue().encode())
    print(f"{count} changes between observed imagery dates for {len(candidates)} contours")


if __name__ == "__main__":
    main()
