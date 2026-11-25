"""List mapped tank candidates inside public cadastral sections named by permits."""
import csv
import io
import json
from collections import defaultdict

from pyproj import Transformer
from shapely import transform
from shapely.geometry import Point, shape

from collect_public import RAW, ROOT, write

# The source PDFs identify legal sections. These polygons can contain several operators.
SECTIONS = {
    "enbridge_cushing": ("OK170170N0050E0", ("10", "14", "15", "22")),
    "enbridge_grandstaff": ("OK170180N0050E0", ("27",)),
    "rose_rock_cushing_north": ("OK170180N0050E0", ("22",)),
    "magellan_cushing_south": ("OK170170N0050E0", ("26",)),
    "magellan_cushing_central": ("OK170170N0050E0", ("14",)),
    "southbow_cushing": ("OK170170N0050E0", ("27",)),
    "plains_cushing": ("OK170170N0050E0", ("23", "24")),
}


def main():
    data = ROOT / "data" / "derived"
    land = json.loads((RAW / "documents" / "blm_plss_sections.geojson").read_bytes())
    tanks = json.loads((data / "osm_tank_candidates.geojson").read_bytes())
    with (ROOT / "data" / "reference" / "permit_documents.csv").open() as stream:
        records = sorted(csv.DictReader(stream), key=lambda row: (row["memo_date"], row["document_id"]))
        documents = {row["facility_key"]: row for row in records}
    to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
    sections = defaultdict(list)
    for section in land["features"]:
        props = section["properties"]
        key = props["PLSSID"], props["FRSTDIVNO"].lstrip("0")
        sections[key].append((props["FRSTDIVID"], shape(section["geometry"])))
    out = io.StringIO(newline="")
    columns = ("candidate_id", "facility_key", "plss_section_id", "centroid_distance_from_reference_m",
               "equivalent_diameter_m", "evidence_type", "permit_source_url")
    writer = csv.writer(out)
    writer.writerow(columns)
    tally = defaultdict(int)
    for candidate in tanks["features"]:
        point = shape(candidate["geometry"]).centroid
        point_metric = transform(point, to_utm.transform, interleaved=False)
        attributes = candidate["properties"]
        for facility, (township, numbers) in SECTIONS.items():
            document = documents[facility]
            anchor = Point(float(document["longitude"]), float(document["latitude"]))
            anchor_metric = transform(anchor, to_utm.transform, interleaved=False)
            for number in numbers:
                for identity, polygon in sections[township, number]:
                    if polygon.covers(point):
                        writer.writerow((attributes["candidate_id"], facility, identity,
                                         round(anchor_metric.distance(point_metric), 2),
                                         attributes["equivalent_diameter_m"], "same_plss_section",
                                         document["source_url"]))
                        tally[facility] += 1
    write(data / "tank_permit_section_candidates.csv", out.getvalue().encode())
    print(dict(tally))


if __name__ == "__main__":
    main()
