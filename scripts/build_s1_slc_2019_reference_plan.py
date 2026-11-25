"""Reconcile Copernicus and ASF burst records for the 2019 radar reference pass."""
import json
from collections import defaultdict
from math import cos, radians
from urllib.parse import urlencode, urlparse

from pyproj import Transformer
from shapely import make_valid, transform
from shapely.geometry import Point, shape

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_public import BBOX, RAW, ROOT, get, write

ROOT_STORE = RAW / "sentinel1_cushing_catalog/bursts"
CDSE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Bursts"
ASF = "https://api.daac.asf.alaska.edu/services/search/param"
WORST_NOMINAL_SLC_SLANT_RESOLUTION_M = 3.5


def source_records(product, product_id):
    west, south, east, north = BBOX.split(",")
    polygon = f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    criteria = f"ParentProductId eq '{product_id}' and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
    query = CDSE + "?" + urlencode({"$filter": criteria, "$count": "true", "$top": 100})
    copernicus_file = ROOT_STORE / f"{product_id}.json"
    response = json.loads(get(query, copernicus_file))
    cdse = response["value"]
    if not cdse or len(cdse) != response["@odata.count"] or response.get("@odata.nextLink"):
        raise ValueError("Incomplete Copernicus burst response")
    if any(b["ParentProductId"] != product_id or b["ParentProductName"] != product for b in cdse):
        raise ValueError("Unexpected Copernicus SLC parent product")
    ids = sorted({f"{b['RelativeOrbitNumber']:03d}_{b['BurstId']:06d}_{b['SwathIdentifier']}" for b in cdse})
    fields = {"processingLevel": "BURST", "fullBurstID": ",".join(ids), "start": "2019-08-13T00:27:00Z",
              "end": "2019-08-13T00:29:00Z", "output": "geojson"}
    asf_file = ROOT_STORE / "asf_20190813_reference.geojson"
    asf = json.loads(get(ASF + "?" + urlencode(fields), asf_file))["features"]
    return cdse, asf, (str(copernicus_file.relative_to(ROOT)), str(asf_file.relative_to(ROOT)))


def reconcile(product, cdse, asf, project):
    nasa = {}
    for item in asf:
        p = item["properties"]
        b = p["burst"]
        key = b["absoluteBurstID"], b["subswath"], p["polarization"]
        url = urlparse(p["url"])
        if (url.netloc != "sentinel1-burst.asf.alaska.edu" or url.path.split("/")[1] != product.removesuffix(".SAFE")
                or p["platform"] != "SENTINEL-1B" or key in nasa):
            raise ValueError("ASF burst identity disagrees with the reference acquisition")
        nasa[key] = item
    grouped = defaultdict(dict)
    for b in cdse:
        burst, swath, pol = b["AbsoluteBurstId"], b["SwathIdentifier"], b["PolarisationChannels"]
        item = nasa[burst, swath, pol]
        p = item["properties"]
        if b["PlatformSerialIdentifier"] != "B" or "/measurement/s1b-" not in b["S3Path"] or pol in grouped[burst, swath]:
            raise ValueError("Duplicate or inconsistent Copernicus burst identity")
        footprints = [transform(make_valid(shape(s)), project, interleaved=False)
                      for s in (b["GeoFootprint"], item["geometry"])]
        grouped[burst, swath][pol] = (b, p, footprints[0].intersection(footprints[1]))
    if len(cdse) != len(nasa) or any(set(group) != {"VV", "VH"} for group in grouped.values()):
        raise ValueError("Expected a complete VV/VH cross-catalog match")
    return grouped


def main():
    calibration = read("s1_naip_2019_calibration_inputs")
    aerial = {r["candidate_id"]: r for r in read("naip_2019_roof_shadow_reference")}
    references = {(r["product_name"], r["product_id"]) for r in read("s1_reference_nearest_products")
                  if r["reference_day"] == "2019-08-12" and r["product_type"] == "IW_SLC__1S"
                  and float(r["maximum_separation_hours"]) <= 24}
    (product, identity), = references
    cdse, asf, paths = source_records(product, identity)
    project = Transformer.from_crs(4326, 32614, always_xy=True).transform
    pairs = reconcile(product, cdse, asf, project)
    footprints = {key: pair["VV"][2].intersection(pair["VH"][2]) for key, pair in pairs.items()}
    output = []
    for row in calibration:
        tank = row["candidate_id"]
        source = aerial[tank]
        center = Point(*project(float(source["longitude"]), float(source["latitude"])))
        area = center.buffer(float(source["radius"]) + 120)
        selected = max(footprints, key=lambda key: (footprints[key].intersection(area).area,
                                                     footprints[key].boundary.distance(center)))
        fraction = area.intersection(footprints[selected]).area / area.area
        if not fraction:
            raise ValueError(f"No matching SLC burst geometry for {tank}")
        vv, vh = (pairs[selected][pol] for pol in ("VV", "VH"))
        gap = float(row["aerial_gap_low_m"]) if row["qualified_aerial_shadow_reference"] == "True" else None
        low = gap * cos(radians(float(row["incidence_angle_deg"]))) if gap is not None else None
        output.append(dict(candidate_id=tank, roof_type=row["roof_type"], qualified_aerial_shadow_reference=gap is not None,
            original_slc_parent_product=product, original_slc_parent_id=identity, selected_absolute_burst_id=selected[0],
            selected_swath=selected[1], context_fraction_in_both_polarizations_and_catalogs=round(fraction, 7),
            full_context_in_one_burst=footprints[selected].covers(area),
            slant_displacement_at_aerial_low_m=low,
            slant_displacement_low_at_least_3_5m=low >= WORST_NOMINAL_SLC_SLANT_RESOLUTION_M if low is not None else None,
            cdse_vv_burst_uuid=vv[0]["Id"], cdse_vh_burst_uuid=vh[0]["Id"], asf_vv_file=vv[1]["fileID"],
            asf_vh_file=vh[1]["fileID"], asf_vv_data_url=vv[1]["url"], asf_vh_data_url=vh[1]["url"],
            asf_vv_annotation_url=vv[1]["additionalUrls"][0], asf_vh_annotation_url=vh[1]["additionalUrls"][0],
            cdse_source_file=paths[0], asf_source_file=paths[1]))
    export("s1_slc_2019_reference_burst_plan", output)
    qualified = [row for row in output if row["qualified_aerial_shadow_reference"]]
    report = dict(parent_product=product, cross_checked_burst_records=len(cdse), dual_pol_burst_footprints=len(pairs),
        cdse_display_names_with_conflicting_satellite_prefix=sum(b["Name"].split("-")[0] != product.split("_")[0] for b in cdse),
        contours=len(output), qualified_aerial_references=len(qualified),
        qualified_aerial_references_with_full_single_burst_context=sum(row["full_context_in_one_burst"] for row in qualified),
        qualified_references_with_lower_slant_displacement_at_least_3_5m=sum(row["slant_displacement_low_at_least_3_5m"] for row in qualified),
        interpretation="The acquisition and polarization records agree by parent SAFE, platform fields, sub-swath "
        "and burst ID. Copernicus display labels have an inconsistent platform prefix for this acquisition. "
        "The proposed windows cover the reference roof plus a 120 m margin when marked full. Slant displacement "
        "is a geometric scale based on the aerial range, not a detected SLC return. No SLC samples were downloaded.")
    write(DERIVED / "s1_slc_2019_reference_burst_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
