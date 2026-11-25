"""Read the reflectance conversion for each local Sentinel-2 product and optical band."""
import json
import math
import re
import xml.etree.ElementTree as ET

from collect_public import RAW, ROOT, write_csv
from collect_s2_l1c import CHANNELS, OUT as L1C, source_metadata_bytes, source_metadata_path

BAND_ID = {"B02": 1, "B03": 2, "B04": 3, "B08": 7}
L2A_STORES = ("imagery_core", "imagery_historical", "sentinel2_supplemental", "sentinel2_supplemental_original")
BASELINE = re.compile(r"_N(\d{4})_")


def source_products():
    products = {}
    for path in sorted(L1C.glob("**/granules/*/*/item.json")):
        manifest = json.loads(path.read_bytes())
        version = manifest["product_name"]
        xml = source_metadata_path(path.parent / "MTD_MSIL1C.xml")
        products.setdefault(("L1C", version), (version, manifest["captured_at_utc"], xml, path, {}))
    for store in L2A_STORES:
        for path in sorted((RAW / store / "sentinel2").glob("*/*/item.json")):
            item = json.loads(path.read_bytes())
            version, props = item["id"], item["properties"]
            xml = source_metadata_path(RAW / "sentinel2_metadata" / version / "product_metadata.xml")
            products.setdefault(("L2A", version), (props["s2:product_uri"], props["datetime"], xml, path, item["assets"]))
    return products


def read_xml(path, level, product):
    if not path.is_file():
        return None, {}, None, None, "missing_product_xml"
    try:
        root = ET.fromstring(source_metadata_bytes(path))
    except (ET.ParseError, OSError, EOFError):
        return None, {}, None, None, "unreadable_product_xml"
    if level == "L1C":
        quant_tag, offset_tag = "QUANTIFICATION_VALUE", "RADIO_ADD_OFFSET"
    else:
        quant_tag, offset_tag = "BOA_QUANTIFICATION_VALUE", "BOA_ADD_OFFSET"
    quant = root.findtext(f".//{{*}}{quant_tag}")
    offsets = {int(node.get("band_id")): float(node.text) for node in root.findall(f".//{{*}}{offset_tag}")}
    specials = {node.findtext("{*}SPECIAL_VALUE_TEXT"): int(node.findtext("{*}SPECIAL_VALUE_INDEX"))
                for node in root.findall(".//{*}Special_Values")}
    baseline = BASELINE.search(product)
    if not offsets and baseline and int(baseline[1]) < 400:
        offsets = {band: 0.0 for band in BAND_ID.values()}
    value = float(quant) if quant is not None and float(quant) > 0 else None
    return value, offsets, specials.get("NODATA"), specials.get("SATURATED"), "read_product_xml"


def rows():
    for (level, version), (product, captured, xml, manifest, assets) in sorted(source_products().items()):
        quant, offsets, nodata, saturation, xml_status = read_xml(xml, level, product)
        match = BASELINE.search(product)
        baseline = f"{match[1][:2]}.{match[1][2:]}" if match else None
        for channel, band in CHANNELS.items():
            add = offsets.get(BAND_ID[band])
            stac = next(iter(assets.get(channel, {}).get("raster:bands", [])), {})
            stac_scale, stac_offset = stac.get("scale"), stac.get("offset")
            scale, offset = None, None
            status = xml_status if xml_status != "read_product_xml" else "incomplete_product_xml"
            if quant is not None and add is not None:
                scale, offset = 1 / quant, add / quant
                status = "product_xml"
                if stac_scale is not None and stac_offset is not None:
                    status = ("product_xml_matches_stac" if math.isclose(scale, stac_scale, abs_tol=1e-12)
                              and math.isclose(offset, stac_offset, abs_tol=1e-12) else "product_xml_stac_disagree")
                    if status == "product_xml_stac_disagree":
                        scale, offset = None, None
            elif stac_scale is not None and stac_offset is not None:
                scale, offset, status = stac_scale, stac_offset, "stac_only"
            yield (level, version, product, captured, baseline, band, channel, quant, add, scale, offset,
                   nodata if nodata is not None else stac.get("nodata"), saturation, stac_scale, stac_offset, status,
                   str(xml.relative_to(ROOT)) if xml.is_file() else None, str(manifest.relative_to(ROOT)))


def main():
    fields = ("level", "version_id", "product_name", "captured_at_utc", "processing_baseline", "band", "channel",
              "quantification_value", "add_offset_dn", "reflectance_scale", "reflectance_offset", "nodata_dn",
              "saturated_dn", "stac_scale", "stac_offset", "metadata_status", "product_xml", "source_manifest")
    output = ROOT / "data" / "derived" / "sentinel2_band_radiometry.csv"
    records = list(rows())
    write_csv(output, fields, records)
    print(f"{len(records)} Sentinel-2 product-band calibration records")


if __name__ == "__main__":
    main()
