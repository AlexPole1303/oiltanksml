"""Run the Umbra geometry checks on the independent public Capella Cushing collect."""
import json
import xml.etree.ElementTree as ET
from collections import Counter
from urllib.parse import urlencode

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from check_umbra_sicd_roof_repeatability import evaluate
from collect_public import RAW, get, write
from collect_umbra_sicd_windows import scene_windows, tank_inputs, with_ellipsoid_heights

STORE = RAW / "capella_cushing"
BUCKET = "https://capella-open-data.s3.us-west-2.amazonaws.com/"
SICD = "CAPELLA_C06_SM_SICD_HH_20240121025802_20240121025806"
SLC = SICD.replace("_SICD_", "_SLC_")
LICENCE = "https://registry.opendata.aws/capella_opendata/"


def verify():
    prefix = "data/2024/1/21/"
    key = f"{prefix}{SICD}/{SICD}.ntf"
    listing = ET.fromstring(get(BUCKET + "?" + urlencode({"list-type": 2, "prefix": key}),
                               STORE / "source_listing.xml"))
    entry, = listing.findall("{*}Contents")
    if entry.findtext("{*}Key") != key or listing.findtext("{*}IsTruncated") != "false":
        raise ValueError("Capella SICD object listing does not match the expected product")
    metadata_url = BUCKET + f"{prefix}{SLC}/{SLC}_extended.json"
    metadata = json.loads(get(metadata_url, STORE / "slc_extended.json"))
    collect = metadata["collect"]
    return dict(product_id=SICD, platform=collect["platform"], capture_start_utc=collect["start_timestamp"],
        capture_end_utc=collect["stop_timestamp"], sicd_url=BUCKET + key,
        sicd_etag=entry.findtext("{*}ETag").strip('"'), sicd_bytes=int(entry.findtext("{*}Size")),
        sicd_s3_last_modified_utc=entry.findtext("{*}LastModified"), companion_slc_metadata_url=metadata_url,
        licence_source=LICENCE)


def main():
    scene = verify()
    tanks = with_ellipsoid_heights(tank_inputs())
    windows = scene_windows(dict(scene, product_name=SICD), list(tanks.values()), store=STORE)
    export("capella_sicd_control_window_index", windows)
    fits = evaluate(read("capella_sicd_control_window_index"))
    export("capella_sicd_control_arc_fits", fits)
    results = {}
    for roof_type in sorted({r["roof_type"] for r in fits}):
        subset = [r for r in fits if r["roof_type"] == roof_type]
        results[roof_type] = dict(tanks=len(subset), local_status=dict(Counter(r["roof_use_status"] for r in subset)),
            scene_registered_status=dict(Counter(r["scene_registered_use_status"] for r in subset)))
    report = dict(scene=scene, native_windows=len({r["window_file"] for r in windows}), roof_types=results,
        interpretation="The same geometric arc tests and quality thresholds as the Umbra pilot are applied to a Capella stripmap. "
        "There is one capture, so cross-sensor repeatability and absolute roof height are not measured. "
        "Source imagery © Capella Space, open data CC BY 4.0.")
    write(DERIVED / "capella_sicd_control_summary.json", (json.dumps(report, indent=2) + "\n").encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
