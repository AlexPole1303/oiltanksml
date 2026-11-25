"""Recover exact Collection 1 T1 B8 crops for Aerial Oklahoma flight dates."""

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import rasterio

from audit_cushing_landsat8_rapid_archive import stage
from audit_cushing_landsat8_rt_tier_controls import final_scenes
from build_aerialok_satellite_calendar import dates_and_roofs, day_offset
from build_cross_sensor_shadow import published_lsf
from build_landsat_pan_roof_angles import prepare_tool
from build_s1_reference_matches import export
from build_weekly_shadow_panel import DERIVED, RESOLVED
from build_weekly_shadow_validation import legacy_geometry
from collect_landsat import GCS_OBJECT
from collect_landsat_c2_registration_context import gcs_version
from collect_public import RAW, ROOT, write
from measure_cushing_landsat8_rapid import measure_scene


STEM = "aerialok_landsat8_t1"
STORE = RAW / "landsat" / STEM


def inventory(geometry):
    dates, first_seen = dates_and_roofs(geometry)
    for final, directory in final_scenes().values():
        capture = final["captured_at_utc"]
        flights = {flight for flight, _ in day_offset(capture, dates)}
        if not flights:
            continue
        product = final["product_id"]
        objects = {Path(r["name"]).name: r for r in json.loads((directory / "objects.json").read_bytes())["records"]}
        assets = {key: objects[f"{product}_{key}"] for key in ("B8.TIF", "BQA.TIF", "MTL.txt", "ANG.txt")}
        with rasterio.open(directory / "pan.tif") as local:
            if local.crs.to_epsg() != 32614:
                raise ValueError(f"Unsupported Landsat geometry {product}")
            west, south, east, north = local.bounds
        selected = []
        for roof, g in geometry.items():
            x, y = (float(g[k]) for k in ("center_east_m", "center_north_m"))
            if west <= x <= east and south <= y <= north and first_seen[roof] <= max(flights):
                selected.append(roof)
        if selected:
            pan = assets["B8.TIF"]
            yield dict(product_id=product, captured_at_utc=capture, week_ending="",
                       archive_bundle_available_utc=max((a["timeCreated"] for a in assets.values()), key=datetime.fromisoformat),
                       source_index=str((directory / "objects.json").relative_to(ROOT)),
                       b8_url=GCS_OBJECT+pan["name"], b8_archive_crc32c=pan["crc32c"], b8_archive_bytes=pan["size"],
                       dated_candidate_ids="|".join(sorted(selected)))


def main():
    geometry = legacy_geometry()
    tool, lsf = prepare_tool(), published_lsf()
    products = sorted(inventory(geometry), key=lambda r: r["captured_at_utc"])

    def fit(product):
        archive = {"size": product["b8_archive_bytes"], "crc32c": product["b8_archive_crc32c"]}
        product = dict(product, b8_archive_generation=gcs_version(product["b8_url"], archive))
        receipt = stage(product, tool, geometry, store=STORE)
        source = {k: v for k, v in receipt.items() if k != "roofs"}
        result = measure_scene(product | source, geometry, lsf, store=STORE)
        print(product["product_id"], len(result), "roof fits", flush=True)
        return product | source, result

    with ThreadPoolExecutor(max_workers=4) as pool:
        completed = list(pool.map(fit, products))
    sources = [r for r, _ in completed]
    records = [r for _, fits in completed for r in fits]
    export(f"{STEM}_sources", sources)
    export(f"{STEM}_fits", records)
    summary = dict(scenes=len(sources), roof_fits=len(records), fit_status=Counter(r["fit_status"] for r in records),
                   passed=sum(r["fit_status"] == RESOLVED and r["center_at_boundary"] is False for r in records),
                   scope="T1 source pixels and angle coefficients for retrospective Aerial Oklahoma controls. "
                         "The Collection 1 T1 archive clock is retained as an input attribute.")
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(summary, indent=2) + "\n").encode())
    print(summary, flush=True)


if __name__ == "__main__":
    main()
