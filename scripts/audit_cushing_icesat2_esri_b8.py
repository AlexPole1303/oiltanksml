"""Recover exact-grid Esri B8 observations for the eight ICESat-2 controls."""

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median

from build_cushing_icesat2_b8_inputs import STEM as ORIGINAL
from build_cushing_inspection_b8 import recover
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import RESOLVED
from collect_landsat import LANDSAT
from collect_public import ROOT, write

STEM = "cushing_icesat2_esri_b8"


def main():
    controls = {(r["product_id"], r["candidate_id"]): r for r in read(f"{ORIGINAL}_laser_comparisons")}
    products = {r["product_id"]: r for r in read(f"{ORIGINAL}_products")}
    tanks = defaultdict(set)
    for name, roof in controls:
        tanks[name].add(roof)
    official = {}
    for name, r in products.items():
        catalog = json.loads((ROOT/r["landsatlook_catalog"]).read_bytes())
        official[name], = [x for x in catalog["features"] if x["id"] == name]
        # Reuse the exact ANG and QA objects already recovered for the original DN route.
        for source_field, hash_field, filename in (("local_angle_file", "source_angle_sha512", "ANG.txt"),
                ("qa_pixel_path", "qa_pixel_sha512", "qa_pixel.tif"), ("qa_radsat_path", "qa_radsat_sha512", "qa_radsat.tif")):
            data = (ROOT/r[source_field]).read_bytes()
            if hashlib.sha512(data).hexdigest() != r[hash_field]:
                raise ValueError(f"Existing {filename} differs from the source checksum for {name}")
            target = LANDSAT/STEM/name/filename
            if not target.is_file():
                write(target, data)
    sources, fits, _ = recover(tanks, official, store=LANDSAT/STEM, native=DERIVED/f"{STEM}_native.sqlite",
                               purpose="retrospective_icesat2_esri_reflectance")
    index = {(r["product_id"], r["candidate_id"]): r for r in fits}
    laser_profiles = {(r["filename"], r["candidate_id"]): r for r in read("cushing_icesat2_native_profiles")
                      if r["passes_roof_profile_checks"] == "True" and r["first_roof_signal_photon_utc"]}
    source_by_product = {r["product_id"]: r for r in sources}
    if not index.keys() <= controls.keys():
        raise ValueError("Unexpected roof added to the ICESat-2 control")
    rows = []
    for key, original in sorted(controls.items()):
        fit = index.get(key, {})
        source = source_by_product[key[0]]
        status = fit.get("fit_status", source["status"])
        gap = float(fit["estimated_gap_m"]) if status == RESOLVED else None
        p = laser_profiles[original["icesat_granule"], key[1]]
        laser = float(p["estimated_wall_to_roof_m"])
        first, last = (p[f"{k}_roof_signal_photon_utc"] for k in ("first", "last"))
        if (p["cmr_id"], p["terminal_id"]) != (original["icesat_cmr_id"], original["terminal_id"]):
            raise ValueError(f"ICESat-2 source and native profile disagree for {key}")
        capture = datetime.fromisoformat(original["landsat_capture_at_utc"])
        maximum_hours = max(abs((capture-datetime.fromisoformat(t)).total_seconds()) for t in (first, last))/3600
        rows.append(dict(product_id=key[0], candidate_id=key[1], terminal_id=original["terminal_id"],
                         landsat_capture_at_utc=original["landsat_capture_at_utc"], icesat_granule=original["icesat_granule"],
                         icesat_cmr_id=original["icesat_cmr_id"], icesat_first_roof_photon_utc=first,
                         icesat_last_roof_photon_utc=last, icesat_photon_file=p["photon_csv"],
                         icesat_native_photon_packet=p["photon_packet"], maximum_separation_hours=maximum_hours, fit_status=status,
                         b8_gap_m=gap, icesat_gap_m=laser, b8_minus_icesat_m=gap-laser if gap is not None else None,
                         reflectance_source=source.get("reflectance_path"), reflectance_sha256=source.get("reflectance_sha256"),
                         original_b8_verified=False, synchronized_operator_label=False))
    errors = [abs(r["b8_minus_icesat_m"]) for r in rows if r["b8_minus_icesat_m"] is not None]
    export(f"{STEM}_sources", sources)
    export(f"{STEM}_comparisons", rows)
    report = dict(scenes=len(sources), windows=len(rows), statuses=Counter(r["fit_status"] for r in rows),
                  resolved=len(errors), median_absolute_difference_m=median(errors) if errors else None)
    payload = json.dumps(report, indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
