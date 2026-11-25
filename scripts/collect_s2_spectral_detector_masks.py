"""Archive verified B03 and B08 detector footprints for the measured Sentinel-2 sample."""
import argparse
import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from audit_shadow_registration_pixels import index
from audit_sentinel2_weekly_volume import PHASES
from build_registered_shadow_development import DATED
from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from collect_l1c_catalog_gaps import AWS, s3_object
from collect_public import ROOT, write
from collect_s2_l1c import OBJECTS, OUT, crc32c_base64, download_source_metadata, source_metadata_path

STEM = "sentinel2_spectral_detector_masks"
CONTROL_STEM = "sentinel2_spectral_control_detector_masks"
HOLDOUT_STEM = "sentinel2_spectral_holdout_detector_masks"
BANDS = ("B03", "B08")
VOLUMES = ("sentinel2_weekly_tank_volumes", "sentinel2_alternate_registration_added_volumes")


def recovery_sources(phase, accepted, *, include_alternates=True):
    tables = [(False, f"weekly_shadow_registered_{phase}_pairs", "offset_status")]
    if include_alternates:
        alternate_table = "sentinel2_alternate_registration_holdout_pairs" if phase == "holdout" else "sentinel2_alternate_registration_pairs"
        tables.append((True, alternate_table, "registration_status"))
    for alternate, table, offset in tables:
        for row in read_csv(table):
            if (row.get("study_phase", phase) == phase and row["geometry_mode"] == DATED
                and row[offset] == "checked_offset" and row["registered_both_shadow_diagnostics_pass"] != "True"
                and row["registered_change_m"] and (row["week_ending"], row["candidate_id"]) not in accepted
            ):
                yield alternate, row


def required_products():
    volumes = [r for table in VOLUMES for r in read_csv(table)]
    versions = {row[f"{side}_sentinel2_version"] for row in volumes for side in ("earlier", "later")}
    for phase in PHASES:
        accepted = {(r["week_ending"], r["candidate_id"]) for r in volumes if r["study_phase"] == phase}
        versions.update(row[f"{side}_version"] for _, row in recovery_sources(phase, accepted)
                        for side in ("earlier", "later"))
    return sorted(versions)


def control_offsets():
    for stem in ("control", "fixed_roof_validation", "fixed_roof_holdout"):
        yield from read_csv(f"shadow_registration_{stem}_offsets")


def control_products():
    return registered_products(control_offsets())


def registered_products(offsets):
    sources, pairs = index("sources", "source_id"), index("scene_pairs", "scene_pair_id")
    products = {}
    for row in offsets:
        if row["registration_status"] != "checked_offset":
            continue
        pair = pairs[row["scene_pair_id"]]
        for side in ("earlier", "later"):
            source = sources[pair[f"{side}_source_id"]]
            products.setdefault(source["version_id"], source["product_manifest"])
    return products


def holdout_alternate_products():
    from audit_sentinel2_alternate_registration import packet_inputs

    products = {}
    for row in read_csv("sentinel2_alternate_registration_holdout_pairs"):
        if row["registration_status"] != "checked_offset":
            continue
        for side in ("earlier", "later"):
            _, observations, scenes, _ = packet_inputs(row[f"{side}_native_packet"])
            obs = observations[int(row[f"{side}_native_observation_id"])]
            scene = scenes[obs["scene_id"]]
            version = row[f"{side}_version"]
            if scene["version_id"] != version:
                raise ValueError(f"Alternate detector product differs from its original source {version}")
            products.setdefault(version, scene["manifest_path"])
    return products


def aws_assets(version, manifest_path):
    manifest = json.loads((ROOT / manifest_path).read_bytes())
    if manifest["archive"] != AWS or manifest["product_name"] != version:
        raise ValueError(f"No exact Google Cloud or AWS metadata for {version}")
    tile = next(row for row in manifest["granule_metadata"] if row["local_name"] == "tileInfo.json")
    actual = s3_object(tile["url"], tile["local_name"])
    if any(actual[key] != tile[key] for key in ("size", "etag", "last_modified_at_utc")):
        raise ValueError(f"AWS tile identity changed after the optical archive was verified: {version}")
    info = json.loads((ROOT / manifest_path).with_name(tile["local_name"]).read_bytes())
    if info["productName"] + ".SAFE" != version:
        raise ValueError(f"Archived AWS tile identity differs for {version}")
    prefix = tile["url"].removesuffix(tile["local_name"]) + "qi/"
    return [dict(s3_object(prefix + f"DETFOO_{band}.jp2", f"MSK_DETFOO_{band}.jp2"), band=band) for band in BANDS]


def jobs(products=None):
    products = {version: None for version in required_products()} if products is None else products
    for version, manifest in sorted(products.items()):
        if not (OUT / "object_indexes" / f"{version}.json").is_file() and manifest:
            for asset in aws_assets(version, manifest):
                yield version, asset["band"], asset
            continue
        assets = json.loads((OUT / "object_indexes" / f"{version}.json").read_bytes())
        for band in BANDS:
            matches = [asset for asset in assets if f"/{version}/GRANULE/" in asset["name"]
                       and re.search(rf"/MSK_DETFOO_{band}\.(gml|jp2)$", asset["name"])]
            if len(matches) != 1:
                raise ValueError(f"Expected one original detector footprint for {version}/{band}")
            yield version, band, matches[0]


def collect(job):
    version, band, asset = job
    google = "name" in asset
    url = OBJECTS + quote(asset["name"], safe="/") if google else asset["url"]
    filename = asset["name"].rsplit("/", 1)[-1] if google else asset["local_name"]
    path = OUT / "spectral_detector_footprints" / version / filename
    data = download_source_metadata(url, path, pack_text=True)
    path = source_metadata_path(path)
    digest = crc32c_base64(data) if google else hashlib.md5(data, usedforsecurity=False).hexdigest()
    if len(data) != int(asset["size"]) or digest != asset.get("crc32c", asset.get("etag")):
        raise ValueError(f"Detector footprint does not match archived bytes for {version}/{band}")
    return dict(version_id=version, band=band, mask_format="gml" if path.name.endswith("gml.gz") else path.suffix[1:],
                source_created_at_utc=asset["timeCreated"] if google else asset["last_modified_at_utc"],
                source_bytes=int(asset["size"]), source_crc32c=asset.get("crc32c"), source_etag=asset.get("etag"),
                source_archive="gcs" if google else "aws", source_sha256=hashlib.sha256(data).hexdigest(), source_url=url,
                local_file=str(path.relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--controls", action="store_true")
    scope.add_argument("--holdout", action="store_true")
    parser.add_argument("--with-alternates", action="store_true", help="Include verified holdout alternate registrations")
    args = parser.parse_args()
    if args.with_alternates and not args.holdout:
        parser.error("--with-alternates requires --holdout")
    products, stem = None, STEM
    if args.controls:
        products, stem = control_products(), CONTROL_STEM
    elif args.holdout:
        products = registered_products(read_csv("shadow_registration_holdout_offsets"))
        if args.with_alternates:
            products.update(holdout_alternate_products())
        stem = HOLDOUT_STEM
    sample = list(jobs(products))
    rows = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, row in enumerate(pool.map(collect, sample), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(sample):
                print(f"Verified {i}/{len(sample)} original detector footprint files", flush=True)
    export(stem, rows)
    report = dict(exact_products=len({row["version_id"] for row in rows}), masks=len(rows),
                  formats=dict(Counter(row["mask_format"] for row in rows)),
                  archives=dict(Counter(row["source_archive"] for row in rows)),
                  total_source_bytes=sum(row["source_bytes"] for row in rows))
    payload = json.dumps(report, indent=2) + "\n"
    write(DERIVED / f"{stem}_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
