"""Fetch and verify the original Landsat Collection 1 B8 angle coefficients."""
import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from collect_landsat import GCS_OBJECT, LANDSAT, crc32c_base64
from collect_public import ROOT, get, write, write_csv

FIELDS = ("product_id", "capture_at_utc", "pan_assets_available_at_utc", "angle_asset_created_at_utc",
          "geometry_inputs_available_at_utc", "source_url", "source_bytes", "source_crc32c", "local_sha256",
          "local_file", "source_index", "pan_pixel_size_m", "pan_image_lines", "pan_image_samples")


def coefficients(path):
    item = json.loads(path.read_bytes())
    product = item["product_id"]
    index = path.parent/"objects.json"
    records = json.loads(index.read_bytes())["records"]
    source, = (row for row in records if row["name"].endswith(f"/{product}_ANG.txt"))
    local = path.parent/"ANG.txt"
    url = GCS_OBJECT+source["name"]
    payload = local.read_bytes() if local.is_file() else get(url)
    if len(payload)!=int(source["size"]) or crc32c_base64(payload)!=source["crc32c"]:
        raise ValueError(f"Angle file differs from the source checksum for {product}")
    text = payload.decode("ascii")
    mtl = (path.parent/"MTL.txt").read_text(encoding="ascii")
    scene_id = re.compile(r'^\s*LANDSAT_SCENE_ID\s*=\s*"([^"]+)"',re.M)
    if scene_id.findall(text)!=scene_id.findall(mtl) or not scene_id.search(text):
        raise ValueError(f"The ANG file and MTL metadata identify different Landsat scenes for {product}")
    dims = re.findall(r"^\s*BAND08_(NUM_L1T_LINES|NUM_L1T_SAMPS|PIXEL_SIZE)\s*=\s*([\d.]+)\s*$",text,re.M)
    dims = {k:float(v) for k,v in dims}
    if (dims.keys()!={"NUM_L1T_LINES","NUM_L1T_SAMPS","PIXEL_SIZE"} or dims["PIXEL_SIZE"]!=15
            or "BAND08_MEAN_SAT_VECTOR" not in text or "BAND08_MEAN_SUN_VECTOR" not in text):
        raise ValueError(f"Original B8 angle coefficient block is incomplete for {product}")
    if not local.is_file():
        write(local,payload)
    ready = max((item["archive_assets_created_by_utc"],source["timeCreated"]),key=datetime.fromisoformat)
    return (product,item["captured_at_utc"],item["archive_assets_created_by_utc"],source["timeCreated"],ready,url,len(payload),
            source["crc32c"],hashlib.sha256(payload).hexdigest(),str(local.relative_to(ROOT)),str(index.relative_to(ROOT)),
            15,int(dims["NUM_L1T_LINES"]),int(dims["NUM_L1T_SAMPS"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    args = parser.parse_args()
    paths = sorted((LANDSAT/"legacy_pan").glob("*/*/item.json"))
    if not paths:
        raise ValueError("The local Collection 1 archive is empty")
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(coefficients,paths):
            rows.append(row)
            if len(rows)%50==0 or len(rows)==len(paths):
                print(f"Verified {len(rows)}/{len(paths)} original Landsat angle files",flush=True)
    write_csv(ROOT/"data/derived/landsat_pan_angle_assets.csv",FIELDS,rows)


if __name__=="__main__":
    main()
