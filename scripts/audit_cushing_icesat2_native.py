"""Check ATL03 photon values directly against HDF5 and inventory retained sources."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_cushing_icesat2_native import BEAM_NAMES, STORE
from collect_public import ROOT, write
from sliderule_h5 import h5

FIELDS = {"lat_ph": "latitude", "lon_ph": "longitude", "h_ph": "height", "delta_time": "delta_time",
          "signal_conf_ph": "atl03_cnf", "quality_ph": "quality_ph", "ph_id_channel": "ph_id_channel",
          "ph_id_pulse": "ph_id_pulse", "ph_id_count": "ph_id_count"}


def checksum(path):
    with (ROOT/path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def audit(row):
    filename = row["filename"]
    path = ROOT / row["csv_path"]
    table = pq.ParquetFile(path).read(columns=["gt", "ph_index", *FIELDS.values()])
    out = []
    for i in sorted({0, table.num_rows//2, table.num_rows-1}):
        p = table.slice(i,1).to_pylist()[0]
        prefix = f"/{BEAM_NAMES[p['gt']]}/heights/"
        arrays, receipt = h5(STORE/"audit", "icesat2", filename, [(prefix+k,p["ph_index"],1,0) for k in FIELDS])
        for name, local in FIELDS.items():
            if not np.array_equal(arrays[prefix+name], np.array([p[local]], dtype=arrays[prefix+name].dtype)):
                raise ValueError(f"Photon differs from original HDF5: {filename}/{p['ph_index']}/{name}")
        out.append(dict(filename=filename, beam=BEAM_NAMES[p["gt"]], ph_index=p["ph_index"], parquet_row=i,
                        fields_matched=len(FIELDS), response_file=receipt["response_file"], response_sha256=receipt["response_sha256"]))
    return out


def main():
    manifests = read("cushing_icesat2_native_extracts")
    for r in manifests:
        if checksum(r["packet_path"]) != r["response_sha256"]:
            raise ValueError(f"Wrong hash of {r['packet_path']}")
        request = Path(r["packet_path"]).with_suffix(".json")
        if checksum(request) != r["request_sha256"]:
            raise ValueError(f"Wrong hash of {request}")
        if r["csv_path"] and checksum(r["csv_path"]) != r["parquet_sha256"]:
            raise ValueError(f"Wrong hash of {r['csv_path']}")
    available = sorted((r for r in manifests if r["csv_path"]), key=lambda r: r["filename"])
    selected = [available[i] for i in sorted({0, len(available)//4, len(available)//2, 3*len(available)//4,len(available)-1})]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [r for group in pool.map(audit, selected) for r in group]
    export("cushing_icesat2_native_direct_h5_audit", rows)
    summary = dict(verified_manifest_files=len(manifests), verified_photon_parquets=len(available), direct_hdf5_granules=len(selected),
                   direct_hdf5_photons=len(rows), direct_hdf5_values=sum(r["fields_matched"] for r in rows))
    payload = json.dumps(summary, indent=2)+"\n"
    write(DERIVED/"cushing_icesat2_native_direct_h5_audit_summary.json", payload.encode())
    print(payload, end="", flush=True)


if __name__ == "__main__":
    main()
