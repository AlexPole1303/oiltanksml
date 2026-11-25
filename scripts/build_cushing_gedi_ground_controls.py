"""Check local GEDI height alignment against the native USGS ground surface."""

import json
from collections import Counter
from statistics import median

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from build_cushing_icesat2_controls import TO_UTM, ground_heights, identities, other_structure_context
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from collect_cushing_gedi_olm import GROUND_CONTEXT_M
from collect_public import ROOT, write

STEM = "cushing_gedi_ground_controls"


def main():
    tanks, _, _ = identities()
    dems = read("cushing_icesat2_ground_assets")
    structures, radii = other_structure_context()
    structure_tree = cKDTree(structures)
    sources = {}
    results = []
    for r in read("cushing_gedi_roof_waveforms_shots"):
        t = tanks[r["candidate_id"]]
        path = r["regional_shots_path"]
        if path not in sources:
            table = pq.ParquetFile(ROOT/path).read()
            arrays = {k: table[k].to_numpy() for k in table.column_names}
            xy = np.column_stack(TO_UTM.transform(arrays["lon_lowestmode"],arrays["lat_lowestmode"]))
            sources[path] = arrays, xy, cKDTree(xy)
        values, xy, tree = sources[path]
        point = (t["east_m"],t["north_m"])
        candidates = np.array(tree.query_ball_point(point, GROUND_CONTEXT_M),dtype=int)
        candidates = candidates[(values["beam"][candidates]==int(r["beam"])) &
            (abs(values["delta_time"][candidates]-float(r["delta_time"]))<=2) &
            (values["l2a_quality_flag_rel3"][candidates]==1)]
        for i in structure_tree.query_ball_point(point, GROUND_CONTEXT_M+max(radii)+22.5):
            candidates = candidates[np.linalg.norm(xy[candidates]-structures[i],axis=1)>radii[i]+22.5]
        offsets = values["elev_lowestmode"][candidates]-ground_heights(xy[candidates],dems)-t["geoid18_interpolated_m"]
        offsets = offsets[np.isfinite(offsets) & (abs(offsets)<=2)]
        count = len(offsets)
        median_offset = float(np.median(offsets)) if count>=3 else None
        spread = float(np.diff(np.percentile(offsets,[10,90]))[0]) if count>=3 else None
        raw = float(r["wall_to_lowest_mode_approx_m"])
        corrected = raw+median_offset if median_offset is not None else None
        results.append(dict(candidate_id=r["candidate_id"],shot_number=r["shot_number"],observed_at_utc=r["observed_at_utc"],
            gedi_acquisition_key=r["gedi_acquisition_key"],beam=r["beam"],native_l2a_quality_flag_rel3=r["l2a_quality_flag_rel3"],
            clean_same_beam_ground_shots=count, local_ground_bias_m=median_offset, local_ground_p90_p10_m=spread,
            native_wall_to_mode_m=raw, wall_to_mode_ground_corrected_m=corrected, native_peaks=int(r["detected_waveform_peaks"]),
            within_physical_wall_limits=corrected is not None and 0<=corrected<=t["lidar_top_navd88_m"]-t["lidar_ground_navd88_m"],
            native_shot_subset=path))
    export(STEM,results)
    biases = [r["local_ground_bias_m"] for r in results if r["local_ground_bias_m"] is not None]
    summary = dict(roof_waveform_rows=len(results), rows_with_at_least_three_ground_shots=len(biases),
        rows_within_physical_wall_limits=sum(r["within_physical_wall_limits"] for r in results),
        median_abs_local_ground_bias_m=median(map(abs,biases)) if biases else None,
        ground_shot_counts=Counter(r["clean_same_beam_ground_shots"] for r in results),
        scope="Same-beam, quality-screened shots outside known structures; native USGS bare-earth DEM and GEOID18. "
        "A ground offset requires at least three returns within two seconds and two metres of the datum-aligned DEM.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
