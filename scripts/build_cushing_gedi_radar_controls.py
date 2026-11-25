"""Match native GEDI roof waveforms to radar scenes and verify existing complex windows."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime

import numpy as np

from build_cushing_gedi_inventory import radar_scenes
from build_cushing_icesat2_controls import identities
from build_s1_reference_matches import export, read
from build_s1_slc_orbit_geometry import look
from build_weekly_shadow_panel import DERIVED
from collect_public import ROOT, write
from collect_s1_slc_orbits import orbit_for

STEM = "cushing_gedi_radar_controls"


def main():
    laser = read("cushing_gedi_roof_waveforms_shots")
    ground = {(r["shot_number"],r["candidate_id"]):r for r in read("cushing_gedi_ground_controls")}
    existing_assets = {r["cdse_uuid"]:r for r in read("ceda_s1_slc_regional_assets")}
    windows = defaultdict(dict)
    for r in read("ceda_s1_slc_regional_windows"):
        windows[r["product_id"],r["candidate_id"]][r["polarization"]] = r
    geometry, _, _ = identities()
    radar = list(radar_scenes())
    matches, local = [], []
    orbit = {}
    for shot in laser:
        t = datetime.fromisoformat(shot["observed_at_utc"])
        candidate = shot["candidate_id"]
        for kind, product, covered, start, stop in radar:
            hours = max(abs((edge-t).total_seconds()) for edge in (start,stop))/3600
            if hours > 24 or candidate not in covered:
                continue
            key = product["product_id"]
            layer = windows[key,candidate]
            alignment = ground[shot["shot_number"],candidate]
            match = dict(candidate_id=candidate, shot_number=shot["shot_number"], gedi_capture_utc=shot["observed_at_utc"],
                gedi_acquisition_key=shot["gedi_acquisition_key"], gedi_peak_count=shot["detected_waveform_peaks"],
                gedi_l2a_quality_flag_rel3=shot["l2a_quality_flag_rel3"], gedi_same_beam_ground_shots=alignment["clean_same_beam_ground_shots"],
                gedi_l2a_lowest_mode_m=shot["elev_lowestmode"], radar_kind=kind, radar_product_id=key,
                radar_product_name=product.get("product_name",""), radar_start_utc=start.isoformat(), radar_end_utc=stop.isoformat(),
                maximum_separation_hours=hours, local_complex_vv_and_vh=set(layer)=={"VV","VH"})
            matches.append(match)
            if not match["local_complex_vv_and_vh"]:
                continue
            if key not in orbit:
                orbit[key] = orbit_for(dict(product,scope="gedi_control"))
            tank = geometry[candidate]
            roof, wall = float(shot["elev_lowestmode"]),tank["top_hae_approx_m"]
            rim = look(tank, wall, orbit[key])
            return_geometry = look(tank, roof, orbit[key])
            if not (rim[3] and return_geometry[3] and return_geometry[1] > rim[1]):
                raise ValueError(f"Invalid Sentinel-1 range prediction for {candidate}/{key}")
            for pol,r in layer.items():
                asset = existing_assets[r["cdse_uuid"]]
                packet = ROOT / asset["local_packet"]
                if hashlib.sha256(packet.read_bytes()).hexdigest() != asset["packet_sha256"]:
                    raise ValueError(f"Complex sample packet hash mismatch for {r['cdse_uuid']}")
                with np.load(packet,allow_pickle=False) as file:
                    values = file["iq_int16"]
                    i,n = int(r["offset_complex_samples"]), int(r["complex_samples"])
                    pixels = values[i:i+n]
                    if (pixels.shape != (n,2) or str(file["cdse_uuid"]) != r["cdse_uuid"]
                            or n != int(r["height"])*int(r["width"]) or not np.any(pixels)):
                        raise ValueError(f"Bad native complex crop for {candidate}/{pol}")
                local.append(dict(match, polarization=pol, cdse_uuid=r["cdse_uuid"], local_packet=asset["local_packet"],
                    local_packet_sha256=asset["packet_sha256"], offset_complex_samples=i, native_complex_samples=n,
                    native_rows=int(r["height"]),native_columns=int(r["width"]), precise_orbit=orbit[key]["local_orbit_gzip"],
                    geometry_wall_to_mode_m=wall-roof, predicted_rim_to_return_slant_m=return_geometry[1]-rim[1]))
    export(STEM,matches)
    export(f"{STEM}_local_windows",local)
    summary = dict(waveform_confirmed_rows=len(laser), catalog_matches_within_24h=len(matches),
        matches_with_local_dual_pol=sum(r["local_complex_vv_and_vh"] for r in matches), verified_native_polarization_windows=len(local),
        scope="GEDI and radar observations can be hours apart. The computed slant shift uses the observed laser mode and wall geometry. "
        "No height has been fitted from the Sentinel-1 windows.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__ == "__main__":
    main()
