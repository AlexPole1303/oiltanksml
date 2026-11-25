"""Compare local SLC coherence with optical roof changes and static ground controls."""

import gzip
import hashlib
import io
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime
from itertools import pairwise, product
from statistics import median

import numpy as np
from scipy.constants import c
from scipy.ndimage import map_coordinates
from scipy.stats import rankdata

from build_ceda_s1_slc_windows import UTM_TO_ECEF
from build_cushing_icesat2_controls import identities
from build_s1_reference_matches import export, read
from build_weekly_shadow_panel import DERIVED
from check_ceda_s1_slc_roof_profiles import ECEF_TO_UTM, geometry, project
from collect_public import ROOT, write

STEM = "ceda_s1_slc_coherence"
INPUT = "ceda_s1_slc_regional"
POLS = ("VV", "VH")
MIN_CELLS = 20
BACKGROUND_MARGIN_M = 25
MAX_OPTICAL_HOURS = 48
MATCH_RADIUS_TOLERANCE_M = 5


class Burst:
    def __init__(self, asset, windows):
        self.asset = asset
        self.windows = windows
        self.burst_id, = {r["relative_burst_id"] for r in windows.values()}
        self.native = geometry(asset, self.burst_id)
        annotation = ET.fromstring(gzip.decompress((ROOT / asset["local_annotation"]).read_bytes()))
        info = annotation.find("generalAnnotation/productInformation")
        self.wavelength = c/float(info.findtext("radarFrequency"))
        self.line_count = int(asset["lines_per_burst"])
        self.mid = self.native.start+(self.line_count-1)*self.native.dt/2
        self.steering_hz_s = 2*np.linalg.norm(self.native.velocity(self.mid))*np.deg2rad(
            float(info.findtext("azimuthSteeringRate")))/self.wavelength
        self.fm = self.polynomial(annotation, "generalAnnotation/azimuthFmRateList/azimuthFmRate", "azimuthFmRatePolynomial")
        self.dc = self.polynomial(annotation, "dopplerCentroid/dcEstimateList/dcEstimate", "dataDcPolynomial")
        self.center_time = -self.dc((int(asset["samples_per_burst"])-1)/2)/self.fm((int(asset["samples_per_burst"])-1)/2)
        self.start = datetime.fromisoformat(asset["capture_start_utc"])
        self.available_at = max(datetime.fromisoformat(asset[k]) for k in ("catalog_publication_utc", "source_zip_last_modified_utc"))
        data = (ROOT / asset["local_packet"]).read_bytes()
        if len(data) != int(asset["packet_bytes"]) or hashlib.sha256(data).hexdigest() != asset["packet_sha256"]:
            raise ValueError(f"Changed I/Q packet for {asset['cdse_uuid']}")
        with np.load(io.BytesIO(data), allow_pickle=False) as archive:
            if str(archive["cdse_uuid"]) != asset["cdse_uuid"]:
                raise ValueError("Wrong I/Q burst identity")
            self.iq = archive["iq_int16"]

    def polynomial(self, root, path, field):
        node = min(root.findall(path), key=lambda n: abs(self.native.seconds_from_origin(n.findtext("azimuthTime"))-self.mid))
        coeffs = np.fromstring(node.findtext(field), sep=" ")
        if len(coeffs) != 3 or not np.isfinite(coeffs).all():
            raise ValueError(f"Invalid SLC polynomial {field}")
        t0 = float(node.findtext("t0"))
        return lambda sample: np.polynomial.polynomial.polyval(2*(self.native.first_range+self.native.dr*sample)/c-t0, coeffs)

    def carrier(self, line, sample):
        fm, dc = self.fm(sample), self.dc(sample)
        shift = (line-(self.line_count-1)/2)*self.native.dt + dc/fm+self.center_time
        kt = fm*self.steering_hz_s/(fm-self.steering_hz_s)
        return np.pi*kt*shift**2+2*np.pi*dc*shift

    def chip(self, row):
        a, b = int(row["offset_complex_samples"]), int(row["complex_samples"])
        iq = self.iq[a:a+b].reshape(int(row["height"]), int(row["width"]), 2)
        origin = np.array([int(row["first_line"]), int(row["first_sample"])])
        grid = np.indices(iq.shape[:2])+origin[:, None, None]
        z = iq[:, :, 0].astype(float)+1j*iq[:, :, 1]
        return z*np.exp(-1j*self.carrier(*grid)), origin


def world_points(xy, z):
    return np.stack(UTM_TO_ECEF(xy[..., 0], xy[..., 1], np.broadcast_to(z, xy.shape[:-1])), axis=-1)


def pixel_grid(burst, row, center, z, offsets):
    target = world_points(center, np.asarray(z).flat[0])
    near = burst.native.start+float(row["rim_center_burst_line"])*burst.native.dt
    when, _ = burst.native.locate(target, near)
    return project(burst.native, world_points(center+offsets, z), when).T


def reference_grid(burst, row, tank, structures):
    center = np.array([tank["east_m"], tank["north_m"]])
    ground = tank["ground_hae_approx_m"]
    rim = tank["top_hae_approx_m"]
    radius = tank["radius_m"]
    target = world_points(center, ground)
    near = burst.native.start+float(row["rim_center_burst_line"])*burst.native.dt
    when, sample = burst.native.locate(target, near)
    native_center = np.array([(when-burst.native.start)/burst.native.dt, sample])
    towards = np.array(ECEF_TO_UTM(*(target+.0001*(burst.native.position(when)-target))))[:2]-center
    towards /= np.linalg.norm(towards)
    rim_and_ground = pixel_grid(burst, row, center, np.array([rim, ground]), np.zeros((2, 2)))
    jacobian = (pixel_grid(burst, row, center, ground, np.eye(2)*10)-native_center).T/10
    inverse = np.linalg.inv(jacobian)
    lo, hi = (np.array([int(row[a]), int(row[b])]) for a, b in (
        ("first_line", "first_sample"), ("end_line_exclusive", "end_sample_exclusive")))
    pixels = np.stack(np.meshgrid(*(np.arange(a+3, b-3) for a,b in zip(lo,hi,strict=True)), indexing="ij"), axis=-1).reshape(-1, 2)
    xy = (pixels-native_center)@inverse.T
    xy += (pixels-pixel_grid(burst, row, center, ground, xy))@inverse.T
    error = np.max(np.abs(pixels-pixel_grid(burst, row, center, ground, xy)))
    if error > .001:
        raise ValueError(f"Ground mapping mismatch {error:.5f} pixels")
    rim_offset = (rim_and_ground[0]-rim_and_ground[1])@inverse.T
    dist = np.linalg.norm(xy, axis=1)
    interior = (dist<=radius*.82) & (np.linalg.norm(xy-rim_offset,axis=1)<=radius*.82)
    background = (dist>=radius+BACKGROUND_MARGIN_M) & (dist<=radius+85)
    for x, y, other_radius in structures:
        offset = np.array([x, y])-center
        if np.linalg.norm(offset)>radius+other_radius+175:
            continue
        relative = xy-offset
        along = np.clip(relative@towards, 0., 40.)
        background &= np.linalg.norm(relative-along[:, None]*towards,axis=1)>other_radius+BACKGROUND_MARGIN_M
    return center, ground, xy, pixels, interior, background, error


def coherence(z0, z1, mask):
    if mask.sum() < MIN_CELLS:
        return None
    a,b = z0[mask],z1[mask]
    power = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(abs(np.vdot(b,a))/power,0,1)) if power else None


def pair_features(first, second, tank, structures):
    key = tank["candidate_id"]
    row0, row1 = first.windows[key], second.windows[key]
    center, ground, xy, pixels, interior, background, error = reference_grid(first, row0, tank, structures)
    native1 = pixel_grid(second, row1, center, ground, xy)
    chip0, origin0 = first.chip(row0)
    chip1, origin1 = second.chip(row1)
    if not np.isclose(first.wavelength, second.wavelength, rtol=1e-9, atol=0):
        raise ValueError("Radar frequencies differ within a nominally repeated geometry")
    coords0 = (pixels-origin0).T
    coords1 = native1-origin1
    # Flat local ground phase in cycles. R2-R1 comes from each native slant-range grid.
    r0 = first.native.first_range+pixels[:,1]*first.native.dr
    r1 = second.native.first_range+native1[:,1]*second.native.dr
    correction = np.exp(1j*(second.carrier(*native1.T)+4*np.pi*(r1-r0)/first.wavelength))
    z0 = chip0[*coords0]*np.exp(1j*first.carrier(*pixels.T))
    z1 = map_coordinates(chip1, coords1.T, order=3, mode="constant", cval=complex(np.nan,np.nan))*correction
    null = map_coordinates(chip1, (coords1+np.array([0,6])).T, order=3,
                           mode="constant", cval=complex(np.nan,np.nan))*correction
    valid = np.isfinite(z1) & np.isfinite(null)
    for mask in (background, interior):
        mask &= valid
    return dict(candidate_id=key, terminal_id=tank["terminal_id"], previous_cdse_uuid=first.asset["cdse_uuid"],
                current_cdse_uuid=second.asset["cdse_uuid"], polarization=first.asset["polarization"],
                previous_acquisition_utc=first.start.isoformat(), current_acquisition_utc=second.start.isoformat(),
                capture_separation_days=(second.start-first.start).total_seconds()/86400,
                source_available_at_utc=max(first.available_at,second.available_at).isoformat(),
                previous_iq_sha256=first.asset["packet_sha256"], current_iq_sha256=second.asset["packet_sha256"],
                radius_m=tank["radius_m"], ground_roundtrip_max_pixel=error,
                roof_zone_cells=int(interior.sum()), ground_control_cells=int(background.sum()),
                roof_zone_coherence=coherence(z0,z1,interior), ground_coherence=coherence(z0,z1,background),
                offset_ground_coherence=coherence(z0,null,background))


def optical_index():
    refs = {}
    for r in read("ceda_s1_concurrent_optical_48h_controls"):
        for p in POLS:
            key = r[f"radar_{p.lower()}_burst_id"],r["candidate_id"]
            if key in refs:
                raise ValueError(f"Duplicate nearest optical control for {key}")
            refs[key] = (float(r["maximum_separation_hours"]),
                float(r["optical_gap_to_rim_m"]), float(r["low_over_all_optical_fits_m"]), float(r["high_over_all_optical_fits_m"]),
                r["optical_capture_start_utc"], r["optical_versions_json"])
    return refs


def optical_comparison(row, optical):
    first, second = (optical.get((row[f"{side}_cdse_uuid"],row["candidate_id"])) for side in ("previous","current"))
    if not first or not second or max(first[0],second[0])>MAX_OPTICAL_HOURS:
        return None
    if datetime.fromisoformat(first[4])>=datetime.fromisoformat(second[4]):
        raise ValueError("Optical controls reversed at distinct radar endpoints")
    rise = first[1]-second[1]
    lo,hi = first[2]-second[3],first[3]-second[2]
    label = "large_with_nonoverlapping_profiles" if abs(rise)>=3 and (lo>0 or hi<0) else (
        "central_change_at_most_1m" if abs(rise)<=1 else "intermediate")
    return dict(row, maximum_optical_separation_h=max(first[0],second[0]), optical_rise_m=rise,
                optical_rise_profile_low_m=lo, optical_rise_profile_high_m=hi, optical_change_class=label,
                previous_optical_capture_utc=first[4], current_optical_capture_utc=second[4],
                previous_optical_versions=first[5], current_optical_versions=second[5])


def change_auc(rows):
    use = [r for r in rows if r["optical_change_class"]!="intermediate"]
    flags = np.array([r["optical_change_class"]=="large_with_nonoverlapping_profiles" for r in use],dtype=bool)
    positive,negative = int(flags.sum()), int((~flags).sum())
    features = {"ground_minus_roof": [r["ground_coherence"]-r["roof_zone_coherence"] for r in use],
                "negative_roof": [-r["roof_zone_coherence"] for r in use],
                "ground_only": [r["ground_coherence"] for r in use],
                "radius_only": [r["radius_m"] for r in use]}
    result = {}
    for key, values in features.items():
        scores = rankdata(values)
        result[key] = (scores[flags].sum()-positive*(positive+1)/2)/(positive*negative) if positive and negative else None
    return dict(large_change=positive, small_change=negative, auc=result)


def size_matched_scores(groups):
    matched = []
    for rows in groups:
        large = [r for r in rows if r["optical_change_class"]=="large_with_nonoverlapping_profiles"]
        small = [r for r in rows if r["optical_change_class"]=="central_change_at_most_1m"]
        for a,b in product(large,small):
            if abs(a["radius_m"]-b["radius_m"])<=MATCH_RADIUS_TOLERANCE_M:
                matched.append((a,b))
    scores = [np.sign(b["roof_zone_coherence"]-a["roof_zone_coherence"]) for a,b in matched]
    return dict(comparisons=len(scores), distinct_large_change_observations=len({(a["candidate_id"],a["current_cdse_uuid"]) for a,_ in matched}),
                distinct_small_change_observations=len({(b["candidate_id"],b["current_cdse_uuid"]) for _,b in matched}),
                maximum_radius_difference_m=MATCH_RADIUS_TOLERANCE_M,
                lower_coherence_for_large_change_fraction=float((1+np.mean(scores))/2) if scores else None)


def control_summary(rows):
    result = {}
    for pol in POLS:
        subset = [r for r in rows if r["polarization"]==pol and r["roof_zone_coherence"] is not None
                  and r["ground_coherence"] is not None and r["offset_ground_coherence"] is not None]
        by_pair = defaultdict(list)
        for r in subset:
            by_pair[r["previous_acquisition_utc"][:10],r["current_acquisition_utc"][:10]].append(r)
        result[pol] = dict(rows=len(subset), by_class=Counter(r["optical_change_class"] for r in subset),
                           **change_auc(subset), by_pair={" / ".join(k):change_auc(v) for k,v in sorted(by_pair.items())},
                           within_date_and_similar_radius=size_matched_scores(by_pair.values()))
    return result


def main():
    tanks, _, _ = identities()
    tanks = {k:dict(r,candidate_id=k) for k,r in tanks.items()}
    structures = [(float(r["footprint_east_m"]),float(r["footprint_north_m"]),float(r["footprint_diameter_m"])/2)
                  for r in read("tank_engineering_registry")]
    windows = defaultdict(dict)
    for row in read(f"{INPUT}_windows"):
        windows[row["cdse_uuid"]][row["candidate_id"]] = row
    assets = read(f"{INPUT}_assets")
    selected = {r["product_id"] for r in assets}
    catalog = {r["product_id"]:r for r in read("s1_cushing_catalog") if r["product_id"] in selected}
    geometry_ids = {(r["satellite"],r["relative_orbit"],r["direction"]) for r in catalog.values()}
    if catalog.keys()!=selected or len(geometry_ids)!=1:
        raise ValueError("SLC input spans inconsistent satellite, relative orbit or direction")
    (satellite, relative_orbit, direction), = geometry_ids
    optical = optical_index()
    rows = []
    for pol in POLS:
        ordered = sorted((r for r in assets if r["polarization"]==pol),key=lambda r:r["capture_start_utc"])
        burst = None
        for previous,current in pairwise(ordered):
            first = burst or Burst(previous,windows[previous["cdse_uuid"]])
            burst = Burst(current,windows[current["cdse_uuid"]])
            if first.burst_id!=burst.burst_id or first.asset["swath"]!=current["swath"]:
                raise ValueError("Cannot compare different repeat-orbit burst identities")
            for key in sorted(first.windows.keys()&burst.windows.keys()):
                rows.append(pair_features(first,burst,tanks[key],structures))
    index = {(r["previous_cdse_uuid"],r["current_cdse_uuid"],r["candidate_id"]):r for r in rows}
    if len(index)!=len(rows):
        raise ValueError("Repeated radar coherence row")
    comparisons = [value for r in rows if (value:=optical_comparison(r,optical)) is not None]
    weekly = []
    for r in read(f"{INPUT}_weekly_pairs"):
        for pol in POLS:
            key = r[f"previous_{pol.lower()}"],r[f"current_{pol.lower()}"],r["candidate_id"]
            features = index[key]
            if datetime.fromisoformat(features["source_available_at_utc"])>datetime.fromisoformat(r["forecast_cutoff_utc"]):
                raise ValueError("Source timestamps disagree with pre-EIA weekly inventory")
            weekly.append(dict(features, week_ending=r["week_ending"], b04_status=r["b04_status"],
                               forecast_cutoff_utc=r["forecast_cutoff_utc"]))
    export(STEM,rows)
    export(f"{STEM}_optical_controls",comparisons)
    export(f"{STEM}_weekly",weekly)
    summary = dict(burst_scenes=len(selected), satellite=satellite, relative_orbit=int(relative_orbit), direction=direction,
                   rows=len(rows), weekly_rows=len(weekly),
                   optical_rows=len(comparisons), optical_controls=control_summary(comparisons),
                   complete_radar_rows=sum(r["roof_zone_coherence"] is not None and r["ground_coherence"] is not None for r in rows),
                   complete_weekly_rows=sum(r["roof_zone_coherence"] is not None and r["ground_coherence"] is not None for r in weekly),
                   complete_weekly_rows_without_clear_b04=sum(r["b04_status"]=="no_clear_pair_with_dated_geometry"
                       and r["roof_zone_coherence"] is not None and r["ground_coherence"] is not None for r in weekly),
                   median_coherence={pol:{key:median([r[key] for r in rows if r["polarization"]==pol and r[key] is not None])
                                           for key in ("roof_zone_coherence","ground_coherence","offset_ground_coherence")}
                                     for pol in POLS},
                   scope="Exploratory local coherence on orbit-registered windows, with TOPS carrier removal and restoration "
                         "and a flat local ground model. No empirical subpixel registration or interferometric displacement. "
                         "Optical controls are asynchronous. Positive and negative classes are defined from optical profiles.")
    payload = json.dumps(summary,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="",flush=True)


if __name__=="__main__":
    main()
