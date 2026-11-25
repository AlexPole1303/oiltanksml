"""Validate differential height estimates on aerial and near-simultaneous controls."""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from functools import lru_cache
from statistics import mean, median

import numpy as np
import rasterio
from pyproj import Transformer
from shapely import transform
from shapely.geometry import Point

from audit_cushing_weekly_change_controls import aerial
from audit_shadow_registration_pixels import check_source, coverage_rows, requested_patches
from audit_sentinel2_alternate_registration import packet_inputs, source_packet
from build_l1c_quality import QUALITY_FIELDS, scene_rows
from build_s1_reference_matches import DERIVED, export, read
from build_shadow_registration_inputs import anchor_grid, build_windows, tank_points
from build_terminal_development_measurements import BARREL_M3
from build_weekly_shadow_validation import legacy_geometry
from collect_cushing_gedi_optical_windows import kernels, shadow_geometry_inputs
from collect_public import ROOT, write
from collect_s2_l1c import parse_product, source_metadata_path
from extract_s2_radiometry import BAND_ID, read_xml
from fit_shadow_geometry import bearing_vector, gap_trials, pixel_grid, response_profiles, shifted_grid, read_only
from fit_shadow_registration import patch_profiles, tank_offset

STEM = "cushing_joint_shadow_change"
CENTRE_SPREAD_M = 2.5
MODES = ("independent", "shared_profile", "shared_change_median")


def differential(before, after):
    a_shifts,a_gaps,a_loss,a_weights = before
    b_shifts,b_gaps,b_loss,b_weights = after
    if a_shifts != b_shifts or not np.array_equal(a_gaps,b_gaps):
        raise ValueError("Differential grid must use a common radius, shell and centre trials")
    a_best = np.unravel_index(a_loss.argmin(),a_loss.shape)[1]
    b_best = np.unravel_index(b_loss.argmin(),b_loss.shape)[1]
    raw = float(a_gaps[a_best]-b_gaps[b_best])
    # Profile the two date-specific radiometries and give both dates equal total weight.
    a = a_loss/a_weights.sum()
    b = b_loss/b_weights.sum()
    joint = a[:,:,None]+b[:,None,:]
    shift,i,j = np.unravel_index(joint.argmin(),joint.shape)
    profile = float(a_gaps[i]-b_gaps[j])

    # Use residual variance to soften the minimum and a LiDAR-centre prior in ground metres.
    # Retain the shape of the evidence at each date rather than shrinking the change to zero.
    def mass(loss, weights):
        n = weights.sum()**2/(weights@weights)
        relative = loss-loss.min()
        return np.exp(-0.5*relative/max(loss.min()/max(n-6,4),1e-9))

    prior = np.exp(-0.5*np.square(np.asarray(a_shifts)/CENTRE_SPREAD_M).sum(axis=1))
    posterior = np.einsum("ci,cj,c->ij",mass(a,a_weights),mass(b,b_weights),prior)
    index = (np.arange(len(a_gaps))[:,None]-np.arange(len(b_gaps))[None,:]+len(a_gaps)-1).ravel()
    distributions = np.bincount(index,weights=posterior.ravel())
    if not distributions.sum() > 0:
        raise ValueError("No joint height probability on a shared centre")
    support = np.arange(-len(a_gaps)+1,len(a_gaps))*(a_gaps[1]-a_gaps[0])
    quantile = np.searchsorted(np.cumsum(distributions),np.array((.1,.5,.9))*distributions.sum())
    return dict(independent=raw,shared_profile=profile,shared_change_median=float(support[quantile[1]]),
                change_mass_p10_m=float(support[quantile[0]]),change_mass_p90_m=float(support[quantile[2]]),
                shared_centre_east_m=a_shifts[shift][0],shared_centre_north_m=a_shifts[shift][1])


def profile(geometry,kernel):
    grid,radius,shell,sun,view = geometry
    gaps = gap_trials(shell,"EFR")
    shifts, loss, _, _ = response_profiles(grid,radius,shell,gaps,sun,view,kernel,reversed_sun=False)
    return shifts,gaps,loss,grid[-1]


def pixel_robust_profile(geometry,kernel):
    """Re-rank the physical shadow responses using capped pixel influence."""
    grid,radius,shell,sun,view = geometry
    gaps = gap_trials(shell,"EFR")
    shifts,pixel_losses,_,_ = response_profiles(grid,radius,shell,gaps,sun,view,kernel,reversed_sun=False,pixel_losses=True)
    weights = grid[-1]
    sums = pixel_losses.sum(axis=-1)
    i,j = np.unravel_index(sums.argmin(),sums.shape)
    # A single scale per endpoint keeps all centres and heights comparable.
    # Remove spatial weights when learning the typical pixel-residual scale,
    # then apply those same fixed weights to the robust loss at every trial.
    residual_squares = pixel_losses/weights[None,None,:]
    scale = max(2.5*np.median(np.sqrt(residual_squares[i,j]))/.67448975,1e-4)
    root = np.sqrt(1+residual_squares/scale**2)
    loss = (2*pixel_losses/(1+root)).sum(axis=-1)
    return shifts,gaps,sums,loss,weights


def robust_and_standard(earlier,later):
    nominal = differential(*(row[:3]+(row[-1],) for row in (earlier,later)))
    robust = differential(*(row[:2]+row[3:] for row in (earlier,later)))
    return nominal | {f"robust_{m}":robust[m] for m in MODES}


def exterior_source(number, version, directory, capture):
    folder = ROOT/directory
    red = folder/"red.tif"
    mask = source_metadata_path(folder/"MSK_CLOUDS_B00.gml")
    basis = "gml_opaque_cirrus" if mask.is_file() else "jp2_opaque_cirrus_snow_ice"
    if basis.startswith("jp2"):
        mask = folder/"MSK_CLASSI_B00.jp2"
    tile_xml = source_metadata_path(folder/"MTD_TL.xml")
    quant,offset,nodata,saturated,_ = read_xml(source_metadata_path(folder/"MTD_MSIL1C.xml"),"L1C",version)
    (_,_,orbit,tile),_,_ = parse_product(version)
    with rasterio.open(red) as raster:
        west,south,east,north = raster.bounds
        if raster.crs.to_epsg()!=32614 or raster.res!=(10,10):
            raise ValueError(f"Unexpected B04 raster grid for {version}")
    return dict(source_id=number,captured_at_utc=capture,relative_orbit=orbit,mgrs_tile=tile,west=west,south=south,east=east,north=north,
                red_raster=str(red.relative_to(ROOT)),red_nodata_dn=nodata,red_saturated_dn=saturated,red_reflectance_scale=1/quant,
                red_reflectance_offset=offset[BAND_ID["B04"]]/quant,cloud_mask_basis=basis,cloud_mask=str(mask.relative_to(ROOT)),
                tile_metadata=str(tile_xml.relative_to(ROOT)))


def exterior_registration(jobs,sources):
    pairs,windows,patches,_ = build_windows(jobs,anchor_grid(*tank_points()))
    pairs = {r["scene_pair_id"]:r for r in pairs}
    patches = {r["anchor_id"]:r for r in patches}
    checked = {(r["source_id"],r["anchor_id"]):r for source,anchors in requested_patches(windows,pairs,patches).items()
               if anchors for r in check_source(sources[source],[patches[a] for a in sorted(anchors)])}
    coverage,_ = coverage_rows(windows,pairs,checked)
    profiles,_ = patch_profiles(coverage,sources,pairs,patches)
    return {r["candidate_id"]:tank_offset(r,profiles) for r in coverage}


def aerial_controls(*, robust=False):
    legacy = legacy_geometry()
    controls = aerial()
    references = {(r["candidate_id"],r["flight_date"],r["satellite_version"],r["satellite_captured_at_utc"],r["satellite_sensor"]):r
                  for r in read("aerialok_native_optical_pairs") if r["satellite_sensor"].startswith("Sentinel-2 B04")}
    staged = defaultdict(set)
    selected,descriptors,jobs = {},{},[]
    fitter = pixel_robust_profile if robust else profile
    compare = robust_and_standard if robust else differential
    for r in controls:
        endpoints = []
        for side in ("first","last"):
            tank,sensor,version = r["candidate_id"],r[f"{side}_satellite_sensor"],r[f"{side}_satellite_version"]
            ref = references[tank,r[f"{side}_flight"],version,r[f"{side}_satellite_capture_utc"],sensor]
            if sensor == "Sentinel-2 B04":
                path,native,scenes,_ = packet_inputs(source_packet(ref["satellite_source"]))
                o = dict(native[int(ref["satellite_observation_id"])])
                g = legacy[tank]
                o.update(center_east_m=float(g["center_east_m"]),center_north_m=float(g["center_north_m"]),radius_m=float(g["diameter_m"])/2)
                with closing(read_only(path)) as db:
                    grid = pixel_grid(db,o)
                shell = float(g["prior_rim_z_p95_native"])-float(g["prior_ground_z_p50_native"])
                angles = [bearing_vector(o[f"{k}_zenith_deg"],o[f"{k}_azimuth_grid_deg"]) for k in ("sun","red_view")]
                selected[tank,version,sensor] = (grid,o["radius_m"],shell,*angles)
                directory = scenes[o["scene_id"]]["source_directory"]
            else:
                directory = ref["satellite_local_source_directory"] or ref["satellite_source"]
                staged[directory].add(tank)
            key = version,directory
            if key not in descriptors:
                descriptors[key] = exterior_source(len(descriptors)+1,version,directory,ref["satellite_captured_at_utc"])
            endpoints.append(descriptors[key])
        jobs.append(("aerial_interval","engineering",r["last_flight"],r["candidate_id"],"EFR",r["terminal_id"],"","","",*endpoints))
    to_geo = Transformer.from_crs(32614,4326,always_xy=True).transform
    for directory, names in staged.items():
        manifest = ROOT/directory/"item.json"
        outlines = []
        points,tanks = {},{}
        for tank in sorted(names):
            g = legacy[tank]
            east,north,diameter = [float(g[k]) for k in ("center_east_m","center_north_m","diameter_m")]
            lon,lat = to_geo(east,north)
            radius = diameter/2
            outlines.append((tank,transform(Point(east,north).buffer(radius,quad_segs=64),to_geo,interleaved=False)))
            points[tank] = lon,lat
            tanks[tank] = dict(longitude=lon,latitude=lat,radius_m=radius,
                               lidar_top_navd88_m=float(g["prior_rim_z_p95_native"]),lidar_ground_navd88_m=float(g["prior_ground_z_p50_native"]))
        for row in scene_rows(manifest,outlines,points,{},None):
            quality = dict(zip(QUALITY_FIELDS,row,strict=True))
            tank,version = quality["candidate_id"],quality["l1c_product_name"]
            selected[tank,version,"Sentinel-2 B04 direct"] = shadow_geometry_inputs(quality,tanks[tank])
    psf = kernels({v[:3] for _,v,_ in selected})
    offsets = exterior_registration(jobs,{r["source_id"]:r for r in descriptors.values()})
    for r in controls:
        ids = [r[f"{side}_satellite_version"] for side in ("first","last")]
        geometries = [selected[r["candidate_id"],v,r[f"{side}_satellite_sensor"]] for v,side in zip(ids,("first","last"),strict=True)]
        fits = [fitter(g,psf[v[:3]]) for g,v in zip(geometries,ids,strict=True)]
        values = compare(*fits)
        if values["independent"] != r["physical"]:
            raise ValueError(f"Original aerial-pair change differs for {r['candidate_id']}")
        offset = offsets[r["candidate_id"]]
        mode_keys = (*MODES,*(f"robust_{m}" for m in MODES)) if robust else MODES
        admitted = dict.fromkeys(f"registered_{m}" for m in mode_keys)
        if offset["registration_status"]=="checked_offset":
            x,y = (float(offset[f"later_minus_earlier_{k}_m"])/2 for k in ("east","north"))
            after_alignment = [fitter((shifted_grid(g[0],sign*x,sign*y),*g[1:]),psf[v[:3]])
                               for g,v,sign in zip(geometries,ids,(-1,1),strict=True)]
            aligned = compare(*after_alignment)
            admitted = {f"registered_{m}":aligned[m] for m in mode_keys}
        yield dict(candidate_id=r["candidate_id"],terminal_id=r["terminal_id"],earlier_version=ids[0],later_version=ids[1],
                   reference_change_m=r["target"],max_aerial_satellite_separation_hours=r["max_gap_hours"],
                   bbl_per_m=float(r["roof_area_m2"])/BARREL_M3,registration_status=offset["registration_status"],**values,**admitted)


def tandem_controls(*, robust=False):
    path, native, _, psf = packet_inputs("shadow_change_controls.sqlite")
    original = {(int(r["observation_id"])):r for r in read("shadow_change_control_estimates")}
    registry = {r["candidate_id"]:r for r in read("tank_engineering_registry")}
    pairs = [r for r in read("shadow_tandem_pass_pairs") if r["selected_for_summary"]==r["both_resolved"]=="True"
             and r["roof_type"]=="EFR" and abs(float(r["other_minus_a_seconds"]))<=60]
    registered = {(r["candidate_id"],r["earlier_version"],r["later_version"]):r for r in read("shadow_registered_control_fit")
                  if r["comparison"]=="tandem" and r["registration_status"]=="checked_offset"}
    fitter = pixel_robust_profile if robust else profile
    compare = robust_and_standard if robust else differential
    mode_keys = (*MODES,*(f"robust_{m}" for m in MODES)) if robust else MODES

    @lru_cache(maxsize=1024)
    def estimate(number,east=0,north=0):
        r = native[number]
        with closing(read_only(path)) as conn:
            grid = shifted_grid(pixel_grid(conn,r),east,north)
        shell = float(original[number]["shell_above_ground_m"])
        angles = [bearing_vector(r[f"{k}_zenith_deg"],r[f"{k}_azimuth_grid_deg"]) for k in ("sun","red_view")]
        return fitter((grid,r["radius_m"],shell,*angles),psf[r["platform"]])

    def solve(r):
        numbers = [int(r[f"{k}_observation_id"]) for k in ("a","other")]
        values = compare(*(estimate(n) for n in numbers))
        if values["independent"] != -float(r["other_minus_a_gap_m"]):
            raise ValueError(f"Original tandem change differs for {r['candidate_id']}")
        g = registry[r["candidate_id"]]
        offset = registered.get((r["candidate_id"],r["a_version"],r["other_version"]))
        sign = 1
        if offset is None:
            offset = registered.get((r["candidate_id"],r["other_version"],r["a_version"]))
            sign = -1
        admitted = dict.fromkeys(f"registered_{m}" for m in mode_keys)
        if offset is not None:
            x,y = (sign*float(offset[f"later_minus_earlier_{k}_m"])/2 for k in ("east","north"))
            aligned = compare(estimate(numbers[0],-x,-y),estimate(numbers[1],x,y))
            if aligned["shared_profile"] != sign*float(offset["shared_registered_change_m"]):
                raise ValueError(f"Saved registered control differs for {r['candidate_id']}")
            admitted = {f"registered_{k}":aligned[k] for k in mode_keys}
        return dict(candidate_id=r["candidate_id"],terminal_id=g["documented_terminal_id"],capture_day=r["capture_day"],
                    earlier_version=r["a_version"],later_version=r["other_version"],reference_change_m=0.,
                    bbl_per_m=float(g["footprint_bbl_per_m"]),**values,**admitted)

    with ThreadPoolExecutor(max_workers=4) as pool:
        yield from pool.map(solve,pairs)


def metrics(rows,modes=MODES):
    return {m:dict(mae_m=mean(abs(r[m]-r["reference_change_m"]) for r in rows),
                   median_ae_m=median(abs(r[m]-r["reference_change_m"]) for r in rows),
                   mae_cylindrical_bbl=mean(abs(r[m]-r["reference_change_m"])*r["bbl_per_m"] for r in rows),
                   mean_signal_m=mean(abs(r[m]) for r in rows),
                   direction=sum(r[m]*r["reference_change_m"]>0 for r in rows)) for m in modes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robust-pixels",action="store_true",help="Compare a robust pixel-response score on the same independent controls")
    args = parser.parse_args()
    robust = args.robust_pixels
    stem = "cushing_robust_shadow_change" if robust else STEM
    modes = (*MODES,*(f"robust_{m}" for m in MODES)) if robust else MODES
    registered_modes = tuple(f"registered_{m}" for m in modes)
    a = list(aerial_controls(robust=robust))
    b = list(tandem_controls(robust=robust))
    admitted = [r for r in b if r["registered_independent"] is not None]
    aerial_admitted = [r for r in a if r["registered_independent"] is not None]
    days = sorted({r["capture_day"] for r in b})
    report = dict(status="retrospective_physical_validation", aerial=dict(rows=len(a),terminals=len({r["terminal_id"] for r in a}),
                  single_flight_interval=True,metrics=metrics(a,modes),by_terminal={k:metrics([r for r in a if r["terminal_id"]==k],modes)
                  for k in sorted({r["terminal_id"] for r in a})},
                  registered_common_sample=dict(rows=len(aerial_admitted),metrics=metrics(aerial_admitted,(*modes,*registered_modes)) if aerial_admitted else {})),
                  tandem=dict(rows=len(b),metrics=metrics(b,modes),
                  by_acquisition={d:metrics([r for r in b if r["capture_day"]==d],modes) for d in days},
                  registered_common_sample=dict(rows=len(admitted),metrics=metrics(admitted,(*modes,*registered_modes)))))
    export(f"{stem}_aerial",a)
    export(f"{stem}_tandem",b)
    payload = json.dumps(report,indent=2,allow_nan=False)+"\n"
    write(DERIVED/f"{stem}_summary.json",payload.encode())
    print({k:(report[k]["rows"],report[k]["metrics"]) for k in ("aerial","tandem")})


if __name__=="__main__":
    main()
