"""Profile the geometric B04 shadow fit without using LiDAR roof heights in fitting."""
import csv
import io
import json
import math
import sqlite3
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing

import numpy as np

from collect_public import ROOT, write

DERIVED = ROOT / "data" / "derived"
CALIBRATION = DERIVED / "shadow_calibration.sqlite"
FIT_FIELDS = (
    "observation_id", "candidate_id", "cohort_id", "roof_type", "version_id", "optical_capture_at_utc", "diameter_m",
    "sun_elevation_deg", "view_zenith_deg", "shell_above_ground_m", "model_maximum_gap_m", "fitted_pixels",
    "fit_status", "estimated_gap_m", "near_optimal_gap_min_m", "near_optimal_gap_max_m", "shift_east_m", "shift_north_m",
    "rim_shadow_reflectance_difference", "pixel_rmse", "flat_rmse", "reversed_sun_rmse", "box_response_gap_m",
)
FIELDS = FIT_FIELDS + (
    "lidar_gap_m", "lidar_flight_lines", "lidar_line_gap_spread_m", "minimum_sensor_gap_hours", "maximum_sensor_gap_hours",
    "signed_error_m", "absolute_error_m", "equivalent_absolute_barrels", "same_day_aerial_top_observation",
    "minimum_aerial_gap_minutes", "maximum_aerial_gap_minutes", "selected_for_pass_metrics",
)
STEP_M = 2
PIXEL_M = 10
SEARCH_STEP_M = 0.5
MAX_GAP_M = 24
SHIFTS_M = (-4, 0, 4)
SHADOW_RMSE_GAIN_FRACTION = 0.05
MAX_NEAR_OPTIMAL_SPAN_M = 4.0


def bearing_vector(zenith, azimuth):
    angle = math.radians(azimuth)
    return math.tan(math.radians(zenith)) * np.array([math.sin(angle), math.cos(angle)])


def read_only(path):
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def inputs(db):
    observations = db.execute("""SELECT o.*,s.version_id,s.platform,s.captured_at_utc FROM observation o
        JOIN scene s USING(scene_id) WHERE s.level='L1C' ORDER BY o.observation_id""").fetchall()
    kernels = {}
    for platform, in db.execute("SELECT DISTINCT platform FROM scene WHERE level='L1C'"):
        values = [value for value, in db.execute("""SELECT coefficient FROM psf_coefficient WHERE platform=? AND band='B04'
            ORDER BY psf_row,psf_col""", (platform,))]
        kernel = np.asarray(values,dtype="float32").reshape(33,33)
        # Average the focal-plane axes because the actual L1C grid response is unavailable.
        kernels[platform] = (kernel + kernel.T) / (2 * kernel.sum())
    return observations, kernels


def pixel_grid(db, observation):
    rows = db.execute("""SELECT tile_row,tile_col,center_east_m,center_north_m,red_reflectance,
        opaque,cirrus,snow_ice,red_detector_bits FROM pixel_red_reflectance WHERE observation_id=?
        ORDER BY tile_row,tile_col""", (observation["observation_id"],)).fetchall()
    first_row, first_col = rows[0][:2]
    last_row,last_col = rows[-1][:2]
    pixel_rows = last_row - first_row + 1
    pixel_cols = last_col - first_col + 1
    if len(rows) != pixel_rows * pixel_cols:
        raise ValueError(f"Nonrectangular native chip for observation {observation['observation_id']}")
    matrix = np.asarray([tuple(r) for r in rows], dtype=float)
    expected_row, expected_col = np.indices((pixel_rows, pixel_cols))
    if (not np.array_equal(matrix[:,0], first_row + expected_row.ravel())
        or not np.array_equal(matrix[:,1], first_col + expected_col.ravel())):
        raise ValueError("Native chip coordinates are not contiguous")
    relative_east = matrix[:,2] - observation["center_east_m"]
    relative_north = matrix[:,3] - observation["center_north_m"]
    radius = observation["radius_m"]
    distance = np.hypot(relative_east,relative_north)
    detector_bit = 1 << (observation["red_detector_id"] - 1)
    usable = (np.isfinite(matrix[:,4:7]).all(axis=1) & (matrix[:,5:7] == 0).all(axis=1)
              & (np.nan_to_num(matrix[:,7],nan=0) == 0)
              & (matrix[:,8] == detector_bit) & (distance <= radius+16))
    background = np.exp(-0.5*(np.maximum(distance[usable]-radius,0)/7)**2)
    xy = (float(relative_east[0]), float(relative_north[0]))
    return pixel_rows,pixel_cols,xy,expected_row.ravel()[usable],expected_col.ravel()[usable],matrix[usable,4],background


def ideal_fields(east, north, radius, shell, gaps, sun, view):
    rim = east*east+north*north <= radius**2
    sun_east,sun_north = sun
    view_east,view_north = view
    ground_east,ground_north = east-shell*view_east,north-shell*view_north
    ground_u = np.clip(-(ground_east*sun_east+ground_north*sun_north)/(sun @ sun), 0, shell)
    exterior = ((ground_east+ground_u*sun_east)**2 + (ground_north+ground_u*sun_north)**2 <= radius**2) & ~rim
    heights = gaps[:,None,None]
    roof_east,roof_north = east-heights*view_east,north-heights*view_north
    roof = rim & (roof_east**2 + roof_north**2 <= radius**2)
    wall = rim & ~roof
    shadow_east,shadow_north = heights*sun_east,heights*sun_north
    shadow = roof & ((roof_east+shadow_east)**2 + (roof_north+shadow_north)**2 > radius**2)
    reversed_shadow = roof & ((roof_east-shadow_east)**2 + (roof_north-shadow_north)**2 > radius**2)
    return np.concatenate(([rim, exterior], shadow | wall, reversed_shadow | wall))


def responses(row_count, col_count, origin, radius, shell, gaps, sun, view, kernel, *,
              pixel_m=PIXEL_M, step_m=STEP_M, shifts=SHIFTS_M, subcell_samples=1):
    ratio = round(pixel_m/step_m)
    pad = kernel.shape[0] // 2 + 2 * max(round(abs(s)/step_m) for s in shifts)
    east = origin[0] + (np.arange(col_count*ratio+2*pad)-pad)*step_m
    north = origin[1] - (np.arange(row_count*ratio+2*pad)-pad)*step_m
    east, north = east[None,:], north[:,None]
    offsets = ((np.arange(subcell_samples)+.5)/subcell_samples-.5)*step_m
    if subcell_samples == 1:
        fields = ideal_fields(east, north, radius, shell, gaps, sun, view).astype("float32")
    else:
        fields = np.zeros((2+2*len(gaps), north.size, east.size), dtype="float32")
        for dx in offsets:
            for dy in offsets:
                fields += ideal_fields(east+dx, north+dy, radius, shell, gaps, sun, view)
        fields /= subcell_samples**2
    height, width = fields.shape[-2:]
    packed = np.zeros((height,width),dtype="float32")
    packed[:kernel.shape[0],:kernel.shape[1]] = kernel
    packed = np.roll(packed,(-(kernel.shape[0]//2),-(kernel.shape[1]//2)),axis=(0,1))
    blurred = np.fft.irfft2(np.fft.rfft2(fields,axes=(-2,-1))*np.fft.rfft2(packed),s=(height,width),axes=(-2,-1))
    return blurred, pad, ratio


def profile_fit(model, background, reflectance, *, pixel_losses=False):
    """Profile intercept, background slope and material contrast at each trial gap."""
    x = model.astype(float) * np.sqrt(background)[None,:,None]
    y = reflectance * np.sqrt(background)
    gram = x.transpose(0,2,1) @ x
    rhs = (x.transpose(0,2,1) @ y[:,None])
    gram[:,np.arange(gram.shape[1]),np.arange(gram.shape[1])] += 1e-9
    coefficients = np.linalg.solve(gram,rhs).squeeze(-1)
    invalid = (coefficients[:,-2:] > 0).any(axis=1)
    if invalid.any():
        trial_gram,trial_rhs,trial_x = gram[invalid],rhs[invalid],x[invalid]
        restricted = []
        for active_columns in ((0,1,2,3,4),(0,1,2,3,5),(0,1,2,3)):
            partial = np.linalg.solve(trial_gram[:,active_columns][:,:,active_columns],trial_rhs[:,active_columns]).squeeze(-1)
            candidate = np.zeros((len(partial),gram.shape[1]))
            candidate[:,active_columns] = partial
            error = np.square((trial_x @ candidate[:,:,None]).squeeze(-1) - y).sum(axis=1)
            error[(candidate[:,-2:] > 0).any(axis=1)] = np.inf
            restricted.append((candidate,error))
        option = np.argmin(np.stack([loss for _,loss in restricted]),axis=0)
        coefficients[invalid] = np.stack([c for c,_ in restricted])[option,np.arange(len(option))]
    losses = np.square((x @ coefficients[:,:,None]).squeeze(-1)-y)
    return (losses if pixel_losses else losses.sum(axis=1)), coefficients[:,-2]


def response_profiles(grid, radius, shell, gaps, sun, view, kernel, *, reversed_sun,
                      pixel_m=PIXEL_M, step_m=STEP_M, shifts=SHIFTS_M, subcell_samples=1, centre_offsets=None,
                      include_exterior_shadow=True, pixel_losses=False):
    rows,cols,origin,sample_rows,sample_cols,signal,weights = grid
    if centre_offsets is not None:
        extent = max(abs(d) for shift in centre_offsets for d in shift)
        shifts = (max(0,math.ceil(extent/step_m)*step_m),)
    rendered,pad,ratio = responses(rows,cols,origin,radius,shell,gaps,sun,view,kernel,pixel_m=pixel_m,step_m=step_m,
                                   shifts=shifts,subcell_samples=subcell_samples)
    shift_pairs = ([(0,0),*((e,n) for e in shifts for n in shifts if e or n)] if centre_offsets is None else centre_offsets)
    relative_x = (origin[0] + sample_cols*pixel_m)/radius
    relative_y = (origin[1] - sample_rows*pixel_m)/radius
    background_model = np.stack((np.ones(len(signal)),relative_x,relative_y),axis=1)
    main_profile, main_contrast, placebo_profile = [],[],[]
    for east,north in shift_pairs:
        dy,dx = north/step_m,-east/step_m
        if centre_offsets is None or (dy.is_integer() and dx.is_integer()):
            sample_y = pad+ratio*sample_rows+round(dy)
            sample_x = pad+ratio*sample_cols+round(dx)
            sampled = rendered[:,sample_y,sample_x]
        else:
            y0,x0 = math.floor(dy),math.floor(dx)
            wy,wx = dy-y0,dx-x0
            y,x = pad+ratio*sample_rows+y0,pad+ratio*sample_cols+x0
            sampled = ((1-wy)*((1-wx)*rendered[:,y,x]+wx*rendered[:,y,x+1])
                       + wy*((1-wx)*rendered[:,y+1,x]+wx*rendered[:,y+1,x+1]))
        rim,external,*shadow = sampled
        shared = np.column_stack((background_model,rim))
        model = np.empty((len(gaps),len(signal),6))
        model[:,:,:4],model[:,:,5] = shared,external if include_exterior_shadow else 0
        model[:,:,4] = shadow[:len(gaps)]
        errors,contrast = profile_fit(model,weights,signal,pixel_losses=pixel_losses)
        main_profile.append(errors)
        main_contrast.append(contrast)
        if reversed_sun:
            model[:,:,4] = shadow[len(gaps):]
            placebo_profile.append(profile_fit(model,weights,signal,pixel_losses=pixel_losses)[0])
    return shift_pairs,np.asarray(main_profile),np.asarray(main_contrast),np.asarray(placebo_profile)


def fit_response(grid, radius, shell, gaps, sun, view, kernel, *, reversed_sun,
                 pixel_m=PIXEL_M, step_m=STEP_M, shifts=SHIFTS_M, centre_offsets=None, subcell_samples=1):
    weights = grid[-1]
    shift_pairs,main_profile,main_contrast,placebo_profile = response_profiles(
        grid,radius,shell,gaps,sun,view,kernel,reversed_sun=reversed_sun,pixel_m=pixel_m,step_m=step_m,
        shifts=shifts,centre_offsets=centre_offsets,subcell_samples=subcell_samples)
    shift_index,gap_index = np.unravel_index(np.argmin(main_profile),main_profile.shape)
    error = main_profile[shift_index,gap_index]
    # This is a descriptive loss range. It is not a confidence interval.
    near = gaps[main_profile.min(axis=0) <= error*(1+1/max(1,len(weights)-6))+1e-10]
    return (gaps[gap_index],float(near.min()),float(near.max()),*shift_pairs[shift_index],
            float(main_contrast[shift_index][gap_index]),math.sqrt(error/weights.sum()),
            math.sqrt(main_profile[:,0].min()/weights.sum()),
            math.sqrt(placebo_profile.min()/weights.sum()) if reversed_sun else None)


def estimate(observation, shell, kernel, *, calibration_db=CALIBRATION):
    usable = (observation["roof_clear_by_source_mask"] and observation["resolved_b04_detector"]
              and observation["roof_cells_expected"] == observation["roof_cells_valid_all_bands"])
    result = dict(observation_id=observation["observation_id"], candidate_id=observation["candidate_id"],
                  cohort_id=observation["cohort_id"], roof_type=observation["roof_type"], version_id=observation["version_id"],
                  optical_capture_at_utc=observation["captured_at_utc"], diameter_m=2*observation["radius_m"],
                  sun_elevation_deg=90-observation["sun_zenith_deg"], view_zenith_deg=observation["red_view_zenith_deg"],
                  shell_above_ground_m=shell,fit_status="incomplete_optical_geometry_or_quality")
    if not usable:
        return result
    with closing(read_only(calibration_db)) as db:
        grid = pixel_grid(db,observation)
    pixels = len(grid[-2])
    result.update(fitted_pixels=pixels,fit_status="insufficient_valid_pixels")
    if pixels < 12:
        return result
    sun = bearing_vector(observation["sun_zenith_deg"],observation["sun_azimuth_grid_deg"])
    view = bearing_vector(observation["red_view_zenith_deg"],observation["red_view_azimuth_grid_deg"])
    return dict(result, **geometry_fit(grid,observation["radius_m"],shell,observation["roof_type"],sun,view,kernel))


def gap_trials(shell, roof_type):
    upper = min(MAX_GAP_M,shell) if shell is not None and shell>=5 and roof_type in (None,"EFR") else MAX_GAP_M
    return np.arange(0,upper+1e-8,SEARCH_STEP_M)


def gap_fit_status(gaps, gap, lower, upper, rmse, flat, placebo):
    if gap in (gaps[0],gaps[-1]):
        return "search_boundary"
    if flat == 0 or rmse > flat*(1-SHADOW_RMSE_GAIN_FRACTION):
        return "weak_improvement_over_flat_roof"
    if placebo <= rmse:
        return "reversed_sun_fits_as_well"
    if upper-lower > MAX_NEAR_OPTIMAL_SPAN_M:
        return "broad_profile"
    return "resolved_in_geometry_diagnostic"


def shifted_grid(grid, east, north):
    rows, cols, origin, *pixels = grid
    return rows, cols, (origin[0]-east, origin[1]-north), *pixels


def shared_centre_minimum(earlier_loss, later_loss):
    before = np.argmin(earlier_loss, axis=1)
    after = np.argmin(later_loss, axis=1)
    indices = np.arange(len(before))
    score = earlier_loss[indices, before] + later_loss[indices, after]
    optimum = int(np.argmin(score))
    return optimum, int(before[optimum]), int(after[optimum]), float(score[optimum])


def geometry_fit(grid, radius, shell, roof_type, sun, view, kernel, *,
                 pixel_m=PIXEL_M, step_m=STEP_M, shifts=SHIFTS_M):
    gaps = gap_trials(shell,roof_type)
    model_shell = shell if shell is not None and shell>=5 else 0
    grid_parameters = dict(pixel_m=pixel_m,step_m=step_m,shifts=shifts)
    fit = fit_response(grid,radius,model_shell,gaps,sun,view,kernel,reversed_sun=True,**grid_parameters)
    box = np.zeros_like(kernel)
    middle = box.shape[0] // 2
    box_side = round(pixel_m/step_m)
    half = box_side//2
    box[middle-half:middle+half+1,middle-half:middle+half+1] = 1/box_side**2
    box_fit = fit_response(grid,radius,model_shell,gaps,sun,view,box,reversed_sun=False,**grid_parameters)
    gap,lower,upper_fit,east,north,contrast,rmse,flat,placebo = fit
    status = gap_fit_status(gaps,gap,lower,upper_fit,rmse,flat,placebo)
    return dict(model_maximum_gap_m=float(gaps[-1]),fit_status=status,estimated_gap_m=float(gap),
                  near_optimal_gap_min_m=lower,near_optimal_gap_max_m=upper_fit,shift_east_m=east,shift_north_m=north,
                  rim_shadow_reflectance_difference=-contrast,pixel_rmse=rmse,flat_rmse=flat,reversed_sun_rmse=placebo,
                  box_response_gap_m=float(box_fit[0]))


def add_reference(db, estimates):
    reference = defaultdict(list)
    for row in db.execute("""SELECT observation_id,rim_minus_center_plane_native_z,
        abs(optical_minus_lidar_start_hours),abs(optical_minus_lidar_end_hours) FROM observation_lidar
        WHERE level='L1C' AND rim_minus_center_plane_native_z IS NOT NULL"""):
        reference[row[0]].append(tuple(row[1:]))
    aerial = {(candidate,version):(top,lower/60,upper/60) for candidate,version,top,lower,upper in db.execute("""
        SELECT candidate_id,version_id,aerial_top_observation,min(minimum_separation_seconds),max(maximum_separation_seconds)
        FROM same_day_aerial_pair WHERE level='L1C' GROUP BY candidate_id,version_id,aerial_top_observation""")}
    for estimate in estimates:
        aerial_pair = aerial.get((estimate["candidate_id"],estimate["version_id"]))
        if aerial_pair:
            top,lower,upper = aerial_pair
            estimate.update(same_day_aerial_top_observation=top,minimum_aerial_gap_minutes=lower,maximum_aerial_gap_minutes=upper)
        values = np.asarray(reference[estimate["observation_id"]])
        if not values.size:
            continue
        gap = float(np.median(values[:,0]))
        estimate.update(lidar_gap_m=gap,lidar_flight_lines=len(values),lidar_line_gap_spread_m=float(np.ptp(values[:,0])),
                        minimum_sensor_gap_hours=float(values[:,1:].min()),maximum_sensor_gap_hours=float(values[:,1:].max()))
        if estimate.get("estimated_gap_m") is not None:
            error = estimate["estimated_gap_m"] - gap
            radius = estimate["diameter_m"]/2
            estimate.update(signed_error_m=error,absolute_error_m=abs(error),
                            equivalent_absolute_barrels=abs(error)*math.pi*radius*radius/0.158987294928)


def choose_primary(rows):
    groups = defaultdict(list)
    for row in rows:
        row["selected_for_pass_metrics"] = False
        if row.get("estimated_gap_m") is not None:
            groups[row["candidate_id"],row["optical_capture_at_utc"][:10]].append(row)
    for candidates in groups.values():
        # Source selection is independent of the loss, estimated gap and reference gap.
        primary = min(candidates, key=lambda r: (-r["fitted_pixels"],r["version_id"]))
        primary["selected_for_pass_metrics"] = True


def metrics(rows):
    errors = np.asarray([r["signed_error_m"] for r in rows],dtype=float)
    output = {"observations":len(rows),"resolved":sum(r["fit_status"]=="resolved_in_geometry_diagnostic" for r in rows)}
    if len(errors):
        response_difference = [abs(r["estimated_gap_m"]-r["box_response_gap_m"]) for r in rows]
        output.update(mae_m=float(abs(errors).mean()),median_abs_error_m=float(np.median(abs(errors))),
                      p90_abs_error_m=float(np.quantile(abs(errors),0.9)),bias_m=float(errors.mean()),
                      median_equivalent_abs_barrels=float(np.median([r["equivalent_absolute_barrels"] for r in rows])),
                      median_psf_to_box_gap_difference_m=float(np.median(response_difference)),
                      p90_psf_to_box_gap_difference_m=float(np.quantile(response_difference,0.9)))
        output.update(search_boundary=sum(r["fit_status"]=="search_boundary" for r in rows),
                      within_one_metre=sum(r["absolute_error_m"]<=1 for r in rows),
                      lidar_gap_in_near_optimal_range=sum(r["near_optimal_gap_min_m"]<=r["lidar_gap_m"]<=r["near_optimal_gap_max_m"] for r in rows),
                      minimum_sun_elevation_deg=min(r["sun_elevation_deg"] for r in rows),
                      maximum_sun_elevation_deg=max(r["sun_elevation_deg"] for r in rows))
        if all(r["shell_above_ground_m"] is not None for r in rows):
            output["half_shell_baseline_mae_m"] = float(np.mean([abs(r["shell_above_ground_m"]/2-r["lidar_gap_m"]) for r in rows]))
    return output


def summarize(rows):
    partitions = defaultdict(list)
    for row in rows:
        if not row["selected_for_pass_metrics"] or row.get("lidar_gap_m") is None:
            continue
        day = row["optical_capture_at_utc"][:10]
        roof = row["roof_type"]
        partitions[day,roof,"all"].append(row)
        if roof == "EFR":
            diameter = row["diameter_m"]
            size = "under_50m" if diameter<50 else "50_to_under_70m" if diameter<70 else "70m_and_over"
            partitions[day,roof,size].append(row)
            if row["fit_status"]=="resolved_in_geometry_diagnostic":
                partitions[day,roof,"resolved"].append(row)
    return [dict(optical_day=day,roof_type=roof,group=group,**metrics(partition))
            for (day,roof,group),partition in sorted(partitions.items())]


def summarize_aerial(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("same_day_aerial_top_observation"):
            groups[row["same_day_aerial_top_observation"]].append(row)
    return [dict(aerial_top_observation=top,observations=len(partition),
                 resolved=sum(r["fit_status"]=="resolved_in_geometry_diagnostic" for r in partition),
                 boundary=sum(r["fit_status"]=="search_boundary" for r in partition))
            for top,partition in sorted(groups.items())]


def overlapping_tiles(rows):
    observations = defaultdict(list)
    for row in rows:
        if row["roof_type"]=="EFR" and row.get("estimated_gap_m") is not None:
            observations[row["optical_capture_at_utc"][:10],row["candidate_id"]].append(row["estimated_gap_m"])
    spreads = defaultdict(list)
    for (day,_),gaps in observations.items():
        if len(gaps)>1:
            spreads[day].append(max(gaps)-min(gaps))
    return [dict(optical_day=day,overlapping_candidates=len(values),different_estimates=sum(v>0 for v in values),
                 maximum_gap_difference_m=max(values)) for day,values in sorted(spreads.items())]


def main():
    with closing(read_only(CALIBRATION)) as db, closing(read_only(DERIVED/"cushing.sqlite")) as registry:
        db.row_factory = sqlite3.Row
        observations,kernels = inputs(db)
        shells = dict(registry.execute("SELECT candidate_id,lidar_rim_to_ground_native_z FROM tank_engineering_registry"))
        with ThreadPoolExecutor(max_workers=3) as pool:
            estimates = []
            for row in pool.map(lambda o: estimate(o,shells[o["candidate_id"]],kernels[o["platform"]]),observations):
                estimates.append(row)
                if len(estimates)%100==0:
                    print(len(estimates),flush=True)
        add_reference(db,estimates)
    choose_primary(estimates)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer,fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(estimates)
    write(DERIVED/"shadow_geometry_estimates.csv",buffer.getvalue().encode())
    summary = {"comparisons":summarize(estimates),"same_day_aerial_controls":summarize_aerial(estimates),
               "overlapping_tile_checks":overlapping_tiles(estimates),
               "optical_observations":len(estimates),
               "status_counts":{status:sum(r["fit_status"]==status for r in estimates)
                                for status in sorted({r["fit_status"] for r in estimates})},
               "sampling_m":STEP_M,"maximum_gap_grid_m":MAX_GAP_M,"gap_grid_step_m":SEARCH_STEP_M,
               "registration_offsets_east_or_north_m":SHIFTS_M,"profile_span_limit_m":MAX_NEAR_OPTIMAL_SPAN_M,
               "required_rmse_improvement_over_flat_fraction":SHADOW_RMSE_GAIN_FRACTION,
               "reference_use":"Retrospective geometry and evaluation only. No LiDAR roof levels in optical fitting."}
    write(DERIVED/"shadow_geometry_diagnostic.json",json.dumps(summary,indent=2).encode())
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
