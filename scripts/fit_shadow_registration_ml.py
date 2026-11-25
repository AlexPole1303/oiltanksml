"""Learn spatial registration from stable exterior patches and apply it to B04 pairs."""
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from scipy.optimize import minimize

from build_weekly_shadow_development_check import read_csv
from build_weekly_shadow_panel import DERIVED, export
from build_weekly_shadow_validation import legacy_geometry
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "shadow_registration_ml"
WEEKLY = "sentinel2_weekly_refined_alignment"
MODES = ("coarse", "scene", "local", "spatial")
FROZEN_YEAR = 2023


def scene_key(row):
    return row["earlier_version"], row["later_version"]


def load():
    patches, pairs = defaultdict(dict), defaultdict(list)
    for stem in (WEEKLY, f"{WEEKLY}_holdout"):
        for r in read_csv(f"{stem}_patches"):
            if r["refinement_status"] != "interior_peak":
                continue
            scene = patches[scene_key(r)]
            anchor = r["anchor_id"]
            if anchor in scene:
                raise ValueError(f"Duplicate exterior patch {scene_key(r)}/{anchor}")
            scene[anchor] = r
        for r in read_csv(f"{stem}_pairs"):
            if not datetime.fromisoformat(r["archive_available_at_utc"]) < datetime.fromisoformat(r["cutoff_utc"]):
                raise ValueError(f"Unavailable source for {r['week_ending']}/{r['candidate_id']}")
            pairs[scene_key(r)].append(r)
    if patches.keys() != pairs.keys():
        raise ValueError("Different scene identities in patch and tank tables")
    return patches, pairs


def displacement(r, name):
    return [float(r[f"{name}_{axis}_m"]) for axis in ("east", "north")]


def coordinates(r):
    return tuple(map(float, r["anchor_id"].split("_")))


def spatial_candidates(donors, points):
    xy = np.asarray([coordinates(r) for r in donors])
    coarse = np.asarray([displacement(r, "coarse") for r in donors])
    refined = np.asarray([displacement(r, "refined") for r in donors])
    scene = np.median(refined, axis=0)
    quality = np.clip([float(r["refined_correlation"]) for r in donors], .2, 1.)
    eligible = np.linalg.norm(refined-scene, axis=1) <= 6
    predictions = []
    for point in np.asarray(points):
        distance = np.linalg.norm(xy-point, axis=1)
        order = np.argsort(distance, kind="stable")
        nearest = order[eligible[order]][:8]
        if not len(nearest):
            nearest = order[:1]
        x = np.column_stack((np.ones(len(nearest)), (xy[nearest]-point)/350))
        weights = quality[nearest]**2/(1+(distance[nearest]/230)**2)
        local = np.average(refined[nearest], axis=0, weights=weights)
        penalty = np.diag([.12, .7, .7])*weights.sum()
        affine = scene + np.linalg.solve(x.T@(weights[:, None]*x)+penalty,
                                        x.T@(weights[:, None]*(refined[nearest]-scene)))[0]
        affine = np.clip(affine, refined[nearest].min(axis=0), refined[nearest].max(axis=0))
        predictions.append((np.median(coarse[nearest[:4]], axis=0), scene, local, affine))
    return np.asarray(predictions)


def scene_samples(patches, pairs):
    data = {}
    for key, values in patches.items():
        fit = [r for r in values.values() if r["role"] == "fit"]
        check = [r for r in values.values() if r["role"] == "spatial_check"]
        if min(len(fit), len(check)) < 3:
            continue
        latest = max(datetime.fromisoformat(r["archive_available_at_utc"]) for r in pairs[key])
        data[key] = dict(fit=fit, check=check, latest=latest, phases={r["study_phase"] for r in pairs[key]},
            x=spatial_candidates(fit, [coordinates(r) for r in check]),
            y=np.asarray([displacement(r, "refined") for r in check]))
    return data


def learn(data, year):
    boundary = datetime.fromisoformat(f"{year}-01-01T00:00:00+00:00")
    # The fixed gap avoids recycling one scene as the endpoint of adjacent-year pairs.
    eligible = {key:s for key,s in data.items()
                if s["phases"] == {"development"} and s["latest"] < boundary-timedelta(days=28)}
    train = list(eligible.values())
    if len(train) < 12:
        return None
    x = np.concatenate([s["x"] for s in train])
    y = np.concatenate([s["y"] for s in train])
    mass = np.concatenate([np.full(len(s["y"]), 1/len(s["y"])) for s in train])
    prior = np.array([0., .35, .35, .30])

    def objective(w):
        residual = np.einsum("nkd,k->nd", x, w)-y
        return np.average(np.sqrt(.5**2+(residual**2).sum(axis=1)), weights=mass)+.10*np.sum((w-prior)**2)

    result = minimize(objective, prior, method="SLSQP", bounds=[(0., 1.)]*len(prior),
                      constraints=[{"type": "eq", "fun": lambda w: w.sum()-1}],
                      options={"ftol": 1e-10, "maxiter": 200})
    if not result.success or not np.isclose(result.x.sum(), 1):
        raise RuntimeError(f"Cannot fit registration for {year}: {result.message}")
    return dict(weights=result.x, scenes=len(train), patches=len(x), latest=max(s["latest"] for s in train),
                source_versions={v for pair in eligible for v in pair})


def baseline_pairs():
    result = {}
    for stem in ("sentinel2_space_time_coverage_tank_weeks", "holdout_optical_tank_weeks"):
        for r in read_csv(stem):
            if r["baseline_b04_resolved"] != "True" or (stem.startswith("sentinel2") and r["study_phase"] == "holdout"):
                continue
            key = r["release_date"], r["week_ending"], r["candidate_id"]
            if key in result:
                raise ValueError(f"Repeated published B04 baseline at {key}")
            result[key] = (r["earlier_sentinel2_version"], r["later_sentinel2_version"]), float(r["capture_change_kbbl"])
    return result


def evaluate(pairs, data, models):
    controls, measurements = [], []
    geometry, baseline = legacy_geometry(), baseline_pairs()
    for key, rows in sorted(pairs.items(), key=lambda item: tuple(v.split("_")[2] for v in item[0])):
        sample = data.get(key)
        by_year = defaultdict(list)
        for r in rows:
            stored = baseline.get((r["release_date"], r["week_ending"], r["candidate_id"]))
            if r["original_resolved"] == "True" and stored and stored[0] == key:
                if not math.isclose(float(r["coarse_change_kbbl"]), stored[1], abs_tol=1e-8, rel_tol=0):
                    raise ValueError(f"Original B04 result differs for {r['release_date']}/{r['candidate_id']}")
                by_year[min(int(r["release_date"][:4]), FROZEN_YEAR)].append(r)
        for year, selected in by_year.items():
            model = models[year]
            shifts = (spatial_candidates(sample["fit"], [(float(geometry[r["candidate_id"]]["center_east_m"]),
                        float(geometry[r["candidate_id"]]["center_north_m"])) for r in selected])
                      if model and sample else [None]*len(selected))
            for r, candidates in zip(selected, shifts, strict=True):
                mode = "coarse"
                if candidates is not None:
                    if model["latest"] >= datetime.fromisoformat(r["cutoff_utc"]) or not model["source_versions"].isdisjoint(key):
                        raise ValueError("Registration model includes a later or evaluated image")
                    expected = model["weights"]@candidates
                    eligible = ["coarse", *(m for m in ("local", "scene") if r[f"{m}_can_use"] == "True")]
                    mode = min(eligible, key=lambda m: np.linalg.norm(np.asarray(displacement(r, m))-expected))
                measurements.append({f:r[f] for f in ("study_phase", "week_ending", "release_date", "candidate_id",
                    "terminal_id", "cutoff_utc", "archive_available_at_utc", "earlier_version", "later_version")} |
                    dict(selected_mode=mode, ml_applied=candidates is not None,
                         coarse_change_kbbl=float(r["coarse_change_kbbl"]), selected_change_kbbl=float(r[f"{mode}_change_kbbl"]),
                         latest_training_image_at_utc=model["latest"].isoformat() if candidates is not None else None))
        if not sample or sample["phases"] not in ({"validation"}, {"holdout"}):
            continue
        model = models[FROZEN_YEAR]
        if model["latest"] >= sample["latest"] or not model["source_versions"].isdisjoint(key):
            raise ValueError("Test scene predates the model or appears in training")
        predicted = np.concatenate((sample["x"], np.einsum("nkd,k->nd", sample["x"], model["weights"])[:, None]), axis=1)
        errors = np.linalg.norm(predicted-sample["y"][:, None], axis=2)
        for patch, error in zip(sample["check"], errors, strict=True):
            controls.append(dict(study_phase=next(iter(sample["phases"])), earlier_version=key[0], later_version=key[1],
                anchor_id=patch["anchor_id"], **{f"{m}_error_m":e for m,e in zip((*MODES, "ml"), error, strict=True)}))
    if len({(r["release_date"],r["week_ending"],r["candidate_id"]) for r in measurements}) != len(measurements):
        raise ValueError("B04 footprint changed under learned registration")
    return controls, measurements


def summarize(controls, measurements):
    result = {}
    for phase in ("validation", "holdout"):
        rows = [r for r in controls if r["study_phase"] == phase]
        by_scene = defaultdict(list)
        for r in rows:
            by_scene[scene_key(r)].append(r)
        errors = {m: [np.mean([r[f"{m}_error_m"] for r in sample]) for sample in by_scene.values()] for m in (*MODES, "ml")}
        result[phase] = dict(scenes=len(by_scene), patches=len(rows), scene_balanced_mean_error_m={m:float(np.mean(v)) for m,v in errors.items()},
            ml_minus_local_ci_m=block_bootstrap_mean(np.array(errors["ml"])-errors["local"]),
            ml_minus_scene_median_ci_m=block_bootstrap_mean(np.array(errors["ml"])-errors["scene"]))
    usage = {}
    for phase in ("development", "validation", "holdout"):
        sample = [r for r in measurements if r["study_phase"] == phase]
        changes = [r for r in sample if r["selected_mode"] != "coarse"]
        usage[phase] = dict(tank_weeks=len(sample), changed_mode=len(changes), affected_weeks=len({r["week_ending"] for r in changes}),
                            median_absolute_change_kbbl=float(np.median([abs(r["coarse_change_kbbl"]-r["selected_change_kbbl"])
                                                                         for r in changes])) if changes else None)
    return result, usage


def main():
    patches, pairs = load()
    data = scene_samples(patches, pairs)
    years = sorted({min(int(r["release_date"][:4]), FROZEN_YEAR) for sample in pairs.values() for r in sample})
    models = {year:learn(data, year) for year in years}
    controls, measurements = evaluate(pairs, data, models)
    errors, usage = summarize(controls, measurements)
    report = dict(exterior_checks=errors, applied_b04=usage, training_models={y:None if v is None else
        dict(scenes=v["scenes"], patches=v["patches"], latest_training_image_at_utc=v["latest"].isoformat(),
             weights=dict(zip(MODES, v["weights"].tolist(), strict=True))) for y,v in models.items()},
        interpretation="Independent check patches measure image displacement, not liquid height. "
        "Tank choices retain the same native endpoints and pass the original optical checks. "
        "No EIA targets or WTI prices are read.")
    export(f"{STEM}_patch_checks", controls)
    export(f"{STEM}_tank_weeks", measurements)
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print(payload, end="")


if __name__ == "__main__":
    main()
