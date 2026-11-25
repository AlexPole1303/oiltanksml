"""Use a second verified pair for a roof while sharing its measurement errors."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import prior_scale
from fit_cushing_late_pair_recovery import accepted_pairs
from fit_cushing_mixed_optical_intervals import calibrated_weight, fitted, model
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import LOCAL_ACTIVITY_M, NOISE_FLOOR_M, PHASES
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import inputs, pass_id
from fit_cushing_robust_stack import indexed
from replay_cushing_weighted_inventory_market import replay_weighted_market

STEM = "cushing_repeated_roof_pairs"
INVENTORY = "cushing_mixed_optical_intervals"
METHODS = (SUPPLEMENT, "second_pair", "shared_roof_endpoint", "shared_endpoint_and_activity", "learned_second_pair")
MARGINAL = METHODS[3]
CURRENT = "second_pair_current_only"
MARKET_METHODS = (*METHODS, CURRENT)


def usable(row, late):
    primary = {r["candidate_id"]:r for r in row["observed"]}
    additions, audit = [], []
    for pair in late:
        key = pair["candidate_id"]
        base = primary.get(key)
        if base is None:
            continue
        if (pair["available_utc"] >= row["cutoff"] or pair["terminal_id"] != base["terminal_id"]
                or not math.isclose(pair["area"], base["area"], rel_tol=1e-8)):
            raise ValueError(f"Inconsistent repeated measurement on {row['release_date']}/{key}")
        before, after = (tuple(pass_id(q[k]) for k in ("scene_before", "scene_after")) for q in (base,pair))
        if before == after:
            continue
        shared = set(before) & set(after)
        additions.append(pair)
        audit.append(dict(release_date=row["release_date"], study_phase=row["study_phase"], tank=key, terminal=pair["terminal_id"],
            primary_before=base["scene_before"], primary_after=base["scene_after"], second_before=pair["scene_before"],
            second_after=pair["scene_after"], shared_passes=len(shared), cutoff_utc=row["cutoff"], available_utc=pair["available_utc"],
            before_volume_mbbl=base["pair_signal"], after_volume_mbbl=pair["pair_signal"]))
    return additions, audit


def interval(pair, total_area):
    if "starting_fraction" in pair:
        return pair["starting_fraction"], 1. + pair["trailing_fraction"]
    exposure = pair["sensitivity"] / (pair["area"] / total_area)
    start = exposure - pair["duration"]
    end = 1. + pair["post_fraction"] if "post_fraction" in pair else exposure
    if "post_fraction" in pair:
        start = -pair["pre_fraction"]
    if not math.isclose(end - start, pair["duration"], abs_tol=1e-9, rel_tol=0):
        raise ValueError(f"The capture interval does not reproduce on {pair['candidate_id']}")
    return start, end


def repeated_errors(current, row, *, activity):
    covariance = current["local"].copy()
    observations = current["observations"]
    loadings = []
    roofs = defaultdict(list)
    for i, q in enumerate(observations):
        tank = q["candidate_id"]
        roofs[tank].append(i)
        # The paired optical variance already includes one floor. Split it
        # over the two endpoints and retain their signs for adjoining pairs.
        sigma = NOISE_FLOOR_M * q["volume_per_m"] / math.sqrt(2)
        covariance[i,i] -= 2 * sigma**2
        loadings.append({(tank,pass_id(q[k])):sign * sigma for k,sign in (("scene_before",1.), ("scene_after",-1.))})
    if activity:
        for rows in roofs.values():
            for j, a in enumerate(rows):
                first, last = interval(observations[a], row["hub_area"])
                for b in rows[j+1:]:
                    start, end = interval(observations[b], row["hub_area"])
                    overlap = max(0., min(last, end) - max(first, start))
                    term = LOCAL_ACTIVITY_M**2 * observations[a]["volume_per_m"] * observations[b]["volume_per_m"] * overlap
                    covariance[a,b] += term
                    covariance[b,a] += term
    if np.any(covariance.diagonal() <= 0):
        raise ValueError(f"Nonpositive residual roof variance on {row['release_date']}")
    return current | dict(local=covariance, endpoint_errors=loadings)


def predictions():
    reference = indexed(f"{INVENTORY}_forecasts")
    late = accepted_pairs()
    past = {m:[] for m in METHODS[:-1]}
    errors, tuning = [], []
    for row in inputs():
        issue, cutoff, known, label = (row[k] for k in ("release_date", "cutoff", "known", "actual"))
        ref = reference[issue]
        if (any(datetime.fromisoformat(ref[k]) != row[k] for k in ("cutoff", "known")) or float(ref["actual"]) != label
                or any(ref[k] != row[k] for k in ("week_ending", "study_phase"))):
            raise ValueError(f"Forecast clocks differ on {issue}")
        additions, audit = usable(row, late[issue])
        values = dict.fromkeys(METHODS, row["public"])
        latest = None
        if row["observed"]:
            targets = (row["public"], label)
            primary = tuple(prepared(row, t, row["spanning"]) for t in targets)
            joint = tuple(model(row, t, p, [], additions, 1.) for t,p in zip(targets,primary,strict=True))
            sigma = prior_scale([q for q in errors if q["known"] < cutoff])
            for m, pair in zip(METHODS[:-1], (primary, joint,
                    tuple(repeated_errors(p,row,activity=False) for p in joint),
                    tuple(repeated_errors(p,row,activity=True) for p in joint)), strict=True):
                values[m], train = fitted(row, *pair, past[m], sigma)
                if train:
                    latest = max(latest or train, train)
        baseline = values[SUPPLEMENT]
        if not math.isclose(baseline, row["reference"], abs_tol=1e-9, rel_tol=0):
            raise ValueError(f"The anchor projection changed on {issue}")
        anchor = float(ref[f"long_{SUPPLEMENT}"])
        gain, prior_gain = calibrated_weight(tuning, cutoff, anchor=0.)
        if additions:
            tuning.append(dict(cutoff=cutoff, known=known, baseline=anchor, delta=values[MARGINAL] - baseline, actual=label))
        values[METHODS[-1]] = baseline + gain * (values[MARGINAL] - baseline)
        errors.append(dict(known=known, error=label-row["public"]))
        yield {k:row[k] for k in ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual")} | values | dict(
            main_roofs=len(row["observed"]), repeated_roofs=len(additions), weighted_gain=gain, latest_gain_label_utc=prior_gain,
            most_recent_optical_label=latest, shared_pass_roofs=sum(bool(q["shared_passes"]) for q in audit),
            **{f"long_{m}":anchor + value - baseline for m,value in values.items()}), audit


def summarize(rows, trades):
    phases = {}
    for phase in PHASES:
        sample = [r for r in rows if r["study_phase"] == phase]
        inventory = {}
        for group, panel in (("all", sample), ("second_pair_available", [r for r in sample if r["repeated_roofs"]])):
            errors = {m:1000 * np.asarray([abs(q[f"long_{m}"]-q["actual"]) for q in panel]) for m in METHODS}
            inventory[group] = {m:dict(weeks=len(e), mae_kbbl=float(e.mean()), paired_95_kbbl=block_bootstrap_mean(e-errors[SUPPLEMENT]))
                                for m,e in errors.items() if len(e)}
        trading = {}
        for method in (*MARKET_METHODS, *(f"{m}_conditional_exit" for m in MARKET_METHODS)):
            group = [r for r in trades if (r["study_phase"],r["method"]) == (phase,method)]
            control = f"{SUPPLEMENT}_conditional_exit" if method.endswith("_conditional_exit") else SUPPLEMENT
            base = [r for r in trades if (r["study_phase"],r["method"]) == (phase,control)]
            if [r["release_date"] for r in group] != [r["release_date"] for r in base]:
                raise ValueError(f"The second pair evaluation uses different trading dates on {phase}/{method}")
            trading[method] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(r["net_340_usd"] for r in group),
                changed_trades=sum(bool(a["near_side"] != b["near_side"] or a["near_side"] and a["exit_at_utc"] != b["exit_at_utc"])
                                   for a,b in zip(group,base,strict=True)),
                paired_delta_95_usd=block_bootstrap_mean([a["net_170_usd"]-b["net_170_usd"] for a,b in zip(group,base,strict=True)]))
        phases[phase] = dict(inventory=inventory, trading=trading, repeated_roofs=sum(q["repeated_roofs"] for q in sample),
                             roofs_with_shared_pass=sum(q["shared_pass_roofs"] for q in sample))
        print(phase, {k:{m:round(v["mae_kbbl"],3) for m,v in q.items()} for k,q in inventory.items()},
              {m:q["net_170_usd"] for m,q in trading.items()})
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def main():
    with threadpool_limits(limits=1):
        work = list(predictions())
        rows = [r for r,_ in work]
        book = {r["release_date"]:r | {CURRENT:r[MARGINAL]} for r in rows}
        trades = list(replay_weighted_market(book, MARKET_METHODS, reference=SUPPLEMENT, historical_inputs={CURRENT:SUPPLEMENT}))
    result = summarize(rows,trades)
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_pairs",[q for _,pairs in work for q in pairs])
    export(f"{STEM}_trades",trades)
    write(DERIVED / f"{STEM}_summary.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
