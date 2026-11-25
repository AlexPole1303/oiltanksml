"""Separate post-survey movement from the signal in repeated roof measurements."""

import json
import math
from datetime import date, datetime, timedelta

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_intraweek import estimate, learned_weights, prior_scale
from fit_cushing_late_pair_recovery import accepted_pairs
from fit_cushing_mixed_optical_intervals import calibrated_weight, model
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import SUPPLEMENT, prepared
from fit_cushing_overpass_covariance import BASE, forecast, inputs
from fit_cushing_repeated_roof_pairs import INVENTORY, interval, repeated_errors, usable
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import MAX_SCENE_AGE, PRIOR_INVENTORY_SIGMA_MBBL, TERMINAL_SIGMA_M
from fit_cushing_weekend_timing import weighted_time
from replay_cushing_weighted_inventory_market import replay_weighted_market

STEM = "cushing_late_pair_timing"
METHODS = (SUPPLEMENT, "same_covariance_delta", "overlap_terminal", "slower_weekend_tail",
           "overlap_and_weekend_tail", "learned_gain", "ratio_weighted_pairs", "qualified_ratio_update")
FULL_DELTAS = METHODS[1:5]


def terminal_overlap(candidate, row):
    observations = candidate["observations"]
    starts, ends = np.array([interval(q,row["hub_area"]) for q in observations]).T
    overlap = np.maximum(0.,np.minimum.outer(ends,ends)-np.maximum.outer(starts,starts))
    days = ends-starts
    volume = np.array([q["volume_per_m"] for q in observations])
    term = np.array([q["terminal_id"] for q in observations])
    weights = TERMINAL_SIGMA_M**2*np.outer(volume,volume)*(term[:,None]==term[None,:])
    # Independent increments within a terminal covary only on shared time.
    return candidate | dict(local=candidate["local"] + weights*(overlap - np.sqrt(np.outer(days,days))))


def weekend_tail(candidate, row):
    week = date.fromisoformat(row["week_ending"])+timedelta(weeks=1)
    if week not in candidate["clocks"]:
        return candidate
    old_terms = candidate["clocks"][week]
    if isinstance(old_terms[0],np.ndarray):
        old_terms = (old_terms,)
    new_terms = tuple((weighted_time(t,week,.5),a) for t,a in old_terms)
    before = sum(t*a for t,a in old_terms)
    after = sum(t*a for t,a in new_terms)
    return candidate | dict(clocks=candidate["clocks"] | {week:new_terms},
        local=candidate["local"]+PRIOR_INVENTORY_SIGMA_MBBL**2*(np.outer(after,after)-np.outer(before,before)))


def ratio_weighted(pairs):
    # Downweight a late measurement in proportion to how much of its
    # interval actually belongs to the target week. This is independent of
    # the prior on post-survey aggregate flow in the covariance model.
    return [q | dict(variance=q["variance"]*(q["duration"]/(1-q["starting_fraction"]))**2) for q in pairs]


def pair_deltas(row, state, pairs):
    cutoff = row["cutoff"]
    history = [r for r in state["measured"] if r["known"] < cutoff]
    recent = [r["models"][BASE] for r in history if cutoff-r["cutoff"]<=MAX_SCENE_AGE]
    ordinary = prepared(row,row["public"],row["spanning"])
    additive = model(row,row["public"],ordinary,[],pairs,1.)
    weighted = model(row,row["public"],ordinary,[],ratio_weighted(pairs),1.)
    sigma = prior_scale([q for q in state["errors"] if q["known"] < cutoff])
    weights = learned_weights([dict(absolute_errors=r["errors_by_timing"][BASE]) for r in history])

    def predict(candidate, past):
        values = estimate(repeated_errors(candidate,row,activity=True),past)
        return float(weights @ np.clip([q[2]/(1/sigma**2+q[1]) for q in values],-1.25,1.25))

    unchanged = [repeated_errors(p,row,activity=False) for p in recent]
    pairs_by_variant = (
        (ordinary,additive),
        (terminal_overlap(ordinary,row),terminal_overlap(additive,row)),
        (weekend_tail(ordinary,row),weekend_tail(additive,row)),
        (terminal_overlap(weekend_tail(ordinary,row),row),terminal_overlap(weekend_tail(additive,row),row)),
        (ordinary,weighted))
    return [predict(b,unchanged)-predict(a,unchanged) for a,b in pairs_by_variant], history[-1]["known"] if history else None


def qualified_gain(previous, row, delta):
    # Require repeated improvement among already scored issues with
    # comparable timing. Default to zero when the measured benefit is small
    # relative to the week-to-week dispersion of paired errors.
    compatible = [q for q in previous if q["known"] < row["cutoff"]
                  and abs(q["tail_share"]-row["tail_share"]) <= .20][-52:]
    if len(compatible) < 12 or not delta:
        return 0.,len(compatible)
    improvement = np.array([abs(q["actual"]-q["baseline"])-abs(q["actual"]-q["baseline"]-.5*q["delta"])
                            for q in compatible])
    mean = improvement.mean()
    noise = improvement.std(ddof=1)/math.sqrt(len(improvement))
    return (.5 if mean > noise else 0.),len(compatible)


def replay():
    prior = indexed(f"{INVENTORY}_forecasts")
    late = accepted_pairs()
    state = dict(measured=[],errors=[])
    tuning = []
    ratio_history = []
    for row in inputs():
        issue, cutoff = row["release_date"], row["cutoff"]
        additions, audit = usable(row,late[issue])
        ref = prior[issue]
        long_anchor = float(ref[f"long_{SUPPLEMENT}"])
        if (any(datetime.fromisoformat(ref[k]) != row[k] for k in ("cutoff","known"))
                or ref["week_ending"] != row["week_ending"]
                or float(ref["actual"]) != row["actual"] or ref["study_phase"] != row["study_phase"]):
            raise ValueError(f"Mismatched history on {issue}")
        deltas, label = pair_deltas(row,state,additions) if additions else ([0.]*(len(FULL_DELTAS)+1),None)
        physical, _, _, _, _ = forecast(row,state)
        base = physical[BASE]
        if not math.isclose(base,float(ref[SUPPLEMENT]),rel_tol=0.,abs_tol=1e-9):
            raise ValueError(f"The frozen trading forecast differs on {issue}")
        gain, known = calibrated_weight(tuning,cutoff,anchor=0.)
        shared_update, ratio_update = deltas[-2:]
        ratio_gain, compatible = 0.,0
        tail_share = None
        if additions:
            scored_fields = dict(cutoff=cutoff,known=row["known"],baseline=long_anchor,actual=row["actual"])
            tuning.append(scored_fields | dict(delta=shared_update))
            tail_share = float(np.median([q["trailing_fraction"]/q["duration"] for q in additions]))
            candidate = scored_fields | dict(delta=ratio_update,tail_share=tail_share)
            ratio_gain,compatible = qualified_gain(ratio_history,candidate,ratio_update)
            ratio_history.append(candidate)
        deltas = dict(zip(METHODS,(0.,*deltas[:-1],gain*shared_update,ratio_update,ratio_gain*ratio_update),strict=True))
        yield {k:row[k] for k in ("release_date","week_ending","study_phase","cutoff","known","actual")} | dict(
            added_pairs=len(additions),shared_pass_roofs=sum(bool(q["shared_passes"]) for q in audit),
            latest_physics_label_utc=label, latest_gain_label_utc=known,gain=gain,
            tail_share=tail_share,ratio_gain=ratio_gain,compatible_ratio_weeks=compatible,
            median_late_hours=168*float(np.median([q["trailing_fraction"] for q in additions])) if additions else None,
            median_inside_hours=168*float(np.median([1-q["starting_fraction"] for q in additions])) if additions else None,
            **{m:base+v for m,v in deltas.items()},**{f"long_{m}":long_anchor+v for m,v in deltas.items()})


def summarize(rows,trades):
    report = {}
    for phase in PHASES:
        group = [r for r in rows if r["study_phase"]==phase]
        observed = [r for r in group if r["added_pairs"]]
        bins = dict(all=group,repeated=observed,short_tail=[r for r in observed if r["median_late_hours"]<=36],
                    long_tail=[r for r in observed if r["median_late_hours"]>36])
        inventory = {}
        for name,sample in bins.items():
            if not sample:
                continue
            errors = {m:1000*np.array([abs(q[f"long_{m}"]-q["actual"]) for q in sample]) for m in METHODS}
            inventory[name] = {m:dict(weeks=len(v),mae_kbbl=float(v.mean()),paired_95_kbbl=block_bootstrap_mean(v-errors[SUPPLEMENT]))
                               for m,v in errors.items()}
        market = {}
        for method in METHODS:
            sample = [q for q in trades if q["study_phase"]==phase and q["method"]==f"{method}_conditional_exit"]
            market[method] = robustness(sample) | daily_risk(sample) | dict(net_340_usd=sum(q["net_340_usd"] for q in sample))
        report[phase] = dict(inventory=inventory,market=market)
        print(phase,{m:(round(inventory["all"][m]["mae_kbbl"],2),v["net_170_usd"]) for m,v in market.items()},flush=True)
    return dict(status="exploratory_on_reviewed_history",phases=report)


def main():
    with threadpool_limits(limits=1):
        rows = list(replay())
        trades = list(replay_weighted_market({r["release_date"]:r for r in rows},METHODS,reference=SUPPLEMENT))
    result = summarize(rows,trades)
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
