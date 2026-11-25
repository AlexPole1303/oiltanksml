"""Estimate trade-signal sensitivity to blocks of the settled training sample."""

import json
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_cl_entry_margin import inputs_with_reference, side
from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import trade
from fit_cushing_cl_regime_strategy import MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from register_cushing_leading_models import prediction

STEM = "cushing_cl_training_uncertainty"
METHODS = ("reference", "block_median", "three_quarters_agreement", "half_standard_error", "one_standard_error")
BLOCK_WEEKS = 13


def jackknife(settled, calibration, current):
    # Drop consecutive releases, with block sizes differing by at most one.
    # All candidate fits still use strictly earlier and settled outcomes.
    count = len(settled)//BLOCK_WEEKS
    if count < 5:
        return [], None
    estimates, blocks = [], []
    for indices in np.array_split(np.arange(len(settled)), count):
        start, stop = int(indices[0]), int(indices[-1])+1
        reduced = settled[:start]+settled[stop:]
        if len(reduced) < MIN_WINDOWS:
            return [], None
        first, last = (settled[i]["release_date"] for i in (start,stop-1))
        local = [r for r in calibration if not first <= r["release_date"] <= last]
        point, correction = prediction(reduced, local, current)
        estimates.append(point+correction)
        blocks.append((first,last))
    return estimates, blocks


def decisions(central, estimates):
    if not estimates:
        return dict.fromkeys(METHODS, central), None, None
    values = np.asarray(estimates)
    k = len(values)
    jack_sd = float(np.sqrt((k-1)/k*np.square(values-values.mean()).sum()))
    incumbent = side(central, HURDLE)
    agreement = sum(side(v, HURDLE) == incumbent for v in estimates)/k
    authority = float(np.sign(central))
    adjusted = tuple(authority*max(0., abs(central)-factor*jack_sd) for factor in (.5, 1.))
    result = (central,float(np.median(values)),central if agreement >= .75 else 0.,*adjusted)
    return dict(zip(METHODS,result,strict=True)),jack_sd,agreement


def replay():
    trades, dependence = [], []
    for r, current, ref, value, settled, calibration in inputs_with_reference():
        estimates, blocks = jackknife(settled, calibration, current)
        chosen, sd, fraction = decisions(value, estimates)
        original_side = side(value,HURDLE)
        if len(estimates):
            for estimate,(first,last) in zip(estimates,blocks,strict=True):
                dependence.append(dict(release_date=r["release_date"], study_phase=r["study_phase"],
                    decision_at_utc=r["decision"], near_month=r["near_month"], far_month=r["far_month"],
                    first_training_release_removed=first, last_training_release_removed=last,
                    expected_long_move_usd=estimate, full_estimate_usd=value, near_side=side(estimate,HURDLE),
                    side_matches_reference=side(estimate,HURDLE)==original_side))
        for m,estimate in chosen.items():
            q = trade(r,m,estimate,3,training_at=settled[-1]["exit_at"],
                      calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if m=="reference" and any(q[k]!=int(ref[k]) for k in ("near_side","gross_usd","net_170_usd")):
                raise ValueError(f"The training-block replay changed the original result on {r['release_date']}")
            trades.append(q | dict(reference_estimate_usd=value, reference_side=original_side,
                model_standard_error_usd=sd, same_action_fold_fraction=fraction, training_blocks=len(estimates),
                training_windows=len(settled), last_training_release=settled[-1]["release_date"], realized_long_move_usd=r["long_gross"],
                net_340_usd=q["gross_usd"]-340*abs(q["near_side"])))
    return trades, dependence


def summarize(trades):
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["method"]].append(r)
    report = dict(status="exploratory_on_reviewed_history", removal_block_target_weeks=BLOCK_WEEKS, phases={})
    for phase in PHASES:
        output = report["phases"][phase] = {}
        base = groups[phase,"reference"]
        measured = [r for r in base if r["model_standard_error_usd"] is not None]
        for m in METHODS:
            g = groups[phase,m]
            pairs = list(zip(g,base,strict=True))
            if any(r["release_date"]!=a["release_date"] for r,a in pairs):
                raise ValueError(f"Trading calendars differ for {phase}/{m}")
            diffs = [r["net_170_usd"]-a["net_170_usd"] for r,a in pairs]
            changed = [r for r,a in pairs if r["near_side"]!=a["near_side"]]
            errors = [abs(r["realized_long_move_usd"]-r["expected_long_move_usd"]) for r in g]
            output[m] = robustness(g) | daily_risk(g) | dict(net_340_usd=sum(r["net_340_usd"] for r in g),
                paired_delta_95_usd=block_bootstrap_mean(diffs), changed_dates=[r["release_date"] for r in changed],
                spread_point_mae_usd=float(np.mean(errors)))
        active = [r for r in measured if r["near_side"]]
        output["training_dependence"] = dict(eligible_releases=len(measured), fragile_reference_trades=sum(r["same_action_fold_fraction"]<.75 for r in active),
            reference_trades=len(active), median_standard_error_usd=float(np.median([r["model_standard_error_usd"] for r in measured])),
            blocks_per_release=dict(Counter(r["training_blocks"] for r in measured)))
        print(phase,{m:(q["trades"],q["net_170_usd"],q["net_340_usd"],q["daily_drawdown_usd"]) for m,q in output.items() if m in METHODS})
    return report


def main():
    with threadpool_limits(limits=1):
        trades,dependence = replay()
    report = summarize(trades)
    export(f"{STEM}_trades",trades)
    export(f"{STEM}_blocks",dependence)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
