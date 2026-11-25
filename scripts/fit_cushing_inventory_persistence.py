"""Learn persistence of inventory forecast errors and its value for dated CL trades."""

import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import trading as base_trades
from fit_cushing_outer_capture_nowcast import SUPPLEMENT
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import fit_correction
from fit_cushing_terminal_weekly_transfers import LIMITED_MARKET as BASE, STEM as TERMINALS
from fit_cushing_volatility_neighbors import STEM as VOLATILITY

STEM = "cushing_inventory_persistence"
RAW_METHODS = ("innovation_memory", "innovation_and_inventory")
ADAPTIVE = "validated_innovation_memory"
INVENTORY = (PUBLIC, SUPPLEMENT, BASE, *RAW_METHODS, ADAPTIVE)
REFERENCE = "weighted_satellite_reference"
AFTER_RELEASE = (REFERENCE, "persistent_surprise", "persistent_deficit")
DIFFERENCE_CI = "mean_absolute_error_change_vs_base_95_kbbl"


def state(week, known):
    weeks = [week-timedelta(days=7*k) for k in range(4,0,-1)]
    completed = [known.get(day) for day in weeks]
    available = [r for r in completed if r is not None]
    if len(available)!=4:
        return None
    error = np.clip([r["actual"]-r[BASE] for r in completed],-2.,2.)
    movement = np.clip([r["actual"] for r in completed],-2.,2.)
    # Distinguish a new one-week error from several weeks of directional bias.
    innovations = (float(error[-1]),float(error[:-1].mean()))
    balance = (float(movement.mean()),float(movement[-1]-movement[:-1].mean()))
    return dict(innovation_design=innovations, balance_design=(*innovations,*balance),
                previous_error_million_bbl=float(error[-1]), previous_change_million_bbl=float(movement[-1]),
                four_week_change_million_bbl=float(movement.sum()),
                latest_lag_label_utc=max(r["known"] for r in available), previous_release=available[-1]["release_date"])


def validated_adjustment(eligible, base, raw, cutoff):
    # Evaluate the forecasts actually produced in earlier weeks, not their
    # in-sample fitted values. Require evidence above a zero-gain prior.
    scored = [r for r in eligible if r["used_memory"]][-52:]
    if len(scored)<26:
        return base, 0., len(scored)
    actual = np.asarray([r["actual"] for r in scored])
    control = np.asarray([r[BASE] for r in scored])
    alternatives = np.array([r[RAW_METHODS[0]] for r in scored])
    difference = np.clip(abs(actual-control)-abs(actual-alternatives),-.3,.3)
    elapsed = np.asarray([(cutoff-r["cutoff"]).total_seconds()/604800 for r in scored])
    w = np.exp2(-elapsed/26.)
    advantage = float(w@difference/(w.sum()+8.))
    # Shrink toward the existing forecast when the assessed gain is small.
    weight = float(np.clip(advantage/.04,0.,1.))
    return base+weight*(raw-base), weight, len(scored)


def inventory():
    archive = indexed(f"{TERMINALS}_forecasts")
    seen = {}
    for day,q in sorted(archive.items()):
        week = date.fromisoformat(q["week_ending"])
        cutoff,known = (datetime.fromisoformat(q[k]) for k in ("cutoff","known"))
        if (week in seen or not cutoff<known or any(q[k] and datetime.fromisoformat(q[k])>=cutoff
                for k in ("latest_training_label_utc",))):
            raise ValueError(f"Noncausal weekly forecast on {day}")
        actual = float(q["actual"])
        base = float(q[BASE])
        eligible = {week:r for week,r in seen.items() if r["known"]<cutoff}
        current_state = state(week,eligible)
        train = [r for r in eligible.values() if r["innovation_design"] is not None][-HISTORY:]
        current = dict(release_date=day,week_ending=q["week_ending"],study_phase=q["study_phase"],cutoff=cutoff,known=known,
                       actual=actual,residual=actual-base, innovation_design=None,balance_design=None,
                       previous_error_million_bbl=None,previous_change_million_bbl=None,four_week_change_million_bbl=None,
                       latest_lag_label_utc=None,previous_release=None,
                       **{m:float(q[m]) for m in INVENTORY[:3]}) | (current_state or {})
        forecasts = [base,base]
        fitted = current_state is not None and len(train)>=MIN_WINDOWS
        if fitted:
            for i,key in enumerate(("innovation_design","balance_design")):
                penalty = (.65,.8,1.1,1.25)[:len(current[key])]
                delta = fit_correction(train,current,key,"residual",penalty=penalty,loss_scale=.65,bound=.40)
                forecasts[i] += delta
        adjusted,weight,count = validated_adjustment(eligible.values(),base,forecasts[0],cutoff)
        result = current | dict(zip(RAW_METHODS,forecasts,strict=True)) | {
            ADAPTIVE:adjusted, "adaptive_memory_weight":weight, "adaptive_scored_weeks":count, "used_memory":fitted,
            "state_training_windows":len(train), "latest_state_training_utc":train[-1]["known"] if train else None,
            "latest_validation_outcome_utc":max((r["known"] for r in eligible.values() if r["used_memory"]),default=None),
        }
        seen[week] = result
        yield result


def after_release(inventory_rows):
    records = {r["release_date"]:r for r in inventory_rows}
    anchors = {r["release_date"]:r for r in read(f"{VOLATILITY}_trades") if r["method"]==f"{SUPPLEMENT}:volatility"}
    history = []
    for row,_ in inputs():
        day = row["release_date"]
        ref = anchors.get(day)
        if ref is None:
            continue
        f = records[day]
        cutoff = row["decision"]
        scale = row["volatility_usd"]
        if (not f["cutoff"]<f["known"]<=cutoff<row["entry"]<row["exit_at"] or f["actual"]!=row["x"][0]
                or any(ref[k]!=row[k] for k in ("near_month","far_month")) or any(
                    datetime.fromisoformat(ref[k])!=row[field] for k,field in (
                        ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")))
                or any(datetime.fromisoformat(ref[k])>=cutoff for k in (
                        "latest_training_exit_utc","latest_calibration_exit_utc") if ref[k])):
            raise ValueError(f"Invalid inventory or market clocks on {day}")
        base = float(ref["expected_long_move_usd"])
        previous = f["previous_error_million_bbl"]
        previous_change = f["previous_change_million_bbl"]
        surprise, lag, balance = np.clip((f["actual"]-f[BASE],previous or 0.,(f["four_week_change_million_bbl"] or 0.)/3.),-2.,2.)
        # Follow-through is represented by signed overlap with the prior
        # week's forecast error; isolated breaks have the opposite term.
        common = math.copysign(min(abs(surprise),abs(lag)),surprise) if surprise*lag>0 else 0.
        reversal = math.copysign(min(abs(surprise),abs(lag)),surprise) if surprise*lag<0 else 0.
        surprise_state = (common,reversal,lag)
        design = (*surprise_state,balance,surprise*balance,float(previous_change or 0.))
        current = row | dict(cutoff=cutoff,shock_state=surprise_state,flow_state=design,
                             market_residual=(row["long_gross"]-base)/scale)
        available = [q for q in history if q["exit_at"]<cutoff][-HISTORY:]
        if previous is not None:
            history.append(current)
        estimates = [base,base,base]
        if previous is not None:
            for j,field in enumerate(("shock_state","flow_state"),1):
                d = fit_correction(available,current,field,"market_residual",penalty=(1.2,1.2,1.5,1.35,2.,2.)[:len(current[field])],
                                   loss_scale=1.,bound=.40)
                estimates[j] += scale*d
        baseline_training = datetime.fromisoformat(ref["latest_training_exit_utc"])
        baseline_calibration = datetime.fromisoformat(ref["latest_calibration_exit_utc"]) if ref["latest_calibration_exit_utc"] else None
        for name,estimate in zip(AFTER_RELEASE,estimates,strict=True):
            stamped = baseline_training if name==REFERENCE or not available else max(baseline_training,available[-1]["exit_at"])
            result = trade(row,name,estimate,3,training_at=stamped,calibration_at=baseline_calibration)
            if name==REFERENCE and (result["near_side"]!=int(ref["near_side"]) or result["gross_usd"]!=int(ref["gross_usd"])):
                raise ValueError(f"Weighted reference failed to reproduce on {day}")
            yield result | dict(last_release_label_utc=f["latest_lag_label_utc"],known_shock_million_bbl=surprise,
                previous_shock_million_bbl=previous,aligned_surprise_million_bbl=common,reversed_surprise_million_bbl=reversal,
                state_fit_windows=len(available),net_340_usd=result["gross_usd"]-340*abs(result["near_side"]))


def score_inventory(rows):
    if not rows:
        return dict(releases=0,models={})
    errors = {m:np.asarray([1000*abs(r[m]-r["actual"]) for r in rows]) for m in INVENTORY}
    return dict(releases=len(rows),models={m:dict(mae_kbbl=float(v.mean()),
        rmse_kbbl=float(np.sqrt(np.mean(v*v))),**{DIFFERENCE_CI:block_bootstrap_mean(v-errors[BASE])}) for m,v in errors.items()})


def main():
    with threadpool_limits(limits=1):
        rows = list(inventory())
        base = list(base_trades(rows,INVENTORY))
        refit = list(replay_conditional_market(rows,base,INVENTORY,references={PUBLIC:f"{PUBLIC}_conditional",SUPPLEMENT:f"{SUPPLEMENT}_conditional"}))
        shock = list(after_release(rows))
    r = dict(status="exploratory_on_reviewed_history",inventory={},trading={})
    by_policy = defaultdict(list)
    for row in (*refit,*shock):
        by_policy[row["study_phase"],row["method"]].append(row)
    for phase in PHASES:
        cohort = [q for q in rows if q["study_phase"]==phase]
        r["inventory"][phase] = {n:score_inventory(subset) for n,subset in (("all",cohort),
            ("prior_draw",[q for q in cohort if q["previous_change_million_bbl"] is not None and q["previous_change_million_bbl"]<0]),
            ("prior_build",[q for q in cohort if q["previous_change_million_bbl"] is not None and q["previous_change_million_bbl"]>0]))}
        market = r["trading"][phase] = {}
        for method in (*INVENTORY,*AFTER_RELEASE):
            sample = by_policy[phase,method]
            comparator = by_policy[phase,BASE if method in INVENTORY else REFERENCE]
            if not sample or [q["release_date"] for q in sample]!=[q["release_date"] for q in comparator]:
                raise ValueError(f"Different backtest calendars on {phase}/{method}")
            pairs = list(zip(sample,comparator,strict=True))
            delta = [a["net_170_usd"]-b["net_170_usd"] for a,b in pairs]
            market[method] = robustness(sample) | daily_risk(sample) | dict(
                net_340_usd=sum(q["gross_usd"]-340*abs(q["near_side"]) for q in sample),
                cash_delta_per_release_95_usd=block_bootstrap_mean(delta),
                changed_entries=sum(a["near_side"]!=b["near_side"] for a,b in pairs))
        print(phase,{n:round(q["mae_kbbl"],2) for n,q in r["inventory"][phase]["all"]["models"].items()},
              {n:(q["trades"],q["net_170_usd"],q["net_340_usd"]) for n,q in market.items()},flush=True)
    fields = ("release_date","week_ending","study_phase","cutoff","known","actual","previous_release","previous_error_million_bbl",
              "previous_change_million_bbl","four_week_change_million_bbl","latest_lag_label_utc",
              "state_training_windows","latest_state_training_utc","latest_validation_outcome_utc",
              "adaptive_memory_weight","adaptive_scored_weeks",*INVENTORY)
    export(f"{STEM}_forecasts",[{k:q[k] for k in fields} for q in rows])
    export(f"{STEM}_trades",refit)
    export(f"{STEM}_response_trades",shock)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(r,indent=2,allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
