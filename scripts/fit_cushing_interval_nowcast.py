"""Nowcast inventory with timestamp-weighted subtraction of previously reported flow."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize_scalar

from audit_cushing_cl_horizons import paths
from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS, inputs
from fit_cushing_cl_volatility import HURDLE, RISK_FLOOR_USD, risk_inputs, risk_prediction
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed
from fit_cushing_weekly_change_ml import VOLUMES

STEM = "cushing_interval_nowcast"
PHASES = ("development", "validation", "holdout")
METHODS = ("public", "public_level_calibrated", "interval_ml", "unadjusted_interval_ml")
MARKET_METHODS = ("robust_public", "calibrated_public", "with_optical")


def observations():
    forecasts = indexed("cushing_robust_stack_forecasts")
    windows = indexed("eia_release_clock")
    by_week = defaultdict(list)
    for issue, f in sorted(forecasts.items()):
        by_week[f["week_ending"]].append((issue, f))
    geometry = indexed("cushing_coverage_nowcast_forecasts")
    parts = defaultdict(list)
    for stem in VOLUMES:
        for r in read(stem):
            parts[r["release_date"]].append(r)
    for issue, f in sorted(forecasts.items()):
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("forecast_cutoff_utc", "target_known_by_utc"))
        if not cutoff < known:
            raise ValueError(f"Invalid public information clock on {issue}")
        measured = parts[issue]
        area = float(geometry[issue]["area_fraction"])
        anchor = math.fsum(float(r["circular_area_m2"]) for r in measured)/area if measured else None
        original = adjusted = weight = 0.
        ids = set()
        for r in measured:
            if r["candidate_id"] in ids:
                raise ValueError(f"Repeated tank observation on {issue}/{r['candidate_id']}")
            ids.add(r["candidate_id"])
            early, start, late, end, available = (datetime.fromisoformat(r[k]) for k in (
                "earlier_capture_at_utc", "previous_measurement_at_utc", "later_capture_at_utc",
                "current_measurement_at_utc", "archive_available_at_utc"))
            if not early <= start < late <= end or not late <= available < cutoff:
                raise ValueError(f"Invalid optical clocks on {issue}/{r['candidate_id']}")
            early_fraction = (start-early).total_seconds()/(end-start).total_seconds()
            current_fraction = (late-start)/(end-start)
            prior_day = start.date().isoformat()
            source = [q for _, q in by_week[prior_day] if datetime.fromisoformat(q["target_known_by_utc"]) < cutoff]
            prior = float(source[-1]["target"]) if source else float(f["prior"])
            if source and start != datetime.fromisoformat(windows[source[-1]["release_date"]]["latest_week_observed_at_utc"]):
                raise ValueError(f"Different prior week boundary on {issue}")
            share = float(r["circular_area_m2"])/anchor
            raw = float(r["capture_change_kbbl"])/1000
            change_width = (float(r["joint_loss_max_change_kbbl"])-float(r["joint_loss_min_change_kbbl"]))/float(r["circular_kbbl_per_m"])
            reliability = 1/(1+(change_width/4)**2)
            original += raw*reliability
            adjusted += (raw-share*early_fraction*prior)*reliability
            weight += share*current_fraction*reliability
        # A fixed minimum exposure bounds amplification in weeks with sparse observed volume.
        denom = math.sqrt(max(weight, .035))
        baseline = float(f["public_ensemble"])
        yield dict(release_date=issue, week_ending=f["week_ending"], study_phase=f["study_phase"], cutoff=cutoff, known=known,
                   actual=float(f["target"]), public=baseline, measured=bool(measured), area_fraction=area,
                   observed_weight=weight, original_signal=original/denom, interval_signal=adjusted/denom,
                   expected_share=weight*baseline/denom)


def optimum(x, y, weights, *, bound, penalty):
    def objective(value):
        return weights@np.hypot(0.55,y-value*x)+penalty*value**2

    result = minimize_scalar(objective, bounds=bound, method="bounded", options={"xatol":1e-8})
    if not result.success:
        raise ValueError("Calibration failed")
    return float(result.x)


def replay(rows):
    seen = []
    for r in rows:
        history = [q for q in seen if q["known"] < r["cutoff"]][-156:]
        optical = [q for q in history if q["measured"]]
        base = r["public"]
        factor = 0.
        if len(history) >= 52:
            forecast, actual = (np.asarray([q[k] for q in history]) for k in ("public","actual"))
            factor = optimum(forecast,actual-forecast,np.full(len(history),1/len(history)),bound=(-.45,.45),penalty=.12)
        r["public_level_calibrated"] = base*(1+factor)
        correction = {}
        for signal in ("interval_signal", "original_signal"):
            beta = 0.
            if r["measured"] and len(optical) >= 32:
                prior_x = np.asarray([q[signal]-q["expected_share"] for q in optical])
                residual = np.asarray([q["actual"]-q["public_level_calibrated"] for q in optical])
                age = np.asarray([(r["cutoff"]-q["cutoff"]).days/7 for q in optical])
                weights = np.exp2(-age/78)
                beta = optimum(prior_x,residual,weights/weights.sum(),bound=(0.,1.),penalty=.11)
            correction[signal] = float(np.clip(beta*(r[signal]-r["expected_share"]),-.75,.75))
        seen.append(r)
        yield {k:v for k,v in r.items() if k not in ("cutoff", "known")} | dict(
            forecast_cutoff_utc=r["cutoff"].isoformat(), target_known_by_utc=r["known"].isoformat(),
            latest_training_label_utc=max((q["known"] for q in history),default=r["cutoff"]).isoformat() if history else None,
            training_weeks=len(history), training_optical_weeks=len(optical),
            interval_ml=r["public_level_calibrated"]+correction["interval_signal"],
            unadjusted_interval_ml=r["public_level_calibrated"]+correction["original_signal"])


def score(rows):
    y = np.asarray([r["actual"] for r in rows])
    errors = {m:np.abs(y-[r[m] for r in rows]) for m in METHODS}
    return dict(weeks=len(rows), methods={m:dict(mae_kbbl=float(1000*err.mean()),
                      correct_direction=int(np.count_nonzero(y*np.array([r[m] for r in rows])>0)),
                      paired_delta_vs_public_95_kbbl=block_bootstrap_mean(1000*(err-errors["public"]))) for m,err in errors.items()},
                interval_vs_unadjusted_95_kbbl=block_bootstrap_mean(1000*(errors["interval_ml"]-errors["unadjusted_interval_ml"])))


def trading(forecasts):
    by_issue = {r["release_date"]:r for r in forecasts}
    previous = {(r["release_date"],r["first_session_number"]):r for r in read("cushing_cl_execution_windows_trades")
                if r["method"]=="public_volatility" and r["last_session_number"]=="5" and r["first_session_number"] in ("1","2")}
    history = defaultdict(list)
    for r,days,times,moves in paths(list(risk_inputs(list(inputs())))):
        f = by_issue[r["release_date"]]
        target, _, carry, curve_change, *_ = r["x"]
        if target != f["actual"] or not datetime.fromisoformat(f["target_known_by_utc"]) <= r["decision"] < times[0]:
            raise ValueError(f"Different EIA vintage on {r['release_date']}")
        public_surprise = target-f["public_level_calibrated"]
        optical = f["interval_ml"]-f["public_level_calibrated"]
        base = (target,public_surprise,carry,curve_change,public_surprise*carry)
        features = (r["x"][:5],base,(*base,optical,optical*carry))
        for first in (1,2):
            sigma = max(RISK_FLOOR_USD,r["volatility_usd"]*math.sqrt((5-first)/4))
            move = moves[-1]-moves[first-1]
            for method,x in zip(MARKET_METHODS,features,strict=True):
                key = first,method
                current = dict(x=x,volatility_usd=sigma,long_gross=move,exit_at=times[-1])
                train = [q for q in history[key] if q["exit_at"]<r["decision"]][-HISTORY:]
                history[key].append(current)
                if len(train)<MIN_WINDOWS:
                    continue
                expected = risk_prediction(train,current,len(x))
                side = int(expected>HURDLE)-int(expected<-HURDLE)
                gross = side*move
                if method==MARKET_METHODS[0] and gross-HURDLE*abs(side)!=int(previous[r["release_date"],str(first)]["net_170_usd"]):
                    raise ValueError(f"Trading baseline changed on {r['release_date']}")
                yield {k:r[k] for k in ("release_date","study_phase","near_month","far_month","source")} | dict(
                    first_session_number=first,entry_session=days[first-1],exit_session=days[-1],method=method,
                    decision_at_utc=r["decision"].isoformat(),latest_training_exit_at_utc=train[-1]["exit_at"].isoformat(),
                    measured=f["measured"],volatility_usd=sigma,expected_long_gross_usd=expected,near_side=side,gross_usd=gross,
                    **{f"cost_{c}_usd":c*abs(side) for c in COSTS},**{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    rows = list(replay(observations()))
    phases = {p:{scope:score([r for r in rows if r["study_phase"]==p and (scope=="all" or r["measured"])])
                 for scope in ("all", "measured")} for p in PHASES}
    trades = list(trading(rows))
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["first_session_number"],r["method"]].append(r)
    market = {}
    for phase in PHASES:
        market[phase] = {}
        for first in (1,2):
            controls = groups[phase,first,"calibrated_public"]
            hybrid = groups[phase,first,"with_optical"]
            delta = [a["net_170_usd"]-b["net_170_usd"] for a,b in zip(hybrid,controls,strict=True)]
            market[phase][str(first)] = {m:robustness(groups[phase,first,m]) for m in MARKET_METHODS} | dict(
                optical_increment_170_usd=sum(delta),paired_mean_optical_increment_95_usd=block_bootstrap_mean(delta))
    report = dict(status="exploratory_walk_forward_on_previously_examined_history", phases=phases, trading=market)
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    for p,g in phases.items():
        print(p,{scope:{m:round(v["mae_kbbl"],1) for m,v in d["methods"].items()} for scope,d in g.items()})
        print({n:{m:v[m]["net_170_usd"] for m in MARKET_METHODS} for n,v in market[p].items()})


if __name__ == "__main__":
    main()
