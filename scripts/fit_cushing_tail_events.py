"""Replay probabilities of large inventory moves and their incremental calendar-spread P&L."""

import json
import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, PHASES, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as delayed_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_prerelease_cl import cases as early_cases
from fit_cushing_robust_stack import indexed
from fit_cushing_volatility_interactions import features

STEM = "cushing_tail_events"
TAIL = 1.0  # Million barrels, common to both decision clocks.
PROBABILITY_METHODS = ("historical_frequency", "empirical_residual", "public_ml", "optical_ml", "lagged_optical_ml")
TRADE_METHODS = ("curve", "inventory", "public_tail", "optical_tail", "lagged_optical_tail")
EVENTS = ("large_draw", "normal", "large_build")


def category(value):
    return int(value>=TAIL)-int(value<=-TAIL)+1


def forecast_cases(early):
    public = indexed("cushing_robust_stack_forecasts")
    observation = indexed("cushing_observation_operator_forecasts")
    last_measured = None
    for issue,o in sorted(observation.items()):
        f = public[issue]
        cutoff,known = (datetime.fromisoformat(o[k]) for k in ("forecast_cutoff_utc","target_known_by_utc"))
        if not cutoff < known or float(f["target"]) != float(o["actual"]):
            raise ValueError(f"Inconsistent publication clock or inventory label on {issue}")
        level = float(o["public_ensemble"])
        measured = int(o["measured_roofs"])
        optical = (float(o["correlated_observations"])-level,math.log1p(float(o["correlated_information"])))
        lagged = last_measured[1] if measured and last_measured and cutoff-last_measured[0]<=timedelta(days=14) else (0.,0.)
        yield dict(clock="report_day", release_date=issue, phase=o["study_phase"], cutoff=cutoff, known=known,
            actual=float(o["actual"]), point=level, measured=measured,
            public=(level,float(f["robust_public"]),float(f["prior"])),optical=optical,lagged=lagged)
        if measured:
            last_measured = cutoff,optical
    last_measured = None
    for r in early:
        measured,cutoff = r["measured_roofs"],r["decision"]
        optical = (r["early_optical_ml"],math.log1p(r["early_optical_information"]))
        lagged = last_measured[1] if measured and last_measured and cutoff-last_measured[0]<=timedelta(days=14) else (0.,0.)
        yield dict(clock="previous_session", release_date=r["issue"], phase=r["study_phase"], cutoff=r["decision"],
            known=r["known"], actual=r["inventory"], point=r["early_public_ml"], measured=measured,
            public=(r["early_public_ml"],r["preliminary_inventory"],r["inventory_scale"]),
            optical=optical,lagged=lagged)
        if measured:
            last_measured = cutoff,optical


def probabilities(rows):
    history = defaultdict(list)
    for r in rows:
        past = [q for q in history[r["clock"]] if q["known"]<r["cutoff"]][-156:]
        history[r["clock"]].append(r)
        if len(past)<52:
            continue
        weights = np.exp2(-np.arange(len(past)-1,-1,-1)/52)
        weights *= len(weights)/weights.sum()
        y = np.asarray([category(q["actual"]) for q in past])
        residual_events = [category(r["point"]+q["actual"]-q["point"]) for q in past]
        result = []
        for labels in (y,residual_events):
            counts = np.bincount(labels,weights=weights,minlength=3)+1
            result.append(counts/counts.sum())
        for extra in (None,"optical","lagged"):
            x,z = (np.asarray([(*q["public"],*(q[extra] if extra else ())) for q in group]) for group in (past,[r]))
            center = np.average(x,axis=0,weights=weights)
            scale = np.sqrt(np.average((x-center)**2,axis=0,weights=weights))
            # Sparse optical observations keep their physical scale rather than receiving
            # a large standardized coefficient just because only a few weeks were measured.
            scale = np.maximum(scale,.25)
            model = LogisticRegression(C=.1,max_iter=500).fit((x-center)/scale,y,sample_weight=weights)
            p = np.full(3,1e-6)
            p[model.classes_] = model.predict_proba((z-center)/scale)[0]
            result.append(p/p.sum())
        yield r | dict(last_training_label=max(q["known"] for q in past),training_releases=len(past),
                       predictions=dict(zip(PROBABILITY_METHODS,result,strict=True)))


def score(rows):
    y = np.asarray([category(r["actual"]) for r in rows])
    selected = np.eye(3)[y]
    report = dict(weeks=len(rows), outcomes=dict(zip(EVENTS,np.bincount(y,minlength=3).tolist(),strict=True)), methods={})
    base = np.array([r["predictions"]["public_ml"] for r in rows])
    for method in PROBABILITY_METHODS:
        p = np.array([r["predictions"][method] for r in rows])
        brier = np.sum((p-selected)**2,axis=1)
        report["methods"][method] = dict(brier=float(np.mean(brier)),
            brier_delta_vs_public_95=block_bootstrap_mean(brier-np.sum((base-selected)**2,axis=1)),
            log_loss=float(-np.log(p[np.arange(len(y)),y]).mean()),
            mean_probabilities=dict(zip(EVENTS,p.mean(axis=0).tolist(),strict=True)),
            average_precision={EVENTS[i]:float(average_precision_score(y==i,p[:,i])) if (y==i).any() else None
                                          for i in (0,2)})
    return report


def market_replay(forecasts, early, delayed):
    by_issue = {(r["clock"],r["release_date"]):r for r in forecasts}
    for clock,cases in (("previous_session",early),("report_day",delayed)):
        seen = []
        for r in cases:
            is_early = clock=="previous_session"
            date = r["issue"] if is_early else r["release_date"]
            f = by_issue.get((clock,date))
            if not f:
                continue
            if not f["last_training_label"]<f["cutoff"]<=r["decision"]<r["entry"]<r["exit_at"]:
                raise ValueError(f"Price and forecast clocks do not align on {clock}/{date}")
            if is_early:
                curve,inventory = r["design"][0],r["design"][4]
                if not r["exit_at"]<f["known"] or r["inventory"]!=f["actual"]:
                    raise ValueError(f"Inconsistent early forecast or mark on {date}")
            else:
                _,_,_,inventory,_,curve = features(r)
                if not f["known"]<=r["decision"] or r["x"][0]!=f["actual"]:
                    raise ValueError(f"Inconsistent post-release information on {date}")
            designs = [curve,inventory]
            for method in PROBABILITY_METHODS[2:]:
                p = f["predictions"][method]
                expected_tail = p[2]-p[0]
                signed_tail = expected_tail if is_early else category(f["actual"])-1-expected_tail
                designs.append((*inventory,signed_tail,float(p[0]+p[2]),signed_tail*math.log(r["volatility_usd"]/1000)))
            train = [q for q in seen if q["exit_at"]<r["decision"]][-HISTORY:]
            current = r | dict(designs=designs)
            seen.append(current)
            if len(train)<MIN_WINDOWS:
                continue
            for i,(method,x) in enumerate(zip(TRADE_METHODS,designs,strict=True)):
                mean = risk_prediction([q | dict(x=q["designs"][i]) for q in train],current | dict(x=x),len(x))
                side = int(mean>HURDLE)-int(mean<-HURDLE)
                gross = side*r["long_gross"]
                trade = dict(clock=clock,release_date=date,study_phase=r["study_phase"],method=method,
                    decision_at=r["decision"].isoformat(),entry_at=r["entry"].isoformat(),exit_at=r["exit_at"].isoformat(),
                    latest_training_exit=train[-1]["exit_at"].isoformat(),near_month=r["near_month"],far_month=r["far_month"],
                    expected_move_usd=mean,near_side=side,gross_usd=gross,measured_roofs=f["measured"],
                    **{f"cost_{c}_usd":c*abs(side) for c in COSTS},
                    **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})
                if not is_early:
                    trade.update(alternate_source=r["alternate_source"],
                        alternate_gross_usd=side*r["alternate_long_gross_usd"] if r["alternate_source"] else None,
                        **{f"session_{n}_net_170_usd":side*v-(170 if n==5 else 85)*abs(side)
                           for n,v in enumerate(r["spread_path_usd"],2)})
                yield trade


def main():
    early,_ = early_cases()
    _,_,delayed = delayed_calendar()
    forecasts = list(probabilities(forecast_cases(early)))
    trades = list(market_replay(forecasts,early,delayed))
    report = dict(status="exploratory_replay_on_reviewed_history",large_event_million_bbl=TAIL,probabilities={},trading={})
    for clock in ("previous_session","report_day"):
        report["probabilities"][clock] = {}
        report["trading"][clock] = {}
        for phase in PHASES:
            group = [r for r in forecasts if r["clock"]==clock and r["phase"]==phase]
            report["probabilities"][clock][phase] = dict(all=score(group), measured=score([r for r in group if r["measured"]]))
            group = [r for r in trades if r["clock"]==clock and r["study_phase"]==phase]
            base = [r for r in group if r["method"]=="public_tail"]
            trading = report["trading"][clock][phase] = {}
            for method in TRADE_METHODS:
                matched = [r for r in group if r["method"]==method]
                if [r["release_date"] for r in matched]!=[r["release_date"] for r in base]:
                    raise ValueError(f"Unequal trading calendars in {clock}/{phase}")
                delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(matched,base,strict=True)]
                risk = daily_risk(matched) if clock=="report_day" else {}
                trading[method] = robustness(matched) | risk | dict(delta_vs_public_tail_usd=sum(delta),
                    delta_per_release_95_usd=block_bootstrap_mean(delta))
            print(clock,phase,{m:(v["windows"],v["net_170_usd"]) for m,v in trading.items()})
    export(f"{STEM}_forecasts", [{k:r[k] for k in ("clock","release_date","phase","actual","point","measured","training_releases")} |
        dict(cutoff_at=r["cutoff"].isoformat(),known_at=r["known"].isoformat(),last_training_label=r["last_training_label"].isoformat(),
             **{f"{method}_{event}":float(value) for method,p in r["predictions"].items() for event,value in zip(EVENTS,p,strict=True)})
        for r in forecasts])
    export(f"{STEM}_trades",[dict(session_2_net_170_usd=None,session_3_net_170_usd=None,session_4_net_170_usd=None,
        session_5_net_170_usd=None,alternate_source=None,alternate_gross_usd=None) | r for r in trades])
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
