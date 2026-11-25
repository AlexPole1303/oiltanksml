"""Replay WPSR balance and relative oil-price corrections for Cushing."""

import argparse
import json
import math
from bisect import bisect_left
from collections import defaultdict
from contextlib import closing
from datetime import date, datetime, timedelta

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_cl_volatility import HURDLE, risk_prediction
from fit_cushing_delayed_calendar import daily_risk, replay as market_calendar
from fit_cushing_nowcast_ensemble import block_bootstrap_mean, transformed
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import forecast_metrics, trading
from fit_cushing_robust_stack import indexed
from publication_clocks import EASTERN

STEM = "cushing_public_state"
OPTICAL = "outer_with_internal_calibration"
STATE = ("cushing_crude","padd2_commercial_crude","padd3_commercial_crude",
         "padd2_crude_refiner_inputs","padd3_crude_refiner_inputs","us_crude_exports",
         "us_crude_net_imports_including_spr","us_gasoline_stocks","us_distillate_stocks")
CANDIDATES = {"balance_public": ("balance", False), "basis_public": ("basis", False),
              "balance_optical": ("balance", True), "basis_optical": ("basis", True),
              "basis_when_unmeasured": ("basis", True), "basis_fit_to_unmeasured": ("basis", True),
              "balance_when_unmeasured": ("balance", True), "price_only_when_unmeasured": ("price_only", True)}
FALLBACK = ("basis_when_unmeasured","basis_fit_to_unmeasured")
ABLATIONS = ("balance_when_unmeasured","price_only_when_unmeasured")
MODELS = (PUBLIC,"carried_scene",OPTICAL,*CANDIDATES)
CONDITIONAL_MODELS = (PUBLIC,OPTICAL,"basis_optical",*FALLBACK,*ABLATIONS)
RELEASE_MODELS = ("regional_release_control","regional_release_with_cushing","regional_release_with_optical")
CONSENSUS_MODELS = ("public_and_hybrid_mean","public_and_hybrid_agreement")


def released_balances():
    clocks = indexed("eia_release_clock")
    values = defaultdict(dict)
    for q in read("eia_control_vintage"):
        if q["metric"] not in STATE:
            continue
        row = values[q["release_date"],q["report_date"]]
        expected_unit = "thousand_bbl_per_day" if q["metric"] in STATE[3:7] else "thousand_bbl"
        value = float(q["value"])/1000
        if q["metric"] in row or q["unit"] != expected_unit or not math.isfinite(value):
            raise ValueError(f"Invalid WPSR observation for {q['release_date']}/{q['metric']}")
        row[q["metric"]] = value
    for issue, q in sorted(clocks.items()):
        week = q["latest_week_in_issue"]
        preceding = (date.fromisoformat(week)-timedelta(days=7)).isoformat()
        now, old = values[issue,week], values[issue,preceding]
        current, previous = (np.array([r[key] for key in STATE]) for r in (now,old))
        current[1] -= current[0]
        previous[1] -= previous[0]
        delta = current - previous
        yield dict(issue=issue,week=week,known=datetime.fromisoformat(q["conservative_postrelease_from_utc"]),
                   current=current,delta=delta)


def spot_reference():
    with closing(read_only(SOURCE)) as db:
        keys = list(db.execute("""SELECT w.date,w.value,b.value FROM eia_series w
            JOIN eia_series b ON w.date=b.date WHERE w.series='wti_spot_daily' AND b.series='brent_spot_daily'
            AND w.value IS NOT NULL AND b.value IS NOT NULL ORDER BY w.date"""))
    days = [d for d,_,_ in keys]
    marks = np.array([(w,b) for _,w,b in keys])
    if days != sorted(set(days)) or not np.isfinite(marks).all():
        raise ValueError("Invalid historical WTI/Brent spot data")
    return days,marks


def state(known, week):
    last = known[-1]
    recent = np.array([r["delta"] for r in known[-4:]])
    old = np.array([r["current"] for r in known[-52:]])
    # Regional throughput relative to its own recent level, stock changes
    # in Cushing's adjacent markets, and the prior report's use of crude.
    delta = last["delta"]
    levels = last["current"][3:7]-np.median(old[:,3:7],axis=0)
    angle = 2*math.pi*date.fromisoformat(week).timetuple().tm_yday/365.25
    return (delta[0],recent[:,0].mean(),delta[1],recent[:,1].mean(),delta[2],recent[:,2].mean(),
            *levels,delta[7],delta[8],math.sin(angle),math.cos(angle))


def basis(cutoff, days, marks):
    day = cutoff.astimezone(EASTERN).date().isoformat()
    # The stored price date is a market date. Leave two subsequent common
    # market sessions behind the cutoff as a publication-lag sensitivity.
    i = bisect_left(days,day)-3
    if i < 63 or days[i+2]>=day:
        raise ValueError(f"Insufficient spot history before {cutoff}")
    spreads = marks[i-62:i+1,0]-marks[i-62:i+1,1]
    median = float(np.median(spreads))
    return (spreads[-1]-median,spreads[-1]-spreads[-6],spreads[-1]-spreads[-21]),days[i],days[i+2]


def source_rows(*, use_archived_basis):
    book = list(released_balances())
    by_issue = {q["issue"]:q for q in book}
    observed = indexed("cushing_outer_capture_nowcast_forecasts")
    days, marks = spot_reference()
    archived = indexed("cushing_spot_vintages") if use_archived_basis else {}
    if use_archived_basis and archived.keys() != observed.keys():
        raise ValueError("The WTI/Brent archive must cover the same dated forecast panel")
    for issue,q in sorted(observed.items()):
        week = q["week_ending"]
        cutoff, known = map(datetime.fromisoformat,(q["cutoff_utc"],q["known_utc"]))
        prior = [r for r in book if r["known"] < cutoff and r["week"] < week]
        target = float(q["actual"])
        raw = by_issue[issue]
        if (not cutoff<known or not prior or raw["week"]!=week or raw["known"]!=known
                or not math.isclose(target,raw["delta"][0],abs_tol=1e-10)
                or any(datetime.fromisoformat(v)>=cutoff for k,v in q.items() if k.endswith("_latest_label_utc") and v)):
            raise ValueError(f"Inconsistent target or optical cutoff on {issue}")
        baseline = state(prior,week)
        relative, spot_day, lag_session = basis(cutoff,days,marks)
        vintage_date = None
        if use_archived_basis:
            vintage = archived[issue]
            expected_latest = (cutoff.astimezone(EASTERN).date()-timedelta(days=1)).isoformat()
            if (datetime.fromisoformat(vintage["cutoff"]) != cutoff or vintage["vintage_date"] != expected_latest
                    or not vintage["archived_reference_date"] <= vintage["stored_reference_date"] == spot_day):
                raise ValueError(f"Archived spot timing differs on {issue}")
            relative = tuple(float(vintage[f"archived_basis_{k}"]) for k in ("level","move_5","move_20"))
            spot_day, vintage_date = vintage["archived_reference_date"],vintage["vintage_date"]
        yield {k:q[k] for k in ("release_date","week_ending","study_phase")} | {
            k:float(q[k]) for k in (PUBLIC,"carried_scene",OPTICAL)} | dict(
            cutoff=cutoff,known=known,actual=target,measured=int(q["baseline_roofs"])+int(q["outer_roofs"])>0,
            balance=baseline,basis=(*baseline,*relative),price_only=relative,last_report=prior[-1]["issue"],
            last_report_known=prior[-1]["known"].isoformat(),spot_reference_day=spot_day,
            spot_lag_session=lag_session,spot_vintage_date=vintage_date)


def adjust(train, row, key, *, emphasize_unmeasured=False):
    if len(train) < 52:
        return 0.
    x,z = transformed(train,row,key)
    residual = np.array([q["actual"]-q[PUBLIC] for q in train])
    age = np.array([(row["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in train])
    decay = np.exp2(-age/52.)
    if emphasize_unmeasured:
        # Retain part of the available history so isolated runs of cloudy
        # weeks cannot determine a separate high-variance forecast.
        decay *= np.where([q["measured"] for q in train],.35,1.)
    penalty = np.eye(x.shape[1])*decay.sum()
    beta = np.zeros(x.shape[1])
    # Smooth weighted absolute error, regularized around the existing model.
    for _ in range(6):
        weight = decay/np.hypot(1.,(residual-x@beta)/.5)
        beta = solve(x.T@(weight[:,None]*x)+penalty,x.T@(weight*residual),assume_a="pos")
    return float(np.clip(z@beta,-.6,.6))


def forecasts(*, use_archived_basis):
    history = []
    for row in source_rows(use_archived_basis=use_archived_basis):
        train = [q for q in history if q["known"] < row["cutoff"]][-156:]
        corrections = {k:adjust(train,row,k) for k in ("balance","basis")}
        values = {m:row[OPTICAL if use_optical else PUBLIC]+corrections[k]
                  for m,(k,use_optical) in CANDIDATES.items() if m not in (*FALLBACK,*ABLATIONS)}
        values.update({name:row[OPTICAL] for name in (*FALLBACK,*ABLATIONS)})
        if not row["measured"]:
            values[FALLBACK[0]] += corrections["basis"]
            values[FALLBACK[1]] += adjust(train,row,"basis",emphasize_unmeasured=True)
            values[ABLATIONS[0]] += corrections["balance"]
            values[ABLATIONS[1]] += adjust(train,row,"price_only")
        history.append(row)
        yield row | values | dict(training_weeks=len(train),
            latest_training_label_utc=train[-1]["known"].isoformat() if train else None)


def conditional_strategies(rows, base):
    book = {q["release_date"]:q for q in rows}
    release_book = list(released_balances())
    current_release = {q["issue"]:q for q in release_book}
    original = {(q["method"],q["release_date"]):q for q in base}
    anchor = {q["release_date"]:q for q in read("cushing_cl_conditional_response_trades") if q["method"]=="inventory_conditional"}
    marks = {(q["release_date"],int(q["session_number"])):datetime.fromisoformat(q["settlement_reference_time"])
             for q in read("wti_cushing_event_bars") if q["leg"]=="near"}
    past = defaultdict(list)
    _,_,cases = market_calendar()
    for r in cases:
        issue, decision = r["release_date"],r["decision"]
        f = book[issue]
        v = current_release[issue]
        if not f["cutoff"] < f["known"] == v["known"] <= decision < r["entry"] or f["actual"]!=r["x"][0]:
            raise ValueError(f"Wrong EIA clock or outcome on {issue}")
        _,_,carry,change,*_ = r["x"]
        volatility = math.log(r["volatility_usd"]/1000.)
        # Regional release features add the concurrent change outside the
        # delivery hub, and changes in refinery inputs in the two regions.
        preceding = [q["delta"] for q in release_book if q["week"]<v["week"] and q["known"]<f["cutoff"]][-8:]
        shock = v["delta"]-np.median(preceding,axis=0)
        context = (carry,change,volatility,carry*volatility,change*volatility,
                   shock[1],shock[2],shock[3],shock[4])
        def release_vector(expected):
            surprise = f["actual"]-expected
            return (*context,f["actual"],surprise,surprise*carry,f["actual"]*volatility,surprise*volatility)
        vectors = (context,release_vector(f[PUBLIC]),release_vector(f["basis_when_unmeasured"]))
        design = r | dict(vectors=vectors)
        eligible = [q for q in past["release"] if q["exit_at"]<decision][-HISTORY:]
        past["release"].append(design)
        if issue not in anchor:
            continue
        enriched = r | dict(stamps=[marks[issue,n] for n in range(2,6)],
            next_issue=None,next_report_clock=None,next_report_clock_basis=None)
        if not r["decision"] < enriched["stamps"][0] == r["entry"] < enriched["stamps"][-1] == r["exit_at"]:
            raise ValueError(f"Invalid settlement calendar on {issue}")
        point_estimates = {}
        label_clocks = {}
        for name in CONDITIONAL_MODELS:
            existing = original[name,issue]
            if any(existing[k]!=r[k] for k in ("near_month","far_month")) or any(
                datetime.fromisoformat(existing[k])!=r[field] for k,field in (
                    ("decision_at_utc","decision"),("entry_at_utc","entry"),("exit_at_utc","exit_at")
                )) or datetime.fromisoformat(existing["settled_training_label_utc"])>=decision:
                raise ValueError(f"Base quotes or training clock differ on {issue}/{name}")
            response = features(carry,r["volatility_usd"],f["actual"]-f[name])
            value = float(existing["expected_long_move_usd"])
            current = r | dict(response_features=response,base_estimate=value)
            train = [q for q in past[name] if q["exit_at"]<decision][-HISTORY:]
            past[name].append(current)
            correction,_ = conditional_residual(train,current) if len(train)>=MIN_WINDOWS else (0.,0.)
            point_estimates[name] = value+correction
            label_clocks[name] = (datetime.fromisoformat(existing["settled_training_label_utc"]),
                                 train[-1]["exit_at"] if train else None)
            booked = trade(enriched,f"conditional_{name}",point_estimates[name],3,
                training_at=label_clocks[name][0],calibration_at=label_clocks[name][1])
            if name==PUBLIC and (booked["net_170_usd"]!=int(anchor[issue]["net_170_usd"]) or not math.isclose(
                    booked["expected_long_move_usd"],float(anchor[issue]["expected_long_move_usd"]),abs_tol=1e-8)):
                raise ValueError(f"Original conditional rule changed on {issue}")
            yield booked
        # Require a shared direction or average both forecasts without
        # selecting a model using that week's realized spread move.
        a,b = (point_estimates[k] for k in (PUBLIC,FALLBACK[0]))
        average = (a+b)/2
        agrees = (a>HURDLE and b>HURDLE) or (a< -HURDLE and b< -HURDLE)
        clocks = [label_clocks[k] for k in (PUBLIC,FALLBACK[0])]
        for name, point in zip(CONSENSUS_MODELS,(average,average if agrees else 0.),strict=True):
            yield trade(enriched,name,point,3,training_at=max(x[0] for x in clocks),
                        calibration_at=max((x[1] for x in clocks if x[1]),default=None))
        for i,(name,x) in enumerate(zip(RELEASE_MODELS,vectors,strict=True)):
            value = risk_prediction([q | dict(x=q["vectors"][i]) for q in eligible],r | dict(x=x),len(x))
            yield trade(enriched,name,value,3,training_at=eligible[-1]["exit_at"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spot-lag-proxy",action="store_true",help="Use end-of-history prices to reproduce the initial diagnostic")
    args = parser.parse_args()
    stem = f"{STEM}_proxy" if args.spot_lag_proxy else STEM
    with threadpool_limits(limits=1):
        rows = list(forecasts(use_archived_basis=not args.spot_lag_proxy))
        trades = list(trading(rows,MODELS))
        conditional = list(conditional_strategies(rows,trades))
    report = dict(status="exploratory_on_reviewed_history",
        spot_source="historical_lag_proxy" if args.spot_lag_proxy else "prior_day_alfred_vintages",
        forecasts={},market={},market_conditional={})
    for phase in PHASES:
        group = [r for r in rows if r["study_phase"]==phase]
        result = report["forecasts"][phase] = {}
        for scope,subset in (("all",group),("no_optical",[r for r in group if not r["measured"]]),
                             ("with_optical",[r for r in group if r["measured"]])):
            result[scope] = forecast_metrics(subset,MODELS)
            for m,(_,use_optical) in CANDIDATES.items():
                ref = OPTICAL if use_optical else PUBLIC
                difference = [1000*(abs(r[m]-r["actual"])-abs(r[ref]-r["actual"])) for r in subset]
                result[scope]["models"][m]["paired_delta_vs_matched_95_kbbl"] = block_bootstrap_mean(difference)
        print(phase,{k:round(v["mae_kbbl"],3) for k,v in result["all"]["models"].items()})
        for output,base_name,ledger in (("market",PUBLIC,trades),("market_conditional","conditional_"+PUBLIC,conditional)):
            stats = report[output][phase] = {}
            by_method = defaultdict(list)
            for r in ledger:
                if r["study_phase"]==phase:
                    by_method[r["method"]].append(r)
            base = by_method[base_name]
            for m,subset in by_method.items():
                if [r["release_date"] for r in subset]!=[r["release_date"] for r in base]:
                    raise ValueError(f"Different calendars on {phase}/{m}")
                delta = [r["net_170_usd"]-q["net_170_usd"] for r,q in zip(subset,base,strict=True)]
                stats[m] = robustness(subset) | daily_risk(subset) | dict(
                    net_340_usd=sum(r["gross_usd"]-340*abs(r["near_side"]) for r in subset),
                    paired_delta_vs_public_95_usd=block_bootstrap_mean(delta))
            print(output,{k:v["net_170_usd"] for k,v in stats.items()})
    export(f"{stem}_forecasts",[{k:v for k,v in r.items() if k not in ("balance","basis","price_only")} for r in rows])
    export(f"{stem}_trades",trades)
    export(f"{stem}_conditional_trades",conditional)
    write(DERIVED/f"{stem}_summary.json",(json.dumps(report,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    main()
