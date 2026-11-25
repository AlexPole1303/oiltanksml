"""Estimate seasonal forecast bias using only previously published inventory outcomes."""

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_extended_market_history import STEM as EXTENDED
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_terminal_consensus import STEM as TERMINALS
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_inventory_seasonal_residual"
BASE = "long_median_and_regional"
MODELS = (BASE, "pooled_bias", "seasonal_bias", "seasonal_and_coverage_bias")
MARKET_REF = "original_satellite"
WINDOW = 260
MAX_SHIFT = .25
HALF_LIFE_WEEKS = 156.
SEASON_BANDWIDTH_WEEKS = 6.


def inputs():
    geometry = indexed(f"{TERMINALS}_forecasts")
    optics = indexed(f"{OUTER}_forecasts")
    for issue, f in sorted(indexed(f"{EXTENDED}_inventory").items()):
        q, o = geometry[issue], optics[issue]
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("cutoff", "known"))
        if (not cutoff < known or cutoff != datetime.fromisoformat(q["cutoff"]) or known != datetime.fromisoformat(q["known"])
                or cutoff != datetime.fromisoformat(o["cutoff_utc"]) or known != datetime.fromisoformat(o["known_utc"])
                or any(r[k] != f[k] for r in (q, o) for k in ("week_ending", "study_phase"))
                or any(float(r["actual"]) != float(f["actual"]) for r in (q, o))
                or any(stamp and datetime.fromisoformat(stamp) >= cutoff for stamp in (
                    f["latest_training_release"], q["latest_calibration_label_utc"], o[f"{SUPPLEMENT}_latest_label_utc"]))):
            raise ValueError(f"Misaligned vintage on {issue}")
        yield {k:f[k] for k in ("release_date", "week_ending", "study_phase")} | dict(cutoff=cutoff, known=known,
            actual=float(f["actual"]), observed_terminals=int(q["terminals"]), **{BASE:float(f[BASE]), SUPPLEMENT:float(o[SUPPLEMENT])})


def week_angle(day):
    dt = date.fromisoformat(day)
    start, stop = date(dt.year,1,1), date(dt.year+1,1,1)
    return math.tau * (dt-start).days/(stop-start).days


def centre(errors, weights):
    def objective(offset):
        # Smooth absolute loss and a prior at the unmodified baseline.
        return float((weights @ np.hypot(errors-offset,.2) + 20. * offset**2)/weights.sum())

    fit = minimize_scalar(objective, method="bounded", bounds=(-MAX_SHIFT,MAX_SHIFT))
    if not fit.success or not math.isfinite(fit.x):
        raise ValueError("The residual calibration failed")
    return float(fit.x)


def estimates(history, current):
    prior = [r for r in history if r["known"] < current["cutoff"]][-WINDOW:]
    base = current[BASE]
    if len(prior) < MIN_WINDOWS:
        return dict.fromkeys(MODELS,base), prior
    errors = np.array([r["actual"]-r[BASE] for r in prior])
    age = np.array([(current["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in prior])
    decay = np.exp2(-age/HALF_LIFE_WEEKS)
    angle = week_angle(current["week_ending"])
    cos_dist = np.cos(np.array([week_angle(r["week_ending"]) for r in prior])-angle)
    seasonal = .3+.7*np.exp(-(1-cos_dist)/(math.tau*SEASON_BANDWIDTH_WEEKS/52)**2)
    coverage = np.where([(r["observed_terminals"]>0)==(current["observed_terminals"]>0) for r in prior],1.,.5)
    weights = (decay,decay*seasonal,decay*seasonal*coverage)
    result = {m:base+centre(errors,w) for m,w in zip(MODELS[1:],weights,strict=True)}
    return {BASE:base,**result}, prior


def replay_inventory():
    history = []
    for r in inputs():
        models, past = estimates(history,r)
        history.append(r)
        yield r | models | dict(training_weeks=len(past),latest_training_outcome=past[-1]["known"] if past else None)


def replay_market(inventory):
    book = {r["release_date"]:r for r in inventory}
    anchors = {r["release_date"]:r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    history = defaultdict(list)
    for case,_ in market_inputs():
        issue = case["release_date"]
        r, decision = book[issue], case["decision"]
        if not r["cutoff"] < r["known"] <= decision < case["entry"] or r["actual"] != case["x"][0]:
            raise ValueError(f"Bad forecast clock on {issue}")
        for method in (MARKET_REF,*MODELS):
            expected = r[SUPPLEMENT if method==MARKET_REF else method]
            current = case | dict(x=replace_expectation(case,expected), response_features=features(
                case["x"][2],case["volatility_usd"],r["actual"]-expected))
            eligible = [q for q in history[method] if q["exit_at"] < decision][-HISTORY:]
            history[method].append(current)
            if len(eligible)<MIN_WINDOWS:
                continue
            point = volatility_forecast(eligible,current)
            residuals = [q for q in eligible if "base_estimate" in q]
            delta,_ = conditional_residual(residuals,current) if len(residuals)>=MIN_WINDOWS else (0.,0.)
            current["base_estimate"] = point
            row = trade(case, method, point+delta,3,training_at=eligible[-1]["exit_at"],
                        calibration_at=residuals[-1]["exit_at"] if residuals else None)
            if method == MARKET_REF:
                a = anchors[issue]
                if (any(a[k]!=row[k] for k in ("near_month","far_month","entry_at_utc","exit_at_utc"))
                        or any(int(a[k])!=row[k] for k in ("near_side","gross_usd","net_170_usd"))
                        or not math.isclose(point+delta,float(a["expected_long_move_usd"]),abs_tol=1e-8,rel_tol=0)):
                    raise ValueError(f"Historical leader differs on {issue}")
            yield row | dict(inventory_expectation_kbbl=1000*expected, realized_long_move_usd=case["long_gross"],
                             net_340_usd=row["gross_usd"]-340*abs(row["near_side"]))


def report(inventory,trades):
    groups = defaultdict(list)
    for r in trades:
        groups[r["study_phase"],r["method"]].append(r)
    result = dict(status="exploratory_on_reviewed_history",phases={})
    for phase in PHASES:
        rows = [r for r in inventory if r["study_phase"]==phase]
        forecasts = {}
        for name, g in (("all",rows),("measured",[r for r in rows if r["observed_terminals"]]),
                        ("unmeasured",[r for r in rows if not r["observed_terminals"]])):
            errors = {m:np.array([abs(r[m]-r["actual"])*1000 for r in g]) for m in (SUPPLEMENT,*MODELS)}
            forecasts[name] = {m:dict(releases=len(e),mae_kbbl=float(e.mean()),
                 change_vs_base_95_kbbl=block_bootstrap_mean(e-errors[BASE])) for m,e in errors.items() if len(e)}
        strategies = {}
        controls = groups[phase, MARKET_REF]
        for m in (MARKET_REF,*MODELS):
            arm = groups[phase,m]
            pair = list(zip(arm,controls,strict=True))
            if any(r["release_date"]!=q["release_date"] for r,q in pair):
                raise ValueError(f"Nonmatching trade calendars for {phase}/{m}")
            differences = [r["net_170_usd"]-q["net_170_usd"] for r,q in pair]
            strategies[m] = robustness(arm) | daily_risk(arm) | dict(net_340_usd=sum(r["net_340_usd"] for r in arm),
                paired_net_change_per_window_95_usd=block_bootstrap_mean(differences),
                changed_sides=sum(r["near_side"]!=q["near_side"] for r,q in pair))
        result["phases"][phase] = dict(inventory=forecasts,trading=strategies)
        print(phase,{m:round(v["mae_kbbl"],2) for m,v in forecasts["all"].items()},
              {m:(v["trades"],v["net_170_usd"],v["net_340_usd"]) for m,v in strategies.items()})
    return result


def forward(issue):
    prefix = DERIVED/f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED/f"cushing_prerelease_forecast_{issue}.json"
    extended_path = DERIVED/f"{EXTENDED}_{issue}_registration.json"
    terminal_path = DERIVED/f"{TERMINALS}_{issue}_registration.json"
    original, extended, terminal = (json.loads(p.read_bytes()) for p in (original_path,extended_path,terminal_path))
    cutoff,deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(q[k] != original[k] for q in (extended,terminal) for k in LINEAGE) or extended["measured_roofs"]!=terminal["measured_roofs"]:
        raise ValueError("The forward input dates or coverage differ")
    for q in (extended,terminal):
        checked_hashes(q["sources_sha256"])
    if not path.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The forecast must be registered before publication")
        current = dict(cutoff=cutoff,week_ending=original["week_ending"],observed_terminals=terminal["terminals"],
                       **{BASE:extended["estimates_kbbl"][BASE]/1000})
        values, past = estimates(list(inputs()),current)
        stamp = datetime.now(timezone.utc)
        if not past[-1]["known"] < cutoff <= stamp < deadline:
            raise ValueError("The forward model used a later outcome or missed its deadline")
        sources=(original_path,extended_path,terminal_path,Path(__file__).resolve(),DERIVED/f"{EXTENDED}_inventory.csv",
                 DERIVED/f"{TERMINALS}_forecasts.csv",DERIVED/f"{OUTER}_forecasts.csv")
        freeze(path,{k:original[k] for k in LINEAGE}|dict(recorded_at_utc=stamp.isoformat(),
            estimates_kbbl={m:1000*v for m,v in values.items()}, latest_training_outcome=past[-1]["known"].isoformat(),
            training_weeks=len(past), measured_roofs=extended["measured_roofs"],
            sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    if (not datetime.fromisoformat(record["latest_training_outcome"]) < cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline
            or any(record[k]!=original[k] for k in LINEAGE) or set(record["estimates_kbbl"])!=set(MODELS)):
        raise ValueError("The saved seasonal forecast failed its date or model checks")
    outcome = score_release(original_path,original)
    score = dict(release_date=issue,status=outcome["status"],registration_sha256=digest(path),estimates_kbbl=record["estimates_kbbl"])
    if score["status"]=="scored_first_release":
        actual=outcome["actual_eia_change_kbbl"]
        score.update(actual_eia_change_kbbl=actual,first_release_sha256=outcome["first_release_sha256"],
                     absolute_error_kbbl={m:abs(v-actual) for m,v in record["estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"),(json.dumps(score,indent=2,allow_nan=False)+"\n").encode())
    print(json.dumps(score,indent=2))


def main():
    inventory = list(replay_inventory())
    with threadpool_limits(limits=1):
        trades = list(replay_market(inventory))
    result = report(inventory,trades)
    export(f"{STEM}_forecasts",inventory)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(result,indent=2,allow_nan=False)+"\n").encode())


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release",type=date.fromisoformat)
    args = parser.parse_args()
    if args.release:
        forward(args.release.isoformat())
    else:
        main()
