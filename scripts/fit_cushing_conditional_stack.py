"""Measure whether a causal forecast stack can reject unhelpful satellite updates."""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from backtest_wti_cushing_spread import statistics
from build_s1_reference_matches import DERIVED, export
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_robust_stack import indexed
from fit_cushing_weekly_change_ml import trade_rows

STEM = "cushing_conditional_stack"
PUBLIC = ("public_ensemble", "public_plus_monthly", "robust_public")
OPTICAL = ("pair_quality_ml", "coverage_ml", "terminal_coverage_ml")
MODELS = ("public_ensemble", "conditional_public", "conditional_optical")
PHASES = ("development", "validation", "holdout")
HALF_LIFE_WEEKS = 78.


def input_rows():
    sources = [indexed(name) for name in ("cushing_coverage_nowcast_forecasts", "cushing_flow_extensions_forecasts",
                                         "cushing_robust_stack_forecasts")]
    if any(book.keys()!=sources[0].keys() for book in sources[1:]):
        raise ValueError("Different release cohorts in stack inputs")
    for issue in sorted(sources[0]):
        satellite,flows,stack = [book[issue] for book in sources]
        fields = ("week_ending","forecast_cutoff_utc","target_known_by_utc","target","public_ensemble")
        if any(source[k]!=satellite[k] for source in (flows,stack) for k in fields):
            raise ValueError(f"Different forecast vintage on {issue}")
        f = satellite | flows | stack
        cutoff,known = map(datetime.fromisoformat,(f["forecast_cutoff_utc"],f["target_known_by_utc"]))
        label_clocks = [satellite["latest_training_label_utc"],flows["latest_training_label_utc"],stack["last_training_label_utc"]]
        if not cutoff<known or any(datetime.fromisoformat(t)>=cutoff for t in label_clocks if t):
            raise ValueError(f"Noncausal precursor forecast on {issue}")
        baseline = float(f["public_ensemble"])
        measured = satellite["measured"]=="True"
        optical = np.array([float(f[k])-baseline for k in OPTICAL])
        if not measured and np.any(optical):
            raise ValueError(f"Unmeasured release has an optical adjustment on {issue}")
        yield dict(release_date=issue,week_ending=f["week_ending"],study_phase=f["study_phase"],cutoff=cutoff,known=known,
                   measured=measured,space_time_fraction=float(satellite["space_time_fraction"]),
                   target=float(f["target"]),public_ensemble=baseline,public=np.array([float(f[k]) for k in PUBLIC]),
                   optical=optical)


def fit(rows, now, feature, *, penalty, localized=False):
    values = np.array([r[feature] for r in rows])
    baseline = np.array([r["public_ensemble"] if feature=="public" else r["conditional_public"] for r in rows])
    if feature=="public":
        values = values-baseline[:,None]
    y = np.array([r["target"] for r in rows])-baseline
    age = np.array([(now["cutoff"]-r["cutoff"]).total_seconds()/604800 for r in rows])
    weights = np.exp2(-age/HALF_LIFE_WEEKS)
    if localized:
        cover = np.array([r["space_time_fraction"] for r in rows])
        distance = np.log1p(cover/0.04)-np.log1p(now["space_time_fraction"]/0.04)
        weights *= 0.35+0.65*np.exp(-distance**2/0.5)
    weights /= weights.sum()

    def objective(beta):
        residual = values@beta-y
        scale = 0.35
        hyp = np.hypot(1,residual/scale)
        return weights@((hyp-1)*scale**2)+penalty*(beta@beta), values.T@(weights*residual/hyp)+2*penalty*beta

    result = minimize(objective,np.zeros(values.shape[1]),jac=True,method="SLSQP",bounds=[(0.,1.)]*values.shape[1],
                      constraints={"type":"ineq","fun":lambda beta:1-beta.sum(),"jac":lambda beta:-np.ones(len(beta))},
                      options={"ftol":1e-10,"maxiter":100})
    if not result.success or not np.isfinite(result.x).all() or result.x.sum()>1.00000001:
        raise ValueError(f"Conditional forecast weights failed on {now['release_date']}: {result.message}")
    return result.x


def replay():
    seen = []
    for r in input_rows():
        available = [q for q in seen if q["known"]<r["cutoff"]][-156:]
        measured = [q for q in available if q["measured"]]
        base = r["public_ensemble"]
        public_weights = fit(available,r,"public",penalty=0.012) if len(available)>=52 else np.zeros(len(PUBLIC))
        optical_weights = fit(measured,r,"optical",penalty=0.006,localized=True) if r["measured"] and len(measured)>=32 else np.zeros(len(OPTICAL))
        r["conditional_public"] = base+float((r["public"]-base)@public_weights)
        hybrid = r["conditional_public"]+float(r["optical"]@optical_weights)
        seen.append(r)
        yield {k:r[k] for k in ("release_date","week_ending","study_phase","target","measured","space_time_fraction","public_ensemble",
                                "conditional_public")} | dict(conditional_optical=hybrid,forecast_cutoff_utc=r["cutoff"].isoformat(),
            target_known_by_utc=r["known"].isoformat(),latest_training_label_utc=available[-1]["known"].isoformat() if available else None,
            training_weeks=len(available),training_measured_weeks=len(measured),
            public_weights_json=json.dumps(dict(zip(PUBLIC,map(float,public_weights),strict=True))),
            optical_weights_json=json.dumps(dict(zip(OPTICAL,map(float,optical_weights),strict=True))))


def score(rows):
    y = np.array([r["target"] for r in rows])
    predictions = {method:np.array([r[method] for r in rows]) for method in MODELS}
    errors = {method:np.abs(y-estimate) for method,estimate in predictions.items()}
    return dict(weeks=len(rows),metrics={m:dict(mae_million_bbl=float(errors[m].mean()),direction=int(np.count_nonzero(y*p>0)),
                                              mae_delta_ci_95=block_bootstrap_mean(errors[m]-errors["public_ensemble"]))
                                          for m,p in predictions.items()},
                optical_vs_conditional_public_mae_delta_ci_95=block_bootstrap_mean(errors["conditional_optical"]-errors["conditional_public"]))


def main():
    rows = list(replay())
    trades = list(trade_rows(rows,methods=MODELS))
    groups = defaultdict(list)
    for r in trades:
        groups[r["scope"],r["method"]].append(r)
    summary = dict(status="exploratory_replay_on_previously_examined_history",
        phases={p:{scope:score([r for r in rows if r["study_phase"]==p and (scope=="all" or r["measured"])])
                   for scope in ("all","measured")} for p in PHASES},
        surprise_trades={f"{scope}:{method}":statistics(sample) for (scope,method),sample in sorted(groups.items())})
    export(f"{STEM}_forecasts",rows)
    export(f"{STEM}_trades",trades)
    write(DERIVED/f"{STEM}_summary.json",(json.dumps(summary,indent=2,allow_nan=False)+"\n").encode())
    for p,g in summary["phases"].items():
        print(p,{k:(v["weeks"],{m:round(x["mae_million_bbl"],4) for m,x in v["metrics"].items()}) for k,v in g.items()})


if __name__=="__main__":
    main()
