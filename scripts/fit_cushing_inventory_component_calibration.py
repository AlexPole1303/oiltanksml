"""Test separate gains for optical and non-optical inventory corrections."""

import json
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import trading as base_trades
from fit_cushing_optical_value import BASE, STEM as OPTICAL
from fit_cushing_outer_capture_nowcast import SUPPLEMENT

STEM = "cushing_inventory_component_calibration"
DESIGNS = {"component_gain": (2, False), "coverage_component_gain": (2, True),
           "coverage_component_and_public_gain": (3, True)}
MODELS = (PUBLIC, SUPPLEMENT, BASE, *DESIGNS)


def gains(sample, row, width, condition_on_coverage):
    x = np.stack([r["components"][:width] for r in sample])
    y = np.array([r["actual"] - r[BASE] for r in sample])
    age = np.array([(row["cutoff"] - r["cutoff"]).total_seconds() / 604800 for r in sample])
    weights = np.exp2(-age / 78)
    if condition_on_coverage:
        weights *= np.array([1. if r["measured"] == row["measured"] else .5 for r in sample])
    weights /= weights.sum()
    penalty = np.array((.02, .02, .12))[:width]

    def loss(delta):
        residual = x @ delta - y
        smooth_absolute = np.hypot(residual, .15)
        return (weights @ smooth_absolute + penalty @ np.square(delta),
                x.T @ (weights * residual / smooth_absolute) + 2 * penalty * delta)

    fit = minimize(loss, np.zeros(width), jac=True, bounds=[(-.5, .5)] * width, method="L-BFGS-B")
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError(f"Inventory component calibration failed on {row['release_date']}: {fit.message}")
    return fit.x


def inventory():
    history = []
    for f in sorted(read(f"{OPTICAL}_forecasts"), key=lambda q: q["release_date"]):
        issue = f["release_date"]
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("cutoff", "known"))
        if not cutoff < known or (f["latest_training_outcome_utc"]
                                  and datetime.fromisoformat(f["latest_training_outcome_utc"]) >= cutoff):
            raise ValueError(f"Invalid inventory vintage on {issue}")
        public, satellite, base = (float(f[m]) for m in MODELS[:3])
        optical = float(f["optical_update"])
        other = float(f["remove_current_optical"]) - public
        measured = int(f["optical_roofs"]) > 0
        if not np.isclose(public + optical + other, base, atol=1e-10, rtol=0) or (not measured and abs(optical) > 1e-10):
            raise ValueError(f"The dated inventory components do not reconcile on {issue}")
        row = {k: f[k] for k in ("release_date", "study_phase", "week_ending")} | dict(
            cutoff=cutoff, known=known, actual=float(f["actual"]), measured=measured, components=np.array((optical, other, public)),
            **{PUBLIC: public, SUPPLEMENT: satellite, BASE: base})
        prior = [r for r in history if r["known"] < cutoff][-HISTORY:]
        history.append(row)
        fitted = {m: gains(prior, row, width, local) if len(prior) >= MIN_WINDOWS else np.zeros(width)
                  for m, (width, local) in DESIGNS.items()}
        yield row | {m: base + float(row["components"][:len(c)] @ c) for m, c in fitted.items()} | dict(
            latest_training_outcome_utc=max((r["known"] for r in prior), default=None), train_weeks=len(prior),
            optical_revision_million_bbl=optical, other_revision_million_bbl=other,
            gain_adjustments_json=json.dumps({m: c.tolist() for m, c in fitted.items()}, separators=(",", ":")))


def score_inventory(rows):
    loss = {m: np.array([1000 * abs(r[m] - r["actual"]) for r in rows]) for m in MODELS}
    return dict(weeks=len(rows), models={m: dict(mae_kbbl=float(e.mean()),
        delta_vs_reference_95_kbbl=block_bootstrap_mean(e - loss[BASE])) for m, e in loss.items()})


def main():
    with threadpool_limits(limits=1):
        rows = list(inventory())
        market = list(replay_conditional_market(rows, base_trades(rows, MODELS), MODELS,
                      references={PUBLIC: f"{PUBLIC}_conditional", SUPPLEMENT: f"{SUPPLEMENT}_conditional"}))
    groups = defaultdict(list)
    for r in market:
        groups[r["study_phase"], r["method"]].append(r)
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        panel = [r for r in rows if r["study_phase"] == phase]
        reference = groups[phase, BASE]
        trading = {}
        for m in MODELS:
            group = groups[phase, m]
            if [r["release_date"] for r in group] != [r["release_date"] for r in reference]:
                raise ValueError(f"Different trading windows in {phase}/{m}")
            delta = [r["net_170_usd"] - b["net_170_usd"] for r, b in zip(group, reference, strict=True)]
            trading[m] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(
                r["gross_usd"] - 340 * abs(r["near_side"]) for r in group), paired_delta_vs_reference_95_usd=block_bootstrap_mean(delta),
                changed_entries=sum(r["near_side"] != b["near_side"] for r, b in zip(group, reference, strict=True)))
        all_weeks = score_inventory(panel)
        report["phases"][phase] = dict(all=all_weeks, measured=score_inventory([r for r in panel if r["measured"]]),
                                      unmeasured=score_inventory([r for r in panel if not r["measured"]]), trading=trading)
        print(phase, {m: round(q["mae_kbbl"], 2) for m, q in all_weeks["models"].items()},
              {m: (q["trades"], q["net_170_usd"], q["net_340_usd"]) for m, q in trading.items()}, flush=True)
    export(f"{STEM}_forecasts", [{k: v for k, v in r.items() if k != "components"} for r in rows])
    export(f"{STEM}_trades", market)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    main()
