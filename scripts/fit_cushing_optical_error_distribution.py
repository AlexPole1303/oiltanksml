"""Weight an optical revision by the conditional distribution of past forecast errors."""

import argparse
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export
from build_terminal_development_measurements import BARREL_M3
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_direction import STEM as DIRECTION
from fit_cushing_optical_value import STEM as OPTICAL_VALUE
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_robust_stack import indexed
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from replay_cushing_weighted_inventory_market import replay_weighted_market

STEM = "cushing_optical_error_distribution"
HEIGHT = "cushing_height_response_curve"
PASSES = "cushing_overpass_covariance"
POOLED = "pooled_distribution"
CONDITIONAL = "conditional_distribution"
SOFT = "conditional_shrinkage"
CURRENT = "conditional_current_only"
METHODS = (SUPPLEMENT, POOLED, CONDITIONAL, SOFT)
TRADE_METHODS = (*METHODS, CURRENT)
CHANNELS = ("market", "inventory")
SCALES = np.array((.75, .20, .65, .20))
GAINS = np.arange(0., 1.51, .25)
UNIT_GAIN = int(np.flatnonzero(GAINS == 1.)[0])
MIN_WEEKS = 40


def inputs():
    end, optical, passes, tallies = (indexed(f"{name}_forecasts") for name in (HEIGHT, OPTICAL_VALUE, PASSES, OUTER))
    for issue, h in sorted(end.items()):
        optical_row, pass_row, o = (book[issue] for book in (optical, passes, tallies))
        cutoff, known = (datetime.fromisoformat(h[k]) for k in ("cutoff", "known"))
        actual = float(h["actual"])
        if (not cutoff < known or any(datetime.fromisoformat(q[ck]) != cutoff or datetime.fromisoformat(q[kn]) != known
                for q,ck,kn in ((optical_row, "cutoff", "known"), (pass_row, "cutoff", "known"), (o, "cutoff_utc", "known_utc")))
                or any(float(q["actual"]) != actual or any(q[k] != h[k] for k in ("week_ending", "study_phase"))
                       for q in (optical_row, pass_row, o))
                or any(q[k] and datetime.fromisoformat(q[k]) >= cutoff for q,k in ((h, "latest_height_label"),
                    (optical_row, "latest_training_outcome_utc"), (pass_row, "last_optical_label"), (o, f"{SUPPLEMENT}_latest_label_utc")))
                or float(h[SUPPLEMENT]) != float(o[SUPPLEMENT]) or int(h["measured_roofs"]) != int(pass_row["measured_roofs"])):
            raise ValueError(f"The inventory histories disagree on {issue}")
        market_point = float(o[SUPPLEMENT])
        market_public = float(o[PUBLIC])
        inventory_point = float(h["long_response_supported_by_labels"])
        # Keep the long-history public/market correction fixed. The added
        # pass covariance and height deltas each belong to current optics.
        optical_delta = float(optical_row["optical_update"]) + inventory_point - float(h["long_outer_supplement"])
        optical_delta += float(pass_row["same_week_shared_pass"]) - market_point
        area = float(pass_row["area_fraction"])
        support = float(pass_row["same_week_shared_pass_posterior_scale_mbbl"])
        for name, base, point, count in (("market", market_public, market_point, int(h["measured_roofs"])),
                ("inventory", inventory_point - optical_delta, inventory_point, max(int(h["measured_roofs"]),int(optical_row["optical_roofs"])))):
            if count == 0 and not math.isclose(base, point, abs_tol=1e-10, rel_tol=0):
                raise ValueError(f"The optical-free branch still contains a satellite correction on {issue}")
            yield {k:h[k] for k in ("release_date", "week_ending", "study_phase")} | dict(channel=name,
                cutoff=cutoff, known=known, actual=actual, baseline=point, unmeasured=base, area_fraction=area,
                posterior_scale=support, roofs=count)


def covariates(row):
    revision = row["baseline"] - row["unmeasured"]
    sign = math.copysign(1., revision)
    return np.array((sign * row["unmeasured"], abs(revision), math.log1p(row["area_fraction"] / .04), row["posterior_scale"]))


def estimates(row, history):
    cutoff, base, initial = (row[k] for k in ("cutoff", "unmeasured", "baseline"))
    update = initial - base
    previous = [q for q in history if q["known"] < cutoff and abs(q["baseline"] - q["unmeasured"]) > 1e-10][-156:]
    result = dict.fromkeys(METHODS, initial)
    details = dict(pooled_gain=1., conditional_gain=1., shrinkage_gain=1., probability_improves=None, pooled_probability_improves=None,
                   conditional_effective_weeks=None, latest_training_outcome=previous[-1]["known"] if previous else None,
                   training_weeks=len(previous))
    if abs(update) <= 1e-10 or len(previous) < MIN_WEEKS:
        return result, details
    sign = math.copysign(1., update)
    # Compare errors in a common signed direction. A reversal is helpful
    # only when the true error exceeds half of the proposed adjustment.
    residuals = np.array([math.copysign(1., q["baseline"] - q["unmeasured"]) * (q["actual"] - q["unmeasured"])
                          for q in previous])
    age = np.array([(cutoff - q["known"]).total_seconds() / 604800 for q in previous])
    recent = np.exp2(-age / 104)
    distances = (np.stack([covariates(q) for q in previous]) - covariates(row)) / SCALES
    local = recent * (.30 + .70 * np.exp(-.5 * np.square(distances).mean(axis=1)))
    gains, losses = [], []
    # Smooth absolute error over the historical distribution, penalizing
    # deviations from the original revision. A flat objective favors gain 1.
    cost = np.hypot(residuals[:,None] - abs(update) * GAINS, .06)
    for weight in (recent, local):
        risk = weight @ cost / weight.sum() + .008 * abs(update) / .20 * (GAINS - 1) ** 2
        best = int(np.argmin(risk))
        gains.append(float(GAINS[best]))
        losses.append(risk)
    soft = 1. + float(np.clip((losses[1][UNIT_GAIN] - losses[1].min()) / .020, 0., 1.)) * (gains[1] - 1.)
    result.update({m:base + g * abs(update) * sign for m,g in zip(METHODS[1:], (*gains,soft), strict=True)})
    details.update(pooled_gain=gains[0], conditional_gain=gains[1], shrinkage_gain=soft,
                   probability_improves=float(local @ (residuals > abs(update) / 2.) / local.sum()),
                   pooled_probability_improves=float(recent @ (residuals > abs(update) / 2.) / recent.sum()),
                   conditional_effective_weeks=float(local.sum()**2 / (local @ local)))
    return result, details


def forecasts(source):
    histories = {name:[] for name in CHANNELS}
    for row in source:
        past = histories[row["channel"]]
        result, details = estimates(row, past)
        past.append(row)
        update = row["baseline"] - row["unmeasured"]
        yield row | result | details | dict(update=update, agreement=update * row["unmeasured"] > 0,
            improves=abs(row["baseline"] - row["actual"]) < abs(row["unmeasured"] - row["actual"]))


def metrics(group):
    if not group:
        return {}
    actual = np.array([r["actual"] for r in group])
    errors = {m:1000 * abs(np.array([r[m] for r in group]) - actual) for m in ("unmeasured", *METHODS)}
    return {m:dict(weeks=len(e), mae_kbbl=float(e.mean()), delta_vs_original_95_kbbl=block_bootstrap_mean(e - errors[SUPPLEMENT]))
            for m,e in errors.items()}


def summary(rows, trades):
    phases = {}
    for phase in PHASES:
        inventory = {}
        for name in CHANNELS:
            group = [r for r in rows if (r["channel"],r["study_phase"]) == (name,phase)]
            sample = [r for r in group if abs(r["update"]) > 1e-10]
            probabilities = [r for r in sample if r["probability_improves"] is not None]
            groups = {"all":group, "measured":sample, "agrees":[r for r in sample if r["agreement"]],
                      "disagrees":[r for r in sample if not r["agreement"]]}
            brier = [float((r["probability_improves"] - r["improves"])**2) for r in probabilities]
            pooled_brier = [float((r["pooled_probability_improves"] - r["improves"])**2) for r in probabilities]
            subgroup_deltas = {key:block_bootstrap_mean([1000 * (abs(r["baseline"] - r["actual"])
                - abs(r["unmeasured"] - r["actual"])) for r in groups[key]]) for key in ("measured", "agrees", "disagrees")}
            inventory[name] = {k:metrics(q) for k,q in groups.items()} | dict(probability_weeks=len(brier),
                clustered_brier=float(np.mean(brier)) if brier else None,
                pooled_brier=float(np.mean(pooled_brier)) if pooled_brier else None,
                paired_brier_delta_95=block_bootstrap_mean(np.array(brier) - pooled_brier),
                brier_at_half=.25 if brier else None,
                original_optical_delta_95_kbbl=subgroup_deltas,
                original_wins={k:sum(r["improves"] for r in q) for k,q in groups.items()})
        pnl = {}
        all_methods = (*TRADE_METHODS, *(f"{m}_conditional_exit" for m in TRADE_METHODS))
        groups = {m:[r for r in trades if (r["study_phase"],r["method"]) == (phase,m)] for m in all_methods}
        for name, group in groups.items():
            control = f"{SUPPLEMENT}_conditional_exit" if name.endswith("_conditional_exit") else SUPPLEMENT
            pairs = list(zip(group, groups[control], strict=True))
            if any(a["release_date"] != b["release_date"] for a,b in pairs):
                raise ValueError(f"Different release calendar for {phase}/{name}")
            pnl[name] = robustness(group) | daily_risk(group) | dict(net_340_usd=sum(r["net_340_usd"] for r in group),
                changed_trades=sum(bool(a["near_side"] != b["near_side"] or a["near_side"] and a["exit_at_utc"] != b["exit_at_utc"]) for a,b in pairs),
                delta_95_usd=block_bootstrap_mean([a["net_170_usd"] - b["net_170_usd"] for a,b in pairs]))
        phases[phase] = dict(inventory=inventory, trading=pnl)
        print(phase, {m:round(q["mae_kbbl"],3) for m,q in inventory["inventory"]["measured"].items()},
                  {m:v["net_170_usd"] for m,v in pnl.items()})
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def forward(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    registration = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    original = json.loads(original_path.read_bytes())
    files = {m:DERIVED / f"{m}_{issue}_registration.json" for m in (HEIGHT,PASSES,DIRECTION)}
    h,p,d = (json.loads(q.read_bytes()) for q in files.values())
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    for q in (h,p,d):
        checked_hashes(q["sources_sha256"])
        if any(q[k] != original[k] for k in LINEAGE) or q["measured_roofs"] != h["measured_roofs"]:
            raise ValueError("The forward forecast snapshots use different dates or roof sets")
    if not registration.exists():
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The model must be frozen before the report")
        market_public = original["public_ensemble_kbbl"] / 1000
        market_point = h["estimates_kbbl"][SUPPLEMENT] / 1000
        inventory_point = h["long_estimates_kbbl"]["response_supported_by_labels"] / 1000
        raw = d["channels"]["inventory"]["estimates_kbbl"]
        # This matches the historical component construction at the frozen
        # panel, including both the pass and height refinements.
        pass_adjusted = p["estimates_kbbl"]["same_week_shared_pass"] / 1000
        delta = (raw["same_week"] - raw["no_revision"] + h["estimates_kbbl"]["response_supported_by_labels"]
                 - h["estimates_kbbl"][SUPPLEMENT]) / 1000 + pass_adjusted - market_point
        recovered_path = original_path.with_name(f"{original_path.stem}_recovered_optical.json")
        native_path = DERIVED / f"cushing_forward_native_optical_{issue}_summary.json"
        recovery, native = (json.loads(q.read_bytes()) for q in (recovered_path,native_path))
        physical = original["measured_pairs"] + recovery["recovered_roofs"]
        factor = float(native["dated_roof_area_m2"])
        fraction = math.fsum(r["volume_per_m"] for r in physical) * BARREL_M3 * 1e6 / factor
        if (len(physical) != h["measured_roofs"] or len({q["candidate_id"] for q in physical}) != len(physical)
                or any(datetime.fromisoformat(q.get("available_utc", q.get("source_available_utc"))) >= cutoff for q in physical)
                or not 0 < fraction <= 1.):
            raise ValueError("The registered spatial sample does not reproduce")
        history = list(inputs())
        projections = {}
        for name,base,point in (("market", market_public, market_point), ("inventory", inventory_point - delta, inventory_point)):
            values, detail = estimates(dict(cutoff=cutoff, unmeasured=base, baseline=point, area_fraction=fraction,
                posterior_scale=p["posterior_scale_kbbl"]["same_week_shared_pass"] / 1000), [q for q in history if q["channel"] == name])
            latest = detail["latest_training_outcome"]
            detail["latest_training_outcome"] = latest.isoformat() if latest else None
            projections[name] = dict(estimates_kbbl={m:1000 * v for m,v in values.items()}, **detail)
        sources = (original_path, recovered_path, native_path, *files.values(), Path(__file__).resolve(),
                   *(DERIVED / f"{m}_forecasts.csv" for m in (HEIGHT,PASSES,OPTICAL_VALUE,OUTER,STEM)))
        stamp = datetime.now(timezone.utc)
        if not cutoff <= stamp < deadline:
            raise ValueError("The forecast registration missed the report deadline")
        freeze(registration, {k:original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(),
            area_fraction=fraction, measured_roofs=h["measured_roofs"], projections=projections,
            sources_sha256={str(q.relative_to(ROOT)):digest(q) for q in sources}))
    frozen = json.loads(registration.read_bytes())
    checked_hashes(frozen["sources_sha256"])
    if (any(frozen[k] != original[k] for k in LINEAGE) or set(frozen["projections"]) != set(CHANNELS)
            or not cutoff <= datetime.fromisoformat(frozen["recorded_at_utc"]) < deadline
            or any(v["latest_training_outcome"] and datetime.fromisoformat(v["latest_training_outcome"]) >= cutoff
                   or set(v["estimates_kbbl"]) != set(METHODS)
                   for v in frozen["projections"].values())):
        raise ValueError("The saved model registration does not reproduce")
    target = score_release(original_path, original)
    score = dict(status=target["status"], registration_sha256=digest(registration),
                 estimates_kbbl={k:q["estimates_kbbl"] for k,q in frozen["projections"].items()})
    if target["status"] == "scored_first_release":
        actual = target["actual_eia_change_kbbl"]
        score.update(actual_kbbl=actual, first_release_sha256=target["first_release_sha256"],
                     errors_kbbl={k:{m:abs(v - actual) for m,v in q.items()} for k,q in score["estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(score, indent=2, allow_nan=False) + "\n").encode())
    print(score)


def main():
    with threadpool_limits(limits=1):
        rows = list(forecasts(inputs()))
        market = {r["release_date"]:r | {CURRENT:r[CONDITIONAL]} for r in rows if r["channel"] == "market"}
        trades = list(replay_weighted_market(market, TRADE_METHODS, reference=SUPPLEMENT, historical_inputs={CURRENT:SUPPLEMENT}))
    result = summary(rows,trades)
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    forward(args.release.isoformat()) if args.release else main()
