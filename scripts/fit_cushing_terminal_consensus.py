"""Test breadth of independently measured terminals as an inventory and CL feature."""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_extended_market_history import STEM as EXTENDED
from fit_cushing_joint_late_pairs import source_rows
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_value import STEM as OPTICAL_VALUE, median_gain
from fit_cushing_outer_capture_nowcast import SUPPLEMENT
from fit_cushing_robust_stack import indexed
from fit_cushing_shared_scene import PRIOR_INVENTORY_SIGMA_MBBL, SCENE_SIGMA_M, parts, scene_design
from fit_cushing_storage_state import fit_correction
from fit_cushing_volatility_neighbors import STEM as WEIGHTED
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_terminal_consensus"
BASE = "long_median_and_regional"
ALT = ("breadth_blend", "learned_breadth", "learned_concentration", "breadth_weighted_optical")
FORECASTS = (PUBLIC, SUPPLEMENT, BASE, *ALT)
POLICIES = ("original_satellite", "terminal_contrast", "terminal_contrast_and_surprise")


def contrast_of(stats):
    return stats["median"] - stats["local_information_mean"] if stats else 0.


def terminal_signals(row):
    obs = row["observed"]
    if not obs:
        return [], dict()
    s, residual, noise = parts(obs, row[PUBLIC])
    tags = np.array([q["terminal_id"] for q in obs])
    scenes = sorted({q[k] for q in obs for k in ("scene_before", "scene_after")})
    design = scene_design(obs, {n: i for i, n in enumerate(scenes)})
    cov = noise + SCENE_SIGMA_M**2 * (design @ design.T)
    # Retain uncertainty about oil moved after the reporting cutoff in a
    # pair with one later image. The same latent change affects these roofs.
    post = np.array([q.get("post_fraction", 0.) * q.get("mapped_hub_fraction", 0.) for q in obs])
    cov += PRIOR_INVENTORY_SIGMA_MBBL**2 * np.outer(post, post)
    data = np.clip(residual, -3 * np.sqrt(cov.diagonal()), 3 * np.sqrt(cov.diagonal()))
    signals = []
    for terminal in sorted(set(tags)):
        mask = tags == terminal
        local = cov[np.ix_(mask, mask)]
        sens = s[mask]
        projected = cho_solve(cho_factor(local), sens)
        info = float(sens @ projected)
        if not math.isfinite(info) or info <= 0:
            raise ValueError(f"Invalid local optical support on {row['release_date']}/{terminal}")
        estimate = float(np.clip(projected @ data[mask] / (info + 1 / PRIOR_INVENTORY_SIGMA_MBBL**2), -1.25, 1.25))
        signals.append(dict(terminal_id=terminal, roofs=int(mask.sum()), information=info, correction_mbbl=estimate,
                            area_time_share=float(sens.sum())))
    values = np.array([q["correction_mbbl"] for q in signals])
    info = np.array([q["information"] for q in signals])
    mass = info / info.sum()
    weights = np.sqrt(info)
    weights /= weights.sum()
    # Keep support for weakly observed terminals small. The weighted median
    # preserves direction voted by multiple sites without a large outlier.
    idx = np.argsort(values, kind="stable")
    median = float(values[idx[np.searchsorted(np.cumsum(weights[idx]), .5)]])
    information_mean = float(mass @ values)
    top = int(np.argmax(info))
    rest = float(np.average(np.delete(values, top), weights=np.delete(info, top))) if len(info) > 1 else information_mean
    return signals, dict(effective_terminals=float(1. / (mass @ mass)), top_terminal_share=float(mass.max()),
        local_information_mean=information_mean, median=median, dominant_vs_rest=float(values[top]) - rest,
        sign_agreement=float(abs(weights @ np.sign(values))), local_rms_dispersion=float(np.sqrt(mass @ (values - information_mean)**2)))


def inventory(terminal_rows):
    longer = indexed(f"{EXTENDED}_inventory")
    optical = indexed(f"{OPTICAL_VALUE}_forecasts")
    seen = []
    for r, _, _, _ in source_rows():
        issue, cutoff, known = (r[k] for k in ("release_date", "cutoff", "known"))
        anchor = longer[issue]
        optical_row = optical[issue]
        if (datetime.fromisoformat(anchor["cutoff"]) != cutoff or datetime.fromisoformat(anchor["known"]) != known
                or float(anchor["actual"]) != r["actual"] or anchor["week_ending"] != r["week_ending"]
                or datetime.fromisoformat(optical_row["cutoff"]) != cutoff or datetime.fromisoformat(optical_row["known"]) != known
                or float(optical_row["actual"]) != r["actual"]
                or anchor["latest_training_release"] and datetime.fromisoformat(anchor["latest_training_release"]) >= cutoff):
            raise ValueError(f"Different weekly training cutoff on {issue}")
        base = float(anchor[BASE])
        signals, stats = terminal_signals(r)
        for q in signals:
            terminal_rows.append(dict(release_date=issue, cutoff=cutoff, **q))
        count = len(signals)
        contrast = contrast_of(stats)
        concentration = stats.get("dominant_vs_rest", 0.) * stats.get("top_terminal_share", 0.) if count > 1 else 0.
        agreement = stats.get("sign_agreement", 0.)
        revision = float(optical_row["optical_update"])
        row = dict(cutoff=cutoff, known=known, actual=r["actual"], contrast=contrast, concentration=concentration,
                   short_design=(contrast, contrast * agreement), full_design=(contrast, contrast * agreement, concentration),
                   residual=r["actual"] - base, optical_update=revision, remove_current_optical=base - revision,
                   diagnostics=np.array((math.log1p(stats.get("effective_terminals", 0.)), stats.get("top_terminal_share", 0.),
                       agreement, math.log1p(sum(q["information"] for q in signals)), abs(revision)/.25, abs(base)/.75,
                       stats.get("local_rms_dispersion", 0.)/.15, float(count <= 2))))
        past = [q for q in seen if q["known"] < cutoff][-HISTORY:]
        if count > 1:
            seen.append(row)
        short, full = (fit_correction(past, row, design, "residual", penalty=penalty, loss_scale=.65, bound=.25)
                       for design, penalty in (("short_design", (.16, .35)), ("full_design", (.16, .35, .55))))
        gain_sample = [q for q in past if abs(q["optical_update"]) > 1e-10][-104:]
        gain = median_gain(gain_sample, row, conditional=True) if count > 1 and abs(revision) > 1e-10 and len(gain_sample) >= 40 else 1.
        results = (base + .5 * contrast, base + short, base + full, base + (gain - 1) * revision)
        yield {k: r[k] for k in ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual", PUBLIC, SUPPLEMENT)} | {
            BASE: base, **dict(zip(ALT, results, strict=True))} | dict(terminals=count, contrast=contrast, concentration=concentration,
            effective_terminals=stats.get("effective_terminals", 0.), dominant_terminal_share=stats.get("top_terminal_share", 0.),
            direction_agreement=agreement, local_rms_dispersion=stats.get("local_rms_dispersion", 0.),
            optical_update=revision, learned_optical_weight=gain, calibration_weeks=len(past),
            latest_calibration_label_utc=past[-1]["known"] if past else None)


def market(rows):
    f = {r["release_date"]: r for r in rows}
    anchor = {r["release_date"]: r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    histories = defaultdict(list)
    for r, _ in inputs():
        issue, decision = r["release_date"], r["decision"]
        a = anchor.get(issue)
        if a is None:
            continue
        q = f[issue]
        if (not q["cutoff"] < q["known"] <= decision < r["entry"] < r["exit_at"] or q["actual"] != r["x"][0]
                or any(a[k] != r[k] for k in ("near_month", "far_month", "study_phase"))
                or any(datetime.fromisoformat(a[k]) != r[field] for k, field in (
                    ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))):
            raise ValueError(f"Inconsistent trading window on {issue}")
        train_at = datetime.fromisoformat(a["latest_training_exit_utc"])
        cal_at = datetime.fromisoformat(a["latest_calibration_exit_utc"]) if a["latest_calibration_exit_utc"] else None
        if train_at >= decision or cal_at and cal_at >= decision:
            raise ValueError(f"The original market rule has an unsettled label on {issue}")
        sigma, point = r["volatility_usd"], float(a["expected_long_move_usd"])
        contrast, concentration = q["contrast"], q["concentration"]
        surprise = float(np.clip(q["actual"] - q[SUPPLEMENT], -2., 2.))
        designs = ((contrast, concentration), (contrast, concentration, surprise * (1 - q["direction_agreement"])))
        for i, method in enumerate(POLICIES):
            history = histories[method]
            train = [x for x in history if x["exit_at"] < decision][-HISTORY:]
            correction = 0.
            if i and q["terminals"] >= 2:
                values = designs[i - 1]
                current = dict(cutoff=decision, design=values, exit_at=r["exit_at"], error=(r["long_gross"] - point) / sigma)
                correction = fit_correction(train, current, "design", "error", penalty=(.2, .4, .9)[:len(values)],
                                            loss_scale=1., bound=.30)
                history.append(current)
            t = trade(r, method, point + sigma * correction, 3,
                      training_at=max(train_at, train[-1]["exit_at"]) if train else train_at, calibration_at=cal_at)
            if not i and any(t[k] != int(a[k]) for k in ("near_side", "gross_usd", "net_170_usd")):
                raise ValueError(f"The original trading position differs on {issue}")
            yield t | dict(terminals=q["terminals"], terminal_signal_training_windows=len(train),
                          net_340_usd=t["gross_usd"] - 340 * abs(t["near_side"]))


def forecast_scores(rows):
    error = {m: np.array([1000 * abs(r[m] - r["actual"]) for r in rows]) for m in FORECASTS}
    return dict(windows=len(rows), models={m: dict(mae_kbbl=float(v.mean()), delta_vs_base_95_kbbl=block_bootstrap_mean(v - error[BASE]))
                                          for m, v in error.items()})


def forward(issue, *, scoring):
    path = DERIVED / f"{STEM}_{issue}_registration.json"
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    recovered_path = DERIVED / f"{original_path.stem}_recovered_optical.json"
    baseline_path = DERIVED / f"{EXTENDED}_{issue}_registration.json"
    original, recovered, base = (json.loads(p.read_bytes()) for p in (original_path, recovered_path, baseline_path))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(q[k] != original[k] for q in (recovered, base) for k in LINEAGE):
        raise ValueError("The frozen observations use different release windows")
    checked_hashes(base["sources_sha256"])
    if not path.exists():
        if scoring or not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The forecast must be written before publication")
        checked_hashes({str((DERIVED / k).relative_to(ROOT)): h for k, h in recovered["source_sha256"].items()})
        observations = original["measured_pairs"] + recovered["recovered_roofs"]
        if (len(observations) != base["measured_roofs"] or len({r["candidate_id"] for r in observations}) != len(observations)
                or any(datetime.fromisoformat(q.get("available_utc", q.get("source_available_utc"))) >= cutoff for q in observations)):
            raise ValueError("The forward terminal measurements are duplicated or unavailable")
        signals, stats = terminal_signals(dict(release_date=issue, observed=observations,
                                              **{PUBLIC: original["public_ensemble_kbbl"] / 1000}))
        estimate = base["estimates_kbbl"][BASE]
        stamp = datetime.now(timezone.utc)
        if not cutoff <= stamp < deadline:
            raise ValueError("The forecast crossed its registration deadline")
        evidence = {str(p.relative_to(ROOT)): digest(p) for p in (original_path, recovered_path, baseline_path,
                    Path(__file__).resolve(), ROOT / "scripts/fit_cushing_shared_scene.py")}
        freeze(path, {k: original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(),
            measured_roofs=len(observations), terminals=len(signals), terminal_diagnostics=stats, terminal_signals=signals,
            estimates_kbbl={BASE: estimate, ALT[0]: estimate + 500 * contrast_of(stats)}, sources_sha256=evidence))
    record = json.loads(path.read_bytes())
    if any(record[k] != original[k] for k in LINEAGE) or not cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline:
        raise ValueError("The terminal breadth registration is inconsistent")
    checked_hashes(record["sources_sha256"])
    if not scoring:
        return record
    outcome = score_release(original_path, original)
    result = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path), estimates_kbbl=record["estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_errors_kbbl={m: abs(actual - v) for m, v in record["estimates_kbbl"].items()})
    write(path.with_name(f"{STEM}_{issue}_score.json"), (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    return result


def main():
    terminal_rows = []
    with threadpool_limits(limits=1):
        rows = list(inventory(terminal_rows))
        trades = list(market(rows))
    report = dict(status="exploratory_on_reviewed_history", accuracy={}, trades={})
    for phase in PHASES:
        panel = [r for r in rows if r["study_phase"] == phase]
        multi = [r for r in panel if r["terminals"] >= 2]
        report["accuracy"][phase] = {key: forecast_scores(sample) for key, sample in (
            ("all", panel), ("at_least_two_terminals", multi)) if sample}
        report["accuracy"][phase]["coverage"] = dict(
            median_effective_terminals=float(np.median([r["effective_terminals"] for r in multi])) if multi else None,
            dominated_windows=sum(r["dominant_terminal_share"] > .5 for r in multi),
            learned_weight_changed_windows=sum(abs(r["learned_optical_weight"] - 1.) > 1e-9 for r in multi))
        selected = {m: [r for r in trades if r["study_phase"] == phase and r["method"] == m] for m in POLICIES}
        control = selected[POLICIES[0]]
        report["trades"][phase] = result = {}
        for method, sample in selected.items():
            if [r["release_date"] for r in sample] != [r["release_date"] for r in control]:
                raise ValueError(f"Different calendar on {phase}/{method}")
            pairs = list(zip(sample, control, strict=True))
            result[method] = robustness(sample) | daily_risk(sample) | dict(net_340_usd=sum(r["net_340_usd"] for r in sample),
                paired_pnl_delta_95_usd=block_bootstrap_mean([r["net_170_usd"] - a["net_170_usd"] for r, a in pairs]),
                changed_side=sum(r["near_side"] != a["near_side"] for r, a in pairs))
        print(phase, {m: round(q["mae_kbbl"], 2) for m, q in report["accuracy"][phase]["all"]["models"].items()},
              {m: (q["trades"], q["net_170_usd"], q["net_340_usd"]) for m, q in result.items()})
    export(f"{STEM}_forecasts", rows)
    export(f"{STEM}_terminals", terminal_rows)
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    if args.score and not args.release:
        parser.error("--score requires --release")
    if args.release:
        value = forward(args.release, scoring=args.score)
        print(json.dumps({k: v for k, v in value.items() if k not in ("terminal_signals", "sources_sha256")}, indent=2))
    else:
        main()
