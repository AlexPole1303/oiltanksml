"""Compare inventory tail probabilities with the same point and release clocks."""

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from build_s1_reference_matches import DERIVED, export, read
from collect_public import ROOT, write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_extended_market_history import STEM as EXTENDED
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_value import STEM as OPTICAL_VALUE
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import fit_correction
from fit_cushing_tail_distribution import LEVELS, distribution
from fit_cushing_terminal_weekly_transfers import LIMITED_MARKET
from fit_cushing_volatility_neighbors import STEM as WEIGHTED
from register_cushing_leading_models import LINEAGE, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release

STEM = "cushing_tail_calibration"
LONG = "long_median_and_regional"
NO_OPTICS = "long_without_current_optical"
EMPIRICAL = {"public_empirical": PUBLIC, "hybrid_empirical": LIMITED_MARKET, "long_empirical": LONG}
NORMAL = {"public_constant_scale": (PUBLIC, False), "long_without_current_optical_scale": (NO_OPTICS, False),
          "long_constant_scale": (LONG, False), "long_disagreement_scale": (LONG, True)}
METHODS = (*EMPIRICAL, *NORMAL)
MARKET_METHODS = ("original_satellite", "public_tail", "long_tail", "disagreement_tail")
CLASSES = ("draw", "normal", "build")
ROOT_PI = math.sqrt(math.pi)


def observations():
    out = indexed(f"{OUTER}_forecasts")
    optical = indexed(f"{OPTICAL_VALUE}_forecasts")
    for issue, r in sorted(indexed(f"{EXTENDED}_inventory").items()):
        q, o = out[issue], optical[issue]
        cutoff, known = (datetime.fromisoformat(r[k]) for k in ("cutoff", "known"))
        if (not cutoff < known or cutoff != datetime.fromisoformat(q["cutoff_utc"])
                or known != datetime.fromisoformat(q["known_utc"]) or float(q["actual"]) != float(r["actual"])
                or datetime.fromisoformat(o["cutoff"]) != cutoff or datetime.fromisoformat(o["known"]) != known
                or float(o["actual"]) != float(r["actual"]) or float(o[LIMITED_MARKET]) != float(r[LIMITED_MARKET])
                or any(r[k] != q[k] for k in ("week_ending", "study_phase"))
                or r["latest_training_release"] and datetime.fromisoformat(r["latest_training_release"]) >= cutoff):
            raise ValueError(f"Inconsistent inventory source on {issue}")
        yield {k: r[k] for k in ("release_date", "week_ending", "study_phase")} | dict(
            cutoff=cutoff, known=known, actual=float(r["actual"]),
            **{PUBLIC: float(q[PUBLIC]), LIMITED_MARKET: float(r[LIMITED_MARKET]), LONG: float(r[LONG]),
               NO_OPTICS: float(r[LONG]) - float(o["optical_update"])})


def spread(row):
    return min(abs(row[LONG] - row[PUBLIC]) / .30, 2.)


def scale_fit(train, current, point, local):
    """Minimize rolling normal CRPS with a shrunk disagreement term."""
    z = np.asarray([spread(q) for q in train])
    errors = np.asarray([q["actual"] - q[point] for q in train])
    age = np.asarray([(current["cutoff"] - q["cutoff"]).total_seconds() / 604800 for q in train])
    w = np.exp2(-age / 78.)
    w /= w.sum()
    centre = w @ z
    design = np.column_stack((np.ones(len(z)), z - centre))[:, :2 if local else 1]

    def objective(beta):
        sigma = np.exp(design @ beta)
        x = errors / sigma
        pdf = np.exp(-.5 * x*x) / math.sqrt(2 * math.pi)
        loss = w @ (errors * (2 * ndtr(x) - 1) + sigma * (2 * pdf - 1 / ROOT_PI))
        gradient = design.T @ (w * sigma * (2 * pdf - 1 / ROOT_PI))
        if local:
            loss += .08 * beta[1]**2
            gradient[1] += .16 * beta[1]
        return loss, gradient

    bounds = [(math.log(.35), math.log(2.5))] + ([(-.35, .35)] if local else [])
    result = minimize(objective, np.zeros(len(bounds)), jac=True, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        raise RuntimeError(f"Scale fit failed on {current['release_date']}: {result.message}")
    here = np.array((1., spread(current) - centre))[:len(bounds)]
    sigma = float(np.clip(np.exp(here @ result.x), .35, 2.5))
    p = {f"{v:g}": (float(ndtr((-v-current[point])/sigma)), float(ndtr((current[point]-v)/sigma))) for v in LEVELS}
    return p, sigma


def probabilities(train, row):
    empirical = {m: distribution(train, row, point, False)[0] for m, point in EMPIRICAL.items()}
    normal = {m: scale_fit(train, row, point, local) for m, (point, local) in NORMAL.items()}
    return empirical | {m: p for m, (p, _) in normal.items()}, {m: s for m, (_, s) in normal.items()}


def replay():
    history = []
    for q in observations():
        train = [p for p in history if p["known"] < q["cutoff"]][-HISTORY:]
        history.append(q)
        if len(train) < MIN_WINDOWS:
            continue
        p, scale = probabilities(train, q)
        yield q | dict(probabilities=p, scales=scale, latest_training_label_utc=train[-1]["known"], training_windows=len(train))


def three(pair):
    low, high = pair
    return np.array((low, 1 - low - high, high))


def category(actual, level):
    return int(actual >= level) - int(actual <= -level) + 1


def accuracy(rows, level):
    y = np.asarray([category(r["actual"], level) for r in rows])
    observed = np.eye(3)[y]
    base = np.asarray([three(r["probabilities"][METHODS[0]][f"{level:g}"]) for r in rows])
    losses = {}
    out = {}
    for m in METHODS:
        p = np.asarray([three(r["probabilities"][m][f"{level:g}"]) for r in rows])
        brier = np.square(p - observed).sum(axis=1)
        losses[m] = brier
        out[m] = dict(brier=float(brier.mean()), log_loss=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.)).mean()),
                      event_average_precision=float(average_precision_score(y != 1, p[:, 0]+p[:, 2])) if (y != 1).any() else None,
                      brier_delta_vs_public_95=block_bootstrap_mean(brier - np.square(base - observed).sum(axis=1)))
    optics = losses["long_constant_scale"] - losses["long_without_current_optical_scale"]
    scale = losses["long_constant_scale"] - losses["public_constant_scale"]
    return dict(releases=len(rows), events=dict(zip(CLASSES, np.bincount(y, minlength=3).tolist(), strict=True)), results=out,
                optical_increment=dict(mean_brier_change=float(optics.mean()), paired_95=block_bootstrap_mean(optics)),
                constant_scale_public_comparison=dict(mean_brier_change=float(scale.mean()), paired_95=block_bootstrap_mean(scale)))


def trading(forecasts):
    by_issue = {r["release_date"]: r for r in forecasts}
    base = {q["release_date"]: q for q in read(f"{WEIGHTED}_trades") if q["method"] == f"{SUPPLEMENT}:volatility"}
    history = defaultdict(list)
    for r, _ in inputs():
        issue, decision = r["release_date"], r["decision"]
        if issue not in by_issue or issue not in base:
            continue
        f, a = by_issue[issue], base[issue]
        if (f["actual"] != r["x"][0] or not f["latest_training_label_utc"] < f["cutoff"] < f["known"] <= decision < r["entry"]
                or any(r[k] != a[k] for k in ("near_month", "far_month", "study_phase"))
                or any(datetime.fromisoformat(a[k]) != r[field] for k, field in (
                    ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at"), ("decision_at_utc", "decision")))
                or any(datetime.fromisoformat(a[k]) >= decision
                       for k in ("latest_training_exit_utc", "latest_calibration_exit_utc") if a[k])):
            raise ValueError(f"Different market window or release vintage on {issue}")
        point, sigma = float(a["expected_long_move_usd"]), r["volatility_usd"]
        observed = category(f["actual"], 1.) - 1
        for m, probability in zip(MARKET_METHODS, (None, "public_empirical", "long_empirical", "long_disagreement_scale"), strict=True):
            train = [q for q in history[m] if q["exit_at"] < decision][-HISTORY:]
            revision = 0.
            surprise = None
            if probability is not None:
                draw, build = f["probabilities"][probability]["1"]
                surprise = observed - (build - draw)
                row = dict(cutoff=decision, exit_at=r["exit_at"], design=(surprise, surprise * (draw + build)),
                           error=(r["long_gross"] - point) / sigma)
                revision = fit_correction(train, row, "design", "error", penalty=(.35, .7), loss_scale=1., bound=.30)
                history[m].append(row)
            training_at = datetime.fromisoformat(a["latest_training_exit_utc"])
            if train:
                training_at = max(training_at, train[-1]["exit_at"])
            calibration = datetime.fromisoformat(a["latest_calibration_exit_utc"]) if a["latest_calibration_exit_utc"] else None
            q = trade(r, m, point + sigma * revision, 3, training_at=training_at, calibration_at=calibration)
            if m == MARKET_METHODS[0] and any(q[k] != int(a[k]) for k in ("near_side", "gross_usd", "net_170_usd")):
                raise ValueError(f"The historical leader did not reproduce on {issue}")
            yield q | dict(tail_surprise=surprise, latest_probability_label_utc=f["latest_training_label_utc"].isoformat(),
                           net_340_usd=q["gross_usd"]-340*abs(q["near_side"]))


def register(issue, scoring):
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    leader_path = DERIVED / f"cushing_forward_leading_models_{issue}_registration.json"
    long_path = DERIVED / f"{EXTENDED}_{issue}_registration.json"
    path = DERIVED / f"{STEM}_{issue}_registration.json"
    original, leader, longer = (json.loads(p.read_bytes()) for p in (original_path, leader_path, long_path))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    if any(q[k] != original[k] for q in (leader, longer) for k in LINEAGE):
        raise ValueError("The component forecasts have different release vintages")
    checked_hashes(leader["sources_sha256"])
    checked_hashes(longer["sources_sha256"])
    if not path.exists():
        if scoring or not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The first EIA report must be unpublished at registration")
        train = [q for q in observations() if q["known"] < cutoff][-HISTORY:]
        control = leader["estimates_kbbl"][PUBLIC]
        components = leader["inventory_components_kbbl"]
        if not original["measured_pairs"]:
            raise ValueError("A direct optical panel is required to isolate the current forecast adjustment")
        optical_update = sum(components[k] for k in ("optical_base", "additional_passes", "terminal_limited_coverage")) - control
        point = longer["estimates_kbbl"][LONG]/1000
        row = dict(release_date=issue, cutoff=cutoff, **{PUBLIC: control/1000, LIMITED_MARKET: leader["estimates_kbbl"][LIMITED_MARKET]/1000,
                                                       LONG: point, NO_OPTICS: point - optical_update/1000})
        if len(train) < MIN_WINDOWS:
            raise ValueError("No eligible calibration history")
        pred, scales = probabilities(train, row)
        now = datetime.now(timezone.utc)
        if not cutoff <= now < deadline:
            raise ValueError("Registration window closed during calibration")
        sources = (original_path, leader_path, long_path, DERIVED / f"{EXTENDED}_inventory.csv",
                   DERIVED / f"{OUTER}_forecasts.csv", DERIVED / f"{OPTICAL_VALUE}_forecasts.csv",
                   Path(__file__).resolve(), ROOT / "scripts/fit_cushing_tail_distribution.py")
        freeze(path, {k: original[k] for k in LINEAGE} | dict(recorded_at_utc=now.isoformat(), training_windows=len(train),
            latest_training_label_utc=train[-1]["known"].isoformat(), probabilities={m: {v: dict(zip(CLASSES, three(p).tolist(), strict=True))
                for v, p in bounds.items()} for m, bounds in pred.items()}, scales_million_bbl=scales,
            sources_sha256={str(s.relative_to(ROOT)): digest(s) for s in sources}))
    result = json.loads(path.read_bytes())
    if (any(result[k] != original[k] for k in LINEAGE) or not datetime.fromisoformat(result["latest_training_label_utc"])
            < cutoff <= datetime.fromisoformat(result["recorded_at_utc"]) < deadline):
        raise ValueError("The probability registration is inconsistent")
    checked_hashes(result["sources_sha256"])
    if not scoring:
        return result
    outcome = score_release(original_path, original)
    out = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path))
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"] / 1000
        out.update(actual_eia_change_kbbl=1000*actual, first_release_sha256=outcome["first_release_sha256"],
            scores={m: {level: sum((value - (label == CLASSES[category(actual, float(level))]))**2 for label, value in p.items())
                        for level, p in bounds.items()} for m, bounds in result["probabilities"].items()})
    write(path.with_name(f"{STEM}_{issue}_score.json"), (json.dumps(out, indent=2, allow_nan=False) + "\n").encode())
    return out


def main():
    with threadpool_limits(limits=1):
        forecasts = list(replay())
        trades = list(trading(forecasts))
    report = dict(status="exploratory_on_reviewed_history", forecasts={}, trading={})
    for phase in PHASES:
        panel = [q for q in forecasts if q["study_phase"] == phase]
        report["forecasts"][phase] = {f"{v:g}": accuracy(panel, v) for v in LEVELS}
        group = {m: [q for q in trades if q["study_phase"] == phase and q["method"] == m] for m in MARKET_METHODS}
        control = group[MARKET_METHODS[0]]
        result = {}
        for m, rows in group.items():
            if [q["release_date"] for q in rows] != [r["release_date"] for r in control]:
                raise ValueError(f"The tail rule changed the calendar on {phase}/{m}")
            pairs = list(zip(rows, control, strict=True))
            result[m] = robustness(rows) | daily_risk(rows) | dict(net_340_usd=sum(q["net_340_usd"] for q in rows),
                changed_sides=sum(q["near_side"] != r["near_side"] for q, r in pairs),
                delta_mean_95_usd=block_bootstrap_mean([q["net_170_usd"] - r["net_170_usd"] for q, r in pairs]))
        report["trading"][phase] = result
        print(phase, {v: {m: round(q["brier"], 6) for m, q in d["results"].items()}
                      for v, d in report["forecasts"][phase].items()}, {m: q["net_170_usd"] for m, q in result.items()})
    export(f"{STEM}_forecasts", [{k: q[k] for k in ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual",
            "training_windows", "latest_training_label_utc", PUBLIC, LIMITED_MARKET, LONG, NO_OPTICS)} | dict(
            **{f"{m}_sigma": s for m, s in q["scales"].items()},
            **{f"{m}_{v}_{label}": probability for m, values in q["probabilities"].items() for v, p in values.items()
               for label, probability in zip(CLASSES, three(p), strict=True)}) for q in forecasts])
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
        print(json.dumps({k: v for k, v in register(args.release, args.score).items() if k != "sources_sha256"}, indent=2))
    else:
        main()
