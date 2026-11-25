"""Compare an early optical rule and the delayed leader with one open spread."""

import json
from collections import defaultdict
from datetime import datetime

from audit_cushing_cl_two_windows import book, results
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_volatility import PHASES
from fit_cushing_early_registered_pairs import STEM as EARLY
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_outer_capture_nowcast import STEM as DELAYED, SUPPLEMENT

STEM = "cushing_early_registered_overlay"
LEADER = f"{SUPPLEMENT}_conditional"
PUBLIC = "public_ensemble_conditional"
POLICIES = {
    "delayed_satellite": (None, LEADER),
    "curve_early_and_delayed_satellite": ("curve", LEADER),
    "registered_early_and_delayed_satellite": ("registered_fallback_ml", LEADER),
    "delayed_public": (None, PUBLIC),
    "curve_early_and_delayed_public": ("curve", PUBLIC),
    "registered_early_and_delayed_public": ("registered_fallback_ml", PUBLIC),
}


def candidates():
    for leg, stem in enumerate((EARLY, DELAYED)):
        models = {m[leg] for m in POLICIES.values()}
        clock = ("previous_session", "report_day")[leg]
        decision_key, train_key = (("cutoff_utc", "latest_training_exit_utc"),
                                    ("decision_at_utc", "settled_training_label_utc"))[leg]
        for r in read(f"{stem}_trades"):
            if r["method"] not in models:
                continue
            decision, entry, exit_at = (datetime.fromisoformat(r[k]) for k in (decision_key, "entry_at_utc", "exit_at_utc"))
            trained = datetime.fromisoformat(r[train_key])
            side, gross = (int(r[k]) for k in ("near_side", "gross_usd"))
            if (not trained < decision < entry < exit_at or abs(side) > 1 or (gross and not side)
                    or int(r["net_170_usd"]) != gross-170*abs(side)):
                raise ValueError(f"Invalid source trade on {r['release_date']}/{r['method']}")
            path = [int(r[f"session_{i}_net_170_usd"]) for i in range(2, 6)] if leg else [-85*abs(side), gross-170*abs(side)]
            yield {k:r[k] for k in ("release_date", "study_phase", "near_month", "far_month")} | dict(
                clock=clock, leg=leg, source_model=r["method"], decision=decision, entry=entry, exit_at=exit_at,
                near_side=side, source_near_side=side, gross_usd=gross, path_170=path)


def main():
    windows = sorted(candidates(), key=lambda r:(r["entry"], r["leg"], r["release_date"]))
    if len(windows) != len({(q["release_date"], q["clock"], q["source_model"]) for q in windows}):
        raise ValueError("Duplicate source rows")
    rows = [row for policy, methods in POLICIES.items() for row in book(windows, policy, methods)]
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["study_phase"], r["portfolio"]].append(r)
    forecast_panel = read("cushing_robust_stack_forecasts")
    report = dict(status="exploratory_on_reviewed_history", max_open_spreads=1, phases={})
    for phase in PHASES:
        releases = sorted(r["release_date"] for r in forecast_panel if r["study_phase"] == phase)
        panel = {m:results(grouped[phase, m], releases) for m in POLICIES}
        metrics = report["phases"][phase] = {}
        for method, (summary, net) in panel.items():
            group = grouped[phase, method]
            active = [r for r in group if r["near_side"]]
            if any(a["exit_at"] >= b["entry"] for a, b in zip(active, active[1:])):
                raise ValueError(f"Overlapping windows in {phase}/{method}")
            if POLICIES[method][0] is None and any(r["blocked_by_open_spread"] for r in group):
                raise ValueError(f"The delayed-only calendar changed in {phase}/{method}")
            metrics[method] = summary | dict(
                early_net_170_usd=sum(r["net_170_usd"] for r in group if not r["leg"]),
                delayed_net_170_usd=sum(r["net_170_usd"] for r in group if r["leg"]),
                early_net_340_usd=sum(r["net_340_usd"] for r in group if not r["leg"]),
                delayed_net_340_usd=sum(r["net_340_usd"] for r in group if r["leg"]),
                suppressed_delayed_trades=sum(bool(r["leg"] and r["blocked_by_open_spread"]) for r in group))
            for control in ("delayed_satellite", "curve_early_and_delayed_satellite"):
                delta = [net[k]-panel[control][1][k] for k in releases]
                metrics[method][f"delta_vs_{control}"] = dict(net_170_usd=sum(delta),
                    net_340_usd=summary["net_340_usd"]-panel[control][0]["net_340_usd"],
                    paired_net_170_per_report_95_usd=block_bootstrap_mean(delta))
        print(phase, {m:(v["trades"], v["net_170_usd"], v["net_340_usd"], v["daily_drawdown_170_usd"],
                         v["suppressed_delayed_trades"]) for m, v in metrics.items()}, flush=True)
    export(f"{STEM}_trades", [{k:v for k,v in r.items() if k not in ("path_170", "leg")} |
        dict(path_170_json=json.dumps(r["path_170"])) for r in rows])
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
