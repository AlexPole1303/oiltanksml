"""Compare one-spread rules with risk-scaled admission at fixed CL execution clocks."""

import json
from collections import defaultdict

import numpy as np

from backtest_wti_cushing_spread import COSTS
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_cl_risk_budget"
PHASES = ("development", "validation", "holdout")
FAMILIES = (("public_volatility", "1"), ("coverage_volatility", "1"), ("public_volatility", "2"), ("coverage_volatility", "2"))
# Maximum modelled one-standard-deviation risk per whole spread in USD.
LIMITS = (1000, 2000, 3000)
# Expected net move per dollar of modelled standard deviation, after the $170 hurdle.
EFFICIENCIES = (0.10, 0.25, 0.50)


def scenario_rows():
    for r in read("cushing_cl_execution_windows_trades"):
        family = r["method"], r["first_session_number"]
        if family not in FAMILIES or r["last_session_number"] != "5":
            continue
        sigma = float(r["volatility_usd"])
        value = float(r["expected_long_gross_usd"])
        original = int(r["near_side"])
        gross_original = int(r["gross_usd"])
        efficiency = (abs(value)-170)/sigma
        rules = {"original":True} | {f"risk_max_{cap}": sigma <= cap for cap in LIMITS} | {
            f"net_over_sigma_{rate:g}": efficiency >= rate for rate in EFFICIENCIES}
        for rule, admit in rules.items():
            side = original if admit else 0
            gross = gross_original if admit else 0
            yield {k:r[k] for k in ("release_date", "study_phase", "near_month", "far_month", "entry_session", "exit_session")} | dict(
                model=family[0], first_session=int(family[1]), rule=rule, near_side=side, volatility_usd=sigma,
                estimate_usd=value, expected_net_over_sigma=efficiency, gross_usd=gross,
                **{f"cost_{c}_usd":c*abs(side) for c in COSTS}, **{f"net_{c}_usd":gross-c*abs(side) for c in COSTS})


def main():
    rows = list(scenario_rows())
    by_phase = defaultdict(dict)
    groups = defaultdict(list)
    for r in rows:
        family = f"{r['first_session']}:{r['model']}"
        groups[r["study_phase"], family, r["rule"]].append(r)
    for (phase, family, rule), group in groups.items():
        control = groups[phase, family, "original"]
        if [r["release_date"] for r in group] != [r["release_date"] for r in control]:
            raise ValueError(f"Calendar mismatch for {phase}/{family}/{rule}")
        delta = np.asarray([r["net_170_usd"]-c["net_170_usd"] for r,c in zip(group,control,strict=True)])
        by_phase[phase][f"{family}:{rule}"] = robustness(group) | dict(delta_vs_original_usd=int(delta.sum()),
            paired_mean_delta_vs_original_ci_95_usd=block_bootstrap_mean(delta))

    def prior_rank(name):
        a,b = [by_phase[p][name] for p in PHASES[:2]]
        return min(a["net_170_usd"],b["net_170_usd"]), a["net_170_usd"]+b["net_170_usd"]

    report = dict(status="exploratory_replay_on_previously_examined_history", families=FAMILIES,
                  prior_choice=max(by_phase[PHASES[0]],key=prior_rank), phases=by_phase)
    payload = json.dumps(report, indent=2, allow_nan=False)+"\n"
    export(f"{STEM}_trades", rows)
    write(DERIVED/f"{STEM}_summary.json", payload.encode())
    print("Pre-2024 choice", report["prior_choice"])
    for name in sorted(by_phase["holdout"], key=lambda k:by_phase["holdout"][k]["net_170_usd"], reverse=True):
        print(name, [(by_phase[p][name]["trades"],by_phase[p][name]["net_170_usd"],
                      by_phase[p][name]["stress_net_excluding_three_largest_wins_usd"]) for p in PHASES])


if __name__ == "__main__":
    main()
