"""Audit the researched CL families on aligned release outcomes and resample by blocks."""

import json
from collections import defaultdict

import numpy as np

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_nowcast_ensemble import block_bootstrap_mean

STEM = "cushing_cl_selection_audit"
SOURCES = {
    "expanded": ("cushing_cl_delayed_calendar_trades", ("method",)),
    "windows": ("cushing_cl_execution_windows_trades", ("first_session_number", "last_session_number", "method")),
    "reaction": ("cushing_cl_release_reaction_trades", ("method",)),
    "risk": ("cushing_cl_risk_budget_trades", ("first_session", "model", "rule")),
    "interval": ("cushing_interval_nowcast_trades", ("first_session_number", "method")),
    "interactions": ("cushing_cl_volatility_interactions_trades", ("method",)),
    "observation": ("cushing_observation_operator_trades", ("method",)),
    "prerelease": ("cushing_cl_prerelease_trades", ("method",)),
    "tail": ("cushing_tail_events_trades", ("clock", "method")),
    "portfolio": ("cushing_cl_two_windows_trades", ("portfolio",)),
    "inventory_level": ("cushing_cl_inventory_level_trades", ("method",)),
    "shared_scene": ("cushing_shared_scene_trades", ("method",)),
    "price_confirmation": ("cushing_cl_price_confirmation_trades", ("method",)),
    "exit_policy": ("cushing_cl_exit_policy_trades", ("method",)),
    "regional_balance": ("cushing_regional_balance_trades", ("method",)),
    "optical_reliability": ("cushing_optical_reliability_trades", ("method",)),
    "spectral_recovery": ("cushing_spectral_recovery_trades", ("method",)),
    "curve_residual": ("cushing_cl_curve_residual_trades", ("method",)),
    "terminal_balance": ("cushing_terminal_balance_trades", ("method",)),
    "terminal_ablation": ("cushing_terminal_balance_ablation_trades", ("method",)),
    "payoff_calibration": ("cushing_cl_payoff_calibration_trades", ("method",)),
    "optical_stability": ("cushing_optical_stability_trades", ("method",)),
    "conditional_response": ("cushing_cl_conditional_response_trades", ("method",)),
    "intraweek": ("cushing_intraweek_trades", ("method",)),
    "dated_carry": ("cushing_cl_dated_carry_trades", ("method",)),
    "tail_prior": ("cushing_inventory_tail_prior_trades", ("method",)),
    "tenors": ("cushing_cl_tenors_trades", ("tenor", "method")),
    "tenor_selection": ("cushing_cl_tenors_selection_trades", ("selection",)),
    "recovered_optical": ("cushing_recovered_optical_trades", ("method",)),
    "prerelease_sensor": ("cushing_prerelease_sensor_trades", ("method",)),
    "forecast_bias": ("cushing_forecast_bias_trades", ("method",)),
    "yearly_selection": ("cushing_cl_transfer_trades", ("method",)),
    "weather_options": ("cushing_weather_and_options_trades", ("method",)),
    "bracketed_inventory": ("cushing_bracketed_inventory_trades", ("method",)),
    "satellite_innovation": ("cushing_satellite_innovation_trades", ("method",)),
    "positioning": ("cushing_cl_positioning_trades", ("method",)),
    "optical_variance": ("cushing_optical_variance_budget_trades", ("method",)),
    "exit_horizon": ("cushing_cl_exit_horizon_trades", ("method",)),
    "following_report": ("cushing_cl_following_report_trades", ("method",)),
    "coverage_scaling": ("cushing_optical_coverage_scaling_trades", ("method",)),
    "inventory_market_bridge": ("cushing_inventory_market_bridge_trades", ("method",)),
    "payoff_distribution": ("cushing_cl_payoff_distribution_trades", ("method",)),
    "joint_band_expansion": ("cushing_joint_band_expansion_trades", ("method",)),
    "outer_capture": ("cushing_outer_capture_nowcast_trades", ("method",)),
    "before_next_release": ("cushing_cl_before_next_release_trades", ("method",)),
    "public_state": ("cushing_public_state_trades", ("method",)),
    "public_state_conditional": ("cushing_public_state_conditional_trades", ("method",)),
    "trade_risk": ("cushing_cl_trade_risk_trades", ("model", "method")),
    "inventory_innovation": ("cushing_cl_inventory_innovation_trades", ("method",)),
    "optical_gate": ("cushing_optical_gate_trades", ("method",)),
    "optical_attribution": ("cushing_optical_attribution_trades", ("method",)),
    "spanning_pair_uncertainty": ("cushing_spanning_pair_uncertainty_trades", ("method",)),
    "multiframe": ("cushing_multiframe_inventory_trades", ("method",)),
    "multiframe_registered": ("cushing_multiframe_inventory_registered_trades", ("method",)),
    "multiframe_independent_errors": ("cushing_multiframe_inventory_registered_independent_trades", ("method",)),
    "multiframe_screening": ("cushing_multiframe_inventory_screening_only_trades", ("method",)),
    "following_report_attribution": ("cushing_cl_rollover_attribution_trades", ("method",)),
    "late_pair_recovery": ("cushing_late_pair_recovery_trades", ("method",)),
    "late_pair_drift": ("cushing_late_pair_market_calibration_trades", ("method",)),
    "following_forecasts": ("cushing_cl_paired_following_forecasts_trades", ("strategy", "method")),
    "tail_distribution": ("cushing_tail_distribution_trades", ("method",)),
    "conditional_loss": ("cushing_cl_conditional_loss_trades", ("strategy", "method")),
    "joint_late_pairs": ("cushing_joint_late_pairs_trades", ("method",)),
    "early_registered_pairs": ("cushing_early_registered_pairs_trades", ("method",)),
    "early_registered_overlay": ("cushing_early_registered_overlay_trades", ("portfolio",)),
    "storage_state": ("cushing_storage_state_trades", ("method",)),
    "volatility_neighbors": ("cushing_volatility_neighbors_trades", ("method",)),
    "terminal_weekly_transfers": ("cushing_terminal_weekly_transfers_trades", ("method",)),
    "optical_value": ("cushing_optical_value_trades", ("method",)),
    "optical_value_disagreement": ("cushing_optical_value_disagreement_trades", ("method",)),
    "inventory_persistence": ("cushing_inventory_persistence_trades", ("method",)),
    "inventory_persistence_response": ("cushing_inventory_persistence_response_trades", ("method",)),
    "inventory_component_calibration": ("cushing_inventory_component_calibration_trades", ("method",)),
    "payoff_scenarios": ("cushing_cl_payoff_scenarios_trades", ("method",)),
    "dated_height_calibration": ("cushing_dated_height_calibration_trades", ("method",)),
    "timing_robust": ("cushing_cl_timing_robust_trades", ("method",)),
    "extended_market_history": ("cushing_extended_market_history_trades", ("method",)),
    "reaction_residual": ("cushing_cl_reaction_residual_trades", ("model", "method")),
    "memory_ensemble": ("cushing_cl_memory_ensemble_trades", ("method",)),
    "tail_calibration": ("cushing_tail_calibration_trades", ("method",)),
    "terminal_consensus": ("cushing_terminal_consensus_trades", ("method",)),
    "daily_activity": ("cushing_cl_daily_activity_trades", ("model", "method")),
    "weighted_optical": ("cushing_cl_weighted_optical_trades", ("method",)),
    "coverage_stack": ("cushing_coverage_stack_trades", ("method",)),
    "national_release": ("cushing_cl_national_release_trades", ("model", "method")),
    "entry_margin": ("cushing_cl_entry_margin_trades", ("method",)),
    "separate_innovation": ("cushing_cl_separate_innovation_trades", ("method",)),
    "optical_noise_floor": ("cushing_optical_noise_floor_trades", ("method",)),
    "training_uncertainty": ("cushing_cl_training_uncertainty_trades", ("method",)),
    "seasonal_residual": ("cushing_inventory_seasonal_residual_trades", ("method",)),
    "optical_direction": ("cushing_optical_direction_trades", ("method",)),
    "regional_imports": ("cushing_regional_imports_trades", ("method",)),
    "matched_curve": ("cushing_cl_equal_model_risk_trades", ("method",)),
    "overpass_covariance": ("cushing_overpass_covariance_trades", ("method",)),
    "weekend_timing": ("cushing_weekend_timing_trades", ("method",)),
    "extension_segments": ("cushing_cl_extension_segments_trades", ("method",)),
    "terminal_inventory_response": ("cushing_terminal_inventory_response_trades", ("method",)),
    "height_response_curve": ("cushing_height_response_curve_trades", ("method",)),
    "mark_to_market_exit": ("cushing_cl_mark_to_market_exit_trades", ("method",)),
    "optical_error_distribution": ("cushing_optical_error_distribution_trades", ("method",)),
    "mixed_optical_intervals": ("cushing_mixed_optical_intervals_trades", ("method",)),
    "repeated_roof_pairs": ("cushing_repeated_roof_pairs_trades", ("method",)),
    "inventory_conditional_scale": ("cushing_inventory_conditional_scale_trades", ("method",)),
    "surprise_calibration": ("cushing_cl_surprise_calibration_trades", ("method",)),
    "earlier_entry": ("cushing_cl_earlier_entry_trades", ("method",)),
    "after_release_short": ("cushing_cl_earlier_entry_short_trades", ("method",)),
    "earlier_entry_combined": ("cushing_cl_earlier_entry_combined_trades", ("method",)),
    "pass_reliability": ("cushing_pass_reliability_trades", ("method",)),
    "late_pair_timing": ("cushing_late_pair_timing_trades", ("method",)),
    "dated_curve_shape": ("cushing_cl_curve_shape_trades", ("method",)),
    "optical_likelihood": ("cushing_optical_likelihood_trades", ("method",)),
    "delivery_location": ("cushing_delivery_location_trades", ("method",)),
    "standardized_location": ("cushing_delivery_standardized_trades", ("method",)),
    "transfer_persistence": ("cushing_transfer_persistence_trades", ("method",)),
}
CONTROL = "interactions:carry_volatility_context"
MATCHED_CURVE = "matched_curve:weighted_curve"
PREVIOUS = "interactions:inventory_interaction"
WEIGHTED = "volatility_neighbors:outer_supplement:volatility"
EXTENSION = "extension_segments:curve_post_report_gate"
BASE = "conditional_response:inventory_conditional"
HURDLES = (250, 340, 500)
OPTICAL = "outer_capture:outer_supplement_conditional"
OPTICAL_HURDLES = (171, 250, 340, 500)
ENTRY_THRESHOLDS = {BASE:HURDLES, OPTICAL:OPTICAL_HURDLES}
SIMULATIONS = 10000


def pnl_table():
    releases = [r["release_date"] for r in read("cushing_robust_stack_forecasts") if r["study_phase"] == "holdout"]
    keys = {d:i for i,d in enumerate(releases)}
    series = defaultdict(lambda: np.zeros((len(releases), 4), dtype=int))
    series.update({f"{model}:hurdle_{hurdle}":np.zeros((len(releases),4),dtype=int)
                   for model, hurdles in ENTRY_THRESHOLDS.items() for hurdle in hurdles})
    seen = set()
    for family, (stem, fields) in SOURCES.items():
        for r in read(stem):
            if r["study_phase"] != "holdout":
                continue
            name = ":".join((family, *(r[f] for f in fields)))
            idx = keys[r["release_date"]]
            window = r.get("clock") if family in ("portfolio", "early_registered_overlay", "earlier_entry_combined") else None
            if (name,idx,window) in seen:
                raise ValueError(f"Repeated release {r['release_date']} in {name}")
            seen.add((name,idx,window))
            side = int(r["near_side"])
            gross = int(r["gross_usd"])
            net = int(r["net_170_usd"])
            if side not in (-1,0,1) or net != gross-170*abs(side) or (side==0 and gross):
                raise ValueError(f"Invalid costs for {name} on {r['release_date']}")
            stress = gross-340*abs(side)
            if r.get("net_340_usd") and int(r["net_340_usd"]) != stress:
                raise ValueError(f"Invalid stressed costs for {name} on {r['release_date']}")
            series[name][idx] += side,gross,net,stress
            if name in ENTRY_THRESHOLDS:
                forecast = abs(float(r["expected_long_move_usd"]))
                for hurdle in ENTRY_THRESHOLDS[name]:
                    if forecast > hurdle:
                        series[f"{name}:hurdle_{hurdle}"][idx] += side,gross,net,stress
    if len(keys) != 148 or int(series["expanded:public"][:,2].sum())!=37640 or not np.array_equal(
        series[PREVIOUS],series["portfolio:delayed_inventory"]):
        raise ValueError("The archived holdout calendar or anchor changed")
    return releases, dict(series)


def resample_index(n, block):
    rng = np.random.default_rng(671016+block)
    starts = rng.integers(n,size=(SIMULATIONS,int(np.ceil(n/block))))
    return ((starts[:,:,None]+np.arange(block))%n).reshape(SIMULATIONS,-1)[:,:n]


def adjusted_max(series, baseline, name, block):
    n = len(baseline)
    # Collapse rules with identical payoffs to keep the effective family count visible.
    matrix, named = {}, {}
    for key, values in series.items():
        diff = (values[:,2]-baseline).astype(float)
        if np.any(diff):
            named[key] = diff
            matrix.setdefault(diff.tobytes(),diff)
    values = np.stack(list(matrix.values()),axis=1)
    center = values.mean(axis=0)
    observed = float(center.max())
    idx = resample_index(n,block)
    maximum = ((values[idx].mean(axis=1))-center).max(axis=1)
    primary = named[name][idx].mean(axis=1)
    return dict(block_releases=block,unique_nonzero_strategies=len(matrix),max_mean_increment_usd=observed,
                family_p_value=float((1+np.count_nonzero(maximum>=observed))/(1+len(maximum))),
                primary_mean_increment_usd=float(named[name].mean()),
                primary_unadjusted_interval_usd=np.quantile(primary,[.025,.975]).tolist(),
                primary_family_adjusted_p_value=float((1+np.count_nonzero(maximum>=named[name].mean()))/(1+len(maximum))))


def regimes(releases, series, primary):
    risk = {r["release_date"]:float(r["volatility_usd"]) for r in read(SOURCES["expanded"][0])
            if r["method"]=="public" and r["study_phase"]=="holdout"}
    rows = []
    for i, day in enumerate(releases):
        if day not in risk:
            continue
        sigma = risk[day]
        regime = "under_1000" if sigma<1000 else "1000_to_3000" if sigma<=3000 else "over_3000"
        p,c = [series[m][i] for m in (primary,CONTROL)]
        rows.append(dict(release_date=day,volatility_usd=sigma,regime=regime,candidate_side=int(p[0]),carry_side=int(c[0]),
                         candidate_net_170_usd=int(p[2]),carry_net_170_usd=int(c[2]),increment_usd=int(p[2]-c[2])))
    groups = defaultdict(list)
    for r in rows:
        groups[r["regime"]].append(r)
    summaries = {k:dict(windows=len(g),trades=sum(r["candidate_side"]!=0 for r in g),
                        **{f:sum(r[f] for r in g) for f in ("candidate_net_170_usd","carry_net_170_usd","increment_usd")})
                 for k,g in groups.items()}
    return rows,summaries


def main():
    releases, series = pnl_table()
    models = {m:v for m,v in series.items() if m not in {CONTROL,"expanded:carry_only","expanded:always_long","prerelease:always_long"}}
    primary = max(models,key=lambda m:int(models[m][:,2].sum()))
    rows, summary = regimes(releases,series,primary)
    maximum = max(int(v[:,2].sum()) for v in models.values())
    stress_max = max(int(v[:,3].sum()) for v in models.values())
    current, previous = (series[k] for k in (primary,PREVIOUS))
    delta = current[:,2]-previous[:,2]
    changes = [dict(release_date=release,previous_side=int(a[0]),candidate_side=int(b[0]),
                   previous_net_170_usd=int(a[2]),candidate_net_170_usd=int(b[2]),increment_usd=int(b[2]-a[2]))
               for release,a,b in zip(releases,previous,current,strict=True) if a[2]!=b[2] or a[0]!=b[0]]
    exit_delta = series[EXTENSION][:,2] - series[WEIGHTED][:,2]
    if not np.array_equal(series[EXTENSION][:,0], series[WEIGHTED][:,0]):
        raise ValueError("The extension policy modifies an entry")
    report = dict(status="historical_sensitivity_to_selection", release_count=len(releases), tested_rules=len(models),
        primary=primary, carry_comparator=CONTROL, matched_carry_comparator=MATCHED_CURVE,
        previous_primary=PREVIOUS,no_admitted_window_payoff_usd=0,
        maximum_net_170_usd=maximum, rules_at_maximum=[m for m,v in models.items() if int(v[:,2].sum())==maximum],
        maximum_net_340_usd=stress_max, rules_at_stressed_maximum=[m for m,v in models.items() if int(v[:,3].sum())==stress_max],
        conditional_entry_hurdles={m:dict(trades=int(np.count_nonzero(v[:,0])),net_170_usd=int(v[:,2].sum()),
                    net_340_usd=int(v[:,3].sum())) for m,v in models.items()
                    if m in ENTRY_THRESHOLDS or any(m.startswith(k+":hurdle_") for k in ENTRY_THRESHOLDS)},
        versus_previous=dict(net_increment_usd=int(delta.sum()),paired_mean_95_usd=block_bootstrap_mean(delta),
                             changes_by_release=changes),
        extension_vs_original_exit=dict(candidate=EXTENSION,original=WEIGHTED,net_increment_usd=int(exit_delta.sum()),
            paired_mean_95_usd=block_bootstrap_mean(exit_delta),
            contribution_by_release={day:int(v) for day,v in zip(releases,exit_delta,strict=True) if v}),
        selection_vs_cash=[adjusted_max(models,np.zeros(len(releases)),primary,b) for b in (4,8,13)],
        selection_vs_carry=[adjusted_max(models,series[CONTROL][:,2],primary,b) for b in (4,8,13)],
        selection_vs_matched_carry=[adjusted_max(models,series[MATCHED_CURVE][:,2],primary,b) for b in (4,8,13)],
        volatility_regimes=summary)
    export(f"{STEM}_regimes",rows)
    payload = json.dumps(report,indent=2)+"\n"
    write(DERIVED/f"{STEM}_summary.json",payload.encode())
    print(payload,end="")


if __name__ == "__main__":
    main()
