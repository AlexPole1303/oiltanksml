"""Estimate the contribution of national WPSR surprises to the dated CL spread."""

import json
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np

from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import trade
from fit_cushing_cl_payoff_scenarios import LEADER, PUBLIC_REFERENCE, market_rows
from fit_cushing_cl_regime_strategy import HISTORY
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import fit_correction

STEM = "cushing_cl_national_release"
STOCKS = ("us_commercial_crude", "cushing_crude", "us_gasoline_stocks", "us_distillate_stocks")
FLOWS = ("us_crude_net_imports_including_spr", "us_crude_refiner_inputs")
MODELS = ("reference", "outside_cushing_and_products", "outside_cushing_products_and_flows")


def reports():
    values = defaultdict(dict)
    for q in read("eia_control_vintage"):
        m = q["metric"]
        if m not in (*STOCKS, *FLOWS):
            continue
        key = q["release_date"], q["report_date"]
        expected_unit = "thousand_bbl" if m in STOCKS else "thousand_bbl_per_day"
        if q["unit"] != expected_unit or m in values[key]:
            raise ValueError(f"Invalid reporting unit or duplicate {key}/{m}")
        values[key][m] = float(q["value"]) / 1000
    history = []
    for issue, clock in sorted(indexed("eia_release_clock").items()):
        week = clock["latest_week_in_issue"]
        previous_week = (date.fromisoformat(week) - timedelta(days=7)).isoformat()
        now, prev = (values[issue, d] for d in (week, previous_week))
        known, cutoff = (datetime.fromisoformat(clock[k]) for k in (
            "conservative_postrelease_from_utc", "conservative_prerelease_before_utc"))
        levels = np.array([now[m] for m in (*STOCKS, *FLOWS)])
        change = levels - np.array([prev[m] for m in (*STOCKS, *FLOWS)])
        actual = change[1]
        # Strip the Cushing component from both the national change and its
        # historical expectation to avoid embedding the same release twice.
        change[0] -= actual
        levels[0] -= levels[1]
        prior = [r for r in history if r["known"] < cutoff and r["week"] < week][-8:]
        if not np.isfinite(change).all() or not cutoff < known:
            raise ValueError(f"Invalid national balance on {issue}")
        history.append(dict(week=week, known=known, change=change, levels=levels))
        if len(prior) < 8:
            continue
        stock_expectation = np.median([r["change"][:4] for r in prior], axis=0)
        flow_expectation = np.median([r["levels"][4:] for r in prior], axis=0)
        surprise = np.r_[(change[:4] - stock_expectation)[[0, 2, 3]], levels[4:] - flow_expectation]
        yield issue, dict(known=known, cutoff=cutoff, actual=actual, national_surprises=surprise,
                          predictors=tuple(np.clip(surprise / np.array((5., 3., 3., .9, .7)), -3., 3.)),
                          latest_prior_report_known=prior[-1]["known"])


def replay():
    book = dict(reports())
    history = defaultdict(list)
    for r in market_rows():
        issue, decision, sigma = (r[k] for k in ("release_date", "decision", "volatility_usd"))
        f = book[issue]
        if not (f["latest_prior_report_known"] < f["cutoff"] < f["known"] <= decision < r["entry"]
                and np.isclose(f["actual"], r["x"][0], atol=1e-10, rtol=0)):
            raise ValueError(f"Mismatched original WPSR time or value on {issue}")
        for kind, anchor in (("satellite", LEADER), ("public", PUBLIC_REFERENCE)):
            reference = r["anchors"][anchor]
            estimate = r["forecasts"][anchor]
            history_for_model = history[kind]
            past = [q for q in history_for_model if q["exit_at"] < decision][-HISTORY:]
            current = r | dict(cutoff=decision, residual=(r["long_gross"]-estimate)/sigma,
                stocks=f["predictors"][:3], full=f["predictors"])
            history_for_model.append(current)
            stock = fit_correction(past, current, "stocks", "residual", penalty=(.3, .4, .4), loss_scale=1., bound=.35)
            full = fit_correction(past, current, "full", "residual", penalty=(.3, .4, .4, .5, .5), loss_scale=1., bound=.35)
            original_stamp = datetime.fromisoformat(reference["latest_training_exit_utc"])
            calibration = reference["latest_calibration_exit_utc"]
            for method, factor in zip(MODELS, (0.,stock,full), strict=True):
                stamp = max(original_stamp, past[-1]["exit_at"]) if past and method != "reference" else original_stamp
                row = trade(r, method, estimate + sigma * factor, 3, training_at=stamp,
                            calibration_at=datetime.fromisoformat(calibration) if calibration else None)
                if method == "reference" and (row["near_side"] != int(reference["near_side"])
                                              or row["gross_usd"] != int(reference["gross_usd"])):
                    raise ValueError(f"The archive did not reproduce on {issue}/{kind}")
                yield row | dict(model=kind, original_expected_usd=estimate, national_correction_usd=sigma * factor,
                        wpsr_known_at=f["known"], latest_prior_report_known=f["latest_prior_report_known"],
                        realized_long_move_usd=r["long_gross"], net_340_usd=row["gross_usd"]-340*abs(row["near_side"])) | {
                            k:float(v) for k,v in zip(("outside_cushing_surprise_mmbbl", "gasoline_surprise_mmbbl",
                                "distillate_surprise_mmbbl", "net_imports_surprise_mmbbl_day", "refinery_surprise_mmbbl_day"),
                                f["national_surprises"], strict=True)}


def main():
    rows = list(replay())
    groups = defaultdict(list)
    for r in rows:
        groups[r["study_phase"], r["model"], r["method"]].append(r)
    summary = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        result = summary["phases"][phase] = {}
        for kind in ("satellite", "public"):
            base = groups[phase, kind, "reference"]
            for method in MODELS:
                arm = groups[phase, kind, method]
                if [r["release_date"] for r in arm] != [r["release_date"] for r in base]:
                    raise ValueError(f"Different market calendar on {phase}/{kind}/{method}")
                pairs = list(zip(arm, base, strict=True))
                delta = [r["net_170_usd"] - b["net_170_usd"] for r,b in pairs]
                err = [abs(r["realized_long_move_usd"]-r["expected_long_move_usd"]) for r in arm]
                base_err = [abs(r["realized_long_move_usd"]-r["expected_long_move_usd"]) for r in base]
                result[f"{kind}:{method}"] = robustness(arm) | daily_risk(arm) | dict(
                    net_340_usd=sum(r["net_340_usd"] for r in arm), net_delta_170_usd=sum(delta),
                    net_delta_per_window_95_usd=block_bootstrap_mean(delta), spread_mae_usd=float(np.mean(err)),
                    spread_mae_delta_95_usd=block_bootstrap_mean(np.array(err)-base_err),
                    changed_release_dates=[r["release_date"] for r,b in pairs if r["near_side"] != b["near_side"]])
        print(phase, {m:(s["trades"],s["net_170_usd"],s["net_340_usd"],round(s["spread_mae_usd"],2)) for m,s in result.items()})
    export(f"{STEM}_trades", rows)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(summary, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
