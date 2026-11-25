"""Test dated storage-state corrections to Cushing forecasts and CL spread payoffs."""

import json
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits

from audit_cushing_optical_attribution import replay_conditional_market
from build_s1_reference_matches import DERIVED, export, read
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_windows, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_late_pair_recovery import COMBINED, STEM as INVENTORY_STEM
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_optical_reliability import trading as base_trades
from fit_cushing_outer_capture_nowcast import STEM as MARKET_STEM, SUPPLEMENT
from fit_cushing_public_state import released_balances
from fit_cushing_robust_stack import indexed

STEM = "cushing_storage_state"
FORECASTS = (PUBLIC, COMBINED, "public_storage", "hybrid_storage", "hybrid_storage_and_optical",
             "hybrid_storage_and_region")
REFERENCES = {"public_reference": PUBLIC, "satellite_reference": SUPPLEMENT}
MARKET_FEATURES = {
    "public_storage": (PUBLIC, "storage", "public_error"),
    "satellite_storage": (SUPPLEMENT, "storage", "satellite_error"),
    "public_storage_and_surprise": (PUBLIC, "public_interactions", "public_error"),
    "satellite_storage_and_surprise": (SUPPLEMENT, "satellite_interactions", "satellite_error"),
    "satellite_optical_revision": (SUPPLEMENT, "satellite_optical", "satellite_error"),
    "satellite_local_surprise": (SUPPLEMENT, "regional", "satellite_error"),
    "satellite_local_surprise_and_storage": (SUPPLEMENT, "regional_with_storage", "satellite_error"),
}
REFITS = (COMBINED, "hybrid_storage", "hybrid_storage_and_optical", "hybrid_storage_and_region")
TRADE_CONTEXT = ("previous_stocks_million_bbl", "previous_report_known_utc", "public_surprise_million_bbl",
                 "satellite_surprise_million_bbl", "optical_revision_million_bbl", "rest_of_padd2_surprise_million_bbl",
                 "localized_surprise_million_bbl")


def fit_correction(history, current, field, target, *, penalty, loss_scale, bound):
    if len(history) < MIN_WINDOWS:
        return 0.
    design = np.asarray([q[field] for q in history])
    residual = np.asarray([q[target] for q in history])
    age_weeks = np.asarray([(current["cutoff"]-q["cutoff"]).total_seconds()/604800 for q in history])
    recency = np.exp2(-age_weeks/78.)
    recency /= recency.sum()
    # Fixed physical scales and a zero intercept keep this a correction to
    # the archived forecast. Shrink the additional slopes toward zero.
    regularizer = np.diag(penalty)
    beta = np.zeros(design.shape[1])
    for _ in range(6):
        weights = recency/np.hypot(1., (residual-design@beta)/loss_scale)
        beta = solve(design.T@(weights[:, None]*design)+regularizer, design.T@(weights*residual), assume_a="pos")
    return float(np.clip(np.asarray(current[field])@beta, -bound, bound))


def storage_coordinates(history):
    if len(history)<52:
        raise ValueError("Storage coordinates require 52 published weeks")
    last = history[-1]
    values = np.array([q["current"][0] for q in history[-52:]])
    level = float(values[-1])
    median = float(np.median(values))
    previous_13 = float(np.median(values[-13:]))
    low = float(np.clip(math.log(35./level), -1., 1.))
    relative = float(np.clip((median-level)/10., -2., 2.))
    short_relative = float(np.clip((previous_13-level)/5., -2., 2.))
    previous_levels = np.array([q["current"][1:3] for q in history[-52:]])
    region = np.clip((np.median(previous_levels,axis=0)-last["current"][1:3])/15., -2., 2.)
    design = (low,relative,short_relative)
    return dict(previous_stocks_million_bbl=level,trailing_52_week_median_million_bbl=median,
                trailing_13_week_median_million_bbl=previous_13,state=design,
                regional_state=(*design,float(region[0]),float(region[1]),float(region[0]-region[1])*relative))


def source_rows():
    vintage = list(released_balances())
    by_release = {q["issue"]:q for q in vintage}
    controls = indexed(f"{INVENTORY_STEM}_forecasts")
    outer = indexed(f"{MARKET_STEM}_forecasts")
    if controls.keys() != outer.keys():
        raise ValueError("Storage comparisons require identical inventory panels")
    for issue, raw in sorted(controls.items()):
        cutoff, known = (datetime.fromisoformat(raw[k]) for k in ("cutoff", "known"))
        week = raw["week_ending"]
        label = by_release[issue]
        o = outer[issue]
        history = [r for r in vintage if r["week"] < week and r["known"] < cutoff]
        if (len(history) < 52 or not cutoff < known == label["known"] or label["week"] != week
                or any(datetime.fromisoformat(o[k]) != v for k, v in (("cutoff_utc", cutoff), ("known_utc", known)))
                or o["week_ending"] != week or o["study_phase"] != raw["study_phase"]
                or float(o["actual"]) != float(raw["actual"])
                or not math.isclose(float(raw["actual"]), label["delta"][0], abs_tol=1e-10)
                or any(datetime.fromisoformat(q[k]) >= cutoff for q in (raw, o) for k in q
                       if q[k] and k.endswith(("latest_label_utc", "training_label_utc", "latest_optical_at_utc")))):
            raise ValueError(f"Incompatible EIA and optical clocks on {issue}")
        last = history[-1]
        coordinates = storage_coordinates(history)
        low,relative,_ = coordinates["state"]
        public = float(raw[PUBLIC])
        hybrid = float(raw[COMBINED])
        if public != float(o[PUBLIC]):
            raise ValueError(f"Public comparator differs on {issue}")
        direct = raw["measured"] == "True" or int(raw["late_roofs"]) > 0
        # On measured weeks the archived hybrid base already includes the
        # native optical correction. On gap weeks it is a public fallback.
        optical_anchor = public if raw["measured"] == "True" else float(raw["basis_when_unmeasured"])
        optical = hybrid-optical_anchor if direct else 0.
        # PADD 2 is already net of Cushing in released_balances(). The
        # refinery inputs are rates and are converted to weekly quantities.
        recent_deltas = np.array([q["delta"][1:5] for q in history[-8:]])
        regional_surprise = (label["delta"][1:5]-np.median(recent_deltas,axis=0))*np.array((1.,1.,7.,7.))
        regional_surprise = tuple(float(v) for v in np.clip(regional_surprise, -8., 8.))
        yield dict(release_date=issue, week_ending=week, study_phase=raw["study_phase"], cutoff=cutoff, known=known,
                   actual=float(raw["actual"]), previous_report=last["issue"], previous_report_known_utc=last["known"],
                   published_stocks_million_bbl=float(label["current"][0]),physical_revision_million_bbl=optical,
                   direct_optical=direct,optical_state=(*coordinates["state"],low*optical,relative*optical),
                   rest_of_padd2_surprise_million_bbl=regional_surprise[0],
                   padd3_surprise_million_bbl=regional_surprise[1],
                   padd2_refinery_surprise_million_bbl=regional_surprise[2],
                   padd3_refinery_surprise_million_bbl=regional_surprise[3],
                   public_residual=float(raw["actual"])-public, hybrid_residual=float(raw["actual"])-hybrid,
                   **coordinates, **{PUBLIC:public, COMBINED:hybrid, SUPPLEMENT:float(o[SUPPLEMENT])})


def regional_adjustment(history, row):
    return fit_correction(history,row,"regional_state","hybrid_residual",penalty=(.12,.24,.24,.35,.35,.55),
                          loss_scale=.65,bound=.55)


def inventory_forecasts():
    history = []
    specs = (("public_storage", PUBLIC, "state", "public_residual"),
             ("hybrid_storage", COMBINED, "state", "hybrid_residual"),
             ("hybrid_storage_and_optical", COMBINED, "optical_state", "hybrid_residual"))
    for row in source_rows():
        available = [q for q in history if q["known"] < row["cutoff"]][-HISTORY:]
        estimate = {name:row[base]+fit_correction(available, row, key, err, penalty=(.12, .24, .24, .35, .35, .55)[:len(row[key])],
                    loss_scale=.65, bound=.55) for name, base, key, err in specs}
        estimate["hybrid_storage_and_region"] = row[COMBINED]+regional_adjustment(available,row)
        history.append(row)
        yield row | estimate | dict(inventory_training_weeks=len(available),
            latest_inventory_training_utc=max((q["known"] for q in available), default=None))


def market_forecasts(forecasts):
    estimates = {q["release_date"]:q for q in forecasts}
    anchors = {m:{r["release_date"]:r for r in read(f"{MARKET_STEM}_trades") if r["method"] == f"{m}_conditional"}
               for m in (PUBLIC, SUPPLEMENT)}
    history = []
    for row, _ in market_windows():
        issue, cutoff, sigma = (row[k] for k in ("release_date", "decision", "volatility_usd"))
        if issue not in anchors[PUBLIC]:
            continue
        f = estimates[issue]
        if not f["cutoff"] < f["known"] <= cutoff < row["entry"] < row["exit_at"] or f["actual"] != row["x"][0]:
            raise ValueError(f"Inventory data unavailable on {issue}")
        reference = {}
        for model in (PUBLIC, SUPPLEMENT):
            q = anchors[model][issue]
            if (any(q[k] != row[k] for k in ("near_month", "far_month"))
                    or any(datetime.fromisoformat(q[k]) != row[field] for k, field in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))
                    or any(datetime.fromisoformat(q[k]) >= cutoff for k in ("settled_training_label_utc", "latest_calibration_exit_utc") if q[k])):
                raise ValueError(f"Different market or training clocks on {issue}/{model}")
            reference[model] = float(q["expected_long_move_usd"])
        low, relative, _ = f["state"]
        carry = float(np.clip(row["x"][2]/.5, -3., 3.))
        state = (low, relative, low*carry)
        raw_public = float(np.clip(f["actual"]-f[PUBLIC], -3., 3.))
        raw_satellite = float(np.clip(f["actual"]-f[SUPPLEMENT], -3., 3.))
        forecast_delta = f[SUPPLEMENT]-f[PUBLIC]
        pub_features = (*state, low*raw_public, relative*raw_public)
        sat_features = (*state, low*raw_satellite, relative*raw_satellite)
        optical_features = (*sat_features, low*forecast_delta, relative*forecast_delta)
        outside = f["rest_of_padd2_surprise_million_bbl"]/2.
        gulf = f["padd3_surprise_million_bbl"]/4.
        refinery = (f["padd2_refinery_surprise_million_bbl"]-f["padd3_refinery_surprise_million_bbl"])/4.
        # A larger same-direction change outside the hub reduces the share
        # of the published surprise that is specific to Cushing.
        localized = raw_satellite/(1.+max(0.,math.copysign(1.,raw_satellite)*outside))
        regional = (outside, gulf, refinery, localized-raw_satellite)
        current = row | dict(cutoff=cutoff, storage=state, public_interactions=pub_features, satellite_interactions=sat_features,
            satellite_optical=optical_features, regional=regional, regional_with_storage=(*regional,low*localized,relative*localized),
            public_error=(row["long_gross"]-reference[PUBLIC])/sigma,
            satellite_error=(row["long_gross"]-reference[SUPPLEMENT])/sigma)
        available = [q for q in history if q["exit_at"] < cutoff][-HISTORY:]
        history.append(current)
        choices = {name:reference[m] for name,m in REFERENCES.items()}
        for name, (baseline, field, target) in MARKET_FEATURES.items():
            adjustment = fit_correction(available, current, field, target, penalty=(1.,1.,1.3,1.1,1.3,1.1,1.3)[:len(current[field])],
                                        loss_scale=1., bound=.45)
            choices[name] = reference[baseline]+sigma*adjustment
        for method, expected in choices.items():
            result = trade(row, method, expected, 3, training_at=available[-1]["exit_at"] if available else None)
            if method in REFERENCES:
                original = anchors[REFERENCES[method]][issue]
                if (result["near_side"] != int(original["near_side"]) or result["gross_usd"] != int(original["gross_usd"])):
                    raise ValueError(f"Reference trade changed on {issue}/{method}")
            yield result | dict(previous_stocks_million_bbl=f["previous_stocks_million_bbl"],
                    previous_report_known_utc=f["previous_report_known_utc"], public_surprise_million_bbl=raw_public,
                    satellite_surprise_million_bbl=raw_satellite, optical_revision_million_bbl=forecast_delta,
                    rest_of_padd2_surprise_million_bbl=f["rest_of_padd2_surprise_million_bbl"],
                    localized_surprise_million_bbl=localized,
                    market_training_windows=len(available), net_340_usd=result["gross_usd"]-340*abs(result["near_side"]))


def score_forecasts(panel):
    if not panel:
        return dict(weeks=0, models={})
    loss = {key:np.asarray([1000*abs(r[key]-r["actual"]) for r in panel]) for key in FORECASTS}
    return dict(weeks=len(panel), models={m:dict(mae_kbbl=float(e.mean()),
        delta_vs_public_95_kbbl=block_bootstrap_mean(e-loss[PUBLIC]),
        delta_vs_hybrid_95_kbbl=block_bootstrap_mean(e-loss[COMBINED]),
        rmse_kbbl=float(np.sqrt(np.mean(e**2)))) for m,e in loss.items()})


def refit_trades(forecasts):
    base = list(base_trades(forecasts, (PUBLIC, *REFITS)))
    source = {r["release_date"]:r for r in read(f"{INVENTORY_STEM}_trades") if r["method"] == COMBINED}
    for row in replay_conditional_market(forecasts, base, REFITS, references={}):
        if row["method"] == COMBINED:
            archived = source[row["release_date"]]
            if (not math.isclose(float(archived["expected_long_move_usd"]), row["expected_long_move_usd"], abs_tol=1e-8)
                    or int(archived["near_side"]) != row["near_side"] or int(archived["gross_usd"]) != row["gross_usd"]):
                raise ValueError(f"Inventory refit control changed on {row['release_date']}")
        yield row | dict(method=f"refit_{row['method']}",
                         net_340_usd=row["gross_usd"]-340*abs(row["near_side"]))


def main():
    with threadpool_limits(limits=1):
        forecasts = list(inventory_forecasts())
        trades = list(market_forecasts(forecasts))
        market_context = {r["release_date"]:r for r in trades if r["method"] == "satellite_reference"}
        for r in refit_trades(forecasts):
            context = market_context[r["release_date"]]
            trades.append(r | {k:context[k] for k in TRADE_CONTEXT}
                          | dict(market_training_windows=None))
    report = dict(status="exploratory_on_reviewed_history", inventory={}, trading={})
    groups = defaultdict(list)
    for q in trades:
        groups[q["study_phase"],q["method"]].append(q)
    for phase in PHASES:
        panel = [q for q in forecasts if q["study_phase"] == phase]
        report["inventory"][phase] = {label:score_forecasts(sample) for label,sample in (
            ("all",panel), ("below_25_million",[q for q in panel if q["previous_stocks_million_bbl"] < 25]),
            ("with_optical_observations",[q for q in panel if q["direct_optical"]]))}
        control = groups[phase,"satellite_reference"]
        trading = report["trading"][phase] = {}
        for method in (*REFERENCES, *MARKET_FEATURES, *(f"refit_{name}" for name in REFITS)):
            sample = groups[phase,method]
            if not sample or [q["release_date"] for q in sample] != [q["release_date"] for q in control]:
                raise ValueError(f"Different dated CL cohort on {phase}/{method}")
            delta = [r["net_170_usd"]-b["net_170_usd"] for r,b in zip(sample,control,strict=True)]
            year = defaultdict(int)
            for r,diff in zip(sample,delta,strict=True):
                year[r["release_date"][:4]] += diff
            trading[method] = robustness(sample) | daily_risk(sample) | dict(net_340_usd=sum(r["net_340_usd"] for r in sample),
                delta_vs_satellite_by_year_usd=dict(year), delta_vs_satellite_per_release_95_usd=block_bootstrap_mean(delta),
                changed_entries=sum(r["near_side"]!=b["near_side"] for r,b in zip(sample,control,strict=True)))
        print(phase,{k:round(v["mae_kbbl"],2) for k,v in report["inventory"][phase]["all"]["models"].items()},
              {k:(v["trades"],v["net_170_usd"],v["net_340_usd"],v["daily_drawdown_usd"]) for k,v in trading.items()}, flush=True)
    fields = ("release_date", "week_ending", "study_phase", "cutoff", "known", "actual", "previous_report", "previous_report_known_utc",
              "previous_stocks_million_bbl", "published_stocks_million_bbl", "trailing_52_week_median_million_bbl", "trailing_13_week_median_million_bbl",
              "rest_of_padd2_surprise_million_bbl", "padd3_surprise_million_bbl",
              "padd2_refinery_surprise_million_bbl", "padd3_refinery_surprise_million_bbl",
              "physical_revision_million_bbl", "direct_optical", "inventory_training_weeks", "latest_inventory_training_utc", *FORECASTS)
    export(f"{STEM}_forecasts", [{k:q[k] for k in fields} for q in forecasts])
    export(f"{STEM}_trades", trades)
    write(DERIVED/f"{STEM}_summary.json", (json.dumps(report, indent=2, allow_nan=False)+"\n").encode())


if __name__ == "__main__":
    main()
