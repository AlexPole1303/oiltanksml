"""Use the closed pre-entry spread path to update the frozen EIA trade forecasts."""

import json
import math
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta

import numpy as np

from backtest_wti_cushing_spread import NEW_YORK, settlement, tick_price
from build_cushing_holdout_release_index import price_rows
from build_s1_reference_matches import DERIVED, export, read
from build_weekly_shadow_panel import SOURCE, read_only
from collect_public import write
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES, PUBLIC
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import fit_correction
from fit_cushing_volatility_neighbors import STEM as WEIGHTED

STEM = "cushing_cl_reaction_residual"
SOURCES = ("archive", "ibkr")
CONTROLS = {"satellite": f"{SUPPLEMENT}:volatility", "public": f"{PUBLIC}:volatility"}
METHODS = ("original", "release_reaction", "first_session", "split_reaction_and_surprise")


def next_day(mark):
    day = mark.astimezone(NEW_YORK) + timedelta(days=1)
    return day.replace(hour=0, minute=0, second=0, microsecond=0)


def signals():
    cases = [r for r, _ in inputs()]
    clocks = indexed("eia_release_clock")
    forecast = indexed(f"{OUTER}_forecasts")
    controls = {(r["release_date"], r["method"]): r for r in read(f"{WEIGHTED}_trades") if r["method"] in CONTROLS.values()}
    event = defaultdict(dict)
    for r in read("wti_cushing_event_bars"):
        if r["session_number"] == "1":
            event[r["release_date"]][r["leg"]] = r
    end = max(legs["near"]["session_date"] for legs in event.values())
    with closing(read_only(SOURCE)) as db:
        price = {source: {(d, month): tick_price(value) for d, month, _, value in price_rows(db, source, numeric_through=end)}
                 for source in SOURCES}
        previous = {r["release_date"]: db.execute("SELECT max(session_date) FROM wti_session_basis WHERE session_date<? "
                    "AND full_cmes_session=1 AND cme_no_settlement_notice_url IS NULL", (r["release_date"],)).fetchone()[0]
                    for r in cases}
    for r in cases:
        issue, decision = r["release_date"], r["decision"]
        if (issue, CONTROLS["satellite"]) not in controls:
            continue
        f = forecast[issue]
        cutoff, known = (datetime.fromisoformat(f[k]) for k in ("cutoff_utc", "known_utc"))
        if (not cutoff < known <= decision or float(f["actual"]) != r["x"][0]
                or f[f"{SUPPLEMENT}_latest_label_utc"] and datetime.fromisoformat(f[f"{SUPPLEMENT}_latest_label_utc"]) >= cutoff):
            raise ValueError(f"Inventory vintage differs on {issue}")
        refs = {m: controls[issue, key] for m, key in CONTROLS.items()}
        for q in refs.values():
            if (any(q[k] != r[k] for k in ("near_month", "far_month", "study_phase"))
                    or any(datetime.fromisoformat(q[k]) != r[field] for k, field in (
                        ("decision_at_utc", "decision"), ("entry_at_utc", "entry"), ("exit_at_utc", "exit_at")))
                    or any(datetime.fromisoformat(q[k]) >= decision for k in ("latest_training_exit_utc", "latest_calibration_exit_utc") if q[k])):
                raise ValueError(f"Invalid archived trading decision on {issue}")
        legs = event[issue]
        months = (r["near_month"], r["far_month"])
        if [legs[k]["delivery_month"] for k in ("near", "far")] != list(months) or len({v["settlement_reference_time"] for v in legs.values()}) != 1:
            raise ValueError(f"First session uses a different contract or clock on {issue}")
        prior_day, first_day = previous[issue], legs["near"]["session_date"]
        first_mark = datetime.fromisoformat(legs["near"]["settlement_reference_time"])
        prior_mark, report_mark = map(settlement, (prior_day, issue))
        updated_at = next_day(first_mark)
        if not prior_mark < report_mark < first_mark < updated_at < r["entry"]:
            raise ValueError(f"Invalid pre-entry mark ordering on {issue}")
        report_time = clocks[issue]["scheduled_table_release_at_utc"]
        # Day-only releases may use the change through the first full session,
        # but cannot identify the mark on publication day as post-report.
        report_eligible = bool(report_time and prior_mark < datetime.fromisoformat(report_time) < report_mark < decision)
        release = quotes(price, months, (prior_day, issue), SOURCES) if report_eligible else None
        split = quotes(price, months, (prior_day, issue, first_day), SOURCES) if report_eligible else None
        two = (split[0], [split[1][0], split[1][-1]]) if split else quotes(price, months, (prior_day, first_day), SOURCES)
        yield r | dict(known=known, refs=refs, forecast=f, updated_at=updated_at,
                       prior_mark=prior_mark, report_mark=report_mark, first_mark=first_mark, release=release, split=split, two=two)


def quotes(get, months, days, sources):
    # Quote identities and their raw references are validated when building
    # the book. Use one provider for all marks that make up a feature.
    for source in sources:
        keys = [(day, month) for day in days for month in months]
        if all(k in get[source] for k in keys):
            observed = [get[source][k] for k in keys]
            spread = [10 * (a - b) for a, b in zip(observed[::2], observed[1::2], strict=True)]
            return source, spread
    return None


def replay():
    histories = defaultdict(list)
    for r in signals():
        sigma = r["volatility_usd"]
        report_pair = r["release"][1] if r["release"] else None
        pieces = r["split"][1] if r["split"] else None
        pair = r["two"][1] if r["two"] else None
        clip = lambda value: float(np.clip(value / sigma, -3., 3.))
        release_move, first_step = (clip(pieces[1] - pieces[0]), clip(pieces[2] - pieces[1])) if pieces else (None, None)
        early_move = clip(report_pair[1] - report_pair[0]) if report_pair else None
        total_move = clip(pair[1] - pair[0]) if pair else None
        for kind, ref in r["refs"].items():
            value = float(ref["expected_long_move_usd"])
            prediction = clip(value)
            expectation = float(r["forecast"][SUPPLEMENT if kind == "satellite" else PUBLIC])
            surprise = float(np.clip((r["x"][0] - expectation) / .8, -3., 3.))
            designs = {
                "release_reaction": (early_move, early_move * abs(prediction)) if report_pair else None,
                "first_session": (total_move, total_move * abs(prediction)) if pair else None,
                "split_reaction_and_surprise": (release_move, first_step, surprise * abs(release_move), prediction * abs(first_step))
                                                if pieces else None,
            }
            for m in METHODS:
                design = designs.get(m)
                update = m not in ("original", "release_reaction") and design is not None
                decision = r["updated_at"] if update else r["decision"]
                history = histories[kind, m]
                past = [q for q in history if q["exit_at"] < decision][-HISTORY:]
                delta = 0.
                if design is not None:
                    row = dict(cutoff=decision, exit_at=r["exit_at"], design=design,
                               residual=(r["long_gross"]-value)/sigma)
                    delta = fit_correction(past, row, "design", "residual", penalty=(.18, .35, .55, .65)[:len(design)],
                                           loss_scale=1., bound=.35)
                    history.append(row)
                stamp = datetime.fromisoformat(ref["latest_training_exit_utc"])
                calibration = datetime.fromisoformat(ref["latest_calibration_exit_utc"]) if ref["latest_calibration_exit_utc"] else None
                if design is not None and past:
                    stamp = max(stamp, past[-1]["exit_at"])
                estimate = value + sigma * delta
                timed = r | dict(decision=decision)
                result = trade(timed, m, estimate, 3, training_at=stamp, calibration_at=calibration)
                if m == "original" and (result["near_side"] != int(ref["near_side"]) or result["gross_usd"] != int(ref["gross_usd"])):
                    raise ValueError(f"Frozen market payoff failed on {r['release_date']}/{kind}")
                latest_mark = r["first_mark"] if update else r["report_mark"] if design is not None else None
                if latest_mark is not None and not latest_mark < decision < r["entry"]:
                    raise ValueError(f"Pre-entry feature was unavailable on {r['release_date']}/{m}")
                source = {"release_reaction": r["release"], "first_session": r["two"],
                          "split_reaction_and_surprise": r["split"]}.get(m)
                release_values = report_pair if m == "release_reaction" else pieces if m == "split_reaction_and_surprise" else None
                yield result | dict(model=kind, original_prediction_usd=value,
                    latest_feature_mark_utc=latest_mark.isoformat() if latest_mark else None,
                    prior_feature_mark_utc=r["prior_mark"].isoformat() if latest_mark else None,
                    reaction_source=source[0] if design is not None else None,
                    reaction_close_usd=release_values[1] - release_values[0] if release_values else None,
                    first_step_usd=pieces[2] - pieces[1] if pieces and m == "split_reaction_and_surprise" else None,
                    preentry_move_usd=source[1][-1] - source[1][0] if source and update else None,
                    latest_original_decision_utc=r["decision"].isoformat(), realized_long_move_usd=r["long_gross"],
                    net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def report(rows):
    group = defaultdict(list)
    for r in rows:
        group[r["study_phase"], r["model"], r["method"]].append(r)
    phases = {}
    for phase in PHASES:
        section = phases[phase] = {}
        for kind in CONTROLS:
            base = group[phase, kind, "original"]
            for m in METHODS:
                arm = group[phase, kind, m]
                if not arm or [r["release_date"] for r in arm] != [r["release_date"] for r in base]:
                    raise ValueError(f"Trading samples differ on {phase}/{kind}/{m}")
                pnl = [r["net_170_usd"] - q["net_170_usd"] for r, q in zip(arm, base, strict=True)]
                error = np.array([abs(r["realized_long_move_usd"] - r["expected_long_move_usd"]) for r in arm])
                reference = np.array([abs(r["realized_long_move_usd"] - r["expected_long_move_usd"]) for r in base])
                section[f"{kind}:{m}"] = robustness(arm) | daily_risk(arm) | dict(
                    net_340_usd=sum(r["net_340_usd"] for r in arm), matched_reaction_windows=sum(bool(r["reaction_source"]) for r in arm),
                    price_move_mae_usd=float(error.mean()), paired_error_change_95_usd=block_bootstrap_mean(error - reference),
                    paired_delta_170_usd=sum(pnl), paired_pnl_change_95_usd=block_bootstrap_mean(pnl),
                    changed_sides=sum(r["near_side"] != q["near_side"] for r, q in zip(arm, base, strict=True)))
    return dict(status="exploratory_on_reviewed_history", phases=phases)


def main():
    rows = list(replay())
    result = report(rows)
    export(f"{STEM}_trades", rows)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    for phase, section in result["phases"].items():
        print(phase, {m: (r["matched_reaction_windows"], r["trades"], r["net_170_usd"], r["net_340_usd"])
                      for m, r in section.items()})


if __name__ == "__main__":
    main()
