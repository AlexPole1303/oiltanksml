"""Forecast Cushing stocks with dated regional imports and refinery inputs."""

import argparse
import json
import math
import re
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pdfplumber
from threadpoolctl import threadpool_limits

from audit_cushing_holdout_eia_vintages import ARCHIVE, archive, report_date
from build_s1_reference_matches import DERIVED, export, read
from collect_public import RAW, ROOT, write
from fit_cushing_cl_conditional_response import conditional_residual, features
from fit_cushing_cl_execution_windows import robustness
from fit_cushing_cl_exit_horizon import inputs as market_inputs, trade
from fit_cushing_cl_regime_strategy import HISTORY, MIN_WINDOWS
from fit_cushing_delayed_calendar import daily_risk
from fit_cushing_inventory_seasonal_residual import BASE, EXTENDED, TERMINALS, inputs as inventory_inputs
from fit_cushing_nowcast_ensemble import block_bootstrap_mean
from fit_cushing_observation_operator import PHASES
from fit_cushing_outer_capture_nowcast import STEM as OUTER, SUPPLEMENT
from fit_cushing_public_state import released_balances
from fit_cushing_regional_balance import replace_expectation
from fit_cushing_robust_stack import indexed
from fit_cushing_storage_state import fit_correction
from fit_cushing_volatility_neighbors import STEM as WEIGHTED, volatility_forecast
from register_cushing_leading_models import LINEAGE, RULE, STEM as LEADING, checked_hashes
from register_cushing_prerelease_forecast import digest, freeze, score as score_release
from register_cushing_public_fallback import latest_forward_balances

STEM = "cushing_regional_imports"
REF = "reference"
ARMS = {"import_shocks": (0, 1, 2, 3), "import_minus_runs": (4, 5, 6, 7),
        "regional_balance": (4, 5, 6, 7, 8, 9)}
METHODS = (REF, *ARMS)
LOOKBACK = 208
REPORT_METHODS = ("release_imports", "release_imports_and_runs")
MARKET_METHODS = (*METHODS, *REPORT_METHODS)


def import_row(report, directory):
    issue, week = (report[k] for k in ("issue", "week"))
    rows, stubs, source = archive(issue, 7, directory=directory)
    dates = tuple(map(report_date, rows[0][1:3]))
    prior = (date.fromisoformat(week) - timedelta(days=7)).isoformat()
    if stubs != 1 or rows[0][3] != "Difference":
        raise ValueError(f"Unexpected weekly import table for {issue}")
    if dates != (week, prior):
        clock = indexed("eia_release_clock").get(issue, {})
        if (clock.get("timing_basis") != "archived_pdf_day_only" or dates[0] >= week
                or dates[1] >= dates[0]):
            raise ValueError(f"Wrong survey week in import table for {issue}")
        path = (ROOT / source["source_path"]).with_suffix(".pdf")
        url = source["source_url"].replace("/csv/", "/pdf/").removesuffix(".csv") + ".pdf"
        if path.with_suffix(".pdf.source").read_text().strip() != url:
            raise ValueError(f"Unexpected source of replacement report for {issue}")
        with pdfplumber.open(path) as pdf:
            lines = pdf.pages[0].extract_text().splitlines()
        header, = [line for line in lines[:8] if re.match(r"\d+/\d+/\d+ ", line)]
        if tuple(map(report_date, header.split()[:2])) != (week, prior):
            raise ValueError(f"The original PDF has incorrect reporting dates for {issue}")
        start, = [i for i, line in enumerate(lines) if line.startswith("Crude Oil Net Imports")]
        if not lines[start + 1].startswith("Commercial"):
            raise ValueError(f"The replacement PDF does not have commercial imports for {issue}")
        rows = [re.fullmatch(label + r"\.{4,}\s*([\d,]+) .+", lines[start + offset])
                for label, offset in ((r"Midwest \(PADD 2\)\s*", 3), (r"Gulf Coast \(PADD 3\)\s*", 4))]
        if not all(rows):
            raise ValueError(f"The regional import labels in the PDF are ambiguous for {issue}")
        values = np.array([int(r[1].replace(",", "")) for r in rows], dtype=float) / 1000
        return report | dict(source_path=str(path.relative_to(ROOT)), source_url=url, source_sha256=digest(path),
                             table_7_week=week, imports=values, status="archived_pdf")
    labels = [r[0].strip() for r in rows]
    sections = [i for i, label in enumerate(labels) if label.startswith("Crude Oil Net Imports")]
    section, = sections
    expected = ["Commercial", "East Coast (PADD 1)", "Midwest (PADD 2)", "Gulf Coast (PADD 3)"]
    if labels[section + 1:section + 5] != expected:
        raise ValueError(f"Could not identify commercial crude imports on {issue}")
    cells = rows[section + 3:section + 5]
    values = np.array([int(row[1].replace(",", "")) for row in cells], dtype=float) / 1000
    return report | source | dict(table_7_week=dates[0], imports=values, status="available")


def history_features(reports, cutoff, week):
    dated = [r for r in reports if r["known"] < cutoff and r["week"] < week]
    latest = dated[-1]
    expected = (date.fromisoformat(week) - timedelta(days=7)).isoformat()
    if latest["week"] != expected:
        return None, latest
    reference = dated[:-1][-26:]
    if len(reference) < 22:
        return None, latest

    def flows(r):
        return np.r_[r["imports"], r["imports"] - r["current"][3:5]]

    current = flows(latest)
    medium = np.median(np.array([flows(r) for r in reference]), axis=0)
    drift = current - medium
    jump = current - np.median(np.array([flows(r) for r in reference[-4:]]), axis=0)
    # PADD 2 excludes Cushing here. The residual reflects everything the
    # two measured weekly flows do not explain, including inter-PADD flows.
    unexplained = latest["delta"][1:3] / 7 - current[2:]
    prior_unexplained = [r["delta"][1:3] / 7 - flows(r)[2:] for r in reference]
    balance_shock = unexplained - np.median(prior_unexplained, axis=0)
    return np.clip(np.r_[drift[:2], jump[:2], drift[2:], jump[2:], balance_shock] / .4, -3, 3), latest


def estimates(row, history):
    eligible = [r for r in history if r["known"] < row["cutoff"] and r["flow_features"] is not None][-LOOKBACK:]
    signal = row["flow_features"]
    results = {}
    for target in (BASE, SUPPLEMENT):
        base = row[target]
        results[target] = {REF: base}
        for name, columns in ARMS.items():
            delta = 0.
            if signal is not None and len(eligible) >= MIN_WINDOWS:
                fields = list(columns)
                sample = [p | dict(design=p["flow_features"][fields], residual=p["actual"] - p[target]) for p in eligible]
                delta = fit_correction(sample, row | dict(design=signal[fields]), "design", "residual",
                                       penalty=(.24,) * len(columns), loss_scale=.75, bound=.30)
            results[target][name] = base + delta
    return results, eligible


def historical(reports):
    panel = []
    for q in inventory_inputs():
        x, prior = history_features(reports, q["cutoff"], q["week_ending"])
        row = q | dict(flow_features=x)
        result, eligible = estimates(row, panel)
        panel.append(row)
        yield row | dict(estimates=result, training_count=len(eligible), prior_issue=prior["issue"],
                         prior_week=prior["week"], prior_known_at=prior["known"], prior_status=prior["status"],
                         table_7_week=prior["table_7_week"], table_7_sha256=prior["source_sha256"],
                         source_path=prior["source_path"], latest_calibration_at=eligible[-1]["known"] if eligible else None)


def release_market_features(reports, panel):
    results = {}
    for r in reports:
        row = panel.get(r["issue"])
        if row is None:
            continue
        if r["known"] != row["known"] or r["week"] != row["week_ending"]:
            raise ValueError(f"Mismatched Table 7 release clock for {r['issue']}")
        prior = [q for q in reports if q["known"] < row["cutoff"] and q["week"] < r["week"]][-8:]
        if len(prior) == 8:
            levels = np.r_[r["imports"], r["current"][3:5]]
            normal = np.median([np.r_[q["imports"], q["current"][3:5]] for q in prior], axis=0)
            results[r["issue"]] = (np.clip((levels - normal) / .4, -3, 3), prior[-1]["known"], r["source_sha256"])
    return results


def market(forecasts, reports):
    book = {r["release_date"]: r for r in forecasts}
    release_features = release_market_features(reports, book)
    anchors = {r["release_date"]: r for r in read(f"{WEIGHTED}_trades") if r["method"] == f"{SUPPLEMENT}:volatility"}
    history = defaultdict(list)
    corrections = []
    for case, _ in market_inputs():
        issue, decision = (case[k] for k in ("release_date", "decision"))
        r = book[issue]
        if not r["cutoff"] < r["known"] <= decision < case["entry"] or r["actual"] != case["x"][0]:
            raise ValueError(f"The inventory and market clocks do not match on {issue}")
        for method, expected in r["estimates"][SUPPLEMENT].items():
            design = replace_expectation(case, expected)
            current = case | dict(x=design, response_features=features(case["x"][2], case["volatility_usd"],
                                                                        r["actual"] - expected))
            settled = [p for p in history[method] if p["exit_at"] < decision][-HISTORY:]
            history[method].append(current)
            if len(settled) < MIN_WINDOWS:
                continue
            point = volatility_forecast(settled, current)
            calibration = [p for p in settled if "base_estimate" in p]
            correction, _ = conditional_residual(calibration, current) if len(calibration) >= MIN_WINDOWS else (0., 0.)
            current["base_estimate"] = point
            value = point + correction
            result = trade(case, method, value, 3, training_at=settled[-1]["exit_at"],
                           calibration_at=calibration[-1]["exit_at"] if calibration else None)
            if method == REF:
                a = anchors[issue]
                if (any(a[k] != result[k] for k in ("near_month", "far_month", "entry_at_utc", "exit_at_utc"))
                        or not math.isclose(value, float(a["expected_long_move_usd"]), abs_tol=1e-8, rel_tol=0)
                        or any(result[k] != int(a[k]) for k in ("near_side", "gross_usd", "net_170_usd"))):
                    raise ValueError(f"Historical leader did not reproduce on {issue}")
            yield result | trade_metadata(r, case, result, expected)
            if method != REF:
                continue
            z, prior_known, sha = release_features[issue]
            if not prior_known < r["cutoff"]:
                raise ValueError(f"The import surprise baseline is late on {issue}")
            sigma = case["volatility_usd"]
            context = dict(cutoff=decision, known=case["exit_at"], imports=z[:2], inputs=z,
                           residual=(case["long_gross"] - value) / sigma)
            prior = [q for q in corrections if q["known"] < decision][-HISTORY:]
            corrections.append(context)
            for name, label in zip(REPORT_METHODS, ("imports", "inputs"), strict=True):
                delta = sigma * fit_correction(prior, context, label, "residual", penalty=(.28,) * len(context[label]),
                                               loss_scale=1., bound=.30)
                stamp = prior[-1]["known"] if prior else None
                arm = trade(case, name, value + delta, 3, training_at=settled[-1]["exit_at"], calibration_at=stamp)
                yield arm | trade_metadata(r, case, arm, expected, correction=delta, source_sha=sha,
                                           latest_report=prior_known)


def trade_metadata(row, case, result, expected, correction=0., source_sha=None, latest_report=None):
    return dict(correction_to_inventory_kbbl=1000 * (expected - row[SUPPLEMENT]),
                correction_to_market_usd=correction, known_at_market_utc=row["known"] if source_sha else None,
                report_sha256=source_sha, latest_prior_report_known_utc=latest_report,
                prior_wpsr_issue=row["prior_issue"], realized_long_move_usd=case["long_gross"],
                net_340_usd=result["gross_usd"] - 340 * abs(result["near_side"]))


def summarize(forecasts, trades):
    report = dict(status="exploratory_on_reviewed_history", phases={})
    for phase in PHASES:
        rows = [r for r in forecasts if r["study_phase"] == phase]
        inventory = {}
        for target in (BASE, SUPPLEMENT):
            actual = np.array([r["actual"] for r in rows])
            losses = {m: 1000 * abs(np.array([r["estimates"][target][m] for r in rows]) - actual) for m in METHODS}
            inventory[target] = {m: dict(mae_kbbl=float(e.mean()), delta_vs_reference_95_kbbl=block_bootstrap_mean(e-losses[REF]))
                                 for m, e in losses.items()}
        groups = {m: [r for r in trades if r["study_phase"] == phase and r["method"] == m] for m in MARKET_METHODS}
        pnl = {}
        for method, group in groups.items():
            pairs = list(zip(group, groups[REF], strict=True))
            if any(a["release_date"] != b["release_date"] for a, b in pairs):
                raise ValueError(f"The trade cohorts differ for {phase}/{method}")
            pnl[method] = robustness(group) | daily_risk(group) | dict(
                net_340_usd=sum(r["net_340_usd"] for r in group),
                paired_net_170_delta_95_usd=block_bootstrap_mean([a["net_170_usd"]-b["net_170_usd"] for a, b in pairs]),
                changed_sides=sum(a["near_side"] != b["near_side"] for a, b in pairs),
                mean_absolute_error_usd=float(np.mean([abs(r["expected_long_move_usd"] - r["realized_long_move_usd"])
                                                       for r in group])))
        report["phases"][phase] = dict(releases=len(rows), releases_with_flows=sum(r["flow_features"] is not None for r in rows),
                                      inventory=inventory, trading=pnl)
        print(phase, {m: round(inventory[BASE][m]["mae_kbbl"], 2) for m in METHODS},
              {m: (pnl[m]["net_170_usd"], pnl[m]["net_340_usd"]) for m in MARKET_METHODS})
    return report


def forward(issue):
    prefix = DERIVED / f"{STEM}_{issue}"
    path = prefix.with_name(f"{prefix.name}_registration.json")
    original_path = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    leader_path = DERIVED / f"{LEADING}_{issue}_registration.json"
    extended_path = DERIVED / f"{EXTENDED}_{issue}_registration.json"
    original, leader, extended = (json.loads(p.read_bytes()) for p in (original_path, leader_path, extended_path))
    cutoff, deadline = (datetime.fromisoformat(original[k]) for k in LINEAGE[-2:])
    for p in (leader, extended):
        checked_hashes(p["sources_sha256"])
        if any(p[k] != original[k] for k in LINEAGE) or p["measured_roofs"] != leader["measured_roofs"]:
            raise ValueError("Inherited forecast clocks or optical panel differ")
    if not path.exists():
        week = original["week_ending"]
        if not cutoff <= datetime.now(timezone.utc) < deadline or issue in indexed("eia_wpsr_forward_releases"):
            raise ValueError("The import-model forecasts must be recorded before publication")
        known, evidence = latest_forward_balances(cutoff, week, original["prior_release"])
        older = [import_row(r, ARCHIVE) for r in released_balances() if r["known"] < cutoff and r["week"] < week]
        recent = [import_row(r, RAW / "eia/wpsr_forward") for r in known]
        if older[-1]["week"] >= recent[0]["week"]:
            raise ValueError("Archived and forward regional flows overlap")
        x, previous = history_features(older + recent, cutoff, week)
        if x is None:
            raise ValueError("No verified regional imports for the previous survey week")
        training = list(historical(older))
        row = dict(cutoff=cutoff, flow_features=x,
                   **{BASE: extended["estimates_kbbl"][BASE] / 1000, SUPPLEMENT: leader["estimates_kbbl"][SUPPLEMENT] / 1000})
        values, past = estimates(row, training)
        for r in older + recent:
            evidence[r["source_path"]] = r["source_sha256"]
        sources = (original_path, leader_path, extended_path, Path(__file__).resolve(), DERIVED / "eia_release_clock.csv",
                   DERIVED / "eia_control_vintage.csv", DERIVED / f"{EXTENDED}_inventory.csv",
                   DERIVED / f"{OUTER}_forecasts.csv", DERIVED / f"{TERMINALS}_forecasts.csv",
                   ROOT / "scripts/fit_cushing_storage_state.py", ROOT / "scripts/fit_cushing_inventory_seasonal_residual.py")
        evidence.update({str(p.relative_to(ROOT)): digest(p) for p in sources})
        stamp = datetime.now(timezone.utc)
        if not (past[-1]["known"] < cutoff <= stamp < deadline and previous["known"] < cutoff):
            raise ValueError("The prerelease model missed its deadline")
        freeze(path, {k: original[k] for k in LINEAGE} | dict(recorded_at_utc=stamp.isoformat(),
            estimates_kbbl={target: {m: 1000 * v for m, v in result.items()} for target, result in values.items()},
            measured_roofs=leader["measured_roofs"], market_rule=RULE, previous_wpsr_issue=previous["issue"],
            previous_wpsr_known_at=previous["known"].isoformat(), latest_training_at=past[-1]["known"].isoformat(),
            training_count=len(past), flow_features=x.tolist(), sources_sha256=evidence))
    record = json.loads(path.read_bytes())
    checked_hashes(record["sources_sha256"])
    if (any(record[k] != original[k] for k in LINEAGE) or record["market_rule"] != RULE
            or record["previous_wpsr_issue"] != original["prior_release"]
            or record["measured_roofs"] != leader["measured_roofs"]
            or set(record["estimates_kbbl"]) != {BASE, SUPPLEMENT}
            or any(set(r) != set(METHODS) for r in record["estimates_kbbl"].values())
            or not cutoff <= datetime.fromisoformat(record["recorded_at_utc"]) < deadline
            or any(datetime.fromisoformat(record[k]) >= cutoff for k in ("latest_training_at", "previous_wpsr_known_at"))):
        raise ValueError("The saved regional forecast failed the prerelease checks")
    outcome = score_release(original_path, original)
    result = dict(release_date=issue, status=outcome["status"], registration_sha256=digest(path),
                  estimates_kbbl=record["estimates_kbbl"])
    if outcome["status"] == "scored_first_release":
        actual = outcome["actual_eia_change_kbbl"]
        result.update(actual_eia_change_kbbl=actual, first_release_sha256=outcome["first_release_sha256"],
                      absolute_error_kbbl={target: {m: abs(v - actual) for m, v in estimates.items()}
                                           for target, estimates in record["estimates_kbbl"].items()})
    write(prefix.with_name(f"{prefix.name}_score.json"), (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(result)


def main():
    with threadpool_limits(limits=1):
        reports = [import_row(r, ARCHIVE) for r in released_balances()]
        forecasts = list(historical(reports))
        trades = list(market(forecasts, reports))
    result = summarize(forecasts, trades)
    export(f"{STEM}_source_panel", [{k: r[k] for k in ("issue", "week", "known", "status", "source_url",
                                                        "source_sha256", "source_path")} |
           dict(padd2_crude_import_kbbl_day=round(1000 * r["imports"][0]), padd3_crude_import_kbbl_day=round(1000 * r["imports"][1]),
                padd2_crude_refiner_inputs_kbbl_day=round(1000 * r["current"][3]),
                padd3_crude_refiner_inputs_kbbl_day=round(1000 * r["current"][4])) for r in reports])
    export(f"{STEM}_forecasts", [{k: v for k, v in r.items() if k not in ("flow_features", "estimates")} |
           dict(features=json.dumps(r["flow_features"].tolist()) if r["flow_features"] is not None else None) |
           {f"{name}:{m}": v for name, values in r["estimates"].items() for m, v in values.items()} for r in forecasts])
    export(f"{STEM}_trades", trades)
    write(DERIVED / f"{STEM}_summary.json", (json.dumps(result, indent=2, allow_nan=False) + "\n").encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=date.fromisoformat)
    args = parser.parse_args()
    if args.release:
        forward(args.release.isoformat())
    else:
        main()
