"""Rebuild the compact weekly neural-network panel from the project's dated EIA and optical records."""
import argparse
import hashlib
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from build_s1_reference_matches import read
from audit_cushing_spot_vintages import fetch, SERIES
from publication_clocks import EASTERN
from collect_public import RAW, ROOT
from fit_cushing_adaptive_nowcast import archived_training_rows
from fit_cushing_coverage_nowcast import coverage_inputs
from fit_cushing_inventory_network import DERIVED, OPTICAL, PANEL, PUBLIC, load_panel, save_json
from fit_cushing_network_calibration import BASIS, REFERENCE, calendar, past_errors
from fit_cushing_public_flows_nowcast import early_table
from fit_cushing_public_state import released_balances
from fit_cushing_robust_stack import indexed


def release_history():
    for r in archived_training_rows():
        issue = r["release_date"]
        if issue >= "2016-01-01":
            break
        values, _ = early_table(RAW / "eia/wpsr_training_history" / f"{issue.replace('-', '_')}_table4.csv",
                                4, issue, [r["week_ending"], r["previous_week_ending"]])
        levels = np.array([values[k] for k in ("cushing_crude", "padd2_commercial_crude", "padd3_commercial_crude")]) / 1000
        levels[1] -= levels[0]
        changes = levels[:, 0] - levels[:, 1]
        if not np.isclose(changes[0], r["target_million_bbl"]):
            raise ValueError(f"Original EIA inventory change disagrees on {issue}")
        yield (issue, r["week_ending"], datetime.fromisoformat(r["forecast_cutoff_utc"]),
               datetime.fromisoformat(r["target_known_by_utc"]), levels[0, 0], changes)
    clocks = indexed("eia_release_clock")
    for r in released_balances():
        cutoff = datetime.fromisoformat(clocks[r["issue"]]["conservative_prerelease_before_utc"])
        yield r["issue"], r["week"], cutoff, r["known"], r["current"][0], r["delta"][:3]


def public_features(past, week):
    """Use the last available releases. The age feature records gaps in reporting."""
    changes = np.array([r[5] for r in past[-53:]])
    last = changes[-1]
    y = changes[:, 0]
    # Shrink the local persistence estimate toward zero where recent changes have little variation.
    slopes = []
    for n in (13, 52):
        x, future = y[-n-1:-1], y[-n:]
        slopes.append(float(np.clip(x @ future / (x @ x + 4), -0.8, 1.0)))
    anchor = last[0] * (slopes[0] + slopes[1]) / 2
    when = date.fromisoformat(week)
    angle = 2 * np.pi * when.timetuple().tm_yday / 365.25
    age = (when - date.fromisoformat(past[-1][1])).days / 7
    features = [*y[-3:][::-1], y[-4:].mean(), y[-13:].mean(), y[-13:].std(), *slopes,
                past[-1][4], past[-1][4] - np.mean([r[4] for r in past[-13:]]),
                *last[1:], *changes[-4:, 1:].mean(axis=0), np.sin(angle), np.cos(angle), age]
    return dict(zip(PUBLIC, features, strict=True)), float(anchor)


def vintage_market(cutoff):
    # ALFRED date vintages bound what had been published by the previous New York calendar day.
    vintage = (cutoff.astimezone(EASTERN).date() - timedelta(days=1)).isoformat()
    books = [fetch(vintage, series)[0] for series in SERIES]
    days = sorted(books[0].keys() & books[1].keys())
    if len(days) < 63 or days[-1] > vintage:
        raise ValueError(f"Invalid ALFRED market history on {vintage}")
    wti, brent = np.array([[b[d] for d in days[-63:]] for b in books])
    basis = wti - brent
    values = [basis[-1] - np.median(basis), basis[-1] - basis[-2], basis[-1] - basis[-6],
              basis[-1] - basis[-21], wti[-1] - wti[-6], brent[-1] - brent[-6],
              np.std(np.diff(wti[-21:])), (date.fromisoformat(vintage) - date.fromisoformat(days[-1])).days]
    return dict(zip(BASIS, values, strict=True))


def build():
    optics = {r["release_date"]: r for r in coverage_inputs()}
    supplement = indexed("cushing_outer_capture_nowcast_forecasts")
    strong = {r["release_date"]: r for r in read("cushing_transfer_persistence_forecasts") if r["channel"] == "inventory"}
    old = indexed("cushing_mlp_experiment_forecasts")
    if not (set(optics) == set(supplement) == set(strong)):
        raise ValueError("The optical and reference forecast calendars differ")
    history, result = [], []
    for rec in release_history():
        issue, week, cutoff, known, _, delta = rec
        if not cutoff < known or (history and (history[-1][1] >= week or history[-1][2] >= cutoff)):
            raise ValueError(f"Invalid release order on {issue}")
        past = [r for r in history if r[3] < cutoff and r[1] < week]
        history.append(rec)
        if len(past) < 53:
            continue
        features, anchor = public_features(past, week)
        row = dict(release_date=issue, week_ending=week, cutoff=cutoff.isoformat(), known=known.isoformat(),
                   latest_public_input=max(r[3] for r in past).isoformat(), target=round(float(delta[0]), 6),
                   phase="archive", public_ensemble=None, optical_reference=None, previous_mlp=None, strong_reference=None,
                   **dict.fromkeys(BASIS),
                   anchor=anchor, **features, **dict.fromkeys(OPTICAL, 0.0))
        if issue in optics:
            o, s = optics[issue], supplement[issue]
            base, optical = o["public_ensemble"], float(s["outer_supplement"])
            last_label = s["outer_supplement_latest_label_utc"]
            if (o["cutoff"] != cutoff or o["known"] != known or o["week_ending"] != week or
                datetime.fromisoformat(s["cutoff_utc"]) != cutoff or datetime.fromisoformat(s["known_utc"]) != known or
                s["week_ending"] != week or float(s["public_ensemble"]) != base or
                not np.allclose([o["target"], float(s["actual"])], delta[0], rtol=0, atol=1e-8) or
                (last_label and datetime.fromisoformat(last_label) >= cutoff)):
                raise ValueError(f"Inconsistent forecast sources on {issue}")
            h = strong[issue]
            if (h["week_ending"] != week or datetime.fromisoformat(h["cutoff"]) != cutoff or
                    datetime.fromisoformat(h["known"]) != known or not np.isclose(float(h["actual"]), delta[0]) or
                    (h["latest_transfer_training_label_utc"] and datetime.fromisoformat(h["latest_transfer_training_label_utc"]) >= cutoff)):
                raise ValueError(f"The incumbent hybrid is misaligned on {issue}")
            row.update(strong_reference=float(h[REFERENCE]), **vintage_market(cutoff))
            row.update(phase=o["study_phase"], public_ensemble=base, optical_reference=optical)
            row.update(dict(zip(OPTICAL, [*o["x"][:5], o["space_time_fraction"], float(o["measured"]),
                                           optical - base, float(s["supplement_roofs"])], strict=True)))
            if o["study_phase"] == "holdout":
                b = old[issue]
                if (datetime.fromisoformat(b["forecast_cutoff_utc"]) != cutoff or float(b["target"]) != row["target"]
                        or float(b["public_ensemble"]) != base):
                    raise ValueError(f"Legacy MLP does not match on {issue}")
                row["previous_mlp"] = float(b["mlp_forecast"])
        result.append(row)
    if {r["release_date"] for r in result if r["phase"] != "archive"} != set(optics):
        raise ValueError("A reference forecast is missing from the prepared panel")
    PANEL.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(PANEL, index=False)
    print(f"{len(result)} weekly rows written to {PANEL}")


def forward_inputs(registration):
    """Reconstruct unlabeled features from the archived information set of an inventory registration."""
    strong = json.loads(registration.read_text())
    issue, week = strong["release_date"], strong["week_ending"]
    original = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    public = json.loads(original.read_text())
    signature = strong["sources_sha256"][str(original.relative_to(ROOT))]
    if hashlib.sha256(original.read_bytes()).hexdigest() != signature:
        raise ValueError("The original public forecast was changed after registration")
    for key in ("release_date", "week_ending", "information_cutoff_utc", "registration_deadline_utc"):
        if strong[key] != public[key]:
            raise ValueError(f"Different forecast vintages for {key}")
    info = datetime.fromisoformat(strong["information_cutoff_utc"])
    available = max(datetime.fromisoformat(r["recorded_at_utc"]) for r in (public, strong))
    if not info <= available < datetime.fromisoformat(strong["registration_deadline_utc"]):
        raise ValueError("The source forecast was not available before the release deadline")
    past = [r for r in release_history() if r[3] < info and r[1] < week]
    by_week = {r[1]: r for r in past}
    for r in read("eia_wpsr_forward_releases"):
        known = datetime.fromisoformat(r["conservative_postrelease_from_utc"])
        observed = r["latest_week_in_issue"]
        if known >= info or observed >= week or observed in by_week:
            continue
        path = ROOT / r["source_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["source_sha256"]:
            raise ValueError("Forward EIA archive checksum mismatch")
        values, _ = early_table(path, 4, r["release_date"], [observed, r["previous_week_ending"]])
        levels = np.array([values[k] for k in ("cushing_crude", "padd2_commercial_crude", "padd3_commercial_crude")]) / 1000
        levels[1] -= levels[0]
        current = levels[:, 0] - levels[:, 1]
        if not np.isclose(current[0], float(r["cushing_change_kbbl"]) / 1000):
            raise ValueError("Forward Cushing release does not reproduce the recorded weekly change")
        by_week[observed] = (r["release_date"], observed, None, known, levels[0, 0], current)
    past = sorted(by_week.values(), key=lambda row: row[1])
    if len(past) < 53 or (date.fromisoformat(week) - date.fromisoformat(past[-1][1])).days != 7:
        raise ValueError("The previous weekly EIA release is missing")
    features, anchor = public_features(past, week)
    panel = load_panel()
    settled = panel[(panel.known < info) & panel.strong_reference.notna()]
    if settled.empty:
        raise ValueError("Past errors are unavailable at the registration cutoff")
    reference = float(strong["estimates_kbbl"]["inventory"][REFERENCE]) / 1000
    vintage_end = datetime.combine(info.astimezone(EASTERN).date(), time.min, EASTERN) - timedelta(microseconds=1)
    return dict(release_date=issue, week_ending=week, cutoff=available.isoformat(), information_cutoff_utc=info.isoformat(),
                deadline_utc=strong["registration_deadline_utc"],
                reference_available_utc=available.isoformat(), latest_public_input=max(r[3] for r in past).isoformat(),
                latest_residual_input=settled.known.max().isoformat(), market_vintage_end_utc=vintage_end.astimezone(timezone.utc).isoformat(),
                strong_reference=reference, reference_minus_public=reference - float(public["public_ensemble_kbbl"]) / 1000,
                anchor=anchor, vintage_present=1.0, **features, **calendar(week), **vintage_market(info),
                **past_errors(panel, info))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, help="Pre-release transfer-persistence registration for one week")
    args = parser.parse_args()
    if args.registration:
        row = forward_inputs(args.registration)
        path = DERIVED / f"cushing_calibrated_network_inputs_{row['release_date']}.json"
        save_json(path, row)
        print(path)
    else:
        build()
