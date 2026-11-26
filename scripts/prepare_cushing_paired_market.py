"""Prepare the no-image forecast from dated public stock balances, spot vintages and CL settlements."""

import argparse
import csv
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import cushing_image_control as control
import numpy as np
from audit_cushing_spot_vintages import inspect
from backtest_wti_cushing_spread import settlement, tick_price
from build_s1_reference_matches import DERIVED
from fit_cushing_cl_curve_shape import curve_coordinates, delivery_months
from fit_cushing_public_state import released_balances
from fit_cushing_storage_state import storage_coordinates
from fit_cushing_volatility_neighbors import archived_coordinates
from register_cushing_prerelease_forecast import digest, freeze
from register_cushing_public_fallback import FALLBACK, ROOT, latest_forward_balances
from threadpoolctl import threadpool_limits


def curve_input(path, issue, cutoff):
    tape = json.loads(path.read_bytes())
    if tape["release_date"] != issue:
        raise ValueError("Different release on futures tape")
    dates = tape["eligible_sessions"]
    if (
        len(dates) != 6
        or dates != sorted(set(dates))
        or settlement(dates[-1]) >= cutoff
    ):
        raise ValueError(
            "Expected six consecutive eligible sessions ending before the information cutoff"
        )
    for key in ("calendar_source", "prices_source"):
        raw = path.parent / tape[key]
        if digest(raw) != tape[key + "_sha256"]:
            raise ValueError(f"Source file changed: {raw.name}")
    months = tuple(tape["contract_months"])
    expected = delivery_months(
        {"near_month": months[0], "far_month": months[-1], "release_date": issue}
    )
    if months != expected:
        raise ValueError(
            "Expected four consecutive dated delivery months from the release window"
        )
    with (path.parent / tape["calendar_source"]).open(newline="") as stream:
        calendar = list(csv.DictReader(stream))
    if (
        not calendar
        or max(r["session_date"] for r in calendar)
        < cutoff.astimezone(settlement(dates[-1]).tzinfo).date().isoformat()
    ):
        raise ValueError("The calendar must cover the information cutoff date")
    sessions = sorted(
        {
            r["session_date"]
            for r in calendar
            if r["eligible"] == "1" and settlement(r["session_date"]) < cutoff
        }
    )
    if sessions[-6:] != dates:
        raise ValueError("Dates differ from the latest six eligible settlements")
    with (path.parent / tape["prices_source"]).open(newline="") as stream:
        quotes = {}
        for q in csv.DictReader(stream):
            key = (q["session_date"], q["delivery_month"])
            if key in quotes:
                raise ValueError("Duplicate dated CL settlement")
            quotes[key] = float(q["usd_per_barrel"])
    legs = np.asarray(
        [[quotes[day, month] for month in months] for day in (dates[0], dates[-1])]
    )
    if legs.shape != (2, 4) or not np.isfinite(legs).all():
        raise ValueError("Expected four prices for each end of the five-session change")
    before, after = [
        curve_coordinates([tick_price(float(v)) for v in prices]) for prices in legs
    ]
    raw = [after[0], after[0] - before[0], *after[1:], *(after - before)[1:]]
    available = datetime.fromisoformat(tape["available_utc"])
    if (
        available.tzinfo is None
        or available < settlement(dates[-1])
        or available > cutoff
    ):
        raise ValueError("The futures source was unavailable at the information cutoff")
    return np.tanh(np.asarray(raw) / [0.7, 0.3, 0.35, 0.20, 0.25, 0.15]), available


def prepare(issue, tape, output):
    source = DERIVED / f"cushing_prerelease_forecast_{issue}.json"
    fallback_path = source.with_stem(source.stem + "_public_fallback")
    original, fallback = [json.loads(p.read_bytes()) for p in (source, fallback_path)]
    week = original["week_ending"]
    if (
        not all(
            fallback[k] == original[k]
            for k in (
                "week_ending",
                "release_date",
                "information_cutoff_utc",
                "registration_deadline_utc",
            )
        )
        or issue != original["release_date"]
    ):
        raise ValueError("Public source receipts have different cutoffs or releases")
    cutoff = datetime.fromisoformat(original["information_cutoff_utc"])
    if not np.isclose(
        fallback["baseline_public_kbbl"], original["public_ensemble_kbbl"]
    ):
        raise ValueError("Different public anchors in source receipts")
    older = [q for q in released_balances() if q["known"] < cutoff and q["week"] < week]
    newer, evidence = latest_forward_balances(cutoff, week, original["prior_release"])
    if (
        older[-1]["week"] >= newer[0]["week"]
        or newer[-1]["week"]
        != (date.fromisoformat(week) - timedelta(days=7)).isoformat()
    ):
        raise ValueError(
            "Stock history must end in the published week immediately preceding the forecast"
        )
    state = storage_coordinates(older + newer)
    spot = inspect(
        {
            "release_date": issue,
            "study_phase": "forward",
            "cutoff_utc": cutoff.isoformat(),
            "spot_reference_day": fallback["spot_reference_date"],
        }
    )
    for k in ("wti", "brent"):
        evidence[spot[k + "_path"]] = spot[k + "_sha256"]
    for path, checksum in fallback["forward_evidence_sha256"].items():
        if digest(ROOT / path) != checksum:
            raise ValueError(f"Public source checksum mismatch: {path}")
    curve, stamp = curve_input(tape, issue, cutoff)
    available = max(
        datetime.fromisoformat(fallback["recorded_at_utc"]),
        datetime.fromisoformat(original["recorded_at_utc"]),
        stamp,
    )
    if available > datetime.now(UTC):
        raise ValueError("Public source receipt is dated in the future")
    market = dict(
        release_date=issue,
        week_ending=week,
        cutoff=cutoff.isoformat(),
        available_utc=available.isoformat(),
        public_ensemble=original["public_ensemble_kbbl"] / 1000,
        basis_public=fallback["conditional_fallbacks_kbbl"][FALLBACK] / 1000,
        **dict(zip(control.REGION, state["regional_state"], strict=True)),
        **dict(
            zip(control.COORD, archived_coordinates(spot, state["state"]), strict=True)
        ),
        **dict(zip(control.CURVE, curve, strict=True)),
        source_sha256=evidence
        | {
            source.name: digest(source),
            fallback_path.name: digest(fallback_path),
            tape.name: digest(tape),
        },
        dated_futures_tape=json.loads(tape.read_bytes()),
    )
    freeze(output, market)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("issue")
    p.add_argument("--tape", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    with threadpool_limits(limits=1):
        prepare(args.issue, args.tape, args.output)
