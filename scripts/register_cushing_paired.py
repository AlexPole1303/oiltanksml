"""Record a dated, prospective Cushing forecast pair and reconcile it to the first archived EIA Table 4."""

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

import cushing_image_control as control
import fit_cushing_survey_balance as full
import numpy as np
import pandas as pd
import requests
from register_cushing_prerelease_forecast import digest, freeze
from threadpoolctl import threadpool_limits

ROOT = control.MODEL_DIR.parents[1]
JOURNAL = control.MODEL_DIR / "prospective"
PROTOCOL = JOURNAL / "protocol.json"
EASTERN = ZoneInfo("America/New_York")


def schedule(issue):
    protocol = json.loads(PROTOCOL.read_text())
    row = next((r for r in protocol["releases"] if r["release_date"] == issue), None)
    if row is None:
        raise ValueError("Release is outside the preregistered cohort")
    return row


def utcnow():
    return datetime.now(UTC)


def window(now, row):
    deadline = datetime.fromisoformat(row["deadline_utc"])
    if not deadline - timedelta(hours=2) <= now < deadline:
        raise ValueError("Register during the two hours preceding the frozen cutoff")


def immutable(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Existing record differs: {path}")
        return
    freeze(path, value)


def predict_pair(args, market):
    record, when, quote, external, products, feedback = full.prediction_inputs(args)
    if (
        record["release_date"] != market["release_date"]
        or record["week_ending"] != market["week_ending"]
    ):
        raise ValueError("Public and full snapshots refer to different releases")
    pub = float(market["public_ensemble"])
    if not np.isclose(
        pub,
        record["strong_reference"] - record["reference_minus_public"],
        rtol=0,
        atol=1e-8,
    ):
        raise ValueError("The public base differs across the two arms")
    r = control.market_row(market)
    if not np.isfinite(
        [
            r["basis_public"],
            *r["regional_state"],
            *r["shape"],
            *r["inventory_coordinates"],
            pub,
        ]
    ).all():
        raise ValueError("Incomplete public physical or market inputs")
    available = pd.Timestamp(market["available_utc"])
    if (
        r["cutoff"] != pd.Timestamp(record["information_cutoff_utc"])
        or available.tzinfo is None
        or r["cutoff"].tzinfo is None
        or not r["cutoff"] <= available <= when
    ):
        raise ValueError("Public input time exceeds the decision clock")
    if "target" in market or "target" in record:
        raise ValueError("Prospective snapshots must be unlabeled")
    panel, history = control.panel_and_history()
    ref, _, _ = control.reference(r, history)
    when_public = max(pd.Timestamp(record["cutoff"]), available)
    # Supply the recorded public inputs and recompute every image-sensitive predictor.
    fields = set(control.calibration.DIRECT) | {
        "release_date",
        "week_ending",
        "information_cutoff_utc",
        "latest_public_input",
        "market_vintage_end_utc",
        "deadline_utc",
        "anchor",
        "national_kbbl",
    }
    clean = {k: v for k, v in record.items() if k in fields}
    clean.update(
        strong_reference=ref,
        reference_minus_public=ref - pub,
        cutoff=when_public.isoformat(),
        reference_available_utc=available.isoformat(),
        latest_residual_input=panel.loc[panel.known < r["cutoff"], "known"]
        .max()
        .isoformat(),
        **control.calibration.past_errors(panel, r["cutoff"]),
    )
    pack = json.loads(control.WEIGHTS.read_text())
    stripped = None if external is None else external.copy()
    if stripped is not None:
        stripped["optical_time"] = None
    no = full.forecast_with_models(
        clean,
        when,
        pack["model"],
        quote,
        stripped,
        products,
        feedback,
        pack["network"],
        pack["survey"],
        {"mix": pack["residual_mix"]},
    )
    yes = full.forecast(
        record,
        when,
        json.loads(full.WEIGHTS.read_text()),
        quote,
        external,
        products,
        feedback,
    )
    return {
        "full_kbbl": yes["estimate_kbbl"],
        "without_images_kbbl": no["estimate_kbbl"],
        "api_kbbl": None if quote is None else float(quote.api_change_kbbl),
    }


def pinned_inputs():
    return [
        full.WEIGHTS,
        full.v5.WEIGHTS,
        full.v4.WEIGHTS,
        full.v3.MIXTURE,
        full.v5.PACKAGE,
        control.WEIGHTS,
        control.INPUTS,
        control.MODEL_DIR / "weekly_inputs.csv",
    ]


def register(args):
    row = schedule(args.issue)
    now = utcnow()
    window(now, row)
    args.as_of = now.isoformat()
    record = json.loads(args.predict.read_text())
    market = json.loads(args.market.read_text())
    if (record["release_date"], record["week_ending"]) != (
        args.issue,
        row["week_ending"],
    ) or pd.Timestamp(record["deadline_utc"]) != pd.Timestamp(row["deadline_utc"]):
        raise ValueError("Forecast date and deadline must match the EIA protocol")
    sources = {
        name: digest(getattr(args, name))
        for name in (
            "predict",
            "market",
            "survey_ledger",
            "confirmed_ledger",
            "wire_ledger",
            "precision_ledger",
            "measurements",
            "balance_inputs",
            "products",
            "settled_feedback",
        )
        if getattr(args, name, None)
    }
    protocol = json.loads(PROTOCOL.read_bytes())
    if {p.name: digest(p) for p in pinned_inputs()} != protocol["frozen_artifacts"]:
        raise ValueError("The preregistered model weights or historical inputs changed")
    if not market.get("source_sha256") or not market.get("dated_futures_tape"):
        raise ValueError(
            "Prepare the public snapshot from dated source evidence before registration"
        )
    with threadpool_limits(limits=1):
        values = predict_pair(args, market)
    if not np.isfinite([v for v in values.values() if v is not None]).all():
        raise ValueError("Nonfinite estimate")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    # A receipt must point to retrievable code and frozen coefficients in the named commit.
    for p in [*pinned_inputs(), PROTOCOL, Path(__file__).resolve()]:
        stored = subprocess.check_output(
            ["git", "show", f"{head}:{p.relative_to(ROOT)}"], cwd=ROOT
        )
        if hashlib.sha256(stored).hexdigest() != digest(p):
            raise ValueError(f"Commit does not match {p.name}")
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "scripts"], cwd=ROOT, text=True
    )
    if changed.strip():
        raise ValueError("Commit changes to the prediction code before registration")
    if any(digest(getattr(args, k)) != v for k, v in sources.items()):
        raise ValueError("Inputs changed during forecast generation")
    payload = dict(
        **row,
        decision_utc=args.as_of,
        recorded_at_utc=utcnow().isoformat(),
        git_commit=head,
        protocol_sha256=digest(PROTOCOL),
        input_sha256=sources,
        full_input=record,
        public_market_input=market,
        **values,
    )
    window(datetime.fromisoformat(payload["recorded_at_utc"]), row)
    freeze(JOURNAL / f"{args.issue}.prediction.json", payload)


def reported_change(raw, week):
    rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"))))
    stubs = sum(cell.startswith("STUB_") for cell in rows[0])
    if stubs not in (1, 2):
        raise ValueError("Unknown EIA Table 4 format")
    dates = [
        datetime.strptime(x, "%m/%d/%y").date() for x in rows[0][stubs : stubs + 2]
    ]
    if dates != [
        date.fromisoformat(week),
        date.fromisoformat(week) - timedelta(days=7),
    ]:
        raise ValueError("Wrong EIA reporting weeks")
    c = [r for r in rows if any(cell.strip() == "Cushing" for cell in r[:stubs])]
    if len(c) != 1:
        raise ValueError("Expected one Cushing row")
    a, b, delta = map(Decimal, c[0][stubs : stubs + 3])
    if a - b != delta or not delta.is_finite():
        raise ValueError("Cushing levels do not reconcile")
    return float(delta * 1000)


def score(issue):
    row = schedule(issue)
    if utcnow() < datetime.fromisoformat(row["scheduled_release_utc"]):
        raise ValueError("EIA publication is still in the future")
    path = JOURNAL / f"{issue}.prediction.json"
    if not path.exists():
        raise ValueError("No prerelease pair was registered")
    receipt = json.loads(path.read_bytes())
    window(datetime.fromisoformat(receipt["recorded_at_utc"]), row)
    if receipt["week_ending"] != row["week_ending"] or receipt[
        "protocol_sha256"
    ] != digest(PROTOCOL):
        raise ValueError("Receipt and protocol differ")
    source = JOURNAL / f"{issue}.table4.csv"
    root = f"https://www.eia.gov/petroleum/supply/weekly/archive/{issue[:4]}/{issue.replace('-', '_')}"
    source_url = JOURNAL / f"{issue}.source.json"
    if source_url.exists():
        meta = json.loads(source_url.read_bytes())
        raw = source.read_bytes()
        url = meta["archive_url"]
        if hashlib.sha256(raw).hexdigest() != meta["archive_sha256"]:
            raise ValueError("Saved EIA archive differs from its acquisition receipt")
    else:
        for folder in ("download", "csv"):
            url = f"{root}/{folder}/table4.csv"
            response = requests.get(url, timeout=30)
            if response.status_code != 404:
                break
        response.raise_for_status()
        raw = response.content
        reported_change(raw, row["week_ending"])
        # Publish the bytes before their acquisition receipt. An interrupted run can resume.
        with NamedTemporaryFile(dir=JOURNAL, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            try:
                os.link(temporary, source)
            except FileExistsError:
                if source.read_bytes() != raw:
                    raise ValueError("Different EIA response was already captured")
        finally:
            temporary.unlink(missing_ok=True)
        immutable(
            source_url,
            {"archive_url": url, "archive_sha256": hashlib.sha256(raw).hexdigest()},
        )
    actual = reported_change(raw, row["week_ending"])
    errors = {
        k: abs(receipt[k] - actual)
        for k in ("full_kbbl", "without_images_kbbl", "api_kbbl")
        if receipt[k] is not None
    }
    immutable(
        JOURNAL / f"{issue}.score.json",
        dict(
            **row,
            eia_kbbl=actual,
            absolute_error_kbbl=errors,
            image_gain_kbbl=errors["without_images_kbbl"] - errors["full_kbbl"],
            forecast_sha256=digest(path),
            archive_url=url,
            archive_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )


def summary():
    cohort = json.loads(PROTOCOL.read_bytes())["releases"]
    errors = []
    status = []
    for row in cohort:
        day = row["release_date"]
        pair = JOURNAL / f"{day}.prediction.json"
        result = JOURNAL / f"{day}.score.json"
        state = (
            "scheduled"
            if utcnow() < datetime.fromisoformat(row["deadline_utc"])
            else "missed"
        )
        if pair.exists():
            state = "registered"
        if result.exists():
            score_record = json.loads(result.read_bytes())
            if score_record["forecast_sha256"] != digest(pair):
                raise ValueError("Score no longer matches the registered pair")
            errors.append(score_record["absolute_error_kbbl"])
            state = "scored"
        status.append((day, row["week_ending"], state))
    metrics = {}
    if errors:
        metrics = {
            key: float(np.mean([item[key] for item in errors if key in item]))
            for key in ("full_kbbl", "without_images_kbbl", "api_kbbl")
            if any(key in item for item in errors)
        }
    return {
        "planned": len(cohort),
        "registered": sum(s[2] in ("scored", "registered") for s in status),
        "scored": len(errors),
        "missed": sum(s[2] == "missed" for s in status),
        "scored_with_api": sum("api_kbbl" in item for item in errors),
        "mae_kbbl": metrics,
        "paired_gain_kbbl": (metrics["without_images_kbbl"] - metrics["full_kbbl"])
        if metrics
        else None,
        "releases": status,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("register")
    q.add_argument("issue")
    q.add_argument("--predict", required=True, type=Path)
    q.add_argument("--market", required=True, type=Path)
    defaults = {
        "survey_ledger": full.v3.SURVEY,
        "confirmed_ledger": full.v4.EXTRA,
        "wire_ledger": full.v5.WIRE,
        "precision_ledger": full.PRECISION,
        "measurements": full.v5.MEASUREMENTS,
        "balance_inputs": full.EXTERNAL,
        "products": control.MODEL_DIR / "product_observations.csv",
        "settled_feedback": None,
    }
    for name, default in defaults.items():
        q.add_argument("--" + name.replace("_", "-"), type=Path, default=default)
    sub.add_parser("score").add_argument("issue")
    sub.add_parser("status")
    args = p.parse_args()
    if args.action == "register":
        register(args)
    elif args.action == "score":
        score(args.issue)
    else:
        print(json.dumps(summary(), indent=2))


if __name__ == "__main__":
    main()
