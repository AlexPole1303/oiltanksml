import json
import sys
import unittest
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cushing_image_control as control
import prepare_cushing_paired_market as prepare
import register_cushing_paired as journal


class PairedControls(unittest.TestCase):
    def test_cohort_deadlines_and_frozen_inputs(self):
        plan = json.loads(journal.PROTOCOL.read_bytes())
        self.assertEqual(len(plan["releases"]), 13)
        self.assertEqual(
            plan["frozen_artifacts"],
            {p.name: journal.digest(p) for p in journal.pinned_inputs()},
        )
        holiday = journal.schedule("2027-01-21")
        end = datetime.fromisoformat(holiday["deadline_utc"])
        journal.window(end - timedelta(hours=2), holiday)
        for t in (end, end - timedelta(hours=2, seconds=1)):
            with self.assertRaises(ValueError):
                journal.window(t, holiday)
        with self.assertRaises(ValueError):
            journal.schedule("2026-11-18")

    def test_summary_and_immutable_records(self):
        with TemporaryDirectory() as tmp, patch.object(journal, "JOURNAL", Path(tmp)):
            self.assertEqual(journal.summary()["scored"], 0)
            path = Path(tmp) / "pair.json"
            journal.immutable(path, {"estimate": 4})
            journal.immutable(path, {"estimate": 4})
            with self.assertRaises(ValueError):
                journal.immutable(path, {"estimate": 5})
            self.assertEqual(json.loads(path.read_bytes()), {"estimate": 4})

    def test_first_release_table(self):
        raw = b"STUB_1,11/6/26,10/30/26,Difference\nCushing,27.356,27.250,0.106\n"
        self.assertEqual(journal.reported_change(raw, "2026-11-06"), 106)
        with self.assertRaises(ValueError):
            journal.reported_change(raw, "2026-10-30")
        with self.assertRaises(ValueError):
            journal.reported_change(raw.replace(b"0.106", b"0.101"), "2026-11-06")

    def test_historical_ledger_and_groups(self):
        forecasts = pd.read_csv(control.POINTS)
        self.assertEqual(forecasts.release_date.nunique(), 148)
        err = 1000 * (forecasts.no_images - forecasts.target).abs()
        full = 1000 * (forecasts.estimate - forecasts.target).abs()
        self.assertAlmostEqual(err.mean(), 250.225940023, 6)
        self.assertAlmostEqual(full.mean(), 247.103106583, 6)
        summary = json.loads(control.REPORT.read_bytes())
        self.assertEqual(summary["survey_and_image"]["n"], 67)
        self.assertAlmostEqual(
            summary["all"]["image_gain_kbbl"], float((err - full).mean()), 8
        )

    def test_forecast_pair_uses_same_survey(self):
        f = journal.full
        m = control.MODEL_DIR
        record = json.loads((m / "nov18_inputs.json").read_bytes())
        args = Namespace(
            predict=m / "nov18_inputs.json",
            as_of="2026-11-18T04:45:00+00:00",
            survey_ledger=f.v3.SURVEY,
            confirmed_ledger=f.v4.EXTRA,
            wire_ledger=f.v5.WIRE,
            precision_ledger=f.PRECISION,
            measurements=f.v5.MEASUREMENTS,
            balance_inputs=f.EXTERNAL,
            products=m / "product_observations.csv",
            settled_feedback=None,
        )
        pub = record["strong_reference"] - record["reference_minus_public"]
        # Neutral public features are an artificial unit-test fixture, never a journal receipt.
        market = {k: record[k] for k in ("release_date", "week_ending")}
        market.update(
            cutoff=record["information_cutoff_utc"],
            available_utc=record["cutoff"],
            public_ensemble=pub,
            basis_public=pub,
            **dict.fromkeys(control.REGION + control.CURVE + control.COORD, 0),
        )
        with threadpool_limits(limits=1):
            pair = journal.predict_pair(args, market)
        with threadpool_limits(limits=1):
            args.as_of = "2026-11-15T20:00:00+00:00"
            missing = journal.predict_pair(args, market)
        self.assertIsNone(missing["api_kbbl"])
        self.assertTrue(
            np.isfinite([missing["full_kbbl"], missing["without_images_kbbl"]]).all()
        )
        self.assertEqual(pair["api_kbbl"], -1691)
        self.assertTrue(np.isfinite(list(pair.values())).all())
        with self.assertRaises(ValueError):
            journal.predict_pair(args, market | {"cutoff": args.as_of})

    def test_market_matches_time_stamped_quotes(self):
        cutoff = datetime.fromisoformat("2026-11-13T23:20:00+00:00")
        sessions = [
            "2026-11-06",
            "2026-11-09",
            "2026-11-10",
            "2026-11-11",
            "2026-11-12",
            "2026-11-13",
        ]
        months = ["202701", "202702", "202703", "202704"]
        with TemporaryDirectory() as name:
            folder = Path(name)
            cal = folder / "calendar.csv"
            marks = folder / "prices.csv"
            tape = folder / "tape.json"
            cal.write_text(
                "session_date,eligible\n" + "".join(f"{day},1\n" for day in sessions)
            )
            marks.write_text(
                "session_date,delivery_month,usd_per_barrel\n"
                + "".join(
                    f"{day},{m},{80 + i}\n"
                    for day in (sessions[0], sessions[-1])
                    for i, m in enumerate(months)
                )
            )
            snapshot = {
                "release_date": "2026-11-18",
                "eligible_sessions": sessions,
                "contract_months": months,
                "available_utc": "2026-11-13T20:00:00+00:00",
                "calendar_source": cal.name,
                "prices_source": marks.name,
                "calendar_source_sha256": journal.digest(cal),
                "prices_source_sha256": journal.digest(marks),
            }
            tape.write_text(json.dumps(snapshot))
            curve, _ = prepare.curve_input(tape, "2026-11-18", cutoff)
            self.assertEqual(tuple(curve[1:]), (0, 0, 0, 0, 0))
            marks.write_text(marks.read_text() + "\n")
            with self.assertRaises(ValueError):
                prepare.curve_input(tape, "2026-11-18", cutoff)
