import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import fit_cushing_survey_residuals as prior
from collect_cushing_verified_bulletins import bulletin, get_value
from cushing_inventory_measurement_filter import state
from fit_cushing_measurement_filters import measurement_at, predict, wire_records
from prepare_cushing_measurement_inputs import national_estimate, radar_features


class MeasurementFilterTest(unittest.TestCase):
    def test_morning_report_requires_the_correct_week_and_source(self):
        from cushing_morning_reports import parse_changes

        quote = "API最新数据显示，截至12月13日当周，美国原油库存-469.4万桶，库欣原油库存+77.8万桶。"
        self.assertEqual(parse_changes(quote, "2024-12-13"), {778.0})
        self.assertFalse(parse_changes(quote, "2024-12-06"))
        self.assertFalse(parse_changes(quote.replace("API", "EIA"), "2024-12-13"))

    def test_actual_chinese_change_is_not_the_previous_number(self):
        text = "美国石油协会（API）数据显示，上周，美国API原油库存 -469.4万桶。上周API库欣原油库存 +77.8万桶，前值 -151.7万桶。"
        self.assertEqual(get_value(text, "cushing"), 778)
        self.assertEqual(get_value(text, "national"), -4694)
        item = {
            "id": 1,
            "create_time": "2024-12-18 05:33:00",
            "update_time": "2024-12-18 05:34:11",
            "rich_text": text,
        }
        self.assertEqual(bulletin(item)["available_utc"], "2024-12-17T21:34:11+00:00")
        self.assertIsNone(bulletin(dict(item, rich_text=text + "EIA")))
        self.assertIsNone(bulletin(dict(item, update_time=None)))

    def test_national_inventory_uses_current_api_clause(self):
        self.assertEqual(
            national_estimate(
                "The American Petroleum Institute reported a draw in crude oil inventories of 2.5 million barrels for the week ending May 3."
            ),
            -2500.0,
        )
        self.assertEqual(
            national_estimate(
                "Crude oil inventories in the US rose this week by 1.2 million barrels, the American Petroleum Institute said."
            ),
            1200.0,
        )
        self.assertIsNone(
            national_estimate(
                "The American Petroleum Institute said crude gasoline inventories fell by 2 million barrels."
            )
        )
        self.assertIsNone(national_estimate("A draw in crude oil inventories of 3 million barrels."))

    def test_radar_and_national_use_separate_clocks(self):
        cutoff = pd.Timestamp("2024-05-15T04:00:00Z")
        row = pd.Series(
            {
                "national_kbbl": 1000,
                "national_time": cutoff.isoformat(),
                "radar_weekly_million": 0.5,
                "radar_age_days": 1.0,
                "radar_available": (cutoff - pd.Timedelta(seconds=1)).isoformat(),
            }
        )
        features = measurement_at(row, cutoff)
        self.assertTrue(np.isnan(features["national_kbbl"]))
        self.assertEqual(features["radar_weekly_million"], 0.5)
        scenes = pd.DataFrame(
            {
                "acq_time": pd.to_datetime(["2024-04-27", "2024-05-09", "2024-05-10"], utc=True),
                "available_at": pd.to_datetime(["2024-04-29", "2024-05-12", "2024-05-16"], utc=True),
                "kbbl": [0, 1200, 99999],
            }
        )
        a = radar_features(scenes, "2024-05-10", cutoff)
        b = radar_features(scenes.iloc[:2], "2024-05-10", cutoff)
        self.assertEqual(a, b)
        self.assertAlmostEqual(a[0], 0.7)

    def test_unreleased_labels_cannot_change_accumulated_error(self):
        known = pd.to_datetime(["2024-05-01", "2024-05-08", "2024-05-15"], utc=True)
        d = pd.DataFrame(
            {
                "known": known,
                "week_ending": ["2024-04-26", "2024-05-03", "2024-05-10"],
                "survey": [1, 1, 1],
                "target": [1.3, 1.2, 100],
            }
        )
        a = state(d, "2024-05-10", known[-1])
        d.loc[2, "target"] = -100
        self.assertEqual(a, state(d, "2024-05-10", known[-1]))
        self.assertAlmostEqual(a["last_gap_local"], 0.2)
        self.assertAlmostEqual(a["error_lag2"], 0.3)
        self.assertEqual(a["error_lag3"], 0.0)

    def test_forecast_uses_new_wire_only_after_it_is_available(self):
        record = json.loads((prior.MODEL_DIR / "nov18_inputs.json").read_text())
        day = record["release_date"]
        when = pd.Timestamp("2026-11-18T04:55:00Z")
        quote = pd.Series(
            {"available_utc": when, "week_ending": record["week_ending"], "api_change_kbbl": -890},
            name=day,
        )
        reference = record["strong_reference"]
        old_weights = json.loads(prior.WEIGHTS.read_text())
        package = {
            "mix": {"missing": {"strong_reference": 1.0}, "present": {"coverage_only": 1.0}},
            "filters": {},
            "settled": old_weights["feedback"],
            "v4_sha256": hashlib.sha256(prior.WEIGHTS.read_bytes()).hexdigest(),
        }
        early = predict(record, when, package, None, quote, None)
        later = predict(record, when + pd.Timedelta(seconds=1), package, None, quote, None)
        self.assertFalse(early["survey_available"])
        self.assertAlmostEqual(early["estimate_kbbl"], 1000 * reference)
        self.assertTrue(later["survey_available"])
        self.assertAlmostEqual(later["estimate_kbbl"], 0.9 * -890 + 0.1 * 1000 * reference)

    def test_revised_wire_is_ineligible_at_its_own_cutoff(self):
        old = pd.DataFrame([{"release_date": "2024-05-15", "api_change_kbbl": 50}]).set_index("release_date")
        when = "2024-05-15T03:45:00Z"
        update = {
            "release_date": "2024-05-15",
            "action": "replace",
            "published_utc": "2024-05-15T03:30:00Z",
            "available_utc": when,
            "api_change_kbbl": 100,
            "week_ending": "2024-05-10",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wire.csv"
            pd.DataFrame([update]).to_csv(path, index=False)
            self.assertEqual(wire_records(old, path, when).api_change_kbbl.iloc[0], 50)
            self.assertEqual(wire_records(old, path).api_change_kbbl.iloc[0], 100)
