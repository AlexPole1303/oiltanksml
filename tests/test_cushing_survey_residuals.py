import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_cushing_survey_confirmation import corroborate
from cushing_survey_dispatches import extract, match_dispatch
from cushing_survey_residual_network import feedback, fit_feedback_head
from fit_cushing_survey_residuals import settled_history, sources


class ResidualForecastTest(unittest.TestCase):
    def setUp(self):
        self.week = "2024-05-10"
        self.deadline = pd.Timestamp("2024-05-15T04:00:00Z")
        self.event = {
            "indicatorsId": 9831,
            "dataTime": self.week,
            "pubValue": "-60.1",
            "pubDate": "2024-05-15",
            "pubTime": "04:30",
            "calendarId": 1,
        }

    def test_calendar_alone_cannot_recover_a_quote(self):
        art = {
            "head": "The American Petroleum Institute reported",
            "mentions": ["Cushing fell by 601,000 barrels."],
            "published": "2024-05-14T20:41:00Z",
            "modified": "2024-05-14T20:42:00Z",
            "url": "example",
            "sha256": "x",
        }
        self.assertEqual(corroborate(art, self.event, self.week, self.deadline)["api_change_kbbl"], -601.0)
        for event in [
            dict(self.event, pubValue="60.1"),
            dict(self.event, dataTime="2024-05-03"),
            dict(self.event, indicatorsId=9832),
        ]:
            self.assertIsNone(corroborate(art, event, self.week, self.deadline))
        self.assertIsNone(
            corroborate(dict(art, modified=self.deadline.isoformat()), self.event, self.week, self.deadline)
        )
        self.assertIsNone(corroborate(dict(art, mentions=[]), self.event, self.week, self.deadline))

    def test_dispatch_needs_current_api_value_and_published_clock(self):
        multi = "US API OIL STOCK CHANGE ACTUAL 1M. US API CUSHING STOCK CHANGE ACTUAL -0.6M (FORECAST -, PREVIOUS 0.5M)"
        self.assertEqual(extract(multi), (-600.0, 50.0))
        self.assertIsNone(extract(multi.replace("API CUSHING", "EIA CUSHING")))
        item = {
            "api_change_kbbl": -600.0,
            "rounding_kbbl": 50.0,
            "published_utc": "2024-05-14T20:43:00Z",
            "source_url": "example",
            "sha256": "x",
        }
        self.assertIsNotNone(match_dispatch(item, self.event, self.week, self.deadline))
        self.assertIsNone(
            match_dispatch(
                dict(item, published_utc=self.deadline.isoformat()), self.event, self.week, self.deadline
            )
        )
        self.assertIsNone(
            match_dispatch(dict(item, api_change_kbbl=600.0), self.event, self.week, self.deadline)
        )

    def test_prior_discrepancy_is_causal_and_expires(self):
        records = [
            {"week_ending": "2024-05-03", "known": "2024-05-08T15:00:00Z", "target": 0.5, "survey": 0.2},
            {"week_ending": "2024-05-10", "known": "2024-05-15T15:00:00Z", "target": 99.0, "survey": -1.0},
        ]
        history = settled_history(records)
        d = feedback(history, self.week, self.deadline)
        self.assertAlmostEqual(d["last_gap"], 0.3)
        history.loc[1, "target"] = -100
        self.assertEqual(d, feedback(history, self.week, self.deadline))
        self.assertEqual(feedback(history, "2024-05-24", pd.Timestamp("2024-05-29T04:00Z"))["last_gap"], 0.0)
        with self.assertRaises(ValueError):
            settled_history(records, pd.DataFrame([dict(records[0], target=-1.0)]))

    def test_replacement_waits_for_its_own_publication(self):
        import tempfile

        oldrow = {
            "release_date": "2024-05-15",
            "week_ending": self.week,
            "api_change_kbbl": 601.0,
            "published_utc": "2024-05-14T20:31:00Z",
            "available_utc": "2024-05-14T20:31:00Z",
        }
        update = dict(oldrow, action="replace", api_change_kbbl=-601.0, available_utc="2024-05-14T20:45:00Z")
        with tempfile.TemporaryDirectory() as tmp:
            a, b = (Path(tmp) / k for k in ("a.csv", "b.csv"))
            pd.DataFrame([oldrow]).to_csv(a, index=False)
            pd.DataFrame([update]).to_csv(b, index=False)
            self.assertEqual(sources(b, a, update["available_utc"]).api_change_kbbl.iat[0], 601.0)
            self.assertEqual(sources(b, a, self.deadline).api_change_kbbl.iat[0], -601.0)

    def test_late_label_cannot_change_learned_coefficient(self):
        n = 75
        known = pd.date_range("2019-01-01", periods=n, tz="UTC", freq="7D")
        x = np.random.default_rng(77).normal(size=(n, 5))
        d = pd.DataFrame(x, columns=["strong_reference", "target", "survey", "last_gap", "gap4"])
        d["known"] = known
        d["cutoff"] = known - pd.Timedelta(days=1)
        original = fit_feedback_head(d, known[65])
        d.loc[d.known >= known[65], "target"] = 500
        self.assertEqual(original, fit_feedback_head(d, known[65]))
