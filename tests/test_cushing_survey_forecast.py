import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from collect_cushing_weekly_survey import extract_changes
from cushing_survey_gate import fit_gate
from fit_cushing_survey_ensemble import add_survey, exact_weights, survey_bias


class SurveyForecastTest(unittest.TestCase):
    def test_current_cushing_change_only(self):
        cases = [
            ("Distillates fell versus last week's draw, while Cushing rose by 749,000 barrels.", {749.}),
            ("In the week prior, the API saw a Cushing decrease of 750,000 barrels.", set()),
            ("Cushing saw a 2.066 million-barrel decrease this week.", {-2066.}),
            ("Cushing was unavailable. Gasoline fell by 3 million barrels.", set()),
            ("Cushing inventories saw no change.", {0.}),
            ("Cushing inventories rose by 882,000 million barrels.", set()),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(set(extract_changes([text])), expected)

    def test_publication_at_cutoff_is_unavailable(self):
        cutoff = pd.Timestamp("2026-11-18T05:00:00Z")
        rows = pd.DataFrame([dict(release_date="2026-11-18", week_ending="2026-11-13", cutoff=cutoff)])
        quote = pd.DataFrame([dict(release_date="2026-11-18", week_ending="2026-11-13", available_utc=cutoff,
                                   api_change_kbbl=125.)]).set_index("release_date")
        self.assertTrue(add_survey(rows, quote).survey.isna().all())
        quote["available_utc"] -= pd.Timedelta(seconds=1)
        self.assertEqual(add_survey(rows, quote).survey.iat[0], .125)

    def test_future_targets_cannot_change_bias(self):
        times = pd.date_range("2020-01-01", periods=26, freq="7D", tz="UTC")
        d = pd.DataFrame(dict(known=times, target=.1, survey=0.))
        d.loc[d.known >= times[15], "target"] = 9999
        self.assertAlmostEqual(survey_bias(d, times[15]), .1)

    def test_gate_ignores_unavailable_or_missing_references(self):
        times = pd.date_range("2020-01-01", periods=64, freq="7D", tz="UTC")
        d = pd.DataFrame(dict(cutoff=times, known=times+pd.Timedelta(days=1), target=.05, survey=.1,
                              strong_reference=0., vol13=1., persistence13=.2, basis_move5=.1))
        d.loc[4, "strong_reference"] = np.nan
        a, _ = fit_gate(d, times[51])
        d.loc[d.known >= times[51], "target"] = 1000
        b, _ = fit_gate(d, times[51])
        self.assertEqual(a, b)
        self.assertEqual(a["training_rows"] + a["stopping_rows"], 50)

    def test_exact_minimum_is_attained(self):
        x = np.array([[1.,0.], [2.,0.], [3.,0.]])
        y = x @ np.array([.7, .3])
        np.testing.assert_allclose(exact_weights(x, y, np.ones(3)), [.7, .3])


if __name__ == "__main__":
    unittest.main()
