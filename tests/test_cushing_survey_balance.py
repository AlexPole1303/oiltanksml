import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cushing_survey_reconciliation as rec
import fit_cushing_survey_balance as fit
from collect_cushing_product_surveys import extract
from collect_cushing_survey_precision import dispatch_rounding, parse
from prepare_cushing_balance_inputs import GROUPS, stamped_values


class SurveyBalanceTest(unittest.TestCase):
    def test_actual_week_and_dispatch_precision(self):
        stamp = "2025-09-09T20:35:05Z"
        detail = "美国至9月5日当周API库欣原油库存 -33.2万桶，前值206.3万桶"
        self.assertEqual(parse(detail, stamp), ("2025-09-05", -332))
        self.assertIsNone(parse(detail.replace("9月5日", "8月29日"), stamp))
        self.assertIsNone(parse(detail.replace("API", "EIA"), stamp))
        dispatch = "美国API原油库存-100万桶，上周API库欣原油库存-30万桶，汽油库存20万桶，馏分油库存30万桶。"
        self.assertEqual(dispatch_rounding(dispatch), 50)
        self.assertEqual(dispatch_rounding(dispatch.replace("馏分油库存30", "馏分油库存30.1")), 0)
        self.assertEqual(dispatch_rounding(detail), 0)

    def test_fuel_ignores_prior_forecast_and_eia(self):
        news = (
            "Gasoline inventories fell by 1.3 million barrels, compared with 2 million barrels last week. "
            "Analysts expected distillates to increase by 3 million barrels. "
            "Distillate stocks rose by 700 thousand barrels. The EIA reported gasoline rose by 2 million barrels."
        )
        self.assertEqual(extract(news), {"gasoline": [-1300], "distillate": [700]})

    def test_inputs_have_separate_strict_clocks(self):
        now = pd.Timestamp("2025-09-10T04:00:00Z")
        wk = "2025-09-05"
        row = {k: 17 for cols in GROUPS.values() for k in cols}
        row.update(dict.fromkeys(GROUPS, now))
        row.update(week_ending=wk, curve_time=now - pd.Timedelta(seconds=1))
        obs = stamped_values(row, now, wk)
        self.assertEqual(obs["slope_change"], 17)
        self.assertTrue(np.isnan(obs["temp_change"]))
        with self.assertRaises(ValueError):
            stamped_values(row, now, "2025-08-29")
        fuel = pd.DataFrame(
            [
                {
                    "week_ending": wk,
                    "product": "gasoline",
                    "value_kbbl": 1500,
                    "available_utc": now,
                    "published_utc": now,
                }
            ]
        )
        self.assertEqual(fit.product_features(fuel, wk, now)["product_balance"], 0)
        self.assertEqual(fit.product_features(fuel, wk, now + pd.Timedelta(seconds=1))["product_balance"], 0.5)
        with self.assertRaises(ValueError):
            fit.product_features(pd.concat([fuel, fuel]), wk, now)

    def test_state_skips_late_labels_and_decays_across_missing_weeks(self):
        now = pd.Timestamp("2025-09-10T04:00:00Z")
        data = pd.DataFrame(
            {
                "known": [now - pd.Timedelta(days=7), now],
                "week_ending": ["2025-08-29", "2025-09-05"],
                "target": [0.5, 75.0],
                "survey": [0.1, -30.0],
            }
        )
        early = rec.states(data, "2025-09-05", now)
        later = rec.states(data, "2025-09-12", now)
        self.assertEqual(later["quick"], 0)
        self.assertEqual(early["previous_survey"], 0.1)
        self.assertTrue(np.isnan(later["previous_survey"]))
        self.assertAlmostEqual(later["persistent"], rec.FILTERS["persistent"][0] * early["persistent"])
        for k, v in rec.states(data.iloc[:1], "2025-09-12", now).items():
            np.testing.assert_equal(later[k], v)
        with self.assertRaises(ValueError):
            rec.apply_overlay({"latest": now.isoformat()}, pd.DataFrame({"cutoff": [now]}))

    def test_ledger_priority_changes_only_after_publication(self):
        day = "2025-09-10"
        now = pd.Timestamp("2025-09-10T03:00:00Z")

        def ledger(t, number):
            return pd.DataFrame(
                [{"release_date": day, "available_utc": t, "api_change_kbbl": number}]
            ).set_index("release_date")

        ledgers = [ledger(now - pd.Timedelta(minutes=2), -300), ledger(now, -332)]
        self.assertEqual(fit.dated_quote(day, now, ledgers).api_change_kbbl, -300)
        self.assertEqual(fit.dated_quote(day, now + pd.Timedelta(seconds=1), ledgers).api_change_kbbl, -332)
        self.assertIsNone(fit.dated_quote(day, now - pd.Timedelta(minutes=3), ledgers))

    def test_published_outputs_use_the_same_original_survey_cohort(self):
        rows = pd.read_csv(fit.MODEL_DIR / "balance_forecasts.csv")
        reference = pd.read_csv(fit.MODEL_DIR / "measurement_forecasts.csv")
        scores = json.loads((fit.MODEL_DIR / "balance_evaluation.json").read_bytes())["scores"]
        self.assertEqual(list(rows.release_date), list(reference.release_date))
        np.testing.assert_array_equal(rows.original_quote.notna(), reference.survey.notna())
        np.testing.assert_allclose(rows.baseline, reference.strict, rtol=0, atol=1e-10)
        for name, g in (("all", rows), ("original_quoted", rows[reference.survey.notna()])):
            self.assertAlmostEqual(
                abs(g.estimate - g.target).mean() * 1000, scores[name]["estimate"]["mae_kbbl"]
            )

    def test_predict_from_saved_model_before_and_after_survey(self):
        args = Namespace(
            predict=fit.MODEL_DIR / "nov18_inputs.json",
            confirmed_ledger=fit.v4.EXTRA,
            survey_ledger=fit.v3.SURVEY,
            wire_ledger=fit.v5.WIRE,
            precision_ledger=fit.PRECISION,
            measurements=fit.v5.MEASUREMENTS,
            balance_inputs=fit.EXTERNAL,
            products=fit.MODEL_DIR / "product_observations.csv",
            settled_feedback=None,
            as_of="2026-11-14T12:00:00Z",
        )
        early = fit.predict_from_files(args)
        args.as_of = "2026-11-18T05:00:00Z"
        late = fit.predict_from_files(args)
        self.assertFalse(early["survey_available"])
        self.assertTrue(late["survey_available"])
        self.assertTrue(np.isfinite([early["estimate_kbbl"], late["estimate_kbbl"]]).all())


if __name__ == "__main__":
    unittest.main()
