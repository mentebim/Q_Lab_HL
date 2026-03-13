from __future__ import annotations

import unittest

from q_lab_hl.backtest import load_strategy
from q_lab_hl.data import DataStore
from strategy_model import LINEAR_CROSS_SECTION_FAMILY, strategy_spec_from_dict


class StatisticalStrategyTests(unittest.TestCase):
    def test_strategy_emits_ranked_scores_and_fit_summary(self):
        store = DataStore.synthetic(n_assets=24, periods=24 * 60, seed=11)
        strategy = load_strategy("strategy.py")
        strategy.reset_state()
        ts = store.index[-2]
        scores = strategy.signals(store, ts)
        self.assertGreaterEqual(len(scores), 8)
        self.assertTrue(scores.is_monotonic_decreasing)
        summary = strategy.last_fit_summary()
        self.assertEqual(summary["model_fit"]["family"], "ols")
        self.assertGreater(summary["model_fit"]["n_train_rows"], 0)
        self.assertIn("funding_8h", summary["model_fit"]["coefficients"])
        self.assertIn("diagnostics", summary["model_fit"])
        self.assertEqual(summary["strategy_spec"]["target"]["kind"], "next_open_to_close_return")

    def test_strategy_family_rejects_unsupported_feature_kind(self):
        strategy = load_strategy("strategy.py")
        with self.assertRaises(ValueError):
            strategy_spec_from_dict(
                {
                    "features": [
                        {
                            "name": "bad_feature",
                            "kind": "orderflow",
                            "lookback": 1,
                            "transform": "zscore",
                        }
                    ]
                },
                base=strategy.DEFAULT_SPEC,
                strategy_family=LINEAR_CROSS_SECTION_FAMILY.family_id,
            )


if __name__ == "__main__":
    unittest.main()
