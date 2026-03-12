from __future__ import annotations

import unittest

from q_lab_hl.backtest import load_strategy
from q_lab_hl.data import DataStore


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


if __name__ == "__main__":
    unittest.main()
