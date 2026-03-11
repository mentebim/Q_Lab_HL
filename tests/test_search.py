from __future__ import annotations

import unittest

from q_lab_hl.data import DataStore
from q_lab_hl.search import TimeSeriesCVConfig, build_time_series_cv_folds, build_walk_forward_folds, enumerate_param_grid, run_walk_forward_grid
from q_lab_hl.search import temporary_module_params

import strategy


class SearchTests(unittest.TestCase):
    def test_enumerate_param_grid_cartesian_product(self):
        grid = {"a": [1, 2], "b": ["x", "y", "z"]}
        combos = enumerate_param_grid(grid)
        self.assertEqual(len(combos), 6)

    def test_build_walk_forward_folds_returns_non_empty_blocks(self):
        store = DataStore.synthetic(n_assets=24, periods=24 * 120, seed=2)
        folds = build_walk_forward_folds(store.index, n_folds=3)
        self.assertEqual(len(folds), 3)
        self.assertTrue(all(len(fold["train"]) > 0 and len(fold["validation"]) > 0 for fold in folds))

    def test_build_expanding_cv_folds(self):
        store = DataStore.synthetic(n_assets=24, periods=24 * 120, seed=2)
        folds = build_time_series_cv_folds(
            store.index,
            cv=TimeSeriesCVConfig(mode="expanding", n_folds=3, train_size=200, validation_size=100, step_size=100, gap_size=10),
        )
        self.assertEqual(len(folds), 3)
        self.assertEqual(len(folds[0]["train"]), 200)
        self.assertEqual(len(folds[0]["validation"]), 100)
        self.assertLess(folds[0]["train"].max(), folds[0]["validation"].min())

    def test_build_rolling_cv_folds(self):
        store = DataStore.synthetic(n_assets=24, periods=24 * 120, seed=2)
        folds = build_time_series_cv_folds(
            store.index,
            cv=TimeSeriesCVConfig(mode="rolling", n_folds=2, train_size=240, validation_size=120, step_size=120, gap_size=24),
        )
        self.assertEqual(len(folds), 2)
        self.assertEqual(len(folds[0]["train"]), 240)
        self.assertEqual(len(folds[0]["validation"]), 120)

    def test_build_expanding_cv_with_purge_and_embargo(self):
        store = DataStore.synthetic(n_assets=24, periods=24 * 120, seed=2)
        folds = build_time_series_cv_folds(
            store.index,
            cv=TimeSeriesCVConfig(
                mode="expanding",
                n_folds=2,
                train_size=300,
                validation_size=120,
                gap_size=12,
                purge_size=12,
                embargo_size=24,
            ),
        )
        self.assertEqual(len(folds), 2)
        self.assertEqual(len(folds[0]["train"]), 300)
        self.assertLess(folds[0]["train"].max(), folds[0]["validation"].min())

    def test_run_walk_forward_grid_smoke(self):
        store = DataStore.synthetic(n_assets=24, periods=24 * 120, seed=2)
        results, folds = run_walk_forward_grid(
            strategy,
            store,
            cv=TimeSeriesCVConfig(mode="expanding", n_folds=2, train_size=300, validation_size=120, step_size=120, gap_size=24),
            param_grid={
                "POSITION_BUCKET": [3, 4],
                "LOOKBACK_SHORT_HOURS": [6],
            },
        )
        self.assertEqual(len(folds), 2)
        self.assertEqual(len(results), 2)
        self.assertIn("validation_score_median", results.columns)

    def test_all_model_families_produce_scores(self):
        store = DataStore.synthetic(n_assets=24, periods=24 * 120, seed=2)
        ts = store.index[-1]
        for family in ["residual_reversal", "funding_dislocation", "beta_neutral_momentum"]:
            with temporary_module_params(
                strategy,
                {
                    "MODEL_FAMILY": family,
                    "POSITION_BUCKET": 6,
                    "LOOKBACK_SHORT_HOURS": 6,
                    "LOOKBACK_MEDIUM_HOURS": 24,
                    "FUNDING_WINDOW_HOURS": 8,
                    "VOL_WINDOW_HOURS": 48,
                },
            ):
                scores = strategy.signals(store, ts)
                self.assertFalse(scores.empty, msg=family)
                self.assertTrue(scores.index.isin(store.assets).all(), msg=family)


if __name__ == "__main__":
    unittest.main()
