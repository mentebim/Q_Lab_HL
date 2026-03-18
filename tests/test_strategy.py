from __future__ import annotations

import unittest
from dataclasses import replace

import pandas as pd

from q_lab_hl.backtest import load_strategy
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore
from strategy_model import (
    LINEAR_CROSS_SECTION_FAMILY,
    FeatureSpec,
    ModelSpec,
    StrategySpec,
    TargetSpec,
    build_training_dataset,
    strategy_spec_from_dict,
    validate_strategy_spec,
)


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
        self.assertEqual(summary["strategy_spec"]["target"]["kind"], "forward_close_return")

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


class TrainWindowValidationTests(unittest.TestCase):
    """Test that train_window_bars < 2 * horizon is rejected at validation."""

    def _make_spec(self, train_window_bars: int, horizon: int) -> StrategySpec:
        return StrategySpec(
            train_window_bars=train_window_bars,
            min_train_rows=20,
            position_bucket=4,
            features=(
                FeatureSpec(name="ret_1h", kind="return", lookback=1, transform="rank"),
            ),
            target=TargetSpec(name="fwd", kind="forward_close_return", horizon=horizon),
            model=ModelSpec(family="ols"),
        )

    def test_rejects_train_window_less_than_2x_horizon(self):
        spec = self._make_spec(train_window_bars=24, horizon=48)
        with self.assertRaises(ValueError, msg="tw=24, h=48 should be rejected"):
            validate_strategy_spec(spec, LINEAR_CROSS_SECTION_FAMILY)

    def test_rejects_train_window_equal_to_horizon(self):
        spec = self._make_spec(train_window_bars=48, horizon=48)
        with self.assertRaises(ValueError, msg="tw=48, h=48 should be rejected"):
            validate_strategy_spec(spec, LINEAR_CROSS_SECTION_FAMILY)

    def test_accepts_train_window_at_2x_horizon(self):
        spec = self._make_spec(train_window_bars=96, horizon=48)
        validate_strategy_spec(spec, LINEAR_CROSS_SECTION_FAMILY)

    def test_accepts_train_window_above_2x_horizon(self):
        spec = self._make_spec(train_window_bars=504, horizon=48)
        validate_strategy_spec(spec, LINEAR_CROSS_SECTION_FAMILY)


class MaturityFilterTests(unittest.TestCase):
    """Test that build_training_dataset only includes rows where sample_ts + horizon <= fit_ts."""

    def test_training_rows_are_mature(self):
        store = DataStore.synthetic(n_assets=20, periods=24 * 30, seed=42)
        horizon = 48
        spec = StrategySpec(
            train_window_bars=200,
            min_train_rows=10,
            position_bucket=4,
            features=(
                FeatureSpec(name="ret_1h", kind="return", lookback=1, transform="rank"),
            ),
            target=TargetSpec(name="fwd", kind="forward_close_return", horizon=horizon),
            model=ModelSpec(family="ols"),
        )
        execution = ExecutionConfig(
            rebalance_every_bars=48,
            min_history_bars=0,
            min_dollar_volume=0.0,
            min_price=0.0,
            listing_cooldown_bars=0,
        )
        ts = store.index[-2]
        dataset = build_training_dataset(store, ts, execution=execution, strategy_spec=spec)
        train_ts = dataset["train_timestamps"]
        if len(train_ts) == 0:
            self.skipTest("No training rows on synthetic data")
        index = store.index
        ts_pos = int(index.get_loc(ts))
        for sample_ts in train_ts:
            sample_pos = int(index.get_loc(sample_ts))
            maturity_pos = sample_pos + horizon
            self.assertLess(maturity_pos, len(index), "maturity pos out of bounds")
            maturity_ts = index[maturity_pos]
            self.assertLessEqual(
                maturity_ts, ts,
                f"Training row at {sample_ts} has maturity {maturity_ts} > fit time {ts} — lookahead",
            )

    def test_maturity_diagnostics_present(self):
        store = DataStore.synthetic(n_assets=20, periods=24 * 30, seed=42)
        spec = StrategySpec(
            train_window_bars=200,
            min_train_rows=10,
            position_bucket=4,
            features=(
                FeatureSpec(name="ret_1h", kind="return", lookback=1, transform="rank"),
            ),
            target=TargetSpec(name="fwd", kind="forward_close_return", horizon=48),
            model=ModelSpec(family="ols"),
        )
        execution = ExecutionConfig(
            rebalance_every_bars=48,
            min_history_bars=0,
            min_dollar_volume=0.0,
            min_price=0.0,
            listing_cooldown_bars=0,
        )
        ts = store.index[-2]
        dataset = build_training_dataset(store, ts, execution=execution, strategy_spec=spec)
        self.assertIn("n_unique_train_timestamps", dataset)
        self.assertIn("matured_train_share", dataset)
        self.assertIsInstance(dataset["n_unique_train_timestamps"], int)
        self.assertGreaterEqual(dataset["matured_train_share"], 0.0)
        self.assertLessEqual(dataset["matured_train_share"], 1.0)


class RuntimeTrainQualityGateTests(unittest.TestCase):
    """Test that signals() returns empty when n_unique_train_timestamps is below threshold."""

    def test_signals_empty_when_insufficient_mature_timestamps(self):
        # Use a small synthetic dataset where horizon=48 and train_window=96
        # leaves very few mature timestamps.
        store = DataStore.synthetic(n_assets=20, periods=150, seed=99)
        strategy = load_strategy("strategy.py")
        strategy.reset_state()
        # Override to a spec where the train window barely passes validation
        # but the short data means few mature timestamps at early rebalance points
        tight_spec = StrategySpec(
            train_window_bars=96,
            min_train_rows=200,
            position_bucket=4,
            features=(
                FeatureSpec(name="ret_1h", kind="return", lookback=1, transform="rank"),
            ),
            target=TargetSpec(name="fwd", kind="forward_close_return", horizon=48),
            model=ModelSpec(family="ols"),
        )
        strategy.SPEC = tight_spec
        strategy.EXECUTION = ExecutionConfig(
            rebalance_every_bars=48,
            min_history_bars=0,
            min_dollar_volume=0.0,
            min_price=0.0,
            listing_cooldown_bars=0,
        )
        # Pick an early timestamp where mature data is still below the
        # runtime threshold of max(24, horizon)=48 unique timestamps.
        ts = store.index[60]
        scores = strategy.signals(store, ts)
        # At pos 60 with tw=96 and h=48, only timestamps up to pos 12 are mature,
        # so the fit has 13 unique mature timestamps, which is below the runtime
        # threshold of 48. signals() should therefore return empty.
        self.assertEqual(len(scores), 0)

    def test_train_quality_in_fit_summary(self):
        store = DataStore.synthetic(n_assets=24, periods=24 * 60, seed=11)
        strategy = load_strategy("strategy.py")
        strategy.reset_state()
        ts = store.index[-2]
        scores = strategy.signals(store, ts)
        if len(scores) == 0:
            self.skipTest("No scores produced on synthetic data")
        summary = strategy.last_fit_summary()
        self.assertIn("train_quality", summary)
        tq = summary["train_quality"]
        self.assertIn("n_unique_train_timestamps", tq)
        self.assertIn("matured_train_share", tq)
        self.assertIn("target_horizon", tq)
        self.assertIn("train_window_bars", tq)
        self.assertGreater(tq["n_unique_train_timestamps"], 0)
        self.assertGreater(tq["matured_train_share"], 0.0)


if __name__ == "__main__":
    unittest.main()
