from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from q_lab_hl.backtest import run_backtest
from q_lab_hl.config import ExecutionConfig
from q_lab_hl.data import DataStore, MarketPanels


def make_store() -> DataStore:
    index = pd.date_range("2025-01-01", periods=4, freq="h")
    open_prices = pd.DataFrame({"A": [100.0, 100.0, 110.0, 110.0], "B": [100.0, 100.0, 100.0, 100.0]}, index=index)
    close_prices = pd.DataFrame({"A": [100.0, 110.0, 110.0, 110.0], "B": [100.0, 100.0, 100.0, 100.0]}, index=index)
    panels = MarketPanels(
        open=open_prices,
        high=close_prices,
        low=open_prices,
        close=close_prices,
        volume=pd.DataFrame({"A": [1_000_000] * 4, "B": [1_000_000] * 4}, index=index),
        funding=pd.DataFrame({"A": [0.0, 0.001, 0.0, 0.0], "B": [0.0, 0.0, 0.0, 0.0]}, index=index),
        open_interest=pd.DataFrame({"A": [10_000_000] * 4, "B": [10_000_000] * 4}, index=index),
        tradable=pd.DataFrame({"A": [True] * 4, "B": [True] * 4}, index=index),
        metadata={"A": {"sector": "L1"}, "B": {"sector": "L2"}},
    )
    return DataStore(panels)


class BacktestTests(unittest.TestCase):
    def test_backtest_uses_next_bar_execution_and_applies_funding(self):
        store = make_store()
        strategy = SimpleNamespace(
            reset_state=lambda: None,
            signals=lambda data, ts: pd.Series({"A": 1.0, "B": -1.0}),
            construct=lambda scores, data, ts: pd.Series({"A": 0.5, "B": -0.5}),
            risk=lambda weights, data, ts: pd.Series(weights),
        )
        result = run_backtest(
            strategy,
            store,
            timestamps=store.index,
            execution=ExecutionConfig(
                rebalance_every_bars=1,
                target_gross_exposure=1.0,
                target_net_exposure=0.0,
                max_gross_exposure=1.0,
                max_abs_weight=0.6,
                max_group_gross=1.0,
                taker_fee_bps=0.0,
                slippage_bps=0.0,
                min_history_bars=0,
                min_dollar_volume=0.0,
                min_price=0.0,
                listing_cooldown_bars=0,
            ),
        )
        self.assertAlmostEqual(float(result.returns.iloc[0]), 0.0, places=9)
        self.assertAlmostEqual(float(result.price_pnl.iloc[1]), 0.05, places=9)
        self.assertAlmostEqual(float(result.funding_pnl.iloc[1]), -0.0005, places=9)
        self.assertGreater(float(result.diagnostics["avg_child_orders_per_rebalance"]), 0.0)
        self.assertEqual(float(result.diagnostics["skipped_notional_ratio"]), 0.0)

    def test_backtest_tracks_implementation_shortfall(self):
        store = make_store()
        strategy = SimpleNamespace(
            reset_state=lambda: None,
            signals=lambda data, ts: pd.Series({"A": 1.0, "B": -1.0}),
            construct=lambda scores, data, ts: pd.Series({"A": 0.01, "B": -0.01}),
            risk=lambda weights, data, ts: pd.Series(weights),
        )
        result = run_backtest(
            strategy,
            store,
            timestamps=store.index,
            execution=ExecutionConfig(
                rebalance_every_bars=1,
                target_gross_exposure=1.0,
                target_net_exposure=0.0,
                max_gross_exposure=1.0,
                max_abs_weight=0.6,
                max_group_gross=1.0,
                taker_fee_bps=0.0,
                slippage_bps=0.0,
                min_history_bars=0,
                min_dollar_volume=0.0,
                min_price=0.0,
                listing_cooldown_bars=0,
                min_trade_notional_usd=10_000.0,
            ),
        )
        self.assertGreater(float(result.diagnostics["skipped_notional_total_usd"]), 0.0)
        self.assertGreater(float(result.diagnostics["skipped_notional_ratio"]), 0.0)


if __name__ == "__main__":
    unittest.main()
