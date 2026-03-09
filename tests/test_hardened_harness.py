from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import prepare
import strategy


def make_store() -> prepare.DataStore:
    dates = pd.bdate_range("2023-01-02", periods=320)
    signal_prices = pd.DataFrame(
        {
            "AAA": np.linspace(100, 140, len(dates)),
            "BBB": np.linspace(90, 118, len(dates)),
            "SPY": np.linspace(400, 420, len(dates)),
        },
        index=dates,
    )
    total_return_prices = signal_prices.copy()
    open_prices = signal_prices.copy()
    volumes = pd.DataFrame(
        {
            "AAA": 200_000,
            "BBB": 180_000,
            "SPY": 1_000_000,
        },
        index=dates,
    )
    market_caps = pd.DataFrame(
        {
            "AAA": 2_000_000_000.0,
            "BBB": 1_500_000_000.0,
            "SPY": 100_000_000_000.0,
        },
        index=dates,
    )

    def panel(values):
        return pd.DataFrame(values, index=dates)

    raw_fundamentals = {
        "net_income": panel({"AAA": 100_000_000.0, "BBB": 75_000_000.0, "SPY": 1.0}),
        "gross_profit": panel({"AAA": 400_000_000.0, "BBB": 280_000_000.0, "SPY": 1.0}),
        "assets": panel({"AAA": 2_000_000_000.0, "BBB": 1_700_000_000.0, "SPY": 1.0}),
        "book_equity": panel({"AAA": 900_000_000.0, "BBB": 700_000_000.0, "SPY": 1.0}),
        "current_assets": panel({"AAA": 700_000_000.0, "BBB": 600_000_000.0, "SPY": 1.0}),
        "current_liabilities": panel({"AAA": 300_000_000.0, "BBB": 310_000_000.0, "SPY": 1.0}),
        "debt": panel({"AAA": 250_000_000.0, "BBB": 350_000_000.0, "SPY": 0.0}),
        "shares_out": panel({"AAA": 20_000_000.0, "BBB": 18_000_000.0, "SPY": 1.0}),
    }
    legacy_fundamentals = {
        "piotroski": panel({"AAA": 7.0, "BBB": 5.0, "SPY": 0.0}),
    }
    macro_vintage_table = pd.DataFrame(
        [
            {
                "field": "cpi",
                "observation_date": pd.Timestamp("2023-12-01"),
                "value": 300.0,
                "vintage_date": pd.Timestamp("2024-01-10"),
                "first_trade_date": dates[8],
            }
        ]
    )
    market_macro = {"vix": pd.Series(np.linspace(12, 20, len(dates)), index=dates)}
    metadata = {
        "AAA": {"country": "US", "sector": "Tech", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
        "BBB": {"country": "US", "sector": "Industrials", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
        "SPY": {"country": "US", "sector": "ETF", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
    }
    sp500_membership = {
        "AAA": [(dates[0], None)],
        "BBB": [(dates[0], None)],
        "SPY": [(dates[0], None)],
    }
    return prepare.DataStore(
        signal_prices=signal_prices,
        total_return_prices=total_return_prices,
        open_prices=open_prices,
        volumes=volumes,
        market_caps=market_caps,
        raw_fundamental_panels=raw_fundamentals,
        legacy_fundamentals=legacy_fundamentals,
        macro_vintage_table=macro_vintage_table,
        market_macro=market_macro,
        metadata=metadata,
        sp500_membership=sp500_membership,
    )


class PrepareMathTests(unittest.TestCase):
    def test_expected_max_increases_with_trial_count(self):
        self.assertGreater(
            prepare.expected_max_standard_normal_exact(10),
            prepare.expected_max_standard_normal_exact(2),
        )

    def test_bootstrap_uses_adaptive_block_length(self):
        short = pd.Series(np.sin(np.linspace(0, 10, 90)) / 100)
        long = pd.Series(np.sin(np.linspace(0, 18, 240)) / 100)
        _, _, short_block, _ = prepare.bootstrap_sharpe_ci(short, n_bootstrap=80)
        _, _, long_block, _ = prepare.bootstrap_sharpe_ci(long, n_bootstrap=80)
        self.assertGreater(short_block, 0)
        self.assertGreater(long_block, 0)
        self.assertNotEqual(short_block, long_block)

    def test_effective_trials_bounded_by_raw_trials(self):
        matrix = pd.DataFrame(
            {
                "a": [0.01, 0.02, -0.01, 0.00],
                "b": [0.01, 0.02, -0.01, 0.00],
                "c": [0.00, -0.01, 0.01, 0.02],
            }
        )
        eff = prepare.estimate_effective_independent_trials(matrix)
        self.assertGreaterEqual(eff, 1.0)
        self.assertLessEqual(eff, matrix.shape[1])


class DataStoreTests(unittest.TestCase):
    def test_latest_fundamental_derives_earnings_yield(self):
        store = make_store()
        series = store.latest_fundamental("earnings_yield", store._signal_prices.index[-1])
        expected = (100_000_000.0 * 4) / 2_000_000_000.0
        self.assertAlmostEqual(series["AAA"], expected, places=6)

    def test_strategy_signals_use_helper_api(self):
        store = make_store()
        limited = prepare.DateLimitedStore(store, store._signal_prices.index[-1])
        old_holdings = strategy.NUM_HOLDINGS
        strategy.NUM_HOLDINGS = 2
        try:
            score = strategy.signals(limited, store._signal_prices.index[-1])
        finally:
            strategy.NUM_HOLDINGS = old_holdings
        self.assertFalse(score.empty)
        self.assertIn("AAA", score.index)


class BacktestTests(unittest.TestCase):
    def test_backtest_trades_next_day_not_same_day(self):
        dates = pd.bdate_range("2024-01-01", periods=12)
        signal_prices = pd.DataFrame(
            {"AAA": [100, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200], "SPY": 100.0},
            index=dates,
        )
        total_return = signal_prices.copy()
        volumes = pd.DataFrame({"AAA": 500_000, "SPY": 1_000_000}, index=dates)
        market_caps = pd.DataFrame({"AAA": 2_000_000_000.0, "SPY": 100_000_000_000.0}, index=dates)
        metadata = {
            "AAA": {"country": "US", "sector": "Tech", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
            "SPY": {"country": "US", "sector": "ETF", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
        }
        store = prepare.DataStore(
            signal_prices=signal_prices,
            total_return_prices=total_return,
            open_prices=signal_prices.copy(),
            volumes=volumes,
            market_caps=market_caps,
            raw_fundamental_panels={},
            legacy_fundamentals={},
            macro_vintage_table=pd.DataFrame(),
            market_macro={},
            metadata=metadata,
            sp500_membership={"AAA": [(dates[0], None)], "SPY": [(dates[0], None)]},
        )

        class OneAssetStrategy:
            REBALANCE_FREQ = "W"

            @staticmethod
            def signals(data, date):
                return pd.Series({"AAA": 1.0})

            @staticmethod
            def construct(scores, data, date):
                return pd.Series({"AAA": 1.0})

            @staticmethod
            def risk(weights, data, date):
                return pd.Series({"AAA": 1.0})

        result = prepare.run_backtest(OneAssetStrategy, store, str(dates[0].date()), str(dates[-1].date()), rebalance_freq="W")
        self.assertAlmostEqual(result.daily_returns.iloc[1], -0.002, places=8)
        self.assertIn(dates[1], result.trade_dates)


if __name__ == "__main__":
    unittest.main()
