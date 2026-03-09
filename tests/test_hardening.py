import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

import prepare
import strategy


class FakeStrategyData:
    def __init__(self):
        self.dates = pd.bdate_range("2024-01-02", periods=260)
        tickers = [f"T{i:03d}" for i in range(140)]
        base = np.linspace(100, 140, len(self.dates))
        self.price_frame = pd.DataFrame(
            {
                ticker: base + i * 3 + np.sin(np.arange(len(self.dates)) / (7 + i))
                for i, ticker in enumerate(tickers)
            },
            index=self.dates,
        )
        self.market_caps = pd.Series({ticker: 10_000_000_000 + i * 1_000_000_000 for i, ticker in enumerate(tickers)})
        sector_cycle = ["Tech", "Health", "Industrials", "Utilities"]
        self.sectors = {ticker: sector_cycle[i % len(sector_cycle)] for i, ticker in enumerate(tickers)}
        self.fundamentals = {
            "book_to_price": pd.Series({ticker: 0.4 + i * 0.1 for i, ticker in enumerate(tickers)}),
            "gross_profitability": pd.Series({ticker: 0.1 + i * 0.02 for i, ticker in enumerate(tickers)}),
            "asset_growth": pd.Series({ticker: 0.05 - i * 0.005 for i, ticker in enumerate(tickers)}),
            "current_ratio": pd.Series({ticker: 1.1 + i * 0.1 for i, ticker in enumerate(tickers)}),
        }
        self.vix = 18.0

    def tradable_universe(self, date, **kwargs):
        return list(self.price_frame.columns)

    def prices_signal(self, tickers=None, start=None, end=None):
        df = self.price_frame
        if tickers is not None:
            df = df[tickers]
        if start is not None:
            df = df.loc[pd.Timestamp(start) :]
        if end is not None:
            df = df.loc[: pd.Timestamp(end)]
        return df

    def latest_fundamental(self, field, date):
        return self.fundamentals[field]

    def latest_macro(self, field, date):
        if field == "vix":
            return self.vix
        raise KeyError(field)

    def factor_rank(self, series, method="pct"):
        return pd.Series(series).rank(pct=True)

    def winsorize_cross_section(self, series, lower_pct=0.02, upper_pct=0.98):
        return pd.Series(series)

    def neutralize_cross_section(self, series, by):
        return pd.Series(series)

    def market_cap(self, date):
        return self.market_caps

    def sector(self, ticker):
        return self.sectors[ticker]

    def fundamental(self, field):
        raise AssertionError("baseline strategy should not call raw fundamental()")


class HardeningTests(unittest.TestCase):
    def test_strategy_signals_use_helper_api(self):
        data = FakeStrategyData()
        scores = strategy.signals(data, data.dates[-1])
        self.assertFalse(scores.empty)
        self.assertTrue(scores.index.isin(data.price_frame.columns).all())

    def test_strategy_construct_and_risk_keep_cash_explicit(self):
        data = FakeStrategyData()
        scores = strategy.signals(data, data.dates[-1])
        weights = strategy.construct(scores, data, data.dates[-1])
        self.assertIn("__CASH__", weights.index)
        risked = strategy.risk(weights, data, data.dates[-1])
        self.assertAlmostEqual(float(risked.sum()), 1.0, places=6)
        self.assertGreaterEqual(float(risked["__CASH__"]), 0.0)

    def test_normalize_target_weights_rejects_negative_cash(self):
        with self.assertRaises(ValueError):
            prepare.normalize_target_weights(pd.Series({"AAA": 1.2, "__CASH__": -0.2}))

    def test_effective_independent_trials_is_bounded(self):
        idx = pd.bdate_range("2024-01-01", periods=30)
        matrix = pd.DataFrame(
            {
                "a": np.linspace(-0.01, 0.02, len(idx)),
                "b": np.linspace(-0.01, 0.02, len(idx)) * 0.9,
                "c": np.linspace(0.02, -0.01, len(idx)),
            },
            index=idx,
        )
        n_eff = prepare.estimate_effective_independent_trials(matrix)
        self.assertGreaterEqual(n_eff, 1.0)
        self.assertLessEqual(n_eff, 3.0)

    def test_run_backtest_enforces_next_day_execution(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        prices = pd.DataFrame(
            {
                "AAA": [100.0, 101.0, 111.1, 111.1, 111.1],
                "SPY": [100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=dates,
        )
        volumes = pd.DataFrame({"AAA": [1_000_000] * 5, "SPY": [1_000_000] * 5}, index=dates)
        market_caps = pd.DataFrame({"AAA": [1_000_000_000] * 5, "SPY": [1_000_000_000] * 5}, index=dates)
        store = prepare.DataStore(
            signal_prices=prices,
            total_return_prices=prices,
            open_prices=prices,
            volumes=volumes,
            market_caps=market_caps,
            raw_fundamental_panels={},
            legacy_fundamentals={},
            macro_vintage_table=pd.DataFrame(),
            market_macro={},
            metadata={"AAA": {"country": "US", "sector": "Tech"}, "SPY": {"country": "US", "sector": "ETF"}},
            sp500_membership={},
        )
        module = SimpleNamespace(
            REBALANCE_FREQ="M",
            signals=lambda data, date: pd.Series({"AAA": 1.0}),
            construct=lambda scores, data, date: pd.Series({"AAA": 1.0}),
            risk=lambda weights, data, date: pd.Series({"AAA": 1.0}),
        )
        result = prepare.run_backtest(module, store, str(dates[0].date()), str(dates[-1].date()))
        self.assertAlmostEqual(float(result.daily_returns.iloc[1]), -0.002, places=9)
        self.assertAlmostEqual(float(result.daily_returns.iloc[2]), 0.10, places=6)


if __name__ == "__main__":
    unittest.main()
