import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import prepare
import strategy


def make_store(num_tickers=70, periods=260):
    dates = pd.bdate_range("2023-01-02", periods=periods)
    tickers = [f"T{i:03d}" for i in range(num_tickers)]

    signal_prices = {}
    total_prices = {}
    open_prices = {}
    volumes = {}
    metadata = {}
    for i, ticker in enumerate(tickers):
        base = 20 + i
        trend = np.linspace(0, 5 + i / 10, periods)
        series = pd.Series(base + trend, index=dates, dtype=float)
        signal_prices[ticker] = series
        total_prices[ticker] = series
        open_prices[ticker] = series
        volumes[ticker] = pd.Series(1_000_000 + i * 10_000, index=dates, dtype=float)
        metadata[ticker] = {
            "country": "US",
            "sector": "Tech" if i % 2 == 0 else "Health",
            "exchange": "NASDAQ",
            "listing_start_date": dates[0],
            "listing_end_date": dates[-1],
        }

    return prepare.DataStore(
        signal_prices=pd.DataFrame(signal_prices),
        total_return_prices=pd.DataFrame(total_prices),
        open_prices=pd.DataFrame(open_prices),
        volumes=pd.DataFrame(volumes),
        market_caps=pd.DataFrame(index=dates),
        raw_fundamental_panels={},
        legacy_fundamentals={},
        macro_vintage_table=pd.DataFrame(),
        market_macro={"vix": pd.Series(15.0, index=dates)},
        metadata=metadata,
        sp500_membership={},
        cache_dir=prepare.CACHE_DIR,
    )


class FixedLongStrategy:
    REBALANCE_FREQ = "W"

    @staticmethod
    def signals(data, date):
        return pd.Series({"A": 1.0})

    @staticmethod
    def construct(scores, data, date):
        return pd.Series({"A": 1.0})

    @staticmethod
    def risk(weights, data, date):
        return pd.Series({"A": 1.0})


class FakeUniverseClient:
    def __init__(self, payloads):
        self.payloads = payloads

    def get(self, endpoint, params=None):
        params = params or {}
        if endpoint == "delisted-companies":
            page = params.get("page", 0)
            return self.payloads.get((endpoint, page), [])
        return self.payloads.get(endpoint, [])

    def load_parquet(self, name):
        payload = self.payloads.get(("parquet", name))
        if payload is None:
            return None
        return pd.DataFrame(payload)

    def save_parquet(self, df, name):
        self.payloads[("parquet", name)] = df.to_dict(orient="records")

    def cache_fresh(self, name, max_age_hours=24):
        return ("parquet", name) in self.payloads


class PrepareTests(unittest.TestCase):
    def test_next_trading_day_is_strictly_after_date(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        self.assertEqual(prepare.next_trading_day(dates, dates[0]), dates[1])

    def test_normalize_target_weights_adds_cash_and_rejects_negative_cash(self):
        weights = prepare.normalize_target_weights(pd.Series({"A": 0.4, "B": 0.5}))
        self.assertAlmostEqual(weights["__CASH__"], 0.1)
        with self.assertRaises(ValueError):
            prepare.normalize_target_weights(pd.Series({"A": 1.2, "B": 0.2}))
        with self.assertRaises(ValueError):
            prepare.normalize_target_weights(pd.Series({"A": 0.8, "B": -0.1, "__CASH__": 0.3}))

    def test_from_cache_hard_fails_on_missing_schema_version(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pd.DataFrame({"ticker": ["AAA"]}).to_parquet(tmp_path / "metadata.parquet", index=False)
            with self.assertRaises(RuntimeError):
                prepare.DataStore.from_cache(tmp_path)

    def test_from_cache_ignores_stale_files_outside_metadata(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pd.DataFrame(
                [{"ticker": "AAA", "schema_version": prepare.CACHE_SCHEMA_VERSION}]
            ).to_parquet(tmp_path / "metadata.parquet", index=False)
            pd.DataFrame(
                {"date": pd.bdate_range("2024-01-01", periods=3), "close": [10.0, 11.0, 12.0], "adjClose": [10.0, 11.0, 12.0], "open": [10.0, 11.0, 12.0], "volume": [1000, 1000, 1000]}
            ).to_parquet(tmp_path / "prices_AAA.parquet", index=False)
            pd.DataFrame(
                {"date": pd.bdate_range("2024-01-01", periods=3), "close": [20.0, 21.0, 22.0], "adjClose": [20.0, 21.0, 22.0], "open": [20.0, 21.0, 22.0], "volume": [1000, 1000, 1000]}
            ).to_parquet(tmp_path / "prices_BBB.parquet", index=False)
            pd.DataFrame(columns=["ticker", "start_date", "end_date"]).to_parquet(tmp_path / "sp500_membership.parquet", index=False)
            store = prepare.DataStore.from_cache(tmp_path)
            self.assertEqual(list(store.prices_signal().columns), ["AAA"])

    def test_legacy_fundamentals_require_real_filing_dates(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "date": "2024-03-31",
                        "earningsYieldTTM": 0.1,
                    }
                ]
            ).to_parquet(tmp_path / "fundamentals_AAA.parquet", index=False)
            dates = pd.bdate_range("2024-01-01", periods=120)
            with self.assertRaises(RuntimeError):
                prepare.DataStore._load_legacy_fundamentals(tmp_path, dates)

    def test_tradable_universe_uses_observation_count_not_consecutive_rows(self):
        dates = pd.bdate_range("2023-01-02", periods=300)
        a_series = pd.Series(np.nan, index=dates, dtype=float)
        observed_dates = dates[:260].delete(slice(0, 16, 2))
        a_series.loc[observed_dates] = np.linspace(20, 30, len(observed_dates))
        b_series = pd.Series(np.linspace(20, 25, len(dates)), index=dates, dtype=float)
        volume = pd.Series(1_000_000, index=dates, dtype=float)

        store = prepare.DataStore(
            signal_prices=pd.DataFrame({"A": a_series, "B": b_series}),
            total_return_prices=pd.DataFrame({"A": a_series, "B": b_series}),
            open_prices=pd.DataFrame({"A": a_series, "B": b_series}),
            volumes=pd.DataFrame({"A": volume, "B": volume}),
            market_caps=pd.DataFrame(index=dates),
            raw_fundamental_panels={},
            legacy_fundamentals={},
            macro_vintage_table=pd.DataFrame(),
            market_macro={},
            metadata={
                "A": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
                "B": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
            },
            sp500_membership={},
            cache_dir=prepare.CACHE_DIR,
        )

        universe = store.tradable_universe(
            date=dates[259],
            min_history_days=252,
            min_price=5.0,
            min_dollar_volume=1_000.0,
            countries=("US",),
        )
        self.assertIn("A", universe)
        self.assertIn("B", universe)

    def test_run_backtest_is_t_plus_1(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.DataFrame(
            {
                "A": [100.0, 120.0, 150.0, 150.0],
                "B": [100.0, 100.0, 100.0, 100.0],
            },
            index=dates,
        )
        volumes = pd.DataFrame({"A": 1_000_000.0, "B": 1_000_000.0}, index=dates)
        metadata = {
            "A": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
            "B": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
        }
        store = prepare.DataStore(
            signal_prices=prices,
            total_return_prices=prices,
            open_prices=prices,
            volumes=volumes,
            market_caps=pd.DataFrame(index=dates),
            raw_fundamental_panels={},
            legacy_fundamentals={},
            macro_vintage_table=pd.DataFrame(),
            market_macro={},
            metadata=metadata,
            sp500_membership={},
            cache_dir=prepare.CACHE_DIR,
        )

        result = prepare.run_backtest(FixedLongStrategy, store, str(dates[0].date()), str(dates[-1].date()), rebalance_freq="W")
        self.assertAlmostEqual(result.daily_returns.iloc[1], -0.002)
        self.assertAlmostEqual(result.daily_returns.iloc[2], 0.25)

    def test_run_backtest_treats_missing_held_returns_as_flat(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = pd.DataFrame(
            {
                "A": [100.0, 100.0, 110.0, np.nan, 121.0],
                "B": [100.0, 100.0, 100.0, 100.0, 100.0],
            },
            index=dates,
        )
        volumes = pd.DataFrame({"A": 1_000_000.0, "B": 1_000_000.0}, index=dates)
        metadata = {
            "A": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
            "B": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
        }
        store = prepare.DataStore(
            signal_prices=prices,
            total_return_prices=prices,
            open_prices=prices,
            volumes=volumes,
            market_caps=pd.DataFrame(index=dates),
            raw_fundamental_panels={},
            legacy_fundamentals={},
            macro_vintage_table=pd.DataFrame(),
            market_macro={},
            metadata=metadata,
            sp500_membership={},
            cache_dir=prepare.CACHE_DIR,
        )

        result = prepare.run_backtest(FixedLongStrategy, store, str(dates[0].date()), str(dates[-1].date()), rebalance_freq="W")
        self.assertFalse(result.daily_returns.isna().any())
        self.assertFalse(result.equity_curve.isna().any())
        self.assertAlmostEqual(float(result.daily_returns.iloc[3]), 0.0)
        self.assertGreaterEqual(int(result.diagnostics.get("missing_asset_return_count", 0)), 1)

    def test_run_backtest_filters_untradeable_target_into_cash(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.DataFrame(
            {
                "A": [100.0, np.nan, 105.0, 105.0],
                "B": [100.0, 100.0, 100.0, 100.0],
            },
            index=dates,
        )
        volumes = pd.DataFrame({"A": 1_000_000.0, "B": 1_000_000.0}, index=dates)
        metadata = {
            "A": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
            "B": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]},
        }
        store = prepare.DataStore(
            signal_prices=prices.ffill(),
            total_return_prices=prices,
            open_prices=prices.ffill(),
            volumes=volumes,
            market_caps=pd.DataFrame(index=dates),
            raw_fundamental_panels={},
            legacy_fundamentals={},
            macro_vintage_table=pd.DataFrame(),
            market_macro={},
            metadata=metadata,
            sp500_membership={},
            cache_dir=prepare.CACHE_DIR,
        )

        result = prepare.run_backtest(FixedLongStrategy, store, str(dates[0].date()), str(dates[-1].date()), rebalance_freq="W")
        self.assertEqual(int(result.diagnostics.get("filtered_trade_target_count", 0)), 1)
        trade_date, weights = result.weights_history[0]
        self.assertEqual(trade_date, dates[1])
        self.assertAlmostEqual(float(weights.get("__CASH__", 0.0)), 1.0)

    def test_build_us_universe_metadata_keeps_historical_only_members(self):
        client = FakeUniverseClient(
            {
                "sp500-constituent": [{"symbol": "AAA", "name": "A", "sector": "Tech", "subSector": "Software", "exchange": "NYSE"}],
                "historical-sp500-constituent": [
                    {"symbol": "AAA", "dateAdded": "2020-01-01", "dateRemoved": None},
                    {"symbol": "OLD", "dateAdded": "2010-01-01", "dateRemoved": "2015-01-01"},
                ],
                ("delisted-companies", 0): [],
            }
        )
        tickers, meta_df, membership_df = prepare.build_us_universe_metadata(client)
        self.assertIn("OLD", tickers)
        self.assertIn("OLD", set(meta_df["ticker"]))
        self.assertIn("OLD", set(membership_df["ticker"]))

    def test_build_us_universe_metadata_paginates_delisted(self):
        page0 = [{"symbol": f"D{i:04d}", "companyName": f"D{i:04d}", "exchange": "NYSE"} for i in range(1000)]
        page1 = [{"symbol": "TAIL", "companyName": "Tail Co", "exchange": "NYSE"}]
        client = FakeUniverseClient(
            {
                "sp500-constituent": [],
                "historical-sp500-constituent": [],
                ("delisted-companies", 0): page0,
                ("delisted-companies", 1): page1,
            }
        )
        tickers, meta_df, _ = prepare.build_us_universe_metadata(client)
        self.assertIn("TAIL", tickers)
        self.assertIn("TAIL", set(meta_df["ticker"]))

    def test_build_us_universe_metadata_requires_explicit_us_registry_country(self):
        client = FakeUniverseClient(
            {
                "sp500-constituent": [],
                "historical-sp500-constituent": [],
                ("delisted-companies", 0): [],
            }
        )
        with patch.object(
            prepare,
            "load_companies_json",
            return_value={
                "MISSING.HK": {"name": "Missing Country"},
                "REALUS": {"name": "US Name", "country": "US"},
            },
        ):
            tickers, meta_df, _ = prepare.build_us_universe_metadata(client)
        self.assertIn("REALUS", tickers)
        self.assertNotIn("MISSING.HK", tickers)
        self.assertNotIn("MISSING.HK", set(meta_df["ticker"]))

    def test_build_us_universe_metadata_filters_foreign_and_fund_like_rows(self):
        client = FakeUniverseClient(
            {
                "sp500-constituent": [],
                "historical-sp500-constituent": [],
                ("delisted-companies", 0): [
                    {"symbol": "GOOD", "companyName": "Good Co", "exchange": "NYSE"},
                    {"symbol": "ASCU.TO", "companyName": "Arizona Sonoran Copper Company Inc.", "exchange": "TSX"},
                    {"symbol": "BFRE", "companyName": "Westwood LBRTY Global Equity ETF", "exchange": "AMEX"},
                ],
            }
        )
        with patch.object(
            prepare,
            "load_companies_json",
            return_value={"GCU.TO": {"name": "Gunnison Copper Corp", "exchange": "TSX", "country": "US"}},
        ):
            tickers, meta_df, _ = prepare.build_us_universe_metadata(client)
        self.assertIn("GOOD", tickers)
        self.assertNotIn("ASCU.TO", tickers)
        self.assertNotIn("BFRE", tickers)
        self.assertNotIn("GCU.TO", tickers)
        self.assertEqual(set(meta_df["ticker"]), {"GOOD"})

    def test_build_coverage_audit_flags_price_support_as_backtest_gate(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pd.DataFrame({"date": ["2024-01-02"], "close": [10.0]}).to_parquet(tmp_path / "prices_AAA.parquet", index=False)
            pd.DataFrame({"date": ["2024-01-02"], "marketCap": [100.0]}).to_parquet(tmp_path / "marketcap_BBB.parquet", index=False)
            meta_df = pd.DataFrame(
                [
                    {"ticker": "AAA", "name": "AAA Co", "exchange": "NYSE"},
                    {"ticker": "BBB", "name": "BBB Co", "exchange": "NYSE"},
                    {"ticker": "CCC", "name": "", "exchange": ""},
                ]
            )
            audit = prepare.build_coverage_audit(meta_df, tmp_path).set_index("ticker")
            self.assertTrue(bool(audit.loc["AAA", "vendor_backtest_supported"]))
            self.assertEqual(audit.loc["AAA", "vendor_support_status"], "backtest_supported")
            self.assertEqual(audit.loc["BBB", "vendor_support_status"], "partial_vendor_support")
            self.assertEqual(audit.loc["CCC", "vendor_support_status"], "unsupported_by_vendor")
            self.assertTrue(bool(audit.loc["CCC", "vendor_identity_weak"]))

    def test_build_coverage_audit_marks_symbol_change_candidates(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pd.DataFrame({"date": ["2024-01-02"], "close": [10.0]}).to_parquet(tmp_path / "prices_NEW.parquet", index=False)
            meta_df = pd.DataFrame([{"ticker": "OLD", "name": "Old Co", "exchange": "NYSE"}])
            client = FakeUniverseClient(
                {
                    ("parquet", "ref_stock_list"): [{"symbol": "NEW", "companyName": "New Co"}],
                    ("parquet", "ref_financial_statement_symbol_list"): [],
                    ("parquet", "ref_symbol_change"): [{"oldSymbol": "OLD", "newSymbol": "NEW", "date": "2025-01-01"}],
                }
            )
            audit = prepare.build_coverage_audit(meta_df, tmp_path, client=client).set_index("ticker")
            self.assertEqual(audit.loc["OLD", "symbol_change_new_symbol"], "NEW")
            self.assertTrue(bool(audit.loc["OLD", "symbol_change_new_symbol_direct_support"]))
            self.assertEqual(audit.loc["OLD", "vendor_support_status"], "resolved_symbol_candidate")

    def test_run_backtest_trade_costs_hit_daily_returns(self):
        dates = pd.bdate_range("2024-01-01", periods=4)
        prices = pd.DataFrame({"A": [100.0, 100.0, 100.0, 100.0]}, index=dates)
        volumes = pd.DataFrame({"A": 1_000_000.0}, index=dates)
        metadata = {"A": {"country": "US", "sector": "Tech", "exchange": "NASDAQ", "listing_start_date": dates[0], "listing_end_date": dates[-1]}}
        store = prepare.DataStore(
            signal_prices=prices,
            total_return_prices=prices,
            open_prices=prices,
            volumes=volumes,
            market_caps=pd.DataFrame(index=dates),
            raw_fundamental_panels={},
            legacy_fundamentals={},
            macro_vintage_table=pd.DataFrame(),
            market_macro={},
            metadata=metadata,
            sp500_membership={},
            cache_dir=prepare.CACHE_DIR,
        )

        result = prepare.run_backtest(FixedLongStrategy, store, str(dates[0].date()), str(dates[-1].date()), rebalance_freq="W")
        self.assertAlmostEqual(float(result.daily_returns.iloc[1]), -0.002, places=6)
        curve_from_returns = (1 + result.daily_returns).cumprod() * prepare.INITIAL_CAPITAL
        self.assertAlmostEqual(float(curve_from_returns.iloc[-1]), float(result.equity_curve.iloc[-1]), places=6)

    def test_strategy_signals_survive_missing_fundamentals(self):
        store = make_store()
        date = store.prices_signal().index[-1]
        limited = prepare.DateLimitedStore(store, date)
        scores = strategy.signals(limited, date)
        self.assertGreater(len(scores), 0)


if __name__ == "__main__":
    unittest.main()
